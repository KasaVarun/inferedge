#include "inferedge/inference_scheduler.hpp"

#include "inferedge/bounded_queue.hpp"
#include "inferedge/concurrent_inference.hpp"
#include "inferedge/tensor_batch.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

namespace inferedge {

static_assert(kMaxSchedulerWorkers == kMaxConcurrentWorkers);

// One accepted request. The queue, the worker that popped it, or last-worker
// cleanup owns it. backstop arms the destructor only until an explicit
// publisher stores the completion. Move clears it so a moved-from promise is
// never published. A successful publisher disarms it, so the normal path calls
// set_value once.
struct RequestEntry {
    std::vector<NamedTensor> inputs;
    std::uint64_t request_id{0};
    std::uint64_t sequence{0};
    std::promise<SchedulerCompletion> completion;
    bool backstop{false};
    bool completion_timing{false};

    RequestEntry() = default;
    RequestEntry(const RequestEntry&) = delete;
    RequestEntry& operator=(const RequestEntry&) = delete;
    RequestEntry(RequestEntry&& other) noexcept
        : inputs(std::move(other.inputs)), request_id(other.request_id), sequence(other.sequence),
          completion(std::move(other.completion)), backstop(std::exchange(other.backstop, false)),
          completion_timing(other.completion_timing) {}
    RequestEntry& operator=(RequestEntry&&) = delete;

    ~RequestEntry() noexcept {
        if (!backstop) {
            return;
        }
        try {
            SchedulerCompletion completion_value;
            completion_value.request_id = request_id;
            completion_value.sequence = sequence;
            completion_value.status = CompletionStatus::kSchedulerFailed;
            completion_value.inference_error.kind = InferenceErrorKind::kExecution;
            // Short enough for libc++ and libstdc++ small-string storage.
            completion_value.inference_error.message = "unsettled";
            if (completion_timing) {
                completion_value.pre_publication_time = std::chrono::steady_clock::now();
            }
            completion.set_value(std::move(completion_value));
        } catch (...) {
            // already satisfied, no shared state, or allocation failure.
            // The first explicit completion, when there was one, stays in place.
        }
    }
};

static_assert(std::is_nothrow_move_constructible_v<RequestEntry>);
static_assert(std::is_nothrow_destructible_v<RequestEntry>);
static_assert(!std::is_copy_constructible_v<RequestEntry>);
static_assert(!std::is_copy_assignable_v<RequestEntry>);
static_assert(!std::is_move_assignable_v<RequestEntry>);

void stamp_pre_publication(SchedulerCompletion& completion, bool enabled) noexcept {
    if (!enabled) {
        return;
    }
    completion.pre_publication_time = std::chrono::steady_clock::now();
}

enum class State {
    kIdle,
    kStarting,
    kRunning,
    kDraining,
    kCanceling,
    kStopped,
    kFailed,
};

enum class StartupSignal {
    kPending,
    kStart,
    kAbort,
};

struct WorkerMetrics {
    std::uint64_t engine_run_count{0};
    std::uint64_t requests_entered_engine_runs{0};
    std::array<std::uint64_t, kSchedulerMetricsHistogramSize> batch_size_histogram{};
    std::uint64_t total_assembly_nanoseconds{0};
    std::uint64_t total_engine_run_nanoseconds{0};
    std::uint64_t total_split_nanoseconds{0};
};

struct WorkerSlot {
    std::unique_ptr<InferenceEngine> engine;
    std::jthread thread;
    bool retired{false};
    WorkerMetrics metrics;
};

namespace {

SchedulerError MakeError(SchedulerErrorKind kind, std::string message) {
    SchedulerError error;
    error.kind = kind;
    error.message = std::move(message);
    return error;
}

SchedulerError ValidateConfig(const SchedulerConfig& config) {
    if (config.model_path.empty()) {
        return MakeError(SchedulerErrorKind::kInvalidConfig, "model path is empty");
    }
    if (config.worker_count == 0 || config.worker_count > kMaxSchedulerWorkers) {
        return MakeError(SchedulerErrorKind::kInvalidConfig,
                         "worker_count must be from 1 through 64");
    }
    if (config.queue_capacity == 0 || config.queue_capacity > kMaxSchedulerQueueCapacity) {
        return MakeError(SchedulerErrorKind::kInvalidConfig,
                         "queue_capacity must be from 1 through 1024");
    }
    if (config.max_batch_size == 0 || config.max_batch_size > 32) {
        return MakeError(SchedulerErrorKind::kInvalidConfig,
                         "max_batch_size must be from 1 through 32");
    }
    if (config.max_batch_delay < std::chrono::milliseconds::zero() ||
        config.max_batch_delay > std::chrono::milliseconds{1000}) {
        return MakeError(SchedulerErrorKind::kInvalidConfig,
                         "max_batch_delay must be from 0 through 1000 milliseconds");
    }
    return {};
}

bool CheckedMultiply(std::size_t left, std::size_t right, std::size_t& product) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    product = left * right;
    return true;
}

bool FitsSize(std::int64_t value, std::size_t& converted) {
    if (value < 0) {
        return false;
    }
    const auto unsigned_value = static_cast<std::uint64_t>(value);
    if (unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    converted = static_cast<std::size_t>(unsigned_value);
    return true;
}

SchedulerError MakeModelLoad(std::string message) {
    SchedulerError error;
    error.kind = SchedulerErrorKind::kModelLoad;
    error.load_error = InferenceError{InferenceErrorKind::kModelInvalid, message};
    error.message = std::move(message);
    return error;
}

// Same shape contract the batch helper will require, including a static
// dimension product that fits in std::size_t. Checked here so start() can
// fail before any worker thread exists.
SchedulerError ValidateBatchSignature(const std::vector<TensorMetadata>& metadata,
                                      const char* role) {
    if (metadata.empty()) {
        return MakeModelLoad(std::string{role} + " metadata is empty");
    }
    for (std::size_t index = 0; index < metadata.size(); ++index) {
        const TensorMetadata& tensor = metadata[index];
        if (tensor.element_type != ElementType::kFloat32) {
            return MakeModelLoad(std::string{role} + " " + tensor.name + " is not float32");
        }
        if (tensor.name.empty()) {
            return MakeModelLoad(std::string{role} + " metadata name is empty");
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (metadata[earlier].name == tensor.name) {
                return MakeModelLoad(std::string{role} +
                                     " metadata name is duplicated: " + tensor.name);
            }
        }
        if (tensor.dimensions.size() < 2) {
            return MakeModelLoad(std::string{role} + " " + tensor.name + " rank is below 2");
        }
        if (tensor.dimensions[0] >= 0) {
            return MakeModelLoad(std::string{role} + " " + tensor.name + " axis 0 is not dynamic");
        }
        std::size_t row = 1;
        for (std::size_t axis = 1; axis < tensor.dimensions.size(); ++axis) {
            const std::int64_t dimension = tensor.dimensions[axis];
            if (dimension < 0) {
                return MakeModelLoad(std::string{role} + " " + tensor.name +
                                     " has an additional dynamic axis");
            }
            if (dimension == 0) {
                return MakeModelLoad(std::string{role} + " " + tensor.name +
                                     " has a non-positive dimension");
            }
            std::size_t converted = 0;
            if (!FitsSize(dimension, converted) || !CheckedMultiply(row, converted, row)) {
                return MakeModelLoad(std::string{role} + " " + tensor.name +
                                     " shape product overflows");
            }
        }
    }
    return {};
}

SchedulerError ValidateBatchCapability(const InferenceEngine& engine) {
    if (SchedulerError inputs = ValidateBatchSignature(engine.inputs(), "input");
        inputs.kind != SchedulerErrorKind::kNone) {
        return inputs;
    }
    return ValidateBatchSignature(engine.outputs(), "output");
}

std::string WorkerFailureMessage(std::string_view text) {
    if (text.size() <= 15) {
        return std::string{text};
    }
    return "worker error";
}

void SaturatingAdd(std::uint64_t& total, std::uint64_t add) noexcept {
    if (add > std::numeric_limits<std::uint64_t>::max() - total) {
        total = std::numeric_limits<std::uint64_t>::max();
        return;
    }
    total += add;
}

void AddElapsed(std::uint64_t& total, std::chrono::steady_clock::time_point start,
                std::chrono::steady_clock::time_point stop) noexcept {
    if (stop < start) {
        return;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    if (elapsed.count() < 0) {
        return;
    }
    SaturatingAdd(total, static_cast<std::uint64_t>(elapsed.count()));
}

void RecordEngineInvocation(WorkerMetrics& metrics, std::size_t batch_size,
                            std::chrono::steady_clock::time_point start,
                            std::chrono::steady_clock::time_point stop) noexcept {
    SaturatingAdd(metrics.engine_run_count, 1);
    SaturatingAdd(metrics.requests_entered_engine_runs, static_cast<std::uint64_t>(batch_size));
    if (batch_size >= 1 && batch_size <= kSchedulerMetricsMaxBatchSize) {
        SaturatingAdd(metrics.batch_size_histogram[batch_size], 1);
    }
    AddElapsed(metrics.total_engine_run_nanoseconds, start, stop);
}

void MergeWorkerMetrics(WorkerMetrics& dest, const WorkerMetrics& src) noexcept {
    SaturatingAdd(dest.engine_run_count, src.engine_run_count);
    SaturatingAdd(dest.requests_entered_engine_runs, src.requests_entered_engine_runs);
    for (std::size_t index = 0; index < dest.batch_size_histogram.size(); ++index) {
        SaturatingAdd(dest.batch_size_histogram[index], src.batch_size_histogram[index]);
    }
    SaturatingAdd(dest.total_assembly_nanoseconds, src.total_assembly_nanoseconds);
    SaturatingAdd(dest.total_engine_run_nanoseconds, src.total_engine_run_nanoseconds);
    SaturatingAdd(dest.total_split_nanoseconds, src.total_split_nanoseconds);
}

enum class BatchStep {
    kContinue,
    kClosed,
    kFault,
};

inline constexpr std::size_t kNoFault = std::numeric_limits<std::size_t>::max();

struct ExecutingGuard {
    std::atomic<std::uint64_t>& executing;

    explicit ExecutingGuard(std::atomic<std::uint64_t>& value) : executing(value) {
        executing.fetch_add(1, std::memory_order_relaxed);
    }

    ~ExecutingGuard() { executing.fetch_sub(1, std::memory_order_relaxed); }
};

const TensorMetadata* FindModelTensor(const std::vector<TensorMetadata>& metadata,
                                      const std::string& name) {
    for (const TensorMetadata& tensor : metadata) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

const NamedTensor* FindNamedTensor(const std::vector<NamedTensor>& tensors,
                                   const std::string& name) {
    for (const NamedTensor& tensor : tensors) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

InferenceError InputError(InferenceErrorKind kind, std::string message) {
    return InferenceError{kind, std::move(message)};
}

// Per-request check used before assembly. A failure here completes only that
// request. The rules match the batch helper's single-request checks.
std::optional<InferenceError> ValidateQueuedRequest(const InferenceEngine& engine,
                                                    const RequestEntry& entry) {
    const std::vector<TensorMetadata>& model = engine.inputs();
    const std::vector<NamedTensor>& tensors = entry.inputs;
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const std::string& name = tensors[index].name;
        if (name.empty()) {
            return InputError(InferenceErrorKind::kEmptyName, "input name is empty");
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (tensors[earlier].name == name) {
                return InputError(InferenceErrorKind::kDuplicateInput,
                                  "duplicate input name: " + name);
            }
        }
        if (FindModelTensor(model, name) == nullptr) {
            return InputError(InferenceErrorKind::kUnexpectedInput,
                              "unexpected input name: " + name);
        }
    }
    for (const TensorMetadata& expected : model) {
        if (FindNamedTensor(tensors, expected.name) == nullptr) {
            return InputError(InferenceErrorKind::kMissingInput,
                              "missing input name: " + expected.name);
        }
    }
    for (const TensorMetadata& expected : model) {
        const NamedTensor* tensor = FindNamedTensor(tensors, expected.name);
        const std::string where = "input tensor " + tensor->name;
        if (tensor->shape.size() != expected.dimensions.size()) {
            return InputError(InferenceErrorKind::kShapeMismatch,
                              where + " rank does not match metadata");
        }
        if (tensor->shape[0] != 1) {
            return InputError(InferenceErrorKind::kShapeMismatch, where + " batch axis is not 1");
        }
        for (std::size_t axis = 1; axis < expected.dimensions.size(); ++axis) {
            if (tensor->shape[axis] != expected.dimensions[axis]) {
                return InputError(InferenceErrorKind::kShapeMismatch,
                                  where + " static dimension does not match metadata");
            }
        }
        std::size_t count = 1;
        for (const std::int64_t dimension : tensor->shape) {
            if (dimension < 0) {
                return InputError(InferenceErrorKind::kShapeMismatch,
                                  where + " dimension is negative");
            }
            std::size_t converted = 0;
            if (!FitsSize(dimension, converted) || !CheckedMultiply(count, converted, count)) {
                return InputError(InferenceErrorKind::kElementCountMismatch,
                                  where + " shape product overflows");
            }
        }
        if (tensor->values.size() != count) {
            return InputError(InferenceErrorKind::kElementCountMismatch,
                              where + " element count does not match its shape");
        }
    }
    return std::nullopt;
}

InferenceError FromBatchError(const TensorBatchError& error) {
    InferenceError mapped;
    mapped.message = error.message;
    switch (error.kind) {
    case TensorBatchErrorKind::kRankMismatch:
    case TensorBatchErrorKind::kBatchAxisMismatch:
    case TensorBatchErrorKind::kShapeMismatch:
        mapped.kind = InferenceErrorKind::kShapeMismatch;
        break;
    case TensorBatchErrorKind::kElementCountMismatch:
        mapped.kind = InferenceErrorKind::kElementCountMismatch;
        break;
    case TensorBatchErrorKind::kEmptyName:
        mapped.kind = InferenceErrorKind::kEmptyName;
        break;
    case TensorBatchErrorKind::kUnsupportedType:
        mapped.kind = InferenceErrorKind::kUnsupportedType;
        break;
    default:
        mapped.kind = InferenceErrorKind::kExecution;
        break;
    }
    return mapped;
}

} // namespace

struct InferenceScheduler::Impl {
    explicit Impl(SchedulerConfig config) : config_(std::move(config)) {}
    ~Impl() noexcept;

    SchedulerError Start();
    Admission Submit(std::vector<NamedTensor> inputs);
    ShutdownResult Shutdown(ShutdownPolicy policy);
    SchedulerStats Stats() const;
    SchedulerMetrics Metrics() const;

    SchedulerError StartWorkers();
    SchedulerError AbortStart(SchedulerError error);
    bool WaitForStart();
    bool GateIsStart() const;
    void WorkerMain(std::uint32_t index, InferenceEngine* engine) noexcept;
    BatchStep RunBatch(std::uint32_t index, InferenceEngine& engine);
    bool PublishInferenceFailure(RequestEntry& entry, InferenceError error);
    bool PublishBatchSuccess(RequestEntry& entry, std::vector<NamedTensor> outputs);
    void NotePublished(CompletionStatus status);
    void FailUnpublished(std::vector<RequestEntry>& owned, std::string_view text);
    bool ConsumeThrow(std::uint32_t index);
    CompletionStatus Execute(InferenceEngine& engine, RequestEntry& entry, WorkerMetrics* metrics);
    bool PublishWorkerFailure(RequestEntry& entry, std::string_view text);
    void PublishSchedulerFailure(RequestEntry& entry) noexcept;
    void PublishCanceled(RequestEntry& entry) noexcept;
    void RecordLocked(CompletionStatus status);
    void Retire(std::uint32_t index, bool faulted) noexcept;
    void FinishLastWorker(bool faulted);
    std::uint64_t PublishQueuedFailures() noexcept;
    void JoinWorkers() noexcept;
    void EmergencyStop() noexcept;
    AdmissionStatus RejectionStatus() const;
    void WaitForSettledShutdown(std::unique_lock<std::mutex>& lock);
    void JoinOnce(std::unique_lock<std::mutex>& lock);
    void PublishJoinFinished(std::unique_lock<std::mutex>& lock, std::uint64_t swept);
    ShutdownResult SettledResult() const;

    SchedulerConfig config_;
    mutable std::mutex admission_;
    mutable std::mutex gate_mu_;
    std::condition_variable state_cv_;
    std::condition_variable gate_cv_;
    State state_{State::kIdle};
    StartupSignal gate_{StartupSignal::kPending};
    bool shutdown_requested_{false};
    bool joined_{false};
    bool join_claimed_{false};
    bool coordinator_claimed_{false};
    bool policy_established_{false};
    ShutdownPolicy user_policy_{ShutdownPolicy::kDrain};
    bool cleanup_claimed_{false};

    std::atomic<bool> fail_close_cancel_{false};
    std::atomic<bool> fail_join_settle_{false};
    // Test-only. any_throw_armed_ keeps an inactive worker to one atomic load.
    // Each armed worker consumes its own flag once, after it has popped.
    std::atomic<bool> any_throw_armed_{false};
    std::array<std::atomic<bool>, kMaxSchedulerWorkers> throw_armed_{};
    std::atomic<bool> fail_batch_guard_creation_{false};
    std::atomic<std::size_t> throw_after_batch_collect_{kNoFault};
    std::atomic<std::size_t> throw_before_batch_publish_{kNoFault};
    std::atomic<std::size_t> fail_worker_failure_publication_at_{kNoFault};
    std::atomic<std::size_t> worker_failure_publication_attempt_{0};
    std::optional<std::uint32_t> stop_after_;
    std::shared_ptr<std::latch> before_pop_;
    // Test-only per-worker pause before wait_pop. The flag stays false unless
    // a test stores a latch, so production does not touch the array.
    std::atomic<bool> worker_pop_gate_armed_{false};
    std::array<std::shared_ptr<std::latch>, kMaxSchedulerWorkers> worker_pop_gate_{};
    std::function<void()> before_loop_hook_;
    std::shared_ptr<std::latch> reached_run_;
    std::shared_ptr<std::latch> before_run_;
    std::function<void(std::size_t)> on_batch_run_;
    std::shared_ptr<std::latch> reached_batch_wait_;
    std::shared_ptr<std::latch> before_batch_wait_;
    std::shared_ptr<std::latch> after_publish_;
    std::shared_ptr<std::latch> hold_other_workers_;
    std::shared_ptr<std::latch> after_drain_closed_;
    std::shared_ptr<std::latch> shutdown_follower_entered_;
    std::function<void(std::future<SchedulerCompletion>&)> on_accepted_;

    std::uint64_t next_id_{0};
    std::uint64_t accepted_{0};
    std::uint64_t rejected_full_{0};
    std::uint64_t rejected_closed_{0};
    std::uint64_t succeeded_{0};
    std::uint64_t failed_inference_{0};
    std::uint64_t failed_worker_{0};
    std::uint64_t failed_scheduler_{0};
    std::uint64_t canceled_{0};
    std::atomic<std::uint64_t> executing_{0};
    std::atomic<std::uint32_t> live_workers_{0};

    // Written only after every joinable worker thread joined successfully.
    // metrics() copies it under admission_ only when metrics_merged_ is true.
    WorkerMetrics published_metrics_{};
    bool metrics_merged_{false};

    // queue_ is declared before workers_ so a joined jthread is destroyed
    // before the queue it waited on.
    std::unique_ptr<BoundedFifoQueue<RequestEntry>> queue_;
    std::vector<WorkerSlot> workers_;
};

InferenceScheduler::Impl::~Impl() noexcept {
    try {
        static_cast<void>(Shutdown(ShutdownPolicy::kDrain));
    } catch (...) {
        EmergencyStop();
    }
}

SchedulerError InferenceScheduler::Impl::Start() {
    {
        std::lock_guard<std::mutex> lock(admission_);
        if (state_ != State::kIdle) {
            return MakeError(SchedulerErrorKind::kInvalidState, "scheduler cannot start");
        }
        if (SchedulerError config_error = ValidateConfig(config_);
            config_error.kind != SchedulerErrorKind::kNone) {
            return config_error;
        }
        state_ = State::kStarting;
    }

    try {
        return StartWorkers();
    } catch (const std::exception& ex) {
        return AbortStart(MakeError(SchedulerErrorKind::kWorkerStart, ex.what()));
    } catch (...) {
        return AbortStart(MakeError(SchedulerErrorKind::kWorkerStart, "worker start failed"));
    }
}

SchedulerError InferenceScheduler::Impl::StartWorkers() {
    queue_ = std::make_unique<BoundedFifoQueue<RequestEntry>>(config_.queue_capacity);

    std::vector<std::unique_ptr<InferenceEngine>> engines;
    engines.reserve(config_.worker_count);
    for (std::uint32_t index = 0; index < config_.worker_count; ++index) {
        EngineLoad loaded = load_inference_engine(config_.model_path);
        if (loaded.error.kind != InferenceErrorKind::kNone || loaded.engine == nullptr) {
            SchedulerError error;
            error.kind = SchedulerErrorKind::kModelLoad;
            error.load_error = std::move(loaded.error);
            if (error.load_error.message.empty()) {
                error.message = "model load failed";
            } else {
                error.message = error.load_error.message;
            }
            return AbortStart(std::move(error));
        }
        engines.push_back(std::move(loaded.engine));
    }

    if (config_.enable_batching) {
        if (SchedulerError capability = ValidateBatchCapability(*engines.front());
            capability.kind != SchedulerErrorKind::kNone) {
            return AbortStart(std::move(capability));
        }
    }

    workers_.reserve(engines.size());
    for (std::uint32_t index = 0; index < static_cast<std::uint32_t>(engines.size()); ++index) {
        if (stop_after_.has_value() && index == *stop_after_) {
            return AbortStart(
                MakeError(SchedulerErrorKind::kWorkerStart, "partial worker startup aborted"));
        }
        workers_.push_back(WorkerSlot{});
        workers_.back().engine = std::move(engines[static_cast<std::size_t>(index)]);
        InferenceEngine* const engine = workers_.back().engine.get();
        // The engine object is owned by this slot's unique_ptr. reserve() keeps
        // the slot stable, and the pointer stays valid across unique_ptr moves.
        workers_.back().thread =
            std::jthread([this, index, engine]() noexcept { WorkerMain(index, engine); });
    }

    bool abort = false;
    {
        std::lock_guard<std::mutex> admission(admission_);
        if (shutdown_requested_) {
            abort = true;
        } else {
            {
                std::lock_guard<std::mutex> gate(gate_mu_);
                gate_ = StartupSignal::kStart;
            }
            live_workers_.store(static_cast<std::uint32_t>(workers_.size()),
                                std::memory_order_release);
            state_ = State::kRunning;
        }
    }
    if (abort) {
        return AbortStart(
            MakeError(SchedulerErrorKind::kShutdownDuringStart, "shutdown overlapped startup"));
    }
    gate_cv_.notify_all();
    return {};
}

SchedulerError InferenceScheduler::Impl::AbortStart(SchedulerError error) {
    {
        std::lock_guard<std::mutex> gate(gate_mu_);
        gate_ = StartupSignal::kAbort;
    }
    gate_cv_.notify_all();
    {
        std::lock_guard<std::mutex> lock(admission_);
        join_claimed_ = true;
    }
    JoinWorkers();
    if (queue_ != nullptr) {
        try {
            if (!queue_->is_closed()) {
                static_cast<void>(queue_->close_drain());
            }
        } catch (...) {
        }
    }
    {
        std::lock_guard<std::mutex> lock(admission_);
        state_ = State::kFailed;
        joined_ = true;
    }
    state_cv_.notify_all();
    return error;
}

bool InferenceScheduler::Impl::PublishInferenceFailure(RequestEntry& entry, InferenceError error) {
    SchedulerCompletion completion;
    completion.request_id = entry.request_id;
    completion.sequence = entry.sequence;
    completion.status = CompletionStatus::kInferenceFailed;
    completion.inference_error = std::move(error);
    stamp_pre_publication(completion, entry.completion_timing);
    try {
        entry.completion.set_value(std::move(completion));
    } catch (const std::future_error&) {
        entry.backstop = false;
        return false;
    }
    entry.backstop = false;
    return true;
}

bool InferenceScheduler::Impl::PublishBatchSuccess(RequestEntry& entry,
                                                   std::vector<NamedTensor> outputs) {
    SchedulerCompletion completion;
    completion.request_id = entry.request_id;
    completion.sequence = entry.sequence;
    completion.status = CompletionStatus::kSucceeded;
    completion.outputs = std::move(outputs);
    stamp_pre_publication(completion, entry.completion_timing);
    try {
        entry.completion.set_value(std::move(completion));
    } catch (const std::future_error&) {
        entry.backstop = false;
        return false;
    }
    entry.backstop = false;
    return true;
}

void InferenceScheduler::Impl::NotePublished(CompletionStatus status) {
    if (const std::shared_ptr<std::latch> after_publish = after_publish_;
        after_publish != nullptr) {
        after_publish->count_down();
    }
    try {
        std::lock_guard<std::mutex> lock(admission_);
        RecordLocked(status);
    } catch (...) {
    }
}

void InferenceScheduler::Impl::FailUnpublished(std::vector<RequestEntry>& owned,
                                               std::string_view text) {
    for (RequestEntry& entry : owned) {
        if (!entry.backstop) {
            continue;
        }
        // One entry's publication failure must not skip the entries after it.
        // A promise that is still armed keeps the destructor backstop.
        try {
            if (PublishWorkerFailure(entry, text)) {
                NotePublished(CompletionStatus::kWorkerFailed);
            }
        } catch (...) {
        }
    }
}

BatchStep InferenceScheduler::Impl::RunBatch(std::uint32_t index, InferenceEngine& engine) {
    // Guards are declared first so a still-armed entry's destructor runs while
    // its executing count is still held.
    std::vector<std::unique_ptr<ExecutingGuard>> guards;
    std::vector<RequestEntry> owned;
    // The request most recently removed from the queue, until it is published
    // or moved into owned. The catch below can still see it.
    std::optional<RequestEntry> incoming;

    auto fail_parked = [&](std::string_view text) {
        try {
            if (incoming.has_value() && incoming->backstop &&
                PublishWorkerFailure(*incoming, text)) {
                NotePublished(CompletionStatus::kWorkerFailed);
            }
        } catch (...) {
        }
        FailUnpublished(owned, text);
    };

    try {
        // Reserve before any pop. A throw here owns no request. Later
        // insertions of a nothrow move cannot allocate.
        const std::size_t limit = static_cast<std::size_t>(config_.max_batch_size);
        owned.reserve(limit);
        guards.reserve(limit);

        QueuePop<RequestEntry> popped = queue_->wait_pop();
        if (popped.result != QueuePopResult::kReady || !popped.value.has_value()) {
            return BatchStep::kClosed;
        }
        incoming.emplace(std::move(*popped.value));
        popped.value.reset();

        const bool wait_for_more = config_.max_batch_delay > std::chrono::milliseconds::zero();
        const std::chrono::steady_clock::time_point deadline =
            std::chrono::steady_clock::now() + config_.max_batch_delay;
        bool signaled_wait = false;

        // false means this worker must retire. incoming stays the owner until
        // the entry is published or moved into owned.
        auto adopt = [&]() -> bool {
            std::unique_ptr<ExecutingGuard> guard;
            try {
                if (fail_batch_guard_creation_.exchange(false, std::memory_order_acq_rel)) {
                    throw std::bad_alloc();
                }
                guard = std::make_unique<ExecutingGuard>(executing_);
            } catch (const std::bad_alloc&) {
                fail_parked("alloc failed");
                return false;
            }

            std::optional<InferenceError> invalid;
            try {
                invalid = ValidateQueuedRequest(engine, *incoming);
            } catch (const std::bad_alloc&) {
                fail_parked("alloc failed");
                return false;
            } catch (const std::length_error&) {
                fail_parked("length error");
                return false;
            } catch (const std::exception& ex) {
                fail_parked(ex.what());
                return false;
            } catch (...) {
                fail_parked("worker error");
                return false;
            }
            if (invalid.has_value()) {
                // The local guard stays alive through this publication and is
                // released when adopt returns. It does not cover the neighbors.
                try {
                    if (PublishInferenceFailure(*incoming, std::move(*invalid))) {
                        NotePublished(CompletionStatus::kInferenceFailed);
                    }
                } catch (const std::bad_alloc&) {
                    fail_parked("alloc failed");
                    return false;
                } catch (const std::length_error&) {
                    fail_parked("length error");
                    return false;
                } catch (const std::exception& ex) {
                    fail_parked(ex.what());
                    return false;
                } catch (...) {
                    fail_parked("worker error");
                    return false;
                }
                incoming.reset();
                return true;
            }
            // The guard joins the vector before the entry moves, so the
            // increment outlives a throw from that move.
            guards.push_back(std::move(guard));
            owned.push_back(std::move(*incoming));
            incoming.reset();
            return true;
        };

        if (!adopt()) {
            return BatchStep::kFault;
        }
        std::size_t collect_fault = owned.size();
        if (throw_after_batch_collect_.compare_exchange_strong(collect_fault, kNoFault,
                                                               std::memory_order_acq_rel)) {
            throw std::runtime_error("collect failed");
        }

        while (owned.size() < limit) {
            // A drain-closed queue still returns kReady while items remain.
            // kClosed means closed and empty, so the group seals without
            // waiting out the deadline. An open empty queue is the only
            // path into wait_pop_until.
            QueueTake<RequestEntry> taken = queue_->try_pop();
            if (taken.result == QueueTakeResult::kReady && taken.value.has_value()) {
                incoming.emplace(std::move(*taken.value));
                taken.value.reset();
                if (!adopt()) {
                    return BatchStep::kFault;
                }
                collect_fault = owned.size();
                if (throw_after_batch_collect_.compare_exchange_strong(collect_fault, kNoFault,
                                                                       std::memory_order_acq_rel)) {
                    throw std::runtime_error("collect failed");
                }
                continue;
            }
            if (taken.result == QueueTakeResult::kClosed) {
                break;
            }
            if (!wait_for_more || std::chrono::steady_clock::now() >= deadline) {
                break;
            }
            if (!signaled_wait) {
                signaled_wait = true;
                if (const std::shared_ptr<std::latch> reached = reached_batch_wait_;
                    reached != nullptr) {
                    reached->count_down();
                }
                if (const std::shared_ptr<std::latch> hold = before_batch_wait_; hold != nullptr) {
                    hold->wait();
                }
            }
            QueueWait<RequestEntry> waited = queue_->wait_pop_until(deadline);
            if (waited.result == QueueWaitResult::kReady && waited.value.has_value()) {
                incoming.emplace(std::move(*waited.value));
                waited.value.reset();
                if (!adopt()) {
                    return BatchStep::kFault;
                }
                collect_fault = owned.size();
                if (throw_after_batch_collect_.compare_exchange_strong(collect_fault, kNoFault,
                                                                       std::memory_order_acq_rel)) {
                    throw std::runtime_error("collect failed");
                }
                continue;
            }
            break;
        }

        if (owned.empty()) {
            return BatchStep::kContinue;
        }

        WorkerMetrics* metrics = nullptr;
        if (config_.enable_metrics && index < workers_.size()) {
            metrics = &workers_[index].metrics;
        }

        std::vector<std::span<const NamedTensor>> requests;
        requests.reserve(owned.size());
        for (const RequestEntry& entry : owned) {
            requests.emplace_back(entry.inputs.data(), entry.inputs.size());
        }
        TensorBatchAssembly assembled;
        if (metrics == nullptr) {
            assembled = AssembleBatchInputs(engine.inputs(), requests);
        } else {
            const auto assembly_start = std::chrono::steady_clock::now();
            try {
                assembled = AssembleBatchInputs(engine.inputs(), requests);
            } catch (...) {
                AddElapsed(metrics->total_assembly_nanoseconds, assembly_start,
                           std::chrono::steady_clock::now());
                throw;
            }
            AddElapsed(metrics->total_assembly_nanoseconds, assembly_start,
                       std::chrono::steady_clock::now());
        }
        if (assembled.error.kind != TensorBatchErrorKind::kNone) {
            FailUnpublished(owned, assembled.error.kind == TensorBatchErrorKind::kOverflow
                                       ? "shape overflow"
                                       : "batch failed");
            return BatchStep::kFault;
        }

        std::vector<NamedTensorView> views;
        views.reserve(assembled.inputs.size());
        for (const NamedTensor& input : assembled.inputs) {
            views.push_back(
                NamedTensorView{input.name, input.shape, std::span<const float>{input.values}});
        }
        if (on_batch_run_) {
            on_batch_run_(owned.size());
        }
        if (const std::shared_ptr<std::latch> reached = reached_run_; reached != nullptr) {
            reached->count_down();
        }
        if (const std::shared_ptr<std::latch> before_run = before_run_; before_run != nullptr) {
            before_run->wait();
        }
        if (ConsumeThrow(index)) {
            FailUnpublished(owned, "injected");
            return BatchStep::kFault;
        }

        InferenceResult result;
        if (metrics == nullptr) {
            result = engine.run(views);
        } else {
            const std::size_t batch_size = owned.size();
            const auto run_start = std::chrono::steady_clock::now();
            try {
                result = engine.run(views);
            } catch (...) {
                RecordEngineInvocation(*metrics, batch_size, run_start,
                                       std::chrono::steady_clock::now());
                throw;
            }
            RecordEngineInvocation(*metrics, batch_size, run_start,
                                   std::chrono::steady_clock::now());
        }
        if (result.error.kind != InferenceErrorKind::kNone) {
            for (RequestEntry& entry : owned) {
                if (PublishInferenceFailure(entry, result.error)) {
                    NotePublished(CompletionStatus::kInferenceFailed);
                }
            }
            return BatchStep::kContinue;
        }

        TensorBatchSplit split;
        if (metrics == nullptr) {
            split = SplitBatchOutputs(engine.outputs(), result.outputs, owned.size());
        } else {
            const auto split_start = std::chrono::steady_clock::now();
            try {
                split = SplitBatchOutputs(engine.outputs(), result.outputs, owned.size());
            } catch (...) {
                AddElapsed(metrics->total_split_nanoseconds, split_start,
                           std::chrono::steady_clock::now());
                throw;
            }
            AddElapsed(metrics->total_split_nanoseconds, split_start,
                       std::chrono::steady_clock::now());
        }
        if (split.error.kind == TensorBatchErrorKind::kOverflow) {
            FailUnpublished(owned, "shape overflow");
            return BatchStep::kFault;
        }
        if (split.error.kind != TensorBatchErrorKind::kNone ||
            split.requests.size() != owned.size()) {
            if (split.error.kind == TensorBatchErrorKind::kNone) {
                FailUnpublished(owned, "batch failed");
                return BatchStep::kFault;
            }
            const InferenceError mapped = FromBatchError(split.error);
            for (RequestEntry& entry : owned) {
                if (PublishInferenceFailure(entry, mapped)) {
                    NotePublished(CompletionStatus::kInferenceFailed);
                }
            }
            return BatchStep::kContinue;
        }
        // A throw after some successes leaves those promises satisfied and
        // their backstops clear. The catch publishes only the entries that
        // are still armed, so a published request is not completed twice
        // and is not counted twice.
        for (std::size_t request_index = 0; request_index < owned.size(); ++request_index) {
            std::size_t expected = request_index;
            if (throw_before_batch_publish_.compare_exchange_strong(expected, kNoFault,
                                                                    std::memory_order_acq_rel)) {
                throw std::runtime_error("publish failed");
            }
            if (PublishBatchSuccess(owned[request_index],
                                    std::move(split.requests[request_index]))) {
                NotePublished(CompletionStatus::kSucceeded);
            }
        }
        return BatchStep::kContinue;
    } catch (const std::bad_alloc&) {
        fail_parked("alloc failed");
        return BatchStep::kFault;
    } catch (const std::length_error&) {
        fail_parked("length error");
        return BatchStep::kFault;
    } catch (const std::exception& ex) {
        fail_parked(ex.what());
        return BatchStep::kFault;
    } catch (...) {
        fail_parked("worker error");
        return BatchStep::kFault;
    }
}

bool InferenceScheduler::Impl::WaitForStart() {
    std::unique_lock<std::mutex> lock(gate_mu_);
    gate_cv_.wait(lock, [this] { return gate_ != StartupSignal::kPending; });
    return gate_ == StartupSignal::kStart;
}

bool InferenceScheduler::Impl::GateIsStart() const {
    std::lock_guard<std::mutex> lock(gate_mu_);
    return gate_ == StartupSignal::kStart;
}

void InferenceScheduler::Impl::WorkerMain(std::uint32_t index, InferenceEngine* engine) noexcept {
    bool entered = false;
    bool faulted = false;
    try {
        if (!WaitForStart()) {
            return;
        }
        entered = true;
        for (;;) {
            if (before_loop_hook_) {
                before_loop_hook_();
            }
            if (const std::shared_ptr<std::latch> hold = hold_other_workers_;
                hold != nullptr && index > 0) {
                hold->wait();
            }
            if (worker_pop_gate_armed_.load(std::memory_order_acquire) &&
                index < worker_pop_gate_.size()) {
                if (const std::shared_ptr<std::latch> gate = worker_pop_gate_[index];
                    gate != nullptr) {
                    gate->wait();
                }
            }
            if (const std::shared_ptr<std::latch> before_pop = before_pop_; before_pop != nullptr) {
                before_pop->wait();
            }

            if (config_.enable_batching) {
                const BatchStep step = RunBatch(index, *engine);
                if (step == BatchStep::kClosed) {
                    break;
                }
                if (step == BatchStep::kFault) {
                    faulted = true;
                    break;
                }
                continue;
            }

            QueuePop<RequestEntry> popped = queue_->wait_pop();
            if (popped.result != QueuePopResult::kReady || !popped.value.has_value()) {
                break;
            }
            RequestEntry entry = std::move(*popped.value);
            popped.value.reset();
            ExecutingGuard executing_guard{executing_};

            bool published = false;
            bool retire_now = false;
            CompletionStatus status = CompletionStatus::kWorkerFailed;
            try {
                if (const std::shared_ptr<std::latch> reached = reached_run_; reached != nullptr) {
                    reached->count_down();
                }
                if (const std::shared_ptr<std::latch> before_run = before_run_;
                    before_run != nullptr) {
                    before_run->wait();
                }
                // The injected fault stays after the pop proof so a test can
                // cancel the queue before this worker throws.
                if (ConsumeThrow(index)) {
                    retire_now = true;
                    throw std::runtime_error("injected");
                }
                WorkerMetrics* metrics = nullptr;
                if (config_.enable_metrics && index < workers_.size()) {
                    metrics = &workers_[index].metrics;
                }
                status = Execute(*engine, entry, metrics);
                published = true;
            } catch (const std::exception& ex) {
                if (!published && PublishWorkerFailure(entry, ex.what())) {
                    published = true;
                    status = CompletionStatus::kWorkerFailed;
                }
                retire_now = true;
            } catch (...) {
                if (!published && PublishWorkerFailure(entry, "worker error")) {
                    published = true;
                    status = CompletionStatus::kWorkerFailed;
                }
                retire_now = true;
            }

            if (const std::shared_ptr<std::latch> after_publish = after_publish_;
                after_publish != nullptr && published) {
                after_publish->count_down();
            }
            try {
                std::lock_guard<std::mutex> lock(admission_);
                if (published) {
                    RecordLocked(status);
                }
            } catch (...) {
            }
            if (retire_now) {
                faulted = true;
                break;
            }
        }
    } catch (...) {
        faulted = true;
    }

    if (!entered) {
        try {
            if (GateIsStart()) {
                entered = true;
            }
        } catch (...) {
        }
    }
    if (entered) {
        Retire(index, faulted);
    }
}

bool InferenceScheduler::Impl::ConsumeThrow(std::uint32_t index) {
    if (!any_throw_armed_.load(std::memory_order_acquire) || index >= throw_armed_.size()) {
        return false;
    }
    bool expected = true;
    return throw_armed_[index].compare_exchange_strong(expected, false, std::memory_order_acq_rel);
}

CompletionStatus InferenceScheduler::Impl::Execute(InferenceEngine& engine, RequestEntry& entry,
                                                   WorkerMetrics* metrics) {
    InferenceResult result;
    {
        // Views borrow entry.inputs and die before entry is destroyed.
        std::vector<NamedTensorView> views;
        views.reserve(entry.inputs.size());
        for (const NamedTensor& input : entry.inputs) {
            views.push_back(
                NamedTensorView{input.name, input.shape, std::span<const float>{input.values}});
        }
        if (metrics == nullptr) {
            result = engine.run(views);
        } else {
            const auto run_start = std::chrono::steady_clock::now();
            try {
                result = engine.run(views);
            } catch (...) {
                RecordEngineInvocation(*metrics, 1, run_start, std::chrono::steady_clock::now());
                throw;
            }
            RecordEngineInvocation(*metrics, 1, run_start, std::chrono::steady_clock::now());
        }
    }

    SchedulerCompletion completion;
    completion.request_id = entry.request_id;
    completion.sequence = entry.sequence;
    CompletionStatus status = CompletionStatus::kSucceeded;
    if (result.error.kind != InferenceErrorKind::kNone) {
        status = CompletionStatus::kInferenceFailed;
        completion.status = status;
        completion.inference_error = std::move(result.error);
    } else {
        completion.status = status;
        completion.outputs = std::move(result.outputs);
    }
    stamp_pre_publication(completion, entry.completion_timing);
    entry.completion.set_value(std::move(completion));
    entry.backstop = false;
    return status;
}

bool InferenceScheduler::Impl::PublishWorkerFailure(RequestEntry& entry, std::string_view text) {
    if (fail_worker_failure_publication_at_.load(std::memory_order_acquire) != kNoFault) {
        const std::size_t attempt =
            worker_failure_publication_attempt_.fetch_add(1, std::memory_order_acq_rel);
        std::size_t expected = attempt;
        if (fail_worker_failure_publication_at_.compare_exchange_strong(
                expected, kNoFault, std::memory_order_acq_rel)) {
            // Test-only pre-publication fault. The promise remains armed.
            throw std::bad_alloc();
        }
    }
    try {
        SchedulerCompletion completion;
        completion.request_id = entry.request_id;
        completion.sequence = entry.sequence;
        completion.status = CompletionStatus::kWorkerFailed;
        completion.inference_error.kind = InferenceErrorKind::kExecution;
        completion.inference_error.message = WorkerFailureMessage(text);
        stamp_pre_publication(completion, entry.completion_timing);
        entry.completion.set_value(std::move(completion));
    } catch (const std::future_error&) {
        // Already satisfied or moved-from. This is not a new completion.
        entry.backstop = false;
        return false;
    }
    entry.backstop = false;
    return true;
}

void InferenceScheduler::Impl::PublishCanceled(RequestEntry& entry) noexcept {
    try {
        SchedulerCompletion completion;
        completion.request_id = entry.request_id;
        completion.sequence = entry.sequence;
        completion.status = CompletionStatus::kCanceled;
        stamp_pre_publication(completion, entry.completion_timing);
        entry.completion.set_value(std::move(completion));
    } catch (const std::future_error&) {
        entry.backstop = false;
        return;
    } catch (...) {
        return;
    }
    entry.backstop = false;
    try {
        std::lock_guard<std::mutex> lock(admission_);
        ++canceled_;
    } catch (...) {
    }
}

void InferenceScheduler::Impl::PublishSchedulerFailure(RequestEntry& entry) noexcept {
    try {
        SchedulerCompletion completion;
        completion.request_id = entry.request_id;
        completion.sequence = entry.sequence;
        completion.status = CompletionStatus::kSchedulerFailed;
        completion.inference_error.kind = InferenceErrorKind::kExecution;
        completion.inference_error.message = "no worker";
        stamp_pre_publication(completion, entry.completion_timing);
        entry.completion.set_value(std::move(completion));
        entry.backstop = false;
    } catch (const std::future_error&) {
        entry.backstop = false;
        return;
    } catch (...) {
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(admission_);
        ++failed_scheduler_;
    } catch (...) {
    }
}

void InferenceScheduler::Impl::RecordLocked(CompletionStatus status) {
    switch (status) {
    case CompletionStatus::kSucceeded:
        ++succeeded_;
        break;
    case CompletionStatus::kInferenceFailed:
        ++failed_inference_;
        break;
    case CompletionStatus::kWorkerFailed:
        ++failed_worker_;
        break;
    case CompletionStatus::kSchedulerFailed:
        ++failed_scheduler_;
        break;
    case CompletionStatus::kCanceled:
        ++canceled_;
        break;
    }
}

void InferenceScheduler::Impl::Retire(std::uint32_t index, bool faulted) noexcept {
    if (index >= workers_.size() || workers_[index].retired) {
        return;
    }
    workers_[index].retired = true;
    const std::uint32_t previous = live_workers_.fetch_sub(1, std::memory_order_acq_rel);
    if (previous != 1) {
        return;
    }
    try {
        FinishLastWorker(faulted);
    } catch (...) {
        try {
            if (queue_ != nullptr && !queue_->is_closed()) {
                static_cast<void>(queue_->close_drain());
            }
        } catch (...) {
        }
    }
}

void InferenceScheduler::Impl::FinishLastWorker(bool faulted) {
    std::unique_lock<std::mutex> lock(admission_);
    if (cleanup_claimed_) {
        return;
    }
    if (!faulted) {
        if (queue_ == nullptr || !queue_->is_closed() || queue_->size() == 0) {
            return;
        }
        cleanup_claimed_ = true;
        state_ = State::kFailed;
        lock.unlock();
        static_cast<void>(PublishQueuedFailures());
        state_cv_.notify_all();
        return;
    }

    cleanup_claimed_ = true;
    if (queue_ == nullptr) {
        state_ = State::kFailed;
        lock.unlock();
        state_cv_.notify_all();
        return;
    }

    const bool already_closed = queue_->is_closed();
    // Shutdown may already have closed an empty queue. The worker exception was
    // published by the worker that owned the request. Do not turn that drain
    // into Failed when nothing is left waiting.
    if (already_closed && queue_->size() == 0) {
        lock.unlock();
        return;
    }
    if (already_closed || state_ != State::kRunning) {
        state_ = State::kFailed;
        lock.unlock();
        static_cast<void>(PublishQueuedFailures());
        state_cv_.notify_all();
        return;
    }

    std::unique_ptr<std::deque<RequestEntry>> canceled;
    bool fallback = false;
    try {
        // The test seam throws before close_cancel, matching an allocation
        // failure that happens before the queue commits its closure.
        if (fail_close_cancel_.exchange(false, std::memory_order_acq_rel)) {
            throw std::bad_alloc();
        }
        QueueClose<RequestEntry> closed = queue_->close_cancel();
        canceled = std::move(closed.canceled);
    } catch (const std::bad_alloc&) {
        fallback = true;
    }

    state_ = State::kFailed;
    if (fallback) {
        try {
            static_cast<void>(queue_->close_drain());
        } catch (...) {
            lock.unlock();
            state_cv_.notify_all();
            throw;
        }
        lock.unlock();
        static_cast<void>(PublishQueuedFailures());
        state_cv_.notify_all();
        return;
    }

    lock.unlock();
    state_cv_.notify_all();
    if (canceled != nullptr) {
        for (RequestEntry& entry : *canceled) {
            PublishSchedulerFailure(entry);
        }
        return;
    }
    static_cast<void>(PublishQueuedFailures());
}

std::uint64_t InferenceScheduler::Impl::PublishQueuedFailures() noexcept {
    if (queue_ == nullptr) {
        return 0;
    }
    try {
        if (!queue_->is_closed()) {
            static_cast<void>(queue_->close_drain());
        }
    } catch (...) {
        return 0;
    }
    try {
        if (!queue_->is_closed()) {
            return 0;
        }
    } catch (...) {
        return 0;
    }

    std::uint64_t count = 0;
    for (;;) {
        try {
            QueuePop<RequestEntry> popped = queue_->wait_pop();
            if (popped.result != QueuePopResult::kReady || !popped.value.has_value()) {
                break;
            }
            PublishSchedulerFailure(*popped.value);
            ++count;
        } catch (...) {
            break;
        }
    }
    return count;
}

void InferenceScheduler::Impl::JoinWorkers() noexcept {
    bool all_joined = true;
    for (WorkerSlot& worker : workers_) {
        try {
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
        } catch (...) {
            all_joined = false;
        }
    }
    // A failed join leaves that thread potentially live. Do not merge
    // worker-local counters, do not claim a finalized snapshot, and do
    // not destroy slots while a worker may still be executing.
    if (!all_joined) {
        return;
    }

    bool published = !config_.enable_metrics;
    if (config_.enable_metrics) {
        if (!metrics_merged_) {
            WorkerMetrics merged;
            for (const WorkerSlot& worker : workers_) {
                MergeWorkerMetrics(merged, worker.metrics);
            }
            auto publish = [this, &merged, &published] {
                std::lock_guard<std::mutex> lock(admission_);
                if (!metrics_merged_) {
                    published_metrics_ = merged;
                    metrics_merged_ = true;
                }
                published = metrics_merged_;
            };
            try {
                publish();
            } catch (...) {
                try {
                    publish();
                } catch (...) {
                }
            }
        } else {
            published = true;
        }
    }
    if (!published) {
        return;
    }
    try {
        workers_.clear();
    } catch (...) {
    }
}

void InferenceScheduler::Impl::EmergencyStop() noexcept {
    try {
        std::lock_guard<std::mutex> gate(gate_mu_);
        if (gate_ == StartupSignal::kPending) {
            gate_ = StartupSignal::kAbort;
        }
    } catch (...) {
    }
    gate_cv_.notify_all();
    try {
        if (queue_ != nullptr && !queue_->is_closed()) {
            static_cast<void>(queue_->close_drain());
        }
    } catch (...) {
    }
    bool perform_join = false;
    try {
        std::lock_guard<std::mutex> lock(admission_);
        if (!joined_) {
            join_claimed_ = true;
            perform_join = true;
        }
    } catch (...) {
    }
    if (perform_join) {
        JoinWorkers();
        static_cast<void>(PublishQueuedFailures());
    }
    try {
        std::lock_guard<std::mutex> lock(admission_);
        joined_ = true;
        if (state_ == State::kDraining || state_ == State::kCanceling ||
            state_ == State::kRunning || state_ == State::kStarting || state_ == State::kIdle) {
            state_ = State::kStopped;
        }
    } catch (...) {
    }
    state_cv_.notify_all();
}

AdmissionStatus InferenceScheduler::Impl::RejectionStatus() const {
    if (state_ == State::kIdle || state_ == State::kStarting) {
        return AdmissionStatus::kRejectedNotRunning;
    }
    return AdmissionStatus::kRejectedClosed;
}

Admission InferenceScheduler::Impl::Submit(std::vector<NamedTensor> inputs) {
    Admission rejected;
    // promise construction can throw. Keep the caller's tensors until the entry
    // and its future both exist.
    std::optional<RequestEntry> prepared;
    std::future<SchedulerCompletion> future;
    try {
        prepared.emplace();
        prepared->inputs = std::move(inputs);
        future = prepared->completion.get_future();
    } catch (const std::exception&) {
        rejected.status = AdmissionStatus::kRejectedNotRunning;
        if (prepared.has_value()) {
            rejected.inputs = std::move(prepared->inputs);
        } else {
            rejected.inputs = std::move(inputs);
        }
        return rejected;
    }
    RequestEntry entry = std::move(*prepared);
    prepared.reset();

    std::lock_guard<std::mutex> lock(admission_);
    if (state_ != State::kRunning) {
        future = std::future<SchedulerCompletion>{};
        rejected.status = RejectionStatus();
        if (rejected.status == AdmissionStatus::kRejectedClosed) {
            ++rejected_closed_;
        }
        rejected.inputs = std::move(entry.inputs);
        return rejected;
    }

    const std::uint64_t id = ++next_id_;
    entry.request_id = id;
    entry.sequence = id;
    ++accepted_;
    entry.backstop = true;
    entry.completion_timing = config_.enable_completion_timing;

    try {
        QueuePush<RequestEntry> pushed = queue_->try_push(std::move(entry));
        if (pushed.result != QueuePushResult::kAccepted) {
            --accepted_;
            if (pushed.result == QueuePushResult::kFull) {
                ++rejected_full_;
                rejected.status = AdmissionStatus::kRejectedFull;
            } else {
                ++rejected_closed_;
                rejected.status = AdmissionStatus::kRejectedClosed;
            }
            future = std::future<SchedulerCompletion>{};
            if (pushed.rejected.has_value()) {
                pushed.rejected->backstop = false;
                rejected.inputs = std::move(pushed.rejected->inputs);
            }
            return rejected;
        }
    } catch (const std::exception&) {
        // The queue argument's destructor publishes when the push throws before
        // the queue takes ownership. Hand that future back. Do not touch entry.
        ++failed_scheduler_;
        Admission accepted;
        accepted.status = AdmissionStatus::kAccepted;
        accepted.completion = std::move(future);
        return accepted;
    }

    // Test seam only. Production leaves this empty, so the admission mutex is
    // not held across InferenceEngine::run. The hook observes the future after
    // the worker has published and before that worker records its counter.
    if (on_accepted_) {
        on_accepted_(future);
    }
    Admission accepted;
    accepted.status = AdmissionStatus::kAccepted;
    accepted.completion = std::move(future);
    return accepted;
}

void InferenceScheduler::Impl::WaitForSettledShutdown(std::unique_lock<std::mutex>& lock) {
    std::shared_ptr<std::latch> entered = shutdown_follower_entered_;
    if (entered != nullptr) {
        lock.unlock();
        try {
            entered->count_down();
        } catch (...) {
        }
        if (!lock.owns_lock()) {
            lock.lock();
        }
    }
    state_cv_.wait(lock, [this] {
        return joined_ && (state_ == State::kStopped || state_ == State::kFailed);
    });
}

void InferenceScheduler::Impl::PublishJoinFinished(std::unique_lock<std::mutex>& lock,
                                                   std::uint64_t swept) {
    auto publish = [this, swept] {
        joined_ = true;
        if (swept > 0 && state_ != State::kFailed) {
            state_ = State::kFailed;
        } else if (state_ == State::kDraining || state_ == State::kCanceling) {
            state_ = State::kStopped;
        }
        state_cv_.notify_all();
    };
    auto attempt = [this, &lock, &publish] {
        if (!lock.owns_lock()) {
            lock.lock();
        }
        // One injected failure matches a mutex lock that throws after join.
        // The retry below still publishes joined_ so followers are not left
        // waiting. A second failure is not retried.
        if (fail_join_settle_.exchange(false, std::memory_order_acq_rel)) {
            lock.unlock();
            throw std::runtime_error("shutdown settle failed");
        }
        publish();
    };
    try {
        attempt();
    } catch (...) {
        try {
            attempt();
        } catch (...) {
            state_cv_.notify_all();
        }
    }
}

void InferenceScheduler::Impl::JoinOnce(std::unique_lock<std::mutex>& lock) {
    if (joined_) {
        return;
    }
    if (join_claimed_) {
        state_cv_.wait(lock, [this] { return joined_; });
        return;
    }
    join_claimed_ = true;
    lock.unlock();
    JoinWorkers();
    const std::uint64_t swept = PublishQueuedFailures();
    PublishJoinFinished(lock, swept);
}

ShutdownResult InferenceScheduler::Impl::SettledResult() const {
    ShutdownResult result;
    result.changed = false;
    result.established = policy_established_;
    if (policy_established_) {
        result.policy = user_policy_;
    }
    return result;
}

ShutdownResult InferenceScheduler::Impl::Shutdown(ShutdownPolicy policy) {
    ShutdownResult result;
    if (policy != ShutdownPolicy::kDrain && policy != ShutdownPolicy::kCancelPending) {
        result.error = MakeError(SchedulerErrorKind::kInvalidState, "unsupported shutdown policy");
        return result;
    }

    std::unique_lock<std::mutex> lock(admission_);
    if (state_ == State::kStarting) {
        shutdown_requested_ = true;
        state_cv_.wait(lock, [this] {
            return joined_ && (state_ == State::kFailed || state_ == State::kStopped);
        });
        result.changed = false;
        result.established = false;
        result.error =
            MakeError(SchedulerErrorKind::kShutdownDuringStart, "shutdown overlapped startup");
        return result;
    }
    if (state_ == State::kDraining || state_ == State::kCanceling) {
        WaitForSettledShutdown(lock);
        return SettledResult();
    }
    if (state_ == State::kStopped || state_ == State::kFailed) {
        JoinOnce(lock);
        return SettledResult();
    }
    if (state_ != State::kRunning) {
        result.error = MakeError(SchedulerErrorKind::kInvalidState, "scheduler is not running");
        return result;
    }
    if (coordinator_claimed_) {
        // The owner stores Draining or Canceling before it releases admission_.
        // Wait for that transition instead of waiting for join while Running.
        state_cv_.wait(lock, [this] { return !coordinator_claimed_ || state_ != State::kRunning; });
        if (state_ == State::kDraining || state_ == State::kCanceling) {
            WaitForSettledShutdown(lock);
            return SettledResult();
        }
        if (state_ == State::kStopped || state_ == State::kFailed) {
            JoinOnce(lock);
            return SettledResult();
        }
        result.error = MakeError(SchedulerErrorKind::kInvalidState, "scheduler is not running");
        return result;
    }

    coordinator_claimed_ = true;
    std::unique_ptr<std::deque<RequestEntry>> canceled_entries;
    try {
        if (policy == ShutdownPolicy::kCancelPending) {
            // Throws before close_cancel, matching an allocation failure that
            // happens before the queue commits its closure.
            if (fail_close_cancel_.exchange(false, std::memory_order_acq_rel)) {
                throw std::bad_alloc();
            }
            QueueClose<RequestEntry> closed = queue_->close_cancel();
            if (!closed.changed || closed.mode != QueueCloseMode::kCancel ||
                closed.canceled == nullptr) {
                throw std::runtime_error("shutdown close failed");
            }
            canceled_entries = std::move(closed.canceled);
            state_ = State::kCanceling;
        } else {
            QueueClose<RequestEntry> closed = queue_->close_drain();
            if (!closed.changed || closed.mode != QueueCloseMode::kDrain) {
                throw std::runtime_error("shutdown close failed");
            }
            state_ = State::kDraining;
        }
        user_policy_ = policy;
        policy_established_ = true;
        join_claimed_ = true;
    } catch (const std::exception&) {
        coordinator_claimed_ = false;
        result.error = MakeError(SchedulerErrorKind::kWorkerStart, "shutdown close failed");
        result.changed = false;
        result.established = false;
        return result;
    }

    std::shared_ptr<std::latch> after_close = after_drain_closed_;
    lock.unlock();

    // Publication and the test latch must not skip the join. PublishCanceled
    // already contains a failing set_value. The deque stays owned here, and
    // its destructor runs the backstop before this lock is released.
    try {
        if (policy == ShutdownPolicy::kCancelPending && canceled_entries != nullptr) {
            for (RequestEntry& entry : *canceled_entries) {
                PublishCanceled(entry);
            }
        }
        if (after_close != nullptr) {
            after_close->count_down();
        }
    } catch (...) {
    }

    JoinWorkers();
    const std::uint64_t swept = policy == ShutdownPolicy::kDrain ? PublishQueuedFailures() : 0;
    PublishJoinFinished(lock, swept);

    result.changed = true;
    result.established = policy_established_;
    result.policy = user_policy_;
    return result;
}

SchedulerMetrics InferenceScheduler::Impl::Metrics() const {
    SchedulerMetrics snapshot;
    snapshot.enabled = config_.enable_metrics;
    std::lock_guard<std::mutex> lock(admission_);
    if (state_ == State::kIdle) {
        snapshot.finalized = true;
        return snapshot;
    }
    if (!snapshot.enabled) {
        snapshot.finalized = joined_ || state_ == State::kStopped || state_ == State::kFailed;
        return snapshot;
    }
    if (metrics_merged_) {
        snapshot.finalized = true;
        snapshot.engine_run_count = published_metrics_.engine_run_count;
        snapshot.requests_entered_engine_runs = published_metrics_.requests_entered_engine_runs;
        snapshot.batch_size_histogram = published_metrics_.batch_size_histogram;
        snapshot.total_assembly_nanoseconds = published_metrics_.total_assembly_nanoseconds;
        snapshot.total_engine_run_nanoseconds = published_metrics_.total_engine_run_nanoseconds;
        snapshot.total_split_nanoseconds = published_metrics_.total_split_nanoseconds;
        return snapshot;
    }
    snapshot.finalized = false;
    return snapshot;
}

SchedulerStats InferenceScheduler::Impl::Stats() const {
    std::lock_guard<std::mutex> lock(admission_);
    SchedulerStats stats;
    stats.accepted = accepted_;
    stats.rejected_full = rejected_full_;
    stats.rejected_closed = rejected_closed_;
    stats.succeeded = succeeded_;
    stats.failed_inference = failed_inference_;
    stats.failed_worker = failed_worker_;
    stats.failed_scheduler = failed_scheduler_;
    stats.canceled = canceled_;
    stats.queue_depth = queue_ == nullptr ? 0 : static_cast<std::uint64_t>(queue_->size());
    stats.executing = executing_.load(std::memory_order_relaxed);
    stats.live_workers = live_workers_.load(std::memory_order_relaxed);
    return stats;
}

InferenceScheduler::InferenceScheduler(SchedulerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
}

InferenceScheduler::~InferenceScheduler() noexcept {
    impl_.reset();
}

SchedulerError InferenceScheduler::start() {
    return impl_->Start();
}

Admission InferenceScheduler::submit(std::vector<NamedTensor> inputs) {
    return impl_->Submit(std::move(inputs));
}

ShutdownResult InferenceScheduler::shutdown(ShutdownPolicy policy) {
    return impl_->Shutdown(policy);
}

SchedulerStats InferenceScheduler::stats() const {
    return impl_->Stats();
}

SchedulerMetrics InferenceScheduler::metrics() const {
    return impl_->Metrics();
}

void SchedulerTestAccess::SetStopAfterStartingWorkers(InferenceScheduler& scheduler,
                                                      std::uint32_t count) {
    scheduler.impl_->stop_after_ = count;
}

void SchedulerTestAccess::SetThrowInWorker(InferenceScheduler& scheduler, std::uint32_t index) {
    if (index >= scheduler.impl_->throw_armed_.size()) {
        return;
    }
    scheduler.impl_->throw_armed_[index].store(true, std::memory_order_release);
    scheduler.impl_->any_throw_armed_.store(true, std::memory_order_release);
}

void SchedulerTestAccess::SetFailNextBatchGuardCreation(InferenceScheduler& scheduler) {
    scheduler.impl_->fail_batch_guard_creation_.store(true, std::memory_order_release);
}

void SchedulerTestAccess::SetThrowAfterBatchCollect(InferenceScheduler& scheduler,
                                                    std::size_t request_count) {
    scheduler.impl_->throw_after_batch_collect_.store(request_count, std::memory_order_release);
}

void SchedulerTestAccess::SetThrowBeforeBatchPublish(InferenceScheduler& scheduler,
                                                     std::size_t request_index) {
    scheduler.impl_->throw_before_batch_publish_.store(request_index, std::memory_order_release);
}

void SchedulerTestAccess::SetFailWorkerFailurePublication(InferenceScheduler& scheduler,
                                                          std::size_t attempt_index) {
    scheduler.impl_->worker_failure_publication_attempt_.store(0, std::memory_order_release);
    scheduler.impl_->fail_worker_failure_publication_at_.store(attempt_index,
                                                               std::memory_order_release);
}

void SchedulerTestAccess::SetBeforePopForWorker(InferenceScheduler& scheduler, std::uint32_t index,
                                                std::shared_ptr<std::latch> latch) {
    if (index >= scheduler.impl_->worker_pop_gate_.size()) {
        return;
    }
    scheduler.impl_->worker_pop_gate_[index] = std::move(latch);
    scheduler.impl_->worker_pop_gate_armed_.store(true, std::memory_order_release);
}

void SchedulerTestAccess::SetBeforePop(InferenceScheduler& scheduler,
                                       std::shared_ptr<std::latch> latch) {
    scheduler.impl_->before_pop_ = std::move(latch);
}

void SchedulerTestAccess::SetBeforeLoop(InferenceScheduler& scheduler, std::function<void()> hook) {
    scheduler.impl_->before_loop_hook_ = std::move(hook);
}

void SchedulerTestAccess::SetReachedRun(InferenceScheduler& scheduler,
                                        std::shared_ptr<std::latch> latch) {
    scheduler.impl_->reached_run_ = std::move(latch);
}

void SchedulerTestAccess::SetBeforeRun(InferenceScheduler& scheduler,
                                       std::shared_ptr<std::latch> latch) {
    scheduler.impl_->before_run_ = std::move(latch);
}

void SchedulerTestAccess::SetOnBatchRun(InferenceScheduler& scheduler,
                                        std::function<void(std::size_t)> callback) {
    scheduler.impl_->on_batch_run_ = std::move(callback);
}

void SchedulerTestAccess::SetReachedBatchWait(InferenceScheduler& scheduler,
                                              std::shared_ptr<std::latch> latch) {
    scheduler.impl_->reached_batch_wait_ = std::move(latch);
}

void SchedulerTestAccess::SetBeforeBatchWait(InferenceScheduler& scheduler,
                                             std::shared_ptr<std::latch> latch) {
    scheduler.impl_->before_batch_wait_ = std::move(latch);
}

SchedulerError SchedulerTestAccess::CheckBatchMetadata(const std::vector<TensorMetadata>& inputs,
                                                       const std::vector<TensorMetadata>& outputs) {
    if (SchedulerError error = ValidateBatchSignature(inputs, "input");
        error.kind != SchedulerErrorKind::kNone) {
        return error;
    }
    return ValidateBatchSignature(outputs, "output");
}

void SchedulerTestAccess::SetAfterPublish(InferenceScheduler& scheduler,
                                          std::shared_ptr<std::latch> latch) {
    scheduler.impl_->after_publish_ = std::move(latch);
}

void SchedulerTestAccess::SetHoldOtherWorkers(InferenceScheduler& scheduler,
                                              std::shared_ptr<std::latch> latch) {
    scheduler.impl_->hold_other_workers_ = std::move(latch);
}

void SchedulerTestAccess::SetAfterDrainClosed(InferenceScheduler& scheduler,
                                              std::shared_ptr<std::latch> latch) {
    scheduler.impl_->after_drain_closed_ = std::move(latch);
}

void SchedulerTestAccess::SetShutdownFollowerEntered(InferenceScheduler& scheduler,
                                                     std::shared_ptr<std::latch> latch) {
    scheduler.impl_->shutdown_follower_entered_ = std::move(latch);
}

void SchedulerTestAccess::SetOnAccepted(
    InferenceScheduler& scheduler,
    std::function<void(std::future<SchedulerCompletion>&)> callback) {
    scheduler.impl_->on_accepted_ = std::move(callback);
}

void SchedulerTestAccess::SetFailNextCloseCancel(InferenceScheduler& scheduler) {
    scheduler.impl_->fail_close_cancel_.store(true, std::memory_order_release);
}

void SchedulerTestAccess::SetFailNextJoinSettle(InferenceScheduler& scheduler) {
    scheduler.impl_->fail_join_settle_.store(true, std::memory_order_release);
}

bool SchedulerTestAccess::IsStopped(const InferenceScheduler& scheduler) {
    std::lock_guard<std::mutex> lock(scheduler.impl_->admission_);
    return scheduler.impl_->state_ == State::kStopped;
}

bool SchedulerTestAccess::IsFailed(const InferenceScheduler& scheduler) {
    std::lock_guard<std::mutex> lock(scheduler.impl_->admission_);
    return scheduler.impl_->state_ == State::kFailed;
}

bool SchedulerTestAccess::ClosedForDrain(const InferenceScheduler& scheduler) {
    std::lock_guard<std::mutex> lock(scheduler.impl_->admission_);
    if (scheduler.impl_->queue_ == nullptr) {
        return false;
    }
    const std::optional<QueueCloseMode> mode = scheduler.impl_->queue_->close_mode();
    return mode.has_value() && *mode == QueueCloseMode::kDrain;
}

bool SchedulerTestAccess::ClosedForCancel(const InferenceScheduler& scheduler) {
    std::lock_guard<std::mutex> lock(scheduler.impl_->admission_);
    if (scheduler.impl_->queue_ == nullptr) {
        return false;
    }
    const std::optional<QueueCloseMode> mode = scheduler.impl_->queue_->close_mode();
    return mode.has_value() && *mode == QueueCloseMode::kCancel;
}

} // namespace inferedge
