#include "inferedge/inference_scheduler.hpp"

#include "inferedge/bounded_queue.hpp"
#include "inferedge/concurrent_inference.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <new>
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

    RequestEntry() = default;
    RequestEntry(const RequestEntry&) = delete;
    RequestEntry& operator=(const RequestEntry&) = delete;
    RequestEntry(RequestEntry&& other) noexcept
        : inputs(std::move(other.inputs)), request_id(other.request_id), sequence(other.sequence),
          completion(std::move(other.completion)), backstop(std::exchange(other.backstop, false)) {}
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

enum class State {
    kIdle,
    kStarting,
    kRunning,
    kDraining,
    kStopped,
    kFailed,
};

enum class StartupSignal {
    kPending,
    kStart,
    kAbort,
};

struct WorkerSlot {
    std::unique_ptr<InferenceEngine> engine;
    std::jthread thread;
    bool retired{false};
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
    return {};
}

std::string WorkerFailureMessage(std::string_view text) {
    if (text.size() <= 15) {
        return std::string{text};
    }
    return "worker error";
}

struct ExecutingGuard {
    std::atomic<std::uint64_t>& executing;

    explicit ExecutingGuard(std::atomic<std::uint64_t>& value) : executing(value) {
        executing.fetch_add(1, std::memory_order_relaxed);
    }

    ~ExecutingGuard() { executing.fetch_sub(1, std::memory_order_relaxed); }
};

} // namespace

struct InferenceScheduler::Impl {
    explicit Impl(SchedulerConfig config) : config_(std::move(config)) {}
    ~Impl() noexcept;

    SchedulerError Start();
    Admission Submit(std::vector<NamedTensor> inputs);
    ShutdownResult Shutdown(ShutdownPolicy policy);
    SchedulerStats Stats() const;

    SchedulerError StartWorkers();
    SchedulerError AbortStart(SchedulerError error);
    bool WaitForStart();
    bool GateIsStart() const;
    void WorkerMain(std::uint32_t index, InferenceEngine* engine) noexcept;
    bool ConsumeThrow(std::uint32_t index);
    CompletionStatus Execute(InferenceEngine& engine, RequestEntry& entry);
    bool PublishWorkerFailure(RequestEntry& entry, std::string_view text);
    void PublishSchedulerFailure(RequestEntry& entry) noexcept;
    void RecordLocked(CompletionStatus status);
    void Retire(std::uint32_t index, bool faulted) noexcept;
    void FinishLastWorker(bool faulted);
    std::uint64_t PublishQueuedFailures() noexcept;
    void JoinWorkers() noexcept;
    void EmergencyStop() noexcept;
    AdmissionStatus RejectionStatus() const;

    SchedulerConfig config_;
    mutable std::mutex admission_;
    mutable std::mutex gate_mu_;
    std::condition_variable state_cv_;
    std::condition_variable gate_cv_;
    State state_{State::kIdle};
    StartupSignal gate_{StartupSignal::kPending};
    bool shutdown_requested_{false};
    bool joined_{false};
    bool cleanup_claimed_{false};

    std::atomic<bool> fail_close_cancel_{false};
    std::atomic<bool> throw_consumed_{false};
    std::optional<std::uint32_t> stop_after_;
    std::optional<std::uint32_t> throw_in_worker_;
    std::shared_ptr<std::latch> before_pop_;
    std::shared_ptr<std::latch> reached_run_;
    std::shared_ptr<std::latch> before_run_;
    std::shared_ptr<std::latch> after_publish_;
    std::shared_ptr<std::latch> hold_other_workers_;
    std::shared_ptr<std::latch> after_drain_closed_;
    std::function<void(std::future<SchedulerCompletion>&)> on_accepted_;

    std::uint64_t next_id_{0};
    std::uint64_t accepted_{0};
    std::uint64_t rejected_full_{0};
    std::uint64_t rejected_closed_{0};
    std::uint64_t succeeded_{0};
    std::uint64_t failed_inference_{0};
    std::uint64_t failed_worker_{0};
    std::uint64_t failed_scheduler_{0};
    std::atomic<std::uint64_t> executing_{0};
    std::atomic<std::uint32_t> live_workers_{0};

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
            if (const std::shared_ptr<std::latch> hold = hold_other_workers_;
                hold != nullptr && index > 0) {
                hold->wait();
            }
            if (const std::shared_ptr<std::latch> before_pop = before_pop_; before_pop != nullptr) {
                before_pop->wait();
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
                if (ConsumeThrow(index)) {
                    retire_now = true;
                    throw std::runtime_error("injected");
                }
                if (const std::shared_ptr<std::latch> reached = reached_run_; reached != nullptr) {
                    reached->count_down();
                }
                if (const std::shared_ptr<std::latch> before_run = before_run_;
                    before_run != nullptr) {
                    before_run->wait();
                }
                status = Execute(*engine, entry);
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
    if (!throw_in_worker_.has_value() || *throw_in_worker_ != index) {
        return false;
    }
    bool expected = false;
    return throw_consumed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
}

CompletionStatus InferenceScheduler::Impl::Execute(InferenceEngine& engine, RequestEntry& entry) {
    InferenceResult result;
    {
        // Views borrow entry.inputs and die before entry is destroyed.
        std::vector<NamedTensorView> views;
        views.reserve(entry.inputs.size());
        for (const NamedTensor& input : entry.inputs) {
            views.push_back(
                NamedTensorView{input.name, input.shape, std::span<const float>{input.values}});
        }
        result = engine.run(views);
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
    entry.completion.set_value(std::move(completion));
    entry.backstop = false;
    return status;
}

bool InferenceScheduler::Impl::PublishWorkerFailure(RequestEntry& entry, std::string_view text) {
    try {
        SchedulerCompletion completion;
        completion.request_id = entry.request_id;
        completion.sequence = entry.sequence;
        completion.status = CompletionStatus::kWorkerFailed;
        completion.inference_error.kind = InferenceErrorKind::kExecution;
        completion.inference_error.message = WorkerFailureMessage(text);
        entry.completion.set_value(std::move(completion));
    } catch (const std::future_error&) {
        // Already satisfied or moved-from. This is not a new completion.
        entry.backstop = false;
        return false;
    }
    entry.backstop = false;
    return true;
}

void InferenceScheduler::Impl::PublishSchedulerFailure(RequestEntry& entry) noexcept {
    try {
        SchedulerCompletion completion;
        completion.request_id = entry.request_id;
        completion.sequence = entry.sequence;
        completion.status = CompletionStatus::kSchedulerFailed;
        completion.inference_error.kind = InferenceErrorKind::kExecution;
        completion.inference_error.message = "no worker";
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
    for (WorkerSlot& worker : workers_) {
        try {
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
        } catch (...) {
        }
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
    JoinWorkers();
    static_cast<void>(PublishQueuedFailures());
    try {
        std::lock_guard<std::mutex> lock(admission_);
        joined_ = true;
        if (state_ == State::kDraining || state_ == State::kRunning || state_ == State::kStarting ||
            state_ == State::kIdle) {
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

ShutdownResult InferenceScheduler::Impl::Shutdown(ShutdownPolicy policy) {
    ShutdownResult result;
    result.policy = policy;
    if (policy != ShutdownPolicy::kDrain) {
        result.error = MakeError(SchedulerErrorKind::kInvalidState, "unsupported shutdown policy");
        return result;
    }

    std::unique_lock<std::mutex> lock(admission_);
    if (state_ == State::kStarting) {
        shutdown_requested_ = true;
        state_cv_.wait(lock,
                       [this] { return state_ == State::kFailed || state_ == State::kStopped; });
        result.changed = false;
        result.error =
            MakeError(SchedulerErrorKind::kShutdownDuringStart, "shutdown overlapped startup");
        return result;
    }
    if (state_ == State::kDraining) {
        state_cv_.wait(lock,
                       [this] { return state_ == State::kStopped || state_ == State::kFailed; });
        result.changed = false;
        return result;
    }
    if (state_ == State::kStopped || state_ == State::kFailed) {
        if (!joined_) {
            lock.unlock();
            JoinWorkers();
            static_cast<void>(PublishQueuedFailures());
            lock.lock();
            joined_ = true;
        }
        result.changed = false;
        return result;
    }
    if (state_ != State::kRunning) {
        result.error = MakeError(SchedulerErrorKind::kInvalidState, "scheduler is not running");
        return result;
    }

    bool drain_started = false;
    try {
        static_cast<void>(queue_->close_drain());
        state_ = State::kDraining;
        drain_started = true;
    } catch (const std::exception& ex) {
        result.error = MakeError(SchedulerErrorKind::kWorkerStart, ex.what());
    }
    std::shared_ptr<std::latch> after_drain = after_drain_closed_;
    lock.unlock();

    if (!drain_started && queue_ != nullptr) {
        try {
            if (!queue_->is_closed()) {
                static_cast<void>(queue_->close_drain());
            }
            drain_started = queue_->is_closed();
        } catch (...) {
        }
    }
    if (drain_started && after_drain != nullptr) {
        after_drain->count_down();
    }

    JoinWorkers();
    const std::uint64_t swept = PublishQueuedFailures();
    lock.lock();
    joined_ = true;
    if (swept > 0) {
        state_ = State::kFailed;
    } else if (state_ == State::kDraining) {
        state_ = State::kStopped;
    }
    state_cv_.notify_all();
    result.changed = drain_started && result.error.kind == SchedulerErrorKind::kNone;
    return result;
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

void SchedulerTestAccess::SetStopAfterStartingWorkers(InferenceScheduler& scheduler,
                                                      std::uint32_t count) {
    scheduler.impl_->stop_after_ = count;
}

void SchedulerTestAccess::SetThrowInWorker(InferenceScheduler& scheduler, std::uint32_t index) {
    scheduler.impl_->throw_in_worker_ = index;
}

void SchedulerTestAccess::SetBeforePop(InferenceScheduler& scheduler,
                                       std::shared_ptr<std::latch> latch) {
    scheduler.impl_->before_pop_ = std::move(latch);
}

void SchedulerTestAccess::SetReachedRun(InferenceScheduler& scheduler,
                                        std::shared_ptr<std::latch> latch) {
    scheduler.impl_->reached_run_ = std::move(latch);
}

void SchedulerTestAccess::SetBeforeRun(InferenceScheduler& scheduler,
                                       std::shared_ptr<std::latch> latch) {
    scheduler.impl_->before_run_ = std::move(latch);
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

void SchedulerTestAccess::SetOnAccepted(
    InferenceScheduler& scheduler,
    std::function<void(std::future<SchedulerCompletion>&)> callback) {
    scheduler.impl_->on_accepted_ = std::move(callback);
}

void SchedulerTestAccess::SetFailNextCloseCancel(InferenceScheduler& scheduler) {
    scheduler.impl_->fail_close_cancel_.store(true, std::memory_order_release);
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
