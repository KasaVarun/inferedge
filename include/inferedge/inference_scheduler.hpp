#pragma once

#include "inferedge/inference_engine.hpp"
#include "inferedge/scheduler_metrics.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <latch>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace inferedge {

// One scheduler loads one model. worker_count is 1 through kMaxSchedulerWorkers.
// It is not derived from hardware_concurrency. queue_capacity is the number of
// waiting requests, 1 through kMaxSchedulerQueueCapacity. An invalid
// configuration does not create worker threads.
inline constexpr std::uint32_t kMaxSchedulerWorkers = 64;
inline constexpr std::size_t kMaxSchedulerQueueCapacity = 1024;

// enable_batching selects worker-local dynamic batching. When it is false,
// each popped request is executed alone. When it is true, start() requires a
// dynamic batch axis, and a worker may run several validated requests in one
// InferenceEngine::run(). max_batch_size is 1 through 32. max_batch_delay is
// 0 through 1000 milliseconds. Both bounds are checked either way.
// adaptive_batching is an opt-in early-seal policy for the fill loop. The
// default is false and preserves fixed-delay waiting. When true, a worker may
// seal once it owns at least kAdaptiveMinTargetBatch validated requests and
// the queue is empty, without waiting out the remainder of max_batch_delay.
// It does not estimate arrival rate. enable_metrics collects worker-local
// engine-run counts, batch-size histogram bins, and assembly/run/split
// duration totals. The default is false: workers do not read a clock and do
// not update metric storage.
inline constexpr std::size_t kAdaptiveMinTargetBatch = 2;

struct SchedulerConfig {
    std::filesystem::path model_path;
    std::uint32_t worker_count{1};
    std::size_t queue_capacity{1};
    bool enable_batching{false};
    std::uint32_t max_batch_size{1};
    std::chrono::milliseconds max_batch_delay{0};
    // Opt-in. When false, RunBatch keeps fixed-delay fill behavior. When true,
    // start() requires enable_batching, max_batch_size >= kAdaptiveMinTargetBatch,
    // and max_batch_delay > 0.
    bool adaptive_batching{false};
    bool enable_metrics{false};
    // When true, each published completion carries one steady_clock time point
    // taken immediately before promise::set_value. The default reads no clock
    // on that path. This is not a post-publication timestamp.
    bool enable_completion_timing{false};
};

enum class SchedulerErrorKind {
    kNone,
    kInvalidConfig,
    kInvalidState,
    kModelLoad,
    kWorkerStart,
    kShutdownDuringStart,
};

// load_error is set for kModelLoad. Other kinds leave it at kNone.
struct SchedulerError {
    SchedulerErrorKind kind{SchedulerErrorKind::kNone};
    InferenceError load_error;
    std::string message;
};

// Terminal status of one accepted request. kCanceled is a waiting request
// extracted by shutdown(kCancelPending). It is not a per-request cancel.
enum class CompletionStatus {
    kSucceeded,
    kInferenceFailed,
    kWorkerFailed,
    kSchedulerFailed,
    kCanceled,
};

// outputs own their floats. inference_error is the engine error when status
// is kInferenceFailed. It is empty when status is kSucceeded or kCanceled.
struct SchedulerCompletion {
    std::uint64_t request_id{0};
    std::uint64_t sequence{0};
    CompletionStatus status{CompletionStatus::kSchedulerFailed};
    InferenceError inference_error;
    std::vector<NamedTensor> outputs;
    // Empty unless the scheduler was constructed with enable_completion_timing.
    // The value is the steady_clock reading taken immediately before the
    // set_value that stored this completion. It is not the future ready
    // transition and it is not a reading taken after set_value returns.
    std::optional<std::chrono::steady_clock::time_point> pre_publication_time;
};

enum class AdmissionStatus {
    kAccepted,
    kRejectedFull,
    kRejectedClosed,
    kRejectedNotRunning,
};

// On kAccepted, completion is engaged and inputs is empty. The future stays
// valid if a worker completes the request before submit returns. On every
// rejection, completion is empty and inputs owns the caller's tensors.
struct Admission {
    AdmissionStatus status{AdmissionStatus::kRejectedNotRunning};
    std::optional<std::future<SchedulerCompletion>> completion;
    std::optional<std::vector<NamedTensor>> inputs;
};

// kDrain finishes work already accepted. kCancelPending completes requests
// still waiting in the queue and leaves a request already popped by a worker
// with that worker. Neither policy interrupts InferenceEngine::run().
enum class ShutdownPolicy {
    kDrain,
    kCancelPending,
};

// established is true when a user shutdown committed. policy is the winner
// only then. changed is true only for the caller that committed that
// transition. An internal worker failure does not establish a user policy.
struct ShutdownResult {
    SchedulerError error;
    ShutdownPolicy policy{ShutdownPolicy::kDrain};
    bool changed{false};
    bool established{false};
};

// A concurrent snapshot. queue_depth, executing, and the counters are not one
// linearizable observation. After shutdown has joined the workers and no
// request is still in flight:
//   accepted == succeeded + failed_inference + failed_worker
//               + failed_scheduler + canceled
//   queue_depth == 0 && executing == 0 && live_workers == 0
// Rejected requests are not part of accepted. Do not require the equation
// while a request is queued, executing, or extracted but not yet published.
// If storing the completion throws and only the request destructor settles
// the promise, that exceptional completion is absent from the terminal
// counters, including canceled.
struct SchedulerStats {
    std::uint64_t accepted{0};
    std::uint64_t rejected_full{0};
    std::uint64_t rejected_closed{0};
    std::uint64_t succeeded{0};
    std::uint64_t failed_inference{0};
    std::uint64_t failed_worker{0};
    std::uint64_t failed_scheduler{0};
    std::uint64_t canceled{0};
    std::uint64_t queue_depth{0};
    std::uint64_t executing{0};
    std::uint64_t live_workers{0};
};

class SchedulerTestAccess;

// One-shot pool of independent inference workers. start() loads one
// InferenceEngine per worker. Those engines are not shared, and run() is never
// called on one engine from two threads.
//
// The lifecycle is Idle, Starting, Running, then Draining or Canceling, and
// then Stopped or Failed. start() works from Idle only. Stopped and Failed do
// not restart; create another scheduler. The destructor drains when no user
// shutdown has committed. It does not replace a committed cancel. It does not
// interrupt an ONNX Runtime call that has already entered run().
//
// Callers must not destroy a scheduler concurrently with start, submit,
// shutdown, stats, or metrics on that same object.
class InferenceScheduler {
public:
    explicit InferenceScheduler(SchedulerConfig config);
    InferenceScheduler(const InferenceScheduler&) = delete;
    InferenceScheduler& operator=(const InferenceScheduler&) = delete;
    InferenceScheduler(InferenceScheduler&&) = delete;
    InferenceScheduler& operator=(InferenceScheduler&&) = delete;
    ~InferenceScheduler() noexcept;

    [[nodiscard]] SchedulerError start();
    [[nodiscard]] Admission submit(std::vector<NamedTensor> inputs);
    [[nodiscard]] ShutdownResult shutdown(ShutdownPolicy policy);
    [[nodiscard]] SchedulerStats stats() const;
    // Copies the cumulative metrics snapshot. The copy does not mutate
    // counters. Meaningful totals require a finalized snapshot after every
    // worker thread has joined successfully, or before start() when no
    // worker exists. Failed or Stopped alone does not finalize enabled
    // metrics. A call while workers may still be running, or after a join
    // failure, returns enabled from the config and zeros with finalized
    // false. Disabled metrics return enabled false and zeros; they become
    // finalized once Idle or after a terminal state because they have no
    // worker-local data. See SchedulerMetrics.
    [[nodiscard]] SchedulerMetrics metrics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class SchedulerTestAccess;
};

// Test-only controls. Production callers leave them unset. Set the startup
// and worker hooks before start(), and the admission hook before submit().
// The admission hook runs on the submit thread while submit still owns the
// future. It must not call back into that scheduler.
class SchedulerTestAccess {
public:
    static void SetStopAfterStartingWorkers(InferenceScheduler& scheduler, std::uint32_t count);
    // Arms that worker to fail the next request it has already popped, once.
    // Calling it for several indices arms each worker independently. An unarmed
    // worker does not throw. Production runs leave every flag false.
    static void SetThrowInWorker(InferenceScheduler& scheduler, std::uint32_t index);
    // The next batching request throws std::bad_alloc immediately before its
    // ExecutingGuard allocation. The popped request is already worker-owned.
    static void SetFailNextBatchGuardCreation(InferenceScheduler& scheduler);
    // Throws once after this many valid requests have joined a worker-local
    // batch and before the worker performs another queue pop.
    static void SetThrowAfterBatchCollect(InferenceScheduler& scheduler, std::size_t request_count);
    // Throws once before publishing the zero-based request in a successfully
    // executed and split batch.
    static void SetThrowBeforeBatchPublish(InferenceScheduler& scheduler,
                                           std::size_t request_index);
    // Throws std::bad_alloc before set_value() on the zero-based worker-failure
    // publication attempt. This does not simulate std::promise internals.
    static void SetFailWorkerFailurePublication(InferenceScheduler& scheduler,
                                                std::size_t attempt_index);
    static void SetBeforePop(InferenceScheduler& scheduler, std::shared_ptr<std::latch> latch);
    // Holds one worker before wait_pop. Set it before start(). A null slot
    // does not wait. This is how a test chooses which worker pops.
    static void SetBeforePopForWorker(InferenceScheduler& scheduler, std::uint32_t index,
                                      std::shared_ptr<std::latch> latch);
    // Runs at the start of every worker loop, before wait_pop. Set it before
    // start(). It must not call back into the scheduler.
    static void SetBeforeLoop(InferenceScheduler& scheduler, std::function<void()> hook);
    static void SetReachedRun(InferenceScheduler& scheduler, std::shared_ptr<std::latch> latch);
    static void SetBeforeRun(InferenceScheduler& scheduler, std::shared_ptr<std::latch> latch);
    // Called on the worker thread immediately before a sealed batch calls
    // InferenceEngine::run, with the number of requests in that run. Set it
    // before start(). It must not call back into the scheduler. The unbatched
    // path does not call it.
    static void SetOnBatchRun(InferenceScheduler& scheduler,
                              std::function<void(std::size_t)> callback);
    // Counted down when a batching worker is about to block in wait_pop_until.
    // Set it before start().
    static void SetReachedBatchWait(InferenceScheduler& scheduler,
                                    std::shared_ptr<std::latch> latch);
    // Holds that worker before wait_pop_until. A null latch does not wait.
    static void SetBeforeBatchWait(InferenceScheduler& scheduler,
                                   std::shared_ptr<std::latch> latch);
    // Runs the startup batch-metadata check on caller-supplied signatures.
    static SchedulerError CheckBatchMetadata(const std::vector<TensorMetadata>& inputs,
                                             const std::vector<TensorMetadata>& outputs);
    static void SetAfterPublish(InferenceScheduler& scheduler, std::shared_ptr<std::latch> latch);
    static void SetHoldOtherWorkers(InferenceScheduler& scheduler,
                                    std::shared_ptr<std::latch> latch);
    static void SetAfterDrainClosed(InferenceScheduler& scheduler,
                                    std::shared_ptr<std::latch> latch);
    // Counts down when a shutdown caller begins waiting for an in-progress
    // user shutdown. The latch means the queue close has already committed.
    static void SetShutdownFollowerEntered(InferenceScheduler& scheduler,
                                           std::shared_ptr<std::latch> latch);
    static void SetOnAccepted(InferenceScheduler& scheduler,
                              std::function<void(std::future<SchedulerCompletion>&)> callback);
    static void SetFailNextCloseCancel(InferenceScheduler& scheduler);
    // The next publication of joined_ throws once, then retries.
    static void SetFailNextJoinSettle(InferenceScheduler& scheduler);
    static bool IsStopped(const InferenceScheduler& scheduler);
    static bool IsFailed(const InferenceScheduler& scheduler);
    static bool ClosedForDrain(const InferenceScheduler& scheduler);
    static bool ClosedForCancel(const InferenceScheduler& scheduler);
};

} // namespace inferedge
