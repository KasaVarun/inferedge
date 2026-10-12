#pragma once

#include "inferedge/inference_engine.hpp"

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

struct SchedulerConfig {
    std::filesystem::path model_path;
    std::uint32_t worker_count{1};
    std::size_t queue_capacity{1};
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
// shutdown, or stats on that same object.
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
    static void SetThrowInWorker(InferenceScheduler& scheduler, std::uint32_t index);
    static void SetBeforePop(InferenceScheduler& scheduler, std::shared_ptr<std::latch> latch);
    // Runs at the start of every worker loop, before wait_pop. Set it before
    // start(). It must not call back into the scheduler.
    static void SetBeforeLoop(InferenceScheduler& scheduler, std::function<void()> hook);
    static void SetReachedRun(InferenceScheduler& scheduler, std::shared_ptr<std::latch> latch);
    static void SetBeforeRun(InferenceScheduler& scheduler, std::shared_ptr<std::latch> latch);
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
