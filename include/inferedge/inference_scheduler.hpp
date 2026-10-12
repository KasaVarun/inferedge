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

// Terminal status of one accepted request. A canceled-pending status is not
// part of this scheduler.
enum class CompletionStatus {
    kSucceeded,
    kInferenceFailed,
    kWorkerFailed,
    kSchedulerFailed,
};

// outputs own their floats. inference_error is the engine error when status
// is kInferenceFailed, and empty when status is kSucceeded.
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

// kDrain finishes work already accepted. Cancel-pending shutdown is not
// implemented.
enum class ShutdownPolicy {
    kDrain,
};

// changed is true only for the call that moves a running scheduler into drain.
struct ShutdownResult {
    SchedulerError error;
    ShutdownPolicy policy{ShutdownPolicy::kDrain};
    bool changed{false};
};

// A concurrent snapshot. queue_depth, executing, and the counters are not one
// linearizable observation. After shutdown has joined the workers and no
// request is still in flight:
//   accepted == succeeded + failed_inference + failed_worker + failed_scheduler
//   queue_depth == 0 && executing == 0 && live_workers == 0
// Rejected requests are not part of accepted. Do not require the equation
// while a request is queued or executing. If storing the completion throws and
// only the request destructor settles the promise, that exceptional completion
// is absent from the terminal counters.
struct SchedulerStats {
    std::uint64_t accepted{0};
    std::uint64_t rejected_full{0};
    std::uint64_t rejected_closed{0};
    std::uint64_t succeeded{0};
    std::uint64_t failed_inference{0};
    std::uint64_t failed_worker{0};
    std::uint64_t failed_scheduler{0};
    std::uint64_t queue_depth{0};
    std::uint64_t executing{0};
    std::uint64_t live_workers{0};
};

class SchedulerTestAccess;

// One-shot pool of independent inference workers. start() loads one
// InferenceEngine per worker. Those engines are not shared, and run() is never
// called on one engine from two threads.
//
// The lifecycle is Idle, Starting, Running, Draining, and then Stopped or
// Failed. start() works from Idle only. Stopped and Failed do not restart;
// create another scheduler. The destructor drains accepted work and joins.
// It does not interrupt an ONNX Runtime call that has already entered run().
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
    static void SetReachedRun(InferenceScheduler& scheduler, std::shared_ptr<std::latch> latch);
    static void SetBeforeRun(InferenceScheduler& scheduler, std::shared_ptr<std::latch> latch);
    static void SetAfterPublish(InferenceScheduler& scheduler, std::shared_ptr<std::latch> latch);
    static void SetHoldOtherWorkers(InferenceScheduler& scheduler,
                                    std::shared_ptr<std::latch> latch);
    static void SetAfterDrainClosed(InferenceScheduler& scheduler,
                                    std::shared_ptr<std::latch> latch);
    static void SetOnAccepted(InferenceScheduler& scheduler,
                              std::function<void(std::future<SchedulerCompletion>&)> callback);
    static void SetFailNextCloseCancel(InferenceScheduler& scheduler);
    static bool IsStopped(const InferenceScheduler& scheduler);
    static bool IsFailed(const InferenceScheduler& scheduler);
    static bool ClosedForDrain(const InferenceScheduler& scheduler);
    static bool ClosedForCancel(const InferenceScheduler& scheduler);
};

} // namespace inferedge
