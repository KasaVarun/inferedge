#pragma once

#include "inferedge/inference_engine.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace inferedge {

// A correctness run uses at most this many workers and this many calls each.
inline constexpr std::uint32_t kMaxConcurrentWorkers = 64;
inline constexpr std::uint32_t kMaxConcurrentRequestsPerWorker = 1000;

// Owned reference output. Comparison is
// abs(actual - expected) <= absolute_tolerance + relative_tolerance * abs(expected).
// Tolerances are finite and non-negative. This is the same numeric contract as
// the synchronous benchmark's ExpectedTensor. The concurrent library does not
// link the benchmark runner.
struct WorkerExpectedTensor {
    std::string name;
    std::vector<std::int64_t> shape;
    std::vector<float> values;
    double relative_tolerance{0.0};
    double absolute_tolerance{0.0};
};

// One worker's inputs and the outputs those calls must produce.
// inputs own their floats. During a call, NamedTensorView objects borrow only
// that worker's buffers. The plan must stay alive and unchanged for the whole
// run_concurrent_inference call. Workers do not share these vectors.
struct WorkerRequest {
    std::vector<NamedTensor> inputs;
    std::vector<WorkerExpectedTensor> expected;
};

// Fixed worker count, fixed calls per worker, and owned per-worker data.
// There is no queue. requests_per_worker is the same for every worker.
//
// stop_after_starting_workers and throw_in_worker are deterministic test seams.
// Leave both unset for a normal run.
// stop_after_starting_workers, when set, must be less than the worker count.
// After that many std::jthread objects have been created, further creation is
// aborted. Threads already created are signalled to stop between calls and are
// joined. This does not detach and does not wait for threads that were never
// created.
// throw_in_worker, when set, must be a worker index. That worker throws before
// its first run(). The thread function catches the exception.
struct ConcurrentPlan {
    std::uint32_t requests_per_worker{1};
    std::vector<WorkerRequest> workers;
    std::optional<std::uint32_t> stop_after_starting_workers;
    std::optional<std::uint32_t> throw_in_worker;
};

enum class ConcurrentErrorKind {
    kNone,
    kInvalidPlan,
    kModelLoad,
    kWorkerStart,
    kInferenceFailed,
    kOutputMismatch,
    kNonFiniteOutput,
    kWorkerException,
};

// One worker-owned failure. request_iteration is the 0-based call that failed.
// A start failure uses the index of the worker that was not created and
// request_iteration 0. inference_kind is kNone unless kind is kInferenceFailed.
struct WorkerError {
    std::uint32_t worker_index{0};
    std::uint32_t request_iteration{0};
    ConcurrentErrorKind kind{ConcurrentErrorKind::kNone};
    InferenceErrorKind inference_kind{InferenceErrorKind::kNone};
    std::string message;
};

// Present only when the whole run succeeded. outputs are copies of the last
// call's tensors. They own their floats and stay valid after every engine is
// destroyed. completed_requests equals the plan's requests_per_worker.
struct WorkerOutput {
    std::uint32_t worker_index{0};
    std::uint32_t completed_requests{0};
    std::vector<NamedTensor> outputs;
};

// outputs is empty unless kind is kNone. A failed run has no partial success.
// load_error carries the engine's load error when kind is kModelLoad, and errors
// also records that load failure. errors holds every worker failure, and the
// start failure when creation stops early. On success, errors is empty.
struct ConcurrentExecution {
    ConcurrentErrorKind kind{ConcurrentErrorKind::kNone};
    InferenceError load_error;
    std::vector<WorkerError> errors;
    std::vector<WorkerOutput> outputs;
};

// Loads one InferenceEngine per worker from model_path before any thread
// starts. Each engine uses the existing CPU session policy. If any load fails,
// the returned kind is kModelLoad, load_error is that engine error, and no
// worker thread is started.
//
// Each worker thread calls run() only on its own engine. Concurrent run() on
// one InferenceEngine is still not supported. The call joins every thread it
// creates. Worker thread functions catch exceptions and return normally.
// There is no measured latency or throughput in this result.
ConcurrentExecution run_concurrent_inference(const std::filesystem::path& model_path,
                                             const ConcurrentPlan& plan);

// Moves engines into the call. engines.size() must equal plan.workers.size(),
// and every pointer must be non-null. A moved-from engine is non-null and
// fails on the first run() with kExecution. The same lifetime, joining, and
// failure rules as the path overload apply. Engines are destroyed before the
// function returns, after outputs have been copied into the result.
ConcurrentExecution run_concurrent_inference(std::vector<std::unique_ptr<InferenceEngine>> engines,
                                             const ConcurrentPlan& plan);

} // namespace inferedge
