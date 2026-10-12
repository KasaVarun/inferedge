#pragma once

#include "inferedge/benchmark_statistics.hpp"
#include "inferedge/concurrent_inference.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace inferedge {

// Warmup and measured counts use the same ceiling as kMaxBenchmarkIterations.
// worker_count uses kMaxConcurrentWorkers. The product of worker_count and
// measured_iterations is checked before any engine is loaded.

struct ConcurrentBenchmarkConfig {
    std::uint32_t worker_count{1};
    std::uint32_t warmup_iterations{10};
    std::uint32_t measured_iterations{100};
    std::uint32_t correctness_check_interval{1};
    std::string workload;
    std::string model;
    // Test seams. Leave them unset for a measured run.
    // stop_after_starting_workers, when set, must be less than worker_count.
    // After that many std::jthread objects exist, creation stops. Threads
    // already created are cancelled and joined. Threads that were never
    // created are not waited on.
    std::optional<std::uint32_t> stop_after_starting_workers;
    // throw_in_worker, when set, must be a worker index. That worker throws
    // before warmup. The thread function catches the exception.
    std::optional<std::uint32_t> throw_in_worker;
    // fail_warmup_worker, when set, must be a worker index. That worker fails
    // after preparation and before its first warmup call, then cancels the
    // start gate. This does not call InferenceEngine::run.
    std::optional<std::uint32_t> fail_warmup_worker;
    // Both must be set together. fail_measured_iteration is the 0-based
    // measured iteration that fails before its clock starts and before
    // InferenceEngine::run. The injection is outside the per-call sample.
    std::optional<std::uint32_t> fail_measured_worker;
    std::optional<std::uint32_t> fail_measured_iteration;
};

// One worker's inputs and the outputs every call on that worker must produce.
// inputs own their floats. Views built for the run borrow only this worker's
// buffers. The assignment must stay alive and unchanged for the whole call.
// Workers do not share these vectors.
struct ConcurrentBenchmarkAssignment {
    std::vector<NamedTensor> inputs;
    std::vector<WorkerExpectedTensor> expected;
};

enum class ConcurrentBenchmarkPhase {
    kNone,
    kWarmup,
    kMeasured,
};

enum class ConcurrentBenchmarkErrorKind {
    kNone,
    kInvalidConfig,
    kModelLoad,
    kWorkerStart,
    kInferenceFailed,
    kOutputMismatch,
    kNonFiniteOutput,
    kWorkerException,
    kInvalidDuration,
    kZeroElapsed,
    kStatistics,
};

// request_iteration is 0-based when phase is kWarmup or kMeasured.
// worker_index and request_iteration are meaningful for a worker failure.
// inference_kind is kNone unless the engine reported an inference error.
// statistics_kind is kNone unless kind is kStatistics.
struct ConcurrentBenchmarkError {
    ConcurrentBenchmarkErrorKind kind{ConcurrentBenchmarkErrorKind::kNone};
    ConcurrentBenchmarkPhase phase{ConcurrentBenchmarkPhase::kNone};
    std::uint32_t worker_index{0};
    std::uint32_t request_iteration{0};
    InferenceErrorKind inference_kind{InferenceErrorKind::kNone};
    StatisticsErrorKind statistics_kind{StatisticsErrorKind::kNone};
    std::string message;
};

// Present only when error.kind is kNone. Samples are one duration per
// successful measured run() on that worker. Output tensors are not retained.
// call_summary is summarize_latencies on the pooled per-request samples.
// Its active_call_throughput_requests_per_second stays the statistics
// definition: sample count divided by the sum of those samples. Overlapping
// calls can make that figure larger than aggregate_completed_requests_per_second.
// Neither figure is server capacity.
// aggregate_completed_requests_per_second is total_completed_requests divided
// by measurement_wall_duration in seconds. The wall interval is one shared
// steady_clock start, taken after warmup and before measured work, through
// the latest worker finish timestamp after that worker's last result is
// destroyed. It is not the sum of per-call durations.
struct ConcurrentBenchmarkReport {
    std::string workload;
    std::string model;
    std::string backend;
    std::uint32_t worker_count{0};
    std::uint32_t warmup_iterations{0};
    std::uint32_t measured_iterations{0};
    std::uint32_t correctness_check_interval{0};
    std::uint64_t total_completed_requests{0};
    std::vector<std::uint32_t> per_worker_completed_requests;
    std::vector<std::vector<std::chrono::nanoseconds>> per_worker_call_durations;
    LatencySummary call_summary;
    std::chrono::nanoseconds measurement_wall_duration{0};
    double aggregate_completed_requests_per_second{0.0};
};

// report is empty unless error.kind is kNone. load_error carries the engine
// error when error.kind is kModelLoad.
struct ConcurrentBenchmarkRun {
    ConcurrentBenchmarkError error;
    InferenceError load_error;
    ConcurrentBenchmarkReport report;
};

// Loads one InferenceEngine per assignment before any thread starts. Each
// engine uses the existing CPU session policy: one intra-op thread and graph
// optimizations disabled. A load failure returns kModelLoad and starts no
// threads.
//
// Each worker calls run() only on its own engine. The call joins every thread
// it creates. Worker thread functions catch exceptions and return normally.
// There is no queue, scheduler, or batch. This function does not call
// run_benchmark or run_concurrent_inference.
//
// assignments and config must stay unchanged until the function returns.
ConcurrentBenchmarkRun
run_concurrent_benchmark(const std::filesystem::path& model_path,
                         std::span<const ConcurrentBenchmarkAssignment> assignments,
                         const ConcurrentBenchmarkConfig& config);

} // namespace inferedge
