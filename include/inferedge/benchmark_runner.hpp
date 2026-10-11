#pragma once

#include "inferedge/benchmark_statistics.hpp"
#include "inferedge/inference_engine.hpp"

#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace inferedge {

// Warmup and measured iteration counts above this value are rejected.
inline constexpr std::uint32_t kMaxBenchmarkIterations = 100000;

struct BenchmarkConfig {
    std::uint32_t warmup_iterations{10};
    std::uint32_t measured_iterations{100};
    std::uint32_t correctness_check_interval{1};
    std::string workload;
    std::string model;
};

// Reference values for one model output. Tolerances are finite and non-negative.
// Numerical comparison is
// abs(actual - expected) <= absolute_tolerance + relative_tolerance * abs(expected).
struct ExpectedTensor {
    std::string name;
    std::vector<std::int64_t> shape;
    std::vector<float> values;
    double relative_tolerance{0.0};
    double absolute_tolerance{0.0};
};

// call_durations and call_summary describe only successful measured run() calls.
// measurement_loop_duration also includes validation, result destruction, and
// loop bookkeeping between those calls. It excludes warmup, model loading,
// fixture parsing, summarize_latencies, and report formatting.
// observed_sequential_throughput_requests_per_second is measured_iterations
// divided by measurement_loop_duration in seconds. It is not server capacity.
// call_summary.active_call_throughput_requests_per_second is the statistics
// figure: completed calls divided by the sum of call_durations.
struct BenchmarkReport {
    std::string workload;
    std::string model;
    std::string backend;
    std::uint32_t warmup_iterations{0};
    std::uint32_t measured_iterations{0};
    std::uint32_t correctness_check_interval{0};
    std::vector<std::chrono::nanoseconds> call_durations;
    LatencySummary call_summary;
    std::chrono::nanoseconds measurement_loop_duration{0};
    double observed_sequential_throughput_requests_per_second{0.0};
};

enum class BenchmarkErrorKind {
    kNone,
    kInvalidConfig,
    kMissingExpectedOutput,
    kInvalidExpectedOutput,
    kInferenceFailed,
    kOutputMismatch,
    kNonFiniteOutput,
    kInvalidDuration,
    kZeroElapsed,
    kStatistics,
};

struct BenchmarkError {
    BenchmarkErrorKind kind{BenchmarkErrorKind::kNone};
    InferenceErrorKind inference_kind{InferenceErrorKind::kNone};
    StatisticsErrorKind statistics_kind{StatisticsErrorKind::kNone};
    std::string message;
};

// report is meaningful only when error.kind is kNone.
struct BenchmarkRun {
    BenchmarkError error;
    BenchmarkReport report;
};

// The caller keeps engine, the input buffers, and the views alive for the
// whole call. Expected tensors are read and not stored. Warmup runs first and
// is not timed. Each measured sample is steady_clock immediately before
// engine.run and immediately after it returns. The outer loop clock starts
// after warmup and stops after the last InferenceResult is destroyed.
// Value comparison uses 1-based iteration numbers, every
// correctness_check_interval iteration, and always the last iteration.
// A failure discards every sample and returns an empty report.
BenchmarkRun run_benchmark(InferenceEngine& engine, std::span<const NamedTensorView> inputs,
                           std::span<const ExpectedTensor> expected, const BenchmarkConfig& config);

} // namespace inferedge
