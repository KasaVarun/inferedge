#pragma once

#include "inferedge/benchmark_statistics.hpp"
#include "inferedge/inference_scheduler.hpp"
#include "inferedge/scheduler_metrics.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace inferedge {

// Caps keep one trial from growing without a bound. They are runner limits.
// SchedulerConfig still enforces its own worker, queue, and batch bounds.
inline constexpr std::uint64_t kMaxSchedulerBenchmarkOffered = 100000;
inline constexpr std::uint32_t kMaxSchedulerBenchmarkDurationMs = 60000;
inline constexpr std::uint32_t kMaxSchedulerBenchmarkProducers = 64;
inline constexpr double kMaxSchedulerBenchmarkRequestsPerSecond = 1000000.0;
inline constexpr std::size_t kSchedulerBenchmarkMaxErrorRecords = 8;
// Open-loop collector resolution. Closed-loop producers block in wait_for
// and do not use this quantum. The value is disclosure, not a latency target.
inline constexpr std::chrono::nanoseconds kSchedulerBenchmarkObservationQuantum{1000000};

// One trial is one workload. Do not compare a closed-loop sample with an
// open-loop sample as if they shared an arrival process.
enum class SchedulerWorkloadKind {
    kOpenLoopConstant,
    kOpenLoopBurst,
    kClosedLoop,
};

// Open-loop catch-up is stop-and-record. The generator sleeps until the
// precomputed timestamp. generator_lag is actual_submit_start minus
// scheduled_arrival, signed. A negative lag is early and does not by itself
// invalidate the trial. A positive lag above lag_tolerance does not submit
// that opportunity or any later one. Those opportunities increment
// generator_skips_lag. The trial is invalid. The schedule is not rewritten
// and overdue arrivals are not compressed into a submitted burst.
//
// The outstanding cap is skip-opportunity. A scheduled opportunity that finds
// the cap full is not passed to submit(). It increments
// generator_skips_outstanding. It is not a scheduler rejection.
//
// Warm-up, when warmup_requests is positive, runs on a separate
// InferenceScheduler that is drained and destroyed before the measured
// scheduler is constructed. Measured SchedulerMetrics are not a subtraction.
//
// hold_collector, test_submit_delay, test_fail_checked_successes, and
// test_submit_after_shutdown are test seams. The executable leaves them
// unset. Any seam marks the trial invalid.
//
// completion_timeout bounds future observation only. shutdown(kDrain) can
// still block inside an InferenceEngine::run that has already started. This
// process has no external hard timeout.
//
// An open-loop outstanding skip is not a scheduler rejection. If those skips
// are at least half of the scheduled opportunities, the trial is invalid:
// the run did not offer the declared schedule. A collector that falls behind
// keeps accepted futures in the cap and causes those skips.
struct SchedulerBenchmarkConfig {
    std::string model_path;
    std::uint32_t worker_count{1};
    std::size_t queue_capacity{1};
    bool enable_batching{false};
    std::uint32_t max_batch_size{1};
    std::chrono::milliseconds max_batch_delay{0};
    bool enable_metrics{true};
    SchedulerWorkloadKind workload{SchedulerWorkloadKind::kOpenLoopConstant};
    double target_requests_per_second{0.0};
    double burst_high_requests_per_second{0.0};
    double burst_low_requests_per_second{0.0};
    std::chrono::milliseconds burst_high{0};
    std::chrono::milliseconds burst_low{0};
    std::uint32_t producer_count{1};
    std::uint64_t max_offered{0};
    std::uint64_t max_outstanding{0};
    std::uint64_t warmup_requests{0};
    std::chrono::milliseconds duration{0};
    std::chrono::milliseconds completion_timeout{0};
    std::chrono::nanoseconds lag_tolerance{0};
    std::uint32_t check_interval{1};
    bool hold_collector{false};
    std::chrono::nanoseconds test_submit_delay{0};
    std::uint32_t test_fail_checked_successes{0};
    // After drain shutdown, submit one more request. It must come back
    // kRejectedClosed with its inputs. The executable leaves this false.
    bool test_submit_after_shutdown{false};
};

struct SchedulerBenchmarkArrival {
    std::chrono::nanoseconds offset{0};
    std::uint32_t phase_index{0};
};

// One alternating burst window, or the single window of a constant plan.
// Arrivals are spaced by that phase's period and stay inside the phase.
// The next phase starts on its own boundary instead of carrying a remainder.
struct SchedulerBenchmarkPhase {
    std::uint32_t index{0};
    double requests_per_second{0.0};
    std::chrono::nanoseconds duration{0};
    std::uint64_t scheduled_opportunities{0};
};

struct SchedulerBenchmarkPlan {
    std::vector<SchedulerBenchmarkArrival> arrivals;
    std::vector<SchedulerBenchmarkPhase> phases;
};

enum class SchedulerBenchmarkPlanError {
    kNone,
    kInvalidRate,
    kInvalidDuration,
    kInvalidBound,
};

struct SchedulerBenchmarkPlanResult {
    SchedulerBenchmarkPlanError error{SchedulerBenchmarkPlanError::kNone};
    SchedulerBenchmarkPlan plan;
};

// lag is actual minus scheduled. exceeds_tolerance is true only when lag is
// greater than tolerance. Early (negative) lag does not exceed a
// non-negative tolerance.
struct SchedulerBenchmarkLag {
    std::chrono::nanoseconds lag{0};
    bool exceeds_tolerance{false};
};

struct SchedulerBenchmarkTensors {
    std::vector<NamedTensor> inputs;
    float expected_c0{0.F};
    float expected_c1{0.F};
};

enum class SchedulerBenchmarkErrorKind {
    kNone,
    kInvalidConfig,
    kModelLoad,
    kRuntime,
};

struct SchedulerBenchmarkRejections {
    std::uint64_t full{0};
    std::uint64_t closed{0};
    std::uint64_t not_running{0};
};

struct SchedulerBenchmarkLagSummary {
    std::uint64_t count{0};
    std::chrono::nanoseconds minimum{0};
    std::chrono::nanoseconds maximum{0};
    double mean_nanoseconds{0.0};
};

// Percentiles come from summarize_latencies. percentiles_reported is false
// when there are no successful samples. precise_tail is false below 100
// samples: the figures are still the nearest-rank statistics, not a claim
// that the tail is stable.
struct SchedulerBenchmarkLatency {
    std::uint64_t sample_count{0};
    std::chrono::nanoseconds minimum{0};
    std::chrono::nanoseconds maximum{0};
    double mean_microseconds{0.0};
    double p50_microseconds{0.0};
    double p95_microseconds{0.0};
    double p99_microseconds{0.0};
    bool percentiles_reported{false};
    bool precise_tail{false};
};

// offered/attempt/accepted/successful_per_arrival_window_second divide by the
// measured arrival window. That window starts when the measured generator is
// released and stops when it makes its last scheduling decision. It excludes
// warm-up and the post-arrival drain.
// successful_per_total_trial_second divides the same successes by the arrival
// window plus the completion drain, including shutdown(kDrain). These two
// successful rates are not interchangeable. The arrival-window rate is an
// offered-cohort rate, not steady-state completion capacity.
struct SchedulerBenchmarkThroughput {
    double arrival_window_seconds{0.0};
    double completion_seconds{0.0};
    double total_trial_seconds{0.0};
    double offered_per_arrival_window_second{0.0};
    double attempts_per_arrival_window_second{0.0};
    double accepted_per_arrival_window_second{0.0};
    double successful_per_arrival_window_second{0.0};
    double successful_per_total_trial_second{0.0};
};

struct SchedulerBenchmarkErrorRecord {
    std::uint64_t attempt_id{0};
    std::string kind;
    std::string message;
};

// report is meaningful when error.kind is kNone. valid can still be false:
// the trial ran, the JSON document is complete, and the methodology failed.
// Primary end-to-end latency starts at steady_clock immediately before
// submit() and stops at steady_clock when this runner observes the future
// ready. It includes admission, queue time, batch wait, inference,
// publication, and observation delay. It is not the promise set_value time.
// Rejected requests are absent from the latency samples.
//
// completion_observation is "open_loop_poll" or "closed_loop_producer_wait".
// Open-loop poll uses kSchedulerBenchmarkObservationQuantum. One collector
// thread scans ready futures with wait_for(0). It does not block in get()
// on an earlier request while a later one is ready. Closed-loop producers
// each wait on their own future. There is not one thread per request.
struct SchedulerBenchmarkReport {
    bool valid{false};
    std::vector<std::string> invalid_reasons;
    std::string model;
    std::string workload;
    std::uint32_t worker_count{0};
    std::uint64_t queue_capacity{0};
    bool enable_batching{false};
    std::uint32_t max_batch_size{0};
    std::chrono::milliseconds max_batch_delay{0};
    bool enable_metrics{false};
    std::uint64_t warmup_requests{0};
    bool warmup_uses_separate_scheduler{true};
    std::chrono::nanoseconds configured_duration{0};
    // Precomputed last open-loop offset. Closed-loop has no arrival schedule
    // and leaves this at zero. It is not the measured arrival window.
    std::chrono::nanoseconds scheduled_span{0};
    // max_offered, configured_duration, or generator_lag. Rates divide by the
    // measured arrival window, not by configured_duration.
    std::string arrival_stop_reason;
    std::chrono::nanoseconds arrival_window{0};
    std::chrono::nanoseconds completion{0};
    std::chrono::nanoseconds total_trial{0};
    std::uint64_t scheduled_opportunities{0};
    std::uint64_t submit_attempts{0};
    std::uint64_t generator_skips_outstanding{0};
    std::uint64_t generator_skips_lag{0};
    std::uint64_t accepted{0};
    SchedulerBenchmarkRejections rejections;
    std::uint64_t succeeded{0};
    std::uint64_t failed_inference{0};
    std::uint64_t failed_worker{0};
    std::uint64_t failed_scheduler{0};
    std::uint64_t canceled{0};
    std::uint64_t timeouts{0};
    SchedulerBenchmarkLagSummary generator_lag;
    SchedulerBenchmarkLatency latency;
    std::string completion_observation;
    std::chrono::nanoseconds observation_quantum{0};
    SchedulerBenchmarkThroughput throughput;
    SchedulerMetrics metrics;
    bool histogram_saturated{false};
    std::uint64_t correctness_checked{0};
    std::uint64_t correctness_mismatches{0};
    std::uint32_t check_interval{1};
    std::vector<SchedulerBenchmarkErrorRecord> errors;
    std::uint64_t error_count{0};
    std::vector<SchedulerBenchmarkPhase> phases;
    std::uint64_t queue_depth{0};
    std::uint64_t executing{0};
    std::uint64_t live_workers{0};
};

struct SchedulerBenchmarkRun {
    SchedulerBenchmarkErrorKind kind{SchedulerBenchmarkErrorKind::kNone};
    InferenceError load_error;
    std::string message;
    SchedulerBenchmarkReport report;
};

[[nodiscard]] const char* scheduler_workload_name(SchedulerWorkloadKind workload) noexcept;

[[nodiscard]] SchedulerBenchmarkPlanResult plan_constant_arrivals(double requests_per_second,
                                                                  std::chrono::nanoseconds duration,
                                                                  std::uint64_t max_offered);

[[nodiscard]] SchedulerBenchmarkPlanResult
plan_burst_arrivals(double high_requests_per_second, double low_requests_per_second,
                    std::chrono::nanoseconds high, std::chrono::nanoseconds low,
                    std::chrono::nanoseconds duration, std::uint64_t max_offered);

[[nodiscard]] SchedulerBenchmarkLag
measure_generator_lag(std::chrono::steady_clock::time_point scheduled,
                      std::chrono::steady_clock::time_point actual,
                      std::chrono::nanoseconds tolerance) noexcept;

// attempt_id selects deterministic float32[1, 2] inputs for batch_add_f32.
// Expected C is the exact integer sum of those inputs.
[[nodiscard]] SchedulerBenchmarkTensors scheduler_benchmark_request(std::uint64_t attempt_id);

// True when this success is checked. one_based_index starts at 1. The last
// success is checked even when it is off the interval.
[[nodiscard]] bool scheduler_benchmark_should_check(std::uint64_t one_based_index,
                                                    std::uint32_t check_interval,
                                                    bool is_last) noexcept;

// Does not block. wait_for(0) is the collector's readiness probe.
[[nodiscard]] bool scheduler_benchmark_future_ready(std::future<SchedulerCompletion>& completion);

// 0 when the trial is valid. 1 when the run failed or the trial is invalid.
[[nodiscard]] int scheduler_benchmark_status(const SchedulerBenchmarkRun& run) noexcept;

// Loads no model until the configuration is accepted. Uses shutdown(kDrain)
// on every scheduler it starts. Does not call SchedulerTestAccess.
[[nodiscard]] SchedulerBenchmarkRun run_scheduler_benchmark(const SchedulerBenchmarkConfig& config);

} // namespace inferedge
