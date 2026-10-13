#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace inferedge {

// Histogram length is 33 so index k is observed batch size k.
// Index 0 is unused and stays zero. Valid observed sizes are 1 through 32,
// matching SchedulerConfig::max_batch_size.
inline constexpr std::size_t kSchedulerMetricsHistogramSize = 33;
inline constexpr std::size_t kSchedulerMetricsMaxBatchSize = 32;

// Cumulative scheduler execution metrics. These are not SchedulerStats and do
// not measure worker utilization, queue wait, or producer end-to-end latency.
//
// enabled is a copy of SchedulerConfig::enable_metrics. When it is false the
// remaining counters stay zero and no clocks are read on the worker path.
//
// finalized is true when this snapshot is the post-join aggregate after
// every worker thread joined successfully, or when no worker was ever
// started. It is false while workers may still be running and after a join
// failure. Failed or Stopped alone does not finalize enabled metrics.
// Counters are meaningful only when finalized is true. A live call during
// Running, Draining, or Canceling returns zeros and must not be treated as a
// partial total.
//
// engine_run_count is the number of InferenceEngine::run invocations that
// were called. A run that returns an inference error is counted. A run that
// throws after the call is entered is counted. Invalid requests completed
// before an engine call, assembly failures, injected faults before run,
// queued cancellations, and rejected submissions are not.
//
// requests_entered_engine_runs is the number of requests in those invocations.
// For the unbatched path each invocation contributes 1. For the batched path
// it is the sealed, validated group size at run(). Configured max_batch_size
// is not recorded.
//
// batch_size_histogram[k] is the number of engine invocations whose group
// size was k. The unbatched path records size 1 and remains a distinct
// Execute path from batching with max_batch_size 1.
//
// After a finalized snapshot of entered calls, and before any counter
// saturates at UINT64_MAX:
//   sum(batch_size_histogram[1..32]) == engine_run_count
//   sum(k * batch_size_histogram[k]) == requests_entered_engine_runs
// Saturation is independent per field; those equalities may not hold after
// a field has reached UINT64_MAX.
//
// total_assembly_nanoseconds is the sum of steady_clock intervals around
// AssembleBatchInputs, including attempts that return an error or throw
// after the start timestamp. The unbatched path does not assemble and
// leaves this zero. A group that never starts assembly contributes nothing.
//
// total_engine_run_nanoseconds is the sum of steady_clock intervals around
// InferenceEngine::run, including validation, copies, and ONNX Runtime
// execution inside that call. Failed and throwing invocations are included
// when a start timestamp was taken. Queue wait, batch formation, assembly,
// split, and publication are excluded.
//
// total_split_nanoseconds is the sum of steady_clock intervals around
// SplitBatchOutputs, including attempts that return an error or throw after
// the start timestamp. The unbatched path does not split and leaves this
// zero. A group that never starts a split contributes nothing.
//
// Counters are cumulative for the scheduler lifetime. They do not reset on
// shutdown. Repeated shutdown or destruction does not merge worker locals
// twice. Unsigned totals saturate at UINT64_MAX. They do not wrap.
//
// metrics() copies this value type. It does not return references to worker
// storage. Callers must not destroy a scheduler concurrently with metrics().
struct SchedulerMetrics {
    bool enabled{false};
    bool finalized{false};
    std::uint64_t engine_run_count{0};
    std::uint64_t requests_entered_engine_runs{0};
    std::array<std::uint64_t, kSchedulerMetricsHistogramSize> batch_size_histogram{};
    std::uint64_t total_assembly_nanoseconds{0};
    std::uint64_t total_engine_run_nanoseconds{0};
    std::uint64_t total_split_nanoseconds{0};
};

} // namespace inferedge
