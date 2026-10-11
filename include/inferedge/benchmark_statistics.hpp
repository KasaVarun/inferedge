#pragma once

#include <chrono>
#include <cstdint>
#include <span>
#include <string>

namespace inferedge {

// Durations are std::chrono::nanoseconds. Mean, percentiles, and the population
// standard deviation are microseconds, including fractions of a microsecond.
// active_call_throughput_requests_per_second is completed samples divided by
// the sum of those durations. It counts only the supplied call intervals. It
// is not observed server throughput: model loading, checks, destruction,
// loop overhead, queues, and transport are absent from the samples.
struct LatencySummary {
    std::uint64_t sample_count{0};
    std::chrono::nanoseconds minimum{0};
    std::chrono::nanoseconds maximum{0};
    double mean_microseconds{0.0};
    double p50_microseconds{0.0};
    double p95_microseconds{0.0};
    double p99_microseconds{0.0};
    double population_standard_deviation_microseconds{0.0};
    double active_call_throughput_requests_per_second{0.0};
    std::chrono::nanoseconds total_duration{0};
};

enum class StatisticsErrorKind {
    kNone,
    kEmptySamples,
    kNegativeDuration,
    kZeroTotalDuration,
    kDurationOverflow,
};

struct StatisticsError {
    StatisticsErrorKind kind{StatisticsErrorKind::kNone};
    std::string message;
};

// summary is meaningful only when error.kind is kNone.
struct StatisticsResult {
    StatisticsError error;
    LatencySummary summary;
};

// Reduces injected durations. Does not read a clock or call InferenceEngine.
// Nearest-rank percentiles use index ceil(p * n / 100) - 1 on the sorted
// samples. P50 is that element, not the average of the two middle elements.
// The deviation divides by n, not n - 1. A zero sample is kept when the total
// is positive. An empty span, a negative sample, a zero total, or a sum that
// does not fit in the nanosecond representation returns an error and an empty
// summary. The span is not reordered.
StatisticsResult summarize_latencies(std::span<const std::chrono::nanoseconds> samples);

} // namespace inferedge
