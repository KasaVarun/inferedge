#include "inferedge/benchmark_statistics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace inferedge {
namespace {

StatisticsResult Fail(StatisticsErrorKind kind, std::string message) {
    return StatisticsResult{StatisticsError{kind, std::move(message)}, LatencySummary{}};
}

double microseconds_from_nanoseconds(std::int64_t nanoseconds) {
    return static_cast<double>(nanoseconds) / 1000.0;
}

// ceil(percent * count / 100) - 1, with integer division. percent is 50, 95,
// or 99. count is at least 1. This is the nearest-rank index.
bool nearest_rank_index(std::uint64_t count, std::uint64_t percent, std::uint64_t& index) {
    if (percent == 0 || count > (std::numeric_limits<std::uint64_t>::max() - 99) / percent) {
        return false;
    }
    const std::uint64_t rank = (percent * count + 99) / 100;
    if (rank == 0) {
        return false;
    }
    index = rank - 1;
    return true;
}

} // namespace

StatisticsResult summarize_latencies(std::span<const std::chrono::nanoseconds> samples) {
    if (samples.empty()) {
        return Fail(StatisticsErrorKind::kEmptySamples, "latency sample collection is empty");
    }

    using Rep = std::chrono::nanoseconds::rep;
    static_assert(std::is_signed_v<Rep>, "nanosecond representation must be signed");

    std::vector<Rep> ordered;
    ordered.reserve(samples.size());
    Rep total = 0;
    for (const std::chrono::nanoseconds sample : samples) {
        const Rep count = sample.count();
        if (count < 0) {
            return Fail(StatisticsErrorKind::kNegativeDuration, "latency sample is negative");
        }
        if (count > std::numeric_limits<Rep>::max() - total) {
            return Fail(StatisticsErrorKind::kDurationOverflow, "sum of latency samples overflows");
        }
        total += count;
        ordered.push_back(count);
    }
    if (total == 0) {
        return Fail(StatisticsErrorKind::kZeroTotalDuration,
                    "total measured call duration is zero");
    }

    std::sort(ordered.begin(), ordered.end());
    const std::uint64_t sample_count = ordered.size();
    std::uint64_t p50_index = 0;
    std::uint64_t p95_index = 0;
    std::uint64_t p99_index = 0;
    if (!nearest_rank_index(sample_count, 50, p50_index) ||
        !nearest_rank_index(sample_count, 95, p95_index) ||
        !nearest_rank_index(sample_count, 99, p99_index) || p50_index >= sample_count ||
        p95_index >= sample_count || p99_index >= sample_count) {
        return Fail(StatisticsErrorKind::kDurationOverflow,
                    "percentile rank does not fit the sample count");
    }

    const double mean_nanoseconds = static_cast<double>(total) / static_cast<double>(sample_count);
    double standard_deviation_microseconds = 0.0;
    if (ordered.front() != ordered.back()) {
        double squared_error = 0.0;
        for (const Rep count : ordered) {
            const double delta = static_cast<double>(count) - mean_nanoseconds;
            squared_error += delta * delta;
        }
        standard_deviation_microseconds =
            std::sqrt(squared_error / static_cast<double>(sample_count)) / 1000.0;
    }

    const double total_seconds = static_cast<double>(total) / 1000000000.0;
    const double throughput = static_cast<double>(sample_count) / total_seconds;
    const double mean_microseconds = mean_nanoseconds / 1000.0;
    const double p50 = microseconds_from_nanoseconds(ordered[static_cast<std::size_t>(p50_index)]);
    const double p95 = microseconds_from_nanoseconds(ordered[static_cast<std::size_t>(p95_index)]);
    const double p99 = microseconds_from_nanoseconds(ordered[static_cast<std::size_t>(p99_index)]);
    if (!std::isfinite(mean_microseconds) || !std::isfinite(standard_deviation_microseconds) ||
        !std::isfinite(throughput) || !std::isfinite(p50) || !std::isfinite(p95) ||
        !std::isfinite(p99)) {
        return Fail(StatisticsErrorKind::kDurationOverflow, "latency summary is not finite");
    }

    LatencySummary summary;
    summary.sample_count = sample_count;
    summary.minimum = std::chrono::nanoseconds{ordered.front()};
    summary.maximum = std::chrono::nanoseconds{ordered.back()};
    summary.mean_microseconds = mean_microseconds;
    summary.p50_microseconds = p50;
    summary.p95_microseconds = p95;
    summary.p99_microseconds = p99;
    summary.population_standard_deviation_microseconds = standard_deviation_microseconds;
    summary.active_call_throughput_requests_per_second = throughput;
    summary.total_duration = std::chrono::nanoseconds{total};
    return StatisticsResult{StatisticsError{}, summary};
}

} // namespace inferedge
