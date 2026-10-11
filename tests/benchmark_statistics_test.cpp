#include "inferedge/benchmark_statistics.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <vector>

namespace {

std::vector<std::chrono::nanoseconds> durations(std::initializer_list<std::int64_t> nanoseconds) {
    std::vector<std::chrono::nanoseconds> samples;
    samples.reserve(nanoseconds.size());
    for (const std::int64_t value : nanoseconds) {
        samples.push_back(std::chrono::nanoseconds{value});
    }
    return samples;
}

inferedge::LatencySummary expect_success(const inferedge::StatisticsResult& result) {
    EXPECT_EQ(result.error.kind, inferedge::StatisticsErrorKind::kNone);
    EXPECT_TRUE(result.error.message.empty());
    return result.summary;
}

void expect_failure(const inferedge::StatisticsResult& result,
                    inferedge::StatisticsErrorKind kind) {
    EXPECT_EQ(result.error.kind, kind);
    EXPECT_FALSE(result.error.message.empty());
    EXPECT_EQ(result.summary.sample_count, 0u);
    EXPECT_EQ(result.summary.total_duration.count(), 0);
}

} // namespace

TEST(BenchmarkStatisticsTest, RejectsAnEmptyCollection) {
    const inferedge::StatisticsResult result =
        inferedge::summarize_latencies(std::span<const std::chrono::nanoseconds>{});
    expect_failure(result, inferedge::StatisticsErrorKind::kEmptySamples);
}

TEST(BenchmarkStatisticsTest, SummarizesOneSample) {
    const std::vector<std::chrono::nanoseconds> samples = durations({1000});
    const inferedge::LatencySummary& summary =
        expect_success(inferedge::summarize_latencies(samples));
    EXPECT_EQ(summary.sample_count, 1u);
    EXPECT_EQ(summary.minimum.count(), 1000);
    EXPECT_EQ(summary.maximum.count(), 1000);
    EXPECT_EQ(summary.total_duration.count(), 1000);
    EXPECT_NEAR(summary.mean_microseconds, 1.0, 1e-12);
    EXPECT_NEAR(summary.p50_microseconds, 1.0, 1e-12);
    EXPECT_NEAR(summary.p95_microseconds, 1.0, 1e-12);
    EXPECT_NEAR(summary.p99_microseconds, 1.0, 1e-12);
    EXPECT_NEAR(summary.population_standard_deviation_microseconds, 0.0, 1e-12);
    EXPECT_NEAR(summary.active_call_throughput_requests_per_second, 1000000.0, 1e-6);
}

TEST(BenchmarkStatisticsTest, UsesTheLowerSampleForNearestRankP50) {
    const std::vector<std::chrono::nanoseconds> samples = durations({1000, 3000});
    const inferedge::LatencySummary& summary =
        expect_success(inferedge::summarize_latencies(samples));
    EXPECT_NEAR(summary.p50_microseconds, 1.0, 1e-12);
    EXPECT_GT(std::abs(summary.p50_microseconds - 2.0), 0.5);
}

TEST(BenchmarkStatisticsTest, MatchesSortedOrderForUnsortedSamples) {
    const std::vector<std::chrono::nanoseconds> scrambled =
        durations({5000, 1000, 4000, 2000, 3000});
    const std::vector<std::chrono::nanoseconds> sorted = durations({1000, 2000, 3000, 4000, 5000});
    const inferedge::StatisticsResult scrambled_result = inferedge::summarize_latencies(scrambled);
    const inferedge::StatisticsResult sorted_result = inferedge::summarize_latencies(sorted);
    const inferedge::LatencySummary& left = expect_success(scrambled_result);
    const inferedge::LatencySummary& right = expect_success(sorted_result);
    EXPECT_EQ(left.sample_count, right.sample_count);
    EXPECT_EQ(left.minimum, right.minimum);
    EXPECT_EQ(left.maximum, right.maximum);
    EXPECT_EQ(left.total_duration, right.total_duration);
    EXPECT_NEAR(left.mean_microseconds, right.mean_microseconds, 1e-12);
    EXPECT_NEAR(left.p50_microseconds, right.p50_microseconds, 1e-12);
    EXPECT_NEAR(left.p95_microseconds, right.p95_microseconds, 1e-12);
    EXPECT_NEAR(left.p99_microseconds, right.p99_microseconds, 1e-12);
    EXPECT_NEAR(left.population_standard_deviation_microseconds,
                right.population_standard_deviation_microseconds, 1e-12);
    EXPECT_NEAR(left.active_call_throughput_requests_per_second,
                right.active_call_throughput_requests_per_second, 1e-6);
}

TEST(BenchmarkStatisticsTest, SummarizesTheKnownFiveSampleSeries) {
    const std::vector<std::chrono::nanoseconds> samples = durations({1000, 2000, 3000, 4000, 5000});
    const inferedge::LatencySummary& summary =
        expect_success(inferedge::summarize_latencies(samples));
    EXPECT_EQ(summary.sample_count, 5u);
    EXPECT_EQ(summary.minimum.count(), 1000);
    EXPECT_EQ(summary.maximum.count(), 5000);
    EXPECT_EQ(summary.total_duration.count(), 15000);
    EXPECT_NEAR(summary.mean_microseconds, 3.0, 1e-12);
    EXPECT_NEAR(summary.p50_microseconds, 3.0, 1e-12);
    EXPECT_NEAR(summary.p95_microseconds, 5.0, 1e-12);
    EXPECT_NEAR(summary.p99_microseconds, 5.0, 1e-12);
    EXPECT_NEAR(summary.population_standard_deviation_microseconds, std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(summary.active_call_throughput_requests_per_second, 5000000000.0 / 15000.0, 1e-6);
}

TEST(BenchmarkStatisticsTest, ComputesActiveCallThroughputFromKnownDurations) {
    const std::vector<std::chrono::nanoseconds> samples = durations({1000000000, 1000000000});
    const inferedge::LatencySummary& summary =
        expect_success(inferedge::summarize_latencies(samples));
    EXPECT_EQ(summary.total_duration.count(), 2000000000);
    EXPECT_NEAR(summary.active_call_throughput_requests_per_second, 1.0, 1e-12);
    EXPECT_NEAR(summary.mean_microseconds, 1000000.0, 1e-9);
}

TEST(BenchmarkStatisticsTest, KeepsAZeroDurationWhenTheTotalIsPositive) {
    const std::vector<std::chrono::nanoseconds> samples = durations({0, 1000});
    const inferedge::LatencySummary& summary =
        expect_success(inferedge::summarize_latencies(samples));
    EXPECT_EQ(summary.sample_count, 2u);
    EXPECT_EQ(summary.minimum.count(), 0);
    EXPECT_EQ(summary.maximum.count(), 1000);
    EXPECT_EQ(summary.total_duration.count(), 1000);
    EXPECT_NEAR(summary.mean_microseconds, 0.5, 1e-12);
    EXPECT_NEAR(summary.p50_microseconds, 0.0, 1e-12);
    EXPECT_NEAR(summary.active_call_throughput_requests_per_second, 2000000.0, 1e-6);
}

TEST(BenchmarkStatisticsTest, RejectsAllZeroSamples) {
    const std::vector<std::chrono::nanoseconds> samples = durations({0, 0, 0});
    const inferedge::StatisticsResult result = inferedge::summarize_latencies(samples);
    expect_failure(result, inferedge::StatisticsErrorKind::kZeroTotalDuration);
}

TEST(BenchmarkStatisticsTest, RejectsANegativeDuration) {
    const std::vector<std::chrono::nanoseconds> samples = durations({1000, -1, 2000});
    const inferedge::StatisticsResult result = inferedge::summarize_latencies(samples);
    expect_failure(result, inferedge::StatisticsErrorKind::kNegativeDuration);
}

TEST(BenchmarkStatisticsTest, SummarizesLargeFiniteDurations) {
    using Rep = std::chrono::nanoseconds::rep;
    const Rep half = std::numeric_limits<Rep>::max() / 2;
    const std::vector<std::chrono::nanoseconds> samples{std::chrono::nanoseconds{half},
                                                        std::chrono::nanoseconds{half}};
    const inferedge::LatencySummary& summary =
        expect_success(inferedge::summarize_latencies(samples));
    EXPECT_EQ(summary.sample_count, 2u);
    EXPECT_EQ(summary.minimum.count(), half);
    EXPECT_EQ(summary.maximum.count(), half);
    EXPECT_EQ(summary.total_duration.count(), half + half);
    EXPECT_NEAR(summary.population_standard_deviation_microseconds, 0.0, 1e-12);
    EXPECT_TRUE(std::isfinite(summary.mean_microseconds));
    EXPECT_GT(summary.mean_microseconds, 0.0);
    EXPECT_TRUE(std::isfinite(summary.active_call_throughput_requests_per_second));
    EXPECT_GT(summary.active_call_throughput_requests_per_second, 0.0);
}

TEST(BenchmarkStatisticsTest, RejectsAnOverflowingSum) {
    using Rep = std::chrono::nanoseconds::rep;
    const Rep too_large = std::numeric_limits<Rep>::max() / 2 + 1;
    const std::vector<std::chrono::nanoseconds> samples{std::chrono::nanoseconds{too_large},
                                                        std::chrono::nanoseconds{too_large}};
    const inferedge::StatisticsResult result = inferedge::summarize_latencies(samples);
    expect_failure(result, inferedge::StatisticsErrorKind::kDurationOverflow);
}

TEST(BenchmarkStatisticsTest, PlacesP95AndP99ForOneHundredSamples) {
    std::vector<std::chrono::nanoseconds> samples;
    samples.reserve(100);
    for (int value = 100; value >= 1; --value) {
        samples.push_back(std::chrono::nanoseconds{value});
    }
    const inferedge::LatencySummary& summary =
        expect_success(inferedge::summarize_latencies(samples));
    EXPECT_EQ(summary.sample_count, 100u);
    EXPECT_NEAR(summary.p95_microseconds, 0.095, 1e-12);
    EXPECT_NEAR(summary.p99_microseconds, 0.099, 1e-12);
}

TEST(BenchmarkStatisticsTest, LeavesTheInputOrderUnchanged) {
    std::vector<std::chrono::nanoseconds> samples = durations({5000, 1000, 4000, 2000, 3000});
    const std::vector<std::chrono::nanoseconds> original = samples;
    expect_success(inferedge::summarize_latencies(samples));
    EXPECT_EQ(samples, original);
}

TEST(BenchmarkStatisticsTest, RepeatsTheSameSummary) {
    const std::vector<std::chrono::nanoseconds> samples = durations({1000, 2000, 3000, 4000, 5000});
    const inferedge::StatisticsResult first = inferedge::summarize_latencies(samples);
    const inferedge::StatisticsResult second = inferedge::summarize_latencies(samples);
    const inferedge::LatencySummary& left = expect_success(first);
    const inferedge::LatencySummary& right = expect_success(second);
    EXPECT_EQ(left.sample_count, right.sample_count);
    EXPECT_EQ(left.minimum, right.minimum);
    EXPECT_EQ(left.maximum, right.maximum);
    EXPECT_EQ(left.total_duration, right.total_duration);
    EXPECT_NEAR(left.mean_microseconds, right.mean_microseconds, 0.0);
    EXPECT_NEAR(left.p50_microseconds, right.p50_microseconds, 0.0);
    EXPECT_NEAR(left.p95_microseconds, right.p95_microseconds, 0.0);
    EXPECT_NEAR(left.p99_microseconds, right.p99_microseconds, 0.0);
    EXPECT_NEAR(left.population_standard_deviation_microseconds,
                right.population_standard_deviation_microseconds, 0.0);
    EXPECT_NEAR(left.active_call_throughput_requests_per_second,
                right.active_call_throughput_requests_per_second, 0.0);
}
