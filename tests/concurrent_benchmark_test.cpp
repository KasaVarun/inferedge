#include "inferedge/benchmark_runner.hpp"
#include "inferedge/concurrent_benchmark.hpp"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

#ifndef INFEREDGE_ADD_MODEL_PATH
#error "CMake must define INFEREDGE_ADD_MODEL_PATH"
#endif

#ifndef INFEREDGE_MNIST_MODEL_PATH
#error "CMake must define INFEREDGE_MNIST_MODEL_PATH"
#endif

#ifndef INFEREDGE_MNIST_FIXTURE_DIR
#error "CMake must define INFEREDGE_MNIST_FIXTURE_DIR"
#endif

namespace {

using inferedge::ConcurrentBenchmarkAssignment;
using inferedge::ConcurrentBenchmarkConfig;
using inferedge::ConcurrentBenchmarkErrorKind;
using inferedge::ConcurrentBenchmarkPhase;
using inferedge::ConcurrentBenchmarkRun;
using inferedge::InferenceErrorKind;
using inferedge::kMaxBenchmarkIterations;
using inferedge::LatencySummary;
using inferedge::NamedTensor;
using inferedge::run_concurrent_benchmark;
using inferedge::StatisticsErrorKind;
using inferedge::summarize_latencies;
using inferedge::WorkerExpectedTensor;

constexpr std::size_t kMnistInputCount = 784;
constexpr std::size_t kMnistLogitCount = 10;
constexpr double kRelativeTolerance = 1e-4;
constexpr double kAbsoluteTolerance = 1e-5;
constexpr const char* kMissingModel = "unavailable-concurrent-benchmark-model.onnx";
constexpr const char* kMnistIds[] = {
    "test_00003", "test_00002", "test_00001", "test_00018",
    "test_00004", "test_00008", "test_00011", "test_00000",
};

class ConcurrentBenchmarkTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

ConcurrentBenchmarkConfig Config(std::uint32_t workers, std::uint32_t warmup,
                                 std::uint32_t measured, std::uint32_t interval = 1) {
    ConcurrentBenchmarkConfig config;
    config.worker_count = workers;
    config.warmup_iterations = warmup;
    config.measured_iterations = measured;
    config.correctness_check_interval = interval;
    config.workload = "add";
    config.model = "add_f32";
    return config;
}

WorkerExpectedTensor ExpectAdd(float first, float second) {
    return WorkerExpectedTensor{"C", {2}, {first, second}, 0.0, 0.0};
}

ConcurrentBenchmarkAssignment AddAssignment(float b0, float b1) {
    ConcurrentBenchmarkAssignment assignment;
    assignment.inputs.push_back(NamedTensor{"A", {2}, {1.0F, 2.0F}});
    assignment.inputs.push_back(NamedTensor{"B", {2}, {b0, b1}});
    assignment.expected.push_back(ExpectAdd(1.0F + b0, 2.0F + b1));
    return assignment;
}

std::vector<ConcurrentBenchmarkAssignment> AddAssignments(std::uint32_t workers, bool identical) {
    std::vector<ConcurrentBenchmarkAssignment> assignments;
    assignments.reserve(workers);
    for (std::uint32_t index = 0; index < workers; ++index) {
        if (identical) {
            assignments.push_back(AddAssignment(3.0F, 4.0F));
        } else {
            const float offset = static_cast<float>(index);
            assignments.push_back(AddAssignment(3.0F + offset, 4.0F + offset));
        }
    }
    return assignments;
}

void ExpectEmptyReport(const ConcurrentBenchmarkRun& run) {
    EXPECT_EQ(run.report.worker_count, 0u);
    EXPECT_EQ(run.report.total_completed_requests, 0u);
    EXPECT_TRUE(run.report.per_worker_completed_requests.empty());
    EXPECT_TRUE(run.report.per_worker_call_durations.empty());
    EXPECT_EQ(run.report.call_summary.sample_count, 0u);
    EXPECT_EQ(run.report.measurement_wall_duration.count(), 0);
    EXPECT_EQ(run.report.aggregate_completed_requests_per_second, 0.0);
}

void ExpectSameSummary(const LatencySummary& actual, const LatencySummary& expected) {
    EXPECT_EQ(actual.sample_count, expected.sample_count);
    EXPECT_EQ(actual.minimum, expected.minimum);
    EXPECT_EQ(actual.maximum, expected.maximum);
    EXPECT_EQ(actual.total_duration, expected.total_duration);
    EXPECT_DOUBLE_EQ(actual.mean_microseconds, expected.mean_microseconds);
    EXPECT_DOUBLE_EQ(actual.p50_microseconds, expected.p50_microseconds);
    EXPECT_DOUBLE_EQ(actual.p95_microseconds, expected.p95_microseconds);
    EXPECT_DOUBLE_EQ(actual.p99_microseconds, expected.p99_microseconds);
    EXPECT_DOUBLE_EQ(actual.population_standard_deviation_microseconds,
                     expected.population_standard_deviation_microseconds);
    EXPECT_DOUBLE_EQ(actual.active_call_throughput_requests_per_second,
                     expected.active_call_throughput_requests_per_second);
}

void ExpectSuccess(const ConcurrentBenchmarkRun& run, const ConcurrentBenchmarkConfig& config) {
    ASSERT_EQ(run.error.kind, ConcurrentBenchmarkErrorKind::kNone) << run.error.message;
    EXPECT_EQ(run.error.phase, ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.load_error.kind, InferenceErrorKind::kNone);
    const auto& report = run.report;
    EXPECT_EQ(report.backend, "ONNX Runtime CPU");
    EXPECT_EQ(report.workload, config.workload);
    EXPECT_EQ(report.model, config.model);
    EXPECT_EQ(report.worker_count, config.worker_count);
    EXPECT_EQ(report.warmup_iterations, config.warmup_iterations);
    EXPECT_EQ(report.measured_iterations, config.measured_iterations);
    EXPECT_EQ(report.correctness_check_interval, config.correctness_check_interval);

    const std::uint64_t total = static_cast<std::uint64_t>(config.worker_count) *
                                static_cast<std::uint64_t>(config.measured_iterations);
    EXPECT_EQ(report.total_completed_requests, total);
    ASSERT_EQ(report.per_worker_completed_requests.size(), config.worker_count);
    ASSERT_EQ(report.per_worker_call_durations.size(), config.worker_count);

    std::vector<std::chrono::nanoseconds> pooled;
    pooled.reserve(static_cast<std::size_t>(total));
    for (std::uint32_t index = 0; index < config.worker_count; ++index) {
        EXPECT_EQ(report.per_worker_completed_requests[index], config.measured_iterations);
        ASSERT_EQ(report.per_worker_call_durations[index].size(), config.measured_iterations);
        pooled.insert(pooled.end(), report.per_worker_call_durations[index].begin(),
                      report.per_worker_call_durations[index].end());
    }
    EXPECT_EQ(pooled.size(), static_cast<std::size_t>(total));
    EXPECT_EQ(report.call_summary.sample_count, total);

    const auto statistics = summarize_latencies(pooled);
    ASSERT_EQ(statistics.error.kind, StatisticsErrorKind::kNone) << statistics.error.message;
    ExpectSameSummary(report.call_summary, statistics.summary);

    EXPECT_GT(report.measurement_wall_duration.count(), 0);
    EXPECT_TRUE(std::isfinite(report.aggregate_completed_requests_per_second));
    EXPECT_GT(report.aggregate_completed_requests_per_second, 0.0);
    const double seconds =
        static_cast<double>(report.measurement_wall_duration.count()) / 1000000000.0;
    const double rate = static_cast<double>(report.total_completed_requests) / seconds;
    EXPECT_DOUBLE_EQ(report.aggregate_completed_requests_per_second, rate);
    EXPECT_TRUE(std::isfinite(report.call_summary.active_call_throughput_requests_per_second));
    EXPECT_GT(report.call_summary.active_call_throughput_requests_per_second, 0.0);
}

void ExpectFailure(const ConcurrentBenchmarkRun& run, ConcurrentBenchmarkErrorKind kind,
                   ConcurrentBenchmarkPhase phase) {
    EXPECT_EQ(run.error.kind, kind) << run.error.message;
    EXPECT_EQ(run.error.phase, phase);
    ExpectEmptyReport(run);
}

struct FloatFile {
    std::string error;
    std::vector<float> values;
};

FloatFile ReadFloats(const std::filesystem::path& path, std::size_t count) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        return FloatFile{"cannot open " + path.string(), {}};
    }
    input.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(input.tellg());
    if (size != count * sizeof(float)) {
        return FloatFile{"unexpected size for " + path.string(), {}};
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        unsigned char bytes[4] = {};
        input.read(reinterpret_cast<char*>(bytes), 4);
        if (!input) {
            return FloatFile{"short read of " + path.string(), {}};
        }
        const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                                   (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                                   (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                                   (static_cast<std::uint32_t>(bytes[3]) << 24U);
        values[index] = std::bit_cast<float>(bits);
    }
    return FloatFile{{}, std::move(values)};
}

ConcurrentBenchmarkAssignment MnistAssignment(const char* fixture) {
    const std::filesystem::path root{INFEREDGE_MNIST_FIXTURE_DIR};
    const FloatFile image =
        ReadFloats(root / "inputs" / (std::string{fixture} + ".f32"), kMnistInputCount);
    const FloatFile reference =
        ReadFloats(root / "onnx" / (std::string{fixture} + ".f32"), kMnistLogitCount);
    EXPECT_TRUE(image.error.empty()) << image.error;
    EXPECT_TRUE(reference.error.empty()) << reference.error;
    ConcurrentBenchmarkAssignment assignment;
    assignment.inputs.push_back(
        NamedTensor{"image", {static_cast<std::int64_t>(kMnistInputCount)}, image.values});
    assignment.expected.push_back(
        WorkerExpectedTensor{"logits",
                             {static_cast<std::int64_t>(kMnistLogitCount)},
                             reference.values,
                             kRelativeTolerance,
                             kAbsoluteTolerance});
    return assignment;
}

std::vector<ConcurrentBenchmarkAssignment> MnistAssignments(std::uint32_t workers, bool identical) {
    std::vector<ConcurrentBenchmarkAssignment> assignments;
    assignments.reserve(workers);
    for (std::uint32_t index = 0; index < workers; ++index) {
        const char* fixture = identical ? kMnistIds[0] : kMnistIds[index];
        assignments.push_back(MnistAssignment(fixture));
    }
    return assignments;
}

ConcurrentBenchmarkConfig MnistConfig(std::uint32_t workers, std::uint32_t warmup,
                                      std::uint32_t measured) {
    ConcurrentBenchmarkConfig config = Config(workers, warmup, measured);
    config.workload = "mnist";
    config.model = "mnist_mlp";
    return config;
}

} // namespace

TEST_F(ConcurrentBenchmarkTest, OneAddWorker) {
    const ConcurrentBenchmarkConfig config = Config(1, 1, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(1, true);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config), config);
}

TEST_F(ConcurrentBenchmarkTest, TwoAddWorkers) {
    const ConcurrentBenchmarkConfig config = Config(2, 1, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2, true);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config), config);
}

TEST_F(ConcurrentBenchmarkTest, FourAddWorkers) {
    const ConcurrentBenchmarkConfig config = Config(4, 1, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(4, true);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config), config);
}

TEST_F(ConcurrentBenchmarkTest, EightAddWorkers) {
    const ConcurrentBenchmarkConfig config = Config(8, 1, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(8, true);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config), config);
}

TEST_F(ConcurrentBenchmarkTest, IdenticalAddInputsUseSeparateAssignments) {
    const ConcurrentBenchmarkConfig config = Config(2, 1, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2, true);
    ASSERT_EQ(assignments.size(), 2u);
    EXPECT_NE(assignments[0].inputs[0].values.data(), assignments[1].inputs[0].values.data());
    EXPECT_EQ(assignments[0].inputs[1].values, assignments[1].inputs[1].values);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config), config);
}

TEST_F(ConcurrentBenchmarkTest, DistinctAddInputs) {
    const ConcurrentBenchmarkConfig config = Config(4, 1, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(4, false);
    EXPECT_NE(assignments[0].expected[0].values, assignments[3].expected[0].values);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config), config);
}

TEST_F(ConcurrentBenchmarkTest, OneMnistWorker) {
    const ConcurrentBenchmarkConfig config = MnistConfig(1, 1, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = MnistAssignments(1, true);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_MNIST_MODEL_PATH, assignments, config),
                  config);
}

TEST_F(ConcurrentBenchmarkTest, TwoMnistWorkers) {
    const ConcurrentBenchmarkConfig config = MnistConfig(2, 1, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = MnistAssignments(2, true);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_MNIST_MODEL_PATH, assignments, config),
                  config);
}

TEST_F(ConcurrentBenchmarkTest, FourMnistWorkers) {
    const ConcurrentBenchmarkConfig config = MnistConfig(4, 0, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = MnistAssignments(4, true);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_MNIST_MODEL_PATH, assignments, config),
                  config);
}

TEST_F(ConcurrentBenchmarkTest, EightMnistWorkers) {
    const ConcurrentBenchmarkConfig config = MnistConfig(8, 0, 1);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = MnistAssignments(8, true);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_MNIST_MODEL_PATH, assignments, config),
                  config);
}

TEST_F(ConcurrentBenchmarkTest, DistinctMnistFixtures) {
    const ConcurrentBenchmarkConfig config = MnistConfig(2, 0, 1);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = MnistAssignments(2, false);
    EXPECT_NE(assignments[0].expected[0].values, assignments[1].expected[0].values);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_MNIST_MODEL_PATH, assignments, config),
                  config);
}

TEST_F(ConcurrentBenchmarkTest, AllowsZeroWarmup) {
    const ConcurrentBenchmarkConfig config = Config(2, 0, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2, true);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config), config);
}

TEST_F(ConcurrentBenchmarkTest, RepeatedRuns) {
    const ConcurrentBenchmarkConfig config = Config(2, 1, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2, true);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config), config);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config), config);
}

TEST_F(ConcurrentBenchmarkTest, AcceptsACheckIntervalGreaterThanOne) {
    const ConcurrentBenchmarkConfig config = Config(2, 0, 4, 3);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2, true);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config), config);
}

TEST_F(ConcurrentBenchmarkTest, ChecksValuesOnTheFinalIteration) {
    ConcurrentBenchmarkConfig config = Config(1, 0, 3, 1000);
    std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(1, true);
    assignments[0].expected[0].values = {0.0F, 0.0F};
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kOutputMismatch,
                  ConcurrentBenchmarkPhase::kMeasured);
    EXPECT_EQ(run.error.worker_index, 0u);
    EXPECT_EQ(run.error.request_iteration, 2u);
}

TEST_F(ConcurrentBenchmarkTest, RejectsZeroWorkers) {
    const ConcurrentBenchmarkConfig config = Config(0, 0, 1);
    const ConcurrentBenchmarkRun run = run_concurrent_benchmark(kMissingModel, {}, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInvalidConfig,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.message, "worker count is zero");
    EXPECT_EQ(run.load_error.kind, InferenceErrorKind::kNone);
}

TEST_F(ConcurrentBenchmarkTest, RejectsTooManyWorkers) {
    const ConcurrentBenchmarkConfig config = Config(65, 0, 1);
    const ConcurrentBenchmarkRun run = run_concurrent_benchmark(kMissingModel, {}, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInvalidConfig,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.message, "worker count exceeds 64");
}

TEST_F(ConcurrentBenchmarkTest, RejectsAssignmentCountMismatch) {
    const ConcurrentBenchmarkConfig config = Config(2, 0, 1);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(1, true);
    const ConcurrentBenchmarkRun run = run_concurrent_benchmark(kMissingModel, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInvalidConfig,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.message, "assignment count does not match worker count");
}

TEST_F(ConcurrentBenchmarkTest, RejectsZeroMeasuredIterations) {
    const ConcurrentBenchmarkConfig config = Config(1, 0, 0);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(1, true);
    const ConcurrentBenchmarkRun run = run_concurrent_benchmark(kMissingModel, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInvalidConfig,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.message, "measured iteration count is zero");
}

TEST_F(ConcurrentBenchmarkTest, RejectsExcessiveWarmup) {
    const ConcurrentBenchmarkConfig config = Config(1, kMaxBenchmarkIterations + 1, 1);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(1, true);
    const ConcurrentBenchmarkRun run = run_concurrent_benchmark(kMissingModel, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInvalidConfig,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.message, "warmup iteration count exceeds 100000");
}

TEST_F(ConcurrentBenchmarkTest, RejectsExcessiveMeasuredIterations) {
    const ConcurrentBenchmarkConfig config = Config(1, 0, kMaxBenchmarkIterations + 1);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(1, true);
    const ConcurrentBenchmarkRun run = run_concurrent_benchmark(kMissingModel, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInvalidConfig,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.message, "measured iteration count exceeds 100000");
}

TEST_F(ConcurrentBenchmarkTest, RejectsZeroCheckInterval) {
    const ConcurrentBenchmarkConfig config = Config(1, 0, 1, 0);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(1, true);
    const ConcurrentBenchmarkRun run = run_concurrent_benchmark(kMissingModel, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInvalidConfig,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.message, "correctness check interval is zero");
}

TEST_F(ConcurrentBenchmarkTest, RejectsAnOverflowingSampleProduct) {
    ConcurrentBenchmarkConfig config = Config(64, 0, 1);
    config.measured_iterations = std::numeric_limits<std::uint32_t>::max() / 64U + 1U;
    std::vector<ConcurrentBenchmarkAssignment> assignments(64);
    const ConcurrentBenchmarkRun run = run_concurrent_benchmark(kMissingModel, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInvalidConfig,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.message, "worker and measured iteration product overflows");
    EXPECT_EQ(run.load_error.kind, InferenceErrorKind::kNone);
}

TEST_F(ConcurrentBenchmarkTest, RejectsEmptyExpectedOutput) {
    const ConcurrentBenchmarkConfig config = Config(1, 0, 1);
    const std::vector<ConcurrentBenchmarkAssignment> assignments(1);
    const ConcurrentBenchmarkRun run = run_concurrent_benchmark(kMissingModel, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInvalidConfig,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.message, "worker 0 expected output list is empty");
}

TEST_F(ConcurrentBenchmarkTest, RejectsInvalidTolerance) {
    const ConcurrentBenchmarkConfig config = Config(1, 0, 1);
    std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(1, true);
    assignments[0].expected[0].absolute_tolerance = -1.0;
    const ConcurrentBenchmarkRun run = run_concurrent_benchmark(kMissingModel, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInvalidConfig,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.worker_index, 0u);
}

TEST_F(ConcurrentBenchmarkTest, ModelLoadFailureStartsNoWorkers) {
    const ConcurrentBenchmarkConfig config = Config(2, 1, 2);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2, true);
    const ConcurrentBenchmarkRun run = run_concurrent_benchmark(kMissingModel, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kModelLoad, ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.worker_index, 0u);
    EXPECT_EQ(run.error.inference_kind, InferenceErrorKind::kModelNotFound);
    EXPECT_EQ(run.load_error.kind, InferenceErrorKind::kModelNotFound);
    EXPECT_EQ(run.error.message, run.load_error.message);
    EXPECT_FALSE(run.load_error.message.empty());
}

TEST_F(ConcurrentBenchmarkTest, WarmupInferenceFailureCancelsTheRun) {
    ConcurrentBenchmarkConfig config = Config(2, 2, 2);
    std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2, true);
    assignments[1].inputs[1].shape = {3};
    assignments[1].inputs[1].values = {3.0F, 4.0F, 5.0F};
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInferenceFailed,
                  ConcurrentBenchmarkPhase::kWarmup);
    EXPECT_EQ(run.error.worker_index, 1u);
    EXPECT_EQ(run.error.request_iteration, 0u);
    EXPECT_EQ(run.error.inference_kind, InferenceErrorKind::kShapeMismatch);
}

TEST_F(ConcurrentBenchmarkTest, WarmupMismatchCancelsBeforeMeasurement) {
    ConcurrentBenchmarkConfig config = Config(2, 1, 2);
    std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2, true);
    assignments[0].expected[0].values = {0.0F, 0.0F};
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kOutputMismatch,
                  ConcurrentBenchmarkPhase::kWarmup);
    EXPECT_EQ(run.error.worker_index, 0u);
    EXPECT_EQ(run.error.request_iteration, 0u);
}

TEST_F(ConcurrentBenchmarkTest, InjectedWarmupFailureCancelsTheGate) {
    ConcurrentBenchmarkConfig config = Config(4, 2, 2);
    config.fail_warmup_worker = 2;
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(4, true);
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInferenceFailed,
                  ConcurrentBenchmarkPhase::kWarmup);
    EXPECT_EQ(run.error.worker_index, 2u);
    EXPECT_EQ(run.error.request_iteration, 0u);
    EXPECT_EQ(run.error.message, "injected warmup failure");
}

TEST_F(ConcurrentBenchmarkTest, MeasuredOutputMismatchFailsWithoutAReport) {
    ConcurrentBenchmarkConfig config = Config(2, 0, 2);
    std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2, false);
    assignments[1].expected[0].values = {0.0F, 0.0F};
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kOutputMismatch,
                  ConcurrentBenchmarkPhase::kMeasured);
    EXPECT_EQ(run.error.worker_index, 1u);
    EXPECT_EQ(run.error.request_iteration, 0u);
}

TEST_F(ConcurrentBenchmarkTest, InjectedMeasuredIterationFailsOutsideTheTimedCall) {
    ConcurrentBenchmarkConfig config = Config(2, 1, 4);
    config.fail_measured_worker = 1;
    config.fail_measured_iteration = 2;
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2, true);
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInferenceFailed,
                  ConcurrentBenchmarkPhase::kMeasured);
    EXPECT_EQ(run.error.worker_index, 1u);
    EXPECT_EQ(run.error.request_iteration, 2u);
    EXPECT_EQ(run.error.inference_kind, InferenceErrorKind::kNone);
    EXPECT_EQ(run.error.message, "injected measured-loop failure");
}

TEST_F(ConcurrentBenchmarkTest, RejectsANonFiniteOutput) {
    ConcurrentBenchmarkConfig config = Config(1, 1, 1);
    std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(1, true);
    assignments[0].inputs[0].values = {std::numeric_limits<float>::infinity(), 2.0F};
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kNonFiniteOutput,
                  ConcurrentBenchmarkPhase::kWarmup);
    EXPECT_EQ(run.error.request_iteration, 0u);
}

TEST_F(ConcurrentBenchmarkTest, InjectedWorkerExceptionIsCaught) {
    ConcurrentBenchmarkConfig config = Config(3, 1, 2);
    config.throw_in_worker = 1;
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(3, true);
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kWorkerException,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.worker_index, 1u);
    EXPECT_EQ(run.error.message, "injected worker exception");
}

TEST_F(ConcurrentBenchmarkTest, PartialStartupBeforeAnyThread) {
    ConcurrentBenchmarkConfig config = Config(4, 1, 2);
    config.stop_after_starting_workers = 0;
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(4, true);
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kWorkerStart, ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.worker_index, 0u);
    EXPECT_EQ(run.error.message, "worker thread was not started");
}

TEST_F(ConcurrentBenchmarkTest, PartialStartupJoinsThreadsAlreadyCreated) {
    ConcurrentBenchmarkConfig config = Config(8, 1, 2);
    config.stop_after_starting_workers = 3;
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(8, true);
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kWorkerStart, ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.worker_index, 3u);
    EXPECT_EQ(run.error.message, "worker thread was not started");
}

TEST_F(ConcurrentBenchmarkTest, FailureThenSuccessCleansUp) {
    ConcurrentBenchmarkConfig failing = Config(2, 0, 1);
    failing.fail_warmup_worker = 0;
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2, true);
    const ConcurrentBenchmarkRun failed =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, failing);
    ExpectFailure(failed, ConcurrentBenchmarkErrorKind::kInferenceFailed,
                  ConcurrentBenchmarkPhase::kWarmup);

    const ConcurrentBenchmarkConfig config = Config(2, 0, 2);
    ExpectSuccess(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config), config);
}

TEST_F(ConcurrentBenchmarkTest, RejectsAPartialMeasuredInjection) {
    ConcurrentBenchmarkConfig config = Config(1, 0, 2);
    config.fail_measured_worker = 0;
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(1, true);
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectFailure(run, ConcurrentBenchmarkErrorKind::kInvalidConfig,
                  ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.message, "measured failure injection requires a worker and an iteration");
}
