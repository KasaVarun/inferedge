#include "inferedge/benchmark_runner.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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

using inferedge::BenchmarkConfig;
using inferedge::BenchmarkErrorKind;
using inferedge::BenchmarkRun;
using inferedge::ExpectedTensor;
using inferedge::InferenceEngine;
using inferedge::InferenceErrorKind;
using inferedge::kMaxBenchmarkIterations;
using inferedge::load_inference_engine;
using inferedge::NamedTensorView;
using inferedge::run_benchmark;
using inferedge::StatisticsErrorKind;
using inferedge::summarize_latencies;

class BenchmarkRunnerTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

ExpectedTensor add_output(std::vector<float> values, std::vector<std::int64_t> shape,
                          std::string name) {
    ExpectedTensor tensor;
    tensor.name = std::move(name);
    tensor.shape = std::move(shape);
    tensor.values = std::move(values);
    return tensor;
}

ExpectedTensor correct_add_output() {
    return add_output({4.0f, 6.0f}, {2}, "C");
}

BenchmarkConfig short_config() {
    BenchmarkConfig config;
    config.warmup_iterations = 0;
    config.measured_iterations = 2;
    config.correctness_check_interval = 1;
    config.workload = "add";
    config.model = "add_f32";
    return config;
}

struct AddInputs {
    std::vector<float> left{1.0f, 2.0f};
    std::vector<float> right{3.0f, 4.0f};

    std::vector<NamedTensorView> views() const {
        return {NamedTensorView{"A", {2}, left}, NamedTensorView{"B", {2}, right}};
    }
};

InferenceEngine load_add() {
    auto loaded = load_inference_engine(INFEREDGE_ADD_MODEL_PATH);
    EXPECT_EQ(loaded.error.kind, InferenceErrorKind::kNone);
    if (loaded.engine == nullptr) {
        throw std::runtime_error(loaded.error.message);
    }
    return std::move(*loaded.engine);
}

InferenceEngine moved_from_add() {
    InferenceEngine engine = load_add();
    InferenceEngine discarded = std::move(engine);
    return engine;
}

void expect_failure(const BenchmarkRun& run, BenchmarkErrorKind kind) {
    EXPECT_EQ(run.error.kind, kind);
    EXPECT_FALSE(run.error.message.empty());
    EXPECT_TRUE(run.report.call_durations.empty());
    EXPECT_EQ(run.report.call_summary.sample_count, 0u);
    EXPECT_EQ(run.report.measurement_loop_duration.count(), 0);
    EXPECT_EQ(run.report.observed_sequential_throughput_requests_per_second, 0.0);
}

inferedge::BenchmarkReport expect_success(const BenchmarkRun& run) {
    EXPECT_EQ(run.error.kind, BenchmarkErrorKind::kNone);
    EXPECT_TRUE(run.error.message.empty());
    return run.report;
}

std::vector<float> read_float32(const std::filesystem::path& path, std::size_t count) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input) << path;
    input.seekg(0, std::ios::end);
    EXPECT_EQ(input.tellg(), static_cast<std::streamoff>(count * 4));
    input.seekg(0, std::ios::beg);
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        unsigned char bytes[4] = {};
        input.read(reinterpret_cast<char*>(bytes), 4);
        EXPECT_TRUE(input);
        const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                                   (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                                   (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                                   (static_cast<std::uint32_t>(bytes[3]) << 24U);
        values[index] = std::bit_cast<float>(bits);
    }
    return values;
}

} // namespace

TEST_F(BenchmarkRunnerTest, RejectsZeroMeasuredIterations) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    BenchmarkConfig config = short_config();
    config.measured_iterations = 0;
    expect_failure(run_benchmark(engine, views, expected, config),
                   BenchmarkErrorKind::kInvalidConfig);
}

TEST_F(BenchmarkRunnerTest, RejectsExcessiveWarmup) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    BenchmarkConfig config = short_config();
    config.warmup_iterations = kMaxBenchmarkIterations + 1;
    expect_failure(run_benchmark(engine, views, expected, config),
                   BenchmarkErrorKind::kInvalidConfig);
}

TEST_F(BenchmarkRunnerTest, RejectsExcessiveMeasuredIterations) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    BenchmarkConfig config = short_config();
    config.measured_iterations = kMaxBenchmarkIterations + 1;
    expect_failure(run_benchmark(engine, views, expected, config),
                   BenchmarkErrorKind::kInvalidConfig);
}

TEST_F(BenchmarkRunnerTest, RejectsZeroCheckInterval) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    BenchmarkConfig config = short_config();
    config.correctness_check_interval = 0;
    expect_failure(run_benchmark(engine, views, expected, config),
                   BenchmarkErrorKind::kInvalidConfig);
}

TEST_F(BenchmarkRunnerTest, RejectsEmptyExpectedOutputs) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    expect_failure(run_benchmark(engine, views, {}, short_config()),
                   BenchmarkErrorKind::kMissingExpectedOutput);
}

TEST_F(BenchmarkRunnerTest, RejectsAnEmptyExpectedName) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{add_output({4.0f, 6.0f}, {2}, "")};
    expect_failure(run_benchmark(engine, views, expected, short_config()),
                   BenchmarkErrorKind::kInvalidExpectedOutput);
}

TEST_F(BenchmarkRunnerTest, RejectsDuplicateExpectedNames) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output(), correct_add_output()};
    expect_failure(run_benchmark(engine, views, expected, short_config()),
                   BenchmarkErrorKind::kInvalidExpectedOutput);
}

TEST_F(BenchmarkRunnerTest, RejectsANegativeExpectedDimension) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{add_output({4.0f}, {-1}, "C")};
    expect_failure(run_benchmark(engine, views, expected, short_config()),
                   BenchmarkErrorKind::kInvalidExpectedOutput);
}

TEST_F(BenchmarkRunnerTest, RejectsAnExpectedElementCountMismatch) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{add_output({4.0f, 6.0f, 0.0f}, {2}, "C")};
    expect_failure(run_benchmark(engine, views, expected, short_config()),
                   BenchmarkErrorKind::kInvalidExpectedOutput);
}

TEST_F(BenchmarkRunnerTest, RejectsAnOverflowingExpectedShape) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::int64_t huge = std::numeric_limits<std::int64_t>::max();
    const std::vector<ExpectedTensor> expected{add_output({1.0f}, {huge, 2}, "C")};
    expect_failure(run_benchmark(engine, views, expected, short_config()),
                   BenchmarkErrorKind::kInvalidExpectedOutput);
}

TEST_F(BenchmarkRunnerTest, RejectsInvalidTolerances) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const double bad_values[] = {-1.0, std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::infinity()};
    for (const double bad : bad_values) {
        ExpectedTensor tensor = correct_add_output();
        tensor.relative_tolerance = bad;
        const std::vector<ExpectedTensor> expected{tensor};
        expect_failure(run_benchmark(engine, views, expected, short_config()),
                       BenchmarkErrorKind::kInvalidExpectedOutput);
        tensor = correct_add_output();
        tensor.absolute_tolerance = bad;
        const std::vector<ExpectedTensor> absolute{tensor};
        expect_failure(run_benchmark(engine, views, absolute, short_config()),
                       BenchmarkErrorKind::kInvalidExpectedOutput);
    }
}

TEST_F(BenchmarkRunnerTest, RejectsNonFiniteExpectedValues) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{
        add_output({std::numeric_limits<float>::quiet_NaN(), 6.0f}, {2}, "C")};
    expect_failure(run_benchmark(engine, views, expected, short_config()),
                   BenchmarkErrorKind::kInvalidExpectedOutput);
}

TEST_F(BenchmarkRunnerTest, RejectsAWrongInputShape) {
    InferenceEngine engine = load_add();
    std::vector<float> left{1.0f, 2.0f, 3.0f};
    const std::vector<float> right{3.0f, 4.0f};
    const std::vector<NamedTensorView> views{NamedTensorView{"A", {3}, left},
                                             NamedTensorView{"B", {2}, right}};
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    const BenchmarkRun run = run_benchmark(engine, views, expected, short_config());
    expect_failure(run, BenchmarkErrorKind::kInferenceFailed);
    EXPECT_EQ(run.error.inference_kind, InferenceErrorKind::kShapeMismatch);
}

TEST_F(BenchmarkRunnerTest, RejectsAMovedFromEngine) {
    InferenceEngine engine = moved_from_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    const BenchmarkRun run = run_benchmark(engine, views, expected, short_config());
    expect_failure(run, BenchmarkErrorKind::kInferenceFailed);
    EXPECT_EQ(run.error.inference_kind, InferenceErrorKind::kExecution);
}

TEST_F(BenchmarkRunnerTest, MeasuresTheAddModel) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    BenchmarkConfig config = short_config();
    config.warmup_iterations = 1;
    config.measured_iterations = 4;
    const inferedge::BenchmarkReport& report =
        expect_success(run_benchmark(engine, views, expected, config));
    EXPECT_EQ(report.workload, "add");
    EXPECT_EQ(report.model, "add_f32");
    EXPECT_EQ(report.backend, "ONNX Runtime CPU");
    EXPECT_EQ(report.warmup_iterations, 1u);
    EXPECT_EQ(report.measured_iterations, 4u);
    EXPECT_EQ(report.correctness_check_interval, 1u);
    EXPECT_EQ(report.call_durations.size(), 4u);
}

TEST_F(BenchmarkRunnerTest, MeasuresTheMnistFixture) {
    auto loaded = load_inference_engine(INFEREDGE_MNIST_MODEL_PATH);
    ASSERT_EQ(loaded.error.kind, InferenceErrorKind::kNone);
    ASSERT_NE(loaded.engine, nullptr);
    const std::filesystem::path directory{INFEREDGE_MNIST_FIXTURE_DIR};
    const std::vector<float> image = read_float32(directory / "inputs" / "test_00003.f32", 784);
    const std::vector<float> logits = read_float32(directory / "onnx" / "test_00003.f32", 10);
    std::size_t best = 0;
    for (std::size_t index = 1; index < logits.size(); ++index) {
        if (logits[index] > logits[best]) {
            best = index;
        }
    }
    ASSERT_EQ(best, 0u);

    const std::vector<NamedTensorView> views{NamedTensorView{"image", {784}, image}};
    ExpectedTensor expected;
    expected.name = "logits";
    expected.shape = {10};
    expected.values = logits;
    expected.relative_tolerance = 1e-4;
    expected.absolute_tolerance = 1e-5;
    const std::vector<ExpectedTensor> outputs{expected};
    BenchmarkConfig config;
    config.warmup_iterations = 1;
    config.measured_iterations = 2;
    config.correctness_check_interval = 1;
    config.workload = "mnist";
    config.model = "mnist_mlp";
    const inferedge::BenchmarkReport& report =
        expect_success(run_benchmark(*loaded.engine, views, outputs, config));
    EXPECT_EQ(report.call_durations.size(), 2u);
    EXPECT_EQ(report.backend, "ONNX Runtime CPU");
}

TEST_F(BenchmarkRunnerTest, RejectsWrongExpectedValues) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{add_output({0.0f, 0.0f}, {2}, "C")};
    expect_failure(run_benchmark(engine, views, expected, short_config()),
                   BenchmarkErrorKind::kOutputMismatch);
}

TEST_F(BenchmarkRunnerTest, RejectsAWrongExpectedName) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{add_output({4.0f, 6.0f}, {2}, "sum")};
    expect_failure(run_benchmark(engine, views, expected, short_config()),
                   BenchmarkErrorKind::kOutputMismatch);
}

TEST_F(BenchmarkRunnerTest, RejectsAWrongExpectedShape) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{add_output({4.0f, 6.0f, 0.0f}, {3}, "C")};
    expect_failure(run_benchmark(engine, views, expected, short_config()),
                   BenchmarkErrorKind::kOutputMismatch);
}

TEST_F(BenchmarkRunnerTest, RepeatsARunOnOneEngine) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    expect_success(run_benchmark(engine, views, expected, short_config()));
    const inferedge::BenchmarkReport& second =
        expect_success(run_benchmark(engine, views, expected, short_config()));
    EXPECT_EQ(second.call_durations.size(), 2u);
}

TEST_F(BenchmarkRunnerTest, RecoversAfterAFailedBenchmark) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> wrong{add_output({0.0f, 0.0f}, {2}, "C")};
    expect_failure(run_benchmark(engine, views, wrong, short_config()),
                   BenchmarkErrorKind::kOutputMismatch);
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    const inferedge::BenchmarkReport& report =
        expect_success(run_benchmark(engine, views, expected, short_config()));
    EXPECT_EQ(report.call_durations.size(), 2u);
}

TEST_F(BenchmarkRunnerTest, RecordsOneSamplePerMeasuredIteration) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    BenchmarkConfig config = short_config();
    config.measured_iterations = 3;
    const inferedge::BenchmarkReport& report =
        expect_success(run_benchmark(engine, views, expected, config));
    EXPECT_EQ(report.call_durations.size(), 3u);
    EXPECT_EQ(report.call_summary.sample_count, 3u);
}

TEST_F(BenchmarkRunnerTest, MatchesTheLatencySummaryToTheStoredSamples) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    const inferedge::BenchmarkReport& report =
        expect_success(run_benchmark(engine, views, expected, short_config()));
    const auto statistics = summarize_latencies(report.call_durations);
    ASSERT_EQ(statistics.error.kind, StatisticsErrorKind::kNone);
    EXPECT_EQ(report.call_summary.sample_count, statistics.summary.sample_count);
    EXPECT_EQ(report.call_summary.minimum, statistics.summary.minimum);
    EXPECT_EQ(report.call_summary.maximum, statistics.summary.maximum);
    EXPECT_EQ(report.call_summary.total_duration, statistics.summary.total_duration);
    EXPECT_NEAR(report.call_summary.mean_microseconds, statistics.summary.mean_microseconds, 1e-12);
    EXPECT_NEAR(report.call_summary.p50_microseconds, statistics.summary.p50_microseconds, 1e-12);
    EXPECT_NEAR(report.call_summary.p95_microseconds, statistics.summary.p95_microseconds, 1e-12);
    EXPECT_NEAR(report.call_summary.p99_microseconds, statistics.summary.p99_microseconds, 1e-12);
    EXPECT_NEAR(report.call_summary.population_standard_deviation_microseconds,
                statistics.summary.population_standard_deviation_microseconds, 1e-12);
    EXPECT_NEAR(report.call_summary.active_call_throughput_requests_per_second,
                statistics.summary.active_call_throughput_requests_per_second, 1e-6);
}

TEST_F(BenchmarkRunnerTest, MatchesObservedThroughputToTheOuterLoop) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    const inferedge::BenchmarkReport& report =
        expect_success(run_benchmark(engine, views, expected, short_config()));
    ASSERT_GT(report.measurement_loop_duration.count(), 0);
    EXPECT_GE(report.measurement_loop_duration, report.call_summary.total_duration);
    const double seconds =
        static_cast<double>(report.measurement_loop_duration.count()) / 1000000000.0;
    EXPECT_NEAR(report.observed_sequential_throughput_requests_per_second,
                static_cast<double>(report.measured_iterations) / seconds, 1e-6);
    EXPECT_TRUE(std::isfinite(report.observed_sequential_throughput_requests_per_second));
}

TEST_F(BenchmarkRunnerTest, LeavesWarmupOutOfTheRecordedSamples) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    BenchmarkConfig config = short_config();
    config.warmup_iterations = 5;
    config.measured_iterations = 2;
    const inferedge::BenchmarkReport& report =
        expect_success(run_benchmark(engine, views, expected, config));
    EXPECT_EQ(report.warmup_iterations, 5u);
    EXPECT_EQ(report.call_durations.size(), 2u);
    EXPECT_EQ(report.call_summary.sample_count, 2u);
}

TEST_F(BenchmarkRunnerTest, AllowsZeroWarmup) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    BenchmarkConfig config = short_config();
    config.warmup_iterations = 0;
    const inferedge::BenchmarkReport& report =
        expect_success(run_benchmark(engine, views, expected, config));
    EXPECT_EQ(report.warmup_iterations, 0u);
    EXPECT_EQ(report.call_durations.size(), 2u);
}

TEST_F(BenchmarkRunnerTest, AcceptsACheckIntervalGreaterThanOne) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{correct_add_output()};
    BenchmarkConfig config = short_config();
    config.measured_iterations = 4;
    config.correctness_check_interval = 3;
    const inferedge::BenchmarkReport& report =
        expect_success(run_benchmark(engine, views, expected, config));
    EXPECT_EQ(report.correctness_check_interval, 3u);
    EXPECT_EQ(report.call_durations.size(), 4u);
}

TEST_F(BenchmarkRunnerTest, ChecksValuesOnTheFinalIteration) {
    InferenceEngine engine = load_add();
    const AddInputs inputs;
    const std::vector<NamedTensorView> views = inputs.views();
    const std::vector<ExpectedTensor> expected{add_output({0.0f, 0.0f}, {2}, "C")};
    BenchmarkConfig config = short_config();
    config.measured_iterations = 3;
    config.correctness_check_interval = 1000;
    expect_failure(run_benchmark(engine, views, expected, config),
                   BenchmarkErrorKind::kOutputMismatch);
}
