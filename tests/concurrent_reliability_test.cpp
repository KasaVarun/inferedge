#include "inferedge/concurrent_benchmark.hpp"
#include "inferedge/concurrent_inference.hpp"
#include "inferedge/inference_engine.hpp"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <type_traits>
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

using inferedge::ConcurrentBenchmarkAssignment;
using inferedge::ConcurrentBenchmarkConfig;
using inferedge::ConcurrentBenchmarkErrorKind;
using inferedge::ConcurrentBenchmarkPhase;
using inferedge::ConcurrentBenchmarkRun;
using inferedge::ConcurrentErrorKind;
using inferedge::ConcurrentExecution;
using inferedge::ConcurrentPlan;
using inferedge::EngineLoad;
using inferedge::InferenceEngine;
using inferedge::InferenceErrorKind;
using inferedge::InferenceResult;
using inferedge::load_inference_engine;
using inferedge::NamedTensor;
using inferedge::NamedTensorView;
using inferedge::run_concurrent_benchmark;
using inferedge::run_concurrent_inference;
using inferedge::WorkerExpectedTensor;
using inferedge::WorkerRequest;

constexpr std::size_t kMnistInputCount = 784;
constexpr std::size_t kMnistLogitCount = 10;
constexpr double kRelativeTolerance = 1e-4;
constexpr double kAbsoluteTolerance = 1e-5;
constexpr const char* kPrimaryMnistFixture = "test_00003";
constexpr const char* kSecondMnistFixture = "test_00002";

static_assert(std::is_same_v<decltype(NamedTensor::values), std::vector<float>>);

class ConcurrentReliabilityTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

WorkerExpectedTensor ExpectAdd(float first, float second) {
    return WorkerExpectedTensor{"C", {2}, {first, second}, 0.0, 0.0};
}

WorkerRequest AddWorker(float b0, float b1) {
    WorkerRequest worker;
    worker.inputs.push_back(NamedTensor{"A", {2}, {1.0F, 2.0F}});
    worker.inputs.push_back(NamedTensor{"B", {2}, {b0, b1}});
    worker.expected.push_back(ExpectAdd(1.0F + b0, 2.0F + b1));
    return worker;
}

ConcurrentPlan AddPlan(std::uint32_t workers, std::uint32_t requests, bool identical) {
    ConcurrentPlan plan;
    plan.requests_per_worker = requests;
    plan.workers.reserve(workers);
    for (std::uint32_t index = 0; index < workers; ++index) {
        if (identical) {
            plan.workers.push_back(AddWorker(3.0F, 4.0F));
        } else {
            const float offset = static_cast<float>(index);
            plan.workers.push_back(AddWorker(3.0F + offset, 4.0F + offset));
        }
    }
    return plan;
}

std::vector<std::vector<float>> InputSnapshot(const ConcurrentPlan& plan) {
    std::vector<std::vector<float>> snapshot;
    for (const WorkerRequest& worker : plan.workers) {
        for (const NamedTensor& input : worker.inputs) {
            snapshot.push_back(input.values);
        }
    }
    return snapshot;
}

void ExpectInputsUnchanged(const ConcurrentPlan& plan,
                           const std::vector<std::vector<float>>& snapshot) {
    std::size_t cursor = 0;
    for (const WorkerRequest& worker : plan.workers) {
        for (const NamedTensor& input : worker.inputs) {
            ASSERT_LT(cursor, snapshot.size());
            EXPECT_EQ(input.values, snapshot[cursor]);
            ++cursor;
        }
    }
    EXPECT_EQ(cursor, snapshot.size());
}

void ExpectIndependentOutputs(ConcurrentExecution& result) {
    ASSERT_GE(result.outputs.size(), 2U);
    std::vector<const float*> pointers;
    pointers.reserve(result.outputs.size());
    for (const auto& output : result.outputs) {
        ASSERT_EQ(output.outputs.size(), 1U);
        ASSERT_NE(output.outputs[0].values.data(), nullptr);
        pointers.push_back(output.outputs[0].values.data());
    }
    for (std::size_t left = 0; left < pointers.size(); ++left) {
        for (std::size_t right = left + 1; right < pointers.size(); ++right) {
            EXPECT_NE(pointers[left], pointers[right]);
        }
    }
    const std::vector<float> second = result.outputs[1].outputs[0].values;
    result.outputs[0].outputs[0].values[0] = -1000.0F;
    EXPECT_EQ(result.outputs[1].outputs[0].values, second);
}

void ExpectAddSuccess(const ConcurrentExecution& result, const ConcurrentPlan& plan) {
    ASSERT_EQ(result.kind, ConcurrentErrorKind::kNone);
    EXPECT_EQ(result.load_error.kind, InferenceErrorKind::kNone);
    EXPECT_TRUE(result.errors.empty());
    ASSERT_EQ(result.outputs.size(), plan.workers.size());
    for (std::uint32_t index = 0; index < plan.workers.size(); ++index) {
        const auto& output = result.outputs[index];
        EXPECT_EQ(output.worker_index, index);
        EXPECT_EQ(output.completed_requests, plan.requests_per_worker);
        ASSERT_EQ(output.outputs.size(), 1U);
        const NamedTensor& tensor = output.outputs[0];
        EXPECT_EQ(tensor.name, "C");
        ASSERT_EQ(tensor.shape.size(), 1U);
        EXPECT_EQ(tensor.shape[0], 2);
        ASSERT_EQ(tensor.values.size(), 2U);
        ASSERT_NE(tensor.values.data(), nullptr);
        EXPECT_FLOAT_EQ(tensor.values[0], plan.workers[index].expected[0].values[0]);
        EXPECT_FLOAT_EQ(tensor.values[1], plan.workers[index].expected[0].values[1]);
    }
}

void ExpectInferenceFailure(const ConcurrentExecution& result, ConcurrentErrorKind kind) {
    EXPECT_EQ(result.kind, kind);
    EXPECT_TRUE(result.outputs.empty());
    ASSERT_FALSE(result.errors.empty());
    EXPECT_EQ(result.errors.front().kind, kind);
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

bool WithinReference(float actual, float expected) {
    const double difference =
        std::fabs(static_cast<double>(actual) - static_cast<double>(expected));
    const double limit =
        kAbsoluteTolerance + kRelativeTolerance * std::fabs(static_cast<double>(expected));
    return difference <= limit;
}

WorkerRequest MnistWorker(const char* fixture) {
    const std::filesystem::path root{INFEREDGE_MNIST_FIXTURE_DIR};
    const FloatFile image =
        ReadFloats(root / "inputs" / (std::string{fixture} + ".f32"), kMnistInputCount);
    const FloatFile reference =
        ReadFloats(root / "onnx" / (std::string{fixture} + ".f32"), kMnistLogitCount);
    EXPECT_TRUE(image.error.empty()) << image.error;
    EXPECT_TRUE(reference.error.empty()) << reference.error;
    WorkerRequest worker;
    worker.inputs.push_back(
        NamedTensor{"image", {static_cast<std::int64_t>(kMnistInputCount)}, image.values});
    worker.expected.push_back(WorkerExpectedTensor{"logits",
                                                   {static_cast<std::int64_t>(kMnistLogitCount)},
                                                   reference.values,
                                                   kRelativeTolerance,
                                                   kAbsoluteTolerance});
    return worker;
}

void ExpectMnistOutput(const inferedge::WorkerOutput& output, const WorkerRequest& worker) {
    EXPECT_EQ(output.completed_requests, 20U);
    ASSERT_EQ(output.outputs.size(), 1U);
    const NamedTensor& tensor = output.outputs[0];
    EXPECT_EQ(tensor.name, "logits");
    ASSERT_EQ(tensor.shape.size(), 1U);
    EXPECT_EQ(tensor.shape[0], static_cast<std::int64_t>(kMnistLogitCount));
    ASSERT_EQ(tensor.values.size(), kMnistLogitCount);
    ASSERT_NE(tensor.values.data(), nullptr);
    ASSERT_EQ(worker.expected.size(), 1U);
    ASSERT_EQ(worker.expected[0].values.size(), kMnistLogitCount);
    for (std::size_t index = 0; index < kMnistLogitCount; ++index) {
        EXPECT_TRUE(std::isfinite(tensor.values[index]));
        EXPECT_TRUE(WithinReference(tensor.values[index], worker.expected[0].values[index]))
            << "logit " << index;
    }
}

void ExpectOneAdd(InferenceEngine& engine) {
    const NamedTensor first{"A", {2}, {1.0F, 2.0F}};
    const NamedTensor second{"B", {2}, {3.0F, 4.0F}};
    const std::vector<NamedTensorView> views{
        NamedTensorView{first.name, first.shape, std::span<const float>{first.values}},
        NamedTensorView{second.name, second.shape, std::span<const float>{second.values}},
    };
    const InferenceResult result = engine.run(views);
    ASSERT_EQ(result.error.kind, InferenceErrorKind::kNone) << result.error.message;
    ASSERT_EQ(result.outputs.size(), 1U);
    EXPECT_EQ(result.outputs[0].name, "C");
    ASSERT_EQ(result.outputs[0].shape.size(), 1U);
    EXPECT_EQ(result.outputs[0].shape[0], 2);
    ASSERT_EQ(result.outputs[0].values.size(), 2U);
    EXPECT_FLOAT_EQ(result.outputs[0].values[0], 4.0F);
    EXPECT_FLOAT_EQ(result.outputs[0].values[1], 6.0F);
}

void ExpectOneMnist(InferenceEngine& engine, const WorkerRequest& worker) {
    ASSERT_EQ(worker.inputs.size(), 1U);
    const NamedTensor& image = worker.inputs[0];
    const std::vector<NamedTensorView> views{
        NamedTensorView{image.name, image.shape, std::span<const float>{image.values}},
    };
    const InferenceResult result = engine.run(views);
    ASSERT_EQ(result.error.kind, InferenceErrorKind::kNone) << result.error.message;
    ASSERT_EQ(result.outputs.size(), 1U);
    EXPECT_EQ(result.outputs[0].name, "logits");
    ASSERT_EQ(result.outputs[0].shape.size(), 1U);
    EXPECT_EQ(result.outputs[0].shape[0], static_cast<std::int64_t>(kMnistLogitCount));
    ASSERT_EQ(result.outputs[0].values.size(), worker.expected[0].values.size());
    for (std::size_t index = 0; index < result.outputs[0].values.size(); ++index) {
        EXPECT_TRUE(std::isfinite(result.outputs[0].values[index]));
        EXPECT_TRUE(
            WithinReference(result.outputs[0].values[index], worker.expected[0].values[index]));
    }
}

ConcurrentBenchmarkConfig AddBenchmarkConfig(std::uint32_t workers, std::uint32_t warmup,
                                             std::uint32_t measured) {
    ConcurrentBenchmarkConfig config;
    config.worker_count = workers;
    config.warmup_iterations = warmup;
    config.measured_iterations = measured;
    config.correctness_check_interval = 1;
    config.workload = "add";
    config.model = "add_f32";
    return config;
}

ConcurrentBenchmarkAssignment AddAssignment(float b0, float b1) {
    ConcurrentBenchmarkAssignment assignment;
    assignment.inputs.push_back(NamedTensor{"A", {2}, {1.0F, 2.0F}});
    assignment.inputs.push_back(NamedTensor{"B", {2}, {b0, b1}});
    assignment.expected.push_back(ExpectAdd(1.0F + b0, 2.0F + b1));
    return assignment;
}

std::vector<ConcurrentBenchmarkAssignment> AddAssignments(std::uint32_t workers) {
    std::vector<ConcurrentBenchmarkAssignment> assignments;
    assignments.reserve(workers);
    for (std::uint32_t index = 0; index < workers; ++index) {
        const float offset = static_cast<float>(index);
        assignments.push_back(AddAssignment(3.0F + offset, 4.0F + offset));
    }
    return assignments;
}

ConcurrentBenchmarkAssignment MnistAssignment(const char* fixture) {
    const WorkerRequest worker = MnistWorker(fixture);
    ConcurrentBenchmarkAssignment assignment;
    assignment.inputs = worker.inputs;
    assignment.expected = worker.expected;
    return assignment;
}

std::vector<ConcurrentBenchmarkAssignment> MnistAssignments(std::uint32_t workers) {
    std::vector<ConcurrentBenchmarkAssignment> assignments;
    assignments.reserve(workers);
    for (std::uint32_t index = 0; index < workers; ++index) {
        const char* fixture = index == 1U ? kSecondMnistFixture : kPrimaryMnistFixture;
        assignments.push_back(MnistAssignment(fixture));
    }
    return assignments;
}

void ExpectDurationReport(const ConcurrentBenchmarkRun& run,
                          const ConcurrentBenchmarkConfig& config) {
    ASSERT_EQ(run.error.kind, ConcurrentBenchmarkErrorKind::kNone) << run.error.message;
    EXPECT_EQ(run.error.phase, ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.load_error.kind, InferenceErrorKind::kNone);
    const auto& report = run.report;
    EXPECT_EQ(report.worker_count, config.worker_count);
    EXPECT_EQ(report.warmup_iterations, config.warmup_iterations);
    EXPECT_EQ(report.measured_iterations, config.measured_iterations);
    EXPECT_EQ(report.correctness_check_interval, config.correctness_check_interval);
    const std::uint64_t total = static_cast<std::uint64_t>(config.worker_count) *
                                static_cast<std::uint64_t>(config.measured_iterations);
    EXPECT_EQ(report.total_completed_requests, total);
    ASSERT_EQ(report.per_worker_completed_requests.size(), config.worker_count);
    ASSERT_EQ(report.per_worker_call_durations.size(), config.worker_count);
    std::uint64_t sample_sum = 0;
    for (std::uint32_t index = 0; index < config.worker_count; ++index) {
        EXPECT_EQ(report.per_worker_completed_requests[index], config.measured_iterations);
        EXPECT_EQ(report.per_worker_call_durations[index].size(), config.measured_iterations);
        sample_sum += report.per_worker_call_durations[index].size();
    }
    EXPECT_EQ(sample_sum, total);
    EXPECT_EQ(report.call_summary.sample_count, total);
}

void ExpectEmptyReport(const ConcurrentBenchmarkRun& run) {
    EXPECT_EQ(run.report.worker_count, 0U);
    EXPECT_EQ(run.report.total_completed_requests, 0U);
    EXPECT_TRUE(run.report.per_worker_completed_requests.empty());
    EXPECT_TRUE(run.report.per_worker_call_durations.empty());
    EXPECT_EQ(run.report.call_summary.sample_count, 0U);
}

void ExpectBenchmarkFailure(const ConcurrentBenchmarkRun& run, ConcurrentBenchmarkErrorKind kind,
                            ConcurrentBenchmarkPhase phase) {
    EXPECT_EQ(run.error.kind, kind) << run.error.message;
    EXPECT_EQ(run.error.phase, phase);
    ExpectEmptyReport(run);
}

} // namespace

TEST_F(ConcurrentReliabilityTest, SustainedAddOneWorker) {
    ConcurrentPlan plan = AddPlan(1, 100, true);
    const std::vector<std::vector<float>> inputs = InputSnapshot(plan);
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectAddSuccess(result, plan);
    ExpectInputsUnchanged(plan, inputs);
    ASSERT_EQ(result.outputs.size(), 1U);
    EXPECT_EQ(result.outputs[0].outputs.size(), 1U);
}

TEST_F(ConcurrentReliabilityTest, SustainedAddEightWorkers) {
    ConcurrentPlan plan = AddPlan(8, 100, false);
    const std::vector<std::vector<float>> inputs = InputSnapshot(plan);
    ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectAddSuccess(result, plan);
    ExpectInputsUnchanged(plan, inputs);
    ExpectIndependentOutputs(result);
}

TEST_F(ConcurrentReliabilityTest, SustainedMnistFourWorkers) {
    ConcurrentPlan plan;
    plan.requests_per_worker = 20;
    plan.workers.push_back(MnistWorker(kPrimaryMnistFixture));
    plan.workers.push_back(MnistWorker(kSecondMnistFixture));
    plan.workers.push_back(MnistWorker(kPrimaryMnistFixture));
    plan.workers.push_back(MnistWorker(kPrimaryMnistFixture));
    ASSERT_EQ(plan.workers.size(), 4U);
    ASSERT_NE(plan.workers[0].expected[0].values, plan.workers[1].expected[0].values);
    const std::vector<std::vector<float>> inputs = InputSnapshot(plan);
    ConcurrentExecution result = run_concurrent_inference(INFEREDGE_MNIST_MODEL_PATH, plan);
    ASSERT_EQ(result.kind, ConcurrentErrorKind::kNone);
    EXPECT_TRUE(result.errors.empty());
    ASSERT_EQ(result.outputs.size(), 4U);
    for (std::uint32_t index = 0; index < 4U; ++index) {
        EXPECT_EQ(result.outputs[index].worker_index, index);
        ExpectMnistOutput(result.outputs[index], plan.workers[index]);
    }
    EXPECT_NE(result.outputs[0].outputs[0].values, result.outputs[1].outputs[0].values);
    ExpectInputsUnchanged(plan, inputs);
    ExpectIndependentOutputs(result);
}

TEST_F(ConcurrentReliabilityTest, ReturnedOutputsOutliveThePlan) {
    ConcurrentPlan plan = AddPlan(2, 5, false);
    ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectAddSuccess(result, plan);
    ExpectIndependentOutputs(result);
    plan = ConcurrentPlan{};
    ASSERT_EQ(result.outputs.size(), 2U);
    EXPECT_FLOAT_EQ(result.outputs[0].outputs[0].values[0], -1000.0F);
    EXPECT_FLOAT_EQ(result.outputs[0].outputs[0].values[1], 6.0F);
    EXPECT_FLOAT_EQ(result.outputs[1].outputs[0].values[0], 5.0F);
    EXPECT_FLOAT_EQ(result.outputs[1].outputs[0].values[1], 7.0F);
    EXPECT_EQ(result.outputs[0].outputs[0].shape, (std::vector<std::int64_t>{2}));
    EXPECT_EQ(result.outputs[1].outputs[0].shape, (std::vector<std::int64_t>{2}));
}

TEST_F(ConcurrentReliabilityTest, AddEngineLoadRunDestroyCycles) {
    for (int cycle = 0; cycle < 16; ++cycle) {
        EngineLoad loaded = load_inference_engine(INFEREDGE_ADD_MODEL_PATH);
        ASSERT_EQ(loaded.error.kind, InferenceErrorKind::kNone) << loaded.error.message;
        ASSERT_NE(loaded.engine, nullptr);
        ExpectOneAdd(*loaded.engine);
        loaded.engine.reset();
    }
}

TEST_F(ConcurrentReliabilityTest, MnistEngineLoadRunDestroyCycles) {
    const WorkerRequest worker = MnistWorker(kPrimaryMnistFixture);
    for (int cycle = 0; cycle < 4; ++cycle) {
        EngineLoad loaded = load_inference_engine(INFEREDGE_MNIST_MODEL_PATH);
        ASSERT_EQ(loaded.error.kind, InferenceErrorKind::kNone) << loaded.error.message;
        ASSERT_NE(loaded.engine, nullptr);
        ExpectOneMnist(*loaded.engine, worker);
        loaded.engine.reset();
    }
}

TEST_F(ConcurrentReliabilityTest, RepeatedPathOverloadLoadsEngines) {
    const ConcurrentPlan plan = AddPlan(4, 10, false);
    const std::vector<std::vector<float>> inputs = InputSnapshot(plan);
    for (int cycle = 0; cycle < 8; ++cycle) {
        const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
        ExpectAddSuccess(result, plan);
    }
    ExpectInputsUnchanged(plan, inputs);
}

TEST_F(ConcurrentReliabilityTest, MovedEnginesLeaveOwnedOutputs) {
    constexpr std::uint32_t kWorkers = 4;
    std::vector<std::unique_ptr<InferenceEngine>> engines;
    engines.reserve(kWorkers);
    for (std::uint32_t index = 0; index < kWorkers; ++index) {
        EngineLoad loaded = load_inference_engine(INFEREDGE_ADD_MODEL_PATH);
        ASSERT_EQ(loaded.error.kind, InferenceErrorKind::kNone) << loaded.error.message;
        ASSERT_NE(loaded.engine, nullptr);
        engines.push_back(std::move(loaded.engine));
    }
    ConcurrentPlan plan = AddPlan(kWorkers, 10, false);
    ConcurrentExecution result = run_concurrent_inference(std::move(engines), plan);
    EXPECT_TRUE(engines.empty());
    ExpectAddSuccess(result, plan);
    plan = ConcurrentPlan{};
    ASSERT_EQ(result.outputs.size(), kWorkers);
    EXPECT_FLOAT_EQ(result.outputs[0].outputs[0].values[0], 4.0F);
    EXPECT_FLOAT_EQ(result.outputs[0].outputs[0].values[1], 6.0F);
    EXPECT_FLOAT_EQ(result.outputs[3].outputs[0].values[0], 7.0F);
    EXPECT_FLOAT_EQ(result.outputs[3].outputs[0].values[1], 9.0F);
}

TEST_F(ConcurrentReliabilityTest, SustainedAddBenchmarkReport) {
    const ConcurrentBenchmarkConfig config = AddBenchmarkConfig(8, 1, 20);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(8);
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectDurationReport(run, config);
}

TEST_F(ConcurrentReliabilityTest, SustainedMnistBenchmarkReport) {
    const ConcurrentBenchmarkConfig config = AddBenchmarkConfig(4, 1, 10);
    ConcurrentBenchmarkConfig mnist = config;
    mnist.workload = "mnist";
    mnist.model = "mnist_mlp";
    const std::vector<ConcurrentBenchmarkAssignment> assignments = MnistAssignments(4);
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_MNIST_MODEL_PATH, assignments, mnist);
    ExpectDurationReport(run, mnist);
}

TEST_F(ConcurrentReliabilityTest, BenchmarkReportsOwnSeparateSamples) {
    const ConcurrentBenchmarkConfig config = AddBenchmarkConfig(2, 1, 8);
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(2);
    ConcurrentBenchmarkRun first =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectDurationReport(first, config);
    const auto first_samples = first.report.per_worker_call_durations;
    const std::uint64_t first_count = first.report.call_summary.sample_count;
    ConcurrentBenchmarkRun second =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectDurationReport(second, config);
    EXPECT_EQ(first.report.per_worker_call_durations, first_samples);
    EXPECT_EQ(first.report.call_summary.sample_count, first_count);
    ASSERT_EQ(first.report.per_worker_call_durations.size(),
              second.report.per_worker_call_durations.size());
    for (std::size_t index = 0; index < first.report.per_worker_call_durations.size(); ++index) {
        ASSERT_FALSE(first.report.per_worker_call_durations[index].empty());
        EXPECT_NE(first.report.per_worker_call_durations[index].data(),
                  second.report.per_worker_call_durations[index].data());
    }
}

TEST_F(ConcurrentReliabilityTest, PartialStartDuringLongAddInference) {
    ConcurrentPlan plan = AddPlan(8, 100, true);
    plan.stop_after_starting_workers = 4;
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectInferenceFailure(result, ConcurrentErrorKind::kWorkerStart);
    EXPECT_EQ(result.errors.front().worker_index, 4U);
    EXPECT_EQ(result.errors.front().request_iteration, 0U);
    EXPECT_NE(result.errors.front().message.find("not started"), std::string::npos);
}

TEST_F(ConcurrentReliabilityTest, PartialStartDuringAddBenchmark) {
    ConcurrentBenchmarkConfig config = AddBenchmarkConfig(8, 1, 20);
    config.stop_after_starting_workers = 4;
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(8);
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectBenchmarkFailure(run, ConcurrentBenchmarkErrorKind::kWorkerStart,
                           ConcurrentBenchmarkPhase::kNone);
    EXPECT_EQ(run.error.worker_index, 4U);
    EXPECT_EQ(run.error.message, "worker thread was not started");
}

TEST_F(ConcurrentReliabilityTest, WarmupInjectionCancelsBeforeMeasurement) {
    ConcurrentBenchmarkConfig config = AddBenchmarkConfig(8, 1, 20);
    config.fail_warmup_worker = 3;
    const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(8);
    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config);
    ExpectBenchmarkFailure(run, ConcurrentBenchmarkErrorKind::kInferenceFailed,
                           ConcurrentBenchmarkPhase::kWarmup);
    EXPECT_EQ(run.error.worker_index, 3U);
    EXPECT_EQ(run.error.request_iteration, 0U);
    EXPECT_EQ(run.error.message, "injected warmup failure");
}

TEST_F(ConcurrentReliabilityTest, InferenceMismatchThenSuccess) {
    ConcurrentPlan failing = AddPlan(4, 10, true);
    failing.workers[2].expected[0].values = {0.0F, 0.0F};
    const ConcurrentExecution failed = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, failing);
    ExpectInferenceFailure(failed, ConcurrentErrorKind::kOutputMismatch);
    bool found = false;
    for (const auto& error : failed.errors) {
        if (error.worker_index == 2U) {
            found = true;
            EXPECT_EQ(error.kind, ConcurrentErrorKind::kOutputMismatch);
            EXPECT_NE(error.message.find("outside tolerance"), std::string::npos);
        }
    }
    EXPECT_TRUE(found);

    const ConcurrentPlan plan = AddPlan(4, 10, false);
    const ConcurrentExecution succeeded = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectAddSuccess(succeeded, plan);
    EXPECT_TRUE(failed.outputs.empty());
}

TEST_F(ConcurrentReliabilityTest, MnistMeasuredInjectionThenSuccess) {
    ConcurrentBenchmarkConfig failing = AddBenchmarkConfig(2, 1, 2);
    failing.workload = "mnist";
    failing.model = "mnist_mlp";
    failing.fail_measured_worker = 1;
    failing.fail_measured_iteration = 1;
    const std::vector<ConcurrentBenchmarkAssignment> assignments = MnistAssignments(2);
    const ConcurrentBenchmarkRun failed =
        run_concurrent_benchmark(INFEREDGE_MNIST_MODEL_PATH, assignments, failing);
    ExpectBenchmarkFailure(failed, ConcurrentBenchmarkErrorKind::kInferenceFailed,
                           ConcurrentBenchmarkPhase::kMeasured);
    EXPECT_EQ(failed.error.worker_index, 1U);
    EXPECT_EQ(failed.error.request_iteration, 1U);
    EXPECT_EQ(failed.error.message, "injected measured-loop failure");

    ConcurrentBenchmarkConfig config = AddBenchmarkConfig(2, 1, 2);
    config.workload = "mnist";
    config.model = "mnist_mlp";
    const ConcurrentBenchmarkRun succeeded =
        run_concurrent_benchmark(INFEREDGE_MNIST_MODEL_PATH, assignments, config);
    ExpectDurationReport(succeeded, config);
}

TEST_F(ConcurrentReliabilityTest, InjectedExceptionThenSuccess) {
    ConcurrentPlan failing = AddPlan(4, 10, true);
    failing.throw_in_worker = 2;
    const ConcurrentExecution failed = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, failing);
    ExpectInferenceFailure(failed, ConcurrentErrorKind::kWorkerException);
    bool found = false;
    for (const auto& error : failed.errors) {
        if (error.worker_index == 2U) {
            found = true;
            EXPECT_NE(error.message.find("injected worker exception"), std::string::npos);
        }
    }
    EXPECT_TRUE(found);
    EXPECT_TRUE(failed.outputs.empty());

    const ConcurrentPlan plan = AddPlan(4, 10, false);
    const ConcurrentExecution succeeded = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectAddSuccess(succeeded, plan);
    EXPECT_TRUE(failed.outputs.empty());
}
