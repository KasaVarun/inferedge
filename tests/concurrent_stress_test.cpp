#include "inferedge/concurrent_benchmark.hpp"
#include "inferedge/concurrent_inference.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
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

using inferedge::ConcurrentBenchmarkAssignment;
using inferedge::ConcurrentBenchmarkConfig;
using inferedge::ConcurrentBenchmarkErrorKind;
using inferedge::ConcurrentBenchmarkRun;
using inferedge::ConcurrentErrorKind;
using inferedge::ConcurrentExecution;
using inferedge::ConcurrentPlan;
using inferedge::InferenceErrorKind;
using inferedge::NamedTensor;
using inferedge::run_concurrent_benchmark;
using inferedge::run_concurrent_inference;
using inferedge::WorkerExpectedTensor;
using inferedge::WorkerRequest;

constexpr std::size_t kMnistInputCount = 784;
constexpr std::size_t kMnistLogitCount = 10;
constexpr double kRelativeTolerance = 1e-4;
constexpr double kAbsoluteTolerance = 1e-5;
constexpr std::uint32_t kWorkerCounts[] = {1U, 2U, 4U, 8U};
constexpr const char* kMnistIds[] = {
    "test_00003", "test_00002", "test_00001", "test_00018",
    "test_00004", "test_00008", "test_00011", "test_00000",
};

class ConcurrentStressTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

WorkerRequest AddWorker(float b0, float b1) {
    WorkerRequest worker;
    worker.inputs.push_back(NamedTensor{"A", {2}, {1.0F, 2.0F}});
    worker.inputs.push_back(NamedTensor{"B", {2}, {b0, b1}});
    worker.expected.push_back(WorkerExpectedTensor{"C", {2}, {1.0F + b0, 2.0F + b1}, 0.0, 0.0});
    return worker;
}

ConcurrentPlan AddPlan(std::uint32_t workers, std::uint32_t requests) {
    ConcurrentPlan plan;
    plan.requests_per_worker = requests;
    plan.workers.reserve(workers);
    for (std::uint32_t index = 0; index < workers; ++index) {
        const float offset = static_cast<float>(index);
        plan.workers.push_back(AddWorker(3.0F + offset, 4.0F + offset));
    }
    return plan;
}

void ExpectAdd(const ConcurrentExecution& result, const ConcurrentPlan& plan) {
    ASSERT_EQ(result.kind, ConcurrentErrorKind::kNone);
    EXPECT_TRUE(result.errors.empty());
    ASSERT_EQ(result.outputs.size(), plan.workers.size());
    for (std::uint32_t index = 0; index < plan.workers.size(); ++index) {
        const auto& output = result.outputs[index];
        EXPECT_EQ(output.completed_requests, plan.requests_per_worker);
        ASSERT_EQ(output.outputs.size(), 1U);
        ASSERT_EQ(output.outputs[0].values.size(), 2U);
        EXPECT_FLOAT_EQ(output.outputs[0].values[0], plan.workers[index].expected[0].values[0]);
        EXPECT_FLOAT_EQ(output.outputs[0].values[1], plan.workers[index].expected[0].values[1]);
        EXPECT_EQ(plan.workers[index].inputs[0].values, (std::vector<float>{1.0F, 2.0F}));
    }
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

ConcurrentPlan MnistPlan(std::uint32_t workers, std::uint32_t requests) {
    ConcurrentPlan plan;
    plan.requests_per_worker = requests;
    plan.workers.reserve(workers);
    for (std::uint32_t index = 0; index < workers; ++index) {
        plan.workers.push_back(MnistWorker(kMnistIds[index]));
    }
    return plan;
}

void ExpectMnist(const ConcurrentExecution& result, const ConcurrentPlan& plan) {
    ASSERT_EQ(result.kind, ConcurrentErrorKind::kNone);
    EXPECT_TRUE(result.errors.empty());
    ASSERT_EQ(result.outputs.size(), plan.workers.size());
    for (std::uint32_t index = 0; index < plan.workers.size(); ++index) {
        const auto& output = result.outputs[index];
        EXPECT_EQ(output.completed_requests, plan.requests_per_worker);
        ASSERT_EQ(output.outputs.size(), 1U);
        ASSERT_EQ(output.outputs[0].values.size(), plan.workers[index].expected[0].values.size());
        EXPECT_EQ(plan.workers[index].inputs[0].values.size(), kMnistInputCount);
        for (std::size_t value = 0; value < output.outputs[0].values.size(); ++value) {
            EXPECT_TRUE(std::isfinite(output.outputs[0].values[value]));
            EXPECT_TRUE(WithinReference(output.outputs[0].values[value],
                                        plan.workers[index].expected[0].values[value]));
        }
    }
}

ConcurrentBenchmarkConfig BenchmarkConfig(std::uint32_t workers, const char* workload,
                                          const char* model) {
    ConcurrentBenchmarkConfig config;
    config.worker_count = workers;
    config.warmup_iterations = 2;
    config.measured_iterations = 100;
    config.correctness_check_interval = 1;
    config.workload = workload;
    config.model = model;
    return config;
}

std::vector<ConcurrentBenchmarkAssignment> AddAssignments(std::uint32_t workers) {
    std::vector<ConcurrentBenchmarkAssignment> assignments;
    assignments.reserve(workers);
    for (std::uint32_t index = 0; index < workers; ++index) {
        const float offset = static_cast<float>(index);
        const WorkerRequest worker = AddWorker(3.0F + offset, 4.0F + offset);
        ConcurrentBenchmarkAssignment assignment;
        assignment.inputs = worker.inputs;
        assignment.expected = worker.expected;
        assignments.push_back(std::move(assignment));
    }
    return assignments;
}

std::vector<ConcurrentBenchmarkAssignment> MnistAssignments(std::uint32_t workers) {
    std::vector<ConcurrentBenchmarkAssignment> assignments;
    assignments.reserve(workers);
    for (std::uint32_t index = 0; index < workers; ++index) {
        const WorkerRequest worker = MnistWorker(kMnistIds[index]);
        ConcurrentBenchmarkAssignment assignment;
        assignment.inputs = worker.inputs;
        assignment.expected = worker.expected;
        assignments.push_back(std::move(assignment));
    }
    return assignments;
}

void ExpectBenchmark(const ConcurrentBenchmarkRun& run, const ConcurrentBenchmarkConfig& config) {
    ASSERT_EQ(run.error.kind, ConcurrentBenchmarkErrorKind::kNone) << run.error.message;
    EXPECT_EQ(run.load_error.kind, InferenceErrorKind::kNone);
    const std::uint64_t total = static_cast<std::uint64_t>(config.worker_count) *
                                static_cast<std::uint64_t>(config.measured_iterations);
    EXPECT_EQ(run.report.total_completed_requests, total);
    ASSERT_EQ(run.report.per_worker_completed_requests.size(), config.worker_count);
    ASSERT_EQ(run.report.per_worker_call_durations.size(), config.worker_count);
    std::uint64_t samples = 0;
    for (std::uint32_t index = 0; index < config.worker_count; ++index) {
        EXPECT_EQ(run.report.per_worker_completed_requests[index], config.measured_iterations);
        EXPECT_EQ(run.report.per_worker_call_durations[index].size(), config.measured_iterations);
        samples += run.report.per_worker_call_durations[index].size();
    }
    EXPECT_EQ(samples, total);
    EXPECT_EQ(run.report.call_summary.sample_count, total);
}

} // namespace

TEST_F(ConcurrentStressTest, AddInferenceAcrossWorkerCounts) {
    for (const std::uint32_t workers : kWorkerCounts) {
        SCOPED_TRACE(workers);
        const ConcurrentPlan plan = AddPlan(workers, 1000);
        ExpectAdd(run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan), plan);
    }
}

TEST_F(ConcurrentStressTest, MnistInferenceAcrossWorkerCounts) {
    for (const std::uint32_t workers : kWorkerCounts) {
        SCOPED_TRACE(workers);
        const ConcurrentPlan plan = MnistPlan(workers, 100);
        ExpectMnist(run_concurrent_inference(INFEREDGE_MNIST_MODEL_PATH, plan), plan);
    }
}

TEST_F(ConcurrentStressTest, AddBenchmarkAcrossWorkerCounts) {
    for (const std::uint32_t workers : kWorkerCounts) {
        SCOPED_TRACE(workers);
        const ConcurrentBenchmarkConfig config = BenchmarkConfig(workers, "add", "add_f32");
        const std::vector<ConcurrentBenchmarkAssignment> assignments = AddAssignments(workers);
        ExpectBenchmark(run_concurrent_benchmark(INFEREDGE_ADD_MODEL_PATH, assignments, config),
                        config);
    }
}

TEST_F(ConcurrentStressTest, MnistBenchmarkAcrossWorkerCounts) {
    for (const std::uint32_t workers : kWorkerCounts) {
        SCOPED_TRACE(workers);
        const ConcurrentBenchmarkConfig config = BenchmarkConfig(workers, "mnist", "mnist_mlp");
        const std::vector<ConcurrentBenchmarkAssignment> assignments = MnistAssignments(workers);
        ExpectBenchmark(run_concurrent_benchmark(INFEREDGE_MNIST_MODEL_PATH, assignments, config),
                        config);
    }
}
