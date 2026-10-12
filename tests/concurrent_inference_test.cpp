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

using inferedge::ConcurrentErrorKind;
using inferedge::ConcurrentExecution;
using inferedge::ConcurrentPlan;
using inferedge::EngineLoad;
using inferedge::InferenceEngine;
using inferedge::InferenceErrorKind;
using inferedge::load_inference_engine;
using inferedge::NamedTensor;
using inferedge::run_concurrent_inference;
using inferedge::WorkerExpectedTensor;
using inferedge::WorkerRequest;

constexpr std::size_t kMnistInputCount = 784;
constexpr std::size_t kMnistLogitCount = 10;
constexpr double kRelativeTolerance = 1e-4;
constexpr double kAbsoluteTolerance = 1e-5;
constexpr const char* kMnistIds[] = {
    "test_00003", "test_00002", "test_00001", "test_00018",
    "test_00004", "test_00008", "test_00011", "test_00000",
};

class ConcurrentInferenceTest : public testing::Test {
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

void ExpectAddSuccess(const ConcurrentExecution& result, const ConcurrentPlan& plan) {
    EXPECT_EQ(result.kind, ConcurrentErrorKind::kNone);
    EXPECT_EQ(result.load_error.kind, InferenceErrorKind::kNone);
    EXPECT_TRUE(result.errors.empty());
    ASSERT_EQ(result.outputs.size(), plan.workers.size());
    for (std::uint32_t index = 0; index < plan.workers.size(); ++index) {
        const auto& output = result.outputs[index];
        EXPECT_EQ(output.worker_index, index);
        EXPECT_EQ(output.completed_requests, plan.requests_per_worker);
        ASSERT_EQ(output.outputs.size(), 1U);
        EXPECT_EQ(output.outputs[0].name, "C");
        ASSERT_EQ(output.outputs[0].values.size(), 2U);
        EXPECT_FLOAT_EQ(output.outputs[0].values[0], plan.workers[index].expected[0].values[0]);
        EXPECT_FLOAT_EQ(output.outputs[0].values[1], plan.workers[index].expected[0].values[1]);
    }
}

void ExpectFailure(const ConcurrentExecution& result, ConcurrentErrorKind kind) {
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

ConcurrentPlan MnistPlan(std::uint32_t workers, bool identical) {
    ConcurrentPlan plan;
    plan.requests_per_worker = 2;
    for (std::uint32_t index = 0; index < workers; ++index) {
        const char* fixture = identical ? kMnistIds[0] : kMnistIds[index];
        plan.workers.push_back(MnistWorker(fixture));
    }
    return plan;
}

bool WithinReference(float actual, float expected) {
    const double difference =
        std::fabs(static_cast<double>(actual) - static_cast<double>(expected));
    const double limit =
        kAbsoluteTolerance + kRelativeTolerance * std::fabs(static_cast<double>(expected));
    return difference <= limit;
}

void ExpectMnistSuccess(const ConcurrentExecution& result, const ConcurrentPlan& plan) {
    EXPECT_EQ(result.kind, ConcurrentErrorKind::kNone);
    EXPECT_TRUE(result.errors.empty());
    ASSERT_EQ(result.outputs.size(), plan.workers.size());
    for (std::uint32_t index = 0; index < plan.workers.size(); ++index) {
        const auto& output = result.outputs[index];
        EXPECT_EQ(output.completed_requests, plan.requests_per_worker);
        ASSERT_EQ(output.outputs.size(), 1U);
        EXPECT_EQ(output.outputs[0].name, "logits");
        ASSERT_EQ(output.outputs[0].shape.size(), 1U);
        EXPECT_EQ(output.outputs[0].shape[0], static_cast<std::int64_t>(kMnistLogitCount));
        ASSERT_EQ(output.outputs[0].values.size(), kMnistLogitCount);
        const auto& expected = plan.workers[index].expected[0].values;
        ASSERT_EQ(expected.size(), kMnistLogitCount);
        for (std::size_t value = 0; value < kMnistLogitCount; ++value) {
            EXPECT_TRUE(std::isfinite(output.outputs[0].values[value]));
            EXPECT_TRUE(WithinReference(output.outputs[0].values[value], expected[value]))
                << "worker " << index << " logit " << value;
        }
    }
}

} // namespace

TEST_F(ConcurrentInferenceTest, OneAddWorker) {
    const ConcurrentPlan plan = AddPlan(1, 1, true);
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectAddSuccess(result, plan);
    EXPECT_FLOAT_EQ(result.outputs[0].outputs[0].values[0], 4.0F);
    EXPECT_FLOAT_EQ(result.outputs[0].outputs[0].values[1], 6.0F);
}

TEST_F(ConcurrentInferenceTest, TwoAddWorkers) {
    const ConcurrentPlan plan = AddPlan(2, 1, true);
    ExpectAddSuccess(run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan), plan);
}

TEST_F(ConcurrentInferenceTest, FourAddWorkers) {
    const ConcurrentPlan plan = AddPlan(4, 1, true);
    ExpectAddSuccess(run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan), plan);
}

TEST_F(ConcurrentInferenceTest, EightAddWorkers) {
    const ConcurrentPlan plan = AddPlan(8, 1, true);
    ExpectAddSuccess(run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan), plan);
}

TEST_F(ConcurrentInferenceTest, IdenticalAddInputsUseSeparateBuffers) {
    ConcurrentPlan plan = AddPlan(4, 1, true);
    ASSERT_NE(plan.workers[0].inputs[0].values.data(), plan.workers[1].inputs[0].values.data());
    ExpectAddSuccess(run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan), plan);
}

TEST_F(ConcurrentInferenceTest, DifferentAddInputsStayWithTheirWorker) {
    const ConcurrentPlan plan = AddPlan(8, 1, false);
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectAddSuccess(result, plan);
    for (std::uint32_t index = 1; index < plan.workers.size(); ++index) {
        EXPECT_NE(result.outputs[index].outputs[0].values[0],
                  result.outputs[0].outputs[0].values[0]);
    }
}

TEST_F(ConcurrentInferenceTest, RepeatedAddCalls) {
    const ConcurrentPlan plan = AddPlan(4, 5, true);
    ExpectAddSuccess(run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan), plan);
}

TEST_F(ConcurrentInferenceTest, OneMnistWorkerMatchesReferenceLogits) {
    const ConcurrentPlan plan = MnistPlan(1, true);
    ExpectMnistSuccess(run_concurrent_inference(INFEREDGE_MNIST_MODEL_PATH, plan), plan);
}

TEST_F(ConcurrentInferenceTest, TwoMnistWorkers) {
    const ConcurrentPlan plan = MnistPlan(2, true);
    ExpectMnistSuccess(run_concurrent_inference(INFEREDGE_MNIST_MODEL_PATH, plan), plan);
}

TEST_F(ConcurrentInferenceTest, FourMnistWorkers) {
    const ConcurrentPlan plan = MnistPlan(4, true);
    ExpectMnistSuccess(run_concurrent_inference(INFEREDGE_MNIST_MODEL_PATH, plan), plan);
}

TEST_F(ConcurrentInferenceTest, EightMnistWorkers) {
    const ConcurrentPlan plan = MnistPlan(8, true);
    ExpectMnistSuccess(run_concurrent_inference(INFEREDGE_MNIST_MODEL_PATH, plan), plan);
}

TEST_F(ConcurrentInferenceTest, IdenticalMnistFixtureInputs) {
    const ConcurrentPlan plan = MnistPlan(4, true);
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_MNIST_MODEL_PATH, plan);
    ExpectMnistSuccess(result, plan);
    ASSERT_GE(result.outputs.size(), 2U);
    EXPECT_EQ(result.outputs[0].outputs[0].values, result.outputs[1].outputs[0].values);
}

TEST_F(ConcurrentInferenceTest, DifferentMnistFixturesStayWithTheirWorker) {
    const ConcurrentPlan plan = MnistPlan(8, false);
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_MNIST_MODEL_PATH, plan);
    ExpectMnistSuccess(result, plan);
    EXPECT_NE(result.outputs[0].outputs[0].values, result.outputs[1].outputs[0].values);
}

TEST_F(ConcurrentInferenceTest, OutputBuffersAreIndependent) {
    const ConcurrentPlan plan = AddPlan(2, 1, true);
    ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ASSERT_EQ(result.kind, ConcurrentErrorKind::kNone);
    ASSERT_EQ(result.outputs.size(), 2U);
    float* first = result.outputs[0].outputs[0].values.data();
    float* second = result.outputs[1].outputs[0].values.data();
    ASSERT_NE(first, second);
    first[0] = 100.0F;
    EXPECT_FLOAT_EQ(second[0], 4.0F);
}

TEST_F(ConcurrentInferenceTest, OutputsOutliveTheEngines) {
    ConcurrentPlan plan = AddPlan(2, 1, false);
    ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ASSERT_EQ(result.kind, ConcurrentErrorKind::kNone);
    plan = ConcurrentPlan{};
    ASSERT_EQ(result.outputs.size(), 2U);
    EXPECT_FLOAT_EQ(result.outputs[0].outputs[0].values[0], 4.0F);
    EXPECT_FLOAT_EQ(result.outputs[0].outputs[0].values[1], 6.0F);
    EXPECT_FLOAT_EQ(result.outputs[1].outputs[0].values[0], 5.0F);
    EXPECT_FLOAT_EQ(result.outputs[1].outputs[0].values[1], 7.0F);
}

TEST_F(ConcurrentInferenceTest, InvalidInputShapeFailsTheRun) {
    ConcurrentPlan plan = AddPlan(4, 3, true);
    plan.workers[2].inputs[0] = NamedTensor{"A", {3}, {1.0F, 2.0F, 3.0F}};
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectFailure(result, ConcurrentErrorKind::kInferenceFailed);
    bool found = false;
    for (const auto& error : result.errors) {
        if (error.worker_index == 2) {
            found = true;
            EXPECT_EQ(error.request_iteration, 0U);
            EXPECT_EQ(error.inference_kind, InferenceErrorKind::kShapeMismatch);
            EXPECT_FALSE(error.message.empty());
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(ConcurrentInferenceTest, MissingInputFailsTheRun) {
    ConcurrentPlan plan = AddPlan(2, 2, true);
    plan.workers[0].inputs.pop_back();
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectFailure(result, ConcurrentErrorKind::kInferenceFailed);
    ASSERT_FALSE(result.errors.empty());
    EXPECT_EQ(result.errors.front().worker_index, 0U);
    EXPECT_EQ(result.errors.front().inference_kind, InferenceErrorKind::kMissingInput);
    EXPECT_NE(result.errors.front().message.find("missing"), std::string::npos);
}

TEST_F(ConcurrentInferenceTest, InferenceErrorIsPreserved) {
    ConcurrentPlan plan = AddPlan(1, 1, true);
    plan.workers[0].inputs[0].shape = {3};
    plan.workers[0].inputs[0].values = {1.0F, 2.0F, 9.0F};
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectFailure(result, ConcurrentErrorKind::kInferenceFailed);
    EXPECT_EQ(result.errors.front().inference_kind, InferenceErrorKind::kShapeMismatch);
    EXPECT_NE(result.errors.front().message.find("dimension"), std::string::npos);
}

TEST_F(ConcurrentInferenceTest, CorrectnessMismatchFailsWithoutPartialSuccess) {
    ConcurrentPlan plan = AddPlan(4, 2, true);
    plan.workers[1].expected[0].values = {0.0F, 0.0F};
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectFailure(result, ConcurrentErrorKind::kOutputMismatch);
    bool found = false;
    for (const auto& error : result.errors) {
        if (error.worker_index == 1) {
            found = true;
            EXPECT_EQ(error.inference_kind, InferenceErrorKind::kNone);
            EXPECT_NE(error.message.find("outside tolerance"), std::string::npos);
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(ConcurrentInferenceTest, MovedFromEngineFailsAndJoins) {
    EngineLoad first = load_inference_engine(INFEREDGE_ADD_MODEL_PATH);
    EngineLoad second = load_inference_engine(INFEREDGE_ADD_MODEL_PATH);
    ASSERT_NE(first.engine, nullptr);
    ASSERT_NE(second.engine, nullptr);
    InferenceEngine held = std::move(*first.engine);
    EXPECT_TRUE(first.engine->inputs().empty());
    EXPECT_FALSE(held.inputs().empty());
    std::vector<std::unique_ptr<InferenceEngine>> engines;
    engines.push_back(std::move(first.engine));
    engines.push_back(std::move(second.engine));
    const ConcurrentPlan plan = AddPlan(2, 2, true);
    const ConcurrentExecution result = run_concurrent_inference(std::move(engines), plan);
    ExpectFailure(result, ConcurrentErrorKind::kInferenceFailed);
    bool found = false;
    for (const auto& error : result.errors) {
        if (error.worker_index == 0) {
            found = true;
            EXPECT_EQ(error.inference_kind, InferenceErrorKind::kExecution);
            EXPECT_NE(error.message.find("empty"), std::string::npos);
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(ConcurrentInferenceTest, OneWorkerFailureDoesNotReportSuccess) {
    ConcurrentPlan plan = AddPlan(8, 4, true);
    plan.workers[5].inputs.clear();
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectFailure(result, ConcurrentErrorKind::kInferenceFailed);
    EXPECT_EQ(result.outputs.size(), 0U);
    EXPECT_EQ(result.errors.front().worker_index, 5U);
    EXPECT_EQ(result.errors.front().request_iteration, 0U);
}

TEST_F(ConcurrentInferenceTest, FailureStopsTheFailingWorker) {
    ConcurrentPlan plan = AddPlan(2, 4, true);
    plan.workers[0].inputs.pop_back();
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectFailure(result, ConcurrentErrorKind::kInferenceFailed);
    EXPECT_EQ(result.errors.front().worker_index, 0U);
    EXPECT_EQ(result.errors.front().request_iteration, 0U);
    std::uint32_t reports_for_worker = 0;
    for (const auto& error : result.errors) {
        if (error.worker_index == 0) {
            ++reports_for_worker;
        }
    }
    EXPECT_EQ(reports_for_worker, 1U);
}

TEST_F(ConcurrentInferenceTest, InjectedWorkerExceptionIsCaught) {
    ConcurrentPlan plan = AddPlan(4, 3, true);
    plan.throw_in_worker = 2;
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectFailure(result, ConcurrentErrorKind::kWorkerException);
    bool found = false;
    for (const auto& error : result.errors) {
        if (error.worker_index == 2) {
            found = true;
            EXPECT_NE(error.message.find("injected worker exception"), std::string::npos);
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(ConcurrentInferenceTest, PartialStartupJoinsThreadsAlreadyCreated) {
    ConcurrentPlan plan = AddPlan(8, 4, true);
    plan.stop_after_starting_workers = 3;
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectFailure(result, ConcurrentErrorKind::kWorkerStart);
    EXPECT_EQ(result.errors.front().worker_index, 3U);
    EXPECT_EQ(result.errors.front().request_iteration, 0U);
    EXPECT_NE(result.errors.front().message.find("not started"), std::string::npos);
}

TEST_F(ConcurrentInferenceTest, StartupFailureBeforeAnyThreadStillReturns) {
    ConcurrentPlan plan = AddPlan(4, 2, true);
    plan.stop_after_starting_workers = 0;
    const ConcurrentExecution result = run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan);
    ExpectFailure(result, ConcurrentErrorKind::kWorkerStart);
    EXPECT_EQ(result.errors.front().worker_index, 0U);
}

TEST_F(ConcurrentInferenceTest, RepeatedConcurrentRuns) {
    const ConcurrentPlan plan = AddPlan(4, 3, false);
    ExpectAddSuccess(run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan), plan);
    ExpectAddSuccess(run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, plan), plan);
}

TEST_F(ConcurrentInferenceTest, ModelLoadFailureStartsNoWorkers) {
    ConcurrentPlan plan = AddPlan(4, 1, true);
    const ConcurrentExecution result =
        run_concurrent_inference(std::string{INFEREDGE_ADD_MODEL_PATH} + ".missing", plan);
    ExpectFailure(result, ConcurrentErrorKind::kModelLoad);
    EXPECT_EQ(result.load_error.kind, InferenceErrorKind::kModelNotFound);
    EXPECT_NE(result.load_error.message.find("missing"), std::string::npos);
    EXPECT_TRUE(result.outputs.empty());
}

TEST_F(ConcurrentInferenceTest, RejectsAnEmptyPlan) {
    const ConcurrentExecution result =
        run_concurrent_inference(INFEREDGE_ADD_MODEL_PATH, ConcurrentPlan{});
    ExpectFailure(result, ConcurrentErrorKind::kInvalidPlan);
}

TEST_F(ConcurrentInferenceTest, RejectsANullEngineBeforeStarting) {
    std::vector<std::unique_ptr<InferenceEngine>> engines;
    engines.push_back(nullptr);
    ConcurrentPlan plan = AddPlan(1, 1, true);
    const ConcurrentExecution result = run_concurrent_inference(std::move(engines), plan);
    ExpectFailure(result, ConcurrentErrorKind::kInvalidPlan);
    EXPECT_NE(result.errors.front().message.find("null"), std::string::npos);
}
