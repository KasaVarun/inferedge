#include "inferedge/inference_engine.hpp"
#include "inferedge/neural_reference.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

#ifndef INFEREDGE_MLP_SMALL_MODEL_PATH
#error "CMake must define INFEREDGE_MLP_SMALL_MODEL_PATH"
#endif
#ifndef INFEREDGE_MLP_MEDIUM_MODEL_PATH
#error "CMake must define INFEREDGE_MLP_MEDIUM_MODEL_PATH"
#endif
#ifndef INFEREDGE_GEMM_MODEL_PATH
#error "CMake must define INFEREDGE_GEMM_MODEL_PATH"
#endif
#ifndef INFEREDGE_MLP_SMALL_WEIGHTS_PATH
#error "CMake must define INFEREDGE_MLP_SMALL_WEIGHTS_PATH"
#endif
#ifndef INFEREDGE_MLP_MEDIUM_WEIGHTS_PATH
#error "CMake must define INFEREDGE_MLP_MEDIUM_WEIGHTS_PATH"
#endif
#ifndef INFEREDGE_GEMM_WEIGHTS_PATH
#error "CMake must define INFEREDGE_GEMM_WEIGHTS_PATH"
#endif
#ifndef INFEREDGE_MLP_SMALL_FIXTURE_DIR
#error "CMake must define INFEREDGE_MLP_SMALL_FIXTURE_DIR"
#endif
#ifndef INFEREDGE_MLP_MEDIUM_FIXTURE_DIR
#error "CMake must define INFEREDGE_MLP_MEDIUM_FIXTURE_DIR"
#endif
#ifndef INFEREDGE_GEMM_FIXTURE_DIR
#error "CMake must define INFEREDGE_GEMM_FIXTURE_DIR"
#endif
#ifndef INFEREDGE_INT64_MODEL_PATH
#error "CMake must define INFEREDGE_INT64_MODEL_PATH"
#endif

namespace {

using inferedge::ElementType;
using inferedge::evaluate_neural_reference;
using inferedge::InferenceEngine;
using inferedge::InferenceErrorKind;
using inferedge::load_float32_file;
using inferedge::load_inference_engine;
using inferedge::NamedTensorView;
using inferedge::neural_contract;
using inferedge::neural_tolerance;
using inferedge::neural_worst_mismatch;
using inferedge::NeuralContract;
using inferedge::NeuralMismatch;
using inferedge::NeuralReferenceErrorKind;
using inferedge::NeuralWorkload;

struct WorkloadPaths {
    NeuralWorkload workload;
    const char* model;
    const char* weights;
    const char* fixtures;
};

constexpr WorkloadPaths kWorkloads[] = {
    {NeuralWorkload::kMlpSmall, INFEREDGE_MLP_SMALL_MODEL_PATH, INFEREDGE_MLP_SMALL_WEIGHTS_PATH,
     INFEREDGE_MLP_SMALL_FIXTURE_DIR},
    {NeuralWorkload::kMlpMedium, INFEREDGE_MLP_MEDIUM_MODEL_PATH, INFEREDGE_MLP_MEDIUM_WEIGHTS_PATH,
     INFEREDGE_MLP_MEDIUM_FIXTURE_DIR},
    {NeuralWorkload::kGemm, INFEREDGE_GEMM_MODEL_PATH, INFEREDGE_GEMM_WEIGHTS_PATH,
     INFEREDGE_GEMM_FIXTURE_DIR},
};

constexpr std::size_t kBatches[] = {1, 2, 4, 8, 16, 32};

const char* WorkloadName(NeuralWorkload workload) {
    switch (workload) {
    case NeuralWorkload::kMlpSmall:
        return "mlp_small_f32";
    case NeuralWorkload::kMlpMedium:
        return "mlp_medium_f32";
    case NeuralWorkload::kGemm:
        return "gemm_f32";
    }
    return "unknown";
}

std::filesystem::path FixtureFile(const char* directory, std::size_t batch, const char* kind) {
    std::string name = "b";
    if (batch < 10) {
        name += "0";
    }
    name += std::to_string(batch);
    name += ".";
    name += kind;
    name += ".f32";
    return std::filesystem::path{directory} / name;
}

class DynamicWorkloadInferenceTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

void ExpectEngineAgrees(const WorkloadPaths& paths) {
    auto loaded = load_inference_engine(paths.model);
    ASSERT_EQ(loaded.error.kind, InferenceErrorKind::kNone) << loaded.error.message;
    const NeuralContract contract = neural_contract(paths.workload);
    ASSERT_EQ(loaded.engine->inputs().size(), 1U);
    ASSERT_EQ(loaded.engine->outputs().size(), 1U);
    EXPECT_EQ(loaded.engine->inputs()[0].name, contract.input_name);
    EXPECT_EQ(loaded.engine->outputs()[0].name, contract.output_name);
    EXPECT_EQ(loaded.engine->inputs()[0].element_type, ElementType::kFloat32);
    EXPECT_EQ(loaded.engine->outputs()[0].element_type, ElementType::kFloat32);
    ASSERT_EQ(loaded.engine->inputs()[0].dimensions.size(), 2U);
    ASSERT_EQ(loaded.engine->outputs()[0].dimensions.size(), 2U);
    EXPECT_LT(loaded.engine->inputs()[0].dimensions[0], 0);
    EXPECT_LT(loaded.engine->outputs()[0].dimensions[0], 0);
    EXPECT_EQ(loaded.engine->inputs()[0].dimensions[1],
              static_cast<std::int64_t>(contract.input_width));
    EXPECT_EQ(loaded.engine->outputs()[0].dimensions[1],
              static_cast<std::int64_t>(contract.output_width));

    const auto weights = load_float32_file(paths.weights, contract.parameter_count);
    ASSERT_EQ(weights.error.kind, NeuralReferenceErrorKind::kNone) << weights.error.message;

    NeuralMismatch worst;
    std::size_t worst_batch = 0;
    float worst_relative = 0.F;
    for (const std::size_t batch : kBatches) {
        const auto input = load_float32_file(FixtureFile(paths.fixtures, batch, "input"),
                                             batch * contract.input_width);
        ASSERT_EQ(input.error.kind, NeuralReferenceErrorKind::kNone) << input.error.message;
        const auto reference =
            evaluate_neural_reference(paths.workload, input.values, batch, weights.values);
        ASSERT_EQ(reference.error.kind, NeuralReferenceErrorKind::kNone) << reference.error.message;
        const NamedTensorView view{
            contract.input_name,
            {static_cast<std::int64_t>(batch), static_cast<std::int64_t>(contract.input_width)},
            std::span<const float>{input.values}};
        const auto inferred = loaded.engine->run(std::span<const NamedTensorView>{&view, 1});
        ASSERT_EQ(inferred.error.kind, InferenceErrorKind::kNone) << inferred.error.message;
        ASSERT_EQ(inferred.outputs.size(), 1U);
        EXPECT_EQ(inferred.outputs[0].name, contract.output_name);
        ASSERT_EQ(inferred.outputs[0].shape.size(), 2U);
        EXPECT_EQ(inferred.outputs[0].shape[0], static_cast<std::int64_t>(batch));
        EXPECT_EQ(inferred.outputs[0].shape[1], static_cast<std::int64_t>(contract.output_width));
        ASSERT_EQ(inferred.outputs[0].values.size(), reference.values.size());
        const NeuralMismatch mismatch =
            neural_worst_mismatch(paths.workload, inferred.outputs[0].values, reference.values);
        EXPECT_TRUE(mismatch.within)
            << WorkloadName(paths.workload) << " batch " << batch << " index " << mismatch.index
            << " absolute " << mismatch.absolute_error << " ratio " << mismatch.ratio;
        if (mismatch.ratio > worst.ratio) {
            worst = mismatch;
            worst_batch = batch;
        }
        for (std::size_t index = 0; index < reference.values.size(); ++index) {
            const float seen = inferred.outputs[0].values[index];
            const float expected = reference.values[index];
            ASSERT_TRUE(std::isfinite(seen));
            ASSERT_TRUE(std::isfinite(expected));
            if (expected != 0.F) {
                const float relative = std::fabs(seen - expected) / std::fabs(expected);
                if (relative > worst_relative) {
                    worst_relative = relative;
                }
            }
        }
    }
    const std::size_t row = worst.index / contract.output_width;
    const std::size_t column = worst.index % contract.output_width;
    std::cout << "neural_tolerance " << WorkloadName(paths.workload)
              << " max_abs=" << worst.absolute_error << " max_rel=" << worst_relative
              << " max_ratio=" << worst.ratio << " batch=" << worst_batch << " row=" << row
              << " output_index=" << column << "\n";
}

} // namespace

TEST_F(DynamicWorkloadInferenceTest, SmallMlpAgreesWithTheReference) {
    ExpectEngineAgrees(kWorkloads[0]);
}

TEST_F(DynamicWorkloadInferenceTest, MediumMlpAgreesWithTheReference) {
    ExpectEngineAgrees(kWorkloads[1]);
}

TEST_F(DynamicWorkloadInferenceTest, GemmAgreesWithTheReference) {
    ExpectEngineAgrees(kWorkloads[2]);
}

TEST_F(DynamicWorkloadInferenceTest, RejectsWrongNameRankWidthAndCount) {
    auto loaded = load_inference_engine(INFEREDGE_MLP_SMALL_MODEL_PATH);
    ASSERT_EQ(loaded.error.kind, InferenceErrorKind::kNone);
    const std::vector<float> values(32, 0.F);
    const NamedTensorView wrong_name{"image", {1, 32}, std::span<const float>{values}};
    EXPECT_EQ(loaded.engine->run(std::span<const NamedTensorView>{&wrong_name, 1}).error.kind,
              InferenceErrorKind::kUnexpectedInput);
    const NamedTensorView wrong_rank{"features", {32}, std::span<const float>{values}};
    EXPECT_EQ(loaded.engine->run(std::span<const NamedTensorView>{&wrong_rank, 1}).error.kind,
              InferenceErrorKind::kShapeMismatch);
    const std::vector<float> narrow(8, 0.F);
    const NamedTensorView wrong_width{"features", {1, 8}, std::span<const float>{narrow}};
    EXPECT_EQ(loaded.engine->run(std::span<const NamedTensorView>{&wrong_width, 1}).error.kind,
              InferenceErrorKind::kShapeMismatch);
    const std::vector<float> short_values(16, 0.F);
    const NamedTensorView wrong_count{"features", {1, 32}, std::span<const float>{short_values}};
    EXPECT_EQ(loaded.engine->run(std::span<const NamedTensorView>{&wrong_count, 1}).error.kind,
              InferenceErrorKind::kElementCountMismatch);
}

TEST_F(DynamicWorkloadInferenceTest, RejectsAMissingModelAndANonFloatModel) {
    const auto missing = load_inference_engine("models/does_not_exist.onnx");
    EXPECT_EQ(missing.error.kind, InferenceErrorKind::kModelNotFound);
    EXPECT_EQ(missing.engine, nullptr);
    const auto int64_model = load_inference_engine(INFEREDGE_INT64_MODEL_PATH);
    EXPECT_EQ(int64_model.error.kind, InferenceErrorKind::kUnsupportedType);
    EXPECT_EQ(int64_model.engine, nullptr);
}

TEST(NeuralToleranceTest, ProposedBoundsStayAtTheReviewedValues) {
    EXPECT_FLOAT_EQ(neural_tolerance(NeuralWorkload::kMlpSmall).absolute, 2.0e-4F);
    EXPECT_FLOAT_EQ(neural_tolerance(NeuralWorkload::kMlpSmall).relative, 2.0e-4F);
    EXPECT_FLOAT_EQ(neural_tolerance(NeuralWorkload::kMlpMedium).absolute, 2.0e-4F);
    EXPECT_FLOAT_EQ(neural_tolerance(NeuralWorkload::kGemm).absolute, 1.0e-4F);
    EXPECT_FLOAT_EQ(neural_tolerance(NeuralWorkload::kGemm).relative, 1.0e-4F);
}
