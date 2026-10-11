#include "inferedge/inference_engine.hpp"

#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#ifndef INFEREDGE_ADD_MODEL_PATH
#error "CMake must define INFEREDGE_ADD_MODEL_PATH"
#endif

#ifndef INFEREDGE_INVALID_MODEL_PATH
#error "CMake must define INFEREDGE_INVALID_MODEL_PATH"
#endif

#ifndef INFEREDGE_INT64_MODEL_PATH
#error "CMake must define INFEREDGE_INT64_MODEL_PATH"
#endif

namespace {

using inferedge::ElementType;
using inferedge::EngineLoad;
using inferedge::InferenceEngine;
using inferedge::InferenceErrorKind;
using inferedge::InferenceResult;
using inferedge::load_inference_engine;
using inferedge::NamedTensorView;

class InferenceEngineTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

EngineLoad LoadAddModel() {
    return load_inference_engine(INFEREDGE_ADD_MODEL_PATH);
}

NamedTensorView View(const std::string& name, std::vector<std::int64_t> shape,
                     const std::vector<float>& values) {
    return NamedTensorView{name, std::move(shape), std::span<const float>{values}};
}

InferenceResult RunAdd(InferenceEngine& engine, const std::vector<float>& left,
                       const std::vector<float>& right, bool right_first) {
    const NamedTensorView first = View("A", {2}, left);
    const NamedTensorView second = View("B", {2}, right);
    if (right_first) {
        const NamedTensorView ordered[] = {second, first};
        return engine.run(ordered);
    }
    const NamedTensorView ordered[] = {first, second};
    return engine.run(ordered);
}

void ExpectAdd(const InferenceResult& result, float first, float second) {
    ASSERT_EQ(result.error.kind, InferenceErrorKind::kNone);
    EXPECT_TRUE(result.error.message.empty());
    ASSERT_EQ(result.outputs.size(), 1U);
    EXPECT_EQ(result.outputs.front().name, "C");
    ASSERT_EQ(result.outputs.front().shape.size(), 1U);
    EXPECT_EQ(result.outputs.front().shape.front(), 2);
    ASSERT_EQ(result.outputs.front().values.size(), 2U);
    EXPECT_EQ(result.outputs.front().values[0], first);
    EXPECT_EQ(result.outputs.front().values[1], second);
}

} // namespace

TEST_F(InferenceEngineTest, LoadsAValidModel) {
    const EngineLoad loaded = LoadAddModel();
    EXPECT_EQ(loaded.error.kind, InferenceErrorKind::kNone);
    EXPECT_TRUE(loaded.error.message.empty());
    ASSERT_NE(loaded.engine, nullptr);
    EXPECT_EQ(loaded.engine->inputs().size(), 2U);
    EXPECT_EQ(loaded.engine->outputs().size(), 1U);
}

TEST_F(InferenceEngineTest, RejectsAMissingModel) {
    const EngineLoad loaded =
        load_inference_engine(std::string{INFEREDGE_ADD_MODEL_PATH} + ".missing");
    EXPECT_EQ(loaded.error.kind, InferenceErrorKind::kModelNotFound);
    EXPECT_NE(loaded.error.message.find("missing"), std::string::npos);
    EXPECT_EQ(loaded.engine, nullptr);
}

TEST_F(InferenceEngineTest, RejectsAnInvalidModel) {
    const EngineLoad loaded = load_inference_engine(INFEREDGE_INVALID_MODEL_PATH);
    EXPECT_EQ(loaded.error.kind, InferenceErrorKind::kModelInvalid);
    EXPECT_FALSE(loaded.error.message.empty());
    EXPECT_EQ(loaded.engine, nullptr);
}

TEST_F(InferenceEngineTest, ReportsAddMetadata) {
    const EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const auto& inputs = loaded.engine->inputs();
    const auto& outputs = loaded.engine->outputs();
    ASSERT_EQ(inputs.size(), 2U);
    EXPECT_EQ(inputs[0].name, "A");
    EXPECT_EQ(inputs[1].name, "B");
    EXPECT_EQ(outputs.size(), 1U);
    EXPECT_EQ(outputs[0].name, "C");
    for (const auto& tensor : inputs) {
        EXPECT_EQ(tensor.element_type, ElementType::kFloat32);
        ASSERT_EQ(tensor.dimensions.size(), 1U);
        EXPECT_EQ(tensor.dimensions.front(), 2);
    }
    EXPECT_EQ(outputs[0].element_type, ElementType::kFloat32);
    ASSERT_EQ(outputs[0].dimensions.size(), 1U);
    EXPECT_EQ(outputs[0].dimensions.front(), 2);
}

TEST_F(InferenceEngineTest, ExecutesPositiveInputs) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> left{1.0f, 2.0f};
    const std::vector<float> right{3.0f, 4.0f};
    ExpectAdd(RunAdd(*loaded.engine, left, right, false), 4.0f, 6.0f);
}

TEST_F(InferenceEngineTest, AcceptsReorderedInputs) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> left{1.0f, 2.0f};
    const std::vector<float> right{3.0f, 4.0f};
    ExpectAdd(RunAdd(*loaded.engine, left, right, true), 4.0f, 6.0f);
}

TEST_F(InferenceEngineTest, ExecutesNegativeAndZeroInputs) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> left{-1.5f, 0.0f};
    const std::vector<float> right{2.5f, -4.0f};
    ExpectAdd(RunAdd(*loaded.engine, left, right, false), 1.0f, -4.0f);
}

TEST_F(InferenceEngineTest, RejectsAMissingInput) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> left{1.0f, 2.0f};
    const NamedTensorView inputs[] = {View("A", {2}, left)};
    const InferenceResult result = loaded.engine->run(inputs);
    EXPECT_EQ(result.error.kind, InferenceErrorKind::kMissingInput);
    EXPECT_NE(result.error.message.find("B"), std::string::npos);
    EXPECT_TRUE(result.outputs.empty());
}

TEST_F(InferenceEngineTest, RejectsAnUnexpectedInput) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> left{1.0f, 2.0f};
    const std::vector<float> right{3.0f, 4.0f};
    const std::vector<float> extra{5.0f, 6.0f};
    const NamedTensorView inputs[] = {View("A", {2}, left), View("B", {2}, right),
                                      View("D", {2}, extra)};
    const InferenceResult result = loaded.engine->run(inputs);
    EXPECT_EQ(result.error.kind, InferenceErrorKind::kUnexpectedInput);
    EXPECT_NE(result.error.message.find("D"), std::string::npos);
}

TEST_F(InferenceEngineTest, RejectsADuplicateInput) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> left{1.0f, 2.0f};
    const std::vector<float> right{3.0f, 4.0f};
    const NamedTensorView inputs[] = {View("A", {2}, left), View("A", {2}, left),
                                      View("B", {2}, right)};
    const InferenceResult result = loaded.engine->run(inputs);
    EXPECT_EQ(result.error.kind, InferenceErrorKind::kDuplicateInput);
    EXPECT_NE(result.error.message.find("A"), std::string::npos);
}

TEST_F(InferenceEngineTest, RejectsAnEmptyName) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> left{1.0f, 2.0f};
    const NamedTensorView inputs[] = {View("", {2}, left)};
    const InferenceResult result = loaded.engine->run(inputs);
    EXPECT_EQ(result.error.kind, InferenceErrorKind::kEmptyName);
    EXPECT_NE(result.error.message.find("empty"), std::string::npos);
}

TEST_F(InferenceEngineTest, RejectsAnIncorrectShape) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> wide{1.0f, 2.0f, 3.0f};
    const std::vector<float> right{3.0f, 4.0f};
    const NamedTensorView wrong_dimension[] = {View("A", {3}, wide), View("B", {2}, right)};
    const InferenceResult dimension = loaded.engine->run(wrong_dimension);
    EXPECT_EQ(dimension.error.kind, InferenceErrorKind::kShapeMismatch);

    const std::vector<float> left{1.0f, 2.0f};
    const NamedTensorView wrong_rank[] = {View("A", {2, 1}, left), View("B", {2}, right)};
    const InferenceResult rank = loaded.engine->run(wrong_rank);
    EXPECT_EQ(rank.error.kind, InferenceErrorKind::kShapeMismatch);
    EXPECT_NE(rank.error.message.find("rank"), std::string::npos);
}

TEST_F(InferenceEngineTest, RejectsAnIncorrectElementCount) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> left{1.0f, 2.0f, 3.0f};
    const std::vector<float> right{3.0f, 4.0f};
    const NamedTensorView inputs[] = {View("A", {2}, left), View("B", {2}, right)};
    const InferenceResult result = loaded.engine->run(inputs);
    EXPECT_EQ(result.error.kind, InferenceErrorKind::kElementCountMismatch);
    EXPECT_NE(result.error.message.find("3"), std::string::npos);
}

TEST_F(InferenceEngineTest, RejectsANegativeDimension) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> left{1.0f};
    const std::vector<float> right{3.0f, 4.0f};
    const NamedTensorView inputs[] = {View("A", {-1}, left), View("B", {2}, right)};
    const InferenceResult result = loaded.engine->run(inputs);
    EXPECT_EQ(result.error.kind, InferenceErrorKind::kShapeMismatch);
    EXPECT_NE(result.error.message.find("negative"), std::string::npos);
}

TEST_F(InferenceEngineTest, RejectsAnOverflowingShape) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> left{1.0f};
    const std::vector<float> right{3.0f, 4.0f};
    const std::int64_t huge = std::numeric_limits<std::int64_t>::max();
    const NamedTensorView inputs[] = {View("A", {huge, huge}, left), View("B", {2}, right)};
    const InferenceResult result = loaded.engine->run(inputs);
    EXPECT_EQ(result.error.kind, InferenceErrorKind::kElementCountMismatch);
    EXPECT_NE(result.error.message.find("overflows"), std::string::npos);
}

TEST_F(InferenceEngineTest, RejectsAnUnsupportedTensorType) {
    const EngineLoad loaded = load_inference_engine(INFEREDGE_INT64_MODEL_PATH);
    EXPECT_EQ(loaded.error.kind, InferenceErrorKind::kUnsupportedType);
    EXPECT_NE(loaded.error.message.find("X"), std::string::npos);
    EXPECT_EQ(loaded.engine, nullptr);
}

TEST_F(InferenceEngineTest, OutputOwnsItsValues) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    std::vector<float> left{1.0f, 2.0f};
    const std::vector<float> right{3.0f, 4.0f};
    InferenceResult result = RunAdd(*loaded.engine, left, right, false);
    ASSERT_EQ(result.error.kind, InferenceErrorKind::kNone);
    inferedge::NamedTensor output = std::move(result.outputs.front());
    left[0] = 0.0f;
    loaded.engine.reset();
    EXPECT_EQ(output.values[0], 4.0f);
    EXPECT_EQ(output.values[1], 6.0f);
}

TEST_F(InferenceEngineTest, RunsSequentialRequests) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> first_left{1.0f, 2.0f};
    const std::vector<float> first_right{3.0f, 4.0f};
    const std::vector<float> second_left{-1.5f, 0.0f};
    const std::vector<float> second_right{2.5f, -4.0f};
    ExpectAdd(RunAdd(*loaded.engine, first_left, first_right, false), 4.0f, 6.0f);
    ExpectAdd(RunAdd(*loaded.engine, second_left, second_right, false), 1.0f, -4.0f);
}

TEST_F(InferenceEngineTest, MoveConstructionTransfersTheEngine) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    InferenceEngine moved = std::move(*loaded.engine);
    const std::vector<float> left{1.0f, 2.0f};
    const std::vector<float> right{3.0f, 4.0f};
    ExpectAdd(RunAdd(moved, left, right, false), 4.0f, 6.0f);
    EXPECT_TRUE(loaded.engine->inputs().empty());
    EXPECT_TRUE(loaded.engine->outputs().empty());
}

TEST_F(InferenceEngineTest, MoveAssignmentTransfersTheEngine) {
    EngineLoad source = LoadAddModel();
    EngineLoad destination = LoadAddModel();
    ASSERT_NE(source.engine, nullptr);
    ASSERT_NE(destination.engine, nullptr);
    *destination.engine = std::move(*source.engine);
    const std::vector<float> left{1.0f, 2.0f};
    const std::vector<float> right{3.0f, 4.0f};
    ExpectAdd(RunAdd(*destination.engine, left, right, false), 4.0f, 6.0f);
    EXPECT_TRUE(source.engine->inputs().empty());
    EXPECT_TRUE(source.engine->outputs().empty());
}

TEST_F(InferenceEngineTest, MovedFromEngineIsSafe) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    InferenceEngine moved = std::move(*loaded.engine);
    EXPECT_EQ(moved.inputs().size(), 2U);
    EXPECT_TRUE(loaded.engine->inputs().empty());
    EXPECT_TRUE(loaded.engine->outputs().empty());
    const InferenceResult result = loaded.engine->run({});
    EXPECT_EQ(result.error.kind, InferenceErrorKind::kExecution);
    EXPECT_NE(result.error.message.find("empty"), std::string::npos);
    EXPECT_TRUE(result.outputs.empty());
}

TEST_F(InferenceEngineTest, StaysUsableAfterARejectedRequest) {
    EngineLoad loaded = LoadAddModel();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> left{1.0f, 2.0f};
    const NamedTensorView missing[] = {View("A", {2}, left)};
    const InferenceResult rejected = loaded.engine->run(missing);
    EXPECT_EQ(rejected.error.kind, InferenceErrorKind::kMissingInput);

    const std::vector<float> right{3.0f, 4.0f};
    ExpectAdd(RunAdd(*loaded.engine, left, right, false), 4.0f, 6.0f);
}
