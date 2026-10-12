#include "inferedge/inference_engine.hpp"
#include "inferedge/tensor_batch.hpp"

#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#ifndef INFEREDGE_BATCH_ADD_MODEL_PATH
#error "CMake must define INFEREDGE_BATCH_ADD_MODEL_PATH"
#endif

namespace {

using inferedge::AssembleBatchInputs;
using inferedge::ElementType;
using inferedge::InferenceErrorKind;
using inferedge::load_inference_engine;
using inferedge::NamedTensor;
using inferedge::NamedTensorView;
using inferedge::SplitBatchOutputs;
using inferedge::TensorBatchAssembly;
using inferedge::TensorBatchErrorKind;
using inferedge::TensorBatchSplit;
using inferedge::TensorMetadata;

TensorMetadata Meta(std::string name, std::vector<std::int64_t> dimensions) {
    return TensorMetadata{std::move(name), ElementType::kFloat32, std::move(dimensions)};
}

NamedTensor Tensor(std::string name, std::vector<std::int64_t> shape, std::vector<float> values) {
    return NamedTensor{std::move(name), std::move(shape), std::move(values)};
}

std::vector<TensorMetadata> AddInputs() {
    return {Meta("A", {-1, 2}), Meta("B", {-1, 2})};
}

std::vector<TensorMetadata> AddOutput() {
    return {Meta("C", {-1, 2})};
}

TensorBatchAssembly Assemble(const std::vector<TensorMetadata>& metadata,
                             std::span<const std::span<const NamedTensor>> requests) {
    return AssembleBatchInputs(metadata, requests);
}

void ExpectFailure(const TensorBatchAssembly& assembled, TensorBatchErrorKind kind) {
    EXPECT_EQ(assembled.error.kind, kind);
    EXPECT_TRUE(assembled.inputs.empty());
}

void ExpectFailure(const TensorBatchSplit& split, TensorBatchErrorKind kind) {
    EXPECT_EQ(split.error.kind, kind);
    EXPECT_TRUE(split.requests.empty());
}

void ExpectTensor(const NamedTensor& tensor, const std::string& name,
                  const std::vector<std::int64_t>& shape, const std::vector<float>& values) {
    EXPECT_EQ(tensor.name, name);
    EXPECT_EQ(tensor.shape, shape);
    EXPECT_EQ(tensor.values, values);
}

class TensorBatchTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

} // namespace

TEST_F(TensorBatchTest, RejectsAnEmptyRequestList) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    ExpectFailure(Assemble(inputs, {}), TensorBatchErrorKind::kEmptyBatch);
}

TEST_F(TensorBatchTest, AssemblesOneAddRequest) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f}),
                                   Tensor("B", {1, 2}, {3.0f, 4.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    const TensorBatchAssembly assembled = Assemble(inputs, requests);
    ASSERT_EQ(assembled.error.kind, TensorBatchErrorKind::kNone);
    ASSERT_EQ(assembled.inputs.size(), 2U);
    ExpectTensor(assembled.inputs[0], "A", {1, 2}, {1.0f, 2.0f});
    ExpectTensor(assembled.inputs[1], "B", {1, 2}, {3.0f, 4.0f});
}

TEST_F(TensorBatchTest, AssemblesTwoAddRequests) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor first[] = {Tensor("A", {1, 2}, {1.0f, 2.0f}),
                                 Tensor("B", {1, 2}, {3.0f, 4.0f})};
    const NamedTensor second[] = {Tensor("A", {1, 2}, {5.0f, 6.0f}),
                                  Tensor("B", {1, 2}, {7.0f, 8.0f})};
    const std::span<const NamedTensor> requests[] = {first, second};
    const TensorBatchAssembly assembled = Assemble(inputs, requests);
    ASSERT_EQ(assembled.error.kind, TensorBatchErrorKind::kNone);
    ASSERT_EQ(assembled.inputs.size(), 2U);
    ExpectTensor(assembled.inputs[0], "A", {2, 2}, {1.0f, 2.0f, 5.0f, 6.0f});
    ExpectTensor(assembled.inputs[1], "B", {2, 2}, {3.0f, 4.0f, 7.0f, 8.0f});
}

TEST_F(TensorBatchTest, AssemblesFourAddRequests) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor first[] = {Tensor("A", {1, 2}, {1.0f, 2.0f}),
                                 Tensor("B", {1, 2}, {3.0f, 4.0f})};
    const NamedTensor second[] = {Tensor("A", {1, 2}, {5.0f, 6.0f}),
                                  Tensor("B", {1, 2}, {7.0f, 8.0f})};
    const NamedTensor third[] = {Tensor("A", {1, 2}, {9.0f, 0.0f}),
                                 Tensor("B", {1, 2}, {1.0f, 1.0f})};
    const NamedTensor fourth[] = {Tensor("A", {1, 2}, {-1.0f, 4.0f}),
                                  Tensor("B", {1, 2}, {2.0f, -3.0f})};
    const std::span<const NamedTensor> requests[] = {first, second, third, fourth};
    const TensorBatchAssembly assembled = Assemble(inputs, requests);
    ASSERT_EQ(assembled.error.kind, TensorBatchErrorKind::kNone);
    ASSERT_EQ(assembled.inputs.size(), 2U);
    ExpectTensor(assembled.inputs[0], "A", {4, 2},
                 {1.0f, 2.0f, 5.0f, 6.0f, 9.0f, 0.0f, -1.0f, 4.0f});
    ExpectTensor(assembled.inputs[1], "B", {4, 2},
                 {3.0f, 4.0f, 7.0f, 8.0f, 1.0f, 1.0f, 2.0f, -3.0f});
}

TEST_F(TensorBatchTest, AssemblesInputsInModelOrder) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor first[] = {Tensor("B", {1, 2}, {3.0f, 4.0f}),
                                 Tensor("A", {1, 2}, {1.0f, 2.0f})};
    const NamedTensor second[] = {Tensor("A", {1, 2}, {5.0f, 6.0f}),
                                  Tensor("B", {1, 2}, {7.0f, 8.0f})};
    const std::span<const NamedTensor> requests[] = {first, second};
    const TensorBatchAssembly assembled = Assemble(inputs, requests);
    ASSERT_EQ(assembled.error.kind, TensorBatchErrorKind::kNone);
    ASSERT_EQ(assembled.inputs.size(), 2U);
    ExpectTensor(assembled.inputs[0], "A", {2, 2}, {1.0f, 2.0f, 5.0f, 6.0f});
    ExpectTensor(assembled.inputs[1], "B", {2, 2}, {3.0f, 4.0f, 7.0f, 8.0f});
}

TEST_F(TensorBatchTest, AssemblesAModelWithOneInput) {
    const std::vector<TensorMetadata> inputs = {Meta("image", {-1, 3})};
    const NamedTensor first[] = {Tensor("image", {1, 3}, {1.0f, 2.0f, 3.0f})};
    const NamedTensor second[] = {Tensor("image", {1, 3}, {4.0f, 5.0f, 6.0f})};
    const std::span<const NamedTensor> requests[] = {first, second};
    const TensorBatchAssembly assembled = Assemble(inputs, requests);
    ASSERT_EQ(assembled.error.kind, TensorBatchErrorKind::kNone);
    ASSERT_EQ(assembled.inputs.size(), 1U);
    ExpectTensor(assembled.inputs[0], "image", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

TEST_F(TensorBatchTest, RejectsAMissingInput) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kMissingTensor);
}

TEST_F(TensorBatchTest, RejectsAnUnexpectedInput) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f}),
                                   Tensor("B", {1, 2}, {3.0f, 4.0f}),
                                   Tensor("D", {1, 2}, {5.0f, 6.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kUnexpectedTensor);
}

TEST_F(TensorBatchTest, RejectsADuplicateInput) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f}),
                                   Tensor("A", {1, 2}, {9.0f, 9.0f}),
                                   Tensor("B", {1, 2}, {3.0f, 4.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kDuplicateName);
}

TEST_F(TensorBatchTest, RejectsARequestRankMismatch) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor request[] = {Tensor("A", {2}, {1.0f, 2.0f}),
                                   Tensor("B", {1, 2}, {3.0f, 4.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kRankMismatch);
}

TEST_F(TensorBatchTest, RejectsAStaticDimensionMismatch) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor request[] = {Tensor("A", {1, 3}, {1.0f, 2.0f, 3.0f}),
                                   Tensor("B", {1, 2}, {4.0f, 5.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kShapeMismatch);
}

TEST_F(TensorBatchTest, RejectsABatchAxisOtherThanOne) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor request[] = {Tensor("A", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}),
                                   Tensor("B", {1, 2}, {5.0f, 6.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kBatchAxisMismatch);
}

TEST_F(TensorBatchTest, RejectsAnElementCountMismatch) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f, 3.0f}),
                                   Tensor("B", {1, 2}, {4.0f, 5.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kElementCountMismatch);
}

TEST_F(TensorBatchTest, RejectsAnEmptyTensorName) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    const NamedTensor request[] = {Tensor("", {1, 2}, {1.0f, 2.0f}),
                                   Tensor("B", {1, 2}, {3.0f, 4.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kEmptyName);
}

TEST_F(TensorBatchTest, RejectsEmptyInputMetadata) {
    const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble({}, requests), TensorBatchErrorKind::kUnsupportedMetadata);
}

TEST_F(TensorBatchTest, RejectsAnEmptyMetadataName) {
    const std::vector<TensorMetadata> inputs = {Meta("", {-1, 2})};
    const NamedTensor request[] = {Tensor("", {1, 2}, {1.0f, 2.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kEmptyName);
}

TEST_F(TensorBatchTest, RejectsDuplicateMetadataNames) {
    const std::vector<TensorMetadata> inputs = {Meta("A", {-1, 2}), Meta("A", {-1, 2})};
    const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kDuplicateName);
}

TEST_F(TensorBatchTest, RejectsMetadataRankBelowTwo) {
    const std::vector<TensorMetadata> inputs = {Meta("A", {-1})};
    const NamedTensor request[] = {Tensor("A", {1}, {1.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kUnsupportedMetadata);
}

TEST_F(TensorBatchTest, RejectsAStaticMetadataBatchAxis) {
    const std::vector<TensorMetadata> inputs = {Meta("A", {1, 2})};
    const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kUnsupportedMetadata);
}

TEST_F(TensorBatchTest, RejectsAnAdditionalDynamicAxis) {
    const std::vector<TensorMetadata> inputs = {Meta("A", {-1, -1})};
    const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kUnsupportedMetadata);
}

TEST_F(TensorBatchTest, RejectsANonPositiveStaticDimension) {
    const std::vector<TensorMetadata> inputs = {Meta("A", {-1, 0})};
    const NamedTensor request[] = {Tensor("A", {1, 0}, {})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kUnsupportedMetadata);
}

TEST_F(TensorBatchTest, RejectsAnUnsupportedElementType) {
    TensorMetadata input = Meta("A", {-1, 2});
    input.element_type = static_cast<ElementType>(1);
    const std::vector<TensorMetadata> inputs = {input};
    const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kUnsupportedType);
}

TEST_F(TensorBatchTest, RejectsAMetadataShapeThatOverflows) {
    const std::int64_t wide = std::int64_t{1} << 62;
    const std::vector<TensorMetadata> inputs = {Meta("A", {-1, wide, 8})};
    const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f})};
    const std::span<const NamedTensor> requests[] = {request};
    ExpectFailure(Assemble(inputs, requests), TensorBatchErrorKind::kOverflow);
}

TEST_F(TensorBatchTest, AssemblyFailsWithoutPublishingAPartialBatch) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    std::vector<NamedTensor> first = {Tensor("A", {1, 2}, {1.0f, 2.0f}),
                                      Tensor("B", {1, 2}, {3.0f, 4.0f})};
    std::vector<NamedTensor> second = {Tensor("A", {1, 2}, {5.0f, 6.0f})};
    std::vector<NamedTensor> third = {Tensor("B", {1, 2}, {7.0f, 8.0f}),
                                      Tensor("A", {1, 2}, {9.0f, 0.0f})};
    const std::vector<float> first_a = first[0].values;
    const std::vector<float> first_b = first[1].values;
    const std::vector<float> second_a = second[0].values;
    const std::vector<float> third_b = third[0].values;
    const std::vector<float> third_a = third[1].values;
    const std::span<const NamedTensor> requests[] = {first, second, third};
    const TensorBatchAssembly assembled = Assemble(inputs, requests);
    ExpectFailure(assembled, TensorBatchErrorKind::kMissingTensor);
    EXPECT_EQ(first[0].values, first_a);
    EXPECT_EQ(first[1].values, first_b);
    EXPECT_EQ(second[0].values, second_a);
    EXPECT_EQ(third[0].values, third_b);
    EXPECT_EQ(third[1].values, third_a);
}

TEST_F(TensorBatchTest, AssembledValuesOutliveTheirInputs) {
    const std::vector<TensorMetadata> inputs = AddInputs();
    TensorBatchAssembly assembled;
    {
        const NamedTensor request[] = {Tensor("A", {1, 2}, {1.0f, 2.0f}),
                                       Tensor("B", {1, 2}, {3.0f, 4.0f})};
        const std::span<const NamedTensor> requests[] = {request};
        assembled = Assemble(inputs, requests);
    }
    ASSERT_EQ(assembled.error.kind, TensorBatchErrorKind::kNone);
    ExpectTensor(assembled.inputs[0], "A", {1, 2}, {1.0f, 2.0f});
    ExpectTensor(assembled.inputs[1], "B", {1, 2}, {3.0f, 4.0f});
}

TEST_F(TensorBatchTest, SplitsTwoAddRows) {
    const std::vector<TensorMetadata> outputs = AddOutput();
    const NamedTensor batched[] = {Tensor("C", {2, 2}, {4.0f, 6.0f, 12.0f, 14.0f})};
    const TensorBatchSplit split = SplitBatchOutputs(outputs, batched, 2);
    ASSERT_EQ(split.error.kind, TensorBatchErrorKind::kNone);
    ASSERT_EQ(split.requests.size(), 2U);
    ASSERT_EQ(split.requests[0].size(), 1U);
    ASSERT_EQ(split.requests[1].size(), 1U);
    ExpectTensor(split.requests[0][0], "C", {1, 2}, {4.0f, 6.0f});
    ExpectTensor(split.requests[1][0], "C", {1, 2}, {12.0f, 14.0f});
}

TEST_F(TensorBatchTest, SplitsFourAddRows) {
    const std::vector<TensorMetadata> outputs = AddOutput();
    const NamedTensor batched[] = {
        Tensor("C", {4, 2}, {4.0f, 6.0f, 12.0f, 14.0f, 10.0f, 1.0f, 1.0f, 1.0f})};
    const TensorBatchSplit split = SplitBatchOutputs(outputs, batched, 4);
    ASSERT_EQ(split.error.kind, TensorBatchErrorKind::kNone);
    ASSERT_EQ(split.requests.size(), 4U);
    ExpectTensor(split.requests[0][0], "C", {1, 2}, {4.0f, 6.0f});
    ExpectTensor(split.requests[1][0], "C", {1, 2}, {12.0f, 14.0f});
    ExpectTensor(split.requests[2][0], "C", {1, 2}, {10.0f, 1.0f});
    ExpectTensor(split.requests[3][0], "C", {1, 2}, {1.0f, 1.0f});
}

TEST_F(TensorBatchTest, SplitsTwoOutputsInModelOrder) {
    const std::vector<TensorMetadata> outputs = {Meta("logits", {-1, 2}), Meta("score", {-1, 1})};
    const NamedTensor batched[] = {Tensor("score", {2, 1}, {8.0f, 9.0f}),
                                   Tensor("logits", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f})};
    const TensorBatchSplit split = SplitBatchOutputs(outputs, batched, 2);
    ASSERT_EQ(split.error.kind, TensorBatchErrorKind::kNone);
    ASSERT_EQ(split.requests.size(), 2U);
    ASSERT_EQ(split.requests[0].size(), 2U);
    ASSERT_EQ(split.requests[1].size(), 2U);
    ExpectTensor(split.requests[0][0], "logits", {1, 2}, {1.0f, 2.0f});
    ExpectTensor(split.requests[0][1], "score", {1, 1}, {8.0f});
    ExpectTensor(split.requests[1][0], "logits", {1, 2}, {3.0f, 4.0f});
    ExpectTensor(split.requests[1][1], "score", {1, 1}, {9.0f});
}

TEST_F(TensorBatchTest, RejectsAnEmptySplit) {
    const std::vector<TensorMetadata> outputs = AddOutput();
    const NamedTensor batched[] = {Tensor("C", {0, 2}, {})};
    ExpectFailure(SplitBatchOutputs(outputs, batched, 0), TensorBatchErrorKind::kEmptyBatch);
}

TEST_F(TensorBatchTest, RejectsAWrongOutputBatchSize) {
    const std::vector<TensorMetadata> outputs = AddOutput();
    const NamedTensor batched[] = {Tensor("C", {1, 2}, {4.0f, 6.0f})};
    ExpectFailure(SplitBatchOutputs(outputs, batched, 2), TensorBatchErrorKind::kBatchAxisMismatch);
}

TEST_F(TensorBatchTest, RejectsAWrongOutputStaticShape) {
    const std::vector<TensorMetadata> outputs = AddOutput();
    const NamedTensor batched[] = {Tensor("C", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})};
    ExpectFailure(SplitBatchOutputs(outputs, batched, 2), TensorBatchErrorKind::kShapeMismatch);
}

TEST_F(TensorBatchTest, RejectsAnOutputRankMismatch) {
    const std::vector<TensorMetadata> outputs = AddOutput();
    const NamedTensor batched[] = {Tensor("C", {4}, {1.0f, 2.0f, 3.0f, 4.0f})};
    ExpectFailure(SplitBatchOutputs(outputs, batched, 2), TensorBatchErrorKind::kRankMismatch);
}

TEST_F(TensorBatchTest, RejectsAMissingOutput) {
    const std::vector<TensorMetadata> outputs = {Meta("C", {-1, 2}), Meta("D", {-1, 1})};
    const NamedTensor batched[] = {Tensor("C", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f})};
    ExpectFailure(SplitBatchOutputs(outputs, batched, 2), TensorBatchErrorKind::kMissingTensor);
}

TEST_F(TensorBatchTest, RejectsAnUnexpectedOutput) {
    const std::vector<TensorMetadata> outputs = AddOutput();
    const NamedTensor batched[] = {Tensor("C", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}),
                                   Tensor("D", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f})};
    ExpectFailure(SplitBatchOutputs(outputs, batched, 2), TensorBatchErrorKind::kUnexpectedTensor);
}

TEST_F(TensorBatchTest, RejectsADuplicateOutput) {
    const std::vector<TensorMetadata> outputs = AddOutput();
    const NamedTensor batched[] = {Tensor("C", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}),
                                   Tensor("C", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f})};
    ExpectFailure(SplitBatchOutputs(outputs, batched, 2), TensorBatchErrorKind::kDuplicateName);
}

TEST_F(TensorBatchTest, RejectsAnOutputElementCountMismatch) {
    const std::vector<TensorMetadata> outputs = AddOutput();
    const NamedTensor batched[] = {Tensor("C", {2, 2}, {1.0f, 2.0f, 3.0f})};
    ExpectFailure(SplitBatchOutputs(outputs, batched, 2),
                  TensorBatchErrorKind::kElementCountMismatch);
}

TEST_F(TensorBatchTest, RejectsUnsupportedOutputMetadata) {
    const std::vector<TensorMetadata> outputs = {Meta("C", {2, 2})};
    const NamedTensor batched[] = {Tensor("C", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f})};
    ExpectFailure(SplitBatchOutputs(outputs, batched, 2),
                  TensorBatchErrorKind::kUnsupportedMetadata);
}

TEST_F(TensorBatchTest, RejectsABatchedShapeThatOverflows) {
    const std::int64_t wide = std::numeric_limits<std::int64_t>::max();
    const std::vector<TensorMetadata> outputs = {Meta("C", {-1, wide})};
    const NamedTensor batched[] = {Tensor("C", {3, wide}, {})};
    ExpectFailure(SplitBatchOutputs(outputs, batched, 3), TensorBatchErrorKind::kOverflow);
}

TEST_F(TensorBatchTest, SplitFailsWithoutPartialRequests) {
    const std::vector<TensorMetadata> outputs = {Meta("C", {-1, 2}), Meta("D", {-1, 2})};
    const NamedTensor batched[] = {Tensor("C", {2, 2}, {4.0f, 6.0f, 12.0f, 14.0f}),
                                   Tensor("D", {1, 2}, {1.0f, 1.0f})};
    ExpectFailure(SplitBatchOutputs(outputs, batched, 2), TensorBatchErrorKind::kBatchAxisMismatch);
}

TEST_F(TensorBatchTest, SplitRowsOutliveTheBatch) {
    const std::vector<TensorMetadata> outputs = AddOutput();
    TensorBatchSplit split;
    {
        const NamedTensor batched[] = {Tensor("C", {2, 2}, {4.0f, 6.0f, 12.0f, 14.0f})};
        split = SplitBatchOutputs(outputs, batched, 2);
    }
    ASSERT_EQ(split.error.kind, TensorBatchErrorKind::kNone);
    ExpectTensor(split.requests[0][0], "C", {1, 2}, {4.0f, 6.0f});
    ExpectTensor(split.requests[1][0], "C", {1, 2}, {12.0f, 14.0f});
}

TEST_F(TensorBatchTest, ModifyingOneSplitDoesNotAffectAnother) {
    const std::vector<TensorMetadata> outputs = AddOutput();
    NamedTensor batched = Tensor("C", {2, 2}, {4.0f, 6.0f, 12.0f, 14.0f});
    const std::vector<float> original = batched.values;
    TensorBatchSplit split =
        SplitBatchOutputs(outputs, std::span<const NamedTensor>{&batched, 1}, 2);
    ASSERT_EQ(split.error.kind, TensorBatchErrorKind::kNone);
    split.requests[0][0].values[0] = 99.0f;
    EXPECT_EQ(split.requests[1][0].values, (std::vector<float>{12.0f, 14.0f}));
    EXPECT_EQ(batched.values, original);
}

TEST_F(TensorBatchTest, RunsAssembledAddThroughTheEngine) {
    auto loaded = load_inference_engine(INFEREDGE_BATCH_ADD_MODEL_PATH);
    ASSERT_NE(loaded.engine, nullptr);
    ASSERT_EQ(loaded.engine->inputs().size(), 2U);
    EXPECT_EQ(loaded.engine->inputs()[0].name, "A");
    EXPECT_EQ(loaded.engine->inputs()[0].dimensions, (std::vector<std::int64_t>{-1, 2}));
    EXPECT_EQ(loaded.engine->inputs()[1].name, "B");
    EXPECT_EQ(loaded.engine->inputs()[1].dimensions, (std::vector<std::int64_t>{-1, 2}));
    EXPECT_EQ(loaded.engine->outputs()[0].name, "C");
    EXPECT_EQ(loaded.engine->outputs()[0].dimensions, (std::vector<std::int64_t>{-1, 2}));

    const NamedTensor first[] = {Tensor("A", {1, 2}, {1.0f, 2.0f}),
                                 Tensor("B", {1, 2}, {3.0f, 4.0f})};
    const NamedTensor second[] = {Tensor("B", {1, 2}, {7.0f, 8.0f}),
                                  Tensor("A", {1, 2}, {5.0f, 6.0f})};
    const std::span<const NamedTensor> requests[] = {first, second};
    TensorBatchAssembly assembled = AssembleBatchInputs(loaded.engine->inputs(), requests);
    ASSERT_EQ(assembled.error.kind, TensorBatchErrorKind::kNone);

    std::vector<NamedTensorView> views;
    views.reserve(assembled.inputs.size());
    for (const NamedTensor& input : assembled.inputs) {
        views.push_back(NamedTensorView{input.name, input.shape, input.values});
    }
    auto inferred = loaded.engine->run(views);
    ASSERT_EQ(inferred.error.kind, InferenceErrorKind::kNone);
    ASSERT_EQ(inferred.outputs.size(), 1U);
    ExpectTensor(inferred.outputs[0], "C", {2, 2}, {4.0f, 6.0f, 12.0f, 14.0f});

    TensorBatchSplit split = SplitBatchOutputs(loaded.engine->outputs(), inferred.outputs, 2);
    ASSERT_EQ(split.error.kind, TensorBatchErrorKind::kNone);
    loaded.engine.reset();
    assembled.inputs.clear();
    inferred.outputs.clear();
    ExpectTensor(split.requests[0][0], "C", {1, 2}, {4.0f, 6.0f});
    ExpectTensor(split.requests[1][0], "C", {1, 2}, {12.0f, 14.0f});
    split.requests[0][0].values[0] = -5.0f;
    EXPECT_EQ(split.requests[1][0].values, (std::vector<float>{12.0f, 14.0f}));
}
