#include "inferedge/inference_engine.hpp"
#include "inferedge/neural_reference.hpp"
#include "inferedge/resnet18_fixtures.hpp"
#include "inferedge/scheduler_benchmark.hpp"
#include "inferedge/tensor_batch.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <vector>

#ifndef INFEREDGE_RESNET18_MODEL_PATH
#error "CMake must define INFEREDGE_RESNET18_MODEL_PATH"
#endif
#ifndef INFEREDGE_RESNET18_FIXTURE_DIR
#error "CMake must define INFEREDGE_RESNET18_FIXTURE_DIR"
#endif
#ifndef INFEREDGE_GEMM_MODEL_PATH
#error "CMake must define INFEREDGE_GEMM_MODEL_PATH"
#endif

namespace {

using inferedge::AssembleBatchInputs;
using inferedge::ElementType;
using inferedge::InferenceErrorKind;
using inferedge::kResNet18AbsTol;
using inferedge::kResNet18FixtureIdMax;
using inferedge::kResNet18FixtureIdMin;
using inferedge::kResNet18ImageElements;
using inferedge::kResNet18InputName;
using inferedge::kResNet18LogitElements;
using inferedge::kResNet18OnnxSha256;
using inferedge::kResNet18OutputName;
using inferedge::kResNet18RelTol;
using inferedge::load_float32_file;
using inferedge::load_inference_engine;
using inferedge::load_resnet18_fixture_table;
using inferedge::NamedTensor;
using inferedge::NamedTensorView;
using inferedge::NeuralReferenceErrorKind;
using inferedge::resnet18_check_logits;
using inferedge::resnet18_fixture_id_valid;
using inferedge::resnet18_metadata_error;
using inferedge::resnet18_request_from_fixture;
using inferedge::ResNet18FixtureRequest;
using inferedge::ResNet18FixtureTable;
using inferedge::scheduler_benchmark_check_output;
using inferedge::scheduler_benchmark_metadata_error;
using inferedge::scheduler_benchmark_request;
using inferedge::scheduler_benchmark_sha256_file;
using inferedge::SchedulerBenchmarkModelId;
using inferedge::SplitBatchOutputs;
using inferedge::TensorBatchErrorKind;
using inferedge::TensorMetadata;

constexpr std::size_t kEngineBatches[] = {1, 2, 4, 8};

std::filesystem::path FixtureFile(std::size_t batch, const char* kind) {
    std::string name = "b";
    if (batch < 10) {
        name += "0";
    }
    name += std::to_string(batch);
    name += ".";
    name += kind;
    name += ".f32";
    return std::filesystem::path{INFEREDGE_RESNET18_FIXTURE_DIR} / name;
}

TensorMetadata ImageMeta() {
    return TensorMetadata{kResNet18InputName, ElementType::kFloat32, {-1, 3, 224, 224}};
}

TensorMetadata LogitMeta() {
    return TensorMetadata{kResNet18OutputName, ElementType::kFloat32, {-1, 1000}};
}

ResNet18FixtureRequest RequireRequest(const ResNet18FixtureTable& table, std::uint64_t fixture_id) {
    ResNet18FixtureRequest request = resnet18_request_from_fixture(table, fixture_id);
    EXPECT_EQ(request.error.kind, NeuralReferenceErrorKind::kNone) << request.error.message;
    EXPECT_EQ(request.images.name, kResNet18InputName);
    EXPECT_EQ(request.images.shape, (std::vector<std::int64_t>{1, 3, 224, 224}));
    EXPECT_EQ(request.images.values.size(), kResNet18ImageElements);
    EXPECT_EQ(request.expected_logits.size(), kResNet18LogitElements);
    return request;
}

class ResNet18InferenceTest : public testing::Test {
protected:
    static void SetUpTestSuite() {
        ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0);
        loaded_ = load_inference_engine(INFEREDGE_RESNET18_MODEL_PATH);
        ASSERT_EQ(loaded_.error.kind, InferenceErrorKind::kNone) << loaded_.error.message;
        ASSERT_NE(loaded_.engine, nullptr);
        table_ = load_resnet18_fixture_table(INFEREDGE_RESNET18_FIXTURE_DIR);
        ASSERT_EQ(table_.error.kind, NeuralReferenceErrorKind::kNone) << table_.error.message;
        ASSERT_EQ(table_.table.images.size(), 8U * kResNet18ImageElements);
        ASSERT_EQ(table_.table.logits.size(), 8U * kResNet18LogitElements);
    }

    static void TearDownTestSuite() { loaded_.engine.reset(); }

    static inferedge::EngineLoad loaded_;
    static inferedge::ResNet18FixtureLoad table_;
};

inferedge::EngineLoad ResNet18InferenceTest::loaded_;
inferedge::ResNet18FixtureLoad ResNet18InferenceTest::table_;

} // namespace

TEST_F(ResNet18InferenceTest, FixtureIdentitiesAreNotAttemptIds) {
    EXPECT_TRUE(resnet18_fixture_id_valid(1000));
    EXPECT_TRUE(resnet18_fixture_id_valid(1007));
    EXPECT_FALSE(resnet18_fixture_id_valid(999));
    EXPECT_FALSE(resnet18_fixture_id_valid(1008));
    EXPECT_FALSE(resnet18_fixture_id_valid(1));
    EXPECT_EQ(kResNet18FixtureIdMin, 1000U);
    EXPECT_EQ(kResNet18FixtureIdMax, 1007U);

    const auto generated =
        scheduler_benchmark_request(SchedulerBenchmarkModelId::kResnet18F32, 1000);
    EXPECT_TRUE(generated.inputs.empty());
    EXPECT_EQ(generated.expected_c0, 0.F);
    EXPECT_EQ(generated.expected_c1, 0.F);

    const ResNet18FixtureRequest row = RequireRequest(table_.table, 1000);
    EXPECT_EQ(row.images.values.size(), kResNet18ImageElements);
}

TEST_F(ResNet18InferenceTest, RejectsUnknownAndOutOfRangeFixtureIds) {
    for (const std::uint64_t fixture_id : {0ULL, 1ULL, 999ULL, 1008ULL, 2000ULL}) {
        const ResNet18FixtureRequest request =
            resnet18_request_from_fixture(table_.table, fixture_id);
        EXPECT_EQ(request.error.kind, NeuralReferenceErrorKind::kShape) << fixture_id;
        EXPECT_EQ(request.error.message, "unrecognized resnet18_f32 fixture identity");
        EXPECT_TRUE(request.images.values.empty());
        EXPECT_TRUE(request.expected_logits.empty());
    }
}

TEST_F(ResNet18InferenceTest, RejectsMissingAndShortFixtureFiles) {
    const std::filesystem::path missing_dir =
        std::filesystem::temp_directory_path() / "inferedge_resnet18_missing";
    std::filesystem::remove_all(missing_dir);
    std::filesystem::create_directories(missing_dir);
    const auto missing = load_resnet18_fixture_table(missing_dir);
    EXPECT_EQ(missing.error.kind, NeuralReferenceErrorKind::kNotFound);
    EXPECT_TRUE(missing.table.images.empty());

    const std::filesystem::path short_dir =
        std::filesystem::temp_directory_path() / "inferedge_resnet18_short";
    std::filesystem::remove_all(short_dir);
    std::filesystem::create_directories(short_dir);
    {
        std::ofstream out(short_dir / "b08.input.f32", std::ios::binary);
        const char bytes[] = {'\0', '\0', '\0', '\0'};
        out.write(bytes, 4);
    }
    const auto short_file = load_resnet18_fixture_table(short_dir);
    EXPECT_EQ(short_file.error.kind, NeuralReferenceErrorKind::kSize);

    const ResNet18FixtureTable incomplete;
    const ResNet18FixtureRequest request = resnet18_request_from_fixture(incomplete, 1000);
    EXPECT_EQ(request.error.kind, NeuralReferenceErrorKind::kSize);

    std::filesystem::remove_all(missing_dir);
    std::filesystem::remove_all(short_dir);
}

TEST_F(ResNet18InferenceTest, ValidatesOnnxMetadataAndRejectsGemm) {
    const auto digest = scheduler_benchmark_sha256_file(INFEREDGE_RESNET18_MODEL_PATH);
    ASSERT_TRUE(digest.ok);
    EXPECT_EQ(digest.hex, kResNet18OnnxSha256);

    EXPECT_TRUE(resnet18_metadata_error(*loaded_.engine).empty());
    EXPECT_TRUE(
        scheduler_benchmark_metadata_error(SchedulerBenchmarkModelId::kResnet18F32, *loaded_.engine)
            .empty());
    EXPECT_FALSE(
        scheduler_benchmark_metadata_error(SchedulerBenchmarkModelId::kGemmF32, *loaded_.engine)
            .empty());

    auto gemm = load_inference_engine(INFEREDGE_GEMM_MODEL_PATH);
    ASSERT_EQ(gemm.error.kind, InferenceErrorKind::kNone) << gemm.error.message;
    EXPECT_TRUE(
        scheduler_benchmark_metadata_error(SchedulerBenchmarkModelId::kGemmF32, *gemm.engine)
            .empty());
    EXPECT_FALSE(resnet18_metadata_error(*gemm.engine).empty());
    EXPECT_FALSE(
        scheduler_benchmark_metadata_error(SchedulerBenchmarkModelId::kResnet18F32, *gemm.engine)
            .empty());
}

TEST_F(ResNet18InferenceTest, MetadataMatchIsNotNumericalSuccess) {
    EXPECT_TRUE(resnet18_metadata_error(*loaded_.engine).empty());
    NamedTensor logits;
    logits.name = kResNet18OutputName;
    logits.shape = {1, 1000};
    logits.values.assign(kResNet18LogitElements, 0.F);
    const auto empty_reference = resnet18_check_logits(logits, {});
    EXPECT_FALSE(empty_reference.matched);
    EXPECT_EQ(empty_reference.message, "no numerical reference");
}

TEST_F(ResNet18InferenceTest, CopiesCachedRowsIntoRank4Requests) {
    const ResNet18FixtureRequest first = RequireRequest(table_.table, 1000);
    const ResNet18FixtureRequest last = RequireRequest(table_.table, 1007);
    EXPECT_NE(first.images.values, last.images.values);
    EXPECT_NE(first.expected_logits, last.expected_logits);
    EXPECT_EQ(first.images.values,
              std::vector<float>(table_.table.images.begin(),
                                 table_.table.images.begin() +
                                     static_cast<std::ptrdiff_t>(kResNet18ImageElements)));
}

TEST_F(ResNet18InferenceTest, ComparesLogitsWithFrozenTolerance) {
    const ResNet18FixtureRequest request = RequireRequest(table_.table, 1003);
    NamedTensor logits;
    logits.name = kResNet18OutputName;
    logits.shape = {1, 1000};
    logits.values = request.expected_logits;
    EXPECT_TRUE(resnet18_check_logits(logits, request.expected_logits).matched);

    logits.values[0] += 5.0e-5F;
    const auto mismatch = resnet18_check_logits(logits, request.expected_logits);
    EXPECT_FALSE(mismatch.matched);
    EXPECT_EQ(mismatch.message, "output exceeded the neural tolerance");
    EXPECT_FALSE(scheduler_benchmark_check_output(SchedulerBenchmarkModelId::kResnet18F32, logits,
                                                  request.expected_logits)
                     .matched);

    logits.values = request.expected_logits;
    logits.values[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(resnet18_check_logits(logits, request.expected_logits).message,
              "output is not finite");

    logits.values[0] = std::numeric_limits<float>::infinity();
    EXPECT_EQ(resnet18_check_logits(logits, request.expected_logits).message,
              "output is not finite");

    logits.values = request.expected_logits;
    logits.name = "product";
    EXPECT_EQ(resnet18_check_logits(logits, request.expected_logits).message,
              "output name mismatch");

    logits.name = kResNet18OutputName;
    logits.shape = {1, 512};
    logits.values.resize(512);
    EXPECT_EQ(resnet18_check_logits(logits, request.expected_logits).message,
              "output shape mismatch");

    EXPECT_FLOAT_EQ(kResNet18AbsTol, 1.0e-5F);
    EXPECT_FLOAT_EQ(kResNet18RelTol, 1.0e-5F);
}

TEST_F(ResNet18InferenceTest, EngineMatchesCommittedBatches) {
    ASSERT_EQ(loaded_.engine->inputs().size(), 1U);
    ASSERT_EQ(loaded_.engine->outputs().size(), 1U);
    EXPECT_EQ(loaded_.engine->inputs()[0].name, kResNet18InputName);
    EXPECT_EQ(loaded_.engine->outputs()[0].name, kResNet18OutputName);
    EXPECT_EQ(loaded_.engine->inputs()[0].dimensions, (std::vector<std::int64_t>{-1, 3, 224, 224}));
    EXPECT_EQ(loaded_.engine->outputs()[0].dimensions, (std::vector<std::int64_t>{-1, 1000}));

    for (const std::size_t batch : kEngineBatches) {
        const auto input =
            load_float32_file(FixtureFile(batch, "input"), batch * kResNet18ImageElements);
        ASSERT_EQ(input.error.kind, NeuralReferenceErrorKind::kNone) << input.error.message;
        const auto expected =
            load_float32_file(FixtureFile(batch, "expected"), batch * kResNet18LogitElements);
        ASSERT_EQ(expected.error.kind, NeuralReferenceErrorKind::kNone) << expected.error.message;

        const NamedTensorView view{kResNet18InputName,
                                   {static_cast<std::int64_t>(batch), 3, 224, 224},
                                   std::span<const float>{input.values}};
        const auto inferred = loaded_.engine->run(std::span<const NamedTensorView>{&view, 1});
        ASSERT_EQ(inferred.error.kind, InferenceErrorKind::kNone) << inferred.error.message;
        ASSERT_EQ(inferred.outputs.size(), 1U);
        EXPECT_EQ(inferred.outputs[0].name, kResNet18OutputName);
        ASSERT_EQ(inferred.outputs[0].shape.size(), 2U);
        EXPECT_EQ(inferred.outputs[0].shape[0], static_cast<std::int64_t>(batch));
        EXPECT_EQ(inferred.outputs[0].shape[1], 1000);
        ASSERT_EQ(inferred.outputs[0].values.size(), expected.values.size());

        const auto split = SplitBatchOutputs(loaded_.engine->outputs(), inferred.outputs, batch);
        ASSERT_EQ(split.error.kind, TensorBatchErrorKind::kNone) << split.error.message;
        ASSERT_EQ(split.requests.size(), batch);
        for (std::size_t row = 0; row < batch; ++row) {
            ASSERT_EQ(split.requests[row].size(), 1U);
            const std::span<const float> golden{
                expected.values.data() + row * kResNet18LogitElements, kResNet18LogitElements};
            const auto checked = resnet18_check_logits(split.requests[row][0], golden);
            EXPECT_TRUE(checked.matched)
                << "batch " << batch << " row " << row << " " << checked.message;
        }
    }
}

TEST_F(ResNet18InferenceTest, AssemblesAndSplitsRank4WithFixtureIdentity) {
    const std::uint64_t ids[] = {1000, 1007};
    std::vector<ResNet18FixtureRequest> rows;
    std::vector<std::vector<NamedTensor>> owned;
    for (const std::uint64_t fixture_id : ids) {
        rows.push_back(RequireRequest(table_.table, fixture_id));
        owned.push_back({rows.back().images});
    }
    const std::span<const NamedTensor> views[] = {owned[0], owned[1]};
    const std::vector<TensorMetadata> inputs = {ImageMeta()};
    const auto assembled = AssembleBatchInputs(inputs, views);
    ASSERT_EQ(assembled.error.kind, TensorBatchErrorKind::kNone) << assembled.error.message;
    ASSERT_EQ(assembled.inputs.size(), 1U);
    EXPECT_EQ(assembled.inputs[0].name, kResNet18InputName);
    EXPECT_EQ(assembled.inputs[0].shape, (std::vector<std::int64_t>{2, 3, 224, 224}));
    EXPECT_EQ(assembled.inputs[0].values.size(), 2U * kResNet18ImageElements);

    NamedTensor batched_logits;
    batched_logits.name = kResNet18OutputName;
    batched_logits.shape = {2, 1000};
    batched_logits.values.insert(batched_logits.values.end(), rows[0].expected_logits.begin(),
                                 rows[0].expected_logits.end());
    batched_logits.values.insert(batched_logits.values.end(), rows[1].expected_logits.begin(),
                                 rows[1].expected_logits.end());
    const std::vector<TensorMetadata> outputs = {LogitMeta()};
    const auto split =
        SplitBatchOutputs(outputs, std::span<const NamedTensor>{&batched_logits, 1}, 2);
    ASSERT_EQ(split.error.kind, TensorBatchErrorKind::kNone) << split.error.message;
    ASSERT_EQ(split.requests.size(), 2U);
    EXPECT_EQ(split.requests[0][0].values, rows[0].expected_logits);
    EXPECT_EQ(split.requests[1][0].values, rows[1].expected_logits);
    EXPECT_NE(split.requests[0][0].values, rows[1].expected_logits);
    EXPECT_TRUE(resnet18_check_logits(split.requests[0][0], rows[0].expected_logits).matched);
    EXPECT_TRUE(resnet18_check_logits(split.requests[1][0], rows[1].expected_logits).matched);
}

TEST_F(ResNet18InferenceTest, AssemblesAndSplitsApprovedBatchSizes) {
    const std::vector<TensorMetadata> inputs = {ImageMeta()};
    const std::vector<TensorMetadata> outputs = {LogitMeta()};
    for (const std::size_t batch : kEngineBatches) {
        std::vector<ResNet18FixtureRequest> rows;
        std::vector<std::vector<NamedTensor>> owned;
        std::vector<std::span<const NamedTensor>> views;
        rows.reserve(batch);
        owned.reserve(batch);
        for (std::size_t index = 0; index < batch; ++index) {
            rows.push_back(RequireRequest(table_.table, kResNet18FixtureIdMin + index));
            owned.push_back({rows.back().images});
        }
        views.reserve(batch);
        for (const auto& request : owned) {
            views.emplace_back(request);
        }
        const auto assembled = AssembleBatchInputs(inputs, views);
        ASSERT_EQ(assembled.error.kind, TensorBatchErrorKind::kNone) << assembled.error.message;
        ASSERT_EQ(assembled.inputs.size(), 1U);
        EXPECT_EQ(assembled.inputs[0].shape,
                  (std::vector<std::int64_t>{static_cast<std::int64_t>(batch), 3, 224, 224}));

        NamedTensor batched;
        batched.name = kResNet18OutputName;
        batched.shape = {static_cast<std::int64_t>(batch), 1000};
        for (const auto& row : rows) {
            batched.values.insert(batched.values.end(), row.expected_logits.begin(),
                                  row.expected_logits.end());
        }
        const auto split =
            SplitBatchOutputs(outputs, std::span<const NamedTensor>{&batched, 1}, batch);
        ASSERT_EQ(split.error.kind, TensorBatchErrorKind::kNone) << split.error.message;
        ASSERT_EQ(split.requests.size(), batch);
        for (std::size_t index = 0; index < batch; ++index) {
            EXPECT_TRUE(resnet18_check_logits(split.requests[index][0], rows[index].expected_logits)
                            .matched)
                << "batch " << batch << " row " << index;
        }
    }
}

TEST_F(ResNet18InferenceTest, RejectsMalformedRank4AssemblyAndSplit) {
    const ResNet18FixtureRequest row = RequireRequest(table_.table, 1000);
    const std::vector<TensorMetadata> inputs = {ImageMeta()};
    const std::vector<TensorMetadata> outputs = {LogitMeta()};

    NamedTensor wrong_rank = row.images;
    wrong_rank.shape = {1, static_cast<std::int64_t>(kResNet18ImageElements)};
    const std::span<const NamedTensor> rank_view[] = {std::span<const NamedTensor>{&wrong_rank, 1}};
    EXPECT_EQ(AssembleBatchInputs(inputs, rank_view).error.kind,
              TensorBatchErrorKind::kRankMismatch);

    NamedTensor wrong_hw = row.images;
    wrong_hw.shape = {1, 3, 32, 32};
    wrong_hw.values.resize(3 * 32 * 32);
    const std::span<const NamedTensor> hw_view[] = {std::span<const NamedTensor>{&wrong_hw, 1}};
    EXPECT_EQ(AssembleBatchInputs(inputs, hw_view).error.kind,
              TensorBatchErrorKind::kShapeMismatch);

    NamedTensor wrong_name = row.images;
    wrong_name.name = "features";
    const std::span<const NamedTensor> name_view[] = {std::span<const NamedTensor>{&wrong_name, 1}};
    EXPECT_EQ(AssembleBatchInputs(inputs, name_view).error.kind,
              TensorBatchErrorKind::kUnexpectedTensor);

    NamedTensor wrong_axis = row.images;
    wrong_axis.shape = {2, 3, 224, 224};
    wrong_axis.values.resize(2 * kResNet18ImageElements);
    const std::span<const NamedTensor> axis_view[] = {std::span<const NamedTensor>{&wrong_axis, 1}};
    EXPECT_EQ(AssembleBatchInputs(inputs, axis_view).error.kind,
              TensorBatchErrorKind::kBatchAxisMismatch);

    NamedTensor bad_count = row.images;
    bad_count.values.resize(16);
    const std::span<const NamedTensor> count_view[] = {std::span<const NamedTensor>{&bad_count, 1}};
    EXPECT_EQ(AssembleBatchInputs(inputs, count_view).error.kind,
              TensorBatchErrorKind::kElementCountMismatch);

    NamedTensor logits;
    logits.name = "product";
    logits.shape = {1, 1000};
    logits.values.assign(kResNet18LogitElements, 0.F);
    EXPECT_EQ(SplitBatchOutputs(outputs, std::span<const NamedTensor>{&logits, 1}, 1).error.kind,
              TensorBatchErrorKind::kUnexpectedTensor);

    logits.name = kResNet18OutputName;
    logits.shape = {1, 10};
    logits.values.resize(10);
    EXPECT_EQ(SplitBatchOutputs(outputs, std::span<const NamedTensor>{&logits, 1}, 1).error.kind,
              TensorBatchErrorKind::kShapeMismatch);
}
