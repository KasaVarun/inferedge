#include "inferedge/neural_reference.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

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

namespace {

using inferedge::evaluate_neural_reference;
using inferedge::load_float32_file;
using inferedge::neural_contract;
using inferedge::neural_worst_mismatch;
using inferedge::NeuralContract;
using inferedge::NeuralMismatch;
using inferedge::NeuralReferenceErrorKind;
using inferedge::NeuralWorkload;

struct WorkloadPaths {
    NeuralWorkload workload;
    const char* weights;
    const char* fixtures;
};

constexpr WorkloadPaths kWorkloads[] = {
    {NeuralWorkload::kMlpSmall, INFEREDGE_MLP_SMALL_WEIGHTS_PATH, INFEREDGE_MLP_SMALL_FIXTURE_DIR},
    {NeuralWorkload::kMlpMedium, INFEREDGE_MLP_MEDIUM_WEIGHTS_PATH,
     INFEREDGE_MLP_MEDIUM_FIXTURE_DIR},
    {NeuralWorkload::kGemm, INFEREDGE_GEMM_WEIGHTS_PATH, INFEREDGE_GEMM_FIXTURE_DIR},
};

constexpr std::size_t kBatches[] = {1, 2, 4, 8, 16, 32};

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

std::filesystem::path TemporaryFile(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

void WriteBytes(const std::filesystem::path& path, const std::vector<unsigned char>& bytes) {
    std::ofstream output{path, std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(output);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(output);
}

std::vector<unsigned char> FloatBits(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return {static_cast<unsigned char>(bits & 0xFFU),
            static_cast<unsigned char>((bits >> 8U) & 0xFFU),
            static_cast<unsigned char>((bits >> 16U) & 0xFFU),
            static_cast<unsigned char>((bits >> 24U) & 0xFFU)};
}

std::size_t LargestAbsIndex(const std::vector<float>& values, std::size_t begin,
                            std::size_t count) {
    std::size_t chosen = begin;
    float best = 0.F;
    for (std::size_t index = begin; index < begin + count; ++index) {
        const float magnitude = std::fabs(values[index]);
        if (magnitude > best) {
            best = magnitude;
            chosen = index;
        }
    }
    return chosen;
}

} // namespace

TEST(NeuralReferenceTest, ReproducesEveryKnownAnswerFixtureExactly) {
    for (const WorkloadPaths& paths : kWorkloads) {
        const NeuralContract contract = neural_contract(paths.workload);
        const auto weights = load_float32_file(paths.weights, contract.parameter_count);
        ASSERT_EQ(weights.error.kind, NeuralReferenceErrorKind::kNone) << weights.error.message;
        for (const std::size_t batch : kBatches) {
            const auto input = load_float32_file(FixtureFile(paths.fixtures, batch, "input"),
                                                 batch * contract.input_width);
            const auto expected = load_float32_file(FixtureFile(paths.fixtures, batch, "expected"),
                                                    batch * contract.output_width);
            ASSERT_EQ(input.error.kind, NeuralReferenceErrorKind::kNone) << input.error.message;
            ASSERT_EQ(expected.error.kind, NeuralReferenceErrorKind::kNone)
                << expected.error.message;
            const auto actual =
                evaluate_neural_reference(paths.workload, input.values, batch, weights.values);
            ASSERT_EQ(actual.error.kind, NeuralReferenceErrorKind::kNone) << actual.error.message;
            ASSERT_EQ(actual.values.size(), expected.values.size());
            for (std::size_t index = 0; index < expected.values.size(); ++index) {
                EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.values[index]),
                          std::bit_cast<std::uint32_t>(expected.values[index]))
                    << paths.fixtures << " batch " << batch << " index " << index;
            }
        }
    }
}

TEST(NeuralReferenceTest, RejectsAShortLongOrNonFiniteSidecar) {
    const auto path = TemporaryFile("inferedge-c2-sidecar.weights");
    WriteBytes(path, {0x00, 0x00, 0x80, 0x3F, 0x00});
    const auto extra = load_float32_file(path, 1);
    EXPECT_EQ(extra.error.kind, NeuralReferenceErrorKind::kSize);
    WriteBytes(path, {0x00, 0x00, 0x80, 0x3F});
    const auto short_file = load_float32_file(path, 2);
    EXPECT_EQ(short_file.error.kind, NeuralReferenceErrorKind::kSize);
    const auto nan_bits = FloatBits(std::numeric_limits<float>::quiet_NaN());
    WriteBytes(path, nan_bits);
    const auto non_finite = load_float32_file(path, 1);
    EXPECT_EQ(non_finite.error.kind, NeuralReferenceErrorKind::kNonFinite);
    std::filesystem::remove(path);
    const auto missing = load_float32_file(path, 1);
    EXPECT_EQ(missing.error.kind, NeuralReferenceErrorKind::kNotFound);
}

TEST(NeuralReferenceTest, RejectsTruncatedAndExtendedFixtures) {
    const auto original =
        load_float32_file(FixtureFile(INFEREDGE_MLP_SMALL_FIXTURE_DIR, 1, "input"), 32);
    ASSERT_EQ(original.error.kind, NeuralReferenceErrorKind::kNone);
    const auto path = TemporaryFile("inferedge-c2-fixture.f32");
    std::vector<unsigned char> bytes;
    for (const float value : original.values) {
        const auto encoded = FloatBits(value);
        bytes.insert(bytes.end(), encoded.begin(), encoded.end());
    }
    bytes.pop_back();
    WriteBytes(path, bytes);
    EXPECT_EQ(load_float32_file(path, 32).error.kind, NeuralReferenceErrorKind::kSize);
    bytes.push_back(0x00);
    bytes.insert(bytes.end(), {0x00, 0x00, 0x00, 0x00});
    WriteBytes(path, bytes);
    EXPECT_EQ(load_float32_file(path, 32).error.kind, NeuralReferenceErrorKind::kSize);
    std::filesystem::remove(path);
}

TEST(NeuralReferenceTest, RejectsAMismatchedInputOrWeightCount) {
    const NeuralContract contract = neural_contract(NeuralWorkload::kMlpSmall);
    const auto weights =
        load_float32_file(INFEREDGE_MLP_SMALL_WEIGHTS_PATH, contract.parameter_count);
    ASSERT_EQ(weights.error.kind, NeuralReferenceErrorKind::kNone);
    const std::vector<float> inputs(contract.input_width, 0.F);
    const auto bad_batch =
        evaluate_neural_reference(NeuralWorkload::kMlpSmall, inputs, 0, weights.values);
    EXPECT_EQ(bad_batch.error.kind, NeuralReferenceErrorKind::kOverflow);
    const auto bad_width =
        evaluate_neural_reference(NeuralWorkload::kMlpSmall, inputs, 2, weights.values);
    EXPECT_EQ(bad_width.error.kind, NeuralReferenceErrorKind::kShape);
    std::vector<float> short_weights = weights.values;
    short_weights.pop_back();
    const auto bad_weights =
        evaluate_neural_reference(NeuralWorkload::kMlpSmall, inputs, 1, short_weights);
    EXPECT_EQ(bad_weights.error.kind, NeuralReferenceErrorKind::kSize);
}

TEST(NeuralReferenceTest, RejectsSwappedRowsAndAnAlteredElement) {
    for (const WorkloadPaths& paths : kWorkloads) {
        const NeuralContract contract = neural_contract(paths.workload);
        const auto weights = load_float32_file(paths.weights, contract.parameter_count);
        ASSERT_EQ(weights.error.kind, NeuralReferenceErrorKind::kNone) << weights.error.message;
        std::vector<float> inputs(contract.input_width * 2, 0.F);
        for (std::size_t index = 0; index < contract.input_width; ++index) {
            inputs[contract.input_width + index] = 1.F;
        }
        const auto reference = evaluate_neural_reference(paths.workload, inputs, 2, weights.values);
        ASSERT_EQ(reference.error.kind, NeuralReferenceErrorKind::kNone) << reference.error.message;
        std::vector<float> swapped = reference.values;
        const std::size_t width = contract.output_width;
        for (std::size_t index = 0; index < width; ++index) {
            std::swap(swapped[index], swapped[width + index]);
        }
        const NeuralMismatch swap_gap =
            neural_worst_mismatch(paths.workload, swapped, reference.values);
        EXPECT_FALSE(swap_gap.within) << paths.weights;
        EXPECT_GT(swap_gap.ratio, 1.F) << paths.weights;

        std::vector<float> altered = reference.values;
        altered[0] = altered[0] + 1.F;
        const NeuralMismatch altered_gap =
            neural_worst_mismatch(paths.workload, altered, reference.values);
        EXPECT_FALSE(altered_gap.within) << paths.weights;
        EXPECT_GT(altered_gap.absolute_error, 0.5F) << paths.weights;
    }
}

TEST(NeuralReferenceTest, RejectsADroppedBiasAndAFlippedWeight) {
    for (const NeuralWorkload workload : {NeuralWorkload::kMlpSmall, NeuralWorkload::kMlpMedium}) {
        const NeuralContract contract = neural_contract(workload);
        const char* weights_path = workload == NeuralWorkload::kMlpSmall
                                       ? INFEREDGE_MLP_SMALL_WEIGHTS_PATH
                                       : INFEREDGE_MLP_MEDIUM_WEIGHTS_PATH;
        const auto weights = load_float32_file(weights_path, contract.parameter_count);
        ASSERT_EQ(weights.error.kind, NeuralReferenceErrorKind::kNone);
        const std::vector<float> input(contract.input_width, 0.25F);
        const auto original = evaluate_neural_reference(workload, input, 1, weights.values);
        ASSERT_EQ(original.error.kind, NeuralReferenceErrorKind::kNone);
        const std::size_t bias_begin = contract.parameter_count - contract.output_width;
        const std::size_t bias_index =
            LargestAbsIndex(weights.values, bias_begin, contract.output_width);
        ASSERT_GT(std::fabs(weights.values[bias_index]), 1.0e-2F);
        std::vector<float> dropped = weights.values;
        dropped[bias_index] = 0.F;
        const auto without_bias = evaluate_neural_reference(workload, input, 1, dropped);
        ASSERT_EQ(without_bias.error.kind, NeuralReferenceErrorKind::kNone);
        const NeuralMismatch bias_gap =
            neural_worst_mismatch(workload, without_bias.values, original.values);
        EXPECT_FALSE(bias_gap.within) << "dropped bias index " << bias_index;
        EXPECT_GT(bias_gap.ratio, 1.F);

        const std::size_t last_fan_in = workload == NeuralWorkload::kMlpSmall ? 32U : 128U;
        const std::size_t matrix_count = last_fan_in * contract.output_width;
        const std::size_t matrix_begin =
            contract.parameter_count - contract.output_width - matrix_count;
        bool found_flip = false;
        std::size_t flipped = matrix_begin;
        float flipped_ratio = 0.F;
        for (std::size_t index = matrix_begin; index < matrix_begin + matrix_count; ++index) {
            if (weights.values[index] == 0.F) {
                continue;
            }
            std::vector<float> negated = weights.values;
            negated[index] = -negated[index];
            const auto flipped_run = evaluate_neural_reference(workload, input, 1, negated);
            ASSERT_EQ(flipped_run.error.kind, NeuralReferenceErrorKind::kNone);
            const NeuralMismatch weight_gap =
                neural_worst_mismatch(workload, flipped_run.values, original.values);
            if (!weight_gap.within && weight_gap.ratio > 1.F) {
                found_flip = true;
                flipped = index;
                flipped_ratio = weight_gap.ratio;
                break;
            }
        }
        EXPECT_TRUE(found_flip) << "no last-layer sign flip exceeded tolerance";
        EXPECT_GT(flipped_ratio, 1.F) << "flipped weight index " << flipped;
    }
}

TEST(NeuralReferenceTest, RejectsNonFiniteComparisons) {
    const std::vector<float> expected{0.F};
    for (const float corrupt :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity()}) {
        const std::vector<float> actual{corrupt};
        const NeuralMismatch mismatch =
            neural_worst_mismatch(NeuralWorkload::kGemm, actual, expected);
        EXPECT_FALSE(mismatch.within);
        EXPECT_TRUE(std::isinf(mismatch.ratio));
    }
}
