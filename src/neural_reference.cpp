#include "inferedge/neural_reference.hpp"

#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <utility>

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace inferedge {
namespace {

NeuralReferenceError Fail(NeuralReferenceErrorKind kind, std::string message) {
    return NeuralReferenceError{kind, std::move(message)};
}

bool Multiply(std::size_t left, std::size_t right, std::size_t& product) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    product = left * right;
    return true;
}

bool AddOffset(std::size_t cursor, std::size_t count, std::size_t& next) {
    if (count > std::numeric_limits<std::size_t>::max() - cursor) {
        return false;
    }
    next = cursor + count;
    return true;
}

float MultiplyFloat(float left, float right) {
    return left * right;
}

float AddFloat(float left, float right) {
    return left + right;
}

NeuralReferenceError Matmul(std::span<const float> input, std::span<const float> weight,
                            std::size_t fan_in, std::size_t fan_out, std::vector<float>& output) {
    std::size_t weight_count = 0;
    if (!Multiply(fan_in, fan_out, weight_count)) {
        return Fail(NeuralReferenceErrorKind::kOverflow, "weight shape product overflows");
    }
    if (input.size() != fan_in || weight.size() != weight_count) {
        return Fail(NeuralReferenceErrorKind::kShape, "matmul shape does not match the weights");
    }
    output.assign(fan_out, 0.F);
    for (std::size_t column = 0; column < fan_out; ++column) {
        float accumulator = 0.F;
        for (std::size_t reduction = 0; reduction < fan_in; ++reduction) {
            std::size_t offset = 0;
            if (!Multiply(reduction, fan_out, offset) || !AddOffset(offset, column, offset)) {
                return Fail(NeuralReferenceErrorKind::kOverflow, "weight index overflows");
            }
            const float product = MultiplyFloat(input[reduction], weight[offset]);
            accumulator = AddFloat(accumulator, product);
        }
        output[column] = accumulator;
    }
    return {};
}

NeuralReferenceError AddBias(std::span<float> row, std::span<const float> bias) {
    if (row.size() != bias.size()) {
        return Fail(NeuralReferenceErrorKind::kShape, "bias width does not match the row");
    }
    for (std::size_t index = 0; index < row.size(); ++index) {
        row[index] = AddFloat(row[index], bias[index]);
    }
    return {};
}

void Relu(std::span<float> row) {
    for (float& value : row) {
        value = value > 0.F ? value : 0.F;
    }
}

struct Layer {
    std::size_t fan_in{0};
    std::size_t fan_out{0};
    bool relu{false};
};

NeuralReferenceError EvaluateLayers(std::span<const float> inputs, std::size_t batch,
                                    std::size_t input_width, std::span<const float> weights,
                                    std::span<const Layer> layers, std::size_t output_width,
                                    std::vector<float>& outputs) {
    std::size_t input_count = 0;
    std::size_t output_count = 0;
    if (batch == 0 || !Multiply(batch, input_width, input_count) ||
        !Multiply(batch, output_width, output_count)) {
        return Fail(NeuralReferenceErrorKind::kOverflow, "batch shape product overflows");
    }
    if (inputs.size() != input_count) {
        return Fail(NeuralReferenceErrorKind::kShape,
                    "input element count does not match the batch");
    }
    std::size_t required_weights = 0;
    for (const Layer& layer : layers) {
        std::size_t matrix = 0;
        std::size_t layer_count = 0;
        if (!Multiply(layer.fan_in, layer.fan_out, matrix) ||
            !AddOffset(matrix, layer.fan_out, layer_count) ||
            !AddOffset(required_weights, layer_count, required_weights)) {
            return Fail(NeuralReferenceErrorKind::kOverflow, "layer weight span overflows");
        }
    }
    if (required_weights != weights.size()) {
        return Fail(NeuralReferenceErrorKind::kSize,
                    "weight sidecar size does not match the layers");
    }
    outputs.assign(output_count, 0.F);
    std::vector<float> current(input_width);
    std::vector<float> activated;
    for (std::size_t row = 0; row < batch; ++row) {
        const std::size_t input_offset = row * input_width;
        current.assign(inputs.begin() + static_cast<std::ptrdiff_t>(input_offset),
                       inputs.begin() + static_cast<std::ptrdiff_t>(input_offset + input_width));
        std::size_t weight_cursor = 0;
        for (const Layer& layer : layers) {
            std::size_t matrix = 0;
            if (!Multiply(layer.fan_in, layer.fan_out, matrix)) {
                return Fail(NeuralReferenceErrorKind::kOverflow, "layer weight product overflows");
            }
            const std::span<const float> matrix_weights = weights.subspan(weight_cursor, matrix);
            weight_cursor += matrix;
            const std::span<const float> bias = weights.subspan(weight_cursor, layer.fan_out);
            weight_cursor += layer.fan_out;
            const NeuralReferenceError multiplied =
                Matmul(current, matrix_weights, layer.fan_in, layer.fan_out, activated);
            if (multiplied.kind != NeuralReferenceErrorKind::kNone) {
                return multiplied;
            }
            const NeuralReferenceError added = AddBias(activated, bias);
            if (added.kind != NeuralReferenceErrorKind::kNone) {
                return added;
            }
            if (layer.relu) {
                Relu(activated);
            }
            current.swap(activated);
        }
        if (current.size() != output_width) {
            return Fail(NeuralReferenceErrorKind::kShape,
                        "layer output width does not match the contract");
        }
        const std::size_t output_offset = row * output_width;
        for (std::size_t column = 0; column < output_width; ++column) {
            const float value = current[column];
            if (!std::isfinite(value)) {
                return Fail(NeuralReferenceErrorKind::kNonFinite, "reference output is not finite");
            }
            outputs[output_offset + column] = value;
        }
    }
    return {};
}

NeuralReferenceError EvaluateGemm(std::span<const float> inputs, std::size_t batch,
                                  std::span<const float> weights, std::vector<float>& outputs) {
    constexpr std::size_t kWidth = 512;
    std::size_t input_count = 0;
    std::size_t output_count = 0;
    std::size_t weight_count = 0;
    if (batch == 0 || !Multiply(batch, kWidth, input_count) ||
        !Multiply(batch, kWidth, output_count) || !Multiply(kWidth, kWidth, weight_count)) {
        return Fail(NeuralReferenceErrorKind::kOverflow, "gemm shape product overflows");
    }
    if (inputs.size() != input_count || weights.size() != weight_count) {
        return Fail(NeuralReferenceErrorKind::kShape, "gemm input or weight size does not match");
    }
    outputs.assign(output_count, 0.F);
    std::vector<float> row(kWidth);
    std::vector<float> product;
    for (std::size_t index = 0; index < batch; ++index) {
        const std::size_t offset = index * kWidth;
        row.assign(inputs.begin() + static_cast<std::ptrdiff_t>(offset),
                   inputs.begin() + static_cast<std::ptrdiff_t>(offset + kWidth));
        const NeuralReferenceError multiplied = Matmul(row, weights, kWidth, kWidth, product);
        if (multiplied.kind != NeuralReferenceErrorKind::kNone) {
            return multiplied;
        }
        for (std::size_t column = 0; column < kWidth; ++column) {
            const float value = product[column];
            if (!std::isfinite(value)) {
                return Fail(NeuralReferenceErrorKind::kNonFinite, "reference output is not finite");
            }
            outputs[offset + column] = value;
        }
    }
    return {};
}

} // namespace

NeuralContract neural_contract(NeuralWorkload workload) noexcept {
    switch (workload) {
    case NeuralWorkload::kMlpSmall:
        return NeuralContract{workload, "features", "logits", 32, 8, 4456};
    case NeuralWorkload::kMlpMedium:
        return NeuralContract{workload, "features", "logits", 256, 16, 100752};
    case NeuralWorkload::kGemm:
        return NeuralContract{workload, "features", "product", 512, 512, 262144};
    }
    return NeuralContract{};
}

NeuralTolerance neural_tolerance(NeuralWorkload workload) noexcept {
    switch (workload) {
    case NeuralWorkload::kMlpSmall:
    case NeuralWorkload::kMlpMedium:
        return NeuralTolerance{2.0e-4F, 2.0e-4F};
    case NeuralWorkload::kGemm:
        return NeuralTolerance{1.0e-4F, 1.0e-4F};
    }
    return NeuralTolerance{};
}

NeuralFloatFile load_float32_file(const std::filesystem::path& path, std::size_t expected_count) {
    std::error_code status;
    if (!std::filesystem::is_regular_file(path, status) || status) {
        return NeuralFloatFile{
            Fail(NeuralReferenceErrorKind::kNotFound, "float32 file is missing: " + path.string()),
            {}};
    }
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        return NeuralFloatFile{Fail(NeuralReferenceErrorKind::kNotFound,
                                    "float32 file cannot be opened: " + path.string()),
                               {}};
    }
    input.seekg(0, std::ios::end);
    const std::streamoff file_size = input.tellg();
    std::size_t byte_count = 0;
    if (expected_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        !Multiply(expected_count, sizeof(float), byte_count)) {
        return NeuralFloatFile{
            Fail(NeuralReferenceErrorKind::kOverflow, "float32 file size overflows"), {}};
    }
    if (file_size < 0 || static_cast<unsigned long long>(file_size) != byte_count) {
        return NeuralFloatFile{Fail(NeuralReferenceErrorKind::kSize,
                                    path.string() + " has " + std::to_string(file_size) +
                                        " bytes, expected " + std::to_string(byte_count)),
                               {}};
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> values(expected_count);
    for (std::size_t index = 0; index < expected_count; ++index) {
        unsigned char bytes[4] = {};
        input.read(reinterpret_cast<char*>(bytes), 4);
        if (!input) {
            return NeuralFloatFile{
                Fail(NeuralReferenceErrorKind::kSize, "short read of " + path.string()), {}};
        }
        const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                                   (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                                   (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                                   (static_cast<std::uint32_t>(bytes[3]) << 24U);
        const float value = std::bit_cast<float>(bits);
        if (!std::isfinite(value)) {
            return NeuralFloatFile{
                Fail(NeuralReferenceErrorKind::kNonFinite, "non-finite value in " + path.string()),
                {}};
        }
        values[index] = value;
    }
    return NeuralFloatFile{{}, std::move(values)};
}

NeuralReferenceResult evaluate_neural_reference(NeuralWorkload workload,
                                                std::span<const float> inputs, std::size_t batch,
                                                std::span<const float> weights) {
    const NeuralContract contract = neural_contract(workload);
    if (weights.size() != contract.parameter_count) {
        return NeuralReferenceResult{
            Fail(NeuralReferenceErrorKind::kSize, "weight count does not match the workload"), {}};
    }
    NeuralReferenceResult result;
    if (workload == NeuralWorkload::kGemm) {
        result.error = EvaluateGemm(inputs, batch, weights, result.values);
        if (result.error.kind != NeuralReferenceErrorKind::kNone) {
            result.values.clear();
        }
        return result;
    }
    const Layer small_layers[] = {{32, 64, true}, {64, 32, true}, {32, 8, false}};
    const Layer medium_layers[] = {{256, 256, true}, {256, 128, true}, {128, 16, false}};
    const std::span<const Layer> layers = workload == NeuralWorkload::kMlpSmall
                                              ? std::span<const Layer>{small_layers}
                                              : std::span<const Layer>{medium_layers};
    result.error = EvaluateLayers(inputs, batch, contract.input_width, weights, layers,
                                  contract.output_width, result.values);
    if (result.error.kind != NeuralReferenceErrorKind::kNone) {
        result.values.clear();
    }
    return result;
}

NeuralMismatch neural_worst_mismatch(NeuralWorkload workload, std::span<const float> actual,
                                     std::span<const float> reference) {
    NeuralMismatch mismatch;
    if (actual.size() != reference.size() || actual.empty()) {
        mismatch.within = false;
        mismatch.ratio = std::numeric_limits<float>::infinity();
        return mismatch;
    }
    const NeuralTolerance tolerance = neural_tolerance(workload);
    mismatch.within = true;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float seen = actual[index];
        const float expected = reference[index];
        if (!std::isfinite(seen) || !std::isfinite(expected)) {
            mismatch.within = false;
            mismatch.index = index;
            mismatch.absolute_error = std::numeric_limits<float>::infinity();
            mismatch.ratio = std::numeric_limits<float>::infinity();
            return mismatch;
        }
        const float absolute = std::fabs(seen - expected);
        const float limit = tolerance.absolute + tolerance.relative * std::fabs(expected);
        const float ratio =
            limit == 0.F ? std::numeric_limits<float>::infinity() : absolute / limit;
        if (ratio > mismatch.ratio) {
            mismatch.index = index;
            mismatch.absolute_error = absolute;
            mismatch.ratio = ratio;
        }
        if (!(absolute <= limit)) {
            mismatch.within = false;
        }
    }
    return mismatch;
}

} // namespace inferedge
