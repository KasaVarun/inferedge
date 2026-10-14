#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace inferedge {

// Independent float32 reference for the three dynamic-batch fixtures.
// It reads a weight sidecar and does not call ONNX Runtime.
enum class NeuralWorkload {
    kMlpSmall,
    kMlpMedium,
    kGemm,
};

enum class NeuralReferenceErrorKind {
    kNone,
    kNotFound,
    kSize,
    kNonFinite,
    kShape,
    kOverflow,
};

struct NeuralReferenceError {
    NeuralReferenceErrorKind kind{NeuralReferenceErrorKind::kNone};
    std::string message;
};

struct NeuralContract {
    NeuralWorkload workload{NeuralWorkload::kMlpSmall};
    const char* input_name{"features"};
    const char* output_name{"logits"};
    std::size_t input_width{0};
    std::size_t output_width{0};
    std::size_t parameter_count{0};
};

struct NeuralTolerance {
    float absolute{0.F};
    float relative{0.F};
};

struct NeuralFloatFile {
    NeuralReferenceError error;
    std::vector<float> values;
};

// The worst finite comparison, or the first non-finite value. within is true
// only when every pair is finite and inside the workload tolerance.
struct NeuralMismatch {
    bool within{false};
    std::size_t index{0};
    float absolute_error{0.F};
    float ratio{0.F};
};

struct NeuralReferenceResult {
    NeuralReferenceError error;
    std::vector<float> values;
};

[[nodiscard]] NeuralContract neural_contract(NeuralWorkload workload) noexcept;

[[nodiscard]] NeuralTolerance neural_tolerance(NeuralWorkload workload) noexcept;

// expected_count is the number of float32 values. The file must contain
// exactly that many little-endian bytes and every value must be finite.
[[nodiscard]] NeuralFloatFile load_float32_file(const std::filesystem::path& path,
                                                std::size_t expected_count);

// inputs holds batch rows of contract.input_width. weights is the sidecar
// order: MLP W1, b1, W2, b2, W3, b3, or GEMM weight. MatMul accumulates from
// +0 in k order with index k * N + n. Bias is a later Add. ReLU follows the
// first two MLP layers only.
[[nodiscard]] NeuralReferenceResult evaluate_neural_reference(NeuralWorkload workload,
                                                              std::span<const float> inputs,
                                                              std::size_t batch,
                                                              std::span<const float> weights);

[[nodiscard]] NeuralMismatch neural_worst_mismatch(NeuralWorkload workload,
                                                   std::span<const float> actual,
                                                   std::span<const float> reference);

} // namespace inferedge
