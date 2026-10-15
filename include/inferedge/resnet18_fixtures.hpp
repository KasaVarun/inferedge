#pragma once

#include "inferedge/inference_engine.hpp"
#include "inferedge/neural_reference.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace inferedge {

// Canonical ResNet-18 fixture identities. These are not scheduler attempt ids.
inline constexpr std::uint64_t kResNet18FixtureIdMin = 1000;
inline constexpr std::uint64_t kResNet18FixtureIdMax = 1007;
inline constexpr std::size_t kResNet18FixtureRows = 8;
inline constexpr std::size_t kResNet18ImageElements = 3 * 224 * 224;
inline constexpr std::size_t kResNet18LogitElements = 1000;
inline constexpr float kResNet18AbsTol = 1.0e-5F;
inline constexpr float kResNet18RelTol = 1.0e-5F;
inline constexpr char kResNet18OnnxSha256[] =
    "cfc843f2939dc6065d95d911ec778f57ca5f0f25faa4988a5b5c61011bebd935";
inline constexpr char kResNet18InputName[] = "images";
inline constexpr char kResNet18OutputName[] = "logits";

// Immutable table loaded once from b08.input.f32 and b08.expected.f32.
struct ResNet18FixtureTable {
    std::vector<float> images;
    std::vector<float> logits;
};

struct ResNet18FixtureLoad {
    NeuralReferenceError error;
    ResNet18FixtureTable table;
};

struct ResNet18FixtureRequest {
    NeuralReferenceError error;
    NamedTensor images;
    std::vector<float> expected_logits;
};

struct ResNet18OutputCheck {
    bool matched{false};
    std::string message;
};

[[nodiscard]] bool resnet18_fixture_id_valid(std::uint64_t fixture_id) noexcept;

// Correctness-testing replay only. attempt_id is 1-based. Images are not unique:
// fixture_id = 1000 + ((attempt_id - 1) % 8). Attempt 0 is rejected.
[[nodiscard]] bool resnet18_fixture_id_for_attempt(std::uint64_t attempt_id,
                                                   std::uint64_t& fixture_id) noexcept;

// Reads b08 packed NCHW images and packed logits. No ORT. No per-request I/O
// after this call. Missing files and wrong sizes fail.
[[nodiscard]] ResNet18FixtureLoad
load_resnet18_fixture_table(const std::filesystem::path& fixture_dir);

// Copies one cached row. fixture_id must be 1000..1007. Does not map attempt
// ids and does not generate images.
[[nodiscard]] ResNet18FixtureRequest
resnet18_request_from_fixture(const ResNet18FixtureTable& table, std::uint64_t fixture_id);

// Empty string means names, ranks, dtypes, and static axes match. That is not
// a numerical success.
[[nodiscard]] std::string resnet18_metadata_error(const InferenceEngine& engine);

// Compares one split request: logits [1,1000], 1000 finite values, frozen
// atol/rtol. matched is never true for metadata-only callers.
[[nodiscard]] ResNet18OutputCheck resnet18_check_logits(const NamedTensor& output,
                                                        std::span<const float> expected);

} // namespace inferedge
