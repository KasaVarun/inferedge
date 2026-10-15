#include "inferedge/resnet18_fixtures.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

namespace inferedge {
namespace {

NeuralReferenceError Fail(NeuralReferenceErrorKind kind, std::string message) {
    return NeuralReferenceError{kind, std::move(message)};
}

} // namespace

bool resnet18_fixture_id_valid(std::uint64_t fixture_id) noexcept {
    return fixture_id >= kResNet18FixtureIdMin && fixture_id <= kResNet18FixtureIdMax;
}

ResNet18FixtureLoad load_resnet18_fixture_table(const std::filesystem::path& fixture_dir) {
    ResNet18FixtureLoad loaded;
    const NeuralFloatFile images = load_float32_file(fixture_dir / "b08.input.f32",
                                                     kResNet18FixtureRows * kResNet18ImageElements);
    if (images.error.kind != NeuralReferenceErrorKind::kNone) {
        loaded.error = images.error;
        return loaded;
    }
    const NeuralFloatFile logits = load_float32_file(fixture_dir / "b08.expected.f32",
                                                     kResNet18FixtureRows * kResNet18LogitElements);
    if (logits.error.kind != NeuralReferenceErrorKind::kNone) {
        loaded.error = logits.error;
        return loaded;
    }
    loaded.table.images = images.values;
    loaded.table.logits = logits.values;
    return loaded;
}

ResNet18FixtureRequest resnet18_request_from_fixture(const ResNet18FixtureTable& table,
                                                     std::uint64_t fixture_id) {
    ResNet18FixtureRequest request;
    if (!resnet18_fixture_id_valid(fixture_id)) {
        request.error =
            Fail(NeuralReferenceErrorKind::kShape, "unrecognized resnet18_f32 fixture identity");
        return request;
    }
    const std::size_t expected_images = kResNet18FixtureRows * kResNet18ImageElements;
    const std::size_t expected_logits = kResNet18FixtureRows * kResNet18LogitElements;
    if (table.images.size() != expected_images || table.logits.size() != expected_logits) {
        request.error =
            Fail(NeuralReferenceErrorKind::kSize, "resnet18_f32 fixture table is incomplete");
        return request;
    }
    const std::size_t row = static_cast<std::size_t>(fixture_id - kResNet18FixtureIdMin);
    const std::size_t image_offset = row * kResNet18ImageElements;
    const std::size_t logit_offset = row * kResNet18LogitElements;
    request.images.name = kResNet18InputName;
    request.images.shape = {1, 3, 224, 224};
    request.images.values.assign(table.images.begin() + static_cast<std::ptrdiff_t>(image_offset),
                                 table.images.begin() + static_cast<std::ptrdiff_t>(image_offset) +
                                     static_cast<std::ptrdiff_t>(kResNet18ImageElements));
    request.expected_logits.assign(table.logits.begin() + static_cast<std::ptrdiff_t>(logit_offset),
                                   table.logits.begin() +
                                       static_cast<std::ptrdiff_t>(logit_offset) +
                                       static_cast<std::ptrdiff_t>(kResNet18LogitElements));
    return request;
}

std::string resnet18_metadata_error(const InferenceEngine& engine) {
    const std::vector<TensorMetadata>& inputs = engine.inputs();
    const std::vector<TensorMetadata>& outputs = engine.outputs();
    if (inputs.size() != 1 || outputs.size() != 1) {
        return "model metadata does not match resnet18_f32";
    }
    const TensorMetadata& images = inputs[0];
    const TensorMetadata& logits = outputs[0];
    if (images.name != kResNet18InputName || logits.name != kResNet18OutputName) {
        return "model metadata does not match resnet18_f32";
    }
    if (images.element_type != ElementType::kFloat32 ||
        logits.element_type != ElementType::kFloat32) {
        return "model metadata does not match resnet18_f32";
    }
    if (images.dimensions.size() != 4 || images.dimensions[0] >= 0 || images.dimensions[1] != 3 ||
        images.dimensions[2] != 224 || images.dimensions[3] != 224) {
        return "model metadata does not match resnet18_f32";
    }
    if (logits.dimensions.size() != 2 || logits.dimensions[0] >= 0 ||
        logits.dimensions[1] != 1000) {
        return "model metadata does not match resnet18_f32";
    }
    return {};
}

ResNet18OutputCheck resnet18_check_logits(const NamedTensor& output,
                                          std::span<const float> expected) {
    ResNet18OutputCheck checked;
    if (expected.empty()) {
        checked.message = "no numerical reference";
        return checked;
    }
    if (output.name != kResNet18OutputName) {
        checked.message = "output name mismatch";
        return checked;
    }
    if (output.shape.size() != 2 || output.shape[0] != 1 || output.shape[1] != 1000 ||
        output.values.size() != kResNet18LogitElements) {
        checked.message = "output shape mismatch";
        return checked;
    }
    if (expected.size() != kResNet18LogitElements) {
        checked.message = "reference width mismatch";
        return checked;
    }
    for (const float value : output.values) {
        if (!std::isfinite(value)) {
            checked.message = "output is not finite";
            return checked;
        }
    }
    for (const float value : expected) {
        if (!std::isfinite(value)) {
            checked.message = "reference is not finite";
            return checked;
        }
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float actual = output.values[index];
        const float golden = expected[index];
        const float limit = kResNet18AbsTol + kResNet18RelTol * std::fabs(golden);
        if (std::fabs(actual - golden) > limit) {
            checked.message = "output exceeded the neural tolerance";
            return checked;
        }
    }
    checked.matched = true;
    return checked;
}

} // namespace inferedge
