#include "inferedge/run_command.hpp"

#include "inferedge/inference_engine.hpp"

#include <charconv>
#include <system_error>
#include <utility>

namespace inferedge {
namespace {

constexpr int kExitSuccess = 0;
constexpr int kExitModel = 3;
constexpr int kExitInput = 4;
constexpr int kExitExecution = 5;

CliResult fail(int exit_code, std::string message) {
    CliResult result;
    result.action = CliAction::kError;
    result.exit_code = exit_code;
    result.message = "error: " + std::move(message) + "\n";
    return result;
}

int exit_code_for(InferenceErrorKind kind) {
    switch (kind) {
    case InferenceErrorKind::kNone:
        return kExitSuccess;
    case InferenceErrorKind::kModelNotFound:
    case InferenceErrorKind::kModelInvalid:
    case InferenceErrorKind::kUnsupportedType:
        return kExitModel;
    case InferenceErrorKind::kEmptyName:
    case InferenceErrorKind::kDuplicateInput:
    case InferenceErrorKind::kUnexpectedInput:
    case InferenceErrorKind::kMissingInput:
    case InferenceErrorKind::kShapeMismatch:
    case InferenceErrorKind::kElementCountMismatch:
        return kExitInput;
    case InferenceErrorKind::kExecution:
        return kExitExecution;
    }
    return kExitExecution;
}

const TensorMetadata* find_model_input(const std::vector<TensorMetadata>& inputs,
                                       const std::string& name) {
    for (const TensorMetadata& input : inputs) {
        if (input.name == name) {
            return &input;
        }
    }
    return nullptr;
}

std::string unsupported_shape_reason(const TensorMetadata& metadata) {
    if (metadata.dimensions.size() != 1) {
        return "rank " + std::to_string(metadata.dimensions.size());
    }
    return "dynamic dimension";
}

bool supported_input_shape(const TensorMetadata& metadata) {
    return metadata.dimensions.size() == 1 && metadata.dimensions[0] >= 0;
}

std::string format_float(float value, bool& formatted) {
    char buffer[64];
    const std::to_chars_result result =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general);
    if (result.ec != std::errc{}) {
        formatted = false;
        return {};
    }
    return std::string{buffer, result.ptr};
}

std::string format_success(const std::string& model_path, const std::vector<NamedTensor>& outputs,
                           bool& formatted) {
    std::string text;
    text += "Model: ";
    text += model_path;
    text += "\nBackend: ONNX Runtime CPU\n\n";
    for (const NamedTensor& output : outputs) {
        text += "Output ";
        text += output.name;
        text += ":\n[";
        for (std::size_t index = 0; index < output.values.size(); ++index) {
            if (index != 0) {
                text += ", ";
            }
            text += format_float(output.values[index], formatted);
            if (!formatted) {
                return {};
            }
        }
        text += "]\n";
    }
    return text;
}

} // namespace

CliResult execute_inference(const CliResult& parsed) {
    const EngineLoad loaded = load_inference_engine(parsed.model_path);
    if (loaded.error.kind != InferenceErrorKind::kNone || loaded.engine == nullptr) {
        const InferenceErrorKind kind = loaded.error.kind == InferenceErrorKind::kNone
                                            ? InferenceErrorKind::kExecution
                                            : loaded.error.kind;
        const std::string message =
            loaded.error.message.empty() ? "model failed to load" : loaded.error.message;
        return fail(exit_code_for(kind), message);
    }

    InferenceEngine& engine = *loaded.engine;
    for (const TensorMetadata& metadata : engine.inputs()) {
        if (!supported_input_shape(metadata)) {
            return fail(kExitInput, "unsupported input shape for " + metadata.name + ": " +
                                        unsupported_shape_reason(metadata));
        }
    }

    std::vector<NamedTensorView> views;
    views.reserve(parsed.inputs.size());
    for (const NamedFloatInput& input : parsed.inputs) {
        NamedTensorView view;
        view.name = input.name;
        view.values = std::span<const float>{input.values.data(), input.values.size()};
        const TensorMetadata* metadata = find_model_input(engine.inputs(), input.name);
        if (metadata != nullptr) {
            view.shape = metadata->dimensions;
        }
        views.push_back(std::move(view));
    }

    const InferenceResult result = engine.run(views);
    if (result.error.kind != InferenceErrorKind::kNone) {
        return fail(exit_code_for(result.error.kind), result.error.message);
    }

    bool formatted = true;
    std::string text = format_success(parsed.model_path, result.outputs, formatted);
    if (!formatted) {
        return fail(kExitExecution, "failed to format an output value");
    }

    CliResult success;
    success.action = CliAction::kRun;
    success.exit_code = kExitSuccess;
    success.message = std::move(text);
    return success;
}

} // namespace inferedge
