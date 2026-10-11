#include "inferedge/inference_engine.hpp"

#include <cpu_provider_factory.h>
#include <limits>
#include <onnxruntime_cxx_api.h>
#include <utility>

namespace inferedge {
namespace {

InferenceError Fail(InferenceErrorKind kind, std::string message) {
    return InferenceError{kind, std::move(message)};
}

const NamedTensorView* FindView(std::span<const NamedTensorView> inputs, const std::string& name) {
    for (const NamedTensorView& input : inputs) {
        if (input.name == name) {
            return &input;
        }
    }
    return nullptr;
}

const TensorMetadata* FindModelInput(const std::vector<TensorMetadata>& inputs,
                                     const std::string& name) {
    for (const TensorMetadata& input : inputs) {
        if (input.name == name) {
            return &input;
        }
    }
    return nullptr;
}

InferenceError ValidateShape(const NamedTensorView& input, const TensorMetadata& model) {
    for (std::size_t index = 0; index < input.shape.size(); ++index) {
        if (input.shape[index] < 0) {
            return Fail(InferenceErrorKind::kShapeMismatch, "input " + input.name + " dimension " +
                                                                std::to_string(index) +
                                                                " is negative");
        }
    }

    // Checked before rank so a wrong-rank shape cannot signed-overflow.
    std::int64_t count = 1;
    for (const std::int64_t dimension : input.shape) {
        if (dimension > 0 && count > std::numeric_limits<std::int64_t>::max() / dimension) {
            return Fail(InferenceErrorKind::kElementCountMismatch,
                        "input " + input.name + " shape product overflows");
        }
        count *= dimension;
    }
    if (static_cast<std::uint64_t>(count) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        static_cast<std::size_t>(count) != input.values.size()) {
        return Fail(InferenceErrorKind::kElementCountMismatch,
                    "input " + input.name + " has " + std::to_string(input.values.size()) +
                        " values, shape requires " + std::to_string(count));
    }

    if (input.shape.size() != model.dimensions.size()) {
        return Fail(InferenceErrorKind::kShapeMismatch,
                    "input " + input.name + " rank is " + std::to_string(input.shape.size()) +
                        ", model rank is " + std::to_string(model.dimensions.size()));
    }
    for (std::size_t index = 0; index < model.dimensions.size(); ++index) {
        const std::int64_t expected = model.dimensions[index];
        if (expected >= 0 && input.shape[index] != expected) {
            return Fail(InferenceErrorKind::kShapeMismatch,
                        "input " + input.name + " dimension " + std::to_string(index) + " is " +
                            std::to_string(input.shape[index]) + ", model dimension is " +
                            std::to_string(expected));
        }
    }
    return {};
}

InferenceError ValidateInputs(const std::vector<TensorMetadata>& model_inputs,
                              std::span<const NamedTensorView> inputs) {
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        if (inputs[index].name.empty()) {
            return Fail(InferenceErrorKind::kEmptyName, "input name is empty");
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (inputs[earlier].name == inputs[index].name) {
                return Fail(InferenceErrorKind::kDuplicateInput,
                            "duplicate input name: " + inputs[index].name);
            }
        }
    }

    for (const NamedTensorView& input : inputs) {
        if (FindModelInput(model_inputs, input.name) == nullptr) {
            return Fail(InferenceErrorKind::kUnexpectedInput,
                        "unexpected input name: " + input.name);
        }
    }
    for (const TensorMetadata& model_input : model_inputs) {
        if (FindView(inputs, model_input.name) == nullptr) {
            return Fail(InferenceErrorKind::kMissingInput,
                        "missing input name: " + model_input.name);
        }
    }
    for (const NamedTensorView& input : inputs) {
        const TensorMetadata* model_input = FindModelInput(model_inputs, input.name);
        const InferenceError shape = ValidateShape(input, *model_input);
        if (shape.kind != InferenceErrorKind::kNone) {
            return shape;
        }
    }
    return {};
}

InferenceError AppendMetadata(const Ort::TypeInfo& type_info, const char* name,
                              std::vector<TensorMetadata>& metadata, const char* role) {
    const std::string tensor_name = name == nullptr ? std::string{} : std::string{name};
    if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
        return Fail(InferenceErrorKind::kUnsupportedType,
                    "model " + std::string{role} + " " + tensor_name + " is not a tensor");
    }
    const Ort::ConstTensorTypeAndShapeInfo tensor = type_info.GetTensorTypeAndShapeInfo();
    if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return Fail(InferenceErrorKind::kUnsupportedType, "model " + std::string{role} + " " +
                                                              tensor_name +
                                                              " element type is not float32");
    }
    metadata.push_back(TensorMetadata{tensor_name, ElementType::kFloat32, tensor.GetShape()});
    return {};
}

} // namespace

struct InferenceEngine::Impl {
    Ort::Env environment;
    Ort::MemoryInfo memory;
    Ort::Session session;
    std::vector<TensorMetadata> inputs;
    std::vector<TensorMetadata> outputs;
    std::vector<std::string> output_names;

    Impl(Ort::Env env, Ort::MemoryInfo mem, Ort::Session sess)
        : environment{std::move(env)}, memory{std::move(mem)}, session{std::move(sess)} {}

    InferenceError ReadMetadata();
    InferenceResult Execute(std::span<const NamedTensorView> supplied);
};

namespace {

const std::vector<TensorMetadata>& EmptyMetadata() {
    static const std::vector<TensorMetadata> empty;
    return empty;
}

} // namespace

InferenceError InferenceEngine::Impl::ReadMetadata() {
    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t input_count = session.GetInputCount();
    inputs.reserve(input_count);
    for (std::size_t index = 0; index < input_count; ++index) {
        const Ort::AllocatedStringPtr name = session.GetInputNameAllocated(index, allocator);
        const InferenceError error =
            AppendMetadata(session.GetInputTypeInfo(index), name.get(), inputs, "input");
        if (error.kind != InferenceErrorKind::kNone) {
            return error;
        }
    }

    const std::size_t output_count = session.GetOutputCount();
    outputs.reserve(output_count);
    output_names.reserve(output_count);
    for (std::size_t index = 0; index < output_count; ++index) {
        const Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(index, allocator);
        const InferenceError error =
            AppendMetadata(session.GetOutputTypeInfo(index), name.get(), outputs, "output");
        if (error.kind != InferenceErrorKind::kNone) {
            return error;
        }
        output_names.push_back(outputs.back().name);
    }
    return {};
}

InferenceResult InferenceEngine::Impl::Execute(std::span<const NamedTensorView> supplied) {
    // CreateTensor borrows mutable memory and does not free it. Each input is
    // copied so the const span is never cast to float*. The copies outlive Run.
    std::vector<std::vector<float>> storage;
    std::vector<Ort::Value> values;
    std::vector<const char*> names;
    storage.reserve(inputs.size());
    values.reserve(inputs.size());
    names.reserve(inputs.size());

    for (const TensorMetadata& model_input : inputs) {
        const NamedTensorView* view = FindView(supplied, model_input.name);
        if (view == nullptr) {
            return InferenceResult{
                Fail(InferenceErrorKind::kExecution, "input name was not validated"), {}};
        }
        storage.emplace_back(view->values.begin(), view->values.end());
        std::vector<float>& buffer = storage.back();
        values.push_back(Ort::Value::CreateTensor<float>(memory, buffer.data(), buffer.size(),
                                                         view->shape.data(), view->shape.size()));
        names.push_back(model_input.name.c_str());
    }

    std::vector<const char*> output_name_pointers;
    output_name_pointers.reserve(output_names.size());
    for (const std::string& name : output_names) {
        output_name_pointers.push_back(name.c_str());
    }

    Ort::RunOptions run_options;
    const std::vector<Ort::Value> ort_outputs =
        session.Run(run_options, names.data(), values.data(), values.size(),
                    output_name_pointers.data(), output_name_pointers.size());
    if (ort_outputs.size() != outputs.size()) {
        return InferenceResult{Fail(InferenceErrorKind::kExecution,
                                    "onnx runtime returned an unexpected output count"),
                               {}};
    }

    InferenceResult result;
    result.outputs.reserve(ort_outputs.size());
    for (std::size_t index = 0; index < ort_outputs.size(); ++index) {
        if (!ort_outputs[index].IsTensor()) {
            return InferenceResult{
                Fail(InferenceErrorKind::kExecution, "onnx runtime output is not a tensor"), {}};
        }
        const Ort::TensorTypeAndShapeInfo info = ort_outputs[index].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return InferenceResult{Fail(InferenceErrorKind::kExecution,
                                        "onnx runtime output element type is not float32"),
                                   {}};
        }
        const std::size_t count = info.GetElementCount();
        std::vector<float> owned(count);
        if (count > 0) {
            const float* data = ort_outputs[index].GetTensorData<float>();
            if (data == nullptr) {
                return InferenceResult{
                    Fail(InferenceErrorKind::kExecution, "onnx runtime output data is missing"),
                    {}};
            }
            owned.assign(data, data + count);
        }
        result.outputs.push_back(
            NamedTensor{output_names[index], info.GetShape(), std::move(owned)});
    }
    return result;
}

InferenceEngine::InferenceEngine(std::unique_ptr<Impl> impl) : impl_{std::move(impl)} {
}

InferenceEngine::InferenceEngine(InferenceEngine&&) noexcept = default;

InferenceEngine& InferenceEngine::operator=(InferenceEngine&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

InferenceEngine::~InferenceEngine() = default;

const std::vector<TensorMetadata>& InferenceEngine::inputs() const {
    if (!impl_) {
        return EmptyMetadata();
    }
    return impl_->inputs;
}

const std::vector<TensorMetadata>& InferenceEngine::outputs() const {
    if (!impl_) {
        return EmptyMetadata();
    }
    return impl_->outputs;
}

InferenceResult InferenceEngine::run(std::span<const NamedTensorView> inputs) {
    if (!impl_) {
        return InferenceResult{Fail(InferenceErrorKind::kExecution, "inference engine is empty"),
                               {}};
    }
    const InferenceError validation = ValidateInputs(impl_->inputs, inputs);
    if (validation.kind != InferenceErrorKind::kNone) {
        return InferenceResult{validation, {}};
    }
    try {
        return impl_->Execute(inputs);
    } catch (const Ort::Exception& error) {
        return InferenceResult{Fail(InferenceErrorKind::kExecution,
                                    std::string{"onnx runtime execution failed: "} + error.what()),
                               {}};
    }
}

EngineLoad load_inference_engine(const std::filesystem::path& model_path) {
    std::error_code status;
    if (!std::filesystem::is_regular_file(model_path, status) || status) {
        return EngineLoad{Fail(InferenceErrorKind::kModelNotFound,
                               "model file is missing: " + model_path.string()),
                          nullptr};
    }

    try {
        Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "inferedge"};
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
        options.AppendExecutionProvider_CPU(1);
        const std::string path = model_path.string();
        Ort::Session session{environment, path.c_str(), options};
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto impl = std::unique_ptr<InferenceEngine::Impl>(new InferenceEngine::Impl{
            std::move(environment), std::move(memory), std::move(session)});
        const InferenceError metadata = impl->ReadMetadata();
        if (metadata.kind != InferenceErrorKind::kNone) {
            return EngineLoad{metadata, nullptr};
        }
        return EngineLoad{InferenceError{},
                          std::unique_ptr<InferenceEngine>(new InferenceEngine{std::move(impl)})};
    } catch (const Ort::Exception& error) {
        return EngineLoad{Fail(InferenceErrorKind::kModelInvalid,
                               "model file is not a valid ONNX model: " + model_path.string() +
                                   ": " + error.what()),
                          nullptr};
    }
}

} // namespace inferedge
