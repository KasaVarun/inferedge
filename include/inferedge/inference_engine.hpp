#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace inferedge {

enum class ElementType {
    kFloat32,
};

// dimensions uses a negative value for a dynamic ONNX dimension.
struct TensorMetadata {
    std::string name;
    ElementType element_type{ElementType::kFloat32};
    std::vector<std::int64_t> dimensions;
};

// name and shape are owned. values borrows caller memory for the duration of
// run(). That call copies the floats: ONNX Runtime 1.31 CreateTensor requires
// a mutable pointer, and this view does not cast away const.
struct NamedTensorView {
    std::string name;
    std::vector<std::int64_t> shape;
    std::span<const float> values;
};

// Owns its floats. The tensor remains valid after the engine is destroyed.
struct NamedTensor {
    std::string name;
    std::vector<std::int64_t> shape;
    std::vector<float> values;
};

enum class InferenceErrorKind {
    kNone,
    kModelNotFound,
    kModelInvalid,
    kUnsupportedType,
    kEmptyName,
    kDuplicateInput,
    kUnexpectedInput,
    kMissingInput,
    kShapeMismatch,
    kElementCountMismatch,
    kExecution,
};

struct InferenceError {
    InferenceErrorKind kind{InferenceErrorKind::kNone};
    std::string message;
};

struct InferenceResult {
    InferenceError error;
    std::vector<NamedTensor> outputs;
};

struct EngineLoad;
class InferenceEngine;

// One loaded ONNX model and one CPU session. Move-only. A moved-from engine
// has no model: metadata accessors return empty vectors, and run() returns
// kExecution. Concurrent run() calls are not supported.
class InferenceEngine {
public:
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) noexcept;
    InferenceEngine& operator=(InferenceEngine&&) noexcept;
    ~InferenceEngine();

    const std::vector<TensorMetadata>& inputs() const;
    const std::vector<TensorMetadata>& outputs() const;

    InferenceResult run(std::span<const NamedTensorView> inputs);

private:
    struct Impl;

    friend EngineLoad load_inference_engine(const std::filesystem::path& model_path);

    explicit InferenceEngine(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

struct EngineLoad {
    InferenceError error;
    std::unique_ptr<InferenceEngine> engine;
};

EngineLoad load_inference_engine(const std::filesystem::path& model_path);

} // namespace inferedge
