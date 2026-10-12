#pragma once

#include "inferedge/inference_engine.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace inferedge {

// Validation failures from batch assembly and splitting. These are request
// and metadata errors, not InferenceEngine execution errors. A count that
// fits in std::size_t is not necessarily allocatable: std::vector throws
// std::length_error when it exceeds max_size(), and std::bad_alloc when the
// allocator cannot obtain memory. Those exceptions are not translated.
enum class TensorBatchErrorKind {
    kNone,
    kEmptyBatch,
    kEmptyName,
    kDuplicateName,
    kMissingTensor,
    kUnexpectedTensor,
    kUnsupportedType,
    kUnsupportedMetadata,
    kRankMismatch,
    kBatchAxisMismatch,
    kShapeMismatch,
    kElementCountMismatch,
    kOverflow,
};

struct TensorBatchError {
    TensorBatchErrorKind kind{TensorBatchErrorKind::kNone};
    std::string message;
};

// On failure, inputs is empty. On success, inputs owns one tensor per model
// input, in model-metadata order, with axis 0 equal to the request count.
struct TensorBatchAssembly {
    TensorBatchError error;
    std::vector<NamedTensor> inputs;
};

// On failure, requests is empty. On success, requests[i] owns that request's
// outputs in model-metadata order. Each output keeps a leading batch axis of 1.
struct TensorBatchSplit {
    TensorBatchError error;
    std::vector<std::vector<NamedTensor>> requests;
};

// Copies single-request tensors into owned batched inputs.
// model_inputs is the canonical name order and shape signature. Each inner
// span borrows one request's tensors for this call only. Every request tensor
// must use batch axis 1. An empty request list is rejected and does not build
// a zero-length batch. Caller-owned tensors are not mutated. Returned tensors
// do not borrow the inputs. std::length_error and std::bad_alloc propagate.
TensorBatchAssembly AssembleBatchInputs(std::span<const TensorMetadata> model_inputs,
                                        std::span<const std::span<const NamedTensor>> requests);

// Copies each request row out of owned batched outputs.
// request_count must be positive. Batched axis 0 must equal that count, and
// the remaining axes must match model_outputs. The returned rows own their
// floats. A validation failure returns no per-request collections.
// std::length_error and std::bad_alloc propagate. This function does not run
// a model.
TensorBatchSplit SplitBatchOutputs(std::span<const TensorMetadata> model_outputs,
                                   std::span<const NamedTensor> batched_outputs,
                                   std::size_t request_count);

} // namespace inferedge
