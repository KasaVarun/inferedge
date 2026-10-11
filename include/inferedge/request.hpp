#pragma once

#include <string>
#include <vector>

namespace inferedge {

enum class RequestPriority {
    kInteractive,
    kBackground,
};

// A value type. Copy and move are the compiler-generated operations from
// std::string and std::vector. An omitted priority stays in the background.
struct InferenceRequest {
    std::string request_id;
    std::string model_id;
    std::vector<float> input_values;
    RequestPriority priority{RequestPriority::kBackground};
};

enum class RequestValidationError {
    kNone,
    kEmptyRequestId,
    kEmptyModelId,
    kEmptyInput,
};

struct RequestValidation {
    RequestValidationError error{RequestValidationError::kNone};
    std::string message;
};

// Returns the first structural problem, in order: request id, model id, input.
// A valid request returns kNone and an empty message.
// This check does not verify model-specific tensor shapes, NaN or infinity, or
// a priority created by casting an arbitrary integer to RequestPriority.
RequestValidation validate_request(const InferenceRequest& request);

} // namespace inferedge
