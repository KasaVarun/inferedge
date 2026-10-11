#include "inferedge/request.hpp"

namespace inferedge {

RequestValidation validate_request(const InferenceRequest& request) {
    if (request.request_id.empty()) {
        return RequestValidation{RequestValidationError::kEmptyRequestId, "request_id is empty"};
    }
    if (request.model_id.empty()) {
        return RequestValidation{RequestValidationError::kEmptyModelId, "model_id is empty"};
    }
    if (request.input_values.empty()) {
        return RequestValidation{RequestValidationError::kEmptyInput, "input_values is empty"};
    }
    return RequestValidation{RequestValidationError::kNone, {}};
}

} // namespace inferedge
