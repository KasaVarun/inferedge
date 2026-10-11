#include "inferedge/request.hpp"

#include <gtest/gtest.h>

namespace {

inferedge::InferenceRequest valid_request() {
    return inferedge::InferenceRequest{
        .request_id = "req-1",
        .model_id = "model-a",
        .input_values = {1.0f, -2.5f, 0.0f},
        .priority = inferedge::RequestPriority::kInteractive,
    };
}

} // namespace

TEST(RequestTest, AcceptsACompleteRequest) {
    const inferedge::InferenceRequest request = valid_request();
    const inferedge::RequestValidation result = inferedge::validate_request(request);
    EXPECT_EQ(result.error, inferedge::RequestValidationError::kNone);
    EXPECT_TRUE(result.message.empty());
}

TEST(RequestTest, RejectsAnEmptyRequestId) {
    inferedge::InferenceRequest request = valid_request();
    request.request_id.clear();
    const inferedge::RequestValidation result = inferedge::validate_request(request);
    EXPECT_EQ(result.error, inferedge::RequestValidationError::kEmptyRequestId);
    EXPECT_EQ(result.message, "request_id is empty");
}

TEST(RequestTest, RejectsAnEmptyModelId) {
    inferedge::InferenceRequest request = valid_request();
    request.model_id.clear();
    const inferedge::RequestValidation result = inferedge::validate_request(request);
    EXPECT_EQ(result.error, inferedge::RequestValidationError::kEmptyModelId);
    EXPECT_EQ(result.message, "model_id is empty");
}

TEST(RequestTest, RejectsAnEmptyInput) {
    inferedge::InferenceRequest request = valid_request();
    request.input_values.clear();
    const inferedge::RequestValidation result = inferedge::validate_request(request);
    EXPECT_EQ(result.error, inferedge::RequestValidationError::kEmptyInput);
    EXPECT_EQ(result.message, "input_values is empty");
}

TEST(RequestTest, KeepsInteractivePriority) {
    inferedge::InferenceRequest request = valid_request();
    request.priority = inferedge::RequestPriority::kInteractive;
    const inferedge::RequestValidation result = inferedge::validate_request(request);
    EXPECT_EQ(request.priority, inferedge::RequestPriority::kInteractive);
    EXPECT_EQ(result.error, inferedge::RequestValidationError::kNone);
    EXPECT_TRUE(result.message.empty());
}

TEST(RequestTest, KeepsBackgroundPriority) {
    inferedge::InferenceRequest request = valid_request();
    request.priority = inferedge::RequestPriority::kBackground;
    const inferedge::RequestValidation result = inferedge::validate_request(request);
    EXPECT_EQ(request.priority, inferedge::RequestPriority::kBackground);
    EXPECT_EQ(result.error, inferedge::RequestValidationError::kNone);
    EXPECT_TRUE(result.message.empty());
}

TEST(RequestTest, DefaultsPriorityToBackground) {
    const inferedge::InferenceRequest omitted{
        .request_id = "req-1",
        .model_id = "model-a",
        .input_values = {0.5f},
    };
    const inferedge::InferenceRequest constructed;
    EXPECT_EQ(omitted.priority, inferedge::RequestPriority::kBackground);
    EXPECT_EQ(constructed.priority, inferedge::RequestPriority::kBackground);
    const inferedge::RequestValidation result = inferedge::validate_request(omitted);
    EXPECT_EQ(result.error, inferedge::RequestValidationError::kNone);
    EXPECT_TRUE(result.message.empty());
}

TEST(RequestTest, ReportsTheFirstErrorWhenSeveralFieldsAreInvalid) {
    inferedge::InferenceRequest request = valid_request();
    request.request_id.clear();
    request.model_id.clear();
    request.input_values.clear();
    const inferedge::RequestValidation all_empty = inferedge::validate_request(request);
    EXPECT_EQ(all_empty.error, inferedge::RequestValidationError::kEmptyRequestId);
    EXPECT_EQ(all_empty.message, "request_id is empty");

    request.request_id = "req-1";
    const inferedge::RequestValidation model_and_input = inferedge::validate_request(request);
    EXPECT_EQ(model_and_input.error, inferedge::RequestValidationError::kEmptyModelId);
    EXPECT_EQ(model_and_input.message, "model_id is empty");
}

TEST(RequestTest, ValidationDoesNotModifyTheRequest) {
    inferedge::InferenceRequest request = valid_request();
    request.request_id.clear();
    const inferedge::InferenceRequest before = request;
    const inferedge::RequestValidation result = inferedge::validate_request(request);
    EXPECT_EQ(result.error, inferedge::RequestValidationError::kEmptyRequestId);
    EXPECT_EQ(request.request_id, before.request_id);
    EXPECT_EQ(request.model_id, before.model_id);
    EXPECT_EQ(request.input_values, before.input_values);
    EXPECT_EQ(request.priority, before.priority);
}
