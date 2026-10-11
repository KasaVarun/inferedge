#include "inferedge/float_buffer.hpp"
#include "inferedge/request.hpp"
#include "inferedge/request_lifecycle.hpp"

#include <gtest/gtest.h>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

// Copy duplicates values into independent storage. Move transfers the vector
// buffer. With std::allocator<float>, that move is noexcept and constant
// time, so the destination data() pointer is the source pointer from before
// the move. A moved-from object stays valid. These tests do not read its
// elements or require it to be empty.
//
// Copy assignment may reuse the destination vector's existing buffer. It
// does not have to allocate. Independence is shown by a distinct source
// pointer and by mutating the copy.
//
// A span borrows the owner's pointer and length. It allocates nothing and
// is used only while that owner is still alive.

constexpr float kFirst = 1.0f;
constexpr float kSecond = -2.5f;
constexpr float kThird = 0.0f;

inferedge::InferenceRequest sample_request() {
    return inferedge::InferenceRequest{
        .request_id = "request-id-long-enough-to-exercise-string-storage",
        .model_id = "model-id-long-enough-to-exercise-string-storage",
        .input_values = {kFirst, kSecond, kThird},
        .priority = inferedge::RequestPriority::kInteractive,
    };
}

void expect_same_payload(const inferedge::InferenceRequest& actual,
                         const inferedge::InferenceRequest& expected) {
    EXPECT_EQ(actual.request_id, expected.request_id);
    EXPECT_EQ(actual.model_id, expected.model_id);
    EXPECT_EQ(actual.priority, expected.priority);
    EXPECT_EQ(actual.input_values, expected.input_values);
}

void mutate_payload(inferedge::InferenceRequest& request) {
    request.request_id.push_back('X');
    request.model_id.push_back('Y');
    request.input_values[0] = 42.0f;
}

// Reads the caller's request through a const reference. The returned pointer
// is the caller's vector storage, not storage belonging to a copy.
struct BorrowedRequest {
    std::string request_id;
    std::size_t value_count = 0;
    const float* data = nullptr;
};

BorrowedRequest inspect_request(const inferedge::InferenceRequest& request) {
    return BorrowedRequest{
        request.request_id,
        request.input_values.size(),
        request.input_values.data(),
    };
}

inferedge::FloatBuffer sample_buffer() {
    return inferedge::FloatBuffer{std::vector<float>{kFirst, kSecond, kThird}};
}

void expect_sample_buffer(const inferedge::FloatBuffer& buffer) {
    ASSERT_EQ(buffer.size(), 3u);
    const inferedge::BufferRead first = buffer.try_at(0);
    const inferedge::BufferRead second = buffer.try_at(1);
    const inferedge::BufferRead third = buffer.try_at(2);
    EXPECT_EQ(first.error, inferedge::BufferReadError::kNone);
    EXPECT_EQ(second.error, inferedge::BufferReadError::kNone);
    EXPECT_EQ(third.error, inferedge::BufferReadError::kNone);
    EXPECT_FLOAT_EQ(first.value, kFirst);
    EXPECT_FLOAT_EQ(second.value, kSecond);
    EXPECT_FLOAT_EQ(third.value, kThird);
}

void expect_valid_moved_from_request(const inferedge::InferenceRequest& source) {
    const std::size_t value_count = source.input_values.size();
    const std::size_t id_size = source.request_id.size();
    const std::size_t model_size = source.model_id.size();
    EXPECT_EQ(source.input_values.size(), value_count);
    EXPECT_EQ(source.request_id.size(), id_size);
    EXPECT_EQ(source.model_id.size(), model_size);
}

void expect_valid_moved_from_buffer(const inferedge::FloatBuffer& source) {
    const std::size_t value_count = source.size();
    EXPECT_EQ(source.view().size(), value_count);
    EXPECT_EQ(source.empty(), value_count == 0);
}

} // namespace

TEST(OwnershipTest, CopyConstructionDuplicatesRequestStorage) {
    const inferedge::InferenceRequest source = sample_request();
    ASSERT_FALSE(source.input_values.empty());
    const float* const source_data = source.input_values.data();
    const std::string source_id = source.request_id;
    const std::string source_model = source.model_id;

    inferedge::InferenceRequest copy = source;

    expect_same_payload(copy, source);
    EXPECT_NE(copy.input_values.data(), source_data);
    EXPECT_EQ(source.input_values.data(), source_data);

    mutate_payload(copy);
    EXPECT_EQ(source.request_id, source_id);
    EXPECT_EQ(source.model_id, source_model);
    EXPECT_FLOAT_EQ(source.input_values[0], kFirst);
}

TEST(OwnershipTest, CopyAssignmentDuplicatesRequestValues) {
    const inferedge::InferenceRequest source = sample_request();
    ASSERT_FALSE(source.input_values.empty());
    const float* const source_data = source.input_values.data();
    const std::string source_id = source.request_id;
    const std::string source_model = source.model_id;

    inferedge::InferenceRequest assigned{
        .request_id = "replacement-request-id-with-different-text",
        .model_id = "replacement-model-id-with-different-text",
        .input_values = {9.0f, 9.0f, 9.0f, 9.0f},
        .priority = inferedge::RequestPriority::kBackground,
    };
    assigned = source;

    expect_same_payload(assigned, source);
    EXPECT_NE(assigned.input_values.data(), source_data);
    EXPECT_EQ(source.input_values.data(), source_data);

    mutate_payload(assigned);
    EXPECT_EQ(source.request_id, source_id);
    EXPECT_EQ(source.model_id, source_model);
    EXPECT_EQ(source.priority, inferedge::RequestPriority::kInteractive);
    EXPECT_FLOAT_EQ(source.input_values[0], kFirst);
}

TEST(OwnershipTest, MoveConstructionTransfersRequestStorage) {
    inferedge::InferenceRequest source = sample_request();
    ASSERT_FALSE(source.input_values.empty());
    const float* const source_data = source.input_values.data();
    const std::string expected_id = source.request_id;
    const std::string expected_model = source.model_id;
    const std::vector<float> expected_values = source.input_values;
    const inferedge::RequestPriority expected_priority = source.priority;

    const inferedge::InferenceRequest destination = std::move(source);

    EXPECT_EQ(destination.request_id, expected_id);
    EXPECT_EQ(destination.model_id, expected_model);
    EXPECT_EQ(destination.priority, expected_priority);
    EXPECT_EQ(destination.input_values, expected_values);
    EXPECT_EQ(destination.input_values.data(), source_data);
    expect_valid_moved_from_request(source);
}

TEST(OwnershipTest, MoveAssignmentTransfersRequestStorage) {
    inferedge::InferenceRequest source = sample_request();
    ASSERT_FALSE(source.input_values.empty());
    const float* const source_data = source.input_values.data();
    const std::string expected_id = source.request_id;
    const std::string expected_model = source.model_id;
    const std::vector<float> expected_values = source.input_values;
    const inferedge::RequestPriority expected_priority = source.priority;

    inferedge::InferenceRequest destination{
        .request_id = "destination-request-id-before-move-assignment",
        .model_id = "destination-model-id-before-move-assignment",
        .input_values = {8.0f, 8.0f},
        .priority = inferedge::RequestPriority::kBackground,
    };
    destination = std::move(source);

    EXPECT_EQ(destination.request_id, expected_id);
    EXPECT_EQ(destination.model_id, expected_model);
    EXPECT_EQ(destination.priority, expected_priority);
    EXPECT_EQ(destination.input_values, expected_values);
    EXPECT_EQ(destination.input_values.data(), source_data);
    expect_valid_moved_from_request(source);
}

TEST(OwnershipTest, ConstReferenceObservesTheCallerStorage) {
    const inferedge::InferenceRequest request = sample_request();
    const float* const source_data = request.input_values.data();
    const std::string source_id = request.request_id;
    const std::vector<float> source_values = request.input_values;

    const BorrowedRequest observed = inspect_request(request);

    EXPECT_EQ(observed.request_id, source_id);
    EXPECT_EQ(observed.value_count, source_values.size());
    EXPECT_EQ(observed.data, source_data);
    EXPECT_EQ(request.input_values.data(), source_data);
    EXPECT_EQ(request.request_id, source_id);
    EXPECT_EQ(request.input_values, source_values);
    ASSERT_NE(observed.data, nullptr);
    EXPECT_FLOAT_EQ(observed.data[0], kFirst);
}

TEST(OwnershipTest, MoveConstructionTransfersBufferStorage) {
    inferedge::FloatBuffer source = sample_buffer();
    const float* const source_data = source.view().data();

    inferedge::FloatBuffer destination{std::move(source)};

    expect_sample_buffer(destination);
    EXPECT_EQ(destination.view().data(), source_data);
    expect_valid_moved_from_buffer(source);
}

TEST(OwnershipTest, MoveAssignmentTransfersBufferStorage) {
    inferedge::FloatBuffer source = sample_buffer();
    const float* const source_data = source.view().data();
    inferedge::FloatBuffer destination{std::vector<float>{9.0f}};

    destination = std::move(source);

    expect_sample_buffer(destination);
    EXPECT_EQ(destination.view().data(), source_data);
    expect_valid_moved_from_buffer(source);
}

TEST(OwnershipTest, CloneCopiesBufferValuesIntoIndependentStorage) {
    const inferedge::FloatBuffer source = sample_buffer();
    const float* const source_data = source.view().data();

    const inferedge::FloatBuffer copy = source.clone();

    expect_sample_buffer(source);
    expect_sample_buffer(copy);
    EXPECT_EQ(source.view().data(), source_data);
    EXPECT_NE(copy.view().data(), source.view().data());
}

TEST(OwnershipTest, SpanBorrowsRequestStorage) {
    const inferedge::InferenceRequest request = sample_request();
    const std::span<const float> borrowed{request.input_values.data(), request.input_values.size()};

    EXPECT_EQ(borrowed.data(), request.input_values.data());
    EXPECT_EQ(borrowed.size(), request.input_values.size());
    ASSERT_FALSE(borrowed.empty());
    EXPECT_FLOAT_EQ(borrowed[0], kFirst);
}

TEST(OwnershipTest, SpanBorrowsBufferStorage) {
    const inferedge::FloatBuffer buffer = sample_buffer();
    const std::span<const float> first = buffer.view();
    const std::span<const float> second = buffer.view();

    EXPECT_EQ(first.data(), buffer.view().data());
    EXPECT_EQ(first.size(), buffer.size());
    EXPECT_EQ(second.data(), first.data());
    EXPECT_EQ(second.size(), first.size());
    ASSERT_FALSE(first.empty());
    EXPECT_FLOAT_EQ(first[0], kFirst);
}

TEST(OwnershipTest, TypeTraitsMatchTheOwnershipPolicy) {
    static_assert(std::is_copy_constructible_v<inferedge::InferenceRequest>);
    static_assert(std::is_copy_assignable_v<inferedge::InferenceRequest>);
    static_assert(!std::is_nothrow_copy_constructible_v<inferedge::InferenceRequest>);
    static_assert(std::is_nothrow_move_constructible_v<inferedge::InferenceRequest>);
    static_assert(std::is_nothrow_move_assignable_v<inferedge::InferenceRequest>);

    static_assert(!std::is_copy_constructible_v<inferedge::FloatBuffer>);
    static_assert(!std::is_copy_assignable_v<inferedge::FloatBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<inferedge::FloatBuffer>);
    static_assert(std::is_nothrow_move_assignable_v<inferedge::FloatBuffer>);

    static_assert(std::is_copy_constructible_v<inferedge::RequestLifecycle>);
    static_assert(std::is_copy_assignable_v<inferedge::RequestLifecycle>);
    static_assert(std::is_nothrow_move_constructible_v<inferedge::RequestLifecycle>);
    static_assert(std::is_nothrow_move_assignable_v<inferedge::RequestLifecycle>);
}
