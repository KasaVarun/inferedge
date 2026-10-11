#include "inferedge/float_buffer.hpp"
#include "inferedge/request.hpp"

#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// count_items and Slot are compiled separately for each type argument.
// HasRequestId is checked when the template is instantiated. id_of() is a
// candidate only when that check succeeds. A virtual function would choose an
// override at run time through a base. These tests do not add a base.

namespace {

template <typename T> std::size_t count_items(const std::vector<T>& items) {
    return items.size();
}

template <typename T> class Slot {
public:
    Slot(std::size_t index, T value) : index_{index}, value_{std::move(value)} {}

    std::size_t index() const noexcept { return index_; }

    const T& value() const noexcept { return value_; }

    T release() { return std::move(value_); }

private:
    std::size_t index_;
    T value_;
};

template <typename T>
concept HasRequestId = requires(const T& item) {
    { item.request_id } -> std::convertible_to<std::string_view>;
};

template <HasRequestId T> std::string_view id_of(const T& item) {
    return item.request_id;
}

inferedge::InferenceRequest make_request() {
    return inferedge::InferenceRequest{
        .request_id = "req-a",
        .model_id = "model",
        .input_values = {1.0f, 2.0f},
        .priority = inferedge::RequestPriority::kInteractive,
    };
}

} // namespace

TEST(TemplateTest, FunctionTemplateInstantiatesPerType) {
    const std::vector<inferedge::InferenceRequest> requests{make_request(), make_request()};
    const std::vector<float> values{1.0f, 2.0f, 3.0f};

    EXPECT_EQ(count_items(requests), 2u);
    EXPECT_EQ(count_items(values), 3u);
}

TEST(TemplateTest, SlotStoresAnIndependentRequest) {
    inferedge::InferenceRequest original = make_request();
    Slot<inferedge::InferenceRequest> slot{4, original};
    original.input_values[0] = 9.0f;

    EXPECT_EQ(slot.index(), 4u);
    EXPECT_EQ(slot.value().request_id, "req-a");
    EXPECT_FLOAT_EQ(slot.value().input_values[0], 1.0f);
    EXPECT_FLOAT_EQ(slot.value().input_values[1], 2.0f);

    Slot<inferedge::InferenceRequest> copy = slot;
    const inferedge::InferenceRequest released = copy.release();
    EXPECT_EQ(released.request_id, "req-a");
    EXPECT_FLOAT_EQ(released.input_values[0], 1.0f);
    EXPECT_FLOAT_EQ(slot.value().input_values[0], 1.0f);
}

TEST(TemplateTest, SlotMovesOutAFloatBuffer) {
    Slot<inferedge::FloatBuffer> slot{8, inferedge::FloatBuffer{std::vector<float>{1.0f, -2.5f}}};

    EXPECT_EQ(slot.index(), 8u);
    ASSERT_EQ(slot.value().size(), 2u);

    const inferedge::FloatBuffer owned = slot.release();
    ASSERT_EQ(owned.view().size(), 2u);
    EXPECT_FLOAT_EQ(owned.view()[0], 1.0f);
    EXPECT_FLOAT_EQ(owned.view()[1], -2.5f);
}

TEST(TemplateTest, SlotTraitsFollowTheStoredType) {
    static_assert(std::is_copy_constructible_v<Slot<inferedge::InferenceRequest>>);
    static_assert(std::is_copy_assignable_v<Slot<inferedge::InferenceRequest>>);
    static_assert(std::is_nothrow_move_constructible_v<Slot<inferedge::InferenceRequest>>);
    static_assert(std::is_nothrow_move_assignable_v<Slot<inferedge::InferenceRequest>>);

    static_assert(!std::is_copy_constructible_v<Slot<inferedge::FloatBuffer>>);
    static_assert(!std::is_copy_assignable_v<Slot<inferedge::FloatBuffer>>);
    static_assert(std::is_nothrow_move_constructible_v<Slot<inferedge::FloatBuffer>>);
    static_assert(std::is_nothrow_move_assignable_v<Slot<inferedge::FloatBuffer>>);
}

TEST(TemplateTest, ConceptAcceptsAnInferenceRequest) {
    static_assert(HasRequestId<inferedge::InferenceRequest>);

    const inferedge::InferenceRequest request = make_request();
    EXPECT_EQ(id_of(request), std::string_view{"req-a"});
}

TEST(TemplateTest, ConceptRejectsTypesWithoutRequestId) {
    static_assert(!HasRequestId<inferedge::FloatBuffer>);
    static_assert(!HasRequestId<int>);
}
