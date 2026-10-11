#include "inferedge/float_buffer.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

std::vector<float> sample_values() {
    return std::vector<float>{1.0f, -2.5f, 0.0f};
}

inferedge::FloatBuffer sample_buffer() {
    return inferedge::FloatBuffer{sample_values()};
}

void expect_sample(const inferedge::FloatBuffer& buffer) {
    ASSERT_EQ(buffer.size(), 3u);
    ASSERT_FALSE(buffer.empty());
    const inferedge::BufferRead first = buffer.try_at(0);
    const inferedge::BufferRead second = buffer.try_at(1);
    const inferedge::BufferRead third = buffer.try_at(2);
    EXPECT_EQ(first.error, inferedge::BufferReadError::kNone);
    EXPECT_EQ(second.error, inferedge::BufferReadError::kNone);
    EXPECT_EQ(third.error, inferedge::BufferReadError::kNone);
    EXPECT_FLOAT_EQ(first.value, 1.0f);
    EXPECT_FLOAT_EQ(second.value, -2.5f);
    EXPECT_FLOAT_EQ(third.value, 0.0f);
}

struct Probe {
    int& live;

    explicit Probe(int& live_count) : live(live_count) { ++live; }

    ~Probe() { --live; }
};

} // namespace

TEST(FloatBufferTest, ConstructsFromAMovedVector) {
    std::vector<float> values = sample_values();
    const float* const source_data = values.data();
    inferedge::FloatBuffer buffer{std::move(values)};
    expect_sample(buffer);
    EXPECT_EQ(buffer.view().data(), source_data);
}

TEST(FloatBufferTest, ReportsItsSize) {
    const inferedge::FloatBuffer buffer = sample_buffer();
    EXPECT_EQ(buffer.size(), 3u);
}

TEST(FloatBufferTest, SupportsAnEmptyBuffer) {
    inferedge::FloatBuffer buffer{std::vector<float>{}};
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0u);
    EXPECT_TRUE(buffer.view().empty());
    const inferedge::BufferRead read = buffer.try_at(0);
    EXPECT_EQ(read.error, inferedge::BufferReadError::kOutOfRange);
    EXPECT_FLOAT_EQ(read.value, 0.0f);
}

TEST(FloatBufferTest, StoresTheGivenValues) {
    const inferedge::FloatBuffer buffer = sample_buffer();
    expect_sample(buffer);
}

TEST(FloatBufferTest, ViewBorrowsTheStoredValues) {
    const inferedge::FloatBuffer buffer = sample_buffer();
    const std::span<const float> borrowed = buffer.view();
    ASSERT_EQ(borrowed.size(), buffer.size());
    EXPECT_FLOAT_EQ(borrowed[0], 1.0f);
    EXPECT_FLOAT_EQ(borrowed[1], -2.5f);
    EXPECT_FLOAT_EQ(borrowed[2], 0.0f);
}

TEST(FloatBufferTest, RepeatedViewsShareStorage) {
    const inferedge::FloatBuffer buffer = sample_buffer();
    const std::span<const float> first = buffer.view();
    const std::span<const float> second = buffer.view();
    EXPECT_EQ(first.data(), second.data());
    EXPECT_EQ(first.size(), second.size());
}

TEST(FloatBufferTest, ReadsAnInRangeElement) {
    const inferedge::FloatBuffer buffer = sample_buffer();
    const inferedge::BufferRead read = buffer.try_at(1);
    EXPECT_EQ(read.error, inferedge::BufferReadError::kNone);
    EXPECT_FLOAT_EQ(read.value, -2.5f);
}

TEST(FloatBufferTest, RejectsAnOutOfRangeIndex) {
    const inferedge::FloatBuffer buffer = sample_buffer();
    const inferedge::BufferRead at_size = buffer.try_at(buffer.size());
    const inferedge::BufferRead past_size = buffer.try_at(99);
    EXPECT_EQ(at_size.error, inferedge::BufferReadError::kOutOfRange);
    EXPECT_EQ(past_size.error, inferedge::BufferReadError::kOutOfRange);
    EXPECT_FLOAT_EQ(at_size.value, 0.0f);
    EXPECT_FLOAT_EQ(past_size.value, 0.0f);
    expect_sample(buffer);
}

TEST(FloatBufferTest, MoveConstructionTransfersTheValues) {
    inferedge::FloatBuffer source = sample_buffer();
    inferedge::FloatBuffer destination{std::move(source)};
    expect_sample(destination);
    EXPECT_EQ(source.view().size(), source.size());
}

TEST(FloatBufferTest, MoveAssignmentTransfersTheValues) {
    inferedge::FloatBuffer source = sample_buffer();
    std::vector<float> existing{9.0f};
    inferedge::FloatBuffer destination{std::move(existing)};
    destination = std::move(source);
    expect_sample(destination);
    EXPECT_EQ(source.view().size(), source.size());
}

TEST(FloatBufferTest, CloneIsAnIndependentCopy) {
    const inferedge::FloatBuffer source = sample_buffer();
    const float* const source_data = source.view().data();
    const inferedge::FloatBuffer copy = source.clone();
    expect_sample(source);
    expect_sample(copy);
    EXPECT_EQ(source.view().data(), source_data);
    EXPECT_NE(copy.view().data(), source.view().data());
}

TEST(FloatBufferTest, IsNotCopyable) {
    static_assert(!std::is_copy_constructible_v<inferedge::FloatBuffer>);
    static_assert(!std::is_copy_assignable_v<inferedge::FloatBuffer>);
    static_assert(!std::is_constructible_v<inferedge::FloatBuffer, std::vector<float>&>);
}

TEST(FloatBufferTest, IsMovable) {
    static_assert(std::is_move_constructible_v<inferedge::FloatBuffer>);
    static_assert(std::is_move_assignable_v<inferedge::FloatBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<inferedge::FloatBuffer>);
    static_assert(std::is_nothrow_move_assignable_v<inferedge::FloatBuffer>);
    static_assert(std::is_nothrow_constructible_v<inferedge::FloatBuffer, std::vector<float>&&>);
}

TEST(FloatBufferTest, UniquePtrReleasesAtEndOfScope) {
    int live = 0;
    {
        const std::unique_ptr<Probe> owned = std::make_unique<Probe>(live);
        EXPECT_EQ(live, 1);
    }
    EXPECT_EQ(live, 0);
}

TEST(FloatBufferTest, UniquePtrReleasesDuringExceptionUnwind) {
    int live = 0;
    try {
        const std::unique_ptr<Probe> owned = std::make_unique<Probe>(live);
        EXPECT_EQ(live, 1);
        throw std::runtime_error("expected");
    } catch (const std::runtime_error&) {
        EXPECT_EQ(live, 0);
    }
}
