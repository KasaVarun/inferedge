#include "inferedge/float_buffer.hpp"
#include "inferedge/request.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <gtest/gtest.h>
#include <queue>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// These tests exercise containers. They do not schedule work.
//
// vector keeps values in one contiguous block. reserve() holds enough
// capacity that addresses stay valid while size remains inside it. Growth
// past that capacity can move every element. The factor is
// implementation-defined, so these tests do not name one, and they do not
// keep a pointer across a push that would exceed capacity.
//
// deque keeps values in separate blocks. push_front and push_back leave
// references to existing elements valid and invalidate iterators. This file
// keeps a pointer across end insertions and does not keep an iterator.
//
// queue is a FIFO adapter over deque. pop() returns void. The caller moves
// from front(), then pops, so a throw during that move cannot drop the only
// copy. priority_queue is a heap. SchedulingOrder decides rank.
// RequestPriority's declaration order does not. top() returns const T&, so
// the heap stores copyable requests. A move-only FloatBuffer cannot be moved
// out through top(). Equal keys are not stable; a sequence number is the
// FIFO tie-break. Always taking interactive work first can leave background
// work waiting. No fairness rule is implemented here.
//
// unordered_map finds a request by id. These tests use find() on known keys.
// They do not walk the table, and they do not treat average-case lookup as a
// worst-case bound.

namespace {

inferedge::InferenceRequest make_request(std::string request_id,
                                         inferedge::RequestPriority priority,
                                         std::vector<float> input_values) {
    return inferedge::InferenceRequest{
        .request_id = std::move(request_id),
        .model_id = "model",
        .input_values = std::move(input_values),
        .priority = priority,
    };
}

struct SequencedRequest {
    inferedge::InferenceRequest request;
    std::uint64_t sequence = 0;
};

// "Less" means lower scheduling rank. priority_queue puts the greatest element
// at top(). Background is less than interactive. Within one priority, a larger
// sequence is less, so the smaller sequence comes out first.
struct SchedulingOrder {
    bool operator()(const SequencedRequest& left, const SequencedRequest& right) const {
        const bool left_interactive =
            left.request.priority == inferedge::RequestPriority::kInteractive;
        const bool right_interactive =
            right.request.priority == inferedge::RequestPriority::kInteractive;
        if (left_interactive != right_interactive) {
            return right_interactive;
        }
        return left.sequence > right.sequence;
    }
};

using RequestHeap =
    std::priority_queue<SequencedRequest, std::vector<SequencedRequest>, SchedulingOrder>;

} // namespace

TEST(StlContainerTest, VectorStoresIndependentRequests) {
    inferedge::InferenceRequest first =
        make_request("first", inferedge::RequestPriority::kBackground, {1.0f, 2.0f});
    inferedge::InferenceRequest second =
        make_request("second", inferedge::RequestPriority::kInteractive, {3.0f});

    std::vector<inferedge::InferenceRequest> stored;
    stored.push_back(first);
    stored.push_back(second);

    first.input_values[0] = 9.0f;
    stored[1].input_values[0] = 8.0f;

    EXPECT_FLOAT_EQ(stored[0].input_values[0], 1.0f);
    EXPECT_FLOAT_EQ(stored[0].input_values[1], 2.0f);
    EXPECT_EQ(stored[0].request_id, "first");
    EXPECT_FLOAT_EQ(second.input_values[0], 3.0f);
    EXPECT_EQ(stored[1].request_id, "second");
    EXPECT_EQ(stored.size(), 2u);
}

TEST(StlContainerTest, VectorReserveKeepsElementAddress) {
    std::vector<inferedge::InferenceRequest> stored;
    stored.reserve(2);
    const std::size_t reserved = stored.capacity();
    ASSERT_GE(reserved, 2u);

    stored.push_back(make_request("first", inferedge::RequestPriority::kBackground, {1.0f}));
    inferedge::InferenceRequest* const first = &stored[0];
    stored.push_back(make_request("second", inferedge::RequestPriority::kInteractive, {2.0f}));

    EXPECT_EQ(stored.size(), 2u);
    EXPECT_GE(stored.capacity(), stored.size());
    EXPECT_EQ(stored.capacity(), reserved);
    EXPECT_EQ(first, &stored[0]);
    EXPECT_EQ(first->request_id, "first");
    EXPECT_EQ(stored[1].request_id, "second");
}

TEST(StlContainerTest, DequePreservesFifoOrder) {
    std::deque<inferedge::InferenceRequest> line;
    line.push_back(make_request("middle", inferedge::RequestPriority::kBackground, {2.0f}));
    inferedge::InferenceRequest* const middle = &line[0];

    line.push_front(make_request("first", inferedge::RequestPriority::kBackground, {1.0f}));
    line.push_back(make_request("last", inferedge::RequestPriority::kInteractive, {3.0f}));

    EXPECT_EQ(middle->request_id, "middle");
    EXPECT_EQ(line.size(), 3u);
    EXPECT_EQ(line.front().request_id, "first");
    line.pop_front();
    EXPECT_EQ(line.front().request_id, "middle");
    line.pop_front();
    EXPECT_EQ(line.front().request_id, "last");
    line.pop_front();
    EXPECT_TRUE(line.empty());
}

TEST(StlContainerTest, QueuePopsInPushOrder) {
    std::queue<inferedge::InferenceRequest> waiting;
    waiting.push(make_request("first", inferedge::RequestPriority::kBackground, {1.0f}));
    waiting.push(make_request("second", inferedge::RequestPriority::kInteractive, {2.0f}));

    EXPECT_EQ(waiting.size(), 2u);
    EXPECT_EQ(waiting.front().request_id, "first");
    EXPECT_EQ(waiting.back().request_id, "second");
    waiting.pop();
    EXPECT_EQ(waiting.size(), 1u);
    EXPECT_EQ(waiting.front().request_id, "second");
    waiting.pop();
    EXPECT_TRUE(waiting.empty());
}

TEST(StlContainerTest, PriorityPopsInteractiveBeforeBackground) {
    RequestHeap heap;
    heap.push(SequencedRequest{
        .request = make_request("bg-a", inferedge::RequestPriority::kBackground, {1.0f}),
        .sequence = 1,
    });
    heap.push(SequencedRequest{
        .request = make_request("bg-b", inferedge::RequestPriority::kBackground, {2.0f}),
        .sequence = 2,
    });
    heap.push(SequencedRequest{
        .request = make_request("interactive", inferedge::RequestPriority::kInteractive, {3.0f}),
        .sequence = 3,
    });

    ASSERT_FALSE(heap.empty());
    EXPECT_EQ(heap.top().request.request_id, "interactive");
    EXPECT_EQ(heap.top().request.priority, inferedge::RequestPriority::kInteractive);
    heap.pop();

    std::vector<std::string> background_ids;
    while (!heap.empty()) {
        EXPECT_EQ(heap.top().request.priority, inferedge::RequestPriority::kBackground);
        background_ids.push_back(heap.top().request.request_id);
        heap.pop();
    }
    std::sort(background_ids.begin(), background_ids.end());
    EXPECT_EQ(background_ids, (std::vector<std::string>{"bg-a", "bg-b"}));
}

TEST(StlContainerTest, EqualPriorityUsesSequenceNumber) {
    RequestHeap heap;
    heap.push(SequencedRequest{
        .request = make_request("third", inferedge::RequestPriority::kBackground, {3.0f}),
        .sequence = 3,
    });
    heap.push(SequencedRequest{
        .request = make_request("first", inferedge::RequestPriority::kBackground, {1.0f}),
        .sequence = 1,
    });
    heap.push(SequencedRequest{
        .request = make_request("second", inferedge::RequestPriority::kBackground, {2.0f}),
        .sequence = 2,
    });

    ASSERT_EQ(heap.size(), 3u);
    EXPECT_EQ(heap.top().request.request_id, "first");
    EXPECT_EQ(heap.top().sequence, 1u);
    heap.pop();
    EXPECT_EQ(heap.top().request.request_id, "second");
    EXPECT_EQ(heap.top().sequence, 2u);
    heap.pop();
    EXPECT_EQ(heap.top().request.request_id, "third");
    EXPECT_EQ(heap.top().sequence, 3u);
    heap.pop();
    EXPECT_TRUE(heap.empty());
}

TEST(StlContainerTest, UnorderedMapFindsRequestById) {
    std::unordered_map<std::string, inferedge::InferenceRequest> by_id;
    by_id.emplace("req-a", make_request("req-a", inferedge::RequestPriority::kInteractive, {1.0f}));
    by_id.emplace("req-b", make_request("req-b", inferedge::RequestPriority::kBackground, {2.0f}));

    const auto found = by_id.find("req-a");
    ASSERT_NE(found, by_id.end());
    EXPECT_EQ(found->second.request_id, "req-a");
    EXPECT_EQ(found->second.priority, inferedge::RequestPriority::kInteractive);
    EXPECT_FLOAT_EQ(found->second.input_values[0], 1.0f);

    found->second.input_values[0] = 50.0f;
    const auto other = by_id.find("req-b");
    ASSERT_NE(other, by_id.end());
    EXPECT_FLOAT_EQ(other->second.input_values[0], 2.0f);
    EXPECT_EQ(by_id.find("missing"), by_id.end());
}

TEST(StlContainerTest, VectorOwnsMovedFloatBuffers) {
    static_assert(!std::is_copy_constructible_v<inferedge::FloatBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<inferedge::FloatBuffer>);
    // vector declares a copy constructor even when T is move-only, so
    // is_copy_constructible can be true. Instantiating that copy is what fails.
    // This test moves the buffers instead.
    static_assert(std::is_nothrow_move_constructible_v<std::vector<inferedge::FloatBuffer>>);

    std::vector<inferedge::FloatBuffer> buffers;
    buffers.push_back(inferedge::FloatBuffer{std::vector<float>{1.0f, 2.0f}});
    buffers.emplace_back(std::vector<float>{3.0f});

    ASSERT_EQ(buffers.size(), 2u);
    inferedge::FloatBuffer owned = std::move(buffers.back());
    buffers.pop_back();

    EXPECT_EQ(buffers.size(), 1u);
    ASSERT_EQ(buffers.front().view().size(), 2u);
    EXPECT_FLOAT_EQ(buffers.front().view()[0], 1.0f);
    EXPECT_FLOAT_EQ(buffers.front().view()[1], 2.0f);
    ASSERT_EQ(owned.view().size(), 1u);
    EXPECT_FLOAT_EQ(owned.view()[0], 3.0f);
}

TEST(StlContainerTest, QueueMovesFloatBufferOutThroughFront) {
    std::queue<inferedge::FloatBuffer> waiting;
    waiting.push(inferedge::FloatBuffer{std::vector<float>{4.0f, 5.0f}});
    ASSERT_EQ(waiting.size(), 1u);

    inferedge::FloatBuffer owned = std::move(waiting.front());
    waiting.pop();

    EXPECT_TRUE(waiting.empty());
    ASSERT_EQ(owned.view().size(), 2u);
    EXPECT_FLOAT_EQ(owned.view()[0], 4.0f);
    EXPECT_FLOAT_EQ(owned.view()[1], 5.0f);
}
