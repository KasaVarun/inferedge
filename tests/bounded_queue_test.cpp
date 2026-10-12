#include "inferedge/bounded_queue.hpp"

#include <algorithm>
#include <atomic>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <latch>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

class JoinThread {
public:
    template <typename Function>
        requires(!std::is_same_v<std::decay_t<Function>, JoinThread>)
    explicit JoinThread(Function&& function) : thread_(std::forward<Function>(function)) {}

    JoinThread(const JoinThread&) = delete;
    JoinThread& operator=(const JoinThread&) = delete;
    JoinThread(JoinThread&& other) noexcept : thread_(std::move(other.thread_)) {}
    JoinThread& operator=(JoinThread&&) = delete;

    ~JoinThread() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::thread thread_;
};

struct CompletionItem {
    int id{0};
    std::promise<int> completion;

    explicit CompletionItem(int item_id) : id{item_id} {}

    CompletionItem(const CompletionItem&) = delete;
    CompletionItem& operator=(const CompletionItem&) = delete;
    CompletionItem(CompletionItem&&) noexcept = default;
    CompletionItem& operator=(CompletionItem&&) noexcept = default;
};

// Move-only element whose move constructor can be armed to throw. A negative
// throw_after disables the throw. Destruction of a moved-from object, whose
// id is negative, is not recorded.
struct ThrowingMove {
    int id{-1};
    static int moves;
    static int throw_after;
    static std::vector<int> destroyed_ids;

    explicit ThrowingMove(int item_id) : id{item_id} {}

    ThrowingMove(const ThrowingMove&) = delete;
    ThrowingMove& operator=(const ThrowingMove&) = delete;
    ThrowingMove& operator=(ThrowingMove&&) = delete;

    ThrowingMove(ThrowingMove&& other) : id{other.id} {
        ++moves;
        if (throw_after >= 0 && moves > throw_after) {
            throw std::runtime_error("element move");
        }
        other.id = -1;
    }

    ~ThrowingMove() {
        if (id >= 0) {
            destroyed_ids.push_back(id);
        }
    }
};

int ThrowingMove::moves = 0;
int ThrowingMove::throw_after = -1;
std::vector<int> ThrowingMove::destroyed_ids;

template <typename T> T take_ready(inferedge::QueuePop<T> popped) {
    EXPECT_EQ(popped.result, inferedge::QueuePopResult::kReady);
    if (!popped.value.has_value()) {
        throw std::logic_error("popped value is empty");
    }
    return std::move(*popped.value);
}

} // namespace

TEST(BoundedFifoQueueTest, RejectsZeroCapacity) {
    EXPECT_THROW(inferedge::BoundedFifoQueue<int>(0), std::invalid_argument);
}

TEST(BoundedFifoQueueTest, AcceptsCapacityOne) {
    inferedge::BoundedFifoQueue<int> queue(1);
    EXPECT_EQ(queue.capacity(), 1U);
    EXPECT_EQ(queue.size(), 0U);
    EXPECT_FALSE(queue.is_closed());

    const inferedge::QueuePush<int> pushed = queue.try_push(4);
    EXPECT_EQ(pushed.result, inferedge::QueuePushResult::kAccepted);
    EXPECT_FALSE(pushed.rejected.has_value());
    EXPECT_EQ(queue.size(), 1U);
    EXPECT_EQ(take_ready(queue.wait_pop()), 4);
    EXPECT_EQ(queue.size(), 0U);
}

TEST(BoundedFifoQueueTest, RejectsAPushWhenFull) {
    inferedge::BoundedFifoQueue<int> queue(1);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);

    inferedge::QueuePush<int> rejected = queue.try_push(2);
    EXPECT_EQ(rejected.result, inferedge::QueuePushResult::kFull);
    ASSERT_TRUE(rejected.rejected.has_value());
    EXPECT_EQ(*rejected.rejected, 2);
    EXPECT_EQ(queue.size(), 1U);
    EXPECT_EQ(take_ready(queue.wait_pop()), 1);
}

TEST(BoundedFifoQueueTest, PopsIntegersInFifoOrder) {
    inferedge::BoundedFifoQueue<int> queue(4);
    EXPECT_EQ(queue.try_push(10).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(20).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(30).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(take_ready(queue.wait_pop()), 10);
    EXPECT_EQ(take_ready(queue.wait_pop()), 20);
    EXPECT_EQ(take_ready(queue.wait_pop()), 30);
}

TEST(BoundedFifoQueueTest, PopsMoveOnlyValuesInFifoOrder) {
    inferedge::BoundedFifoQueue<std::unique_ptr<int>> queue(3);
    EXPECT_EQ(queue.try_push(std::make_unique<int>(1)).result,
              inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(std::make_unique<int>(2)).result,
              inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(std::make_unique<int>(3)).result,
              inferedge::QueuePushResult::kAccepted);

    const std::unique_ptr<int> first = take_ready(queue.wait_pop());
    const std::unique_ptr<int> second = take_ready(queue.wait_pop());
    const std::unique_ptr<int> third = take_ready(queue.wait_pop());
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_NE(third, nullptr);
    EXPECT_EQ(*first, 1);
    EXPECT_EQ(*second, 2);
    EXPECT_EQ(*third, 3);
}

TEST(BoundedFifoQueueTest, QueuedValuesOwnTheirStorage) {
    inferedge::BoundedFifoQueue<std::unique_ptr<int>> queue(1);
    auto original = std::make_unique<int>(42);
    const int* const address = original.get();
    EXPECT_EQ(queue.try_push(std::move(original)).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(original, nullptr);

    const std::unique_ptr<int> owned = take_ready(queue.wait_pop());
    ASSERT_NE(owned, nullptr);
    EXPECT_EQ(owned.get(), address);
    EXPECT_EQ(*owned, 42);
}

TEST(BoundedFifoQueueTest, WakesAWaitingPopOnPush) {
    inferedge::BoundedFifoQueue<int> queue(1);
    std::promise<void> started;
    std::promise<inferedge::QueuePop<int>> finished;
    JoinThread consumer([&] {
        started.set_value();
        finished.set_value(queue.wait_pop());
    });
    started.get_future().wait();

    EXPECT_EQ(queue.try_push(7).result, inferedge::QueuePushResult::kAccepted);
    const inferedge::QueuePop<int> popped = finished.get_future().get();
    EXPECT_EQ(take_ready(popped), 7);
}

TEST(BoundedFifoQueueTest, WakesAWaitingPopOnDrain) {
    inferedge::BoundedFifoQueue<int> queue(1);
    std::promise<void> started;
    std::promise<inferedge::QueuePop<int>> finished;
    JoinThread consumer([&] {
        started.set_value();
        finished.set_value(queue.wait_pop());
    });
    started.get_future().wait();

    const inferedge::QueueClose<int> closed = queue.close_drain();
    EXPECT_TRUE(closed.changed);
    EXPECT_EQ(closed.mode, inferedge::QueueCloseMode::kDrain);
    const inferedge::QueuePop<int> popped = finished.get_future().get();
    EXPECT_EQ(popped.result, inferedge::QueuePopResult::kClosed);
    EXPECT_FALSE(popped.value.has_value());
}

TEST(BoundedFifoQueueTest, WakesAWaitingPopOnCancel) {
    inferedge::BoundedFifoQueue<int> queue(1);
    std::promise<void> started;
    std::promise<inferedge::QueuePop<int>> finished;
    JoinThread consumer([&] {
        started.set_value();
        finished.set_value(queue.wait_pop());
    });
    started.get_future().wait();

    const inferedge::QueueClose<int> closed = queue.close_cancel();
    EXPECT_TRUE(closed.changed);
    EXPECT_EQ(closed.mode, inferedge::QueueCloseMode::kCancel);
    EXPECT_TRUE(closed.canceled.empty());
    const inferedge::QueuePop<int> popped = finished.get_future().get();
    EXPECT_EQ(popped.result, inferedge::QueuePopResult::kClosed);
    EXPECT_FALSE(popped.value.has_value());
}

TEST(BoundedFifoQueueTest, DrainPreservesQueuedValues) {
    inferedge::BoundedFifoQueue<int> queue(3);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(3).result, inferedge::QueuePushResult::kAccepted);

    const inferedge::QueueClose<int> closed = queue.close_drain();
    EXPECT_TRUE(closed.changed);
    EXPECT_TRUE(closed.canceled.empty());
    EXPECT_EQ(queue.size(), 3U);
    EXPECT_EQ(take_ready(queue.wait_pop()), 1);
    EXPECT_EQ(take_ready(queue.wait_pop()), 2);
    EXPECT_EQ(take_ready(queue.wait_pop()), 3);
    EXPECT_EQ(queue.wait_pop().result, inferedge::QueuePopResult::kClosed);
}

TEST(BoundedFifoQueueTest, CancelReturnsPendingValuesInOrder) {
    inferedge::BoundedFifoQueue<int> queue(3);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(3).result, inferedge::QueuePushResult::kAccepted);

    const inferedge::QueueClose<int> closed = queue.close_cancel();
    EXPECT_TRUE(closed.changed);
    EXPECT_EQ(closed.mode, inferedge::QueueCloseMode::kCancel);
    ASSERT_EQ(closed.canceled.size(), 3U);
    EXPECT_EQ(closed.canceled[0], 1);
    EXPECT_EQ(closed.canceled[1], 2);
    EXPECT_EQ(closed.canceled[2], 3);
    EXPECT_EQ(queue.size(), 0U);
    EXPECT_EQ(queue.wait_pop().result, inferedge::QueuePopResult::kClosed);
}

TEST(BoundedFifoQueueTest, AlreadyPoppedValuesStayWithTheConsumer) {
    inferedge::BoundedFifoQueue<std::unique_ptr<int>> queue(2);
    EXPECT_EQ(queue.try_push(std::make_unique<int>(8)).result,
              inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(std::make_unique<int>(9)).result,
              inferedge::QueuePushResult::kAccepted);

    const std::unique_ptr<int> popped = take_ready(queue.wait_pop());
    const inferedge::QueueClose<std::unique_ptr<int>> closed = queue.close_cancel();
    ASSERT_EQ(closed.canceled.size(), 1U);
    ASSERT_NE(popped, nullptr);
    ASSERT_NE(closed.canceled[0], nullptr);
    EXPECT_EQ(*popped, 8);
    EXPECT_EQ(*closed.canceled[0], 9);
}

TEST(BoundedFifoQueueTest, RepeatedDrainIsIdempotent) {
    inferedge::BoundedFifoQueue<int> queue(2);
    EXPECT_EQ(queue.try_push(5).result, inferedge::QueuePushResult::kAccepted);
    const inferedge::QueueClose<int> first = queue.close_drain();
    const inferedge::QueueClose<int> second = queue.close_drain();
    EXPECT_TRUE(first.changed);
    EXPECT_FALSE(second.changed);
    EXPECT_EQ(first.mode, inferedge::QueueCloseMode::kDrain);
    EXPECT_EQ(second.mode, inferedge::QueueCloseMode::kDrain);
    EXPECT_TRUE(second.canceled.empty());
    EXPECT_EQ(take_ready(queue.wait_pop()), 5);
}

TEST(BoundedFifoQueueTest, RepeatedCancelIsIdempotent) {
    inferedge::BoundedFifoQueue<int> queue(2);
    EXPECT_EQ(queue.try_push(5).result, inferedge::QueuePushResult::kAccepted);
    const inferedge::QueueClose<int> first = queue.close_cancel();
    const inferedge::QueueClose<int> second = queue.close_cancel();
    EXPECT_TRUE(first.changed);
    EXPECT_FALSE(second.changed);
    EXPECT_EQ(second.mode, inferedge::QueueCloseMode::kCancel);
    ASSERT_EQ(first.canceled.size(), 1U);
    EXPECT_EQ(first.canceled[0], 5);
    EXPECT_TRUE(second.canceled.empty());
}

TEST(BoundedFifoQueueTest, CancelAfterDrainPreservesDrain) {
    inferedge::BoundedFifoQueue<int> queue(2);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.close_drain().mode, inferedge::QueueCloseMode::kDrain);

    const inferedge::QueueClose<int> later = queue.close_cancel();
    EXPECT_FALSE(later.changed);
    EXPECT_EQ(later.mode, inferedge::QueueCloseMode::kDrain);
    EXPECT_TRUE(later.canceled.empty());
    EXPECT_EQ(queue.close_mode(), inferedge::QueueCloseMode::kDrain);
    EXPECT_EQ(take_ready(queue.wait_pop()), 1);
    EXPECT_EQ(take_ready(queue.wait_pop()), 2);
}

TEST(BoundedFifoQueueTest, DrainAfterCancelPreservesCancel) {
    inferedge::BoundedFifoQueue<int> queue(2);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    const inferedge::QueueClose<int> canceled = queue.close_cancel();
    const inferedge::QueueClose<int> later = queue.close_drain();
    EXPECT_FALSE(later.changed);
    EXPECT_EQ(later.mode, inferedge::QueueCloseMode::kCancel);
    EXPECT_TRUE(later.canceled.empty());
    ASSERT_EQ(canceled.canceled.size(), 1U);
    EXPECT_EQ(canceled.canceled[0], 1);
    EXPECT_EQ(queue.wait_pop().result, inferedge::QueuePopResult::kClosed);
    EXPECT_EQ(queue.close_mode(), inferedge::QueueCloseMode::kCancel);
}

TEST(BoundedFifoQueueTest, PushAfterDrainIsRejected) {
    inferedge::BoundedFifoQueue<int> queue(1);
    EXPECT_TRUE(queue.close_drain().changed);
    inferedge::QueuePush<int> rejected = queue.try_push(9);
    EXPECT_EQ(rejected.result, inferedge::QueuePushResult::kClosed);
    ASSERT_TRUE(rejected.rejected.has_value());
    EXPECT_EQ(*rejected.rejected, 9);
    EXPECT_EQ(queue.size(), 0U);
}

TEST(BoundedFifoQueueTest, PushAfterCancelIsRejected) {
    inferedge::BoundedFifoQueue<std::unique_ptr<int>> queue(1);
    EXPECT_TRUE(queue.close_cancel().changed);
    auto value = std::make_unique<int>(9);
    const int* const address = value.get();
    inferedge::QueuePush<std::unique_ptr<int>> rejected = queue.try_push(std::move(value));
    EXPECT_EQ(rejected.result, inferedge::QueuePushResult::kClosed);
    ASSERT_TRUE(rejected.rejected.has_value());
    EXPECT_EQ(rejected.rejected->get(), address);
    EXPECT_EQ(**rejected.rejected, 9);
}

TEST(BoundedFifoQueueTest, ConcurrentProducersAccountForEveryValue) {
    constexpr int kProducers = 4;
    constexpr int kEach = 20;
    inferedge::BoundedFifoQueue<int> queue(static_cast<std::size_t>(kProducers * kEach));
    std::latch start(static_cast<std::ptrdiff_t>(kProducers));
    std::mutex accepted_mutex;
    std::vector<int> accepted;

    std::vector<JoinThread> producers;
    producers.reserve(kProducers);
    for (int producer = 0; producer < kProducers; ++producer) {
        producers.emplace_back([&, producer] {
            start.arrive_and_wait();
            for (int offset = 0; offset < kEach; ++offset) {
                const int value = producer * kEach + offset;
                const inferedge::QueuePush<int> pushed = queue.try_push(value);
                EXPECT_EQ(pushed.result, inferedge::QueuePushResult::kAccepted);
                std::lock_guard<std::mutex> lock(accepted_mutex);
                accepted.push_back(value);
            }
        });
    }
    producers.clear();

    std::vector<int> popped;
    popped.reserve(accepted.size());
    for (std::size_t index = 0; index < accepted.size(); ++index) {
        popped.push_back(take_ready(queue.wait_pop()));
    }
    std::sort(accepted.begin(), accepted.end());
    std::sort(popped.begin(), popped.end());
    EXPECT_EQ(popped, accepted);
    EXPECT_EQ(accepted.size(), static_cast<std::size_t>(kProducers * kEach));
}

TEST(BoundedFifoQueueTest, OrderedProducersKeepAdmissionOrder) {
    inferedge::BoundedFifoQueue<int> queue(2);
    std::promise<void> first_admitted;
    JoinThread first([&] {
        EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
        first_admitted.set_value();
    });
    first_admitted.get_future().wait();
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);

    EXPECT_EQ(take_ready(queue.wait_pop()), 1);
    EXPECT_EQ(take_ready(queue.wait_pop()), 2);
}

TEST(BoundedFifoQueueTest, ConcurrentConsumersTakeEachValueOnce) {
    constexpr int kItems = 50;
    constexpr int kConsumers = 4;
    inferedge::BoundedFifoQueue<int> queue(static_cast<std::size_t>(kItems));
    for (int value = 0; value < kItems; ++value) {
        EXPECT_EQ(queue.try_push(value).result, inferedge::QueuePushResult::kAccepted);
    }

    std::mutex popped_mutex;
    std::vector<int> popped;
    std::vector<JoinThread> consumers;
    consumers.reserve(kConsumers);
    for (int consumer = 0; consumer < kConsumers; ++consumer) {
        consumers.emplace_back([&] {
            for (;;) {
                inferedge::QueuePop<int> item = queue.wait_pop();
                if (item.result == inferedge::QueuePopResult::kClosed) {
                    return;
                }
                if (!item.value.has_value()) {
                    ADD_FAILURE() << "ready pop has no value";
                    return;
                }
                std::lock_guard<std::mutex> lock(popped_mutex);
                popped.push_back(*item.value);
            }
        });
    }
    EXPECT_TRUE(queue.close_drain().changed);
    consumers.clear();

    std::sort(popped.begin(), popped.end());
    EXPECT_EQ(popped.size(), static_cast<std::size_t>(kItems));
    EXPECT_EQ(std::unique(popped.begin(), popped.end()), popped.end());
    EXPECT_EQ(popped.front(), 0);
    EXPECT_EQ(popped.back(), kItems - 1);
}

TEST(BoundedFifoQueueTest, CapacityHoldsUnderContention) {
    constexpr int kProducers = 4;
    constexpr int kEach = 30;
    constexpr std::size_t kCapacity = 2;
    inferedge::BoundedFifoQueue<int> queue(kCapacity);
    std::atomic<int> violations{0};
    std::mutex record_mutex;
    std::vector<int> accepted;
    std::vector<int> rejected;
    std::vector<int> popped;

    auto note_size = [&] {
        if (queue.size() > queue.capacity()) {
            violations.fetch_add(1, std::memory_order_relaxed);
        }
    };

    std::vector<JoinThread> consumers;
    for (int consumer = 0; consumer < 2; ++consumer) {
        consumers.emplace_back([&] {
            for (;;) {
                inferedge::QueuePop<int> item = queue.wait_pop();
                note_size();
                if (item.result == inferedge::QueuePopResult::kClosed) {
                    return;
                }
                if (!item.value.has_value()) {
                    ADD_FAILURE() << "ready pop has no value";
                    return;
                }
                std::lock_guard<std::mutex> lock(record_mutex);
                popped.push_back(*item.value);
            }
        });
    }

    std::atomic<int> next_id{0};
    std::vector<JoinThread> producers;
    for (int producer = 0; producer < kProducers; ++producer) {
        producers.emplace_back([&] {
            for (int count = 0; count < kEach; ++count) {
                const int id = next_id.fetch_add(1, std::memory_order_relaxed);
                inferedge::QueuePush<int> result = queue.try_push(id);
                note_size();
                std::lock_guard<std::mutex> lock(record_mutex);
                if (result.result == inferedge::QueuePushResult::kAccepted) {
                    accepted.push_back(id);
                } else {
                    EXPECT_EQ(result.result, inferedge::QueuePushResult::kFull);
                    if (!result.rejected.has_value()) {
                        ADD_FAILURE() << "rejected push has no value";
                        return;
                    }
                    rejected.push_back(*result.rejected);
                }
            }
        });
    }

    JoinThread sampler([&] {
        for (int sample = 0; sample < 1000; ++sample) {
            note_size();
        }
    });

    producers.clear();
    EXPECT_TRUE(queue.close_drain().changed);
    consumers.clear();

    EXPECT_EQ(violations.load(), 0);
    EXPECT_EQ(accepted.size() + rejected.size(), static_cast<std::size_t>(kProducers * kEach));
    std::sort(accepted.begin(), accepted.end());
    std::sort(rejected.begin(), rejected.end());
    std::sort(popped.begin(), popped.end());
    EXPECT_EQ(popped, accepted);
    std::vector<int> overlap;
    std::set_intersection(accepted.begin(), accepted.end(), rejected.begin(), rejected.end(),
                          std::back_inserter(overlap));
    EXPECT_TRUE(overlap.empty());
}

TEST(BoundedFifoQueueTest, MoveOnlyCompletionHasOneOwner) {
    inferedge::BoundedFifoQueue<CompletionItem> queue(2);
    CompletionItem first{11};
    CompletionItem second{22};
    std::future<int> first_future = first.completion.get_future();
    std::future<int> second_future = second.completion.get_future();
    EXPECT_EQ(queue.try_push(std::move(first)).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(std::move(second)).result, inferedge::QueuePushResult::kAccepted);

    CompletionItem popped = take_ready(queue.wait_pop());
    EXPECT_EQ(popped.id, 11);
    popped.completion.set_value(popped.id);

    inferedge::QueueClose<CompletionItem> canceled = queue.close_cancel();
    ASSERT_EQ(canceled.canceled.size(), 1U);
    EXPECT_EQ(canceled.canceled[0].id, 22);
    canceled.canceled[0].completion.set_value(canceled.canceled[0].id);

    EXPECT_EQ(first_future.get(), 11);
    EXPECT_EQ(second_future.get(), 22);
    EXPECT_THROW(popped.completion.set_value(11), std::future_error);
    EXPECT_THROW(canceled.canceled[0].completion.set_value(22), std::future_error);
}

TEST(BoundedFifoQueueTest, CancelDoesNotMoveElements) {
    static_assert(!std::is_nothrow_move_constructible_v<ThrowingMove>);
    static_assert(std::is_nothrow_move_constructible_v<inferedge::QueueClose<ThrowingMove>>);
    static_assert(std::is_nothrow_swappable_v<std::deque<ThrowingMove>>);

    ThrowingMove::moves = 0;
    ThrowingMove::throw_after = -1;
    ThrowingMove::destroyed_ids.clear();

    inferedge::BoundedFifoQueue<ThrowingMove> queue(3);
    {
        ThrowingMove first{1};
        ThrowingMove second{2};
        ThrowingMove third{3};
        EXPECT_EQ(queue.try_push(std::move(first)).result, inferedge::QueuePushResult::kAccepted);
        EXPECT_EQ(queue.try_push(std::move(second)).result, inferedge::QueuePushResult::kAccepted);
        EXPECT_EQ(queue.try_push(std::move(third)).result, inferedge::QueuePushResult::kAccepted);
    }
    EXPECT_GT(ThrowingMove::moves, 0);
    EXPECT_TRUE(ThrowingMove::destroyed_ids.empty());
    EXPECT_EQ(queue.size(), 3U);

    const int moves_after_enqueue = ThrowingMove::moves;
    ThrowingMove::throw_after = moves_after_enqueue;
    {
        const inferedge::QueueClose<ThrowingMove> closed = queue.close_cancel();
        EXPECT_EQ(ThrowingMove::moves, moves_after_enqueue);
        EXPECT_TRUE(ThrowingMove::destroyed_ids.empty());
        EXPECT_TRUE(closed.changed);
        EXPECT_EQ(closed.mode, inferedge::QueueCloseMode::kCancel);
        EXPECT_TRUE(queue.is_closed());
        EXPECT_EQ(queue.close_mode(), inferedge::QueueCloseMode::kCancel);
        EXPECT_EQ(queue.size(), 0U);
        ASSERT_EQ(closed.canceled.size(), 3U);
        EXPECT_EQ(closed.canceled[0].id, 1);
        EXPECT_EQ(closed.canceled[1].id, 2);
        EXPECT_EQ(closed.canceled[2].id, 3);
        EXPECT_EQ(queue.wait_pop().result, inferedge::QueuePopResult::kClosed);
    }

    std::vector<int> destroyed = ThrowingMove::destroyed_ids;
    std::sort(destroyed.begin(), destroyed.end());
    ASSERT_EQ(destroyed.size(), 3U);
    EXPECT_EQ(destroyed[0], 1);
    EXPECT_EQ(destroyed[1], 2);
    EXPECT_EQ(destroyed[2], 3);
}

TEST(BoundedFifoQueueTest, CancelTransfersEveryPromise) {
    inferedge::BoundedFifoQueue<CompletionItem> queue(3);
    std::vector<std::future<int>> futures;
    for (const int id : {1, 2, 3}) {
        CompletionItem item{id};
        futures.push_back(item.completion.get_future());
        EXPECT_EQ(queue.try_push(std::move(item)).result, inferedge::QueuePushResult::kAccepted);
    }

    inferedge::QueueClose<CompletionItem> canceled = queue.close_cancel();
    ASSERT_EQ(canceled.canceled.size(), 3U);
    EXPECT_TRUE(queue.is_closed());
    EXPECT_EQ(queue.size(), 0U);
    for (std::size_t index = 0; index < canceled.canceled.size(); ++index) {
        const int id = static_cast<int>(index) + 1;
        EXPECT_EQ(canceled.canceled[index].id, id);
        canceled.canceled[index].completion.set_value(id);
        try {
            canceled.canceled[index].completion.set_value(id);
            ADD_FAILURE() << "canceled promise accepted a second completion";
        } catch (const std::future_error& error) {
            EXPECT_EQ(error.code(),
                      std::make_error_code(std::future_errc::promise_already_satisfied));
        }
        EXPECT_EQ(futures[index].get(), id);
    }
}

TEST(BoundedFifoQueueTest, ClosedEmptyPopReturnsClosed) {
    inferedge::BoundedFifoQueue<int> queue(1);
    EXPECT_TRUE(queue.close_drain().changed);
    const inferedge::QueuePop<int> popped = queue.wait_pop();
    EXPECT_EQ(popped.result, inferedge::QueuePopResult::kClosed);
    EXPECT_FALSE(popped.value.has_value());
}

TEST(BoundedFifoQueueTest, MetadataReadsASynchronizedSnapshot) {
    inferedge::BoundedFifoQueue<int> queue(3);
    EXPECT_EQ(queue.capacity(), 3U);
    EXPECT_EQ(queue.size(), 0U);
    EXPECT_FALSE(queue.is_closed());
    EXPECT_FALSE(queue.close_mode().has_value());

    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.size(), 2U);
    EXPECT_EQ(queue.capacity(), 3U);

    EXPECT_EQ(queue.close_drain().mode, inferedge::QueueCloseMode::kDrain);
    EXPECT_TRUE(queue.is_closed());
    EXPECT_EQ(queue.close_mode(), inferedge::QueueCloseMode::kDrain);
    EXPECT_EQ(queue.size(), 2U);
    EXPECT_EQ(take_ready(queue.wait_pop()), 1);
    EXPECT_EQ(queue.size(), 1U);
}

TEST(BoundedFifoQueueTest, ConcurrentCloseChoosesOneMode) {
    inferedge::BoundedFifoQueue<int> queue(3);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(3).result, inferedge::QueuePushResult::kAccepted);

    std::latch start(2);
    std::promise<inferedge::QueueClose<int>> drain_result;
    std::promise<inferedge::QueueClose<int>> cancel_result;
    JoinThread drain([&] {
        start.arrive_and_wait();
        drain_result.set_value(queue.close_drain());
    });
    JoinThread cancel([&] {
        start.arrive_and_wait();
        cancel_result.set_value(queue.close_cancel());
    });
    inferedge::QueueClose<int> drained = drain_result.get_future().get();
    inferedge::QueueClose<int> canceled = cancel_result.get_future().get();

    EXPECT_EQ(drained.mode, canceled.mode);
    EXPECT_NE(drained.changed, canceled.changed);
    EXPECT_TRUE(drained.canceled.empty());
    if (drained.mode == inferedge::QueueCloseMode::kDrain) {
        EXPECT_TRUE(canceled.canceled.empty());
        EXPECT_EQ(take_ready(queue.wait_pop()), 1);
        EXPECT_EQ(take_ready(queue.wait_pop()), 2);
        EXPECT_EQ(take_ready(queue.wait_pop()), 3);
    } else {
        ASSERT_EQ(canceled.canceled.size(), 3U);
        EXPECT_EQ(canceled.canceled[0], 1);
        EXPECT_EQ(canceled.canceled[1], 2);
        EXPECT_EQ(canceled.canceled[2], 3);
        EXPECT_EQ(queue.wait_pop().result, inferedge::QueuePopResult::kClosed);
    }
}

TEST(BoundedFifoQueueTest, DrainWakesEveryBlockedConsumer) {
    constexpr int kConsumers = 3;
    inferedge::BoundedFifoQueue<int> queue(1);
    std::latch started(kConsumers);
    std::mutex results_mutex;
    std::vector<inferedge::QueuePopResult> results;
    std::vector<JoinThread> consumers;
    for (int consumer = 0; consumer < kConsumers; ++consumer) {
        consumers.emplace_back([&] {
            started.arrive_and_wait();
            const inferedge::QueuePop<int> popped = queue.wait_pop();
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(popped.result);
        });
    }
    started.wait();
    EXPECT_TRUE(queue.close_drain().changed);
    consumers.clear();
    EXPECT_EQ(results.size(), static_cast<std::size_t>(kConsumers));
    EXPECT_TRUE(std::all_of(results.begin(), results.end(), [](inferedge::QueuePopResult result) {
        return result == inferedge::QueuePopResult::kClosed;
    }));
}
