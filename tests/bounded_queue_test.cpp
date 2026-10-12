#include "inferedge/bounded_queue.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
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
// id is negative, is not recorded. The destructor does not allocate.
struct ThrowingMove {
    int id{-1};
    static constexpr int kMaxDestroyed = 8;
    static int moves;
    static int throw_after;
    static int destroyed_ids[kMaxDestroyed];
    static int destroyed_count;

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

    ~ThrowingMove() noexcept {
        if (id < 0 || destroyed_count >= kMaxDestroyed) {
            return;
        }
        destroyed_ids[destroyed_count] = id;
        ++destroyed_count;
    }
};

int ThrowingMove::moves = 0;
int ThrowingMove::throw_after = -1;
int ThrowingMove::destroyed_ids[ThrowingMove::kMaxDestroyed] = {};
int ThrowingMove::destroyed_count = 0;

void reset_throwing_move() {
    ThrowingMove::moves = 0;
    ThrowingMove::throw_after = -1;
    ThrowingMove::destroyed_count = 0;
}

template <typename T> T take_ready(inferedge::QueuePop<T> popped) {
    EXPECT_EQ(popped.result, inferedge::QueuePopResult::kReady);
    if (!popped.value.has_value()) {
        throw std::logic_error("popped value is empty");
    }
    return std::move(*popped.value);
}

std::chrono::steady_clock::time_point expired_deadline() {
    return std::chrono::steady_clock::now() - std::chrono::seconds(1);
}

std::chrono::steady_clock::time_point future_deadline() {
    return std::chrono::steady_clock::now() + std::chrono::seconds(10);
}

// Unblocks a consumer that is still inside the queue, then returns its result.
// The five-second bound is a deadlock guard. A successful wake returns sooner.
template <typename T, typename Queue> T finish_consumer(std::future<T>& future, Queue& queue) {
    if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        try {
            if (!queue.is_closed()) {
                static_cast<void>(queue.close_drain());
            }
        } catch (...) {
        }
        ADD_FAILURE() << "consumer did not finish";
    }
    return future.get();
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
    ASSERT_NE(closed.canceled, nullptr);
    EXPECT_TRUE(closed.canceled->empty());
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
    EXPECT_EQ(closed.canceled, nullptr);
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
    ASSERT_NE(closed.canceled, nullptr);
    ASSERT_EQ(closed.canceled->size(), 3U);
    EXPECT_EQ((*closed.canceled)[0], 1);
    EXPECT_EQ((*closed.canceled)[1], 2);
    EXPECT_EQ((*closed.canceled)[2], 3);
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
    ASSERT_NE(closed.canceled, nullptr);
    ASSERT_EQ(closed.canceled->size(), 1U);
    ASSERT_NE(popped, nullptr);
    ASSERT_NE((*closed.canceled)[0], nullptr);
    EXPECT_EQ(*popped, 8);
    EXPECT_EQ(*(*closed.canceled)[0], 9);
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
    EXPECT_EQ(first.canceled, nullptr);
    EXPECT_EQ(second.canceled, nullptr);
    EXPECT_EQ(take_ready(queue.wait_pop()), 5);
}

TEST(BoundedFifoQueueTest, RepeatedCancelIsIdempotent) {
    inferedge::BoundedFifoQueue<int> queue(2);
    EXPECT_EQ(queue.try_push(5).result, inferedge::QueuePushResult::kAccepted);
    const inferedge::QueueClose<int> first = queue.close_cancel();
    const inferedge::QueueClose<int> second = queue.close_cancel();
    EXPECT_TRUE(first.changed);
    EXPECT_FALSE(second.changed);
    EXPECT_EQ(first.mode, inferedge::QueueCloseMode::kCancel);
    EXPECT_EQ(second.mode, inferedge::QueueCloseMode::kCancel);
    ASSERT_NE(first.canceled, nullptr);
    ASSERT_EQ(first.canceled->size(), 1U);
    EXPECT_EQ((*first.canceled)[0], 5);
    EXPECT_EQ(second.canceled, nullptr);
}

TEST(BoundedFifoQueueTest, CancelAfterDrainPreservesDrain) {
    inferedge::BoundedFifoQueue<int> queue(2);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.close_drain().mode, inferedge::QueueCloseMode::kDrain);

    const inferedge::QueueClose<int> later = queue.close_cancel();
    EXPECT_FALSE(later.changed);
    EXPECT_EQ(later.mode, inferedge::QueueCloseMode::kDrain);
    EXPECT_EQ(later.canceled, nullptr);
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
    EXPECT_EQ(later.canceled, nullptr);
    ASSERT_NE(canceled.canceled, nullptr);
    ASSERT_EQ(canceled.canceled->size(), 1U);
    EXPECT_EQ((*canceled.canceled)[0], 1);
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
    ASSERT_NE(canceled.canceled, nullptr);
    const std::deque<CompletionItem>* const storage = canceled.canceled.get();
    inferedge::QueueClose<CompletionItem> delivered = std::move(canceled);
    ASSERT_EQ(delivered.canceled.get(), storage);
    EXPECT_EQ(canceled.canceled, nullptr);
    ASSERT_EQ(delivered.canceled->size(), 1U);
    EXPECT_EQ((*delivered.canceled)[0].id, 22);
    (*delivered.canceled)[0].completion.set_value((*delivered.canceled)[0].id);

    EXPECT_EQ(first_future.get(), 11);
    EXPECT_EQ(second_future.get(), 22);
    EXPECT_THROW(popped.completion.set_value(11), std::future_error);
    EXPECT_THROW((*delivered.canceled)[0].completion.set_value(22), std::future_error);
}

TEST(BoundedFifoQueueTest, QueueCloseIsNothrowMoveOnly) {
    static_assert(!std::is_nothrow_move_constructible_v<ThrowingMove>);
    static_assert(std::is_nothrow_destructible_v<ThrowingMove>);
    static_assert(std::is_nothrow_move_constructible_v<CompletionItem>);
    static_assert(std::is_nothrow_destructible_v<CompletionItem>);
    static_assert(std::is_nothrow_default_constructible_v<inferedge::QueueClose<ThrowingMove>>);
    static_assert(std::is_nothrow_move_constructible_v<inferedge::QueueClose<ThrowingMove>>);
    static_assert(std::is_nothrow_move_constructible_v<inferedge::QueueClose<CompletionItem>>);
    static_assert(std::is_nothrow_move_constructible_v<inferedge::QueueClose<int>>);
    static_assert(!std::is_copy_constructible_v<inferedge::QueueClose<ThrowingMove>>);
    static_assert(!std::is_copy_assignable_v<inferedge::QueueClose<ThrowingMove>>);
    static_assert(!std::is_move_assignable_v<inferedge::QueueClose<ThrowingMove>>);
    static_assert(std::is_nothrow_swappable_v<std::deque<ThrowingMove>>);
}

TEST(BoundedFifoQueueTest, CancelDoesNotMoveElements) {
    reset_throwing_move();

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
    EXPECT_EQ(ThrowingMove::destroyed_count, 0);
    EXPECT_EQ(queue.size(), 3U);

    const int moves_after_enqueue = ThrowingMove::moves;
    ThrowingMove::throw_after = moves_after_enqueue;
    {
        const inferedge::QueueClose<ThrowingMove> closed = queue.close_cancel();
        EXPECT_EQ(ThrowingMove::moves, moves_after_enqueue);
        EXPECT_EQ(ThrowingMove::destroyed_count, 0);
        EXPECT_TRUE(closed.changed);
        EXPECT_EQ(closed.mode, inferedge::QueueCloseMode::kCancel);
        EXPECT_TRUE(queue.is_closed());
        EXPECT_EQ(queue.close_mode(), inferedge::QueueCloseMode::kCancel);
        EXPECT_EQ(queue.size(), 0U);
        ASSERT_NE(closed.canceled, nullptr);
        ASSERT_EQ(closed.canceled->size(), 3U);
        EXPECT_EQ((*closed.canceled)[0].id, 1);
        EXPECT_EQ((*closed.canceled)[1].id, 2);
        EXPECT_EQ((*closed.canceled)[2].id, 3);
        EXPECT_EQ(queue.wait_pop().result, inferedge::QueuePopResult::kClosed);
    }

    std::vector<int> destroyed(ThrowingMove::destroyed_ids,
                               ThrowingMove::destroyed_ids + ThrowingMove::destroyed_count);
    std::sort(destroyed.begin(), destroyed.end());
    ASSERT_EQ(destroyed.size(), 3U);
    EXPECT_EQ(destroyed[0], 1);
    EXPECT_EQ(destroyed[1], 2);
    EXPECT_EQ(destroyed[2], 3);
}

TEST(BoundedFifoQueueTest, CancelHandleTransfersWithoutMovingElements) {
    reset_throwing_move();
    inferedge::BoundedFifoQueue<ThrowingMove> queue(3);
    {
        ThrowingMove first{1};
        ThrowingMove second{2};
        ThrowingMove third{3};
        EXPECT_EQ(queue.try_push(std::move(first)).result, inferedge::QueuePushResult::kAccepted);
        EXPECT_EQ(queue.try_push(std::move(second)).result, inferedge::QueuePushResult::kAccepted);
        EXPECT_EQ(queue.try_push(std::move(third)).result, inferedge::QueuePushResult::kAccepted);
    }

    const int moves_after_enqueue = ThrowingMove::moves;
    ThrowingMove::throw_after = moves_after_enqueue;
    inferedge::QueueClose<ThrowingMove> closed = queue.close_cancel();
    ASSERT_NE(closed.canceled, nullptr);
    const std::deque<ThrowingMove>* const storage = closed.canceled.get();
    const ThrowingMove* const first_element = &(*closed.canceled)[0];
    EXPECT_EQ(ThrowingMove::moves, moves_after_enqueue);
    EXPECT_EQ(ThrowingMove::destroyed_count, 0);

    inferedge::QueueClose<ThrowingMove> delivered = std::move(closed);
    EXPECT_EQ(delivered.canceled.get(), storage);
    EXPECT_EQ(&(*delivered.canceled)[0], first_element);
    EXPECT_EQ(closed.canceled, nullptr);
    EXPECT_EQ(ThrowingMove::moves, moves_after_enqueue);
    EXPECT_EQ(ThrowingMove::destroyed_count, 0);
    ASSERT_EQ(delivered.canceled->size(), 3U);
    EXPECT_EQ((*delivered.canceled)[0].id, 1);
    EXPECT_EQ((*delivered.canceled)[1].id, 2);
    EXPECT_EQ((*delivered.canceled)[2].id, 3);

    delivered.canceled.reset();
    std::vector<int> destroyed(ThrowingMove::destroyed_ids,
                               ThrowingMove::destroyed_ids + ThrowingMove::destroyed_count);
    std::sort(destroyed.begin(), destroyed.end());
    ASSERT_EQ(destroyed.size(), 3U);
    EXPECT_EQ(destroyed[0], 1);
    EXPECT_EQ(destroyed[1], 2);
    EXPECT_EQ(destroyed[2], 3);
}

TEST(BoundedFifoQueueTest, EmptyCancelOwnsAnEmptyDeque) {
    reset_throwing_move();
    inferedge::BoundedFifoQueue<ThrowingMove> queue(1);
    inferedge::QueueClose<ThrowingMove> closed = queue.close_cancel();
    EXPECT_TRUE(closed.changed);
    EXPECT_EQ(closed.mode, inferedge::QueueCloseMode::kCancel);
    ASSERT_NE(closed.canceled, nullptr);
    EXPECT_TRUE(closed.canceled->empty());
    EXPECT_EQ(ThrowingMove::moves, 0);
    EXPECT_EQ(ThrowingMove::destroyed_count, 0);
    EXPECT_TRUE(queue.is_closed());
    EXPECT_EQ(queue.size(), 0U);

    const std::deque<ThrowingMove>* const storage = closed.canceled.get();
    inferedge::QueueClose<ThrowingMove> delivered = std::move(closed);
    EXPECT_EQ(delivered.canceled.get(), storage);
    EXPECT_EQ(closed.canceled, nullptr);
    EXPECT_EQ(ThrowingMove::moves, 0);
    EXPECT_EQ(ThrowingMove::destroyed_count, 0);

    const inferedge::QueueClose<ThrowingMove> repeated = queue.close_cancel();
    EXPECT_FALSE(repeated.changed);
    EXPECT_EQ(repeated.mode, inferedge::QueueCloseMode::kCancel);
    EXPECT_EQ(repeated.canceled, nullptr);

    const inferedge::QueueClose<ThrowingMove> crossed = queue.close_drain();
    EXPECT_FALSE(crossed.changed);
    EXPECT_EQ(crossed.mode, inferedge::QueueCloseMode::kCancel);
    EXPECT_EQ(crossed.canceled, nullptr);
    EXPECT_EQ(queue.wait_pop().result, inferedge::QueuePopResult::kClosed);
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
    ASSERT_NE(canceled.canceled, nullptr);
    const std::deque<CompletionItem>* const storage = canceled.canceled.get();
    inferedge::QueueClose<CompletionItem> delivered = std::move(canceled);
    ASSERT_EQ(delivered.canceled.get(), storage);
    EXPECT_EQ(canceled.canceled, nullptr);
    ASSERT_EQ(delivered.canceled->size(), 3U);
    EXPECT_TRUE(queue.is_closed());
    EXPECT_EQ(queue.size(), 0U);
    for (std::size_t index = 0; index < delivered.canceled->size(); ++index) {
        const int id = static_cast<int>(index) + 1;
        CompletionItem& item = (*delivered.canceled)[index];
        EXPECT_EQ(item.id, id);
        item.completion.set_value(id);
        try {
            item.completion.set_value(id);
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
    EXPECT_EQ(drained.canceled, nullptr);
    if (drained.mode == inferedge::QueueCloseMode::kDrain) {
        EXPECT_EQ(canceled.canceled, nullptr);
        EXPECT_EQ(take_ready(queue.wait_pop()), 1);
        EXPECT_EQ(take_ready(queue.wait_pop()), 2);
        EXPECT_EQ(take_ready(queue.wait_pop()), 3);
    } else {
        ASSERT_NE(canceled.canceled, nullptr);
        ASSERT_EQ(canceled.canceled->size(), 3U);
        EXPECT_EQ((*canceled.canceled)[0], 1);
        EXPECT_EQ((*canceled.canceled)[1], 2);
        EXPECT_EQ((*canceled.canceled)[2], 3);
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

TEST(BoundedFifoQueueTest, TryPopOnEmptyQueueReturnsEmpty) {
    inferedge::BoundedFifoQueue<int> queue(2);
    const auto started = std::chrono::steady_clock::now();
    const inferedge::QueueTake<int> taken = queue.try_pop();
    const auto elapsed = std::chrono::steady_clock::now() - started;
    EXPECT_LT(elapsed, std::chrono::seconds(1));
    EXPECT_EQ(taken.result, inferedge::QueueTakeResult::kEmpty);
    EXPECT_FALSE(taken.value.has_value());
    EXPECT_FALSE(queue.is_closed());
    EXPECT_EQ(queue.size(), 0U);
    EXPECT_EQ(queue.capacity(), 2U);
}

TEST(BoundedFifoQueueTest, TryPopRemovesMoveOnlyValuesInFifoOrder) {
    inferedge::BoundedFifoQueue<std::unique_ptr<int>> queue(3);
    EXPECT_EQ(queue.try_push(std::make_unique<int>(1)).result,
              inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(std::make_unique<int>(2)).result,
              inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(std::make_unique<int>(3)).result,
              inferedge::QueuePushResult::kAccepted);

    const inferedge::QueueTake<std::unique_ptr<int>> first = queue.try_pop();
    EXPECT_EQ(first.result, inferedge::QueueTakeResult::kReady);
    ASSERT_TRUE(first.value.has_value());
    ASSERT_NE(*first.value, nullptr);
    EXPECT_EQ(**first.value, 1);
    EXPECT_EQ(queue.size(), 2U);

    const inferedge::QueueTake<std::unique_ptr<int>> second = queue.try_pop();
    EXPECT_EQ(second.result, inferedge::QueueTakeResult::kReady);
    ASSERT_TRUE(second.value.has_value());
    ASSERT_NE(*second.value, nullptr);
    EXPECT_EQ(**second.value, 2);
    EXPECT_EQ(queue.size(), 1U);

    const inferedge::QueueTake<std::unique_ptr<int>> third = queue.try_pop();
    EXPECT_EQ(third.result, inferedge::QueueTakeResult::kReady);
    ASSERT_TRUE(third.value.has_value());
    ASSERT_NE(*third.value, nullptr);
    EXPECT_EQ(**third.value, 3);
    EXPECT_EQ(queue.size(), 0U);

    const inferedge::QueueTake<std::unique_ptr<int>> empty = queue.try_pop();
    EXPECT_EQ(empty.result, inferedge::QueueTakeResult::kEmpty);
    EXPECT_FALSE(empty.value.has_value());
    EXPECT_EQ(queue.capacity(), 3U);
    EXPECT_FALSE(queue.is_closed());
}

TEST(BoundedFifoQueueTest, TryPopAfterDrainReturnsReadyUntilEmpty) {
    inferedge::BoundedFifoQueue<int> queue(3);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(3).result, inferedge::QueuePushResult::kAccepted);

    const inferedge::QueueClose<int> closed = queue.close_drain();
    EXPECT_TRUE(closed.changed);
    EXPECT_EQ(closed.mode, inferedge::QueueCloseMode::kDrain);
    EXPECT_EQ(closed.canceled, nullptr);
    EXPECT_EQ(queue.size(), 3U);

    const inferedge::QueueTake<int> first = queue.try_pop();
    EXPECT_EQ(first.result, inferedge::QueueTakeResult::kReady);
    ASSERT_TRUE(first.value.has_value());
    EXPECT_EQ(*first.value, 1);
    const inferedge::QueueTake<int> second = queue.try_pop();
    EXPECT_EQ(second.result, inferedge::QueueTakeResult::kReady);
    ASSERT_TRUE(second.value.has_value());
    EXPECT_EQ(*second.value, 2);
    const inferedge::QueueTake<int> third = queue.try_pop();
    EXPECT_EQ(third.result, inferedge::QueueTakeResult::kReady);
    ASSERT_TRUE(third.value.has_value());
    EXPECT_EQ(*third.value, 3);

    const inferedge::QueueTake<int> done = queue.try_pop();
    EXPECT_EQ(done.result, inferedge::QueueTakeResult::kClosed);
    EXPECT_FALSE(done.value.has_value());
    EXPECT_EQ(queue.size(), 0U);
}

TEST(BoundedFifoQueueTest, TryPopAfterCancelReturnsClosed) {
    inferedge::BoundedFifoQueue<std::unique_ptr<int>> queue(3);
    EXPECT_EQ(queue.try_push(std::make_unique<int>(1)).result,
              inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(std::make_unique<int>(2)).result,
              inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(std::make_unique<int>(3)).result,
              inferedge::QueuePushResult::kAccepted);

    inferedge::QueueClose<std::unique_ptr<int>> closed = queue.close_cancel();
    EXPECT_TRUE(closed.changed);
    ASSERT_NE(closed.canceled, nullptr);
    ASSERT_EQ(closed.canceled->size(), 3U);
    ASSERT_NE((*closed.canceled)[0], nullptr);
    ASSERT_NE((*closed.canceled)[1], nullptr);
    ASSERT_NE((*closed.canceled)[2], nullptr);
    EXPECT_EQ(*(*closed.canceled)[0], 1);
    EXPECT_EQ(*(*closed.canceled)[1], 2);
    EXPECT_EQ(*(*closed.canceled)[2], 3);

    const inferedge::QueueTake<std::unique_ptr<int>> taken = queue.try_pop();
    EXPECT_EQ(taken.result, inferedge::QueueTakeResult::kClosed);
    EXPECT_FALSE(taken.value.has_value());
    EXPECT_EQ(queue.size(), 0U);
    EXPECT_EQ(closed.canceled->size(), 3U);
}

TEST(BoundedFifoQueueTest, WaitPopUntilReturnsQueuedItemBeforeDeadline) {
    inferedge::BoundedFifoQueue<int> queue(1);
    EXPECT_EQ(queue.try_push(7).result, inferedge::QueuePushResult::kAccepted);
    const inferedge::QueueWait<int> waited = queue.wait_pop_until(future_deadline());
    EXPECT_EQ(waited.result, inferedge::QueueWaitResult::kReady);
    ASSERT_TRUE(waited.value.has_value());
    EXPECT_EQ(*waited.value, 7);
    EXPECT_EQ(queue.size(), 0U);
    EXPECT_FALSE(queue.is_closed());
}

TEST(BoundedFifoQueueTest, WaitPopUntilPrefersQueuedItemOverExpiredDeadline) {
    inferedge::BoundedFifoQueue<int> queue(1);
    EXPECT_EQ(queue.try_push(7).result, inferedge::QueuePushResult::kAccepted);
    const auto started = std::chrono::steady_clock::now();
    const inferedge::QueueWait<int> waited = queue.wait_pop_until(expired_deadline());
    const auto elapsed = std::chrono::steady_clock::now() - started;
    EXPECT_LT(elapsed, std::chrono::seconds(1));
    EXPECT_EQ(waited.result, inferedge::QueueWaitResult::kReady);
    ASSERT_TRUE(waited.value.has_value());
    EXPECT_EQ(*waited.value, 7);
    EXPECT_EQ(queue.size(), 0U);
}

TEST(BoundedFifoQueueTest, WaitPopUntilTimesOutWhenOpenAndEmpty) {
    inferedge::BoundedFifoQueue<int> queue(1);
    const auto started = std::chrono::steady_clock::now();
    const inferedge::QueueWait<int> waited = queue.wait_pop_until(expired_deadline());
    const auto elapsed = std::chrono::steady_clock::now() - started;
    EXPECT_LT(elapsed, std::chrono::seconds(1));
    EXPECT_EQ(waited.result, inferedge::QueueWaitResult::kTimeout);
    EXPECT_FALSE(waited.value.has_value());
    EXPECT_FALSE(queue.is_closed());
    EXPECT_EQ(queue.size(), 0U);
    EXPECT_EQ(queue.capacity(), 1U);
}

TEST(BoundedFifoQueueTest, WaitPopUntilAfterDrainReturnsReadyThenClosed) {
    inferedge::BoundedFifoQueue<int> queue(3);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(3).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_TRUE(queue.close_drain().changed);

    const inferedge::QueueWait<int> first = queue.wait_pop_until(future_deadline());
    EXPECT_EQ(first.result, inferedge::QueueWaitResult::kReady);
    ASSERT_TRUE(first.value.has_value());
    EXPECT_EQ(*first.value, 1);

    const inferedge::QueueWait<int> second = queue.wait_pop_until(expired_deadline());
    EXPECT_EQ(second.result, inferedge::QueueWaitResult::kReady);
    ASSERT_TRUE(second.value.has_value());
    EXPECT_EQ(*second.value, 2);

    const inferedge::QueueWait<int> third = queue.wait_pop_until(future_deadline());
    EXPECT_EQ(third.result, inferedge::QueueWaitResult::kReady);
    ASSERT_TRUE(third.value.has_value());
    EXPECT_EQ(*third.value, 3);

    const inferedge::QueueWait<int> closed_future = queue.wait_pop_until(future_deadline());
    EXPECT_EQ(closed_future.result, inferedge::QueueWaitResult::kClosed);
    EXPECT_FALSE(closed_future.value.has_value());
    const auto started = std::chrono::steady_clock::now();
    const inferedge::QueueWait<int> closed_expired = queue.wait_pop_until(expired_deadline());
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(1));
    EXPECT_EQ(closed_expired.result, inferedge::QueueWaitResult::kClosed);
    EXPECT_FALSE(closed_expired.value.has_value());
}

TEST(BoundedFifoQueueTest, WaitPopUntilAfterCancelReturnsClosed) {
    inferedge::BoundedFifoQueue<int> queue(3);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(3).result, inferedge::QueuePushResult::kAccepted);

    inferedge::QueueClose<int> closed = queue.close_cancel();
    ASSERT_NE(closed.canceled, nullptr);
    ASSERT_EQ(closed.canceled->size(), 3U);

    const inferedge::QueueWait<int> future = queue.wait_pop_until(future_deadline());
    EXPECT_EQ(future.result, inferedge::QueueWaitResult::kClosed);
    EXPECT_FALSE(future.value.has_value());
    const inferedge::QueueWait<int> expired = queue.wait_pop_until(expired_deadline());
    EXPECT_EQ(expired.result, inferedge::QueueWaitResult::kClosed);
    EXPECT_FALSE(expired.value.has_value());
    EXPECT_EQ((*closed.canceled)[0], 1);
    EXPECT_EQ((*closed.canceled)[1], 2);
    EXPECT_EQ((*closed.canceled)[2], 3);
    EXPECT_EQ(queue.size(), 0U);
}

TEST(BoundedFifoQueueTest, WaitingTimedConsumerWakesOnPush) {
    inferedge::BoundedFifoQueue<int> queue(1);
    std::promise<void> started;
    std::promise<inferedge::QueueWait<int>> finished;
    JoinThread consumer([&] {
        started.set_value();
        finished.set_value(queue.wait_pop_until(future_deadline()));
    });
    started.get_future().wait();

    EXPECT_EQ(queue.try_push(7).result, inferedge::QueuePushResult::kAccepted);
    std::future<inferedge::QueueWait<int>> future = finished.get_future();
    const inferedge::QueueWait<int> waited = finish_consumer(future, queue);
    EXPECT_EQ(waited.result, inferedge::QueueWaitResult::kReady);
    ASSERT_TRUE(waited.value.has_value());
    EXPECT_EQ(*waited.value, 7);
    EXPECT_EQ(queue.size(), 0U);
}

TEST(BoundedFifoQueueTest, WaitingTimedConsumerWakesOnDrain) {
    inferedge::BoundedFifoQueue<int> queue(1);
    std::promise<void> started;
    std::promise<inferedge::QueueWait<int>> finished;
    JoinThread consumer([&] {
        started.set_value();
        finished.set_value(queue.wait_pop_until(future_deadline()));
    });
    started.get_future().wait();

    const inferedge::QueueClose<int> closed = queue.close_drain();
    EXPECT_TRUE(closed.changed);
    EXPECT_EQ(closed.canceled, nullptr);
    std::future<inferedge::QueueWait<int>> future = finished.get_future();
    const inferedge::QueueWait<int> waited = finish_consumer(future, queue);
    EXPECT_EQ(waited.result, inferedge::QueueWaitResult::kClosed);
    EXPECT_FALSE(waited.value.has_value());
}

TEST(BoundedFifoQueueTest, WaitingTimedConsumerWakesOnCancel) {
    inferedge::BoundedFifoQueue<int> queue(1);
    std::promise<void> started;
    std::promise<inferedge::QueueWait<int>> finished;
    JoinThread consumer([&] {
        started.set_value();
        finished.set_value(queue.wait_pop_until(future_deadline()));
    });
    started.get_future().wait();

    inferedge::QueueClose<int> closed = queue.close_cancel();
    EXPECT_TRUE(closed.changed);
    ASSERT_NE(closed.canceled, nullptr);
    EXPECT_TRUE(closed.canceled->empty());
    std::future<inferedge::QueueWait<int>> future = finished.get_future();
    const inferedge::QueueWait<int> waited = finish_consumer(future, queue);
    EXPECT_EQ(waited.result, inferedge::QueueWaitResult::kClosed);
    EXPECT_FALSE(waited.value.has_value());
    EXPECT_TRUE(closed.canceled->empty());
}

TEST(BoundedFifoQueueTest, TimeoutVersusPushKeepsSingleOwnership) {
    inferedge::BoundedFifoQueue<int> queue(1);
    std::atomic<int> outcome{0};
    {
        JoinThread consumer([&] {
            const inferedge::QueueWait<int> waited =
                queue.wait_pop_until(std::chrono::steady_clock::now());
            if (waited.result == inferedge::QueueWaitResult::kReady && waited.value.has_value() &&
                *waited.value == 7) {
                outcome.store(1, std::memory_order_relaxed);
            } else if (waited.result == inferedge::QueueWaitResult::kTimeout &&
                       !waited.value.has_value()) {
                outcome.store(2, std::memory_order_relaxed);
            } else {
                outcome.store(3, std::memory_order_relaxed);
            }
        });
        JoinThread producer(
            [&] { EXPECT_EQ(queue.try_push(7).result, inferedge::QueuePushResult::kAccepted); });
    }

    const inferedge::QueueTake<int> leftover = queue.try_pop();
    const int observed = outcome.load(std::memory_order_relaxed);
    if (observed == 1) {
        EXPECT_EQ(leftover.result, inferedge::QueueTakeResult::kEmpty);
        EXPECT_FALSE(leftover.value.has_value());
    } else if (observed == 2) {
        EXPECT_EQ(leftover.result, inferedge::QueueTakeResult::kReady);
        ASSERT_TRUE(leftover.value.has_value());
        EXPECT_EQ(*leftover.value, 7);
    } else {
        ADD_FAILURE() << "timed pop returned an unexpected result";
    }
    EXPECT_EQ(queue.try_pop().result, inferedge::QueueTakeResult::kEmpty);
    EXPECT_FALSE(queue.is_closed());
    EXPECT_EQ(queue.size(), 0U);
}

TEST(BoundedFifoQueueTest, MixedConsumersTakeEachValueOnce) {
    constexpr int kItems = 40;
    inferedge::BoundedFifoQueue<int> queue(static_cast<std::size_t>(kItems));
    for (int value = 0; value < kItems; ++value) {
        EXPECT_EQ(queue.try_push(value).result, inferedge::QueuePushResult::kAccepted);
    }

    std::vector<int> taken;
    taken.reserve(static_cast<std::size_t>(kItems));
    for (int expected = 0; expected < 5; ++expected) {
        const inferedge::QueueTake<int> item = queue.try_pop();
        EXPECT_EQ(item.result, inferedge::QueueTakeResult::kReady);
        ASSERT_TRUE(item.value.has_value());
        EXPECT_EQ(*item.value, expected);
        taken.push_back(*item.value);
    }

    std::mutex popped_mutex;
    std::vector<int> popped;
    std::vector<JoinThread> consumers;
    consumers.reserve(4);
    for (int consumer = 0; consumer < 2; ++consumer) {
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
    for (int consumer = 0; consumer < 2; ++consumer) {
        consumers.emplace_back([&] {
            for (;;) {
                const inferedge::QueueWait<int> item = queue.wait_pop_until(future_deadline());
                if (item.result == inferedge::QueueWaitResult::kClosed) {
                    return;
                }
                if (item.result == inferedge::QueueWaitResult::kTimeout) {
                    continue;
                }
                if (!item.value.has_value()) {
                    ADD_FAILURE() << "ready timed pop has no value";
                    return;
                }
                std::lock_guard<std::mutex> lock(popped_mutex);
                popped.push_back(*item.value);
            }
        });
    }

    EXPECT_TRUE(queue.close_drain().changed);
    consumers.clear();
    taken.insert(taken.end(), popped.begin(), popped.end());
    std::sort(taken.begin(), taken.end());
    EXPECT_EQ(taken.size(), static_cast<std::size_t>(kItems));
    EXPECT_EQ(std::unique(taken.begin(), taken.end()), taken.end());
    EXPECT_EQ(taken.front(), 0);
    EXPECT_EQ(taken.back(), kItems - 1);
}

TEST(BoundedFifoQueueTest, TryPopLeavesFrontWhenMoveThrows) {
    reset_throwing_move();
    inferedge::BoundedFifoQueue<ThrowingMove> queue(2);
    {
        ThrowingMove first{1};
        ThrowingMove second{2};
        EXPECT_EQ(queue.try_push(std::move(first)).result, inferedge::QueuePushResult::kAccepted);
        EXPECT_EQ(queue.try_push(std::move(second)).result, inferedge::QueuePushResult::kAccepted);
    }

    ThrowingMove::throw_after = ThrowingMove::moves;
    EXPECT_THROW(static_cast<void>(queue.try_pop()), std::runtime_error);
    EXPECT_EQ(queue.size(), 2U);
    EXPECT_FALSE(queue.is_closed());

    ThrowingMove::throw_after = -1;
    const inferedge::QueueTake<ThrowingMove> first = queue.try_pop();
    EXPECT_EQ(first.result, inferedge::QueueTakeResult::kReady);
    ASSERT_TRUE(first.value.has_value());
    EXPECT_EQ(first.value->id, 1);
    const inferedge::QueueTake<ThrowingMove> second = queue.try_pop();
    EXPECT_EQ(second.result, inferedge::QueueTakeResult::kReady);
    ASSERT_TRUE(second.value.has_value());
    EXPECT_EQ(second.value->id, 2);
    EXPECT_EQ(queue.try_pop().result, inferedge::QueueTakeResult::kEmpty);
    EXPECT_EQ(queue.size(), 0U);
}

TEST(BoundedFifoQueueTest, WaitPopUntilLeavesFrontWhenMoveThrows) {
    reset_throwing_move();
    inferedge::BoundedFifoQueue<ThrowingMove> queue(2);
    {
        ThrowingMove first{4};
        ThrowingMove second{5};
        EXPECT_EQ(queue.try_push(std::move(first)).result, inferedge::QueuePushResult::kAccepted);
        EXPECT_EQ(queue.try_push(std::move(second)).result, inferedge::QueuePushResult::kAccepted);
    }

    ThrowingMove::throw_after = ThrowingMove::moves;
    EXPECT_THROW(static_cast<void>(queue.wait_pop_until(expired_deadline())), std::runtime_error);
    EXPECT_EQ(queue.size(), 2U);
    EXPECT_FALSE(queue.is_closed());

    ThrowingMove::throw_after = -1;
    const inferedge::QueueWait<ThrowingMove> first = queue.wait_pop_until(expired_deadline());
    EXPECT_EQ(first.result, inferedge::QueueWaitResult::kReady);
    ASSERT_TRUE(first.value.has_value());
    EXPECT_EQ(first.value->id, 4);
    const inferedge::QueueWait<ThrowingMove> second = queue.wait_pop_until(future_deadline());
    EXPECT_EQ(second.result, inferedge::QueueWaitResult::kReady);
    ASSERT_TRUE(second.value.has_value());
    EXPECT_EQ(second.value->id, 5);
    EXPECT_EQ(queue.size(), 0U);
}

TEST(BoundedFifoQueueTest, NewPopsObserveDrainThatWonOverCancel) {
    inferedge::BoundedFifoQueue<int> queue(2);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.close_drain().mode, inferedge::QueueCloseMode::kDrain);

    const inferedge::QueueClose<int> later = queue.close_cancel();
    EXPECT_FALSE(later.changed);
    EXPECT_EQ(later.mode, inferedge::QueueCloseMode::kDrain);
    EXPECT_EQ(later.canceled, nullptr);
    EXPECT_EQ(queue.close_mode(), inferedge::QueueCloseMode::kDrain);

    const inferedge::QueueTake<int> first = queue.try_pop();
    EXPECT_EQ(first.result, inferedge::QueueTakeResult::kReady);
    ASSERT_TRUE(first.value.has_value());
    EXPECT_EQ(*first.value, 1);
    const inferedge::QueueWait<int> second = queue.wait_pop_until(expired_deadline());
    EXPECT_EQ(second.result, inferedge::QueueWaitResult::kReady);
    ASSERT_TRUE(second.value.has_value());
    EXPECT_EQ(*second.value, 2);
    EXPECT_EQ(queue.try_pop().result, inferedge::QueueTakeResult::kClosed);
    const inferedge::QueueWait<int> closed = queue.wait_pop_until(future_deadline());
    EXPECT_EQ(closed.result, inferedge::QueueWaitResult::kClosed);
    EXPECT_FALSE(closed.value.has_value());
}

TEST(BoundedFifoQueueTest, NewPopsObserveCancelThatWonOverDrain) {
    inferedge::BoundedFifoQueue<int> queue(2);
    EXPECT_EQ(queue.try_push(1).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(2).result, inferedge::QueuePushResult::kAccepted);
    inferedge::QueueClose<int> canceled = queue.close_cancel();
    const inferedge::QueueClose<int> later = queue.close_drain();
    EXPECT_FALSE(later.changed);
    EXPECT_EQ(later.mode, inferedge::QueueCloseMode::kCancel);
    EXPECT_EQ(later.canceled, nullptr);
    EXPECT_EQ(queue.close_mode(), inferedge::QueueCloseMode::kCancel);
    ASSERT_NE(canceled.canceled, nullptr);
    ASSERT_EQ(canceled.canceled->size(), 2U);

    const inferedge::QueueTake<int> taken = queue.try_pop();
    EXPECT_EQ(taken.result, inferedge::QueueTakeResult::kClosed);
    EXPECT_FALSE(taken.value.has_value());
    const inferedge::QueueWait<int> waited = queue.wait_pop_until(future_deadline());
    EXPECT_EQ(waited.result, inferedge::QueueWaitResult::kClosed);
    EXPECT_FALSE(waited.value.has_value());
    const inferedge::QueueWait<int> expired = queue.wait_pop_until(expired_deadline());
    EXPECT_EQ(expired.result, inferedge::QueueWaitResult::kClosed);
    EXPECT_FALSE(expired.value.has_value());
    EXPECT_EQ((*canceled.canceled)[0], 1);
    EXPECT_EQ((*canceled.canceled)[1], 2);
}

TEST(BoundedFifoQueueTest, NewPopsTransferMoveOnlyCompletion) {
    inferedge::BoundedFifoQueue<CompletionItem> queue(2);
    CompletionItem first{11};
    CompletionItem second{22};
    std::future<int> first_future = first.completion.get_future();
    std::future<int> second_future = second.completion.get_future();
    EXPECT_EQ(queue.try_push(std::move(first)).result, inferedge::QueuePushResult::kAccepted);
    EXPECT_EQ(queue.try_push(std::move(second)).result, inferedge::QueuePushResult::kAccepted);

    inferedge::QueueTake<CompletionItem> taken = queue.try_pop();
    EXPECT_EQ(taken.result, inferedge::QueueTakeResult::kReady);
    ASSERT_TRUE(taken.value.has_value());
    EXPECT_EQ(taken.value->id, 11);
    taken.value->completion.set_value(11);

    inferedge::QueueWait<CompletionItem> waited = queue.wait_pop_until(expired_deadline());
    EXPECT_EQ(waited.result, inferedge::QueueWaitResult::kReady);
    ASSERT_TRUE(waited.value.has_value());
    EXPECT_EQ(waited.value->id, 22);
    waited.value->completion.set_value(22);

    EXPECT_EQ(first_future.get(), 11);
    EXPECT_EQ(second_future.get(), 22);
    EXPECT_THROW(taken.value->completion.set_value(11), std::future_error);
    EXPECT_THROW(waited.value->completion.set_value(22), std::future_error);
    EXPECT_EQ(queue.size(), 0U);
    EXPECT_EQ(queue.try_pop().result, inferedge::QueueTakeResult::kEmpty);
}
