#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace inferedge {

// Result of one nonblocking admission attempt.
enum class QueuePushResult {
    kAccepted,
    kFull,
    kClosed,
};

// Result of one waiting removal. kClosed means admission is closed and no
// waiting item remains. It is not an empty-value sentinel.
enum class QueuePopResult {
    kReady,
    kClosed,
};

// The first successful close selects one mode for the rest of the queue lifetime.
enum class QueueCloseMode {
    kDrain,
    kCancel,
};

// On kAccepted, rejected is empty and the queue owns the item.
// On kFull or kClosed, rejected owns the same item and the queue does not.
template <typename T> struct QueuePush {
    QueuePushResult result{QueuePushResult::kFull};
    std::optional<T> rejected;
};

// On kReady, value owns the removed item. On kClosed, value is empty.
template <typename T> struct QueuePop {
    QueuePopResult result{QueuePopResult::kClosed};
    std::optional<T> value;
};

// changed is true only for the call that performs the open-to-closed transition.
// canceled owns items removed by a winning close_cancel, in FIFO order.
// Ownership moves by swapping deques, not by move-constructing each T.
// A losing close, a repeated close, and close_drain leave canceled empty.
template <typename T> struct QueueClose {
    QueueCloseMode mode{QueueCloseMode::kDrain};
    bool changed{false};
    std::deque<T> canceled;
};

// A bounded FIFO queue. Capacity counts items waiting in the queue. A popped
// item no longer occupies a slot. try_push never blocks: a full queue returns
// kFull and a closed queue returns kClosed.
//
// T must be move-constructible and destructible. It does not need to be
// copyable or move-assignable. try_push and wait_pop move-construct T while
// the queue holds its mutex, and those moves must not call back into the
// same queue.
//
// try_push and wait_pop provide the basic exception guarantee when T's move
// constructor can throw. They are not strongly exception-safe for every
// movable T, and they are not noexcept. If inserting into the deque throws,
// try_push leaves the queue unchanged, but its by-value argument may already
// be moved-from. If extracting the front element throws, wait_pop leaves that
// moved-from element at the front and does not pop it.
//
// close_drain does not move or destroy T. If the mutex lock throws, admission
// stays open. Otherwise the queue closes in drain mode and the waiting items
// remain in place. close_cancel also does not move or destroy T. Lock failure
// leaves the queue open. The winning call swaps the waiting deque into
// QueueClose::canceled, then closes admission and wakes consumers.
// std::deque::swap and the move of that deque are noexcept for std::allocator
// and do not invoke T's move constructor, so a throwing T move cannot destroy
// or drop an accepted item during cancellation. A later close returns the
// winning mode and an empty deque. close_drain and close_cancel are not
// noexcept, because locking the mutex can throw.
//
// capacity() only reads the fixed capacity and is noexcept. size(),
// is_closed(), and close_mode() lock the mutex and are not noexcept.
// T's destructor must not throw.
//
// The queue does not own producer or consumer threads. Before destroying it,
// the owner must close it, join every thread that calls into it, and only
// then destroy the queue. The destructor does not wake or join external
// threads. Destroying a queue that still holds items destroys those items.
template <typename T> class BoundedFifoQueue {
public:
    explicit BoundedFifoQueue(std::size_t capacity);

    BoundedFifoQueue(const BoundedFifoQueue&) = delete;
    BoundedFifoQueue& operator=(const BoundedFifoQueue&) = delete;
    BoundedFifoQueue(BoundedFifoQueue&&) = delete;
    BoundedFifoQueue& operator=(BoundedFifoQueue&&) = delete;
    ~BoundedFifoQueue() = default;

    [[nodiscard]] QueuePush<T> try_push(T item);
    [[nodiscard]] QueuePop<T> wait_pop();
    [[nodiscard]] QueueClose<T> close_drain();
    [[nodiscard]] QueueClose<T> close_cancel();

    // Synchronized snapshot of the number of waiting items.
    std::size_t size() const;
    // Fixed for the lifetime of the queue. This does not take the mutex.
    std::size_t capacity() const noexcept;
    bool is_closed() const;
    // Empty while the queue accepts pushes. After closure, the winning mode.
    std::optional<QueueCloseMode> close_mode() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<T> items_;
    const std::size_t capacity_;
    std::optional<QueueCloseMode> mode_;
};

template <typename T>
BoundedFifoQueue<T>::BoundedFifoQueue(std::size_t capacity) : capacity_{capacity} {
    if (capacity == 0) {
        throw std::invalid_argument("bounded queue capacity must be greater than zero");
    }
}

template <typename T> QueuePush<T> BoundedFifoQueue<T>::try_push(T item) {
    std::lock_guard<std::mutex> lock(mutex_);
    QueuePush<T> outcome;
    if (mode_.has_value()) {
        outcome.result = QueuePushResult::kClosed;
        outcome.rejected.emplace(std::move(item));
        return outcome;
    }
    if (items_.size() == capacity_) {
        outcome.result = QueuePushResult::kFull;
        outcome.rejected.emplace(std::move(item));
        return outcome;
    }
    items_.push_back(std::move(item));
    cv_.notify_one();
    outcome.result = QueuePushResult::kAccepted;
    return outcome;
}

template <typename T> QueuePop<T> BoundedFifoQueue<T>::wait_pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !items_.empty() || mode_.has_value(); });
    QueuePop<T> outcome;
    if (items_.empty()) {
        outcome.result = QueuePopResult::kClosed;
        return outcome;
    }
    outcome.result = QueuePopResult::kReady;
    outcome.value.emplace(std::move(items_.front()));
    items_.pop_front();
    return outcome;
}

template <typename T> QueueClose<T> BoundedFifoQueue<T>::close_drain() {
    std::lock_guard<std::mutex> lock(mutex_);
    QueueClose<T> outcome;
    if (mode_.has_value()) {
        outcome.mode = *mode_;
        return outcome;
    }
    mode_ = QueueCloseMode::kDrain;
    outcome.mode = QueueCloseMode::kDrain;
    outcome.changed = true;
    cv_.notify_all();
    return outcome;
}

template <typename T> QueueClose<T> BoundedFifoQueue<T>::close_cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    QueueClose<T> outcome;
    if (mode_.has_value()) {
        outcome.mode = *mode_;
        return outcome;
    }
    outcome.canceled.swap(items_);
    mode_ = QueueCloseMode::kCancel;
    outcome.mode = QueueCloseMode::kCancel;
    outcome.changed = true;
    cv_.notify_all();
    return outcome;
}

template <typename T> std::size_t BoundedFifoQueue<T>::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return items_.size();
}

template <typename T> std::size_t BoundedFifoQueue<T>::capacity() const noexcept {
    return capacity_;
}

template <typename T> bool BoundedFifoQueue<T>::is_closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_.has_value();
}

template <typename T> std::optional<QueueCloseMode> BoundedFifoQueue<T>::close_mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
}

} // namespace inferedge
