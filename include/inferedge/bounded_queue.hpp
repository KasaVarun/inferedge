#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>
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

// Result of one nonblocking removal.
enum class QueueTakeResult {
    kReady,
    kEmpty,
    kClosed,
};

// Result of one deadline-based removal. The deadline uses steady_clock.
enum class QueueWaitResult {
    kReady,
    kTimeout,
    kClosed,
};

// On kReady, value owns the removed front item. On kEmpty and kClosed, value
// is empty. kEmpty means the queue is open and has nothing waiting.
template <typename T> struct QueueTake {
    QueueTakeResult result{QueueTakeResult::kClosed};
    std::optional<T> value;
};

// On kReady, value owns the removed front item. On kTimeout and kClosed, value
// is empty. kTimeout means the queue is open, empty, and the deadline passed.
template <typename T> struct QueueWait {
    QueueWaitResult result{QueueWaitResult::kClosed};
    std::optional<T> value;
};

// changed is true only for the call that performs the open-to-closed transition.
//
// canceled is null unless this call won close_cancel. A winning close_cancel
// owns the waiting items in FIFO order, including when that collection is
// empty. The deque is allocated before cancellation commits. QueueClose then
// carries it by std::unique_ptr. Moving a QueueClose moves that pointer: the
// move is noexcept, allocates nothing, and does not move or destroy T.
//
// QueueClose is move-only. Move assignment is deleted. A losing close, a
// repeated close, and close_drain leave canceled null.
template <typename T> struct QueueClose {
    QueueCloseMode mode{QueueCloseMode::kDrain};
    bool changed{false};
    std::unique_ptr<std::deque<T>> canceled;

    QueueClose() noexcept = default;
    QueueClose(const QueueClose&) = delete;
    QueueClose& operator=(const QueueClose&) = delete;
    QueueClose(QueueClose&&) noexcept = default;
    QueueClose& operator=(QueueClose&&) = delete;
};

// A bounded FIFO queue. Capacity counts items waiting in the queue. A popped
// item no longer occupies a slot. try_push never blocks: a full queue returns
// kFull and a closed queue returns kClosed.
//
// T must be move-constructible and destructible. It does not need to be
// copyable or move-assignable. try_push, wait_pop, try_pop, and
// wait_pop_until move-construct T while the queue holds its mutex, and those
// moves must not call back into the same queue.
//
// try_push, wait_pop, try_pop, and wait_pop_until provide the basic exception
// guarantee when T's move constructor can throw. They are not strongly
// exception-safe for every movable T, and they are not noexcept. If inserting
// into the deque throws, try_push leaves the queue unchanged, but its by-value
// argument may already be moved-from. If extracting the front element throws,
// wait_pop, try_pop, and wait_pop_until leave that moved-from element at the
// front and do not pop it. A nothrow move makes that transfer strong: the
// caller owns the item, or the queue still does.
//
// try_pop does not wait. An open empty queue returns kEmpty. A closed empty
// queue returns kClosed. A waiting item returns kReady, including after
// close_drain. kEmpty and kClosed do not move T and do not allocate.
//
// wait_pop_until waits on the same condition variable as wait_pop, using
// steady_clock. Spurious wakeups recheck the queue while the mutex is held.
// The mutex-held decision is: a waiting item is kReady, then a closed empty
// queue is kClosed, then an open empty queue whose deadline has passed is
// kTimeout. A queued item is not discarded because the deadline expired. A
// closed empty queue is not reported as a timeout, and the wait does not
// continue until the deadline after closure. kTimeout and kClosed do not move
// T and do not allocate. The mutex is not held after the function returns.
//
// close_drain does not move or destroy T. If the mutex lock throws, admission
// stays open. Otherwise the queue closes in drain mode and the waiting items
// remain in place. The returned handle is null. Moving that QueueClose is
// noexcept, so constructing the return value cannot throw after admission
// has closed.
//
// close_cancel does not move or destroy T. If the mutex lock throws, admission
// stays open. A queue that is already closed allocates nothing and returns the
// winning mode with a null handle. Otherwise the winning call allocates an
// empty deque while admission is still open and the waiting items are still
// owned by the queue. std::deque's default constructor can allocate, on both
// libstdc++ and libc++. If that allocation throws, the queue is unchanged.
//
// Cancellation commits when the waiting deque is swapped into that allocated
// deque and kCancel is stored. deque::swap is unconditionally noexcept and
// does not move, copy, or destroy T. The mode store is noexcept. The mutex is
// still held, so other callers cannot observe a partial transition.
//
// After that commit the function only move-assigns the unique_ptr, stores the
// result fields, calls notify_all, and move-constructs the return value. Those
// operations are noexcept. They do not allocate, and they do not move or
// destroy T. A later close returns the winning mode and a null handle.
//
// close_drain and close_cancel are not noexcept. Locking the mutex can throw,
// and the winning close_cancel can throw std::bad_alloc before it commits.
//
// The return moves of try_push, wait_pop, try_pop, and wait_pop_until are
// separate. When the returned optional owns a T, that return move-constructs T
// and provides only the basic guarantee above.
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
    // Nonblocking removal of the front item. Does not notify producers.
    [[nodiscard]] QueueTake<T> try_pop();
    // Deadline removal of the front item. deadline is a steady_clock time point.
    [[nodiscard]] QueueWait<T> wait_pop_until(std::chrono::steady_clock::time_point deadline);
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

template <typename T> QueueTake<T> BoundedFifoQueue<T>::try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    QueueTake<T> outcome;
    if (items_.empty()) {
        outcome.result = mode_.has_value() ? QueueTakeResult::kClosed : QueueTakeResult::kEmpty;
        return outcome;
    }
    outcome.result = QueueTakeResult::kReady;
    outcome.value.emplace(std::move(items_.front()));
    items_.pop_front();
    return outcome;
}

template <typename T>
QueueWait<T> BoundedFifoQueue<T>::wait_pop_until(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_until(lock, deadline, [this] { return !items_.empty() || mode_.has_value(); });
    QueueWait<T> outcome;
    if (!items_.empty()) {
        outcome.result = QueueWaitResult::kReady;
        outcome.value.emplace(std::move(items_.front()));
        items_.pop_front();
        return outcome;
    }
    if (mode_.has_value()) {
        outcome.result = QueueWaitResult::kClosed;
        return outcome;
    }
    outcome.result = QueueWaitResult::kTimeout;
    return outcome;
}

template <typename T> QueueClose<T> BoundedFifoQueue<T>::close_drain() {
    static_assert(std::is_nothrow_default_constructible_v<QueueClose<T>>);
    static_assert(std::is_nothrow_move_constructible_v<QueueClose<T>>);
    static_assert(!std::is_copy_constructible_v<QueueClose<T>>);
    static_assert(!std::is_move_assignable_v<QueueClose<T>>);

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
    static_assert(std::is_nothrow_default_constructible_v<QueueClose<T>>);
    static_assert(std::is_nothrow_move_constructible_v<QueueClose<T>>);
    static_assert(!std::is_copy_constructible_v<QueueClose<T>>);
    static_assert(!std::is_move_assignable_v<QueueClose<T>>);

    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_.has_value()) {
        QueueClose<T> outcome;
        outcome.mode = *mode_;
        return outcome;
    }

    // This allocation is the last operation that may throw. It runs before
    // items_ is disturbed and before admission closes.
    auto transferred = std::make_unique<std::deque<T>>();
    transferred->swap(items_);
    mode_ = QueueCloseMode::kCancel;

    QueueClose<T> outcome;
    outcome.mode = QueueCloseMode::kCancel;
    outcome.changed = true;
    outcome.canceled = std::move(transferred);
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
