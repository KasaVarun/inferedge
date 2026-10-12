#pragma once

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
// The return moves of try_push and wait_pop are separate. When the returned
// optional owns a T, that return move-constructs T and provides only the basic
// guarantee above.
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
