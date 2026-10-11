#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

// A C library returns a handle and requires a matching destroy function.
// unique_ptr with NativeDeleter is the RAII shape of that contract.
// ONNX Runtime's C API uses the same shape: a create call returns an
// OrtSession*, and a release call destroys it. The official C++ wrapper
// Ort::Session already owns that pointer. Real model integration should
// use Ort::Session, Ort::Env, and Ort::Value rather than a project-local
// unique_ptr around every ONNX handle.

namespace {

constexpr int kHandleId = 7;

struct NativeHandle {
    int id = 0;
    int* live = nullptr;
    int* destroyed = nullptr;
};

NativeHandle* create_native_handle(int& live, int& destroyed) {
    auto* handle = new NativeHandle{};
    handle->id = kHandleId;
    handle->live = &live;
    handle->destroyed = &destroyed;
    ++live;
    return handle;
}

void destroy_native_handle(NativeHandle* handle) noexcept {
    if (handle == nullptr) {
        return;
    }
    if (handle->destroyed != nullptr) {
        ++(*handle->destroyed);
    }
    if (handle->live != nullptr) {
        --(*handle->live);
    }
    delete handle;
}

struct NativeDeleter {
    void operator()(NativeHandle* handle) const noexcept { destroy_native_handle(handle); }
};

using NativeOwner = std::unique_ptr<NativeHandle, NativeDeleter>;

NativeOwner acquire_native_handle(int& live, int& destroyed) {
    return NativeOwner{create_native_handle(live, destroyed)};
}

// partner is a shared_ptr, so two nodes can keep each other alive after the
// local owners are dropped. A weak_ptr in that member would not increment
// the count, and the node would die with its last external owner.
struct Node {
    int* live = nullptr;
    std::shared_ptr<Node> partner;

    explicit Node(int& live_count) : live{&live_count} { ++(*live); }

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    ~Node() {
        if (live != nullptr) {
            --(*live);
        }
    }
};

} // namespace

TEST(NativeResourceTest, AcquiresAHandle) {
    int live = 0;
    int destroyed = 0;
    const NativeOwner owner = acquire_native_handle(live, destroyed);
    ASSERT_NE(owner.get(), nullptr);
    EXPECT_EQ(owner.get()->id, kHandleId);
    EXPECT_EQ(live, 1);
    EXPECT_EQ(destroyed, 0);
}

TEST(NativeResourceTest, DestroysTheHandleAtScopeExit) {
    int live = 0;
    int destroyed = 0;
    {
        const NativeOwner owner = acquire_native_handle(live, destroyed);
        EXPECT_EQ(live, 1);
        EXPECT_EQ(destroyed, 0);
    }
    EXPECT_EQ(live, 0);
    EXPECT_EQ(destroyed, 1);
}

TEST(NativeResourceTest, MoveTransfersUniqueOwnership) {
    int live = 0;
    int destroyed = 0;
    NativeOwner source = acquire_native_handle(live, destroyed);
    NativeHandle* const raw = source.get();

    NativeOwner destination = std::move(source);

    EXPECT_EQ(source.get(), nullptr);
    EXPECT_EQ(destination.get(), raw);
    ASSERT_NE(destination.get(), nullptr);
    EXPECT_EQ(destination.get()->id, kHandleId);
    EXPECT_EQ(live, 1);
    EXPECT_EQ(destroyed, 0);
}

TEST(NativeResourceTest, DestroysTheHandleDuringExceptionUnwind) {
    int live = 0;
    int destroyed = 0;
    try {
        const NativeOwner owner = acquire_native_handle(live, destroyed);
        EXPECT_EQ(live, 1);
        throw std::runtime_error("expected");
    } catch (const std::runtime_error&) {
        EXPECT_EQ(live, 0);
        EXPECT_EQ(destroyed, 1);
    }
}

TEST(NativeResourceTest, ResetReleasesTheOwnedHandle) {
    int live = 0;
    int destroyed = 0;
    NativeOwner owner = acquire_native_handle(live, destroyed);
    owner.reset();
    EXPECT_EQ(owner.get(), nullptr);
    EXPECT_EQ(live, 0);
    EXPECT_EQ(destroyed, 1);
}

TEST(NativeResourceTest, ResetAdoptsAReplacementHandle) {
    int live = 0;
    int destroyed = 0;
    NativeOwner owner = acquire_native_handle(live, destroyed);
    NativeHandle* const replacement = create_native_handle(live, destroyed);
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(live, 2);

    owner.reset(replacement);

    EXPECT_EQ(owner.get(), replacement);
    EXPECT_EQ(owner.get()->id, kHandleId);
    EXPECT_EQ(live, 1);
    EXPECT_EQ(destroyed, 1);
}

TEST(NativeResourceTest, GetBorrowsWithoutReleasingOwnership) {
    int live = 0;
    int destroyed = 0;
    const NativeOwner owner = acquire_native_handle(live, destroyed);
    NativeHandle* const borrowed = owner.get();
    ASSERT_NE(borrowed, nullptr);
    EXPECT_EQ(borrowed, owner.get());
    EXPECT_EQ(borrowed->id, kHandleId);
    EXPECT_EQ(live, 1);
    EXPECT_EQ(destroyed, 0);
}

TEST(NativeResourceTest, ReleaseRequiresExplicitDestruction) {
    int live = 0;
    int destroyed = 0;
    NativeOwner owner = acquire_native_handle(live, destroyed);
    NativeHandle* const raw = owner.release();
    EXPECT_EQ(owner.get(), nullptr);
    EXPECT_EQ(live, 1);
    EXPECT_EQ(destroyed, 0);
    ASSERT_NE(raw, nullptr);
    EXPECT_EQ(raw->id, kHandleId);

    destroy_native_handle(raw);
    EXPECT_EQ(live, 0);
    EXPECT_EQ(destroyed, 1);
}

TEST(NativeResourceTest, CustomDeleterInvokesDestroy) {
    int live = 0;
    int destroyed = 0;
    NativeHandle* const raw = create_native_handle(live, destroyed);
    ASSERT_NE(raw, nullptr);
    NativeDeleter{}(raw);
    EXPECT_EQ(live, 0);
    EXPECT_EQ(destroyed, 1);

    NativeDeleter{}(nullptr);
    EXPECT_EQ(destroyed, 1);

    {
        const NativeOwner owner = acquire_native_handle(live, destroyed);
        EXPECT_EQ(live, 1);
    }
    EXPECT_EQ(live, 0);
    EXPECT_EQ(destroyed, 2);
}

TEST(NativeResourceTest, UniqueOwnerIsMovableAndNotCopyable) {
    static_assert(!std::is_copy_constructible_v<NativeOwner>);
    static_assert(!std::is_copy_assignable_v<NativeOwner>);
    static_assert(std::is_nothrow_move_constructible_v<NativeOwner>);
    static_assert(std::is_nothrow_move_assignable_v<NativeOwner>);
}

TEST(NativeResourceTest, SharedOwnersDestroyAfterTheLastRelease) {
    int live = 0;
    int destroyed = 0;
    std::shared_ptr<NativeHandle> first{create_native_handle(live, destroyed),
                                        destroy_native_handle};
    EXPECT_EQ(first.use_count(), 1);
    EXPECT_EQ(live, 1);

    std::shared_ptr<NativeHandle> second = first;
    EXPECT_EQ(first.use_count(), 2);
    EXPECT_EQ(second.use_count(), 2);
    EXPECT_EQ(live, 1);
    EXPECT_EQ(destroyed, 0);

    second.reset();
    EXPECT_EQ(first.use_count(), 1);
    EXPECT_EQ(live, 1);
    EXPECT_EQ(destroyed, 0);

    first.reset();
    EXPECT_EQ(live, 0);
    EXPECT_EQ(destroyed, 1);
}

TEST(NativeResourceTest, WeakPointerDoesNotExtendLifetime) {
    int live = 0;
    int destroyed = 0;
    std::shared_ptr<NativeHandle> owner{create_native_handle(live, destroyed),
                                        destroy_native_handle};
    std::weak_ptr<NativeHandle> observer = owner;
    EXPECT_FALSE(observer.expired());
    EXPECT_EQ(observer.use_count(), 1);

    {
        const std::shared_ptr<NativeHandle> locked = observer.lock();
        ASSERT_NE(locked, nullptr);
        EXPECT_EQ(locked->id, kHandleId);
        EXPECT_EQ(owner.use_count(), 2);
    }
    EXPECT_EQ(live, 1);
    EXPECT_EQ(destroyed, 0);

    owner.reset();
    EXPECT_TRUE(observer.expired());
    EXPECT_EQ(observer.use_count(), 0);
    EXPECT_EQ(observer.lock(), nullptr);
    EXPECT_EQ(live, 0);
    EXPECT_EQ(destroyed, 1);
}

TEST(NativeResourceTest, NullHandlesAreSafe) {
    int live = 0;
    int destroyed = 0;
    NativeOwner owner;
    EXPECT_EQ(owner.get(), nullptr);
    owner.reset();
    EXPECT_EQ(owner.release(), nullptr);

    std::shared_ptr<NativeHandle> shared;
    EXPECT_EQ(shared.get(), nullptr);
    EXPECT_EQ(shared.use_count(), 0);
    shared.reset();

    const std::weak_ptr<NativeHandle> observer;
    EXPECT_TRUE(observer.expired());
    EXPECT_EQ(observer.lock(), nullptr);

    destroy_native_handle(nullptr);
    EXPECT_EQ(live, 0);
    EXPECT_EQ(destroyed, 0);
}

TEST(NativeResourceTest, SharedCycleRetainsNodesUntilBroken) {
    int live = 0;
    std::shared_ptr<Node> first = std::make_shared<Node>(live);
    std::shared_ptr<Node> second = std::make_shared<Node>(live);
    EXPECT_EQ(live, 2);

    first->partner = second;
    second->partner = first;
    EXPECT_EQ(first.use_count(), 2);
    EXPECT_EQ(second.use_count(), 2);

    first.reset();
    EXPECT_EQ(live, 2);
    EXPECT_EQ(second.use_count(), 2);

    second->partner.reset();
    EXPECT_EQ(live, 1);

    second.reset();
    EXPECT_EQ(live, 0);
}
