#include "inferedge/request_lifecycle.hpp"

#include <gtest/gtest.h>

namespace {

inferedge::RequestLifecycle lifecycle_in(inferedge::RequestState state) {
    inferedge::RequestLifecycle lifecycle;
    if (state == inferedge::RequestState::kCreated) {
        return lifecycle;
    }

    EXPECT_EQ(lifecycle.transition_to(inferedge::RequestState::kQueued).error,
              inferedge::LifecycleError::kNone);
    if (state == inferedge::RequestState::kQueued) {
        return lifecycle;
    }

    EXPECT_EQ(lifecycle.transition_to(inferedge::RequestState::kRunning).error,
              inferedge::LifecycleError::kNone);
    if (state == inferedge::RequestState::kRunning) {
        return lifecycle;
    }

    if (state == inferedge::RequestState::kCompleted) {
        EXPECT_EQ(lifecycle.transition_to(inferedge::RequestState::kCompleted).error,
                  inferedge::LifecycleError::kNone);
        return lifecycle;
    }

    EXPECT_EQ(lifecycle.transition_to(inferedge::RequestState::kFailed).error,
              inferedge::LifecycleError::kNone);
    return lifecycle;
}

void expect_rejected(inferedge::RequestLifecycle& lifecycle, inferedge::RequestState next) {
    const inferedge::RequestState before = lifecycle.state();
    const inferedge::LifecycleResult result = lifecycle.transition_to(next);
    EXPECT_EQ(result.error, inferedge::LifecycleError::kInvalidTransition);
    EXPECT_FALSE(result.message.empty());
    EXPECT_EQ(lifecycle.state(), before);
}

} // namespace

TEST(RequestLifecycleTest, DefaultsToCreated) {
    const inferedge::RequestLifecycle lifecycle;
    EXPECT_EQ(lifecycle.state(), inferedge::RequestState::kCreated);
}

TEST(RequestLifecycleTest, AcceptsEveryLegalEdge) {
    inferedge::RequestLifecycle created;
    const inferedge::LifecycleResult queued =
        created.transition_to(inferedge::RequestState::kQueued);
    EXPECT_EQ(queued.error, inferedge::LifecycleError::kNone);
    EXPECT_TRUE(queued.message.empty());
    EXPECT_EQ(created.state(), inferedge::RequestState::kQueued);

    inferedge::RequestLifecycle running_source = lifecycle_in(inferedge::RequestState::kQueued);
    const inferedge::LifecycleResult running =
        running_source.transition_to(inferedge::RequestState::kRunning);
    EXPECT_EQ(running.error, inferedge::LifecycleError::kNone);
    EXPECT_TRUE(running.message.empty());
    EXPECT_EQ(running_source.state(), inferedge::RequestState::kRunning);

    inferedge::RequestLifecycle completed_source = lifecycle_in(inferedge::RequestState::kRunning);
    const inferedge::LifecycleResult completed =
        completed_source.transition_to(inferedge::RequestState::kCompleted);
    EXPECT_EQ(completed.error, inferedge::LifecycleError::kNone);
    EXPECT_TRUE(completed.message.empty());
    EXPECT_EQ(completed_source.state(), inferedge::RequestState::kCompleted);

    inferedge::RequestLifecycle failed_source = lifecycle_in(inferedge::RequestState::kRunning);
    const inferedge::LifecycleResult failed =
        failed_source.transition_to(inferedge::RequestState::kFailed);
    EXPECT_EQ(failed.error, inferedge::LifecycleError::kNone);
    EXPECT_TRUE(failed.message.empty());
    EXPECT_EQ(failed_source.state(), inferedge::RequestState::kFailed);
}

TEST(RequestLifecycleTest, WalksCreatedToCompleted) {
    inferedge::RequestLifecycle lifecycle;
    EXPECT_EQ(lifecycle.transition_to(inferedge::RequestState::kQueued).error,
              inferedge::LifecycleError::kNone);
    EXPECT_EQ(lifecycle.transition_to(inferedge::RequestState::kRunning).error,
              inferedge::LifecycleError::kNone);
    EXPECT_EQ(lifecycle.transition_to(inferedge::RequestState::kCompleted).error,
              inferedge::LifecycleError::kNone);
    EXPECT_EQ(lifecycle.state(), inferedge::RequestState::kCompleted);
}

TEST(RequestLifecycleTest, WalksCreatedToFailed) {
    inferedge::RequestLifecycle lifecycle;
    EXPECT_EQ(lifecycle.transition_to(inferedge::RequestState::kQueued).error,
              inferedge::LifecycleError::kNone);
    EXPECT_EQ(lifecycle.transition_to(inferedge::RequestState::kRunning).error,
              inferedge::LifecycleError::kNone);
    EXPECT_EQ(lifecycle.transition_to(inferedge::RequestState::kFailed).error,
              inferedge::LifecycleError::kNone);
    EXPECT_EQ(lifecycle.state(), inferedge::RequestState::kFailed);
}

TEST(RequestLifecycleTest, RejectsSelfTransitions) {
    const inferedge::RequestState states[] = {
        inferedge::RequestState::kCreated, inferedge::RequestState::kQueued,
        inferedge::RequestState::kRunning, inferedge::RequestState::kCompleted,
        inferedge::RequestState::kFailed,
    };
    for (const inferedge::RequestState state : states) {
        inferedge::RequestLifecycle lifecycle = lifecycle_in(state);
        expect_rejected(lifecycle, state);
    }
}

TEST(RequestLifecycleTest, RejectsSkippedStates) {
    inferedge::RequestLifecycle created;
    expect_rejected(created, inferedge::RequestState::kRunning);
    expect_rejected(created, inferedge::RequestState::kCompleted);
    expect_rejected(created, inferedge::RequestState::kFailed);

    inferedge::RequestLifecycle queued = lifecycle_in(inferedge::RequestState::kQueued);
    expect_rejected(queued, inferedge::RequestState::kCompleted);
    expect_rejected(queued, inferedge::RequestState::kFailed);
}

TEST(RequestLifecycleTest, RejectsBackwardTransitions) {
    inferedge::RequestLifecycle queued = lifecycle_in(inferedge::RequestState::kQueued);
    expect_rejected(queued, inferedge::RequestState::kCreated);

    inferedge::RequestLifecycle running = lifecycle_in(inferedge::RequestState::kRunning);
    expect_rejected(running, inferedge::RequestState::kQueued);
    expect_rejected(running, inferedge::RequestState::kCreated);
}

TEST(RequestLifecycleTest, CompletedIsTerminal) {
    inferedge::RequestLifecycle lifecycle = lifecycle_in(inferedge::RequestState::kCompleted);
    const inferedge::RequestState targets[] = {
        inferedge::RequestState::kCreated, inferedge::RequestState::kQueued,
        inferedge::RequestState::kRunning, inferedge::RequestState::kCompleted,
        inferedge::RequestState::kFailed,
    };
    for (const inferedge::RequestState target : targets) {
        expect_rejected(lifecycle, target);
    }
    EXPECT_EQ(lifecycle.state(), inferedge::RequestState::kCompleted);
}

TEST(RequestLifecycleTest, FailedIsTerminal) {
    inferedge::RequestLifecycle lifecycle = lifecycle_in(inferedge::RequestState::kFailed);
    const inferedge::RequestState targets[] = {
        inferedge::RequestState::kCreated, inferedge::RequestState::kQueued,
        inferedge::RequestState::kRunning, inferedge::RequestState::kCompleted,
        inferedge::RequestState::kFailed,
    };
    for (const inferedge::RequestState target : targets) {
        expect_rejected(lifecycle, target);
    }
    EXPECT_EQ(lifecycle.state(), inferedge::RequestState::kFailed);
}

TEST(RequestLifecycleTest, RejectsAnUnknownEnumerator) {
    inferedge::RequestLifecycle lifecycle;
    const auto unknown = static_cast<inferedge::RequestState>(99);
    const inferedge::LifecycleResult result = lifecycle.transition_to(unknown);
    EXPECT_EQ(result.error, inferedge::LifecycleError::kInvalidTransition);
    EXPECT_EQ(result.message, "invalid transition from created to unknown");
    EXPECT_EQ(lifecycle.state(), inferedge::RequestState::kCreated);

    inferedge::RequestLifecycle running = lifecycle_in(inferedge::RequestState::kRunning);
    const inferedge::LifecycleResult from_running = running.transition_to(unknown);
    EXPECT_EQ(from_running.error, inferedge::LifecycleError::kInvalidTransition);
    EXPECT_EQ(from_running.message, "invalid transition from running to unknown");
    EXPECT_EQ(running.state(), inferedge::RequestState::kRunning);
}

TEST(RequestLifecycleTest, RejectedTransitionPreservesState) {
    inferedge::RequestLifecycle lifecycle;
    const inferedge::LifecycleResult result =
        lifecycle.transition_to(inferedge::RequestState::kRunning);
    EXPECT_EQ(result.error, inferedge::LifecycleError::kInvalidTransition);
    EXPECT_EQ(result.message, "invalid transition from created to running");
    EXPECT_EQ(lifecycle.state(), inferedge::RequestState::kCreated);
}

TEST(RequestLifecycleTest, ConstGetterReadsTheState) {
    inferedge::RequestLifecycle lifecycle;
    const inferedge::RequestLifecycle& view = lifecycle;
    EXPECT_EQ(view.state(), inferedge::RequestState::kCreated);

    EXPECT_EQ(lifecycle.transition_to(inferedge::RequestState::kQueued).error,
              inferedge::LifecycleError::kNone);
    EXPECT_EQ(view.state(), inferedge::RequestState::kQueued);
}

TEST(RequestLifecycleTest, CopyHasAnIndependentState) {
    inferedge::RequestLifecycle original;
    EXPECT_EQ(original.transition_to(inferedge::RequestState::kQueued).error,
              inferedge::LifecycleError::kNone);

    inferedge::RequestLifecycle copy = original;
    EXPECT_EQ(copy.state(), inferedge::RequestState::kQueued);

    EXPECT_EQ(copy.transition_to(inferedge::RequestState::kRunning).error,
              inferedge::LifecycleError::kNone);
    EXPECT_EQ(original.state(), inferedge::RequestState::kQueued);
    EXPECT_EQ(copy.state(), inferedge::RequestState::kRunning);

    EXPECT_EQ(original.transition_to(inferedge::RequestState::kRunning).error,
              inferedge::LifecycleError::kNone);
    EXPECT_EQ(original.transition_to(inferedge::RequestState::kCompleted).error,
              inferedge::LifecycleError::kNone);
    EXPECT_EQ(original.state(), inferedge::RequestState::kCompleted);
    EXPECT_EQ(copy.state(), inferedge::RequestState::kRunning);
}
