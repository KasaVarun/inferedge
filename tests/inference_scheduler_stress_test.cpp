#include "inferedge/inference_scheduler.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <latch>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#ifndef INFEREDGE_ADD_MODEL_PATH
#error "CMake must define INFEREDGE_ADD_MODEL_PATH"
#endif

namespace {

using inferedge::Admission;
using inferedge::AdmissionStatus;
using inferedge::CompletionStatus;
using inferedge::InferenceErrorKind;
using inferedge::InferenceScheduler;
using inferedge::NamedTensor;
using inferedge::SchedulerCompletion;
using inferedge::SchedulerConfig;
using inferedge::SchedulerErrorKind;
using inferedge::SchedulerStats;
using inferedge::SchedulerTestAccess;
using inferedge::ShutdownPolicy;
using inferedge::ShutdownResult;

constexpr auto kCompletionBound = std::chrono::seconds{10};

class InferenceSchedulerStressTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

struct LatchRelease {
    std::shared_ptr<std::latch> latch;
    bool armed{false};

    explicit LatchRelease(std::shared_ptr<std::latch> held)
        : latch(std::move(held)), armed(latch != nullptr) {}

    void release() {
        if (!armed) {
            return;
        }
        armed = false;
        latch->count_down();
    }

    ~LatchRelease() { release(); }
};

struct JoinThread {
    std::thread thread;

    ~JoinThread() {
        if (thread.joinable()) {
            thread.join();
        }
    }
};

// Threads block until every participant arrives. abandon unblocks them if the
// test leaves before that happens. Declare the abandon guard after the join
// guards so it runs first.
struct StartGate {
    std::mutex mu;
    std::condition_variable cv;
    int arrived{0};
    int participants{0};
    bool open{false};
    bool abandon{false};

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mu);
        ++arrived;
        if (arrived >= participants) {
            open = true;
            cv.notify_all();
        }
        cv.wait(lock, [this] { return open || abandon; });
    }
};

struct AbandonGate {
    StartGate& gate;

    ~AbandonGate() {
        std::lock_guard<std::mutex> lock(gate.mu);
        gate.abandon = true;
        gate.cv.notify_all();
    }
};

struct Attempt {
    float tag{0.F};
    Admission admission;
};

std::vector<NamedTensor> AddInputs(float tag) {
    return {NamedTensor{"A", {2}, {tag, 0.F}}, NamedTensor{"B", {2}, {1.F, 0.F}}};
}

std::unique_ptr<InferenceScheduler> MakeAdd(std::uint32_t workers, std::size_t capacity) {
    return std::make_unique<InferenceScheduler>(
        SchedulerConfig{INFEREDGE_ADD_MODEL_PATH, workers, capacity});
}

SchedulerCompletion TakeBounded(Admission& admission) {
    SchedulerCompletion completion;
    EXPECT_EQ(admission.status, AdmissionStatus::kAccepted);
    EXPECT_FALSE(admission.inputs.has_value());
    if (!admission.completion.has_value()) {
        ADD_FAILURE() << "accepted admission has no future";
        return completion;
    }
    if (admission.completion->wait_for(kCompletionBound) != std::future_status::ready) {
        ADD_FAILURE() << "completion was not ready";
        return completion;
    }
    EXPECT_NO_THROW(completion = admission.completion->get());
    return completion;
}

void ExpectSum(const SchedulerCompletion& completion, float tag) {
    ASSERT_EQ(completion.status, CompletionStatus::kSucceeded);
    EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kNone);
    ASSERT_EQ(completion.outputs.size(), 1U);
    EXPECT_EQ(completion.outputs[0].name, "C");
    ASSERT_EQ(completion.outputs[0].values.size(), 2U);
    EXPECT_FLOAT_EQ(completion.outputs[0].values[0], tag + 1.F);
    EXPECT_FLOAT_EQ(completion.outputs[0].values[1], 0.F);
}

void ExpectRejected(const Attempt& attempt, AdmissionStatus status) {
    EXPECT_EQ(attempt.admission.status, status);
    EXPECT_FALSE(attempt.admission.completion.has_value());
    ASSERT_TRUE(attempt.admission.inputs.has_value());
    ASSERT_EQ(attempt.admission.inputs->size(), 2U);
    EXPECT_FLOAT_EQ(attempt.admission.inputs->front().values[0], attempt.tag);
    EXPECT_FLOAT_EQ(attempt.admission.inputs->front().values[1], 0.F);
    EXPECT_FLOAT_EQ(attempt.admission.inputs->back().values[0], 1.F);
    EXPECT_FLOAT_EQ(attempt.admission.inputs->back().values[1], 0.F);
}

void ExpectUnique(const std::vector<std::uint64_t>& values) {
    std::vector<std::uint64_t> ordered = values;
    std::sort(ordered.begin(), ordered.end());
    EXPECT_EQ(std::adjacent_find(ordered.begin(), ordered.end()), ordered.end());
}

void ExpectQuiescent(const InferenceScheduler& scheduler, std::uint64_t succeeded,
                     std::uint64_t failed_worker, std::uint64_t canceled) {
    const SchedulerStats stats = scheduler.stats();
    EXPECT_EQ(stats.succeeded, succeeded);
    EXPECT_EQ(stats.failed_inference, 0U);
    EXPECT_EQ(stats.failed_worker, failed_worker);
    EXPECT_EQ(stats.failed_scheduler, 0U);
    EXPECT_EQ(stats.canceled, canceled);
    EXPECT_EQ(stats.accepted, succeeded + failed_worker + canceled);
    EXPECT_EQ(stats.queue_depth, 0U);
    EXPECT_EQ(stats.executing, 0U);
    EXPECT_EQ(stats.live_workers, 0U);
}

void ExpectAgreedShutdown(const std::vector<ShutdownResult>& results) {
    ASSERT_FALSE(results.empty());
    int changed = 0;
    for (const ShutdownResult& result : results) {
        EXPECT_EQ(result.error.kind, SchedulerErrorKind::kNone);
        EXPECT_TRUE(result.established);
        EXPECT_EQ(result.policy, results.front().policy);
        if (result.changed) {
            ++changed;
        }
    }
    EXPECT_EQ(changed, 1);
}

TEST_F(InferenceSchedulerStressTest, StressProducersThenDrain) {
    constexpr int kProducers = 8;
    constexpr int kAttempts = 32;
    auto scheduler = MakeAdd(4, 32);
    std::vector<std::vector<Attempt>> records(static_cast<std::size_t>(kProducers));
    for (std::vector<Attempt>& row : records) {
        row.resize(static_cast<std::size_t>(kAttempts));
    }
    std::vector<JoinThread> producers(static_cast<std::size_t>(kProducers));
    const auto started = scheduler->start();
    ASSERT_EQ(started.kind, SchedulerErrorKind::kNone) << started.message;

    for (int producer = 0; producer < kProducers; ++producer) {
        producers[static_cast<std::size_t>(producer)].thread = std::thread([&, producer] {
            auto& row = records[static_cast<std::size_t>(producer)];
            for (int attempt = 0; attempt < kAttempts; ++attempt) {
                Attempt& slot = row[static_cast<std::size_t>(attempt)];
                slot.tag = static_cast<float>(producer * kAttempts + attempt);
                slot.admission = scheduler->submit(AddInputs(slot.tag));
            }
        });
    }
    for (JoinThread& producer : producers) {
        producer.thread.join();
    }

    int accepted = 0;
    int rejected_full = 0;
    int rejected_closed = 0;
    for (const std::vector<Attempt>& row : records) {
        for (const Attempt& attempt : row) {
            if (attempt.admission.status == AdmissionStatus::kAccepted) {
                ++accepted;
            } else if (attempt.admission.status == AdmissionStatus::kRejectedFull) {
                ++rejected_full;
                ExpectRejected(attempt, AdmissionStatus::kRejectedFull);
            } else {
                ++rejected_closed;
                ExpectRejected(attempt, AdmissionStatus::kRejectedClosed);
            }
        }
    }
    EXPECT_EQ(accepted + rejected_full + rejected_closed, kProducers * kAttempts);
    EXPECT_EQ(rejected_closed, 0);

    const ShutdownResult shutdown = scheduler->shutdown(ShutdownPolicy::kDrain);
    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(shutdown.established);
    EXPECT_TRUE(shutdown.changed);
    EXPECT_EQ(shutdown.policy, ShutdownPolicy::kDrain);

    std::vector<std::uint64_t> ids;
    std::vector<std::uint64_t> sequences;
    int succeeded = 0;
    for (std::vector<Attempt>& row : records) {
        for (Attempt& attempt : row) {
            if (attempt.admission.status != AdmissionStatus::kAccepted) {
                continue;
            }
            const SchedulerCompletion completion = TakeBounded(attempt.admission);
            EXPECT_EQ(completion.request_id, completion.sequence);
            ids.push_back(completion.request_id);
            sequences.push_back(completion.sequence);
            if (completion.status == CompletionStatus::kSucceeded) {
                ++succeeded;
                ExpectSum(completion, attempt.tag);
            } else {
                ADD_FAILURE() << "drain completion was not successful";
            }
        }
    }
    ExpectUnique(ids);
    ExpectUnique(sequences);
    EXPECT_EQ(static_cast<int>(ids.size()), accepted);
    ExpectQuiescent(*scheduler, static_cast<std::uint64_t>(succeeded), 0, 0);
    EXPECT_EQ(scheduler->stats().rejected_full, static_cast<std::uint64_t>(rejected_full));
    EXPECT_EQ(scheduler->stats().rejected_closed, 0U);
}

TEST_F(InferenceSchedulerStressTest, StressMixedShutdown) {
    constexpr int kProducers = 8;
    constexpr int kAttempts = 64;
    constexpr int kCallers = 4;
    auto scheduler = MakeAdd(4, 32);
    StartGate gate;
    gate.participants = kProducers + kCallers;
    std::array<ShutdownResult, kCallers> results{};
    const std::array<ShutdownPolicy, kCallers> policies{
        ShutdownPolicy::kDrain,
        ShutdownPolicy::kCancelPending,
        ShutdownPolicy::kDrain,
        ShutdownPolicy::kCancelPending,
    };
    std::vector<std::vector<Attempt>> records(static_cast<std::size_t>(kProducers));
    for (std::vector<Attempt>& row : records) {
        row.resize(static_cast<std::size_t>(kAttempts));
    }
    std::vector<JoinThread> producers(static_cast<std::size_t>(kProducers));
    std::vector<JoinThread> callers(static_cast<std::size_t>(kCallers));
    // Opens the gate before the join guards run, while policies and results
    // are still alive for any thread that has not passed the gate.
    AbandonGate abandon{gate};
    const auto started = scheduler->start();
    ASSERT_EQ(started.kind, SchedulerErrorKind::kNone) << started.message;
    for (int producer = 0; producer < kProducers; ++producer) {
        producers[static_cast<std::size_t>(producer)].thread = std::thread([&, producer] {
            gate.arrive_and_wait();
            auto& row = records[static_cast<std::size_t>(producer)];
            for (int attempt = 0; attempt < kAttempts; ++attempt) {
                Attempt& slot = row[static_cast<std::size_t>(attempt)];
                slot.tag = static_cast<float>(producer * kAttempts + attempt);
                slot.admission = scheduler->submit(AddInputs(slot.tag));
            }
        });
    }
    for (int caller = 0; caller < kCallers; ++caller) {
        callers[static_cast<std::size_t>(caller)].thread = std::thread([&, caller] {
            gate.arrive_and_wait();
            results[static_cast<std::size_t>(caller)] = scheduler->shutdown(policies[caller]);
        });
    }
    for (JoinThread& producer : producers) {
        producer.thread.join();
    }
    for (JoinThread& caller : callers) {
        caller.thread.join();
    }

    ExpectAgreedShutdown({results.begin(), results.end()});
    const ShutdownPolicy winner = results.front().policy;

    int accepted = 0;
    int rejected_full = 0;
    int rejected_closed = 0;
    int succeeded = 0;
    int canceled = 0;
    std::vector<std::uint64_t> ids;
    std::vector<std::uint64_t> sequences;
    for (std::vector<Attempt>& row : records) {
        for (Attempt& attempt : row) {
            if (attempt.admission.status == AdmissionStatus::kAccepted) {
                ++accepted;
                const SchedulerCompletion completion = TakeBounded(attempt.admission);
                EXPECT_EQ(completion.request_id, completion.sequence);
                ids.push_back(completion.request_id);
                sequences.push_back(completion.sequence);
                if (completion.status == CompletionStatus::kSucceeded) {
                    ++succeeded;
                    ExpectSum(completion, attempt.tag);
                } else if (completion.status == CompletionStatus::kCanceled) {
                    ++canceled;
                    EXPECT_EQ(winner, ShutdownPolicy::kCancelPending);
                    EXPECT_TRUE(completion.outputs.empty());
                    EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kNone);
                } else {
                    ADD_FAILURE() << "unexpected terminal status";
                }
            } else if (attempt.admission.status == AdmissionStatus::kRejectedFull) {
                ++rejected_full;
                ExpectRejected(attempt, AdmissionStatus::kRejectedFull);
            } else {
                ++rejected_closed;
                ExpectRejected(attempt, AdmissionStatus::kRejectedClosed);
            }
        }
    }
    EXPECT_EQ(accepted + rejected_full + rejected_closed, kProducers * kAttempts);
    ExpectUnique(ids);
    ExpectUnique(sequences);
    EXPECT_EQ(static_cast<int>(ids.size()), accepted);
    if (winner == ShutdownPolicy::kDrain) {
        EXPECT_EQ(canceled, 0);
    }
    ExpectQuiescent(*scheduler, static_cast<std::uint64_t>(succeeded), 0,
                    static_cast<std::uint64_t>(canceled));
    EXPECT_EQ(scheduler->stats().rejected_full, static_cast<std::uint64_t>(rejected_full));
    EXPECT_EQ(scheduler->stats().rejected_closed, static_cast<std::uint64_t>(rejected_closed));
    EXPECT_EQ(scheduler->stats().live_workers, 0U);
}

TEST_F(InferenceSchedulerStressTest, StressWorkerFailure) {
    constexpr int kRequests = 64;
    auto scheduler = MakeAdd(4, static_cast<std::size_t>(kRequests));
    auto gate1 = std::make_shared<std::latch>(1);
    auto gate2 = std::make_shared<std::latch>(1);
    auto gate3 = std::make_shared<std::latch>(1);
    LatchRelease release1{gate1};
    LatchRelease release2{gate2};
    LatchRelease release3{gate3};
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 1, gate1);
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 2, gate2);
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 3, gate3);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    const auto started = scheduler->start();
    ASSERT_EQ(started.kind, SchedulerErrorKind::kNone) << started.message;

    std::vector<Attempt> attempts(static_cast<std::size_t>(kRequests));
    for (int index = 0; index < kRequests; ++index) {
        Attempt& attempt = attempts[static_cast<std::size_t>(index)];
        attempt.tag = static_cast<float>(index);
        attempt.admission = scheduler->submit(AddInputs(attempt.tag));
        ASSERT_EQ(attempt.admission.status, AdmissionStatus::kAccepted);
    }

    std::optional<std::size_t> failed;
    const auto deadline = std::chrono::steady_clock::now() + kCompletionBound;
    while (std::chrono::steady_clock::now() < deadline && !failed.has_value()) {
        for (std::size_t index = 0; index < attempts.size(); ++index) {
            std::future<SchedulerCompletion>& completion = *attempts[index].admission.completion;
            if (completion.wait_for(std::chrono::milliseconds{1}) == std::future_status::ready) {
                failed = index;
                break;
            }
        }
    }
    ASSERT_TRUE(failed.has_value());
    const SchedulerCompletion failure = TakeBounded(attempts[*failed].admission);
    EXPECT_EQ(failure.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failure.inference_error.message, "injected");
    EXPECT_TRUE(failure.outputs.empty());

    release1.release();
    release2.release();
    release3.release();
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);

    int succeeded = 0;
    std::vector<std::uint64_t> ids;
    std::vector<std::uint64_t> sequences;
    ids.push_back(failure.request_id);
    sequences.push_back(failure.sequence);
    EXPECT_EQ(failure.request_id, failure.sequence);
    for (std::size_t index = 0; index < attempts.size(); ++index) {
        if (index == *failed) {
            continue;
        }
        const SchedulerCompletion completion = TakeBounded(attempts[index].admission);
        EXPECT_EQ(completion.request_id, completion.sequence);
        ids.push_back(completion.request_id);
        sequences.push_back(completion.sequence);
        if (completion.status == CompletionStatus::kSucceeded) {
            ++succeeded;
            ExpectSum(completion, attempts[index].tag);
        } else {
            ADD_FAILURE() << "healthy request did not succeed";
        }
    }
    ExpectUnique(ids);
    ExpectUnique(sequences);
    EXPECT_EQ(succeeded, kRequests - 1);
    ExpectQuiescent(*scheduler, static_cast<std::uint64_t>(succeeded), 1, 0);
}

TEST_F(InferenceSchedulerStressTest, StressRepeatedLifetime) {
    constexpr int kCycles = 32;
    constexpr int kRequests = 8;
    for (int cycle = 0; cycle < kCycles; ++cycle) {
        SCOPED_TRACE(cycle);
        auto scheduler = MakeAdd(1, static_cast<std::size_t>(kRequests));
        const auto started = scheduler->start();
        ASSERT_EQ(started.kind, SchedulerErrorKind::kNone) << started.message;

        std::vector<float> tags(static_cast<std::size_t>(kRequests));
        std::vector<std::future<SchedulerCompletion>> futures(static_cast<std::size_t>(kRequests));
        for (int index = 0; index < kRequests; ++index) {
            const float tag = static_cast<float>(cycle * kRequests + index);
            Admission admission = scheduler->submit(AddInputs(tag));
            ASSERT_EQ(admission.status, AdmissionStatus::kAccepted);
            ASSERT_TRUE(admission.completion.has_value());
            tags[static_cast<std::size_t>(index)] = tag;
            futures[static_cast<std::size_t>(index)] = std::move(*admission.completion);
        }
        scheduler.reset();
        for (int index = 0; index < kRequests; ++index) {
            std::future<SchedulerCompletion>& future = futures[static_cast<std::size_t>(index)];
            ASSERT_EQ(future.wait_for(std::chrono::seconds{0}), std::future_status::ready);
            SchedulerCompletion completion;
            EXPECT_NO_THROW(completion = future.get());
            ExpectSum(completion, tags[static_cast<std::size_t>(index)]);
        }
    }
}

} // namespace
