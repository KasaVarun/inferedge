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
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef INFEREDGE_ADD_MODEL_PATH
#error "CMake must define INFEREDGE_ADD_MODEL_PATH"
#endif

#ifndef INFEREDGE_BATCH_ADD_MODEL_PATH
#error "CMake must define INFEREDGE_BATCH_ADD_MODEL_PATH"
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
using inferedge::SchedulerError;
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

struct OutcomeCounts {
    std::uint64_t accepted{0};
    std::uint64_t rejected_full{0};
    std::uint64_t rejected_closed{0};
    std::uint64_t succeeded{0};
    std::uint64_t failed_inference{0};
    std::uint64_t failed_worker{0};
    std::uint64_t failed_scheduler{0};
    std::uint64_t canceled{0};
    std::vector<std::uint64_t> ids;
    std::vector<std::uint64_t> sequences;
    std::vector<std::string> worker_messages;
    std::vector<SchedulerCompletion> successful_completions;
};

struct BatchRuns {
    std::mutex mutex;
    std::vector<std::size_t> sizes;

    void Record(std::size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        sizes.push_back(size);
    }

    std::vector<std::size_t> Snapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        return sizes;
    }
};

std::vector<NamedTensor> AddInputs(float tag) {
    return {NamedTensor{"A", {2}, {tag, 0.F}}, NamedTensor{"B", {2}, {1.F, 0.F}}};
}

std::vector<NamedTensor> BatchAddInputs(float tag) {
    return {NamedTensor{"A", {1, 2}, {tag, 0.F}}, NamedTensor{"B", {1, 2}, {1.F, 0.F}}};
}

std::unique_ptr<InferenceScheduler> MakeAdd(std::uint32_t workers, std::size_t capacity) {
    return std::make_unique<InferenceScheduler>(
        SchedulerConfig{INFEREDGE_ADD_MODEL_PATH, workers, capacity});
}

std::unique_ptr<InferenceScheduler> MakeBatch(std::uint32_t workers, std::size_t capacity,
                                              std::uint32_t batch_size,
                                              std::chrono::milliseconds delay) {
    SchedulerConfig config{INFEREDGE_BATCH_ADD_MODEL_PATH, workers, capacity};
    config.enable_batching = true;
    config.max_batch_size = batch_size;
    config.max_batch_delay = delay;
    return std::make_unique<InferenceScheduler>(std::move(config));
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

void ExpectBatchSum(const SchedulerCompletion& completion, float tag) {
    ASSERT_EQ(completion.status, CompletionStatus::kSucceeded);
    EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kNone);
    ASSERT_EQ(completion.outputs.size(), 1U);
    EXPECT_EQ(completion.outputs[0].name, "C");
    EXPECT_EQ(completion.outputs[0].shape, (std::vector<std::int64_t>{1, 2}));
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

OutcomeCounts CollectBatchOutcomes(std::vector<std::vector<Attempt>>& records) {
    OutcomeCounts counts;
    for (std::vector<Attempt>& row : records) {
        for (Attempt& attempt : row) {
            if (attempt.admission.status == AdmissionStatus::kRejectedFull) {
                ++counts.rejected_full;
                ExpectRejected(attempt, AdmissionStatus::kRejectedFull);
                continue;
            }
            if (attempt.admission.status == AdmissionStatus::kRejectedClosed) {
                ++counts.rejected_closed;
                ExpectRejected(attempt, AdmissionStatus::kRejectedClosed);
                continue;
            }
            if (attempt.admission.status != AdmissionStatus::kAccepted) {
                ADD_FAILURE() << "unexpected admission status";
                continue;
            }

            ++counts.accepted;
            SchedulerCompletion completion = TakeBounded(attempt.admission);
            EXPECT_EQ(completion.request_id, completion.sequence);
            counts.ids.push_back(completion.request_id);
            counts.sequences.push_back(completion.sequence);
            switch (completion.status) {
            case CompletionStatus::kSucceeded:
                ++counts.succeeded;
                ExpectBatchSum(completion, attempt.tag);
                if (completion.outputs.size() == 1U &&
                    completion.outputs.front().values.size() == 2U) {
                    counts.successful_completions.push_back(std::move(completion));
                }
                break;
            case CompletionStatus::kInferenceFailed:
                ++counts.failed_inference;
                ADD_FAILURE() << "valid batch input failed inference";
                break;
            case CompletionStatus::kWorkerFailed:
                ++counts.failed_worker;
                EXPECT_TRUE(completion.outputs.empty());
                EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kExecution);
                counts.worker_messages.push_back(completion.inference_error.message);
                break;
            case CompletionStatus::kSchedulerFailed:
                ++counts.failed_scheduler;
                EXPECT_TRUE(completion.outputs.empty());
                break;
            case CompletionStatus::kCanceled:
                ++counts.canceled;
                EXPECT_TRUE(completion.outputs.empty());
                EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kNone);
                break;
            }
        }
    }
    ExpectUnique(counts.ids);
    ExpectUnique(counts.sequences);
    EXPECT_EQ(counts.ids.size(), counts.accepted);
    for (std::size_t left = 0; left < counts.successful_completions.size(); ++left) {
        const std::vector<float>& left_values =
            counts.successful_completions[left].outputs.front().values;
        for (std::size_t right = left + 1; right < counts.successful_completions.size(); ++right) {
            const std::vector<float>& right_values =
                counts.successful_completions[right].outputs.front().values;
            EXPECT_NE(left_values.data(), right_values.data());
        }
    }
    return counts;
}

void ExpectBatchAccounting(const InferenceScheduler& scheduler, const OutcomeCounts& counts) {
    const SchedulerStats stats = scheduler.stats();
    EXPECT_EQ(stats.accepted, counts.accepted);
    EXPECT_EQ(stats.succeeded, counts.succeeded);
    EXPECT_EQ(stats.failed_inference, counts.failed_inference);
    EXPECT_EQ(stats.failed_worker, counts.failed_worker);
    EXPECT_EQ(stats.failed_scheduler, counts.failed_scheduler);
    EXPECT_EQ(stats.canceled, counts.canceled);
    EXPECT_EQ(stats.accepted, stats.succeeded + stats.failed_inference + stats.failed_worker +
                                  stats.failed_scheduler + stats.canceled);
    EXPECT_EQ(stats.rejected_full, counts.rejected_full);
    EXPECT_EQ(stats.rejected_closed, counts.rejected_closed);
    EXPECT_EQ(stats.queue_depth, 0U);
    EXPECT_EQ(stats.executing, 0U);
    EXPECT_EQ(stats.live_workers, 0U);
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

TEST_F(InferenceSchedulerStressTest, StressBatchProducersThenDrain) {
    constexpr int kProducers = 8;
    constexpr int kAttempts = 32;
    constexpr std::uint32_t kWorkers = 4;
    constexpr std::size_t kCapacity = 8;
    constexpr std::uint32_t kBatchSize = 8;
    BatchRuns runs;
    auto before_pop = std::make_shared<std::latch>(1);
    auto scheduler = MakeBatch(kWorkers, kCapacity, kBatchSize, std::chrono::milliseconds{5});
    std::vector<JoinThread> producers(static_cast<std::size_t>(kProducers));
    LatchRelease release_pop{before_pop};
    SchedulerTestAccess::SetBeforePop(*scheduler, before_pop);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t size) { runs.Record(size); });
    const SchedulerError started = scheduler->start();
    ASSERT_EQ(started.kind, SchedulerErrorKind::kNone) << started.message;
    EXPECT_EQ(scheduler->stats().live_workers, kWorkers);

    std::vector<std::vector<Attempt>> records(static_cast<std::size_t>(kProducers));
    for (std::vector<Attempt>& row : records) {
        row.resize(static_cast<std::size_t>(kAttempts));
    }
    for (int producer = 0; producer < kProducers; ++producer) {
        producers[static_cast<std::size_t>(producer)].thread = std::thread([&, producer] {
            std::vector<Attempt>& row = records[static_cast<std::size_t>(producer)];
            for (int attempt = 0; attempt < kAttempts; ++attempt) {
                Attempt& slot = row[static_cast<std::size_t>(attempt)];
                slot.tag = static_cast<float>(producer * kAttempts + attempt);
                slot.admission = scheduler->submit(BatchAddInputs(slot.tag));
            }
        });
    }
    for (JoinThread& producer : producers) {
        producer.thread.join();
    }

    release_pop.release();
    const ShutdownResult shutdown = scheduler->shutdown(ShutdownPolicy::kDrain);
    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(shutdown.established);
    EXPECT_TRUE(shutdown.changed);
    EXPECT_EQ(shutdown.policy, ShutdownPolicy::kDrain);

    const OutcomeCounts counts = CollectBatchOutcomes(records);
    EXPECT_EQ(counts.accepted + counts.rejected_full + counts.rejected_closed,
              static_cast<std::uint64_t>(kProducers * kAttempts));
    EXPECT_GT(counts.accepted, 0U);
    EXPECT_EQ(counts.succeeded, counts.accepted);
    EXPECT_EQ(counts.rejected_closed, 0U);
    EXPECT_EQ(counts.failed_inference + counts.failed_worker + counts.failed_scheduler +
                  counts.canceled,
              0U);
    const std::vector<std::size_t> sizes = runs.Snapshot();
    ASSERT_FALSE(sizes.empty());
    std::size_t batched_requests = 0;
    for (const std::size_t size : sizes) {
        EXPECT_GE(size, 1U);
        EXPECT_LE(size, kBatchSize);
        batched_requests += size;
    }
    EXPECT_EQ(batched_requests, counts.succeeded);
    ExpectBatchAccounting(*scheduler, counts);
}

TEST_F(InferenceSchedulerStressTest, StressBatchProducersThenCancelPending) {
    constexpr int kProducers = 4;
    constexpr int kAttempts = 8;
    constexpr std::uint32_t kWorkers = 2;
    constexpr std::size_t kCapacity = 16;
    constexpr std::uint32_t kBatchSize = 4;
    BatchRuns runs;
    auto before_pop = std::make_shared<std::latch>(1);
    auto before_run = std::make_shared<std::latch>(1);
    auto first_sealed = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    std::atomic<bool> signal_first{true};
    auto scheduler = MakeBatch(kWorkers, kCapacity, kBatchSize, std::chrono::milliseconds{10});
    std::vector<JoinThread> producers(static_cast<std::size_t>(kProducers));
    JoinThread stopper;
    LatchRelease release_pop{before_pop};
    LatchRelease release_run{before_run};
    SchedulerTestAccess::SetBeforePop(*scheduler, before_pop);
    SchedulerTestAccess::SetBeforeRun(*scheduler, before_run);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetOnBatchRun(*scheduler, [&](std::size_t size) {
        runs.Record(size);
        if (signal_first.exchange(false)) {
            first_sealed->count_down();
        }
    });
    const SchedulerError started = scheduler->start();
    ASSERT_EQ(started.kind, SchedulerErrorKind::kNone) << started.message;

    std::vector<std::vector<Attempt>> records(static_cast<std::size_t>(kProducers));
    for (std::vector<Attempt>& row : records) {
        row.resize(static_cast<std::size_t>(kAttempts));
    }
    for (int producer = 0; producer < kProducers; ++producer) {
        producers[static_cast<std::size_t>(producer)].thread = std::thread([&, producer] {
            std::vector<Attempt>& row = records[static_cast<std::size_t>(producer)];
            for (int attempt = 0; attempt < kAttempts; ++attempt) {
                Attempt& slot = row[static_cast<std::size_t>(attempt)];
                slot.tag = static_cast<float>(producer * kAttempts + attempt);
                slot.admission = scheduler->submit(BatchAddInputs(slot.tag));
            }
        });
    }
    for (JoinThread& producer : producers) {
        producer.thread.join();
    }

    release_pop.release();
    first_sealed->wait();
    ShutdownResult shutdown;
    stopper.thread =
        std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    release_run.release();
    stopper.thread.join();

    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(shutdown.established);
    EXPECT_TRUE(shutdown.changed);
    EXPECT_EQ(shutdown.policy, ShutdownPolicy::kCancelPending);
    const OutcomeCounts counts = CollectBatchOutcomes(records);
    EXPECT_EQ(counts.accepted, kCapacity);
    EXPECT_GT(counts.succeeded, 0U);
    EXPECT_GT(counts.canceled, 0U);
    EXPECT_EQ(counts.succeeded + counts.canceled, counts.accepted);
    EXPECT_EQ(counts.failed_inference + counts.failed_worker + counts.failed_scheduler, 0U);
    std::size_t batched_requests = 0;
    for (const std::size_t size : runs.Snapshot()) {
        EXPECT_GE(size, 1U);
        EXPECT_LE(size, kBatchSize);
        batched_requests += size;
    }
    EXPECT_EQ(batched_requests, counts.succeeded);
    ExpectBatchAccounting(*scheduler, counts);
}

TEST_F(InferenceSchedulerStressTest, StressBatchProducersRaceMixedShutdown) {
    constexpr int kInitialProducers = 4;
    constexpr int kRaceProducers = 4;
    constexpr int kInitialAttempts = 4;
    constexpr int kRaceAttempts = 16;
    constexpr int kCallers = 4;
    constexpr std::uint32_t kWorkers = 2;
    constexpr std::size_t kCapacity = 8;
    constexpr std::uint32_t kBatchSize = 4;
    BatchRuns runs;
    auto before_pop = std::make_shared<std::latch>(1);
    auto before_run = std::make_shared<std::latch>(1);
    auto first_sealed = std::make_shared<std::latch>(1);
    auto followers_entered = std::make_shared<std::latch>(kCallers - 1);
    std::atomic<bool> signal_first{true};
    StartGate start_gate;
    start_gate.participants = kRaceProducers + kCallers;
    auto scheduler = MakeBatch(kWorkers, kCapacity, kBatchSize, std::chrono::milliseconds{5});
    std::vector<JoinThread> initial_producers(static_cast<std::size_t>(kInitialProducers));
    std::vector<JoinThread> race_producers(static_cast<std::size_t>(kRaceProducers));
    std::vector<JoinThread> callers(static_cast<std::size_t>(kCallers));
    LatchRelease release_pop{before_pop};
    LatchRelease release_run{before_run};
    AbandonGate abandon{start_gate};
    SchedulerTestAccess::SetBeforePop(*scheduler, before_pop);
    SchedulerTestAccess::SetBeforeRun(*scheduler, before_run);
    SchedulerTestAccess::SetShutdownFollowerEntered(*scheduler, followers_entered);
    SchedulerTestAccess::SetOnBatchRun(*scheduler, [&](std::size_t size) {
        runs.Record(size);
        if (signal_first.exchange(false)) {
            first_sealed->count_down();
        }
    });
    const SchedulerError started = scheduler->start();
    ASSERT_EQ(started.kind, SchedulerErrorKind::kNone) << started.message;

    std::vector<std::vector<Attempt>> records(
        static_cast<std::size_t>(kInitialProducers + kRaceProducers));
    for (int producer = 0; producer < kInitialProducers; ++producer) {
        records[static_cast<std::size_t>(producer)].resize(
            static_cast<std::size_t>(kInitialAttempts));
        initial_producers[static_cast<std::size_t>(producer)].thread = std::thread([&, producer] {
            std::vector<Attempt>& row = records[static_cast<std::size_t>(producer)];
            for (int attempt = 0; attempt < kInitialAttempts; ++attempt) {
                Attempt& slot = row[static_cast<std::size_t>(attempt)];
                slot.tag = static_cast<float>(producer * kInitialAttempts + attempt);
                slot.admission = scheduler->submit(BatchAddInputs(slot.tag));
            }
        });
    }
    for (JoinThread& producer : initial_producers) {
        producer.thread.join();
    }
    release_pop.release();
    first_sealed->wait();

    std::array<ShutdownResult, kCallers> results{};
    const std::array<ShutdownPolicy, kCallers> policies{
        ShutdownPolicy::kDrain,
        ShutdownPolicy::kCancelPending,
        ShutdownPolicy::kDrain,
        ShutdownPolicy::kCancelPending,
    };
    for (int producer = 0; producer < kRaceProducers; ++producer) {
        const int row_index = kInitialProducers + producer;
        records[static_cast<std::size_t>(row_index)].resize(
            static_cast<std::size_t>(kRaceAttempts));
        race_producers[static_cast<std::size_t>(producer)].thread =
            std::thread([&, producer, row_index] {
                start_gate.arrive_and_wait();
                std::vector<Attempt>& row = records[static_cast<std::size_t>(row_index)];
                for (int attempt = 0; attempt < kRaceAttempts; ++attempt) {
                    Attempt& slot = row[static_cast<std::size_t>(attempt)];
                    slot.tag = static_cast<float>(100 + producer * kRaceAttempts + attempt);
                    slot.admission = scheduler->submit(BatchAddInputs(slot.tag));
                }
            });
    }
    for (int caller = 0; caller < kCallers; ++caller) {
        callers[static_cast<std::size_t>(caller)].thread = std::thread([&, caller] {
            start_gate.arrive_and_wait();
            results[static_cast<std::size_t>(caller)] =
                scheduler->shutdown(policies[static_cast<std::size_t>(caller)]);
        });
    }
    for (JoinThread& producer : race_producers) {
        producer.thread.join();
    }
    followers_entered->wait();
    release_run.release();
    for (JoinThread& caller : callers) {
        caller.thread.join();
    }

    ExpectAgreedShutdown({results.begin(), results.end()});
    const OutcomeCounts counts = CollectBatchOutcomes(records);
    EXPECT_EQ(counts.accepted + counts.rejected_full + counts.rejected_closed,
              static_cast<std::uint64_t>(kInitialProducers * kInitialAttempts +
                                         kRaceProducers * kRaceAttempts));
    EXPECT_GT(counts.accepted, 0U);
    EXPECT_EQ(counts.failed_inference + counts.failed_worker + counts.failed_scheduler, 0U);
    if (results.front().policy == ShutdownPolicy::kDrain) {
        EXPECT_EQ(counts.canceled, 0U);
        EXPECT_EQ(counts.succeeded, counts.accepted);
    } else {
        EXPECT_EQ(counts.succeeded + counts.canceled, counts.accepted);
    }
    std::size_t batched_requests = 0;
    for (const std::size_t size : runs.Snapshot()) {
        EXPECT_GE(size, 1U);
        EXPECT_LE(size, kBatchSize);
        batched_requests += size;
    }
    EXPECT_EQ(batched_requests, counts.succeeded);
    ExpectBatchAccounting(*scheduler, counts);
}

TEST_F(InferenceSchedulerStressTest, StressBatchWorkerFailureStages) {
    constexpr int kRequests = 12;
    constexpr std::uint32_t kBatchSize = 4;
    for (int scenario = 0; scenario < 3; ++scenario) {
        SCOPED_TRACE(scenario);
        const std::size_t expected_failures = scenario == 0 ? 1U : kBatchSize;
        auto gate0 = std::make_shared<std::latch>(1);
        auto gate1 = std::make_shared<std::latch>(1);
        auto scheduler = MakeBatch(2, kRequests, kBatchSize, std::chrono::milliseconds{5});
        LatchRelease release0{gate0};
        LatchRelease release1{gate1};
        SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 0, gate0);
        SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 1, gate1);
        if (scenario < 2) {
            SchedulerTestAccess::SetThrowAfterBatchCollect(*scheduler, expected_failures);
        } else {
            SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
        }
        const SchedulerError started = scheduler->start();
        ASSERT_EQ(started.kind, SchedulerErrorKind::kNone) << started.message;

        std::vector<std::vector<Attempt>> records(1);
        records.front().resize(kRequests);
        for (int index = 0; index < kRequests; ++index) {
            Attempt& attempt = records.front()[static_cast<std::size_t>(index)];
            attempt.tag = static_cast<float>(scenario * 100 + index);
            attempt.admission = scheduler->submit(BatchAddInputs(attempt.tag));
            ASSERT_EQ(attempt.admission.status, AdmissionStatus::kAccepted);
        }

        release0.release();
        std::size_t ready = 0;
        const auto deadline = std::chrono::steady_clock::now() + kCompletionBound;
        while (std::chrono::steady_clock::now() < deadline && ready < expected_failures) {
            ready = 0;
            for (Attempt& attempt : records.front()) {
                if (attempt.admission.completion->wait_for(std::chrono::milliseconds{1}) ==
                    std::future_status::ready) {
                    ++ready;
                }
            }
        }
        ASSERT_EQ(ready, expected_failures);
        release1.release();
        const ShutdownResult shutdown = scheduler->shutdown(ShutdownPolicy::kDrain);
        EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
        EXPECT_TRUE(shutdown.established);
        EXPECT_EQ(shutdown.policy, ShutdownPolicy::kDrain);

        const OutcomeCounts counts = CollectBatchOutcomes(records);
        EXPECT_EQ(counts.accepted, kRequests);
        EXPECT_EQ(counts.failed_worker, expected_failures);
        EXPECT_EQ(counts.succeeded + counts.failed_worker, counts.accepted);
        EXPECT_EQ(counts.failed_inference + counts.failed_scheduler + counts.canceled, 0U);
        ASSERT_EQ(counts.worker_messages.size(), expected_failures);
        const std::string expected_message = scenario < 2 ? "collect failed" : "injected";
        for (const std::string& message : counts.worker_messages) {
            EXPECT_EQ(message, expected_message);
        }
        ExpectBatchAccounting(*scheduler, counts);
    }
}

TEST_F(InferenceSchedulerStressTest, StressRepeatedBatchLifetimes) {
    constexpr int kCycles = 12;
    constexpr int kProducers = 2;
    constexpr int kAttempts = 8;
    constexpr std::array<std::uint32_t, 3> kWorkers{1, 2, 4};
    constexpr std::array<std::uint32_t, 4> kBatchSizes{1, 4, 8, 32};
    constexpr std::array<std::size_t, 3> kCapacities{2, 4, 8};

    for (int cycle = 0; cycle < kCycles; ++cycle) {
        SCOPED_TRACE(cycle);
        BatchRuns runs;
        const std::uint32_t workers = kWorkers[static_cast<std::size_t>(cycle) % kWorkers.size()];
        const std::uint32_t batch_size =
            kBatchSizes[static_cast<std::size_t>(cycle) % kBatchSizes.size()];
        const std::size_t capacity =
            kCapacities[static_cast<std::size_t>(cycle) % kCapacities.size()];
        const std::chrono::milliseconds delay =
            cycle % 2 == 0 ? std::chrono::milliseconds{0} : std::chrono::milliseconds{5};
        auto scheduler = MakeBatch(workers, capacity, batch_size, delay);
        std::vector<JoinThread> producers(static_cast<std::size_t>(kProducers));
        SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                           [&runs](std::size_t size) { runs.Record(size); });
        const SchedulerError started = scheduler->start();
        ASSERT_EQ(started.kind, SchedulerErrorKind::kNone) << started.message;

        std::vector<std::vector<Attempt>> records(static_cast<std::size_t>(kProducers));
        for (std::vector<Attempt>& row : records) {
            row.resize(static_cast<std::size_t>(kAttempts));
        }
        for (int producer = 0; producer < kProducers; ++producer) {
            producers[static_cast<std::size_t>(producer)].thread = std::thread([&, producer] {
                std::vector<Attempt>& row = records[static_cast<std::size_t>(producer)];
                for (int attempt = 0; attempt < kAttempts; ++attempt) {
                    Attempt& slot = row[static_cast<std::size_t>(attempt)];
                    slot.tag = static_cast<float>(cycle * 100 + producer * kAttempts + attempt);
                    slot.admission = scheduler->submit(BatchAddInputs(slot.tag));
                }
            });
        }
        for (JoinThread& producer : producers) {
            producer.thread.join();
        }

        if (cycle % 5 == 4) {
            scheduler.reset();
            const OutcomeCounts counts = CollectBatchOutcomes(records);
            EXPECT_GT(counts.accepted, 0U);
            EXPECT_EQ(counts.succeeded, counts.accepted);
            EXPECT_EQ(counts.failed_inference + counts.failed_worker + counts.failed_scheduler +
                          counts.canceled,
                      0U);
            continue;
        }

        const ShutdownPolicy policy =
            cycle % 2 == 0 ? ShutdownPolicy::kDrain : ShutdownPolicy::kCancelPending;
        const ShutdownResult shutdown = scheduler->shutdown(policy);
        EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
        EXPECT_TRUE(shutdown.established);
        EXPECT_EQ(shutdown.policy, policy);
        const OutcomeCounts counts = CollectBatchOutcomes(records);
        EXPECT_GT(counts.accepted, 0U);
        EXPECT_EQ(counts.failed_inference + counts.failed_worker + counts.failed_scheduler, 0U);
        if (policy == ShutdownPolicy::kDrain) {
            EXPECT_EQ(counts.succeeded, counts.accepted);
            EXPECT_EQ(counts.canceled, 0U);
        } else {
            EXPECT_EQ(counts.succeeded + counts.canceled, counts.accepted);
        }
        ExpectBatchAccounting(*scheduler, counts);
    }
}

} // namespace
