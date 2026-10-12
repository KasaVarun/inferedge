#include "inferedge/inference_scheduler.hpp"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <latch>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef INFEREDGE_ADD_MODEL_PATH
#error "CMake must define INFEREDGE_ADD_MODEL_PATH"
#endif
#ifndef INFEREDGE_INVALID_MODEL_PATH
#error "CMake must define INFEREDGE_INVALID_MODEL_PATH"
#endif
#ifndef INFEREDGE_MNIST_MODEL_PATH
#error "CMake must define INFEREDGE_MNIST_MODEL_PATH"
#endif
#ifndef INFEREDGE_MNIST_FIXTURE_DIR
#error "CMake must define INFEREDGE_MNIST_FIXTURE_DIR"
#endif

namespace {

using inferedge::Admission;
using inferedge::AdmissionStatus;
using inferedge::CompletionStatus;
using inferedge::InferenceErrorKind;
using inferedge::InferenceScheduler;
using inferedge::kMaxSchedulerQueueCapacity;
using inferedge::kMaxSchedulerWorkers;
using inferedge::NamedTensor;
using inferedge::SchedulerCompletion;
using inferedge::SchedulerConfig;
using inferedge::SchedulerError;
using inferedge::SchedulerErrorKind;
using inferedge::SchedulerStats;
using inferedge::SchedulerTestAccess;
using inferedge::ShutdownPolicy;
using inferedge::ShutdownResult;

class InferenceSchedulerTest : public testing::Test {
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

std::vector<NamedTensor> AddInputs(float a0, float a1, float b0, float b1) {
    return {NamedTensor{"A", {2}, {a0, a1}}, NamedTensor{"B", {2}, {b0, b1}}};
}

std::unique_ptr<InferenceScheduler> MakeScheduler(SchedulerConfig config) {
    return std::make_unique<InferenceScheduler>(std::move(config));
}

std::unique_ptr<InferenceScheduler> MakeAdd(std::uint32_t workers, std::size_t capacity) {
    return MakeScheduler(SchedulerConfig{INFEREDGE_ADD_MODEL_PATH, workers, capacity});
}

void RequireStart(InferenceScheduler& scheduler) {
    const SchedulerError error = scheduler.start();
    ASSERT_EQ(error.kind, SchedulerErrorKind::kNone) << error.message;
}

SchedulerCompletion Take(Admission admission) {
    EXPECT_EQ(admission.status, AdmissionStatus::kAccepted);
    EXPECT_FALSE(admission.inputs.has_value());
    EXPECT_TRUE(admission.completion.has_value());
    return admission.completion->get();
}

void ExpectSum(const SchedulerCompletion& completion, float c0, float c1) {
    ASSERT_EQ(completion.status, CompletionStatus::kSucceeded);
    EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kNone);
    ASSERT_EQ(completion.outputs.size(), 1U);
    EXPECT_EQ(completion.outputs[0].name, "C");
    ASSERT_EQ(completion.outputs[0].values.size(), 2U);
    EXPECT_FLOAT_EQ(completion.outputs[0].values[0], c0);
    EXPECT_FLOAT_EQ(completion.outputs[0].values[1], c1);
}

void ExpectQuiescent(const InferenceScheduler& scheduler, std::uint64_t succeeded,
                     std::uint64_t failed_inference, std::uint64_t failed_worker,
                     std::uint64_t failed_scheduler) {
    const SchedulerStats stats = scheduler.stats();
    EXPECT_EQ(stats.succeeded, succeeded);
    EXPECT_EQ(stats.failed_inference, failed_inference);
    EXPECT_EQ(stats.failed_worker, failed_worker);
    EXPECT_EQ(stats.failed_scheduler, failed_scheduler);
    EXPECT_EQ(stats.accepted, succeeded + failed_inference + failed_worker + failed_scheduler);
    EXPECT_EQ(stats.queue_depth, 0U);
    EXPECT_EQ(stats.executing, 0U);
    EXPECT_EQ(stats.live_workers, 0U);
}

std::vector<float> ReadFloats(const std::filesystem::path& path, std::size_t count) {
    std::ifstream input{path, std::ios::binary};
    EXPECT_TRUE(input) << path;
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        unsigned char bytes[4] = {};
        input.read(reinterpret_cast<char*>(bytes), 4);
        EXPECT_TRUE(input) << path;
        const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                                   (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                                   (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                                   (static_cast<std::uint32_t>(bytes[3]) << 24U);
        values[index] = std::bit_cast<float>(bits);
    }
    return values;
}

int ArgMax(const std::vector<float>& values) {
    std::size_t best = 0;
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return static_cast<int>(best);
}

TEST_F(InferenceSchedulerTest, RejectsZeroWorkers) {
    const auto scheduler = MakeAdd(0, 1);
    const SchedulerError error = scheduler->start();
    EXPECT_EQ(error.kind, SchedulerErrorKind::kInvalidConfig);
    EXPECT_FALSE(SchedulerTestAccess::IsFailed(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::IsStopped(*scheduler));
}

TEST_F(InferenceSchedulerTest, RejectsTooManyWorkers) {
    const auto scheduler = MakeAdd(kMaxSchedulerWorkers + 1, 1);
    const SchedulerError error = scheduler->start();
    EXPECT_EQ(error.kind, SchedulerErrorKind::kInvalidConfig);
}

TEST_F(InferenceSchedulerTest, RejectsZeroQueueCapacity) {
    const auto scheduler = MakeAdd(1, 0);
    const SchedulerError error = scheduler->start();
    EXPECT_EQ(error.kind, SchedulerErrorKind::kInvalidConfig);
}

TEST_F(InferenceSchedulerTest, RejectsExcessiveQueueCapacity) {
    const auto scheduler = MakeAdd(1, kMaxSchedulerQueueCapacity + 1);
    const SchedulerError error = scheduler->start();
    EXPECT_EQ(error.kind, SchedulerErrorKind::kInvalidConfig);
}

TEST_F(InferenceSchedulerTest, RejectsEmptyModelPath) {
    const auto scheduler = MakeScheduler(SchedulerConfig{});
    const SchedulerError error = scheduler->start();
    EXPECT_EQ(error.kind, SchedulerErrorKind::kInvalidConfig);
    const SchedulerError again = scheduler->start();
    EXPECT_EQ(again.kind, SchedulerErrorKind::kInvalidConfig);
}

TEST_F(InferenceSchedulerTest, ModelLoadFailureLeavesNoWorkers) {
    const auto scheduler = MakeScheduler(SchedulerConfig{INFEREDGE_INVALID_MODEL_PATH, 2, 1});
    const SchedulerError error = scheduler->start();
    EXPECT_EQ(error.kind, SchedulerErrorKind::kModelLoad);
    EXPECT_EQ(error.load_error.kind, InferenceErrorKind::kModelInvalid);
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
    const SchedulerStats stats = scheduler->stats();
    EXPECT_EQ(stats.accepted, 0U);
    EXPECT_EQ(stats.live_workers, 0U);
}

TEST_F(InferenceSchedulerTest, PartialWorkerStartupAborts) {
    const auto scheduler = MakeAdd(2, 1);
    SchedulerTestAccess::SetStopAfterStartingWorkers(*scheduler, 1);
    const SchedulerError error = scheduler->start();
    EXPECT_EQ(error.kind, SchedulerErrorKind::kWorkerStart);
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));

    std::vector<NamedTensor> inputs = AddInputs(1.F, 2.F, 3.F, 4.F);
    const Admission rejected = scheduler->submit(std::move(inputs));
    EXPECT_EQ(rejected.status, AdmissionStatus::kRejectedClosed);
    EXPECT_FALSE(rejected.completion.has_value());
    ASSERT_TRUE(rejected.inputs.has_value());
    ASSERT_EQ(rejected.inputs->size(), 2U);
    EXPECT_FLOAT_EQ(rejected.inputs->front().values[0], 1.F);

    const SchedulerStats stats = scheduler->stats();
    EXPECT_EQ(stats.accepted, 0U);
    EXPECT_EQ(stats.live_workers, 0U);
    EXPECT_EQ(stats.executing, 0U);

    const SchedulerError restart = scheduler->start();
    EXPECT_EQ(restart.kind, SchedulerErrorKind::kInvalidState);
}

TEST_F(InferenceSchedulerTest, OneWorkerAddInference) {
    const auto scheduler = MakeAdd(1, 1);
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    const ShutdownResult shutdown = scheduler->shutdown(ShutdownPolicy::kDrain);
    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(shutdown.changed);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, FourWorkerAddInference) {
    const auto scheduler = MakeAdd(4, 4);
    RequireStart(*scheduler);
    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(5.F, 6.F, 7.F, 8.F));
    Admission third = scheduler->submit(AddInputs(0.F, 1.F, 0.F, 1.F));
    Admission fourth = scheduler->submit(AddInputs(9.F, 0.F, 1.F, 1.F));
    ExpectSum(Take(std::move(first)), 4.F, 6.F);
    ExpectSum(Take(std::move(second)), 12.F, 14.F);
    ExpectSum(Take(std::move(third)), 0.F, 2.F);
    ExpectSum(Take(std::move(fourth)), 10.F, 1.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 4, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, MnistInference) {
    constexpr std::size_t kInputCount = 784;
    constexpr std::size_t kLogitCount = 10;
    const std::filesystem::path root{INFEREDGE_MNIST_FIXTURE_DIR};
    const std::vector<float> image = ReadFloats(root / "inputs" / "test_00000.f32", kInputCount);
    const std::vector<float> reference = ReadFloats(root / "onnx" / "test_00000.f32", kLogitCount);
    ASSERT_EQ(image.size(), kInputCount);
    ASSERT_EQ(reference.size(), kLogitCount);

    const auto scheduler = MakeScheduler(SchedulerConfig{INFEREDGE_MNIST_MODEL_PATH, 1, 1});
    RequireStart(*scheduler);
    const SchedulerCompletion completion = Take(
        scheduler->submit({NamedTensor{"image", {static_cast<std::int64_t>(kInputCount)}, image}}));
    ASSERT_EQ(completion.status, CompletionStatus::kSucceeded);
    ASSERT_EQ(completion.outputs.size(), 1U);
    EXPECT_EQ(completion.outputs[0].name, "logits");
    ASSERT_EQ(completion.outputs[0].values.size(), kLogitCount);
    EXPECT_EQ(ArgMax(completion.outputs[0].values), 7);
    for (std::size_t index = 0; index < kLogitCount; ++index) {
        const double actual = completion.outputs[0].values[index];
        const double expected = reference[index];
        const double difference = std::fabs(actual - expected);
        const double limit = 1e-5 + 1e-4 * std::fabs(expected);
        EXPECT_LE(difference, limit) << index;
    }
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
}

TEST_F(InferenceSchedulerTest, InvalidAddInputThenValidAdd) {
    const auto scheduler = MakeAdd(1, 2);
    RequireStart(*scheduler);
    const SchedulerCompletion failed = Take(scheduler->submit({NamedTensor{"A", {2}, {1.F, 2.F}}}));
    EXPECT_EQ(failed.status, CompletionStatus::kInferenceFailed);
    EXPECT_EQ(failed.inference_error.kind, InferenceErrorKind::kMissingInput);
    EXPECT_TRUE(failed.outputs.empty());
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 1, 0, 0);
}

TEST_F(InferenceSchedulerTest, CallerClearsInputsAfterSubmit) {
    const auto scheduler = MakeAdd(1, 1);
    RequireStart(*scheduler);
    std::vector<NamedTensor> inputs = AddInputs(1.F, 2.F, 3.F, 4.F);
    Admission admission = scheduler->submit(inputs);
    inputs.clear();
    inputs.shrink_to_fit();
    ExpectSum(Take(std::move(admission)), 4.F, 6.F);
}

TEST_F(InferenceSchedulerTest, ReturnedOutputsRemainOwned) {
    auto scheduler = MakeAdd(1, 1);
    RequireStart(*scheduler);
    SchedulerCompletion completion = Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F)));
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    scheduler.reset();
    ASSERT_EQ(completion.outputs.size(), 1U);
    const std::vector<float> values = std::move(completion.outputs[0].values);
    ASSERT_EQ(values.size(), 2U);
    EXPECT_FLOAT_EQ(values[0], 4.F);
    EXPECT_FLOAT_EQ(values[1], 6.F);
}

TEST_F(InferenceSchedulerTest, EveryAcceptedRequestCompletesOnce) {
    constexpr int kRequests = 8;
    const auto scheduler = MakeAdd(4, static_cast<std::size_t>(kRequests));
    RequireStart(*scheduler);
    std::vector<Admission> admissions;
    admissions.reserve(static_cast<std::size_t>(kRequests));
    for (int index = 0; index < kRequests; ++index) {
        const float value = static_cast<float>(index);
        admissions.push_back(scheduler->submit(AddInputs(value, 0.F, 1.F, 0.F)));
        EXPECT_EQ(admissions.back().status, AdmissionStatus::kAccepted);
    }
    std::vector<std::uint64_t> ids;
    for (int index = 0; index < kRequests; ++index) {
        const SchedulerCompletion completion =
            Take(std::move(admissions[static_cast<std::size_t>(index)]));
        ExpectSum(completion, static_cast<float>(index) + 1.F, 0.F);
        ids.push_back(completion.request_id);
        EXPECT_EQ(completion.sequence, completion.request_id);
    }
    EXPECT_EQ(ids.size(), static_cast<std::size_t>(kRequests));
    for (std::size_t left = 0; left < ids.size(); ++left) {
        for (std::size_t right = left + 1; right < ids.size(); ++right) {
            EXPECT_NE(ids[left], ids[right]);
        }
    }
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, static_cast<std::uint64_t>(kRequests), 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, QueueFullRejectionReturnsInputs) {
    auto scheduler = MakeAdd(1, 1);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);

    Admission accepted = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    ASSERT_EQ(accepted.status, AdmissionStatus::kAccepted);
    const SchedulerStats waiting = scheduler->stats();
    EXPECT_EQ(waiting.queue_depth, 1U);
    EXPECT_EQ(waiting.accepted, 1U);

    std::vector<NamedTensor> second = AddInputs(8.F, 7.F, 6.F, 5.F);
    const Admission rejected = scheduler->submit(second);
    EXPECT_EQ(rejected.status, AdmissionStatus::kRejectedFull);
    EXPECT_FALSE(rejected.completion.has_value());
    ASSERT_TRUE(rejected.inputs.has_value());
    ASSERT_EQ(rejected.inputs->size(), 2U);
    EXPECT_FLOAT_EQ((*rejected.inputs)[0].values[0], 8.F);
    EXPECT_FLOAT_EQ((*rejected.inputs)[1].values[1], 5.F);
    EXPECT_EQ(scheduler->stats().rejected_full, 1U);
    EXPECT_EQ(scheduler->stats().accepted, 1U);

    release.release();
    ExpectSum(Take(std::move(accepted)), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
    EXPECT_EQ(scheduler->stats().rejected_full, 1U);
}

TEST_F(InferenceSchedulerTest, RejectedAdmissionHasNoFuture) {
    const auto scheduler = MakeAdd(1, 1);
    const Admission rejected = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    EXPECT_EQ(rejected.status, AdmissionStatus::kRejectedNotRunning);
    EXPECT_FALSE(rejected.completion.has_value());
    ASSERT_TRUE(rejected.inputs.has_value());
    EXPECT_EQ(rejected.inputs->front().name, "A");
    EXPECT_EQ(scheduler->stats().accepted, 0U);
}

TEST_F(InferenceSchedulerTest, SubmitAfterShutdownIsRejected) {
    const auto scheduler = MakeAdd(1, 1);
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    const ShutdownResult shutdown = scheduler->shutdown(ShutdownPolicy::kDrain);
    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));

    const Admission rejected = scheduler->submit(AddInputs(1.F, 1.F, 1.F, 1.F));
    EXPECT_EQ(rejected.status, AdmissionStatus::kRejectedClosed);
    EXPECT_FALSE(rejected.completion.has_value());
    ASSERT_TRUE(rejected.inputs.has_value());
    EXPECT_EQ(scheduler->stats().rejected_closed, 1U);
}

TEST_F(InferenceSchedulerTest, DrainCompletesQueuedWork) {
    auto scheduler = MakeAdd(1, 4);
    auto blocked = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    JoinThread stopper;
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    RequireStart(*scheduler);

    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(0.F, 1.F, 1.F, 1.F));
    Admission third = scheduler->submit(AddInputs(2.F, 0.F, 2.F, 0.F));
    EXPECT_EQ(scheduler->stats().queue_depth, 3U);

    ShutdownResult shutdown;
    stopper.thread = std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kDrain); });
    closed->wait();
    release.release();
    stopper.thread.join();

    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(shutdown.changed);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectSum(Take(std::move(first)), 4.F, 6.F);
    ExpectSum(Take(std::move(second)), 1.F, 2.F);
    ExpectSum(Take(std::move(third)), 4.F, 0.F);
    ExpectQuiescent(*scheduler, 3, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, DrainCompletesInFlightWork) {
    auto scheduler = MakeAdd(1, 1);
    auto reached = std::make_shared<std::latch>(1);
    auto release_run = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    JoinThread stopper;
    LatchRelease release{release_run};
    SchedulerTestAccess::SetReachedRun(*scheduler, reached);
    SchedulerTestAccess::SetBeforeRun(*scheduler, release_run);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    RequireStart(*scheduler);

    Admission admission = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    reached->wait();

    ShutdownResult shutdown;
    stopper.thread = std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kDrain); });
    closed->wait();
    release.release();
    stopper.thread.join();

    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(shutdown.changed);
    ASSERT_TRUE(admission.completion.has_value());
    EXPECT_EQ(admission.completion->wait_for(std::chrono::seconds{0}), std::future_status::ready);
    ExpectSum(Take(std::move(admission)), 4.F, 6.F);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
}

TEST_F(InferenceSchedulerTest, RepeatedDrainShutdownIsIdempotent) {
    const auto scheduler = MakeAdd(1, 1);
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 0.F, 1.F, 0.F))), 2.F, 0.F);
    const ShutdownResult first = scheduler->shutdown(ShutdownPolicy::kDrain);
    const ShutdownResult second = scheduler->shutdown(ShutdownPolicy::kDrain);
    EXPECT_EQ(first.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(first.changed);
    EXPECT_EQ(first.policy, ShutdownPolicy::kDrain);
    EXPECT_EQ(second.error.kind, SchedulerErrorKind::kNone);
    EXPECT_FALSE(second.changed);
    EXPECT_EQ(second.policy, ShutdownPolicy::kDrain);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
}

TEST_F(InferenceSchedulerTest, WorkerExceptionIsContained) {
    auto scheduler = MakeAdd(2, 2);
    auto hold = std::make_shared<std::latch>(1);
    LatchRelease release{hold};
    SchedulerTestAccess::SetHoldOtherWorkers(*scheduler, hold);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    RequireStart(*scheduler);

    const SchedulerCompletion failed = Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F)));
    EXPECT_EQ(failed.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed.inference_error.kind, InferenceErrorKind::kExecution);
    EXPECT_EQ(failed.inference_error.message, "injected");
    EXPECT_TRUE(failed.outputs.empty());

    release.release();
    const ShutdownResult shutdown = scheduler->shutdown(ShutdownPolicy::kDrain);
    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectQuiescent(*scheduler, 0, 0, 1, 0);
}

TEST_F(InferenceSchedulerTest, RetiredWorkerLeavesSiblingsRunning) {
    auto scheduler = MakeAdd(2, 2);
    auto hold = std::make_shared<std::latch>(1);
    LatchRelease release{hold};
    SchedulerTestAccess::SetHoldOtherWorkers(*scheduler, hold);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    RequireStart(*scheduler);

    Admission failed_admission = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission live_admission = scheduler->submit(AddInputs(5.F, 0.F, 1.F, 0.F));
    const SchedulerCompletion failed = Take(std::move(failed_admission));
    EXPECT_EQ(failed.status, CompletionStatus::kWorkerFailed);

    release.release();
    ExpectSum(Take(std::move(live_admission)), 6.F, 0.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectQuiescent(*scheduler, 1, 0, 1, 0);
}

TEST_F(InferenceSchedulerTest, LastWorkerRetiresWhileRunning) {
    auto scheduler = MakeAdd(1, 3);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    RequireStart(*scheduler);

    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(1.F, 0.F, 1.F, 0.F));
    Admission third = scheduler->submit(AddInputs(2.F, 0.F, 2.F, 0.F));
    EXPECT_EQ(scheduler->stats().queue_depth, 3U);
    release.release();

    const SchedulerCompletion worker_failed = Take(std::move(first));
    const SchedulerCompletion left = Take(std::move(second));
    const SchedulerCompletion right = Take(std::move(third));
    EXPECT_EQ(worker_failed.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(left.status, CompletionStatus::kSchedulerFailed);
    EXPECT_EQ(right.status, CompletionStatus::kSchedulerFailed);
    EXPECT_EQ(left.inference_error.message, "no worker");
    EXPECT_TRUE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::ClosedForDrain(*scheduler));

    const ShutdownResult shutdown = scheduler->shutdown(ShutdownPolicy::kDrain);
    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
    ExpectQuiescent(*scheduler, 0, 0, 1, 2);
}

TEST_F(InferenceSchedulerTest, LastWorkerRetiresWhileDraining) {
    auto scheduler = MakeAdd(1, 2);
    auto blocked = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    JoinThread stopper;
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    RequireStart(*scheduler);

    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(4.F, 0.F, 1.F, 0.F));

    ShutdownResult shutdown;
    stopper.thread = std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kDrain); });
    closed->wait();
    EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    release.release();
    stopper.thread.join();

    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(shutdown.changed);
    EXPECT_EQ(Take(std::move(first)).status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(Take(std::move(second)).status, CompletionStatus::kSchedulerFailed);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
    ExpectQuiescent(*scheduler, 0, 0, 1, 1);
}

TEST_F(InferenceSchedulerTest, AcceptedFutureRemainsValidIfWorkerFinishesFirst) {
    auto scheduler = MakeAdd(1, 1);
    auto published = std::make_shared<std::latch>(1);
    SchedulerTestAccess::SetAfterPublish(*scheduler, published);
    bool ready_before_return = false;
    SchedulerTestAccess::SetOnAccepted(*scheduler, [&](std::future<SchedulerCompletion>& future) {
        published->wait();
        ready_before_return = future.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
    });
    RequireStart(*scheduler);

    Admission admission = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    EXPECT_TRUE(ready_before_return);
    ASSERT_TRUE(admission.completion.has_value());
    EXPECT_EQ(admission.completion->wait_for(std::chrono::seconds{0}), std::future_status::ready);
    ExpectSum(Take(std::move(admission)), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, DestructorDrainsAndJoins) {
    std::optional<std::future<SchedulerCompletion>> future;
    {
        auto scheduler = MakeAdd(1, 2);
        RequireStart(*scheduler);
        Admission admission = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
        Admission second = scheduler->submit(AddInputs(1.F, 0.F, 0.F, 1.F));
        future = std::move(admission.completion);
        std::future<SchedulerCompletion> other = std::move(*second.completion);
        scheduler.reset();
        EXPECT_EQ(other.wait_for(std::chrono::seconds{0}), std::future_status::ready);
        ExpectSum(other.get(), 1.F, 1.F);
    }
    ASSERT_TRUE(future.has_value());
    EXPECT_EQ(future->wait_for(std::chrono::seconds{0}), std::future_status::ready);
    ExpectSum(future->get(), 4.F, 6.F);
}

TEST_F(InferenceSchedulerTest, FailedStartupDestroysSafely) {
    {
        auto scheduler = MakeAdd(3, 1);
        SchedulerTestAccess::SetStopAfterStartingWorkers(*scheduler, 1);
        const SchedulerError error = scheduler->start();
        EXPECT_EQ(error.kind, SchedulerErrorKind::kWorkerStart);
        EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
    }
    const auto missing = MakeScheduler(SchedulerConfig{INFEREDGE_INVALID_MODEL_PATH, 1, 1});
    EXPECT_EQ(missing->start().kind, SchedulerErrorKind::kModelLoad);
}

TEST_F(InferenceSchedulerTest, RestartAfterStoppedIsRejected) {
    const auto scheduler = MakeAdd(1, 1);
    RequireStart(*scheduler);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    const SchedulerError restart = scheduler->start();
    EXPECT_EQ(restart.kind, SchedulerErrorKind::kInvalidState);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
}

TEST_F(InferenceSchedulerTest, RestartAfterFailedIsRejected) {
    const auto scheduler = MakeScheduler(SchedulerConfig{INFEREDGE_INVALID_MODEL_PATH, 1, 1});
    EXPECT_EQ(scheduler->start().kind, SchedulerErrorKind::kModelLoad);
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
    const SchedulerError restart = scheduler->start();
    EXPECT_EQ(restart.kind, SchedulerErrorKind::kInvalidState);
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
}

TEST_F(InferenceSchedulerTest, QuiescentAccountingMatchesCompletions) {
    const auto scheduler = MakeAdd(2, 4);
    RequireStart(*scheduler);
    const SchedulerCompletion failed = Take(scheduler->submit({NamedTensor{"A", {2}, {1.F, 2.F}}}));
    EXPECT_EQ(failed.status, CompletionStatus::kInferenceFailed);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    ExpectSum(Take(scheduler->submit(AddInputs(0.F, 2.F, 0.F, 3.F))), 0.F, 5.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 2, 1, 0, 0);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
}

TEST_F(InferenceSchedulerTest, LastWorkerCloseCancelAllocationUsesDrain) {
    auto scheduler = MakeAdd(1, 2);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    SchedulerTestAccess::SetFailNextCloseCancel(*scheduler);
    RequireStart(*scheduler);

    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(1.F, 1.F, 1.F, 1.F));
    release.release();

    EXPECT_EQ(Take(std::move(first)).status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(Take(std::move(second)).status, CompletionStatus::kSchedulerFailed);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));

    const ShutdownResult shutdown = scheduler->shutdown(ShutdownPolicy::kDrain);
    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 0, 0, 1, 1);
}

} // namespace
