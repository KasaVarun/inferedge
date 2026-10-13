#include "inferedge/inference_scheduler.hpp"

#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <latch>
#include <memory>
#include <mutex>
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
#ifndef INFEREDGE_BATCH_ADD_MODEL_PATH
#error "CMake must define INFEREDGE_BATCH_ADD_MODEL_PATH"
#endif

namespace {

using inferedge::Admission;
using inferedge::AdmissionStatus;
using inferedge::CompletionStatus;
using inferedge::ElementType;
using inferedge::InferenceErrorKind;
using inferedge::InferenceScheduler;
using inferedge::kMaxSchedulerQueueCapacity;
using inferedge::kMaxSchedulerWorkers;
using inferedge::kSchedulerMetricsMaxBatchSize;
using inferedge::NamedTensor;
using inferedge::SchedulerCompletion;
using inferedge::SchedulerConfig;
using inferedge::SchedulerError;
using inferedge::SchedulerErrorKind;
using inferedge::SchedulerMetrics;
using inferedge::SchedulerStats;
using inferedge::SchedulerTestAccess;
using inferedge::ShutdownPolicy;
using inferedge::ShutdownResult;
using inferedge::TensorMetadata;

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
                     std::uint64_t failed_scheduler, std::uint64_t canceled = 0) {
    const SchedulerStats stats = scheduler.stats();
    EXPECT_EQ(stats.succeeded, succeeded);
    EXPECT_EQ(stats.failed_inference, failed_inference);
    EXPECT_EQ(stats.failed_worker, failed_worker);
    EXPECT_EQ(stats.failed_scheduler, failed_scheduler);
    EXPECT_EQ(stats.canceled, canceled);
    EXPECT_EQ(stats.accepted,
              succeeded + failed_inference + failed_worker + failed_scheduler + canceled);
    EXPECT_EQ(stats.queue_depth, 0U);
    EXPECT_EQ(stats.executing, 0U);
    EXPECT_EQ(stats.live_workers, 0U);
}

void ExpectWinner(const ShutdownResult& result, ShutdownPolicy policy, bool changed) {
    EXPECT_EQ(result.error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(result.established, true);
    EXPECT_EQ(result.changed, changed);
    EXPECT_EQ(result.policy, policy);
}

void ExpectCanceled(const SchedulerCompletion& completion, std::uint64_t sequence) {
    EXPECT_EQ(completion.status, CompletionStatus::kCanceled);
    EXPECT_EQ(completion.sequence, sequence);
    EXPECT_EQ(completion.request_id, sequence);
    EXPECT_TRUE(completion.outputs.empty());
    EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kNone);
    EXPECT_TRUE(completion.inference_error.message.empty());
}

// Close must commit while the worker is still held, then the hold is released
// so shutdown can join.
ShutdownResult CancelWhileBlocked(InferenceScheduler& scheduler, LatchRelease& release) {
    auto closed = std::make_shared<std::latch>(1);
    SchedulerTestAccess::SetAfterDrainClosed(scheduler, closed);
    JoinThread stopper;
    ShutdownResult shutdown;
    stopper.thread =
        std::thread([&] { shutdown = scheduler.shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    release.release();
    stopper.thread.join();
    return shutdown;
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
    EXPECT_FALSE(shutdown.established);
    ExpectQuiescent(*scheduler, 0, 0, 1, 1);
}

TEST_F(InferenceSchedulerTest, CancelEmptyQueue) {
    const auto scheduler = MakeAdd(1, 1);
    RequireStart(*scheduler);
    const ShutdownResult shutdown = scheduler->shutdown(ShutdownPolicy::kCancelPending);
    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    ExpectQuiescent(*scheduler, 0, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, CancelSeveralWaitingRequests) {
    auto scheduler = MakeAdd(1, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);

    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(0.F, 1.F, 1.F, 1.F));
    Admission third = scheduler->submit(AddInputs(2.F, 0.F, 2.F, 0.F));
    EXPECT_EQ(scheduler->stats().queue_depth, 3U);

    const ShutdownResult shutdown = CancelWhileBlocked(*scheduler, release);
    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    ExpectCanceled(Take(std::move(first)), 1);
    ExpectCanceled(Take(std::move(second)), 2);
    ExpectCanceled(Take(std::move(third)), 3);
    ExpectQuiescent(*scheduler, 0, 0, 0, 0, 3);
}

TEST_F(InferenceSchedulerTest, PoppedRequestSurvivesCancel) {
    auto scheduler = MakeAdd(1, 2);
    auto reached = std::make_shared<std::latch>(1);
    auto release_run = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    JoinThread stopper;
    LatchRelease release{release_run};
    SchedulerTestAccess::SetReachedRun(*scheduler, reached);
    SchedulerTestAccess::SetBeforeRun(*scheduler, release_run);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    RequireStart(*scheduler);

    Admission popped = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    reached->wait();
    Admission waiting = scheduler->submit(AddInputs(1.F, 0.F, 1.F, 0.F));

    ShutdownResult shutdown;
    stopper.thread =
        std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    release.release();
    stopper.thread.join();

    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    ExpectSum(Take(std::move(popped)), 4.F, 6.F);
    ExpectCanceled(Take(std::move(waiting)), 2);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0, 1);
}

TEST_F(InferenceSchedulerTest, InFlightInferenceSurvivesCancel) {
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
    stopper.thread =
        std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    release.release();
    stopper.thread.join();

    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    ExpectSum(Take(std::move(admission)), 4.F, 6.F);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, FutureReturnsCanceled) {
    auto scheduler = MakeAdd(1, 1);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(AddInputs(1.F, 1.F, 1.F, 1.F));
    const ShutdownResult shutdown = CancelWhileBlocked(*scheduler, release);
    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    const SchedulerCompletion completion = Take(std::move(admission));
    EXPECT_EQ(completion.status, CompletionStatus::kCanceled);
}

TEST_F(InferenceSchedulerTest, CanceledResultHasNoOutputs) {
    auto scheduler = MakeAdd(1, 1);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(AddInputs(1.F, 1.F, 1.F, 1.F));
    EXPECT_EQ(CancelWhileBlocked(*scheduler, release).error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(Take(std::move(admission)).outputs.empty());
}

TEST_F(InferenceSchedulerTest, CanceledInferenceErrorIsNone) {
    auto scheduler = MakeAdd(1, 1);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(AddInputs(1.F, 1.F, 1.F, 1.F));
    EXPECT_EQ(CancelWhileBlocked(*scheduler, release).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(Take(std::move(admission)).inference_error.kind, InferenceErrorKind::kNone);
}

TEST_F(InferenceSchedulerTest, SubmissionAfterCancelIsRejected) {
    const auto scheduler = MakeAdd(1, 1);
    RequireStart(*scheduler);
    ExpectWinner(scheduler->shutdown(ShutdownPolicy::kCancelPending),
                 ShutdownPolicy::kCancelPending, true);
    const Admission rejected = scheduler->submit(AddInputs(1.F, 1.F, 1.F, 1.F));
    EXPECT_EQ(rejected.status, AdmissionStatus::kRejectedClosed);
    EXPECT_FALSE(rejected.completion.has_value());
    EXPECT_EQ(scheduler->stats().rejected_closed, 1U);
}

TEST_F(InferenceSchedulerTest, RejectedSubmissionAfterCancelReturnsInputs) {
    const auto scheduler = MakeAdd(1, 1);
    RequireStart(*scheduler);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kCancelPending).error.kind,
              SchedulerErrorKind::kNone);
    const Admission rejected = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    EXPECT_EQ(rejected.status, AdmissionStatus::kRejectedClosed);
    EXPECT_FALSE(rejected.completion.has_value());
    ASSERT_TRUE(rejected.inputs.has_value());
    ASSERT_EQ(rejected.inputs->size(), 2U);
    EXPECT_EQ(rejected.inputs->front().name, "A");
    EXPECT_FLOAT_EQ(rejected.inputs->front().values[0], 1.F);
}

TEST_F(InferenceSchedulerTest, DrainThenCancelDrainWins) {
    auto scheduler = MakeAdd(1, 2);
    auto blocked = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto follower = std::make_shared<std::latch>(1);
    JoinThread drain_call;
    JoinThread cancel_call;
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetShutdownFollowerEntered(*scheduler, follower);
    RequireStart(*scheduler);
    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(1.F, 0.F, 1.F, 0.F));

    ShutdownResult drain;
    ShutdownResult cancel;
    drain_call.thread = std::thread([&] { drain = scheduler->shutdown(ShutdownPolicy::kDrain); });
    closed->wait();
    cancel_call.thread =
        std::thread([&] { cancel = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    follower->wait();
    release.release();
    drain_call.thread.join();
    cancel_call.thread.join();

    ExpectWinner(drain, ShutdownPolicy::kDrain, true);
    ExpectWinner(cancel, ShutdownPolicy::kDrain, false);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    ExpectSum(Take(std::move(first)), 4.F, 6.F);
    ExpectSum(Take(std::move(second)), 2.F, 0.F);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
}

TEST_F(InferenceSchedulerTest, CancelThenDrainCancelWins) {
    auto scheduler = MakeAdd(1, 2);
    auto blocked = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto follower = std::make_shared<std::latch>(1);
    JoinThread cancel_call;
    JoinThread drain_call;
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetShutdownFollowerEntered(*scheduler, follower);
    RequireStart(*scheduler);
    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(4.F, 0.F, 1.F, 0.F));

    ShutdownResult cancel;
    ShutdownResult drain;
    cancel_call.thread =
        std::thread([&] { cancel = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    drain_call.thread = std::thread([&] { drain = scheduler->shutdown(ShutdownPolicy::kDrain); });
    follower->wait();
    release.release();
    cancel_call.thread.join();
    drain_call.thread.join();

    ExpectWinner(cancel, ShutdownPolicy::kCancelPending, true);
    ExpectWinner(drain, ShutdownPolicy::kCancelPending, false);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    ExpectCanceled(Take(std::move(first)), 1);
    ExpectCanceled(Take(std::move(second)), 2);
}

TEST_F(InferenceSchedulerTest, ConcurrentMixedPolicyShutdownAgrees) {
    auto scheduler = MakeAdd(2, 1);
    JoinThread drain_call;
    JoinThread cancel_call;
    RequireStart(*scheduler);
    ShutdownResult drain;
    ShutdownResult cancel;
    drain_call.thread = std::thread([&] { drain = scheduler->shutdown(ShutdownPolicy::kDrain); });
    cancel_call.thread =
        std::thread([&] { cancel = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    drain_call.thread.join();
    cancel_call.thread.join();

    EXPECT_EQ(drain.error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(cancel.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(drain.established);
    EXPECT_TRUE(cancel.established);
    EXPECT_EQ(drain.policy, cancel.policy);
    EXPECT_NE(drain.changed, cancel.changed);
    if (drain.policy == ShutdownPolicy::kDrain) {
        EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    } else {
        EXPECT_TRUE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    }
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
}

TEST_F(InferenceSchedulerTest, RepeatedCancelShutdownIsIdempotent) {
    const auto scheduler = MakeAdd(1, 1);
    RequireStart(*scheduler);
    const ShutdownResult first = scheduler->shutdown(ShutdownPolicy::kCancelPending);
    const ShutdownResult second = scheduler->shutdown(ShutdownPolicy::kCancelPending);
    ExpectWinner(first, ShutdownPolicy::kCancelPending, true);
    ExpectWinner(second, ShutdownPolicy::kCancelPending, false);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
}

TEST_F(InferenceSchedulerTest, ShutdownFollowersObserveWinner) {
    auto scheduler = MakeAdd(1, 1);
    auto blocked = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto followers = std::make_shared<std::latch>(3);
    JoinThread leader;
    JoinThread follower_a;
    JoinThread follower_b;
    JoinThread follower_c;
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetShutdownFollowerEntered(*scheduler, followers);
    RequireStart(*scheduler);

    ShutdownResult lead;
    ShutdownResult first;
    ShutdownResult second;
    ShutdownResult third;
    leader.thread =
        std::thread([&] { lead = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    follower_a.thread = std::thread([&] { first = scheduler->shutdown(ShutdownPolicy::kDrain); });
    follower_b.thread =
        std::thread([&] { second = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    follower_c.thread = std::thread([&] { third = scheduler->shutdown(ShutdownPolicy::kDrain); });
    followers->wait();
    release.release();
    leader.thread.join();
    follower_a.thread.join();
    follower_b.thread.join();
    follower_c.thread.join();

    ExpectWinner(lead, ShutdownPolicy::kCancelPending, true);
    ExpectWinner(first, ShutdownPolicy::kCancelPending, false);
    ExpectWinner(second, ShutdownPolicy::kCancelPending, false);
    ExpectWinner(third, ShutdownPolicy::kCancelPending, false);
}

TEST_F(InferenceSchedulerTest, WorkerExceptionDuringCancel) {
    auto scheduler = MakeAdd(1, 2);
    auto reached = std::make_shared<std::latch>(1);
    auto release_run = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    JoinThread stopper;
    LatchRelease release{release_run};
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    SchedulerTestAccess::SetReachedRun(*scheduler, reached);
    SchedulerTestAccess::SetBeforeRun(*scheduler, release_run);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    RequireStart(*scheduler);

    Admission popped = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    reached->wait();
    Admission waiting = scheduler->submit(AddInputs(1.F, 0.F, 1.F, 0.F));
    ShutdownResult shutdown;
    stopper.thread =
        std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    release.release();
    stopper.thread.join();

    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    const SchedulerCompletion failed = Take(std::move(popped));
    EXPECT_EQ(failed.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed.inference_error.message, "injected");
    ExpectCanceled(Take(std::move(waiting)), 2);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::IsFailed(*scheduler));
    ExpectQuiescent(*scheduler, 0, 0, 1, 0, 1);
}

TEST_F(InferenceSchedulerTest, LastWorkerRetiresAfterCancelCommits) {
    auto scheduler = MakeAdd(1, 2);
    auto blocked = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    JoinThread stopper;
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    RequireStart(*scheduler);
    Admission first = scheduler->submit(AddInputs(1.F, 0.F, 1.F, 0.F));
    Admission second = scheduler->submit(AddInputs(2.F, 0.F, 2.F, 0.F));

    ShutdownResult shutdown;
    stopper.thread =
        std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    EXPECT_TRUE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    release.release();
    stopper.thread.join();

    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    ExpectCanceled(Take(std::move(first)), 1);
    ExpectCanceled(Take(std::move(second)), 2);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::IsFailed(*scheduler));
    ExpectQuiescent(*scheduler, 0, 0, 0, 0, 2);
}

TEST_F(InferenceSchedulerTest, CompletedResultUnchangedByCancel) {
    const auto scheduler = MakeAdd(1, 1);
    RequireStart(*scheduler);
    const SchedulerCompletion completed = Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F)));
    ExpectSum(completed, 4.F, 6.F);
    ExpectWinner(scheduler->shutdown(ShutdownPolicy::kCancelPending),
                 ShutdownPolicy::kCancelPending, true);
    ExpectSum(completed, 4.F, 6.F);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, DestructorAfterCancelPreservesCompletion) {
    std::optional<std::future<SchedulerCompletion>> kept;
    {
        auto scheduler = MakeAdd(1, 1);
        auto blocked = std::make_shared<std::latch>(1);
        LatchRelease release{blocked};
        SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
        RequireStart(*scheduler);
        Admission admission = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
        kept = std::move(admission.completion);
        ExpectWinner(CancelWhileBlocked(*scheduler, release), ShutdownPolicy::kCancelPending, true);
    }
    ASSERT_TRUE(kept.has_value());
    ExpectCanceled(kept->get(), 1);
}

TEST_F(InferenceSchedulerTest, FailedStartupShutdownEstablishesNoPolicy) {
    const auto scheduler = MakeScheduler(SchedulerConfig{INFEREDGE_INVALID_MODEL_PATH, 1, 1});
    EXPECT_EQ(scheduler->start().kind, SchedulerErrorKind::kModelLoad);
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
    const ShutdownResult cancel = scheduler->shutdown(ShutdownPolicy::kCancelPending);
    EXPECT_EQ(cancel.error.kind, SchedulerErrorKind::kNone);
    EXPECT_FALSE(cancel.established);
    EXPECT_FALSE(cancel.changed);
    const ShutdownResult drain = scheduler->shutdown(ShutdownPolicy::kDrain);
    EXPECT_EQ(drain.error.kind, SchedulerErrorKind::kNone);
    EXPECT_FALSE(drain.established);
    EXPECT_FALSE(drain.changed);
    EXPECT_EQ(scheduler->start().kind, SchedulerErrorKind::kInvalidState);
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
}

TEST_F(InferenceSchedulerTest, QuiescentAccountingIncludesCanceled) {
    std::mutex gate_mu;
    std::condition_variable gate_cv;
    int permits = 0;
    bool release_all = false;
    bool waiting = false;
    auto scheduler = MakeAdd(1, 4);
    struct ReleaseGate {
        std::mutex& mu;
        std::condition_variable& cv;
        bool& release_all;
        ~ReleaseGate() {
            std::lock_guard<std::mutex> lock(mu);
            release_all = true;
            cv.notify_all();
        }
    } release_gate{gate_mu, gate_cv, release_all};
    SchedulerTestAccess::SetBeforeLoop(*scheduler, [&] {
        std::unique_lock<std::mutex> lock(gate_mu);
        waiting = true;
        gate_cv.notify_all();
        gate_cv.wait(lock, [&] { return permits > 0 || release_all; });
        waiting = false;
        if (permits > 0) {
            --permits;
        }
        gate_cv.notify_all();
    });
    RequireStart(*scheduler);
    {
        std::unique_lock<std::mutex> lock(gate_mu);
        gate_cv.wait(lock, [&] { return waiting; });
    }
    Admission failed = scheduler->submit({NamedTensor{"A", {2}, {1.F, 2.F}}});
    Admission success = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission canceled = scheduler->submit(AddInputs(1.F, 0.F, 1.F, 0.F));
    EXPECT_EQ(scheduler->stats().queue_depth, 3U);

    auto allow_one = [&] {
        std::unique_lock<std::mutex> lock(gate_mu);
        permits = 1;
        waiting = false;
        gate_cv.notify_all();
        gate_cv.wait(lock, [&] { return waiting && permits == 0; });
    };
    allow_one();
    EXPECT_EQ(Take(std::move(failed)).status, CompletionStatus::kInferenceFailed);
    allow_one();
    ExpectSum(Take(std::move(success)), 4.F, 6.F);

    auto closed = std::make_shared<std::latch>(1);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    JoinThread stopper;
    ShutdownResult shutdown;
    stopper.thread =
        std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    {
        std::lock_guard<std::mutex> lock(gate_mu);
        release_all = true;
    }
    gate_cv.notify_all();
    stopper.thread.join();
    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    ExpectCanceled(Take(std::move(canceled)), 3);
    ExpectQuiescent(*scheduler, 1, 1, 0, 0, 1);
}

TEST_F(InferenceSchedulerTest, CanceledRequestCompletesOnce) {
    auto scheduler = MakeAdd(1, 1);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(AddInputs(2.F, 0.F, 2.F, 0.F));
    EXPECT_EQ(CancelWhileBlocked(*scheduler, release).error.kind, SchedulerErrorKind::kNone);
    ExpectCanceled(Take(std::move(admission)), 1);
    ExpectQuiescent(*scheduler, 0, 0, 0, 0, 1);
    scheduler.reset();
}

TEST_F(InferenceSchedulerTest, OrdinaryCancelDoesNotBreakPromise) {
    auto scheduler = MakeAdd(1, 1);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(AddInputs(1.F, 1.F, 1.F, 1.F));
    EXPECT_EQ(CancelWhileBlocked(*scheduler, release).error.kind, SchedulerErrorKind::kNone);
    ASSERT_TRUE(admission.completion.has_value());
    SchedulerCompletion completion;
    EXPECT_NO_THROW(completion = admission.completion->get());
    ExpectCanceled(completion, 1);
}

TEST_F(InferenceSchedulerTest, CanceledRequestsPreserveAdmissionSequence) {
    auto scheduler = MakeAdd(1, 3);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission first = scheduler->submit(AddInputs(1.F, 0.F, 0.F, 1.F));
    Admission second = scheduler->submit(AddInputs(1.F, 1.F, 1.F, 1.F));
    Admission third = scheduler->submit(AddInputs(2.F, 2.F, 2.F, 2.F));
    EXPECT_EQ(CancelWhileBlocked(*scheduler, release).error.kind, SchedulerErrorKind::kNone);
    const SchedulerCompletion left = Take(std::move(first));
    const SchedulerCompletion middle = Take(std::move(second));
    const SchedulerCompletion right = Take(std::move(third));
    ExpectCanceled(left, 1);
    ExpectCanceled(middle, 2);
    ExpectCanceled(right, 3);
    EXPECT_LT(left.sequence, middle.sequence);
    EXPECT_LT(middle.sequence, right.sequence);
}

TEST_F(InferenceSchedulerTest, CloseCancelFailureBeforeCommitStaysRunning) {
    auto scheduler = MakeAdd(1, 1);
    auto blocked = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    JoinThread stopper;
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetFailNextCloseCancel(*scheduler);
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    EXPECT_EQ(scheduler->stats().queue_depth, 1U);

    const ShutdownResult failed = scheduler->shutdown(ShutdownPolicy::kCancelPending);
    EXPECT_EQ(failed.error.kind, SchedulerErrorKind::kWorkerStart);
    EXPECT_EQ(failed.error.message, "shutdown close failed");
    EXPECT_FALSE(failed.established);
    EXPECT_FALSE(failed.changed);
    EXPECT_FALSE(SchedulerTestAccess::IsStopped(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::IsFailed(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    EXPECT_EQ(scheduler->stats().queue_depth, 1U);

    ShutdownResult shutdown;
    stopper.thread =
        std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    release.release();
    stopper.thread.join();
    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    ExpectCanceled(Take(std::move(admission)), 1);
    ExpectQuiescent(*scheduler, 0, 0, 0, 0, 1);
}

TEST_F(InferenceSchedulerTest, JoinSettleFailureStillReleasesFollowers) {
    auto scheduler = MakeAdd(1, 1);
    auto blocked = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto follower = std::make_shared<std::latch>(1);
    JoinThread leader;
    JoinThread follower_call;
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetShutdownFollowerEntered(*scheduler, follower);
    SchedulerTestAccess::SetFailNextJoinSettle(*scheduler);
    RequireStart(*scheduler);

    ShutdownResult lead;
    ShutdownResult follow;
    leader.thread =
        std::thread([&] { lead = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    follower_call.thread =
        std::thread([&] { follow = scheduler->shutdown(ShutdownPolicy::kDrain); });
    follower->wait();
    release.release();
    leader.thread.join();
    follower_call.thread.join();

    ExpectWinner(lead, ShutdownPolicy::kCancelPending, true);
    ExpectWinner(follow, ShutdownPolicy::kCancelPending, false);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
}

constexpr auto kCompletionBound = std::chrono::seconds{10};

void ExpectRejectedInputs(const Admission& admission, AdmissionStatus status, float a0) {
    EXPECT_EQ(admission.status, status);
    EXPECT_FALSE(admission.completion.has_value());
    ASSERT_TRUE(admission.inputs.has_value());
    ASSERT_EQ(admission.inputs->size(), 2U);
    EXPECT_FLOAT_EQ(admission.inputs->front().values[0], a0);
    EXPECT_FLOAT_EQ(admission.inputs->front().values[1], 0.F);
    EXPECT_FLOAT_EQ(admission.inputs->back().values[0], 1.F);
    EXPECT_FLOAT_EQ(admission.inputs->back().values[1], 0.F);
}

SchedulerCompletion TakeBounded(Admission admission) {
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

void ExpectUnique(const std::vector<std::uint64_t>& values) {
    for (std::size_t left = 0; left < values.size(); ++left) {
        for (std::size_t right = left + 1; right < values.size(); ++right) {
            EXPECT_NE(values[left], values[right]);
        }
    }
}

struct BeforePopThrow {
    std::shared_ptr<std::latch> arrived;
    std::shared_ptr<std::latch> release;
};

BeforePopThrow ArmThrowBeforePop(InferenceScheduler& scheduler, int workers, int throws) {
    BeforePopThrow gate;
    gate.arrived = std::make_shared<std::latch>(static_cast<std::ptrdiff_t>(workers));
    gate.release = std::make_shared<std::latch>(1);
    auto arrived = gate.arrived;
    auto release = gate.release;
    auto budget = std::make_shared<std::atomic<int>>(throws);
    auto visits = std::make_shared<std::atomic<int>>(0);
    // The hook runs on every loop. Only the first visit of each worker may
    // touch the latches; a later count_down would pass zero.
    SchedulerTestAccess::SetBeforeLoop(scheduler, [arrived, release, budget, visits, workers] {
        if (visits->fetch_add(1) >= workers) {
            return;
        }
        arrived->count_down();
        release->wait();
        if (budget->fetch_sub(1) > 0) {
            throw std::runtime_error("before pop");
        }
    });
    return gate;
}

TEST_F(InferenceSchedulerTest, ProducersFillABlockedQueue) {
    for (const int producers : {2, 4, 8, 16}) {
        SCOPED_TRACE(producers);
        auto scheduler = MakeAdd(4, 4);
        auto blocked = std::make_shared<std::latch>(1);
        LatchRelease release{blocked};
        SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
        RequireStart(*scheduler);

        const auto count = static_cast<std::size_t>(producers);
        std::vector<Admission> admissions(count);
        std::vector<JoinThread> threads(count);
        for (int index = 0; index < producers; ++index) {
            threads[static_cast<std::size_t>(index)].thread = std::thread([&, index] {
                admissions[static_cast<std::size_t>(index)] =
                    scheduler->submit(AddInputs(static_cast<float>(index), 0.F, 1.F, 0.F));
            });
        }
        for (JoinThread& thread : threads) {
            thread.thread.join();
        }

        const int accepted = producers < 4 ? producers : 4;
        const int rejected = producers - accepted;
        int accepted_count = 0;
        int rejected_count = 0;
        for (int index = 0; index < producers; ++index) {
            const Admission& admission = admissions[static_cast<std::size_t>(index)];
            if (admission.status == AdmissionStatus::kAccepted) {
                ++accepted_count;
                EXPECT_TRUE(admission.completion.has_value());
                EXPECT_FALSE(admission.inputs.has_value());
            } else {
                ++rejected_count;
                ExpectRejectedInputs(admission, AdmissionStatus::kRejectedFull,
                                     static_cast<float>(index));
            }
        }
        EXPECT_EQ(accepted_count, accepted);
        EXPECT_EQ(rejected_count, rejected);
        EXPECT_EQ(scheduler->stats().queue_depth, static_cast<std::uint64_t>(accepted));
        EXPECT_EQ(scheduler->stats().executing, 0U);
        EXPECT_EQ(scheduler->stats().rejected_full, static_cast<std::uint64_t>(rejected));

        release.release();
        std::vector<std::uint64_t> ids;
        std::vector<std::uint64_t> sequences;
        for (int index = 0; index < producers; ++index) {
            Admission& admission = admissions[static_cast<std::size_t>(index)];
            if (admission.status != AdmissionStatus::kAccepted) {
                continue;
            }
            const SchedulerCompletion completion = TakeBounded(std::move(admission));
            ExpectSum(completion, static_cast<float>(index) + 1.F, 0.F);
            EXPECT_EQ(completion.request_id, completion.sequence);
            ids.push_back(completion.request_id);
            sequences.push_back(completion.sequence);
        }
        ExpectUnique(ids);
        ExpectUnique(sequences);
        EXPECT_EQ(ids.size(), static_cast<std::size_t>(accepted));
        EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind,
                  SchedulerErrorKind::kNone);
        ExpectQuiescent(*scheduler, static_cast<std::uint64_t>(accepted), 0, 0, 0);
        EXPECT_EQ(scheduler->stats().rejected_full, static_cast<std::uint64_t>(rejected));
    }
}

TEST_F(InferenceSchedulerTest, ProducersSubmitUntilShutdown) {
    auto scheduler = MakeAdd(2, 4);
    constexpr int kProducers = 4;
    constexpr int kAttempts = 8;
    ShutdownResult drain;
    ShutdownResult cancel;
    std::vector<std::vector<Admission>> records(kProducers);
    std::vector<JoinThread> producers(kProducers);
    JoinThread drain_call;
    JoinThread cancel_call;
    RequireStart(*scheduler);
    for (int producer = 0; producer < kProducers; ++producer) {
        producers[static_cast<std::size_t>(producer)].thread = std::thread([&, producer] {
            auto& mine = records[static_cast<std::size_t>(producer)];
            mine.reserve(kAttempts);
            for (int attempt = 0; attempt < kAttempts; ++attempt) {
                const float tag = static_cast<float>(producer * kAttempts + attempt);
                mine.push_back(scheduler->submit(AddInputs(tag, 0.F, 1.F, 0.F)));
            }
        });
    }
    drain_call.thread = std::thread([&] { drain = scheduler->shutdown(ShutdownPolicy::kDrain); });
    cancel_call.thread =
        std::thread([&] { cancel = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    for (JoinThread& thread : producers) {
        thread.thread.join();
    }
    drain_call.thread.join();
    cancel_call.thread.join();

    EXPECT_EQ(drain.error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(cancel.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(drain.established);
    EXPECT_TRUE(cancel.established);
    EXPECT_EQ(drain.policy, cancel.policy);
    EXPECT_NE(drain.changed, cancel.changed);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));

    std::uint64_t accepted = 0;
    std::uint64_t rejected_full = 0;
    std::uint64_t rejected_closed = 0;
    std::uint64_t succeeded = 0;
    std::uint64_t canceled = 0;
    std::vector<std::uint64_t> ids;
    for (int producer = 0; producer < kProducers; ++producer) {
        for (Admission& admission : records[static_cast<std::size_t>(producer)]) {
            const float tag = static_cast<float>(
                producer * kAttempts +
                static_cast<int>(&admission - records[static_cast<std::size_t>(producer)].data()));
            if (admission.status == AdmissionStatus::kAccepted) {
                ++accepted;
                const SchedulerCompletion completion = TakeBounded(std::move(admission));
                EXPECT_EQ(completion.request_id, completion.sequence);
                ids.push_back(completion.request_id);
                if (completion.status == CompletionStatus::kSucceeded) {
                    ++succeeded;
                    ExpectSum(completion, tag + 1.F, 0.F);
                } else if (completion.status == CompletionStatus::kCanceled) {
                    ++canceled;
                    EXPECT_TRUE(completion.outputs.empty());
                    EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kNone);
                } else {
                    ADD_FAILURE() << "unexpected terminal status";
                }
            } else if (admission.status == AdmissionStatus::kRejectedFull) {
                ++rejected_full;
                ExpectRejectedInputs(admission, AdmissionStatus::kRejectedFull, tag);
            } else {
                ++rejected_closed;
                ExpectRejectedInputs(admission, AdmissionStatus::kRejectedClosed, tag);
            }
        }
    }
    EXPECT_EQ(accepted + rejected_full + rejected_closed,
              static_cast<std::uint64_t>(kProducers * kAttempts));
    ExpectUnique(ids);
    EXPECT_EQ(scheduler->stats().rejected_full, rejected_full);
    EXPECT_EQ(scheduler->stats().rejected_closed, rejected_closed);
    ExpectQuiescent(*scheduler, succeeded, 0, 0, 0, canceled);
}

TEST_F(InferenceSchedulerTest, CapacityExcludesExecutingRequests) {
    auto scheduler = MakeAdd(1, 2);
    auto before_pop = std::make_shared<std::latch>(1);
    auto before_run = std::make_shared<std::latch>(1);
    auto reached = std::make_shared<std::latch>(1);
    LatchRelease release_pop{before_pop};
    LatchRelease release_run{before_run};
    SchedulerTestAccess::SetBeforePop(*scheduler, before_pop);
    SchedulerTestAccess::SetBeforeRun(*scheduler, before_run);
    SchedulerTestAccess::SetReachedRun(*scheduler, reached);
    RequireStart(*scheduler);

    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(5.F, 6.F, 7.F, 8.F));
    EXPECT_EQ(first.status, AdmissionStatus::kAccepted);
    EXPECT_EQ(second.status, AdmissionStatus::kAccepted);
    EXPECT_EQ(scheduler->stats().queue_depth, 2U);
    EXPECT_EQ(scheduler->stats().executing, 0U);
    ExpectRejectedInputs(scheduler->submit(AddInputs(8.F, 0.F, 1.F, 0.F)),
                         AdmissionStatus::kRejectedFull, 8.F);

    release_pop.release();
    reached->wait();
    EXPECT_EQ(scheduler->stats().queue_depth, 1U);
    EXPECT_EQ(scheduler->stats().executing, 1U);
    Admission third = scheduler->submit(AddInputs(1.F, 0.F, 1.F, 0.F));
    EXPECT_EQ(third.status, AdmissionStatus::kAccepted);
    EXPECT_EQ(scheduler->stats().queue_depth, 2U);
    EXPECT_EQ(scheduler->stats().executing, 1U);
    ExpectRejectedInputs(scheduler->submit(AddInputs(9.F, 0.F, 1.F, 0.F)),
                         AdmissionStatus::kRejectedFull, 9.F);

    release_run.release();
    ExpectSum(TakeBounded(std::move(first)), 4.F, 6.F);
    ExpectSum(TakeBounded(std::move(second)), 12.F, 14.F);
    ExpectSum(TakeBounded(std::move(third)), 2.F, 0.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 3, 0, 0, 0);
    EXPECT_EQ(scheduler->stats().rejected_full, 2U);
}

TEST_F(InferenceSchedulerTest, Capacity1024RejectsTheNextWaitingRequest) {
    constexpr int kCapacity = 1024;
    auto scheduler = MakeAdd(1, static_cast<std::size_t>(kCapacity));
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);

    std::vector<Admission> admissions;
    admissions.reserve(static_cast<std::size_t>(kCapacity));
    for (int index = 0; index < kCapacity; ++index) {
        admissions.push_back(
            scheduler->submit(AddInputs(static_cast<float>(index), 0.F, 1.F, 0.F)));
        EXPECT_EQ(admissions.back().status, AdmissionStatus::kAccepted);
    }
    EXPECT_EQ(scheduler->stats().queue_depth, static_cast<std::uint64_t>(kCapacity));
    EXPECT_EQ(scheduler->stats().executing, 0U);
    EXPECT_EQ(scheduler->stats().accepted, static_cast<std::uint64_t>(kCapacity));
    ExpectRejectedInputs(scheduler->submit(AddInputs(static_cast<float>(kCapacity), 0.F, 1.F, 0.F)),
                         AdmissionStatus::kRejectedFull, static_cast<float>(kCapacity));

    release.release();
    std::vector<std::uint64_t> ids;
    for (int index = 0; index < kCapacity; ++index) {
        const SchedulerCompletion completion =
            TakeBounded(std::move(admissions[static_cast<std::size_t>(index)]));
        ExpectSum(completion, static_cast<float>(index) + 1.F, 0.F);
        EXPECT_EQ(completion.request_id, completion.sequence);
        ids.push_back(completion.request_id);
    }
    ExpectUnique(ids);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, static_cast<std::uint64_t>(kCapacity), 0, 0, 0);
    EXPECT_EQ(scheduler->stats().rejected_full, 1U);
}

TEST_F(InferenceSchedulerTest, DrainWithProducersAndFollowers) {
    auto scheduler = MakeAdd(1, 4);
    auto blocked = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto followers = std::make_shared<std::latch>(2);
    ShutdownResult lead;
    ShutdownResult follow_a;
    ShutdownResult follow_b;
    JoinThread leader;
    JoinThread follower_a;
    JoinThread follower_b;
    constexpr int kLater = 4;
    std::vector<Admission> later(static_cast<std::size_t>(kLater));
    std::vector<JoinThread> producers(static_cast<std::size_t>(kLater));
    // Release the worker before joining a shutdown thread. Joining first would
    // wait for that worker while it is still blocked on this latch.
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetShutdownFollowerEntered(*scheduler, followers);
    RequireStart(*scheduler);
    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(0.F, 1.F, 1.F, 1.F));
    EXPECT_EQ(scheduler->stats().queue_depth, 2U);

    leader.thread = std::thread([&] { lead = scheduler->shutdown(ShutdownPolicy::kDrain); });
    closed->wait();
    follower_a.thread =
        std::thread([&] { follow_a = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    follower_b.thread =
        std::thread([&] { follow_b = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    followers->wait();

    for (int index = 0; index < kLater; ++index) {
        producers[static_cast<std::size_t>(index)].thread = std::thread([&, index] {
            later[static_cast<std::size_t>(index)] =
                scheduler->submit(AddInputs(10.F + static_cast<float>(index), 0.F, 1.F, 0.F));
        });
    }
    for (JoinThread& thread : producers) {
        thread.thread.join();
    }
    for (int index = 0; index < kLater; ++index) {
        ExpectRejectedInputs(later[static_cast<std::size_t>(index)],
                             AdmissionStatus::kRejectedClosed, 10.F + static_cast<float>(index));
    }

    release.release();
    leader.thread.join();
    follower_a.thread.join();
    follower_b.thread.join();
    ExpectWinner(lead, ShutdownPolicy::kDrain, true);
    ExpectWinner(follow_a, ShutdownPolicy::kDrain, false);
    ExpectWinner(follow_b, ShutdownPolicy::kDrain, false);
    ExpectSum(TakeBounded(std::move(first)), 4.F, 6.F);
    ExpectSum(TakeBounded(std::move(second)), 1.F, 2.F);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
    EXPECT_EQ(scheduler->stats().canceled, 0U);
    EXPECT_EQ(scheduler->stats().rejected_closed, static_cast<std::uint64_t>(kLater));
}

TEST_F(InferenceSchedulerTest, CancelWithProducersAndInFlight) {
    auto scheduler = MakeAdd(1, 4);
    auto reached = std::make_shared<std::latch>(1);
    auto before_run = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto followers = std::make_shared<std::latch>(2);
    ShutdownResult lead;
    ShutdownResult follow_a;
    ShutdownResult follow_b;
    JoinThread leader;
    JoinThread follower_a;
    JoinThread follower_b;
    constexpr int kLater = 4;
    std::vector<Admission> later(static_cast<std::size_t>(kLater));
    std::vector<JoinThread> producers(static_cast<std::size_t>(kLater));
    // Release the in-flight worker before joining shutdown.
    LatchRelease release{before_run};
    SchedulerTestAccess::SetReachedRun(*scheduler, reached);
    SchedulerTestAccess::SetBeforeRun(*scheduler, before_run);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetShutdownFollowerEntered(*scheduler, followers);
    RequireStart(*scheduler);

    Admission popped = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    reached->wait();
    Admission waiting_a = scheduler->submit(AddInputs(1.F, 0.F, 1.F, 0.F));
    Admission waiting_b = scheduler->submit(AddInputs(2.F, 0.F, 2.F, 0.F));
    EXPECT_EQ(scheduler->stats().queue_depth, 2U);
    EXPECT_EQ(scheduler->stats().executing, 1U);

    leader.thread =
        std::thread([&] { lead = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    EXPECT_TRUE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    follower_a.thread =
        std::thread([&] { follow_a = scheduler->shutdown(ShutdownPolicy::kDrain); });
    follower_b.thread =
        std::thread([&] { follow_b = scheduler->shutdown(ShutdownPolicy::kDrain); });
    followers->wait();

    for (int index = 0; index < kLater; ++index) {
        producers[static_cast<std::size_t>(index)].thread = std::thread([&, index] {
            later[static_cast<std::size_t>(index)] =
                scheduler->submit(AddInputs(20.F + static_cast<float>(index), 0.F, 1.F, 0.F));
        });
    }
    for (JoinThread& thread : producers) {
        thread.thread.join();
    }

    release.release();
    leader.thread.join();
    follower_a.thread.join();
    follower_b.thread.join();

    ExpectWinner(lead, ShutdownPolicy::kCancelPending, true);
    ExpectWinner(follow_a, ShutdownPolicy::kCancelPending, false);
    ExpectWinner(follow_b, ShutdownPolicy::kCancelPending, false);
    ExpectSum(TakeBounded(std::move(popped)), 4.F, 6.F);
    ExpectCanceled(TakeBounded(std::move(waiting_a)), 2);
    ExpectCanceled(TakeBounded(std::move(waiting_b)), 3);
    for (int index = 0; index < kLater; ++index) {
        ExpectRejectedInputs(later[static_cast<std::size_t>(index)],
                             AdmissionStatus::kRejectedClosed, 20.F + static_cast<float>(index));
    }
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectQuiescent(*scheduler, 1, 0, 0, 0, 2);
    EXPECT_EQ(scheduler->stats().rejected_closed, static_cast<std::uint64_t>(kLater));
}

TEST_F(InferenceSchedulerTest, UnorderedMixedShutdownSettlesEveryFuture) {
    auto scheduler = MakeAdd(2, 4);
    constexpr int kProducers = 4;
    constexpr int kAttempts = 8;
    ShutdownResult drain;
    ShutdownResult cancel;
    std::vector<std::vector<Admission>> records(kProducers);
    std::vector<JoinThread> producers(kProducers);
    JoinThread drain_call;
    JoinThread cancel_call;
    RequireStart(*scheduler);
    for (int producer = 0; producer < kProducers; ++producer) {
        producers[static_cast<std::size_t>(producer)].thread = std::thread([&, producer] {
            auto& mine = records[static_cast<std::size_t>(producer)];
            mine.reserve(kAttempts);
            for (int attempt = 0; attempt < kAttempts; ++attempt) {
                const float tag = static_cast<float>(producer * kAttempts + attempt);
                mine.push_back(scheduler->submit(AddInputs(tag, 0.F, 1.F, 0.F)));
            }
        });
    }
    drain_call.thread = std::thread([&] { drain = scheduler->shutdown(ShutdownPolicy::kDrain); });
    cancel_call.thread =
        std::thread([&] { cancel = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    for (JoinThread& thread : producers) {
        thread.thread.join();
    }
    drain_call.thread.join();
    cancel_call.thread.join();

    EXPECT_EQ(drain.error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(cancel.error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(drain.established);
    EXPECT_TRUE(cancel.established);
    EXPECT_EQ(drain.policy, cancel.policy);
    EXPECT_NE(drain.changed, cancel.changed);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    if (drain.policy == ShutdownPolicy::kDrain) {
        EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    } else {
        EXPECT_TRUE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    }

    std::uint64_t accepted = 0;
    std::uint64_t rejected_full = 0;
    std::uint64_t rejected_closed = 0;
    std::uint64_t succeeded = 0;
    std::uint64_t canceled = 0;
    std::vector<std::uint64_t> ids;
    for (int producer = 0; producer < kProducers; ++producer) {
        int attempt = 0;
        for (Admission& admission : records[static_cast<std::size_t>(producer)]) {
            const float tag = static_cast<float>(producer * kAttempts + attempt);
            ++attempt;
            if (admission.status == AdmissionStatus::kAccepted) {
                ++accepted;
                const SchedulerCompletion completion = TakeBounded(std::move(admission));
                EXPECT_EQ(completion.request_id, completion.sequence);
                ids.push_back(completion.request_id);
                if (completion.status == CompletionStatus::kSucceeded) {
                    ++succeeded;
                    ExpectSum(completion, tag + 1.F, 0.F);
                } else if (completion.status == CompletionStatus::kCanceled) {
                    ++canceled;
                    EXPECT_TRUE(completion.outputs.empty());
                    EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kNone);
                    EXPECT_EQ(drain.policy, ShutdownPolicy::kCancelPending);
                } else {
                    ADD_FAILURE() << "unexpected terminal status";
                }
            } else if (admission.status == AdmissionStatus::kRejectedFull) {
                ++rejected_full;
                ExpectRejectedInputs(admission, AdmissionStatus::kRejectedFull, tag);
            } else {
                ++rejected_closed;
                ExpectRejectedInputs(admission, AdmissionStatus::kRejectedClosed, tag);
            }
        }
    }
    EXPECT_EQ(accepted + rejected_full + rejected_closed,
              static_cast<std::uint64_t>(kProducers * kAttempts));
    ExpectUnique(ids);
    EXPECT_EQ(scheduler->stats().rejected_full, rejected_full);
    EXPECT_EQ(scheduler->stats().rejected_closed, rejected_closed);
    ExpectQuiescent(*scheduler, succeeded, 0, 0, 0, canceled);
}

TEST_F(InferenceSchedulerTest, OneWorkerRetiresBeforePop) {
    auto scheduler = MakeAdd(2, 2);
    const BeforePopThrow gate = ArmThrowBeforePop(*scheduler, 2, 1);
    LatchRelease release{gate.release};
    RequireStart(*scheduler);
    gate.arrived->wait();
    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(1.F, 0.F, 1.F, 0.F));
    EXPECT_EQ(scheduler->stats().queue_depth, 2U);
    release.release();
    ExpectSum(TakeBounded(std::move(first)), 4.F, 6.F);
    ExpectSum(TakeBounded(std::move(second)), 2.F, 0.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::IsFailed(*scheduler));
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, SeveralWorkersRetireBeforePop) {
    auto scheduler = MakeAdd(4, 4);
    const BeforePopThrow gate = ArmThrowBeforePop(*scheduler, 4, 3);
    LatchRelease release{gate.release};
    RequireStart(*scheduler);
    gate.arrived->wait();
    std::vector<Admission> admissions;
    for (int index = 0; index < 4; ++index) {
        admissions.push_back(
            scheduler->submit(AddInputs(static_cast<float>(index), 0.F, 1.F, 0.F)));
    }
    EXPECT_EQ(scheduler->stats().queue_depth, 4U);
    release.release();
    for (int index = 0; index < 4; ++index) {
        ExpectSum(TakeBounded(std::move(admissions[static_cast<std::size_t>(index)])),
                  static_cast<float>(index) + 1.F, 0.F);
    }
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::IsFailed(*scheduler));
    ExpectQuiescent(*scheduler, 4, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, LastWorkerThrowsBeforePop) {
    auto scheduler = MakeAdd(1, 2);
    const BeforePopThrow gate = ArmThrowBeforePop(*scheduler, 1, 1);
    LatchRelease release{gate.release};
    RequireStart(*scheduler);
    gate.arrived->wait();
    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(4.F, 0.F, 1.F, 0.F));
    EXPECT_EQ(scheduler->stats().queue_depth, 2U);
    release.release();
    const SchedulerCompletion left = TakeBounded(std::move(first));
    const SchedulerCompletion right = TakeBounded(std::move(second));
    EXPECT_EQ(left.status, CompletionStatus::kSchedulerFailed);
    EXPECT_EQ(right.status, CompletionStatus::kSchedulerFailed);
    EXPECT_EQ(left.inference_error.message, "no worker");
    EXPECT_EQ(right.inference_error.message, "no worker");
    EXPECT_TRUE(left.outputs.empty());
    EXPECT_TRUE(right.outputs.empty());
    const ShutdownResult shutdown = scheduler->shutdown(ShutdownPolicy::kDrain);
    EXPECT_EQ(shutdown.error.kind, SchedulerErrorKind::kNone);
    EXPECT_FALSE(shutdown.established);
    EXPECT_FALSE(shutdown.changed);
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    ExpectQuiescent(*scheduler, 0, 0, 0, 2);
}

TEST_F(InferenceSchedulerTest, TwoPoppedWorkersFailWhileSiblingContinues) {
    auto scheduler = MakeAdd(3, 3);
    auto gate0 = std::make_shared<std::latch>(1);
    auto gate1 = std::make_shared<std::latch>(1);
    auto gate2 = std::make_shared<std::latch>(1);
    LatchRelease release0{gate0};
    LatchRelease release1{gate1};
    LatchRelease release2{gate2};
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 0, gate0);
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 1, gate1);
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 2, gate2);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 1);
    RequireStart(*scheduler);

    release0.release();
    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    const SchedulerCompletion failed_first = TakeBounded(std::move(first));
    EXPECT_EQ(failed_first.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed_first.inference_error.message, "injected");
    EXPECT_TRUE(failed_first.outputs.empty());

    release1.release();
    Admission second = scheduler->submit(AddInputs(5.F, 0.F, 1.F, 0.F));
    const SchedulerCompletion failed_second = TakeBounded(std::move(second));
    EXPECT_EQ(failed_second.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed_second.inference_error.message, "injected");

    release2.release();
    Admission third = scheduler->submit(AddInputs(2.F, 0.F, 2.F, 0.F));
    ExpectSum(TakeBounded(std::move(third)), 4.F, 0.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    EXPECT_FALSE(SchedulerTestAccess::IsFailed(*scheduler));
    ExpectQuiescent(*scheduler, 1, 0, 2, 0);
}

TEST_F(InferenceSchedulerTest, DroppedFutureStillCompletes) {
    auto scheduler = MakeAdd(1, 2);
    RequireStart(*scheduler);
    Admission kept = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission dropped = scheduler->submit(AddInputs(1.F, 0.F, 0.F, 1.F));
    ASSERT_TRUE(dropped.completion.has_value());
    dropped.completion.reset();
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectSum(TakeBounded(std::move(kept)), 4.F, 6.F);
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
    EXPECT_EQ(scheduler->stats().canceled, 0U);
}

TEST_F(InferenceSchedulerTest, AcceptedOutputsDoNotAlias) {
    auto scheduler = MakeAdd(1, 2);
    RequireStart(*scheduler);
    SchedulerCompletion left = TakeBounded(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F)));
    SchedulerCompletion right = TakeBounded(scheduler->submit(AddInputs(5.F, 6.F, 7.F, 8.F)));
    ASSERT_EQ(left.outputs.size(), 1U);
    ASSERT_EQ(right.outputs.size(), 1U);
    const float right_first = right.outputs[0].values[0];
    const float right_second = right.outputs[0].values[1];
    left.outputs[0].values[0] = 42.F;
    EXPECT_FLOAT_EQ(right.outputs[0].values[0], right_first);
    EXPECT_FLOAT_EQ(right.outputs[0].values[1], right_second);
    EXPECT_FLOAT_EQ(left.outputs[0].values[0], 42.F);
    scheduler.reset();
    EXPECT_FLOAT_EQ(left.outputs[0].values[0], 42.F);
    EXPECT_FLOAT_EQ(right.outputs[0].values[0], right_first);
    EXPECT_FLOAT_EQ(right.outputs[0].values[1], right_second);
}

TEST_F(InferenceSchedulerTest, MnistInputReleasedAfterAccept) {
    constexpr std::size_t kInputCount = 784;
    constexpr std::size_t kLogitCount = 10;
    const std::filesystem::path root{INFEREDGE_MNIST_FIXTURE_DIR};
    std::vector<float> image = ReadFloats(root / "inputs" / "test_00000.f32", kInputCount);
    const std::vector<float> reference = ReadFloats(root / "onnx" / "test_00000.f32", kLogitCount);
    ASSERT_EQ(image.size(), kInputCount);
    ASSERT_EQ(reference.size(), kLogitCount);

    auto scheduler = MakeScheduler(SchedulerConfig{INFEREDGE_MNIST_MODEL_PATH, 1, 1});
    RequireStart(*scheduler);
    std::vector<NamedTensor> caller{
        NamedTensor{"image", {static_cast<std::int64_t>(kInputCount)}, image}};
    Admission admission = scheduler->submit(caller);
    caller.clear();
    caller.shrink_to_fit();
    image.clear();
    image.shrink_to_fit();
    SchedulerCompletion completion = TakeBounded(std::move(admission));
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
    const float first_logit = completion.outputs[0].values[0];
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
    scheduler.reset();
    EXPECT_FLOAT_EQ(completion.outputs[0].values[0], first_logit);
    EXPECT_EQ(ArgMax(completion.outputs[0].values), 7);
}

TEST_F(InferenceSchedulerTest, MnistQueuedCancelHasEmptyOutputs) {
    constexpr std::size_t kInputCount = 784;
    const std::filesystem::path root{INFEREDGE_MNIST_FIXTURE_DIR};
    const std::vector<float> image = ReadFloats(root / "inputs" / "test_00000.f32", kInputCount);
    ASSERT_EQ(image.size(), kInputCount);
    auto scheduler = MakeScheduler(SchedulerConfig{INFEREDGE_MNIST_MODEL_PATH, 1, 1});
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission admission =
        scheduler->submit({NamedTensor{"image", {static_cast<std::int64_t>(kInputCount)}, image}});
    EXPECT_EQ(scheduler->stats().queue_depth, 1U);
    EXPECT_EQ(scheduler->stats().executing, 0U);
    ExpectWinner(CancelWhileBlocked(*scheduler, release), ShutdownPolicy::kCancelPending, true);
    ExpectCanceled(TakeBounded(std::move(admission)), 1);
    ExpectQuiescent(*scheduler, 0, 0, 0, 0, 1);
}

SchedulerConfig WithBatching(std::filesystem::path model, bool enabled, std::uint32_t batch_size,
                             std::chrono::milliseconds delay, std::uint32_t workers = 1) {
    SchedulerConfig config{std::move(model), workers, 1};
    config.enable_batching = enabled;
    config.max_batch_size = batch_size;
    config.max_batch_delay = delay;
    return config;
}

void ExpectIdleInvalidConfig(InferenceScheduler& scheduler) {
    const SchedulerError error = scheduler.start();
    EXPECT_EQ(error.kind, SchedulerErrorKind::kInvalidConfig);
    EXPECT_EQ(error.load_error.kind, InferenceErrorKind::kNone);
    EXPECT_FALSE(SchedulerTestAccess::IsFailed(scheduler));
    EXPECT_FALSE(SchedulerTestAccess::IsStopped(scheduler));
    EXPECT_EQ(scheduler.start().kind, SchedulerErrorKind::kInvalidConfig);
}

std::vector<NamedTensor> DynamicAddInputs(float a0, float a1, float b0, float b1) {
    return {NamedTensor{"A", {1, 2}, {a0, a1}}, NamedTensor{"B", {1, 2}, {b0, b1}}};
}

void ExpectDynamicSum(const SchedulerCompletion& completion, float c0, float c1) {
    ASSERT_EQ(completion.status, CompletionStatus::kSucceeded);
    EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kNone);
    ASSERT_EQ(completion.outputs.size(), 1U);
    EXPECT_EQ(completion.outputs[0].name, "C");
    EXPECT_EQ(completion.outputs[0].shape, (std::vector<std::int64_t>{1, 2}));
    ASSERT_EQ(completion.outputs[0].values.size(), 2U);
    EXPECT_FLOAT_EQ(completion.outputs[0].values[0], c0);
    EXPECT_FLOAT_EQ(completion.outputs[0].values[1], c1);
}

void ExpectRequestIdentity(const SchedulerCompletion& completion, std::uint64_t expected) {
    EXPECT_EQ(completion.request_id, expected);
    EXPECT_EQ(completion.sequence, expected);
}

TEST_F(InferenceSchedulerTest, AggregateInitializationLeavesBatchingDisabled) {
    const SchedulerConfig config{INFEREDGE_ADD_MODEL_PATH, 1, 1};
    EXPECT_FALSE(config.enable_batching);
    EXPECT_EQ(config.max_batch_size, 1U);
    EXPECT_EQ(config.max_batch_delay, std::chrono::milliseconds::zero());
    EXPECT_FALSE(config.enable_metrics);
}

TEST_F(InferenceSchedulerTest, RejectsZeroBatchSize) {
    const auto scheduler = MakeScheduler(
        WithBatching(INFEREDGE_ADD_MODEL_PATH, false, 0, std::chrono::milliseconds{0}));
    ExpectIdleInvalidConfig(*scheduler);
}

TEST_F(InferenceSchedulerTest, RejectsBatchSizeAboveMaximum) {
    const auto scheduler = MakeScheduler(
        WithBatching(INFEREDGE_ADD_MODEL_PATH, false, 33, std::chrono::milliseconds{0}));
    ExpectIdleInvalidConfig(*scheduler);
}

TEST_F(InferenceSchedulerTest, RejectsNegativeBatchDelay) {
    const auto scheduler = MakeScheduler(
        WithBatching(INFEREDGE_ADD_MODEL_PATH, false, 1, std::chrono::milliseconds{-1}));
    ExpectIdleInvalidConfig(*scheduler);
}

TEST_F(InferenceSchedulerTest, RejectsExcessiveBatchDelay) {
    const auto scheduler = MakeScheduler(
        WithBatching(INFEREDGE_ADD_MODEL_PATH, false, 1, std::chrono::milliseconds{1001}));
    ExpectIdleInvalidConfig(*scheduler);
}

TEST_F(InferenceSchedulerTest, DisabledBatchingStartsStaticAdd) {
    const auto scheduler = MakeScheduler(
        WithBatching(INFEREDGE_ADD_MODEL_PATH, false, 32, std::chrono::milliseconds{1000}));
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, EnabledBatchingAcceptsMinimumBounds) {
    const auto scheduler = MakeScheduler(
        WithBatching(INFEREDGE_BATCH_ADD_MODEL_PATH, true, 1, std::chrono::milliseconds{0}));
    RequireStart(*scheduler);
    ExpectDynamicSum(Take(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, EnabledBatchingAcceptsMaximumBounds) {
    const auto scheduler = MakeScheduler(
        WithBatching(INFEREDGE_BATCH_ADD_MODEL_PATH, true, 32, std::chrono::milliseconds{1000}));
    RequireStart(*scheduler);
    ExpectDynamicSum(Take(scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F))), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, EnabledBatchingRejectsStaticAdd) {
    const auto scheduler = MakeScheduler(
        WithBatching(INFEREDGE_ADD_MODEL_PATH, true, 1, std::chrono::milliseconds{0}, 2));
    const SchedulerError error = scheduler->start();
    EXPECT_EQ(error.kind, SchedulerErrorKind::kModelLoad);
    EXPECT_EQ(error.load_error.kind, InferenceErrorKind::kModelInvalid);
    EXPECT_FALSE(error.message.empty());
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));

    const SchedulerStats stats = scheduler->stats();
    EXPECT_EQ(stats.accepted, 0U);
    EXPECT_EQ(stats.live_workers, 0U);
    EXPECT_EQ(stats.executing, 0U);

    std::vector<NamedTensor> inputs = AddInputs(1.F, 2.F, 3.F, 4.F);
    const Admission rejected = scheduler->submit(std::move(inputs));
    EXPECT_EQ(rejected.status, AdmissionStatus::kRejectedClosed);
    EXPECT_FALSE(rejected.completion.has_value());
    ASSERT_TRUE(rejected.inputs.has_value());
    EXPECT_EQ(scheduler->stats().accepted, 0U);

    EXPECT_EQ(scheduler->start().kind, SchedulerErrorKind::kInvalidState);
}

TEST_F(InferenceSchedulerTest, EnabledBatchingRejectsStaticMnist) {
    const auto scheduler = MakeScheduler(
        WithBatching(INFEREDGE_MNIST_MODEL_PATH, true, 1, std::chrono::milliseconds{0}));
    const SchedulerError error = scheduler->start();
    EXPECT_EQ(error.kind, SchedulerErrorKind::kModelLoad);
    EXPECT_EQ(error.load_error.kind, InferenceErrorKind::kModelInvalid);
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
    EXPECT_EQ(scheduler->stats().live_workers, 0U);
    EXPECT_EQ(scheduler->stats().accepted, 0U);
}

std::unique_ptr<InferenceScheduler> MakeBatch(std::uint32_t batch_size,
                                              std::chrono::milliseconds delay, std::size_t capacity,
                                              std::uint32_t workers = 1) {
    SchedulerConfig config{INFEREDGE_BATCH_ADD_MODEL_PATH, workers, capacity};
    config.enable_batching = true;
    config.max_batch_size = batch_size;
    config.max_batch_delay = delay;
    return MakeScheduler(std::move(config));
}

struct BatchRuns {
    std::mutex mutex;
    std::vector<std::size_t> sizes;

    void Record(std::size_t count) {
        std::lock_guard<std::mutex> lock(mutex);
        sizes.push_back(count);
    }

    std::vector<std::size_t> Snapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        return sizes;
    }
};

void ExpectInferenceFailure(const SchedulerCompletion& completion, InferenceErrorKind kind) {
    EXPECT_EQ(completion.status, CompletionStatus::kInferenceFailed);
    EXPECT_EQ(completion.inference_error.kind, kind);
    EXPECT_FALSE(completion.inference_error.message.empty());
    EXPECT_TRUE(completion.outputs.empty());
}

ShutdownResult ShutdownWhileBlocked(InferenceScheduler& scheduler, LatchRelease& release,
                                    ShutdownPolicy policy) {
    auto closed = std::make_shared<std::latch>(1);
    SchedulerTestAccess::SetAfterDrainClosed(scheduler, closed);
    JoinThread stopper;
    ShutdownResult shutdown;
    stopper.thread = std::thread([&] { shutdown = scheduler.shutdown(policy); });
    closed->wait();
    release.release();
    stopper.thread.join();
    return shutdown;
}

TEST_F(InferenceSchedulerTest, DisabledBatchingDoesNotRunABatch) {
    std::atomic<int> runs{0};
    auto scheduler = MakeScheduler(
        WithBatching(INFEREDGE_ADD_MODEL_PATH, false, 4, std::chrono::milliseconds{0}));
    SchedulerTestAccess::SetOnBatchRun(*scheduler, [&runs](std::size_t) { runs.fetch_add(1); });
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    ExpectSum(Take(scheduler->submit(AddInputs(5.F, 6.F, 7.F, 8.F))), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.load(), 0);
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, RunsOneRequestAsOneBatch) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 2);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    ExpectDynamicSum(Take(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{1}));
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, RunsTwoReadyRequestsInOneBatch) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    ExpectDynamicSum(Take(std::move(first)), 4.F, 6.F);
    ExpectDynamicSum(Take(std::move(second)), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{2}));
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, RunsFourReadyRequestsInOneBatch) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    std::vector<Admission> admissions;
    admissions.push_back(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(9.F, 0.F, 1.F, 1.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(-1.F, 4.F, 2.F, -3.F)));
    release.release();
    std::vector<SchedulerCompletion> completions;
    for (Admission& admission : admissions) {
        completions.push_back(Take(std::move(admission)));
    }
    ExpectDynamicSum(completions[0], 4.F, 6.F);
    ExpectDynamicSum(completions[1], 12.F, 14.F);
    ExpectDynamicSum(completions[2], 10.F, 1.F);
    ExpectDynamicSum(completions[3], 1.F, 1.F);
    const float kept = completions[1].outputs[0].values[0];
    completions[0].outputs[0].values[0] = 42.F;
    EXPECT_FLOAT_EQ(completions[1].outputs[0].values[0], kept);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 4, 0, 0, 0);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{4}));
    scheduler.reset();
    EXPECT_FLOAT_EQ(completions[0].outputs[0].values[0], 42.F);
    EXPECT_FLOAT_EQ(completions[1].outputs[0].values[0], kept);
}

TEST_F(InferenceSchedulerTest, EnforcesMaxBatchSize) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 8);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    std::vector<Admission> admissions;
    admissions.push_back(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(9.F, 0.F, 1.F, 1.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(-1.F, 4.F, 2.F, -3.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(8.F, 1.F, 1.F, 1.F)));
    release.release();
    ExpectDynamicSum(Take(std::move(admissions[0])), 4.F, 6.F);
    ExpectDynamicSum(Take(std::move(admissions[1])), 12.F, 14.F);
    ExpectDynamicSum(Take(std::move(admissions[2])), 10.F, 1.F);
    ExpectDynamicSum(Take(std::move(admissions[3])), 1.F, 1.F);
    ExpectDynamicSum(Take(std::move(admissions[4])), 9.F, 2.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{4, 1}));
    ExpectQuiescent(*scheduler, 5, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, ZeroDelayDoesNotWaitForALaterRequest) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 4);
    auto before_run = std::make_shared<std::latch>(1);
    auto sealed = std::make_shared<std::latch>(1);
    auto timed_wait = std::make_shared<std::latch>(1);
    LatchRelease release{before_run};
    std::atomic<int> calls{0};
    SchedulerTestAccess::SetBeforeRun(*scheduler, before_run);
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, timed_wait);
    SchedulerTestAccess::SetOnBatchRun(*scheduler, [&](std::size_t count) {
        runs.Record(count);
        if (calls.fetch_add(1) == 0) {
            sealed->count_down();
        }
    });
    RequireStart(*scheduler);
    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    sealed->wait();
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    ExpectDynamicSum(Take(std::move(first)), 4.F, 6.F);
    ExpectDynamicSum(Take(std::move(second)), 12.F, 14.F);
    EXPECT_FALSE(timed_wait->try_wait());
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{1, 1}));
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, PositiveDelayAdmitsTheNextRequest) {
    BatchRuns runs;
    auto scheduler = MakeBatch(2, std::chrono::milliseconds{1000}, 4);
    auto reached = std::make_shared<std::latch>(1);
    auto hold = std::make_shared<std::latch>(1);
    LatchRelease release{hold};
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, reached);
    SchedulerTestAccess::SetBeforeBatchWait(*scheduler, hold);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    reached->wait();
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    ExpectDynamicSum(Take(std::move(first)), 4.F, 6.F);
    ExpectDynamicSum(Take(std::move(second)), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{2}));
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, PositiveDelayFlushesOneRequest) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{30}, 2);
    auto reached = std::make_shared<std::latch>(1);
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, reached);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    reached->wait();
    ExpectDynamicSum(Take(std::move(admission)), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{1}));
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, DrainDuringBatchWaitRunsTheOwnedRequest) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{1000}, 2);
    auto reached = std::make_shared<std::latch>(1);
    auto hold = std::make_shared<std::latch>(1);
    LatchRelease release{hold};
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, reached);
    SchedulerTestAccess::SetBeforeBatchWait(*scheduler, hold);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    // The worker is held immediately before wait_pop_until. ShutdownWhileBlocked
    // commits close_drain before releasing that hold, so the timed pop observes
    // a closed queue and returns without the 1000 ms deadline.
    reached->wait();
    const ShutdownResult shutdown =
        ShutdownWhileBlocked(*scheduler, release, ShutdownPolicy::kDrain);
    ExpectWinner(shutdown, ShutdownPolicy::kDrain, true);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    ASSERT_TRUE(admission.completion.has_value());
    ASSERT_EQ(admission.completion->wait_for(std::chrono::seconds{5}), std::future_status::ready);
    ExpectDynamicSum(Take(std::move(admission)), 4.F, 6.F);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{1}));
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, DrainDuringBatchWaitCollectsAQueuedRequest) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{1000}, 4);
    auto reached = std::make_shared<std::latch>(1);
    auto hold = std::make_shared<std::latch>(1);
    LatchRelease release{hold};
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, reached);
    SchedulerTestAccess::SetBeforeBatchWait(*scheduler, hold);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    reached->wait();
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    const ShutdownResult shutdown =
        ShutdownWhileBlocked(*scheduler, release, ShutdownPolicy::kDrain);
    ExpectWinner(shutdown, ShutdownPolicy::kDrain, true);
    ExpectDynamicSum(Take(std::move(first)), 4.F, 6.F);
    ExpectDynamicSum(Take(std::move(second)), 12.F, 14.F);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{2}));
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, CancelDuringBatchWaitKeepsTheOwnedRequest) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{1000}, 4);
    auto reached = std::make_shared<std::latch>(1);
    auto hold = std::make_shared<std::latch>(1);
    LatchRelease release{hold};
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, reached);
    SchedulerTestAccess::SetBeforeBatchWait(*scheduler, hold);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission owned = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    reached->wait();
    Admission queued = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    const ShutdownResult shutdown =
        ShutdownWhileBlocked(*scheduler, release, ShutdownPolicy::kCancelPending);
    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    ExpectDynamicSum(Take(std::move(owned)), 4.F, 6.F);
    ExpectCanceled(Take(std::move(queued)), 2);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{1}));
    ExpectQuiescent(*scheduler, 1, 0, 0, 0, 1);
}

TEST_F(InferenceSchedulerTest, IsolatesADuplicateInputFromTheBatch) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission valid_a = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission invalid = scheduler->submit({NamedTensor{"A", {1, 2}, {1.F, 2.F}},
                                           NamedTensor{"A", {1, 2}, {3.F, 4.F}},
                                           NamedTensor{"B", {1, 2}, {5.F, 6.F}}});
    Admission valid_c = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    ExpectDynamicSum(Take(std::move(valid_a)), 4.F, 6.F);
    ExpectInferenceFailure(Take(std::move(invalid)), InferenceErrorKind::kDuplicateInput);
    ExpectDynamicSum(Take(std::move(valid_c)), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{2}));
    ExpectQuiescent(*scheduler, 2, 1, 0, 0);
}

TEST_F(InferenceSchedulerTest, IsolatesAMissingInputFromTheBatch) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission valid_a = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission invalid = scheduler->submit({NamedTensor{"A", {1, 2}, {1.F, 2.F}}});
    Admission valid_c = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    ExpectDynamicSum(Take(std::move(valid_a)), 4.F, 6.F);
    ExpectInferenceFailure(Take(std::move(invalid)), InferenceErrorKind::kMissingInput);
    ExpectDynamicSum(Take(std::move(valid_c)), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{2}));
    ExpectQuiescent(*scheduler, 2, 1, 0, 0);
}

TEST_F(InferenceSchedulerTest, IsolatesAWrongShapeFromTheBatch) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission valid_a = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission invalid = scheduler->submit(
        {NamedTensor{"A", {2, 2}, {1.F, 2.F, 3.F, 4.F}}, NamedTensor{"B", {1, 2}, {5.F, 6.F}}});
    Admission valid_c = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    ExpectDynamicSum(Take(std::move(valid_a)), 4.F, 6.F);
    ExpectInferenceFailure(Take(std::move(invalid)), InferenceErrorKind::kShapeMismatch);
    ExpectDynamicSum(Take(std::move(valid_c)), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{2}));
    ExpectQuiescent(*scheduler, 2, 1, 0, 0);
}

TEST_F(InferenceSchedulerTest, ThrowsBeforeBatchRunFailsTheGroup) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    const SchedulerCompletion failed_first = Take(std::move(first));
    const SchedulerCompletion failed_second = Take(std::move(second));
    EXPECT_EQ(failed_first.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed_second.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed_first.inference_error.message, "injected");
    EXPECT_EQ(failed_second.inference_error.message, "injected");
    EXPECT_TRUE(failed_first.outputs.empty());
    EXPECT_TRUE(failed_second.outputs.empty());
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{2}));
    ExpectQuiescent(*scheduler, 0, 0, 2, 0);
}

TEST_F(InferenceSchedulerTest, BatchFaultGuardCreationKeepsPoppedRequestOwned) {
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 2);
    SchedulerTestAccess::SetFailNextBatchGuardCreation(*scheduler);
    RequireStart(*scheduler);

    const SchedulerCompletion completion =
        TakeBounded(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F)));
    EXPECT_EQ(completion.request_id, 1U);
    EXPECT_EQ(completion.sequence, 1U);
    EXPECT_EQ(completion.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kExecution);
    EXPECT_EQ(completion.inference_error.message, "alloc failed");
    EXPECT_TRUE(completion.outputs.empty());

    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 0, 0, 1, 0);
}

TEST_F(InferenceSchedulerTest, BatchFaultDuringCollectionSeparatesOwnedAndQueuedRequests) {
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetThrowAfterBatchCollect(*scheduler, 1);
    RequireStart(*scheduler);

    Admission owned = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission queued = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();

    const SchedulerCompletion failed = TakeBounded(std::move(owned));
    const SchedulerCompletion swept = TakeBounded(std::move(queued));
    EXPECT_EQ(failed.request_id, 1U);
    EXPECT_EQ(failed.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed.inference_error.message, "collect failed");
    EXPECT_EQ(swept.request_id, 2U);
    EXPECT_EQ(swept.status, CompletionStatus::kSchedulerFailed);
    EXPECT_EQ(swept.inference_error.message, "no worker");

    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
    ExpectQuiescent(*scheduler, 0, 0, 1, 1);
}

TEST_F(InferenceSchedulerTest, BatchFaultPartialPublicationPreservesPublishedResults) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetThrowBeforeBatchPublish(*scheduler, 2);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);

    std::vector<Admission> admissions;
    admissions.push_back(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(9.F, 0.F, 1.F, 1.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(-1.F, 4.F, 2.F, -3.F)));
    release.release();

    std::vector<SchedulerCompletion> completions;
    for (Admission& admission : admissions) {
        completions.push_back(TakeBounded(std::move(admission)));
    }
    ExpectDynamicSum(completions[0], 4.F, 6.F);
    ExpectDynamicSum(completions[1], 12.F, 14.F);
    for (std::size_t index = 2; index < completions.size(); ++index) {
        EXPECT_EQ(completions[index].status, CompletionStatus::kWorkerFailed);
        EXPECT_EQ(completions[index].inference_error.kind, InferenceErrorKind::kExecution);
        EXPECT_EQ(completions[index].inference_error.message, "publish failed");
        EXPECT_TRUE(completions[index].outputs.empty());
    }
    for (std::size_t index = 0; index < completions.size(); ++index) {
        EXPECT_EQ(completions[index].request_id, index + 1);
        EXPECT_EQ(completions[index].sequence, index + 1);
    }

    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{4}));
    ExpectQuiescent(*scheduler, 2, 0, 2, 0);
    scheduler.reset();
    EXPECT_FLOAT_EQ(completions[0].outputs[0].values[0], 4.F);
    EXPECT_FLOAT_EQ(completions[1].outputs[0].values[0], 12.F);
}

TEST_F(InferenceSchedulerTest, BatchFaultFailurePublicationContinuesAfterOneAttemptThrows) {
    BatchRuns runs;
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    SchedulerTestAccess::SetFailWorkerFailurePublication(*scheduler, 1);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);

    std::vector<Admission> admissions;
    admissions.push_back(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(9.F, 0.F, 1.F, 1.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(-1.F, 4.F, 2.F, -3.F)));
    release.release();

    std::vector<SchedulerCompletion> completions;
    for (Admission& admission : admissions) {
        completions.push_back(TakeBounded(std::move(admission)));
    }
    EXPECT_EQ(completions[0].status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(completions[0].inference_error.message, "injected");
    EXPECT_EQ(completions[1].status, CompletionStatus::kSchedulerFailed);
    EXPECT_EQ(completions[1].inference_error.message, "unsettled");
    EXPECT_EQ(completions[2].status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(completions[2].inference_error.message, "injected");
    EXPECT_EQ(completions[3].status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(completions[3].inference_error.message, "injected");
    for (std::size_t index = 0; index < completions.size(); ++index) {
        EXPECT_EQ(completions[index].request_id, index + 1);
        EXPECT_EQ(completions[index].sequence, index + 1);
        EXPECT_TRUE(completions[index].outputs.empty());
    }

    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{4}));
    const SchedulerStats stats = scheduler->stats();
    EXPECT_EQ(stats.accepted, 4U);
    EXPECT_EQ(stats.succeeded, 0U);
    EXPECT_EQ(stats.failed_inference, 0U);
    EXPECT_EQ(stats.failed_worker, 3U);
    // The second completion came from RequestEntry's destructor backstop.
    EXPECT_EQ(stats.failed_scheduler, 0U);
    EXPECT_EQ(stats.canceled, 0U);
    EXPECT_EQ(stats.queue_depth, 0U);
    EXPECT_EQ(stats.executing, 0U);
    EXPECT_EQ(stats.live_workers, 0U);
}

TEST_F(InferenceSchedulerTest, BatchFaultCancelDuringSealedFailureKeepsOwnedRequests) {
    BatchRuns runs;
    auto scheduler = MakeBatch(2, std::chrono::milliseconds{0}, 4);
    auto before_pop = std::make_shared<std::latch>(1);
    auto before_run = std::make_shared<std::latch>(1);
    auto reached_run = std::make_shared<std::latch>(1);
    LatchRelease pop_release{before_pop};
    LatchRelease run_release{before_run};
    SchedulerTestAccess::SetBeforePop(*scheduler, before_pop);
    SchedulerTestAccess::SetBeforeRun(*scheduler, before_run);
    SchedulerTestAccess::SetReachedRun(*scheduler, reached_run);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);

    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    pop_release.release();
    reached_run->wait();
    const ShutdownResult shutdown =
        ShutdownWhileBlocked(*scheduler, run_release, ShutdownPolicy::kCancelPending);

    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    const SchedulerCompletion failed_first = TakeBounded(std::move(first));
    const SchedulerCompletion failed_second = TakeBounded(std::move(second));
    EXPECT_EQ(failed_first.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed_second.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed_first.request_id, 1U);
    EXPECT_EQ(failed_second.request_id, 2U);
    EXPECT_EQ(failed_first.inference_error.message, "injected");
    EXPECT_EQ(failed_second.inference_error.message, "injected");
    EXPECT_TRUE(failed_first.outputs.empty());
    EXPECT_TRUE(failed_second.outputs.empty());
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{2}));
    EXPECT_TRUE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectQuiescent(*scheduler, 0, 0, 2, 0);
}

TEST_F(InferenceSchedulerTest, MultiWorkerBatchDrainCompletesOwnedAndQueuedRequests) {
    BatchRuns runs;
    auto first_sealed = std::make_shared<std::latch>(1);
    std::atomic<bool> signal_first{true};
    auto gate0 = std::make_shared<std::latch>(1);
    auto gate1 = std::make_shared<std::latch>(1);
    auto reached_wait = std::make_shared<std::latch>(1);
    auto before_wait = std::make_shared<std::latch>(1);
    auto before_run = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto scheduler = MakeBatch(2, std::chrono::milliseconds{1000}, 5, 2);
    JoinThread stopper;
    LatchRelease release0{gate0};
    LatchRelease release1{gate1};
    LatchRelease release_wait{before_wait};
    LatchRelease release_run{before_run};
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 0, gate0);
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 1, gate1);
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, reached_wait);
    SchedulerTestAccess::SetBeforeBatchWait(*scheduler, before_wait);
    SchedulerTestAccess::SetBeforeRun(*scheduler, before_run);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetOnBatchRun(*scheduler, [&](std::size_t count) {
        runs.Record(count);
        if (signal_first.exchange(false)) {
            first_sealed->count_down();
        }
    });
    RequireStart(*scheduler);

    std::vector<Admission> admissions;
    admissions.push_back(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F)));
    release0.release();
    first_sealed->wait();

    admissions.push_back(scheduler->submit(DynamicAddInputs(9.F, 0.F, 1.F, 1.F)));
    release1.release();
    reached_wait->wait();
    // Worker 0 is held before run and worker 1 is held before its only timed
    // wait. The fourth request seals worker 1's size-two batch after drain;
    // the fifth is then popped from the closed queue without another wait.
    admissions.push_back(scheduler->submit(DynamicAddInputs(-1.F, 4.F, 2.F, -3.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(3.F, 3.F, 4.F, 4.F)));

    ShutdownResult shutdown;
    stopper.thread = std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kDrain); });
    closed->wait();
    release_wait.release();
    release_run.release();
    stopper.thread.join();

    ExpectWinner(shutdown, ShutdownPolicy::kDrain, true);
    const SchedulerCompletion first = TakeBounded(std::move(admissions[0]));
    const SchedulerCompletion second = TakeBounded(std::move(admissions[1]));
    const SchedulerCompletion third = TakeBounded(std::move(admissions[2]));
    const SchedulerCompletion fourth = TakeBounded(std::move(admissions[3]));
    const SchedulerCompletion fifth = TakeBounded(std::move(admissions[4]));
    ExpectDynamicSum(first, 4.F, 6.F);
    ExpectDynamicSum(second, 12.F, 14.F);
    ExpectDynamicSum(third, 10.F, 1.F);
    ExpectDynamicSum(fourth, 1.F, 1.F);
    ExpectDynamicSum(fifth, 7.F, 7.F);
    const std::vector<SchedulerCompletion> completions{first, second, third, fourth, fifth};
    for (std::size_t index = 0; index < completions.size(); ++index) {
        ExpectRequestIdentity(completions[index], index + 1);
    }
    const std::vector<std::size_t> sizes = runs.Snapshot();
    ASSERT_EQ(sizes.size(), 3U);
    std::size_t total = 0;
    for (const std::size_t size : sizes) {
        EXPECT_LE(size, 2U);
        total += size;
    }
    EXPECT_EQ(total, 5U);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectQuiescent(*scheduler, 5, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, MultiWorkerBatchCancelSeparatesOwnedAndQueuedRequests) {
    BatchRuns runs;
    auto first_sealed = std::make_shared<std::latch>(1);
    std::atomic<bool> signal_first{true};
    auto gate0 = std::make_shared<std::latch>(1);
    auto gate1 = std::make_shared<std::latch>(1);
    auto reached_wait = std::make_shared<std::latch>(1);
    auto before_wait = std::make_shared<std::latch>(1);
    auto before_run = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto scheduler = MakeBatch(2, std::chrono::milliseconds{1000}, 5, 2);
    JoinThread stopper;
    LatchRelease release0{gate0};
    LatchRelease release1{gate1};
    LatchRelease release_wait{before_wait};
    LatchRelease release_run{before_run};
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 0, gate0);
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 1, gate1);
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, reached_wait);
    SchedulerTestAccess::SetBeforeBatchWait(*scheduler, before_wait);
    SchedulerTestAccess::SetBeforeRun(*scheduler, before_run);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetOnBatchRun(*scheduler, [&](std::size_t count) {
        runs.Record(count);
        if (signal_first.exchange(false)) {
            first_sealed->count_down();
        }
    });
    RequireStart(*scheduler);

    std::vector<Admission> admissions;
    admissions.push_back(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F)));
    release0.release();
    first_sealed->wait();

    admissions.push_back(scheduler->submit(DynamicAddInputs(9.F, 0.F, 1.F, 1.F)));
    release1.release();
    reached_wait->wait();
    // Both following requests remain queue-owned until cancel transfers them.
    // Releasing the partial collector therefore cannot enter another timed wait.
    admissions.push_back(scheduler->submit(DynamicAddInputs(-1.F, 4.F, 2.F, -3.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(3.F, 3.F, 4.F, 4.F)));

    ShutdownResult shutdown;
    stopper.thread =
        std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    release_wait.release();
    release_run.release();
    stopper.thread.join();

    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    const SchedulerCompletion first = TakeBounded(std::move(admissions[0]));
    const SchedulerCompletion second = TakeBounded(std::move(admissions[1]));
    const SchedulerCompletion third = TakeBounded(std::move(admissions[2]));
    ExpectDynamicSum(first, 4.F, 6.F);
    ExpectDynamicSum(second, 12.F, 14.F);
    ExpectDynamicSum(third, 10.F, 1.F);
    ExpectRequestIdentity(first, 1);
    ExpectRequestIdentity(second, 2);
    ExpectRequestIdentity(third, 3);
    ExpectCanceled(TakeBounded(std::move(admissions[3])), 4);
    ExpectCanceled(TakeBounded(std::move(admissions[4])), 5);
    const std::vector<std::size_t> sizes = runs.Snapshot();
    ASSERT_EQ(sizes.size(), 2U);
    EXPECT_EQ(sizes[0] + sizes[1], 3U);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectQuiescent(*scheduler, 3, 0, 0, 0, 2);
}

TEST_F(InferenceSchedulerTest, BatchDrainWinsAgainstCancelAndRejectsLateSubmissions) {
    auto reached_wait = std::make_shared<std::latch>(1);
    auto before_wait = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto follower = std::make_shared<std::latch>(1);
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{1000}, 4);
    JoinThread drain_call;
    JoinThread cancel_call;
    LatchRelease release_wait{before_wait};
    std::vector<JoinThread> producers(2);
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, reached_wait);
    SchedulerTestAccess::SetBeforeBatchWait(*scheduler, before_wait);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetShutdownFollowerEntered(*scheduler, follower);
    RequireStart(*scheduler);

    Admission owned = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    reached_wait->wait();
    Admission queued = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    ShutdownResult drain;
    ShutdownResult cancel;
    drain_call.thread = std::thread([&] { drain = scheduler->shutdown(ShutdownPolicy::kDrain); });
    closed->wait();
    cancel_call.thread =
        std::thread([&] { cancel = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    follower->wait();

    std::vector<Admission> rejected(2);
    for (std::size_t index = 0; index < producers.size(); ++index) {
        producers[index].thread = std::thread([&, index] {
            const float tag = 20.F + static_cast<float>(index);
            rejected[index] = scheduler->submit(DynamicAddInputs(tag, 0.F, 1.F, 0.F));
        });
    }
    for (JoinThread& producer : producers) {
        producer.thread.join();
    }
    ExpectRejectedInputs(rejected[0], AdmissionStatus::kRejectedClosed, 20.F);
    ExpectRejectedInputs(rejected[1], AdmissionStatus::kRejectedClosed, 21.F);

    release_wait.release();
    drain_call.thread.join();
    cancel_call.thread.join();
    ExpectWinner(drain, ShutdownPolicy::kDrain, true);
    ExpectWinner(cancel, ShutdownPolicy::kDrain, false);
    const SchedulerCompletion owned_completion = TakeBounded(std::move(owned));
    const SchedulerCompletion queued_completion = TakeBounded(std::move(queued));
    ExpectDynamicSum(owned_completion, 4.F, 6.F);
    ExpectDynamicSum(queued_completion, 12.F, 14.F);
    ExpectRequestIdentity(owned_completion, 1);
    ExpectRequestIdentity(queued_completion, 2);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
    EXPECT_EQ(scheduler->stats().rejected_closed, 2U);
}

TEST_F(InferenceSchedulerTest, BatchCancelWinsAgainstDrain) {
    auto reached_wait = std::make_shared<std::latch>(1);
    auto before_wait = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto follower = std::make_shared<std::latch>(1);
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{1000}, 2);
    JoinThread cancel_call;
    JoinThread drain_call;
    LatchRelease release_wait{before_wait};
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, reached_wait);
    SchedulerTestAccess::SetBeforeBatchWait(*scheduler, before_wait);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetShutdownFollowerEntered(*scheduler, follower);
    RequireStart(*scheduler);

    Admission owned = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    reached_wait->wait();
    Admission queued = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    ShutdownResult cancel;
    ShutdownResult drain;
    cancel_call.thread =
        std::thread([&] { cancel = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    drain_call.thread = std::thread([&] { drain = scheduler->shutdown(ShutdownPolicy::kDrain); });
    follower->wait();

    release_wait.release();
    cancel_call.thread.join();
    drain_call.thread.join();
    ExpectWinner(cancel, ShutdownPolicy::kCancelPending, true);
    ExpectWinner(drain, ShutdownPolicy::kCancelPending, false);
    const SchedulerCompletion owned_completion = TakeBounded(std::move(owned));
    ExpectDynamicSum(owned_completion, 4.F, 6.F);
    ExpectRequestIdentity(owned_completion, 1);
    ExpectCanceled(TakeBounded(std::move(queued)), 2);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForCancel(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectQuiescent(*scheduler, 1, 0, 0, 0, 1);
}

TEST_F(InferenceSchedulerTest, BatchWorkerFaultDuringDrainLetsSiblingFinish) {
    BatchRuns runs;
    auto first_sealed = std::make_shared<std::latch>(1);
    std::atomic<bool> signal_first{true};
    auto gate0 = std::make_shared<std::latch>(1);
    auto gate1 = std::make_shared<std::latch>(1);
    auto before_run = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto scheduler = MakeBatch(2, std::chrono::milliseconds{0}, 4, 2);
    JoinThread stopper;
    LatchRelease release0{gate0};
    LatchRelease release1{gate1};
    LatchRelease release_run{before_run};
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 0, gate0);
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 1, gate1);
    SchedulerTestAccess::SetBeforeRun(*scheduler, before_run);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    SchedulerTestAccess::SetOnBatchRun(*scheduler, [&](std::size_t count) {
        runs.Record(count);
        if (signal_first.exchange(false)) {
            first_sealed->count_down();
        }
    });
    RequireStart(*scheduler);

    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release0.release();
    first_sealed->wait();
    Admission third = scheduler->submit(DynamicAddInputs(9.F, 0.F, 1.F, 1.F));
    Admission fourth = scheduler->submit(DynamicAddInputs(-1.F, 4.F, 2.F, -3.F));

    ShutdownResult shutdown;
    stopper.thread = std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kDrain); });
    closed->wait();
    release1.release();
    release_run.release();
    stopper.thread.join();

    ExpectWinner(shutdown, ShutdownPolicy::kDrain, true);
    const SchedulerCompletion failed_first = TakeBounded(std::move(first));
    const SchedulerCompletion failed_second = TakeBounded(std::move(second));
    EXPECT_EQ(failed_first.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed_second.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed_first.inference_error.message, "injected");
    EXPECT_EQ(failed_second.inference_error.message, "injected");
    ExpectRequestIdentity(failed_first, 1);
    ExpectRequestIdentity(failed_second, 2);
    const SchedulerCompletion third_completion = TakeBounded(std::move(third));
    const SchedulerCompletion fourth_completion = TakeBounded(std::move(fourth));
    ExpectDynamicSum(third_completion, 10.F, 1.F);
    ExpectDynamicSum(fourth_completion, 1.F, 1.F);
    ExpectRequestIdentity(third_completion, 3);
    ExpectRequestIdentity(fourth_completion, 4);
    const std::vector<std::size_t> sizes = runs.Snapshot();
    ASSERT_EQ(sizes.size(), 2U);
    EXPECT_EQ(sizes[0] + sizes[1], 4U);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::IsStopped(*scheduler));
    ExpectQuiescent(*scheduler, 2, 0, 2, 0);
}

TEST_F(InferenceSchedulerTest, BatchLastWorkerFaultDuringDrainSweepsQueuedTail) {
    auto sealed = std::make_shared<std::latch>(1);
    auto before_pop = std::make_shared<std::latch>(1);
    auto before_run = std::make_shared<std::latch>(1);
    auto closed = std::make_shared<std::latch>(1);
    auto scheduler = MakeBatch(2, std::chrono::milliseconds{0}, 3);
    JoinThread stopper;
    LatchRelease release_pop{before_pop};
    LatchRelease release_run{before_run};
    SchedulerTestAccess::SetBeforePop(*scheduler, before_pop);
    SchedulerTestAccess::SetBeforeRun(*scheduler, before_run);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    SchedulerTestAccess::SetOnBatchRun(*scheduler, [sealed](std::size_t) { sealed->count_down(); });
    RequireStart(*scheduler);

    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    Admission queued = scheduler->submit(DynamicAddInputs(9.F, 0.F, 1.F, 1.F));
    release_pop.release();
    sealed->wait();

    ShutdownResult shutdown;
    stopper.thread = std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kDrain); });
    closed->wait();
    release_run.release();
    stopper.thread.join();

    ExpectWinner(shutdown, ShutdownPolicy::kDrain, true);
    const SchedulerCompletion failed_first = TakeBounded(std::move(first));
    const SchedulerCompletion failed_second = TakeBounded(std::move(second));
    const SchedulerCompletion swept = TakeBounded(std::move(queued));
    EXPECT_EQ(failed_first.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed_second.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(failed_first.inference_error.message, "injected");
    EXPECT_EQ(failed_second.inference_error.message, "injected");
    EXPECT_EQ(swept.status, CompletionStatus::kSchedulerFailed);
    EXPECT_EQ(swept.inference_error.message, "no worker");
    ExpectRequestIdentity(failed_first, 1);
    ExpectRequestIdentity(failed_second, 2);
    ExpectRequestIdentity(swept, 3);
    EXPECT_TRUE(SchedulerTestAccess::ClosedForDrain(*scheduler));
    EXPECT_TRUE(SchedulerTestAccess::IsFailed(*scheduler));
    ExpectQuiescent(*scheduler, 0, 0, 2, 1);
}

TEST_F(InferenceSchedulerTest, RepeatedBatchSchedulerLifetimes) {
    for (int cycle = 0; cycle < 9; ++cycle) {
        SCOPED_TRACE(cycle);
        const std::uint32_t workers = 1U + static_cast<std::uint32_t>(cycle % 2);
        const std::uint32_t batch_size = 1U + static_cast<std::uint32_t>(cycle % 4);
        const std::size_t capacity = 2U + static_cast<std::size_t>(cycle % 3);
        const std::chrono::milliseconds delay =
            cycle % 2 == 0 ? std::chrono::milliseconds{0} : std::chrono::milliseconds{1000};

        if (cycle % 3 == 2) {
            auto scheduler = MakeBatch(batch_size, delay, capacity, workers);
            RequireStart(*scheduler);
            Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
            Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
            ASSERT_TRUE(first.completion.has_value());
            ASSERT_TRUE(second.completion.has_value());
            std::future<SchedulerCompletion> left = std::move(*first.completion);
            std::future<SchedulerCompletion> right = std::move(*second.completion);
            scheduler.reset();
            ASSERT_EQ(left.wait_for(std::chrono::seconds{0}), std::future_status::ready);
            ASSERT_EQ(right.wait_for(std::chrono::seconds{0}), std::future_status::ready);
            const SchedulerCompletion left_completion = left.get();
            const SchedulerCompletion right_completion = right.get();
            ExpectDynamicSum(left_completion, 4.F, 6.F);
            ExpectDynamicSum(right_completion, 12.F, 14.F);
            ExpectRequestIdentity(left_completion, 1);
            ExpectRequestIdentity(right_completion, 2);
            continue;
        }

        auto before_pop = std::make_shared<std::latch>(1);
        auto scheduler = MakeBatch(batch_size, delay, capacity, workers);
        LatchRelease release{before_pop};
        SchedulerTestAccess::SetBeforePop(*scheduler, before_pop);
        RequireStart(*scheduler);
        Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
        Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
        const ShutdownPolicy policy =
            cycle % 3 == 0 ? ShutdownPolicy::kDrain : ShutdownPolicy::kCancelPending;
        const ShutdownResult shutdown = ShutdownWhileBlocked(*scheduler, release, policy);
        ExpectWinner(shutdown, policy, true);
        if (policy == ShutdownPolicy::kDrain) {
            const SchedulerCompletion first_completion = TakeBounded(std::move(first));
            const SchedulerCompletion second_completion = TakeBounded(std::move(second));
            ExpectDynamicSum(first_completion, 4.F, 6.F);
            ExpectDynamicSum(second_completion, 12.F, 14.F);
            ExpectRequestIdentity(first_completion, 1);
            ExpectRequestIdentity(second_completion, 2);
            ExpectQuiescent(*scheduler, 2, 0, 0, 0);
        } else {
            ExpectCanceled(TakeBounded(std::move(first)), 1);
            ExpectCanceled(TakeBounded(std::move(second)), 2);
            ExpectQuiescent(*scheduler, 0, 0, 0, 0, 2);
        }
        scheduler.reset();
    }
}

TEST_F(InferenceSchedulerTest, RejectsAMetadataShapeProductOverflow) {
    const TensorMetadata valid_input{"A", ElementType::kFloat32, {-1, 2}};
    const TensorMetadata valid_output{"C", ElementType::kFloat32, {-1, 2}};
    const SchedulerError accepted =
        SchedulerTestAccess::CheckBatchMetadata({valid_input}, {valid_output});
    EXPECT_EQ(accepted.kind, SchedulerErrorKind::kNone);

    const TensorMetadata overflow{"A", ElementType::kFloat32, {-1, std::int64_t{1} << 62, 8}};
    const SchedulerError rejected =
        SchedulerTestAccess::CheckBatchMetadata({overflow}, {valid_output});
    EXPECT_EQ(rejected.kind, SchedulerErrorKind::kModelLoad);
    EXPECT_EQ(rejected.load_error.kind, InferenceErrorKind::kModelInvalid);
    EXPECT_FALSE(rejected.message.empty());

    const TensorMetadata output_overflow{
        "C", ElementType::kFloat32, {-1, std::int64_t{1} << 62, 8}};
    const SchedulerError rejected_output =
        SchedulerTestAccess::CheckBatchMetadata({valid_input}, {output_overflow});
    EXPECT_EQ(rejected_output.kind, SchedulerErrorKind::kModelLoad);
    EXPECT_EQ(rejected_output.load_error.kind, InferenceErrorKind::kModelInvalid);
}

TEST_F(InferenceSchedulerTest, RepeatedSchedulerLifetime) {
    for (int cycle = 0; cycle < 8; ++cycle) {
        SCOPED_TRACE(cycle);
        auto scheduler = MakeAdd(1, 2);
        RequireStart(*scheduler);
        const float base = static_cast<float>(cycle);
        Admission first = scheduler->submit(AddInputs(base, 0.F, 1.F, 0.F));
        Admission second = scheduler->submit(AddInputs(base + 1.F, 0.F, 1.F, 0.F));
        ASSERT_TRUE(first.completion.has_value());
        ASSERT_TRUE(second.completion.has_value());
        std::future<SchedulerCompletion> left = std::move(*first.completion);
        std::future<SchedulerCompletion> right = std::move(*second.completion);
        scheduler.reset();
        ASSERT_EQ(left.wait_for(std::chrono::seconds{0}), std::future_status::ready);
        ASSERT_EQ(right.wait_for(std::chrono::seconds{0}), std::future_status::ready);
        SchedulerCompletion left_completion;
        SchedulerCompletion right_completion;
        EXPECT_NO_THROW(left_completion = left.get());
        EXPECT_NO_THROW(right_completion = right.get());
        ExpectSum(left_completion, base + 1.F, 0.F);
        ExpectSum(right_completion, base + 2.F, 0.F);
    }
}

void ExpectEmptyMetrics(const SchedulerMetrics& metrics, bool enabled, bool finalized) {
    EXPECT_EQ(metrics.enabled, enabled);
    EXPECT_EQ(metrics.finalized, finalized);
    EXPECT_EQ(metrics.engine_run_count, 0U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 0U);
    for (const std::uint64_t bin : metrics.batch_size_histogram) {
        EXPECT_EQ(bin, 0U);
    }
    EXPECT_EQ(metrics.total_assembly_nanoseconds, 0U);
    EXPECT_EQ(metrics.total_engine_run_nanoseconds, 0U);
    EXPECT_EQ(metrics.total_split_nanoseconds, 0U);
}

void ExpectHistogramInvariants(const SchedulerMetrics& metrics) {
    EXPECT_EQ(metrics.batch_size_histogram[0], 0U);
    std::uint64_t runs = 0;
    std::uint64_t weighted = 0;
    for (std::size_t size = 1; size <= kSchedulerMetricsMaxBatchSize; ++size) {
        const std::uint64_t count = metrics.batch_size_histogram[size];
        runs += count;
        weighted += count * static_cast<std::uint64_t>(size);
    }
    EXPECT_EQ(runs, metrics.engine_run_count);
    EXPECT_EQ(weighted, metrics.requests_entered_engine_runs);
}

SchedulerConfig MetricsAddConfig(std::uint32_t workers, std::size_t capacity) {
    SchedulerConfig config{INFEREDGE_ADD_MODEL_PATH, workers, capacity};
    config.enable_metrics = true;
    return config;
}

std::unique_ptr<InferenceScheduler> MakeMetricsBatch(std::uint32_t batch_size,
                                                     std::chrono::milliseconds delay,
                                                     std::size_t capacity,
                                                     std::uint32_t workers = 1) {
    SchedulerConfig config{INFEREDGE_BATCH_ADD_MODEL_PATH, workers, capacity};
    config.enable_batching = true;
    config.max_batch_size = batch_size;
    config.max_batch_delay = delay;
    config.enable_metrics = true;
    return MakeScheduler(std::move(config));
}

TEST_F(InferenceSchedulerTest, MetricsDisabledByDefault) {
    const auto scheduler = MakeAdd(1, 1);
    ExpectEmptyMetrics(scheduler->metrics(), false, true);
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
    ExpectEmptyMetrics(scheduler->metrics(), false, true);
}

TEST_F(InferenceSchedulerTest, UnbatchedEnabledRecordsOneRunPerRequest) {
    const auto scheduler = MakeScheduler(MetricsAddConfig(1, 2));
    ExpectEmptyMetrics(scheduler->metrics(), true, true);
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    ExpectSum(Take(scheduler->submit(AddInputs(5.F, 6.F, 7.F, 8.F))), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_TRUE(metrics.enabled);
    EXPECT_TRUE(metrics.finalized);
    EXPECT_EQ(metrics.engine_run_count, 2U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 2U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 2U);
    EXPECT_EQ(metrics.total_assembly_nanoseconds, 0U);
    EXPECT_EQ(metrics.total_split_nanoseconds, 0U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, BatchedTwoRequestsShareOneEngineRun) {
    auto scheduler = MakeMetricsBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    ExpectDynamicSum(Take(std::move(first)), 4.F, 6.F);
    ExpectDynamicSum(Take(std::move(second)), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_TRUE(metrics.finalized);
    EXPECT_EQ(metrics.engine_run_count, 1U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 2U);
    EXPECT_EQ(metrics.batch_size_histogram[2], 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, BatchedFourRequestsShareOneEngineRun) {
    auto scheduler = MakeMetricsBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    std::vector<Admission> admissions;
    admissions.push_back(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(9.F, 0.F, 1.F, 1.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(-1.F, 4.F, 2.F, -3.F)));
    release.release();
    ExpectDynamicSum(Take(std::move(admissions[0])), 4.F, 6.F);
    ExpectDynamicSum(Take(std::move(admissions[1])), 12.F, 14.F);
    ExpectDynamicSum(Take(std::move(admissions[2])), 10.F, 1.F);
    ExpectDynamicSum(Take(std::move(admissions[3])), 1.F, 1.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 4, 0, 0, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_EQ(metrics.engine_run_count, 1U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 4U);
    EXPECT_EQ(metrics.batch_size_histogram[4], 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, FiveRequestsCapFourRecordsGroupsFourAndOne) {
    auto scheduler = MakeMetricsBatch(4, std::chrono::milliseconds{0}, 8);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    std::vector<Admission> admissions;
    admissions.push_back(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(9.F, 0.F, 1.F, 1.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(-1.F, 4.F, 2.F, -3.F)));
    admissions.push_back(scheduler->submit(DynamicAddInputs(8.F, 1.F, 1.F, 1.F)));
    release.release();
    ExpectDynamicSum(Take(std::move(admissions[0])), 4.F, 6.F);
    ExpectDynamicSum(Take(std::move(admissions[1])), 12.F, 14.F);
    ExpectDynamicSum(Take(std::move(admissions[2])), 10.F, 1.F);
    ExpectDynamicSum(Take(std::move(admissions[3])), 1.F, 1.F);
    ExpectDynamicSum(Take(std::move(admissions[4])), 9.F, 2.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 5, 0, 0, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_EQ(metrics.engine_run_count, 2U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 5U);
    EXPECT_EQ(metrics.batch_size_histogram[4], 1U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, InvalidRequestExcludedFromEngineMetrics) {
    auto scheduler = MakeMetricsBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission valid_a = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission invalid = scheduler->submit({NamedTensor{"A", {1, 2}, {1.F, 2.F}},
                                           NamedTensor{"A", {1, 2}, {3.F, 4.F}},
                                           NamedTensor{"B", {1, 2}, {5.F, 6.F}}});
    Admission valid_c = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    ExpectDynamicSum(Take(std::move(valid_a)), 4.F, 6.F);
    ExpectInferenceFailure(Take(std::move(invalid)), InferenceErrorKind::kDuplicateInput);
    ExpectDynamicSum(Take(std::move(valid_c)), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 2, 1, 0, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_EQ(metrics.engine_run_count, 1U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 2U);
    EXPECT_EQ(metrics.batch_size_histogram[2], 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, FullQueueRejectionDoesNotIncrementEngineMetrics) {
    auto scheduler = MakeScheduler(MetricsAddConfig(1, 1));
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission accepted = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    std::vector<NamedTensor> second = AddInputs(8.F, 7.F, 6.F, 5.F);
    const Admission rejected = scheduler->submit(std::move(second));
    EXPECT_EQ(rejected.status, AdmissionStatus::kRejectedFull);
    EXPECT_FALSE(rejected.completion.has_value());
    release.release();
    ExpectSum(Take(std::move(accepted)), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(scheduler->stats().rejected_full, 1U);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_EQ(metrics.engine_run_count, 1U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 1U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, DrainKeepsCompletedEngineMetrics) {
    const auto scheduler = MakeScheduler(MetricsAddConfig(1, 1));
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    ExpectWinner(scheduler->shutdown(ShutdownPolicy::kDrain), ShutdownPolicy::kDrain, true);
    const SchedulerMetrics first = scheduler->metrics();
    EXPECT_EQ(first.engine_run_count, 1U);
    ExpectHistogramInvariants(first);
    ExpectWinner(scheduler->shutdown(ShutdownPolicy::kDrain), ShutdownPolicy::kDrain, false);
    const SchedulerMetrics second = scheduler->metrics();
    EXPECT_EQ(second.engine_run_count, first.engine_run_count);
    EXPECT_EQ(second.requests_entered_engine_runs, first.requests_entered_engine_runs);
    EXPECT_EQ(second.batch_size_histogram, first.batch_size_histogram);
    EXPECT_EQ(second.total_engine_run_nanoseconds, first.total_engine_run_nanoseconds);
}

TEST_F(InferenceSchedulerTest, CancelQueuedDoesNotCreateEngineRun) {
    auto scheduler = MakeScheduler(MetricsAddConfig(1, 1));
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    ExpectWinner(CancelWhileBlocked(*scheduler, release), ShutdownPolicy::kCancelPending, true);
    ExpectCanceled(TakeBounded(std::move(admission)), 1);
    ExpectQuiescent(*scheduler, 0, 0, 0, 0, 1);
    ExpectEmptyMetrics(scheduler->metrics(), true, true);
}

TEST_F(InferenceSchedulerTest, CancelPreservesEarlierEngineRun) {
    auto scheduler = MakeScheduler(MetricsAddConfig(1, 2));
    std::atomic<int> loops{0};
    auto reached_second = std::make_shared<std::latch>(1);
    auto hold_second = std::make_shared<std::latch>(1);
    LatchRelease release{hold_second};
    SchedulerTestAccess::SetBeforeLoop(*scheduler, [&loops, reached_second, hold_second] {
        if (loops.fetch_add(1) >= 1) {
            reached_second->count_down();
            hold_second->wait();
        }
    });
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    reached_second->wait();
    Admission queued = scheduler->submit(AddInputs(5.F, 6.F, 7.F, 8.F));
    ExpectWinner(CancelWhileBlocked(*scheduler, release), ShutdownPolicy::kCancelPending, true);
    ExpectCanceled(TakeBounded(std::move(queued)), 2);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0, 1);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_EQ(metrics.engine_run_count, 1U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 1U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, MultiWorkerMetricsAggregate) {
    auto scheduler = MakeScheduler(MetricsAddConfig(2, 2));
    auto gate0 = std::make_shared<std::latch>(1);
    auto gate1 = std::make_shared<std::latch>(1);
    LatchRelease release0{gate0};
    LatchRelease release1{gate1};
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 0, gate0);
    SchedulerTestAccess::SetBeforePopForWorker(*scheduler, 1, gate1);
    RequireStart(*scheduler);
    Admission first = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(AddInputs(5.F, 6.F, 7.F, 8.F));
    release0.release();
    release1.release();
    ExpectSum(Take(std::move(first)), 4.F, 6.F);
    ExpectSum(Take(std::move(second)), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_EQ(metrics.engine_run_count, 2U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 2U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 2U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, MetricsBeforeStartAreFinalizedEmpty) {
    const auto scheduler = MakeScheduler(MetricsAddConfig(1, 1));
    ExpectEmptyMetrics(scheduler->metrics(), true, true);
}

TEST_F(InferenceSchedulerTest, MetricsWhileRunningAreNotFinalized) {
    auto scheduler = MakeScheduler(MetricsAddConfig(1, 1));
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    ExpectEmptyMetrics(scheduler->metrics(), true, false);
    release.release();
    ExpectSum(Take(std::move(admission)), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_TRUE(metrics.finalized);
    EXPECT_EQ(metrics.engine_run_count, 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, MetricsDoNotChangeCompletionStatus) {
    auto scheduler = MakeMetricsBatch(2, std::chrono::milliseconds{0}, 2);
    RequireStart(*scheduler);
    const SchedulerCompletion completion =
        Take(scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F)));
    ExpectDynamicSum(completion, 4.F, 6.F);
    EXPECT_EQ(completion.status, CompletionStatus::kSucceeded);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
}

TEST_F(InferenceSchedulerTest, UnbatchedDurationsHaveNoAssemblyOrSplit) {
    const auto scheduler = MakeScheduler(MetricsAddConfig(1, 1));
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_EQ(metrics.total_assembly_nanoseconds, 0U);
    EXPECT_EQ(metrics.total_split_nanoseconds, 0U);
    EXPECT_GE(metrics.total_engine_run_nanoseconds, 0U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, BatchedDurationsAreNonnegative) {
    auto scheduler = MakeMetricsBatch(2, std::chrono::milliseconds{0}, 2);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    ExpectDynamicSum(Take(std::move(first)), 4.F, 6.F);
    ExpectDynamicSum(Take(std::move(second)), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_GE(metrics.total_assembly_nanoseconds, 0U);
    EXPECT_GE(metrics.total_engine_run_nanoseconds, 0U);
    EXPECT_GE(metrics.total_split_nanoseconds, 0U);
    EXPECT_EQ(metrics.engine_run_count, 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, InjectedUnbatchedThrowDoesNotCountEngineRun) {
    auto scheduler = MakeScheduler(MetricsAddConfig(1, 1));
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    RequireStart(*scheduler);
    const SchedulerCompletion failed = Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F)));
    EXPECT_EQ(failed.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 0, 0, 1, 0);
    ExpectEmptyMetrics(scheduler->metrics(), true, true);
}

TEST_F(InferenceSchedulerTest, CollectFaultDoesNotCountEngineRun) {
    auto scheduler = MakeMetricsBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetThrowAfterBatchCollect(*scheduler, 1);
    RequireStart(*scheduler);
    Admission owned = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission queued = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    EXPECT_EQ(TakeBounded(std::move(owned)).status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(TakeBounded(std::move(queued)).status, CompletionStatus::kSchedulerFailed);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 0, 0, 1, 1);
    ExpectEmptyMetrics(scheduler->metrics(), true, true);
}

TEST_F(InferenceSchedulerTest, ThrowBeforeBatchRunDoesNotCountEngineRun) {
    auto scheduler = MakeMetricsBatch(4, std::chrono::milliseconds{0}, 4);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    RequireStart(*scheduler);
    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    EXPECT_EQ(Take(std::move(first)).status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(Take(std::move(second)).status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 0, 0, 2, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_EQ(metrics.engine_run_count, 0U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 0U);
    EXPECT_GE(metrics.total_assembly_nanoseconds, 0U);
    EXPECT_EQ(metrics.total_engine_run_nanoseconds, 0U);
    EXPECT_EQ(metrics.total_split_nanoseconds, 0U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, LaterWorkerFailureKeepsEarlierEngineRun) {
    auto scheduler = MakeScheduler(MetricsAddConfig(1, 2));
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    const SchedulerCompletion failed = Take(scheduler->submit(AddInputs(5.F, 6.F, 7.F, 8.F)));
    EXPECT_EQ(failed.status, CompletionStatus::kWorkerFailed);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 1, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_EQ(metrics.engine_run_count, 1U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 1U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, MetricsStayUnfinalizedUntilJoinAfterWorkerFailure) {
    auto scheduler = MakeScheduler(MetricsAddConfig(1, 2));
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 0);
    const SchedulerCompletion failed = Take(scheduler->submit(AddInputs(5.F, 6.F, 7.F, 8.F)));
    EXPECT_EQ(failed.status, CompletionStatus::kWorkerFailed);
    const SchedulerMetrics before_join = scheduler->metrics();
    EXPECT_TRUE(before_join.enabled);
    EXPECT_FALSE(before_join.finalized);
    EXPECT_EQ(before_join.engine_run_count, 0U);
    EXPECT_EQ(before_join.requests_entered_engine_runs, 0U);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 1, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_TRUE(metrics.finalized);
    EXPECT_EQ(metrics.engine_run_count, 1U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 1U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, CapOneBatchingRecordsSizeOnePerRequest) {
    BatchRuns runs;
    auto scheduler = MakeMetricsBatch(1, std::chrono::milliseconds{0}, 2);
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    ExpectDynamicSum(Take(std::move(first)), 4.F, 6.F);
    ExpectDynamicSum(Take(std::move(second)), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{1, 1}));
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
    const SchedulerMetrics first_read = scheduler->metrics();
    const SchedulerMetrics second_read = scheduler->metrics();
    EXPECT_TRUE(first_read.finalized);
    EXPECT_EQ(first_read.engine_run_count, 2U);
    EXPECT_EQ(first_read.requests_entered_engine_runs, 2U);
    EXPECT_EQ(first_read.batch_size_histogram[1], 2U);
    EXPECT_EQ(first_read.batch_size_histogram[2], 0U);
    EXPECT_GE(first_read.total_assembly_nanoseconds, 0U);
    EXPECT_GE(first_read.total_engine_run_nanoseconds, 0U);
    EXPECT_GE(first_read.total_split_nanoseconds, 0U);
    ExpectHistogramInvariants(first_read);
    EXPECT_EQ(second_read.enabled, first_read.enabled);
    EXPECT_EQ(second_read.finalized, first_read.finalized);
    EXPECT_EQ(second_read.engine_run_count, first_read.engine_run_count);
    EXPECT_EQ(second_read.requests_entered_engine_runs, first_read.requests_entered_engine_runs);
    EXPECT_EQ(second_read.batch_size_histogram, first_read.batch_size_histogram);
    EXPECT_EQ(second_read.total_assembly_nanoseconds, first_read.total_assembly_nanoseconds);
    EXPECT_EQ(second_read.total_engine_run_nanoseconds, first_read.total_engine_run_nanoseconds);
    EXPECT_EQ(second_read.total_split_nanoseconds, first_read.total_split_nanoseconds);
}

TEST_F(InferenceSchedulerTest, TimedPartialBatchRecordsTheSingleOwnedRequest) {
    BatchRuns runs;
    auto scheduler = MakeMetricsBatch(4, std::chrono::milliseconds{30}, 2);
    auto reached = std::make_shared<std::latch>(1);
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, reached);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    reached->wait();
    ExpectDynamicSum(TakeBounded(std::move(admission)), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{1}));
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_TRUE(metrics.finalized);
    EXPECT_EQ(metrics.engine_run_count, 1U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 1U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 1U);
    EXPECT_EQ(metrics.batch_size_histogram[4], 0U);
    EXPECT_GE(metrics.total_assembly_nanoseconds, 0U);
    EXPECT_GE(metrics.total_engine_run_nanoseconds, 0U);
    EXPECT_GE(metrics.total_split_nanoseconds, 0U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, DrainDuringBatchWaitFinalizesOnlyTheOwnedRun) {
    BatchRuns runs;
    auto hold = std::make_shared<std::latch>(1);
    auto scheduler = MakeMetricsBatch(4, std::chrono::milliseconds{1000}, 2);
    ShutdownResult shutdown;
    JoinThread stopper;
    auto reached = std::make_shared<std::latch>(1);
    LatchRelease release{hold};
    SchedulerTestAccess::SetReachedBatchWait(*scheduler, reached);
    SchedulerTestAccess::SetBeforeBatchWait(*scheduler, hold);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    reached->wait();
    auto closed = std::make_shared<std::latch>(1);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    stopper.thread = std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kDrain); });
    closed->wait();
    ExpectEmptyMetrics(scheduler->metrics(), true, false);
    release.release();
    stopper.thread.join();
    ExpectWinner(shutdown, ShutdownPolicy::kDrain, true);
    ExpectDynamicSum(Take(std::move(admission)), 4.F, 6.F);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{1}));
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_TRUE(metrics.finalized);
    EXPECT_EQ(metrics.engine_run_count, 1U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 1U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, CancelPendingMetricsStayUnfinalizedUntilJoin) {
    auto blocked = std::make_shared<std::latch>(1);
    auto scheduler = MakeScheduler(MetricsAddConfig(1, 1));
    ShutdownResult shutdown;
    JoinThread stopper;
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    RequireStart(*scheduler);
    Admission admission = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    auto closed = std::make_shared<std::latch>(1);
    SchedulerTestAccess::SetAfterDrainClosed(*scheduler, closed);
    stopper.thread =
        std::thread([&] { shutdown = scheduler->shutdown(ShutdownPolicy::kCancelPending); });
    closed->wait();
    ExpectEmptyMetrics(scheduler->metrics(), true, false);
    release.release();
    stopper.thread.join();
    ExpectWinner(shutdown, ShutdownPolicy::kCancelPending, true);
    ExpectCanceled(TakeBounded(std::move(admission)), 1);
    ExpectQuiescent(*scheduler, 0, 0, 0, 0, 1);
    ExpectEmptyMetrics(scheduler->metrics(), true, true);
}

TEST_F(InferenceSchedulerTest, UnbatchedInferenceErrorIsAnEngineRun) {
    const auto scheduler = MakeScheduler(MetricsAddConfig(1, 4));
    RequireStart(*scheduler);
    const SchedulerCompletion missing =
        Take(scheduler->submit({NamedTensor{"A", {2}, {1.F, 2.F}}}));
    EXPECT_EQ(missing.status, CompletionStatus::kInferenceFailed);
    EXPECT_EQ(missing.inference_error.kind, InferenceErrorKind::kMissingInput);
    const SchedulerCompletion shape = Take(scheduler->submit(
        {NamedTensor{"A", {3}, {1.F, 2.F, 3.F}}, NamedTensor{"B", {2}, {3.F, 4.F}}}));
    EXPECT_EQ(shape.status, CompletionStatus::kInferenceFailed);
    EXPECT_EQ(shape.inference_error.kind, InferenceErrorKind::kShapeMismatch);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 2, 0, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_TRUE(metrics.finalized);
    EXPECT_EQ(metrics.engine_run_count, 3U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 3U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 3U);
    EXPECT_EQ(metrics.total_assembly_nanoseconds, 0U);
    EXPECT_EQ(metrics.total_split_nanoseconds, 0U);
    EXPECT_GE(metrics.total_engine_run_nanoseconds, 0U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, DisabledMetricsSkipABatchedRun) {
    BatchRuns runs;
    auto blocked = std::make_shared<std::latch>(1);
    auto scheduler = MakeBatch(4, std::chrono::milliseconds{0}, 4);
    LatchRelease release{blocked};
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&runs](std::size_t count) { runs.Record(count); });
    RequireStart(*scheduler);
    ExpectEmptyMetrics(scheduler->metrics(), false, false);
    Admission first = scheduler->submit(DynamicAddInputs(1.F, 2.F, 3.F, 4.F));
    Admission second = scheduler->submit(DynamicAddInputs(5.F, 6.F, 7.F, 8.F));
    release.release();
    ExpectDynamicSum(Take(std::move(first)), 4.F, 6.F);
    ExpectDynamicSum(Take(std::move(second)), 12.F, 14.F);
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    EXPECT_EQ(runs.Snapshot(), (std::vector<std::size_t>{2}));
    ExpectQuiescent(*scheduler, 2, 0, 0, 0);
    ExpectEmptyMetrics(scheduler->metrics(), false, true);
}

TEST_F(InferenceSchedulerTest, RejectionAfterShutdownDoesNotCreateAnEngineRun) {
    const auto scheduler = MakeScheduler(MetricsAddConfig(1, 1));
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(8.F, 0.F, 1.F, 0.F))), 9.F, 0.F);
    ExpectWinner(scheduler->shutdown(ShutdownPolicy::kDrain), ShutdownPolicy::kDrain, true);
    const SchedulerMetrics before = scheduler->metrics();
    EXPECT_TRUE(before.finalized);
    EXPECT_EQ(before.engine_run_count, 1U);
    EXPECT_EQ(before.batch_size_histogram[1], 1U);
    ExpectHistogramInvariants(before);
    const Admission rejected = scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F));
    EXPECT_EQ(rejected.status, AdmissionStatus::kRejectedClosed);
    EXPECT_FALSE(rejected.completion.has_value());
    ASSERT_TRUE(rejected.inputs.has_value());
    ASSERT_EQ(rejected.inputs->size(), 2U);
    EXPECT_EQ(rejected.inputs->front().name, "A");
    ASSERT_EQ(rejected.inputs->front().values.size(), 2U);
    EXPECT_FLOAT_EQ(rejected.inputs->front().values[0], 1.F);
    EXPECT_FLOAT_EQ(rejected.inputs->front().values[1], 2.F);
    EXPECT_EQ(rejected.inputs->back().name, "B");
    ASSERT_EQ(rejected.inputs->back().values.size(), 2U);
    EXPECT_FLOAT_EQ(rejected.inputs->back().values[0], 3.F);
    EXPECT_FLOAT_EQ(rejected.inputs->back().values[1], 4.F);
    EXPECT_EQ(scheduler->stats().accepted, 1U);
    EXPECT_EQ(scheduler->stats().rejected_closed, 1U);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
    const SchedulerMetrics after = scheduler->metrics();
    EXPECT_EQ(after.enabled, before.enabled);
    EXPECT_EQ(after.finalized, before.finalized);
    EXPECT_EQ(after.engine_run_count, before.engine_run_count);
    EXPECT_EQ(after.requests_entered_engine_runs, before.requests_entered_engine_runs);
    EXPECT_EQ(after.batch_size_histogram, before.batch_size_histogram);
    EXPECT_EQ(after.total_assembly_nanoseconds, before.total_assembly_nanoseconds);
    EXPECT_EQ(after.total_engine_run_nanoseconds, before.total_engine_run_nanoseconds);
    EXPECT_EQ(after.total_split_nanoseconds, before.total_split_nanoseconds);
}

TEST_F(InferenceSchedulerTest, IdleSiblingLeavesCompletedRunUnfinalizedUntilJoin) {
    std::atomic<bool> saw_published{false};
    auto hold_sibling = std::make_shared<std::latch>(1);
    auto scheduler = MakeScheduler(MetricsAddConfig(2, 1));
    JoinThread reader;
    LatchRelease release_sibling{hold_sibling};
    SchedulerTestAccess::SetHoldOtherWorkers(*scheduler, hold_sibling);
    RequireStart(*scheduler);
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    reader.thread = std::thread([&] {
        for (int sample = 0; sample < 100; ++sample) {
            const SchedulerMetrics snapshot = scheduler->metrics();
            if (!snapshot.enabled || snapshot.finalized || snapshot.engine_run_count != 0 ||
                snapshot.requests_entered_engine_runs != 0 ||
                snapshot.total_assembly_nanoseconds != 0 ||
                snapshot.total_engine_run_nanoseconds != 0 ||
                snapshot.total_split_nanoseconds != 0) {
                saw_published.store(true, std::memory_order_relaxed);
            }
            for (const std::uint64_t bin : snapshot.batch_size_histogram) {
                if (bin != 0) {
                    saw_published.store(true, std::memory_order_relaxed);
                }
            }
        }
    });
    reader.thread.join();
    EXPECT_FALSE(saw_published.load());
    ExpectEmptyMetrics(scheduler->metrics(), true, false);
    release_sibling.release();
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 0, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_TRUE(metrics.finalized);
    EXPECT_EQ(metrics.engine_run_count, 1U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 1U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 1U);
    ExpectHistogramInvariants(metrics);
}

TEST_F(InferenceSchedulerTest, FailedSiblingKeepsTheOtherWorkersEngineRun) {
    auto both_entered = std::make_shared<std::latch>(2);
    auto success_held = std::make_shared<std::latch>(1);
    auto hold_sibling = std::make_shared<std::latch>(1);
    auto hold_success_worker = std::make_shared<std::latch>(1);
    std::atomic<int> entries{0};
    auto scheduler = MakeScheduler(MetricsAddConfig(2, 1));
    LatchRelease release_sibling{hold_sibling};
    LatchRelease release_success{hold_success_worker};
    SchedulerTestAccess::SetHoldOtherWorkers(*scheduler, hold_sibling);
    SchedulerTestAccess::SetBeforeLoop(*scheduler, [&] {
        const int seen = entries.fetch_add(1);
        if (seen < 2) {
            both_entered->count_down();
            return;
        }
        if (seen == 2) {
            success_held->count_down();
        }
        hold_success_worker->wait();
    });
    RequireStart(*scheduler);
    both_entered->wait();
    ExpectSum(Take(scheduler->submit(AddInputs(1.F, 2.F, 3.F, 4.F))), 4.F, 6.F);
    success_held->wait();
    SchedulerTestAccess::SetThrowInWorker(*scheduler, 1);
    release_sibling.release();
    const SchedulerCompletion failed = Take(scheduler->submit(AddInputs(5.F, 6.F, 7.F, 8.F)));
    EXPECT_EQ(failed.status, CompletionStatus::kWorkerFailed);
    ExpectEmptyMetrics(scheduler->metrics(), true, false);
    release_success.release();
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    ExpectQuiescent(*scheduler, 1, 0, 1, 0);
    const SchedulerMetrics metrics = scheduler->metrics();
    EXPECT_TRUE(metrics.finalized);
    EXPECT_EQ(metrics.engine_run_count, 1U);
    EXPECT_EQ(metrics.requests_entered_engine_runs, 1U);
    EXPECT_EQ(metrics.batch_size_histogram[1], 1U);
    ExpectHistogramInvariants(metrics);
}

} // namespace
