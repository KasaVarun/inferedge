#include "inferedge/scheduler_benchmark_cli.hpp"
#include "inferedge/scheduler_benchmark_json.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <spawn.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#ifndef INFEREDGE_BENCH_SCHEDULER_EXECUTABLE
#error "CMake must define INFEREDGE_BENCH_SCHEDULER_EXECUTABLE"
#endif

#ifndef INFEREDGE_BATCH_ADD_MODEL_PATH
#error "CMake must define INFEREDGE_BATCH_ADD_MODEL_PATH"
#endif

#ifndef INFEREDGE_INVALID_MODEL_PATH
#error "CMake must define INFEREDGE_INVALID_MODEL_PATH"
#endif

extern "C" char** environ;

namespace {

using namespace std::chrono_literals;
using inferedge::SchedulerBenchmarkCliAction;
using inferedge::SchedulerBenchmarkConfig;
using inferedge::SchedulerBenchmarkErrorKind;
using inferedge::SchedulerBenchmarkPlanError;
using inferedge::SchedulerBenchmarkReport;
using inferedge::SchedulerBenchmarkRun;
using inferedge::SchedulerWorkloadKind;

SchedulerBenchmarkConfig MeasuredConfig() {
    SchedulerBenchmarkConfig config;
    config.model_path = INFEREDGE_BATCH_ADD_MODEL_PATH;
    config.worker_count = 1;
    config.queue_capacity = 8;
    config.enable_batching = false;
    config.max_batch_size = 1;
    config.max_batch_delay = 0ms;
    config.enable_metrics = true;
    config.workload = SchedulerWorkloadKind::kOpenLoopConstant;
    config.target_requests_per_second = 1000000.0;
    config.producer_count = 1;
    config.max_offered = 4;
    config.max_outstanding = 4;
    config.duration = 1000ms;
    config.completion_timeout = 5000ms;
    config.lag_tolerance = 5000ms;
    config.check_interval = 1;
    return config;
}

void ExpectAccounting(const SchedulerBenchmarkReport& report) {
    const std::uint64_t rejections =
        report.rejections.full + report.rejections.closed + report.rejections.not_running;
    const std::uint64_t terminals = report.succeeded + report.failed_inference +
                                    report.failed_worker + report.failed_scheduler +
                                    report.canceled + report.timeouts;
    EXPECT_EQ(report.scheduled_opportunities, report.submit_attempts +
                                                  report.generator_skips_outstanding +
                                                  report.generator_skips_lag);
    EXPECT_EQ(report.submit_attempts, report.accepted + rejections);
    EXPECT_EQ(report.accepted, terminals);
}

void ExpectHistogram(const SchedulerBenchmarkReport& report) {
    ASSERT_EQ(report.metrics.batch_size_histogram.size(), 33U);
    EXPECT_EQ(report.metrics.batch_size_histogram[0], 0U);
    std::uint64_t runs = 0;
    std::uint64_t weighted = 0;
    for (std::size_t size = 1; size < report.metrics.batch_size_histogram.size(); ++size) {
        const std::uint64_t bin = report.metrics.batch_size_histogram[size];
        runs += bin;
        weighted += static_cast<std::uint64_t>(size) * bin;
    }
    EXPECT_EQ(runs, report.metrics.engine_run_count);
    EXPECT_EQ(weighted, report.metrics.requests_entered_engine_runs);
    EXPECT_TRUE(report.metrics.enabled);
    EXPECT_TRUE(report.metrics.finalized);
    EXPECT_FALSE(report.histogram_saturated);
}

struct ProcessOutput {
    int status{0};
    std::string output;
    std::string error;
};

std::string ReadPipe(int fd) {
    std::string text;
    char buffer[256];
    while (true) {
        const ssize_t count = ::read(fd, buffer, sizeof(buffer));
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (count == 0) {
            break;
        }
        text.append(buffer, static_cast<std::size_t>(count));
    }
    return text;
}

ProcessOutput RunProcess(const std::vector<std::string>& args) {
    int output_pipe[2] = {-1, -1};
    int error_pipe[2] = {-1, -1};
    EXPECT_EQ(::pipe(output_pipe), 0);
    EXPECT_EQ(::pipe(error_pipe), 0);
    posix_spawn_file_actions_t actions;
    EXPECT_EQ(::posix_spawn_file_actions_init(&actions), 0);
    EXPECT_EQ(::posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDOUT_FILENO), 0);
    EXPECT_EQ(::posix_spawn_file_actions_adddup2(&actions, error_pipe[1], STDERR_FILENO), 0);
    EXPECT_EQ(::posix_spawn_file_actions_addclose(&actions, output_pipe[0]), 0);
    EXPECT_EQ(::posix_spawn_file_actions_addclose(&actions, error_pipe[0]), 0);
    EXPECT_EQ(::posix_spawn_file_actions_addclose(&actions, output_pipe[1]), 0);
    EXPECT_EQ(::posix_spawn_file_actions_addclose(&actions, error_pipe[1]), 0);
    std::vector<std::string> storage = args;
    std::vector<char*> argv;
    argv.reserve(storage.size() + 1);
    for (std::string& arg : storage) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    pid_t pid = 0;
    const int spawned = ::posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
    EXPECT_EQ(spawned, 0);
    ::posix_spawn_file_actions_destroy(&actions);
    ::close(output_pipe[1]);
    ::close(error_pipe[1]);
    ProcessOutput captured;
    captured.output = ReadPipe(output_pipe[0]);
    captured.error = ReadPipe(error_pipe[0]);
    ::close(output_pipe[0]);
    ::close(error_pipe[0]);
    int status = 0;
    EXPECT_EQ(::waitpid(pid, &status, 0), pid);
    captured.status = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return captured;
}

} // namespace

TEST(SchedulerBenchmarkCliTest, HelpShowsUsage) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli({});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kHelp);
    EXPECT_EQ(parsed.exit_code, 0);
    EXPECT_NE(parsed.message.find("Usage: inferedge-bench-scheduler"), std::string::npos);
}

TEST(SchedulerBenchmarkCliTest, VersionShowsToolName) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli({"--version"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kVersion);
    EXPECT_NE(parsed.message.find("inferedge-bench-scheduler 0.1.0"), std::string::npos);
}

TEST(SchedulerBenchmarkCliTest, RejectsUnknownOption) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli({"--not-a-flag"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kError);
    EXPECT_EQ(parsed.exit_code, 2);
}

TEST(SchedulerBenchmarkCliTest, RejectsMissingRequiredOption) {
    const auto parsed =
        inferedge::parse_scheduler_benchmark_cli({"--model", "model.onnx", "--workers", "1"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kError);
    EXPECT_EQ(parsed.exit_code, 2);
}

TEST(SchedulerBenchmarkCliTest, RejectsInvalidInteger) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli(
        {"--model", "model.onnx", "--workers", "-1", "--queue-capacity", "1", "--workload",
         "closed_loop", "--max-offered", "1", "--max-outstanding", "1", "--duration-ms", "10",
         "--completion-timeout-ms", "10"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kError);
    EXPECT_NE(parsed.message.find("invalid integer"), std::string::npos);
}

TEST(SchedulerBenchmarkCliTest, RejectsUnsupportedWorkload) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli(
        {"--model", "model.onnx", "--workers", "1", "--queue-capacity", "1", "--workload", "batch",
         "--max-offered", "1", "--max-outstanding", "1", "--duration-ms", "10",
         "--completion-timeout-ms", "10"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kError);
    EXPECT_NE(parsed.message.find("unsupported workload"), std::string::npos);
}

TEST(SchedulerBenchmarkCliTest, RejectsBatchSizeOutOfRange) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli(
        {"--model", "model.onnx", "--workers", "1", "--queue-capacity", "1", "--workload",
         "closed_loop", "--max-offered", "1", "--max-outstanding", "1", "--duration-ms", "10",
         "--completion-timeout-ms", "10", "--max-batch-size", "33"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kError);
}

TEST(SchedulerBenchmarkCliTest, RejectsOutstandingAboveOffered) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli(
        {"--model", "model.onnx", "--workers", "1", "--queue-capacity", "1", "--workload",
         "closed_loop", "--max-offered", "1", "--max-outstanding", "2", "--duration-ms", "10",
         "--completion-timeout-ms", "10"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kError);
    EXPECT_EQ(parsed.exit_code, 2);
}

TEST(SchedulerBenchmarkCliTest, RejectsClosedLoopSchedule) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli(
        {"--model", "model.onnx", "--workers", "1", "--queue-capacity", "1", "--workload",
         "closed_loop", "--target-rps", "10", "--max-offered", "1", "--max-outstanding", "1",
         "--duration-ms", "10", "--completion-timeout-ms", "10"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kError);
}

TEST(SchedulerBenchmarkCliTest, AdaptiveBatchingDefaultsOff) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli(
        {"--model", "model.onnx", "--workers", "1", "--queue-capacity", "1", "--workload",
         "closed_loop", "--max-offered", "1", "--max-outstanding", "1", "--duration-ms", "10",
         "--completion-timeout-ms", "10"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kRun);
    EXPECT_FALSE(parsed.config.adaptive_batching);
    EXPECT_NE(inferedge::scheduler_benchmark_help_message().find("--adaptive-batching"),
              std::string::npos);
}

TEST(SchedulerBenchmarkCliTest, AdaptiveBatchingFlagSetsTrue) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli({"--model",
                                                                  "model.onnx",
                                                                  "--workers",
                                                                  "1",
                                                                  "--queue-capacity",
                                                                  "1",
                                                                  "--workload",
                                                                  "closed_loop",
                                                                  "--max-offered",
                                                                  "1",
                                                                  "--max-outstanding",
                                                                  "1",
                                                                  "--duration-ms",
                                                                  "10",
                                                                  "--completion-timeout-ms",
                                                                  "10",
                                                                  "--enable-batching",
                                                                  "--max-batch-size",
                                                                  "4",
                                                                  "--max-batch-delay-ms",
                                                                  "2",
                                                                  "--adaptive-batching"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kRun);
    EXPECT_TRUE(parsed.config.adaptive_batching);
    EXPECT_TRUE(parsed.config.enable_batching);
}

TEST(SchedulerBenchmarkCliTest, RejectsAdaptiveWithoutBatching) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli({"--model",
                                                                  "model.onnx",
                                                                  "--workers",
                                                                  "1",
                                                                  "--queue-capacity",
                                                                  "1",
                                                                  "--workload",
                                                                  "closed_loop",
                                                                  "--max-offered",
                                                                  "1",
                                                                  "--max-outstanding",
                                                                  "1",
                                                                  "--duration-ms",
                                                                  "10",
                                                                  "--completion-timeout-ms",
                                                                  "10",
                                                                  "--max-batch-size",
                                                                  "4",
                                                                  "--max-batch-delay-ms",
                                                                  "2",
                                                                  "--adaptive-batching"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kError);
    EXPECT_NE(parsed.message.find("--enable-batching"), std::string::npos);
}

TEST(SchedulerBenchmarkCliTest, RejectsAdaptiveWithBatchSizeOne) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli({"--model",
                                                                  "model.onnx",
                                                                  "--workers",
                                                                  "1",
                                                                  "--queue-capacity",
                                                                  "1",
                                                                  "--workload",
                                                                  "closed_loop",
                                                                  "--max-offered",
                                                                  "1",
                                                                  "--max-outstanding",
                                                                  "1",
                                                                  "--duration-ms",
                                                                  "10",
                                                                  "--completion-timeout-ms",
                                                                  "10",
                                                                  "--enable-batching",
                                                                  "--max-batch-size",
                                                                  "1",
                                                                  "--max-batch-delay-ms",
                                                                  "2",
                                                                  "--adaptive-batching"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kError);
    EXPECT_NE(parsed.message.find("--max-batch-size"), std::string::npos);
}

TEST(SchedulerBenchmarkCliTest, RejectsAdaptiveWithZeroDelay) {
    const auto parsed = inferedge::parse_scheduler_benchmark_cli({"--model",
                                                                  "model.onnx",
                                                                  "--workers",
                                                                  "1",
                                                                  "--queue-capacity",
                                                                  "1",
                                                                  "--workload",
                                                                  "closed_loop",
                                                                  "--max-offered",
                                                                  "1",
                                                                  "--max-outstanding",
                                                                  "1",
                                                                  "--duration-ms",
                                                                  "10",
                                                                  "--completion-timeout-ms",
                                                                  "10",
                                                                  "--enable-batching",
                                                                  "--max-batch-size",
                                                                  "4",
                                                                  "--max-batch-delay-ms",
                                                                  "0",
                                                                  "--adaptive-batching"});
    EXPECT_EQ(parsed.action, SchedulerBenchmarkCliAction::kError);
    EXPECT_NE(parsed.message.find("--max-batch-delay-ms"), std::string::npos);
}

TEST(SchedulerBenchmarkPlanTest, ConstantScheduleIsSpaced) {
    const auto planned = inferedge::plan_constant_arrivals(100.0, 50ms, 10);
    ASSERT_EQ(planned.error, SchedulerBenchmarkPlanError::kNone);
    ASSERT_EQ(planned.plan.arrivals.size(), 5U);
    EXPECT_EQ(planned.plan.arrivals[0].offset, 0ms);
    EXPECT_EQ(planned.plan.arrivals[1].offset, 10ms);
    EXPECT_EQ(planned.plan.arrivals[4].offset, 40ms);
    ASSERT_EQ(planned.plan.phases.size(), 1U);
    EXPECT_EQ(planned.plan.phases[0].scheduled_opportunities, 5U);
}

TEST(SchedulerBenchmarkPlanTest, BurstScheduleRecordsPhaseBoundaries) {
    const auto planned = inferedge::plan_burst_arrivals(100.0, 50.0, 50ms, 50ms, 100ms, 20);
    ASSERT_EQ(planned.error, SchedulerBenchmarkPlanError::kNone);
    ASSERT_EQ(planned.plan.phases.size(), 2U);
    EXPECT_EQ(planned.plan.phases[0].scheduled_opportunities, 5U);
    EXPECT_EQ(planned.plan.phases[1].scheduled_opportunities, 3U);
    ASSERT_EQ(planned.plan.arrivals.size(), 8U);
    EXPECT_EQ(planned.plan.arrivals[5].offset, 50ms);
    EXPECT_EQ(planned.plan.arrivals[5].phase_index, 1U);
    EXPECT_EQ(planned.plan.arrivals[7].offset, 90ms);
}

TEST(SchedulerBenchmarkPlanTest, LagAboveToleranceIsLate) {
    const auto scheduled = std::chrono::steady_clock::time_point{10ms};
    const auto actual = scheduled + 5ms;
    const auto lag = inferedge::measure_generator_lag(scheduled, actual, 1ms);
    EXPECT_EQ(lag.lag, 5ms);
    EXPECT_TRUE(lag.exceeds_tolerance);
}

TEST(SchedulerBenchmarkPlanTest, EarlyLagDoesNotExceedTolerance) {
    const auto scheduled = std::chrono::steady_clock::time_point{10ms};
    const auto actual = scheduled - 1ms;
    const auto lag = inferedge::measure_generator_lag(scheduled, actual, 1ms);
    EXPECT_EQ(lag.lag, -1ms);
    EXPECT_FALSE(lag.exceeds_tolerance);
}

TEST(SchedulerBenchmarkPlanTest, FutureReadyDoesNotBlock) {
    std::promise<inferedge::SchedulerCompletion> pending;
    std::future<inferedge::SchedulerCompletion> waiting = pending.get_future();
    const auto started = std::chrono::steady_clock::now();
    EXPECT_FALSE(inferedge::scheduler_benchmark_future_ready(waiting));
    EXPECT_LT(std::chrono::steady_clock::now() - started, 50ms);
    std::promise<inferedge::SchedulerCompletion> ready_promise;
    std::future<inferedge::SchedulerCompletion> ready = ready_promise.get_future();
    ready_promise.set_value(inferedge::SchedulerCompletion{});
    EXPECT_TRUE(inferedge::scheduler_benchmark_future_ready(ready));
}

TEST(SchedulerBenchmarkPlanTest, RequestTensorMatchesAttempt) {
    const auto request = inferedge::scheduler_benchmark_request(1001);
    ASSERT_EQ(request.inputs.size(), 2U);
    EXPECT_EQ(request.inputs[0].shape, (std::vector<std::int64_t>{1, 2}));
    EXPECT_FLOAT_EQ(request.inputs[0].values[0], 1.F);
    EXPECT_FLOAT_EQ(request.inputs[0].values[1], 1.F);
    EXPECT_FLOAT_EQ(request.expected_c0, 2.F);
    EXPECT_FLOAT_EQ(request.expected_c1, 3.F);
    EXPECT_FALSE(inferedge::scheduler_benchmark_should_check(1, 2, false));
    EXPECT_TRUE(inferedge::scheduler_benchmark_should_check(2, 2, false));
    EXPECT_TRUE(inferedge::scheduler_benchmark_should_check(1, 2, true));
}

TEST(SchedulerBenchmarkTest, OpenLoopConstantCompletes) {
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(MeasuredConfig());
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.valid) << (run.report.invalid_reasons.empty()
                                          ? std::string{}
                                          : run.report.invalid_reasons.front());
    EXPECT_EQ(run.report.workload, "open_loop_constant");
    EXPECT_EQ(run.report.succeeded, 4U);
    EXPECT_EQ(run.report.correctness_mismatches, 0U);
    EXPECT_EQ(run.report.correctness_checked, 4U);
    EXPECT_EQ(run.report.completion_observation, "open_loop_poll");
    EXPECT_GT(run.report.latency.sample_count, 0U);
    EXPECT_TRUE(run.report.latency.percentiles_reported);
    EXPECT_FALSE(run.report.latency.precise_tail);
    EXPECT_GT(run.report.throughput.total_trial_seconds, 0.0);
    EXPECT_EQ(run.report.queue_depth, 0U);
    EXPECT_EQ(run.report.live_workers, 0U);
    ExpectAccounting(run.report);
    ExpectHistogram(run.report);
    EXPECT_EQ(inferedge::scheduler_benchmark_status(run), 0);
}

TEST(SchedulerBenchmarkTest, BurstTrialRunsBothPhases) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.workload = SchedulerWorkloadKind::kOpenLoopBurst;
    config.burst_high_requests_per_second = 200.0;
    config.burst_low_requests_per_second = 100.0;
    config.burst_high = 20ms;
    config.burst_low = 20ms;
    config.duration = 40ms;
    config.max_offered = 20;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.valid);
    ASSERT_GE(run.report.phases.size(), 2U);
    EXPECT_GT(run.report.phases[0].scheduled_opportunities, 0U);
    EXPECT_GT(run.report.phases[1].scheduled_opportunities, 0U);
    ExpectAccounting(run.report);
    ExpectHistogram(run.report);
}

TEST(SchedulerBenchmarkTest, ClosedLoopStaysBounded) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.workload = SchedulerWorkloadKind::kClosedLoop;
    config.producer_count = 2;
    config.max_outstanding = 2;
    config.max_offered = 4;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.valid);
    EXPECT_EQ(run.report.succeeded, 4U);
    EXPECT_EQ(run.report.generator_skips_outstanding, 0U);
    EXPECT_EQ(run.report.completion_observation, "closed_loop_producer_wait");
    ExpectAccounting(run.report);
    ExpectHistogram(run.report);
}

TEST(SchedulerBenchmarkTest, RejectionAfterShutdownReturnsInputs) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.max_offered = 1;
    config.max_outstanding = 1;
    config.test_submit_after_shutdown = true;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_FALSE(run.report.valid);
    EXPECT_EQ(run.report.succeeded, 1U);
    EXPECT_EQ(run.report.rejections.closed, 1U);
    EXPECT_EQ(run.report.rejections.full, 0U);
    EXPECT_EQ(run.report.generator_skips_outstanding, 0U);
    EXPECT_EQ(run.report.error_count, 0U);
    EXPECT_EQ(run.report.metrics.engine_run_count, 1U);
    ExpectAccounting(run.report);
    ExpectHistogram(run.report);
}

TEST(SchedulerBenchmarkTest, OutstandingCapSkipsWithoutSubmitting) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.hold_collector = true;
    config.max_outstanding = 1;
    config.max_offered = 4;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_FALSE(run.report.valid);
    EXPECT_EQ(run.report.submit_attempts, 1U);
    EXPECT_EQ(run.report.generator_skips_outstanding, 3U);
    EXPECT_NE(std::find(run.report.invalid_reasons.begin(), run.report.invalid_reasons.end(),
                        "outstanding_cap_suppressed_schedule"),
              run.report.invalid_reasons.end());
    EXPECT_EQ(run.report.rejections.full, 0U);
    EXPECT_EQ(run.report.succeeded, 1U);
    ExpectAccounting(run.report);
}

TEST(SchedulerBenchmarkTest, WarmupUsesASeparateScheduler) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.warmup_requests = 2;
    config.max_offered = 1;
    config.max_outstanding = 1;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.valid);
    EXPECT_EQ(run.report.succeeded, 1U);
    EXPECT_EQ(run.report.metrics.engine_run_count, 1U);
    EXPECT_EQ(run.report.warmup_requests, 2U);
    EXPECT_TRUE(run.report.warmup_uses_separate_scheduler);
    ExpectHistogram(run.report);
}

TEST(SchedulerBenchmarkTest, CapOneHistogramIsSizeOne) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.enable_batching = true;
    config.max_batch_size = 1;
    config.max_offered = 2;
    config.max_outstanding = 2;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.valid);
    EXPECT_EQ(run.report.metrics.batch_size_histogram[1], 2U);
    EXPECT_EQ(run.report.metrics.engine_run_count, 2U);
    EXPECT_EQ(run.report.metrics.requests_entered_engine_runs, 2U);
    ExpectHistogram(run.report);
}

TEST(SchedulerBenchmarkTest, CheckIntervalKeepsTheLastSuccess) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.check_interval = 2;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.valid);
    EXPECT_EQ(run.report.succeeded, 4U);
    EXPECT_EQ(run.report.correctness_checked, 2U);
    EXPECT_EQ(run.report.correctness_mismatches, 0U);
}

TEST(SchedulerBenchmarkTest, MissingModelReturnsNonzero) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.model_path = INFEREDGE_INVALID_MODEL_PATH;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    EXPECT_EQ(run.kind, SchedulerBenchmarkErrorKind::kInvalidConfig);
    EXPECT_NE(run.message.find("SHA-256"), std::string::npos);
    EXPECT_EQ(inferedge::scheduler_benchmark_status(run), 2);
}

TEST(SchedulerBenchmarkTest, MalformedConfigIsRejected) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.worker_count = 0;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    EXPECT_EQ(run.kind, SchedulerBenchmarkErrorKind::kInvalidConfig);
    EXPECT_EQ(inferedge::scheduler_benchmark_status(run), 2);
}

TEST(SchedulerBenchmarkTest, EarlyMaxOfferedStopIsNotTheConfiguredDuration) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.max_offered = 2;
    config.max_outstanding = 2;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.valid);
    EXPECT_EQ(run.report.arrival_stop_reason, "max_offered");
    EXPECT_EQ(run.report.succeeded, 2U);
    EXPECT_LT(run.report.scheduled_span, run.report.configured_duration);
    EXPECT_LT(run.report.arrival_window, run.report.configured_duration);
    EXPECT_GT(run.report.throughput.arrival_window_seconds, 0.0);
    EXPECT_EQ(run.report.generator_skips_outstanding, 0U);
}

TEST(SchedulerBenchmarkTest, JsonSchemaIsVersionThree) {
    SchedulerBenchmarkReport report;
    report.valid = true;
    report.model = "models/batch_add_f32.onnx";
    report.workload = "closed_loop";
    report.completion_observation = "closed_loop_producer_wait";
    report.metrics.enabled = true;
    report.metrics.finalized = true;
    report.warmup_uses_separate_scheduler = true;
    const auto json = inferedge::write_scheduler_benchmark_json(report, "0.1.0");
    EXPECT_TRUE(json.error.empty());
    EXPECT_NE(json.text.find("\"schema_version\": 3"), std::string::npos);
    EXPECT_NE(json.text.find("\"adaptive_batching\": false"), std::string::npos);
    EXPECT_NE(json.text.find("\"arrival_stop_reason\""), std::string::npos);
    EXPECT_NE(json.text.find("\"completion_timeout_bound\": \"future_observation_only\""),
              std::string::npos);
    EXPECT_NE(json.text.find("\"scheduled_span_nanoseconds\""), std::string::npos);
    EXPECT_EQ(json.text.back(), '\n');
    report.throughput.successful_per_total_trial_second = std::numeric_limits<double>::infinity();
    const auto rejected = inferedge::write_scheduler_benchmark_json(report, "0.1.0");
    EXPECT_FALSE(rejected.error.empty());
    EXPECT_TRUE(rejected.text.empty());
}

TEST(SchedulerBenchmarkTest, AdaptiveFalseIsReportedAndSerialized) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.max_offered = 1;
    config.max_outstanding = 1;
    EXPECT_FALSE(config.adaptive_batching);
    EXPECT_FALSE(inferedge::scheduler_config(config).adaptive_batching);
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_FALSE(run.report.adaptive_batching);
    EXPECT_EQ(run.report.adaptive_batching, inferedge::scheduler_config(config).adaptive_batching);
    const auto json = inferedge::write_scheduler_benchmark_json(run.report, "0.1.0");
    EXPECT_TRUE(json.error.empty());
    EXPECT_NE(json.text.find("\"schema_version\": 3"), std::string::npos);
    EXPECT_NE(json.text.find("\"adaptive_batching\": false"), std::string::npos);
}

TEST(SchedulerBenchmarkTest, AdaptiveTrueIsReportedAndSerialized) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.max_offered = 2;
    config.max_outstanding = 2;
    config.enable_batching = true;
    config.max_batch_size = 4;
    config.max_batch_delay = 2ms;
    config.adaptive_batching = true;
    config.model_path = INFEREDGE_BATCH_ADD_MODEL_PATH;
    EXPECT_TRUE(inferedge::scheduler_config(config).adaptive_batching);
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.adaptive_batching);
    EXPECT_EQ(run.report.adaptive_batching, inferedge::scheduler_config(config).adaptive_batching);
    const auto json = inferedge::write_scheduler_benchmark_json(run.report, "0.1.0");
    EXPECT_TRUE(json.error.empty());
    EXPECT_NE(json.text.find("\"schema_version\": 3"), std::string::npos);
    EXPECT_NE(json.text.find("\"adaptive_batching\": true"), std::string::npos);
}

TEST(SchedulerBenchmarkTest, AdaptiveTrueKeepsSchemaFourTimingKeys) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.max_offered = 1;
    config.max_outstanding = 1;
    config.enable_batching = true;
    config.max_batch_size = 4;
    config.max_batch_delay = 2ms;
    config.adaptive_batching = true;
    config.enable_completion_timing = true;
    config.model_path = INFEREDGE_BATCH_ADD_MODEL_PATH;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.adaptive_batching);
    EXPECT_TRUE(run.report.completion_timing_enabled);
    const auto json = inferedge::write_scheduler_benchmark_json(run.report, "0.1.0");
    EXPECT_TRUE(json.error.empty());
    EXPECT_NE(json.text.find("\"schema_version\": 4"), std::string::npos);
    EXPECT_NE(json.text.find("\"adaptive_batching\": true"), std::string::npos);
    EXPECT_NE(json.text.find("\"enable_completion_timing\": true"), std::string::npos);
    EXPECT_NE(json.text.find("\"pre_publication_latency\""), std::string::npos);
}

TEST(SchedulerBenchmarkTest, InjectedCheckFailureExitsNonzero) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.max_offered = 1;
    config.max_outstanding = 1;
    config.test_fail_checked_successes = 1;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_FALSE(run.report.valid);
    EXPECT_GT(run.report.correctness_mismatches, 0U);
    EXPECT_EQ(inferedge::scheduler_benchmark_status(run), 1);
}

TEST(SchedulerBenchmarkTest, LagStopDoesNotSubmitTheTail) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.max_offered = 3;
    config.max_outstanding = 3;
    config.lag_tolerance = 1ms;
    config.test_submit_delay = 20ms;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_FALSE(run.report.valid);
    EXPECT_EQ(run.report.submit_attempts, 0U);
    EXPECT_EQ(run.report.generator_skips_lag, 3U);
    EXPECT_EQ(run.report.arrival_stop_reason, "generator_lag");
    EXPECT_EQ(run.report.metrics.engine_run_count, 0U);
    ExpectAccounting(run.report);
}

TEST(SchedulerBenchmarkProcessTest, HelpProcessExitsZero) {
    const ProcessOutput process = RunProcess({INFEREDGE_BENCH_SCHEDULER_EXECUTABLE, "--help"});
    EXPECT_EQ(process.status, 0);
    EXPECT_NE(process.output.find("Usage: inferedge-bench-scheduler"), std::string::npos);
}

TEST(SchedulerBenchmarkProcessTest, UsageErrorExitsNonzero) {
    const ProcessOutput process =
        RunProcess({INFEREDGE_BENCH_SCHEDULER_EXECUTABLE, "--workers", "0"});
    EXPECT_EQ(process.status, 2);
    EXPECT_NE(process.error.find("error:"), std::string::npos);
}

TEST(SchedulerBenchmarkProcessTest, MissingModelProcessExitsNonzero) {
    const ProcessOutput process = RunProcess({INFEREDGE_BENCH_SCHEDULER_EXECUTABLE,
                                              "--model",
                                              INFEREDGE_INVALID_MODEL_PATH,
                                              "--workers",
                                              "1",
                                              "--queue-capacity",
                                              "1",
                                              "--workload",
                                              "closed_loop",
                                              "--producer-count",
                                              "1",
                                              "--max-offered",
                                              "1",
                                              "--max-outstanding",
                                              "1",
                                              "--duration-ms",
                                              "50",
                                              "--completion-timeout-ms",
                                              "1000",
                                              "--format",
                                              "json"});
    EXPECT_EQ(process.status, 2);
    EXPECT_NE(process.output.find("\"schema_version\": 3"), std::string::npos);
    EXPECT_NE(process.output.find("\"error\": \"invalid_config\""), std::string::npos);
    EXPECT_NE(process.output.find("SHA-256"), std::string::npos);
    EXPECT_EQ(process.output.find("\"model_id\""), std::string::npos);
}

TEST(SchedulerBenchmarkProcessTest, JsonTrialExitsZero) {
    const ProcessOutput process = RunProcess({INFEREDGE_BENCH_SCHEDULER_EXECUTABLE,
                                              "--model",
                                              INFEREDGE_BATCH_ADD_MODEL_PATH,
                                              "--workers",
                                              "1",
                                              "--queue-capacity",
                                              "4",
                                              "--workload",
                                              "closed_loop",
                                              "--producer-count",
                                              "1",
                                              "--max-offered",
                                              "2",
                                              "--max-outstanding",
                                              "1",
                                              "--duration-ms",
                                              "1000",
                                              "--completion-timeout-ms",
                                              "5000",
                                              "--format",
                                              "json"});
    EXPECT_EQ(process.status, 0) << process.error << process.output;
    EXPECT_NE(process.output.find("\"schema_version\": 3"), std::string::npos);
    EXPECT_NE(process.output.find("\"workload\": \"closed_loop\""), std::string::npos);
    EXPECT_NE(process.output.find("\"valid\": true"), std::string::npos);
    EXPECT_EQ(process.output.back(), '\n');
}

TEST(SchedulerBenchmarkTimingTest, SignedLagPercentilesKeepEarlyArrivals) {
    using namespace std::chrono_literals;
    const std::chrono::nanoseconds lags[] = {-20ns, -10ns, 0ns, 30ns, 40ns};
    const inferedge::SchedulerBenchmarkLagSummary summary =
        inferedge::scheduler_benchmark_summarize_lag(lags);
    EXPECT_EQ(summary.count, 5U);
    EXPECT_EQ(summary.minimum.count(), -20);
    EXPECT_EQ(summary.maximum.count(), 40);
    EXPECT_TRUE(summary.percentiles_reported);
    EXPECT_FALSE(summary.precise_tail);
    EXPECT_EQ(summary.p50_nanoseconds, 0);
    EXPECT_EQ(summary.p95_nanoseconds, 40);
    EXPECT_EQ(summary.p99_nanoseconds, 40);
}

TEST(SchedulerBenchmarkTimingTest, Schema3OmitsTimingFields) {
    inferedge::SchedulerBenchmarkReport report;
    report.valid = true;
    report.workload = "closed_loop";
    report.arrival_stop_reason = "configured_duration";
    report.completion_observation = "closed_loop_producer_wait";
    report.model = "model";
    const inferedge::SchedulerBenchmarkJsonResult json =
        inferedge::write_scheduler_benchmark_json(report, "0.1.0");
    EXPECT_TRUE(json.error.empty()) << json.error;
    EXPECT_NE(json.text.find("\"schema_version\": 3"), std::string::npos);
    EXPECT_EQ(json.text.find("pre_publication_latency"), std::string::npos);
    EXPECT_EQ(json.text.find("p50_nanoseconds"), std::string::npos);
    EXPECT_EQ(json.text.find("post_publication"), std::string::npos);
    EXPECT_EQ(json.text.find("completion_timing_enabled"), std::string::npos);
}

TEST(SchedulerBenchmarkTimingTest, Schema4RequiresTimingObjectsAndRejectsNonfiniteMeans) {
    inferedge::SchedulerBenchmarkReport report;
    report.completion_timing_enabled = true;
    report.valid = true;
    report.workload = "open_loop_constant";
    report.arrival_stop_reason = "configured_duration";
    report.completion_observation = "open_loop_poll";
    report.model = "model";
    report.steady_clock_period_num = 1;
    report.steady_clock_period_den = 1000000000;
    report.succeeded = 1;
    report.pre_publication_latency.sample_count = 1;
    report.pre_publication_latency.percentiles_reported = true;
    report.pre_publication_latency.mean_nanoseconds = 10.0;
    report.pre_publication_to_observation_delay.sample_count = 1;
    report.pre_publication_to_observation_delay.mean_nanoseconds = 20.0;
    report.generator_lag.count = 1;
    report.generator_lag.p50_nanoseconds = -5;
    report.generator_lag.p95_nanoseconds = -5;
    report.generator_lag.p99_nanoseconds = -5;
    report.generator_lag.percentiles_reported = true;
    report.generator_lag.mean_nanoseconds = -5.0;
    const inferedge::SchedulerBenchmarkJsonResult json =
        inferedge::write_scheduler_benchmark_json(report, "0.1.0");
    EXPECT_TRUE(json.error.empty()) << json.error;
    EXPECT_NE(json.text.find("\"schema_version\": 4"), std::string::npos);
    EXPECT_NE(json.text.find("\"completion_timing_enabled\": true"), std::string::npos);
    EXPECT_NE(json.text.find("\"pre_publication_latency\""), std::string::npos);
    EXPECT_NE(json.text.find("\"pre_publication_to_observation_delay\""), std::string::npos);
    EXPECT_NE(json.text.find("\"p50_nanoseconds\": -5"), std::string::npos);
    EXPECT_EQ(json.text.find("post_publication"), std::string::npos);
    EXPECT_NE(json.text.find("\"phase_boundary_fidelity\": \"unavailable\""), std::string::npos);
    report.pre_publication_latency.mean_nanoseconds = std::numeric_limits<double>::infinity();
    const inferedge::SchedulerBenchmarkJsonResult rejected =
        inferedge::write_scheduler_benchmark_json(report, "0.1.0");
    EXPECT_FALSE(rejected.error.empty());
}

std::optional<std::int64_t> JsonIntAfter(const std::string& text, std::string_view object,
                                         std::string_view field) {
    const std::string object_key = "\"" + std::string{object} + "\"";
    const auto object_at = text.find(object_key);
    if (object_at == std::string::npos) {
        return std::nullopt;
    }
    const std::string field_key = "\"" + std::string{field} + "\": ";
    const auto field_at = text.find(field_key, object_at);
    if (field_at == std::string::npos) {
        return std::nullopt;
    }
    const char* begin = text.data() + field_at + field_key.size();
    const char* end = text.data() + text.size();
    std::int64_t value = 0;
    const std::from_chars_result parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc{}) {
        return std::nullopt;
    }
    return value;
}

TEST(SchedulerBenchmarkProcessTest, CompletionTimingSelectsSchema4) {
    const ProcessOutput process = RunProcess({INFEREDGE_BENCH_SCHEDULER_EXECUTABLE,
                                              "--model",
                                              INFEREDGE_BATCH_ADD_MODEL_PATH,
                                              "--workers",
                                              "1",
                                              "--queue-capacity",
                                              "4",
                                              "--workload",
                                              "closed_loop",
                                              "--producer-count",
                                              "1",
                                              "--max-offered",
                                              "1",
                                              "--max-outstanding",
                                              "1",
                                              "--duration-ms",
                                              "1000",
                                              "--completion-timeout-ms",
                                              "5000",
                                              "--completion-timing",
                                              "--format",
                                              "json"});
    EXPECT_EQ(process.status, 0) << process.error << process.output;
    EXPECT_NE(process.output.find("\"schema_version\": 4"), std::string::npos);
    EXPECT_NE(process.output.find("\"completion_timing_enabled\": true"), std::string::npos);
    EXPECT_NE(process.output.find("\"sample_count\": 1"), std::string::npos);
    EXPECT_EQ(process.output.find("post_publication"), std::string::npos);
    const auto producer =
        JsonIntAfter(process.output, "successful_e2e_latency", "minimum_nanoseconds");
    const auto pre = JsonIntAfter(process.output, "pre_publication_latency", "minimum_nanoseconds");
    const auto delay =
        JsonIntAfter(process.output, "pre_publication_to_observation_delay", "minimum_nanoseconds");
    ASSERT_TRUE(producer.has_value());
    ASSERT_TRUE(pre.has_value());
    ASSERT_TRUE(delay.has_value());
    EXPECT_EQ(*pre + *delay, *producer);
    EXPECT_EQ(JsonIntAfter(process.output, "pre_publication_latency", "sample_count"),
              std::optional<std::int64_t>{1});
    EXPECT_EQ(JsonIntAfter(process.output, "pre_publication_to_observation_delay", "sample_count"),
              std::optional<std::int64_t>{1});
    EXPECT_EQ(JsonIntAfter(process.output, "successful_e2e_latency", "sample_count"),
              std::optional<std::int64_t>{1});
}

std::filesystem::path ModelsDirectory() {
    return std::filesystem::path{INFEREDGE_BATCH_ADD_MODEL_PATH}.parent_path();
}

std::filesystem::path ModelFile(const char* name) {
    return ModelsDirectory() / name;
}

void ExpectDigest(const std::filesystem::path& path, const char* expected) {
    const inferedge::SchedulerBenchmarkDigest digest =
        inferedge::scheduler_benchmark_sha256_file(path);
    ASSERT_TRUE(digest.ok) << digest.message;
    EXPECT_EQ(digest.hex, expected);
    EXPECT_EQ(digest.hex.size(), 64U);
    const inferedge::SchedulerBenchmarkDigest again =
        inferedge::scheduler_benchmark_sha256_file(path);
    ASSERT_TRUE(again.ok);
    EXPECT_EQ(again.hex, digest.hex);
}

class TempDirectory {
public:
    TempDirectory() {
        path_ = std::filesystem::temp_directory_path() /
                ("inferedge-c4-" + std::to_string(static_cast<long long>(::getpid())));
        std::filesystem::remove_all(path_);
        std::filesystem::create_directory(path_);
    }

    ~TempDirectory() { std::filesystem::remove_all(path_); }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

void WriteBytes(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream output(path, std::ios::binary);
    ASSERT_TRUE(output);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(output);
}

SchedulerBenchmarkConfig NeuralRun(const char* model_file, inferedge::SchedulerBenchmarkModelId id,
                                   bool batching) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.model_path = ModelFile(model_file).string();
    config.model_id = id;
    config.max_offered = batching ? 4 : 2;
    config.max_outstanding = config.max_offered;
    config.queue_capacity = 8;
    config.enable_batching = batching;
    config.max_batch_size = batching ? 4 : 1;
    config.warmup_requests = 0;
    return config;
}

TEST(SchedulerBenchmarkSha256Test, KnownVectorsAndBlockBoundaries) {
    TempDirectory directory;
    const std::pair<const char*, std::string> cases[] = {
        {"empty", std::string{}},
        {"abc", std::string{"abc"}},
        {"nist",
         std::string{"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmn"
                     "opjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"}},
        {"nulls", std::string("\x00\x00\xff\x00", 4)},
        {"a55", std::string(55, 'a')},
        {"a56", std::string(56, 'a')},
        {"a63", std::string(63, 'a')},
        {"a64", std::string(64, 'a')},
        {"a65", std::string(65, 'a')},
    };
    const char* expected[] = {
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
        "bf4571113cd8c2262425e5bc5c38ace8909f4215569f49ad115f87445d596d02",
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
        "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
        "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34",
        "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
        "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0",
    };
    for (std::size_t index = 0; index < 9; ++index) {
        const std::filesystem::path file = directory.path() / cases[index].first;
        WriteBytes(file, cases[index].second);
        ExpectDigest(file, expected[index]);
    }
    const std::filesystem::path abc = directory.path() / "abc";
    std::fstream mutated(abc, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(mutated);
    mutated.seekp(0);
    mutated.put('d');
    mutated.close();
    const inferedge::SchedulerBenchmarkDigest changed =
        inferedge::scheduler_benchmark_sha256_file(abc);
    ASSERT_TRUE(changed.ok);
    EXPECT_NE(changed.hex, expected[1]);
    EXPECT_FALSE(inferedge::scheduler_benchmark_sha256_file(directory.path() / "missing").ok);
    EXPECT_EQ(inferedge::scheduler_benchmark_sha256_file(directory.path() / "missing").message,
              "file is missing");
    EXPECT_FALSE(inferedge::scheduler_benchmark_sha256_file(directory.path()).ok);
    EXPECT_EQ(inferedge::scheduler_benchmark_sha256_file(directory.path()).message,
              "file is not a regular file");
    const std::filesystem::path locked = directory.path() / "locked";
    WriteBytes(locked, std::string{"abc"});
    ASSERT_EQ(::chmod(locked.c_str(), 0), 0);
    const inferedge::SchedulerBenchmarkDigest unreadable =
        inferedge::scheduler_benchmark_sha256_file(locked);
    EXPECT_FALSE(unreadable.ok);
    EXPECT_EQ(::chmod(locked.c_str(), 0600), 0);
}

TEST(SchedulerBenchmarkSha256Test, PinnedArtifactsMatchReviewedDigests) {
    ExpectDigest(INFEREDGE_BATCH_ADD_MODEL_PATH,
                 "f33bbe0ad5420419dd2dbb5322f4172e213b854f1124c660b1d732e8f8ce3b23");
    ExpectDigest(ModelFile("mlp_small_f32.onnx"),
                 "bcf02cac789c65cb12785ccfd175827cb21f89574ef298af009b74d7131df1bd");
    ExpectDigest(ModelFile("mlp_medium_f32.onnx"),
                 "8d43a97b8d44d3d1f601570c8b09b08c90dda4977e6f8b7123eea8c0455024dd");
    ExpectDigest(ModelFile("gemm_f32.onnx"),
                 "a6203020b870eb04f9fd8e836cf4ccd930a57156a026d2a50163121d83debf32");
    ExpectDigest(ModelsDirectory() / "fixtures" / "mlp_small_f32.weights",
                 "b6335064c4e2c89add3ed54452e27dacf3ea7d485be6adbdd24fae1e2e5f1899");
    ExpectDigest(ModelsDirectory() / "fixtures" / "mlp_medium_f32.weights",
                 "d3dd2f6148a6684288d4f25b0c426eefde7227468be9120ddd08cd99cbb9059e");
    ExpectDigest(ModelsDirectory() / "fixtures" / "gemm_f32.weights",
                 "a335c8b12defcb885f8bada9ef1092655d024d5586fe011a9336ad2d472f82b6");
}

TEST(SchedulerBenchmarkCliTest, ModelIdDefaultsToAddAndRejectsBadSelectors) {
    const auto omitted = inferedge::parse_scheduler_benchmark_cli(
        {"--model", "model.onnx", "--workers", "1", "--queue-capacity", "1", "--workload",
         "closed_loop", "--max-offered", "1", "--max-outstanding", "1", "--duration-ms", "10",
         "--completion-timeout-ms", "10"});
    EXPECT_EQ(omitted.action, SchedulerBenchmarkCliAction::kRun);
    EXPECT_EQ(omitted.config.model_id, inferedge::SchedulerBenchmarkModelId::kBatchAddF32);
    const auto explicit_add = inferedge::parse_scheduler_benchmark_cli(
        {"--model-id=batch_add_f32", "--model", "model.onnx", "--workers", "1", "--queue-capacity",
         "1", "--workload", "closed_loop", "--max-offered", "1", "--max-outstanding", "1",
         "--duration-ms", "10", "--completion-timeout-ms", "10"});
    EXPECT_EQ(explicit_add.config.model_id, inferedge::SchedulerBenchmarkModelId::kBatchAddF32);
    const auto small = inferedge::parse_scheduler_benchmark_cli(
        {"--model", "model.onnx", "--model-id", "mlp_small_f32", "--workers", "1",
         "--queue-capacity", "1", "--workload", "closed_loop", "--max-offered", "1",
         "--max-outstanding", "1", "--duration-ms", "10", "--completion-timeout-ms", "10"});
    EXPECT_EQ(small.config.model_id, inferedge::SchedulerBenchmarkModelId::kMlpSmallF32);
    const auto missing = inferedge::parse_scheduler_benchmark_cli({"--model-id"});
    EXPECT_EQ(missing.action, SchedulerBenchmarkCliAction::kError);
    EXPECT_NE(missing.message.find("missing value for --model-id"), std::string::npos);
    const auto unknown = inferedge::parse_scheduler_benchmark_cli(
        {"--model-id", "resnet", "--model", "model.onnx", "--workers", "1", "--queue-capacity", "1",
         "--workload", "closed_loop", "--max-offered", "1", "--max-outstanding", "1",
         "--duration-ms", "10", "--completion-timeout-ms", "10"});
    EXPECT_EQ(unknown.action, SchedulerBenchmarkCliAction::kError);
    EXPECT_NE(unknown.message.find("unsupported model id"), std::string::npos);
    const auto duplicated = inferedge::parse_scheduler_benchmark_cli({"--model-id",
                                                                      "gemm_f32",
                                                                      "--model-id",
                                                                      "gemm_f32",
                                                                      "--model",
                                                                      "model.onnx",
                                                                      "--workers",
                                                                      "1",
                                                                      "--queue-capacity",
                                                                      "1",
                                                                      "--workload",
                                                                      "closed_loop",
                                                                      "--max-offered",
                                                                      "1",
                                                                      "--max-outstanding",
                                                                      "1",
                                                                      "--duration-ms",
                                                                      "10",
                                                                      "--completion-timeout-ms",
                                                                      "10"});
    EXPECT_EQ(duplicated.action, SchedulerBenchmarkCliAction::kError);
    EXPECT_NE(duplicated.message.find("specified more than once"), std::string::npos);
}

TEST(SchedulerBenchmarkNeuralTest, RequestsUseTheAttemptFormula) {
    const auto first = inferedge::scheduler_benchmark_request(
        inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, 1000);
    ASSERT_EQ(first.inputs.size(), 1U);
    EXPECT_EQ(first.inputs[0].name, "features");
    EXPECT_EQ(first.inputs[0].shape, (std::vector<std::int64_t>{1, 32}));
    EXPECT_FLOAT_EQ(first.inputs[0].values[0], static_cast<float>(1000 % 251) * 0x1p-10F);
    EXPECT_FLOAT_EQ(first.inputs[0].values[31], static_cast<float>(1000 % 251 + 31) * 0x1p-10F);
    const auto wrapped = inferedge::scheduler_benchmark_request(
        inferedge::SchedulerBenchmarkModelId::kGemmF32, 1000 + 251);
    EXPECT_EQ(wrapped.inputs[0].values[0], first.inputs[0].values[0]);
    const auto add = inferedge::scheduler_benchmark_request(
        inferedge::SchedulerBenchmarkModelId::kBatchAddF32, 1001);
    EXPECT_EQ(add.inputs.size(), 2U);
    EXPECT_FLOAT_EQ(add.expected_c0, 2.F);
}

TEST(SchedulerBenchmarkNeuralTest, OutputAndRejectionChecksRejectCorruptRows) {
    const auto request = inferedge::scheduler_benchmark_request(
        inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, 3);
    inferedge::NamedTensor output{"logits", {1, 8}, std::vector<float>(8, 0.F)};
    const std::vector<float> reference(8, 0.F);
    EXPECT_TRUE(inferedge::scheduler_benchmark_check_output(
                    inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, output, reference)
                    .matched);
    output.values[0] = 1e-6F;
    EXPECT_TRUE(inferedge::scheduler_benchmark_check_output(
                    inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, output, reference)
                    .matched);
    output.values[0] = 1.F;
    EXPECT_FALSE(inferedge::scheduler_benchmark_check_output(
                     inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, output, reference)
                     .matched);
    output.values[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(inferedge::scheduler_benchmark_check_output(
                     inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, output, reference)
                     .matched);
    output.values.assign(8, 0.F);
    output.values[3] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(inferedge::scheduler_benchmark_check_output(
                     inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, output, reference)
                     .matched);
    output.name = "scores";
    output.values.assign(8, 0.F);
    EXPECT_FALSE(inferedge::scheduler_benchmark_check_output(
                     inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, output, reference)
                     .matched);
    output.name = "logits";
    output.shape = {1, 7};
    output.values.assign(7, 0.F);
    EXPECT_FALSE(inferedge::scheduler_benchmark_check_output(
                     inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, output, reference)
                     .matched);
    const inferedge::NamedTensor submitted = request.inputs[0];
    EXPECT_TRUE(inferedge::scheduler_benchmark_check_returned_inputs(
                    std::span<const inferedge::NamedTensor>{&submitted, 1}, request.inputs)
                    .matched);
    inferedge::NamedTensor renamed = submitted;
    renamed.name = "image";
    EXPECT_FALSE(inferedge::scheduler_benchmark_check_returned_inputs(
                     std::span<const inferedge::NamedTensor>{&renamed, 1}, request.inputs)
                     .matched);
    inferedge::NamedTensor reshaped = submitted;
    reshaped.shape = {1, 16};
    reshaped.values.resize(16);
    EXPECT_FALSE(inferedge::scheduler_benchmark_check_returned_inputs(
                     std::span<const inferedge::NamedTensor>{&reshaped, 1}, request.inputs)
                     .matched);
    inferedge::NamedTensor altered = submitted;
    altered.values[0] += 1.F;
    EXPECT_FALSE(inferedge::scheduler_benchmark_check_returned_inputs(
                     std::span<const inferedge::NamedTensor>{&altered, 1}, request.inputs)
                     .matched);
    EXPECT_FALSE(inferedge::scheduler_benchmark_check_returned_inputs({}, request.inputs).matched);
    const float add_expected[] = {2.F, 3.F};
    inferedge::NamedTensor add_output{"C", {1, 2}, {2.F, 3.F}};
    EXPECT_TRUE(inferedge::scheduler_benchmark_check_output(
                    inferedge::SchedulerBenchmarkModelId::kBatchAddF32, add_output, add_expected)
                    .matched);
    add_output.values[1] = 4.F;
    EXPECT_FALSE(inferedge::scheduler_benchmark_check_output(
                     inferedge::SchedulerBenchmarkModelId::kBatchAddF32, add_output, add_expected)
                     .matched);
}

void ExpectNeuralTrial(const char* file, inferedge::SchedulerBenchmarkModelId id, bool batching) {
    const SchedulerBenchmarkRun run =
        inferedge::run_scheduler_benchmark(NeuralRun(file, id, batching));
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.valid) << (run.report.invalid_reasons.empty()
                                          ? std::string{}
                                          : run.report.invalid_reasons.front());
    EXPECT_EQ(run.report.succeeded, batching ? 4U : 2U);
    EXPECT_EQ(run.report.correctness_mismatches, 0U);
    EXPECT_EQ(run.report.correctness_checked, run.report.succeeded);
    EXPECT_EQ(run.report.failed_inference, 0U);
    EXPECT_TRUE(run.report.metrics.finalized);
    ExpectAccounting(run.report);
    ExpectHistogram(run.report);
    if (!batching) {
        EXPECT_EQ(run.report.metrics.batch_size_histogram[1], run.report.succeeded);
    }
    for (std::size_t size = batching ? 5 : 2; size < run.report.metrics.batch_size_histogram.size();
         ++size) {
        EXPECT_EQ(run.report.metrics.batch_size_histogram[size], 0U);
    }
    const auto json = inferedge::write_scheduler_benchmark_json(run.report, "0.1.0");
    EXPECT_TRUE(json.error.empty());
    EXPECT_NE(json.text.find("\"schema_version\": 3"), std::string::npos);
    EXPECT_EQ(json.text.find("\"model_id\""), std::string::npos);
}

TEST(SchedulerBenchmarkNeuralTest, UnbatchedAndBatchedRunsMatchTheReference) {
    ExpectNeuralTrial("mlp_small_f32.onnx", inferedge::SchedulerBenchmarkModelId::kMlpSmallF32,
                      false);
    ExpectNeuralTrial("mlp_small_f32.onnx", inferedge::SchedulerBenchmarkModelId::kMlpSmallF32,
                      true);
    ExpectNeuralTrial("mlp_medium_f32.onnx", inferedge::SchedulerBenchmarkModelId::kMlpMediumF32,
                      false);
    ExpectNeuralTrial("mlp_medium_f32.onnx", inferedge::SchedulerBenchmarkModelId::kMlpMediumF32,
                      true);
    ExpectNeuralTrial("gemm_f32.onnx", inferedge::SchedulerBenchmarkModelId::kGemmF32, false);
    ExpectNeuralTrial("gemm_f32.onnx", inferedge::SchedulerBenchmarkModelId::kGemmF32, true);
}

TEST(SchedulerBenchmarkNeuralTest, WrongIdentityDoesNotWarmUpOrSubmit) {
    SchedulerBenchmarkConfig missing = MeasuredConfig();
    missing.model_path = (ModelsDirectory() / "missing-model.onnx").string();
    missing.warmup_requests = 3;
    const SchedulerBenchmarkRun absent = inferedge::run_scheduler_benchmark(missing);
    EXPECT_EQ(absent.kind, SchedulerBenchmarkErrorKind::kInvalidConfig);
    EXPECT_NE(absent.message.find("file is missing"), std::string::npos);
    EXPECT_EQ(absent.report.succeeded, 0U);

    SchedulerBenchmarkConfig swapped =
        NeuralRun("mlp_small_f32.onnx", inferedge::SchedulerBenchmarkModelId::kGemmF32, false);
    swapped.warmup_requests = 2;
    const SchedulerBenchmarkRun wrong_id = inferedge::run_scheduler_benchmark(swapped);
    EXPECT_EQ(wrong_id.kind, SchedulerBenchmarkErrorKind::kInvalidConfig);
    EXPECT_NE(wrong_id.message.find("SHA-256"), std::string::npos);
    EXPECT_EQ(wrong_id.report.succeeded, 0U);

    TempDirectory directory;
    const std::filesystem::path copied = directory.path() / "mlp_small_f32.onnx";
    std::filesystem::copy_file(ModelFile("mlp_small_f32.onnx"), copied);
    std::fstream model(copied, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(model);
    model.seekp(-1, std::ios::end);
    model.put('\x5a');
    model.close();
    SchedulerBenchmarkConfig mutated =
        NeuralRun("mlp_small_f32.onnx", inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, false);
    mutated.model_path = copied.string();
    mutated.warmup_requests = 4;
    const SchedulerBenchmarkRun bad_model = inferedge::run_scheduler_benchmark(mutated);
    EXPECT_EQ(bad_model.kind, SchedulerBenchmarkErrorKind::kInvalidConfig);
    EXPECT_NE(bad_model.message.find("model SHA-256 mismatch"), std::string::npos);
    EXPECT_EQ(bad_model.report.succeeded, 0U);

    const std::filesystem::path good_model = directory.path() / "mlp_small_f32.onnx";
    std::filesystem::remove(good_model);
    std::filesystem::copy_file(ModelFile("mlp_small_f32.onnx"), good_model);
    std::filesystem::create_directory(directory.path() / "fixtures");
    const std::filesystem::path sidecar = directory.path() / "fixtures" / "mlp_small_f32.weights";
    std::filesystem::copy_file(ModelsDirectory() / "fixtures" / "mlp_small_f32.weights", sidecar);
    std::fstream weights(sidecar, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(weights);
    weights.seekp(0);
    weights.put('\x11');
    weights.close();
    mutated.model_path = good_model.string();
    const SchedulerBenchmarkRun bad_weights = inferedge::run_scheduler_benchmark(mutated);
    EXPECT_EQ(bad_weights.kind, SchedulerBenchmarkErrorKind::kInvalidConfig);
    EXPECT_NE(bad_weights.message.find("weight sidecar SHA-256 mismatch"), std::string::npos);
    EXPECT_EQ(bad_weights.report.succeeded, 0U);
}

TEST(SchedulerBenchmarkNeuralTest, ClosedRejectionRestoresTheFeaturesRow) {
    SchedulerBenchmarkConfig config =
        NeuralRun("mlp_small_f32.onnx", inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, false);
    config.max_offered = 1;
    config.max_outstanding = 1;
    config.test_submit_after_shutdown = true;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_EQ(run.report.succeeded, 1U);
    EXPECT_EQ(run.report.rejections.closed, 1U);
    EXPECT_EQ(run.report.correctness_mismatches, 0U);
    EXPECT_EQ(run.report.error_count, 0U);
    EXPECT_FALSE(run.report.valid);
}

TEST(SchedulerBenchmarkNeuralTest, ClosedLoopAndBatchedOpenLoopStayCorrect) {
    SchedulerBenchmarkConfig closed =
        NeuralRun("mlp_small_f32.onnx", inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, false);
    closed.workload = SchedulerWorkloadKind::kClosedLoop;
    closed.producer_count = 1;
    closed.max_offered = 3;
    closed.max_outstanding = 2;
    closed.target_requests_per_second = 0.0;
    const SchedulerBenchmarkRun closed_run = inferedge::run_scheduler_benchmark(closed);
    ASSERT_EQ(closed_run.kind, SchedulerBenchmarkErrorKind::kNone) << closed_run.message;
    EXPECT_TRUE(closed_run.report.valid);
    EXPECT_EQ(closed_run.report.succeeded, 3U);
    EXPECT_EQ(closed_run.report.correctness_mismatches, 0U);
    EXPECT_EQ(closed_run.report.correctness_checked, 3U);
    EXPECT_EQ(closed_run.report.failed_inference, 0U);
    ExpectAccounting(closed_run.report);

    SchedulerBenchmarkConfig gemm =
        NeuralRun("gemm_f32.onnx", inferedge::SchedulerBenchmarkModelId::kGemmF32, true);
    gemm.workload = SchedulerWorkloadKind::kClosedLoop;
    gemm.producer_count = 1;
    gemm.max_offered = 2;
    gemm.max_outstanding = 1;
    gemm.target_requests_per_second = 0.0;
    const SchedulerBenchmarkRun gemm_run = inferedge::run_scheduler_benchmark(gemm);
    ASSERT_EQ(gemm_run.kind, SchedulerBenchmarkErrorKind::kNone) << gemm_run.message;
    EXPECT_TRUE(gemm_run.report.valid);
    EXPECT_EQ(gemm_run.report.succeeded, 2U);
    EXPECT_EQ(gemm_run.report.correctness_mismatches, 0U);
    ExpectAccounting(gemm_run.report);
}

TEST(SchedulerBenchmarkNeuralTest, CheckIntervalStillChecksTheLastSuccess) {
    SchedulerBenchmarkConfig config =
        NeuralRun("mlp_small_f32.onnx", inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, false);
    config.max_offered = 4;
    config.max_outstanding = 4;
    config.check_interval = 3;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.valid);
    EXPECT_EQ(run.report.succeeded, 4U);
    EXPECT_EQ(run.report.correctness_mismatches, 0U);
    EXPECT_GE(run.report.correctness_checked, 2U);
    EXPECT_LE(run.report.correctness_checked, 4U);
}

TEST(SchedulerBenchmarkNeuralTest, SchemaFourKeysStayUnchangedForNeural) {
    SchedulerBenchmarkConfig config =
        NeuralRun("mlp_small_f32.onnx", inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, false);
    config.enable_completion_timing = true;
    config.max_offered = 1;
    config.max_outstanding = 1;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.valid);
    EXPECT_EQ(run.report.succeeded, 1U);
    EXPECT_EQ(run.report.correctness_mismatches, 0U);
    const auto json = inferedge::write_scheduler_benchmark_json(run.report, "0.1.0");
    EXPECT_TRUE(json.error.empty());
    EXPECT_NE(json.text.find("\"schema_version\": 4"), std::string::npos);
    EXPECT_NE(json.text.find("\"pre_publication_latency\""), std::string::npos);
    EXPECT_NE(json.text.find("\"pre_publication_to_observation_delay\""), std::string::npos);
    EXPECT_EQ(json.text.find("\"model_id\""), std::string::npos);
    EXPECT_EQ(json.text.find("post_publication"), std::string::npos);
}

TEST(SchedulerBenchmarkNeuralTest, AddRegressionStillUsesExactEquality) {
    SchedulerBenchmarkConfig config = MeasuredConfig();
    config.max_offered = 3;
    config.max_outstanding = 3;
    config.model_id = inferedge::SchedulerBenchmarkModelId::kBatchAddF32;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_TRUE(run.report.valid);
    EXPECT_EQ(run.report.succeeded, 3U);
    EXPECT_EQ(run.report.correctness_mismatches, 0U);
    const auto json = inferedge::write_scheduler_benchmark_json(run.report, "0.1.0");
    EXPECT_TRUE(json.error.empty());
    EXPECT_NE(json.text.find("\"schema_version\": 3"), std::string::npos);
    EXPECT_EQ(json.text.find("\"model_id\""), std::string::npos);
}

TEST(SchedulerBenchmarkNeuralTest, InjectedCheckFailureDoesNotFalsePass) {
    SchedulerBenchmarkConfig config =
        NeuralRun("mlp_small_f32.onnx", inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, false);
    config.max_offered = 1;
    config.max_outstanding = 1;
    config.test_fail_checked_successes = 1;
    const SchedulerBenchmarkRun run = inferedge::run_scheduler_benchmark(config);
    ASSERT_EQ(run.kind, SchedulerBenchmarkErrorKind::kNone) << run.message;
    EXPECT_FALSE(run.report.valid);
    EXPECT_GT(run.report.correctness_mismatches, 0U);
    EXPECT_EQ(inferedge::scheduler_benchmark_status(run), 1);
}

TEST(SchedulerBenchmarkNeuralTest, DistinctAttemptRowsFailWhenSwapped) {
    const auto early = inferedge::scheduler_benchmark_request(
        inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, 1);
    const auto late = inferedge::scheduler_benchmark_request(
        inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, 200);
    ASSERT_NE(early.inputs[0].values[0], late.inputs[0].values[0]);
    // Distinct stored expected rows must not match under the MLP tolerance.
    // Full end-to-end prep-before-submit association is covered by open/closed
    // neural trials that pass only when each Inflight keeps its own expected row.
    const std::vector<float> expected_early(8, 0.F);
    std::vector<float> expected_late(8, 0.F);
    expected_late[0] = 1.F;
    inferedge::NamedTensor late_as_early{"logits", {1, 8}, expected_late};
    EXPECT_FALSE(
        inferedge::scheduler_benchmark_check_output(
            inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, late_as_early, expected_early)
            .matched);
    inferedge::NamedTensor early_as_early{"logits", {1, 8}, expected_early};
    EXPECT_TRUE(
        inferedge::scheduler_benchmark_check_output(
            inferedge::SchedulerBenchmarkModelId::kMlpSmallF32, early_as_early, expected_early)
            .matched);
}
