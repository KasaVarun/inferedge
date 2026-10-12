#include "inferedge/concurrent_benchmark.hpp"

#include "inferedge/benchmark_runner.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <ratio>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace inferedge {
namespace {

constexpr const char* kBackend = "ONNX Runtime CPU";

struct StartGate {
    std::mutex mutex;
    std::condition_variable cv;
    std::uint32_t arrived{0};
    bool release{false};
    bool cancel{false};
};

struct WorkerSlot {
    bool failed{false};
    bool finished{false};
    std::uint32_t completed_requests{0};
    ConcurrentBenchmarkError error;
    std::vector<std::chrono::nanoseconds> samples;
    std::chrono::steady_clock::time_point finished_at{};
};

struct StartAborted : std::exception {
    std::uint32_t worker_index{0};
    explicit StartAborted(std::uint32_t index) : worker_index{index} {}
};

struct OutputProblem {
    ConcurrentBenchmarkErrorKind kind{ConcurrentBenchmarkErrorKind::kNone};
    std::string message;
};

void cancel_gate(StartGate& gate, std::atomic<bool>& stop) noexcept {
    stop.store(true, std::memory_order_release);
    try {
        std::lock_guard<std::mutex> lock(gate.mutex);
        gate.cancel = true;
    } catch (...) {
    }
    gate.cv.notify_all();
}

ConcurrentBenchmarkError
MakeError(ConcurrentBenchmarkErrorKind kind, std::string message,
          ConcurrentBenchmarkPhase phase = ConcurrentBenchmarkPhase::kNone,
          std::uint32_t worker_index = 0, std::uint32_t request_iteration = 0,
          InferenceErrorKind inference_kind = InferenceErrorKind::kNone,
          StatisticsErrorKind statistics_kind = StatisticsErrorKind::kNone) {
    ConcurrentBenchmarkError error;
    error.kind = kind;
    error.phase = phase;
    error.worker_index = worker_index;
    error.request_iteration = request_iteration;
    error.inference_kind = inference_kind;
    error.statistics_kind = statistics_kind;
    error.message = std::move(message);
    return error;
}

ConcurrentBenchmarkRun Fail(ConcurrentBenchmarkError error) {
    ConcurrentBenchmarkRun run;
    run.error = std::move(error);
    return run;
}

ConcurrentBenchmarkRun FailConfig(std::string message, std::uint32_t worker_index = 0) {
    return Fail(MakeError(ConcurrentBenchmarkErrorKind::kInvalidConfig, std::move(message),
                          ConcurrentBenchmarkPhase::kNone, worker_index));
}

enum class ShapeCountKind { kOk, kNegative, kOverflow };

ShapeCountKind shape_element_count(const std::vector<std::int64_t>& shape, std::int64_t& count) {
    count = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension < 0) {
            return ShapeCountKind::kNegative;
        }
        if (dimension > 0 && count > std::numeric_limits<std::int64_t>::max() / dimension) {
            return ShapeCountKind::kOverflow;
        }
        count *= dimension;
    }
    return ShapeCountKind::kOk;
}

bool finite_tolerance(double value) {
    return std::isfinite(value) && value >= 0.0;
}

bool product_fits(std::uint32_t workers, std::uint32_t measured) {
    if (measured != 0 && workers > std::numeric_limits<std::uint32_t>::max() / measured) {
        return false;
    }
    const std::uint64_t total_samples =
        static_cast<std::uint64_t>(workers) * static_cast<std::uint64_t>(measured);
    constexpr std::uint64_t bytes_per_sample = sizeof(std::chrono::nanoseconds);
    if (bytes_per_sample != 0 &&
        total_samples > std::numeric_limits<std::uint64_t>::max() / bytes_per_sample) {
        return false;
    }
    const std::uint64_t bytes = total_samples * bytes_per_sample;
    return bytes <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
}

ConcurrentBenchmarkRun validate_expected(std::uint32_t worker_index,
                                         const ConcurrentBenchmarkAssignment& assignment) {
    const std::string label = "worker " + std::to_string(worker_index);
    if (assignment.expected.empty()) {
        return FailConfig(label + " expected output list is empty", worker_index);
    }
    for (std::size_t index = 0; index < assignment.expected.size(); ++index) {
        const WorkerExpectedTensor& tensor = assignment.expected[index];
        if (tensor.name.empty()) {
            return FailConfig(label + " expected output name is empty", worker_index);
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (assignment.expected[earlier].name == tensor.name) {
                return FailConfig(label + " duplicate expected output name: " + tensor.name,
                                  worker_index);
            }
        }
        std::int64_t count = 0;
        const ShapeCountKind shape = shape_element_count(tensor.shape, count);
        if (shape == ShapeCountKind::kNegative) {
            return FailConfig(label + " expected output " + tensor.name +
                                  " has a negative dimension",
                              worker_index);
        }
        if (shape == ShapeCountKind::kOverflow) {
            return FailConfig(label + " expected output " + tensor.name +
                                  " shape product overflows",
                              worker_index);
        }
        if (static_cast<std::uint64_t>(count) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            static_cast<std::size_t>(count) != tensor.values.size()) {
            return FailConfig(label + " expected output " + tensor.name +
                                  " element count does not match its shape",
                              worker_index);
        }
        for (const float value : tensor.values) {
            if (!std::isfinite(value)) {
                return FailConfig(label + " expected output " + tensor.name +
                                      " has a non-finite value",
                                  worker_index);
            }
        }
        if (!finite_tolerance(tensor.relative_tolerance) ||
            !finite_tolerance(tensor.absolute_tolerance)) {
            return FailConfig(label + " expected output " + tensor.name +
                                  " has an invalid tolerance",
                              worker_index);
        }
    }
    return ConcurrentBenchmarkRun{};
}

ConcurrentBenchmarkRun validate_config(const ConcurrentBenchmarkConfig& config,
                                       std::span<const ConcurrentBenchmarkAssignment> assignments) {
    if (config.worker_count == 0) {
        return FailConfig("worker count is zero");
    }
    if (config.worker_count > kMaxConcurrentWorkers) {
        return FailConfig("worker count exceeds 64");
    }
    if (assignments.size() != config.worker_count) {
        return FailConfig("assignment count does not match worker count");
    }
    if (config.measured_iterations == 0) {
        return FailConfig("measured iteration count is zero");
    }
    if (!product_fits(config.worker_count, config.measured_iterations)) {
        return FailConfig("worker and measured iteration product overflows");
    }
    if (config.measured_iterations > kMaxBenchmarkIterations) {
        return FailConfig("measured iteration count exceeds 100000");
    }
    if (config.warmup_iterations > kMaxBenchmarkIterations) {
        return FailConfig("warmup iteration count exceeds 100000");
    }
    if (config.correctness_check_interval == 0) {
        return FailConfig("correctness check interval is zero");
    }
    if (config.stop_after_starting_workers.has_value() &&
        *config.stop_after_starting_workers >= config.worker_count) {
        return FailConfig("stop_after_starting_workers is outside the worker range");
    }
    if (config.throw_in_worker.has_value() && *config.throw_in_worker >= config.worker_count) {
        return FailConfig("throw_in_worker is outside the worker range");
    }
    if (config.fail_warmup_worker.has_value() &&
        *config.fail_warmup_worker >= config.worker_count) {
        return FailConfig("fail_warmup_worker is outside the worker range");
    }
    const bool has_measured_worker = config.fail_measured_worker.has_value();
    const bool has_measured_iteration = config.fail_measured_iteration.has_value();
    if (has_measured_worker != has_measured_iteration) {
        return FailConfig("measured failure injection requires a worker and an iteration");
    }
    if (has_measured_worker && *config.fail_measured_worker >= config.worker_count) {
        return FailConfig("fail_measured_worker is outside the worker range");
    }
    if (has_measured_iteration && *config.fail_measured_iteration >= config.measured_iterations) {
        return FailConfig("fail_measured_iteration is outside the measured range");
    }
    for (std::uint32_t index = 0; index < config.worker_count; ++index) {
        const ConcurrentBenchmarkRun expected = validate_expected(index, assignments[index]);
        if (expected.error.kind != ConcurrentBenchmarkErrorKind::kNone) {
            return expected;
        }
    }
    return ConcurrentBenchmarkRun{};
}

const NamedTensor* find_output(const std::vector<NamedTensor>& outputs, const std::string& name) {
    for (const NamedTensor& output : outputs) {
        if (output.name == name) {
            return &output;
        }
    }
    return nullptr;
}

bool within_tolerance(float actual, float expected, double absolute_tolerance,
                      double relative_tolerance) {
    const double difference =
        std::fabs(static_cast<double>(actual) - static_cast<double>(expected));
    const double limit =
        absolute_tolerance + relative_tolerance * std::fabs(static_cast<double>(expected));
    return difference <= limit;
}

// Same conversion contract as the sequential benchmark runner. A negative
// duration or a value that does not fit in nanoseconds is rejected.
bool nanoseconds_from(const std::chrono::steady_clock::duration& duration,
                      std::chrono::nanoseconds& out) {
    if (duration.count() < 0) {
        return false;
    }
    using Duration = std::chrono::steady_clock::duration;
    using Ratio = std::ratio_divide<typename Duration::period, std::nano>;
    const auto count = static_cast<std::uint64_t>(duration.count());
    constexpr auto numerator = static_cast<std::uint64_t>(Ratio::num);
    constexpr auto denominator = static_cast<std::uint64_t>(Ratio::den);
    if (denominator == 0 ||
        (numerator != 0 && count > std::numeric_limits<std::uint64_t>::max() / numerator)) {
        return false;
    }
    const std::uint64_t scaled = count * numerator;
    const std::uint64_t nanos = scaled / denominator;
    if (nanos > static_cast<std::uint64_t>(std::chrono::nanoseconds::max().count())) {
        return false;
    }
    out = std::chrono::nanoseconds{static_cast<std::chrono::nanoseconds::rep>(nanos)};
    return true;
}

bool compares_values(std::uint32_t iteration, std::uint32_t measured, std::uint32_t interval) {
    return iteration == measured || iteration % interval == 0;
}

bool outputs_ok(const InferenceResult& result, const std::vector<WorkerExpectedTensor>& expected,
                bool compare_values, OutputProblem& problem) {
    if (result.outputs.size() != expected.size()) {
        problem.kind = ConcurrentBenchmarkErrorKind::kOutputMismatch;
        problem.message = "output count is " + std::to_string(result.outputs.size()) +
                          ", expected " + std::to_string(expected.size());
        return false;
    }
    for (const WorkerExpectedTensor& tensor : expected) {
        const NamedTensor* output = find_output(result.outputs, tensor.name);
        if (output == nullptr) {
            problem.kind = ConcurrentBenchmarkErrorKind::kOutputMismatch;
            problem.message = "missing output name: " + tensor.name;
            return false;
        }
        if (output->shape != tensor.shape) {
            problem.kind = ConcurrentBenchmarkErrorKind::kOutputMismatch;
            problem.message = "output " + tensor.name + " shape does not match";
            return false;
        }
        if (output->values.size() != tensor.values.size()) {
            problem.kind = ConcurrentBenchmarkErrorKind::kOutputMismatch;
            problem.message = "output " + tensor.name + " element count does not match";
            return false;
        }
        for (std::size_t index = 0; index < output->values.size(); ++index) {
            if (!std::isfinite(output->values[index])) {
                problem.kind = ConcurrentBenchmarkErrorKind::kNonFiniteOutput;
                problem.message =
                    "output " + tensor.name + " value " + std::to_string(index) + " is non-finite";
                return false;
            }
        }
        if (!compare_values) {
            continue;
        }
        for (std::size_t index = 0; index < output->values.size(); ++index) {
            if (!within_tolerance(output->values[index], tensor.values[index],
                                  tensor.absolute_tolerance, tensor.relative_tolerance)) {
                problem.kind = ConcurrentBenchmarkErrorKind::kOutputMismatch;
                problem.message = "output " + tensor.name + " value " + std::to_string(index) +
                                  " is outside tolerance";
                return false;
            }
        }
    }
    return true;
}

void fail_slot(WorkerSlot& slot, StartGate& gate, std::atomic<bool>& stop,
               std::uint32_t worker_index, std::uint32_t iteration, ConcurrentBenchmarkPhase phase,
               ConcurrentBenchmarkErrorKind kind, InferenceErrorKind inference_kind,
               std::string message) {
    if (!slot.failed) {
        slot.failed = true;
        slot.error.worker_index = worker_index;
        slot.error.request_iteration = iteration;
        slot.error.phase = phase;
        slot.error.kind = kind;
        slot.error.inference_kind = inference_kind;
    }
    cancel_gate(gate, stop);
    try {
        if (slot.error.message.empty()) {
            slot.error.message = std::move(message);
        }
    } catch (...) {
    }
}

void run_worker(std::uint32_t worker_index, InferenceEngine& engine,
                const ConcurrentBenchmarkAssignment& assignment,
                const ConcurrentBenchmarkConfig& config, bool inject_exception, bool inject_warmup,
                bool inject_measured, std::uint32_t injected_iteration, StartGate& gate,
                std::atomic<bool>& stop, WorkerSlot& slot) {
    try {
        if (inject_exception) {
            throw std::runtime_error("injected worker exception");
        }

        std::vector<NamedTensorView> views;
        views.reserve(assignment.inputs.size());
        for (const NamedTensor& input : assignment.inputs) {
            views.push_back(
                NamedTensorView{input.name, input.shape, std::span<const float>{input.values}});
        }

        if (inject_warmup) {
            fail_slot(slot, gate, stop, worker_index, 0, ConcurrentBenchmarkPhase::kWarmup,
                      ConcurrentBenchmarkErrorKind::kInferenceFailed, InferenceErrorKind::kNone,
                      "injected warmup failure");
            return;
        }

        for (std::uint32_t iteration = 0; iteration < config.warmup_iterations; ++iteration) {
            if (stop.load(std::memory_order_acquire)) {
                return;
            }
            InferenceResult result = engine.run(views);
            if (result.error.kind != InferenceErrorKind::kNone) {
                const InferenceErrorKind inference_kind = result.error.kind;
                std::string message = std::move(result.error.message);
                fail_slot(slot, gate, stop, worker_index, iteration,
                          ConcurrentBenchmarkPhase::kWarmup,
                          ConcurrentBenchmarkErrorKind::kInferenceFailed, inference_kind,
                          std::move(message));
                return;
            }
            OutputProblem problem;
            if (!outputs_ok(result, assignment.expected, true, problem)) {
                fail_slot(slot, gate, stop, worker_index, iteration,
                          ConcurrentBenchmarkPhase::kWarmup, problem.kind,
                          InferenceErrorKind::kNone, std::move(problem.message));
                return;
            }
        }

        // Capacity is reserved after warmup and before the start gate, so the
        // measured loop does not allocate sample storage and the allocation
        // stays outside the shared wall interval.
        slot.samples.reserve(config.measured_iterations);

        {
            std::unique_lock<std::mutex> lock(gate.mutex);
            if (gate.cancel || stop.load(std::memory_order_acquire)) {
                return;
            }
            ++gate.arrived;
            gate.cv.notify_all();
            gate.cv.wait(lock, [&] {
                return gate.release || gate.cancel || stop.load(std::memory_order_acquire);
            });
            if (!gate.release || gate.cancel || stop.load(std::memory_order_acquire)) {
                return;
            }
        }

        for (std::uint32_t iteration = 1; iteration <= config.measured_iterations; ++iteration) {
            if (stop.load(std::memory_order_acquire)) {
                return;
            }
            const std::uint32_t iteration_index = iteration - 1;
            if (inject_measured && iteration_index == injected_iteration) {
                fail_slot(slot, gate, stop, worker_index, iteration_index,
                          ConcurrentBenchmarkPhase::kMeasured,
                          ConcurrentBenchmarkErrorKind::kInferenceFailed, InferenceErrorKind::kNone,
                          "injected measured-loop failure");
                return;
            }

            {
                const auto call_start = std::chrono::steady_clock::now();
                InferenceResult result = engine.run(views);
                const auto call_stop = std::chrono::steady_clock::now();

                if (result.error.kind != InferenceErrorKind::kNone) {
                    const InferenceErrorKind inference_kind = result.error.kind;
                    std::string message = std::move(result.error.message);
                    fail_slot(slot, gate, stop, worker_index, iteration_index,
                              ConcurrentBenchmarkPhase::kMeasured,
                              ConcurrentBenchmarkErrorKind::kInferenceFailed, inference_kind,
                              std::move(message));
                    return;
                }
                std::chrono::nanoseconds call_duration{0};
                if (!nanoseconds_from(call_stop - call_start, call_duration)) {
                    fail_slot(slot, gate, stop, worker_index, iteration_index,
                              ConcurrentBenchmarkPhase::kMeasured,
                              ConcurrentBenchmarkErrorKind::kInvalidDuration,
                              InferenceErrorKind::kNone, "measured call duration is invalid");
                    return;
                }
                OutputProblem problem;
                if (!outputs_ok(result, assignment.expected,
                                compares_values(iteration, config.measured_iterations,
                                                config.correctness_check_interval),
                                problem)) {
                    fail_slot(slot, gate, stop, worker_index, iteration_index,
                              ConcurrentBenchmarkPhase::kMeasured, problem.kind,
                              InferenceErrorKind::kNone, std::move(problem.message));
                    return;
                }
                slot.samples.push_back(call_duration);
            }
        }

        slot.finished_at = std::chrono::steady_clock::now();
        slot.completed_requests = config.measured_iterations;
        slot.finished = true;
    } catch (const std::exception& error) {
        fail_slot(slot, gate, stop, worker_index, 0, ConcurrentBenchmarkPhase::kNone,
                  ConcurrentBenchmarkErrorKind::kWorkerException, InferenceErrorKind::kNone,
                  error.what());
    } catch (...) {
        fail_slot(slot, gate, stop, worker_index, 0, ConcurrentBenchmarkPhase::kNone,
                  ConcurrentBenchmarkErrorKind::kWorkerException, InferenceErrorKind::kNone,
                  "unknown worker exception");
    }
}

ConcurrentBenchmarkRun finish(std::vector<WorkerSlot>& slots,
                              const ConcurrentBenchmarkConfig& config,
                              std::chrono::steady_clock::time_point measurement_start,
                              bool released) {
    for (const WorkerSlot& slot : slots) {
        if (slot.failed) {
            return Fail(slot.error);
        }
    }
    if (!released) {
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kWorkerException,
                              "worker stopped before completing its requests"));
    }

    auto latest = slots.front().finished_at;
    for (std::uint32_t index = 0; index < slots.size(); ++index) {
        const WorkerSlot& slot = slots[index];
        if (!slot.finished || slot.completed_requests != config.measured_iterations ||
            slot.samples.size() != config.measured_iterations) {
            return Fail(MakeError(ConcurrentBenchmarkErrorKind::kWorkerException,
                                  "worker stopped before completing its requests",
                                  ConcurrentBenchmarkPhase::kNone, index));
        }
        if (slot.finished_at > latest) {
            latest = slot.finished_at;
        }
    }

    std::chrono::nanoseconds wall{0};
    if (!nanoseconds_from(latest - measurement_start, wall)) {
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kInvalidDuration,
                              "measurement wall duration is invalid"));
    }
    if (wall.count() <= 0) {
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kZeroElapsed,
                              "measurement wall duration is zero"));
    }

    const std::uint64_t total_samples = static_cast<std::uint64_t>(config.worker_count) *
                                        static_cast<std::uint64_t>(config.measured_iterations);
    std::uint64_t completed_sum = 0;
    std::uint64_t sample_sum = 0;
    for (const WorkerSlot& slot : slots) {
        completed_sum += slot.completed_requests;
        sample_sum += slot.samples.size();
    }
    if (completed_sum != total_samples || sample_sum != total_samples) {
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kWorkerException,
                              "completed request count does not match the measured sample count"));
    }

    std::vector<std::chrono::nanoseconds> pooled;
    pooled.reserve(static_cast<std::size_t>(total_samples));
    for (const WorkerSlot& slot : slots) {
        pooled.insert(pooled.end(), slot.samples.begin(), slot.samples.end());
    }
    if (pooled.size() != total_samples) {
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kWorkerException,
                              "pooled latency sample count does not match completed requests"));
    }
    const StatisticsResult statistics = summarize_latencies(pooled);
    if (statistics.error.kind != StatisticsErrorKind::kNone) {
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kStatistics, statistics.error.message,
                              ConcurrentBenchmarkPhase::kNone, 0, 0, InferenceErrorKind::kNone,
                              statistics.error.kind));
    }
    if (statistics.summary.sample_count != total_samples) {
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kWorkerException,
                              "latency summary sample count does not match completed requests"));
    }

    const double wall_seconds = static_cast<double>(wall.count()) / 1000000000.0;
    const double aggregate = static_cast<double>(total_samples) / wall_seconds;
    if (!std::isfinite(aggregate) || aggregate <= 0.0) {
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kZeroElapsed,
                              "aggregate completed throughput is not finite"));
    }

    ConcurrentBenchmarkRun run;
    run.report.workload = config.workload;
    run.report.model = config.model;
    run.report.backend = kBackend;
    run.report.worker_count = config.worker_count;
    run.report.warmup_iterations = config.warmup_iterations;
    run.report.measured_iterations = config.measured_iterations;
    run.report.correctness_check_interval = config.correctness_check_interval;
    run.report.total_completed_requests = total_samples;
    run.report.per_worker_completed_requests.reserve(slots.size());
    run.report.per_worker_call_durations.reserve(slots.size());
    for (WorkerSlot& slot : slots) {
        run.report.per_worker_completed_requests.push_back(slot.completed_requests);
        run.report.per_worker_call_durations.push_back(std::move(slot.samples));
    }
    run.report.call_summary = statistics.summary;
    run.report.measurement_wall_duration = wall;
    run.report.aggregate_completed_requests_per_second = aggregate;
    return run;
}

ConcurrentBenchmarkRun execute(std::vector<std::unique_ptr<InferenceEngine>> engines,
                               std::span<const ConcurrentBenchmarkAssignment> assignments,
                               const ConcurrentBenchmarkConfig& config) {
    const std::uint32_t worker_count = config.worker_count;
    std::vector<WorkerSlot> slots(worker_count);
    std::atomic<bool> stop{false};
    StartGate gate;
    std::vector<std::jthread> threads;
    threads.reserve(worker_count);

    std::chrono::steady_clock::time_point measurement_start{};
    bool released = false;
    try {
        for (std::uint32_t index = 0; index < worker_count; ++index) {
            if (config.stop_after_starting_workers.has_value() &&
                index == *config.stop_after_starting_workers) {
                cancel_gate(gate, stop);
                throw StartAborted{index};
            }
            const bool inject_exception =
                config.throw_in_worker.has_value() && *config.throw_in_worker == index;
            const bool inject_warmup =
                config.fail_warmup_worker.has_value() && *config.fail_warmup_worker == index;
            const bool inject_measured =
                config.fail_measured_worker.has_value() && *config.fail_measured_worker == index;
            const std::uint32_t injected_iteration = config.fail_measured_iteration.value_or(0);
            InferenceEngine* engine = engines[index].get();
            const ConcurrentBenchmarkAssignment* assignment = &assignments[index];
            threads.emplace_back([index, engine, assignment, &config, inject_exception,
                                  inject_warmup, inject_measured, injected_iteration, &gate, &stop,
                                  &slots]() {
                try {
                    run_worker(index, *engine, *assignment, config, inject_exception, inject_warmup,
                               inject_measured, injected_iteration, gate, stop, slots[index]);
                } catch (...) {
                    if (!slots[index].failed) {
                        slots[index].failed = true;
                        slots[index].error.worker_index = index;
                        slots[index].error.kind = ConcurrentBenchmarkErrorKind::kWorkerException;
                    }
                    cancel_gate(gate, stop);
                    try {
                        if (slots[index].error.message.empty()) {
                            slots[index].error.message = "worker exception";
                        }
                    } catch (...) {
                    }
                }
            });
        }

        {
            std::unique_lock<std::mutex> lock(gate.mutex);
            gate.cv.wait(lock, [&] {
                return gate.cancel || stop.load(std::memory_order_acquire) ||
                       gate.arrived == worker_count;
            });
            if (!gate.cancel && !stop.load(std::memory_order_acquire) &&
                gate.arrived == worker_count) {
                measurement_start = std::chrono::steady_clock::now();
                gate.release = true;
                released = true;
                gate.cv.notify_all();
            }
        }
    } catch (const StartAborted& aborted) {
        threads.clear();
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kWorkerStart,
                              "worker thread was not started", ConcurrentBenchmarkPhase::kNone,
                              aborted.worker_index));
    } catch (const std::exception& error) {
        cancel_gate(gate, stop);
        const auto failed_index = static_cast<std::uint32_t>(threads.size());
        threads.clear();
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kWorkerStart, error.what(),
                              ConcurrentBenchmarkPhase::kNone, failed_index));
    } catch (...) {
        cancel_gate(gate, stop);
        const auto failed_index = static_cast<std::uint32_t>(threads.size());
        threads.clear();
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kWorkerStart,
                              "unknown exception while starting workers",
                              ConcurrentBenchmarkPhase::kNone, failed_index));
    }

    threads.clear();
    return finish(slots, config, measurement_start, released);
}

} // namespace

ConcurrentBenchmarkRun
run_concurrent_benchmark(const std::filesystem::path& model_path,
                         std::span<const ConcurrentBenchmarkAssignment> assignments,
                         const ConcurrentBenchmarkConfig& config) {
    try {
        const ConcurrentBenchmarkRun invalid = validate_config(config, assignments);
        if (invalid.error.kind != ConcurrentBenchmarkErrorKind::kNone) {
            return invalid;
        }

        std::vector<std::unique_ptr<InferenceEngine>> engines;
        engines.reserve(config.worker_count);
        for (std::uint32_t index = 0; index < config.worker_count; ++index) {
            EngineLoad loaded = load_inference_engine(model_path);
            if (loaded.error.kind != InferenceErrorKind::kNone || loaded.engine == nullptr) {
                ConcurrentBenchmarkRun result;
                result.error.kind = ConcurrentBenchmarkErrorKind::kModelLoad;
                result.error.worker_index = index;
                if (loaded.error.kind == InferenceErrorKind::kNone) {
                    result.load_error = InferenceError{InferenceErrorKind::kModelInvalid,
                                                       "model load returned no engine"};
                } else {
                    result.load_error = std::move(loaded.error);
                }
                result.error.inference_kind = result.load_error.kind;
                result.error.message = result.load_error.message;
                return result;
            }
            engines.push_back(std::move(loaded.engine));
        }
        return execute(std::move(engines), assignments, config);
    } catch (const std::exception& error) {
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kWorkerException, error.what()));
    } catch (...) {
        return Fail(MakeError(ConcurrentBenchmarkErrorKind::kWorkerException, "unknown exception"));
    }
}

} // namespace inferedge
