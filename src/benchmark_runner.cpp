#include "inferedge/benchmark_runner.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

namespace inferedge {
namespace {

constexpr const char* kBackend = "ONNX Runtime CPU";

BenchmarkRun Fail(BenchmarkErrorKind kind, std::string message,
                  InferenceErrorKind inference_kind = InferenceErrorKind::kNone,
                  StatisticsErrorKind statistics_kind = StatisticsErrorKind::kNone) {
    BenchmarkRun run;
    run.error.kind = kind;
    run.error.inference_kind = inference_kind;
    run.error.statistics_kind = statistics_kind;
    run.error.message = std::move(message);
    return run;
}

bool finite_tolerance(double value) {
    return std::isfinite(value) && value >= 0.0;
}

enum class ShapeCountKind {
    kOk,
    kNegative,
    kOverflow,
};

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

BenchmarkRun validate_config(const BenchmarkConfig& config,
                             std::span<const ExpectedTensor> expected) {
    if (config.measured_iterations == 0) {
        return Fail(BenchmarkErrorKind::kInvalidConfig, "measured iteration count is zero");
    }
    if (config.measured_iterations > kMaxBenchmarkIterations) {
        return Fail(BenchmarkErrorKind::kInvalidConfig, "measured iteration count exceeds 100000");
    }
    if (config.warmup_iterations > kMaxBenchmarkIterations) {
        return Fail(BenchmarkErrorKind::kInvalidConfig, "warmup iteration count exceeds 100000");
    }
    if (config.correctness_check_interval == 0) {
        return Fail(BenchmarkErrorKind::kInvalidConfig, "correctness check interval is zero");
    }
    if (expected.empty()) {
        return Fail(BenchmarkErrorKind::kMissingExpectedOutput, "expected output list is empty");
    }

    for (std::size_t index = 0; index < expected.size(); ++index) {
        const ExpectedTensor& tensor = expected[index];
        if (tensor.name.empty()) {
            return Fail(BenchmarkErrorKind::kInvalidExpectedOutput,
                        "expected output name is empty");
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (expected[earlier].name == tensor.name) {
                return Fail(BenchmarkErrorKind::kInvalidExpectedOutput,
                            "duplicate expected output name: " + tensor.name);
            }
        }
        std::int64_t count = 0;
        const ShapeCountKind shape = shape_element_count(tensor.shape, count);
        if (shape == ShapeCountKind::kNegative) {
            return Fail(BenchmarkErrorKind::kInvalidExpectedOutput,
                        "expected output " + tensor.name + " has a negative dimension");
        }
        if (shape == ShapeCountKind::kOverflow) {
            return Fail(BenchmarkErrorKind::kInvalidExpectedOutput,
                        "expected output " + tensor.name + " shape product overflows");
        }
        if (static_cast<std::uint64_t>(count) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            static_cast<std::size_t>(count) != tensor.values.size()) {
            return Fail(BenchmarkErrorKind::kInvalidExpectedOutput,
                        "expected output " + tensor.name +
                            " element count does not match its shape");
        }
        for (const float value : tensor.values) {
            if (!std::isfinite(value)) {
                return Fail(BenchmarkErrorKind::kInvalidExpectedOutput,
                            "expected output " + tensor.name + " has a non-finite value");
            }
        }
        if (!finite_tolerance(tensor.relative_tolerance) ||
            !finite_tolerance(tensor.absolute_tolerance)) {
            return Fail(BenchmarkErrorKind::kInvalidExpectedOutput,
                        "expected output " + tensor.name + " has an invalid tolerance");
        }
    }
    return BenchmarkRun{};
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

BenchmarkRun check_outputs(const InferenceResult& result, std::span<const ExpectedTensor> expected,
                           bool compare_values) {
    if (result.error.kind != InferenceErrorKind::kNone) {
        return Fail(BenchmarkErrorKind::kInferenceFailed, result.error.message, result.error.kind);
    }
    if (result.outputs.size() != expected.size()) {
        return Fail(BenchmarkErrorKind::kOutputMismatch,
                    "output count is " + std::to_string(result.outputs.size()) + ", expected " +
                        std::to_string(expected.size()));
    }
    for (const ExpectedTensor& tensor : expected) {
        const NamedTensor* output = find_output(result.outputs, tensor.name);
        if (output == nullptr) {
            return Fail(BenchmarkErrorKind::kOutputMismatch, "missing output name: " + tensor.name);
        }
        if (output->shape != tensor.shape) {
            return Fail(BenchmarkErrorKind::kOutputMismatch,
                        "output " + tensor.name + " shape does not match");
        }
        for (std::size_t index = 0; index < output->values.size(); ++index) {
            if (!std::isfinite(output->values[index])) {
                return Fail(BenchmarkErrorKind::kNonFiniteOutput,
                            "output " + tensor.name + " value " + std::to_string(index) +
                                " is non-finite");
            }
        }
        if (!compare_values) {
            continue;
        }
        for (std::size_t index = 0; index < output->values.size(); ++index) {
            if (!within_tolerance(output->values[index], tensor.values[index],
                                  tensor.absolute_tolerance, tensor.relative_tolerance)) {
                return Fail(BenchmarkErrorKind::kOutputMismatch,
                            "output " + tensor.name + " value " + std::to_string(index) +
                                " is outside tolerance");
            }
        }
    }
    return BenchmarkRun{};
}

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

} // namespace

BenchmarkRun run_benchmark(InferenceEngine& engine, std::span<const NamedTensorView> inputs,
                           std::span<const ExpectedTensor> expected,
                           const BenchmarkConfig& config) {
    const BenchmarkRun config_error = validate_config(config, expected);
    if (config_error.error.kind != BenchmarkErrorKind::kNone) {
        return config_error;
    }

    for (std::uint32_t iteration = 0; iteration < config.warmup_iterations; ++iteration) {
        InferenceResult result = engine.run(inputs);
        const BenchmarkRun checked = check_outputs(result, expected, true);
        if (checked.error.kind != BenchmarkErrorKind::kNone) {
            return checked;
        }
    }

    std::vector<std::chrono::nanoseconds> call_durations;
    call_durations.reserve(config.measured_iterations);
    const auto measurement_start = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 1; iteration <= config.measured_iterations; ++iteration) {
        const auto call_start = std::chrono::steady_clock::now();
        InferenceResult result = engine.run(inputs);
        const auto call_stop = std::chrono::steady_clock::now();

        if (result.error.kind != InferenceErrorKind::kNone) {
            return Fail(BenchmarkErrorKind::kInferenceFailed, result.error.message,
                        result.error.kind);
        }
        std::chrono::nanoseconds call_duration{0};
        if (!nanoseconds_from(call_stop - call_start, call_duration)) {
            return Fail(BenchmarkErrorKind::kInvalidDuration, "measured call duration is invalid");
        }
        call_durations.push_back(call_duration);

        const BenchmarkRun checked =
            check_outputs(result, expected,
                          compares_values(iteration, config.measured_iterations,
                                          config.correctness_check_interval));
        if (checked.error.kind != BenchmarkErrorKind::kNone) {
            return checked;
        }
    }
    const auto measurement_stop = std::chrono::steady_clock::now();

    std::chrono::nanoseconds measurement_loop_duration{0};
    if (!nanoseconds_from(measurement_stop - measurement_start, measurement_loop_duration)) {
        return Fail(BenchmarkErrorKind::kInvalidDuration, "measurement loop duration is invalid");
    }
    if (measurement_loop_duration.count() <= 0) {
        return Fail(BenchmarkErrorKind::kZeroElapsed, "measurement loop duration is zero");
    }

    const StatisticsResult statistics = summarize_latencies(call_durations);
    if (statistics.error.kind != StatisticsErrorKind::kNone) {
        return Fail(BenchmarkErrorKind::kStatistics, statistics.error.message,
                    InferenceErrorKind::kNone, statistics.error.kind);
    }

    const double loop_seconds =
        static_cast<double>(measurement_loop_duration.count()) / 1000000000.0;
    const double observed = static_cast<double>(config.measured_iterations) / loop_seconds;
    if (!std::isfinite(observed)) {
        return Fail(BenchmarkErrorKind::kZeroElapsed,
                    "observed sequential throughput is not finite");
    }

    BenchmarkRun run;
    run.report.workload = config.workload;
    run.report.model = config.model;
    run.report.backend = kBackend;
    run.report.warmup_iterations = config.warmup_iterations;
    run.report.measured_iterations = config.measured_iterations;
    run.report.correctness_check_interval = config.correctness_check_interval;
    run.report.call_durations = std::move(call_durations);
    run.report.call_summary = statistics.summary;
    run.report.measurement_loop_duration = measurement_loop_duration;
    run.report.observed_sequential_throughput_requests_per_second = observed;
    return run;
}

} // namespace inferedge
