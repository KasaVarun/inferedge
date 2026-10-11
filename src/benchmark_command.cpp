#include "inferedge/benchmark_cli.hpp"
#include "inferedge/benchmark_runner.hpp"
#include "inferedge/inference_engine.hpp"

#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef INFEREDGE_MNIST_FIXTURE_DIR
#error "CMake must define INFEREDGE_MNIST_FIXTURE_DIR"
#endif

namespace inferedge {
namespace {

constexpr int kExitUsage = 2;
constexpr int kExitModel = 3;
constexpr int kExitFixture = 4;
constexpr int kExitInference = 5;
constexpr int kExitMetadata = 6;
constexpr int kExitCorrectness = 7;
constexpr int kExitReport = 8;
constexpr std::size_t kMnistInputCount = 784;
constexpr std::size_t kMnistLogitCount = 10;

struct PreparedInputs {
    std::string error;
    int exit_code{kExitFixture};
    std::vector<std::string> names;
    std::vector<std::vector<std::int64_t>> shapes;
    std::vector<std::vector<float>> buffers;
    std::vector<ExpectedTensor> expected;
};

struct FloatLoad {
    std::string error;
    std::vector<float> values;
};

BenchmarkCliResult fail(int exit_code, std::string message) {
    BenchmarkCliResult result;
    result.action = BenchmarkCliAction::kError;
    result.exit_code = exit_code;
    result.message = "error: " + std::move(message) + "\n";
    return result;
}

std::string format_number(double value) {
    char buffer[128];
    const std::to_chars_result result =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general);
    if (result.ec != std::errc{}) {
        return "unavailable";
    }
    return std::string(buffer, result.ptr);
}

std::string format_benchmark_report(const BenchmarkReport& report, std::string_view fixture_id) {
    const LatencySummary& summary = report.call_summary;
    std::string text;
    text += "InferEdge benchmark\n";
    text += "Model: ";
    text += report.model;
    text += "\nWorkload: ";
    text += report.workload;
    text += "\n";
    if (!fixture_id.empty()) {
        text += "Fixture: ";
        text += fixture_id;
        text += "\n";
    }
    text += "Backend: ";
    text += report.backend;
    text += "\nExecution policy: one benchmark calling thread; one ONNX intra-op thread; ";
    text += "graph optimizations disabled\n";
    text += "Warmup iterations: " + std::to_string(report.warmup_iterations) + "\n";
    text += "Measured iterations: " + std::to_string(report.measured_iterations) + "\n";
    text +=
        "Correctness-check interval: " + std::to_string(report.correctness_check_interval) + "\n";
    text += "\nLatency:\n";
    text += "  minimum: " + std::to_string(summary.minimum.count()) + " ns (" +
            format_number(static_cast<double>(summary.minimum.count()) / 1000.0) + " us)\n";
    text += "  mean: " + format_number(summary.mean_microseconds) + " us\n";
    text += "  p50: " + format_number(summary.p50_microseconds) + " us\n";
    text += "  p95: " + format_number(summary.p95_microseconds) + " us\n";
    text += "  p99: " + format_number(summary.p99_microseconds) + " us\n";
    text += "  maximum: " + std::to_string(summary.maximum.count()) + " ns (" +
            format_number(static_cast<double>(summary.maximum.count()) / 1000.0) + " us)\n";
    text += "  population standard deviation: " +
            format_number(summary.population_standard_deviation_microseconds) + " us\n";
    text += "\nThroughput:\n";
    text += "  active-call: " + format_number(summary.active_call_throughput_requests_per_second) +
            " requests/second\n";
    text += "  observed sequential: " +
            format_number(report.observed_sequential_throughput_requests_per_second) +
            " requests/second\n";
    text += "\nCorrectness: passed\n";
    text += "Active-call throughput counts only measured run() durations. ";
    text += "Observed sequential throughput counts the measurement loop, including validation ";
    text += "and result destruction. Neither figure is server capacity.\n";
    return text;
}

int exit_code_for(BenchmarkErrorKind kind) {
    switch (kind) {
    case BenchmarkErrorKind::kNone:
        return 0;
    case BenchmarkErrorKind::kInferenceFailed:
        return kExitInference;
    case BenchmarkErrorKind::kOutputMismatch:
    case BenchmarkErrorKind::kNonFiniteOutput:
        return kExitCorrectness;
    case BenchmarkErrorKind::kInvalidConfig:
    case BenchmarkErrorKind::kMissingExpectedOutput:
    case BenchmarkErrorKind::kInvalidExpectedOutput:
    case BenchmarkErrorKind::kInvalidDuration:
    case BenchmarkErrorKind::kZeroElapsed:
    case BenchmarkErrorKind::kStatistics:
        return kExitReport;
    }
    return kExitReport;
}

const TensorMetadata* find_tensor(const std::vector<TensorMetadata>& tensors,
                                  std::string_view name) {
    for (const TensorMetadata& tensor : tensors) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

bool is_float32_vector(const TensorMetadata& tensor, std::int64_t length) {
    return tensor.element_type == ElementType::kFloat32 && tensor.dimensions.size() == 1 &&
           tensor.dimensions.front() == length;
}

bool metadata_matches(const InferenceEngine& engine, std::string_view workload) {
    const std::vector<TensorMetadata>& inputs = engine.inputs();
    const std::vector<TensorMetadata>& outputs = engine.outputs();
    if (workload == "add") {
        if (inputs.size() != 2 || outputs.size() != 1) {
            return false;
        }
        const TensorMetadata* left = find_tensor(inputs, "A");
        const TensorMetadata* right = find_tensor(inputs, "B");
        const TensorMetadata* sum = find_tensor(outputs, "C");
        return left != nullptr && right != nullptr && sum != nullptr &&
               is_float32_vector(*left, 2) && is_float32_vector(*right, 2) &&
               is_float32_vector(*sum, 2);
    }
    if (inputs.size() != 1 || outputs.size() != 1) {
        return false;
    }
    const TensorMetadata* image = find_tensor(inputs, "image");
    const TensorMetadata* logits = find_tensor(outputs, "logits");
    return image != nullptr && logits != nullptr && is_float32_vector(*image, 784) &&
           is_float32_vector(*logits, 10);
}

FloatLoad read_float32_file(const std::filesystem::path& path, std::size_t count) {
    std::error_code status;
    if (!std::filesystem::is_regular_file(path, status) || status) {
        return FloatLoad{"fixture file is missing: " + path.string(), {}};
    }
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        return FloatLoad{"cannot open " + path.string(), {}};
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    const auto expected = static_cast<std::streamoff>(count * sizeof(std::uint32_t));
    if (size != expected) {
        return FloatLoad{path.string() + " has " + std::to_string(size) + " bytes, expected " +
                             std::to_string(expected),
                         {}};
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        unsigned char bytes[4] = {};
        input.read(reinterpret_cast<char*>(bytes), 4);
        if (!input) {
            return FloatLoad{"short read of " + path.string(), {}};
        }
        const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                                   (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                                   (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                                   (static_cast<std::uint32_t>(bytes[3]) << 24U);
        const float value = std::bit_cast<float>(bits);
        if (!std::isfinite(value)) {
            return FloatLoad{"non-finite value in " + path.string(), {}};
        }
        values[index] = value;
    }
    return FloatLoad{{}, std::move(values)};
}

int argmax(const std::vector<float>& values) {
    std::size_t best = 0;
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return static_cast<int>(best);
}

PreparedInputs prepare_add() {
    PreparedInputs prepared;
    prepared.names = {"A", "B"};
    prepared.shapes = {{2}, {2}};
    prepared.buffers = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    ExpectedTensor sum;
    sum.name = "C";
    sum.shape = {2};
    sum.values = {4.0f, 6.0f};
    sum.relative_tolerance = 0.0;
    sum.absolute_tolerance = 0.0;
    prepared.expected.push_back(std::move(sum));
    return prepared;
}

PreparedInputs prepare_mnist(const std::filesystem::path& directory, std::string_view fixture_id) {
    PreparedInputs prepared;
    const MnistFixture* fixture = find_mnist_fixture(fixture_id);
    if (fixture == nullptr || fixture_id.find('/') != std::string_view::npos ||
        fixture_id.find('\\') != std::string_view::npos ||
        fixture_id.find("..") != std::string_view::npos) {
        prepared.error = "unknown fixture id '" + std::string{fixture_id} + "'";
        prepared.exit_code = kExitUsage;
        return prepared;
    }
    const std::filesystem::path input_path =
        directory / "inputs" / (std::string{fixture->id} + ".f32");
    const std::filesystem::path logits_path =
        directory / "onnx" / (std::string{fixture->id} + ".f32");
    FloatLoad image = read_float32_file(input_path, kMnistInputCount);
    if (!image.error.empty()) {
        prepared.error = std::move(image.error);
        return prepared;
    }
    FloatLoad logits = read_float32_file(logits_path, kMnistLogitCount);
    if (!logits.error.empty()) {
        prepared.error = std::move(logits.error);
        return prepared;
    }
    const int predicted = argmax(logits.values);
    if (predicted != fixture->expected_class) {
        prepared.error = "fixture " + std::string{fixture->id} + " reference argmax is " +
                         std::to_string(predicted) + ", catalog expects " +
                         std::to_string(fixture->expected_class);
        return prepared;
    }
    prepared.names = {"image"};
    prepared.shapes = {{static_cast<std::int64_t>(kMnistInputCount)}};
    prepared.buffers.push_back(std::move(image.values));
    ExpectedTensor expected;
    expected.name = "logits";
    expected.shape = {static_cast<std::int64_t>(kMnistLogitCount)};
    expected.values = std::move(logits.values);
    expected.relative_tolerance = 1e-4;
    expected.absolute_tolerance = 1e-5;
    prepared.expected.push_back(std::move(expected));
    return prepared;
}

std::filesystem::path fixture_directory_for(const BenchmarkCliResult& parsed) {
    if (parsed.fixture_directory_specified) {
        return parsed.fixture_directory;
    }
    return std::filesystem::path{INFEREDGE_MNIST_FIXTURE_DIR};
}

} // namespace

BenchmarkCliResult execute_benchmark(const BenchmarkCliResult& parsed) {
    if (parsed.action != BenchmarkCliAction::kRun) {
        return parsed;
    }
    if (parsed.workload != "add" && parsed.workload != "mnist") {
        return fail(kExitUsage, "unknown workload '" + parsed.workload + "'");
    }
    if (parsed.workload == "add" &&
        (parsed.fixture_specified || parsed.fixture_directory_specified)) {
        return fail(kExitUsage, "fixture flags apply only to the mnist workload");
    }

    // Buffers live in this object. Views are created below, after preparation
    // returns, and are destroyed only after run_benchmark returns.
    PreparedInputs prepared = parsed.workload == "add"
                                  ? prepare_add()
                                  : prepare_mnist(fixture_directory_for(parsed), parsed.fixture_id);
    if (!prepared.error.empty()) {
        return fail(prepared.exit_code, std::move(prepared.error));
    }

    EngineLoad loaded = load_inference_engine(parsed.model_path);
    if (loaded.error.kind != InferenceErrorKind::kNone || loaded.engine == nullptr) {
        const std::string message =
            loaded.error.message.empty() ? "model failed to load" : loaded.error.message;
        return fail(kExitModel, message);
    }
    if (!metadata_matches(*loaded.engine, parsed.workload)) {
        return fail(kExitMetadata,
                    "model metadata does not match the " + parsed.workload + " workload");
    }

    BenchmarkConfig config;
    config.warmup_iterations = parsed.warmup_iterations;
    config.measured_iterations = parsed.measured_iterations;
    config.correctness_check_interval = parsed.correctness_check_interval;
    config.workload = parsed.workload;
    config.model = parsed.model_path;

    std::vector<NamedTensorView> views;
    views.reserve(prepared.buffers.size());
    for (std::size_t index = 0; index < prepared.buffers.size(); ++index) {
        views.push_back(NamedTensorView{prepared.names[index], prepared.shapes[index],
                                        std::span<const float>{prepared.buffers[index]}});
    }
    const BenchmarkRun run = run_benchmark(*loaded.engine, views, prepared.expected, config);
    if (run.error.kind != BenchmarkErrorKind::kNone) {
        const std::string message =
            run.error.message.empty() ? "benchmark failed" : run.error.message;
        return fail(exit_code_for(run.error.kind), message);
    }

    BenchmarkCliResult success = parsed;
    const std::string_view fixture_id =
        parsed.workload == "mnist" ? std::string_view{parsed.fixture_id} : std::string_view{};
    success.action = BenchmarkCliAction::kRun;
    success.exit_code = 0;
    success.message = format_benchmark_report(run.report, fixture_id);
    return success;
}

} // namespace inferedge
