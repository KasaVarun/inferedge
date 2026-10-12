#include "inferedge/benchmark_cli.hpp"
#include "inferedge/concurrent_benchmark.hpp"
#include "inferedge/concurrent_benchmark_cli.hpp"
#include "inferedge/concurrent_benchmark_json.hpp"
#include "inferedge/concurrent_inference.hpp"
#include "inferedge/inference_engine.hpp"
#include "inferedge/version.hpp"

#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
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
    std::vector<WorkerExpectedTensor> expected;
};

struct FloatLoad {
    std::string error;
    std::vector<float> values;
};

ConcurrentBenchmarkCliResult fail(int exit_code, std::string message) {
    ConcurrentBenchmarkCliResult result;
    result.action = ConcurrentBenchmarkCliAction::kError;
    result.exit_code = exit_code;
    result.message = "error: " + std::move(message) + "\n";
    return result;
}

std::string format_number(double value) {
    char buffer[128];
    const std::to_chars_result result =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general);
    if (result.ec != std::errc{}) {
        return {};
    }
    return std::string(buffer, result.ptr);
}

bool append_labeled_number(std::string& text, std::string_view label, double value,
                           std::string_view suffix) {
    const std::string formatted = format_number(value);
    if (formatted.empty()) {
        return false;
    }
    text += label;
    text += formatted;
    text += suffix;
    return true;
}

std::string format_concurrent_benchmark_report(const ConcurrentBenchmarkReport& report,
                                               std::string_view fixture_id) {
    const LatencySummary& summary = report.call_summary;
    std::string text;
    text += "InferEdge concurrent benchmark\n";
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
    text += "\nWorkers: " + std::to_string(report.worker_count) + "\n";
    text += "Execution policy: one InferenceEngine per worker; one ONNX intra-op thread per ";
    text += "session; graph optimizations disabled\n";
    text += "Warmup iterations per worker: " + std::to_string(report.warmup_iterations) + "\n";
    text += "Measured iterations per worker: " + std::to_string(report.measured_iterations) + "\n";
    text +=
        "Correctness-check interval: " + std::to_string(report.correctness_check_interval) + "\n";
    text += "Total completed requests: " + std::to_string(report.total_completed_requests) + "\n";
    text += "Completed requests per worker:";
    for (const std::uint32_t count : report.per_worker_completed_requests) {
        text += " " + std::to_string(count);
    }
    text += "\n";
    text += "\nLatency:\n";
    text += "  minimum: " + std::to_string(summary.minimum.count()) + " ns (";
    if (!append_labeled_number(text, "", static_cast<double>(summary.minimum.count()) / 1000.0,
                               " us)\n")) {
        return {};
    }
    if (!append_labeled_number(text, "  mean: ", summary.mean_microseconds, " us\n") ||
        !append_labeled_number(text, "  p50: ", summary.p50_microseconds, " us\n") ||
        !append_labeled_number(text, "  p95: ", summary.p95_microseconds, " us\n") ||
        !append_labeled_number(text, "  p99: ", summary.p99_microseconds, " us\n")) {
        return {};
    }
    text += "  maximum: " + std::to_string(summary.maximum.count()) + " ns (";
    if (!append_labeled_number(text, "", static_cast<double>(summary.maximum.count()) / 1000.0,
                               " us)\n")) {
        return {};
    }
    if (!append_labeled_number(text, "  population standard deviation: ",
                               summary.population_standard_deviation_microseconds, " us\n")) {
        return {};
    }
    text +=
        "\nMeasurement wall duration: " + std::to_string(report.measurement_wall_duration.count()) +
        " ns (";
    if (!append_labeled_number(
            text, "", static_cast<double>(report.measurement_wall_duration.count()) / 1000.0,
            " us)\n")) {
        return {};
    }
    text += "Aggregate throughput: ";
    if (!append_labeled_number(text, "", report.aggregate_completed_requests_per_second,
                               " requests/second\n")) {
        return {};
    }
    text += "Active-call throughput: ";
    if (!append_labeled_number(text, "", summary.active_call_throughput_requests_per_second,
                               " requests/second\n")) {
        return {};
    }
    text += "\nCorrectness: passed\n";
    text += "Aggregate throughput divides completed requests by the shared wall-clock duration. ";
    text += "Active-call throughput divides completed requests by the sum of per-call durations. ";
    text += "Neither figure is production server capacity.\n";
    return text;
}

int exit_code_for(ConcurrentBenchmarkErrorKind kind) {
    switch (kind) {
    case ConcurrentBenchmarkErrorKind::kNone:
        return 0;
    case ConcurrentBenchmarkErrorKind::kModelLoad:
        return kExitModel;
    case ConcurrentBenchmarkErrorKind::kInferenceFailed:
        return kExitInference;
    case ConcurrentBenchmarkErrorKind::kOutputMismatch:
    case ConcurrentBenchmarkErrorKind::kNonFiniteOutput:
        return kExitCorrectness;
    case ConcurrentBenchmarkErrorKind::kInvalidConfig:
    case ConcurrentBenchmarkErrorKind::kWorkerStart:
    case ConcurrentBenchmarkErrorKind::kWorkerException:
    case ConcurrentBenchmarkErrorKind::kInvalidDuration:
    case ConcurrentBenchmarkErrorKind::kZeroElapsed:
    case ConcurrentBenchmarkErrorKind::kStatistics:
        return kExitReport;
    }
    return kExitReport;
}

std::string describe_run_error(const ConcurrentBenchmarkError& error) {
    const std::string detail = error.message.empty() ? "benchmark failed" : error.message;
    const bool worker_scoped = error.phase != ConcurrentBenchmarkPhase::kNone ||
                               error.kind == ConcurrentBenchmarkErrorKind::kWorkerStart ||
                               error.kind == ConcurrentBenchmarkErrorKind::kWorkerException;
    if (!worker_scoped) {
        return detail;
    }
    std::string prefix = "worker " + std::to_string(error.worker_index);
    if (error.phase == ConcurrentBenchmarkPhase::kWarmup) {
        prefix += " warmup iteration " + std::to_string(error.request_iteration);
    } else if (error.phase == ConcurrentBenchmarkPhase::kMeasured) {
        prefix += " measured iteration " + std::to_string(error.request_iteration);
    }
    prefix += ": ";
    prefix += detail;
    return prefix;
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
    std::filesystem::file_status file_status;
    try {
        file_status = std::filesystem::status(path);
    } catch (const std::filesystem::filesystem_error& error) {
        return FloatLoad{std::string{"fixture filesystem error: "} + error.what(), {}};
    }
    if (!std::filesystem::is_regular_file(file_status)) {
        return FloatLoad{"fixture file is missing: " + path.string(), {}};
    }
    try {
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
    } catch (const std::filesystem::filesystem_error& error) {
        return FloatLoad{std::string{"fixture filesystem error: "} + error.what(), {}};
    } catch (const std::ios_base::failure& error) {
        return FloatLoad{std::string{"fixture I/O error: "} + error.what(), {}};
    }
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
    WorkerExpectedTensor sum;
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
    std::filesystem::file_status directory_status;
    try {
        directory_status = std::filesystem::status(directory);
    } catch (const std::filesystem::filesystem_error& error) {
        prepared.error = std::string{"fixture filesystem error: "} + error.what();
        return prepared;
    }
    if (!std::filesystem::is_directory(directory_status)) {
        prepared.error = "fixture directory is missing: " + directory.string();
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
    WorkerExpectedTensor expected;
    expected.name = "logits";
    expected.shape = {static_cast<std::int64_t>(kMnistLogitCount)};
    expected.values = std::move(logits.values);
    expected.relative_tolerance = 1e-4;
    expected.absolute_tolerance = 1e-5;
    prepared.expected.push_back(std::move(expected));
    return prepared;
}

std::filesystem::path fixture_directory_for(const ConcurrentBenchmarkCliResult& parsed) {
    if (parsed.fixture_directory_specified) {
        return parsed.fixture_directory;
    }
    return std::filesystem::path{INFEREDGE_MNIST_FIXTURE_DIR};
}

std::vector<ConcurrentBenchmarkAssignment> copy_assignments(const PreparedInputs& prepared,
                                                            std::uint32_t worker_count) {
    std::vector<ConcurrentBenchmarkAssignment> assignments;
    assignments.reserve(worker_count);
    for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
        ConcurrentBenchmarkAssignment assignment;
        assignment.inputs.reserve(prepared.buffers.size());
        for (std::size_t index = 0; index < prepared.buffers.size(); ++index) {
            NamedTensor tensor;
            tensor.name = prepared.names[index];
            tensor.shape = prepared.shapes[index];
            tensor.values = prepared.buffers[index];
            assignment.inputs.push_back(std::move(tensor));
        }
        assignment.expected = prepared.expected;
        assignments.push_back(std::move(assignment));
    }
    return assignments;
}

bool inputs_are_independent(const std::vector<ConcurrentBenchmarkAssignment>& assignments) {
    if (assignments.size() < 2 || assignments[0].inputs.empty()) {
        return true;
    }
    const float* const first = assignments[0].inputs[0].values.data();
    for (std::size_t index = 1; index < assignments.size(); ++index) {
        if (assignments[index].inputs.empty()) {
            return false;
        }
        if (assignments[index].inputs[0].values.data() == first) {
            return false;
        }
    }
    return true;
}

} // namespace

ConcurrentBenchmarkCliResult
execute_concurrent_benchmark(const ConcurrentBenchmarkCliResult& parsed) {
    if (parsed.action != ConcurrentBenchmarkCliAction::kRun) {
        return parsed;
    }
    if (parsed.workload != "add" && parsed.workload != "mnist") {
        return fail(kExitUsage, "unknown workload '" + parsed.workload + "'");
    }
    if (parsed.workload == "add" &&
        (parsed.fixture_specified || parsed.fixture_directory_specified)) {
        return fail(kExitUsage, "fixture flags apply only to the mnist workload");
    }
    if (parsed.worker_count < 1 || parsed.worker_count > kMaxConcurrentWorkers) {
        return fail(kExitUsage, "concurrency must be from 1 through 64");
    }

    PreparedInputs prepared;
    try {
        prepared = parsed.workload == "add"
                       ? prepare_add()
                       : prepare_mnist(fixture_directory_for(parsed), parsed.fixture_id);
    } catch (const std::filesystem::filesystem_error& error) {
        return fail(kExitFixture, std::string{"fixture filesystem error: "} + error.what());
    } catch (const std::ios_base::failure& error) {
        return fail(kExitFixture, std::string{"fixture I/O error: "} + error.what());
    }
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
    // The probe session is not part of the measurement and is not shared.
    loaded.engine.reset();

    std::vector<ConcurrentBenchmarkAssignment> assignments =
        copy_assignments(prepared, parsed.worker_count);
    if (!inputs_are_independent(assignments)) {
        return fail(kExitReport, "worker input buffers are not independent");
    }

    ConcurrentBenchmarkConfig config;
    config.worker_count = parsed.worker_count;
    config.warmup_iterations = parsed.warmup_iterations;
    config.measured_iterations = parsed.measured_iterations;
    config.correctness_check_interval = parsed.correctness_check_interval;
    config.workload = parsed.workload;
    config.model = parsed.model_path;
    // Fault-injection seams stay unset for a normal command.

    const ConcurrentBenchmarkRun run =
        run_concurrent_benchmark(parsed.model_path, assignments, config);
    if (run.error.kind == ConcurrentBenchmarkErrorKind::kModelLoad) {
        const std::string message =
            run.load_error.message.empty()
                ? (run.error.message.empty() ? "model failed to load" : run.error.message)
                : run.load_error.message;
        return fail(kExitModel, message);
    }
    if (run.error.kind != ConcurrentBenchmarkErrorKind::kNone) {
        return fail(exit_code_for(run.error.kind), describe_run_error(run.error));
    }

    ConcurrentBenchmarkCliResult success = parsed;
    const std::string_view fixture_id =
        parsed.workload == "mnist" ? std::string_view{parsed.fixture_id} : std::string_view{};
    success.action = ConcurrentBenchmarkCliAction::kRun;
    success.exit_code = 0;
    if (parsed.output_format == ConcurrentBenchmarkOutputFormat::kJson) {
        const ConcurrentBenchmarkJsonResult written =
            write_concurrent_benchmark_json(run.report, fixture_id, version());
        if (!written.error.empty()) {
            return fail(kExitReport, written.error);
        }
        success.message = written.text;
    } else {
        success.message = format_concurrent_benchmark_report(run.report, fixture_id);
        if (success.message.empty()) {
            return fail(kExitReport, "concurrent benchmark report number could not be formatted");
        }
    }
    return success;
}

} // namespace inferedge
