#include "inferedge/concurrent_benchmark_json.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>

namespace inferedge {
namespace {

constexpr std::string_view kToolName = "inferedge-bench-concurrent";
constexpr std::string_view kSessionOwnership = "one_inference_engine_per_worker";
constexpr std::string_view kBackendName = "ONNX Runtime CPU";
constexpr std::string_view kGraphOptimizations = "disabled";
constexpr std::uint64_t kIntraOpThreadsPerSession = 1;

bool is_valid_utf8(std::string_view text) {
    for (std::size_t index = 0; index < text.size();) {
        const auto lead = static_cast<unsigned char>(text[index]);
        if (lead <= 0x7FU) {
            ++index;
            continue;
        }
        std::size_t length = 0;
        unsigned int code_point = 0;
        unsigned int minimum = 0;
        if (lead >= 0xC2U && lead <= 0xDFU) {
            length = 2;
            code_point = lead & 0x1FU;
            minimum = 0x80U;
        } else if (lead >= 0xE0U && lead <= 0xEFU) {
            length = 3;
            code_point = lead & 0x0FU;
            minimum = 0x800U;
        } else if (lead >= 0xF0U && lead <= 0xF4U) {
            length = 4;
            code_point = lead & 0x07U;
            minimum = 0x10000U;
        } else {
            return false;
        }
        if (index + length > text.size()) {
            return false;
        }
        for (std::size_t byte_index = 1; byte_index < length; ++byte_index) {
            const auto continuation = static_cast<unsigned char>(text[index + byte_index]);
            if ((continuation & 0xC0U) != 0x80U) {
                return false;
            }
            code_point = (code_point << 6U) | (continuation & 0x3FU);
        }
        if (code_point < minimum || code_point > 0x10FFFFU ||
            (code_point >= 0xD800U && code_point <= 0xDFFFU)) {
            return false;
        }
        index += length;
    }
    return true;
}

void append_json_string(std::string& out, std::string_view text) {
    constexpr char kHex[] = "0123456789abcdef";
    out.push_back('"');
    for (const char character : text) {
        const auto byte = static_cast<unsigned char>(character);
        if (byte == '"') {
            out += "\\\"";
        } else if (byte == '\\') {
            out += "\\\\";
        } else if (byte <= 0x1FU) {
            out += "\\u00";
            out.push_back(kHex[byte >> 4U]);
            out.push_back(kHex[byte & 0x0FU]);
        } else {
            out.push_back(character);
        }
    }
    out.push_back('"');
}

bool append_number(std::string& out, double value) {
    char buffer[128];
    const std::to_chars_result result =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general);
    if (result.ec != std::errc{}) {
        return false;
    }
    out.append(buffer, static_cast<std::size_t>(result.ptr - buffer));
    return true;
}

bool append_integer(std::string& out, std::uint64_t value) {
    char buffer[32];
    const std::to_chars_result result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (result.ec != std::errc{}) {
        return false;
    }
    out.append(buffer, static_cast<std::size_t>(result.ptr - buffer));
    return true;
}

double microseconds_from(std::chrono::nanoseconds duration) {
    return static_cast<double>(duration.count()) / 1000.0;
}

// Accepts an inversion no larger than one part in a billion, which covers a
// rounding difference between equivalent representations. A reversal of
// distinct sample magnitudes still fails.
bool ordered_metric(double earlier, double later) {
    if (earlier <= later) {
        return true;
    }
    const double scale = std::max(std::abs(earlier), std::abs(later));
    const double tolerance = std::max(scale, 1.0) * 1e-9;
    return (earlier - later) <= tolerance;
}

bool nonnegative_metric(double value) {
    return std::isfinite(value) && !std::signbit(value);
}

std::string validate_report(const ConcurrentBenchmarkReport& report, std::string_view fixture_id,
                            std::string_view tool_version, double minimum_us, double maximum_us,
                            double wall_us) {
    if (report.worker_count < 1) {
        return "concurrent benchmark report worker count is not positive";
    }
    if (report.measured_iterations < 1) {
        return "concurrent benchmark report measured iteration count is zero";
    }
    if (report.correctness_check_interval < 1) {
        return "concurrent benchmark report correctness-check interval is zero";
    }
    if (report.model.empty() || report.workload.empty() || tool_version.empty()) {
        return "concurrent benchmark report string is empty";
    }
    if (report.backend != kBackendName) {
        return "concurrent benchmark report backend is not ONNX Runtime CPU";
    }

    const std::uint64_t workers = report.worker_count;
    const std::uint64_t measured = report.measured_iterations;
    if (workers > std::numeric_limits<std::uint64_t>::max() / measured) {
        return "concurrent benchmark report completed request count overflows";
    }
    const std::uint64_t expected_total = workers * measured;
    if (report.total_completed_requests != expected_total) {
        return "concurrent benchmark report completed request count does not match the measured "
               "iteration count";
    }
    if (report.per_worker_completed_requests.size() != workers) {
        return "concurrent benchmark report per-worker completed request count does not match the "
               "worker count";
    }
    std::uint64_t completed_sum = 0;
    for (const std::uint32_t count : report.per_worker_completed_requests) {
        if (count != report.measured_iterations) {
            return "concurrent benchmark report per-worker completed request count does not match "
                   "measured iterations";
        }
        if (completed_sum > std::numeric_limits<std::uint64_t>::max() - count) {
            return "concurrent benchmark report completed request count overflows";
        }
        completed_sum += count;
    }
    if (completed_sum != report.total_completed_requests) {
        return "concurrent benchmark report completed request count does not match the per-worker "
               "counts";
    }
    if (report.call_summary.sample_count != report.total_completed_requests) {
        return "concurrent benchmark report sample count does not match completed requests";
    }

    const LatencySummary& summary = report.call_summary;
    const struct NamedDouble {
        const char* name;
        double value;
    } numbers[] = {
        {"minimum latency", minimum_us},
        {"mean latency", summary.mean_microseconds},
        {"p50 latency", summary.p50_microseconds},
        {"p95 latency", summary.p95_microseconds},
        {"p99 latency", summary.p99_microseconds},
        {"maximum latency", maximum_us},
        {"population standard deviation", summary.population_standard_deviation_microseconds},
        {"measurement wall duration", wall_us},
        {"aggregate throughput", report.aggregate_completed_requests_per_second},
        {"active-call throughput", summary.active_call_throughput_requests_per_second},
    };
    for (const NamedDouble& number : numbers) {
        if (!std::isfinite(number.value)) {
            return std::string{"concurrent benchmark report "} + number.name + " is not finite";
        }
    }
    if (report.measurement_wall_duration.count() <= 0 || wall_us <= 0.0 || std::signbit(wall_us)) {
        return "concurrent benchmark report measurement wall duration is not positive";
    }
    if (summary.minimum.count() < 0 || std::signbit(minimum_us)) {
        return "concurrent benchmark report minimum latency is negative";
    }
    if (summary.maximum.count() < summary.minimum.count()) {
        return "concurrent benchmark report maximum latency is below the minimum";
    }
    if (!nonnegative_metric(summary.mean_microseconds) ||
        !nonnegative_metric(summary.p50_microseconds) ||
        !nonnegative_metric(summary.p95_microseconds) ||
        !nonnegative_metric(summary.p99_microseconds) || !nonnegative_metric(maximum_us)) {
        return "concurrent benchmark report latency metric is negative";
    }
    if (!ordered_metric(minimum_us, summary.p50_microseconds) ||
        !ordered_metric(summary.p50_microseconds, summary.p95_microseconds) ||
        !ordered_metric(summary.p95_microseconds, summary.p99_microseconds) ||
        !ordered_metric(summary.p99_microseconds, maximum_us)) {
        return "concurrent benchmark report latency metrics are out of order";
    }
    // Any negative sign is rejected, including a tiny residue and negative zero.
    // The stored value is not rewritten as zero.
    if (std::signbit(summary.population_standard_deviation_microseconds)) {
        return "concurrent benchmark report standard deviation is negative";
    }
    if (!nonnegative_metric(report.aggregate_completed_requests_per_second) ||
        report.aggregate_completed_requests_per_second <= 0.0) {
        return "concurrent benchmark report aggregate throughput is not positive";
    }
    if (!nonnegative_metric(summary.active_call_throughput_requests_per_second) ||
        summary.active_call_throughput_requests_per_second <= 0.0) {
        return "concurrent benchmark report active-call throughput is not positive";
    }

    const double wall_seconds =
        static_cast<double>(report.measurement_wall_duration.count()) / 1000000000.0;
    if (!(wall_seconds > 0.0) || !std::isfinite(wall_seconds)) {
        return "concurrent benchmark report measurement wall duration is not positive";
    }
    const double expected_aggregate =
        static_cast<double>(report.total_completed_requests) / wall_seconds;
    if (!std::isfinite(expected_aggregate) || expected_aggregate <= 0.0) {
        return "concurrent benchmark report aggregate throughput is not positive";
    }
    const double aggregate_scale = std::max(
        std::abs(expected_aggregate), std::abs(report.aggregate_completed_requests_per_second));
    const double aggregate_tolerance = std::max(aggregate_scale, 1.0) * 1e-9;
    if (std::abs(report.aggregate_completed_requests_per_second - expected_aggregate) >
        aggregate_tolerance) {
        return "concurrent benchmark report aggregate throughput does not match the wall duration";
    }

    const std::string_view strings[] = {report.model,       report.workload, report.backend,
                                        tool_version,       fixture_id,      kSessionOwnership,
                                        kGraphOptimizations};
    for (const std::string_view text : strings) {
        if (!is_valid_utf8(text)) {
            return "concurrent benchmark report string is not valid UTF-8";
        }
    }
    return {};
}

ConcurrentBenchmarkJsonResult failed(std::string error) {
    return ConcurrentBenchmarkJsonResult{std::move(error), {}};
}

} // namespace

ConcurrentBenchmarkJsonResult
write_concurrent_benchmark_json(const ConcurrentBenchmarkReport& report,
                                std::string_view fixture_id, std::string_view tool_version) {
    const LatencySummary& summary = report.call_summary;
    const double minimum_us = microseconds_from(summary.minimum);
    const double maximum_us = microseconds_from(summary.maximum);
    const double wall_us = microseconds_from(report.measurement_wall_duration);
    const std::string problem =
        validate_report(report, fixture_id, tool_version, minimum_us, maximum_us, wall_us);
    if (!problem.empty()) {
        return failed(problem);
    }

    std::string text;
    text.reserve(1024);
    text += "{\n";
    text += "  \"schema_version\": 2,\n";
    text += "  \"tool\": {\n";
    text += "    \"name\": ";
    append_json_string(text, kToolName);
    text += ",\n";
    text += "    \"version\": ";
    append_json_string(text, tool_version);
    text += "\n";
    text += "  },\n";
    text += "  \"model\": ";
    append_json_string(text, report.model);
    text += ",\n";
    text += "  \"workload\": ";
    append_json_string(text, report.workload);
    text += ",\n";
    text += "  \"fixture\": ";
    if (fixture_id.empty()) {
        text += "null";
    } else {
        append_json_string(text, fixture_id);
    }
    text += ",\n";
    text += "  \"backend\": ";
    append_json_string(text, report.backend);
    text += ",\n";
    text += "  \"configuration\": {\n";
    text += "    \"worker_count\": ";
    if (!append_integer(text, report.worker_count)) {
        return failed("concurrent benchmark report integer could not be formatted");
    }
    text += ",\n    \"session_ownership\": ";
    append_json_string(text, kSessionOwnership);
    text += ",\n    \"onnx_intra_op_threads_per_session\": ";
    if (!append_integer(text, kIntraOpThreadsPerSession)) {
        return failed("concurrent benchmark report integer could not be formatted");
    }
    text += ",\n    \"graph_optimizations\": ";
    append_json_string(text, kGraphOptimizations);
    text += ",\n    \"warmup_iterations_per_worker\": ";
    if (!append_integer(text, report.warmup_iterations)) {
        return failed("concurrent benchmark report integer could not be formatted");
    }
    text += ",\n    \"measured_iterations_per_worker\": ";
    if (!append_integer(text, report.measured_iterations)) {
        return failed("concurrent benchmark report integer could not be formatted");
    }
    text += ",\n    \"correctness_check_interval\": ";
    if (!append_integer(text, report.correctness_check_interval)) {
        return failed("concurrent benchmark report integer could not be formatted");
    }
    text += "\n";
    text += "  },\n";
    text += "  \"latency_microseconds\": {\n";
    text += "    \"sample_count\": ";
    if (!append_integer(text, summary.sample_count)) {
        return failed("concurrent benchmark report integer could not be formatted");
    }
    const struct NamedDouble {
        const char* key;
        double value;
    } latency[] = {
        {"minimum", minimum_us},
        {"mean", summary.mean_microseconds},
        {"p50", summary.p50_microseconds},
        {"p95", summary.p95_microseconds},
        {"p99", summary.p99_microseconds},
        {"maximum", maximum_us},
        {"population_standard_deviation", summary.population_standard_deviation_microseconds},
    };
    for (const NamedDouble& field : latency) {
        text += ",\n    \"";
        text += field.key;
        text += "\": ";
        if (!append_number(text, field.value)) {
            return failed("concurrent benchmark report number could not be formatted");
        }
    }
    text += "\n";
    text += "  },\n";
    text += "  \"measurement_wall_microseconds\": ";
    if (!append_number(text, wall_us)) {
        return failed("concurrent benchmark report number could not be formatted");
    }
    text += ",\n";
    text += "  \"total_completed_requests\": ";
    if (!append_integer(text, report.total_completed_requests)) {
        return failed("concurrent benchmark report integer could not be formatted");
    }
    text += ",\n";
    text += "  \"per_worker_completed_requests\": [";
    for (std::size_t index = 0; index < report.per_worker_completed_requests.size(); ++index) {
        if (index != 0) {
            text += ", ";
        }
        if (!append_integer(text, report.per_worker_completed_requests[index])) {
            return failed("concurrent benchmark report integer could not be formatted");
        }
    }
    text += "],\n";
    text += "  \"aggregate_completed_requests_per_second\": ";
    if (!append_number(text, report.aggregate_completed_requests_per_second)) {
        return failed("concurrent benchmark report number could not be formatted");
    }
    text += ",\n";
    text += "  \"active_call_requests_per_second\": ";
    if (!append_number(text, summary.active_call_throughput_requests_per_second)) {
        return failed("concurrent benchmark report number could not be formatted");
    }
    text += ",\n";
    text += "  \"correctness\": \"passed\"\n";
    text += "}\n";
    return ConcurrentBenchmarkJsonResult{{}, std::move(text)};
}

} // namespace inferedge
