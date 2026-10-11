#include "inferedge/benchmark_json.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>

namespace inferedge {
namespace {

// Fixed session policy from InferenceEngine: SetIntraOpNumThreads(1),
// SetGraphOptimizationLevel(ORT_DISABLE_ALL), and the CPU provider.
// run_benchmark calls that session on the calling thread. These are the
// current configuration values, not a probe of the host.
constexpr int kCallingThreads = 1;
constexpr int kOnnxIntraOpThreads = 1;
constexpr const char* kGraphOptimizations = "disabled";
constexpr const char* kToolName = "inferedge-bench";

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

std::string validate_report(const BenchmarkReport& report, std::string_view fixture_id,
                            std::string_view tool_version, double minimum_us, double maximum_us,
                            double loop_us) {
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
        {"measurement loop duration", loop_us},
        {"active-call throughput", summary.active_call_throughput_requests_per_second},
        {"observed sequential throughput",
         report.observed_sequential_throughput_requests_per_second},
    };
    for (const NamedDouble& number : numbers) {
        if (!std::isfinite(number.value)) {
            return std::string{"benchmark report "} + number.name + " is not finite";
        }
    }
    if (summary.sample_count == 0) {
        return "benchmark report sample count is not positive";
    }
    if (summary.sample_count != report.measured_iterations) {
        return "benchmark report sample count does not match measured iterations";
    }
    if (report.measurement_loop_duration.count() <= 0) {
        return "benchmark report measurement loop duration is not positive";
    }
    if (summary.minimum.count() < 0) {
        return "benchmark report minimum latency is negative";
    }
    if (summary.maximum.count() < summary.minimum.count()) {
        return "benchmark report maximum latency is below the minimum";
    }
    if (!ordered_metric(minimum_us, summary.p50_microseconds) ||
        !ordered_metric(summary.p50_microseconds, summary.p95_microseconds) ||
        !ordered_metric(summary.p95_microseconds, summary.p99_microseconds) ||
        !ordered_metric(summary.p99_microseconds, maximum_us)) {
        return "benchmark report latency metrics are out of order";
    }
    // Any negative sign is rejected, including a tiny residue and negative zero.
    // The stored value is not rewritten as zero.
    if (std::signbit(summary.population_standard_deviation_microseconds)) {
        return "benchmark report standard deviation is negative";
    }
    if (loop_us <= 0.0) {
        return "benchmark report measurement loop duration is not positive";
    }
    if (summary.active_call_throughput_requests_per_second <= 0.0) {
        return "benchmark report active-call throughput is not positive";
    }
    if (report.observed_sequential_throughput_requests_per_second <= 0.0) {
        return "benchmark report observed sequential throughput is not positive";
    }

    const std::string_view strings[] = {report.model, report.workload, report.backend, tool_version,
                                        fixture_id};
    for (const std::string_view text : strings) {
        if (!is_valid_utf8(text)) {
            return "benchmark report string is not valid UTF-8";
        }
    }
    return {};
}

} // namespace

BenchmarkJsonResult write_benchmark_json(const BenchmarkReport& report, std::string_view fixture_id,
                                         std::string_view tool_version) {
    const LatencySummary& summary = report.call_summary;
    const double minimum_us = microseconds_from(summary.minimum);
    const double maximum_us = microseconds_from(summary.maximum);
    const double loop_us = microseconds_from(report.measurement_loop_duration);
    const std::string problem =
        validate_report(report, fixture_id, tool_version, minimum_us, maximum_us, loop_us);
    if (!problem.empty()) {
        return BenchmarkJsonResult{problem, {}};
    }

    std::string text;
    text.reserve(768);
    text += "{\n";
    text += "  \"schema_version\": 1,\n";
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
    text += "    \"warmup_iterations\": ";
    if (!append_integer(text, report.warmup_iterations)) {
        return BenchmarkJsonResult{"benchmark report integer could not be formatted", {}};
    }
    text += ",\n    \"measured_iterations\": ";
    if (!append_integer(text, report.measured_iterations)) {
        return BenchmarkJsonResult{"benchmark report integer could not be formatted", {}};
    }
    text += ",\n    \"correctness_check_interval\": ";
    if (!append_integer(text, report.correctness_check_interval)) {
        return BenchmarkJsonResult{"benchmark report integer could not be formatted", {}};
    }
    text += ",\n    \"calling_threads\": ";
    if (!append_integer(text, kCallingThreads)) {
        return BenchmarkJsonResult{"benchmark report integer could not be formatted", {}};
    }
    text += ",\n    \"onnx_intra_op_threads\": ";
    if (!append_integer(text, kOnnxIntraOpThreads)) {
        return BenchmarkJsonResult{"benchmark report integer could not be formatted", {}};
    }
    text += ",\n";
    text += "    \"graph_optimizations\": ";
    append_json_string(text, kGraphOptimizations);
    text += "\n";
    text += "  },\n";
    text += "  \"latency_microseconds\": {\n";
    text += "    \"sample_count\": ";
    if (!append_integer(text, summary.sample_count)) {
        return BenchmarkJsonResult{"benchmark report integer could not be formatted", {}};
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
            return BenchmarkJsonResult{"benchmark report number could not be formatted", {}};
        }
    }
    text += "\n";
    text += "  },\n";
    text += "  \"measurement_loop_microseconds\": ";
    if (!append_number(text, loop_us)) {
        return BenchmarkJsonResult{"benchmark report number could not be formatted", {}};
    }
    text += ",\n";
    text += "  \"throughput_requests_per_second\": {\n";
    text += "    \"active_call\": ";
    if (!append_number(text, summary.active_call_throughput_requests_per_second)) {
        return BenchmarkJsonResult{"benchmark report number could not be formatted", {}};
    }
    text += ",\n";
    text += "    \"observed_sequential\": ";
    if (!append_number(text, report.observed_sequential_throughput_requests_per_second)) {
        return BenchmarkJsonResult{"benchmark report number could not be formatted", {}};
    }
    text += "\n";
    text += "  },\n";
    text += "  \"correctness\": \"passed\"\n";
    text += "}\n";
    return BenchmarkJsonResult{{}, std::move(text)};
}

} // namespace inferedge
