#include "inferedge/concurrent_benchmark_json.hpp"
#include "inferedge/version.hpp"

#include <bit>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>

namespace {

using inferedge::ConcurrentBenchmarkJsonResult;
using inferedge::ConcurrentBenchmarkReport;
using inferedge::write_concurrent_benchmark_json;

std::string bytes(std::initializer_list<unsigned char> values) {
    std::string text;
    text.reserve(values.size());
    for (const unsigned char value : values) {
        text.push_back(static_cast<char>(value));
    }
    return text;
}

double aggregate_for(std::uint64_t requests, std::chrono::nanoseconds wall) {
    const double wall_seconds = static_cast<double>(wall.count()) / 1000000000.0;
    return static_cast<double>(requests) / wall_seconds;
}

ConcurrentBenchmarkReport synthetic_report() {
    ConcurrentBenchmarkReport report;
    report.workload = "add";
    report.model = "models/add_f32.onnx";
    report.backend = "ONNX Runtime CPU";
    report.worker_count = 2;
    report.warmup_iterations = 1;
    report.measured_iterations = 4;
    report.correctness_check_interval = 2;
    report.total_completed_requests = 8;
    report.per_worker_completed_requests = {4, 4};
    report.call_summary.sample_count = 8;
    report.call_summary.minimum = std::chrono::nanoseconds{1500};
    report.call_summary.maximum = std::chrono::nanoseconds{4000};
    report.call_summary.mean_microseconds = 2.5;
    report.call_summary.p50_microseconds = 2.0;
    report.call_summary.p95_microseconds = 3.0;
    report.call_summary.p99_microseconds = 3.5;
    report.call_summary.population_standard_deviation_microseconds = 0.1;
    report.call_summary.active_call_throughput_requests_per_second = 1000.0;
    report.measurement_wall_duration = std::chrono::nanoseconds{20000};
    report.aggregate_completed_requests_per_second =
        aggregate_for(report.total_completed_requests, report.measurement_wall_duration);
    report.per_worker_call_durations.push_back({std::chrono::nanoseconds{1500}});
    return report;
}

std::string raw_field(const std::string& text, std::string_view key) {
    const std::string pattern = "\"" + std::string{key} + "\": ";
    const std::size_t at = text.find(pattern);
    EXPECT_NE(at, std::string::npos) << key << "\n" << text;
    if (at == std::string::npos) {
        return {};
    }
    const std::size_t start = at + pattern.size();
    std::size_t end = start;
    while (end < text.size() && text[end] != ',' && text[end] != '\n') {
        ++end;
    }
    return text.substr(start, end - start);
}

void expect_integer(const std::string& text, std::string_view key, std::uint64_t expected) {
    const std::string token = raw_field(text, key);
    ASSERT_FALSE(token.empty()) << key;
    for (const char character : token) {
        EXPECT_TRUE(std::isdigit(static_cast<unsigned char>(character)) != 0) << token;
    }
    std::uint64_t value = 0;
    const char* const begin = token.data();
    const std::from_chars_result result = std::from_chars(begin, begin + token.size(), value);
    EXPECT_EQ(result.ec, std::errc{});
    EXPECT_EQ(result.ptr, begin + token.size());
    EXPECT_EQ(value, expected) << token;
}

void expect_round_trip(const std::string& text, std::string_view key, double expected) {
    const std::string token = raw_field(text, key);
    ASSERT_FALSE(token.empty()) << key;
    EXPECT_EQ(token.find(','), std::string::npos) << token;
    double value = 0.0;
    const char* const begin = token.data();
    const std::from_chars_result result = std::from_chars(begin, begin + token.size(), value);
    EXPECT_EQ(result.ec, std::errc{}) << token;
    EXPECT_EQ(result.ptr, begin + token.size()) << token;
    EXPECT_EQ(std::bit_cast<std::uint64_t>(value), std::bit_cast<std::uint64_t>(expected)) << token;
}

void expect_rejected(const ConcurrentBenchmarkJsonResult& written, std::string_view fragment) {
    EXPECT_TRUE(written.text.empty()) << written.text;
    EXPECT_EQ(written.text.find('{'), std::string::npos);
    EXPECT_EQ(written.error.find('{'), std::string::npos) << written.error;
    EXPECT_NE(written.error.find(fragment), std::string::npos) << written.error;
    EXPECT_EQ(written.error.find("error:"), std::string::npos) << written.error;
}

const char* kPythonCheck = R"PY(
import json
import math
import sys

def reject_constant(token):
    raise ValueError("constant " + token)

path, workers_text, measured_text, fixture_mode = sys.argv[1:5]
workers = int(workers_text)
measured = int(measured_text)
with open(path, "r", encoding="utf-8") as handle:
    raw = handle.read()
if "per_worker_call_durations" in raw:
    sys.exit("raw samples")
if "\"schema_version\": 1" in raw or "measurement_loop_microseconds" in raw:
    sys.exit("schema 1 field")
if "throughput_requests_per_second" in raw or "observed_sequential" in raw:
    sys.exit("schema 1 throughput")
if "calling_threads" in raw or "onnx_intra_op_threads\"" in raw:
    sys.exit("schema 1 threads")
document = json.loads(raw, parse_constant=reject_constant)
keys = [
    "schema_version", "tool", "model", "workload", "fixture", "backend",
    "configuration", "latency_microseconds", "measurement_wall_microseconds",
    "total_completed_requests", "per_worker_completed_requests",
    "aggregate_completed_requests_per_second", "active_call_requests_per_second",
    "correctness",
]
if list(document) != keys:
    sys.exit("keys")
if type(document["schema_version"]) is not int or document["schema_version"] != 2:
    sys.exit("schema_version")
tool = document["tool"]
if list(tool) != ["name", "version"]:
    sys.exit("tool keys")
if tool["name"] != "inferedge-bench-concurrent" or type(tool["version"]) is not str:
    sys.exit("tool")
if type(document["model"]) is not str or type(document["workload"]) is not str:
    sys.exit("identity type")
if document["backend"] != "ONNX Runtime CPU" or document["correctness"] != "passed":
    sys.exit("identity")
if fixture_mode == "null":
    if document["fixture"] is not None:
        sys.exit("fixture")
elif type(document["fixture"]) is not str or document["fixture"] != fixture_mode:
    sys.exit("fixture")
config = document["configuration"]
config_keys = [
    "worker_count", "session_ownership", "onnx_intra_op_threads_per_session",
    "graph_optimizations", "warmup_iterations_per_worker",
    "measured_iterations_per_worker", "correctness_check_interval",
]
if list(config) != config_keys:
    sys.exit("configuration keys")
if type(config["worker_count"]) is not int or config["worker_count"] != workers:
    sys.exit("worker_count")
if config["session_ownership"] != "one_inference_engine_per_worker":
    sys.exit("session")
if type(config["onnx_intra_op_threads_per_session"]) is not int:
    sys.exit("threads type")
if config["onnx_intra_op_threads_per_session"] != 1:
    sys.exit("threads")
if config["graph_optimizations"] != "disabled":
    sys.exit("optimizations")
if type(config["measured_iterations_per_worker"]) is not int:
    sys.exit("measured type")
if config["measured_iterations_per_worker"] != measured:
    sys.exit("measured")
if type(config["warmup_iterations_per_worker"]) is not int:
    sys.exit("warmup type")
if type(config["correctness_check_interval"]) is not int:
    sys.exit("interval type")
latency = document["latency_microseconds"]
latency_keys = [
    "sample_count", "minimum", "mean", "p50", "p95", "p99", "maximum",
    "population_standard_deviation",
]
if list(latency) != latency_keys:
    sys.exit("latency keys")
if type(latency["sample_count"]) is not int or latency["sample_count"] != workers * measured:
    sys.exit("sample_count")
for key in ("minimum", "mean", "p50", "p95", "p99", "maximum", "population_standard_deviation"):
    value = latency[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        sys.exit(key)
    if value < 0 or math.copysign(1.0, float(value)) < 0:
        sys.exit("negative " + key)
if not (latency["minimum"] <= latency["p50"] <= latency["p95"] <= latency["p99"] <= latency["maximum"]):
    sys.exit("order")
wall = document["measurement_wall_microseconds"]
if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
    sys.exit("wall")
total = document["total_completed_requests"]
if type(total) is not int or total != workers * measured:
    sys.exit("total")
counts = document["per_worker_completed_requests"]
if type(counts) is not list or len(counts) != workers:
    sys.exit("per worker")
if sum(counts) != total:
    sys.exit("per worker sum")
for count in counts:
    if type(count) is not int or count != measured:
        sys.exit("per worker value")
aggregate = document["aggregate_completed_requests_per_second"]
active = document["active_call_requests_per_second"]
for name, value in (("aggregate", aggregate), ("active", active)):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        sys.exit(name + " type")
    if not math.isfinite(value) or value <= 0 or math.copysign(1.0, float(value)) < 0:
        sys.exit(name)
expected = total / (wall / 1e6)
if not math.isclose(aggregate, expected, rel_tol=1e-6, abs_tol=1e-6):
    sys.exit("aggregate mismatch")
print("ok")
)PY";

int python_status(const std::string& document, std::uint32_t workers, std::uint32_t measured,
                  std::string_view fixture_mode, std::string& output, std::string& error) {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("inferedge-concurrent-json-check-" + std::to_string(::getpid()));
    std::filesystem::create_directories(directory);
    const std::filesystem::path document_path = directory / "document.json";
    const std::filesystem::path script_path = directory / "check.py";
    const std::filesystem::path output_path = directory / "stdout.txt";
    const std::filesystem::path error_path = directory / "stderr.txt";
    {
        std::ofstream stream{document_path, std::ios::binary};
        stream << document;
    }
    {
        std::ofstream stream{script_path, std::ios::binary};
        stream << kPythonCheck;
    }
    const std::string command = "python3 \"" + script_path.string() + "\" \"" +
                                document_path.string() + "\" " + std::to_string(workers) + " " +
                                std::to_string(measured) + " \"" + std::string{fixture_mode} +
                                "\" > \"" + output_path.string() + "\" 2> \"" +
                                error_path.string() + "\"";
    const int status = std::system(command.c_str());
    {
        std::ifstream stream{output_path};
        output.assign(std::istreambuf_iterator<char>{stream}, {});
    }
    {
        std::ifstream stream{error_path};
        error.assign(std::istreambuf_iterator<char>{stream}, {});
    }
    return status;
}

bool python3_available() {
    const int status = std::system("python3 -c \"import json, math\" >/dev/null 2>&1");
    return status == 0;
}

} // namespace

TEST(ConcurrentBenchmarkJson, WritesSchemaVersion2) {
    const ConcurrentBenchmarkReport report = synthetic_report();
    const ConcurrentBenchmarkJsonResult written =
        write_concurrent_benchmark_json(report, {}, inferedge::version());
    ASSERT_TRUE(written.error.empty()) << written.error;
    ASSERT_FALSE(written.text.empty());
    EXPECT_EQ(written.text.front(), '{');
    EXPECT_EQ(written.text.back(), '\n');
    EXPECT_EQ(written.text.find("\n{"), std::string::npos) << written.text;
    EXPECT_NE(written.text.find("\"schema_version\": 2,\n"), std::string::npos);
    EXPECT_EQ(written.text.find("\"schema_version\": 1"), std::string::npos);
    EXPECT_EQ(written.text.find("measurement_loop_microseconds"), std::string::npos);
    EXPECT_EQ(written.text.find("throughput_requests_per_second"), std::string::npos);
    EXPECT_EQ(written.text.find("observed_sequential"), std::string::npos);
    EXPECT_EQ(written.text.find("calling_threads"), std::string::npos);
    EXPECT_EQ(written.text.find("per_worker_call_durations"), std::string::npos);
    EXPECT_NE(written.text.find("\"name\": \"inferedge-bench-concurrent\""), std::string::npos);
    EXPECT_NE(written.text.find("\"session_ownership\": \"one_inference_engine_per_worker\""),
              std::string::npos);
    EXPECT_NE(written.text.find("\"onnx_intra_op_threads_per_session\": 1"), std::string::npos);
    EXPECT_NE(written.text.find("\"graph_optimizations\": \"disabled\""), std::string::npos);
    EXPECT_NE(written.text.find("\"per_worker_completed_requests\": [4, 4]"), std::string::npos);
    EXPECT_EQ(raw_field(written.text, "fixture"), "null");
    expect_integer(written.text, "schema_version", 2);
    expect_integer(written.text, "worker_count", 2);
    expect_integer(written.text, "warmup_iterations_per_worker", 1);
    expect_integer(written.text, "measured_iterations_per_worker", 4);
    expect_integer(written.text, "correctness_check_interval", 2);
    expect_integer(written.text, "sample_count", 8);
    expect_integer(written.text, "total_completed_requests", 8);
    expect_round_trip(written.text, "mean", 2.5);
    expect_round_trip(written.text, "p50", 2.0);
    expect_round_trip(written.text, "p95", 3.0);
    expect_round_trip(written.text, "p99", 3.5);
    expect_round_trip(written.text, "population_standard_deviation", 0.1);
    expect_round_trip(written.text, "aggregate_completed_requests_per_second",
                      report.aggregate_completed_requests_per_second);
    expect_round_trip(written.text, "active_call_requests_per_second", 1000.0);
    expect_round_trip(written.text, "measurement_wall_microseconds", 20.0);
    expect_round_trip(written.text, "minimum", 1.5);
    expect_round_trip(written.text, "maximum", 4.0);
}

TEST(ConcurrentBenchmarkJson, PythonParsesAValidDocument) {
    if (!python3_available()) {
        GTEST_SKIP() << "python3 is not available";
    }
    const ConcurrentBenchmarkReport report = synthetic_report();
    const ConcurrentBenchmarkJsonResult written =
        write_concurrent_benchmark_json(report, {}, inferedge::version());
    ASSERT_TRUE(written.error.empty()) << written.error;
    std::string output;
    std::string error;
    const int status = python_status(written.text, 2, 4, "null", output, error);
    EXPECT_EQ(status, 0) << output << error;
    EXPECT_EQ(output, "ok\n") << output;
}

TEST(ConcurrentBenchmarkJson, WritesAMnistFixtureString) {
    const ConcurrentBenchmarkJsonResult written =
        write_concurrent_benchmark_json(synthetic_report(), "test_00003", inferedge::version());
    ASSERT_TRUE(written.error.empty()) << written.error;
    EXPECT_EQ(raw_field(written.text, "fixture"), "\"test_00003\"");
    if (!python3_available()) {
        GTEST_SKIP() << "python3 is not available";
    }
    std::string output;
    std::string error;
    const int status = python_status(written.text, 2, 4, "test_00003", output, error);
    EXPECT_EQ(status, 0) << output << error;
    EXPECT_EQ(output, "ok\n") << output;
}

TEST(ConcurrentBenchmarkJson, EscapesStrings) {
    ConcurrentBenchmarkReport report = synthetic_report();
    report.model = "quote\"slash\\line\n";
    const ConcurrentBenchmarkJsonResult written =
        write_concurrent_benchmark_json(report, {}, inferedge::version());
    ASSERT_TRUE(written.error.empty()) << written.error;
    EXPECT_NE(written.text.find("\"quote\\\"slash\\\\line\\u000a\""), std::string::npos)
        << written.text;
    EXPECT_EQ(written.text.find("quote\"slash"), std::string::npos) << written.text;
}

TEST(ConcurrentBenchmarkJson, RejectsInvalidUtf8) {
    ConcurrentBenchmarkReport report = synthetic_report();
    report.model = bytes({0xFF});
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "not valid UTF-8");
    report = synthetic_report();
    expect_rejected(
        write_concurrent_benchmark_json(report, bytes({0xC3, 0x28}), inferedge::version()),
        "not valid UTF-8");
}

TEST(ConcurrentBenchmarkJson, RejectsNonFiniteMetrics) {
    ConcurrentBenchmarkReport report = synthetic_report();
    report.call_summary.mean_microseconds = std::numeric_limits<double>::quiet_NaN();
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "mean latency is not finite");
    report = synthetic_report();
    report.call_summary.p99_microseconds = std::numeric_limits<double>::infinity();
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "p99 latency is not finite");
    report = synthetic_report();
    report.aggregate_completed_requests_per_second = std::numeric_limits<double>::infinity();
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "aggregate throughput is not finite");
}

TEST(ConcurrentBenchmarkJson, RejectsANegativeDuration) {
    ConcurrentBenchmarkReport report = synthetic_report();
    report.call_summary.minimum = std::chrono::nanoseconds{-1};
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "minimum latency is negative");
    report = synthetic_report();
    report.measurement_wall_duration = std::chrono::nanoseconds{-5};
    report.aggregate_completed_requests_per_second = 1.0;
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "measurement wall duration is not positive");
}

TEST(ConcurrentBenchmarkJson, RejectsNegativeZeroStandardDeviation) {
    ConcurrentBenchmarkReport report = synthetic_report();
    report.call_summary.population_standard_deviation_microseconds = std::copysign(0.0, -1.0);
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "standard deviation is negative");
    EXPECT_TRUE(std::signbit(report.call_summary.population_standard_deviation_microseconds));
}

TEST(ConcurrentBenchmarkJson, RejectsInvalidPercentileOrder) {
    ConcurrentBenchmarkReport report = synthetic_report();
    report.call_summary.p50_microseconds = 3.0;
    report.call_summary.p95_microseconds = 1.0;
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "latency metrics are out of order");
}

TEST(ConcurrentBenchmarkJson, RejectsInconsistentCompletionCounts) {
    ConcurrentBenchmarkReport report = synthetic_report();
    report.per_worker_completed_requests = {4, 3};
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "does not match measured iterations");
    report = synthetic_report();
    report.total_completed_requests = 9;
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "does not match the measured iteration count");
    report = synthetic_report();
    report.per_worker_completed_requests = {4};
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "does not match the worker count");
    report = synthetic_report();
    report.worker_count = 0;
    report.per_worker_completed_requests.clear();
    report.total_completed_requests = 0;
    report.call_summary.sample_count = 0;
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "worker count is not positive");
}

TEST(ConcurrentBenchmarkJson, RejectsAnIncorrectSampleCount) {
    ConcurrentBenchmarkReport report = synthetic_report();
    report.call_summary.sample_count = 7;
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "sample count does not match completed requests");
}

TEST(ConcurrentBenchmarkJson, RejectsNonPositiveThroughput) {
    ConcurrentBenchmarkReport report = synthetic_report();
    report.aggregate_completed_requests_per_second = 0.0;
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "aggregate throughput");
    report = synthetic_report();
    report.aggregate_completed_requests_per_second = 1.0;
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "does not match the wall duration");
    report = synthetic_report();
    report.call_summary.active_call_throughput_requests_per_second = -1.0;
    expect_rejected(write_concurrent_benchmark_json(report, {}, inferedge::version()),
                    "active-call throughput is not positive");
}
