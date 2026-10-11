#include "inferedge/benchmark_json.hpp"

#include <bit>
#include <cctype>
#include <charconv>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>

namespace {

using inferedge::BenchmarkJsonResult;
using inferedge::BenchmarkReport;
using inferedge::write_benchmark_json;

std::string bytes(std::initializer_list<unsigned char> values) {
    std::string text;
    text.reserve(values.size());
    for (const unsigned char value : values) {
        text.push_back(static_cast<char>(value));
    }
    return text;
}

BenchmarkReport synthetic_report() {
    BenchmarkReport report;
    report.workload = "add";
    report.model = "models/add_f32.onnx";
    report.backend = "ONNX Runtime CPU";
    report.warmup_iterations = 1;
    report.measured_iterations = 4;
    report.correctness_check_interval = 2;
    report.call_summary.sample_count = 4;
    report.call_summary.minimum = std::chrono::nanoseconds{1500};
    report.call_summary.maximum = std::chrono::nanoseconds{4000};
    report.call_summary.mean_microseconds = 2.5;
    report.call_summary.p50_microseconds = 2.0;
    report.call_summary.p95_microseconds = 3.0;
    report.call_summary.p99_microseconds = 3.5;
    report.call_summary.population_standard_deviation_microseconds = 0.1;
    report.call_summary.active_call_throughput_requests_per_second = 1000.0;
    report.measurement_loop_duration = std::chrono::nanoseconds{20000};
    report.observed_sequential_throughput_requests_per_second = 500.0;
    report.call_durations.push_back(std::chrono::nanoseconds{1500});
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
    const char* begin = token.data();
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
    const char* begin = token.data();
    const std::from_chars_result result = std::from_chars(begin, begin + token.size(), value);
    EXPECT_EQ(result.ec, std::errc{}) << token;
    EXPECT_EQ(result.ptr, begin + token.size()) << token;
    EXPECT_EQ(std::bit_cast<std::uint64_t>(value), std::bit_cast<std::uint64_t>(expected)) << token;
}

void expect_rejected(const BenchmarkJsonResult& written, std::string_view fragment) {
    EXPECT_TRUE(written.text.empty()) << written.text;
    EXPECT_EQ(written.text.find('{'), std::string::npos);
    EXPECT_EQ(written.error.find('{'), std::string::npos) << written.error;
    EXPECT_NE(written.error.find(fragment), std::string::npos) << written.error;
    EXPECT_EQ(written.error.find("error:"), std::string::npos) << written.error;
}

std::size_t count_char(const std::string& text, char character) {
    std::size_t count = 0;
    for (const char item : text) {
        if (item == character) {
            ++count;
        }
    }
    return count;
}

bool has_trailing_comma(const std::string& text) {
    for (std::size_t index = 0; index < text.size(); ++index) {
        if (text[index] != ',') {
            continue;
        }
        std::size_t next = index + 1;
        while (next < text.size() && (text[next] == ' ' || text[next] == '\n')) {
            ++next;
        }
        if (next < text.size() && text[next] == '}') {
            return true;
        }
    }
    return false;
}

class NumericLocale {
public:
    explicit NumericLocale(const char* name) {
        const char* previous = std::setlocale(LC_NUMERIC, nullptr);
        saved_ = previous == nullptr ? "C" : previous;
        applied_ = std::setlocale(LC_NUMERIC, name) != nullptr;
    }

    ~NumericLocale() { std::setlocale(LC_NUMERIC, saved_.c_str()); }

    NumericLocale(const NumericLocale&) = delete;
    NumericLocale& operator=(const NumericLocale&) = delete;

    bool applied() const { return applied_; }

private:
    std::string saved_;
    bool applied_{false};
};

} // namespace

TEST(BenchmarkJsonTest, SerializesASyntheticReport) {
    const BenchmarkJsonResult written = write_benchmark_json(synthetic_report(), {}, "9.9.9-test");
    ASSERT_TRUE(written.error.empty()) << written.error;
    EXPECT_EQ(written.text.front(), '{');
    EXPECT_EQ(written.text.back(), '\n');
    EXPECT_NE(written.text.size(), 0u);
    EXPECT_NE(written.text[written.text.size() - 2], '\n');
    EXPECT_EQ(count_char(written.text, '{'), 5u);
    EXPECT_EQ(count_char(written.text, '}'), 5u);
    EXPECT_FALSE(has_trailing_comma(written.text)) << written.text;
    EXPECT_EQ(written.text.find("call_duration"), std::string::npos) << written.text;
    EXPECT_EQ(written.text.find("InferEdge benchmark"), std::string::npos);
    EXPECT_EQ(written.text.find("unavailable"), std::string::npos);
}

TEST(BenchmarkJsonTest, WritesSchemaVersionOne) {
    const BenchmarkJsonResult written = write_benchmark_json(synthetic_report(), {}, "9.9.9-test");
    ASSERT_TRUE(written.error.empty()) << written.error;
    EXPECT_EQ(raw_field(written.text, "schema_version"), "1");
    EXPECT_TRUE(written.text.starts_with("{\n  \"schema_version\": 1,\n"));
}

TEST(BenchmarkJsonTest, WritesEveryRequiredField) {
    const BenchmarkJsonResult written = write_benchmark_json(synthetic_report(), {}, "9.9.9-test");
    ASSERT_TRUE(written.error.empty()) << written.error;
    const std::string_view keys[] = {
        "schema_version",
        "tool",
        "name",
        "version",
        "model",
        "workload",
        "fixture",
        "backend",
        "configuration",
        "warmup_iterations",
        "measured_iterations",
        "correctness_check_interval",
        "calling_threads",
        "onnx_intra_op_threads",
        "graph_optimizations",
        "latency_microseconds",
        "sample_count",
        "minimum",
        "mean",
        "p50",
        "p95",
        "p99",
        "maximum",
        "population_standard_deviation",
        "measurement_loop_microseconds",
        "throughput_requests_per_second",
        "active_call",
        "observed_sequential",
        "correctness",
    };
    std::size_t position = 0;
    for (const std::string_view key : keys) {
        const std::size_t found =
            written.text.find(std::string{"\""} + std::string{key} + "\"", position);
        EXPECT_NE(found, std::string::npos) << key;
        if (found != std::string::npos) {
            position = found + 1;
        }
    }
}

TEST(BenchmarkJsonTest, UsesJsonTypes) {
    const BenchmarkJsonResult written =
        write_benchmark_json(synthetic_report(), "test_00003", "9.9.9-test");
    ASSERT_TRUE(written.error.empty()) << written.error;
    EXPECT_EQ(raw_field(written.text, "name"), "\"inferedge-bench\"");
    EXPECT_EQ(raw_field(written.text, "version"), "\"9.9.9-test\"");
    EXPECT_EQ(raw_field(written.text, "workload"), "\"add\"");
    EXPECT_EQ(raw_field(written.text, "backend"), "\"ONNX Runtime CPU\"");
    EXPECT_EQ(raw_field(written.text, "graph_optimizations"), "\"disabled\"");
    EXPECT_EQ(raw_field(written.text, "correctness"), "\"passed\"");
    EXPECT_EQ(raw_field(written.text, "fixture"), "\"test_00003\"");
    EXPECT_EQ(raw_field(written.text, "schema_version").find('.'), std::string::npos);
}

TEST(BenchmarkJsonTest, WritesIntegerCounts) {
    const BenchmarkJsonResult written = write_benchmark_json(synthetic_report(), {}, "9.9.9-test");
    ASSERT_TRUE(written.error.empty()) << written.error;
    expect_integer(written.text, "warmup_iterations", 1);
    expect_integer(written.text, "measured_iterations", 4);
    expect_integer(written.text, "correctness_check_interval", 2);
    expect_integer(written.text, "calling_threads", 1);
    expect_integer(written.text, "onnx_intra_op_threads", 1);
    expect_integer(written.text, "sample_count", 4);
}

TEST(BenchmarkJsonTest, RoundTripsFloatingPointMetrics) {
    BenchmarkReport report = synthetic_report();
    const double fraction = static_cast<double>(1) / 1000.0;
    report.call_summary.minimum = std::chrono::nanoseconds{1};
    report.call_summary.maximum = std::chrono::nanoseconds{1};
    report.call_summary.mean_microseconds = 0.1;
    report.call_summary.p50_microseconds = fraction;
    report.call_summary.p95_microseconds = fraction;
    report.call_summary.p99_microseconds = fraction;
    report.measurement_loop_duration = std::chrono::nanoseconds{1};
    const BenchmarkJsonResult written = write_benchmark_json(report, {}, "9.9.9-test");
    ASSERT_TRUE(written.error.empty()) << written.error;
    expect_round_trip(written.text, "minimum", fraction);
    expect_round_trip(written.text, "maximum", fraction);
    expect_round_trip(written.text, "mean", 0.1);
    expect_round_trip(written.text, "p50", fraction);
    expect_round_trip(written.text, "p95", fraction);
    expect_round_trip(written.text, "p99", fraction);
    expect_round_trip(written.text, "population_standard_deviation", 0.1);
    expect_round_trip(written.text, "measurement_loop_microseconds", fraction);
    expect_round_trip(written.text, "active_call", 1000.0);
    expect_round_trip(written.text, "observed_sequential", 500.0);

    for (const char* name : {"de_DE.UTF-8", "de_DE", "fr_FR.UTF-8", "fr_FR"}) {
        const NumericLocale locale{name};
        if (!locale.applied()) {
            continue;
        }
        const BenchmarkJsonResult localized = write_benchmark_json(synthetic_report(), {}, "0.1.0");
        ASSERT_TRUE(localized.error.empty()) << localized.error;
        EXPECT_EQ(raw_field(localized.text, "mean"), "2.5") << localized.text;
        break;
    }
    const BenchmarkJsonResult decimal = write_benchmark_json(synthetic_report(), {}, "0.1.0");
    ASSERT_TRUE(decimal.error.empty()) << decimal.error;
    EXPECT_EQ(raw_field(decimal.text, "minimum"), "1.5");
    EXPECT_EQ(raw_field(decimal.text, "mean"), "2.5");
}

TEST(BenchmarkJsonTest, WritesNullFixtureForAdd) {
    const BenchmarkJsonResult written = write_benchmark_json(synthetic_report(), {}, "0.1.0");
    ASSERT_TRUE(written.error.empty()) << written.error;
    EXPECT_EQ(raw_field(written.text, "fixture"), "null");
}

TEST(BenchmarkJsonTest, WritesMnistFixtureString) {
    const BenchmarkJsonResult written =
        write_benchmark_json(synthetic_report(), "test_00003", "0.1.0");
    ASSERT_TRUE(written.error.empty()) << written.error;
    EXPECT_EQ(raw_field(written.text, "fixture"), "\"test_00003\"");
}

TEST(BenchmarkJsonTest, EscapesQuotes) {
    BenchmarkReport report = synthetic_report();
    report.model = "say \"hi\"";
    const BenchmarkJsonResult written = write_benchmark_json(report, {}, "0.1.0");
    ASSERT_TRUE(written.error.empty()) << written.error;
    EXPECT_EQ(raw_field(written.text, "model"), "\"say \\\"hi\\\"\"");
}

TEST(BenchmarkJsonTest, EscapesBackslashes) {
    BenchmarkReport report = synthetic_report();
    report.model = "a\\b\\\\c";
    const BenchmarkJsonResult written = write_benchmark_json(report, {}, "0.1.0");
    ASSERT_TRUE(written.error.empty()) << written.error;
    EXPECT_EQ(raw_field(written.text, "model"), "\"a\\\\b\\\\\\\\c\"");
}

TEST(BenchmarkJsonTest, EscapesAsciiControls) {
    BenchmarkReport report = synthetic_report();
    report.model = "a";
    report.model.push_back('\0');
    report.model += "b\t c\n d\r e";
    const BenchmarkJsonResult clean = write_benchmark_json(synthetic_report(), {}, "0.1.0");
    const BenchmarkJsonResult written = write_benchmark_json(report, {}, "0.1.0");
    ASSERT_TRUE(written.error.empty()) << written.error;
    EXPECT_NE(written.text.find("\"model\": \"a\\u0000b\\u0009 c\\u000a d\\u000d e\""),
              std::string::npos)
        << written.text;
    EXPECT_EQ(written.text.find("\\n"), std::string::npos);
    EXPECT_EQ(written.text.find("\\t"), std::string::npos);
    EXPECT_EQ(written.text.find("\\r"), std::string::npos);
    EXPECT_EQ(count_char(written.text, '\n'), count_char(clean.text, '\n'));
}

TEST(BenchmarkJsonTest, PreservesValidUtf8) {
    BenchmarkReport report = synthetic_report();
    const std::string euro = bytes({0xE2, 0x82, 0xAC});
    const std::string emoji = bytes({0xF0, 0x9F, 0x98, 0x80});
    const std::string copyright = bytes({0xC2, 0xA9});
    report.model = "price " + euro;
    const BenchmarkJsonResult written = write_benchmark_json(report, emoji + copyright, "0.1.0");
    ASSERT_TRUE(written.error.empty()) << written.error;
    EXPECT_NE(written.text.find(euro), std::string::npos);
    EXPECT_NE(written.text.find(emoji), std::string::npos);
    EXPECT_NE(written.text.find(copyright), std::string::npos);
    EXPECT_EQ(written.text.find("\\u"), std::string::npos) << written.text;
}

TEST(BenchmarkJsonTest, RejectsInvalidUtf8) {
    const std::string sequences[] = {
        bytes({0x80}),
        bytes({0xC2, 0x20}),
        bytes({0xE2, 0x82, 0x20}),
        bytes({0xF4, 0x90, 0x80, 0x80}),
        bytes({0xF5, 0x80, 0x80, 0x80}),
        bytes({0xFF}),
        std::string("ok") + bytes({0x80}),
    };
    for (const std::string& sequence : sequences) {
        BenchmarkReport report = synthetic_report();
        report.model = sequence;
        expect_rejected(write_benchmark_json(report, {}, "0.1.0"), "UTF-8");
    }
}

TEST(BenchmarkJsonTest, RejectsOverlongUtf8) {
    const std::string sequences[] = {
        bytes({0xC0, 0x80}),
        bytes({0xC1, 0x81}),
        bytes({0xE0, 0x80, 0x80}),
        bytes({0xE0, 0x9F, 0xBF}),
        bytes({0xF0, 0x80, 0x80, 0x80}),
        bytes({0xF0, 0x8F, 0xBF, 0xBF}),
    };
    for (const std::string& sequence : sequences) {
        BenchmarkReport report = synthetic_report();
        report.model = sequence;
        expect_rejected(write_benchmark_json(report, {}, "0.1.0"), "UTF-8");
    }
}

TEST(BenchmarkJsonTest, RejectsUtf8Surrogates) {
    const std::string sequences[] = {
        bytes({0xED, 0xA0, 0x80}),
        bytes({0xED, 0xBF, 0xBF}),
    };
    for (const std::string& sequence : sequences) {
        BenchmarkReport report = synthetic_report();
        report.workload = sequence;
        expect_rejected(write_benchmark_json(report, {}, "0.1.0"), "UTF-8");
    }
}

TEST(BenchmarkJsonTest, RejectsIncompleteUtf8) {
    const std::string sequences[] = {
        bytes({0xC2}),
        bytes({0xE2, 0x82}),
        bytes({0xF0, 0x9F, 0x98}),
    };
    for (const std::string& sequence : sequences) {
        expect_rejected(write_benchmark_json(synthetic_report(), sequence, "0.1.0"), "UTF-8");
    }
}

TEST(BenchmarkJsonTest, RejectsNan) {
    BenchmarkReport report = synthetic_report();
    report.call_summary.mean_microseconds = std::numeric_limits<double>::quiet_NaN();
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"), "mean latency is not finite");

    report = synthetic_report();
    report.call_summary.p99_microseconds = std::numeric_limits<double>::quiet_NaN();
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"), "p99 latency is not finite");
}

TEST(BenchmarkJsonTest, RejectsInfinity) {
    BenchmarkReport report = synthetic_report();
    report.call_summary.active_call_throughput_requests_per_second =
        std::numeric_limits<double>::infinity();
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"),
                    "active-call throughput is not finite");

    report = synthetic_report();
    report.observed_sequential_throughput_requests_per_second =
        -std::numeric_limits<double>::infinity();
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"),
                    "observed sequential throughput is not finite");
}

TEST(BenchmarkJsonTest, RejectsInvalidMetricOrdering) {
    BenchmarkReport report = synthetic_report();
    report.call_summary.p50_microseconds = 3.5;
    report.call_summary.p95_microseconds = 3.0;
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"), "out of order");

    report = synthetic_report();
    report.call_summary.p99_microseconds = 100.0;
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"), "out of order");

    report = synthetic_report();
    report.call_summary.minimum = std::chrono::nanoseconds{-1000};
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"), "minimum latency is negative");

    report = synthetic_report();
    report.call_summary.minimum = std::chrono::nanoseconds{4000};
    report.call_summary.maximum = std::chrono::nanoseconds{1500};
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"), "below the minimum");

    report = synthetic_report();
    report.call_summary.population_standard_deviation_microseconds = -0.001;
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"), "standard deviation is negative");

    report = synthetic_report();
    report.call_summary.p95_microseconds = report.call_summary.p50_microseconds - 1e-12;
    const BenchmarkJsonResult tiny = write_benchmark_json(report, {}, "0.1.0");
    EXPECT_TRUE(tiny.error.empty()) << tiny.error;
    EXPECT_FALSE(tiny.text.empty());

    report = synthetic_report();
    report.call_summary.active_call_throughput_requests_per_second = 0.0;
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"),
                    "active-call throughput is not positive");
    report = synthetic_report();
    report.observed_sequential_throughput_requests_per_second = -5.0;
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"),
                    "observed sequential throughput is not positive");
}

TEST(BenchmarkJsonTest, RejectsTinyNegativeStandardDeviation) {
    BenchmarkReport report = synthetic_report();
    report.call_summary.population_standard_deviation_microseconds = -1e-12;
    const BenchmarkJsonResult tiny = write_benchmark_json(report, {}, "0.1.0");
    expect_rejected(tiny, "standard deviation is negative");
    EXPECT_EQ(tiny.text.find("-1e-12"), std::string::npos);
    EXPECT_EQ(tiny.text.find("\"population_standard_deviation\""), std::string::npos);

    report = synthetic_report();
    report.call_summary.population_standard_deviation_microseconds = -0.0;
    const BenchmarkJsonResult negative_zero = write_benchmark_json(report, {}, "0.1.0");
    expect_rejected(negative_zero, "standard deviation is negative");
    EXPECT_EQ(negative_zero.text.find("-0"), std::string::npos);

    report = synthetic_report();
    report.call_summary.population_standard_deviation_microseconds = 0.0;
    const BenchmarkJsonResult zero = write_benchmark_json(report, {}, "0.1.0");
    ASSERT_TRUE(zero.error.empty()) << zero.error;
    EXPECT_EQ(raw_field(zero.text, "population_standard_deviation"), "0");
}

TEST(BenchmarkJsonTest, RejectsInvalidSampleCount) {
    BenchmarkReport report = synthetic_report();
    report.call_summary.sample_count = 0;
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"), "sample count is not positive");

    report = synthetic_report();
    report.call_summary.sample_count = 3;
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"),
                    "sample count does not match measured iterations");
}

TEST(BenchmarkJsonTest, RejectsNonPositiveLoopDuration) {
    BenchmarkReport report = synthetic_report();
    report.measurement_loop_duration = std::chrono::nanoseconds{0};
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"),
                    "measurement loop duration is not positive");

    report = synthetic_report();
    report.measurement_loop_duration = std::chrono::nanoseconds{-20};
    expect_rejected(write_benchmark_json(report, {}, "0.1.0"),
                    "measurement loop duration is not positive");
}

TEST(BenchmarkJsonTest, ReturnsNoPartialOutputOnError) {
    BenchmarkReport report = synthetic_report();
    report.model = bytes({0xC0, 0x80});
    report.call_summary.mean_microseconds = std::numeric_limits<double>::quiet_NaN();
    const BenchmarkJsonResult written = write_benchmark_json(report, "bad", "0.1.0");
    expect_rejected(written, "not finite");
    EXPECT_EQ(written.text.find("schema_version"), std::string::npos);
}

TEST(BenchmarkJsonTest, RepeatsTheSameDocument) {
    const BenchmarkReport report = synthetic_report();
    const BenchmarkJsonResult first = write_benchmark_json(report, "test_00003", "0.1.0");
    const BenchmarkJsonResult second = write_benchmark_json(report, "test_00003", "0.1.0");
    ASSERT_TRUE(first.error.empty()) << first.error;
    EXPECT_EQ(first.text, second.text);
    EXPECT_NE(first.text.find("\"version\": \"0.1.0\""), std::string::npos);
}
