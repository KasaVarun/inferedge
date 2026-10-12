#include "inferedge/concurrent_benchmark_cli.hpp"
#include "inferedge/version.hpp"

#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <poll.h>
#include <spawn.h>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#ifndef INFEREDGE_BENCH_CONCURRENT_EXECUTABLE
#error "CMake must define INFEREDGE_BENCH_CONCURRENT_EXECUTABLE"
#endif

#ifndef INFEREDGE_BENCH_EXECUTABLE
#error "CMake must define INFEREDGE_BENCH_EXECUTABLE"
#endif

#ifndef INFEREDGE_ADD_MODEL_PATH
#error "CMake must define INFEREDGE_ADD_MODEL_PATH"
#endif

#ifndef INFEREDGE_MNIST_MODEL_PATH
#error "CMake must define INFEREDGE_MNIST_MODEL_PATH"
#endif

#ifndef INFEREDGE_RANK2_MODEL_PATH
#error "CMake must define INFEREDGE_RANK2_MODEL_PATH"
#endif

#ifndef INFEREDGE_INT64_MODEL_PATH
#error "CMake must define INFEREDGE_INT64_MODEL_PATH"
#endif

#ifndef INFEREDGE_INVALID_MODEL_PATH
#error "CMake must define INFEREDGE_INVALID_MODEL_PATH"
#endif

#ifndef INFEREDGE_MNIST_FIXTURE_DIR
#error "CMake must define INFEREDGE_MNIST_FIXTURE_DIR"
#endif

extern "C" char** environ;

namespace {

using inferedge::ConcurrentBenchmarkCliAction;
using inferedge::ConcurrentBenchmarkCliResult;
using inferedge::ConcurrentBenchmarkOutputFormat;
using inferedge::execute_concurrent_benchmark;
using inferedge::parse_concurrent_benchmark_cli;

class ConcurrentBenchmarkCommandTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

class TempTree {
public:
    TempTree()
        : root_(std::filesystem::temp_directory_path() /
                ("inferedge-bench-concurrent-fixture-" + std::to_string(::getpid()))) {
        std::filesystem::remove_all(root_);
        std::filesystem::create_directories(root_ / "inputs");
        std::filesystem::create_directories(root_ / "onnx");
    }

    ~TempTree() { std::filesystem::remove_all(root_); }

    TempTree(const TempTree&) = delete;
    TempTree& operator=(const TempTree&) = delete;

    const std::filesystem::path& root() const { return root_; }

private:
    std::filesystem::path root_;
};

ConcurrentBenchmarkCliResult parse_words(const std::vector<std::string>& args) {
    std::vector<std::string_view> views;
    views.reserve(args.size());
    for (const std::string& arg : args) {
        views.emplace_back(arg);
    }
    return parse_concurrent_benchmark_cli(views);
}

ConcurrentBenchmarkCliResult run_words(const std::vector<std::string>& args) {
    const ConcurrentBenchmarkCliResult parsed = parse_words(args);
    if (parsed.action != ConcurrentBenchmarkCliAction::kRun) {
        return parsed;
    }
    return execute_concurrent_benchmark(parsed);
}

void expect_usage(const ConcurrentBenchmarkCliResult& result, std::string_view fragment) {
    EXPECT_EQ(result.action, ConcurrentBenchmarkCliAction::kError);
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_EQ(result.message.compare(0, 7, "error: "), 0) << result.message;
    EXPECT_NE(result.message.find(fragment), std::string::npos) << result.message;
    EXPECT_NE(result.message.find("Usage: inferedge-bench-concurrent"), std::string::npos)
        << result.message;
}

void expect_failure(const ConcurrentBenchmarkCliResult& result, int exit_code,
                    std::string_view fragment) {
    EXPECT_EQ(result.action, ConcurrentBenchmarkCliAction::kError);
    EXPECT_EQ(result.exit_code, exit_code);
    EXPECT_EQ(result.message.compare(0, 7, "error: "), 0) << result.message;
    EXPECT_NE(result.message.find(fragment), std::string::npos) << result.message;
    EXPECT_EQ(result.message.find("Correctness: passed"), std::string::npos) << result.message;
    EXPECT_EQ(result.message.find("InferEdge concurrent benchmark"), std::string::npos)
        << result.message;
    EXPECT_EQ(result.message.find('{'), std::string::npos) << result.message;
}

void expect_text_report(const ConcurrentBenchmarkCliResult& result, std::string_view workload,
                        std::string_view model, std::uint32_t workers, std::uint32_t warmup,
                        std::uint32_t measured, std::string_view fixture) {
    EXPECT_EQ(result.action, ConcurrentBenchmarkCliAction::kRun);
    EXPECT_EQ(result.exit_code, 0);
    const std::string& text = result.message;
    EXPECT_NE(text.find("InferEdge concurrent benchmark\n"), std::string::npos) << text;
    EXPECT_NE(text.find("Model: " + std::string{model}), std::string::npos) << text;
    EXPECT_NE(text.find("Workload: " + std::string{workload}), std::string::npos) << text;
    EXPECT_NE(text.find("Backend: ONNX Runtime CPU\n"), std::string::npos) << text;
    EXPECT_NE(text.find("Workers: " + std::to_string(workers) + "\n"), std::string::npos) << text;
    EXPECT_NE(text.find("one InferenceEngine per worker"), std::string::npos) << text;
    EXPECT_NE(text.find("one ONNX intra-op thread per session"), std::string::npos) << text;
    EXPECT_NE(text.find("graph optimizations disabled"), std::string::npos) << text;
    EXPECT_NE(text.find("Warmup iterations per worker: " + std::to_string(warmup) + "\n"),
              std::string::npos);
    EXPECT_NE(text.find("Measured iterations per worker: " + std::to_string(measured) + "\n"),
              std::string::npos);
    EXPECT_NE(text.find("Correctness-check interval: "), std::string::npos) << text;
    const std::uint64_t total = static_cast<std::uint64_t>(workers) * measured;
    EXPECT_NE(text.find("Total completed requests: " + std::to_string(total) + "\n"),
              std::string::npos)
        << text;
    EXPECT_NE(text.find("  minimum: "), std::string::npos) << text;
    EXPECT_NE(text.find(" ns ("), std::string::npos) << text;
    EXPECT_NE(text.find(" us)"), std::string::npos) << text;
    EXPECT_NE(text.find("  mean: "), std::string::npos) << text;
    EXPECT_NE(text.find("  p50: "), std::string::npos) << text;
    EXPECT_NE(text.find("  p95: "), std::string::npos) << text;
    EXPECT_NE(text.find("  p99: "), std::string::npos) << text;
    EXPECT_NE(text.find("  maximum: "), std::string::npos) << text;
    EXPECT_NE(text.find("Measurement wall duration: "), std::string::npos) << text;
    EXPECT_NE(text.find("Aggregate throughput: "), std::string::npos) << text;
    EXPECT_NE(text.find("Active-call throughput: "), std::string::npos) << text;
    EXPECT_NE(text.find("requests/second"), std::string::npos) << text;
    EXPECT_NE(text.find("Correctness: passed\n"), std::string::npos) << text;
    EXPECT_NE(text.find("shared wall-clock duration"), std::string::npos) << text;
    EXPECT_NE(text.find("sum of per-call durations"), std::string::npos) << text;
    EXPECT_NE(text.find("production server capacity"), std::string::npos) << text;
    EXPECT_EQ(text.find("schema_version"), std::string::npos) << text;
    EXPECT_EQ(text.find("GPU"), std::string::npos) << text;
    if (fixture.empty()) {
        EXPECT_EQ(text.find("Fixture:"), std::string::npos) << text;
    } else {
        EXPECT_NE(text.find("Fixture: " + std::string{fixture} + "\n"), std::string::npos) << text;
    }
}

void write_f32(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream output{path, std::ios::binary};
    ASSERT_TRUE(output) << path;
    for (const float value : values) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        const unsigned char bytes[4] = {
            static_cast<unsigned char>(bits & 0xffU),
            static_cast<unsigned char>((bits >> 8U) & 0xffU),
            static_cast<unsigned char>((bits >> 16U) & 0xffU),
            static_cast<unsigned char>((bits >> 24U) & 0xffU),
        };
        output.write(reinterpret_cast<const char*>(bytes), 4);
    }
}

class PipeEnds {
public:
    PipeEnds() {
        if (::pipe(fds_) != 0) {
            fds_[0] = -1;
            fds_[1] = -1;
        }
    }

    ~PipeEnds() {
        close_read();
        close_write();
    }

    PipeEnds(const PipeEnds&) = delete;
    PipeEnds& operator=(const PipeEnds&) = delete;

    int read_end() const { return fds_[0]; }
    int write_end() const { return fds_[1]; }
    bool valid() const { return fds_[0] >= 0 && fds_[1] >= 0; }
    void close_read() { close_at(0); }
    void close_write() { close_at(1); }

private:
    void close_at(int index) {
        if (fds_[index] >= 0) {
            ::close(fds_[index]);
            fds_[index] = -1;
        }
    }

    int fds_[2]{-1, -1};
};

struct CapturedProcess {
    int exit_code{-1};
    std::string standard_output;
    std::string standard_error;
};

enum class Drain {
    kOpen,
    kClosed,
    kFailed,
};

Drain drain(int fd, std::string& destination) {
    char buffer[4096];
    while (true) {
        const ssize_t count = ::read(fd, buffer, sizeof(buffer));
        if (count > 0) {
            destination.append(buffer, static_cast<std::size_t>(count));
            continue;
        }
        if (count == 0) {
            return Drain::kClosed;
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno == EAGAIN) {
            return Drain::kOpen;
        }
        return Drain::kFailed;
    }
}

bool finish_pipe(const pollfd& ready, int fd, std::string& destination, bool& open) {
    if (!open || (ready.revents & (POLLIN | POLLHUP | POLLERR)) == 0) {
        return true;
    }
    const Drain state = drain(fd, destination);
    if (state == Drain::kFailed) {
        return false;
    }
    if (state == Drain::kClosed || (ready.revents & (POLLHUP | POLLERR)) != 0) {
        open = false;
    }
    return true;
}

CapturedProcess run_captured(const char* executable, const std::vector<std::string>& args) {
    CapturedProcess captured;
    PipeEnds output;
    PipeEnds error;
    if (!output.valid() || !error.valid()) {
        ADD_FAILURE() << "pipe failed: " << std::strerror(errno);
        return captured;
    }

    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) {
        ADD_FAILURE() << "posix_spawn_file_actions_init failed";
        return captured;
    }
    posix_spawn_file_actions_adddup2(&actions, output.write_end(), STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, error.write_end(), STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, output.read_end());
    posix_spawn_file_actions_addclose(&actions, output.write_end());
    posix_spawn_file_actions_addclose(&actions, error.read_end());
    posix_spawn_file_actions_addclose(&actions, error.write_end());

    std::vector<std::string> arguments;
    arguments.emplace_back(executable);
    arguments.insert(arguments.end(), args.begin(), args.end());
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 1);
    for (std::string& argument : arguments) {
        argv.push_back(argument.data());
    }
    argv.push_back(nullptr);

    std::vector<std::string> environment;
    bool saw_telemetry = false;
    if (environ != nullptr) {
        for (char** entry = environ; *entry != nullptr; ++entry) {
            std::string item{*entry};
            if (item.rfind("ORT_DISABLE_TELEMETRY=", 0) == 0) {
                item = "ORT_DISABLE_TELEMETRY=1";
                saw_telemetry = true;
            }
            environment.push_back(std::move(item));
        }
    }
    if (!saw_telemetry) {
        environment.emplace_back("ORT_DISABLE_TELEMETRY=1");
    }
    std::vector<char*> envp;
    envp.reserve(environment.size() + 1);
    for (std::string& item : environment) {
        envp.push_back(item.data());
    }
    envp.push_back(nullptr);

    pid_t pid = 0;
    const int spawned = posix_spawn(&pid, executable, &actions, nullptr, argv.data(), envp.data());
    posix_spawn_file_actions_destroy(&actions);
    output.close_write();
    error.close_write();
    if (spawned != 0) {
        ADD_FAILURE() << "posix_spawn failed: " << std::strerror(spawned);
        return captured;
    }

    ::fcntl(output.read_end(), F_SETFL, O_NONBLOCK);
    ::fcntl(error.read_end(), F_SETFL, O_NONBLOCK);
    bool output_open = true;
    bool error_open = true;
    while (output_open || error_open) {
        pollfd ready[2]{};
        nfds_t count = 0;
        pollfd* output_ready = nullptr;
        pollfd* error_ready = nullptr;
        if (output_open) {
            ready[count].fd = output.read_end();
            ready[count].events = POLLIN;
            output_ready = &ready[count];
            ++count;
        }
        if (error_open) {
            ready[count].fd = error.read_end();
            ready[count].events = POLLIN;
            error_ready = &ready[count];
            ++count;
        }
        const int polled = ::poll(ready, count, -1);
        if (polled < 0) {
            if (errno == EINTR) {
                continue;
            }
            ADD_FAILURE() << "poll failed: " << std::strerror(errno);
            break;
        }
        if (output_ready != nullptr &&
            !finish_pipe(*output_ready, output.read_end(), captured.standard_output, output_open)) {
            ADD_FAILURE() << "reading stdout failed";
            break;
        }
        if (error_ready != nullptr &&
            !finish_pipe(*error_ready, error.read_end(), captured.standard_error, error_open)) {
            ADD_FAILURE() << "reading stderr failed";
            break;
        }
    }
    output.close_read();
    error.close_read();

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            ADD_FAILURE() << "waitpid failed: " << std::strerror(errno);
            return captured;
        }
    }
    if (WIFEXITED(status)) {
        captured.exit_code = WEXITSTATUS(status);
    } else {
        ADD_FAILURE() << "child did not exit normally";
    }
    return captured;
}

CapturedProcess run_concurrent(const std::vector<std::string>& args) {
    return run_captured(INFEREDGE_BENCH_CONCURRENT_EXECUTABLE, args);
}

void expect_process_failure(const CapturedProcess& captured, int exit_code,
                            std::string_view fragment) {
    EXPECT_EQ(captured.exit_code, exit_code);
    EXPECT_TRUE(captured.standard_output.empty()) << captured.standard_output;
    ASSERT_FALSE(captured.standard_error.empty());
    EXPECT_EQ(captured.standard_error.compare(0, 7, "error: "), 0) << captured.standard_error;
    EXPECT_NE(captured.standard_error.find(fragment), std::string::npos) << captured.standard_error;
    EXPECT_EQ(captured.standard_error.find("Correctness: passed"), std::string::npos)
        << captured.standard_error;
    EXPECT_EQ(captured.standard_output.find('{'), std::string::npos);
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
document = json.loads(raw, parse_constant=reject_constant)
if type(document["schema_version"]) is not int or document["schema_version"] != 2:
    sys.exit("schema")
if document["tool"]["name"] != "inferedge-bench-concurrent":
    sys.exit("tool")
config = document["configuration"]
if config["worker_count"] != workers or config["measured_iterations_per_worker"] != measured:
    sys.exit("counts")
if config["session_ownership"] != "one_inference_engine_per_worker":
    sys.exit("session")
if config["onnx_intra_op_threads_per_session"] != 1 or config["graph_optimizations"] != "disabled":
    sys.exit("policy")
total = document["total_completed_requests"]
if type(total) is not int or total != workers * measured:
    sys.exit("total")
if document["latency_microseconds"]["sample_count"] != total:
    sys.exit("samples")
counts = document["per_worker_completed_requests"]
if counts != [measured] * workers:
    sys.exit("workers")
if fixture_mode == "null":
    if document["fixture"] is not None:
        sys.exit("fixture")
elif document["fixture"] != fixture_mode:
    sys.exit("fixture")
wall = document["measurement_wall_microseconds"]
aggregate = document["aggregate_completed_requests_per_second"]
active = document["active_call_requests_per_second"]
if not math.isfinite(wall) or wall <= 0:
    sys.exit("wall")
if not math.isfinite(aggregate) or aggregate <= 0:
    sys.exit("aggregate")
if not math.isfinite(active) or active <= 0:
    sys.exit("active")
expected = total / (wall / 1e6)
if not math.isclose(aggregate, expected, rel_tol=1e-6, abs_tol=1e-6):
    sys.exit("aggregate mismatch")
if document["correctness"] != "passed" or document["backend"] != "ONNX Runtime CPU":
    sys.exit("identity")
print("ok")
)PY";

bool python3_available() {
    return std::system("python3 -c \"import json, math\" >/dev/null 2>&1") == 0;
}

void expect_python_json(const std::string& document, std::uint32_t workers, std::uint32_t measured,
                        std::string_view fixture_mode) {
    if (!python3_available()) {
        GTEST_SKIP() << "python3 is not available";
    }
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("inferedge-concurrent-cli-json-" + std::to_string(::getpid()));
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
    std::string output;
    std::string error;
    {
        std::ifstream stream{output_path};
        output.assign(std::istreambuf_iterator<char>{stream}, {});
    }
    {
        std::ifstream stream{error_path};
        error.assign(std::istreambuf_iterator<char>{stream}, {});
    }
    EXPECT_EQ(status, 0) << output << error;
    EXPECT_EQ(output, "ok\n") << output;
}

std::vector<std::string> add_command(std::string_view concurrency, std::string_view format) {
    return {"--model",       INFEREDGE_ADD_MODEL_PATH,
            "--workload",    "add",
            "--concurrency", std::string{concurrency},
            "--warmup",      "0",
            "--iterations",  "2",
            "--format",      std::string{format}};
}

} // namespace

TEST(ConcurrentBenchmarkCli, HelpAndVersionStandAlone) {
    const ConcurrentBenchmarkCliResult empty = parse_words({});
    const ConcurrentBenchmarkCliResult help = parse_words({"--help"});
    const ConcurrentBenchmarkCliResult short_help = parse_words({"-h"});
    EXPECT_EQ(empty.action, ConcurrentBenchmarkCliAction::kHelp);
    EXPECT_EQ(empty.exit_code, 0);
    EXPECT_EQ(empty.message, inferedge::concurrent_benchmark_help_message());
    EXPECT_EQ(help.message, empty.message);
    EXPECT_EQ(short_help.message, empty.message);
    EXPECT_NE(empty.message.find("--concurrency <count>"), std::string::npos);
    EXPECT_NE(empty.message.find("Workers, required, from 1 through 64"), std::string::npos);
    EXPECT_NE(empty.message.find("--concurrency has no default."), std::string::npos);
    EXPECT_NE(empty.message.find("schema version 2"), std::string::npos);

    const ConcurrentBenchmarkCliResult version = parse_words({"--version"});
    const ConcurrentBenchmarkCliResult short_version = parse_words({"-V"});
    EXPECT_EQ(version.action, ConcurrentBenchmarkCliAction::kVersion);
    EXPECT_EQ(version.exit_code, 0);
    EXPECT_EQ(version.message,
              "inferedge-bench-concurrent " + std::string{inferedge::version()} + "\n");
    EXPECT_EQ(short_version.message, version.message);
}

TEST(ConcurrentBenchmarkCli, RejectsHelpOrVersionWithExtraArguments) {
    expect_usage(parse_words({"--help", "--model", "m.onnx"}), "unexpected argument");
    expect_usage(parse_words({"--version", "--help"}), "unexpected argument");
    expect_usage(parse_words({"-h", "extra"}), "unexpected argument");
    expect_usage(parse_words({"-V", "--concurrency", "1"}), "unexpected argument");
    expect_usage(parse_words({"--model", "m.onnx", "--help"}), "unexpected argument '--help'");
}

TEST(ConcurrentBenchmarkCli, RequiresModelWorkloadAndConcurrency) {
    expect_usage(parse_words({"--workload", "add", "--concurrency", "1"}), "missing --model");
    expect_usage(parse_words({"--model", "m.onnx", "--concurrency", "1"}), "missing --workload");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add"}), "missing --concurrency");
    expect_usage(parse_words({"--model=m.onnx", "--workload=add"}), "missing --concurrency");
}

TEST(ConcurrentBenchmarkCli, AppliesDefaultsWhenConcurrencyIsPresent) {
    const ConcurrentBenchmarkCliResult spaced =
        parse_words({"--model", "m.onnx", "--workload", "mnist", "--concurrency", "4"});
    const ConcurrentBenchmarkCliResult attached =
        parse_words({"--model=m.onnx", "--workload=mnist", "--concurrency=4"});
    for (const ConcurrentBenchmarkCliResult& result : {spaced, attached}) {
        EXPECT_EQ(result.action, ConcurrentBenchmarkCliAction::kRun);
        EXPECT_EQ(result.exit_code, 0);
        EXPECT_EQ(result.model_path, "m.onnx");
        EXPECT_EQ(result.workload, "mnist");
        EXPECT_EQ(result.worker_count, 4U);
        EXPECT_EQ(result.warmup_iterations, 10U);
        EXPECT_EQ(result.measured_iterations, 100U);
        EXPECT_EQ(result.correctness_check_interval, 1U);
        EXPECT_EQ(result.fixture_id, "test_00003");
        EXPECT_FALSE(result.fixture_specified);
        EXPECT_FALSE(result.fixture_directory_specified);
        EXPECT_EQ(result.output_format, ConcurrentBenchmarkOutputFormat::kText);
    }
}

TEST(ConcurrentBenchmarkCli, AcceptsConcurrencyBoundsInBothForms) {
    EXPECT_EQ(
        parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1"}).worker_count,
        1U);
    EXPECT_EQ(parse_words({"--model=m.onnx", "--workload=add", "--concurrency=1"}).worker_count,
              1U);
    EXPECT_EQ(
        parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "64"}).worker_count,
        64U);
    EXPECT_EQ(parse_words({"--model=m.onnx", "--workload=add", "--concurrency=64"}).worker_count,
              64U);
}

TEST(ConcurrentBenchmarkCli, RejectsConcurrencyOutsideTheRange) {
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "0"}),
                 "concurrency must be at least 1");
    expect_usage(parse_words({"--model=m.onnx", "--workload=add", "--concurrency=0"}),
                 "concurrency must be at least 1");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "65"}),
                 "--concurrency exceeds 64");
    expect_usage(parse_words({"--model=m.onnx", "--workload=add", "--concurrency=65"}),
                 "--concurrency exceeds 64");
}

TEST(ConcurrentBenchmarkCli, RejectsNegativeAndOverflowingIntegers) {
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency=-1"}),
                 "invalid value for --concurrency");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "-1"}),
                 "missing value for --concurrency");
    expect_usage(
        parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency=4294967296"}),
        "--concurrency overflows");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--iterations=-3"}),
                 "invalid value for --iterations");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1",
                              "--warmup", "-1"}),
                 "missing value for --warmup");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1",
                              "--iterations=4294967296"}),
                 "--iterations overflows");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1",
                              "--check-interval=0"}),
                 "correctness-check interval must be at least 1");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1",
                              "--iterations=0"}),
                 "measured iterations must be at least 1");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency=+2"}),
                 "invalid value for --concurrency");
}

TEST(ConcurrentBenchmarkCli, RejectsUnknownFlagsDuplicatesAndMissingValues) {
    expect_usage(parse_words({"--gpu"}), "unknown option '--gpu'");
    expect_usage(parse_words({"model.onnx"}), "unexpected argument 'model.onnx'");
    expect_usage(parse_words({"--model", "m.onnx", "--model", "other.onnx", "--workload", "add",
                              "--concurrency", "1"}),
                 "--model specified more than once");
    expect_usage(
        parse_words({"--model=m.onnx", "--model=other.onnx", "--workload=add", "--concurrency=1"}),
        "--model specified more than once");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--workload", "mnist",
                              "--concurrency", "1"}),
                 "--workload specified more than once");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1",
                              "--concurrency", "2"}),
                 "--concurrency specified more than once");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency=1",
                              "--warmup", "1", "--warmup=2"}),
                 "--warmup specified more than once");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "mnist", "--concurrency", "1",
                              "--fixture", "test_00003", "--fixture=test_00002"}),
                 "--fixture specified more than once");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "mnist", "--concurrency", "1",
                              "--fixture-dir", "a", "--fixture-dir=b"}),
                 "--fixture-dir specified more than once");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1",
                              "--format", "text", "--format=json"}),
                 "--format specified more than once");
    expect_usage(parse_words({"--model"}), "missing value for --model");
    expect_usage(parse_words({"--model="}), "empty model path");
    expect_usage(parse_words({"--workload"}), "missing value for --workload");
    expect_usage(parse_words({"--concurrency"}), "missing value for --concurrency");
    expect_usage(parse_words({"--concurrency="}), "invalid value for --concurrency");
    expect_usage(
        parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1", "--format"}),
        "missing value for --format");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1",
                              "--iterations"}),
                 "missing value for --iterations");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "mnist", "--concurrency", "1",
                              "--fixture"}),
                 "missing value for --fixture");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "mnist", "--concurrency", "1",
                              "--fixture-dir="}),
                 "empty fixture directory");
}

TEST(ConcurrentBenchmarkCli, RejectsUnsupportedFormatAndWorkload) {
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1",
                              "--format", "yaml"}),
                 "unknown format 'yaml'");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1",
                              "--format=yaml"}),
                 "unknown format 'yaml'");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "bert", "--concurrency", "1"}),
                 "unknown workload 'bert'");
    expect_usage(parse_words({"--model=m.onnx", "--workload=cuda", "--concurrency=1"}),
                 "unknown workload 'cuda'");
}

TEST(ConcurrentBenchmarkCli, ChecksMnistFixtureIdsAndRejectsThemForAdd) {
    const ConcurrentBenchmarkCliResult defaults =
        parse_words({"--model", "m.onnx", "--workload", "mnist", "--concurrency", "2"});
    EXPECT_EQ(defaults.action, ConcurrentBenchmarkCliAction::kRun);
    EXPECT_EQ(defaults.fixture_id, "test_00003");
    EXPECT_FALSE(defaults.fixture_specified);

    const ConcurrentBenchmarkCliResult selected = parse_words(
        {"--model=m.onnx", "--workload=mnist", "--concurrency=2", "--fixture=test_00002"});
    EXPECT_EQ(selected.action, ConcurrentBenchmarkCliAction::kRun);
    EXPECT_EQ(selected.fixture_id, "test_00002");
    EXPECT_TRUE(selected.fixture_specified);

    expect_usage(parse_words({"--model", "m.onnx", "--workload", "mnist", "--concurrency", "1",
                              "--fixture", ""}),
                 "empty fixture id");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "mnist", "--concurrency", "1",
                              "--fixture", "../secret"}),
                 "fixture id contains a path");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "mnist", "--concurrency", "1",
                              "--fixture=a/b"}),
                 "fixture id contains a path");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "mnist", "--concurrency", "1",
                              "--fixture", "not_a_fixture"}),
                 "unknown fixture id 'not_a_fixture'");
    expect_usage(parse_words({"--model", "m.onnx", "--workload", "add", "--concurrency", "1",
                              "--fixture", "test_00003"}),
                 "fixture flags apply only to the mnist workload");
    expect_usage(
        parse_words({"--model=m.onnx", "--workload=add", "--concurrency=1", "--fixture-dir=/tmp"}),
        "fixture flags apply only to the mnist workload");
}

TEST(ConcurrentBenchmarkCli, AcceptsBothFormsForTheRemainingFlags) {
    const ConcurrentBenchmarkCliResult result =
        parse_words({"--model", "m.onnx", "--workload", "mnist", "--concurrency", "3", "--warmup=0",
                     "--iterations", "5", "--check-interval=2", "--fixture", "synthetic",
                     "--fixture-dir", "fixtures", "--format=json"});
    EXPECT_EQ(result.action, ConcurrentBenchmarkCliAction::kRun);
    EXPECT_EQ(result.worker_count, 3U);
    EXPECT_EQ(result.warmup_iterations, 0U);
    EXPECT_EQ(result.measured_iterations, 5U);
    EXPECT_EQ(result.correctness_check_interval, 2U);
    EXPECT_EQ(result.fixture_id, "synthetic");
    EXPECT_EQ(result.fixture_directory, "fixtures");
    EXPECT_EQ(result.output_format, ConcurrentBenchmarkOutputFormat::kJson);
}

TEST_F(ConcurrentBenchmarkCommandTest, RunsAddAtConcurrencyOneAndTwo) {
    const ConcurrentBenchmarkCliResult one = run_words(add_command("1", "text"));
    const ConcurrentBenchmarkCliResult two = run_words(add_command("2", "text"));
    expect_text_report(one, "add", INFEREDGE_ADD_MODEL_PATH, 1, 0, 2, {});
    expect_text_report(two, "add", INFEREDGE_ADD_MODEL_PATH, 2, 0, 2, {});
}

TEST_F(ConcurrentBenchmarkCommandTest, RunsMnistAtConcurrencyOneAndTwo) {
    const std::vector<std::string> one = {"--model",       INFEREDGE_MNIST_MODEL_PATH,
                                          "--workload",    "mnist",
                                          "--concurrency", "1",
                                          "--warmup",      "0",
                                          "--iterations",  "1",
                                          "--fixture",     "test_00003"};
    std::vector<std::string> two = one;
    two[5] = "2";
    expect_text_report(run_words(one), "mnist", INFEREDGE_MNIST_MODEL_PATH, 1, 0, 1, "test_00003");
    expect_text_report(run_words(two), "mnist", INFEREDGE_MNIST_MODEL_PATH, 2, 0, 1, "test_00003");
}

TEST_F(ConcurrentBenchmarkCommandTest, RejectsModelAndMetadataFailures) {
    expect_failure(run_words({"--model", std::string{INFEREDGE_ADD_MODEL_PATH} + ".missing",
                              "--workload", "add", "--concurrency", "1", "--iterations", "1"}),
                   3, "model file is missing");
    expect_failure(run_words({"--model", INFEREDGE_INVALID_MODEL_PATH, "--workload", "add",
                              "--concurrency", "1", "--iterations", "1"}),
                   3, "not a valid ONNX model");
    expect_failure(run_words({"--model", INFEREDGE_INT64_MODEL_PATH, "--workload", "add",
                              "--concurrency", "1", "--iterations", "1"}),
                   3, "not float32");
    expect_failure(run_words({"--model", INFEREDGE_RANK2_MODEL_PATH, "--workload", "add",
                              "--concurrency", "1", "--warmup", "0", "--iterations", "1"}),
                   6, "does not match the add workload");
    expect_failure(run_words({"--model", INFEREDGE_ADD_MODEL_PATH, "--workload", "mnist",
                              "--concurrency", "1", "--warmup", "0", "--iterations", "1"}),
                   6, "does not match the mnist workload");
    expect_failure(run_words({"--model", INFEREDGE_MNIST_MODEL_PATH, "--workload", "add",
                              "--concurrency", "2", "--warmup", "0", "--iterations", "1"}),
                   6, "does not match the add workload");
}

std::vector<std::string> mnist_fixture_command(const std::filesystem::path& directory) {
    return {"--model",       INFEREDGE_MNIST_MODEL_PATH,
            "--workload",    "mnist",
            "--concurrency", "1",
            "--fixture",     "test_00003",
            "--fixture-dir", directory.string(),
            "--warmup",      "0",
            "--iterations",  "1",
            "--format",      "json"};
}

TEST_F(ConcurrentBenchmarkCommandTest, RejectsAnInvalidFixtureDirectory) {
    const TempTree tree;
    expect_failure(run_words(mnist_fixture_command(tree.root())), 4, "fixture file is missing");
    expect_failure(run_words(mnist_fixture_command(tree.root() / "absent")), 4,
                   "fixture directory is missing");
}

TEST_F(ConcurrentBenchmarkCommandTest, RejectsFixtureBytesClassAndFilesystemErrors) {
    const TempTree tree;
    const std::filesystem::path input = tree.root() / "inputs" / "test_00003.f32";
    {
        std::ofstream output{input, std::ios::binary};
        ASSERT_TRUE(output);
        const char bytes[4] = {};
        output.write(bytes, 4);
    }
    expect_failure(run_words(mnist_fixture_command(tree.root())), 4, "bytes, expected");

    std::vector<float> image(784, 0.0f);
    image[0] = std::numeric_limits<float>::quiet_NaN();
    write_f32(input, image);
    expect_failure(run_words(mnist_fixture_command(tree.root())), 4, "non-finite value");

    std::filesystem::remove(input);
    std::filesystem::remove_all(tree.root() / "inputs");
    std::filesystem::create_symlink("inputs", tree.root() / "inputs");
    const ConcurrentBenchmarkCliResult loop = run_words(mnist_fixture_command(tree.root()));
    expect_failure(loop, 4, "fixture filesystem error");
    EXPECT_NE(loop.message.find("Too many levels of symbolic links"), std::string::npos)
        << loop.message;
}

TEST_F(ConcurrentBenchmarkCommandTest, FixtureFailuresLeaveStdoutEmpty) {
    const TempTree tree;
    const CapturedProcess missing = run_concurrent(mnist_fixture_command(tree.root() / "absent"));
    expect_process_failure(missing, 4, "fixture directory is missing");
    EXPECT_EQ(missing.standard_error.find("schema_version"), std::string::npos);
    EXPECT_EQ(missing.standard_error.find("Correctness: passed"), std::string::npos);

    std::filesystem::remove_all(tree.root() / "inputs");
    std::filesystem::create_symlink("inputs", tree.root() / "inputs");
    const CapturedProcess loop = run_concurrent(mnist_fixture_command(tree.root()));
    expect_process_failure(loop, 4, "fixture filesystem error");
    EXPECT_NE(loop.standard_error.find("Too many levels of symbolic links"), std::string::npos)
        << loop.standard_error;
    EXPECT_EQ(loop.standard_output.find("InferEdge concurrent benchmark"), std::string::npos);
}

TEST_F(ConcurrentBenchmarkCommandTest, RejectsACorrectnessMismatchFromAFixture) {
    const TempTree tree;
    std::filesystem::copy_file(std::filesystem::path{INFEREDGE_MNIST_FIXTURE_DIR} / "inputs" /
                                   "test_00003.f32",
                               tree.root() / "inputs" / "test_00003.f32");
    std::vector<float> logits(10, 0.0f);
    logits[0] = 1.0e6f;
    write_f32(tree.root() / "onnx" / "test_00003.f32", logits);
    const ConcurrentBenchmarkCliResult result =
        run_words({"--model", INFEREDGE_MNIST_MODEL_PATH, "--workload", "mnist", "--concurrency",
                   "2", "--fixture", "test_00003", "--fixture-dir", tree.root().string(),
                   "--warmup", "0", "--iterations", "1"});
    expect_failure(result, 7, "outside tolerance");
    EXPECT_NE(result.message.find("worker "), std::string::npos) << result.message;
}

TEST_F(ConcurrentBenchmarkCommandTest, JsonSuccessAndFailureUseSeparateStreams) {
    const CapturedProcess missing =
        run_concurrent({"--model", std::string{INFEREDGE_ADD_MODEL_PATH} + ".missing", "--workload",
                        "add", "--concurrency", "1", "--format", "json", "--iterations", "1"});
    expect_process_failure(missing, 3, "model file is missing");
    EXPECT_EQ(missing.standard_error.find("schema_version"), std::string::npos);

    const CapturedProcess usage = run_concurrent({"--format", "yaml"});
    expect_process_failure(usage, 2, "unknown format 'yaml'");
    EXPECT_NE(usage.standard_error.find("Usage: inferedge-bench-concurrent"), std::string::npos);

    const CapturedProcess add = run_concurrent(add_command("1", "json"));
    EXPECT_EQ(add.exit_code, 0);
    EXPECT_TRUE(add.standard_error.empty()) << add.standard_error;
    ASSERT_FALSE(add.standard_output.empty());
    EXPECT_EQ(add.standard_output.front(), '{');
    EXPECT_EQ(add.standard_output.back(), '\n');
    EXPECT_EQ(add.standard_output.find("InferEdge concurrent benchmark"), std::string::npos);
    expect_python_json(add.standard_output, 1, 2, "null");

    const CapturedProcess mnist = run_concurrent(
        {"--model", INFEREDGE_MNIST_MODEL_PATH, "--workload", "mnist", "--concurrency=2",
         "--fixture=test_00003", "--warmup=0", "--iterations=1", "--format=json"});
    EXPECT_EQ(mnist.exit_code, 0);
    EXPECT_TRUE(mnist.standard_error.empty()) << mnist.standard_error;
    expect_python_json(mnist.standard_output, 2, 1, "test_00003");
}

TEST(ConcurrentBenchmarkProcess, SequentialCliContractStaysSeparate) {
    const CapturedProcess version = run_captured(INFEREDGE_BENCH_EXECUTABLE, {"--version"});
    EXPECT_EQ(version.exit_code, 0);
    EXPECT_EQ(version.standard_output,
              "inferedge-bench " + std::string{inferedge::version()} + "\n");
    EXPECT_TRUE(version.standard_error.empty()) << version.standard_error;

    const CapturedProcess help = run_captured(INFEREDGE_BENCH_EXECUTABLE, {"--help"});
    EXPECT_EQ(help.exit_code, 0);
    EXPECT_NE(help.standard_output.find("Usage: inferedge-bench [options]\n"), std::string::npos);
    EXPECT_EQ(help.standard_output.find("inferedge-bench-concurrent"), std::string::npos);
    EXPECT_EQ(help.standard_output.find("--concurrency"), std::string::npos)
        << help.standard_output;

    const CapturedProcess concurrent_help =
        run_captured(INFEREDGE_BENCH_CONCURRENT_EXECUTABLE, {"--help"});
    EXPECT_EQ(concurrent_help.exit_code, 0);
    EXPECT_NE(concurrent_help.standard_output.find("Usage: inferedge-bench-concurrent [options]\n"),
              std::string::npos);
    EXPECT_TRUE(concurrent_help.standard_error.empty()) << concurrent_help.standard_error;
}
