#include "inferedge/benchmark_cli.hpp"

#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <poll.h>
#include <spawn.h>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

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

using inferedge::BenchmarkCliAction;
using inferedge::BenchmarkCliResult;
using inferedge::execute_benchmark;
using inferedge::parse_benchmark_cli;

class BenchmarkCommandTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

class TempTree {
public:
    TempTree() : root_(std::filesystem::temp_directory_path() / "inferedge-bench-fixture") {
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

BenchmarkCliResult run_words(const std::vector<std::string>& args) {
    std::vector<std::string_view> views;
    views.reserve(args.size());
    for (const std::string& arg : args) {
        views.emplace_back(arg);
    }
    const BenchmarkCliResult parsed = parse_benchmark_cli(views);
    if (parsed.action != BenchmarkCliAction::kRun) {
        return parsed;
    }
    return execute_benchmark(parsed);
}

void write_f32(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream output{path, std::ios::binary};
    if (!output) {
        ADD_FAILURE() << path;
        return;
    }
    for (float value : values) {
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

void expect_labels(const BenchmarkCliResult& result, std::string_view workload,
                   std::string_view model, std::uint32_t warmup, std::uint32_t measured,
                   std::string_view fixture) {
    EXPECT_EQ(result.action, BenchmarkCliAction::kRun);
    EXPECT_EQ(result.exit_code, 0);
    const std::string& text = result.message;
    EXPECT_NE(text.find("InferEdge benchmark\n"), std::string::npos) << text;
    EXPECT_NE(text.find("Model: " + std::string{model}), std::string::npos) << text;
    EXPECT_NE(text.find("Workload: " + std::string{workload}), std::string::npos) << text;
    EXPECT_NE(text.find("Backend: ONNX Runtime CPU\n"), std::string::npos) << text;
    EXPECT_NE(text.find("one benchmark calling thread"), std::string::npos) << text;
    EXPECT_NE(text.find("one ONNX intra-op thread"), std::string::npos) << text;
    EXPECT_NE(text.find("graph optimizations disabled"), std::string::npos) << text;
    EXPECT_NE(text.find("Warmup iterations: " + std::to_string(warmup) + "\n"), std::string::npos);
    EXPECT_NE(text.find("Measured iterations: " + std::to_string(measured) + "\n"),
              std::string::npos);
    EXPECT_NE(text.find("Latency:\n"), std::string::npos);
    EXPECT_NE(text.find("  minimum: "), std::string::npos);
    EXPECT_NE(text.find(" ns ("), std::string::npos);
    EXPECT_NE(text.find(" us)"), std::string::npos);
    EXPECT_NE(text.find("  mean: "), std::string::npos);
    EXPECT_NE(text.find("  p50: "), std::string::npos);
    EXPECT_NE(text.find("  p95: "), std::string::npos);
    EXPECT_NE(text.find("  p99: "), std::string::npos);
    EXPECT_NE(text.find("  maximum: "), std::string::npos);
    EXPECT_NE(text.find("  population standard deviation: "), std::string::npos);
    EXPECT_NE(text.find("  active-call: "), std::string::npos);
    EXPECT_NE(text.find("  observed sequential: "), std::string::npos);
    EXPECT_NE(text.find("requests/second"), std::string::npos);
    EXPECT_NE(text.find("Correctness: passed\n"), std::string::npos);
    EXPECT_NE(text.find("server capacity"), std::string::npos);
    EXPECT_EQ(text.find("GPU"), std::string::npos) << text;
    if (fixture.empty()) {
        EXPECT_EQ(text.find("Fixture:"), std::string::npos) << text;
    } else {
        EXPECT_NE(text.find("Fixture: " + std::string{fixture} + "\n"), std::string::npos) << text;
    }
}

void expect_failure(const BenchmarkCliResult& result, int exit_code, std::string_view fragment) {
    EXPECT_EQ(result.action, BenchmarkCliAction::kError);
    EXPECT_EQ(result.exit_code, exit_code);
    EXPECT_EQ(result.message.compare(0, 7, "error: "), 0) << result.message;
    EXPECT_NE(result.message.find(fragment), std::string::npos) << result.message;
    EXPECT_EQ(result.message.find("Correctness: passed"), std::string::npos) << result.message;
    EXPECT_EQ(result.message.find("InferEdge benchmark"), std::string::npos) << result.message;
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

CapturedProcess run_bench(const std::vector<std::string>& args) {
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
    arguments.emplace_back(INFEREDGE_BENCH_EXECUTABLE);
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
    const int spawned =
        posix_spawn(&pid, INFEREDGE_BENCH_EXECUTABLE, &actions, nullptr, argv.data(), envp.data());
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
        ADD_FAILURE() << "inferedge-bench did not exit normally";
    }
    return captured;
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
}

} // namespace

TEST_F(BenchmarkCommandTest, RejectsAMissingModel) {
    const BenchmarkCliResult result =
        run_words({"--model", std::string{INFEREDGE_ADD_MODEL_PATH} + ".missing", "--workload",
                   "add", "--warmup", "0", "--iterations", "1"});
    expect_failure(result, 3, "model file is missing");
}

TEST_F(BenchmarkCommandTest, RejectsAnInvalidModelFile) {
    const BenchmarkCliResult result = run_words(
        {"--model", INFEREDGE_INVALID_MODEL_PATH, "--workload", "add", "--iterations", "1"});
    expect_failure(result, 3, "not a valid ONNX model");
}

TEST_F(BenchmarkCommandTest, RejectsAnUnsupportedTensorTypeAtLoad) {
    const BenchmarkCliResult result = run_words(
        {"--model", INFEREDGE_INT64_MODEL_PATH, "--workload", "add", "--iterations", "1"});
    expect_failure(result, 3, "not float32");
}

TEST_F(BenchmarkCommandTest, RejectsIncompatibleModelMetadata) {
    const BenchmarkCliResult rank2 = run_words({"--model", INFEREDGE_RANK2_MODEL_PATH, "--workload",
                                                "add", "--warmup", "0", "--iterations", "1"});
    const BenchmarkCliResult add_as_mnist =
        run_words({"--model", INFEREDGE_ADD_MODEL_PATH, "--workload", "mnist", "--warmup", "0",
                   "--iterations", "1"});
    const BenchmarkCliResult mnist_as_add =
        run_words({"--model", INFEREDGE_MNIST_MODEL_PATH, "--workload", "add", "--warmup", "0",
                   "--iterations", "1"});
    expect_failure(rank2, 6, "does not match the add workload");
    expect_failure(add_as_mnist, 6, "does not match the mnist workload");
    expect_failure(mnist_as_add, 6, "does not match the add workload");
}

TEST_F(BenchmarkCommandTest, RejectsAMissingFixture) {
    const TempTree tree;
    const BenchmarkCliResult result =
        run_words({"--model", INFEREDGE_MNIST_MODEL_PATH, "--workload", "mnist", "--fixture",
                   "test_00003", "--fixture-dir", tree.root().string(), "--iterations", "1"});
    expect_failure(result, 4, "fixture file is missing");
}

TEST_F(BenchmarkCommandTest, RejectsATruncatedAndOversizedFixture) {
    const TempTree truncated;
    {
        std::ofstream output{truncated.root() / "inputs" / "test_00003.f32", std::ios::binary};
        output << "xxxx";
    }
    const BenchmarkCliResult short_file =
        run_words({"--model", INFEREDGE_MNIST_MODEL_PATH, "--workload", "mnist", "--fixture",
                   "test_00003", "--fixture-dir", truncated.root().string(), "--iterations", "1"});
    expect_failure(short_file, 4, "bytes, expected");

    const TempTree oversized;
    write_f32(oversized.root() / "inputs" / "test_00003.f32", std::vector<float>(785, 0.0f));
    const BenchmarkCliResult long_file =
        run_words({"--model", INFEREDGE_MNIST_MODEL_PATH, "--workload", "mnist", "--fixture",
                   "test_00003", "--fixture-dir", oversized.root().string(), "--iterations", "1"});
    expect_failure(long_file, 4, "bytes, expected");
}

TEST_F(BenchmarkCommandTest, RejectsANonFiniteFixtureAndAWrongArgmax) {
    const TempTree non_finite;
    std::vector<float> image(784, 0.0f);
    image[0] = std::bit_cast<float>(std::uint32_t{0x7fc00000U});
    write_f32(non_finite.root() / "inputs" / "test_00003.f32", image);
    const BenchmarkCliResult nan_file =
        run_words({"--model", INFEREDGE_MNIST_MODEL_PATH, "--workload", "mnist", "--fixture-dir",
                   non_finite.root().string(), "--iterations", "1"});
    expect_failure(nan_file, 4, "non-finite");

    const TempTree wrong_class;
    write_f32(wrong_class.root() / "inputs" / "test_00003.f32", std::vector<float>(784, 0.0f));
    std::vector<float> logits(10, 0.0f);
    logits[1] = 1.0f;
    write_f32(wrong_class.root() / "onnx" / "test_00003.f32", logits);
    const BenchmarkCliResult argmax =
        run_words({"--model", INFEREDGE_MNIST_MODEL_PATH, "--workload", "mnist", "--fixture-dir",
                   wrong_class.root().string(), "--iterations", "1"});
    expect_failure(argmax, 4, "catalog expects 0");
}

TEST_F(BenchmarkCommandTest, MeasuresTheAddModel) {
    const BenchmarkCliResult result =
        run_words({"--model", INFEREDGE_ADD_MODEL_PATH, "--workload", "add", "--warmup", "1",
                   "--iterations", "4", "--check-interval", "1"});
    expect_labels(result, "add", INFEREDGE_ADD_MODEL_PATH, 1, 4, "");
}

TEST_F(BenchmarkCommandTest, MeasuresTheMnistFixture) {
    const BenchmarkCliResult result =
        run_words({"--model", INFEREDGE_MNIST_MODEL_PATH, "--workload", "mnist", "--fixture",
                   "test_00003", "--warmup", "1", "--iterations", "2"});
    expect_labels(result, "mnist", INFEREDGE_MNIST_MODEL_PATH, 1, 2, "test_00003");
}

TEST_F(BenchmarkCommandTest, AllowsZeroWarmupAndAWideCheckInterval) {
    const BenchmarkCliResult result =
        run_words({"--model", INFEREDGE_ADD_MODEL_PATH, "--workload", "add", "--warmup", "0",
                   "--iterations", "2", "--check-interval", "1000"});
    expect_labels(result, "add", INFEREDGE_ADD_MODEL_PATH, 0, 2, "");
    EXPECT_NE(result.message.find("Correctness-check interval: 1000\n"), std::string::npos);
}

TEST(BenchmarkProcess, HelpAndVersionWriteStdout) {
    const CapturedProcess help = run_bench({"--help"});
    const CapturedProcess no_arguments = run_bench({});
    const CapturedProcess version = run_bench({"--version"});
    EXPECT_EQ(help.exit_code, 0);
    EXPECT_TRUE(help.standard_error.empty()) << help.standard_error;
    EXPECT_NE(help.standard_output.find("Usage: inferedge-bench"), std::string::npos);
    EXPECT_EQ(no_arguments.exit_code, 0);
    EXPECT_TRUE(no_arguments.standard_error.empty());
    EXPECT_EQ(no_arguments.standard_output, help.standard_output);
    EXPECT_EQ(version.exit_code, 0);
    EXPECT_TRUE(version.standard_error.empty()) << version.standard_error;
    EXPECT_EQ(version.standard_output, "inferedge-bench 0.1.0\n");
}

TEST(BenchmarkProcess, UsageAndModelErrorsWriteStderr) {
    const CapturedProcess usage = run_bench({"--workload", "add"});
    const CapturedProcess missing =
        run_bench({"--model", std::string{INFEREDGE_ADD_MODEL_PATH} + ".missing", "--workload",
                   "add", "--iterations", "1"});
    expect_process_failure(usage, 2, "missing --model");
    EXPECT_NE(usage.standard_error.find("Usage: inferedge-bench"), std::string::npos);
    expect_process_failure(missing, 3, "model file is missing");
    EXPECT_EQ(missing.standard_error.find("Usage: inferedge-bench"), std::string::npos);
}

TEST(BenchmarkProcess, AddAndMnistSucceedOnStdout) {
    const CapturedProcess add = run_bench({"--model", INFEREDGE_ADD_MODEL_PATH, "--workload", "add",
                                           "--warmup", "0", "--iterations", "2"});
    const CapturedProcess mnist =
        run_bench({"--model", INFEREDGE_MNIST_MODEL_PATH, "--workload", "mnist", "--fixture",
                   "test_00003", "--warmup", "0", "--iterations", "2"});
    EXPECT_EQ(add.exit_code, 0);
    EXPECT_TRUE(add.standard_error.empty()) << add.standard_error;
    EXPECT_NE(add.standard_output.find("Workload: add\n"), std::string::npos);
    EXPECT_NE(add.standard_output.find("Correctness: passed\n"), std::string::npos);
    EXPECT_NE(add.standard_output.find("requests/second"), std::string::npos);
    EXPECT_EQ(mnist.exit_code, 0);
    EXPECT_TRUE(mnist.standard_error.empty()) << mnist.standard_error;
    EXPECT_NE(mnist.standard_output.find("Fixture: test_00003\n"), std::string::npos);
    EXPECT_NE(mnist.standard_output.find("Correctness: passed\n"), std::string::npos);
    EXPECT_EQ(mnist.standard_output.find("GPU"), std::string::npos);
}
