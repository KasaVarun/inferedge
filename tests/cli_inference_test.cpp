#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <spawn.h>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

// posix_spawn runs the real inferedge binary without a shell.

extern "C" char** environ;

namespace {

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

CapturedProcess run_inferedge(const std::vector<std::string>& args) {
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
    arguments.emplace_back(INFEREDGE_EXECUTABLE);
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
        posix_spawn(&pid, INFEREDGE_EXECUTABLE, &actions, nullptr, argv.data(), envp.data());
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
        ADD_FAILURE() << "inferedge did not exit normally";
    }
    return captured;
}

std::string add_stdout(std::string_view values) {
    std::string text;
    text += "Model: ";
    text += INFEREDGE_ADD_MODEL_PATH;
    text += "\nBackend: ONNX Runtime CPU\n\nOutput C:\n";
    text += values;
    text += "\n";
    return text;
}

void expect_failure(const CapturedProcess& captured, int exit_code, std::string_view fragment) {
    EXPECT_EQ(captured.exit_code, exit_code);
    EXPECT_TRUE(captured.standard_output.empty());
    ASSERT_FALSE(captured.standard_error.empty());
    EXPECT_EQ(captured.standard_error.compare(0, 7, "error: "), 0);
    EXPECT_NE(captured.standard_error.find(fragment), std::string::npos);
}

} // namespace

TEST(CliProcess, HelpAndNoArgumentsWriteStdout) {
    const CapturedProcess help = run_inferedge({"--help"});
    const CapturedProcess short_help = run_inferedge({"-h"});
    const CapturedProcess no_arguments = run_inferedge({});
    EXPECT_EQ(help.exit_code, 0);
    EXPECT_TRUE(help.standard_error.empty());
    EXPECT_NE(help.standard_output.find("Usage: inferedge"), std::string::npos);
    EXPECT_EQ(help.standard_output, short_help.standard_output);
    EXPECT_EQ(no_arguments.exit_code, 0);
    EXPECT_TRUE(no_arguments.standard_error.empty());
    EXPECT_EQ(no_arguments.standard_output, help.standard_output);
}

TEST(CliProcess, VersionWritesStdout) {
    const CapturedProcess version = run_inferedge({"--version"});
    const CapturedProcess short_version = run_inferedge({"-V"});
    EXPECT_EQ(version.exit_code, 0);
    EXPECT_TRUE(version.standard_error.empty());
    EXPECT_EQ(version.standard_output, "inferedge 0.1.0\n");
    EXPECT_EQ(short_version.standard_output, version.standard_output);
    EXPECT_EQ(short_version.exit_code, 0);
}

TEST(CliProcess, AddInferencePrintsEngineOutput) {
    const CapturedProcess captured = run_inferedge(
        {"--model", INFEREDGE_ADD_MODEL_PATH, "--input", "A=1,2", "--input", "B=3,4"});
    EXPECT_EQ(captured.exit_code, 0);
    EXPECT_TRUE(captured.standard_error.empty());
    EXPECT_EQ(captured.standard_output, add_stdout("[4, 6]"));
}

TEST(CliProcess, ReorderedInputsMatch) {
    const CapturedProcess captured = run_inferedge(
        {"--model", INFEREDGE_ADD_MODEL_PATH, "--input", "B=3,4", "--input", "A=1,2"});
    EXPECT_EQ(captured.exit_code, 0);
    EXPECT_TRUE(captured.standard_error.empty());
    EXPECT_EQ(captured.standard_output, add_stdout("[4, 6]"));
}

TEST(CliProcess, NegativeAndZeroValues) {
    const CapturedProcess captured =
        run_inferedge({"--model=" + std::string{INFEREDGE_ADD_MODEL_PATH}, "--input=A=-1.5,0",
                       "--input=B=2.5,-4"});
    EXPECT_EQ(captured.exit_code, 0);
    EXPECT_TRUE(captured.standard_error.empty());
    EXPECT_EQ(captured.standard_output, add_stdout("[1, -4]"));
}

TEST(CliProcess, UsageErrorsWriteStderr) {
    const CapturedProcess missing_model = run_inferedge({"--input", "A=1,2"});
    const CapturedProcess duplicate = run_inferedge(
        {"--model", INFEREDGE_ADD_MODEL_PATH, "--input", "A=1,2", "--input", "A=3,4"});
    const CapturedProcess malformed = run_inferedge(
        {"--model", INFEREDGE_ADD_MODEL_PATH, "--input", "A=1,xyz", "--input", "B=3,4"});
    const CapturedProcess non_finite = run_inferedge(
        {"--model", INFEREDGE_ADD_MODEL_PATH, "--input", "A=inf,1", "--input", "B=3,4"});
    const CapturedProcess empty_values =
        run_inferedge({"--model", INFEREDGE_ADD_MODEL_PATH, "--input", "A="});
    const CapturedProcess repeated_comma = run_inferedge(
        {"--model", INFEREDGE_ADD_MODEL_PATH, "--input", "A=1,,2", "--input", "B=3,4"});
    const CapturedProcess unknown = run_inferedge({"--scheduler"});

    expect_failure(missing_model, 2, "missing --model");
    expect_failure(duplicate, 2, "duplicate input name: A");
    expect_failure(malformed, 2, "invalid floating-point value 'xyz'");
    expect_failure(non_finite, 2, "non-finite");
    expect_failure(empty_values, 2, "missing input values");
    expect_failure(repeated_comma, 2, "empty value");
    expect_failure(unknown, 2, "unknown option '--scheduler'");
}

TEST(CliProcess, ModelFailuresWriteStderr) {
    const CapturedProcess missing_file = run_inferedge(
        {"--model", std::string{INFEREDGE_ADD_MODEL_PATH} + ".missing", "--input", "A=1,2"});
    const CapturedProcess invalid_file =
        run_inferedge({"--model", INFEREDGE_INVALID_MODEL_PATH, "--input", "A=1,2"});
    expect_failure(missing_file, 3, "model file is missing");
    expect_failure(invalid_file, 3, "not a valid ONNX model");
}

TEST(CliProcess, InputFailuresComeFromTheEngine) {
    const CapturedProcess missing_input =
        run_inferedge({"--model", INFEREDGE_ADD_MODEL_PATH, "--input", "A=1,2"});
    const CapturedProcess unexpected =
        run_inferedge({"--model", INFEREDGE_ADD_MODEL_PATH, "--input", "A=1,2", "--input", "B=3,4",
                       "--input", "D=5,6"});
    const CapturedProcess count = run_inferedge(
        {"--model", INFEREDGE_ADD_MODEL_PATH, "--input", "A=1,2,3", "--input", "B=3,4"});
    expect_failure(missing_input, 4, "missing input name: B");
    expect_failure(unexpected, 4, "unexpected input name: D");
    expect_failure(count, 4, "has 3 values, shape requires 2");
}

TEST(CliProcess, UnsupportedShapeIsRejected) {
    const CapturedProcess captured =
        run_inferedge({"--model", INFEREDGE_RANK2_MODEL_PATH, "--input", "M=1,2"});
    expect_failure(captured, 4, "unsupported input shape for M: rank 2");
}
