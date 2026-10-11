#include "inferedge/benchmark_cli.hpp"

#include "inferedge/benchmark_runner.hpp"
#include "inferedge/version.hpp"

#include <charconv>
#include <string>
#include <system_error>
#include <utility>

namespace inferedge {
namespace {

constexpr int kExitSuccess = 0;
constexpr int kExitUsage = 2;

enum class CountKind {
    kWarmup,
    kMeasured,
    kInterval,
};

std::string flag_name(CountKind kind) {
    switch (kind) {
    case CountKind::kWarmup:
        return "--warmup";
    case CountKind::kMeasured:
        return "--iterations";
    case CountKind::kInterval:
        return "--check-interval";
    }
    return "--iterations";
}

BenchmarkCliResult make_result(BenchmarkCliAction action, int exit_code, std::string message) {
    BenchmarkCliResult result;
    result.action = action;
    result.exit_code = exit_code;
    result.message = std::move(message);
    return result;
}

BenchmarkCliResult error_result(std::string message) {
    message += benchmark_help_message();
    return make_result(BenchmarkCliAction::kError, kExitUsage, std::move(message));
}

BenchmarkCliResult unexpected_argument(std::string_view argument) {
    std::string message = "error: unexpected argument '";
    message += argument;
    message += "'\n";
    return error_result(std::move(message));
}

bool is_help(std::string_view argument) {
    return argument == "--help" || argument == "-h";
}

bool is_version(std::string_view argument) {
    return argument == "--version" || argument == "-V";
}

bool is_option_token(std::string_view argument) {
    return argument.starts_with("-");
}

bool fixture_id_contains_path(std::string_view id) {
    return id.find('/') != std::string_view::npos || id.find('\\') != std::string_view::npos ||
           id.find("..") != std::string_view::npos;
}

std::string take_value(const std::vector<std::string_view>& args, std::size_t& index,
                       std::string_view flag, std::string_view attached_prefix,
                       std::string_view& value, bool& present) {
    const std::string_view argument = args[index];
    if (argument.starts_with(attached_prefix)) {
        value = argument.substr(attached_prefix.size());
        present = true;
        return {};
    }
    if (argument != flag) {
        present = false;
        return {};
    }
    if (index + 1 >= args.size() || is_option_token(args[index + 1])) {
        return "error: missing value for " + std::string{flag} + "\n";
    }
    ++index;
    value = args[index];
    present = true;
    return {};
}

std::string assign_once(bool& seen, std::string_view flag) {
    if (seen) {
        return "error: " + std::string{flag} + " specified more than once\n";
    }
    seen = true;
    return {};
}

enum class CountError {
    kNone,
    kInvalid,
    kOverflow,
    kExceedsMaximum,
    kZero,
};

CountError parse_count(std::string_view text, CountKind kind, std::uint32_t& value) {
    if (text.empty() || text.front() == '+' || text.front() == '-') {
        return CountError::kInvalid;
    }
    std::uint32_t parsed = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const std::from_chars_result result = std::from_chars(begin, end, parsed);
    if (result.ec == std::errc::result_out_of_range) {
        return CountError::kOverflow;
    }
    if (result.ec != std::errc{} || result.ptr != end) {
        return CountError::kInvalid;
    }
    if (parsed > kMaxBenchmarkIterations) {
        return CountError::kExceedsMaximum;
    }
    if (parsed == 0 && kind != CountKind::kWarmup) {
        return CountError::kZero;
    }
    value = parsed;
    return CountError::kNone;
}

std::string count_error_message(CountKind kind, CountError error) {
    const std::string name = flag_name(kind);
    switch (error) {
    case CountError::kNone:
        return {};
    case CountError::kInvalid:
        return "error: invalid value for " + name + "\n";
    case CountError::kOverflow:
        return "error: " + name + " overflows\n";
    case CountError::kExceedsMaximum:
        return "error: " + name + " exceeds 100000\n";
    case CountError::kZero:
        if (kind == CountKind::kMeasured) {
            return "error: measured iterations must be at least 1\n";
        }
        return "error: correctness-check interval must be at least 1\n";
    }
    return "error: invalid value for " + name + "\n";
}

struct Taken {
    std::string error;
    bool consumed{false};
};

Taken take_count(const std::vector<std::string_view>& args, std::size_t& index,
                 std::string_view flag, CountKind kind, bool& seen, std::uint32_t& destination) {
    std::string_view text;
    bool present = false;
    const std::string missing =
        take_value(args, index, flag, std::string{flag} + "=", text, present);
    if (!missing.empty()) {
        return Taken{missing, true};
    }
    if (!present) {
        return {};
    }
    const std::string duplicate = assign_once(seen, flag);
    if (!duplicate.empty()) {
        return Taken{duplicate, true};
    }
    std::uint32_t parsed = 0;
    const CountError error = parse_count(text, kind, parsed);
    if (error != CountError::kNone) {
        return Taken{count_error_message(kind, error), true};
    }
    destination = parsed;
    return Taken{{}, true};
}

std::string validate_fixture(const BenchmarkCliResult& parsed) {
    if (parsed.workload == "add") {
        if (parsed.fixture_specified || parsed.fixture_directory_specified) {
            return "error: fixture flags apply only to the mnist workload\n";
        }
        return {};
    }
    if (parsed.workload != "mnist") {
        return "error: unknown workload '" + parsed.workload + "'\n";
    }
    if (!parsed.fixture_specified) {
        return {};
    }
    if (parsed.fixture_id.empty()) {
        return "error: empty fixture id\n";
    }
    if (fixture_id_contains_path(parsed.fixture_id)) {
        return "error: fixture id contains a path\n";
    }
    if (find_mnist_fixture(parsed.fixture_id) == nullptr) {
        return "error: unknown fixture id '" + parsed.fixture_id + "'\n";
    }
    return {};
}

} // namespace

std::string benchmark_help_message() {
    std::string message;
    message += "Usage: inferedge-bench [options]\n";
    message += "\n";
    message += "Measure one loaded InferenceEngine on the calling thread.\n";
    message += "\n";
    message += "Options:\n";
    message += "  -h, --help                 Show this help and exit\n";
    message += "  -V, --version              Show the version and exit\n";
    message += "  --model <path>             ONNX model to benchmark\n";
    message += "  --workload <name>          add or mnist\n";
    message += "  --warmup <count>           Unmeasured calls before timing (default 10)\n";
    message += "  --iterations <count>       Measured calls (default 100)\n";
    message += "  --check-interval <count>   Numeric check interval (default 1)\n";
    message += "  --fixture <id>             MNIST fixture id (default test_00003)\n";
    message += "  --fixture-dir <path>       MNIST fixture directory\n";
    message += "\n";
    message += "With no arguments, this help is shown. --help and --version cannot be\n";
    message += "combined with other arguments. --fixture and --fixture-dir apply only to\n";
    message += "the mnist workload. The default fixture directory is the source-tree path\n";
    message += "chosen when CMake configured the build. A copied executable may need\n";
    message += "--fixture-dir. Warmup may be 0. Measured iterations and the check interval\n";
    message += "must be from 1 through 100000.\n";
    return message;
}

std::string benchmark_version_message() {
    std::string message = "inferedge-bench ";
    message += version();
    message += "\n";
    return message;
}

BenchmarkCliResult parse_benchmark_cli(const std::vector<std::string_view>& args) {
    if (args.empty()) {
        return make_result(BenchmarkCliAction::kHelp, kExitSuccess, benchmark_help_message());
    }
    if (is_help(args.front()) || is_version(args.front())) {
        if (args.size() > 1) {
            return unexpected_argument(args[1]);
        }
        if (is_help(args.front())) {
            return make_result(BenchmarkCliAction::kHelp, kExitSuccess, benchmark_help_message());
        }
        return make_result(BenchmarkCliAction::kVersion, kExitSuccess, benchmark_version_message());
    }

    BenchmarkCliResult parsed;
    parsed.action = BenchmarkCliAction::kRun;
    parsed.exit_code = kExitSuccess;
    bool saw_model = false;
    bool saw_workload = false;
    bool saw_warmup = false;
    bool saw_iterations = false;
    bool saw_interval = false;

    for (std::size_t index = 0; index < args.size(); ++index) {
        const std::string_view argument = args[index];
        if (is_help(argument) || is_version(argument)) {
            return unexpected_argument(argument);
        }

        std::string_view text;
        bool present = false;
        const std::string model_error =
            take_value(args, index, "--model", "--model=", text, present);
        if (!model_error.empty()) {
            return error_result(model_error);
        }
        if (present) {
            const std::string duplicate = assign_once(saw_model, "--model");
            if (!duplicate.empty()) {
                return error_result(duplicate);
            }
            if (text.empty()) {
                return error_result("error: empty model path\n");
            }
            parsed.model_path.assign(text);
            continue;
        }

        const std::string workload_error =
            take_value(args, index, "--workload", "--workload=", text, present);
        if (!workload_error.empty()) {
            return error_result(workload_error);
        }
        if (present) {
            const std::string duplicate = assign_once(saw_workload, "--workload");
            if (!duplicate.empty()) {
                return error_result(duplicate);
            }
            parsed.workload.assign(text);
            continue;
        }

        const Taken warmup = take_count(args, index, "--warmup", CountKind::kWarmup, saw_warmup,
                                        parsed.warmup_iterations);
        if (!warmup.error.empty()) {
            return error_result(warmup.error);
        }
        if (warmup.consumed) {
            continue;
        }

        const Taken iterations = take_count(args, index, "--iterations", CountKind::kMeasured,
                                            saw_iterations, parsed.measured_iterations);
        if (!iterations.error.empty()) {
            return error_result(iterations.error);
        }
        if (iterations.consumed) {
            continue;
        }

        const Taken interval = take_count(args, index, "--check-interval", CountKind::kInterval,
                                          saw_interval, parsed.correctness_check_interval);
        if (!interval.error.empty()) {
            return error_result(interval.error);
        }
        if (interval.consumed) {
            continue;
        }

        const std::string fixture_dir_error =
            take_value(args, index, "--fixture-dir", "--fixture-dir=", text, present);
        if (!fixture_dir_error.empty()) {
            return error_result(fixture_dir_error);
        }
        if (present) {
            const std::string duplicate =
                assign_once(parsed.fixture_directory_specified, "--fixture-dir");
            if (!duplicate.empty()) {
                return error_result(duplicate);
            }
            if (text.empty()) {
                return error_result("error: empty fixture directory\n");
            }
            parsed.fixture_directory.assign(text);
            continue;
        }

        const std::string fixture_error =
            take_value(args, index, "--fixture", "--fixture=", text, present);
        if (!fixture_error.empty()) {
            return error_result(fixture_error);
        }
        if (present) {
            const std::string duplicate = assign_once(parsed.fixture_specified, "--fixture");
            if (!duplicate.empty()) {
                return error_result(duplicate);
            }
            parsed.fixture_id.assign(text);
            continue;
        }

        if (is_option_token(argument)) {
            std::string message = "error: unknown option '";
            message += argument;
            message += "'\n";
            return error_result(std::move(message));
        }
        return unexpected_argument(argument);
    }

    if (!saw_model) {
        return error_result("error: missing --model\n");
    }
    if (!saw_workload) {
        return error_result("error: missing --workload\n");
    }
    const std::string fixture_problem = validate_fixture(parsed);
    if (!fixture_problem.empty()) {
        return error_result(fixture_problem);
    }
    return parsed;
}

} // namespace inferedge
