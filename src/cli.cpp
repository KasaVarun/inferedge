#include "inferedge/cli.hpp"

#include "inferedge/version.hpp"

#include <string>
#include <utility>

namespace inferedge {
namespace {

constexpr int kExitSuccess = 0;
constexpr int kExitUsage = 2;

std::string help_message() {
    std::string message;
    message += "Usage: ";
    message += project_name();
    message += " [options]\n";
    message += "\n";
    message += "InferEdge is a high-performance C++ inference runtime.\n";
    message += "This executable is the Phase 0 command-line foundation.\n";
    message += "\n";
    message += "Options:\n";
    message += "  -h, --help       Show this help and exit\n";
    message += "  -V, --version    Show the version and exit\n";
    return message;
}

std::string version_message() {
    std::string message;
    message += project_name();
    message += " ";
    message += version();
    message += "\n";
    return message;
}

CliResult error_result(std::string message) {
    return CliResult{CliAction::kError, kExitUsage, std::move(message)};
}

bool is_help(std::string_view argument) {
    return argument == "--help" || argument == "-h";
}

bool is_version(std::string_view argument) {
    return argument == "--version" || argument == "-V";
}

} // namespace

CliResult parse_cli(const std::vector<std::string_view>& args) {
    if (args.empty()) {
        return CliResult{CliAction::kHelp, kExitSuccess, help_message()};
    }

    const std::string_view option = args.front();
    if (args.size() > 1) {
        std::string message = "error: unexpected argument '";
        message += args[1];
        message += "'\n";
        message += help_message();
        return error_result(std::move(message));
    }

    if (is_help(option)) {
        return CliResult{CliAction::kHelp, kExitSuccess, help_message()};
    }
    if (is_version(option)) {
        return CliResult{CliAction::kVersion, kExitSuccess, version_message()};
    }

    std::string message = "error: unknown option '";
    message += option;
    message += "'\n";
    message += help_message();
    return error_result(std::move(message));
}

} // namespace inferedge
