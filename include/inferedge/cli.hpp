#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace inferedge {

// What the program should do after arguments are parsed.
// Printing stays in main so this result can be tested without capturing stdout.
enum class CliAction {
    kHelp,
    kVersion,
    kError,
};

struct CliResult {
    CliAction action{};
    int exit_code{0};
    std::string message;
};

// args does not include the program name. An empty list means "show help".
CliResult parse_cli(const std::vector<std::string_view>& args);

} // namespace inferedge
