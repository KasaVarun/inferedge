#pragma once

#include <cstddef>
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
    kRun,
};

// Owned parsed input. execute_inference borrows values for the duration of run().
struct NamedFloatInput {
    std::string name;
    std::vector<float> values;
};

// One invocation may name this many inputs and this many finite floats.
// Each numeric token is at most kMaxCharsPerNumber characters. The values
// text of one --input is at most kMaxInputValuesLength characters, which
// fits one maximum-length token and a comma for every allowed value.
inline constexpr std::size_t kMaxNamedInputs = 64;
inline constexpr std::size_t kMaxFloatsPerInput = 1'048'576;
inline constexpr std::size_t kMaxTotalFloats = 1'048'576;
inline constexpr std::size_t kMaxCharsPerNumber = 64;
inline constexpr std::size_t kMaxInputNameLength = 1024;
inline constexpr std::size_t kMaxInputValuesLength = kMaxFloatsPerInput * (kMaxCharsPerNumber + 1);

struct CliResult {
    CliAction action{};
    int exit_code{0};
    std::string message;
    std::string model_path;
    std::vector<NamedFloatInput> inputs;
};

// args does not include the program name. An empty list means "show help".
CliResult parse_cli(const std::vector<std::string_view>& args);

} // namespace inferedge
