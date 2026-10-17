#pragma once

#include "inferedge/model_package_manifest.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace inferedge {

enum class GenerateCliAction {
    kHelp,
    kVersion,
    kError,
    kGenerate,
};

inline constexpr std::size_t kGenerateMaxOptionLength = 4096;
inline constexpr std::size_t kGenerateMaxPromptLength = kGenerativeMaxPromptBytes;
inline constexpr int kGenerateExitSuccess = 0;
inline constexpr int kGenerateExitUsage = 2;
inline constexpr int kGenerateExitFailure = 1;

struct GenerateCliResult {
    GenerateCliAction action{};
    int exit_code{0};
    std::string message;
    std::string model_package;
    std::string prompt;
    std::string prompt_file;
    std::uint32_t max_new_tokens{0};
    std::uint32_t threads{0};
};

// --prompt may appear in process listings and shell history. Prefer --prompt-file
// for sensitive text. Empty and ASCII-whitespace-only prompts (space, tab, LF)
// are rejected. Unicode whitespace is not treated as empty. Stdin is not read.
// InferEdge does not persist or network prompt content.
GenerateCliResult parse_generate_cli(const std::vector<std::string_view>& args);

} // namespace inferedge
