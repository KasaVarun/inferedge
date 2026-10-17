#include "inferedge/generate_cli.hpp"

#include "inferedge/version.hpp"

#include <charconv>
#include <system_error>
#include <utility>

namespace inferedge {
namespace {

GenerateCliResult make_result(GenerateCliAction action, int exit_code, std::string message) {
    GenerateCliResult result;
    result.action = action;
    result.exit_code = exit_code;
    result.message = std::move(message);
    return result;
}

std::string help_message() {
    std::string message;
    message += "Usage: inferedge-generate [options]\n";
    message += "\n";
    message += "Correctness-only generative CLI. Default builds do not include this executable.\n";
    message += "No model is downloaded. Provide a local package directory.\n";
    message += "\n";
    message += "Options:\n";
    message += "  -h, --help                         Show this help and exit\n";
    message += "  -V, --version                      Show the version and exit\n";
    message += "  --model-package <absolute-dir>     Canonical model-package directory\n";
    message += "  --prompt <text>                    Prompt text (visible in process listings)\n";
    message += "  --prompt-file <absolute-file>      Recommended for sensitive prompts\n";
    message += "  --max-new-tokens <integer>         Maximum generated tokens\n";
    message += "  --threads <integer>                CPU thread count\n";
    message += "\n";
    message += "Stdin is not read. Empty and ASCII-whitespace-only prompts (space, tab, LF)\n";
    message += "are rejected. Unicode whitespace is not treated as empty. --prompt may appear\n";
    message += "in process listings and shell history. InferEdge does not persist or network\n";
    message += "prompts. --help and --version cannot be combined with other arguments.\n";
    return message;
}

std::string version_message() {
    std::string message = "inferedge-generate ";
    message += version();
    message += "\n";
    return message;
}

GenerateCliResult error_result(std::string message) {
    message += help_message();
    return make_result(GenerateCliAction::kError, kGenerateExitUsage, std::move(message));
}

bool parse_u32_token(std::string_view token, std::uint32_t& value) {
    if (token.empty() || token.size() > 10 || token[0] == '+' || token[0] == '-') {
        return false;
    }
    std::uint32_t parsed = 0;
    const auto result = std::from_chars(token.data(), token.data() + token.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
        return false;
    }
    value = parsed;
    return true;
}

} // namespace

GenerateCliResult parse_generate_cli(const std::vector<std::string_view>& args) {
    if (args.empty()) {
        return make_result(GenerateCliAction::kHelp, kGenerateExitSuccess, help_message());
    }
    if (args.size() == 1 && (args[0] == "--help" || args[0] == "-h")) {
        return make_result(GenerateCliAction::kHelp, kGenerateExitSuccess, help_message());
    }
    if (args.size() == 1 && (args[0] == "--version" || args[0] == "-V")) {
        return make_result(GenerateCliAction::kVersion, kGenerateExitSuccess, version_message());
    }
    GenerateCliResult parsed;
    parsed.action = GenerateCliAction::kGenerate;
    parsed.exit_code = kGenerateExitSuccess;
    bool saw_package = false;
    bool saw_prompt = false;
    bool saw_prompt_file = false;
    bool saw_max = false;
    bool saw_threads = false;
    for (std::size_t index = 0; index < args.size(); ++index) {
        const std::string_view arg = args[index];
        if (arg.size() > kGenerateMaxOptionLength) {
            return error_result("error: argument exceeds maximum length\n");
        }
        if (arg == "--help" || arg == "-h" || arg == "--version" || arg == "-V") {
            return error_result(
                "error: --help and --version cannot be combined with other arguments\n");
        }
        if (arg == "--model-package" || arg == "--prompt" || arg == "--prompt-file" ||
            arg == "--max-new-tokens" || arg == "--threads") {
            if (index + 1 >= args.size()) {
                return error_result("error: missing value for " + std::string{arg} + "\n");
            }
            const std::string_view value = args[++index];
            if (value.size() >
                (arg == "--prompt" ? kGenerateMaxPromptLength : kGenerateMaxOptionLength)) {
                return error_result("error: argument exceeds maximum length\n");
            }
            if (value.empty()) {
                return error_result("error: empty value for " + std::string{arg} + "\n");
            }
            if (arg == "--model-package") {
                if (saw_package) {
                    return error_result("error: duplicate option --model-package\n");
                }
                saw_package = true;
                parsed.model_package = std::string{value};
            } else if (arg == "--prompt") {
                if (saw_prompt) {
                    return error_result("error: duplicate option --prompt\n");
                }
                saw_prompt = true;
                if (!user_prompt_is_usable(value)) {
                    return error_result("error: empty or invalid --prompt\n");
                }
                parsed.prompt = std::string{value};
            } else if (arg == "--prompt-file") {
                if (saw_prompt_file) {
                    return error_result("error: duplicate option --prompt-file\n");
                }
                saw_prompt_file = true;
                parsed.prompt_file = std::string{value};
            } else if (arg == "--max-new-tokens") {
                if (saw_max) {
                    return error_result("error: duplicate option --max-new-tokens\n");
                }
                saw_max = true;
                if (!parse_u32_token(value, parsed.max_new_tokens) || parsed.max_new_tokens == 0 ||
                    parsed.max_new_tokens > kGenerativeMaxContext) {
                    return error_result("error: invalid --max-new-tokens\n");
                }
            } else if (arg == "--threads") {
                if (saw_threads) {
                    return error_result("error: duplicate option --threads\n");
                }
                saw_threads = true;
                if (!parse_u32_token(value, parsed.threads) || parsed.threads == 0 ||
                    parsed.threads > 256) {
                    return error_result("error: invalid --threads\n");
                }
            }
            continue;
        }
        return error_result("error: unexpected argument '" + std::string{arg} + "'\n");
    }
    if (!saw_package) {
        return error_result("error: --model-package is required\n");
    }
    if (saw_prompt == saw_prompt_file) {
        return error_result("error: exactly one of --prompt or --prompt-file is required\n");
    }
    if (!saw_max) {
        return error_result("error: --max-new-tokens is required\n");
    }
    if (!saw_threads) {
        return error_result("error: --threads is required\n");
    }
    return parsed;
}

} // namespace inferedge
