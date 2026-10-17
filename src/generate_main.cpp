#include "inferedge/generate_cli.hpp"
#include "inferedge/generative_session.hpp"

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::vector<std::string_view> arguments_after_program_name(int argc, char* argv[]) {
    std::vector<std::string_view> args;
    if (argc <= 1) {
        return args;
    }
    args.reserve(static_cast<std::size_t>(argc - 1));
    for (int index = 1; index < argc; ++index) {
        args.emplace_back(argv[index]);
    }
    return args;
}

inferedge::GenerateCliResult load_prompt_file(inferedge::GenerateCliResult result) {
    if (result.action != inferedge::GenerateCliAction::kGenerate || result.prompt_file.empty()) {
        return result;
    }
    inferedge::PackageStatus loaded =
        inferedge::load_untrusted_prompt_file(result.prompt_file, result.prompt);
    if (!loaded.ok) {
        result.action = inferedge::GenerateCliAction::kError;
        result.exit_code = inferedge::kGenerateExitUsage;
        result.message = "error: " + loaded.message + "\n";
        return result;
    }
    return result;
}

} // namespace

int main(int argc, char* argv[]) {
    inferedge::GenerateCliResult parsed =
        inferedge::parse_generate_cli(arguments_after_program_name(argc, argv));
    if (parsed.action == inferedge::GenerateCliAction::kError) {
        std::cerr << parsed.message;
        return parsed.exit_code;
    }
    if (parsed.action != inferedge::GenerateCliAction::kGenerate) {
        std::cout << parsed.message;
        return parsed.exit_code;
    }
    parsed = load_prompt_file(std::move(parsed));
    if (parsed.action == inferedge::GenerateCliAction::kError) {
        std::cerr << parsed.message;
        return parsed.exit_code;
    }
    inferedge::GenerativeSession session;
    inferedge::GenerationStatus loaded = session.load(parsed.model_package);
    if (!loaded.ok) {
        std::cerr << "error: " << loaded.message << "\n";
        return inferedge::kGenerateExitFailure;
    }
    inferedge::GenerationRequest request;
    request.prompt = parsed.prompt;
    request.max_new_tokens = parsed.max_new_tokens;
    request.threads = parsed.threads;
    inferedge::GenerationResult generated = session.generate(request);
    if (!generated.status.ok) {
        std::cerr << "error: " << generated.status.message << "\n";
        return inferedge::kGenerateExitFailure;
    }
    std::cout << generated.text;
    if (!generated.text.empty() && generated.text.back() != '\n') {
        std::cout << '\n';
    }
    return inferedge::kGenerateExitSuccess;
}
