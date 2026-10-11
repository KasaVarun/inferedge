#include "inferedge/cli.hpp"

#include <iostream>
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

} // namespace

int main(int argc, char* argv[]) {
    const inferedge::CliResult result =
        inferedge::parse_cli(arguments_after_program_name(argc, argv));
    if (result.action == inferedge::CliAction::kError) {
        std::cerr << result.message;
    } else {
        std::cout << result.message;
    }
    return result.exit_code;
}
