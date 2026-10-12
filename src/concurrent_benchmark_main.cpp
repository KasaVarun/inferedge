#include "inferedge/concurrent_benchmark_cli.hpp"

#include <exception>
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
    try {
        inferedge::ConcurrentBenchmarkCliResult result =
            inferedge::parse_concurrent_benchmark_cli(arguments_after_program_name(argc, argv));
        if (result.action == inferedge::ConcurrentBenchmarkCliAction::kRun) {
            result = inferedge::execute_concurrent_benchmark(result);
        }
        if (result.action == inferedge::ConcurrentBenchmarkCliAction::kError) {
            std::cerr << result.message;
        } else {
            std::cout << result.message;
        }
        return result.exit_code;
    } catch (const std::exception& error) {
        std::cerr << "error: internal error: " << error.what() << '\n';
        return 8;
    } catch (...) {
        std::cerr << "error: internal error: unknown exception\n";
        return 8;
    }
}
