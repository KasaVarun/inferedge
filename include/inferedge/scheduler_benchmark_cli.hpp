#pragma once

#include "inferedge/scheduler_benchmark.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace inferedge {

enum class SchedulerBenchmarkCliAction {
    kHelp,
    kVersion,
    kError,
    kRun,
};

enum class SchedulerBenchmarkOutputFormat {
    kText,
    kJson,
};

// Parsing does not open a model file. execute_scheduler_benchmark does.
struct SchedulerBenchmarkCliResult {
    SchedulerBenchmarkCliAction action{};
    int exit_code{0};
    std::string message;
    SchedulerBenchmarkConfig config;
    SchedulerBenchmarkOutputFormat output_format{SchedulerBenchmarkOutputFormat::kText};
};

[[nodiscard]] std::string scheduler_benchmark_help_message();
[[nodiscard]] std::string scheduler_benchmark_version_message();

// args does not include the program name. An empty list means "show help".
[[nodiscard]] SchedulerBenchmarkCliResult
parse_scheduler_benchmark_cli(const std::vector<std::string_view>& args);

// Runs one trial. JSON mode writes schema version 3 to message, including an
// invalid trial. Usage errors stay on message with exit code 2.
[[nodiscard]] SchedulerBenchmarkCliResult
execute_scheduler_benchmark(const SchedulerBenchmarkCliResult& parsed);

} // namespace inferedge
