#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace inferedge {

// Parsing does not open a model or a fixture file.
enum class ConcurrentBenchmarkCliAction {
    kHelp,
    kVersion,
    kError,
    kRun,
};

// text is the human report. json is one schema-version-2 document.
enum class ConcurrentBenchmarkOutputFormat {
    kText,
    kJson,
};

// Owned parsed request. worker_count stays 0 until --concurrency is accepted.
// execute_concurrent_benchmark copies input buffers after this result exists.
struct ConcurrentBenchmarkCliResult {
    ConcurrentBenchmarkCliAction action{};
    int exit_code{0};
    std::string message;
    std::string model_path;
    std::string workload;
    std::uint32_t worker_count{0};
    std::uint32_t warmup_iterations{10};
    std::uint32_t measured_iterations{100};
    std::uint32_t correctness_check_interval{1};
    std::string fixture_id{"test_00003"};
    bool fixture_specified{false};
    std::string fixture_directory;
    bool fixture_directory_specified{false};
    ConcurrentBenchmarkOutputFormat output_format{ConcurrentBenchmarkOutputFormat::kText};
};

std::string concurrent_benchmark_help_message();
std::string concurrent_benchmark_version_message();

// args does not include the program name. An empty list means "show help".
ConcurrentBenchmarkCliResult
parse_concurrent_benchmark_cli(const std::vector<std::string_view>& args);

// Prepares one workload, checks metadata, copies one assignment per worker,
// and calls run_concurrent_benchmark. On failure, message is stderr text and
// the report is absent.
ConcurrentBenchmarkCliResult
execute_concurrent_benchmark(const ConcurrentBenchmarkCliResult& parsed);

} // namespace inferedge
