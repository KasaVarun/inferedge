#pragma once

#include "inferedge/scheduler_benchmark.hpp"

#include <string>
#include <string_view>

namespace inferedge {

// text is one JSON document and a trailing newline when error is empty.
// The writer does not run a scheduler, read a clock, or recompute percentiles.
struct SchedulerBenchmarkJsonResult {
    std::string error;
    std::string text;
};

[[nodiscard]] SchedulerBenchmarkJsonResult
write_scheduler_benchmark_json(const SchedulerBenchmarkReport& report,
                               std::string_view tool_version);

// Used when the runner fails before a report exists. schema_version is 3.
[[nodiscard]] SchedulerBenchmarkJsonResult
write_scheduler_benchmark_failure_json(std::string_view error_kind, std::string_view message,
                                       std::string_view tool_version);

} // namespace inferedge
