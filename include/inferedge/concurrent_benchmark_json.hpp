#pragma once

#include "inferedge/concurrent_benchmark.hpp"

#include <string>
#include <string_view>

namespace inferedge {

// text is one JSON document and a trailing newline when error is empty.
// On failure text is empty. error has no "error: " prefix and no trailing newline.
// The writer does not run inference, read a clock, or recompute percentiles.
struct ConcurrentBenchmarkJsonResult {
    std::string error;
    std::string text;
};

// fixture_id empty becomes JSON null. tool_version is copied into the document.
// Schema version 1 is a different function and is not called here.
ConcurrentBenchmarkJsonResult
write_concurrent_benchmark_json(const ConcurrentBenchmarkReport& report,
                                std::string_view fixture_id, std::string_view tool_version);

} // namespace inferedge
