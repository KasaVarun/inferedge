#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace inferedge {

// Parsing does not open a model or a fixture file.
enum class BenchmarkCliAction {
    kHelp,
    kVersion,
    kError,
    kRun,
};

// Owned parsed request. execute_benchmark builds input buffers and views after
// this result exists. Views are not stored here.
struct BenchmarkCliResult {
    BenchmarkCliAction action{};
    int exit_code{0};
    std::string message;
    std::string model_path;
    std::string workload;
    std::uint32_t warmup_iterations{10};
    std::uint32_t measured_iterations{100};
    std::uint32_t correctness_check_interval{1};
    std::string fixture_id{"test_00003"};
    bool fixture_specified{false};
    std::string fixture_directory;
    bool fixture_directory_specified{false};
};

// Ids and expected classes copied from models/fixtures/mnist/manifest.json.
// The manifest remains the metadata source. A test checks this table against it.
struct MnistFixture {
    std::string_view id;
    int expected_class;
};

inline constexpr MnistFixture kMnistFixtureCatalog[] = {
    {"test_00003", 0}, {"test_00002", 1}, {"test_00001", 2}, {"test_00018", 3}, {"test_00004", 4},
    {"test_00008", 5}, {"test_00011", 6}, {"test_00000", 7}, {"test_00061", 8}, {"test_00007", 9},
    {"test_00151", 8}, {"test_00217", 5}, {"test_00241", 8}, {"synthetic", 5},
};

inline constexpr std::size_t kMnistFixtureCount =
    sizeof(kMnistFixtureCatalog) / sizeof(kMnistFixtureCatalog[0]);

static_assert(kMnistFixtureCount == 14);

inline const MnistFixture* find_mnist_fixture(std::string_view id) {
    for (const MnistFixture& fixture : kMnistFixtureCatalog) {
        if (fixture.id == id) {
            return &fixture;
        }
    }
    return nullptr;
}

std::string benchmark_help_message();
std::string benchmark_version_message();

// args does not include the program name. An empty list means "show help".
BenchmarkCliResult parse_benchmark_cli(const std::vector<std::string_view>& args);

// Loads fixtures, loads one model, checks metadata, and calls run_benchmark.
// On failure, message is stderr text and the report is absent.
BenchmarkCliResult execute_benchmark(const BenchmarkCliResult& parsed);

} // namespace inferedge
