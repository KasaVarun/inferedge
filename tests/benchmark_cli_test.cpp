#include "inferedge/benchmark_cli.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef INFEREDGE_MNIST_FIXTURE_DIR
#error "CMake must define INFEREDGE_MNIST_FIXTURE_DIR"
#endif

namespace {

using inferedge::BenchmarkCliAction;
using inferedge::BenchmarkCliResult;
using inferedge::find_mnist_fixture;
using inferedge::kMnistFixtureCatalog;
using inferedge::kMnistFixtureCount;
using inferedge::parse_benchmark_cli;

BenchmarkCliResult parse(std::initializer_list<std::string_view> args) {
    return parse_benchmark_cli(std::vector<std::string_view>{args});
}

void expect_usage(const BenchmarkCliResult& result, std::string_view fragment) {
    EXPECT_EQ(result.action, BenchmarkCliAction::kError);
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_EQ(result.message.compare(0, 7, "error: "), 0) << result.message;
    EXPECT_NE(result.message.find(fragment), std::string::npos) << result.message;
    EXPECT_NE(result.message.find("Usage: inferedge-bench"), std::string::npos) << result.message;
}

} // namespace

TEST(BenchmarkCliTest, NoArgumentsShowHelp) {
    const BenchmarkCliResult result = parse({});
    EXPECT_EQ(result.action, BenchmarkCliAction::kHelp);
    EXPECT_EQ(result.exit_code, 0);
    EXPECT_NE(result.message.find("Usage: inferedge-bench"), std::string::npos);
    EXPECT_EQ(result.message, inferedge::benchmark_help_message());
}

TEST(BenchmarkCliTest, HelpAndVersionAreStandalone) {
    const BenchmarkCliResult help = parse({"--help"});
    const BenchmarkCliResult short_help = parse({"-h"});
    const BenchmarkCliResult version = parse({"--version"});
    const BenchmarkCliResult short_version = parse({"-V"});
    EXPECT_EQ(help.exit_code, 0);
    EXPECT_EQ(help.action, BenchmarkCliAction::kHelp);
    EXPECT_EQ(short_help.message, help.message);
    EXPECT_EQ(version.exit_code, 0);
    EXPECT_EQ(version.action, BenchmarkCliAction::kVersion);
    EXPECT_EQ(version.message, "inferedge-bench 0.1.0\n");
    EXPECT_EQ(short_version.message, version.message);
    expect_usage(parse({"--help", "--model", "m"}), "unexpected argument");
    expect_usage(parse({"--version", "--workload", "add"}), "unexpected argument");
    expect_usage(parse({"--model", "m", "--help"}), "unexpected argument");
}

TEST(BenchmarkCliTest, RejectsMissingRequiredFlags) {
    expect_usage(parse({"--workload", "add"}), "missing --model");
    expect_usage(parse({"--model", "models/add_f32.onnx"}), "missing --workload");
    expect_usage(parse({"--model"}), "missing value for --model");
    expect_usage(parse({"--model", "--workload"}), "missing value for --model");
    expect_usage(parse({"--model="}), "empty model path");
}

TEST(BenchmarkCliTest, RejectsAnUnknownWorkload) {
    expect_usage(parse({"--model", "m", "--workload", "bert"}), "unknown workload 'bert'");
    expect_usage(parse({"--model", "m", "--workload", "ADD"}), "unknown workload 'ADD'");
    expect_usage(parse({"--workload="}), "missing --model");
}

TEST(BenchmarkCliTest, RejectsUnknownAndDuplicateOptions) {
    expect_usage(parse({"--scheduler"}), "unknown option '--scheduler'");
    expect_usage(parse({"--model", "m", "--workload", "add", "extra"}),
                 "unexpected argument 'extra'");
    expect_usage(parse({"--model", "m", "--model", "n", "--workload", "add"}),
                 "--model specified more than once");
    expect_usage(parse({"--model", "m", "--workload", "add", "--workload", "mnist"}),
                 "--workload specified more than once");
    expect_usage(parse({"--model", "m", "--workload", "add", "--warmup", "1", "--warmup", "2"}),
                 "--warmup specified more than once");
    expect_usage(parse({"--model", "m", "--workload", "mnist", "--fixture", "synthetic",
                        "--fixture", "test_00003"}),
                 "--fixture specified more than once");
}

TEST(BenchmarkCliTest, RejectsInvalidCounts) {
    expect_usage(parse({"--model", "m", "--workload", "add", "--iterations", "abc"}),
                 "invalid value for --iterations");
    expect_usage(parse({"--model", "m", "--workload", "add", "--iterations=-1"}),
                 "invalid value for --iterations");
    expect_usage(parse({"--model", "m", "--workload", "add", "--warmup=+3"}),
                 "invalid value for --warmup");
    expect_usage(parse({"--model", "m", "--workload", "add", "--iterations", "-1"}),
                 "missing value for --iterations");
    expect_usage(parse({"--model", "m", "--workload", "add", "--iterations=4294967296"}),
                 "--iterations overflows");
    expect_usage(parse({"--model", "m", "--workload", "add", "--warmup=100001"}),
                 "--warmup exceeds 100000");
    expect_usage(parse({"--model", "m", "--workload", "add", "--iterations=100001"}),
                 "--iterations exceeds 100000");
    expect_usage(parse({"--model", "m", "--workload", "add", "--iterations=0"}),
                 "measured iterations must be at least 1");
    expect_usage(parse({"--model", "m", "--workload", "add", "--check-interval=0"}),
                 "correctness-check interval must be at least 1");
    expect_usage(parse({"--model", "m", "--workload", "add", "--check-interval"}),
                 "missing value for --check-interval");
}

TEST(BenchmarkCliTest, AcceptsDefaultsAndAttachedForms) {
    const BenchmarkCliResult result =
        parse({"--model=models/add_f32.onnx", "--workload=add", "--warmup=0", "--iterations=100000",
               "--check-interval=1000"});
    EXPECT_EQ(result.action, BenchmarkCliAction::kRun);
    EXPECT_EQ(result.exit_code, 0);
    EXPECT_TRUE(result.message.empty());
    EXPECT_EQ(result.model_path, "models/add_f32.onnx");
    EXPECT_EQ(result.workload, "add");
    EXPECT_EQ(result.warmup_iterations, 0u);
    EXPECT_EQ(result.measured_iterations, 100000u);
    EXPECT_EQ(result.correctness_check_interval, 1000u);
    EXPECT_FALSE(result.fixture_specified);

    const BenchmarkCliResult defaults = parse({"--model", "m", "--workload", "add"});
    EXPECT_EQ(defaults.action, BenchmarkCliAction::kRun);
    EXPECT_EQ(defaults.warmup_iterations, 10u);
    EXPECT_EQ(defaults.measured_iterations, 100u);
    EXPECT_EQ(defaults.correctness_check_interval, 1u);
}

TEST(BenchmarkCliTest, RejectsFixtureFlagsOnTheAddWorkload) {
    expect_usage(parse({"--model", "m", "--workload", "add", "--fixture", "test_00003"}),
                 "fixture flags apply only to the mnist workload");
    expect_usage(
        parse({"--model", "m", "--workload", "add", "--fixture-dir", "models/fixtures/mnist"}),
        "fixture flags apply only to the mnist workload");
}

TEST(BenchmarkCliTest, DefaultsTheMnistFixtureAndRejectsUnknownIds) {
    const BenchmarkCliResult result = parse({"--model", "missing.onnx", "--workload", "mnist"});
    EXPECT_EQ(result.action, BenchmarkCliAction::kRun);
    EXPECT_EQ(result.exit_code, 0);
    EXPECT_FALSE(result.fixture_specified);
    EXPECT_EQ(result.fixture_id, "test_00003");
    EXPECT_TRUE(result.message.empty());

    const BenchmarkCliResult synthetic =
        parse({"--model", "m", "--workload", "mnist", "--fixture=synthetic"});
    EXPECT_EQ(synthetic.action, BenchmarkCliAction::kRun);
    EXPECT_TRUE(synthetic.fixture_specified);
    EXPECT_EQ(synthetic.fixture_id, "synthetic");

    expect_usage(parse({"--model", "m", "--workload", "mnist", "--fixture", "missing_case"}),
                 "unknown fixture id 'missing_case'");
    expect_usage(parse({"--model", "m", "--workload", "mnist", "--fixture="}), "empty fixture id");
    expect_usage(parse({"--model", "m", "--workload", "mnist", "--fixture", "../test_00003"}),
                 "fixture id contains a path");
    expect_usage(
        parse({"--model", "m", "--workload", "mnist", "--fixture", "test_00003/../../etc"}),
        "fixture id contains a path");
    expect_usage(parse({"--model", "m", "--workload", "mnist", "--fixture", ".."}),
                 "fixture id contains a path");
    expect_usage(parse({"--model", "m", "--workload", "mnist", "--fixture-dir="}),
                 "empty fixture directory");
}

TEST(BenchmarkCliTest, CatalogMatchesTheManifest) {
    const std::filesystem::path manifest =
        std::filesystem::path{INFEREDGE_MNIST_FIXTURE_DIR} / "manifest.json";
    std::ifstream input{manifest};
    ASSERT_TRUE(input) << manifest;
    const std::string text{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
    EXPECT_NE(text.find("\"input_elements\": 784"), std::string::npos);
    EXPECT_NE(text.find("\"logit_elements\": 10"), std::string::npos);

    const std::string name_key = "\"name\": \"";
    const std::string class_key = "\"expected_predicted_class\": ";
    std::vector<std::pair<std::string, int>> cases;
    std::string last_name;
    std::size_t position = 0;
    while (position < text.size()) {
        const std::size_t name_at = text.find(name_key, position);
        const std::size_t class_at = text.find(class_key, position);
        if (name_at == std::string::npos && class_at == std::string::npos) {
            break;
        }
        if (class_at == std::string::npos || (name_at != std::string::npos && name_at < class_at)) {
            const std::size_t start = name_at + name_key.size();
            const std::size_t end = text.find('"', start);
            ASSERT_NE(end, std::string::npos);
            last_name = text.substr(start, end - start);
            position = end + 1;
            continue;
        }
        const std::size_t start = class_at + class_key.size();
        std::size_t end = start;
        while (end < text.size() &&
               (std::isdigit(static_cast<unsigned char>(text[end])) != 0 || text[end] == '-')) {
            ++end;
        }
        cases.emplace_back(last_name, std::stoi(text.substr(start, end - start)));
        position = end;
    }

    ASSERT_EQ(cases.size(), kMnistFixtureCount);
    ASSERT_EQ(kMnistFixtureCount, 14u);
    for (const inferedge::MnistFixture& fixture : kMnistFixtureCatalog) {
        const auto found = std::find_if(cases.begin(), cases.end(),
                                        [&](const auto& item) { return item.first == fixture.id; });
        ASSERT_NE(found, cases.end()) << fixture.id;
        EXPECT_EQ(found->second, fixture.expected_class) << fixture.id;
        EXPECT_EQ(find_mnist_fixture(fixture.id), &fixture);
    }
    EXPECT_NE(inferedge::benchmark_help_message().find("--format <name>"), std::string::npos);
    for (const auto& item : cases) {
        const inferedge::MnistFixture* fixture = find_mnist_fixture(item.first);
        ASSERT_NE(fixture, nullptr) << item.first;
        EXPECT_EQ(fixture->expected_class, item.second) << item.first;
    }
}

TEST(BenchmarkCliTest, DefaultsToTextAndAcceptsFormatFlags) {
    const BenchmarkCliResult defaults = parse({"--model", "m", "--workload", "add"});
    EXPECT_EQ(defaults.action, BenchmarkCliAction::kRun);
    EXPECT_EQ(defaults.output_format, inferedge::BenchmarkOutputFormat::kText);

    const BenchmarkCliResult text =
        parse({"--model", "m", "--workload", "add", "--format", "text"});
    EXPECT_EQ(text.action, BenchmarkCliAction::kRun);
    EXPECT_EQ(text.output_format, inferedge::BenchmarkOutputFormat::kText);
    EXPECT_TRUE(text.message.empty());

    const BenchmarkCliResult json =
        parse({"--model", "m", "--workload", "add", "--format", "json"});
    EXPECT_EQ(json.action, BenchmarkCliAction::kRun);
    EXPECT_EQ(json.output_format, inferedge::BenchmarkOutputFormat::kJson);

    const BenchmarkCliResult attached =
        parse({"--model", "m", "--workload", "add", "--format=json"});
    EXPECT_EQ(attached.output_format, inferedge::BenchmarkOutputFormat::kJson);
    const BenchmarkCliResult attached_text =
        parse({"--model", "m", "--workload", "mnist", "--format=text"});
    EXPECT_EQ(attached_text.output_format, inferedge::BenchmarkOutputFormat::kText);
    EXPECT_EQ(attached_text.fixture_id, "test_00003");
}

TEST(BenchmarkCliTest, RejectsInvalidFormatFlags) {
    expect_usage(parse({"--format", "yaml"}), "unknown format 'yaml'");
    expect_usage(parse({"--format", "JSON", "--model", "m", "--workload", "add"}),
                 "unknown format 'JSON'");
    expect_usage(parse({"--model", "m", "--workload", "add", "--format"}),
                 "missing value for --format");
    expect_usage(parse({"--format", "--model", "m", "--workload", "add"}),
                 "missing value for --format");
    expect_usage(parse({"--model", "m", "--workload", "add", "--format="}), "empty format");
    expect_usage(parse({"--model", "m", "--workload", "add", "--format", ""}), "empty format");
    expect_usage(
        parse({"--model", "m", "--workload", "add", "--format", "json", "--format", "text"}),
        "--format specified more than once");
    expect_usage(parse({"--format=json", "--model", "m", "--workload", "add", "--format=text"}),
                 "--format specified more than once");
    expect_usage(parse({"--help", "--format", "json"}), "unexpected argument");
    expect_usage(parse({"--format", "json", "--version"}), "unexpected argument");
    expect_usage(parse({"--version", "--format=text"}), "unexpected argument");
}
