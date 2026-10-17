#include "inferedge/generate_cli.hpp"
#include "inferedge/version.hpp"

#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

inferedge::GenerateCliResult parse(std::initializer_list<std::string_view> args) {
    return inferedge::parse_generate_cli(std::vector<std::string_view>{args});
}

} // namespace

TEST(GenerateCliTest, HelpAndVersion) {
    const auto help = parse({"--help"});
    EXPECT_EQ(help.action, inferedge::GenerateCliAction::kHelp);
    EXPECT_EQ(help.exit_code, 0);
    EXPECT_NE(help.message.find("Usage: inferedge-generate"), std::string::npos);
    EXPECT_EQ(parse({"-h"}).message, help.message);
    const auto version = parse({"--version"});
    EXPECT_EQ(version.action, inferedge::GenerateCliAction::kVersion);
    EXPECT_EQ(version.exit_code, 0);
    EXPECT_NE(version.message.find(std::string{inferedge::version()}), std::string::npos);
    EXPECT_EQ(parse({"-V"}).message, version.message);
}

TEST(GenerateCliTest, EmptyArgsAreHelp) {
    const auto result = parse({});
    EXPECT_EQ(result.action, inferedge::GenerateCliAction::kHelp);
    EXPECT_EQ(result.exit_code, 0);
}

TEST(GenerateCliTest, RequiresExplicitOptions) {
    const auto missing = parse({"--prompt", "hi", "--max-new-tokens", "8", "--threads", "1"});
    EXPECT_EQ(missing.action, inferedge::GenerateCliAction::kError);
    EXPECT_EQ(missing.exit_code, inferedge::kGenerateExitUsage);
    const auto ok = parse({"--model-package", "/tmp/pkg", "--prompt", "hi", "--max-new-tokens", "8",
                           "--threads", "1"});
    EXPECT_EQ(ok.action, inferedge::GenerateCliAction::kGenerate);
    EXPECT_EQ(ok.model_package, "/tmp/pkg");
    EXPECT_EQ(ok.prompt, "hi");
    EXPECT_EQ(ok.max_new_tokens, 8U);
    EXPECT_EQ(ok.threads, 1U);
}

TEST(GenerateCliTest, RejectsMissingRequiredOptions) {
    EXPECT_EQ(
        parse({"--model-package", "/tmp/pkg", "--max-new-tokens", "8", "--threads", "1"}).action,
        inferedge::GenerateCliAction::kError);
    EXPECT_EQ(parse({"--model-package", "/tmp/pkg", "--prompt", "hi", "--threads", "1"}).action,
              inferedge::GenerateCliAction::kError);
    EXPECT_EQ(
        parse({"--model-package", "/tmp/pkg", "--prompt", "hi", "--max-new-tokens", "8"}).action,
        inferedge::GenerateCliAction::kError);
}

TEST(GenerateCliTest, RejectsDuplicateAndUnknownOptions) {
    const auto dup = parse({"--model-package", "/tmp/a", "--model-package", "/tmp/b", "--prompt",
                            "hi", "--max-new-tokens", "8", "--threads", "1"});
    EXPECT_EQ(dup.action, inferedge::GenerateCliAction::kError);
    const auto unknown = parse({"--download-model", "qwen"});
    EXPECT_EQ(unknown.action, inferedge::GenerateCliAction::kError);
}

TEST(GenerateCliTest, RejectsInvalidNumericValues) {
    EXPECT_EQ(parse({"--model-package", "/tmp/pkg", "--prompt", "hi", "--max-new-tokens", "-1",
                     "--threads", "1"})
                  .action,
              inferedge::GenerateCliAction::kError);
    EXPECT_EQ(parse({"--model-package", "/tmp/pkg", "--prompt", "hi", "--max-new-tokens", "8",
                     "--threads", "0"})
                  .action,
              inferedge::GenerateCliAction::kError);
    EXPECT_EQ(parse({"--model-package", "/tmp/pkg", "--prompt", "hi", "--max-new-tokens",
                     "18446744073709551616", "--threads", "1"})
                  .action,
              inferedge::GenerateCliAction::kError);
}

TEST(GenerateCliTest, RejectsOversizedArguments) {
    const std::string huge(inferedge::kGenerateMaxOptionLength + 1, 'a');
    EXPECT_EQ(parse({huge}).action, inferedge::GenerateCliAction::kError);
}

TEST(GenerateCliTest, PromptAndPromptFileAreExclusive) {
    const auto both = parse({"--model-package", "/tmp/pkg", "--prompt", "hi", "--prompt-file",
                             "/tmp/p.txt", "--max-new-tokens", "8", "--threads", "1"});
    EXPECT_EQ(both.action, inferedge::GenerateCliAction::kError);
    const auto file_only = parse({"--model-package", "/tmp/pkg", "--prompt-file", "/tmp/p.txt",
                                  "--max-new-tokens", "8", "--threads", "1"});
    EXPECT_EQ(file_only.action, inferedge::GenerateCliAction::kGenerate);
    EXPECT_EQ(file_only.prompt_file, "/tmp/p.txt");
    EXPECT_TRUE(file_only.prompt.empty());
}

TEST(GenerateCliTest, DoesNotReadStdinImplicitly) {
    const auto result =
        parse({"--model-package", "/tmp/pkg", "--max-new-tokens", "8", "--threads", "1"});
    EXPECT_EQ(result.action, inferedge::GenerateCliAction::kError);
    EXPECT_NE(result.message.find("exactly one of --prompt or --prompt-file"), std::string::npos);
}

TEST(GenerateCliTest, HelpCannotCombineWithGenerateOptions) {
    const auto result = parse({"--help", "--prompt", "hi"});
    EXPECT_EQ(result.action, inferedge::GenerateCliAction::kError);
}

TEST(GenerateCliTest, RejectsEmptyAndWhitespacePrompt) {
    EXPECT_EQ(parse({"--model-package", "/tmp/pkg", "--prompt", "", "--max-new-tokens", "8",
                     "--threads", "1"})
                  .action,
              inferedge::GenerateCliAction::kError);
    EXPECT_EQ(parse({"--model-package", "/tmp/pkg", "--prompt", "   ", "--max-new-tokens", "8",
                     "--threads", "1"})
                  .action,
              inferedge::GenerateCliAction::kError);
    EXPECT_EQ(parse({"--model-package", "/tmp/pkg", "--prompt", "\t\n", "--max-new-tokens", "8",
                     "--threads", "1"})
                  .action,
              inferedge::GenerateCliAction::kError);
}

TEST(GenerateCliTest, HelpDocumentsPromptPrivacy) {
    const auto help = parse({"--help"});
    EXPECT_NE(help.message.find("process listings"), std::string::npos);
    EXPECT_NE(help.message.find("sensitive prompts"), std::string::npos);
    EXPECT_NE(help.message.find("does not persist or network"), std::string::npos);
    EXPECT_NE(help.message.find("ASCII-whitespace-only"), std::string::npos);
}
