#include "inferedge/cli.hpp"
#include "inferedge/version.hpp"

#include <gtest/gtest.h>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace {

inferedge::CliResult parse(std::initializer_list<std::string_view> args) {
    return inferedge::parse_cli(std::vector<std::string_view>{args});
}

std::string expected_version_line() {
    std::string line;
    line += inferedge::project_name();
    line += " ";
    line += inferedge::version();
    line += "\n";
    return line;
}

} // namespace

TEST(CliTest, NoArgumentsShowsHelpAndSucceeds) {
    const inferedge::CliResult result = parse({});
    EXPECT_EQ(result.action, inferedge::CliAction::kHelp);
    EXPECT_EQ(result.exit_code, 0);
    EXPECT_NE(result.message.find("Usage: inferedge"), std::string::npos);
    EXPECT_NE(result.message.find("--help"), std::string::npos);
    EXPECT_NE(result.message.find("--version"), std::string::npos);
}

TEST(CliTest, LongAndShortHelpMatch) {
    const inferedge::CliResult long_form = parse({"--help"});
    const inferedge::CliResult short_form = parse({"-h"});
    EXPECT_EQ(long_form.action, inferedge::CliAction::kHelp);
    EXPECT_EQ(short_form.action, inferedge::CliAction::kHelp);
    EXPECT_EQ(long_form.exit_code, 0);
    EXPECT_EQ(short_form.exit_code, 0);
    EXPECT_EQ(long_form.message, short_form.message);
}

TEST(CliTest, VersionUsesTheSameStringAsTheLibrary) {
    const inferedge::CliResult result = parse({"--version"});
    EXPECT_EQ(result.action, inferedge::CliAction::kVersion);
    EXPECT_EQ(result.exit_code, 0);
    EXPECT_EQ(result.message, expected_version_line());
    EXPECT_EQ(parse({"-V"}).message, result.message);
}

TEST(CliTest, UnknownOptionIsAUsageError) {
    const inferedge::CliResult result = parse({"--scheduler"});
    EXPECT_EQ(result.action, inferedge::CliAction::kError);
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_NE(result.message.find("unknown option"), std::string::npos);
    EXPECT_NE(result.message.find("--scheduler"), std::string::npos);
    EXPECT_NE(result.message.find("Usage: inferedge"), std::string::npos);
}

TEST(CliTest, BareWordsAreNotCommands) {
    const inferedge::CliResult result = parse({"help"});
    EXPECT_EQ(result.action, inferedge::CliAction::kError);
    EXPECT_EQ(result.exit_code, 2);
}

TEST(CliTest, ExtraArgumentsAreRejected) {
    const inferedge::CliResult help = parse({"--help", "more"});
    const inferedge::CliResult version = parse({"--version", "0.1.0"});
    EXPECT_EQ(help.action, inferedge::CliAction::kError);
    EXPECT_EQ(version.action, inferedge::CliAction::kError);
    EXPECT_EQ(help.exit_code, 2);
    EXPECT_EQ(version.exit_code, 2);
    EXPECT_NE(help.message.find("unexpected argument"), std::string::npos);
    EXPECT_NE(version.message.find("unexpected argument"), std::string::npos);
}
