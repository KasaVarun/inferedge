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

inferedge::CliResult parse_owned(const std::vector<std::string>& args) {
    std::vector<std::string_view> views;
    views.reserve(args.size());
    for (const std::string& arg : args) {
        views.emplace_back(arg);
    }
    return inferedge::parse_cli(views);
}

std::string zeros(std::size_t count) {
    std::string text;
    if (count == 0) {
        return text;
    }
    text.reserve(count * 2 - 1);
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0) {
            text.push_back(',');
        }
        text.push_back('0');
    }
    return text;
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

TEST(CliTest, ParsesModelAndRepeatedInputs) {
    const inferedge::CliResult result =
        parse({"--model", "models/add_f32.onnx", "--input", "A=1,2", "--input", "B=3,4"});
    ASSERT_EQ(result.action, inferedge::CliAction::kRun);
    EXPECT_EQ(result.exit_code, 0);
    EXPECT_EQ(result.model_path, "models/add_f32.onnx");
    ASSERT_EQ(result.inputs.size(), 2u);
    EXPECT_EQ(result.inputs[0].name, "A");
    ASSERT_EQ(result.inputs[0].values.size(), 2u);
    EXPECT_FLOAT_EQ(result.inputs[0].values[0], 1.0f);
    EXPECT_FLOAT_EQ(result.inputs[0].values[1], 2.0f);
    EXPECT_EQ(result.inputs[1].name, "B");
    EXPECT_FLOAT_EQ(result.inputs[1].values[0], 3.0f);
    EXPECT_FLOAT_EQ(result.inputs[1].values[1], 4.0f);
}

TEST(CliTest, AcceptsAttachedOptionForms) {
    const inferedge::CliResult result =
        parse({"--input=B=-1.5,0", "--model=models/add_f32.onnx", "--input=A=2.5,-4"});
    ASSERT_EQ(result.action, inferedge::CliAction::kRun);
    EXPECT_EQ(result.model_path, "models/add_f32.onnx");
    ASSERT_EQ(result.inputs.size(), 2u);
    EXPECT_EQ(result.inputs[0].name, "B");
    EXPECT_FLOAT_EQ(result.inputs[0].values[0], -1.5f);
    EXPECT_FLOAT_EQ(result.inputs[0].values[1], 0.0f);
    EXPECT_EQ(result.inputs[1].name, "A");
    EXPECT_FLOAT_EQ(result.inputs[1].values[0], 2.5f);
    EXPECT_FLOAT_EQ(result.inputs[1].values[1], -4.0f);
}

TEST(CliTest, MissingModelOptionIsAUsageError) {
    const inferedge::CliResult result = parse({"--input", "A=1,2"});
    EXPECT_EQ(result.action, inferedge::CliAction::kError);
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_NE(result.message.find("missing --model"), std::string::npos);
}

TEST(CliTest, MissingModelPathIsAUsageError) {
    const inferedge::CliResult omitted = parse({"--model"});
    const inferedge::CliResult followed_by_option = parse({"--model", "--input", "A=1"});
    EXPECT_EQ(omitted.exit_code, 2);
    EXPECT_EQ(followed_by_option.exit_code, 2);
    EXPECT_NE(omitted.message.find("missing model path"), std::string::npos);
    EXPECT_NE(followed_by_option.message.find("missing model path"), std::string::npos);
}

TEST(CliTest, EmptyModelPathIsAUsageError) {
    const inferedge::CliResult separate = parse({"--model", "", "--input", "A=1"});
    const inferedge::CliResult attached = parse({"--model=", "--input", "A=1"});
    EXPECT_EQ(separate.exit_code, 2);
    EXPECT_EQ(attached.exit_code, 2);
    EXPECT_NE(separate.message.find("empty model path"), std::string::npos);
    EXPECT_NE(attached.message.find("empty model path"), std::string::npos);
}

TEST(CliTest, MissingInputOptionIsAUsageError) {
    const inferedge::CliResult result = parse({"--model", "models/add_f32.onnx"});
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_NE(result.message.find("missing --input"), std::string::npos);
}

TEST(CliTest, RejectsMalformedInputSyntax) {
    const inferedge::CliResult missing_equals = parse({"--model", "m", "--input", "A"});
    const inferedge::CliResult empty_name = parse({"--model", "m", "--input", "=1,2"});
    const inferedge::CliResult empty_values = parse({"--model", "m", "--input", "A="});
    const inferedge::CliResult repeated_comma = parse({"--model", "m", "--input", "A=1,,2"});
    const inferedge::CliResult trailing_comma = parse({"--model", "m", "--input", "A=1,2,"});
    const inferedge::CliResult duplicate =
        parse({"--model", "m", "--input", "A=1", "--input", "A=2"});
    EXPECT_EQ(missing_equals.exit_code, 2);
    EXPECT_EQ(empty_name.exit_code, 2);
    EXPECT_EQ(empty_values.exit_code, 2);
    EXPECT_EQ(repeated_comma.exit_code, 2);
    EXPECT_EQ(trailing_comma.exit_code, 2);
    EXPECT_EQ(duplicate.exit_code, 2);
    EXPECT_NE(missing_equals.message.find("missing '='"), std::string::npos);
    EXPECT_NE(empty_name.message.find("empty input name"), std::string::npos);
    EXPECT_NE(empty_values.message.find("missing input values"), std::string::npos);
    EXPECT_NE(repeated_comma.message.find("empty value"), std::string::npos);
    EXPECT_NE(trailing_comma.message.find("trailing comma"), std::string::npos);
    EXPECT_NE(duplicate.message.find("duplicate input name: A"), std::string::npos);
}

TEST(CliTest, RejectsMalformedAndNonFiniteNumbers) {
    const inferedge::CliResult malformed = parse({"--model", "m", "--input", "A=1,xyz"});
    const inferedge::CliResult partial = parse({"--model", "m", "--input", "A=1abc"});
    const inferedge::CliResult non_finite = parse({"--model", "m", "--input", "A=nan,1"});
    const inferedge::CliResult overflow = parse({"--model", "m", "--input", "A=1e39"});
    const inferedge::CliResult underflow = parse({"--model", "m", "--input", "A=1e-99"});
    EXPECT_EQ(malformed.exit_code, 2);
    EXPECT_EQ(partial.exit_code, 2);
    EXPECT_EQ(non_finite.exit_code, 2);
    EXPECT_EQ(overflow.exit_code, 2);
    EXPECT_EQ(underflow.exit_code, 2);
    EXPECT_NE(malformed.message.find("invalid floating-point value 'xyz'"), std::string::npos);
    EXPECT_NE(partial.message.find("invalid floating-point value '1abc'"), std::string::npos);
    EXPECT_NE(non_finite.message.find("non-finite"), std::string::npos);
    EXPECT_NE(overflow.message.find("out of range"), std::string::npos);
    EXPECT_NE(underflow.message.find("out of range"), std::string::npos);
    EXPECT_EQ(malformed.inputs.size(), 0u);
}

TEST(CliTest, HelpStaysExclusive) {
    const inferedge::CliResult result = parse({"--model", "m", "--help"});
    EXPECT_EQ(result.action, inferedge::CliAction::kError);
    EXPECT_EQ(result.exit_code, 2);
    EXPECT_NE(result.message.find("unexpected argument"), std::string::npos);
}

TEST(CliTest, NamedInputLimitIs64) {
    std::vector<std::string> accepted{"--model", "m"};
    for (int index = 0; index < static_cast<int>(inferedge::kMaxNamedInputs); ++index) {
        accepted.push_back("--input");
        accepted.push_back("n" + std::to_string(index) + "=0");
    }
    const inferedge::CliResult within_limit = parse_owned(accepted);
    EXPECT_EQ(within_limit.action, inferedge::CliAction::kRun);
    EXPECT_EQ(within_limit.inputs.size(), inferedge::kMaxNamedInputs);

    accepted.push_back("--input");
    accepted.push_back("overflow=0");
    const inferedge::CliResult too_many = parse_owned(accepted);
    EXPECT_EQ(too_many.exit_code, 2);
    EXPECT_NE(too_many.message.find("too many named inputs"), std::string::npos);
}

TEST(CliTest, ValueCountLimitIs1048576) {
    const std::string at_limit = zeros(inferedge::kMaxFloatsPerInput);
    const inferedge::CliResult accepted = parse_owned({"--model", "m", "--input", "A=" + at_limit});
    ASSERT_EQ(accepted.action, inferedge::CliAction::kRun);
    ASSERT_EQ(accepted.inputs.size(), 1u);
    EXPECT_EQ(accepted.inputs[0].values.size(), inferedge::kMaxFloatsPerInput);

    const std::string over_limit = zeros(inferedge::kMaxFloatsPerInput + 1);
    const inferedge::CliResult rejected =
        parse_owned({"--model", "m", "--input", "A=" + over_limit});
    EXPECT_EQ(rejected.exit_code, 2);
    EXPECT_NE(rejected.message.find("exceeds " + std::to_string(inferedge::kMaxFloatsPerInput)),
              std::string::npos);

    const inferedge::CliResult total =
        parse_owned({"--model", "m", "--input", "A=0", "--input", "B=" + at_limit});
    EXPECT_EQ(total.exit_code, 2);
    EXPECT_NE(total.message.find("total values"), std::string::npos);
}

TEST(CliTest, NumericTokenAndArgumentLengthLimits) {
    const std::string token(inferedge::kMaxCharsPerNumber, '0');
    const inferedge::CliResult accepted = parse_owned({"--model", "m", "--input", "A=" + token});
    ASSERT_EQ(accepted.action, inferedge::CliAction::kRun);
    ASSERT_EQ(accepted.inputs.size(), 1u);
    EXPECT_FLOAT_EQ(accepted.inputs[0].values[0], 0.0f);

    const std::string too_long(inferedge::kMaxCharsPerNumber + 1, '0');
    const inferedge::CliResult rejected = parse_owned({"--model", "m", "--input", "A=" + too_long});
    EXPECT_EQ(rejected.exit_code, 2);
    EXPECT_NE(rejected.message.find("maximum length"), std::string::npos);

    const std::string name(inferedge::kMaxInputNameLength, 'n');
    const inferedge::CliResult named = parse_owned({"--model", "m", "--input", name + "=1"});
    EXPECT_EQ(named.action, inferedge::CliAction::kRun);
    const std::string longer_name(inferedge::kMaxInputNameLength + 1, 'n');
    const inferedge::CliResult long_name =
        parse_owned({"--model", "m", "--input", longer_name + "=1"});
    EXPECT_EQ(long_name.exit_code, 2);
    EXPECT_NE(long_name.message.find("input name exceeds maximum length"), std::string::npos);

    const std::string values(inferedge::kMaxInputValuesLength + 1, '0');
    const inferedge::CliResult argument = parse_owned({"--model", "m", "--input", "A=" + values});
    EXPECT_EQ(argument.exit_code, 2);
    EXPECT_NE(argument.message.find("input argument exceeds maximum length"), std::string::npos);
}
