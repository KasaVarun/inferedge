#include "inferedge/cli.hpp"

#include "inferedge/version.hpp"

#include <charconv>
#include <cmath>
#include <system_error>
#include <utility>

namespace inferedge {
namespace {

constexpr int kExitSuccess = 0;
constexpr int kExitUsage = 2;

std::string help_message() {
    std::string message;
    message += "Usage: ";
    message += project_name();
    message += " [options]\n";
    message += "\n";
    message += "InferEdge is a high-performance C++ inference runtime.\n";
    message += "\n";
    message += "Options:\n";
    message += "  -h, --help                 Show this help and exit\n";
    message += "  -V, --version              Show the version and exit\n";
    message += "  --model <path>             ONNX model to run\n";
    message += "  --input <name>=<values>    Named float32 input, comma-separated\n";
    message += "\n";
    message += "Repeat --input once for each model input. With no arguments, this help\n";
    message += "is shown. --help and --version cannot be combined with other arguments.\n";
    return message;
}

std::string version_message() {
    std::string message;
    message += project_name();
    message += " ";
    message += version();
    message += "\n";
    return message;
}

CliResult make_result(CliAction action, int exit_code, std::string message) {
    CliResult result;
    result.action = action;
    result.exit_code = exit_code;
    result.message = std::move(message);
    return result;
}

CliResult error_result(std::string message) {
    message += help_message();
    return make_result(CliAction::kError, kExitUsage, std::move(message));
}

CliResult unexpected_argument(std::string_view argument) {
    std::string message = "error: unexpected argument '";
    message += argument;
    message += "'\n";
    return error_result(std::move(message));
}

bool is_help(std::string_view argument) {
    return argument == "--help" || argument == "-h";
}

bool is_version(std::string_view argument) {
    return argument == "--version" || argument == "-V";
}

bool is_option_token(std::string_view argument) {
    return argument.starts_with("-");
}

bool has_input_name(const std::vector<NamedFloatInput>& inputs, std::string_view name) {
    for (const NamedFloatInput& input : inputs) {
        if (input.name == name) {
            return true;
        }
    }
    return false;
}

std::string parse_float_token(std::string_view token, std::string_view name, float& value) {
    if (token.size() > kMaxCharsPerNumber) {
        return "error: floating-point token exceeds maximum length in input " + std::string{name} +
               "\n";
    }

    const char* const first = token.data();
    const char* const last = token.data() + token.size();
    const std::from_chars_result parsed =
        std::from_chars(first, last, value, std::chars_format::general);
    if (parsed.ptr != last) {
        std::string message = "error: invalid floating-point value '";
        message += token;
        message += "' in input ";
        message += name;
        message += "\n";
        return message;
    }
    if (parsed.ec == std::errc::result_out_of_range) {
        return "error: floating-point value out of range in input " + std::string{name} + "\n";
    }
    if (parsed.ec != std::errc{}) {
        std::string message = "error: invalid floating-point value '";
        message += token;
        message += "' in input ";
        message += name;
        message += "\n";
        return message;
    }
    if (!std::isfinite(value)) {
        return "error: non-finite floating-point value in input " + std::string{name} + "\n";
    }
    return {};
}

std::string parse_input_values(std::string_view name, std::string_view values,
                               std::vector<float>& parsed_values) {
    if (values.empty()) {
        return "error: missing input values for " + std::string{name} + "\n";
    }
    if (values.size() > kMaxInputValuesLength) {
        return "error: input argument exceeds maximum length for " + std::string{name} + "\n";
    }

    std::size_t begin = 0;
    while (true) {
        const std::size_t comma = values.find(',', begin);
        const bool last = comma == std::string_view::npos;
        const std::string_view token =
            values.substr(begin, last ? std::string_view::npos : comma - begin);
        if (token.empty()) {
            if (last) {
                return "error: trailing comma in input " + std::string{name} + "\n";
            }
            return "error: empty value in input " + std::string{name} + "\n";
        }
        if (parsed_values.size() == kMaxFloatsPerInput) {
            return "error: input " + std::string{name} + " exceeds " +
                   std::to_string(kMaxFloatsPerInput) + " values\n";
        }

        float value = 0.0f;
        std::string error = parse_float_token(token, name, value);
        if (!error.empty()) {
            return error;
        }
        parsed_values.push_back(value);
        if (last) {
            return {};
        }
        begin = comma + 1;
    }
}

std::string take_attached_or_next(const std::vector<std::string_view>& args, std::size_t& index,
                                  std::string_view flag, std::string_view attached_prefix,
                                  std::string_view& value, bool& present) {
    const std::string_view argument = args[index];
    if (argument.starts_with(attached_prefix)) {
        value = argument.substr(attached_prefix.size());
        present = true;
        return {};
    }
    if (argument != flag) {
        present = false;
        return {};
    }
    if (index + 1 >= args.size() || is_option_token(args[index + 1])) {
        if (flag == "--model") {
            return "error: missing model path\n";
        }
        return "error: missing value for --input\n";
    }
    ++index;
    value = args[index];
    present = true;
    return {};
}

} // namespace

CliResult parse_cli(const std::vector<std::string_view>& args) {
    if (args.empty()) {
        return make_result(CliAction::kHelp, kExitSuccess, help_message());
    }

    if (is_help(args.front()) || is_version(args.front())) {
        if (args.size() > 1) {
            return unexpected_argument(args[1]);
        }
        if (is_help(args.front())) {
            return make_result(CliAction::kHelp, kExitSuccess, help_message());
        }
        return make_result(CliAction::kVersion, kExitSuccess, version_message());
    }

    std::string model_path;
    bool saw_model = false;
    std::vector<NamedFloatInput> inputs;
    std::size_t total_values = 0;

    for (std::size_t index = 0; index < args.size(); ++index) {
        const std::string_view argument = args[index];
        if (is_help(argument) || is_version(argument)) {
            return unexpected_argument(argument);
        }

        std::string_view model_value;
        bool model_present = false;
        const std::string model_error =
            take_attached_or_next(args, index, "--model", "--model=", model_value, model_present);
        if (!model_error.empty()) {
            return error_result(model_error);
        }
        if (model_present) {
            if (saw_model) {
                return error_result("error: model path specified more than once\n");
            }
            if (model_value.empty()) {
                return error_result("error: empty model path\n");
            }
            saw_model = true;
            model_path.assign(model_value);
            continue;
        }

        std::string_view input_value;
        bool input_present = false;
        const std::string input_error =
            take_attached_or_next(args, index, "--input", "--input=", input_value, input_present);
        if (!input_error.empty()) {
            return error_result(input_error);
        }
        if (input_present) {
            const std::size_t equals = input_value.find('=');
            if (equals == std::string_view::npos) {
                return error_result("error: missing '=' in input\n");
            }
            const std::string_view name = input_value.substr(0, equals);
            const std::string_view values = input_value.substr(equals + 1);
            if (name.empty()) {
                return error_result("error: empty input name\n");
            }
            if (name.size() > kMaxInputNameLength) {
                return error_result("error: input name exceeds maximum length\n");
            }
            if (inputs.size() == kMaxNamedInputs) {
                return error_result("error: too many named inputs\n");
            }
            if (has_input_name(inputs, name)) {
                std::string message = "error: duplicate input name: ";
                message += name;
                message += "\n";
                return error_result(std::move(message));
            }

            NamedFloatInput input;
            input.name.assign(name);
            const std::string value_error = parse_input_values(name, values, input.values);
            if (!value_error.empty()) {
                return error_result(value_error);
            }
            if (total_values + input.values.size() > kMaxTotalFloats) {
                return error_result("error: inputs exceed " + std::to_string(kMaxTotalFloats) +
                                    " total values\n");
            }
            total_values += input.values.size();
            inputs.push_back(std::move(input));
            continue;
        }

        if (is_option_token(argument)) {
            std::string message = "error: unknown option '";
            message += argument;
            message += "'\n";
            return error_result(std::move(message));
        }
        return unexpected_argument(argument);
    }

    if (!saw_model) {
        return error_result("error: missing --model\n");
    }
    if (inputs.empty()) {
        return error_result("error: missing --input\n");
    }

    CliResult result;
    result.action = CliAction::kRun;
    result.exit_code = kExitSuccess;
    result.model_path = std::move(model_path);
    result.inputs = std::move(inputs);
    return result;
}

} // namespace inferedge
