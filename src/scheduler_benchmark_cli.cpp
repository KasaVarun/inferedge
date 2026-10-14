#include "inferedge/scheduler_benchmark_cli.hpp"

#include "inferedge/version.hpp"

#include <charconv>
#include <cmath>
#include <string>
#include <system_error>

namespace inferedge {
namespace {

constexpr int kExitSuccess = 0;
constexpr int kExitUsage = 2;

SchedulerBenchmarkCliResult make_result(SchedulerBenchmarkCliAction action, int exit_code,
                                        std::string message) {
    SchedulerBenchmarkCliResult result;
    result.action = action;
    result.exit_code = exit_code;
    result.message = std::move(message);
    return result;
}

SchedulerBenchmarkCliResult error_result(std::string message) {
    message += scheduler_benchmark_help_message();
    return make_result(SchedulerBenchmarkCliAction::kError, kExitUsage, std::move(message));
}

bool is_help(std::string_view argument) {
    return argument == "--help" || argument == "-h";
}

bool is_version(std::string_view argument) {
    return argument == "--version" || argument == "-V";
}

bool is_option_token(std::string_view argument) {
    if (!argument.starts_with("-") || argument.size() < 2) {
        return false;
    }
    const unsigned char next = static_cast<unsigned char>(argument[1]);
    return next != '.' && (next < '0' || next > '9');
}

std::string take_value(const std::vector<std::string_view>& args, std::size_t& index,
                       std::string_view flag, std::string_view& value) {
    const std::string_view argument = args[index];
    const std::string prefix = std::string{flag} + "=";
    if (argument.starts_with(prefix)) {
        value = argument.substr(prefix.size());
        return {};
    }
    if (argument != flag) {
        return {};
    }
    if (index + 1 >= args.size() || is_option_token(args[index + 1])) {
        return "error: missing value for " + std::string{flag} + "\n";
    }
    ++index;
    value = args[index];
    return {};
}

std::string assign_once(bool& seen, std::string_view flag) {
    if (seen) {
        return "error: " + std::string{flag} + " specified more than once\n";
    }
    seen = true;
    return {};
}

enum class NumberKind {
    kAny,
    kNonZero,
};

std::string parse_u64(std::string_view text, std::uint64_t& value, std::uint64_t maximum,
                      NumberKind kind, std::string_view flag) {
    if (text.empty() || text.front() == '+' || text.front() == '-') {
        return "error: invalid integer for " + std::string{flag} + "\n";
    }
    std::uint64_t parsed = 0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const std::from_chars_result result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end) {
        return "error: invalid integer for " + std::string{flag} + "\n";
    }
    if (kind == NumberKind::kNonZero && parsed == 0) {
        return "error: " + std::string{flag} + " must be positive\n";
    }
    if (parsed > maximum) {
        return "error: " + std::string{flag} + " exceeds the supported maximum\n";
    }
    value = parsed;
    return {};
}

std::string parse_rate(std::string_view text, double& value, std::string_view flag) {
    if (text.empty() || text.front() == '+' || text.front() == '-') {
        return "error: invalid number for " + std::string{flag} + "\n";
    }
    double parsed = 0.0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();
    const std::from_chars_result result = std::from_chars(begin, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end || !std::isfinite(parsed) || parsed <= 0.0 ||
        parsed > kMaxSchedulerBenchmarkRequestsPerSecond) {
        return "error: invalid number for " + std::string{flag} + "\n";
    }
    value = parsed;
    return {};
}

} // namespace

std::string scheduler_benchmark_help_message() {
    return "Usage: inferedge-bench-scheduler [options]\n"
           "\n"
           "Benchmark one InferenceScheduler. --format json is schema 3.\n"
           "--completion-timing selects schema 4 and one pre-publication stamp.\n"
           "Warm-up uses a separate scheduler and is excluded from the report.\n"
           "Open-loop lag above --lag-tolerance-ms stops the rest of the schedule.\n"
           "An outstanding-cap hit skips that opportunity and is not a rejection.\n"
           "\n"
           "Options:\n"
           "  --model <path>                 ONNX model (required)\n"
           "  --model-id <id>                batch_add_f32 (default), mlp_small_f32,\n"
           "                                 mlp_medium_f32, or gemm_f32\n"
           "  --workers <n>                  1..64 (required)\n"
           "  --queue-capacity <n>           1..1024 (required)\n"
           "  --workload <name>              open_loop_constant, open_loop_burst, or closed_loop\n"
           "  --target-rps <rate>            open_loop_constant arrival rate\n"
           "  --burst-high-rps <rate>        open_loop_burst high rate\n"
           "  --burst-low-rps <rate>         open_loop_burst low rate\n"
           "  --burst-high-ms <n>            high-phase length\n"
           "  --burst-low-ms <n>             low-phase length\n"
           "  --producer-count <n>           closed_loop producers, 1..64\n"
           "  --max-offered <n>              measured opportunity cap, 1..100000 (required)\n"
           "  --max-outstanding <n>          unresolved accepted cap (required)\n"
           "  --duration-ms <n>              arrival window, 1..60000 (required)\n"
           "  --warmup <n>                   separate-scheduler warm-up requests (default 0)\n"
           "  --completion-timeout-ms <n>    1..60000 (required)\n"
           "  --lag-tolerance-ms <n>         open-loop lateness limit (default 1000)\n"
           "  --check-interval <n>           success checks (default 1)\n"
           "  --enable-batching              batch ready requests up to --max-batch-size\n"
           "  --max-batch-size <n>           1..32 (default 1)\n"
           "  --max-batch-delay-ms <n>       0..1000 (default 0)\n"
           "  --disable-metrics              leave scheduler metrics off\n"
           "  --completion-timing            schema 4 pre-publication stamps (default off)\n"
           "  --format <text|json>           default text\n"
           "  --help, -h\n"
           "  --version, -V\n";
}

std::string scheduler_benchmark_version_message() {
    std::string message = "inferedge-bench-scheduler ";
    message += version();
    message += "\n";
    return message;
}

SchedulerBenchmarkCliResult
parse_scheduler_benchmark_cli(const std::vector<std::string_view>& args) {
    if (args.empty()) {
        return make_result(SchedulerBenchmarkCliAction::kHelp, kExitSuccess,
                           scheduler_benchmark_help_message());
    }
    for (const std::string_view argument : args) {
        if (is_help(argument)) {
            return make_result(SchedulerBenchmarkCliAction::kHelp, kExitSuccess,
                               scheduler_benchmark_help_message());
        }
        if (is_version(argument)) {
            return make_result(SchedulerBenchmarkCliAction::kVersion, kExitSuccess,
                               scheduler_benchmark_version_message());
        }
    }

    SchedulerBenchmarkCliResult parsed;
    parsed.action = SchedulerBenchmarkCliAction::kRun;
    parsed.exit_code = kExitSuccess;
    parsed.config.enable_metrics = true;
    parsed.config.producer_count = 1;
    parsed.config.max_batch_size = 1;
    parsed.config.check_interval = 1;
    parsed.config.lag_tolerance = std::chrono::milliseconds{1000};
    bool saw_model = false;
    bool saw_model_id = false;
    bool saw_workers = false;
    bool saw_queue = false;
    bool saw_workload = false;
    bool saw_target = false;
    bool saw_high_rps = false;
    bool saw_low_rps = false;
    bool saw_high_ms = false;
    bool saw_low_ms = false;
    bool saw_producers = false;
    bool saw_offered = false;
    bool saw_outstanding = false;
    bool saw_duration = false;
    bool saw_warmup = false;
    bool saw_timeout = false;
    bool saw_lag = false;
    bool saw_interval = false;
    bool saw_batch = false;
    bool saw_delay = false;
    bool saw_format = false;
    bool saw_metrics = false;
    bool saw_timing = false;

    for (std::size_t index = 0; index < args.size(); ++index) {
        const std::string_view argument = args[index];
        std::string_view value;
        auto take = [&](std::string_view flag, bool& seen) -> std::string {
            if (argument != flag && !argument.starts_with(std::string{flag} + "=")) {
                return {};
            }
            if (const std::string duplicated = assign_once(seen, flag); !duplicated.empty()) {
                return duplicated;
            }
            return take_value(args, index, flag, value);
        };
        if (argument == "--enable-batching") {
            parsed.config.enable_batching = true;
            continue;
        }
        if (argument == "--completion-timing") {
            if (const std::string duplicated = assign_once(saw_timing, argument);
                !duplicated.empty()) {
                return error_result(duplicated);
            }
            parsed.config.enable_completion_timing = true;
            continue;
        }
        if (argument == "--disable-metrics") {
            if (const std::string duplicated = assign_once(saw_metrics, argument);
                !duplicated.empty()) {
                return error_result(duplicated);
            }
            parsed.config.enable_metrics = false;
            continue;
        }
        if (argument == "--model-id" || argument.starts_with("--model-id=")) {
            if (const std::string error = take("--model-id", saw_model_id); !error.empty()) {
                return error_result(error);
            }
            if (value == "batch_add_f32") {
                parsed.config.model_id = SchedulerBenchmarkModelId::kBatchAddF32;
            } else if (value == "mlp_small_f32") {
                parsed.config.model_id = SchedulerBenchmarkModelId::kMlpSmallF32;
            } else if (value == "mlp_medium_f32") {
                parsed.config.model_id = SchedulerBenchmarkModelId::kMlpMediumF32;
            } else if (value == "gemm_f32") {
                parsed.config.model_id = SchedulerBenchmarkModelId::kGemmF32;
            } else {
                return error_result("error: unsupported model id '" + std::string{value} + "'\n");
            }
            continue;
        }
        if (argument == "--model" || argument.starts_with("--model=")) {
            if (const std::string error = take("--model", saw_model); !error.empty()) {
                return error_result(error);
            }
            if (value.empty()) {
                return error_result("error: missing value for --model\n");
            }
            parsed.config.model_path = std::string{value};
            continue;
        }
        if (argument == "--workload" || argument.starts_with("--workload=")) {
            if (const std::string error = take("--workload", saw_workload); !error.empty()) {
                return error_result(error);
            }
            if (value == "open_loop_constant") {
                parsed.config.workload = SchedulerWorkloadKind::kOpenLoopConstant;
            } else if (value == "open_loop_burst") {
                parsed.config.workload = SchedulerWorkloadKind::kOpenLoopBurst;
            } else if (value == "closed_loop") {
                parsed.config.workload = SchedulerWorkloadKind::kClosedLoop;
            } else {
                return error_result("error: unsupported workload '" + std::string{value} + "'\n");
            }
            continue;
        }
        if (argument == "--format" || argument.starts_with("--format=")) {
            if (const std::string error = take("--format", saw_format); !error.empty()) {
                return error_result(error);
            }
            if (value == "json") {
                parsed.output_format = SchedulerBenchmarkOutputFormat::kJson;
            } else if (value == "text") {
                parsed.output_format = SchedulerBenchmarkOutputFormat::kText;
            } else {
                return error_result("error: unsupported format '" + std::string{value} + "'\n");
            }
            continue;
        }

        struct U64Flag {
            std::string_view name;
            bool* seen;
            std::uint64_t* destination;
            std::uint64_t maximum;
            NumberKind kind;
        };
        bool matched = false;
        const U64Flag flags[] = {
            {"--workers", &saw_workers, nullptr, kMaxSchedulerWorkers, NumberKind::kNonZero},
            {"--queue-capacity", &saw_queue, nullptr, kMaxSchedulerQueueCapacity,
             NumberKind::kNonZero},
            {"--max-offered", &saw_offered, &parsed.config.max_offered,
             kMaxSchedulerBenchmarkOffered, NumberKind::kNonZero},
            {"--max-outstanding", &saw_outstanding, &parsed.config.max_outstanding,
             kMaxSchedulerBenchmarkOffered, NumberKind::kNonZero},
            {"--duration-ms", &saw_duration, nullptr, kMaxSchedulerBenchmarkDurationMs,
             NumberKind::kNonZero},
            {"--warmup", &saw_warmup, &parsed.config.warmup_requests, kMaxSchedulerBenchmarkOffered,
             NumberKind::kAny},
            {"--completion-timeout-ms", &saw_timeout, nullptr, kMaxSchedulerBenchmarkDurationMs,
             NumberKind::kNonZero},
            {"--lag-tolerance-ms", &saw_lag, nullptr, 60000, NumberKind::kAny},
            {"--check-interval", &saw_interval, nullptr, 1000000, NumberKind::kNonZero},
            {"--max-batch-size", &saw_batch, nullptr, kSchedulerMetricsMaxBatchSize,
             NumberKind::kNonZero},
            {"--max-batch-delay-ms", &saw_delay, nullptr, 1000, NumberKind::kAny},
            {"--producer-count", &saw_producers, nullptr, kMaxSchedulerBenchmarkProducers,
             NumberKind::kNonZero},
            {"--burst-high-ms", &saw_high_ms, nullptr, kMaxSchedulerBenchmarkDurationMs,
             NumberKind::kNonZero},
            {"--burst-low-ms", &saw_low_ms, nullptr, kMaxSchedulerBenchmarkDurationMs,
             NumberKind::kNonZero},
        };
        for (const U64Flag& flag : flags) {
            if (argument != flag.name && !argument.starts_with(std::string{flag.name} + "=")) {
                continue;
            }
            matched = true;
            if (const std::string error = take(flag.name, *flag.seen); !error.empty()) {
                return error_result(error);
            }
            std::uint64_t number = 0;
            if (const std::string error =
                    parse_u64(value, number, flag.maximum, flag.kind, flag.name);
                !error.empty()) {
                return error_result(error);
            }
            if (flag.destination != nullptr) {
                *flag.destination = number;
            } else if (flag.name == "--workers") {
                parsed.config.worker_count = static_cast<std::uint32_t>(number);
            } else if (flag.name == "--queue-capacity") {
                parsed.config.queue_capacity = static_cast<std::size_t>(number);
            } else if (flag.name == "--duration-ms") {
                parsed.config.duration =
                    std::chrono::milliseconds{static_cast<std::int64_t>(number)};
            } else if (flag.name == "--completion-timeout-ms") {
                parsed.config.completion_timeout =
                    std::chrono::milliseconds{static_cast<std::int64_t>(number)};
            } else if (flag.name == "--lag-tolerance-ms") {
                parsed.config.lag_tolerance =
                    std::chrono::milliseconds{static_cast<std::int64_t>(number)};
            } else if (flag.name == "--check-interval") {
                parsed.config.check_interval = static_cast<std::uint32_t>(number);
            } else if (flag.name == "--max-batch-size") {
                parsed.config.max_batch_size = static_cast<std::uint32_t>(number);
            } else if (flag.name == "--max-batch-delay-ms") {
                parsed.config.max_batch_delay =
                    std::chrono::milliseconds{static_cast<std::int64_t>(number)};
            } else if (flag.name == "--producer-count") {
                parsed.config.producer_count = static_cast<std::uint32_t>(number);
            } else if (flag.name == "--burst-high-ms") {
                parsed.config.burst_high =
                    std::chrono::milliseconds{static_cast<std::int64_t>(number)};
            } else if (flag.name == "--burst-low-ms") {
                parsed.config.burst_low =
                    std::chrono::milliseconds{static_cast<std::int64_t>(number)};
            }
            break;
        }
        if (matched) {
            continue;
        }
        if (argument == "--target-rps" || argument.starts_with("--target-rps=")) {
            if (const std::string error = take("--target-rps", saw_target); !error.empty()) {
                return error_result(error);
            }
            if (const std::string error =
                    parse_rate(value, parsed.config.target_requests_per_second, "--target-rps");
                !error.empty()) {
                return error_result(error);
            }
            continue;
        }
        if (argument == "--burst-high-rps" || argument.starts_with("--burst-high-rps=")) {
            if (const std::string error = take("--burst-high-rps", saw_high_rps); !error.empty()) {
                return error_result(error);
            }
            if (const std::string error = parse_rate(
                    value, parsed.config.burst_high_requests_per_second, "--burst-high-rps");
                !error.empty()) {
                return error_result(error);
            }
            continue;
        }
        if (argument == "--burst-low-rps" || argument.starts_with("--burst-low-rps=")) {
            if (const std::string error = take("--burst-low-rps", saw_low_rps); !error.empty()) {
                return error_result(error);
            }
            if (const std::string error = parse_rate(
                    value, parsed.config.burst_low_requests_per_second, "--burst-low-rps");
                !error.empty()) {
                return error_result(error);
            }
            continue;
        }
        return error_result("error: unexpected argument '" + std::string{argument} + "'\n");
    }

    if (!saw_model || !saw_workers || !saw_queue || !saw_workload || !saw_offered ||
        !saw_outstanding || !saw_duration || !saw_timeout) {
        return error_result("error: missing required option\n");
    }
    if (parsed.config.workload == SchedulerWorkloadKind::kOpenLoopConstant) {
        if (!saw_target || saw_high_rps || saw_low_rps || saw_high_ms || saw_low_ms) {
            return error_result("error: open_loop_constant requires --target-rps only\n");
        }
        if (parsed.config.producer_count != 1) {
            return error_result("error: open-loop traffic uses one generator\n");
        }
    } else if (parsed.config.workload == SchedulerWorkloadKind::kOpenLoopBurst) {
        if (!saw_high_rps || !saw_low_rps || !saw_high_ms || !saw_low_ms || saw_target) {
            return error_result("error: open_loop_burst requires high and low rate and duration\n");
        }
        if (parsed.config.producer_count != 1) {
            return error_result("error: open-loop traffic uses one generator\n");
        }
    } else if (saw_target || saw_high_rps || saw_low_rps || saw_high_ms || saw_low_ms) {
        return error_result("error: closed_loop does not take an open-loop schedule\n");
    }
    if (!saw_producers) {
        parsed.config.producer_count = 1;
    }
    if (parsed.config.max_outstanding > parsed.config.max_offered) {
        return error_result("error: --max-outstanding cannot exceed --max-offered\n");
    }
    return parsed;
}

} // namespace inferedge
