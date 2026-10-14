#include "inferedge/scheduler_benchmark_cli.hpp"
#include "inferedge/scheduler_benchmark_json.hpp"
#include "inferedge/version.hpp"

#include <string>
#include <utility>

namespace inferedge {
namespace {

std::string text_report(const SchedulerBenchmarkReport& report) {
    std::string text = "workload ";
    text += report.workload;
    text += "\nvalid ";
    text += report.valid ? "true\n" : "false\n";
    text += "succeeded ";
    text += std::to_string(report.succeeded);
    text += "\nengine_run_count ";
    text += std::to_string(report.metrics.engine_run_count);
    text += "\ncompletion_timing ";
    text += report.completion_timing_enabled ? "true\n" : "false\n";
    return text;
}

const char* error_kind_name(SchedulerBenchmarkErrorKind kind) {
    switch (kind) {
    case SchedulerBenchmarkErrorKind::kNone:
        return "none";
    case SchedulerBenchmarkErrorKind::kInvalidConfig:
        return "invalid_config";
    case SchedulerBenchmarkErrorKind::kModelLoad:
        return "model_load";
    case SchedulerBenchmarkErrorKind::kRuntime:
        return "runtime";
    }
    return "runtime";
}

} // namespace

SchedulerBenchmarkCliResult execute_scheduler_benchmark(const SchedulerBenchmarkCliResult& parsed) {
    const SchedulerBenchmarkRun run = run_scheduler_benchmark(parsed.config);
    SchedulerBenchmarkCliResult result = parsed;
    if (parsed.output_format == SchedulerBenchmarkOutputFormat::kJson) {
        SchedulerBenchmarkJsonResult json;
        if (run.kind == SchedulerBenchmarkErrorKind::kNone) {
            json = write_scheduler_benchmark_json(run.report, version());
        } else {
            json = write_scheduler_benchmark_failure_json(error_kind_name(run.kind), run.message,
                                                          version());
        }
        if (!json.error.empty()) {
            result.action = SchedulerBenchmarkCliAction::kError;
            result.exit_code = 1;
            result.message = json.error + "\n";
            return result;
        }
        result.action = SchedulerBenchmarkCliAction::kRun;
        result.exit_code = scheduler_benchmark_status(run);
        result.message = std::move(json.text);
        return result;
    }
    if (run.kind != SchedulerBenchmarkErrorKind::kNone || !run.report.valid) {
        result.action = SchedulerBenchmarkCliAction::kError;
        result.exit_code = scheduler_benchmark_status(run);
        result.message =
            run.message.empty() ? "scheduler benchmark trial is invalid\n" : run.message + "\n";
        return result;
    }
    result.action = SchedulerBenchmarkCliAction::kRun;
    result.exit_code = 0;
    result.message = text_report(run.report);
    return result;
}

} // namespace inferedge
