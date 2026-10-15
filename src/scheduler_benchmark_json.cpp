#include "inferedge/scheduler_benchmark_json.hpp"

#include <charconv>
#include <cmath>
#include <string>
#include <system_error>

namespace inferedge {
namespace {

constexpr const char* kToolName = "inferedge-bench-scheduler";

struct Writer {
    std::string text;
    bool ok{true};

    void raw(std::string_view value) { text += value; }

    void string(std::string_view value) {
        constexpr char kHex[] = "0123456789abcdef";
        text.push_back('"');
        for (const char character : value) {
            const auto byte = static_cast<unsigned char>(character);
            if (byte == '"') {
                text += "\\\"";
            } else if (byte == '\\') {
                text += "\\\\";
            } else if (byte <= 0x1FU) {
                text += "\\u00";
                text.push_back(kHex[byte >> 4U]);
                text.push_back(kHex[byte & 0x0FU]);
            } else {
                text.push_back(character);
            }
        }
        text.push_back('"');
    }

    void boolean(bool value) { text += value ? "true" : "false"; }

    void u64(std::uint64_t value) {
        char buffer[32];
        const std::to_chars_result result = std::to_chars(buffer, buffer + sizeof(buffer), value);
        if (result.ec != std::errc{}) {
            ok = false;
            return;
        }
        text.append(buffer, static_cast<std::size_t>(result.ptr - buffer));
    }

    void i64(std::int64_t value) {
        char buffer[32];
        const std::to_chars_result result = std::to_chars(buffer, buffer + sizeof(buffer), value);
        if (result.ec != std::errc{}) {
            ok = false;
            return;
        }
        text.append(buffer, static_cast<std::size_t>(result.ptr - buffer));
    }

    void number(double value) {
        if (!std::isfinite(value)) {
            ok = false;
            return;
        }
        char buffer[128];
        const std::to_chars_result result =
            std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general);
        if (result.ec != std::errc{}) {
            ok = false;
            return;
        }
        text.append(buffer, static_cast<std::size_t>(result.ptr - buffer));
    }
};

bool finite_report(const SchedulerBenchmarkReport& report) {
    const SchedulerBenchmarkThroughput& throughput = report.throughput;
    if (report.completion_timing_enabled &&
        (!std::isfinite(report.pre_publication_latency.mean_nanoseconds) ||
         !std::isfinite(report.pre_publication_to_observation_delay.mean_nanoseconds))) {
        return false;
    }
    if (!std::isfinite(report.generator_lag.mean_nanoseconds) ||
        !std::isfinite(report.latency.mean_microseconds) ||
        !std::isfinite(report.latency.p50_microseconds) ||
        !std::isfinite(report.latency.p95_microseconds) ||
        !std::isfinite(report.latency.p99_microseconds) ||
        !std::isfinite(throughput.arrival_window_seconds) ||
        !std::isfinite(throughput.completion_seconds) ||
        !std::isfinite(throughput.total_trial_seconds) ||
        !std::isfinite(throughput.offered_per_arrival_window_second) ||
        !std::isfinite(throughput.attempts_per_arrival_window_second) ||
        !std::isfinite(throughput.accepted_per_arrival_window_second) ||
        !std::isfinite(throughput.successful_per_arrival_window_second) ||
        !std::isfinite(throughput.successful_per_total_trial_second)) {
        return false;
    }
    for (const SchedulerBenchmarkPhase& phase : report.phases) {
        if (!std::isfinite(phase.requests_per_second)) {
            return false;
        }
    }
    return true;
}

SchedulerBenchmarkJsonResult failure(const char* message) {
    SchedulerBenchmarkJsonResult result;
    result.error = message;
    return result;
}

} // namespace

SchedulerBenchmarkJsonResult write_scheduler_benchmark_failure_json(std::string_view error_kind,
                                                                    std::string_view message,
                                                                    std::string_view tool_version) {
    Writer writer;
    writer.raw("{\n  \"schema_version\": 3,\n  \"tool\": ");
    writer.string(kToolName);
    writer.raw(",\n  \"tool_version\": ");
    writer.string(tool_version);
    writer.raw(",\n  \"valid\": false,\n  \"error\": ");
    writer.string(error_kind);
    writer.raw(",\n  \"message\": ");
    writer.string(message);
    writer.raw("\n}\n");
    SchedulerBenchmarkJsonResult result;
    result.text = std::move(writer.text);
    return result;
}

SchedulerBenchmarkJsonResult write_scheduler_benchmark_json(const SchedulerBenchmarkReport& report,
                                                            std::string_view tool_version) {
    if (!finite_report(report)) {
        return failure("JSON numbers must be finite");
    }
    Writer writer;
    writer.raw("{\n  \"schema_version\": ");
    writer.u64(report.completion_timing_enabled ? 4 : 3);
    writer.raw(",\n  \"tool\": ");
    writer.string(kToolName);
    writer.raw(",\n  \"tool_version\": ");
    writer.string(tool_version);
    writer.raw(",\n  \"valid\": ");
    writer.boolean(report.valid);
    writer.raw(",\n  \"invalid_reasons\": [");
    for (std::size_t index = 0; index < report.invalid_reasons.size(); ++index) {
        if (index != 0) {
            writer.raw(", ");
        }
        writer.string(report.invalid_reasons[index]);
    }
    writer.raw("],\n  \"model\": ");
    writer.string(report.model);
    writer.raw(",\n  \"workload\": ");
    writer.string(report.workload);
    writer.raw(",\n  \"e2e_latency_definition\": \"steady_clock immediately before submit() until "
               "the future is observed ready\",\n  \"completion_observation\": ");
    writer.string(report.completion_observation);
    writer.raw(",\n  \"observation_quantum_nanoseconds\": ");
    writer.i64(report.observation_quantum.count());
    writer.raw(",\n  \"scheduler\": {\n    \"worker_count\": ");
    writer.u64(report.worker_count);
    writer.raw(",\n    \"queue_capacity\": ");
    writer.u64(report.queue_capacity);
    writer.raw(",\n    \"enable_batching\": ");
    writer.boolean(report.enable_batching);
    writer.raw(",\n    \"max_batch_size\": ");
    writer.u64(report.max_batch_size);
    writer.raw(",\n    \"max_batch_delay_ms\": ");
    writer.i64(report.max_batch_delay.count());
    writer.raw(",\n    \"adaptive_batching\": ");
    writer.boolean(report.adaptive_batching);
    writer.raw(",\n    \"arrival_aware_batching\": ");
    writer.boolean(report.arrival_aware_batching);
    writer.raw(",\n    \"enable_metrics\": ");
    writer.boolean(report.enable_metrics);
    if (report.completion_timing_enabled) {
        writer.raw(",\n    \"enable_completion_timing\": true");
    }
    writer.raw("\n  },\n  \"generator\": {\n    \"lag_definition\": "
               "\"actual_submit_start_minus_scheduled_arrival\",\n    \"lag_policy\": "
               "\"stop_and_record\",\n    \"outstanding_policy\": \"skip_opportunity\",\n    "
               "\"warmup_requests\": ");
    writer.u64(report.warmup_requests);
    writer.raw(",\n    \"warmup_separate_scheduler\": ");
    writer.boolean(report.warmup_uses_separate_scheduler);
    writer.raw(",\n    \"check_interval\": ");
    writer.u64(report.check_interval);
    writer.raw("\n  },\n  \"configured_duration_nanoseconds\": ");
    writer.i64(report.configured_duration.count());
    writer.raw(",\n  \"scheduled_span_nanoseconds\": ");
    writer.i64(report.scheduled_span.count());
    writer.raw(",\n  \"arrival_stop_reason\": ");
    writer.string(report.arrival_stop_reason);
    writer.raw(",\n  \"arrival_window_nanoseconds\": ");
    writer.i64(report.arrival_window.count());
    writer.raw(",\n  \"completion_nanoseconds\": ");
    writer.i64(report.completion.count());
    writer.raw(",\n  \"total_trial_nanoseconds\": ");
    writer.i64(report.total_trial.count());
    writer.raw(",\n  \"completion_timeout_bound\": \"future_observation_only\",\n  "
               "\"completion_timeout_note\": \"completion_timeout bounds future observation. "
               "shutdown(kDrain) can still block inside an InferenceEngine::run that has already "
               "started. This process has no external hard timeout.\",\n  "
               "\"outstanding_cap_note\": \"A collector that falls behind keeps accepted futures "
               "inside the cap and skips later scheduled opportunities. Those skips are not "
               "scheduler rejections. Suppressing at least half of the schedule invalidates an "
               "open-loop trial.\",");
    writer.raw("\n  \"scheduled_opportunities\": ");
    writer.u64(report.scheduled_opportunities);
    writer.raw(",\n  \"submit_attempts\": ");
    writer.u64(report.submit_attempts);
    writer.raw(",\n  \"generator_skips_outstanding\": ");
    writer.u64(report.generator_skips_outstanding);
    writer.raw(",\n  \"generator_skips_lag\": ");
    writer.u64(report.generator_skips_lag);
    writer.raw(",\n  \"accepted\": ");
    writer.u64(report.accepted);
    writer.raw(",\n  \"rejections\": {\n    \"full\": ");
    writer.u64(report.rejections.full);
    writer.raw(",\n    \"closed\": ");
    writer.u64(report.rejections.closed);
    writer.raw(",\n    \"not_running\": ");
    writer.u64(report.rejections.not_running);
    writer.raw("\n  },\n  \"succeeded\": ");
    writer.u64(report.succeeded);
    writer.raw(",\n  \"failed_inference\": ");
    writer.u64(report.failed_inference);
    writer.raw(",\n  \"failed_worker\": ");
    writer.u64(report.failed_worker);
    writer.raw(",\n  \"failed_scheduler\": ");
    writer.u64(report.failed_scheduler);
    writer.raw(",\n  \"canceled\": ");
    writer.u64(report.canceled);
    writer.raw(",\n  \"timeouts\": ");
    writer.u64(report.timeouts);
    writer.raw(",\n  \"generator_lag\": {\n    \"count\": ");
    writer.u64(report.generator_lag.count);
    writer.raw(",\n    \"minimum_nanoseconds\": ");
    writer.i64(report.generator_lag.minimum.count());
    writer.raw(",\n    \"maximum_nanoseconds\": ");
    writer.i64(report.generator_lag.maximum.count());
    writer.raw(",\n    \"mean_nanoseconds\": ");
    writer.number(report.generator_lag.mean_nanoseconds);
    if (report.completion_timing_enabled) {
        writer.raw(",\n    \"p50_nanoseconds\": ");
        writer.i64(report.generator_lag.p50_nanoseconds);
        writer.raw(",\n    \"p95_nanoseconds\": ");
        writer.i64(report.generator_lag.p95_nanoseconds);
        writer.raw(",\n    \"p99_nanoseconds\": ");
        writer.i64(report.generator_lag.p99_nanoseconds);
        writer.raw(",\n    \"percentiles_reported\": ");
        writer.boolean(report.generator_lag.percentiles_reported);
        writer.raw(",\n    \"precise_tail\": ");
        writer.boolean(report.generator_lag.precise_tail);
    }
    writer.raw("\n  },\n  \"successful_e2e_latency\": {\n    \"sample_count\": ");
    writer.u64(report.latency.sample_count);
    writer.raw(",\n    \"minimum_nanoseconds\": ");
    writer.i64(report.latency.minimum.count());
    writer.raw(",\n    \"maximum_nanoseconds\": ");
    writer.i64(report.latency.maximum.count());
    writer.raw(",\n    \"mean_microseconds\": ");
    writer.number(report.latency.mean_microseconds);
    writer.raw(",\n    \"p50_microseconds\": ");
    writer.number(report.latency.p50_microseconds);
    writer.raw(",\n    \"p95_microseconds\": ");
    writer.number(report.latency.p95_microseconds);
    writer.raw(",\n    \"p99_microseconds\": ");
    writer.number(report.latency.p99_microseconds);
    writer.raw(",\n    \"percentiles_reported\": ");
    writer.boolean(report.latency.percentiles_reported);
    writer.raw(",\n    \"precise_tail\": ");
    writer.boolean(report.latency.precise_tail);
    writer.raw("\n  }");
    if (report.completion_timing_enabled) {
        auto write_nanos = [&writer](const char* name, const char* definition,
                                     const SchedulerBenchmarkNanosecondSummary& summary) {
            writer.raw(",\n  \"");
            writer.raw(name);
            writer.raw("\": {\n    \"definition\": ");
            writer.string(definition);
            writer.raw(",\n    \"unit\": \"nanoseconds\",\n    \"sample_count\": ");
            writer.u64(summary.sample_count);
            writer.raw(",\n    \"minimum_nanoseconds\": ");
            writer.i64(summary.minimum.count());
            writer.raw(",\n    \"maximum_nanoseconds\": ");
            writer.i64(summary.maximum.count());
            writer.raw(",\n    \"mean_nanoseconds\": ");
            writer.number(summary.mean_nanoseconds);
            writer.raw(",\n    \"p50_nanoseconds\": ");
            writer.i64(summary.p50_nanoseconds);
            writer.raw(",\n    \"p95_nanoseconds\": ");
            writer.i64(summary.p95_nanoseconds);
            writer.raw(",\n    \"p99_nanoseconds\": ");
            writer.i64(summary.p99_nanoseconds);
            writer.raw(",\n    \"percentiles_reported\": ");
            writer.boolean(summary.percentiles_reported);
            writer.raw(",\n    \"precise_tail\": ");
            writer.boolean(summary.precise_tail);
            writer.raw("\n  }");
        };
        write_nanos("pre_publication_latency",
                    "steady_clock immediately before submit() until the pre-publication stamp",
                    report.pre_publication_latency);
        write_nanos("pre_publication_to_observation_delay",
                    "pre-publication stamp until the future is observed ready; includes set_value "
                    "and collector delay, not pure polling delay",
                    report.pre_publication_to_observation_delay);
        writer.raw(
            ",\n  \"completion_timing_enabled\": true,\n  \"steady_clock_period_numerator\": ");
        writer.i64(static_cast<std::int64_t>(report.steady_clock_period_num));
        writer.raw(",\n  \"steady_clock_period_denominator\": ");
        writer.i64(static_cast<std::int64_t>(report.steady_clock_period_den));
        writer.raw(",\n  \"steady_clock_period_note\": \"nominal steady_clock tick, not calibrated "
                   "accuracy\",\n  \"phase_boundary_fidelity\": \"unavailable\",\n  "
                   "\"timing_outcome_counts\": {\n    \"inference_failure\": ");
        writer.u64(report.timing_inference_failures);
        writer.raw(",\n    \"worker_failure\": ");
        writer.u64(report.timing_worker_failures);
        writer.raw(",\n    \"scheduler_failure\": ");
        writer.u64(report.timing_scheduler_failures);
        writer.raw(",\n    \"canceled\": ");
        writer.u64(report.timing_cancellations);
        writer.raw(",\n    \"backstop\": ");
        writer.u64(report.timing_backstops);
        writer.raw("\n  }");
    }
    writer.raw(",\n  \"throughput\": {\n    \"arrival_window_seconds\": ");
    writer.number(report.throughput.arrival_window_seconds);
    writer.raw(",\n    \"completion_seconds\": ");
    writer.number(report.throughput.completion_seconds);
    writer.raw(",\n    \"total_trial_seconds\": ");
    writer.number(report.throughput.total_trial_seconds);
    writer.raw(",\n    \"offered_opportunities_per_arrival_window_second\": ");
    writer.number(report.throughput.offered_per_arrival_window_second);
    writer.raw(",\n    \"submit_attempts_per_arrival_window_second\": ");
    writer.number(report.throughput.attempts_per_arrival_window_second);
    writer.raw(",\n    \"accepted_per_arrival_window_second\": ");
    writer.number(report.throughput.accepted_per_arrival_window_second);
    writer.raw(",\n    \"successful_attempts_per_arrival_window_second\": ");
    writer.number(report.throughput.successful_per_arrival_window_second);
    writer.raw(",\n    \"successful_per_total_trial_second\": ");
    writer.number(report.throughput.successful_per_total_trial_second);
    writer.raw(",\n    \"successful_attempts_per_arrival_window_second_note\": \"offered-cohort "
               "rate; not steady-state completion capacity\",\n    "
               "\"successful_per_total_trial_second_note\": \"measured successes divided by the "
               "arrival window plus the completion drain\"\n  },\n  \"metrics\": {\n    "
               "\"enabled\": ");
    writer.boolean(report.metrics.enabled);
    writer.raw(",\n    \"finalized\": ");
    writer.boolean(report.metrics.finalized);
    writer.raw(",\n    \"engine_run_count\": ");
    writer.u64(report.metrics.engine_run_count);
    writer.raw(",\n    \"requests_entered_engine_runs\": ");
    writer.u64(report.metrics.requests_entered_engine_runs);
    writer.raw(",\n    \"batch_size_histogram\": [");
    for (std::size_t index = 0; index < report.metrics.batch_size_histogram.size(); ++index) {
        if (index != 0) {
            writer.raw(", ");
        }
        writer.u64(report.metrics.batch_size_histogram[index]);
    }
    writer.raw("],\n    \"total_assembly_nanoseconds\": ");
    writer.u64(report.metrics.total_assembly_nanoseconds);
    writer.raw(",\n    \"total_engine_run_nanoseconds\": ");
    writer.u64(report.metrics.total_engine_run_nanoseconds);
    writer.raw(",\n    \"total_split_nanoseconds\": ");
    writer.u64(report.metrics.total_split_nanoseconds);
    writer.raw(",\n    \"histogram_saturated\": ");
    writer.boolean(report.histogram_saturated);
    writer.raw("\n  },\n  \"correctness_checked\": ");
    writer.u64(report.correctness_checked);
    writer.raw(",\n  \"correctness_mismatches\": ");
    writer.u64(report.correctness_mismatches);
    writer.raw(",\n  \"error_count\": ");
    writer.u64(report.error_count);
    writer.raw(",\n  \"errors\": [");
    for (std::size_t index = 0; index < report.errors.size(); ++index) {
        if (index != 0) {
            writer.raw(", ");
        }
        const SchedulerBenchmarkErrorRecord& error = report.errors[index];
        writer.raw("{\"attempt_id\": ");
        writer.u64(error.attempt_id);
        writer.raw(", \"kind\": ");
        writer.string(error.kind);
        writer.raw(", \"message\": ");
        writer.string(error.message);
        writer.raw("}");
    }
    writer.raw("],\n  \"burst_phases\": [");
    for (std::size_t index = 0; index < report.phases.size(); ++index) {
        if (index != 0) {
            writer.raw(", ");
        }
        const SchedulerBenchmarkPhase& phase = report.phases[index];
        writer.raw("{\"index\": ");
        writer.u64(phase.index);
        writer.raw(", \"requests_per_second\": ");
        writer.number(phase.requests_per_second);
        writer.raw(", \"duration_nanoseconds\": ");
        writer.i64(phase.duration.count());
        writer.raw(", \"scheduled_opportunities\": ");
        writer.u64(phase.scheduled_opportunities);
        writer.raw("}");
    }
    writer.raw("],\n  \"queue_depth\": ");
    writer.u64(report.queue_depth);
    writer.raw(",\n  \"executing\": ");
    writer.u64(report.executing);
    writer.raw(",\n  \"live_workers\": ");
    writer.u64(report.live_workers);
    writer.raw(",\n  \"exceptional_promise_backstop\": \"a completion stored only by the request "
               "destructor is absent from scheduler terminal counters\"\n}\n");
    if (!writer.ok) {
        return failure("failed to format a JSON number");
    }
    SchedulerBenchmarkJsonResult result;
    result.text = std::move(writer.text);
    return result;
}

} // namespace inferedge
