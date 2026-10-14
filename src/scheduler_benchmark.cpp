#include "inferedge/scheduler_benchmark.hpp"

#include "inferedge/neural_reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <thread>
#include <utility>

namespace inferedge {
namespace {

struct JoinThread {
    std::thread thread;
    ~JoinThread() {
        if (!thread.joinable()) {
            return;
        }
        try {
            thread.join();
        } catch (const std::exception&) {
        }
    }
    JoinThread() = default;
    JoinThread(const JoinThread&) = delete;
    JoinThread& operator=(const JoinThread&) = delete;
};

// Joins every started producer if constructing a later thread throws.
struct JoinThreads {
    std::vector<std::thread> threads;
    JoinThreads() = default;
    JoinThreads(const JoinThreads&) = delete;
    JoinThreads& operator=(const JoinThreads&) = delete;
    ~JoinThreads() {
        for (std::thread& thread : threads) {
            if (!thread.joinable()) {
                continue;
            }
            try {
                thread.join();
            } catch (const std::exception&) {
            }
        }
    }
};

struct PreparedModel {
    SchedulerBenchmarkModelId id{SchedulerBenchmarkModelId::kBatchAddF32};
    bool neural{false};
    NeuralWorkload workload{NeuralWorkload::kMlpSmall};
    std::vector<float> weights;
    std::filesystem::path weights_path;
    std::string model_digest;
    std::string weights_digest;
};

struct Inflight {
    std::uint64_t attempt_id{0};
    std::chrono::steady_clock::time_point scheduled{};
    std::chrono::steady_clock::time_point submit_start{};
    float expected_c0{0.F};
    float expected_c1{0.F};
    // Neural expected row prepared before submit_start. Empty for Add.
    std::vector<float> expected_outputs;
    std::vector<NamedTensor> inputs;
    std::future<SchedulerCompletion> completion;
};

struct SharedTrial {
    std::shared_ptr<const PreparedModel> model;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<Inflight> inflight;
    std::vector<std::chrono::nanoseconds> latencies;
    std::vector<std::chrono::nanoseconds> pre_publication_latencies;
    std::vector<std::chrono::nanoseconds> observation_delays;
    std::vector<std::chrono::nanoseconds> lags;
    std::uint64_t missing_pre_stamps{0};
    std::uint64_t unrepresentable_timing{0};
    std::uint64_t negative_timing{0};
    std::uint64_t timing_inference_failures{0};
    std::uint64_t timing_worker_failures{0};
    std::uint64_t timing_scheduler_failures{0};
    std::uint64_t timing_cancellations{0};
    std::uint64_t timing_backstops{0};
    std::vector<SchedulerBenchmarkErrorRecord> errors;
    std::uint64_t scheduled{0};
    std::uint64_t attempts{0};
    std::uint64_t skips_outstanding{0};
    std::uint64_t skips_lag{0};
    std::uint64_t accepted{0};
    std::uint64_t rejected_full{0};
    std::uint64_t rejected_closed{0};
    std::uint64_t rejected_not_running{0};
    std::uint64_t succeeded{0};
    std::uint64_t failed_inference{0};
    std::uint64_t failed_worker{0};
    std::uint64_t failed_scheduler{0};
    std::uint64_t canceled{0};
    std::uint64_t timeouts{0};
    std::uint64_t checked{0};
    std::uint64_t mismatches{0};
    std::uint64_t error_count{0};
    std::uint32_t injected_check_faults{0};
    std::uint64_t outstanding{0};
    std::uint64_t next_id{0};
    bool have_unchecked_success{false};
    std::uint64_t unchecked_attempt{0};
    float unchecked_c0{0.F};
    float unchecked_c1{0.F};
    std::vector<float> unchecked_expected_outputs;
    std::vector<NamedTensor> unchecked_inputs;
    std::vector<NamedTensor> unchecked_outputs;
    bool generators_done{false};
    bool shutdown_done{false};
    bool runtime_failed{false};
    std::string runtime_message;
};

bool finite_rate(double rate) {
    return std::isfinite(rate) && rate > 0.0 && rate <= kMaxSchedulerBenchmarkRequestsPerSecond;
}

std::chrono::nanoseconds period_for(double requests_per_second) {
    const double nanos = 1000000000.0 / requests_per_second;
    if (!std::isfinite(nanos) || nanos < 1.0 ||
        nanos > static_cast<double>(std::chrono::nanoseconds::max().count())) {
        return std::chrono::nanoseconds{-1};
    }
    return std::chrono::nanoseconds{
        static_cast<std::chrono::nanoseconds::rep>(std::llround(nanos))};
}

bool push_arrival(SchedulerBenchmarkPlan& plan, std::chrono::nanoseconds offset,
                  std::uint32_t phase_index, std::uint64_t max_offered) {
    if (plan.arrivals.size() >= max_offered) {
        return false;
    }
    plan.arrivals.push_back(SchedulerBenchmarkArrival{offset, phase_index});
    return true;
}

void fill_phase(SchedulerBenchmarkPlan& plan, std::uint32_t phase_index, double rate,
                std::chrono::nanoseconds phase_begin, std::chrono::nanoseconds phase_end,
                std::uint64_t max_offered) {
    SchedulerBenchmarkPhase phase;
    phase.index = phase_index;
    phase.requests_per_second = rate;
    phase.duration = phase_end - phase_begin;
    const std::chrono::nanoseconds period = period_for(rate);
    if (period.count() < 1 || phase_end < phase_begin) {
        plan.phases.push_back(phase);
        return;
    }
    for (std::chrono::nanoseconds cursor = phase_begin; cursor < phase_end;) {
        if (!push_arrival(plan, cursor, phase_index, max_offered)) {
            break;
        }
        ++phase.scheduled_opportunities;
        if (period > phase_end - cursor) {
            break;
        }
        cursor += period;
    }
    plan.phases.push_back(phase);
}

SchedulerBenchmarkPlanResult invalid_plan(SchedulerBenchmarkPlanError error) {
    SchedulerBenchmarkPlanResult result;
    result.error = error;
    return result;
}

bool batch_bounds_ok(const SchedulerBenchmarkConfig& config) {
    return config.max_batch_size >= 1 && config.max_batch_size <= kSchedulerMetricsMaxBatchSize &&
           config.max_batch_delay.count() >= 0 && config.max_batch_delay.count() <= 1000;
}

std::string validate_config(const SchedulerBenchmarkConfig& config) {
    if (config.model_path.empty()) {
        return "model path is empty";
    }
    if (config.worker_count < 1 || config.worker_count > kMaxSchedulerWorkers) {
        return "worker_count is outside 1..64";
    }
    if (config.queue_capacity < 1 || config.queue_capacity > kMaxSchedulerQueueCapacity) {
        return "queue_capacity is outside 1..1024";
    }
    if (!batch_bounds_ok(config)) {
        return "batch bounds are outside the scheduler limits";
    }
    if (config.adaptive_batching) {
        if (!config.enable_batching) {
            return "adaptive_batching requires enable_batching";
        }
        if (config.max_batch_size < 2) {
            return "adaptive_batching requires max_batch_size >= 2";
        }
        if (config.max_batch_delay <= std::chrono::milliseconds::zero()) {
            return "adaptive_batching requires max_batch_delay > 0";
        }
    }
    if (config.max_offered < 1 || config.max_offered > kMaxSchedulerBenchmarkOffered) {
        return "max_offered is outside 1..100000";
    }
    if (config.max_outstanding < 1 || config.max_outstanding > config.max_offered) {
        return "max_outstanding must be from 1 through max_offered";
    }
    if (config.duration.count() < 1 ||
        config.duration.count() > static_cast<std::int64_t>(kMaxSchedulerBenchmarkDurationMs)) {
        return "duration is outside 1..60000 milliseconds";
    }
    if (config.completion_timeout.count() < 1 ||
        config.completion_timeout.count() >
            static_cast<std::int64_t>(kMaxSchedulerBenchmarkDurationMs)) {
        return "completion_timeout is outside 1..60000 milliseconds";
    }
    if (config.lag_tolerance.count() < 0 || config.lag_tolerance > std::chrono::seconds{60}) {
        return "lag_tolerance is outside 0..60 seconds";
    }
    if (config.check_interval < 1) {
        return "check_interval must be at least 1";
    }
    if (config.warmup_requests > kMaxSchedulerBenchmarkOffered) {
        return "warmup_requests exceeds 100000";
    }
    if (config.test_submit_delay.count() < 0) {
        return "test_submit_delay must be non-negative";
    }
    if (config.workload == SchedulerWorkloadKind::kOpenLoopConstant) {
        if (!finite_rate(config.target_requests_per_second)) {
            return "target_requests_per_second must be finite and in (0, 1000000]";
        }
        if (config.producer_count != 1) {
            return "open-loop constant traffic uses one generator";
        }
    } else if (config.workload == SchedulerWorkloadKind::kOpenLoopBurst) {
        if (!finite_rate(config.burst_high_requests_per_second) ||
            !finite_rate(config.burst_low_requests_per_second)) {
            return "burst rates must be finite and in (0, 1000000]";
        }
        if (config.burst_high.count() < 1 || config.burst_low.count() < 1 ||
            config.burst_high.count() >
                static_cast<std::int64_t>(kMaxSchedulerBenchmarkDurationMs) ||
            config.burst_low.count() >
                static_cast<std::int64_t>(kMaxSchedulerBenchmarkDurationMs)) {
            return "burst phase lengths must be from 1 through 60000 milliseconds";
        }
        if (config.producer_count != 1) {
            return "open-loop burst traffic uses one generator";
        }
    } else if (config.workload == SchedulerWorkloadKind::kClosedLoop) {
        if (config.producer_count < 1 || config.producer_count > kMaxSchedulerBenchmarkProducers) {
            return "producer_count is outside 1..64";
        }
        if (config.producer_count > config.max_outstanding) {
            return "closed-loop producer_count cannot exceed max_outstanding";
        }
    } else {
        return "workload is unsupported";
    }
    return {};
}

void remember_error(SharedTrial& trial, std::uint64_t attempt_id, std::string kind,
                    std::string message) {
    ++trial.error_count;
    if (trial.errors.size() >= kSchedulerBenchmarkMaxErrorRecords) {
        return;
    }
    trial.errors.push_back(
        SchedulerBenchmarkErrorRecord{attempt_id, std::move(kind), std::move(message)});
}

// Builds inputs and, for neural models, the independent expected row. Must run
// before submit_start. Does not call the engine. weights are borrowed from the
// prepared model for the whole trial.
std::string prepare_inflight_expectation(const PreparedModel& model,
                                         SchedulerBenchmarkModelId model_id,
                                         std::uint64_t attempt_id, Inflight& item) {
    SchedulerBenchmarkTensors tensors = scheduler_benchmark_request(model_id, attempt_id);
    item.inputs = tensors.inputs;
    item.expected_c0 = tensors.expected_c0;
    item.expected_c1 = tensors.expected_c1;
    item.expected_outputs.clear();
    if (!model.neural) {
        return {};
    }
    if (item.inputs.size() != 1 || item.inputs.front().values.empty()) {
        return "neural request input row is empty";
    }
    const NeuralReferenceResult reference =
        evaluate_neural_reference(model.workload, item.inputs.front().values, 1, model.weights);
    if (reference.error.kind != NeuralReferenceErrorKind::kNone) {
        return reference.error.message.empty() ? "neural reference preparation failed"
                                               : reference.error.message;
    }
    if (reference.values.empty()) {
        return "neural reference preparation returned no values";
    }
    item.expected_outputs = std::move(reference.values);
    return {};
}

SchedulerBenchmarkOutputCheck match_completion(const PreparedModel& model, const Inflight& item,
                                               const SchedulerCompletion& completion) {
    SchedulerBenchmarkOutputCheck failed;
    if (completion.outputs.size() != 1) {
        failed.message = "output count is not 1";
        return failed;
    }
    if (!model.neural) {
        const float expected[] = {item.expected_c0, item.expected_c1};
        return scheduler_benchmark_check_output(model.id, completion.outputs.front(), expected);
    }
    if (item.expected_outputs.empty()) {
        failed.message = "neural expected output was not prepared before submit";
        return failed;
    }
    return scheduler_benchmark_check_output(model.id, completion.outputs.front(),
                                            item.expected_outputs);
}

void note_terminal(SharedTrial& trial, CompletionStatus status) {
    switch (status) {
    case CompletionStatus::kSucceeded:
        ++trial.succeeded;
        break;
    case CompletionStatus::kInferenceFailed:
        ++trial.failed_inference;
        break;
    case CompletionStatus::kWorkerFailed:
        ++trial.failed_worker;
        break;
    case CompletionStatus::kSchedulerFailed:
        ++trial.failed_scheduler;
        break;
    case CompletionStatus::kCanceled:
        ++trial.canceled;
        break;
    }
}

void record_success_sample(SharedTrial& trial, const SchedulerBenchmarkConfig& config,
                           const Inflight& item, std::chrono::nanoseconds latency,
                           const SchedulerCompletion& completion, bool is_last,
                           const std::optional<SchedulerBenchmarkOutputCheck>& prechecked) {
    trial.latencies.push_back(latency);
    const std::uint64_t index = trial.succeeded;
    const bool due = scheduler_benchmark_should_check(index, config.check_interval, is_last) ||
                     trial.injected_check_faults < config.test_fail_checked_successes;
    if (!due) {
        trial.have_unchecked_success = true;
        trial.unchecked_attempt = item.attempt_id;
        trial.unchecked_c0 = item.expected_c0;
        trial.unchecked_c1 = item.expected_c1;
        trial.unchecked_expected_outputs = item.expected_outputs;
        trial.unchecked_inputs = item.inputs;
        trial.unchecked_outputs = completion.outputs;
        return;
    }
    trial.have_unchecked_success = false;
    ++trial.checked;
    if (trial.injected_check_faults < config.test_fail_checked_successes) {
        ++trial.injected_check_faults;
        ++trial.mismatches;
        remember_error(trial, item.attempt_id, "correctness", "injected check failure");
        return;
    }
    if (!prechecked.has_value()) {
        ++trial.mismatches;
        remember_error(trial, item.attempt_id, "correctness", "completion was not checked");
        return;
    }
    if (!prechecked->matched) {
        ++trial.mismatches;
        remember_error(trial, item.attempt_id, "correctness",
                       prechecked->message.empty() ? "output did not match the attempt"
                                                   : prechecked->message);
    }
}

bool take_rejection(SharedTrial& trial, Admission& admission, std::uint64_t attempt_id,
                    const std::vector<NamedTensor>& submitted) {
    ++trial.attempts;
    if (!admission.inputs.has_value() || admission.completion.has_value()) {
        remember_error(trial, attempt_id, "admission_ownership",
                       "rejection did not return input ownership");
        return false;
    }
    const SchedulerBenchmarkOutputCheck owned =
        scheduler_benchmark_check_returned_inputs(*admission.inputs, submitted);
    if (!owned.matched) {
        remember_error(trial, attempt_id, "admission_ownership",
                       owned.message.empty() ? "returned inputs did not match the attempt"
                                             : owned.message);
        return false;
    }
    if (admission.status == AdmissionStatus::kRejectedFull) {
        ++trial.rejected_full;
    } else if (admission.status == AdmissionStatus::kRejectedClosed) {
        ++trial.rejected_closed;
    } else {
        ++trial.rejected_not_running;
    }
    return true;
}

bool take_acceptance(SharedTrial& trial, Admission& admission, Inflight item) {
    ++trial.attempts;
    ++trial.accepted;
    if (admission.status != AdmissionStatus::kAccepted || !admission.completion.has_value() ||
        admission.inputs.has_value()) {
        ++trial.timeouts;
        remember_error(trial, item.attempt_id, "admission_ownership",
                       "accepted admission did not own a future");
        return false;
    }
    item.completion = std::move(*admission.completion);
    trial.inflight.push_back(std::move(item));
    return true;
}

void account_admission(SharedTrial& trial, Admission admission, Inflight item) {
    if (admission.status == AdmissionStatus::kAccepted) {
        take_acceptance(trial, admission, std::move(item));
        return;
    }
    take_rejection(trial, admission, item.attempt_id, item.inputs);
}

std::chrono::steady_clock::time_point
wait_until_scheduled(std::chrono::steady_clock::time_point scheduled) {
    std::this_thread::sleep_until(scheduled);
    auto now = std::chrono::steady_clock::now();
    const auto spin_limit = scheduled + std::chrono::milliseconds{2};
    while (now < scheduled && now < spin_limit) {
        std::this_thread::sleep_for(std::chrono::microseconds{50});
        now = std::chrono::steady_clock::now();
    }
    return std::chrono::steady_clock::now();
}

void run_open_loop_generator(InferenceScheduler& scheduler, const SchedulerBenchmarkConfig& config,
                             const SchedulerBenchmarkPlan& plan, SharedTrial& trial,
                             std::chrono::steady_clock::time_point window_start) {
    try {
        const auto window_end = window_start + config.duration;
        for (std::size_t index = 0; index < plan.arrivals.size(); ++index) {
            const auto scheduled = window_start + plan.arrivals[index].offset;
            if (scheduled > window_end) {
                std::lock_guard<std::mutex> lock(trial.mutex);
                trial.skips_lag += plan.arrivals.size() - index;
                break;
            }
            (void)wait_until_scheduled(scheduled);
            if (config.test_submit_delay.count() > 0) {
                std::this_thread::sleep_for(config.test_submit_delay);
            }
            const auto decision = std::chrono::steady_clock::now();
            const SchedulerBenchmarkLag lag =
                measure_generator_lag(scheduled, decision, config.lag_tolerance);
            if (lag.exceeds_tolerance) {
                std::lock_guard<std::mutex> lock(trial.mutex);
                trial.lags.push_back(lag.lag);
                trial.skips_lag += plan.arrivals.size() - index;
                break;
            }
            bool capped = false;
            {
                std::lock_guard<std::mutex> lock(trial.mutex);
                ++trial.scheduled;
                if (trial.inflight.size() >= config.max_outstanding) {
                    ++trial.skips_outstanding;
                    capped = true;
                }
            }
            if (capped) {
                continue;
            }
            const std::uint64_t attempt_id = static_cast<std::uint64_t>(index) + 1;
            if (!trial.model) {
                std::lock_guard<std::mutex> lock(trial.mutex);
                trial.runtime_failed = true;
                trial.runtime_message = "model identity was not prepared";
                trial.generators_done = true;
                trial.ready.notify_all();
                return;
            }
            Inflight item;
            item.attempt_id = attempt_id;
            item.scheduled = scheduled;
            const std::string prepared =
                prepare_inflight_expectation(*trial.model, config.model_id, attempt_id, item);
            if (!prepared.empty()) {
                std::lock_guard<std::mutex> lock(trial.mutex);
                trial.runtime_failed = true;
                trial.runtime_message = prepared;
                trial.generators_done = true;
                trial.ready.notify_all();
                return;
            }
            // Expected outputs are fixed before this stamp. Preparation cost can
            // make the generator late for this or later arrivals.
            item.submit_start = std::chrono::steady_clock::now();
            const SchedulerBenchmarkLag submit_lag =
                measure_generator_lag(scheduled, item.submit_start, config.lag_tolerance);
            Admission admission = scheduler.submit(item.inputs);
            {
                std::lock_guard<std::mutex> lock(trial.mutex);
                trial.lags.push_back(submit_lag.lag);
                account_admission(trial, std::move(admission), std::move(item));
                if (submit_lag.exceeds_tolerance) {
                    trial.skips_lag += plan.arrivals.size() - index - 1;
                    break;
                }
            }
            trial.ready.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(trial.mutex);
            const std::uint64_t accounted =
                trial.attempts + trial.skips_outstanding + trial.skips_lag;
            if (accounted < plan.arrivals.size()) {
                trial.skips_lag += plan.arrivals.size() - accounted;
            }
            trial.scheduled = plan.arrivals.size();
            trial.generators_done = true;
        }
        trial.ready.notify_all();
    } catch (const std::exception& exception) {
        std::lock_guard<std::mutex> lock(trial.mutex);
        trial.runtime_failed = true;
        trial.runtime_message = exception.what();
        trial.generators_done = true;
        trial.ready.notify_all();
    }
}

std::optional<std::int64_t> exact_nanoseconds(std::chrono::steady_clock::duration delta) {
    using Nanoseconds = std::chrono::nanoseconds;
    if (delta > Nanoseconds::max() || delta < Nanoseconds::min()) {
        return std::nullopt;
    }
    const auto nanos = std::chrono::duration_cast<Nanoseconds>(delta);
    if (std::chrono::duration_cast<std::chrono::steady_clock::duration>(nanos) != delta) {
        return std::nullopt;
    }
    return nanos.count();
}

void record_observed(SharedTrial& trial, const SchedulerBenchmarkConfig& config, Inflight item,
                     std::chrono::steady_clock::time_point observed) {
    SchedulerCompletion completion;
    try {
        completion = item.completion.get();
    } catch (const std::exception& exception) {
        std::lock_guard<std::mutex> lock(trial.mutex);
        ++trial.failed_scheduler;
        remember_error(trial, item.attempt_id, "completion_exception", exception.what());
        return;
    }
    // The observation stamp is already fixed. Neural expected outputs were
    // prepared before submit_start. Matching here only compares stored floats
    // to the completion and must not call evaluate_neural_reference.
    std::optional<SchedulerBenchmarkOutputCheck> prechecked;
    if (completion.status == CompletionStatus::kSucceeded) {
        try {
            if (!trial.model) {
                prechecked =
                    SchedulerBenchmarkOutputCheck{false, "model identity was not prepared"};
            } else {
                prechecked = match_completion(*trial.model, item, completion);
            }
        } catch (const std::exception& exception) {
            prechecked = SchedulerBenchmarkOutputCheck{false, exception.what()};
        }
    }
    std::lock_guard<std::mutex> lock(trial.mutex);
    if (observed < item.submit_start) {
        ++trial.failed_scheduler;
        remember_error(trial, item.attempt_id, "clock", "observation preceded submit");
        return;
    }
    if (completion.request_id == 0 || completion.sequence == 0) {
        remember_error(trial, item.attempt_id, "request_identity",
                       "completion omitted the scheduler request id");
    }
    note_terminal(trial, completion.status);
    if (completion.status == CompletionStatus::kSucceeded) {
        record_success_sample(trial, config, item, observed - item.submit_start, completion, false,
                              prechecked);
    }
    if (!config.enable_completion_timing) {
        return;
    }
    const bool backstop = completion.status == CompletionStatus::kSchedulerFailed &&
                          completion.inference_error.message == "unsettled";
    if (backstop) {
        ++trial.timing_backstops;
    }
    if (!completion.pre_publication_time.has_value()) {
        ++trial.missing_pre_stamps;
        remember_error(trial, item.attempt_id, "pre_publication_stamp",
                       "timing-enabled completion omitted the pre-publication stamp");
        return;
    }
    const auto pre_latency =
        exact_nanoseconds(*completion.pre_publication_time - item.submit_start);
    const auto delay = exact_nanoseconds(observed - *completion.pre_publication_time);
    if (!pre_latency.has_value() || !delay.has_value()) {
        ++trial.unrepresentable_timing;
        remember_error(trial, item.attempt_id, "timing_duration",
                       "a timing duration is not an exact int64 nanosecond value");
        return;
    }
    if (*pre_latency < 0 || *delay < 0) {
        ++trial.negative_timing;
        remember_error(trial, item.attempt_id, "timing_duration",
                       "a pre-publication interval moved backwards");
        return;
    }
    if (completion.status == CompletionStatus::kSucceeded) {
        // The stored producer-observed sample is this same duration converted to
        // nanoseconds. Exact stamps must add to it; overflow is not a sample.
        constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
        const auto producer_observed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(observed - item.submit_start);
        if (*pre_latency > kMax - *delay ||
            std::chrono::nanoseconds{*pre_latency + *delay} != producer_observed) {
            ++trial.unrepresentable_timing;
            remember_error(trial, item.attempt_id, "timing_duration",
                           "pre-publication latency and observation delay do not add to the "
                           "producer-observed duration");
            return;
        }
    }
    if (completion.status != CompletionStatus::kSucceeded) {
        if (completion.status == CompletionStatus::kInferenceFailed) {
            ++trial.timing_inference_failures;
        } else if (completion.status == CompletionStatus::kWorkerFailed) {
            ++trial.timing_worker_failures;
        } else if (completion.status == CompletionStatus::kCanceled) {
            ++trial.timing_cancellations;
        } else if (!backstop) {
            ++trial.timing_scheduler_failures;
        }
        return;
    }
    trial.pre_publication_latencies.push_back(std::chrono::nanoseconds{*pre_latency});
    trial.observation_delays.push_back(std::chrono::nanoseconds{*delay});
}

// Readiness uses wait_for(0) under the trial mutex. That probe does not call
// the scheduler. A ready future is removed before get(); a pending future stays
// in place, so a later request is not hidden behind an earlier one.
void collect_ready(SharedTrial& trial, const SchedulerBenchmarkConfig& config) {
    std::vector<Inflight> ready_items;
    std::vector<std::chrono::steady_clock::time_point> observed_at;
    {
        std::lock_guard<std::mutex> lock(trial.mutex);
        for (std::size_t index = 0; index < trial.inflight.size();) {
            if (trial.inflight[index].completion.wait_for(std::chrono::nanoseconds{0}) !=
                std::future_status::ready) {
                ++index;
                continue;
            }
            observed_at.push_back(std::chrono::steady_clock::now());
            ready_items.push_back(std::move(trial.inflight[index]));
            trial.inflight.erase(trial.inflight.begin() + static_cast<std::ptrdiff_t>(index));
        }
    }
    for (std::size_t index = 0; index < ready_items.size(); ++index) {
        record_observed(trial, config, std::move(ready_items[index]), observed_at[index]);
    }
}

void collect_until(SharedTrial& trial, const SchedulerBenchmarkConfig& config,
                   std::chrono::steady_clock::time_point deadline) {
    while (true) {
        Inflight item;
        {
            std::lock_guard<std::mutex> lock(trial.mutex);
            if (trial.inflight.empty()) {
                return;
            }
            item = std::move(trial.inflight.back());
            trial.inflight.pop_back();
        }
        const auto now = std::chrono::steady_clock::now();
        std::future_status status = std::future_status::timeout;
        if (now < deadline) {
            status = item.completion.wait_for(deadline - now);
        }
        if (status != std::future_status::ready) {
            std::lock_guard<std::mutex> lock(trial.mutex);
            ++trial.timeouts;
            remember_error(trial, item.attempt_id, "timeout",
                           "completion was not ready before the deadline");
            continue;
        }
        record_observed(trial, config, std::move(item), std::chrono::steady_clock::now());
    }
}

void run_open_loop_collector(SharedTrial& trial, const SchedulerBenchmarkConfig& config) {
    while (true) {
        bool done = false;
        bool hold = false;
        bool after_shutdown = false;
        {
            std::unique_lock<std::mutex> lock(trial.mutex);
            if (config.hold_collector && !trial.generators_done) {
                trial.ready.wait(lock, [&trial] { return trial.generators_done; });
            }
            hold = trial.inflight.empty();
            done = hold && trial.generators_done && trial.shutdown_done;
            after_shutdown = trial.shutdown_done;
            if (hold && !done) {
                trial.ready.wait_for(lock, kSchedulerBenchmarkObservationQuantum, [&trial] {
                    return !trial.inflight.empty() || trial.shutdown_done;
                });
                continue;
            }
        }
        if (done) {
            return;
        }
        if (after_shutdown) {
            const auto deadline = std::chrono::steady_clock::now() + config.completion_timeout;
            collect_until(trial, config, deadline);
            std::lock_guard<std::mutex> lock(trial.mutex);
            if (trial.inflight.empty()) {
                return;
            }
            continue;
        }
        collect_ready(trial, config);
        bool still = false;
        {
            std::lock_guard<std::mutex> lock(trial.mutex);
            still = !trial.inflight.empty();
        }
        if (still) {
            std::this_thread::sleep_for(kSchedulerBenchmarkObservationQuantum);
        }
    }
}

void run_closed_producer(InferenceScheduler& scheduler, const SchedulerBenchmarkConfig& config,
                         SharedTrial& trial, std::chrono::steady_clock::time_point window_start) {
    const auto window_end = window_start + config.duration;
    try {
        while (true) {
            std::uint64_t attempt_id = 0;
            {
                std::unique_lock<std::mutex> lock(trial.mutex);
                if (trial.runtime_failed || trial.scheduled >= config.max_offered ||
                    std::chrono::steady_clock::now() >= window_end) {
                    return;
                }
                if (trial.outstanding >= config.max_outstanding) {
                    const auto remaining = window_end - std::chrono::steady_clock::now();
                    if (remaining.count() <= 0) {
                        return;
                    }
                    trial.ready.wait_for(lock, remaining, [&trial, &config] {
                        return trial.outstanding < config.max_outstanding ||
                               trial.scheduled >= config.max_offered || trial.runtime_failed;
                    });
                    if (trial.runtime_failed || trial.scheduled >= config.max_offered) {
                        return;
                    }
                    if (std::chrono::steady_clock::now() >= window_end) {
                        ++trial.scheduled;
                        ++trial.skips_outstanding;
                        return;
                    }
                    if (trial.outstanding >= config.max_outstanding) {
                        continue;
                    }
                }
                if (trial.scheduled >= config.max_offered) {
                    return;
                }
                ++trial.scheduled;
                ++trial.outstanding;
                attempt_id = ++trial.next_id;
            }
            if (config.test_submit_delay.count() > 0) {
                std::this_thread::sleep_for(config.test_submit_delay);
            }
            if (!trial.model) {
                std::lock_guard<std::mutex> lock(trial.mutex);
                trial.runtime_failed = true;
                trial.runtime_message = "model identity was not prepared";
                trial.ready.notify_all();
                return;
            }
            Inflight item;
            item.attempt_id = attempt_id;
            const std::string prepared =
                prepare_inflight_expectation(*trial.model, config.model_id, attempt_id, item);
            if (!prepared.empty()) {
                std::lock_guard<std::mutex> lock(trial.mutex);
                trial.runtime_failed = true;
                trial.runtime_message = prepared;
                trial.ready.notify_all();
                return;
            }
            // Expected outputs are fixed before this stamp. Preparation can
            // delay the next closed-loop submit on this producer.
            item.submit_start = std::chrono::steady_clock::now();
            Admission admission = scheduler.submit(item.inputs);
            if (admission.status != AdmissionStatus::kAccepted) {
                std::lock_guard<std::mutex> lock(trial.mutex);
                account_admission(trial, std::move(admission), std::move(item));
                if (trial.outstanding > 0) {
                    --trial.outstanding;
                }
                trial.ready.notify_all();
                continue;
            }
            if (!admission.completion.has_value() || admission.inputs.has_value()) {
                std::lock_guard<std::mutex> lock(trial.mutex);
                ++trial.attempts;
                ++trial.accepted;
                ++trial.timeouts;
                remember_error(trial, attempt_id, "admission_ownership",
                               "accepted admission did not own a future");
                if (trial.outstanding > 0) {
                    --trial.outstanding;
                }
                trial.ready.notify_all();
                continue;
            }
            const auto status = admission.completion->wait_for(config.completion_timeout);
            const auto observed = std::chrono::steady_clock::now();
            if (status != std::future_status::ready) {
                std::lock_guard<std::mutex> lock(trial.mutex);
                ++trial.attempts;
                ++trial.accepted;
                ++trial.timeouts;
                remember_error(trial, attempt_id, "timeout",
                               "closed-loop wait exceeded completion_timeout");
                if (trial.outstanding > 0) {
                    --trial.outstanding;
                }
            } else {
                item.completion = std::move(*admission.completion);
                {
                    std::lock_guard<std::mutex> lock(trial.mutex);
                    ++trial.attempts;
                    ++trial.accepted;
                    if (trial.outstanding > 0) {
                        --trial.outstanding;
                    }
                }
                record_observed(trial, config, std::move(item), observed);
            }
            trial.ready.notify_all();
        }
    } catch (const std::exception& exception) {
        std::lock_guard<std::mutex> lock(trial.mutex);
        trial.runtime_failed = true;
        trial.runtime_message = exception.what();
        trial.ready.notify_all();
    }
}

bool histogram_holds(const SchedulerMetrics& metrics, bool& saturated) {
    saturated = metrics.engine_run_count == UINT64_MAX ||
                metrics.requests_entered_engine_runs == UINT64_MAX;
    if (metrics.batch_size_histogram[0] != 0) {
        return false;
    }
    std::uint64_t runs = 0;
    std::uint64_t weighted = 0;
    for (std::size_t size = 1; size < metrics.batch_size_histogram.size(); ++size) {
        const std::uint64_t bin = metrics.batch_size_histogram[size];
        if (bin == UINT64_MAX) {
            saturated = true;
        }
        if (runs > UINT64_MAX - bin) {
            saturated = true;
            return true;
        }
        runs += bin;
        if (bin > UINT64_MAX / size) {
            saturated = true;
            return true;
        }
        const std::uint64_t product = bin * static_cast<std::uint64_t>(size);
        if (weighted > UINT64_MAX - product) {
            saturated = true;
            return true;
        }
        weighted += product;
    }
    if (saturated) {
        return true;
    }
    return runs == metrics.engine_run_count && weighted == metrics.requests_entered_engine_runs;
}

void add_reason(std::vector<std::string>& reasons, std::string reason) {
    reasons.push_back(std::move(reason));
}

bool signed_nearest_rank(std::uint64_t count, std::uint64_t percent, std::uint64_t& index) {
    if (percent == 0 || count == 0 ||
        count > (std::numeric_limits<std::uint64_t>::max() - 99) / percent) {
        return false;
    }
    const std::uint64_t rank = (percent * count + 99) / 100;
    if (rank == 0 || rank > count) {
        return false;
    }
    index = rank - 1;
    return true;
}

void fill_signed_percentiles(const std::vector<std::chrono::nanoseconds>& sorted, std::int64_t& p50,
                             std::int64_t& p95, std::int64_t& p99, bool& reported) {
    std::uint64_t p50_index = 0;
    std::uint64_t p95_index = 0;
    std::uint64_t p99_index = 0;
    const auto count = static_cast<std::uint64_t>(sorted.size());
    if (!signed_nearest_rank(count, 50, p50_index) || !signed_nearest_rank(count, 95, p95_index) ||
        !signed_nearest_rank(count, 99, p99_index)) {
        reported = false;
        return;
    }
    p50 = sorted[static_cast<std::size_t>(p50_index)].count();
    p95 = sorted[static_cast<std::size_t>(p95_index)].count();
    p99 = sorted[static_cast<std::size_t>(p99_index)].count();
    reported = true;
}

SchedulerBenchmarkNanosecondSummary
summarize_nonnegative_nanos(const std::vector<std::chrono::nanoseconds>& samples) {
    SchedulerBenchmarkNanosecondSummary summary;
    summary.sample_count = samples.size();
    if (samples.empty()) {
        return summary;
    }
    summary.minimum = *std::min_element(samples.begin(), samples.end());
    summary.maximum = *std::max_element(samples.begin(), samples.end());
    long double total = 0;
    for (const std::chrono::nanoseconds sample : samples) {
        total += static_cast<long double>(sample.count());
    }
    summary.mean_nanoseconds =
        static_cast<double>(total / static_cast<long double>(samples.size()));
    std::vector<std::chrono::nanoseconds> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    fill_signed_percentiles(sorted, summary.p50_nanoseconds, summary.p95_nanoseconds,
                            summary.p99_nanoseconds, summary.percentiles_reported);
    summary.precise_tail = summary.sample_count >= 100 && summary.percentiles_reported;
    return summary;
}

SchedulerBenchmarkLagSummary summarize_lag(const std::vector<std::chrono::nanoseconds>& lags) {
    SchedulerBenchmarkLagSummary summary;
    summary.count = lags.size();
    if (lags.empty()) {
        return summary;
    }
    summary.minimum = *std::min_element(lags.begin(), lags.end());
    summary.maximum = *std::max_element(lags.begin(), lags.end());
    long double total = 0;
    for (const std::chrono::nanoseconds lag : lags) {
        total += static_cast<long double>(lag.count());
    }
    summary.mean_nanoseconds = static_cast<double>(total / static_cast<long double>(lags.size()));
    std::vector<std::chrono::nanoseconds> sorted = lags;
    std::sort(sorted.begin(), sorted.end());
    fill_signed_percentiles(sorted, summary.p50_nanoseconds, summary.p95_nanoseconds,
                            summary.p99_nanoseconds, summary.percentiles_reported);
    summary.precise_tail = summary.count >= 100 && summary.percentiles_reported;
    return summary;
}

double seconds_of(std::chrono::nanoseconds duration) {
    return static_cast<double>(duration.count()) / 1000000000.0;
}

double rate_of(std::uint64_t count, double seconds) {
    if (!(seconds > 0.0) || !std::isfinite(seconds)) {
        return 0.0;
    }
    const double rate = static_cast<double>(count) / seconds;
    if (!std::isfinite(rate)) {
        return 0.0;
    }
    return rate;
}

void check_last_success(SharedTrial& trial, const SchedulerBenchmarkConfig& config) {
    if (!trial.have_unchecked_success) {
        return;
    }
    ++trial.checked;
    Inflight item;
    item.attempt_id = trial.unchecked_attempt;
    item.expected_c0 = trial.unchecked_c0;
    item.expected_c1 = trial.unchecked_c1;
    item.expected_outputs = trial.unchecked_expected_outputs;
    item.inputs = trial.unchecked_inputs;
    SchedulerCompletion completion;
    completion.status = CompletionStatus::kSucceeded;
    completion.outputs = trial.unchecked_outputs;
    SchedulerBenchmarkOutputCheck checked;
    try {
        if (!trial.model) {
            checked.message = "model identity was not prepared";
        } else {
            checked = match_completion(*trial.model, item, completion);
        }
    } catch (const std::exception& exception) {
        checked.message = exception.what();
    }
    if (!checked.matched) {
        ++trial.mismatches;
        remember_error(trial, trial.unchecked_attempt, "correctness",
                       checked.message.empty() ? "output did not match the attempt"
                                               : checked.message);
    }
    (void)config;
}

SchedulerBenchmarkReport make_report(const SchedulerBenchmarkConfig& config, SharedTrial& trial,
                                     const SchedulerBenchmarkPlan& plan,
                                     std::chrono::nanoseconds arrival_window,
                                     std::chrono::nanoseconds completion,
                                     const SchedulerMetrics& metrics, const SchedulerStats& stats) {
    check_last_success(trial, config);
    SchedulerBenchmarkReport report;
    report.model = config.model_path;
    report.workload = scheduler_workload_name(config.workload);
    report.worker_count = config.worker_count;
    report.queue_capacity = config.queue_capacity;
    report.enable_batching = config.enable_batching;
    report.max_batch_size = config.max_batch_size;
    report.max_batch_delay = config.max_batch_delay;
    report.adaptive_batching = config.adaptive_batching;
    report.enable_metrics = config.enable_metrics;
    report.completion_timing_enabled = config.enable_completion_timing;
    report.steady_clock_period_num = std::chrono::steady_clock::period::num;
    report.steady_clock_period_den = std::chrono::steady_clock::period::den;
    report.warmup_requests = config.warmup_requests;
    report.warmup_uses_separate_scheduler = true;
    report.configured_duration = config.duration;
    if (!plan.arrivals.empty()) {
        report.scheduled_span = plan.arrivals.back().offset;
    }
    if (trial.skips_lag > 0) {
        report.arrival_stop_reason = "generator_lag";
    } else if (config.workload == SchedulerWorkloadKind::kClosedLoop) {
        report.arrival_stop_reason =
            trial.scheduled >= config.max_offered && arrival_window < config.duration
                ? "max_offered"
                : "configured_duration";
    } else if (!plan.phases.empty() && plan.arrivals.size() == config.max_offered) {
        const std::chrono::nanoseconds period = period_for(plan.phases.back().requests_per_second);
        report.arrival_stop_reason =
            period.count() > 0 && report.scheduled_span + period < config.duration
                ? "max_offered"
                : "configured_duration";
    } else {
        report.arrival_stop_reason = "configured_duration";
    }
    report.arrival_window = arrival_window;
    report.completion = completion;
    report.total_trial = arrival_window + completion;
    report.scheduled_opportunities = trial.scheduled;
    report.submit_attempts = trial.attempts;
    report.generator_skips_outstanding = trial.skips_outstanding;
    report.generator_skips_lag = trial.skips_lag;
    report.accepted = trial.accepted;
    report.rejections.full = trial.rejected_full;
    report.rejections.closed = trial.rejected_closed;
    report.rejections.not_running = trial.rejected_not_running;
    report.succeeded = trial.succeeded;
    report.failed_inference = trial.failed_inference;
    report.failed_worker = trial.failed_worker;
    report.failed_scheduler = trial.failed_scheduler;
    report.canceled = trial.canceled;
    report.timeouts = trial.timeouts;
    report.generator_lag = summarize_lag(trial.lags);
    // Schema 3 historically replaced a nonfinite lag mean with zero. Schema 4
    // keeps the nonfinite value so the trial is rejected instead.
    if (!report.completion_timing_enabled &&
        !std::isfinite(report.generator_lag.mean_nanoseconds)) {
        report.generator_lag.mean_nanoseconds = 0.0;
    }
    report.pre_publication_latency = summarize_nonnegative_nanos(trial.pre_publication_latencies);
    report.pre_publication_to_observation_delay =
        summarize_nonnegative_nanos(trial.observation_delays);
    report.timing_inference_failures = trial.timing_inference_failures;
    report.timing_worker_failures = trial.timing_worker_failures;
    report.timing_scheduler_failures = trial.timing_scheduler_failures;
    report.timing_cancellations = trial.timing_cancellations;
    report.timing_backstops = trial.timing_backstops;
    report.metrics = metrics;
    report.correctness_checked = trial.checked;
    report.correctness_mismatches = trial.mismatches;
    report.check_interval = config.check_interval;
    report.errors = std::move(trial.errors);
    report.error_count = trial.error_count;
    report.phases = plan.phases;
    report.queue_depth = stats.queue_depth;
    report.executing = stats.executing;
    report.live_workers = stats.live_workers;
    if (config.workload == SchedulerWorkloadKind::kClosedLoop) {
        report.completion_observation = "closed_loop_producer_wait";
        report.observation_quantum = std::chrono::nanoseconds{0};
    } else {
        report.completion_observation = "open_loop_poll";
        report.observation_quantum = kSchedulerBenchmarkObservationQuantum;
    }
    const double arrival_seconds = seconds_of(arrival_window);
    const double completion_seconds = seconds_of(completion);
    const double total_seconds = seconds_of(report.total_trial);
    report.throughput.arrival_window_seconds = arrival_seconds;
    report.throughput.completion_seconds = completion_seconds;
    report.throughput.total_trial_seconds = total_seconds;
    report.throughput.offered_per_arrival_window_second =
        rate_of(report.scheduled_opportunities, arrival_seconds);
    report.throughput.attempts_per_arrival_window_second =
        rate_of(report.submit_attempts, arrival_seconds);
    report.throughput.accepted_per_arrival_window_second =
        rate_of(report.accepted, arrival_seconds);
    report.throughput.successful_per_arrival_window_second =
        rate_of(report.succeeded, arrival_seconds);
    report.throughput.successful_per_total_trial_second = rate_of(report.succeeded, total_seconds);

    if (!trial.latencies.empty()) {
        const StatisticsResult statistics = summarize_latencies(trial.latencies);
        if (statistics.error.kind != StatisticsErrorKind::kNone) {
            add_reason(report.invalid_reasons, "latency_summary");
        } else {
            report.latency.sample_count = statistics.summary.sample_count;
            report.latency.minimum = statistics.summary.minimum;
            report.latency.maximum = statistics.summary.maximum;
            report.latency.mean_microseconds = statistics.summary.mean_microseconds;
            report.latency.p50_microseconds = statistics.summary.p50_microseconds;
            report.latency.p95_microseconds = statistics.summary.p95_microseconds;
            report.latency.p99_microseconds = statistics.summary.p99_microseconds;
            report.latency.percentiles_reported = true;
            report.latency.precise_tail = statistics.summary.sample_count >= 100;
        }
    }

    if (config.hold_collector) {
        add_reason(report.invalid_reasons, "collector_held");
    }
    if (config.test_submit_delay.count() > 0) {
        add_reason(report.invalid_reasons, "submit_delay_injected");
    }
    if (config.test_fail_checked_successes > 0) {
        add_reason(report.invalid_reasons, "check_fault_injected");
    }
    if (config.test_submit_after_shutdown) {
        add_reason(report.invalid_reasons, "post_shutdown_submit_injected");
    }
    if (trial.mismatches > 0) {
        add_reason(report.invalid_reasons, "correctness");
    }
    if (trial.timeouts > 0) {
        add_reason(report.invalid_reasons, "timeout");
    }
    if (trial.skips_lag > 0) {
        add_reason(report.invalid_reasons, "generator_lag");
    }
    if (config.workload != SchedulerWorkloadKind::kClosedLoop && trial.skips_outstanding > 0 &&
        trial.attempts * 2 <= trial.scheduled) {
        add_reason(report.invalid_reasons, "outstanding_cap_suppressed_schedule");
    }
    bool saw_request_identity = false;
    bool saw_admission_ownership = false;
    for (const SchedulerBenchmarkErrorRecord& error : report.errors) {
        if (!saw_request_identity && error.kind == "request_identity") {
            add_reason(report.invalid_reasons, "missing_request_identity");
            saw_request_identity = true;
        } else if (!saw_admission_ownership && error.kind == "admission_ownership") {
            add_reason(report.invalid_reasons, "admission_ownership");
            saw_admission_ownership = true;
        }
    }
    if (arrival_window.count() <= 0 || report.total_trial.count() <= 0) {
        add_reason(report.invalid_reasons, "zero_arrival_window");
    }
    if (stats.queue_depth != 0 || stats.executing != 0 || stats.live_workers != 0) {
        add_reason(report.invalid_reasons, "not_quiescent");
    }
    const std::uint64_t terminals = trial.succeeded + trial.failed_inference + trial.failed_worker +
                                    trial.failed_scheduler + trial.canceled + trial.timeouts;
    const std::uint64_t rejection_total =
        trial.rejected_full + trial.rejected_closed + trial.rejected_not_running;
    if (trial.scheduled != trial.attempts + trial.skips_outstanding + trial.skips_lag ||
        trial.attempts != trial.accepted + rejection_total || trial.accepted != terminals) {
        add_reason(report.invalid_reasons, "runner_accounting");
    }
    if (trial.timeouts == 0 &&
        (stats.accepted != trial.accepted || stats.succeeded != trial.succeeded ||
         stats.failed_inference != trial.failed_inference ||
         stats.failed_worker != trial.failed_worker ||
         stats.failed_scheduler != trial.failed_scheduler || stats.canceled != trial.canceled ||
         stats.rejected_full != trial.rejected_full ||
         stats.rejected_closed != trial.rejected_closed)) {
        add_reason(report.invalid_reasons, "terminal_accounting");
    }
    if (!histogram_holds(metrics, report.histogram_saturated)) {
        add_reason(report.invalid_reasons, "histogram");
    }
    if (config.enable_completion_timing) {
        if (trial.missing_pre_stamps != 0) {
            add_reason(report.invalid_reasons, "missing_pre_publication_stamp");
        }
        if (trial.unrepresentable_timing != 0) {
            add_reason(report.invalid_reasons, "timing_unrepresentable");
        }
        if (trial.negative_timing != 0) {
            add_reason(report.invalid_reasons, "negative_timing_duration");
        }
        if (report.pre_publication_latency.sample_count != trial.succeeded ||
            report.pre_publication_to_observation_delay.sample_count != trial.succeeded) {
            add_reason(report.invalid_reasons, "timing_sample_count");
        }
        if (!std::isfinite(report.pre_publication_latency.mean_nanoseconds) ||
            !std::isfinite(report.pre_publication_to_observation_delay.mean_nanoseconds) ||
            !std::isfinite(report.generator_lag.mean_nanoseconds)) {
            add_reason(report.invalid_reasons, "timing_summary");
        }
    }
    if (config.enable_metrics) {
        if (!metrics.enabled || !metrics.finalized) {
            add_reason(report.invalid_reasons, "metrics_not_finalized");
        }
    } else if (metrics.enabled || !metrics.finalized || metrics.engine_run_count != 0 ||
               metrics.requests_entered_engine_runs != 0 ||
               metrics.total_assembly_nanoseconds != 0 ||
               metrics.total_engine_run_nanoseconds != 0 || metrics.total_split_nanoseconds != 0) {
        add_reason(report.invalid_reasons, "disabled_metrics");
    }
    report.valid = report.invalid_reasons.empty();
    return report;
}

SchedulerBenchmarkRun fail_run(SchedulerBenchmarkErrorKind kind, std::string message,
                               InferenceError load = {}) {
    SchedulerBenchmarkRun run;
    run.kind = kind;
    run.message = std::move(message);
    run.load_error = std::move(load);
    return run;
}

SchedulerBenchmarkRun warmup(const SchedulerBenchmarkConfig& config) {
    if (config.warmup_requests == 0) {
        return {};
    }
    InferenceScheduler scheduler(scheduler_config(config));
    const SchedulerError started = scheduler.start();
    if (started.kind == SchedulerErrorKind::kModelLoad) {
        return fail_run(SchedulerBenchmarkErrorKind::kModelLoad, started.message,
                        started.load_error);
    }
    if (started.kind != SchedulerErrorKind::kNone) {
        return fail_run(SchedulerBenchmarkErrorKind::kRuntime, started.message);
    }
    for (std::uint64_t index = 0; index < config.warmup_requests; ++index) {
        SchedulerBenchmarkTensors tensors = scheduler_benchmark_request(config.model_id, index + 1);
        Admission admission = scheduler.submit(std::move(tensors.inputs));
        if (admission.status != AdmissionStatus::kAccepted || !admission.completion.has_value()) {
            (void)scheduler.shutdown(ShutdownPolicy::kDrain);
            return fail_run(SchedulerBenchmarkErrorKind::kRuntime,
                            "warm-up request was not accepted");
        }
        if (admission.completion->wait_for(config.completion_timeout) !=
            std::future_status::ready) {
            (void)scheduler.shutdown(ShutdownPolicy::kDrain);
            return fail_run(SchedulerBenchmarkErrorKind::kRuntime, "warm-up completion timed out");
        }
        try {
            (void)admission.completion->get();
        } catch (const std::exception& exception) {
            (void)scheduler.shutdown(ShutdownPolicy::kDrain);
            return fail_run(SchedulerBenchmarkErrorKind::kRuntime, exception.what());
        }
    }
    const ShutdownResult shutdown = scheduler.shutdown(ShutdownPolicy::kDrain);
    if (shutdown.error.kind != SchedulerErrorKind::kNone) {
        return fail_run(SchedulerBenchmarkErrorKind::kRuntime, shutdown.error.message);
    }
    return {};
}

void submit_after_shutdown(InferenceScheduler& scheduler, SharedTrial& trial,
                           const SchedulerBenchmarkConfig& config) {
    if (!config.test_submit_after_shutdown) {
        return;
    }
    Inflight item;
    item.attempt_id = 1;
    if (trial.model) {
        const std::string prepared =
            prepare_inflight_expectation(*trial.model, config.model_id, 1, item);
        if (!prepared.empty()) {
            std::lock_guard<std::mutex> lock(trial.mutex);
            remember_error(trial, 1, "reference_preparation", prepared);
            return;
        }
    } else {
        SchedulerBenchmarkTensors tensors = scheduler_benchmark_request(config.model_id, 1);
        item.inputs = tensors.inputs;
        item.expected_c0 = tensors.expected_c0;
        item.expected_c1 = tensors.expected_c1;
    }
    Admission admission = scheduler.submit(item.inputs);
    std::lock_guard<std::mutex> lock(trial.mutex);
    ++trial.scheduled;
    account_admission(trial, std::move(admission), std::move(item));
}

SchedulerBenchmarkRun fail_from_start(const SchedulerError& started) {
    if (started.kind == SchedulerErrorKind::kModelLoad) {
        return fail_run(SchedulerBenchmarkErrorKind::kModelLoad, started.message,
                        started.load_error);
    }
    if (started.kind == SchedulerErrorKind::kInvalidConfig) {
        return fail_run(SchedulerBenchmarkErrorKind::kInvalidConfig, started.message);
    }
    return fail_run(SchedulerBenchmarkErrorKind::kRuntime, started.message);
}

constexpr std::uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

constexpr char kAddOnnxSha256[] =
    "f33bbe0ad5420419dd2dbb5322f4172e213b854f1124c660b1d732e8f8ce3b23";
constexpr char kSmallOnnxSha256[] =
    "bcf02cac789c65cb12785ccfd175827cb21f89574ef298af009b74d7131df1bd";
constexpr char kMediumOnnxSha256[] =
    "8d43a97b8d44d3d1f601570c8b09b08c90dda4977e6f8b7123eea8c0455024dd";
constexpr char kGemmOnnxSha256[] =
    "a6203020b870eb04f9fd8e836cf4ccd930a57156a026d2a50163121d83debf32";
constexpr char kSmallWeightsSha256[] =
    "b6335064c4e2c89add3ed54452e27dacf3ea7d485be6adbdd24fae1e2e5f1899";
constexpr char kMediumWeightsSha256[] =
    "d3dd2f6148a6684288d4f25b0c426eefde7227468be9120ddd08cd99cbb9059e";
constexpr char kGemmWeightsSha256[] =
    "a335c8b12defcb885f8bada9ef1092655d024d5586fe011a9336ad2d472f82b6";

struct ModelPin {
    const char* input_name;
    const char* output_name;
    std::size_t input_width;
    std::size_t output_width;
    const char* onnx_sha256;
    const char* weights_file;
    const char* weights_sha256;
    bool neural;
    NeuralWorkload workload;
};

ModelPin pin_for(SchedulerBenchmarkModelId id) {
    switch (id) {
    case SchedulerBenchmarkModelId::kMlpSmallF32:
        return {"features",
                "logits",
                32,
                8,
                kSmallOnnxSha256,
                "mlp_small_f32.weights",
                kSmallWeightsSha256,
                true,
                NeuralWorkload::kMlpSmall};
    case SchedulerBenchmarkModelId::kMlpMediumF32:
        return {"features",
                "logits",
                256,
                16,
                kMediumOnnxSha256,
                "mlp_medium_f32.weights",
                kMediumWeightsSha256,
                true,
                NeuralWorkload::kMlpMedium};
    case SchedulerBenchmarkModelId::kGemmF32:
        return {"features",
                "product",
                512,
                512,
                kGemmOnnxSha256,
                "gemm_f32.weights",
                kGemmWeightsSha256,
                true,
                NeuralWorkload::kGemm};
    case SchedulerBenchmarkModelId::kBatchAddF32:
        break;
    }
    return {"A", "C", 2, 2, kAddOnnxSha256, nullptr, nullptr, false, NeuralWorkload::kMlpSmall};
}

std::uint32_t rotate_right(std::uint32_t value, std::uint32_t bits) noexcept {
    return (value >> bits) | (value << (32U - bits));
}

struct Sha256State {
    std::uint32_t h[8]{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                       0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::uint64_t bits{0};
    std::uint8_t block[64]{};
    std::size_t filled{0};
    bool overflow{false};
};

void transform_block(Sha256State& state, const std::uint8_t block[64]) {
    std::uint32_t words[64];
    for (int index = 0; index < 16; ++index) {
        const std::size_t offset = static_cast<std::size_t>(index) * 4U;
        words[index] = (std::uint32_t{block[offset]} << 24U) |
                       (std::uint32_t{block[offset + 1]} << 16U) |
                       (std::uint32_t{block[offset + 2]} << 8U) | std::uint32_t{block[offset + 3]};
    }
    for (int index = 16; index < 64; ++index) {
        const std::uint32_t small = rotate_right(words[index - 15], 7) ^
                                    rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3U);
        const std::uint32_t large = rotate_right(words[index - 2], 17) ^
                                    rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10U);
        words[index] = words[index - 16] + small + words[index - 7] + large;
    }
    std::uint32_t a = state.h[0];
    std::uint32_t b = state.h[1];
    std::uint32_t c = state.h[2];
    std::uint32_t d = state.h[3];
    std::uint32_t e = state.h[4];
    std::uint32_t f = state.h[5];
    std::uint32_t g = state.h[6];
    std::uint32_t h = state.h[7];
    for (int index = 0; index < 64; ++index) {
        const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const std::uint32_t choose = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + sum1 + choose + kSha256K[index] + words[index];
        const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state.h[0] += a;
    state.h[1] += b;
    state.h[2] += c;
    state.h[3] += d;
    state.h[4] += e;
    state.h[5] += f;
    state.h[6] += g;
    state.h[7] += h;
}

bool absorb(Sha256State& state, const std::uint8_t* data, std::size_t length) {
    for (std::size_t index = 0; index < length; ++index) {
        state.block[state.filled++] = data[index];
        if (state.filled != 64) {
            continue;
        }
        if (state.bits > std::numeric_limits<std::uint64_t>::max() - 512U) {
            state.overflow = true;
            return false;
        }
        transform_block(state, state.block);
        state.bits += 512U;
        state.filled = 0;
    }
    return !state.overflow;
}

bool finish_sha256(Sha256State& state, std::uint8_t out[32]) {
    if (state.overflow) {
        return false;
    }
    const std::uint64_t total_bits = state.bits + static_cast<std::uint64_t>(state.filled) * 8U;
    if (state.filled > 64) {
        return false;
    }
    state.block[state.filled++] = 0x80U;
    if (state.filled > 56) {
        while (state.filled < 64) {
            state.block[state.filled++] = 0;
        }
        transform_block(state, state.block);
        state.filled = 0;
    }
    while (state.filled < 56) {
        state.block[state.filled++] = 0;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        state.block[state.filled++] = static_cast<std::uint8_t>(total_bits >> shift);
    }
    transform_block(state, state.block);
    for (int word = 0; word < 8; ++word) {
        const std::uint32_t value = state.h[word];
        out[word * 4] = static_cast<std::uint8_t>(value >> 24U);
        out[word * 4 + 1] = static_cast<std::uint8_t>(value >> 16U);
        out[word * 4 + 2] = static_cast<std::uint8_t>(value >> 8U);
        out[word * 4 + 3] = static_cast<std::uint8_t>(value);
    }
    return true;
}

std::string hex_digest(const std::uint8_t bytes[32]) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string hex(64, '0');
    for (int index = 0; index < 32; ++index) {
        hex[static_cast<std::size_t>(index) * 2U] = kHex[bytes[index] >> 4U];
        hex[static_cast<std::size_t>(index) * 2U + 1U] = kHex[bytes[index] & 0x0fU];
    }
    return hex;
}

SchedulerBenchmarkDigest hash_regular_file(const std::filesystem::path& path) {
    SchedulerBenchmarkDigest digest;
    std::error_code status_error;
    const std::filesystem::file_status status = std::filesystem::symlink_status(path, status_error);
    if (status_error) {
        digest.message = status_error == std::errc::no_such_file_or_directory
                             ? "file is missing"
                             : "file status could not be read";
        return digest;
    }
    if (status.type() == std::filesystem::file_type::not_found) {
        digest.message = "file is missing";
        return digest;
    }
    if (status.type() != std::filesystem::file_type::regular) {
        digest.message = "file is not a regular file";
        return digest;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        digest.message = "file could not be opened";
        return digest;
    }
    Sha256State state;
    std::array<char, 8192> buffer{};
    while (true) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            if (!absorb(state, reinterpret_cast<const std::uint8_t*>(buffer.data()),
                        static_cast<std::size_t>(count))) {
                digest.message = "file length exceeds SHA-256";
                return digest;
            }
        }
        if (input.eof()) {
            break;
        }
        if (!input) {
            digest.message = "file could not be read";
            return digest;
        }
    }
    std::uint8_t raw[32];
    if (!finish_sha256(state, raw)) {
        digest.message = "file length exceeds SHA-256";
        return digest;
    }
    digest.ok = true;
    digest.hex = hex_digest(raw);
    return digest;
}

bool static_batch_row(const TensorMetadata& metadata, std::int64_t width) {
    return metadata.element_type == ElementType::kFloat32 && metadata.dimensions.size() == 2 &&
           metadata.dimensions[0] < 0 && metadata.dimensions[1] == width;
}

std::string metadata_matches(const InferenceEngine& engine, const ModelPin& pin,
                             SchedulerBenchmarkModelId id) {
    const std::vector<TensorMetadata>& inputs = engine.inputs();
    const std::vector<TensorMetadata>& outputs = engine.outputs();
    if (!pin.neural) {
        if (inputs.size() != 2 || outputs.size() != 1 || inputs[0].name != "A" ||
            inputs[1].name != "B" || outputs[0].name != "C" || !static_batch_row(inputs[0], 2) ||
            !static_batch_row(inputs[1], 2) || !static_batch_row(outputs[0], 2)) {
            return "model metadata does not match batch_add_f32";
        }
        return {};
    }
    if (inputs.size() != 1 || outputs.size() != 1 || inputs[0].name != pin.input_name ||
        outputs[0].name != pin.output_name ||
        !static_batch_row(inputs[0], static_cast<std::int64_t>(pin.input_width)) ||
        !static_batch_row(outputs[0], static_cast<std::int64_t>(pin.output_width))) {
        return std::string{"model metadata does not match "} +
               (id == SchedulerBenchmarkModelId::kMlpSmallF32    ? "mlp_small_f32"
                : id == SchedulerBenchmarkModelId::kMlpMediumF32 ? "mlp_medium_f32"
                                                                 : "gemm_f32");
    }
    return {};
}

struct PrepareResult {
    std::string error;
    std::shared_ptr<PreparedModel> model;
};

PrepareResult prepare_model(const SchedulerBenchmarkConfig& config) {
    PrepareResult result;
    const ModelPin pin = pin_for(config.model_id);
    const SchedulerBenchmarkDigest model_digest = hash_regular_file(config.model_path);
    if (!model_digest.ok) {
        result.error = "model " + model_digest.message;
        return result;
    }
    if (model_digest.hex != pin.onnx_sha256) {
        result.error = "model SHA-256 mismatch for the selected model id";
        return result;
    }
    EngineLoad loaded = load_inference_engine(config.model_path);
    if (loaded.error.kind != InferenceErrorKind::kNone || !loaded.engine) {
        result.error = loaded.error.message.empty() ? "model metadata could not be loaded"
                                                    : loaded.error.message;
        return result;
    }
    const std::string metadata = metadata_matches(*loaded.engine, pin, config.model_id);
    loaded.engine.reset();
    if (!metadata.empty()) {
        result.error = metadata;
        return result;
    }
    auto prepared = std::make_shared<PreparedModel>();
    prepared->id = config.model_id;
    prepared->neural = pin.neural;
    prepared->workload = pin.workload;
    prepared->model_digest = model_digest.hex;
    if (pin.neural) {
        prepared->weights_path =
            std::filesystem::path{config.model_path}.parent_path() / "fixtures" / pin.weights_file;
        const SchedulerBenchmarkDigest weights_digest = hash_regular_file(prepared->weights_path);
        if (!weights_digest.ok) {
            result.error = "weight sidecar " + weights_digest.message;
            return result;
        }
        if (weights_digest.hex != pin.weights_sha256) {
            result.error = "weight sidecar SHA-256 mismatch for the selected model id";
            return result;
        }
        const NeuralContract contract = neural_contract(pin.workload);
        const NeuralFloatFile weights =
            load_float32_file(prepared->weights_path, contract.parameter_count);
        if (weights.error.kind != NeuralReferenceErrorKind::kNone) {
            result.error = weights.error.message.empty() ? "weight sidecar could not be loaded"
                                                         : weights.error.message;
            return result;
        }
        prepared->weights = std::move(weights.values);
        prepared->weights_digest = weights_digest.hex;
    }
    result.model = std::move(prepared);
    return result;
}

std::string recheck_identity(const PreparedModel& model, const SchedulerBenchmarkConfig& config) {
    const SchedulerBenchmarkDigest model_digest = hash_regular_file(config.model_path);
    if (!model_digest.ok) {
        return "model " + model_digest.message;
    }
    if (model_digest.hex != model.model_digest) {
        return "model bytes changed before measured execution";
    }
    if (!model.neural) {
        return {};
    }
    const SchedulerBenchmarkDigest weights_digest = hash_regular_file(model.weights_path);
    if (!weights_digest.ok) {
        return "weight sidecar " + weights_digest.message;
    }
    if (weights_digest.hex != model.weights_digest) {
        return "weight sidecar bytes changed before measured execution";
    }
    return {};
}

} // namespace

SchedulerConfig scheduler_config(const SchedulerBenchmarkConfig& config) {
    SchedulerConfig built;
    built.model_path = config.model_path;
    built.worker_count = config.worker_count;
    built.queue_capacity = config.queue_capacity;
    built.enable_batching = config.enable_batching;
    built.max_batch_size = config.max_batch_size;
    built.max_batch_delay = config.max_batch_delay;
    built.adaptive_batching = config.adaptive_batching;
    built.enable_metrics = config.enable_metrics;
    built.enable_completion_timing = config.enable_completion_timing;
    return built;
}

const char* scheduler_workload_name(SchedulerWorkloadKind workload) noexcept {
    switch (workload) {
    case SchedulerWorkloadKind::kOpenLoopConstant:
        return "open_loop_constant";
    case SchedulerWorkloadKind::kOpenLoopBurst:
        return "open_loop_burst";
    case SchedulerWorkloadKind::kClosedLoop:
        return "closed_loop";
    }
    return "unsupported";
}

SchedulerBenchmarkPlanResult plan_constant_arrivals(double requests_per_second,
                                                    std::chrono::nanoseconds duration,
                                                    std::uint64_t max_offered) {
    if (!finite_rate(requests_per_second) || period_for(requests_per_second).count() < 1) {
        return invalid_plan(SchedulerBenchmarkPlanError::kInvalidRate);
    }
    if (duration.count() <= 0) {
        return invalid_plan(SchedulerBenchmarkPlanError::kInvalidDuration);
    }
    if (max_offered < 1) {
        return invalid_plan(SchedulerBenchmarkPlanError::kInvalidBound);
    }
    SchedulerBenchmarkPlanResult result;
    fill_phase(result.plan, 0, requests_per_second, std::chrono::nanoseconds{0}, duration,
               max_offered);
    return result;
}

SchedulerBenchmarkPlanResult
plan_burst_arrivals(double high_requests_per_second, double low_requests_per_second,
                    std::chrono::nanoseconds high, std::chrono::nanoseconds low,
                    std::chrono::nanoseconds duration, std::uint64_t max_offered) {
    if (!finite_rate(high_requests_per_second) || !finite_rate(low_requests_per_second) ||
        period_for(high_requests_per_second).count() < 1 ||
        period_for(low_requests_per_second).count() < 1) {
        return invalid_plan(SchedulerBenchmarkPlanError::kInvalidRate);
    }
    if (duration.count() <= 0 || high.count() <= 0 || low.count() <= 0) {
        return invalid_plan(SchedulerBenchmarkPlanError::kInvalidDuration);
    }
    if (max_offered < 1) {
        return invalid_plan(SchedulerBenchmarkPlanError::kInvalidBound);
    }
    SchedulerBenchmarkPlanResult result;
    std::chrono::nanoseconds cursor{0};
    bool high_phase = true;
    std::uint32_t phase_index = 0;
    while (cursor < duration && result.plan.arrivals.size() < max_offered) {
        const std::chrono::nanoseconds length = high_phase ? high : low;
        const double rate = high_phase ? high_requests_per_second : low_requests_per_second;
        const std::chrono::nanoseconds end =
            cursor > duration - length ? duration : cursor + length;
        fill_phase(result.plan, phase_index, rate, cursor, end, max_offered);
        cursor = end;
        high_phase = !high_phase;
        ++phase_index;
        if (phase_index > 100000) {
            break;
        }
    }
    return result;
}

SchedulerBenchmarkLagSummary
scheduler_benchmark_summarize_lag(std::span<const std::chrono::nanoseconds> lags) {
    return summarize_lag(std::vector<std::chrono::nanoseconds>{lags.begin(), lags.end()});
}

SchedulerBenchmarkNanosecondSummary
scheduler_benchmark_summarize_timing(std::span<const std::chrono::nanoseconds> samples) {
    return summarize_nonnegative_nanos(
        std::vector<std::chrono::nanoseconds>{samples.begin(), samples.end()});
}

SchedulerBenchmarkLag measure_generator_lag(std::chrono::steady_clock::time_point scheduled,
                                            std::chrono::steady_clock::time_point actual,
                                            std::chrono::nanoseconds tolerance) noexcept {
    SchedulerBenchmarkLag measured;
    measured.lag = std::chrono::duration_cast<std::chrono::nanoseconds>(actual - scheduled);
    measured.exceeds_tolerance = measured.lag > tolerance;
    return measured;
}

SchedulerBenchmarkDigest scheduler_benchmark_sha256_file(const std::filesystem::path& path) {
    return hash_regular_file(path);
}

SchedulerBenchmarkTensors scheduler_benchmark_request(SchedulerBenchmarkModelId model,
                                                      std::uint64_t attempt_id) {
    const ModelPin pin = pin_for(model);
    SchedulerBenchmarkTensors tensors;
    if (!pin.neural) {
        const float a0 = static_cast<float>(attempt_id % 1000U);
        const float a1 = static_cast<float>((attempt_id / 1000U) % 1000U);
        constexpr float b0 = 1.F;
        constexpr float b1 = 2.F;
        tensors.inputs = {NamedTensor{"A", {1, 2}, {a0, a1}}, NamedTensor{"B", {1, 2}, {b0, b1}}};
        tensors.expected_c0 = a0 + b0;
        tensors.expected_c1 = a1 + b1;
        return tensors;
    }
    std::vector<float> values(pin.input_width);
    for (std::size_t index = 0; index < pin.input_width; ++index) {
        const std::uint32_t code =
            static_cast<std::uint32_t>(attempt_id % 251U) + static_cast<std::uint32_t>(index);
        values[index] = static_cast<float>(code) * 0x1p-10F;
    }
    tensors.inputs = {NamedTensor{
        pin.input_name, {1, static_cast<std::int64_t>(pin.input_width)}, std::move(values)}};
    return tensors;
}

SchedulerBenchmarkTensors scheduler_benchmark_request(std::uint64_t attempt_id) {
    return scheduler_benchmark_request(SchedulerBenchmarkModelId::kBatchAddF32, attempt_id);
}

SchedulerBenchmarkOutputCheck scheduler_benchmark_check_output(SchedulerBenchmarkModelId model,
                                                               const NamedTensor& output,
                                                               std::span<const float> reference) {
    const ModelPin pin = pin_for(model);
    SchedulerBenchmarkOutputCheck checked;
    if (output.name != pin.output_name) {
        checked.message = "output name mismatch";
        return checked;
    }
    if (output.shape.size() != 2 || output.shape[0] != 1 ||
        output.shape[1] != static_cast<std::int64_t>(pin.output_width) ||
        output.values.size() != pin.output_width) {
        checked.message = "output shape mismatch";
        return checked;
    }
    if (reference.size() != pin.output_width) {
        checked.message = "reference width mismatch";
        return checked;
    }
    if (!pin.neural) {
        if (output.values[0] != reference[0] || output.values[1] != reference[1]) {
            checked.message = "output C did not match the attempt";
            return checked;
        }
        checked.matched = true;
        return checked;
    }
    for (const float value : output.values) {
        if (!std::isfinite(value)) {
            checked.message = "output is not finite";
            return checked;
        }
    }
    for (const float value : reference) {
        if (!std::isfinite(value)) {
            checked.message = "reference is not finite";
            return checked;
        }
    }
    const NeuralMismatch mismatch = neural_worst_mismatch(pin.workload, output.values, reference);
    if (!mismatch.within) {
        checked.message = "output exceeded the neural tolerance";
        return checked;
    }
    checked.matched = true;
    return checked;
}

SchedulerBenchmarkOutputCheck
scheduler_benchmark_check_returned_inputs(std::span<const NamedTensor> returned,
                                          std::span<const NamedTensor> submitted) {
    SchedulerBenchmarkOutputCheck checked;
    if (returned.size() != submitted.size()) {
        checked.message = "returned input count mismatch";
        return checked;
    }
    for (std::size_t index = 0; index < returned.size(); ++index) {
        if (returned[index].name != submitted[index].name) {
            checked.message = "returned input name mismatch";
            return checked;
        }
        if (returned[index].shape != submitted[index].shape ||
            returned[index].values.size() != submitted[index].values.size()) {
            checked.message = "returned input shape mismatch";
            return checked;
        }
        if (returned[index].values != submitted[index].values) {
            checked.message = "returned input value mismatch";
            return checked;
        }
    }
    checked.matched = true;
    return checked;
}

bool scheduler_benchmark_should_check(std::uint64_t one_based_index, std::uint32_t check_interval,
                                      bool is_last) noexcept {
    if (check_interval <= 1 || is_last) {
        return true;
    }
    if (one_based_index == 0) {
        return false;
    }
    return one_based_index % check_interval == 0;
}

bool scheduler_benchmark_future_ready(std::future<SchedulerCompletion>& completion) {
    return completion.wait_for(std::chrono::nanoseconds{0}) == std::future_status::ready;
}

int scheduler_benchmark_status(const SchedulerBenchmarkRun& run) noexcept {
    if (run.kind == SchedulerBenchmarkErrorKind::kInvalidConfig) {
        return 2;
    }
    if (run.kind != SchedulerBenchmarkErrorKind::kNone || !run.report.valid) {
        return 1;
    }
    return 0;
}

SchedulerBenchmarkRun run_scheduler_benchmark(const SchedulerBenchmarkConfig& config) {
    const std::string problem = validate_config(config);
    if (!problem.empty()) {
        return fail_run(SchedulerBenchmarkErrorKind::kInvalidConfig, problem);
    }
    SchedulerBenchmarkPlanResult planned;
    if (config.workload == SchedulerWorkloadKind::kOpenLoopConstant) {
        planned = plan_constant_arrivals(config.target_requests_per_second, config.duration,
                                         config.max_offered);
    } else if (config.workload == SchedulerWorkloadKind::kOpenLoopBurst) {
        planned = plan_burst_arrivals(config.burst_high_requests_per_second,
                                      config.burst_low_requests_per_second, config.burst_high,
                                      config.burst_low, config.duration, config.max_offered);
    }
    if (config.workload != SchedulerWorkloadKind::kClosedLoop &&
        planned.error != SchedulerBenchmarkPlanError::kNone) {
        return fail_run(SchedulerBenchmarkErrorKind::kInvalidConfig, "arrival plan was rejected");
    }
    PrepareResult prepared;
    try {
        prepared = prepare_model(config);
    } catch (const std::bad_alloc&) {
        return fail_run(SchedulerBenchmarkErrorKind::kRuntime, "allocation failed");
    } catch (const std::exception& exception) {
        return fail_run(SchedulerBenchmarkErrorKind::kRuntime, exception.what());
    }
    if (!prepared.error.empty() || !prepared.model) {
        return fail_run(SchedulerBenchmarkErrorKind::kInvalidConfig,
                        prepared.error.empty() ? "model identity was not prepared"
                                               : prepared.error);
    }
    if (const SchedulerBenchmarkRun warmed = warmup(config);
        warmed.kind != SchedulerBenchmarkErrorKind::kNone) {
        return warmed;
    }
    if (const std::string changed = recheck_identity(*prepared.model, config); !changed.empty()) {
        return fail_run(SchedulerBenchmarkErrorKind::kInvalidConfig, changed);
    }

    InferenceScheduler scheduler(scheduler_config(config));
    const SchedulerError started = scheduler.start();
    if (started.kind != SchedulerErrorKind::kNone) {
        return fail_from_start(started);
    }

    SharedTrial trial;
    trial.model = prepared.model;
    trial.inflight.reserve(config.max_outstanding);
    trial.latencies.reserve(static_cast<std::size_t>(
        std::min(config.max_offered, static_cast<std::uint64_t>(config.max_outstanding))));
    if (config.enable_completion_timing) {
        trial.pre_publication_latencies.reserve(static_cast<std::size_t>(config.max_offered));
        trial.observation_delays.reserve(static_cast<std::size_t>(config.max_offered));
    }
    const auto window_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point arrival_end = window_start;
    try {
        if (config.workload == SchedulerWorkloadKind::kClosedLoop) {
            trial.latencies.reserve(static_cast<std::size_t>(config.max_offered));
            JoinThreads producers;
            producers.threads.reserve(config.producer_count);
            for (std::uint32_t index = 0; index < config.producer_count; ++index) {
                producers.threads.emplace_back(run_closed_producer, std::ref(scheduler),
                                               std::cref(config), std::ref(trial), window_start);
            }
            for (std::thread& producer : producers.threads) {
                if (producer.joinable()) {
                    producer.join();
                }
            }
            arrival_end = std::chrono::steady_clock::now();
            if (trial.runtime_failed) {
                (void)scheduler.shutdown(ShutdownPolicy::kDrain);
                return fail_run(SchedulerBenchmarkErrorKind::kRuntime, trial.runtime_message);
            }
        } else {
            JoinThread collector;
            JoinThread generator;
            // Destroyed before those joins. An exception sets the collector's
            // exit flags first so join cannot wait for a shutdown that never
            // published them.
            struct ReleaseCollectors {
                SharedTrial& trial;
                ~ReleaseCollectors() {
                    {
                        std::lock_guard<std::mutex> lock(trial.mutex);
                        trial.generators_done = true;
                        trial.shutdown_done = true;
                    }
                    trial.ready.notify_all();
                }
            } release{trial};
            collector.thread =
                std::thread(run_open_loop_collector, std::ref(trial), std::cref(config));
            generator.thread =
                std::thread(run_open_loop_generator, std::ref(scheduler), std::cref(config),
                            std::cref(planned.plan), std::ref(trial), window_start);
            if (generator.thread.joinable()) {
                generator.thread.join();
            }
            arrival_end = std::chrono::steady_clock::now();
            const ShutdownResult shutdown = scheduler.shutdown(ShutdownPolicy::kDrain);
            {
                std::lock_guard<std::mutex> lock(trial.mutex);
                trial.shutdown_done = true;
                if (shutdown.error.kind != SchedulerErrorKind::kNone) {
                    trial.runtime_failed = true;
                    trial.runtime_message = shutdown.error.message;
                }
            }
            trial.ready.notify_all();
            if (collector.thread.joinable()) {
                collector.thread.join();
            }
            if (trial.runtime_failed) {
                return fail_run(SchedulerBenchmarkErrorKind::kRuntime, trial.runtime_message);
            }
            submit_after_shutdown(scheduler, trial, config);
            const auto completion = std::chrono::steady_clock::now() - arrival_end;
            const SchedulerMetrics metrics = scheduler.metrics();
            const SchedulerStats stats = scheduler.stats();
            SchedulerBenchmarkRun run;
            run.report = make_report(config, trial, planned.plan, arrival_end - window_start,
                                     completion, metrics, stats);
            return run;
        }
    } catch (const std::exception& exception) {
        return fail_run(SchedulerBenchmarkErrorKind::kRuntime, exception.what());
    }

    const ShutdownResult shutdown = scheduler.shutdown(ShutdownPolicy::kDrain);
    if (shutdown.error.kind != SchedulerErrorKind::kNone) {
        return fail_run(SchedulerBenchmarkErrorKind::kRuntime, shutdown.error.message);
    }
    submit_after_shutdown(scheduler, trial, config);
    const auto completion = std::chrono::steady_clock::now() - arrival_end;
    const SchedulerMetrics metrics = scheduler.metrics();
    const SchedulerStats stats = scheduler.stats();
    SchedulerBenchmarkRun run;
    run.report = make_report(config, trial, planned.plan, arrival_end - window_start, completion,
                             metrics, stats);
    return run;
}

} // namespace inferedge
