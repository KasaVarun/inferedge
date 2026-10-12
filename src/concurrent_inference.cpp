#include "inferedge/concurrent_inference.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace inferedge {
namespace {

struct WorkerSlot {
    bool failed{false};
    std::uint32_t completed_requests{0};
    WorkerError error;
    std::vector<NamedTensor> last_outputs;
};

struct StartAborted : std::exception {
    std::uint32_t worker_index{0};
    explicit StartAborted(std::uint32_t index) : worker_index{index} {}
};

ConcurrentExecution FailPlan(std::string message) {
    ConcurrentExecution result;
    result.kind = ConcurrentErrorKind::kInvalidPlan;
    WorkerError error;
    error.kind = ConcurrentErrorKind::kInvalidPlan;
    error.message = std::move(message);
    result.errors.push_back(std::move(error));
    return result;
}

enum class ShapeCountKind { kOk, kNegative, kOverflow };

ShapeCountKind shape_element_count(const std::vector<std::int64_t>& shape, std::int64_t& count) {
    count = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension < 0) {
            return ShapeCountKind::kNegative;
        }
        if (dimension > 0 && count > std::numeric_limits<std::int64_t>::max() / dimension) {
            return ShapeCountKind::kOverflow;
        }
        count *= dimension;
    }
    return ShapeCountKind::kOk;
}

bool finite_tolerance(double value) {
    return std::isfinite(value) && value >= 0.0;
}

ConcurrentExecution validate_plan(const ConcurrentPlan& plan) {
    if (plan.workers.empty()) {
        return FailPlan("worker count is zero");
    }
    if (plan.workers.size() > kMaxConcurrentWorkers) {
        return FailPlan("worker count exceeds 64");
    }
    if (plan.requests_per_worker == 0) {
        return FailPlan("requests per worker is zero");
    }
    if (plan.requests_per_worker > kMaxConcurrentRequestsPerWorker) {
        return FailPlan("requests per worker exceeds 1000");
    }
    if (plan.stop_after_starting_workers.has_value() &&
        *plan.stop_after_starting_workers >= plan.workers.size()) {
        return FailPlan("stop_after_starting_workers is outside the worker range");
    }
    if (plan.throw_in_worker.has_value() && *plan.throw_in_worker >= plan.workers.size()) {
        return FailPlan("throw_in_worker is outside the worker range");
    }

    for (std::size_t worker_index = 0; worker_index < plan.workers.size(); ++worker_index) {
        const WorkerRequest& worker = plan.workers[worker_index];
        const std::string label = "worker " + std::to_string(worker_index);
        if (worker.expected.empty()) {
            return FailPlan(label + " expected output list is empty");
        }
        for (std::size_t index = 0; index < worker.expected.size(); ++index) {
            const WorkerExpectedTensor& tensor = worker.expected[index];
            if (tensor.name.empty()) {
                return FailPlan(label + " expected output name is empty");
            }
            for (std::size_t earlier = 0; earlier < index; ++earlier) {
                if (worker.expected[earlier].name == tensor.name) {
                    return FailPlan(label + " duplicate expected output name: " + tensor.name);
                }
            }
            std::int64_t count = 0;
            const ShapeCountKind shape = shape_element_count(tensor.shape, count);
            if (shape == ShapeCountKind::kNegative) {
                return FailPlan(label + " expected output " + tensor.name +
                                " has a negative dimension");
            }
            if (shape == ShapeCountKind::kOverflow) {
                return FailPlan(label + " expected output " + tensor.name +
                                " shape product overflows");
            }
            if (static_cast<std::uint64_t>(count) >
                    static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
                static_cast<std::size_t>(count) != tensor.values.size()) {
                return FailPlan(label + " expected output " + tensor.name +
                                " element count does not match its shape");
            }
            for (const float value : tensor.values) {
                if (!std::isfinite(value)) {
                    return FailPlan(label + " expected output " + tensor.name +
                                    " has a non-finite value");
                }
            }
            if (!finite_tolerance(tensor.relative_tolerance) ||
                !finite_tolerance(tensor.absolute_tolerance)) {
                return FailPlan(label + " expected output " + tensor.name +
                                " has an invalid tolerance");
            }
        }
    }
    return ConcurrentExecution{};
}

const NamedTensor* find_output(const std::vector<NamedTensor>& outputs, const std::string& name) {
    for (const NamedTensor& output : outputs) {
        if (output.name == name) {
            return &output;
        }
    }
    return nullptr;
}

bool within_tolerance(float actual, float expected, double absolute_tolerance,
                      double relative_tolerance) {
    const double difference =
        std::fabs(static_cast<double>(actual) - static_cast<double>(expected));
    const double limit =
        absolute_tolerance + relative_tolerance * std::fabs(static_cast<double>(expected));
    return difference <= limit;
}

void record_failure(WorkerSlot& slot, std::uint32_t worker_index, std::uint32_t iteration,
                    ConcurrentErrorKind kind, InferenceErrorKind inference_kind,
                    std::string message, std::atomic<bool>& stop) {
    if (slot.failed) {
        return;
    }
    slot.failed = true;
    slot.last_outputs.clear();
    slot.error.worker_index = worker_index;
    slot.error.request_iteration = iteration;
    slot.error.kind = kind;
    slot.error.inference_kind = inference_kind;
    slot.error.message = std::move(message);
    stop.store(true, std::memory_order_release);
}

bool outputs_match(const InferenceResult& result, const WorkerRequest& request,
                   std::string& message, ConcurrentErrorKind& kind) {
    if (result.outputs.size() != request.expected.size()) {
        kind = ConcurrentErrorKind::kOutputMismatch;
        message = "output count is " + std::to_string(result.outputs.size()) + ", expected " +
                  std::to_string(request.expected.size());
        return false;
    }
    for (const WorkerExpectedTensor& tensor : request.expected) {
        const NamedTensor* output = find_output(result.outputs, tensor.name);
        if (output == nullptr) {
            kind = ConcurrentErrorKind::kOutputMismatch;
            message = "missing output name: " + tensor.name;
            return false;
        }
        if (output->shape != tensor.shape) {
            kind = ConcurrentErrorKind::kOutputMismatch;
            message = "output " + tensor.name + " shape does not match";
            return false;
        }
        if (output->values.size() != tensor.values.size()) {
            kind = ConcurrentErrorKind::kOutputMismatch;
            message = "output " + tensor.name + " element count does not match";
            return false;
        }
        for (std::size_t index = 0; index < output->values.size(); ++index) {
            if (!std::isfinite(output->values[index])) {
                kind = ConcurrentErrorKind::kNonFiniteOutput;
                message =
                    "output " + tensor.name + " value " + std::to_string(index) + " is non-finite";
                return false;
            }
        }
        for (std::size_t index = 0; index < output->values.size(); ++index) {
            if (!within_tolerance(output->values[index], tensor.values[index],
                                  tensor.absolute_tolerance, tensor.relative_tolerance)) {
                kind = ConcurrentErrorKind::kOutputMismatch;
                message = "output " + tensor.name + " value " + std::to_string(index) +
                          " is outside tolerance";
                return false;
            }
        }
    }
    return true;
}

void run_worker(std::uint32_t worker_index, InferenceEngine& engine, const WorkerRequest& request,
                std::uint32_t request_count, bool inject_exception, std::atomic<bool>& stop,
                WorkerSlot& slot) {
    try {
        if (inject_exception) {
            throw std::runtime_error("injected worker exception");
        }
        for (std::uint32_t iteration = 0; iteration < request_count; ++iteration) {
            if (stop.load(std::memory_order_acquire)) {
                break;
            }
            // Views borrow this worker's NamedTensor storage. They die before
            // the next iteration. The plan keeps that storage alive until join.
            std::vector<NamedTensorView> views;
            views.reserve(request.inputs.size());
            for (const NamedTensor& input : request.inputs) {
                views.push_back(
                    NamedTensorView{input.name, input.shape, std::span<const float>{input.values}});
            }
            const InferenceResult result = engine.run(views);
            if (result.error.kind != InferenceErrorKind::kNone) {
                record_failure(slot, worker_index, iteration, ConcurrentErrorKind::kInferenceFailed,
                               result.error.kind, result.error.message, stop);
                break;
            }
            std::string message;
            ConcurrentErrorKind kind = ConcurrentErrorKind::kNone;
            if (!outputs_match(result, request, message, kind)) {
                record_failure(slot, worker_index, iteration, kind, InferenceErrorKind::kNone,
                               std::move(message), stop);
                break;
            }
            slot.last_outputs = result.outputs;
            slot.completed_requests = iteration + 1;
        }
    } catch (const std::exception& error) {
        record_failure(slot, worker_index, 0, ConcurrentErrorKind::kWorkerException,
                       InferenceErrorKind::kNone, error.what(), stop);
    } catch (...) {
        record_failure(slot, worker_index, 0, ConcurrentErrorKind::kWorkerException,
                       InferenceErrorKind::kNone, "unknown worker exception", stop);
    }
}

WorkerError StartError(std::uint32_t worker_index, std::string message) {
    WorkerError error;
    error.worker_index = worker_index;
    error.request_iteration = 0;
    error.kind = ConcurrentErrorKind::kWorkerStart;
    error.message = std::move(message);
    return error;
}

ConcurrentExecution finish(std::vector<WorkerSlot>& slots, const ConcurrentPlan& plan,
                           std::vector<WorkerError> start_errors) {
    ConcurrentExecution result;
    result.errors = std::move(start_errors);
    for (std::uint32_t index = 0; index < slots.size(); ++index) {
        if (slots[index].failed) {
            result.errors.push_back(std::move(slots[index].error));
        }
    }
    if (!result.errors.empty()) {
        result.kind = result.errors.front().kind;
        return result;
    }
    result.outputs.reserve(slots.size());
    for (std::uint32_t index = 0; index < slots.size(); ++index) {
        if (slots[index].completed_requests != plan.requests_per_worker) {
            result.kind = ConcurrentErrorKind::kWorkerException;
            WorkerError error;
            error.worker_index = index;
            error.kind = ConcurrentErrorKind::kWorkerException;
            error.message = "worker stopped before completing its requests";
            result.errors.push_back(std::move(error));
            result.outputs.clear();
            return result;
        }
        WorkerOutput output;
        output.worker_index = index;
        output.completed_requests = slots[index].completed_requests;
        output.outputs = std::move(slots[index].last_outputs);
        result.outputs.push_back(std::move(output));
    }
    return result;
}

ConcurrentExecution execute(std::vector<std::unique_ptr<InferenceEngine>> engines,
                            const ConcurrentPlan& plan) {
    const auto worker_count = static_cast<std::uint32_t>(plan.workers.size());
    std::vector<WorkerSlot> slots(worker_count);
    std::atomic<bool> stop{false};
    std::vector<std::jthread> threads;
    threads.reserve(worker_count);
    std::vector<WorkerError> start_errors;

    try {
        for (std::uint32_t index = 0; index < worker_count; ++index) {
            if (plan.stop_after_starting_workers.has_value() &&
                index == *plan.stop_after_starting_workers) {
                stop.store(true, std::memory_order_release);
                throw StartAborted{index};
            }
            const bool inject = plan.throw_in_worker.has_value() && *plan.throw_in_worker == index;
            InferenceEngine* engine = engines[index].get();
            threads.emplace_back([index, engine, &plan, &stop, &slots, inject]() {
                try {
                    run_worker(index, *engine, plan.workers[index], plan.requests_per_worker,
                               inject, stop, slots[index]);
                } catch (...) {
                    // Assignments above the flag are non-throwing. The message
                    // string is best-effort so this handler cannot escape into
                    // std::terminate while the thread function is unwinding.
                    if (!slots[index].failed) {
                        slots[index].failed = true;
                        slots[index].error.worker_index = index;
                        slots[index].error.kind = ConcurrentErrorKind::kWorkerException;
                    }
                    stop.store(true, std::memory_order_release);
                    try {
                        if (slots[index].error.message.empty()) {
                            slots[index].error.message = "worker exception";
                        }
                    } catch (...) {
                    }
                }
            });
        }
    } catch (const StartAborted& aborted) {
        threads.clear();
        start_errors.push_back(StartError(aborted.worker_index, "worker thread was not started"));
        return finish(slots, plan, std::move(start_errors));
    } catch (const std::exception& error) {
        stop.store(true, std::memory_order_release);
        const auto failed_index = static_cast<std::uint32_t>(threads.size());
        threads.clear();
        start_errors.push_back(StartError(failed_index, error.what()));
        return finish(slots, plan, std::move(start_errors));
    } catch (...) {
        stop.store(true, std::memory_order_release);
        const auto failed_index = static_cast<std::uint32_t>(threads.size());
        threads.clear();
        start_errors.push_back(
            StartError(failed_index, "unknown exception while starting workers"));
        return finish(slots, plan, std::move(start_errors));
    }

    threads.clear();
    return finish(slots, plan, std::move(start_errors));
}

} // namespace

ConcurrentExecution run_concurrent_inference(const std::filesystem::path& model_path,
                                             const ConcurrentPlan& plan) {
    const ConcurrentExecution invalid = validate_plan(plan);
    if (invalid.kind != ConcurrentErrorKind::kNone) {
        return invalid;
    }

    std::vector<std::unique_ptr<InferenceEngine>> engines;
    engines.reserve(plan.workers.size());
    for (std::size_t index = 0; index < plan.workers.size(); ++index) {
        EngineLoad loaded = load_inference_engine(model_path);
        if (loaded.error.kind != InferenceErrorKind::kNone || loaded.engine == nullptr) {
            ConcurrentExecution result;
            result.kind = ConcurrentErrorKind::kModelLoad;
            if (loaded.error.kind == InferenceErrorKind::kNone) {
                result.load_error = InferenceError{InferenceErrorKind::kModelInvalid,
                                                   "model load returned no engine"};
            } else {
                result.load_error = std::move(loaded.error);
            }
            WorkerError error;
            error.worker_index = static_cast<std::uint32_t>(index);
            error.kind = ConcurrentErrorKind::kModelLoad;
            error.inference_kind = result.load_error.kind;
            error.message = result.load_error.message;
            result.errors.push_back(std::move(error));
            return result;
        }
        engines.push_back(std::move(loaded.engine));
    }
    return run_concurrent_inference(std::move(engines), plan);
}

ConcurrentExecution run_concurrent_inference(std::vector<std::unique_ptr<InferenceEngine>> engines,
                                             const ConcurrentPlan& plan) {
    const ConcurrentExecution invalid = validate_plan(plan);
    if (invalid.kind != ConcurrentErrorKind::kNone) {
        return invalid;
    }
    if (engines.size() != plan.workers.size()) {
        return FailPlan("engine count does not match worker count");
    }
    for (std::size_t index = 0; index < engines.size(); ++index) {
        if (engines[index] == nullptr) {
            return FailPlan("worker " + std::to_string(index) + " engine is null");
        }
    }
    try {
        return execute(std::move(engines), plan);
    } catch (const std::exception& error) {
        ConcurrentExecution result;
        result.kind = ConcurrentErrorKind::kWorkerException;
        WorkerError failure;
        failure.kind = ConcurrentErrorKind::kWorkerException;
        failure.message = error.what();
        result.errors.push_back(std::move(failure));
        return result;
    } catch (...) {
        ConcurrentExecution result;
        result.kind = ConcurrentErrorKind::kWorkerException;
        WorkerError failure;
        failure.kind = ConcurrentErrorKind::kWorkerException;
        failure.message = "unknown exception";
        result.errors.push_back(std::move(failure));
        return result;
    }
}

} // namespace inferedge
