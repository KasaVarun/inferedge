#include "inferedge/inference_engine.hpp"
#include "inferedge/inference_scheduler.hpp"
#include "inferedge/neural_reference.hpp"
#include "inferedge/tensor_batch.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <latch>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#ifndef INFEREDGE_MLP_SMALL_MODEL_PATH
#error "CMake must define INFEREDGE_MLP_SMALL_MODEL_PATH"
#endif
#ifndef INFEREDGE_MLP_MEDIUM_MODEL_PATH
#error "CMake must define INFEREDGE_MLP_MEDIUM_MODEL_PATH"
#endif
#ifndef INFEREDGE_GEMM_MODEL_PATH
#error "CMake must define INFEREDGE_GEMM_MODEL_PATH"
#endif
#ifndef INFEREDGE_MLP_SMALL_WEIGHTS_PATH
#error "CMake must define INFEREDGE_MLP_SMALL_WEIGHTS_PATH"
#endif
#ifndef INFEREDGE_MLP_MEDIUM_WEIGHTS_PATH
#error "CMake must define INFEREDGE_MLP_MEDIUM_WEIGHTS_PATH"
#endif
#ifndef INFEREDGE_GEMM_WEIGHTS_PATH
#error "CMake must define INFEREDGE_GEMM_WEIGHTS_PATH"
#endif

namespace {

using inferedge::Admission;
using inferedge::AdmissionStatus;
using inferedge::AssembleBatchInputs;
using inferedge::CompletionStatus;
using inferedge::ElementType;
using inferedge::evaluate_neural_reference;
using inferedge::InferenceErrorKind;
using inferedge::load_float32_file;
using inferedge::load_inference_engine;
using inferedge::NamedTensor;
using inferedge::neural_contract;
using inferedge::neural_worst_mismatch;
using inferedge::NeuralContract;
using inferedge::NeuralMismatch;
using inferedge::NeuralReferenceErrorKind;
using inferedge::NeuralWorkload;
using inferedge::SchedulerCompletion;
using inferedge::SchedulerConfig;
using inferedge::SchedulerErrorKind;
using inferedge::SchedulerTestAccess;
using inferedge::ShutdownPolicy;
using inferedge::SplitBatchOutputs;
using inferedge::TensorBatchErrorKind;
using inferedge::TensorMetadata;

constexpr std::size_t kBatchSizes[] = {1, 2, 4, 8, 16, 32};

struct ModelCase {
    NeuralWorkload workload;
    const char* model;
    const char* weights;
};

constexpr ModelCase kModels[] = {
    {NeuralWorkload::kMlpSmall, INFEREDGE_MLP_SMALL_MODEL_PATH, INFEREDGE_MLP_SMALL_WEIGHTS_PATH},
    {NeuralWorkload::kMlpMedium, INFEREDGE_MLP_MEDIUM_MODEL_PATH,
     INFEREDGE_MLP_MEDIUM_WEIGHTS_PATH},
    {NeuralWorkload::kGemm, INFEREDGE_GEMM_MODEL_PATH, INFEREDGE_GEMM_WEIGHTS_PATH},
};

const char* WorkloadName(NeuralWorkload workload) {
    switch (workload) {
    case NeuralWorkload::kMlpSmall:
        return "mlp_small_f32";
    case NeuralWorkload::kMlpMedium:
        return "mlp_medium_f32";
    case NeuralWorkload::kGemm:
        return "gemm_f32";
    }
    return "unknown";
}

struct RatioRecord {
    float ratio{0.F};
    std::size_t batch{0};
    std::size_t row{0};
    std::size_t column{0};
    bool saw_nonfinite{false};
};

struct ObservedBatches {
    mutable std::mutex mutex;
    std::vector<std::size_t> sizes;

    void Record(std::size_t count) {
        std::lock_guard<std::mutex> lock(mutex);
        sizes.push_back(count);
    }

    std::vector<std::size_t> Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        return sizes;
    }
};

struct LatchRelease {
    std::shared_ptr<std::latch> latch;

    void Release() {
        if (latch) {
            latch->count_down();
            latch.reset();
        }
    }

    ~LatchRelease() { Release(); }
};

class SchedulerNeuralBatchTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

float RowScale(std::size_t row) {
    if (row == 0) {
        return 0.F;
    }
    return std::ldexp(1.F, static_cast<int>(row));
}

std::vector<float> InputRow(std::size_t width, std::size_t row) {
    return std::vector<float>(width, RowScale(row));
}

NamedTensor InputTensor(const NeuralContract& contract, std::size_t row) {
    return NamedTensor{contract.input_name,
                       {1, static_cast<std::int64_t>(contract.input_width)},
                       InputRow(contract.input_width, row)};
}

std::vector<float> ReferenceRow(NeuralWorkload workload, const std::vector<float>& weights,
                                const std::vector<float>& row) {
    const auto evaluated = evaluate_neural_reference(workload, row, 1, weights);
    EXPECT_EQ(evaluated.error.kind, NeuralReferenceErrorKind::kNone) << evaluated.error.message;
    return evaluated.values;
}

bool SwapExceedsTolerance(NeuralWorkload workload, const std::vector<float>& left,
                          const std::vector<float>& right) {
    std::vector<float> original = left;
    original.insert(original.end(), right.begin(), right.end());
    std::vector<float> swapped = right;
    swapped.insert(swapped.end(), left.begin(), left.end());
    const NeuralMismatch mismatch = neural_worst_mismatch(workload, swapped, original);
    return !mismatch.within && mismatch.ratio > 1.F;
}

void RequireSeparableRows(NeuralWorkload workload, const std::vector<std::vector<float>>& rows) {
    for (std::size_t left = 0; left < rows.size(); ++left) {
        for (std::size_t right = left + 1; right < rows.size(); ++right) {
            ASSERT_TRUE(SwapExceedsTolerance(workload, rows[left], rows[right]))
                << WorkloadName(workload) << " rows " << left << " and " << right
                << " are closer than the tolerance";
        }
    }
}

void ExpectMatchesReference(NeuralWorkload workload, const SchedulerCompletion& completion,
                            std::uint64_t request_id, std::size_t batch,
                            const NeuralContract& contract, const std::vector<float>& expected,
                            RatioRecord& worst) {
    EXPECT_EQ(completion.status, CompletionStatus::kSucceeded);
    EXPECT_EQ(completion.request_id, request_id);
    EXPECT_EQ(completion.sequence, request_id);
    EXPECT_EQ(completion.inference_error.kind, InferenceErrorKind::kNone);
    ASSERT_EQ(completion.outputs.size(), 1U);
    EXPECT_EQ(completion.outputs[0].name, contract.output_name);
    EXPECT_EQ(completion.outputs[0].shape,
              (std::vector<std::int64_t>{1, static_cast<std::int64_t>(contract.output_width)}));
    ASSERT_EQ(completion.outputs[0].values.size(), expected.size());
    for (const float value : completion.outputs[0].values) {
        if (!std::isfinite(value)) {
            worst.saw_nonfinite = true;
        }
        EXPECT_TRUE(std::isfinite(value));
    }
    const NeuralMismatch mismatch =
        neural_worst_mismatch(workload, completion.outputs[0].values, expected);
    EXPECT_TRUE(mismatch.within) << WorkloadName(workload) << " request " << request_id << " ratio "
                                 << mismatch.ratio;
    if (mismatch.ratio > worst.ratio) {
        worst.ratio = mismatch.ratio;
        worst.batch = batch;
        worst.row = request_id;
        worst.column = mismatch.index;
    }
}

SchedulerConfig BatchConfig(const char* model, std::uint32_t max_batch, std::size_t capacity) {
    SchedulerConfig config{model, 1, capacity};
    config.enable_batching = true;
    config.max_batch_size = max_batch;
    config.max_batch_delay = std::chrono::milliseconds{0};
    config.enable_metrics = true;
    return config;
}

std::vector<std::size_t> RunGroup(const ModelCase& model, std::uint32_t max_batch,
                                  const std::vector<std::vector<NamedTensor>>& requests,
                                  std::vector<SchedulerCompletion>& completions) {
    ObservedBatches observed;
    auto blocked = std::make_shared<std::latch>(1);
    LatchRelease release{blocked};
    auto scheduler = std::make_unique<inferedge::InferenceScheduler>(
        BatchConfig(model.model, max_batch, requests.size()));
    SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
    SchedulerTestAccess::SetOnBatchRun(*scheduler,
                                       [&observed](std::size_t count) { observed.Record(count); });
    if (scheduler->start().kind != SchedulerErrorKind::kNone) {
        ADD_FAILURE() << "scheduler start failed";
        return {};
    }
    std::vector<Admission> admissions;
    admissions.reserve(requests.size());
    for (const std::vector<NamedTensor>& request : requests) {
        admissions.push_back(scheduler->submit(request));
        EXPECT_EQ(admissions.back().status, AdmissionStatus::kAccepted);
    }
    release.Release();
    completions.clear();
    completions.reserve(admissions.size());
    for (Admission& admission : admissions) {
        if (!admission.completion.has_value()) {
            ADD_FAILURE() << "accepted request has no future";
            return {};
        }
        completions.push_back(admission.completion->get());
    }
    EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind, SchedulerErrorKind::kNone);
    std::vector<std::vector<float>> owned;
    for (const SchedulerCompletion& completion : completions) {
        if (completion.status == CompletionStatus::kSucceeded && !completion.outputs.empty()) {
            owned.push_back(completion.outputs[0].values);
        }
    }
    scheduler.reset();
    std::size_t owned_index = 0;
    for (const SchedulerCompletion& completion : completions) {
        if (completion.status == CompletionStatus::kSucceeded && !completion.outputs.empty()) {
            EXPECT_EQ(completion.outputs[0].values, owned[owned_index]);
            ++owned_index;
        }
    }
    return observed.Snapshot();
}

} // namespace

TEST_F(SchedulerNeuralBatchTest, AssemblesDistinctRowsExactly) {
    for (const ModelCase& model : kModels) {
        auto loaded = load_inference_engine(model.model);
        ASSERT_EQ(loaded.error.kind, InferenceErrorKind::kNone) << loaded.error.message;
        const NeuralContract contract = neural_contract(model.workload);
        ASSERT_EQ(loaded.engine->inputs().size(), 1U);
        EXPECT_EQ(loaded.engine->inputs()[0].element_type, ElementType::kFloat32);
        for (const std::size_t batch : kBatchSizes) {
            std::vector<std::vector<NamedTensor>> requests(batch);
            std::vector<std::span<const NamedTensor>> views;
            views.reserve(batch);
            for (std::size_t row = 0; row < batch; ++row) {
                requests[row] = {InputTensor(contract, row)};
            }
            for (const std::vector<NamedTensor>& request : requests) {
                views.emplace_back(request);
            }
            const auto assembled = AssembleBatchInputs(loaded.engine->inputs(), views);
            ASSERT_EQ(assembled.error.kind, TensorBatchErrorKind::kNone) << assembled.error.message;
            ASSERT_EQ(assembled.inputs.size(), 1U);
            EXPECT_EQ(assembled.inputs[0].name, contract.input_name);
            EXPECT_EQ(assembled.inputs[0].shape,
                      (std::vector<std::int64_t>{static_cast<std::int64_t>(batch),
                                                 static_cast<std::int64_t>(contract.input_width)}));
            ASSERT_EQ(assembled.inputs[0].values.size(), batch * contract.input_width);
            for (std::size_t row = 0; row < batch; ++row) {
                for (std::size_t column = 0; column < contract.input_width; ++column) {
                    EXPECT_EQ(assembled.inputs[0].values[row * contract.input_width + column],
                              requests[row][0].values[column])
                        << WorkloadName(model.workload) << " row " << row;
                }
            }
        }
        const std::vector<TensorMetadata> rank1{
            TensorMetadata{"features", ElementType::kFloat32, {32}}};
        const std::vector<NamedTensor> one{InputTensor(contract, 0)};
        const std::span<const NamedTensor> view{one};
        EXPECT_EQ(
            AssembleBatchInputs(rank1, std::span<const std::span<const NamedTensor>>{&view, 1})
                .error.kind,
            TensorBatchErrorKind::kUnsupportedMetadata);
        EXPECT_EQ(AssembleBatchInputs(loaded.engine->inputs(), {}).error.kind,
                  TensorBatchErrorKind::kEmptyBatch);
    }
}

TEST_F(SchedulerNeuralBatchTest, SplitsDistinctRowsWithoutAliasing) {
    for (const ModelCase& model : kModels) {
        auto loaded = load_inference_engine(model.model);
        ASSERT_EQ(loaded.error.kind, InferenceErrorKind::kNone);
        const NeuralContract contract = neural_contract(model.workload);
        ASSERT_EQ(loaded.engine->outputs().size(), 1U);
        EXPECT_EQ(loaded.engine->outputs()[0].element_type, ElementType::kFloat32);
        for (const std::size_t batch : kBatchSizes) {
            std::vector<float> values(batch * contract.output_width);
            for (std::size_t row = 0; row < batch; ++row) {
                for (std::size_t column = 0; column < contract.output_width; ++column) {
                    values[row * contract.output_width + column] =
                        static_cast<float>(1000 + row * contract.output_width + column);
                }
            }
            NamedTensor batched{contract.output_name,
                                {static_cast<std::int64_t>(batch),
                                 static_cast<std::int64_t>(contract.output_width)},
                                values};
            const float first = batched.values[0];
            const auto split = SplitBatchOutputs(loaded.engine->outputs(),
                                                 std::span<const NamedTensor>{&batched, 1}, batch);
            ASSERT_EQ(split.error.kind, TensorBatchErrorKind::kNone) << split.error.message;
            ASSERT_EQ(split.requests.size(), batch);
            batched.values[0] = -12345.F;
            for (std::size_t row = 0; row < batch; ++row) {
                ASSERT_EQ(split.requests[row].size(), 1U);
                EXPECT_EQ(split.requests[row][0].name, contract.output_name);
                EXPECT_EQ(split.requests[row][0].shape,
                          (std::vector<std::int64_t>{
                              1, static_cast<std::int64_t>(contract.output_width)}));
                ASSERT_EQ(split.requests[row][0].values.size(), contract.output_width);
                EXPECT_NE(split.requests[row][0].values.data(), batched.values.data());
                for (std::size_t column = 0; column < contract.output_width; ++column) {
                    EXPECT_EQ(split.requests[row][0].values[column],
                              static_cast<float>(1000 + row * contract.output_width + column));
                }
            }
            EXPECT_EQ(split.requests[0][0].values[0], first);
        }
    }
}

TEST_F(SchedulerNeuralBatchTest, SchedulerGroupsMatchIndependentReferences) {
    for (const ModelCase& model : kModels) {
        const NeuralContract contract = neural_contract(model.workload);
        const auto weights = load_float32_file(model.weights, contract.parameter_count);
        ASSERT_EQ(weights.error.kind, NeuralReferenceErrorKind::kNone) << weights.error.message;
        RatioRecord worst;
        for (const std::size_t batch : kBatchSizes) {
            std::vector<std::vector<float>> expected;
            std::vector<std::vector<NamedTensor>> requests;
            expected.reserve(batch);
            requests.reserve(batch);
            for (std::size_t row = 0; row < batch; ++row) {
                const std::vector<float> input = InputRow(contract.input_width, row);
                expected.push_back(ReferenceRow(model.workload, weights.values, input));
                requests.push_back({InputTensor(contract, row)});
            }
            if (batch > 1) {
                RequireSeparableRows(model.workload, expected);
            }
            std::vector<SchedulerCompletion> completions;
            const std::vector<std::size_t> groups =
                RunGroup(model, static_cast<std::uint32_t>(batch), requests, completions);
            ASSERT_EQ(groups, (std::vector<std::size_t>{batch})) << WorkloadName(model.workload);
            ASSERT_EQ(completions.size(), batch);
            for (std::size_t row = 0; row < batch; ++row) {
                ExpectMatchesReference(model.workload, completions[row], row + 1, batch, contract,
                                       expected[row], worst);
            }
            if (batch > 1) {
                const float kept = completions[1].outputs[0].values[0];
                completions[0].outputs[0].values[0] = 42.F;
                EXPECT_EQ(completions[1].outputs[0].values[0], kept);
                EXPECT_NE(completions[0].outputs[0].values[0], kept);
            }
        }
        std::cout << "neural_batch " << WorkloadName(model.workload) << " max_ratio=" << worst.ratio
                  << " batch=" << worst.batch << " request=" << worst.row
                  << " output_index=" << worst.column
                  << " nonfinite=" << (worst.saw_nonfinite ? 1 : 0) << "\n";
        EXPECT_FALSE(worst.saw_nonfinite);
    }
}

TEST_F(SchedulerNeuralBatchTest, ThirtyThreeRequestsSealAsThirtyTwoThenOne) {
    for (const ModelCase& model : kModels) {
        const NeuralContract contract = neural_contract(model.workload);
        const auto weights = load_float32_file(model.weights, contract.parameter_count);
        ASSERT_EQ(weights.error.kind, NeuralReferenceErrorKind::kNone);
        constexpr std::size_t kCount = 33;
        std::vector<std::vector<float>> expected;
        std::vector<std::vector<NamedTensor>> requests;
        for (std::size_t row = 0; row < kCount; ++row) {
            const std::vector<float> input = InputRow(contract.input_width, row);
            expected.push_back(ReferenceRow(model.workload, weights.values, input));
            requests.push_back({NamedTensor{
                contract.input_name, {1, static_cast<std::int64_t>(contract.input_width)}, input}});
        }
        RequireSeparableRows(model.workload, expected);
        std::vector<SchedulerCompletion> completions;
        const std::vector<std::size_t> groups = RunGroup(model, 32, requests, completions);
        EXPECT_EQ(groups, (std::vector<std::size_t>{32, 1})) << WorkloadName(model.workload);
        for (const std::size_t size : groups) {
            EXPECT_LE(size, 32U);
        }
        ASSERT_EQ(completions.size(), kCount);
        RatioRecord worst;
        for (std::size_t row = 0; row < kCount; ++row) {
            ExpectMatchesReference(model.workload, completions[row], row + 1,
                                   groups.empty() ? 0 : groups.front(), contract, expected[row],
                                   worst);
        }
        const float kept = completions[1].outputs[0].values[0];
        completions[0].outputs[0].values[0] = 42.F;
        EXPECT_EQ(completions[1].outputs[0].values[0], kept);
        std::cout << "neural_batch_boundary " << WorkloadName(model.workload)
                  << " max_ratio=" << worst.ratio << " batch=" << worst.batch
                  << " request=" << worst.row << " output_index=" << worst.column
                  << " nonfinite=" << (worst.saw_nonfinite ? 1 : 0) << "\n";
        EXPECT_FALSE(worst.saw_nonfinite);
    }
}

TEST_F(SchedulerNeuralBatchTest, InvalidRequestDoesNotReplaceItsNeighbors) {
    for (const ModelCase& model : kModels) {
        const NeuralContract contract = neural_contract(model.workload);
        const auto weights = load_float32_file(model.weights, contract.parameter_count);
        ASSERT_EQ(weights.error.kind, NeuralReferenceErrorKind::kNone);
        const std::vector<float> left_input = InputRow(contract.input_width, 0);
        const std::vector<float> right_input = InputRow(contract.input_width, 3);
        const std::vector<float> left_expected =
            ReferenceRow(model.workload, weights.values, left_input);
        const std::vector<float> right_expected =
            ReferenceRow(model.workload, weights.values, right_input);
        ASSERT_TRUE(SwapExceedsTolerance(model.workload, left_expected, right_expected));

        const std::vector<NamedTensor> wrong_width{
            NamedTensor{contract.input_name,
                        {1, static_cast<std::int64_t>(contract.input_width - 1)},
                        std::vector<float>(contract.input_width - 1, 1.F)}};
        const std::vector<NamedTensor> wrong_name{
            NamedTensor{"image",
                        {1, static_cast<std::int64_t>(contract.input_width)},
                        std::vector<float>(contract.input_width, 1.F)}};
        const std::vector<NamedTensor> duplicate{
            InputTensor(contract, 1),
            NamedTensor{contract.input_name,
                        {1, static_cast<std::int64_t>(contract.input_width)},
                        std::vector<float>(contract.input_width, 3.F)}};
        const std::vector<std::vector<NamedTensor>> invalids{wrong_width, wrong_name, duplicate};
        const InferenceErrorKind kinds[] = {InferenceErrorKind::kShapeMismatch,
                                            InferenceErrorKind::kUnexpectedInput,
                                            InferenceErrorKind::kDuplicateInput};
        for (std::size_t kind = 0; kind < invalids.size(); ++kind) {
            ObservedBatches observed;
            auto blocked = std::make_shared<std::latch>(1);
            LatchRelease release{blocked};
            auto scheduler =
                std::make_unique<inferedge::InferenceScheduler>(BatchConfig(model.model, 4, 4));
            SchedulerTestAccess::SetBeforePop(*scheduler, blocked);
            SchedulerTestAccess::SetOnBatchRun(
                *scheduler, [&observed](std::size_t count) { observed.Record(count); });
            ASSERT_EQ(scheduler->start().kind, SchedulerErrorKind::kNone);
            Admission first = scheduler->submit({InputTensor(contract, 0)});
            Admission bad = scheduler->submit(invalids[kind]);
            Admission third = scheduler->submit({InputTensor(contract, 3)});
            EXPECT_EQ(first.status, AdmissionStatus::kAccepted);
            EXPECT_EQ(bad.status, AdmissionStatus::kAccepted);
            EXPECT_EQ(third.status, AdmissionStatus::kAccepted);
            release.Release();
            const SchedulerCompletion left = first.completion->get();
            const SchedulerCompletion failed = bad.completion->get();
            const SchedulerCompletion right = third.completion->get();
            EXPECT_EQ(scheduler->shutdown(ShutdownPolicy::kDrain).error.kind,
                      SchedulerErrorKind::kNone);
            EXPECT_EQ(observed.Snapshot(), (std::vector<std::size_t>{2}))
                << WorkloadName(model.workload);
            RatioRecord worst;
            ExpectMatchesReference(model.workload, left, 1, 2, contract, left_expected, worst);
            ExpectMatchesReference(model.workload, right, 3, 2, contract, right_expected, worst);
            EXPECT_EQ(failed.status, CompletionStatus::kInferenceFailed);
            EXPECT_EQ(failed.request_id, 2U);
            EXPECT_EQ(failed.inference_error.kind, kinds[kind]);
            EXPECT_TRUE(failed.outputs.empty());
            EXPECT_FALSE(worst.saw_nonfinite);
        }
    }
}
