#include <array>
#include <cpu_provider_factory.h>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#ifndef INFEREDGE_ADD_MODEL_PATH
#error "CMake must define INFEREDGE_ADD_MODEL_PATH"
#endif

#ifndef INFEREDGE_INVALID_MODEL_PATH
#error "CMake must define INFEREDGE_INVALID_MODEL_PATH"
#endif

namespace {

// Environment is declared first, then options, then the session. Members are
// initialized in that order and destroyed in reverse, so the environment
// outlives the session and the options stay alive through session creation.
struct CpuSession {
    Ort::Env environment;
    Ort::SessionOptions options;
    Ort::Session session;

    explicit CpuSession(const char* model_path)
        : environment{ORT_LOGGING_LEVEL_WARNING, "inferedge-add"}, options{},
          session{Open(environment, options, model_path)} {}

    static Ort::Session Open(const Ort::Env& environment, Ort::SessionOptions& options,
                             const char* model_path) {
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
        // The CPU provider is the default. use_arena == 1 keeps its default arena.
        options.AppendExecutionProvider_CPU(1);
        return Ort::Session{environment, model_path, options};
    }
};

template <typename TensorInfo> void ExpectFloatShape2(const TensorInfo& tensor) {
    EXPECT_EQ(tensor.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    const std::vector<int64_t> shape = tensor.GetShape();
    ASSERT_EQ(shape.size(), 1U);
    EXPECT_EQ(shape.front(), static_cast<int64_t>(2));
}

void ExpectModelFloatShape2(const Ort::TypeInfo& type_info) {
    ExpectFloatShape2(type_info.GetTensorTypeAndShapeInfo());
}

// CreateTensor borrows the caller-owned vectors. It does not copy or free them.
// The vectors must already have their final size so a later push_back cannot
// reallocate the storage while an Ort::Value still points at it.
std::vector<Ort::Value> BorrowFloatInputs(const Ort::MemoryInfo& memory, std::vector<float>& left,
                                          std::vector<float>& right, int64_t left_length) {
    const std::array<int64_t, 1> left_shape{left_length};
    const std::array<int64_t, 1> right_shape{static_cast<int64_t>(right.size())};
    std::vector<Ort::Value> inputs;
    inputs.reserve(2);
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, left.data(), left.size(),
                                                     left_shape.data(), left_shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, right.data(), right.size(),
                                                     right_shape.data(), right_shape.size()));
    return inputs;
}

void ExpectAddOutput(std::vector<float>& left, std::vector<float>& right, float first,
                     float second) {
    CpuSession loaded{INFEREDGE_ADD_MODEL_PATH};
    ASSERT_EQ(left.size(), 2U);
    ASSERT_EQ(right.size(), 2U);

    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs =
        BorrowFloatInputs(memory, left, right, static_cast<int64_t>(left.size()));

    const char* input_names[] = {"A", "B"};
    const char* output_names[] = {"C"};
    Ort::RunOptions run_options;
    // The returned values own the output buffer allocated by Run.
    const std::vector<Ort::Value> outputs =
        loaded.session.Run(run_options, input_names, inputs.data(), inputs.size(), output_names, 1);

    ASSERT_EQ(outputs.size(), 1U);
    ASSERT_TRUE(outputs.front().IsTensor());
    const Ort::TensorTypeAndShapeInfo output_info = outputs.front().GetTensorTypeAndShapeInfo();
    ExpectFloatShape2(output_info);
    ASSERT_EQ(output_info.GetElementCount(), 2U);
    const float* values = outputs.front().GetTensorData<float>();
    ASSERT_NE(values, nullptr);
    EXPECT_EQ(values[0], first);
    EXPECT_EQ(values[1], second);
}

} // namespace

class OnnxAddInferenceTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

TEST_F(OnnxAddInferenceTest, DescribesTheAddGraph) {
    const CpuSession loaded{INFEREDGE_ADD_MODEL_PATH};
    const Ort::Session& session = loaded.session;

    ASSERT_EQ(session.GetInputCount(), 2U);
    ASSERT_EQ(session.GetOutputCount(), 1U);

    Ort::AllocatorWithDefaultOptions allocator;
    const Ort::AllocatedStringPtr input_a = session.GetInputNameAllocated(0, allocator);
    const Ort::AllocatedStringPtr input_b = session.GetInputNameAllocated(1, allocator);
    const Ort::AllocatedStringPtr output = session.GetOutputNameAllocated(0, allocator);
    EXPECT_STREQ(input_a.get(), "A");
    EXPECT_STREQ(input_b.get(), "B");
    EXPECT_STREQ(output.get(), "C");

    const Ort::TypeInfo first_input = session.GetInputTypeInfo(0);
    const Ort::TypeInfo second_input = session.GetInputTypeInfo(1);
    const Ort::TypeInfo model_output = session.GetOutputTypeInfo(0);
    ExpectModelFloatShape2(first_input);
    ExpectModelFloatShape2(second_input);
    ExpectModelFloatShape2(model_output);
}

TEST_F(OnnxAddInferenceTest, ExecutesPositiveInputs) {
    std::vector<float> left{1.0f, 2.0f};
    std::vector<float> right{3.0f, 4.0f};
    ExpectAddOutput(left, right, 4.0f, 6.0f);
}

TEST_F(OnnxAddInferenceTest, ExecutesNegativeAndZeroInputs) {
    std::vector<float> left{-1.5f, 0.0f};
    std::vector<float> right{2.5f, -4.0f};
    ExpectAddOutput(left, right, 1.0f, -4.0f);
}

TEST_F(OnnxAddInferenceTest, MissingModelThrows) {
    const std::string missing = std::string{INFEREDGE_ADD_MODEL_PATH} + ".missing";
    EXPECT_THROW((CpuSession{missing.c_str()}), Ort::Exception);
}

TEST_F(OnnxAddInferenceTest, InvalidModelThrows) {
    EXPECT_THROW((CpuSession{INFEREDGE_INVALID_MODEL_PATH}), Ort::Exception);
}

TEST_F(OnnxAddInferenceTest, RejectsWrongInputShape) {
    CpuSession loaded{INFEREDGE_ADD_MODEL_PATH};
    std::vector<float> left{1.0f, 2.0f, 3.0f};
    std::vector<float> right{3.0f, 4.0f};
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs = BorrowFloatInputs(memory, left, right, 3);

    const char* input_names[] = {"A", "B"};
    const char* output_names[] = {"C"};
    Ort::RunOptions run_options;
    EXPECT_THROW((loaded.session.Run(run_options, input_names, inputs.data(), inputs.size(),
                                     output_names, 1)),
                 Ort::Exception);
}

TEST_F(OnnxAddInferenceTest, RejectsWrongInputType) {
    CpuSession loaded{INFEREDGE_ADD_MODEL_PATH};
    std::vector<std::int64_t> left{1, 2};
    std::vector<float> right{3.0f, 4.0f};
    const std::array<int64_t, 1> shape{2};
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    inputs.reserve(2);
    inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(memory, left.data(), left.size(),
                                                            shape.data(), shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, right.data(), right.size(),
                                                     shape.data(), shape.size()));

    const char* input_names[] = {"A", "B"};
    const char* output_names[] = {"C"};
    Ort::RunOptions run_options;
    EXPECT_THROW((loaded.session.Run(run_options, input_names, inputs.data(), inputs.size(),
                                     output_names, 1)),
                 Ort::Exception);
}
