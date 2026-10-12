#include "inferedge/inference_engine.hpp"

#include <atomic>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

#ifndef INFEREDGE_MNIST_MODEL_PATH
#error "CMake must define INFEREDGE_MNIST_MODEL_PATH"
#endif

#ifndef INFEREDGE_MNIST_FIXTURE_DIR
#error "CMake must define INFEREDGE_MNIST_FIXTURE_DIR"
#endif

namespace {

using inferedge::ElementType;
using inferedge::EngineLoad;
using inferedge::InferenceEngine;
using inferedge::InferenceErrorKind;
using inferedge::InferenceResult;
using inferedge::load_inference_engine;
using inferedge::NamedTensor;
using inferedge::NamedTensorView;

constexpr std::size_t kInputCount = 784;
constexpr std::size_t kLogitCount = 10;
constexpr double kRelativeTolerance = 1e-4;
constexpr double kAbsoluteTolerance = 1e-5;

// Fixture ids and expected argmax classes. models/fixtures/mnist/manifest.json
// records the checksums. These tests read the ONNX reference files and do not
// copy those logits into source.
struct FixtureCase {
    const char* name;
    int expected_class;
};

constexpr FixtureCase kCases[] = {
    {"test_00003", 0}, {"test_00002", 1}, {"test_00001", 2}, {"test_00018", 3}, {"test_00004", 4},
    {"test_00008", 5}, {"test_00011", 6}, {"test_00000", 7}, {"test_00061", 8}, {"test_00007", 9},
    {"test_00151", 8}, {"test_00217", 5}, {"test_00241", 8}, {"synthetic", 5},
};

struct FloatFile {
    std::string error;
    std::vector<float> values;
};

class MnistInferenceTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_EQ(::setenv("ORT_DISABLE_TELEMETRY", "1", 1), 0); }
};

std::filesystem::path FixturePath(const char* folder, const char* name) {
    return std::filesystem::path{INFEREDGE_MNIST_FIXTURE_DIR} / folder /
           (std::string{name} + ".f32");
}

FloatFile ReadLittleEndianFloat32(const std::filesystem::path& path, std::size_t count) {
    std::ifstream input{path, std::ios::binary};
    if (!input) {
        return FloatFile{"cannot open " + path.string(), {}};
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    const std::streamoff expected = static_cast<std::streamoff>(count * sizeof(std::uint32_t));
    if (size != expected) {
        return FloatFile{path.string() + " has " + std::to_string(size) + " bytes, expected " +
                             std::to_string(expected),
                         {}};
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> values(count);
    for (std::size_t index = 0; index < count; ++index) {
        unsigned char bytes[4] = {};
        input.read(reinterpret_cast<char*>(bytes), 4);
        if (!input) {
            return FloatFile{"short read of " + path.string(), {}};
        }
        const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                                   (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                                   (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                                   (static_cast<std::uint32_t>(bytes[3]) << 24U);
        const float value = std::bit_cast<float>(bits);
        if (!std::isfinite(value)) {
            return FloatFile{"non-finite value in " + path.string(), {}};
        }
        values[index] = value;
    }
    return FloatFile{{}, std::move(values)};
}

EngineLoad LoadMnist() {
    return load_inference_engine(INFEREDGE_MNIST_MODEL_PATH);
}

NamedTensorView ImageView(const std::vector<float>& values) {
    return NamedTensorView{
        "image", {static_cast<std::int64_t>(values.size())}, std::span<const float>{values}};
}

int ArgMax(const std::vector<float>& values) {
    std::size_t best = 0;
    for (std::size_t index = 1; index < values.size(); ++index) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return static_cast<int>(best);
}

void ExpectLogits(const std::string& name, const std::vector<float>& actual,
                  const std::vector<float>& expected, int expected_class, double& maximum,
                  double& total) {
    ASSERT_EQ(actual.size(), kLogitCount);
    ASSERT_EQ(expected.size(), kLogitCount);
    for (float value : actual) {
        EXPECT_TRUE(std::isfinite(value)) << name;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double difference =
            std::fabs(static_cast<double>(actual[index]) - static_cast<double>(expected[index]));
        const double limit = kAbsoluteTolerance +
                             kRelativeTolerance * std::fabs(static_cast<double>(expected[index]));
        maximum = std::max(maximum, difference);
        total += difference;
        EXPECT_LE(difference, limit)
            << name << " logit " << index << " actual " << actual[index] << " reference "
            << expected[index] << " absolute " << difference;
    }
    EXPECT_EQ(ArgMax(actual), expected_class) << name;
}

} // namespace

TEST_F(MnistInferenceTest, LoadsTheModelAndDescribesItsTensors) {
    const EngineLoad loaded = LoadMnist();
    EXPECT_EQ(loaded.error.kind, InferenceErrorKind::kNone);
    EXPECT_TRUE(loaded.error.message.empty());
    ASSERT_NE(loaded.engine, nullptr);

    const std::vector<inferedge::TensorMetadata>& inputs = loaded.engine->inputs();
    ASSERT_EQ(inputs.size(), 1U);
    EXPECT_EQ(inputs.front().name, "image");
    EXPECT_EQ(inputs.front().element_type, ElementType::kFloat32);
    ASSERT_EQ(inputs.front().dimensions.size(), 1U);
    EXPECT_EQ(inputs.front().dimensions.front(), static_cast<std::int64_t>(kInputCount));

    const std::vector<inferedge::TensorMetadata>& outputs = loaded.engine->outputs();
    ASSERT_EQ(outputs.size(), 1U);
    EXPECT_EQ(outputs.front().name, "logits");
    EXPECT_EQ(outputs.front().element_type, ElementType::kFloat32);
    ASSERT_EQ(outputs.front().dimensions.size(), 1U);
    EXPECT_EQ(outputs.front().dimensions.front(), static_cast<std::int64_t>(kLogitCount));
}

TEST_F(MnistInferenceTest, MatchesTheOnnxReferenceForEveryFixture) {
    EngineLoad loaded = LoadMnist();
    ASSERT_EQ(loaded.error.kind, InferenceErrorKind::kNone);
    ASSERT_NE(loaded.engine, nullptr);

    double maximum = 0.0;
    double total = 0.0;
    std::size_t agreements = 0;
    for (const FixtureCase& fixture : kCases) {
        const FloatFile image =
            ReadLittleEndianFloat32(FixturePath("inputs", fixture.name), kInputCount);
        const FloatFile reference =
            ReadLittleEndianFloat32(FixturePath("onnx", fixture.name), kLogitCount);
        ASSERT_TRUE(image.error.empty()) << image.error;
        ASSERT_TRUE(reference.error.empty()) << reference.error;
        ASSERT_EQ(image.values.size(), kInputCount);
        ASSERT_EQ(reference.values.size(), kLogitCount);

        const NamedTensorView input = ImageView(image.values);
        const InferenceResult result =
            loaded.engine->run(std::span<const NamedTensorView>{&input, 1});
        ASSERT_EQ(result.error.kind, InferenceErrorKind::kNone)
            << fixture.name << " " << result.error.message;
        ASSERT_EQ(result.outputs.size(), 1U);
        EXPECT_EQ(result.outputs.front().name, "logits");
        ASSERT_EQ(result.outputs.front().shape.size(), 1U);
        EXPECT_EQ(result.outputs.front().shape.front(), static_cast<std::int64_t>(kLogitCount));
        const std::vector<float>& actual = result.outputs.front().values;
        ExpectLogits(fixture.name, actual, reference.values, fixture.expected_class, maximum,
                     total);
        if (ArgMax(result.outputs.front().values) == fixture.expected_class) {
            ++agreements;
        }
    }

    constexpr std::size_t case_count = sizeof(kCases) / sizeof(kCases[0]);
    EXPECT_EQ(agreements, case_count);
    const double mean = total / static_cast<double>(case_count * kLogitCount);
    std::printf("mnist_cpp cases %zu agreements %zu maximum_absolute_error %.17g "
                "mean_absolute_error %.17g\n",
                case_count, agreements, maximum, mean);
}

TEST_F(MnistInferenceTest, RepeatsARequestOnOneEngine) {
    EngineLoad loaded = LoadMnist();
    ASSERT_NE(loaded.engine, nullptr);
    const FloatFile image =
        ReadLittleEndianFloat32(FixturePath("inputs", "synthetic"), kInputCount);
    ASSERT_TRUE(image.error.empty()) << image.error;
    const NamedTensorView input = ImageView(image.values);
    const InferenceResult first = loaded.engine->run(std::span<const NamedTensorView>{&input, 1});
    const InferenceResult second = loaded.engine->run(std::span<const NamedTensorView>{&input, 1});
    ASSERT_EQ(first.error.kind, InferenceErrorKind::kNone);
    ASSERT_EQ(second.error.kind, InferenceErrorKind::kNone);
    ASSERT_EQ(first.outputs.size(), 1U);
    ASSERT_EQ(second.outputs.size(), 1U);
    EXPECT_EQ(first.outputs.front().values, second.outputs.front().values);
}

TEST_F(MnistInferenceTest, RejectsTheWrongNumberOfFloats) {
    EngineLoad loaded = LoadMnist();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> too_few(kInputCount - 1, 0.0f);
    const std::vector<float> too_many(kInputCount + 1, 0.0f);
    const NamedTensorView short_input =
        NamedTensorView{"image", {static_cast<std::int64_t>(kInputCount)}, too_few};
    const NamedTensorView long_input =
        NamedTensorView{"image", {static_cast<std::int64_t>(kInputCount)}, too_many};
    const InferenceResult few =
        loaded.engine->run(std::span<const NamedTensorView>{&short_input, 1});
    const InferenceResult many =
        loaded.engine->run(std::span<const NamedTensorView>{&long_input, 1});
    EXPECT_EQ(few.error.kind, InferenceErrorKind::kElementCountMismatch);
    EXPECT_EQ(many.error.kind, InferenceErrorKind::kElementCountMismatch);
    EXPECT_TRUE(few.outputs.empty());
    EXPECT_TRUE(many.outputs.empty());
}

TEST_F(MnistInferenceTest, RejectsTheWrongInputName) {
    EngineLoad loaded = LoadMnist();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> values(kInputCount, 0.0f);
    const NamedTensorView input =
        NamedTensorView{"pixels", {static_cast<std::int64_t>(kInputCount)}, values};
    const InferenceResult result = loaded.engine->run(std::span<const NamedTensorView>{&input, 1});
    EXPECT_EQ(result.error.kind, InferenceErrorKind::kUnexpectedInput);
    EXPECT_TRUE(result.outputs.empty());
}

TEST_F(MnistInferenceTest, OutputRemainsValidAfterTheEngineIsDestroyed) {
    EngineLoad loaded = LoadMnist();
    ASSERT_NE(loaded.engine, nullptr);
    const FloatFile image =
        ReadLittleEndianFloat32(FixturePath("inputs", "test_00000"), kInputCount);
    const FloatFile reference =
        ReadLittleEndianFloat32(FixturePath("onnx", "test_00000"), kLogitCount);
    ASSERT_TRUE(image.error.empty()) << image.error;
    ASSERT_TRUE(reference.error.empty()) << reference.error;
    const NamedTensorView input = ImageView(image.values);
    InferenceResult result = loaded.engine->run(std::span<const NamedTensorView>{&input, 1});
    ASSERT_EQ(result.error.kind, InferenceErrorKind::kNone);
    NamedTensor output = std::move(result.outputs.front());
    loaded.engine.reset();
    double maximum = 0.0;
    double total = 0.0;
    ExpectLogits("test_00000", output.values, reference.values, 7, maximum, total);
}

TEST_F(MnistInferenceTest, StaysUsableAfterARejectedInput) {
    EngineLoad loaded = LoadMnist();
    ASSERT_NE(loaded.engine, nullptr);
    const std::vector<float> too_few(kInputCount - 1, 0.0f);
    const NamedTensorView rejected_input =
        NamedTensorView{"image", {static_cast<std::int64_t>(kInputCount)}, too_few};
    const InferenceResult rejected =
        loaded.engine->run(std::span<const NamedTensorView>{&rejected_input, 1});
    EXPECT_EQ(rejected.error.kind, InferenceErrorKind::kElementCountMismatch);

    const FloatFile image =
        ReadLittleEndianFloat32(FixturePath("inputs", "synthetic"), kInputCount);
    const FloatFile reference =
        ReadLittleEndianFloat32(FixturePath("onnx", "synthetic"), kLogitCount);
    ASSERT_TRUE(image.error.empty()) << image.error;
    ASSERT_TRUE(reference.error.empty()) << reference.error;
    const NamedTensorView input = ImageView(image.values);
    const InferenceResult result = loaded.engine->run(std::span<const NamedTensorView>{&input, 1});
    ASSERT_EQ(result.error.kind, InferenceErrorKind::kNone);
    double maximum = 0.0;
    double total = 0.0;
    ExpectLogits("synthetic", result.outputs.front().values, reference.values, 5, maximum, total);
}

namespace {

constexpr std::string_view kTruncatedFixturePrefix = "inferedge-mnist-truncated-";
constexpr int kTruncatedFixtureAttempts = 32;

std::atomic<std::uint64_t>& TruncatedFixtureIds() {
    static std::atomic<std::uint64_t> next{0};
    return next;
}

// native() and string_view do not allocate. filename().string() can throw, so
// the destructor must not call it.
bool IsOwnedTruncatedFixture(const std::filesystem::path& path) noexcept {
    const std::filesystem::path::string_type& native = path.native();
    if (native.size() < 2 || native.front() != '/') {
        return false;
    }
    const auto separator = native.find_last_of('/');
    if (separator == std::filesystem::path::string_type::npos || separator == 0 ||
        separator + 1 >= native.size()) {
        return false;
    }
    const std::string_view name(native.data() + separator + 1, native.size() - separator - 1);
    return name.starts_with(kTruncatedFixturePrefix) && name.ends_with(".f32");
}

void RemoveOwnedTruncatedFixture(const std::filesystem::path& path) noexcept {
    if (!IsOwnedTruncatedFixture(path)) {
        return;
    }
    std::error_code error;
    std::filesystem::remove(path, error);
}

class TruncatedFixtureFile {
public:
    TruncatedFixtureFile() {
        const std::filesystem::path temp = std::filesystem::temp_directory_path();
        const std::string pid = std::to_string(static_cast<long>(::getpid()));
        for (int attempt = 0; attempt < kTruncatedFixtureAttempts; ++attempt) {
            const std::uint64_t id = TruncatedFixtureIds().fetch_add(1, std::memory_order_relaxed);
            const std::filesystem::path candidate =
                temp /
                (std::string{kTruncatedFixturePrefix} + pid + "-" + std::to_string(id) + ".f32");
            const int fd = ::open(candidate.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
            if (fd < 0) {
                continue;
            }
            const char bytes[8] = {};
            const ssize_t written = ::write(fd, bytes, sizeof(bytes));
            const int write_error = errno;
            ::close(fd);
            if (written != static_cast<ssize_t>(sizeof(bytes))) {
                RemoveOwnedTruncatedFixture(candidate);
                throw std::system_error(write_error, std::generic_category(),
                                        "truncated fixture write failed");
            }
            path_ = candidate;
            owned_ = true;
            return;
        }
        throw std::runtime_error("could not create a unique truncated fixture file");
    }

    ~TruncatedFixtureFile() noexcept {
        if (!owned_) {
            return;
        }
        owned_ = false;
        RemoveOwnedTruncatedFixture(path_);
    }

    TruncatedFixtureFile(const TruncatedFixtureFile&) = delete;
    TruncatedFixtureFile& operator=(const TruncatedFixtureFile&) = delete;
    TruncatedFixtureFile(TruncatedFixtureFile&&) = delete;
    TruncatedFixtureFile& operator=(TruncatedFixtureFile&&) = delete;

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
    bool owned_{false};
};

} // namespace

TEST_F(MnistInferenceTest, RejectsATruncatedFixtureFile) {
    const TruncatedFixtureFile truncated;
    const FloatFile file = ReadLittleEndianFloat32(truncated.path(), kInputCount);
    EXPECT_FALSE(file.error.empty());
    EXPECT_TRUE(file.values.empty());
    std::error_code error;
    EXPECT_TRUE(std::filesystem::is_regular_file(truncated.path(), error));
}
