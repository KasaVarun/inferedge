#include "inferedge/generate_cli.hpp"
#include "inferedge/generative_session.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <unistd.h>

namespace {

std::filesystem::path unique_dir(const std::string& tag) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto path = std::filesystem::weakly_canonical(std::filesystem::temp_directory_path()) /
                ("inferedge-9b1-session-" + tag + "-" + std::to_string(::getpid()) + "-" +
                 std::to_string(now));
    std::filesystem::create_directories(path);
    return std::filesystem::weakly_canonical(path);
}

void write_bytes(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string hex_of(std::string_view bytes) {
    return inferedge::sha256_hex(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
}

std::filesystem::path write_invalid_gguf_package() {
    auto root = unique_dir("gguf");
    const std::string gguf = "not-a-real-gguf-weight-file";
    const std::string tmpl = "User:{user}";
    std::string manifest;
    manifest += "schema_version=1\n";
    manifest += "model_id=qwen3-test\n";
    manifest += "source_repository=Qwen/Qwen3-test\n";
    manifest += "source_revision=deadbeef\n";
    manifest += "runtime_project=llama.cpp\n";
    manifest += "runtime_tag=v0.7.0\n";
    manifest += "runtime_commit=620d2225cc436ab0b001e3200e590e3b286ef9ab\n";
    manifest += "gguf_file=model.gguf\n";
    manifest += "gguf_sha256=" + hex_of(gguf) + "\n";
    manifest += "gguf_size_bytes=" + std::to_string(gguf.size()) + "\n";
    manifest += "gguf_architecture=qwen3\n";
    manifest += "quantization=Q4_K_M\n";
    manifest += "license_id=apache-2.0\n";
    manifest += "context_limit=4096\n";
    manifest += "chat_template_file=qwen3-non-thinking-v1.txt\n";
    manifest += "chat_template_sha256=" + hex_of(tmpl) + "\n";
    manifest += "thinking_mode=disabled\n";
    manifest += "license_file=LICENSE.txt\n";
    manifest += "license_sha256=" + hex_of("Apache-2.0") + "\n";
    manifest += "notice_file=none\n";
    manifest += "notice_sha256=none\n";
    write_bytes(root / "manifest.txt", manifest);
    write_bytes(root / "model.gguf", gguf);
    write_bytes(root / "qwen3-non-thinking-v1.txt", tmpl);
    write_bytes(root / "LICENSE.txt", "Apache-2.0");
    return root;
}

} // namespace

TEST(GenerativeSessionTest, CancellationAdapterIsIdempotent) {
    inferedge::CancellationFlag flag;
    EXPECT_FALSE(flag.is_canceled());
    EXPECT_FALSE(inferedge::cancellation_adapter_should_abort(flag));
    flag.request_cancel();
    flag.request_cancel();
    EXPECT_TRUE(flag.is_canceled());
    EXPECT_TRUE(inferedge::cancellation_adapter_should_abort(flag));
}

TEST(GenerativeSessionTest, FailedLoadDoesNotLeaveLoadedSession) {
    inferedge::GenerativeSession session;
    const auto missing = unique_dir("missing");
    const inferedge::GenerationStatus first = session.load(missing);
    EXPECT_FALSE(first.ok);
    EXPECT_FALSE(session.loaded());
    const inferedge::GenerationStatus second = session.load(missing);
    EXPECT_FALSE(second.ok);
    EXPECT_FALSE(session.loaded());
    inferedge::GenerativeSession moved = std::move(session);
    EXPECT_FALSE(moved.loaded());
    std::filesystem::remove_all(missing);
}

TEST(GenerativeSessionTest, InvalidGgufIsAModelLoadFailure) {
    const auto root = write_invalid_gguf_package();
    inferedge::GenerativeSession session;
    const inferedge::GenerationStatus status = session.load(root);
    EXPECT_FALSE(status.ok);
    EXPECT_EQ(status.error, inferedge::GenerationError::kModelLoad);
    EXPECT_FALSE(session.loaded());
    const inferedge::GenerationStatus again = session.load(root);
    EXPECT_FALSE(again.ok);
    EXPECT_FALSE(session.loaded());
    std::filesystem::remove_all(root);
}

TEST(GenerativeSessionTest, RepeatedFailedLoadsThenDestroyAreSafe) {
    inferedge::GenerativeSession session;
    for (int index = 0; index < 3; ++index) {
        const auto root = write_invalid_gguf_package();
        EXPECT_FALSE(session.load(root).ok);
        EXPECT_FALSE(session.loaded());
        std::filesystem::remove_all(root);
    }
}

TEST(GenerativeSessionTest, GenerateWithoutLoadFails) {
    inferedge::GenerativeSession session;
    inferedge::GenerationRequest request;
    request.prompt = "hello";
    request.max_new_tokens = 4;
    request.threads = 1;
    const inferedge::GenerationResult result = session.generate(request);
    EXPECT_FALSE(result.status.ok);
}

TEST(GenerativeSessionTest, DoesNotPersistPromptFiles) {
    const auto cwd_before = std::filesystem::current_path();
    const auto root = write_invalid_gguf_package();
    inferedge::GenerativeSession session;
    (void)session.load(root);
    inferedge::GenerationRequest request;
    request.prompt = "unique-prompt-9b1-not-for-storage";
    request.max_new_tokens = 4;
    request.threads = 1;
    (void)session.generate(request);
    EXPECT_FALSE(std::filesystem::exists(cwd_before / "unique-prompt-9b1-not-for-storage"));
    std::filesystem::remove_all(root);
}

#ifdef INFEREDGE_GENERATE_EXECUTABLE
TEST(GenerativeSessionTest, GenerateExecutableHelpAndVersion) {
    ASSERT_TRUE(std::filesystem::exists(INFEREDGE_GENERATE_EXECUTABLE));
}
#endif

TEST(GenerativeSessionTest, ContextBudgetRejectsOverflowAndOverfill) {
    EXPECT_TRUE(inferedge::evaluate_generation_context_budget(8, 8, 16).ok);
    EXPECT_FALSE(inferedge::evaluate_generation_context_budget(8, 9, 16).ok);
    EXPECT_FALSE(inferedge::evaluate_generation_context_budget(17, 1, 16).ok);
    EXPECT_FALSE(inferedge::evaluate_generation_context_budget(16, 1, 16).ok);
    EXPECT_FALSE(inferedge::evaluate_generation_context_budget(1, 0, 16).ok);
    EXPECT_FALSE(inferedge::evaluate_generation_context_budget(0, 8, 16).ok);
    EXPECT_FALSE(inferedge::evaluate_generation_context_budget(1, 0xffffffffU, 16).ok);
    EXPECT_FALSE(inferedge::evaluate_generation_context_budget(1, 8, 0).ok);
}

TEST(GenerativeSessionTest, GenerateWithoutLoadFailsForEmptyAndNonEmptyPrompts) {
    inferedge::GenerativeSession session;
    inferedge::GenerationRequest request;
    request.prompt = "";
    request.max_new_tokens = 4;
    request.threads = 1;
    EXPECT_FALSE(session.generate(request).status.ok);
    request.prompt = "   ";
    EXPECT_FALSE(session.generate(request).status.ok);
}

TEST(GenerativeSessionTest, BackendLeaseCountIsZeroAfterFailedLoads) {
    EXPECT_EQ(inferedge::GenerativeSessionTestAccess::BackendLeaseCount(), 0);
    inferedge::GenerativeSession session;
    const auto root = write_invalid_gguf_package();
    EXPECT_FALSE(session.load(root).ok);
    EXPECT_EQ(inferedge::GenerativeSessionTestAccess::BackendLeaseCount(), 0);
    std::filesystem::remove_all(root);
}

TEST(GenerativeSessionTest, TokenizeProbeRejectsInt32Min) {
    std::uint32_t count = 99;
    EXPECT_FALSE(
        inferedge::token_count_from_llama_probe(std::numeric_limits<std::int32_t>::min(), count)
            .ok);
    ASSERT_TRUE(inferedge::token_count_from_llama_probe(-8, count).ok);
    EXPECT_EQ(count, 8U);
    ASSERT_TRUE(inferedge::token_count_from_llama_probe(4, count).ok);
    EXPECT_EQ(count, 4U);
    EXPECT_FALSE(inferedge::token_count_from_llama_probe(0, count).ok);
}

TEST(GenerativeSessionTest, GeneratedTextBoundRejectsOverflow) {
    std::string text(inferedge::kGenerativeMaxGeneratedBytes, 'a');
    EXPECT_FALSE(inferedge::append_generated_text(text, "x").ok);
    std::string small;
    ASSERT_TRUE(inferedge::append_generated_text(small, "ok").ok);
    EXPECT_EQ(small, "ok");
}

TEST(GenerativeSessionTest, LlamaContextSizingMatchesContextLimitAndCappedMicrobatch) {
    const std::uint32_t limits[] = {1, 2, 511, 512, 513, 4095, 4096};
    for (const std::uint32_t limit : limits) {
        const inferedge::GenerativeContextSizing sizing =
            inferedge::GenerativeSessionTestAccess::ContextSizing(limit);
        EXPECT_EQ(sizing.n_ctx, limit);
        EXPECT_EQ(sizing.n_batch, limit);
        const std::uint32_t expected_ubatch = limit < 512U ? limit : 512U;
        EXPECT_EQ(sizing.n_ubatch, expected_ubatch);
        EXPECT_GE(sizing.n_ubatch, 1U);
        EXPECT_LE(sizing.n_ubatch, sizing.n_batch);
        EXPECT_EQ(sizing.n_seq_max, 1U);
        EXPECT_GE(sizing.n_batch, limit);
    }
}

TEST(GenerativeSessionTest, LlamaContextSizingCanonical4096Is4096By4096By512By1) {
    const inferedge::GenerativeContextSizing sizing =
        inferedge::GenerativeSessionTestAccess::ContextSizing(4096);
    EXPECT_EQ(sizing.n_ctx, 4096U);
    EXPECT_EQ(sizing.n_batch, 4096U);
    EXPECT_EQ(sizing.n_ubatch, 512U);
    EXPECT_EQ(sizing.n_seq_max, 1U);
}

TEST(GenerativeSessionTest, LlamaContextSizingMicrobatchShrinksBelow512AndCapsAt512) {
    const inferedge::GenerativeContextSizing small =
        inferedge::GenerativeSessionTestAccess::ContextSizing(511);
    EXPECT_EQ(small.n_batch, 511U);
    EXPECT_EQ(small.n_ubatch, 511U);
    const inferedge::GenerativeContextSizing exact =
        inferedge::GenerativeSessionTestAccess::ContextSizing(512);
    EXPECT_EQ(exact.n_batch, 512U);
    EXPECT_EQ(exact.n_ubatch, 512U);
    const inferedge::GenerativeContextSizing above =
        inferedge::GenerativeSessionTestAccess::ContextSizing(513);
    EXPECT_EQ(above.n_batch, 513U);
    EXPECT_EQ(above.n_ubatch, 512U);
}

TEST(GenerativeSessionTest, SequentialGenerateWithoutLoadRemainsRejected) {
    inferedge::GenerativeSession session;
    inferedge::GenerationRequest request;
    request.prompt = "hello";
    request.max_new_tokens = 4;
    request.threads = 1;
    EXPECT_FALSE(session.generate(request).status.ok);
    EXPECT_FALSE(session.generate(request).status.ok);
    inferedge::CancellationFlag flag;
    request.cancel = &flag;
    EXPECT_FALSE(session.generate(request).status.ok);
    flag.request_cancel();
    EXPECT_FALSE(session.generate(request).status.ok);
    request.cancel = nullptr;
    EXPECT_FALSE(session.generate(request).status.ok);
}
