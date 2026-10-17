#pragma once

#include "inferedge/model_package_manifest.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace inferedge {

enum class GenerationFinishReason {
    kNone,
    kEos,
    kMaximumTokens,
    kCanceled,
};

enum class GenerationError {
    kNone,
    kInvalidPackage,
    kManifest,
    kPathSecurity,
    kHashMismatch,
    kUnsupportedModel,
    kUnsupportedArchitecture,
    kUnsupportedQuantization,
    kLicense,
    kModelLoad,
    kTokenization,
    kContextLimit,
    kDecode,
    kInvalidUtf8,
    kInternal,
};

struct GenerationStatus {
    bool ok{false};
    GenerationError error{GenerationError::kNone};
    std::string message;
};

// Caller-owned cancellation signal. request_cancel() is safe from another thread.
// The object referenced by GenerationRequest::cancel must outlive generate().
class CancellationFlag {
public:
    void request_cancel() noexcept;
    bool is_canceled() const noexcept;

private:
    std::atomic<bool> canceled_{false};
};

bool cancellation_adapter_should_abort(const CancellationFlag& flag) noexcept;

// Request data must outlive the synchronous generate() call. cancel may be null.
struct GenerationRequest {
    std::string prompt;
    std::uint32_t max_new_tokens{16};
    std::uint32_t threads{1};
    const CancellationFlag* cancel{nullptr};
};

struct GenerationResult {
    GenerationStatus status;
    GenerationFinishReason finish_reason{GenerationFinishReason::kNone};
    std::vector<std::int32_t> prompt_token_ids;
    std::vector<std::int32_t> generated_token_ids;
    std::string text;
    std::uint32_t prompt_token_count{0};
    std::uint32_t generated_token_count{0};
};

// Overflow-safe: prompt_tokens <= effective and max_new <= effective - prompt_tokens.
// Zero generated tokens are not supported.
GenerationStatus evaluate_generation_context_budget(std::uint32_t prompt_tokens,
                                                    std::uint32_t max_new_tokens,
                                                    std::uint32_t effective_context);

// Rejects INT32_MIN before negation. Overflow-safe conversion to uint32.
GenerationStatus token_count_from_llama_probe(std::int32_t probe, std::uint32_t& count);

// Overflow-safe append against kGenerativeMaxGeneratedBytes. Does not truncate.
GenerationStatus append_generated_text(std::string& text, std::string_view piece);

// One loaded model. Sequential generate() calls are supported. Concurrent
// generate(), load(), or destruction overlapping generate() is unsupported.
class GenerativeSession {
public:
    GenerativeSession();
    GenerativeSession(const GenerativeSession&) = delete;
    GenerativeSession& operator=(const GenerativeSession&) = delete;
    GenerativeSession(GenerativeSession&& other) noexcept;
    GenerativeSession& operator=(GenerativeSession&& other) noexcept;
    ~GenerativeSession();

    GenerationStatus load(const std::filesystem::path& package_directory);
    bool loaded() const noexcept;
    GenerationResult generate(const GenerationRequest& request);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class GenerativeSessionTestAccess;
};

// Test-only view of the llama.cpp context sizes applied at load. Production
// callers must not use this type.
struct GenerativeContextSizing {
    std::uint32_t n_ctx{0};
    std::uint32_t n_batch{0};
    std::uint32_t n_ubatch{0};
    std::uint32_t n_seq_max{0};
};

// Test-only lease accounting and context-sizing access. Production callers
// must not use this type.
class GenerativeSessionTestAccess {
public:
    static int BackendLeaseCount() noexcept;
    static GenerativeContextSizing ContextSizing(std::uint32_t context_limit) noexcept;
};

} // namespace inferedge
