#include "inferedge/generative_session.hpp"

#include <algorithm>
#include <limits>
#include <llama.h>
#include <mutex>
#include <utility>

namespace inferedge {
namespace {

GenerationStatus fail_status(GenerationError error, std::string message) {
    GenerationStatus status;
    status.error = error;
    status.message = std::move(message);
    return status;
}

GenerationStatus ok_status() {
    GenerationStatus status;
    status.ok = true;
    return status;
}

GenerationError map_package_error(PackageError error) {
    switch (error) {
    case PackageError::kNone:
        return GenerationError::kNone;
    case PackageError::kManifest:
        return GenerationError::kManifest;
    case PackageError::kPathSecurity:
        return GenerationError::kPathSecurity;
    case PackageError::kHashMismatch:
        return GenerationError::kHashMismatch;
    case PackageError::kUnsupportedModel:
        return GenerationError::kUnsupportedModel;
    case PackageError::kUnsupportedArchitecture:
        return GenerationError::kUnsupportedArchitecture;
    case PackageError::kUnsupportedQuantization:
        return GenerationError::kUnsupportedQuantization;
    case PackageError::kLicense:
        return GenerationError::kLicense;
    case PackageError::kInternal:
        break;
    }
    return GenerationError::kInvalidPackage;
}

bool llama_abort_adapter(void* data) {
    const auto* flag = static_cast<const CancellationFlag*>(data);
    return flag != nullptr && flag->is_canceled();
}

std::mutex& backend_mutex() {
    static std::mutex instance;
    return instance;
}

int& backend_count() {
    static int value = 0;
    return value;
}

class BackendLease {
public:
    BackendLease() {
        std::lock_guard<std::mutex> lock(backend_mutex());
        if (backend_count()++ == 0) {
            llama_backend_init();
        }
    }

    ~BackendLease() {
        std::lock_guard<std::mutex> lock(backend_mutex());
        if (--backend_count() == 0) {
            llama_backend_free();
        }
    }

    BackendLease(const BackendLease&) = delete;
    BackendLease& operator=(const BackendLease&) = delete;
};

// Pinned llama.cpp default physical microbatch. One constant for production
// load and the test accessor.
constexpr std::uint32_t kLlamaPhysicalMicrobatch = 512;

GenerativeContextSizing calculate_llama_context_sizing(std::uint32_t context_limit) noexcept {
    GenerativeContextSizing sizing;
    sizing.n_ctx = context_limit;
    sizing.n_batch = context_limit;
    sizing.n_ubatch = std::min(context_limit, kLlamaPhysicalMicrobatch);
    sizing.n_seq_max = 1;
    return sizing;
}

PackageStatus decode_token_piece(const llama_vocab* vocab, llama_token token, std::string& piece) {
    char probe[16];
    const int32_t first = llama_token_to_piece(vocab, token, probe, sizeof(probe), 0, false);
    if (first >= 0) {
        piece.assign(probe, static_cast<std::size_t>(first));
        return PackageStatus{true, PackageError::kNone, {}};
    }
    if (first == std::numeric_limits<int32_t>::min()) {
        return PackageStatus{false, PackageError::kInternal, "token piece size overflow"};
    }
    const int32_t required = -first;
    PackageStatus resized =
        resize_token_piece_buffer(piece, required, kGenerativeMaxTokenPieceBytes);
    if (!resized.ok) {
        return resized;
    }
    const int32_t wrote = llama_token_to_piece(vocab, token, piece.data(),
                                               static_cast<int32_t>(piece.size()), 0, false);
    if (wrote < 0 || static_cast<std::size_t>(wrote) > piece.size()) {
        return PackageStatus{false, PackageError::kInternal, "token piece decode failed"};
    }
    piece.resize(static_cast<std::size_t>(wrote));
    return PackageStatus{true, PackageError::kNone, {}};
}

} // namespace

void CancellationFlag::request_cancel() noexcept {
    canceled_.store(true, std::memory_order_release);
}

bool CancellationFlag::is_canceled() const noexcept {
    return canceled_.load(std::memory_order_acquire);
}

bool cancellation_adapter_should_abort(const CancellationFlag& flag) noexcept {
    return llama_abort_adapter(const_cast<CancellationFlag*>(&flag));
}

int GenerativeSessionTestAccess::BackendLeaseCount() noexcept {
    std::lock_guard<std::mutex> lock(backend_mutex());
    return backend_count();
}

GenerativeContextSizing
GenerativeSessionTestAccess::ContextSizing(std::uint32_t context_limit) noexcept {
    return calculate_llama_context_sizing(context_limit);
}

GenerationStatus evaluate_generation_context_budget(std::uint32_t prompt_tokens,
                                                    std::uint32_t max_new_tokens,
                                                    std::uint32_t effective_context) {
    if (effective_context == 0 || effective_context > kGenerativeMaxContext ||
        max_new_tokens == 0 || max_new_tokens > kGenerativeMaxContext) {
        return fail_status(GenerationError::kContextLimit, "invalid context budget");
    }
    if (prompt_tokens == 0 || prompt_tokens > effective_context) {
        return fail_status(GenerationError::kContextLimit, "prompt exceeds context");
    }
    if (max_new_tokens > effective_context - prompt_tokens) {
        return fail_status(GenerationError::kContextLimit, "requested output exceeds context");
    }
    return ok_status();
}

GenerationStatus token_count_from_llama_probe(std::int32_t probe, std::uint32_t& count) {
    if (probe == std::numeric_limits<std::int32_t>::min()) {
        return fail_status(GenerationError::kTokenization, "tokenization size overflow");
    }
    const std::int32_t needed = probe < 0 ? static_cast<std::int32_t>(-probe) : probe;
    if (needed <= 0) {
        return fail_status(GenerationError::kTokenization, "tokenization failed");
    }
    count = static_cast<std::uint32_t>(needed);
    return ok_status();
}

GenerationStatus append_generated_text(std::string& text, std::string_view piece) {
    if (piece.empty()) {
        return ok_status();
    }
    if (text.size() > kGenerativeMaxGeneratedBytes ||
        piece.size() > kGenerativeMaxGeneratedBytes - text.size()) {
        return fail_status(GenerationError::kDecode, "generated text exceeds maximum");
    }
    text.append(piece);
    return ok_status();
}

struct GenerativeSession::Impl {
    std::unique_ptr<BackendLease> backend;
    llama_model* model{nullptr};
    llama_context* context{nullptr};
    ValidatedModelPackage package;
    std::atomic<bool> generating{false};

    void reset() noexcept {
        if (context != nullptr) {
            llama_set_abort_callback(context, nullptr, nullptr);
            llama_free(context);
            context = nullptr;
        }
        if (model != nullptr) {
            llama_model_free(model);
            model = nullptr;
        }
        backend.reset();
        generating.store(false, std::memory_order_release);
    }

    ~Impl() { reset(); }
};

GenerativeSession::GenerativeSession() = default;
GenerativeSession::GenerativeSession(GenerativeSession&& other) noexcept = default;
GenerativeSession& GenerativeSession::operator=(GenerativeSession&& other) noexcept = default;

GenerativeSession::~GenerativeSession() = default;

bool GenerativeSession::loaded() const noexcept {
    return impl_ && impl_->model != nullptr && impl_->context != nullptr;
}

GenerationStatus GenerativeSession::load(const std::filesystem::path& package_directory) {
    if (impl_ && impl_->generating.load(std::memory_order_acquire)) {
        return fail_status(GenerationError::kInternal, "concurrent session use is unsupported");
    }
    impl_.reset();
    PackageLoadResult loaded = load_model_package(package_directory);
    if (!loaded.status.ok) {
        return fail_status(map_package_error(loaded.status.error), loaded.status.message);
    }
    auto impl = std::make_unique<Impl>();
    impl->package = std::move(loaded.package);
    impl->backend = std::make_unique<BackendLease>();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    impl->model =
        llama_model_load_from_file(impl->package.gguf_path.string().c_str(), model_params);
    if (impl->model == nullptr) {
        impl->reset();
        return fail_status(GenerationError::kModelLoad, "llama.cpp rejected the GGUF");
    }
    llama_context_params context_params = llama_context_default_params();
    const GenerativeContextSizing sizing =
        calculate_llama_context_sizing(impl->package.manifest.context_limit);
    context_params.n_ctx = sizing.n_ctx;
    context_params.n_batch = sizing.n_batch;
    context_params.n_ubatch = sizing.n_ubatch;
    context_params.n_seq_max = sizing.n_seq_max;
    context_params.n_threads = 1;
    context_params.n_threads_batch = 1;
    context_params.embeddings = false;
    impl->context = llama_init_from_model(impl->model, context_params);
    if (impl->context == nullptr) {
        impl->reset();
        return fail_status(GenerationError::kModelLoad, "llama.cpp context creation failed");
    }
    impl_ = std::move(impl);
    return ok_status();
}

GenerationResult GenerativeSession::generate(const GenerationRequest& request) {
    GenerationResult result;
    if (!loaded()) {
        result.status = fail_status(GenerationError::kModelLoad, "session is not loaded");
        return result;
    }
    bool expected = false;
    if (!impl_->generating.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        result.status =
            fail_status(GenerationError::kInternal, "concurrent generate is unsupported");
        return result;
    }
    struct GeneratingGuard {
        Impl* impl;
        ~GeneratingGuard() {
            if (impl != nullptr && impl->context != nullptr) {
                llama_set_abort_callback(impl->context, nullptr, nullptr);
            }
            if (impl != nullptr) {
                impl->generating.store(false, std::memory_order_release);
            }
        }
    } guard{impl_.get()};

    if (request.threads == 0 || request.threads > 256 || !user_prompt_is_usable(request.prompt)) {
        result.status = fail_status(GenerationError::kInternal, "invalid generation request");
        return result;
    }
    if (request.cancel != nullptr && request.cancel->is_canceled()) {
        result.status.ok = true;
        result.finish_reason = GenerationFinishReason::kCanceled;
        return result;
    }
    std::string rendered;
    PackageStatus templated =
        apply_frozen_template(impl_->package.template_text, request.prompt, rendered);
    if (!templated.ok) {
        result.status = fail_status(GenerationError::kTokenization, templated.message);
        return result;
    }
    llama_set_n_threads(impl_->context, static_cast<int32_t>(request.threads),
                        static_cast<int32_t>(request.threads));
    if (request.cancel != nullptr) {
        llama_set_abort_callback(impl_->context, llama_abort_adapter,
                                 const_cast<CancellationFlag*>(request.cancel));
    } else {
        llama_set_abort_callback(impl_->context, nullptr, nullptr);
    }
    const llama_vocab* vocab = llama_model_get_vocab(impl_->model);
    if (rendered.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        result.status = fail_status(GenerationError::kTokenization, "prompt too large to tokenize");
        return result;
    }
    const int32_t probe = llama_tokenize(
        vocab, rendered.c_str(), static_cast<int32_t>(rendered.size()), nullptr, 0, true, true);
    std::uint32_t needed = 0;
    result.status = token_count_from_llama_probe(probe, needed);
    if (!result.status.ok) {
        return result;
    }
    const auto llama_ctx = static_cast<std::uint32_t>(llama_n_ctx(impl_->context));
    const std::uint32_t effective =
        std::min(std::min(impl_->package.manifest.context_limit, llama_ctx), kGenerativeMaxContext);
    result.status = evaluate_generation_context_budget(needed, request.max_new_tokens, effective);
    if (!result.status.ok) {
        return result;
    }
    result.prompt_token_ids.resize(static_cast<std::size_t>(needed));
    const int32_t wrote =
        llama_tokenize(vocab, rendered.c_str(), static_cast<int32_t>(rendered.size()),
                       result.prompt_token_ids.data(), static_cast<int32_t>(needed), true, true);
    if (wrote != static_cast<int32_t>(needed)) {
        result.status = fail_status(GenerationError::kTokenization, "tokenization failed");
        return result;
    }
    result.prompt_token_count = static_cast<std::uint32_t>(wrote);
    llama_memory_clear(llama_get_memory(impl_->context), true);
    llama_sampler* sampler = llama_sampler_init_greedy();
    if (sampler == nullptr) {
        result.status = fail_status(GenerationError::kInternal, "greedy sampler creation failed");
        return result;
    }
    struct SamplerGuard {
        llama_sampler* sampler;
        ~SamplerGuard() {
            if (sampler != nullptr) {
                llama_sampler_free(sampler);
            }
        }
    } sampler_guard{sampler};
    llama_batch batch = llama_batch_get_one(result.prompt_token_ids.data(), wrote);
    const int32_t prompt_rc = llama_decode(impl_->context, batch);
    if (prompt_rc == 2) {
        result.status.ok = true;
        result.finish_reason = GenerationFinishReason::kCanceled;
        return result;
    }
    if (prompt_rc != 0) {
        result.status = fail_status(GenerationError::kDecode, "prompt decode failed");
        return result;
    }
    std::string pieces;
    for (std::uint32_t produced = 0; produced < request.max_new_tokens; ++produced) {
        if (request.cancel != nullptr && request.cancel->is_canceled()) {
            result.status.ok = true;
            result.finish_reason = GenerationFinishReason::kCanceled;
            result.generated_token_count =
                static_cast<std::uint32_t>(result.generated_token_ids.size());
            pieces.resize(complete_utf8_prefix_length(pieces));
            result.text = std::move(pieces);
            return result;
        }
        const llama_token token = llama_sampler_sample(sampler, impl_->context, -1);
        llama_sampler_accept(sampler, token);
        result.generated_token_ids.push_back(token);
        if (llama_vocab_is_eog(vocab, token)) {
            result.generated_token_count =
                static_cast<std::uint32_t>(result.generated_token_ids.size());
            if (!is_strict_utf8(pieces)) {
                result.status =
                    fail_status(GenerationError::kInvalidUtf8, "generated text is not UTF-8");
                return result;
            }
            result.status.ok = true;
            result.finish_reason = GenerationFinishReason::kEos;
            result.text = std::move(pieces);
            return result;
        }
        std::string piece;
        const PackageStatus decoded = decode_token_piece(vocab, token, piece);
        if (!decoded.ok) {
            result.status = fail_status(GenerationError::kDecode, decoded.message);
            return result;
        }
        result.status = append_generated_text(pieces, piece);
        if (!result.status.ok) {
            return result;
        }
        llama_batch next = llama_batch_get_one(&result.generated_token_ids.back(), 1);
        const int32_t rc = llama_decode(impl_->context, next);
        if (rc == 2) {
            result.status.ok = true;
            result.finish_reason = GenerationFinishReason::kCanceled;
            result.generated_token_count =
                static_cast<std::uint32_t>(result.generated_token_ids.size());
            pieces.resize(complete_utf8_prefix_length(pieces));
            result.text = std::move(pieces);
            return result;
        }
        if (rc != 0) {
            result.status = fail_status(GenerationError::kDecode, "token decode failed");
            return result;
        }
    }
    if (!is_strict_utf8(pieces)) {
        result.status = fail_status(GenerationError::kInvalidUtf8, "generated text is not UTF-8");
        return result;
    }
    result.status.ok = true;
    result.finish_reason = GenerationFinishReason::kMaximumTokens;
    result.generated_token_count = static_cast<std::uint32_t>(result.generated_token_ids.size());
    result.text = std::move(pieces);
    return result;
}

} // namespace inferedge
