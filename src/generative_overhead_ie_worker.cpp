#include "inferedge/generative_overhead_schema.hpp"
#include "inferedge/generative_session.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <llama.h>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using inferedge::GenerationError;
using inferedge::GenerationFinishReason;
using inferedge::GenerationRequest;
using inferedge::GenerationResult;
using inferedge::GenerativeSession;
using inferedge::generative_overhead::json_escape;
using inferedge::generative_overhead::json_extract_int;
using inferedge::generative_overhead::json_extract_string;
using inferedge::generative_overhead::json_token_array;
using inferedge::generative_overhead::kEvidencePathContract;

constexpr std::size_t kLogLimit = 1048576;

struct LogSink {
    std::mutex mutex;
    std::string buffer;
    bool overflow{false};
};

LogSink g_log;

void bounded_log(enum ggml_log_level, const char* text, void*) {
    if (text == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_log.mutex);
    const std::size_t n = std::strlen(text);
    if (g_log.buffer.size() + n > kLogLimit) {
        g_log.overflow = true;
        return;
    }
    g_log.buffer.append(text, n);
}

void install_log_sink() {
    llama_log_set(bounded_log, nullptr);
}

const char* error_name(GenerationError error) {
    switch (error) {
    case GenerationError::kNone:
        return "none";
    case GenerationError::kInvalidPackage:
        return "invalid_package";
    case GenerationError::kManifest:
        return "manifest";
    case GenerationError::kPathSecurity:
        return "path_security";
    case GenerationError::kHashMismatch:
        return "hash_mismatch";
    case GenerationError::kUnsupportedModel:
        return "unsupported_model";
    case GenerationError::kUnsupportedArchitecture:
        return "unsupported_architecture";
    case GenerationError::kUnsupportedQuantization:
        return "unsupported_quantization";
    case GenerationError::kLicense:
        return "license";
    case GenerationError::kModelLoad:
        return "model_load";
    case GenerationError::kTokenization:
        return "tokenization";
    case GenerationError::kContextLimit:
        return "context_limit";
    case GenerationError::kDecode:
        return "decode";
    case GenerationError::kInvalidUtf8:
        return "invalid_utf8";
    case GenerationError::kInternal:
        return "internal";
    }
    return "unexpected_error";
}

const char* finish_name(GenerationFinishReason reason) {
    switch (reason) {
    case GenerationFinishReason::kEos:
        return "eos";
    case GenerationFinishReason::kMaximumTokens:
        return "maximum_tokens";
    case GenerationFinishReason::kNone:
    case GenerationFinishReason::kCanceled:
        break;
    }
    return "none";
}

std::int64_t ns_from(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

void fail_line(const char* code, std::string_view message) {
    std::cout << "{\"ok\":false,\"error_code\":\"" << code << "\",\"error_message\":\""
              << json_escape(message) << "\"}\n";
}

bool tokenize_vocab_only(const std::string& gguf, const std::string& rendered,
                         std::vector<std::int32_t>& ids, std::string& error) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.vocab_only = true;
    llama_model* model = llama_model_load_from_file(gguf.c_str(), params);
    if (model == nullptr) {
        error = "vocab-only model load failed";
        return false;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (rendered.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        llama_model_free(model);
        error = "rendered text too large";
        return false;
    }
    const int32_t probe = llama_tokenize(
        vocab, rendered.c_str(), static_cast<int32_t>(rendered.size()), nullptr, 0, true, true);
    if (probe == std::numeric_limits<int32_t>::min() || probe >= 0) {
        llama_model_free(model);
        error = "tokenization probe failed";
        return false;
    }
    const int32_t needed = -probe;
    ids.assign(static_cast<std::size_t>(needed), 0);
    const int32_t wrote =
        llama_tokenize(vocab, rendered.c_str(), static_cast<int32_t>(rendered.size()), ids.data(),
                       needed, true, true);
    llama_model_free(model);
    if (wrote != needed) {
        error = "tokenization write failed";
        return false;
    }
    return true;
}

bool tokenize_loaded(const llama_vocab* vocab, const std::string& rendered,
                     std::vector<std::int32_t>& ids, std::string& error) {
    if (rendered.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        error = "rendered text too large";
        return false;
    }
    const int32_t probe = llama_tokenize(
        vocab, rendered.c_str(), static_cast<int32_t>(rendered.size()), nullptr, 0, true, true);
    if (probe == std::numeric_limits<int32_t>::min() || probe >= 0) {
        error = "tokenization probe failed";
        return false;
    }
    const int32_t needed = -probe;
    ids.assign(static_cast<std::size_t>(needed), 0);
    const int32_t wrote =
        llama_tokenize(vocab, rendered.c_str(), static_cast<int32_t>(rendered.size()), ids.data(),
                       needed, true, true);
    if (wrote != needed) {
        error = "tokenization write failed";
        return false;
    }
    return true;
}

int run_tokenize_stdio(const std::string& gguf) {
    install_log_sink();
    llama_backend_init();
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.vocab_only = true;
    llama_model* model = llama_model_load_from_file(gguf.c_str(), params);
    if (model == nullptr) {
        fail_line("tokenization", "vocab-only model load failed");
        llama_backend_free();
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::cout << std::unitbuf;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        std::string text;
        if (!json_extract_string(line, "text", text) &&
            !json_extract_string(line, "rendered_utf8", text)) {
            fail_line("tokenization", "missing text");
            continue;
        }
        std::vector<std::int32_t> ids;
        std::string error;
        if (!tokenize_loaded(vocab, text, ids, error)) {
            fail_line("tokenization", error);
            continue;
        }
        std::cout << "{\"ok\":true,\"token_ids\":" << json_token_array(ids) << "}\n";
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

void emit_generation(const GenerationResult& result, std::int64_t t0, std::int64_t t1,
                     std::uint32_t threads, const std::string& worker_id, std::int64_t pid,
                     bool log_overflow) {
    const char* error = error_name(result.status.error);
    if (log_overflow && result.status.ok) {
        error = "internal";
    }
    std::cout << "{\"ok\":" << (result.status.ok && !log_overflow ? "true" : "false")
              << ",\"error_code\":\"" << error << "\",\"error_message\":\""
              << json_escape(log_overflow ? "log sink overflow" : result.status.message)
              << "\",\"finish_reason\":\"" << finish_name(result.finish_reason)
              << "\",\"prompt_token_ids\":" << json_token_array(result.prompt_token_ids)
              << ",\"generated_token_ids\":" << json_token_array(result.generated_token_ids)
              << ",\"generated_text_utf8\":\"" << json_escape(result.text)
              << "\",\"prompt_token_count\":" << result.prompt_token_count
              << ",\"generated_token_count\":" << result.generated_token_count
              << ",\"threads_requested\":" << threads << ",\"threads_effective\":" << threads
              << ",\"n_ctx\":4096,\"n_batch\":4096,\"n_ubatch\":512,\"n_seq_max\":1"
              << ",\"worker_instance_id\":\"" << json_escape(worker_id) << "\",\"pid\":" << pid
              << ",\"t0_ns\":" << t0 << ",\"t1_ns\":" << t1 << ",\"duration_ns\":" << (t1 - t0)
              << "}\n";
}

int run_worker_stdio(const std::string& package) {
    install_log_sink();
    const std::string worker_id = "ie-" + std::to_string(static_cast<long long>(getpid()));
    const std::int64_t pid = static_cast<std::int64_t>(getpid());
    GenerativeSession session;
    bool loaded = false;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        std::string op;
        if (!json_extract_string(line, "op", op)) {
            fail_line("internal", "missing op");
            continue;
        }
        if (op == "identity") {
            std::cout << "{\"ok\":true,\"implementation\":\"inferedge\",\"worker_instance_id\":\""
                      << json_escape(worker_id) << "\",\"pid\":" << pid
                      << ",\"evidence_path_contract\":\"" << kEvidencePathContract << "\"}\n";
            continue;
        }
        if (op == "shutdown") {
            std::cout << "{\"ok\":true}\n";
            break;
        }
        if (op == "tokenize") {
            std::string text;
            std::string gguf = std::string(inferedge::generative_overhead::kPackageDirectory) +
                               "/" + std::string(inferedge::generative_overhead::kGgufFile);
            json_extract_string(line, "gguf_path", gguf);
            if (!json_extract_string(line, "text", text) &&
                !json_extract_string(line, "rendered_utf8", text)) {
                fail_line("tokenization", "missing text");
                continue;
            }
            llama_backend_init();
            std::vector<std::int32_t> ids;
            std::string error;
            const bool ok = tokenize_vocab_only(gguf, text, ids, error);
            llama_backend_free();
            if (!ok) {
                fail_line("tokenization", error);
                continue;
            }
            std::cout << "{\"ok\":true,\"token_ids\":" << json_token_array(ids) << "}\n";
            continue;
        }
        if (op == "load") {
            const auto status = session.load(package);
            loaded = status.ok;
            std::cout << "{\"ok\":" << (status.ok ? "true" : "false") << ",\"error_code\":\""
                      << error_name(status.error) << "\",\"error_message\":\""
                      << json_escape(status.message) << "\",\"worker_instance_id\":\""
                      << json_escape(worker_id) << "\",\"pid\":" << pid << "}\n";
            continue;
        }
        if (op == "l01") {
            GenerativeSession load_session;
            install_log_sink();
            const auto t0 = std::chrono::steady_clock::now();
            const auto status = load_session.load(package);
            const auto t1 = std::chrono::steady_clock::now();
            const bool overflow = g_log.overflow;
            std::cout << "{\"ok\":" << (status.ok && !overflow ? "true" : "false")
                      << ",\"error_code\":\"" << (overflow ? "internal" : error_name(status.error))
                      << "\",\"error_message\":\""
                      << json_escape(overflow ? "log sink overflow" : status.message)
                      << "\",\"finish_reason\":\"" << (status.ok ? "load_success" : "none")
                      << "\",\"prompt_token_ids\":[],\"generated_token_ids\":[]"
                      << ",\"generated_text_utf8\":\"\",\"prompt_token_count\":0"
                      << ",\"generated_token_count\":0,\"threads_requested\":1"
                      << ",\"threads_effective\":1,\"n_ctx\":4096,\"n_batch\":4096"
                      << ",\"n_ubatch\":512,\"n_seq_max\":1,\"worker_instance_id\":\""
                      << json_escape(worker_id) << "\",\"pid\":" << pid
                      << ",\"t0_ns\":" << ns_from(t0) << ",\"t1_ns\":" << ns_from(t1)
                      << ",\"duration_ns\":" << (ns_from(t1) - ns_from(t0)) << "}\n";
            continue;
        }
        if (op == "warm_request") {
            if (!loaded) {
                fail_line("model_load", "session is not loaded");
                continue;
            }
            std::string user;
            std::int64_t max_new = 0;
            std::int64_t threads = 0;
            if (!json_extract_string(line, "user_text_utf8", user) ||
                !json_extract_int(line, "max_new_tokens", max_new) ||
                !json_extract_int(line, "threads", threads)) {
                fail_line("internal", "invalid warm_request");
                continue;
            }
            GenerationRequest request;
            request.prompt = user;
            request.max_new_tokens = static_cast<std::uint32_t>(max_new);
            request.threads = static_cast<std::uint32_t>(threads);
            const auto t0 = std::chrono::steady_clock::now();
            const GenerationResult result = session.generate(request);
            const auto t1 = std::chrono::steady_clock::now();
            emit_generation(result, ns_from(t0), ns_from(t1), request.threads, worker_id, pid,
                            g_log.overflow);
            continue;
        }
        fail_line("internal", "unknown op");
    }
    return 0;
}

void usage() {
    std::cout << "Usage: inferedge-overhead-ie --help|--version|--tokenize-only --gguf "
                 "PATH|--worker-stdio "
                 "--package PATH\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string mode;
    std::string gguf;
    std::string package;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            usage();
            return 0;
        }
        if (arg == "--version") {
            std::cout << "inferedge-overhead-ie 0.1.0 " << kEvidencePathContract << "\n";
            return 0;
        }
        if (arg == "--tokenize-only" || arg == "--worker-stdio") {
            mode = arg;
        } else if (arg == "--gguf" && i + 1 < argc) {
            gguf = argv[++i];
        } else if (arg == "--package" && i + 1 < argc) {
            package = argv[++i];
        } else {
            fail_line("internal", "unknown argument");
            return 2;
        }
    }
    if (mode == "--tokenize-only") {
        if (gguf.empty()) {
            fail_line("internal", "--gguf is required");
            return 2;
        }
        return run_tokenize_stdio(gguf);
    }
    if (mode == "--worker-stdio") {
        if (package.empty()) {
            fail_line("internal", "--package is required");
            return 2;
        }
        return run_worker_stdio(package);
    }
    fail_line("internal", "missing required mode");
    return 2;
}
