#include "inferedge/generative_overhead_schema.hpp"
#include "inferedge/model_package_manifest.hpp"

// This worker must not include generative_session.hpp or link inferedge_generative.

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

using inferedge::apply_frozen_template;
using inferedge::is_strict_utf8;
using inferedge::kGenerativeMaxContext;
using inferedge::load_model_package;
using inferedge::PackageStatus;
using inferedge::resize_token_piece_buffer;
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

const char* token_probe_count(int32_t probe, std::uint32_t& count) {
    if (probe == std::numeric_limits<int32_t>::min() || probe >= 0) {
        return "tokenization";
    }
    count = static_cast<std::uint32_t>(-probe);
    if (count == 0) {
        return "tokenization";
    }
    return nullptr;
}

const char* context_budget(std::uint32_t prompt_tokens, std::uint32_t max_new_tokens) {
    if (prompt_tokens == 0 || prompt_tokens > kGenerativeMaxContext || max_new_tokens == 0 ||
        max_new_tokens > kGenerativeMaxContext - prompt_tokens) {
        return "context_limit";
    }
    return nullptr;
}

std::int64_t ns_from(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

void fail_line(const char* code, std::string_view message) {
    std::cout << "{\"ok\":false,\"error_code\":\"" << code << "\",\"error_message\":\""
              << json_escape(message) << "\"}\n";
}

void fill_l01_model_params(llama_model_params& params) {
    params.devices = nullptr;
    params.tensor_buft_overrides = nullptr;
    params.n_gpu_layers = 0;
    params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    params.load_mode = LLAMA_LOAD_MODE_AUTO;
    params.lazy_mode = LLAMA_LAZY_MODE_AUTO;
    params.main_gpu = 0;
    params.tensor_split = nullptr;
    params.progress_callback = nullptr;
    params.progress_callback_user_data = nullptr;
    params.kv_overrides = nullptr;
    params.vocab_only = false;
    params.check_tensors = false;
    params.use_extra_bufts = true;
    params.no_host = false;
    params.no_alloc = false;
    params.load_mtp = false;
}

void fill_l01_context_params(llama_context_params& params) {
    params.n_ctx = 4096;
    params.n_batch = 4096;
    params.n_ubatch = 512;
    params.n_seq_max = 1;
    params.n_rs_seq = 0;
    params.n_outputs_max = 0;
    params.n_outputs_max_per_seq = 1;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
    params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    params.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
    params.attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    params.rope_freq_base = 0.0f;
    params.rope_freq_scale = 0.0f;
    params.yarn_ext_factor = -1.0f;
    params.yarn_attn_factor = -1.0f;
    params.yarn_beta_fast = -1.0f;
    params.yarn_beta_slow = -1.0f;
    params.yarn_orig_ctx = 0;
    params.defrag_thold = -1.0f;
    params.cb_eval = nullptr;
    params.cb_eval_user_data = nullptr;
    params.type_k = GGML_TYPE_F16;
    params.type_v = GGML_TYPE_F16;
    params.moe_cache_size = 0;
    params.abort_callback = nullptr;
    params.abort_callback_data = nullptr;
    params.embeddings = false;
    params.offload_kqv = true;
    params.no_perf = true;
    params.op_offload = true;
    params.swa_full = true;
    params.kv_unified = false;
    params.samplers = nullptr;
    params.n_samplers = 0;
    params.ctx_other = nullptr;
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

struct LoadedRef {
    llama_model* model{nullptr};
    llama_context* context{nullptr};
    inferedge::ValidatedModelPackage package;
    bool backend_owned{false};

    void reset() {
        if (context != nullptr) {
            llama_free(context);
            context = nullptr;
        }
        if (model != nullptr) {
            llama_model_free(model);
            model = nullptr;
        }
        if (backend_owned) {
            llama_backend_free();
            backend_owned = false;
        }
    }

    ~LoadedRef() { reset(); }
};

bool decode_piece(const llama_vocab* vocab, llama_token token, std::string& piece) {
    char probe[16];
    const int32_t first = llama_token_to_piece(vocab, token, probe, sizeof(probe), 0, false);
    if (first >= 0) {
        piece.assign(probe, static_cast<std::size_t>(first));
        return true;
    }
    if (first == std::numeric_limits<int32_t>::min()) {
        return false;
    }
    const int32_t required = -first;
    const PackageStatus resized =
        resize_token_piece_buffer(piece, required, inferedge::kGenerativeMaxTokenPieceBytes);
    if (!resized.ok) {
        return false;
    }
    const int32_t wrote = llama_token_to_piece(vocab, token, piece.data(),
                                               static_cast<int32_t>(piece.size()), 0, false);
    if (wrote < 0 || static_cast<std::size_t>(wrote) > piece.size()) {
        return false;
    }
    piece.resize(static_cast<std::size_t>(wrote));
    return true;
}

int run_worker_stdio(const std::string& package_dir) {
    install_log_sink();
    const std::string worker_id = "ref-" + std::to_string(static_cast<long long>(getpid()));
    const std::int64_t pid = static_cast<std::int64_t>(getpid());
    LoadedRef loaded;
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
            std::cout << "{\"ok\":true,\"implementation\":\"reference\",\"worker_instance_id\":\""
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
            loaded.reset();
            const auto pack = load_model_package(package_dir);
            if (!pack.status.ok) {
                fail_line("model_load", pack.status.message);
                continue;
            }
            llama_backend_init();
            loaded.backend_owned = true;
            loaded.package = pack.package;
            llama_model_params model_params = llama_model_default_params();
            model_params.n_gpu_layers = 0;
            loaded.model =
                llama_model_load_from_file(loaded.package.gguf_path.string().c_str(), model_params);
            if (loaded.model == nullptr) {
                loaded.reset();
                fail_line("model_load", "llama.cpp rejected the GGUF");
                continue;
            }
            llama_context_params context_params = llama_context_default_params();
            context_params.n_ctx = 4096;
            context_params.n_batch = 4096;
            context_params.n_ubatch = 512;
            context_params.n_seq_max = 1;
            context_params.n_threads = 1;
            context_params.n_threads_batch = 1;
            context_params.embeddings = false;
            loaded.context = llama_init_from_model(loaded.model, context_params);
            if (loaded.context == nullptr) {
                loaded.reset();
                fail_line("model_load", "llama.cpp context creation failed");
                continue;
            }
            std::cout << "{\"ok\":true,\"error_code\":\"none\",\"error_message\":\"\","
                      << "\"worker_instance_id\":\"" << json_escape(worker_id)
                      << "\",\"pid\":" << pid << "}\n";
            continue;
        }
        if (op == "l01") {
            install_log_sink();
            const auto pack = load_model_package(package_dir);
            if (!pack.status.ok) {
                fail_line("model_load", pack.status.message);
                continue;
            }
            llama_model_params model_params = llama_model_default_params();
            fill_l01_model_params(model_params);
            llama_context_params context_params = llama_context_default_params();
            fill_l01_context_params(context_params);
            const std::string gguf = pack.package.gguf_path.string();
            llama_model* model = nullptr;
            llama_context* context = nullptr;
            const auto t0 = std::chrono::steady_clock::now();
            llama_backend_init();
            model = llama_model_load_from_file(gguf.c_str(), model_params);
            if (model != nullptr) {
                context = llama_init_from_model(model, context_params);
            }
            std::chrono::steady_clock::time_point t1;
            const bool ok = model != nullptr && context != nullptr;
            if (ok) {
                t1 = std::chrono::steady_clock::now();
            } else {
                t1 = std::chrono::steady_clock::now();
            }
            if (context != nullptr) {
                llama_free(context);
            }
            if (model != nullptr) {
                llama_model_free(model);
            }
            llama_backend_free();
            const bool overflow = g_log.overflow;
            std::cout << "{\"ok\":" << (ok && !overflow ? "true" : "false") << ",\"error_code\":\""
                      << (overflow ? "internal" : (ok ? "none" : "model_load"))
                      << "\",\"error_message\":\""
                      << json_escape(overflow ? "log sink overflow" : (ok ? "" : "l01 load failed"))
                      << "\",\"finish_reason\":\"" << (ok ? "load_success" : "none")
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
            if (loaded.model == nullptr || loaded.context == nullptr) {
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
            const auto t0 = std::chrono::steady_clock::now();
            std::string rendered;
            const PackageStatus templated =
                apply_frozen_template(loaded.package.template_text, user, rendered);
            if (!templated.ok) {
                const auto t1 = std::chrono::steady_clock::now();
                fail_line("tokenization", templated.message);
                (void)t1;
                continue;
            }
            llama_set_n_threads(loaded.context, static_cast<int32_t>(threads),
                                static_cast<int32_t>(threads));
            const llama_vocab* vocab = llama_model_get_vocab(loaded.model);
            const int32_t probe =
                llama_tokenize(vocab, rendered.c_str(), static_cast<int32_t>(rendered.size()),
                               nullptr, 0, true, true);
            std::uint32_t needed = 0;
            const char* error = token_probe_count(probe, needed);
            std::string message;
            std::vector<std::int32_t> prompt_ids;
            std::vector<std::int32_t> generated_ids;
            std::string text;
            const char* finish = "none";
            if (error == nullptr) {
                error = context_budget(needed, static_cast<std::uint32_t>(max_new));
            }
            if (error == nullptr) {
                prompt_ids.resize(needed);
                const int32_t wrote =
                    llama_tokenize(vocab, rendered.c_str(), static_cast<int32_t>(rendered.size()),
                                   prompt_ids.data(), static_cast<int32_t>(needed), true, true);
                if (wrote != static_cast<int32_t>(needed)) {
                    error = "tokenization";
                    message = "tokenization failed";
                }
            }
            if (error == nullptr) {
                llama_memory_clear(llama_get_memory(loaded.context), true);
                llama_sampler* sampler = llama_sampler_init_greedy();
                if (sampler == nullptr) {
                    error = "internal";
                    message = "greedy sampler creation failed";
                } else {
                    llama_batch batch = llama_batch_get_one(
                        prompt_ids.data(), static_cast<int32_t>(prompt_ids.size()));
                    const int32_t prompt_rc = llama_decode(loaded.context, batch);
                    if (prompt_rc != 0) {
                        error = "decode";
                        message = "prompt decode failed";
                    } else {
                        for (std::int64_t produced = 0; produced < max_new; ++produced) {
                            const llama_token token =
                                llama_sampler_sample(sampler, loaded.context, -1);
                            llama_sampler_accept(sampler, token);
                            generated_ids.push_back(token);
                            if (llama_vocab_is_eog(vocab, token)) {
                                finish = "eos";
                                break;
                            }
                            std::string piece;
                            if (!decode_piece(vocab, token, piece)) {
                                error = "decode";
                                message = "token piece decode failed";
                                break;
                            }
                            text += piece;
                            llama_batch next = llama_batch_get_one(&generated_ids.back(), 1);
                            if (llama_decode(loaded.context, next) != 0) {
                                error = "decode";
                                message = "token decode failed";
                                break;
                            }
                        }
                        if (error == nullptr && std::strcmp(finish, "eos") != 0) {
                            finish = "maximum_tokens";
                        }
                        if (error == nullptr && !is_strict_utf8(text)) {
                            error = "invalid_utf8";
                            message = "generated text is not UTF-8";
                            finish = "none";
                        }
                    }
                    llama_sampler_free(sampler);
                }
            }
            const auto t1 = std::chrono::steady_clock::now();
            const bool overflow = g_log.overflow;
            const bool ok = error == nullptr && !overflow;
            std::cout << "{\"ok\":" << (ok ? "true" : "false") << ",\"error_code\":\""
                      << (overflow ? "internal" : (error == nullptr ? "none" : error))
                      << "\",\"error_message\":\""
                      << json_escape(overflow ? "log sink overflow" : message)
                      << "\",\"finish_reason\":\"" << (ok ? finish : "none")
                      << "\",\"prompt_token_ids\":" << json_token_array(prompt_ids)
                      << ",\"generated_token_ids\":" << json_token_array(generated_ids)
                      << ",\"generated_text_utf8\":\"" << json_escape(text)
                      << "\",\"prompt_token_count\":" << prompt_ids.size()
                      << ",\"generated_token_count\":" << generated_ids.size()
                      << ",\"threads_requested\":" << threads
                      << ",\"threads_effective\":" << threads
                      << ",\"n_ctx\":4096,\"n_batch\":4096,\"n_ubatch\":512,\"n_seq_max\":1"
                      << ",\"worker_instance_id\":\"" << json_escape(worker_id)
                      << "\",\"pid\":" << pid << ",\"t0_ns\":" << ns_from(t0)
                      << ",\"t1_ns\":" << ns_from(t1)
                      << ",\"duration_ns\":" << (ns_from(t1) - ns_from(t0)) << "}\n";
            continue;
        }
        fail_line("internal", "unknown op");
    }
    return 0;
}

void usage() {
    std::cout << "Usage: inferedge-overhead-ref --help|--version|--tokenize-only --gguf "
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
            std::cout << "inferedge-overhead-ref 0.1.0 " << kEvidencePathContract << "\n";
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
