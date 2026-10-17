#include "inferedge/model_package_manifest.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <system_error>
#include <utility>

namespace inferedge {
namespace {

constexpr char kHex[] = "0123456789abcdef";

constexpr std::uint32_t kSha256K[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U,
};

PackageStatus fail(PackageError error, std::string message) {
    PackageStatus status;
    status.error = error;
    status.message = std::move(message);
    return status;
}

PackageStatus ok_status() {
    PackageStatus status;
    status.ok = true;
    return status;
}

bool contains_nul_or_cr(std::string_view text) {
    return text.find('\0') != std::string_view::npos || text.find('\r') != std::string_view::npos;
}

bool decode_strict_utf8(std::string_view text, bool allow_incomplete_tail, std::size_t& complete) {
    complete = 0;
    std::size_t index = 0;
    while (index < text.size()) {
        const unsigned char lead = static_cast<unsigned char>(text[index]);
        std::size_t need = 0;
        std::uint32_t min_code = 0;
        std::uint32_t code = 0;
        if (lead <= 0x7fU) {
            ++index;
            complete = index;
            continue;
        }
        if ((lead & 0xe0U) == 0xc0U) {
            need = 1;
            min_code = 0x80U;
            code = lead & 0x1fU;
        } else if ((lead & 0xf0U) == 0xe0U) {
            need = 2;
            min_code = 0x800U;
            code = lead & 0x0fU;
        } else if ((lead & 0xf8U) == 0xf0U) {
            need = 3;
            min_code = 0x10000U;
            code = lead & 0x07U;
        } else {
            return false;
        }
        if (index + need >= text.size()) {
            return allow_incomplete_tail;
        }
        for (std::size_t offset = 1; offset <= need; ++offset) {
            const unsigned char cont = static_cast<unsigned char>(text[index + offset]);
            if ((cont & 0xc0U) != 0x80U) {
                return false;
            }
            code = (code << 6U) | (cont & 0x3fU);
        }
        if (code < min_code || code > 0x10ffffU || (code >= 0xd800U && code <= 0xdfffU)) {
            return false;
        }
        index += need + 1;
        complete = index;
    }
    return true;
}

bool is_valid_utf8(std::string_view text) {
    std::size_t complete = 0;
    return decode_strict_utf8(text, false, complete) && complete == text.size();
}

bool is_hex_sha256(std::string_view value) {
    if (value.size() != 64) {
        return false;
    }
    for (char ch : value) {
        if (!std::isxdigit(static_cast<unsigned char>(ch)) ||
            std::isupper(static_cast<unsigned char>(ch))) {
            return false;
        }
    }
    return true;
}

bool is_safe_file_name(std::string_view name) {
    if (name.empty() || name == "." || name == ".." || name.size() > 255) {
        return false;
    }
    if (name.find('/') != std::string_view::npos || name.find('\\') != std::string_view::npos ||
        name.find('%') != std::string_view::npos) {
        return false;
    }
    if (name.front() == '.') {
        return false;
    }
    for (char ch : name) {
        const unsigned char byte = static_cast<unsigned char>(ch);
        if (byte < 0x20 || byte == 0x7f) {
            return false;
        }
        if (!(std::isalnum(byte) || ch == '.' || ch == '_' || ch == '-')) {
            return false;
        }
    }
    return true;
}

bool is_ident(std::string_view value) {
    if (value.empty() || value.size() > 128) {
        return false;
    }
    for (char ch : value) {
        const unsigned char byte = static_cast<unsigned char>(ch);
        if (!(std::isalnum(byte) || ch == '.' || ch == '_' || ch == '-' || ch == '/')) {
            return false;
        }
    }
    if (value.find("..") != std::string_view::npos || value.front() == '/' || value.back() == '/') {
        return false;
    }
    return true;
}

bool parse_u64(std::string_view text, std::uint64_t& value) {
    if (text.empty() || text.size() > 20 || text[0] == '+' || text[0] == '-') {
        return false;
    }
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return false;
    }
    value = parsed;
    return true;
}

bool parse_u32(std::string_view text, std::uint32_t& value) {
    std::uint64_t parsed = 0;
    if (!parse_u64(text, parsed) || parsed > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    value = static_cast<std::uint32_t>(parsed);
    return true;
}

PackageStatus parse_key_values(std::string_view text, std::size_t max_bytes,
                               const std::vector<std::string_view>& required,
                               bool allow_optional_notice,
                               std::map<std::string, std::string>& values) {
    if (text.size() > max_bytes) {
        return fail(PackageError::kManifest, "document exceeds maximum size");
    }
    if (contains_nul_or_cr(text)) {
        return fail(PackageError::kManifest, "document contains CR or NUL");
    }
    if (!is_valid_utf8(text)) {
        return fail(PackageError::kManifest, "document is not valid UTF-8");
    }
    if (text.empty()) {
        return fail(PackageError::kManifest, "document is empty");
    }
    std::size_t begin = 0;
    while (begin < text.size()) {
        std::size_t end = text.find('\n', begin);
        const bool last = end == std::string_view::npos;
        std::string_view line = text.substr(begin, last ? std::string_view::npos : end - begin);
        if (line.empty()) {
            return fail(PackageError::kManifest, "empty line");
        }
        if (line.front() == ' ' || line.back() == ' ' || line.front() == '\t' ||
            line.back() == '\t') {
            return fail(PackageError::kManifest, "leading or trailing whitespace");
        }
        const std::size_t eq = line.find('=');
        if (eq == std::string_view::npos || eq == 0) {
            return fail(PackageError::kManifest, "line is not key=value");
        }
        const std::string_view key = line.substr(0, eq);
        const std::string_view value = line.substr(eq + 1);
        if (key.find(' ') != std::string_view::npos || key.find('\t') != std::string_view::npos) {
            return fail(PackageError::kManifest, "whitespace in key");
        }
        if (value.find(' ') != std::string_view::npos ||
            value.find('\t') != std::string_view::npos) {
            return fail(PackageError::kManifest, "whitespace in value");
        }
        if (value.empty()) {
            return fail(PackageError::kManifest, "empty value");
        }
        if (values.contains(std::string{key})) {
            return fail(PackageError::kManifest, "duplicate key");
        }
        values.emplace(std::string{key}, std::string{value});
        if (last) {
            break;
        }
        begin = end + 1;
        if (begin == text.size()) {
            break;
        }
    }
    for (std::string_view key : required) {
        if (!values.contains(std::string{key})) {
            return fail(PackageError::kManifest, "missing key");
        }
    }
    for (const auto& [key, value] : values) {
        bool known = false;
        for (std::string_view required_key : required) {
            if (key == required_key) {
                known = true;
                break;
            }
        }
        if (!known && allow_optional_notice && key == "notice_file") {
            known = true;
        }
        if (!known) {
            return fail(PackageError::kManifest, "unknown key");
        }
        (void)value;
    }
    return ok_status();
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
    if (state.overflow || state.filled > 64) {
        return false;
    }
    const std::uint64_t total_bits = state.bits + static_cast<std::uint64_t>(state.filled) * 8U;
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

bool read_regular_file(const std::filesystem::path& path, std::size_t max_bytes, std::string& out) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }
    out.assign(max_bytes + 1, '\0');
    input.read(out.data(), static_cast<std::streamsize>(out.size()));
    const auto got = static_cast<std::size_t>(input.gcount());
    if (got > max_bytes) {
        return false;
    }
    out.resize(got);
    return true;
}

PackageStatus require_regular_unique_file(const std::filesystem::path& path) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error || status.type() == std::filesystem::file_type::not_found) {
        return fail(PackageError::kPathSecurity, "file is missing");
    }
    if (status.type() == std::filesystem::file_type::symlink) {
        return fail(PackageError::kPathSecurity, "symlink file");
    }
    if (status.type() != std::filesystem::file_type::regular) {
        return fail(PackageError::kPathSecurity, "file is not regular");
    }
    const auto links = std::filesystem::hard_link_count(path, error);
    if (error) {
        return fail(PackageError::kPathSecurity, "link count unavailable");
    }
    if (links > 1) {
        return fail(PackageError::kPathSecurity, "hard-linked file");
    }
    return ok_status();
}

bool is_under_root(const std::filesystem::path& root, const std::filesystem::path& child) {
    const auto root_text = root.lexically_normal().generic_string();
    const auto child_text = child.lexically_normal().generic_string();
    if (child_text == root_text) {
        return true;
    }
    return child_text.size() > root_text.size() &&
           child_text.compare(0, root_text.size(), root_text) == 0 &&
           child_text[root_text.size()] == '/';
}

} // namespace

bool is_strict_utf8(std::string_view text) noexcept {
    return is_valid_utf8(text);
}

std::size_t complete_utf8_prefix_length(std::string_view text) noexcept {
    std::size_t complete = 0;
    (void)decode_strict_utf8(text, true, complete);
    return complete;
}

bool user_prompt_is_usable(std::string_view prompt) noexcept {
    if (prompt.empty() || prompt.size() > kGenerativeMaxPromptBytes || contains_nul_or_cr(prompt) ||
        !is_valid_utf8(prompt)) {
        return false;
    }
    for (char ch : prompt) {
        if (ch != ' ' && ch != '\t' && ch != '\n') {
            return true;
        }
    }
    return false;
}

bool path_is_under_root(const std::filesystem::path& root, const std::filesystem::path& child) {
    return is_under_root(root, child);
}

PackageStatus resize_token_piece_buffer(std::string& buffer, int32_t required,
                                        std::size_t maximum) {
    if (required <= 0) {
        return fail(PackageError::kInternal, "token piece size is invalid");
    }
    const auto needed = static_cast<std::size_t>(required);
    if (needed > maximum) {
        return fail(PackageError::kInternal, "token piece exceeds maximum");
    }
    buffer.assign(needed, '\0');
    return ok_status();
}

std::string sha256_hex(const std::uint8_t* data, std::size_t length) {
    Sha256State state;
    std::uint8_t raw[32]{};
    if (!absorb(state, data, length) || !finish_sha256(state, raw)) {
        return {};
    }
    std::string hex(64, '0');
    for (int index = 0; index < 32; ++index) {
        hex[static_cast<std::size_t>(index) * 2U] = kHex[raw[index] >> 4U];
        hex[static_cast<std::size_t>(index) * 2U + 1U] = kHex[raw[index] & 0x0fU];
    }
    return hex;
}

bool hashes_equal(std::string_view left, std::string_view right) noexcept {
    if (left.size() != right.size()) {
        return false;
    }
    unsigned char diff = 0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        diff = static_cast<unsigned char>(diff | (static_cast<unsigned char>(left[index]) ^
                                                  static_cast<unsigned char>(right[index])));
    }
    return diff == 0;
}

PackageStatus sha256_file(const std::filesystem::path& path, std::uint64_t expected_size,
                          std::string& hex) {
    const PackageStatus file_status = require_regular_unique_file(path);
    if (!file_status.ok) {
        return file_status;
    }
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size != expected_size) {
        return fail(PackageError::kHashMismatch, "size mismatch");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return fail(PackageError::kHashMismatch, "file could not be read");
    }
    Sha256State state;
    std::array<std::uint8_t, 4096> buffer{};
    std::uint64_t total = 0;
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(buffer.size()));
        const auto got = static_cast<std::size_t>(input.gcount());
        if (got == 0) {
            break;
        }
        total += got;
        if (total > expected_size || !absorb(state, buffer.data(), got)) {
            return fail(PackageError::kHashMismatch, "hash overflow");
        }
    }
    if (total != expected_size) {
        return fail(PackageError::kHashMismatch, "size mismatch");
    }
    std::uint8_t raw[32]{};
    if (!finish_sha256(state, raw)) {
        return fail(PackageError::kHashMismatch, "hash failed");
    }
    hex.assign(64, '0');
    for (int index = 0; index < 32; ++index) {
        hex[static_cast<std::size_t>(index) * 2U] = kHex[raw[index] >> 4U];
        hex[static_cast<std::size_t>(index) * 2U + 1U] = kHex[raw[index] & 0x0fU];
    }
    return ok_status();
}

bool is_canonical_absolute_path(const std::filesystem::path& path) {
    if (!path.is_absolute() || path.empty()) {
        return false;
    }
    const std::string generic = path.generic_string();
    if (generic.find('\\') != std::string::npos || generic.find("%") != std::string::npos) {
        return false;
    }
    if (generic.find("/./") != std::string::npos || generic.find("/../") != std::string::npos ||
        generic.ends_with("/.") || generic.ends_with("/..")) {
        return false;
    }
    std::error_code error;
    const auto canonical = std::filesystem::weakly_canonical(path, error);
    if (error) {
        return false;
    }
    return canonical.lexically_normal().generic_string() ==
           path.lexically_normal().generic_string();
}

PackageStatus reject_symlink_ancestry(const std::filesystem::path& path) {
    if (!path.is_absolute()) {
        return fail(PackageError::kPathSecurity, "path is not absolute");
    }
    std::filesystem::path current;
    for (const auto& part : path) {
        if (current.empty()) {
            current = part;
        } else {
            current /= part;
        }
        std::error_code error;
        const auto status = std::filesystem::symlink_status(current, error);
        if (error) {
            if (current == path) {
                return fail(PackageError::kPathSecurity, "path status unavailable");
            }
            continue;
        }
        if (status.type() == std::filesystem::file_type::symlink) {
            return fail(PackageError::kPathSecurity, "symlink ancestor");
        }
    }
    return ok_status();
}

PackageStatus parse_llama_cpp_provenance(std::string_view text, LlamaCppProvenance& out) {
    if (text.size() > kGenerativeMaxProvenanceBytes) {
        return fail(PackageError::kManifest, "document exceeds maximum size");
    }
    if (contains_nul_or_cr(text) || !is_valid_utf8(text)) {
        return fail(PackageError::kManifest, "provenance encoding rejected");
    }
    if (text != kLlamaCppProvenanceExact) {
        return fail(PackageError::kManifest, "provenance format rejected");
    }
    LlamaCppProvenance provenance;
    provenance.schema_version = "1";
    provenance.project = std::string{kLlamaCppProject};
    provenance.tag = std::string{kLlamaCppTag};
    provenance.commit = std::string{kLlamaCppCommit};
    provenance.source_archive_sha256 = std::string{kLlamaCppArchiveSha256};
    provenance.build_type = "Release";
    provenance.architecture = "arm64";
    provenance.backend = "cpu";
    provenance.metal = "off";
    provenance.server = "off";
    out = std::move(provenance);
    return ok_status();
}

PackageStatus load_llama_cpp_provenance_file(const std::filesystem::path& path,
                                             LlamaCppProvenance& out) {
    if (!is_canonical_absolute_path(path)) {
        return fail(PackageError::kPathSecurity, "provenance path is not canonical");
    }
    PackageStatus ancestry = reject_symlink_ancestry(path);
    if (!ancestry.ok) {
        return ancestry;
    }
    ancestry = require_regular_unique_file(path);
    if (!ancestry.ok) {
        return ancestry;
    }
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > kGenerativeMaxProvenanceBytes) {
        return fail(PackageError::kManifest, "provenance file too large");
    }
    std::string text;
    if (!read_regular_file(path, kGenerativeMaxProvenanceBytes, text)) {
        return fail(PackageError::kManifest, "provenance file could not be read");
    }
    return parse_llama_cpp_provenance(text, out);
}

PackageStatus parse_model_package_manifest(std::string_view text, ModelPackageManifest& out) {
    static const std::vector<std::string_view> keys{
        "schema_version",  "model_id",        "source_repository",  "source_revision",
        "runtime_project", "runtime_tag",     "runtime_commit",     "gguf_file",
        "gguf_sha256",     "gguf_size_bytes", "gguf_architecture",  "quantization",
        "license_id",      "context_limit",   "chat_template_file", "chat_template_sha256",
        "thinking_mode",   "license_file",    "license_sha256",     "notice_file",
        "notice_sha256",
    };
    std::map<std::string, std::string> values;
    const PackageStatus parsed =
        parse_key_values(text, kGenerativeMaxManifestBytes, keys, false, values);
    if (!parsed.ok) {
        return parsed;
    }
    ModelPackageManifest manifest;
    manifest.schema_version = values["schema_version"];
    manifest.model_id = values["model_id"];
    manifest.source_repository = values["source_repository"];
    manifest.source_revision = values["source_revision"];
    manifest.runtime_project = values["runtime_project"];
    manifest.runtime_tag = values["runtime_tag"];
    manifest.runtime_commit = values["runtime_commit"];
    manifest.gguf_file = values["gguf_file"];
    manifest.gguf_sha256 = values["gguf_sha256"];
    manifest.gguf_architecture = values["gguf_architecture"];
    manifest.quantization = values["quantization"];
    manifest.license_id = values["license_id"];
    manifest.chat_template_file = values["chat_template_file"];
    manifest.chat_template_sha256 = values["chat_template_sha256"];
    manifest.thinking_mode = values["thinking_mode"];
    manifest.license_file = values["license_file"];
    manifest.license_sha256 = values["license_sha256"];
    manifest.notice_file = values["notice_file"];
    manifest.notice_sha256 = values["notice_sha256"];
    if (manifest.schema_version != "1") {
        return fail(PackageError::kManifest, "unsupported schema");
    }
    if (manifest.runtime_project != kLlamaCppProject || manifest.runtime_tag != kLlamaCppTag ||
        manifest.runtime_commit != kLlamaCppCommit) {
        return fail(PackageError::kManifest, "wrong runtime pin");
    }
    if (!parse_u64(values["gguf_size_bytes"], manifest.gguf_size_bytes) ||
        manifest.gguf_size_bytes == 0 || manifest.gguf_size_bytes > kGenerativeMaxGgufBytes) {
        return fail(PackageError::kManifest, "invalid GGUF size");
    }
    if (!parse_u32(values["context_limit"], manifest.context_limit) ||
        manifest.context_limit == 0 || manifest.context_limit > kGenerativeMaxContext) {
        return fail(PackageError::kManifest, "invalid context limit");
    }
    if (!is_hex_sha256(manifest.gguf_sha256) || !is_hex_sha256(manifest.chat_template_sha256) ||
        !is_hex_sha256(manifest.license_sha256)) {
        return fail(PackageError::kManifest, "invalid SHA-256");
    }
    if (!is_safe_file_name(manifest.gguf_file) || !manifest.gguf_file.ends_with(".gguf") ||
        !is_safe_file_name(manifest.chat_template_file) ||
        !is_safe_file_name(manifest.license_file)) {
        return fail(PackageError::kManifest, "unsafe file name");
    }
    if (manifest.notice_file == "none") {
        if (manifest.notice_sha256 != "none") {
            return fail(PackageError::kManifest, "NOTICE hash must be none");
        }
    } else if (!is_safe_file_name(manifest.notice_file) || !is_hex_sha256(manifest.notice_sha256)) {
        return fail(PackageError::kManifest, "invalid NOTICE file or hash");
    }
    if (!is_ident(manifest.model_id) || !is_ident(manifest.source_repository) ||
        !is_ident(manifest.source_revision) || !is_ident(manifest.gguf_architecture) ||
        !is_ident(manifest.quantization) || !is_ident(manifest.license_id)) {
        return fail(PackageError::kManifest, "invalid identifier");
    }
    if (manifest.thinking_mode != "disabled") {
        return fail(PackageError::kUnsupportedModel, "thinking mode must be disabled");
    }
    if (manifest.gguf_architecture != "qwen3" && manifest.gguf_architecture != "llama") {
        return fail(PackageError::kUnsupportedArchitecture, "unsupported GGUF architecture");
    }
    if (manifest.quantization != "Q4_K_M" && manifest.quantization != "Q5_K_M" &&
        manifest.quantization != "Q8_0" && manifest.quantization != "F16" &&
        manifest.quantization != "F32") {
        return fail(PackageError::kUnsupportedQuantization, "unsupported quantization");
    }
    const std::array<std::string_view, 5> names{
        manifest.gguf_file,
        manifest.chat_template_file,
        manifest.license_file,
        manifest.notice_file == "none" ? std::string_view{}
                                       : std::string_view{manifest.notice_file},
        "manifest.txt",
    };
    for (std::size_t left = 0; left < names.size(); ++left) {
        if (names[left].empty()) {
            continue;
        }
        for (std::size_t right = left + 1; right < names.size(); ++right) {
            if (!names[right].empty() && names[left] == names[right]) {
                return fail(PackageError::kManifest, "duplicate package file name");
            }
        }
    }
    out = std::move(manifest);
    return ok_status();
}

PackageStatus apply_frozen_template(std::string_view tmpl, std::string_view user_prompt,
                                    std::string& rendered) {
    if (tmpl.size() > kGenerativeMaxTemplateBytes) {
        return fail(PackageError::kManifest, "template too large");
    }
    if (contains_nul_or_cr(tmpl) || !is_valid_utf8(tmpl)) {
        return fail(PackageError::kManifest, "invalid template encoding");
    }
    if (tmpl.find("{{") != std::string_view::npos || tmpl.find("}}") != std::string_view::npos ||
        tmpl.find("{%") != std::string_view::npos || tmpl.find("%}") != std::string_view::npos ||
        tmpl.find("${") != std::string_view::npos || tmpl.find("$") != std::string_view::npos ||
        tmpl.find("`") != std::string_view::npos || tmpl.find("http:") != std::string_view::npos ||
        tmpl.find("https:") != std::string_view::npos ||
        tmpl.find("../") != std::string_view::npos || tmpl.find("..\\") != std::string_view::npos) {
        return fail(PackageError::kManifest, "template contains forbidden syntax");
    }
    const std::string_view placeholder = "{user}";
    const std::size_t pos = tmpl.find(placeholder);
    if (pos == std::string_view::npos) {
        return fail(PackageError::kManifest, "template missing {user}");
    }
    if (tmpl.find(placeholder, pos + placeholder.size()) != std::string_view::npos) {
        return fail(PackageError::kManifest, "repeated {user}");
    }
    const auto prefix = tmpl.substr(0, pos);
    const auto suffix = tmpl.substr(pos + placeholder.size());
    if (prefix.find('{') != std::string_view::npos || prefix.find('}') != std::string_view::npos ||
        suffix.find('{') != std::string_view::npos || suffix.find('}') != std::string_view::npos) {
        return fail(PackageError::kManifest, "unknown placeholder");
    }
    if (!user_prompt_is_usable(user_prompt)) {
        return fail(PackageError::kManifest, "empty or invalid user prompt");
    }
    if (prefix.size() > kGenerativeMaxExpandedPromptBytes ||
        user_prompt.size() > kGenerativeMaxExpandedPromptBytes - prefix.size()) {
        return fail(PackageError::kManifest, "expanded prompt too large");
    }
    const std::size_t after_user = prefix.size() + user_prompt.size();
    if (suffix.size() > kGenerativeMaxExpandedPromptBytes - after_user) {
        return fail(PackageError::kManifest, "expanded prompt too large");
    }
    std::string built;
    built.append(prefix);
    built.append(user_prompt);
    built.append(suffix);
    rendered = std::move(built);
    return ok_status();
}

PackageLoadResult load_model_package(const std::filesystem::path& package_directory) {
    PackageLoadResult result;
    if (!is_canonical_absolute_path(package_directory)) {
        result.status = fail(PackageError::kPathSecurity, "package path is not canonical");
        return result;
    }
    result.status = reject_symlink_ancestry(package_directory);
    if (!result.status.ok) {
        return result;
    }
    std::error_code error;
    const auto root_status = std::filesystem::symlink_status(package_directory, error);
    if (error || root_status.type() != std::filesystem::file_type::directory) {
        result.status = fail(PackageError::kPathSecurity, "package root is not a directory");
        return result;
    }
    if (root_status.type() == std::filesystem::file_type::symlink) {
        result.status = fail(PackageError::kPathSecurity, "symlink package root");
        return result;
    }
    const auto manifest_path = package_directory / "manifest.txt";
    result.status = require_regular_unique_file(manifest_path);
    if (!result.status.ok) {
        return result;
    }
    std::string manifest_text;
    if (!read_regular_file(manifest_path, kGenerativeMaxManifestBytes, manifest_text)) {
        result.status = fail(PackageError::kManifest, "manifest could not be read");
        return result;
    }
    result.status = parse_model_package_manifest(manifest_text, result.package.manifest);
    if (!result.status.ok) {
        return result;
    }
    const auto& manifest = result.package.manifest;
    std::vector<std::filesystem::path> allowed{
        manifest_path,
        package_directory / manifest.gguf_file,
        package_directory / manifest.chat_template_file,
        package_directory / manifest.license_file,
    };
    if (manifest.notice_file != "none") {
        allowed.push_back(package_directory / manifest.notice_file);
    }
    for (const auto& path : allowed) {
        if (!is_under_root(package_directory, path) || path.parent_path() != package_directory) {
            result.status = fail(PackageError::kPathSecurity, "path traversal");
            return result;
        }
        result.status = reject_symlink_ancestry(path);
        if (!result.status.ok) {
            return result;
        }
        result.status = require_regular_unique_file(path);
        if (!result.status.ok) {
            return result;
        }
    }
    std::size_t entries = 0;
    for (const auto& entry : std::filesystem::directory_iterator(
             package_directory, std::filesystem::directory_options::skip_permission_denied)) {
        ++entries;
        const auto name = entry.path().filename().string();
        if (name.starts_with('.')) {
            result.status = fail(PackageError::kPathSecurity, "hidden file");
            return result;
        }
        std::error_code entry_error;
        const auto type = entry.symlink_status(entry_error).type();
        if (entry_error) {
            result.status = fail(PackageError::kPathSecurity, "entry status unavailable");
            return result;
        }
        if (type == std::filesystem::file_type::directory) {
            result.status = fail(PackageError::kPathSecurity, "unexpected directory");
            return result;
        }
        if (type == std::filesystem::file_type::symlink) {
            result.status = fail(PackageError::kPathSecurity, "symlink file");
            return result;
        }
        if (type != std::filesystem::file_type::regular) {
            result.status = fail(PackageError::kPathSecurity, "special file");
            return result;
        }
        bool allowed_name = name == "manifest.txt" || name == manifest.gguf_file ||
                            name == manifest.chat_template_file || name == manifest.license_file ||
                            (manifest.notice_file != "none" && name == manifest.notice_file);
        if (!allowed_name) {
            result.status = fail(PackageError::kPathSecurity, "unexpected file");
            return result;
        }
    }
    if (entries != allowed.size()) {
        result.status = fail(PackageError::kPathSecurity, "package file count mismatch");
        return result;
    }
    std::string gguf_hex;
    result.status = sha256_file(allowed[1], manifest.gguf_size_bytes, gguf_hex);
    if (!result.status.ok) {
        return result;
    }
    if (!hashes_equal(gguf_hex, manifest.gguf_sha256)) {
        result.status = fail(PackageError::kHashMismatch, "GGUF hash mismatch");
        return result;
    }
    std::error_code size_error;
    const auto template_size = std::filesystem::file_size(allowed[2], size_error);
    if (size_error || template_size == 0 || template_size > kGenerativeMaxTemplateBytes) {
        result.status = fail(PackageError::kHashMismatch, "template size rejected");
        return result;
    }
    std::string template_hex;
    result.status = sha256_file(allowed[2], template_size, template_hex);
    if (!result.status.ok) {
        return result;
    }
    if (!hashes_equal(template_hex, manifest.chat_template_sha256)) {
        result.status = fail(PackageError::kHashMismatch, "template hash mismatch");
        return result;
    }
    if (!read_regular_file(allowed[2], kGenerativeMaxTemplateBytes, result.package.template_text)) {
        result.status = fail(PackageError::kManifest, "template could not be read");
        return result;
    }
    std::error_code license_size_error;
    const auto license_size = std::filesystem::file_size(allowed[3], license_size_error);
    if (license_size_error || license_size == 0 || license_size > kGenerativeMaxLicenseBytes) {
        result.status = fail(PackageError::kHashMismatch, "license size rejected");
        return result;
    }
    std::string license_hex;
    result.status = sha256_file(allowed[3], license_size, license_hex);
    if (!result.status.ok) {
        return result;
    }
    if (!hashes_equal(license_hex, manifest.license_sha256)) {
        result.status = fail(PackageError::kHashMismatch, "license hash mismatch");
        return result;
    }
    if (manifest.notice_file != "none") {
        std::error_code notice_size_error;
        const auto notice_size = std::filesystem::file_size(allowed[4], notice_size_error);
        if (notice_size_error || notice_size == 0 || notice_size > kGenerativeMaxNoticeBytes) {
            result.status = fail(PackageError::kHashMismatch, "NOTICE size rejected");
            return result;
        }
        std::string notice_hex;
        result.status = sha256_file(allowed[4], notice_size, notice_hex);
        if (!result.status.ok) {
            return result;
        }
        if (!hashes_equal(notice_hex, manifest.notice_sha256)) {
            result.status = fail(PackageError::kHashMismatch, "NOTICE hash mismatch");
            return result;
        }
    }
    std::string rendered;
    result.status = apply_frozen_template(result.package.template_text, "probe", rendered);
    if (!result.status.ok) {
        return result;
    }
    result.package.package_root = package_directory;
    result.package.manifest_path = allowed[0];
    result.package.gguf_path = allowed[1];
    result.package.template_path = allowed[2];
    result.package.license_path = allowed[3];
    if (manifest.notice_file != "none") {
        result.package.notice_path = allowed[4];
    }
    result.status = ok_status();
    return result;
}

PackageStatus load_untrusted_prompt_file(const std::filesystem::path& path, std::string& out) {
    if (!is_canonical_absolute_path(path)) {
        return fail(PackageError::kPathSecurity, "prompt path is not canonical");
    }
    PackageStatus status = reject_symlink_ancestry(path);
    if (!status.ok) {
        return status;
    }
    status = require_regular_unique_file(path);
    if (!status.ok) {
        return status;
    }
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size == 0 || size > kGenerativeMaxPromptBytes) {
        return fail(PackageError::kManifest, "prompt file size rejected");
    }
    std::string text;
    if (!read_regular_file(path, kGenerativeMaxPromptBytes, text)) {
        return fail(PackageError::kManifest, "prompt file could not be read");
    }
    if (!user_prompt_is_usable(text)) {
        return fail(PackageError::kManifest, "empty or invalid prompt file");
    }
    out = std::move(text);
    return ok_status();
}

} // namespace inferedge
