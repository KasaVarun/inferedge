#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace inferedge {

inline constexpr std::string_view kLlamaCppTag = "v0.7.0";
inline constexpr std::string_view kLlamaCppCommit = "620d2225cc436ab0b001e3200e590e3b286ef9ab";
inline constexpr std::string_view kLlamaCppArchiveSha256 =
    "55d67acf3ad2ad747a9bbeb75f9f0e173dbcc0d1753a471038a90c819421c5e1";
inline constexpr std::string_view kLlamaCppProject = "llama.cpp";
inline constexpr std::uint32_t kGenerativeMaxContext = 4096;
inline constexpr std::uint64_t kGenerativeMaxGgufBytes = 2147483648ULL;
inline constexpr std::size_t kGenerativeMaxManifestBytes = 8192;
inline constexpr std::size_t kGenerativeMaxProvenanceBytes = 1024;
inline constexpr std::size_t kGenerativeMaxTemplateBytes = 65536;
inline constexpr std::size_t kGenerativeMaxPromptBytes = 65536;
inline constexpr std::size_t kGenerativeMaxExpandedPromptBytes = 65536;
inline constexpr std::uint64_t kGenerativeMaxLicenseBytes = 262144ULL;
inline constexpr std::uint64_t kGenerativeMaxNoticeBytes = 262144ULL;
inline constexpr std::size_t kGenerativeMaxTokenPieceBytes = 4096;
inline constexpr std::size_t kGenerativeMaxGeneratedBytes =
    static_cast<std::size_t>(kGenerativeMaxContext) * kGenerativeMaxTokenPieceBytes;
inline constexpr std::size_t kLlamaCppProvenanceExactBytes = 252;

inline constexpr std::string_view kLlamaCppProvenanceExact =
    "schema_version=1\n"
    "project=llama.cpp\n"
    "tag=v0.7.0\n"
    "commit=620d2225cc436ab0b001e3200e590e3b286ef9ab\n"
    "source_archive_sha256=55d67acf3ad2ad747a9bbeb75f9f0e173dbcc0d1753a471038a90c819421c5e1\n"
    "build_type=Release\n"
    "architecture=arm64\n"
    "backend=cpu\n"
    "metal=off\n"
    "server=off\n";
static_assert(kLlamaCppProvenanceExact.size() == kLlamaCppProvenanceExactBytes);

enum class PackageError {
    kNone,
    kManifest,
    kPathSecurity,
    kHashMismatch,
    kUnsupportedModel,
    kUnsupportedArchitecture,
    kUnsupportedQuantization,
    kLicense,
    kInternal,
};

struct PackageStatus {
    bool ok{false};
    PackageError error{PackageError::kNone};
    std::string message;
};

struct LlamaCppProvenance {
    std::string schema_version;
    std::string project;
    std::string tag;
    std::string commit;
    std::string source_archive_sha256;
    std::string build_type;
    std::string architecture;
    std::string backend;
    std::string metal;
    std::string server;
};

struct ModelPackageManifest {
    std::string schema_version;
    std::string model_id;
    std::string source_repository;
    std::string source_revision;
    std::string runtime_project;
    std::string runtime_tag;
    std::string runtime_commit;
    std::string gguf_file;
    std::string gguf_sha256;
    std::uint64_t gguf_size_bytes{0};
    std::string gguf_architecture;
    std::string quantization;
    std::string license_id;
    std::uint32_t context_limit{0};
    std::string chat_template_file;
    std::string chat_template_sha256;
    std::string thinking_mode;
    std::string license_file;
    std::string license_sha256;
    std::string notice_file;
    std::string notice_sha256;
};

struct ValidatedModelPackage {
    ModelPackageManifest manifest;
    std::filesystem::path package_root;
    std::filesystem::path manifest_path;
    std::filesystem::path gguf_path;
    std::filesystem::path template_path;
    std::filesystem::path license_path;
    std::filesystem::path notice_path;
    std::string template_text;
};

struct PackageLoadResult {
    PackageStatus status;
    ValidatedModelPackage package;
};

PackageStatus parse_llama_cpp_provenance(std::string_view text, LlamaCppProvenance& out);
PackageStatus load_llama_cpp_provenance_file(const std::filesystem::path& path,
                                             LlamaCppProvenance& out);

PackageStatus parse_model_package_manifest(std::string_view text, ModelPackageManifest& out);
PackageLoadResult load_model_package(const std::filesystem::path& package_directory);

std::string sha256_hex(const std::uint8_t* data, std::size_t length);
PackageStatus sha256_file(const std::filesystem::path& path, std::uint64_t expected_size,
                          std::string& hex);
bool hashes_equal(std::string_view left, std::string_view right) noexcept;

bool is_strict_utf8(std::string_view text) noexcept;
std::size_t complete_utf8_prefix_length(std::string_view text) noexcept;
// Rejects empty text and ASCII-only space/tab/LF. Other Unicode whitespace
// does not make a prompt empty.
bool user_prompt_is_usable(std::string_view prompt) noexcept;
bool path_is_under_root(const std::filesystem::path& root, const std::filesystem::path& child);

PackageStatus apply_frozen_template(std::string_view tmpl, std::string_view user_prompt,
                                    std::string& rendered);
PackageStatus load_untrusted_prompt_file(const std::filesystem::path& path, std::string& out);
PackageStatus resize_token_piece_buffer(std::string& buffer, int32_t required, std::size_t maximum);

bool is_canonical_absolute_path(const std::filesystem::path& path);
PackageStatus reject_symlink_ancestry(const std::filesystem::path& path);

} // namespace inferedge
