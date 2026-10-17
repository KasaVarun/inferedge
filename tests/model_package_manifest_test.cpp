#include "inferedge/model_package_manifest.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

std::filesystem::path canonical_temp() {
    return std::filesystem::weakly_canonical(std::filesystem::temp_directory_path());
}

std::filesystem::path unique_dir(const std::string& tag) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto path = canonical_temp() / ("inferedge-9b1-" + tag + "-" + std::to_string(::getpid()) +
                                    "-" + std::to_string(now));
    std::filesystem::create_directories(path);
    return std::filesystem::weakly_canonical(path);
}

void write_bytes(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string valid_provenance() {
    return "schema_version=1\n"
           "project=llama.cpp\n"
           "tag=v0.7.0\n"
           "commit=620d2225cc436ab0b001e3200e590e3b286ef9ab\n"
           "source_archive_sha256="
           "55d67acf3ad2ad747a9bbeb75f9f0e173dbcc0d1753a471038a90c819421c5e1\n"
           "build_type=Release\n"
           "architecture=arm64\n"
           "backend=cpu\n"
           "metal=off\n"
           "server=off\n";
}

std::string hex_of(std::string_view bytes) {
    return inferedge::sha256_hex(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
}

std::string valid_manifest(std::string_view gguf_hex, std::uint64_t gguf_size,
                           std::string_view template_hex, std::string_view extra = {}) {
    std::string text;
    text += "schema_version=1\n";
    text += "model_id=qwen3-test\n";
    text += "source_repository=Qwen/Qwen3-test\n";
    text += "source_revision=deadbeef\n";
    text += "runtime_project=llama.cpp\n";
    text += "runtime_tag=v0.7.0\n";
    text += "runtime_commit=620d2225cc436ab0b001e3200e590e3b286ef9ab\n";
    text += "gguf_file=model.gguf\n";
    text += "gguf_sha256=";
    text += gguf_hex;
    text += "\n";
    text += "gguf_size_bytes=";
    text += std::to_string(gguf_size);
    text += "\n";
    text += "gguf_architecture=qwen3\n";
    text += "quantization=Q4_K_M\n";
    text += "license_id=apache-2.0\n";
    text += "context_limit=4096\n";
    text += "chat_template_file=qwen3-non-thinking-v1.txt\n";
    text += "chat_template_sha256=";
    text += template_hex;
    text += "\n";
    text += "thinking_mode=disabled\n";
    text += "license_file=LICENSE.txt\n";
    text += "license_sha256=";
    text += hex_of("Apache-2.0");
    text += "\n";
    text += "notice_file=none\n";
    text += "notice_sha256=none\n";
    text += extra;
    return text;
}

struct PackageFiles {
    std::filesystem::path root;
    std::string gguf{"GGUF-not-a-model"};
    std::string tmpl{"User:{user}"};
};

void write_valid_package(PackageFiles& pkg) {
    const auto gguf_hex = hex_of(pkg.gguf);
    const auto tmpl_hex = hex_of(pkg.tmpl);
    write_bytes(pkg.root / "manifest.txt", valid_manifest(gguf_hex, pkg.gguf.size(), tmpl_hex));
    write_bytes(pkg.root / "model.gguf", pkg.gguf);
    write_bytes(pkg.root / "qwen3-non-thinking-v1.txt", pkg.tmpl);
    write_bytes(pkg.root / "LICENSE.txt", "Apache-2.0");
}

} // namespace

TEST(ModelPackageManifestTest, ExactProvenanceHasFixedByteLength) {
    EXPECT_EQ(inferedge::kLlamaCppProvenanceExact.size(), inferedge::kLlamaCppProvenanceExactBytes);
    EXPECT_EQ(inferedge::kLlamaCppProvenanceExact.find('\r'), std::string_view::npos);
}

TEST(ModelPackageManifestTest, ParsesValidProvenance) {
    inferedge::LlamaCppProvenance provenance;
    const inferedge::PackageStatus status =
        inferedge::parse_llama_cpp_provenance(valid_provenance(), provenance);
    ASSERT_TRUE(status.ok);
    EXPECT_EQ(provenance.tag, "v0.7.0");
    EXPECT_EQ(provenance.commit, "620d2225cc436ab0b001e3200e590e3b286ef9ab");
}

TEST(ModelPackageManifestTest, RejectsDuplicateProvenanceKeys) {
    inferedge::LlamaCppProvenance provenance;
    auto text = valid_provenance();
    text += "metal=off\n";
    EXPECT_FALSE(inferedge::parse_llama_cpp_provenance(text, provenance).ok);
}

TEST(ModelPackageManifestTest, RejectsUnknownProvenanceKeys) {
    inferedge::LlamaCppProvenance provenance;
    auto text = valid_provenance();
    text += "extra=1\n";
    EXPECT_FALSE(inferedge::parse_llama_cpp_provenance(text, provenance).ok);
}

TEST(ModelPackageManifestTest, RejectsWrongProvenanceValues) {
    inferedge::LlamaCppProvenance provenance;
    auto text = valid_provenance();
    const auto replaced = text.replace(text.find("cpu"), 3, "gpu");
    (void)replaced;
    EXPECT_FALSE(inferedge::parse_llama_cpp_provenance(text, provenance).ok);
}

TEST(ModelPackageManifestTest, RejectsProvenanceWhitespaceAndCr) {
    inferedge::LlamaCppProvenance provenance;
    EXPECT_FALSE(inferedge::parse_llama_cpp_provenance("schema_version =1\n", provenance).ok);
    const std::string cr = "schema_version=1\r\nproject=llama.cpp\n";
    EXPECT_FALSE(inferedge::parse_llama_cpp_provenance(cr, provenance).ok);
}

TEST(ModelPackageManifestTest, LoadsCanonicalProvenanceFile) {
    const auto dir = unique_dir("prov");
    const auto path = dir / "llama-cpp-provenance.txt";
    write_bytes(path, valid_provenance());
    inferedge::LlamaCppProvenance provenance;
    ASSERT_TRUE(inferedge::load_llama_cpp_provenance_file(path, provenance).ok);
    std::filesystem::remove_all(dir);
}

TEST(ModelPackageManifestTest, RejectsMissingProvenanceFile) {
    inferedge::LlamaCppProvenance provenance;
    const auto path = unique_dir("missing") / "llama-cpp-provenance.txt";
    EXPECT_FALSE(inferedge::load_llama_cpp_provenance_file(path, provenance).ok);
    std::filesystem::remove_all(path.parent_path());
}

TEST(ModelPackageManifestTest, RejectsSymlinkProvenanceFile) {
    const auto dir = unique_dir("prov-link");
    const auto real = dir / "real.txt";
    const auto link = dir / "llama-cpp-provenance.txt";
    write_bytes(real, valid_provenance());
    std::filesystem::create_symlink(real, link);
    inferedge::LlamaCppProvenance provenance;
    EXPECT_FALSE(inferedge::load_llama_cpp_provenance_file(link, provenance).ok);
    std::filesystem::remove_all(dir);
}

TEST(ModelPackageManifestTest, ParsesValidManifest) {
    inferedge::ModelPackageManifest manifest;
    const auto text = valid_manifest(hex_of("abc"), 3, hex_of("User:{user}"));
    ASSERT_TRUE(inferedge::parse_model_package_manifest(text, manifest).ok);
    EXPECT_EQ(manifest.model_id, "qwen3-test");
    EXPECT_EQ(manifest.notice_file, "none");
    EXPECT_EQ(manifest.context_limit, 4096U);
}

TEST(ModelPackageManifestTest, RejectsMissingManifestKeys) {
    inferedge::ModelPackageManifest manifest;
    EXPECT_FALSE(inferedge::parse_model_package_manifest("schema_version=1\n", manifest).ok);
}

TEST(ModelPackageManifestTest, RejectsDuplicateAndUnknownManifestKeys) {
    inferedge::ModelPackageManifest manifest;
    auto text = valid_manifest(hex_of("abc"), 3, hex_of("t"));
    EXPECT_FALSE(inferedge::parse_model_package_manifest(text + "model_id=again\n", manifest).ok);
    EXPECT_FALSE(inferedge::parse_model_package_manifest(text + "extra=1\n", manifest).ok);
}

TEST(ModelPackageManifestTest, RejectsInvalidUtf8AndNul) {
    inferedge::ModelPackageManifest manifest;
    std::string bad = valid_manifest(hex_of("abc"), 3, hex_of("t"));
    bad.push_back(static_cast<char>(0xff));
    EXPECT_FALSE(inferedge::parse_model_package_manifest(bad, manifest).ok);
    std::string nul = "schema_version=1";
    nul.push_back('\0');
    EXPECT_FALSE(inferedge::parse_model_package_manifest(nul, manifest).ok);
}

TEST(ModelPackageManifestTest, RejectsNumericOverflowAndInvalidSha) {
    inferedge::ModelPackageManifest manifest;
    auto overflow = valid_manifest(hex_of("abc"), 3, hex_of("t"));
    overflow.replace(overflow.find("gguf_size_bytes=3"), std::string("gguf_size_bytes=3").size(),
                     "gguf_size_bytes=18446744073709551616");
    EXPECT_FALSE(inferedge::parse_model_package_manifest(overflow, manifest).ok);
    auto sha = valid_manifest("zzzz", 3, hex_of("t"));
    EXPECT_FALSE(inferedge::parse_model_package_manifest(sha, manifest).ok);
}

TEST(ModelPackageManifestTest, RejectsWrongRuntimePinAndContext) {
    inferedge::ModelPackageManifest manifest;
    auto pin = valid_manifest(hex_of("abc"), 3, hex_of("t"));
    pin.replace(pin.find("v0.7.0"), 6, "v0.6.0");
    EXPECT_FALSE(inferedge::parse_model_package_manifest(pin, manifest).ok);
    auto ctx = valid_manifest(hex_of("abc"), 3, hex_of("t"));
    ctx.replace(ctx.find("context_limit=4096"), std::string("context_limit=4096").size(),
                "context_limit=8192");
    EXPECT_FALSE(inferedge::parse_model_package_manifest(ctx, manifest).ok);
}

TEST(ModelPackageManifestTest, RejectsUnsafeFileNames) {
    inferedge::ModelPackageManifest manifest;
    auto text = valid_manifest(hex_of("abc"), 3, hex_of("t"));
    text.replace(text.find("model.gguf"), 10, "../x.gguf");
    EXPECT_FALSE(inferedge::parse_model_package_manifest(text, manifest).ok);
}

TEST(ModelPackageManifestTest, LoadsExactAllowlistPackage) {
    PackageFiles pkg;
    pkg.root = unique_dir("pkg-ok");
    write_valid_package(pkg);
    const inferedge::PackageLoadResult loaded = inferedge::load_model_package(pkg.root);
    ASSERT_TRUE(loaded.status.ok) << loaded.status.message;
    EXPECT_EQ(loaded.package.manifest.model_id, "qwen3-test");
    std::filesystem::remove_all(pkg.root);
}

TEST(ModelPackageManifestTest, RejectsRelativePackagePath) {
    const inferedge::PackageLoadResult loaded = inferedge::load_model_package("models/llm/local");
    EXPECT_FALSE(loaded.status.ok);
    EXPECT_EQ(loaded.status.error, inferedge::PackageError::kPathSecurity);
}

TEST(ModelPackageManifestTest, RejectsSymlinkPackageRoot) {
    PackageFiles pkg;
    pkg.root = unique_dir("pkg-real");
    write_valid_package(pkg);
    const auto parent = pkg.root.parent_path();
    const auto link = parent / (pkg.root.filename().string() + "-link");
    std::filesystem::create_directory_symlink(pkg.root, link);
    const auto loaded = inferedge::load_model_package(link);
    EXPECT_FALSE(loaded.status.ok);
    std::filesystem::remove_all(pkg.root);
    std::filesystem::remove_all(link);
}

TEST(ModelPackageManifestTest, RejectsSymlinkAncestor) {
    const auto base = unique_dir("ancestor");
    const auto real = base / "real";
    std::filesystem::create_directories(real);
    const auto link = base / "link";
    std::filesystem::create_directory_symlink(real, link);
    PackageFiles pkg;
    pkg.root = link / "pkg";
    std::filesystem::create_directories(pkg.root);
    write_valid_package(pkg);
    const auto loaded = inferedge::load_model_package(std::filesystem::weakly_canonical(pkg.root));
    EXPECT_FALSE(inferedge::load_model_package(pkg.root).status.ok);
    (void)loaded;
    std::filesystem::remove_all(base);
}

TEST(ModelPackageManifestTest, RejectsSymlinkLeafFiles) {
    const auto workspace = unique_dir("pkg-leaf-ws");
    PackageFiles pkg;
    pkg.root = workspace / "pkg";
    std::filesystem::create_directories(pkg.root);
    write_valid_package(pkg);
    std::filesystem::remove(pkg.root / "model.gguf");
    const auto outside = workspace / "outside.gguf";
    write_bytes(outside, pkg.gguf);
    std::filesystem::create_symlink(outside, pkg.root / "model.gguf");
    EXPECT_FALSE(inferedge::load_model_package(pkg.root).status.ok);
    std::filesystem::remove_all(workspace);
}

TEST(ModelPackageManifestTest, RejectsHardLinkedFilesWhenSupported) {
    const auto workspace = unique_dir("pkg-hard-ws");
    PackageFiles pkg;
    pkg.root = workspace / "pkg";
    std::filesystem::create_directories(pkg.root);
    write_valid_package(pkg);
    const auto extra = workspace / "hard-gguf";
    std::error_code error;
    std::filesystem::create_hard_link(pkg.root / "model.gguf", extra, error);
    if (error) {
        std::filesystem::remove_all(workspace);
        GTEST_SKIP() << "hard links unavailable";
    }
    EXPECT_FALSE(inferedge::load_model_package(pkg.root).status.ok);
    std::filesystem::remove_all(workspace);
}

TEST(ModelPackageManifestTest, RejectsUnexpectedFilesDirectoriesAndHidden) {
    PackageFiles pkg;
    pkg.root = unique_dir("pkg-extra");
    write_valid_package(pkg);
    write_bytes(pkg.root / "extra.txt", "no");
    EXPECT_FALSE(inferedge::load_model_package(pkg.root).status.ok);
    std::filesystem::remove(pkg.root / "extra.txt");
    std::filesystem::create_directory(pkg.root / "nested");
    EXPECT_FALSE(inferedge::load_model_package(pkg.root).status.ok);
    std::filesystem::remove(pkg.root / "nested");
    write_bytes(pkg.root / ".hidden", "no");
    EXPECT_FALSE(inferedge::load_model_package(pkg.root).status.ok);
    std::filesystem::remove_all(pkg.root);
}

TEST(ModelPackageManifestTest, RejectsSpecialFileWhenSafe) {
#ifdef __APPLE__
    PackageFiles pkg;
    pkg.root = unique_dir("pkg-fifo");
    write_valid_package(pkg);
    const auto fifo = pkg.root / "extra.fifo";
    ASSERT_EQ(::mkfifo(fifo.c_str(), 0600), 0);
    EXPECT_FALSE(inferedge::load_model_package(pkg.root).status.ok);
    std::filesystem::remove_all(pkg.root);
#else
    GTEST_SKIP();
#endif
}

TEST(ModelPackageManifestTest, RejectsSizeAndHashMismatch) {
    PackageFiles pkg;
    pkg.root = unique_dir("pkg-hash");
    write_valid_package(pkg);
    auto manifest = valid_manifest(hex_of(pkg.gguf), pkg.gguf.size() + 1, hex_of(pkg.tmpl));
    write_bytes(pkg.root / "manifest.txt", manifest);
    EXPECT_FALSE(inferedge::load_model_package(pkg.root).status.ok);
    write_bytes(pkg.root / "manifest.txt",
                valid_manifest(std::string(64, 'a'), pkg.gguf.size(), hex_of(pkg.tmpl)));
    EXPECT_EQ(inferedge::load_model_package(pkg.root).status.error,
              inferedge::PackageError::kHashMismatch);
    write_bytes(pkg.root / "manifest.txt",
                valid_manifest(hex_of(pkg.gguf), pkg.gguf.size(), std::string(64, 'b')));
    EXPECT_EQ(inferedge::load_model_package(pkg.root).status.error,
              inferedge::PackageError::kHashMismatch);
    std::filesystem::remove_all(pkg.root);
}

TEST(ModelPackageManifestTest, RejectsPathTraversalNames) {
    inferedge::ModelPackageManifest manifest;
    auto text = valid_manifest(hex_of("abc"), 3, hex_of("t"));
    text.replace(text.find("LICENSE.txt"), 11, "LICENSE%2etxt");
    EXPECT_FALSE(inferedge::parse_model_package_manifest(text, manifest).ok);
}

TEST(ModelPackageManifestTest, FrozenTemplateAcceptsUserOnly) {
    std::string rendered;
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "hi", rendered).ok);
    EXPECT_EQ(rendered, "AhiB");
    EXPECT_FALSE(inferedge::apply_frozen_template("{{user}}", "hi", rendered).ok);
    EXPECT_FALSE(inferedge::apply_frozen_template("A{user}{extra}", "hi", rendered).ok);
    EXPECT_FALSE(inferedge::apply_frozen_template("${HOME}{user}", "hi", rendered).ok);
    std::string harmless;
    ASSERT_TRUE(
        inferedge::apply_frozen_template("Please include important notes:{user}", "hi", harmless)
            .ok);
    EXPECT_EQ(harmless, "Please include important notes:hi");
}

TEST(ModelPackageManifestTest, UserTextIsInertAfterTemplatePass) {
    std::string rendered;
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "x{y}z", rendered).ok);
    EXPECT_EQ(rendered, "Ax{y}zB");
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "{user}", rendered).ok);
    EXPECT_EQ(rendered, "A{user}B");
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "${HOME}", rendered).ok);
    EXPECT_EQ(rendered, "A${HOME}B");
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "http://example.com", rendered).ok);
    EXPECT_EQ(rendered, "Ahttp://example.comB");
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "../secret", rendered).ok);
    EXPECT_EQ(rendered, "A../secretB");
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "{{example}}", rendered).ok);
    EXPECT_EQ(rendered, "A{{example}}B");
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "{", rendered).ok);
    EXPECT_EQ(rendered, "A{B");
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "}", rendered).ok);
    EXPECT_EQ(rendered, "A}B");
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "{}", rendered).ok);
    EXPECT_EQ(rendered, "A{}B");
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "https://example.com", rendered).ok);
    EXPECT_EQ(rendered, "Ahttps://example.comB");
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", "a\\b%c", rendered).ok);
    EXPECT_EQ(rendered, "Aa\\b%cB");
    EXPECT_FALSE(inferedge::apply_frozen_template("AB", "hi", rendered).ok);
    EXPECT_FALSE(inferedge::apply_frozen_template("A{foo}B", "hi", rendered).ok);
    EXPECT_FALSE(inferedge::apply_frozen_template("{user}{user}", "hi", rendered).ok);
    EXPECT_FALSE(inferedge::apply_frozen_template("A{user}B", "", rendered).ok);
    EXPECT_FALSE(inferedge::apply_frozen_template("A{user}B", "   ", rendered).ok);
}

TEST(ModelPackageManifestTest, ExpandedPromptBound) {
    std::string rendered;
    const std::string almost(inferedge::kGenerativeMaxExpandedPromptBytes - 2, 'x');
    ASSERT_TRUE(inferedge::apply_frozen_template("A{user}B", almost, rendered).ok);
    EXPECT_EQ(rendered.size(), inferedge::kGenerativeMaxExpandedPromptBytes);
    const std::string over(inferedge::kGenerativeMaxExpandedPromptBytes - 1, 'x');
    EXPECT_FALSE(inferedge::apply_frozen_template("A{user}B", over, rendered).ok);
}

TEST(ModelPackageManifestTest, StrictUtf8RejectsOverlongSurrogateAndTruncated) {
    const char overlong[] = {'\xc0', '\x80'};
    EXPECT_FALSE(inferedge::is_strict_utf8(std::string_view{overlong, 2}));
    const char surrogate[] = {'\xed', '\xa0', '\x80'};
    EXPECT_FALSE(inferedge::is_strict_utf8(std::string_view{surrogate, 3}));
    const char high[] = {'\xf5', '\x80', '\x80', '\x80'};
    EXPECT_FALSE(inferedge::is_strict_utf8(std::string_view{high, 4}));
    const char truncated[] = {'\xe2', '\x82'};
    EXPECT_FALSE(inferedge::is_strict_utf8(std::string_view{truncated, 2}));
    EXPECT_EQ(inferedge::complete_utf8_prefix_length(std::string_view{truncated, 2}), 0U);
    EXPECT_TRUE(inferedge::is_strict_utf8("ok"));
    const char prefix_plus_tail[] = {'o', 'k', '\xe2', '\x82'};
    EXPECT_EQ(inferedge::complete_utf8_prefix_length(std::string_view{prefix_plus_tail, 4}), 2U);
    EXPECT_FALSE(inferedge::is_strict_utf8(std::string_view{prefix_plus_tail, 4}));
}

TEST(ModelPackageManifestTest, RejectsReorderedAndNewlineProvenance) {
    inferedge::LlamaCppProvenance provenance;
    auto exact = valid_provenance();
    ASSERT_TRUE(inferedge::parse_llama_cpp_provenance(exact, provenance).ok);
    auto missing_nl = exact;
    missing_nl.pop_back();
    EXPECT_FALSE(inferedge::parse_llama_cpp_provenance(missing_nl, provenance).ok);
    EXPECT_FALSE(inferedge::parse_llama_cpp_provenance(exact + "\n", provenance).ok);
    EXPECT_FALSE(inferedge::parse_llama_cpp_provenance(exact + "\nbackend=cpu\n", provenance).ok);
    auto reordered = exact;
    reordered.replace(reordered.find("project=llama.cpp\ntag=v0.7.0\n"),
                      std::string("project=llama.cpp\ntag=v0.7.0\n").size(),
                      "tag=v0.7.0\nproject=llama.cpp\n");
    EXPECT_FALSE(inferedge::parse_llama_cpp_provenance(reordered, provenance).ok);
    auto blank = exact;
    blank.insert(blank.find("project="), "\n");
    EXPECT_FALSE(inferedge::parse_llama_cpp_provenance(blank, provenance).ok);
    auto crlf = exact;
    crlf.replace(crlf.find("\n"), 1, "\r\n");
    EXPECT_FALSE(inferedge::parse_llama_cpp_provenance(crlf, provenance).ok);
}

TEST(ModelPackageManifestTest, SiblingPrefixIsNotContainment) {
    EXPECT_FALSE(inferedge::path_is_under_root("/tmp/model", "/tmp/model-evil/x.gguf"));
    EXPECT_TRUE(inferedge::path_is_under_root("/tmp/model", "/tmp/model/x.gguf"));
}

TEST(ModelPackageManifestTest, NoticeNoneRejectsFileNamedNone) {
    PackageFiles pkg;
    pkg.root = unique_dir("pkg-none");
    write_valid_package(pkg);
    write_bytes(pkg.root / "none", "unexpected");
    EXPECT_FALSE(inferedge::load_model_package(pkg.root).status.ok);
    std::filesystem::remove_all(pkg.root);
}

TEST(ModelPackageManifestTest, RejectsLicenseAndNoticeHashMismatch) {
    PackageFiles pkg;
    pkg.root = unique_dir("pkg-lic");
    write_valid_package(pkg);
    auto bad = valid_manifest(hex_of(pkg.gguf), pkg.gguf.size(), hex_of(pkg.tmpl));
    bad.replace(bad.find(hex_of("Apache-2.0")), 64, std::string(64, 'a'));
    write_bytes(pkg.root / "manifest.txt", bad);
    EXPECT_EQ(inferedge::load_model_package(pkg.root).status.error,
              inferedge::PackageError::kHashMismatch);
    inferedge::ModelPackageManifest manifest;
    auto none_hash = valid_manifest(hex_of("abc"), 3, hex_of("t"));
    none_hash.replace(
        none_hash.find("notice_sha256=none"), std::string("notice_sha256=none").size(),
        "notice_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    EXPECT_FALSE(inferedge::parse_model_package_manifest(none_hash, manifest).ok);
    auto notice_none = valid_manifest(hex_of("abc"), 3, hex_of("t"));
    notice_none.replace(notice_none.find("notice_file=none"),
                        std::string("notice_file=none").size(), "notice_file=NOTICE.txt");
    EXPECT_FALSE(inferedge::parse_model_package_manifest(notice_none, manifest).ok);
    PackageFiles notice_pkg;
    notice_pkg.root = unique_dir("pkg-notice");
    write_valid_package(notice_pkg);
    write_bytes(notice_pkg.root / "NOTICE.txt", "notice-body");
    auto notice_manifest =
        valid_manifest(hex_of(notice_pkg.gguf), notice_pkg.gguf.size(), hex_of(notice_pkg.tmpl));
    notice_manifest.replace(notice_manifest.find("notice_file=none"),
                            std::string("notice_file=none").size(), "notice_file=NOTICE.txt");
    notice_manifest.replace(notice_manifest.find("notice_sha256=none"),
                            std::string("notice_sha256=none").size(),
                            "notice_sha256=" + std::string(64, 'b'));
    write_bytes(notice_pkg.root / "manifest.txt", notice_manifest);
    EXPECT_EQ(inferedge::load_model_package(notice_pkg.root).status.error,
              inferedge::PackageError::kHashMismatch);
    std::filesystem::remove_all(pkg.root);
    std::filesystem::remove_all(notice_pkg.root);
}

TEST(ModelPackageManifestTest, TokenPieceBufferResize) {
    std::string buffer;
    ASSERT_TRUE(inferedge::resize_token_piece_buffer(buffer, 8, 16).ok);
    EXPECT_EQ(buffer.size(), 8U);
    EXPECT_FALSE(inferedge::resize_token_piece_buffer(buffer, 0, 16).ok);
    EXPECT_FALSE(inferedge::resize_token_piece_buffer(buffer, 32, 16).ok);
}

TEST(ModelPackageManifestTest, PromptFileSecurity) {
    const auto workspace = unique_dir("prompt-ws");
    const auto file = workspace / "prompt.txt";
    write_bytes(file, "hello prompt");
    std::string out;
    ASSERT_TRUE(inferedge::load_untrusted_prompt_file(file, out).ok);
    EXPECT_EQ(out, "hello prompt");
    EXPECT_FALSE(inferedge::load_untrusted_prompt_file("prompt.txt", out).ok);
    write_bytes(file, "");
    EXPECT_FALSE(inferedge::load_untrusted_prompt_file(file, out).ok);
    write_bytes(file, "   \n");
    EXPECT_FALSE(inferedge::load_untrusted_prompt_file(file, out).ok);
    const auto link = workspace / "prompt-link.txt";
    write_bytes(workspace / "real-prompt.txt", "hello prompt");
    std::filesystem::create_symlink(workspace / "real-prompt.txt", link);
    EXPECT_FALSE(inferedge::load_untrusted_prompt_file(link, out).ok);
    const auto huge = workspace / "huge.txt";
    write_bytes(huge, std::string(inferedge::kGenerativeMaxPromptBytes + 1, 'a'));
    EXPECT_FALSE(inferedge::load_untrusted_prompt_file(huge, out).ok);
    const char bad_utf8[] = {'\xff'};
    write_bytes(workspace / "badutf.txt", std::string_view{bad_utf8, 1});
    EXPECT_FALSE(inferedge::load_untrusted_prompt_file(workspace / "badutf.txt", out).ok);
    const auto fifo = workspace / "prompt.fifo";
    ASSERT_EQ(::mkfifo(fifo.c_str(), 0600), 0);
    EXPECT_FALSE(inferedge::load_untrusted_prompt_file(fifo, out).ok);
    const auto hard = workspace / "hard-prompt.txt";
    write_bytes(file, "hello prompt");
    std::error_code error;
    std::filesystem::create_hard_link(file, hard, error);
    if (!error) {
        EXPECT_FALSE(inferedge::load_untrusted_prompt_file(file, out).ok);
    }
    const auto ancestor = workspace / "linkdir";
    std::filesystem::create_directory_symlink(workspace, ancestor);
    EXPECT_FALSE(inferedge::load_untrusted_prompt_file(ancestor / "prompt.txt", out).ok);
    std::filesystem::remove_all(workspace);
}

TEST(ModelPackageManifestTest, DoesNotPersistPromptOrOutput) {
    PackageFiles pkg;
    pkg.root = unique_dir("pkg-persist");
    write_valid_package(pkg);
    ASSERT_TRUE(inferedge::load_model_package(pkg.root).status.ok);
    bool found_prompt = false;
    for (const auto& entry : std::filesystem::directory_iterator(pkg.root)) {
        std::ifstream in(entry.path(), std::ios::binary);
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (body.find("secret-prompt-token") != std::string::npos) {
            found_prompt = true;
        }
    }
    EXPECT_FALSE(found_prompt);
    std::filesystem::remove_all(pkg.root);
}
