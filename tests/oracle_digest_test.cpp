#include "inferedge/oracle_digest.hpp"

#include <gtest/gtest.h>
#include <string>

namespace {

using inferedge::ClosedSchema;
using inferedge::FullFileDigest;
using inferedge::OracleError;
using inferedge::RawDigest;
using inferedge::SelfChecksum;

constexpr std::string_view kAbc =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
constexpr std::string_view kAbd =
    "a52d159f262b2c6ddb724a61840befc36eb30c88877a4030b65cbe86298449c9";
constexpr std::string_view kLong =
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
constexpr std::string_view kBinary =
    "787a4bce1c8b1ef7f111820f3840c98b4ddf51d55f5e7a9b007fb6224bb563bc";
constexpr std::string_view kPath =
    "b91ece48c52bc109db9e61d6d4ac17b369d51e6595bbb8f8068b48f0d5398f3a";
constexpr std::string_view kHexText =
    "cd372fb85148700fa88095e3492d3f9f5beb43e555e5ff26d95f5a6adc36f8e6";
constexpr std::string_view kDemoPrefix = "schema_version=1\nrecord_kind=demo\n";
constexpr std::string_view kDemoSelf =
    "ba417b90b6b618d9301e7791487e8eeb1b762b555d374c5eab912a604237cf3a";
constexpr std::string_view kDemoFull =
    "86b66575a20c7635f5d14cf013ad57d6f3c7b45fa6cfecc5984380592c90a866";

ClosedSchema demo_schema() {
    return ClosedSchema{{"schema_version", "record_kind", "record_sha256"}, "demo"};
}

std::string demo_file() {
    return std::string{kDemoPrefix} + "record_sha256=" + std::string{kDemoSelf} + "\n";
}

std::string binary_bytes() {
    return std::string("\x00\xff\x00\x80", 4);
}

} // namespace

TEST(OracleDigest, KnownVectors) {
    RawDigest raw;
    raw.sha256 = "KEEP";
    EXPECT_EQ(inferedge::hash_raw("", raw), OracleError::kOk);
    EXPECT_EQ(raw.byte_count, 0u);
    EXPECT_EQ(raw.sha256, inferedge::kEmptyByteDigest);
    EXPECT_EQ(inferedge::hash_raw("abc", raw), OracleError::kOk);
    EXPECT_EQ(raw.sha256, kAbc);
    EXPECT_EQ(inferedge::hash_raw("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", raw),
              OracleError::kOk);
    EXPECT_EQ(raw.byte_count, 56u);
    EXPECT_EQ(raw.sha256, kLong);
    EXPECT_EQ(inferedge::hash_raw(binary_bytes(), raw), OracleError::kOk);
    EXPECT_EQ(raw.byte_count, 4u);
    EXPECT_EQ(raw.sha256, kBinary);
    EXPECT_NE(raw.sha256, inferedge::kZeroDigest);
}

TEST(OracleDigest, RawDigestBytes) {
    RawDigest raw;
    EXPECT_EQ(inferedge::hash_raw("abc", raw), OracleError::kOk);
    EXPECT_EQ(raw.sha256, kAbc);
    EXPECT_EQ(inferedge::hash_raw("abd", raw), OracleError::kOk);
    EXPECT_EQ(raw.sha256, kAbd);
    EXPECT_NE(std::string{kAbc}, std::string{kAbd});
    EXPECT_EQ(inferedge::hash_raw("", raw), OracleError::kOk);
    EXPECT_EQ(raw.sha256, inferedge::kEmptyByteDigest);
    EXPECT_EQ(inferedge::hash_raw("admission/blob.bin", raw), OracleError::kOk);
    EXPECT_EQ(raw.sha256, kPath);
    EXPECT_EQ(inferedge::hash_raw(inferedge::kEmptyByteDigest, raw), OracleError::kOk);
    EXPECT_EQ(raw.sha256, kHexText);
    EXPECT_NE(raw.sha256, inferedge::kEmptyByteDigest);
}

TEST(OracleDigest, FullFileDigest) {
    const std::string file = demo_file();
    FullFileDigest full;
    full.sha256 = "KEEP";
    EXPECT_EQ(inferedge::hash_full_file(file, full), OracleError::kOk);
    EXPECT_EQ(full.byte_count, file.size());
    EXPECT_EQ(full.sha256, kDemoFull);
    EXPECT_NE(full.sha256, kDemoSelf);
    std::string changed = file;
    changed[0] = 'S';
    FullFileDigest altered;
    EXPECT_EQ(inferedge::hash_full_file(changed, altered), OracleError::kOk);
    EXPECT_NE(altered.sha256, full.sha256);
    std::string no_lf = file.substr(0, file.size() - 1);
    EXPECT_EQ(inferedge::hash_full_file(no_lf, altered), OracleError::kOk);
    EXPECT_NE(altered.sha256, full.sha256);
    EXPECT_EQ(inferedge::verify_full_file_digest(file, kDemoFull), OracleError::kOk);
    EXPECT_EQ(inferedge::verify_full_file_digest(changed, kDemoFull), OracleError::kMismatch);
}

TEST(OracleDigest, SelfChecksumBoundary) {
    const std::string file = demo_file();
    SelfChecksum self;
    self.prefix_byte_count = 7;
    self.sha256 = "KEEP";
    EXPECT_EQ(inferedge::hash_self_checksum(file, demo_schema(), self), OracleError::kOk);
    EXPECT_EQ(self.prefix_byte_count, kDemoPrefix.size());
    EXPECT_EQ(self.sha256, kDemoSelf);
    EXPECT_EQ(kDemoPrefix.back(), '\n');
    RawDigest prefix;
    EXPECT_EQ(inferedge::hash_raw(kDemoPrefix, prefix), OracleError::kOk);
    EXPECT_EQ(prefix.sha256, self.sha256);
    FullFileDigest full;
    EXPECT_EQ(inferedge::hash_full_file(file, full), OracleError::kOk);
    EXPECT_NE(full.sha256, self.sha256);
    EXPECT_GT(full.byte_count, self.prefix_byte_count);
    EXPECT_EQ(inferedge::verify_self_checksum(file, demo_schema(), kDemoSelf), OracleError::kOk);
}

TEST(OracleDigest, SelfChecksumRejections) {
    const ClosedSchema schema = demo_schema();
    SelfChecksum kept;
    kept.prefix_byte_count = 4;
    kept.sha256 = "KEEP";
    EXPECT_EQ(inferedge::hash_self_checksum(kDemoPrefix, schema, kept), OracleError::kMissingKey);
    EXPECT_EQ(kept.sha256, "KEEP");
    const std::string duplicated = demo_file() + "record_sha256=" + std::string{kDemoSelf} + "\n";
    EXPECT_EQ(inferedge::hash_self_checksum(duplicated, schema, kept), OracleError::kDuplicateKey);
    const std::string non_final =
        "schema_version=1\nrecord_sha256=" + std::string(64, 'a') + "\nrecord_kind=demo\n";
    EXPECT_EQ(inferedge::hash_self_checksum(non_final, schema, kept), OracleError::kSchema);
    std::string malformed = demo_file();
    malformed.erase(malformed.find(kDemoSelf) + 63, 1);
    EXPECT_EQ(inferedge::hash_self_checksum(malformed, schema, kept), OracleError::kLexical);
    std::string upper = demo_file();
    upper.replace(upper.find(kDemoSelf), 1, "B");
    EXPECT_EQ(inferedge::hash_self_checksum(upper, schema, kept), OracleError::kLexical);
    const std::string no_lf = demo_file().substr(0, demo_file().size() - 1);
    EXPECT_EQ(inferedge::hash_self_checksum(no_lf, schema, kept), OracleError::kMalformed);
    std::string crlf = demo_file();
    crlf.insert(crlf.find('\n'), "\r");
    EXPECT_EQ(inferedge::hash_self_checksum(crlf, schema, kept), OracleError::kMalformed);
    std::string wrong = demo_file();
    wrong[wrong.size() - 2] = 'b';
    EXPECT_EQ(inferedge::hash_self_checksum(wrong, schema, kept), OracleError::kChecksum);
    EXPECT_EQ(kept.prefix_byte_count, 4u);
    EXPECT_EQ(kept.sha256, "KEEP");
}

TEST(OracleDigest, ByteCount) {
    EXPECT_EQ(inferedge::verify_byte_count("abc", 3), OracleError::kOk);
    EXPECT_EQ(inferedge::verify_byte_count("abc", 2), OracleError::kSize);
    EXPECT_EQ(inferedge::verify_byte_count("abc", 4), OracleError::kSize);
    EXPECT_EQ(inferedge::verify_byte_count("", 0), OracleError::kOk);
    EXPECT_EQ(inferedge::verify_byte_count("", 1), OracleError::kSize);
    EXPECT_EQ(inferedge::verify_byte_count("", 4294967296ull), OracleError::kSize);
    EXPECT_EQ(inferedge::verify_byte_count("a", 4294967297ull), OracleError::kSize);
}

TEST(OracleDigest, DigestVerification) {
    EXPECT_EQ(inferedge::verify_raw_digest("abc", kAbc), OracleError::kOk);
    EXPECT_EQ(inferedge::verify_raw_digest("abc", kAbd), OracleError::kMismatch);
    EXPECT_EQ(inferedge::verify_raw_digest("abc", std::string(63, 'a')), OracleError::kLexical);
    EXPECT_EQ(inferedge::verify_raw_digest("abc", std::string(65, 'a')), OracleError::kLexical);
    EXPECT_EQ(inferedge::verify_raw_digest("abc", std::string(63, 'a') + "g"),
              OracleError::kLexical);
    const std::string upper_abc =
        "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD";
    EXPECT_EQ(inferedge::verify_raw_digest("abc", upper_abc), OracleError::kLexical);
    EXPECT_EQ(inferedge::verify_raw_digest("", inferedge::kZeroDigest),
              OracleError::kZeroIneligible);
    EXPECT_EQ(inferedge::verify_raw_digest("", inferedge::kEmptyByteDigest), OracleError::kOk);
    EXPECT_EQ(inferedge::verify_full_file_digest(demo_file(), inferedge::kZeroDigest),
              OracleError::kZeroIneligible);
    EXPECT_EQ(inferedge::verify_self_checksum(demo_file(), demo_schema(), inferedge::kZeroDigest),
              OracleError::kZeroIneligible);
}

TEST(OracleDigest, ZeroAndEmpty) {
    EXPECT_NE(std::string{inferedge::kZeroDigest}, std::string{inferedge::kEmptyByteDigest});
    EXPECT_EQ(inferedge::accept_digest_reference(inferedge::kEmptyByteDigest, 0, false),
              OracleError::kOk);
    EXPECT_EQ(inferedge::verify_raw_digest("", inferedge::kEmptyByteDigest), OracleError::kOk);
    EXPECT_EQ(inferedge::verify_byte_count("", 0), OracleError::kOk);
    EXPECT_EQ(inferedge::verify_raw_digest("", inferedge::kZeroDigest),
              OracleError::kZeroIneligible);
    EXPECT_EQ(inferedge::accept_digest_reference(inferedge::kZeroDigest, 0, true),
              OracleError::kOk);
    EXPECT_EQ(inferedge::accept_digest_reference(inferedge::kZeroDigest, 0, false),
              OracleError::kZeroIneligible);
    EXPECT_EQ(inferedge::accept_digest_reference(inferedge::kZeroDigest, 1, true),
              OracleError::kSize);
    EXPECT_EQ(inferedge::accept_digest_reference(inferedge::kZeroDigest, 9, false),
              OracleError::kZeroIneligible);
    EXPECT_EQ(inferedge::accept_digest_reference(std::string(64, 'A'), 0, true),
              OracleError::kLexical);
    EXPECT_EQ(inferedge::accept_digest_reference(std::string(63, '0'), 0, true),
              OracleError::kLexical);
}

TEST(OracleDigest, KindSeparation) {
    const std::string file = demo_file();
    RawDigest raw_prefix;
    FullFileDigest full;
    SelfChecksum self;
    EXPECT_EQ(inferedge::hash_raw(kDemoPrefix, raw_prefix), OracleError::kOk);
    EXPECT_EQ(inferedge::hash_full_file(file, full), OracleError::kOk);
    EXPECT_EQ(inferedge::hash_self_checksum(file, demo_schema(), self), OracleError::kOk);
    EXPECT_EQ(raw_prefix.sha256, kDemoSelf);
    EXPECT_EQ(full.sha256, kDemoFull);
    EXPECT_EQ(self.sha256, kDemoSelf);
    EXPECT_NE(full.sha256, self.sha256);
    EXPECT_NE(full.byte_count, self.prefix_byte_count);
    RawDigest as_text;
    EXPECT_EQ(inferedge::hash_raw(file, as_text), OracleError::kOk);
    EXPECT_EQ(as_text.sha256, full.sha256);
    EXPECT_NE(as_text.sha256, self.sha256);
}

TEST(OracleDigest, FailureLeavesOutput) {
    RawDigest raw;
    raw.byte_count = 11;
    raw.sha256 = "KEEP";
    EXPECT_EQ(inferedge::verify_raw_digest("abc", "zz"), OracleError::kLexical);
    EXPECT_EQ(raw.byte_count, 11u);
    EXPECT_EQ(raw.sha256, "KEEP");
    FullFileDigest full;
    full.byte_count = 12;
    full.sha256 = "KEEP";
    EXPECT_EQ(inferedge::verify_full_file_digest(demo_file(), std::string(64, 'A')),
              OracleError::kLexical);
    EXPECT_EQ(full.sha256, "KEEP");
    SelfChecksum self;
    self.prefix_byte_count = 13;
    self.sha256 = "KEEP";
    EXPECT_EQ(inferedge::hash_self_checksum("", demo_schema(), self), OracleError::kSize);
    EXPECT_EQ(self.prefix_byte_count, 13u);
    EXPECT_EQ(self.sha256, "KEEP");
    EXPECT_STREQ(inferedge::oracle_error_name(OracleError::kMismatch), "mismatch");
    EXPECT_STREQ(inferedge::oracle_error_name(OracleError::kZeroIneligible), "zero_ineligible");
}
