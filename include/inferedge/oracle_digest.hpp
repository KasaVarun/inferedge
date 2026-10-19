#pragma once

#include "inferedge/admission_record.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace inferedge {

// Phase 9-C5 digest foundation.
//
// hash_raw hashes the supplied bytes and nothing else. hash_full_file hashes every byte of
// the supplied structured file, including the checksum line and the final LF. hash_self_checksum
// hashes only the prefix selected by parse_canonical_record: every byte before the final
// record_sha256= line, including the LF that ends the preceding line and excluding that
// checksum line. A self-checksum is not a digest of the file that contains the checksum.
//
// The three result types are not interchangeable. This module does not open files, does not
// read a path or an index in place of raw bytes, and does not own the section 8 field matrix.
// Zero-digest eligibility stays an explicit caller argument. The zero digest is absence, and
// it is not the empty-byte digest.

enum class OracleError {
    kOk = 0,
    kSize,
    kMalformed,
    kSchema,
    kDuplicateKey,
    kUnknownKey,
    kMissingKey,
    kLexical,
    kChecksum,
    kMismatch,
    kZeroIneligible,
    kInternal,
};

struct RawDigest {
    std::uint64_t byte_count{0};
    std::string sha256;
};

struct FullFileDigest {
    std::uint64_t byte_count{0};
    std::string sha256;
};

struct SelfChecksum {
    std::uint64_t prefix_byte_count{0};
    std::string sha256;
};

const char* oracle_error_name(OracleError error) noexcept;

// On failure, the output parameter is unchanged.
OracleError hash_raw(std::string_view raw_bytes, RawDigest& out);
OracleError hash_full_file(std::string_view file_bytes, FullFileDigest& out);
OracleError hash_self_checksum(std::string_view file_bytes, const ClosedSchema& schema,
                               SelfChecksum& out);

// Compares the exact supplied byte count with expected_size. No integer narrowing.
OracleError verify_byte_count(std::string_view bytes, std::uint64_t expected_size);

// Content checks. The zero digest is absence and is rejected here. It is not a content hash.
OracleError verify_raw_digest(std::string_view raw_bytes, std::string_view expected_sha256);
OracleError verify_full_file_digest(std::string_view file_bytes, std::string_view expected_sha256);
OracleError verify_self_checksum(std::string_view file_bytes, const ClosedSchema& schema,
                                 std::string_view expected_sha256);

// Lexical sha256 text plus the absence sentinel. The zero digest is accepted only when
// zero_eligible is true and paired_size is 0. Any other lowercase sha256 text, including
// the empty-byte digest, is ordinary digest text and does not mean absence.
OracleError accept_digest_reference(std::string_view digest_text, std::uint64_t paired_size,
                                    bool zero_eligible);

} // namespace inferedge
