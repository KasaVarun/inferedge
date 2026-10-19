#include "inferedge/oracle_digest.hpp"

namespace inferedge {
namespace {

OracleError from_admission(AdmissionError error) {
    switch (error) {
    case AdmissionError::kOk:
        return OracleError::kOk;
    case AdmissionError::kSize:
        return OracleError::kSize;
    case AdmissionError::kMalformed:
        return OracleError::kMalformed;
    case AdmissionError::kSchema:
        return OracleError::kSchema;
    case AdmissionError::kDuplicateKey:
        return OracleError::kDuplicateKey;
    case AdmissionError::kUnknownKey:
        return OracleError::kUnknownKey;
    case AdmissionError::kMissingKey:
        return OracleError::kMissingKey;
    case AdmissionError::kLexical:
    case AdmissionError::kChecksumFormat:
        return OracleError::kLexical;
    case AdmissionError::kChecksumMismatch:
        return OracleError::kChecksum;
    case AdmissionError::kOverflow:
    case AdmissionError::kInvalidDate:
    case AdmissionError::kInternal:
        return OracleError::kInternal;
    }
    return OracleError::kInternal;
}

OracleError reject_expected_text(std::string_view expected_sha256) {
    if (validate_sha256(expected_sha256) != AdmissionError::kOk) {
        return OracleError::kLexical;
    }
    if (is_zero_digest(expected_sha256)) {
        return OracleError::kZeroIneligible;
    }
    return OracleError::kOk;
}

} // namespace

const char* oracle_error_name(OracleError error) noexcept {
    switch (error) {
    case OracleError::kOk:
        return "ok";
    case OracleError::kSize:
        return "size";
    case OracleError::kMalformed:
        return "malformed";
    case OracleError::kSchema:
        return "schema";
    case OracleError::kDuplicateKey:
        return "duplicate_key";
    case OracleError::kUnknownKey:
        return "unknown_key";
    case OracleError::kMissingKey:
        return "missing_key";
    case OracleError::kLexical:
        return "lexical";
    case OracleError::kChecksum:
        return "checksum";
    case OracleError::kMismatch:
        return "mismatch";
    case OracleError::kZeroIneligible:
        return "zero_ineligible";
    case OracleError::kInternal:
        return "internal";
    }
    return "internal";
}

OracleError hash_raw(std::string_view raw_bytes, RawDigest& out) {
    RawDigest accepted;
    accepted.byte_count = static_cast<std::uint64_t>(raw_bytes.size());
    accepted.sha256 = digest_hex(raw_bytes);
    out = std::move(accepted);
    return OracleError::kOk;
}

OracleError hash_full_file(std::string_view file_bytes, FullFileDigest& out) {
    FullFileDigest accepted;
    accepted.byte_count = static_cast<std::uint64_t>(file_bytes.size());
    accepted.sha256 = digest_hex(file_bytes);
    out = std::move(accepted);
    return OracleError::kOk;
}

OracleError hash_self_checksum(std::string_view file_bytes, const ClosedSchema& schema,
                               SelfChecksum& out) {
    CanonicalRecord record;
    const OracleError parsed = from_admission(parse_canonical_record(file_bytes, schema, record));
    if (parsed != OracleError::kOk) {
        return parsed;
    }
    const std::string checksum_line = std::string{kChecksumKey} + "=" + record.self_checksum + "\n";
    if (file_bytes.size() < checksum_line.size() ||
        file_bytes.substr(file_bytes.size() - checksum_line.size()) != checksum_line) {
        return OracleError::kInternal;
    }
    SelfChecksum accepted;
    accepted.prefix_byte_count =
        static_cast<std::uint64_t>(file_bytes.size() - checksum_line.size());
    accepted.sha256 = record.self_checksum;
    out = std::move(accepted);
    return OracleError::kOk;
}

OracleError verify_byte_count(std::string_view bytes, std::uint64_t expected_size) {
    if (static_cast<std::uint64_t>(bytes.size()) != expected_size) {
        return OracleError::kSize;
    }
    return OracleError::kOk;
}

OracleError verify_raw_digest(std::string_view raw_bytes, std::string_view expected_sha256) {
    if (const OracleError text = reject_expected_text(expected_sha256); text != OracleError::kOk) {
        return text;
    }
    RawDigest digest;
    if (hash_raw(raw_bytes, digest) != OracleError::kOk) {
        return OracleError::kInternal;
    }
    if (digest.sha256 != expected_sha256) {
        return OracleError::kMismatch;
    }
    return OracleError::kOk;
}

OracleError verify_full_file_digest(std::string_view file_bytes, std::string_view expected_sha256) {
    if (const OracleError text = reject_expected_text(expected_sha256); text != OracleError::kOk) {
        return text;
    }
    FullFileDigest digest;
    if (hash_full_file(file_bytes, digest) != OracleError::kOk) {
        return OracleError::kInternal;
    }
    if (digest.sha256 != expected_sha256) {
        return OracleError::kMismatch;
    }
    return OracleError::kOk;
}

OracleError verify_self_checksum(std::string_view file_bytes, const ClosedSchema& schema,
                                 std::string_view expected_sha256) {
    if (const OracleError text = reject_expected_text(expected_sha256); text != OracleError::kOk) {
        return text;
    }
    SelfChecksum digest;
    if (const OracleError hashed = hash_self_checksum(file_bytes, schema, digest);
        hashed != OracleError::kOk) {
        return hashed;
    }
    if (digest.sha256 != expected_sha256) {
        return OracleError::kMismatch;
    }
    return OracleError::kOk;
}

OracleError accept_digest_reference(std::string_view digest_text, std::uint64_t paired_size,
                                    bool zero_eligible) {
    if (validate_sha256(digest_text) != AdmissionError::kOk) {
        return OracleError::kLexical;
    }
    if (is_zero_digest(digest_text)) {
        if (!zero_eligible) {
            return OracleError::kZeroIneligible;
        }
        if (paired_size != 0) {
            return OracleError::kSize;
        }
        return OracleError::kOk;
    }
    return OracleError::kOk;
}

} // namespace inferedge
