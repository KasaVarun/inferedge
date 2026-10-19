#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace inferedge {

// Phase 9-C2 canonical admission-record foundation.
//
// This layer parses and serializes one caller-selected closed key order. It does not open
// files, walk a trust root, verify signatures, canonicalize tensors, execute a model, or
// choose which of several closed orders a discriminator selects. Callers that later own
// telemetry_log or deletion_authorization pass exactly one of those orders. A key that
// belongs only to another order is unknown.
//
// List-record element lines stay caller-owned. Their grammars contain spaces and a
// sort key defined by later contract sections. This parser accepts only `key=value`
// lines in the supplied order. It does not reorder input, and it does not treat an
// element line or any other extra line as a key. The checksum below covers the bytes
// the caller actually produced; it never sorts them.
//
// Zero-digest eligibility is not assigned to fields here. A caller that knows a field
// is zero-eligible passes that policy to validate_digest_or_zero. The zero digest and
// the empty-byte digest are both lexically sha256 and are not interchangeable.
//
// Ordinary invalid input returns AdmissionError. On failure, every output parameter is
// left unchanged. There is no partially accepted record and no partially serialized file.

inline constexpr std::size_t kMaxCanonicalRecordBytes = 16384;
inline constexpr std::size_t kInt32ListMax = 4096;
inline constexpr std::size_t kUint64ListMax = 8;
inline constexpr std::string_view kChecksumKey{"record_sha256"};
inline constexpr std::string_view kZeroDigest{
    "0000000000000000000000000000000000000000000000000000000000000000"};
inline constexpr std::string_view kEmptyByteDigest{
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"};

enum class AdmissionError {
    kOk = 0,
    kSize,
    kMalformed,
    kSchema,
    kDuplicateKey,
    kUnknownKey,
    kMissingKey,
    kLexical,
    kChecksumFormat,
    kChecksumMismatch,
    kOverflow,
    kInvalidDate,
    kInternal,
};

const char* admission_error_name(AdmissionError error) noexcept;

struct ClosedSchema {
    // Includes record_sha256 as the final key. Order is the closed serialization order.
    std::vector<std::string> keys;
    // Empty when this schema does not require a record_kind match.
    std::string record_kind;
};

struct CanonicalRecord {
    std::vector<std::string> keys;
    std::vector<std::string> values;
    std::string bytes;
    std::string self_checksum;
    std::string full_file_digest;
};

struct Int32List {
    bool is_none{false};
    std::vector<std::int32_t> values;
};

struct Uint64List {
    bool is_none{false};
    std::vector<std::uint64_t> values;
};

// SHA-256 of the supplied bytes, lowercase hex. This is neither a self-checksum nor a
// full-file digest until the caller chooses those exact bytes.
std::string digest_hex(std::string_view bytes);

AdmissionError check_closed_schema(const ClosedSchema& schema);

// `values` are the non-checksum fields, in schema order. The checksum line is appended.
AdmissionError serialize_canonical_record(const ClosedSchema& schema,
                                          std::span<const std::string_view> values,
                                          std::string& out);

// Accepts only a complete canonical file whose self-checksum matches.
AdmissionError parse_canonical_record(std::string_view bytes, const ClosedSchema& schema,
                                      CanonicalRecord& out);

AdmissionError validate_project_id(std::string_view text);
AdmissionError validate_id_or_none(std::string_view text);
AdmissionError make_compound_id(std::string_view prefix, std::string_view component,
                                std::string& out);

AdmissionError parse_uint(std::string_view text, std::uint32_t& out);
AdmissionError parse_uint64(std::string_view text, std::uint64_t& out);
AdmissionError parse_int32(std::string_view text, std::int32_t& out);
AdmissionError parse_int32_list(std::string_view text, Int32List& out);
AdmissionError parse_uint64_list(std::string_view text, Uint64List& out);

AdmissionError validate_sha256(std::string_view text);
bool is_zero_digest(std::string_view text) noexcept;
bool is_empty_byte_digest(std::string_view text) noexcept;
// `zero_eligible` is caller policy. This function does not know field names.
AdmissionError validate_digest_or_zero(std::string_view text, bool zero_eligible);

AdmissionError validate_date(std::string_view text);
AdmissionError add_calendar_days(std::string_view date, std::uint64_t days, std::string& out);
AdmissionError add_uint64(std::uint64_t left, std::uint64_t right, std::uint64_t& sum);

AdmissionError validate_filename(std::string_view text);
AdmissionError validate_upstream_name(std::string_view text);
AdmissionError validate_yes_no(std::string_view text);
AdmissionError validate_phase_id(std::string_view text);
AdmissionError validate_pattern_hex(std::string_view text);
AdmissionError validate_name_hex(std::string_view text, std::string& decoded);

} // namespace inferedge
