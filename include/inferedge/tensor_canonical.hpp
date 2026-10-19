#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace inferedge {

// Phase 9-C4 tensor-canonical foundation.
//
// This layer decodes pattern_hex and name_hex, matches one whole decoded name against one
// decoded pattern, and parses or serializes the four in-memory tensor records from section 26.
// It does not open files, walk a directory, load a model, parse GGUF, or admit a provenance.
//
// Whole-name matching is provided only by match_tensor_name. 0x2A matches zero or more bytes.
// Matching is anchored at both ends. Rule-list order is preserved as record data. Phase 9-C4
// does not define cross-rule classification or precedence. Callers must not infer first-match,
// last-match, or most-specific precedence. No closed tensor type-name vocabulary is defined.

enum class TensorError {
    kOk = 0,
    kMalformed,
    kSize,
    kSchema,
    kChecksum,
    kLexical,
    kCountMismatch,
    kDuplicate,
    kInvalidRole,
    kInvalidRank,
    kDimensionMismatch,
    kInvalidUtf8,
    kPattern,
    kOverflow,
    kHistogramMismatch,
    kChildIdMismatch,
    kChildSizeMismatch,
    kChildDigestMismatch,
    kInternal,
};

enum class TensorRole {
    kWeight,
    kNonWeight,
};

struct TensorRule {
    std::string pattern_hex;
    std::string pattern_bytes;
    TensorRole role{TensorRole::kWeight};
};

struct TensorRuleList {
    std::string list_id;
    std::uint32_t rule_count{0};
    std::vector<TensorRule> rules;
    std::string bytes;
    std::string self_checksum;
    std::string full_file_digest;
};

struct TensorInventoryEntry {
    std::string name_hex;
    std::string name_bytes;
    std::uint32_t rank{0};
    std::vector<std::uint64_t> dimensions;
};

struct TensorInventory {
    std::string inventory_id;
    std::uint32_t tensor_count{0};
    std::vector<TensorInventoryEntry> entries;
    std::string bytes;
    std::string self_checksum;
    std::string full_file_digest;
};

struct TensorHistogramEntry {
    std::string type_name;
    std::uint32_t count{0};
};

struct TensorHistogram {
    std::string histogram_id;
    std::uint32_t type_count{0};
    std::uint32_t weight_tensor_count{0};
    std::vector<TensorHistogramEntry> entries;
    std::string bytes;
    std::string self_checksum;
    std::string full_file_digest;
};

struct TensorSchema {
    std::string schema_id;
    std::string rule_list_id;
    std::uint32_t rule_list_file_size{0};
    std::string rule_list_file_sha256;
    std::string inventory_id;
    std::uint32_t inventory_file_size{0};
    std::string inventory_file_sha256;
    std::string histogram_id;
    std::uint32_t histogram_file_size{0};
    std::string histogram_file_sha256;
    std::string bytes;
    std::string self_checksum;
    std::string full_file_digest;
};

struct TensorBundle {
    TensorSchema schema;
    TensorRuleList rules;
    TensorInventory inventory;
    TensorHistogram histogram;
};

const char* tensor_error_name(TensorError error) noexcept;

// Decodes pattern_hex into exact bytes. 0x2A remains a metacharacter; there is no escape.
// On failure, `decoded` is unchanged.
TensorError decode_pattern_hex(std::string_view pattern_hex, std::string& decoded);

// Decodes name_hex into the exact shortest-form UTF-8 bytes. On failure, `decoded` is unchanged.
TensorError decode_name_hex(std::string_view name_hex, std::string& decoded);

// Anchored whole-name match. A false match is kOk with `matched` set false. On failure,
// `matched` is unchanged. An empty name or pattern is kPattern.
TensorError match_tensor_name(std::string_view pattern_bytes, std::string_view name_bytes,
                              bool& matched);

TensorError parse_tensor_schema(std::string_view bytes, TensorSchema& out);
TensorError serialize_tensor_schema(const TensorSchema& schema, std::string& out);

TensorError parse_tensor_rule_list(std::string_view bytes, TensorRuleList& out);
TensorError serialize_tensor_rule_list(std::string_view list_id, std::span<const TensorRule> rules,
                                       std::string& out);

TensorError parse_tensor_inventory(std::string_view bytes, TensorInventory& out);
TensorError serialize_tensor_inventory(std::string_view inventory_id,
                                       std::span<const TensorInventoryEntry> entries,
                                       std::string& out);

// `weight_tensor_count` is caller-supplied. This function does not inspect a model.
TensorError parse_tensor_histogram(std::string_view bytes, std::uint32_t weight_tensor_count,
                                   TensorHistogram& out);
TensorError serialize_tensor_histogram(std::string_view histogram_id,
                                       std::span<const TensorHistogramEntry> entries,
                                       std::string& out);

// In-memory bundle. Child digests are full-file digests. Failure leaves `out` unchanged.
// A mismatch uses the child-id, child-size, child-digest, or histogram error. This is the
// section 26 provenance_unresolved condition for these four byte strings, not a loader.
TensorError validate_tensor_bundle(std::string_view schema_bytes, std::string_view rule_bytes,
                                   std::string_view inventory_bytes,
                                   std::string_view histogram_bytes,
                                   std::uint32_t weight_tensor_count, TensorBundle& out);

} // namespace inferedge
