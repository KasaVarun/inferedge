#include "inferedge/tensor_canonical.hpp"

#include <chrono>
#include <gtest/gtest.h>
#include <span>
#include <string>

namespace {

using inferedge::TensorBundle;
using inferedge::TensorError;
using inferedge::TensorHistogram;
using inferedge::TensorHistogramEntry;
using inferedge::TensorInventory;
using inferedge::TensorInventoryEntry;
using inferedge::TensorRole;
using inferedge::TensorRule;
using inferedge::TensorRuleList;
using inferedge::TensorSchema;

const char* kRules =
    "schema_version=1\n"
    "record_kind=tensor_rule_list\n"
    "list_id=rules1\n"
    "rule_count=1\n"
    "2a weight\n"
    "record_sha256=c26cf6d3706a90b0e41b7f211a6c1e88e921dbe594b78ff1d524eaee84670f66\n";
const char* kRulesSelf = "c26cf6d3706a90b0e41b7f211a6c1e88e921dbe594b78ff1d524eaee84670f66";
const char* kRulesFull = "38d74868310d832c625c9db03f0082494568482212491118cfe38046b8b8a4b1";

const char* kInventory =
    "schema_version=1\n"
    "record_kind=tensor_inventory\n"
    "inventory_id=inv1\n"
    "tensor_count=1\n"
    "61 1 1\n"
    "record_sha256=e5d8f82fb675822eb086933675857451663a5a17b817df289b34ba8214c17c0d\n";
const char* kInventorySelf = "e5d8f82fb675822eb086933675857451663a5a17b817df289b34ba8214c17c0d";
const char* kInventoryFull = "573f4434c63429205ad37e8e5ea967b0cfcb2c78f267dffc68f734da87a22918";

const char* kHistogram =
    "schema_version=1\n"
    "record_kind=tensor_histogram\n"
    "histogram_id=hist1\n"
    "type_count=1\n"
    "f32 1\n"
    "record_sha256=2d939f1e812f66177604df1b8a19cf846550dc5aae2ec474ccdb373852fa7643\n";
const char* kHistogramSelf = "2d939f1e812f66177604df1b8a19cf846550dc5aae2ec474ccdb373852fa7643";
const char* kHistogramFull = "45ea77678d0d9b564f25778d92790898d89f979882a1ae0c9e3da8c3c7357e7b";

const char* kSchema =
    "schema_version=1\n"
    "record_kind=tensor_schema\n"
    "schema_id=schema1\n"
    "rule_list_id=rules1\n"
    "rule_list_file_size=163\n"
    "rule_list_file_sha256=38d74868310d832c625c9db03f0082494568482212491118cfe38046b8b8a4b1\n"
    "inventory_id=inv1\n"
    "inventory_file_size=165\n"
    "inventory_file_sha256=573f4434c63429205ad37e8e5ea967b0cfcb2c78f267dffc68f734da87a22918\n"
    "histogram_id=hist1\n"
    "histogram_file_size=163\n"
    "histogram_file_sha256=45ea77678d0d9b564f25778d92790898d89f979882a1ae0c9e3da8c3c7357e7b\n"
    "record_sha256=5228ede6cf8614f1fbcaf5bd6dce4e4d5b7fb614db8f873d9c8770f76d5ad534\n";
const char* kSchemaSelf = "5228ede6cf8614f1fbcaf5bd6dce4e4d5b7fb614db8f873d9c8770f76d5ad534";
const char* kSchemaFull = "c68a9dcfd39dacf625468ef555c3eb659859abb9a432c693043abcc97d52f898";

} // namespace

TEST(TensorCanonical, PatternHexBoundaries) {
    std::string decoded = "KEEP";
    EXPECT_EQ(inferedge::decode_pattern_hex("", decoded), TensorError::kLexical);
    EXPECT_EQ(decoded, "KEEP");
    EXPECT_EQ(inferedge::decode_pattern_hex("a", decoded), TensorError::kLexical);
    EXPECT_EQ(inferedge::decode_pattern_hex("abc", decoded), TensorError::kLexical);
    EXPECT_EQ(inferedge::decode_pattern_hex(std::string(514, 'a'), decoded), TensorError::kLexical);
    EXPECT_EQ(inferedge::decode_pattern_hex("2A", decoded), TensorError::kLexical);
    EXPECT_EQ(inferedge::decode_pattern_hex("2g", decoded), TensorError::kLexical);
    EXPECT_EQ(decoded, "KEEP");
    EXPECT_EQ(inferedge::decode_pattern_hex("2a", decoded), TensorError::kOk);
    EXPECT_EQ(decoded, std::string(1, static_cast<char>(0x2A)));
    EXPECT_EQ(inferedge::decode_pattern_hex(std::string(512, 'a'), decoded), TensorError::kOk);
    EXPECT_EQ(decoded.size(), 256u);
}

TEST(TensorCanonical, NameHexAndUtf8) {
    std::string decoded = "KEEP";
    EXPECT_EQ(inferedge::decode_name_hex("", decoded), TensorError::kLexical);
    EXPECT_EQ(inferedge::decode_name_hex("zz", decoded), TensorError::kLexical);
    EXPECT_EQ(decoded, "KEEP");
    EXPECT_EQ(inferedge::decode_name_hex("61", decoded), TensorError::kOk);
    EXPECT_EQ(decoded, "a");
    EXPECT_EQ(inferedge::decode_name_hex("c3a9", decoded), TensorError::kOk);
    EXPECT_EQ(decoded, "\xC3\xA9");
    EXPECT_EQ(inferedge::decode_name_hex("f09f9880", decoded), TensorError::kOk);
    EXPECT_EQ(decoded, "\xF0\x9F\x98\x80");
    EXPECT_EQ(inferedge::decode_name_hex("00", decoded), TensorError::kInvalidUtf8);
    EXPECT_EQ(decoded, "\xF0\x9F\x98\x80");
    EXPECT_EQ(inferedge::decode_name_hex("c080", decoded), TensorError::kInvalidUtf8);
    EXPECT_EQ(inferedge::decode_name_hex("80", decoded), TensorError::kInvalidUtf8);
    EXPECT_EQ(inferedge::decode_name_hex("c3", decoded), TensorError::kInvalidUtf8);
    EXPECT_EQ(inferedge::decode_name_hex("eda080", decoded), TensorError::kInvalidUtf8);
    EXPECT_EQ(inferedge::decode_name_hex("f4908080", decoded), TensorError::kInvalidUtf8);
    EXPECT_EQ(inferedge::decode_name_hex("f48fbfbf", decoded), TensorError::kOk);
}

TEST(TensorCanonical, WholeNameMatch) {
    bool matched = true;
    EXPECT_EQ(inferedge::match_tensor_name("", "a", matched), TensorError::kPattern);
    EXPECT_TRUE(matched);
    matched = false;
    EXPECT_EQ(inferedge::match_tensor_name("a", "a", matched), TensorError::kOk);
    EXPECT_TRUE(matched);
    EXPECT_EQ(inferedge::match_tensor_name(std::string(1, '*'), "abc", matched), TensorError::kOk);
    EXPECT_TRUE(matched);
    EXPECT_EQ(inferedge::match_tensor_name(std::string(1, '*') + "b", "zzb", matched),
              TensorError::kOk);
    EXPECT_TRUE(matched);
    EXPECT_EQ(inferedge::match_tensor_name(std::string(1, '*') + "b", "bzz", matched),
              TensorError::kOk);
    EXPECT_FALSE(matched);
    EXPECT_EQ(inferedge::match_tensor_name("a" + std::string(1, '*'), "azz", matched),
              TensorError::kOk);
    EXPECT_TRUE(matched);
    EXPECT_EQ(inferedge::match_tensor_name("a" + std::string(1, '*') + "b", "ab", matched),
              TensorError::kOk);
    EXPECT_TRUE(matched);
    EXPECT_EQ(inferedge::match_tensor_name("a" + std::string(1, '*') + "b", "axyb", matched),
              TensorError::kOk);
    EXPECT_TRUE(matched);
    EXPECT_EQ(inferedge::match_tensor_name(std::string(2, '*'), "ab", matched), TensorError::kOk);
    EXPECT_TRUE(matched);
    EXPECT_EQ(inferedge::match_tensor_name(
                  "a" + std::string(1, '*') + "b" + std::string(1, '*') + "c", "aXXbYYc", matched),
              TensorError::kOk);
    EXPECT_TRUE(matched);
    EXPECT_EQ(inferedge::match_tensor_name("a", "ab", matched), TensorError::kOk);
    EXPECT_FALSE(matched);
    EXPECT_EQ(inferedge::match_tensor_name("a", "ba", matched), TensorError::kOk);
    EXPECT_FALSE(matched);
    const std::string utf8 = "\xC3\xA9";
    EXPECT_EQ(inferedge::match_tensor_name(utf8, utf8, matched), TensorError::kOk);
    EXPECT_TRUE(matched);
    EXPECT_EQ(inferedge::match_tensor_name(std::string(1, '*'), utf8, matched), TensorError::kOk);
    EXPECT_TRUE(matched);
    const std::string wide_pattern(256, '*');
    const std::string wide_name(256, 'a');
    const auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(inferedge::match_tensor_name(wide_pattern, wide_name, matched), TensorError::kOk);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    EXPECT_TRUE(matched);
    EXPECT_LT(elapsed, std::chrono::milliseconds(100));
    EXPECT_EQ(inferedge::match_tensor_name(std::string(257, '*'), wide_name, matched),
              TensorError::kPattern);
}

TEST(TensorCanonical, SchemaKnownDigestAndRoundTrip) {
    TensorSchema parsed;
    EXPECT_EQ(inferedge::parse_tensor_schema(kSchema, parsed), TensorError::kOk);
    EXPECT_EQ(parsed.schema_id, "schema1");
    EXPECT_EQ(parsed.rule_list_file_size, 163u);
    EXPECT_EQ(parsed.self_checksum, kSchemaSelf);
    EXPECT_EQ(parsed.full_file_digest, kSchemaFull);
    EXPECT_NE(parsed.self_checksum, parsed.full_file_digest);
    EXPECT_EQ(parsed.bytes, kSchema);
    std::string serialized;
    EXPECT_EQ(inferedge::serialize_tensor_schema(parsed, serialized), TensorError::kOk);
    EXPECT_EQ(serialized, kSchema);
    TensorSchema kept;
    kept.schema_id = "KEEP";
    std::string bad = kSchema;
    bad[0] = 'S';
    EXPECT_EQ(inferedge::parse_tensor_schema(bad, kept), TensorError::kSchema);
    EXPECT_EQ(kept.schema_id, "KEEP");
    std::string reordered = kSchema;
    const std::string first = "schema_version=1\n";
    const std::string second = "record_kind=tensor_schema\n";
    reordered.replace(0, first.size() + second.size(), second + first);
    EXPECT_EQ(inferedge::parse_tensor_schema(reordered, kept), TensorError::kSchema);
    std::string missing = kSchema;
    missing.erase(0, first.size());
    EXPECT_EQ(inferedge::parse_tensor_schema(missing, kept), TensorError::kSchema);
    std::string duplicate = std::string(kSchema).insert(first.size(), first);
    EXPECT_EQ(inferedge::parse_tensor_schema(duplicate, kept), TensorError::kSchema);
    std::string unknown = std::string(kSchema).insert(first.size(), "extra=1\n");
    EXPECT_EQ(inferedge::parse_tensor_schema(unknown, kept), TensorError::kSchema);
    std::string version = kSchema;
    version.replace(version.find("schema_version=1"), 16, "schema_version=2");
    EXPECT_EQ(inferedge::parse_tensor_schema(version, kept), TensorError::kSchema);
    std::string kind = kSchema;
    kind.replace(kind.find("tensor_schema"), 13, "tensor_schemx");
    EXPECT_EQ(inferedge::parse_tensor_schema(kind, kept), TensorError::kSchema);
    TensorSchema lexical = parsed;
    lexical.schema_id = "none";
    std::string rejected;
    EXPECT_EQ(inferedge::serialize_tensor_schema(lexical, rejected), TensorError::kLexical);
    lexical = parsed;
    lexical.rule_list_file_sha256 = std::string(64, '0');
    EXPECT_EQ(inferedge::serialize_tensor_schema(lexical, rejected), TensorError::kLexical);
    std::string checksum = kSchema;
    checksum[checksum.rfind('4')] = '5';
    EXPECT_EQ(inferedge::parse_tensor_schema(checksum, kept), TensorError::kChecksum);
    EXPECT_EQ(kept.schema_id, "KEEP");
}

TEST(TensorCanonical, RuleListKnownDigest) {
    TensorRuleList parsed;
    EXPECT_EQ(inferedge::parse_tensor_rule_list(kRules, parsed), TensorError::kOk);
    EXPECT_EQ(parsed.rule_count, 1u);
    EXPECT_EQ(parsed.rules[0].pattern_bytes, "*");
    EXPECT_EQ(parsed.rules[0].role, TensorRole::kWeight);
    EXPECT_EQ(parsed.self_checksum, kRulesSelf);
    EXPECT_EQ(parsed.full_file_digest, kRulesFull);
    EXPECT_NE(parsed.self_checksum, parsed.full_file_digest);
    std::string serialized;
    EXPECT_EQ(inferedge::serialize_tensor_rule_list(parsed.list_id, parsed.rules, serialized),
              TensorError::kOk);
    EXPECT_EQ(serialized, kRules);
    const char* zero =
        "schema_version=1\n"
        "record_kind=tensor_rule_list\n"
        "list_id=rules0\n"
        "rule_count=0\n"
        "record_sha256=bd730482dcb4ca631e467dade118cfe20b4eef47bfb7f24c870efba99d787046\n";
    TensorRuleList empty;
    EXPECT_EQ(inferedge::parse_tensor_rule_list(zero, empty), TensorError::kOk);
    EXPECT_EQ(empty.rule_count, 0u);
    EXPECT_TRUE(empty.rules.empty());
    TensorRule second{"61", "a", TensorRole::kNonWeight};
    std::vector<TensorRule> many = parsed.rules;
    many.push_back(second);
    std::string multi;
    EXPECT_EQ(inferedge::serialize_tensor_rule_list("rules1", many, multi), TensorError::kOk);
    TensorRuleList parsed_many;
    EXPECT_EQ(inferedge::parse_tensor_rule_list(multi, parsed_many), TensorError::kOk);
    EXPECT_EQ(parsed_many.rules.size(), 2u);
    EXPECT_EQ(parsed_many.rules[1].role, TensorRole::kNonWeight);
    many.push_back(parsed.rules[0]);
    EXPECT_EQ(inferedge::serialize_tensor_rule_list("rules1", many, multi),
              TensorError::kDuplicate);
    std::string bad_pattern = kRules;
    bad_pattern.replace(bad_pattern.find("2a weight"), 2, "2A");
    TensorRuleList kept;
    kept.list_id = "KEEP";
    EXPECT_EQ(inferedge::parse_tensor_rule_list(bad_pattern, kept), TensorError::kPattern);
    EXPECT_EQ(kept.list_id, "KEEP");
    std::string bad_role = kRules;
    bad_role.replace(bad_role.find("weight"), 6, "widget");
    EXPECT_EQ(inferedge::parse_tensor_rule_list(bad_role, kept), TensorError::kInvalidRole);
    std::string count = kRules;
    count.replace(count.find("rule_count=1"), 12, "rule_count=2");
    EXPECT_EQ(inferedge::parse_tensor_rule_list(count, kept), TensorError::kCountMismatch);
    std::string spacing = kRules;
    spacing.replace(spacing.find("2a weight"), 9, "2a  weight");
    EXPECT_EQ(inferedge::parse_tensor_rule_list(spacing, kept), TensorError::kMalformed);
    EXPECT_EQ(inferedge::parse_tensor_rule_list(std::string(16385, 'a'), kept), TensorError::kSize);
    EXPECT_EQ(inferedge::parse_tensor_rule_list(std::string(16384, 'a'), kept),
              TensorError::kMalformed);
    EXPECT_EQ(kept.list_id, "KEEP");
}

TEST(TensorCanonical, InventoryRanksUtf8AndDuplicates) {
    TensorInventory parsed;
    EXPECT_EQ(inferedge::parse_tensor_inventory(kInventory, parsed), TensorError::kOk);
    EXPECT_EQ(parsed.self_checksum, kInventorySelf);
    EXPECT_EQ(parsed.full_file_digest, kInventoryFull);
    EXPECT_EQ(parsed.entries[0].name_bytes, "a");
    EXPECT_EQ(parsed.entries[0].dimensions[0], 1u);
    TensorInventoryEntry rank8;
    rank8.name_hex = "62";
    rank8.name_bytes = "b";
    rank8.rank = 8;
    rank8.dimensions = {1, 0, 2, 3, 4, 5, 6, 7};
    std::vector<TensorInventoryEntry> entries = {parsed.entries[0], rank8};
    std::string file;
    EXPECT_EQ(inferedge::serialize_tensor_inventory("inv1", entries, file), TensorError::kOk);
    TensorInventory round;
    EXPECT_EQ(inferedge::parse_tensor_inventory(file, round), TensorError::kOk);
    EXPECT_EQ(round.entries[1].rank, 8u);
    EXPECT_EQ(round.entries[1].dimensions[1], 0u);
    TensorInventoryEntry bad = rank8;
    bad.rank = 0;
    EXPECT_EQ(inferedge::serialize_tensor_inventory(
                  "inv1", std::span<const TensorInventoryEntry>(&bad, 1), file),
              TensorError::kInvalidRank);
    bad.rank = 9;
    EXPECT_EQ(inferedge::serialize_tensor_inventory(
                  "inv1", std::span<const TensorInventoryEntry>(&bad, 1), file),
              TensorError::kInvalidRank);
    bad.rank = 2;
    bad.dimensions = {1};
    EXPECT_EQ(inferedge::serialize_tensor_inventory(
                  "inv1", std::span<const TensorInventoryEntry>(&bad, 1), file),
              TensorError::kDimensionMismatch);
    entries[1].name_hex = entries[0].name_hex;
    entries[1].name_bytes = entries[0].name_bytes;
    EXPECT_EQ(inferedge::serialize_tensor_inventory("inv1", entries, file),
              TensorError::kDuplicate);
    TensorInventoryEntry utf8;
    utf8.name_hex = "c3a9";
    utf8.name_bytes = "\xC3\xA9";
    utf8.rank = 1;
    utf8.dimensions = {4};
    EXPECT_EQ(inferedge::serialize_tensor_inventory(
                  "inv1", std::span<const TensorInventoryEntry>(&utf8, 1), file),
              TensorError::kOk);
    EXPECT_EQ(inferedge::parse_tensor_inventory(file, round), TensorError::kOk);
    EXPECT_EQ(round.entries[0].name_bytes, utf8.name_bytes);
    std::string count = kInventory;
    count.replace(count.find("tensor_count=1"), 14, "tensor_count=2");
    TensorInventory kept;
    kept.inventory_id = "KEEP";
    EXPECT_EQ(inferedge::parse_tensor_inventory(count, kept), TensorError::kCountMismatch);
    std::string line = kInventory;
    line.replace(line.find("61 1 1"), 6, "61 1");
    EXPECT_EQ(inferedge::parse_tensor_inventory(line, kept), TensorError::kMalformed);
    std::string none = kInventory;
    none.replace(none.find("61 1 1"), 6, "61 1 none");
    EXPECT_EQ(inferedge::parse_tensor_inventory(none, kept), TensorError::kLexical);
    std::string rank0 = kInventory;
    rank0.replace(rank0.find("61 1 1"), 6, "61 0 1");
    EXPECT_EQ(inferedge::parse_tensor_inventory(rank0, kept), TensorError::kInvalidRank);
    EXPECT_EQ(kept.inventory_id, "KEEP");
}

TEST(TensorCanonical, HistogramSums) {
    TensorHistogram parsed;
    EXPECT_EQ(inferedge::parse_tensor_histogram(kHistogram, 1, parsed), TensorError::kOk);
    EXPECT_EQ(parsed.self_checksum, kHistogramSelf);
    EXPECT_EQ(parsed.full_file_digest, kHistogramFull);
    TensorHistogram kept;
    kept.histogram_id = "KEEP";
    EXPECT_EQ(inferedge::parse_tensor_histogram(kHistogram, 2, kept),
              TensorError::kHistogramMismatch);
    EXPECT_EQ(kept.histogram_id, "KEEP");
    const char* empty = "schema_version=1\n"
                        "record_kind=tensor_histogram\n"
                        "histogram_id=hist0\n"
                        "type_count=0\n";
    std::string empty_file = empty;
    // The checksum is produced by the serializer so the zero-type grammar is accepted.
    EXPECT_EQ(inferedge::serialize_tensor_histogram("hist0", {}, empty_file), TensorError::kOk);
    EXPECT_EQ(inferedge::parse_tensor_histogram(empty_file, 0, parsed), TensorError::kOk);
    EXPECT_EQ(parsed.type_count, 0u);
    EXPECT_EQ(inferedge::parse_tensor_histogram(empty_file, 1, kept),
              TensorError::kHistogramMismatch);
    TensorHistogramEntry dup{"f32", 1};
    std::vector<TensorHistogramEntry> entries{dup, dup};
    std::string file;
    EXPECT_EQ(inferedge::serialize_tensor_histogram("hist1", entries, file),
              TensorError::kDuplicate);
    TensorHistogramEntry bad{"a=b", 1};
    EXPECT_EQ(inferedge::serialize_tensor_histogram(
                  "hist1", std::span<const TensorHistogramEntry>(&bad, 1), file),
              TensorError::kLexical);
    entries = {TensorHistogramEntry{"f32", 3000000000u}, TensorHistogramEntry{"q8", 3000000000u}};
    EXPECT_EQ(inferedge::serialize_tensor_histogram("hist1", entries, file), TensorError::kOk);
    EXPECT_EQ(inferedge::parse_tensor_histogram(file, 0, kept), TensorError::kOverflow);
    std::string overflow = file;
    overflow.replace(overflow.find("3000000000"), 10, "4294967296");
    EXPECT_EQ(inferedge::parse_tensor_histogram(overflow, 0, kept), TensorError::kOverflow);
    std::string count = kHistogram;
    count.replace(count.find("type_count=1"), 12, "type_count=2");
    EXPECT_EQ(inferedge::parse_tensor_histogram(count, 1, kept), TensorError::kCountMismatch);
    EXPECT_EQ(kept.histogram_id, "KEEP");
}

TEST(TensorCanonical, BundleCrossCheck) {
    TensorBundle bundle;
    EXPECT_EQ(inferedge::validate_tensor_bundle(kSchema, kRules, kInventory, kHistogram, 1, bundle),
              TensorError::kOk);
    EXPECT_EQ(bundle.schema.full_file_digest, kSchemaFull);
    EXPECT_EQ(bundle.rules.full_file_digest, kRulesFull);
    TensorBundle kept;
    kept.schema.schema_id = "KEEP";
    TensorSchema wrong = bundle.schema;
    wrong.rule_list_id = "rules2";
    std::string schema;
    EXPECT_EQ(inferedge::serialize_tensor_schema(wrong, schema), TensorError::kOk);
    EXPECT_EQ(inferedge::validate_tensor_bundle(schema, kRules, kInventory, kHistogram, 1, kept),
              TensorError::kChildIdMismatch);
    wrong = bundle.schema;
    wrong.rule_list_file_size = 1;
    EXPECT_EQ(inferedge::serialize_tensor_schema(wrong, schema), TensorError::kOk);
    EXPECT_EQ(inferedge::validate_tensor_bundle(schema, kRules, kInventory, kHistogram, 1, kept),
              TensorError::kChildSizeMismatch);
    wrong = bundle.schema;
    wrong.rule_list_file_sha256 = std::string(64, 'a');
    EXPECT_EQ(inferedge::serialize_tensor_schema(wrong, schema), TensorError::kOk);
    EXPECT_EQ(inferedge::validate_tensor_bundle(schema, kRules, kInventory, kHistogram, 1, kept),
              TensorError::kChildDigestMismatch);
    EXPECT_EQ(inferedge::validate_tensor_bundle(kSchema, kInventory, kRules, kHistogram, 1, kept),
              TensorError::kSchema);
    std::string altered = kRules;
    altered[altered.find('2')] = '3';
    EXPECT_EQ(inferedge::validate_tensor_bundle(kSchema, altered, kInventory, kHistogram, 1, kept),
              TensorError::kChecksum);
    EXPECT_EQ(inferedge::validate_tensor_bundle(kSchema, "", kInventory, kHistogram, 1, kept),
              TensorError::kMalformed);
    EXPECT_EQ(inferedge::validate_tensor_bundle(kSchema, kRules, kInventory, kHistogram, 9, kept),
              TensorError::kHistogramMismatch);
    EXPECT_EQ(kept.schema.schema_id, "KEEP");
}

TEST(TensorCanonical, CanonicalByteFailures) {
    TensorRuleList kept;
    kept.list_id = "KEEP";
    const std::string text = kRules;
    EXPECT_EQ(inferedge::parse_tensor_rule_list(text.substr(0, text.size() - 1), kept),
              TensorError::kMalformed);
    EXPECT_EQ(inferedge::parse_tensor_rule_list(text + "\n", kept), TensorError::kMalformed);
    std::string crlf = text;
    crlf.insert(crlf.find('\n'), "\r");
    EXPECT_EQ(inferedge::parse_tensor_rule_list(crlf, kept), TensorError::kMalformed);
    std::string blank = text;
    blank.insert(blank.find('\n') + 1, "\n");
    EXPECT_EQ(inferedge::parse_tensor_rule_list(blank, kept), TensorError::kMalformed);
    std::string tab = text;
    tab.insert(tab.find('='), "\t");
    EXPECT_EQ(inferedge::parse_tensor_rule_list(tab, kept), TensorError::kMalformed);
    std::string nul = text;
    nul.insert(nul.find('='), std::string(1, '\0'));
    EXPECT_EQ(inferedge::parse_tensor_rule_list(nul, kept), TensorError::kMalformed);
    EXPECT_EQ(inferedge::parse_tensor_rule_list(text + "x", kept), TensorError::kMalformed);
    const auto checksum = text.rfind("record_sha256=");
    std::string early = text.substr(0, text.find("2a weight")) + text.substr(checksum) +
                        text.substr(text.find("2a weight"), checksum - text.find("2a weight"));
    EXPECT_EQ(inferedge::parse_tensor_rule_list(early, kept), TensorError::kChecksum);
    EXPECT_EQ(inferedge::parse_tensor_rule_list(text + text.substr(checksum), kept),
              TensorError::kChecksum);
    EXPECT_EQ(kept.list_id, "KEEP");
}
