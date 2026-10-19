#include "inferedge/admission_record.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace {
using inferedge::AdmissionError;
using inferedge::CanonicalRecord;
using inferedge::ClosedSchema;
using inferedge::Int32List;
using inferedge::Uint64List;

constexpr std::string_view kDemoPrefix = "schema_version=1\nrecord_kind=demo\n";
constexpr std::string_view kDemoSelf =
    "ba417b90b6b618d9301e7791487e8eeb1b762b555d374c5eab912a604237cf3a";
constexpr std::string_view kDemoFull =
    "86b66575a20c7635f5d14cf013ad57d6f3c7b45fa6cfecc5984380592c90a866";
constexpr std::string_view kAbcDigest =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
constexpr std::string_view kLongDigest =
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
constexpr std::string_view kBoundSelf =
    "a92d95b92d5d5c4c57ec745292389ad80ecc7441818747c4b90ae480197052c0";
constexpr std::string_view kBoundFull =
    "3e7ba5854a18054f8333891f2191a73d0a35ab23b623683632ab1ee491cc3495";

ClosedSchema demo_schema() {
    return ClosedSchema{{"schema_version", "record_kind", "record_sha256"}, "demo"};
}

CanonicalRecord sentinel_record() {
    CanonicalRecord record;
    record.keys = {"sentinel-key"};
    record.values = {"sentinel-value"};
    record.bytes = "SENTINEL";
    record.self_checksum = "sentinel-self";
    record.full_file_digest = "sentinel-full";
    return record;
}

bool same_record(const CanonicalRecord& left, const CanonicalRecord& right) {
    return left.keys == right.keys && left.values == right.values && left.bytes == right.bytes &&
           left.self_checksum == right.self_checksum &&
           left.full_file_digest == right.full_file_digest;
}

std::string demo_file() {
    return std::string{kDemoPrefix} + "record_sha256=" + std::string{kDemoSelf} + "\n";
}

} // namespace

TEST(AdmissionRecord, DigestHexMatchesIndependentVectors) {
    EXPECT_EQ(inferedge::digest_hex(""), inferedge::kEmptyByteDigest);
    EXPECT_EQ(inferedge::digest_hex("abc"), kAbcDigest);
    EXPECT_EQ(inferedge::digest_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
              kLongDigest);
    EXPECT_EQ(inferedge::digest_hex(kDemoPrefix), kDemoSelf);
    const std::string full = demo_file();
    EXPECT_EQ(inferedge::digest_hex(full), kDemoFull);
    EXPECT_NE(std::string{kDemoSelf}, std::string{kDemoFull});
    EXPECT_NE(std::string{inferedge::kZeroDigest}, std::string{inferedge::kEmptyByteDigest});
}

TEST(AdmissionRecord, ValidMinimalClosedRecord) {
    const ClosedSchema schema = demo_schema();
    const std::string file = demo_file();
    CanonicalRecord record = sentinel_record();
    EXPECT_EQ(inferedge::parse_canonical_record(file, schema, record), AdmissionError::kOk);
    EXPECT_EQ(record.keys, schema.keys);
    EXPECT_EQ(record.values.at(0), "1");
    EXPECT_EQ(record.values.at(1), "demo");
    EXPECT_EQ(record.values.at(2), kDemoSelf);
    EXPECT_EQ(record.bytes, file);
    EXPECT_EQ(record.self_checksum, kDemoSelf);
    EXPECT_EQ(record.full_file_digest, kDemoFull);
    EXPECT_NE(record.self_checksum, record.full_file_digest);
}

TEST(AdmissionRecord, SerializeIsDeterministicAndRoundTrips) {
    const ClosedSchema schema = demo_schema();
    const std::string_view values[] = {"1", "demo"};
    std::string first = "KEEP";
    std::string second = "KEEP";
    EXPECT_EQ(inferedge::serialize_canonical_record(schema, values, first), AdmissionError::kOk);
    EXPECT_EQ(inferedge::serialize_canonical_record(schema, values, second), AdmissionError::kOk);
    EXPECT_EQ(first, second);
    EXPECT_EQ(first, demo_file());

    CanonicalRecord record = sentinel_record();
    EXPECT_EQ(inferedge::parse_canonical_record(first, schema, record), AdmissionError::kOk);
    const std::string_view again[] = {record.values[0], record.values[1]};
    std::string rewritten = "KEEP";
    EXPECT_EQ(inferedge::serialize_canonical_record(schema, again, rewritten), AdmissionError::kOk);
    EXPECT_EQ(rewritten, first);
}

TEST(AdmissionRecord, SchemaOrderIsPreserved) {
    ClosedSchema schema;
    schema.keys = {"schema_version", "zebra", "apple", "record_kind", "record_sha256"};
    schema.record_kind = "demo";
    const std::string_view values[] = {"1", "z", "a", "demo"};
    std::string file;
    EXPECT_EQ(inferedge::serialize_canonical_record(schema, values, file), AdmissionError::kOk);
    EXPECT_LT(file.find("zebra="), file.find("apple="));
    CanonicalRecord record;
    EXPECT_EQ(inferedge::parse_canonical_record(file, schema, record), AdmissionError::kOk);
    EXPECT_EQ(record.keys, schema.keys);
    EXPECT_EQ(record.values[1], "z");
    EXPECT_EQ(record.values[2], "a");
}

TEST(AdmissionRecord, OtherClosedOrderRejectsForeignKey) {
    ClosedSchema order_a;
    order_a.keys = {"schema_version", "record_kind", "only_a", "record_sha256"};
    order_a.record_kind = "demo";
    ClosedSchema order_b;
    order_b.keys = {"schema_version", "record_kind", "only_b", "record_sha256"};
    order_b.record_kind = "demo";
    const std::string_view values[] = {"1", "demo", "present"};
    std::string file_a;
    EXPECT_EQ(inferedge::serialize_canonical_record(order_a, values, file_a), AdmissionError::kOk);
    CanonicalRecord accepted;
    EXPECT_EQ(inferedge::parse_canonical_record(file_a, order_a, accepted), AdmissionError::kOk);

    CanonicalRecord untouched = sentinel_record();
    const CanonicalRecord before = untouched;
    EXPECT_EQ(inferedge::parse_canonical_record(file_a, order_b, untouched),
              AdmissionError::kUnknownKey);
    EXPECT_TRUE(same_record(untouched, before));
}

TEST(AdmissionRecord, ListElementLineIsNotAKeyAndIsNotReordered) {
    const std::string scrambled = "record_kind=demo\n"
                                  "schema_version=1\n"
                                  "entry_kind entry_id entry_path\n";
    CanonicalRecord untouched = sentinel_record();
    const CanonicalRecord before = untouched;
    EXPECT_EQ(inferedge::parse_canonical_record(scrambled, demo_schema(), untouched),
              AdmissionError::kMalformed);
    EXPECT_TRUE(same_record(untouched, before));
    EXPECT_NE(scrambled, demo_file());
}

TEST(AdmissionRecord, FailedParseDoesNotMutateAcceptedRecord) {
    CanonicalRecord accepted;
    EXPECT_EQ(inferedge::parse_canonical_record(demo_file(), demo_schema(), accepted),
              AdmissionError::kOk);
    const CanonicalRecord before = accepted;
    EXPECT_EQ(inferedge::parse_canonical_record("schema_version=1", demo_schema(), accepted),
              AdmissionError::kMalformed);
    EXPECT_TRUE(same_record(accepted, before));
}

TEST(AdmissionRecord, FailedSerializeDoesNotExposePartialBytes) {
    std::string out = "KEEP";
    const std::string_view values[] = {"01", "demo"};
    EXPECT_EQ(inferedge::serialize_canonical_record(demo_schema(), values, out),
              AdmissionError::kSchema);
    EXPECT_EQ(out, "KEEP");

    const std::string_view spaced[] = {"1", "de mo"};
    EXPECT_EQ(inferedge::serialize_canonical_record(demo_schema(), spaced, out),
              AdmissionError::kMalformed);
    EXPECT_EQ(out, "KEEP");
}

TEST(AdmissionRecord, RejectsEmptyAndOversizeBeforeAccepting) {
    CanonicalRecord untouched = sentinel_record();
    const CanonicalRecord before = untouched;
    EXPECT_EQ(inferedge::parse_canonical_record("", demo_schema(), untouched),
              AdmissionError::kSize);
    EXPECT_TRUE(same_record(untouched, before));

    const std::string too_big(inferedge::kMaxCanonicalRecordBytes + 1, 'a');
    EXPECT_EQ(inferedge::parse_canonical_record(too_big, demo_schema(), untouched),
              AdmissionError::kSize);
    EXPECT_TRUE(same_record(untouched, before));

    const std::string exact_garbage(inferedge::kMaxCanonicalRecordBytes, 'a');
    EXPECT_EQ(inferedge::parse_canonical_record(exact_garbage, demo_schema(), untouched),
              AdmissionError::kMalformed);
    EXPECT_TRUE(same_record(untouched, before));
}

TEST(AdmissionRecord, AcceptsExactSizeLimitAndRejectsOnePastIt) {
    ClosedSchema schema;
    schema.keys = {"schema_version", "record_kind", "note", "record_sha256"};
    schema.record_kind = "demo";
    const std::string note(16265, 'a');
    const std::string_view values[] = {"1", "demo", note};
    std::string file = "KEEP";
    EXPECT_EQ(inferedge::serialize_canonical_record(schema, values, file), AdmissionError::kOk);
    EXPECT_EQ(file.size(), inferedge::kMaxCanonicalRecordBytes);
    EXPECT_EQ(file.substr(file.size() - 65, 64), kBoundSelf);
    EXPECT_EQ(inferedge::digest_hex(file), kBoundFull);
    EXPECT_NE(std::string{kBoundSelf}, std::string{kBoundFull});

    CanonicalRecord record;
    EXPECT_EQ(inferedge::parse_canonical_record(file, schema, record), AdmissionError::kOk);
    EXPECT_EQ(record.self_checksum, kBoundSelf);
    EXPECT_EQ(record.full_file_digest, kBoundFull);

    std::string bigger = file;
    bigger.push_back('x');
    EXPECT_EQ(bigger.size(), inferedge::kMaxCanonicalRecordBytes + 1);
    CanonicalRecord untouched = sentinel_record();
    const CanonicalRecord before = untouched;
    EXPECT_EQ(inferedge::parse_canonical_record(bigger, schema, untouched), AdmissionError::kSize);
    EXPECT_TRUE(same_record(untouched, before));

    const std::string too_long(16266, 'a');
    const std::string_view oversized[] = {"1", "demo", too_long};
    std::string kept = "KEEP";
    EXPECT_EQ(inferedge::serialize_canonical_record(schema, oversized, kept),
              AdmissionError::kSize);
    EXPECT_EQ(kept, "KEEP");
}

TEST(AdmissionRecord, RejectsMalformedEncodings) {
    const std::string file = demo_file();
    const CanonicalRecord before = sentinel_record();
    const std::vector<std::string> samples = {
        file.substr(0, file.size() - 1),
        file + "\n",
        "schema_version=1\n\nrecord_kind=demo\nrecord_sha256=" + std::string(64, '0') + "\n",
        std::string{"schema_version=1\r\nrecord_kind=demo\n"},
        std::string{"schema_version=1\nrecord_kind=de\rmo\n"},
        std::string{"schema_version=1\nrecord_kind=de mo\n"},
        std::string{"schema_version=1\nrecord_kind=de\tmo\n"},
        std::string{"# comment\nschema_version=1\n"},
        std::string{"schema_version=1\nnot-a-pair\n"},
        std::string{"=demo\n"},
        std::string{"schema_version=1\nrecord_kind=demo\nrecord_sha256="} + std::string(64, 'a') +
            std::string{"\nextra"},
        std::string{"schema_version=1\nrecord_kind=demo\xC3\n"},
    };
    for (const std::string& sample : samples) {
        CanonicalRecord untouched = before;
        EXPECT_EQ(inferedge::parse_canonical_record(sample, demo_schema(), untouched),
                  AdmissionError::kMalformed);
        EXPECT_TRUE(same_record(untouched, before));
    }

    std::string with_nul = file;
    with_nul.insert(with_nul.begin() + 4, '\0');
    CanonicalRecord untouched = before;
    EXPECT_EQ(inferedge::parse_canonical_record(with_nul, demo_schema(), untouched),
              AdmissionError::kMalformed);
    EXPECT_TRUE(same_record(untouched, before));
}

TEST(AdmissionRecord, RejectsStructuralKeyFailures) {
    const CanonicalRecord before = sentinel_record();
    auto expect = [&](std::string_view bytes, AdmissionError error) {
        CanonicalRecord untouched = before;
        EXPECT_EQ(inferedge::parse_canonical_record(bytes, demo_schema(), untouched), error);
        EXPECT_TRUE(same_record(untouched, before));
    };

    expect("schema_version=1\nrecord_sha256=" + std::string(64, 'a') + "\n",
           AdmissionError::kMissingKey);
    expect("schema_version=1\nrecord_kind=demo\nrecord_kind=demo\nrecord_sha256=" +
               std::string(64, 'a') + "\n",
           AdmissionError::kDuplicateKey);
    expect("schema_version=1\nrecord_kind=demo\nextra=1\nrecord_sha256=" + std::string(64, 'a') +
               "\n",
           AdmissionError::kUnknownKey);
    expect("record_kind=demo\nschema_version=1\nrecord_sha256=" + std::string(64, 'a') + "\n",
           AdmissionError::kSchema);
    expect("schema_version=1\nrecord_sha256=" + std::string(64, 'a') + "\nrecord_kind=demo\n",
           AdmissionError::kSchema);
    expect("schema_version=1\nrecord_kind=demo\nrecord_sha256=" + std::string(64, 'b') +
               "\nrecord_sha256=" + std::string(64, 'c') + "\n",
           AdmissionError::kDuplicateKey);
    expect("schema_version=2\nrecord_kind=demo\nrecord_sha256=" + std::string(64, 'd') + "\n",
           AdmissionError::kSchema);
    expect("schema_version=1\nrecord_kind=other\nrecord_sha256=" + std::string(64, 'e') + "\n",
           AdmissionError::kSchema);
    expect("Schema_version=1\nrecord_kind=demo\nrecord_sha256=" + std::string(64, 'f') + "\n",
           AdmissionError::kUnknownKey);
}

TEST(AdmissionRecord, RejectsChecksumFormatAndMismatch) {
    const CanonicalRecord before = sentinel_record();
    auto expect = [&](std::string checksum, AdmissionError error) {
        const std::string bytes = std::string{kDemoPrefix} + "record_sha256=" + checksum + "\n";
        CanonicalRecord untouched = before;
        EXPECT_EQ(inferedge::parse_canonical_record(bytes, demo_schema(), untouched), error);
        EXPECT_TRUE(same_record(untouched, before));
    };
    expect(std::string(64, 'A'), AdmissionError::kChecksumFormat);
    expect(std::string(63, 'a'), AdmissionError::kChecksumFormat);
    expect(std::string(65, 'a'), AdmissionError::kChecksumFormat);
    expect(std::string(64, '0'), AdmissionError::kChecksumMismatch);
}

TEST(AdmissionRecord, SchemaWithoutFinalChecksumIsRejected) {
    ClosedSchema schema = demo_schema();
    schema.keys = {"record_sha256", "schema_version", "record_kind"};
    CanonicalRecord untouched = sentinel_record();
    const CanonicalRecord before = untouched;
    EXPECT_EQ(inferedge::check_closed_schema(schema), AdmissionError::kSchema);
    EXPECT_EQ(inferedge::parse_canonical_record(demo_file(), schema, untouched),
              AdmissionError::kSchema);
    EXPECT_TRUE(same_record(untouched, before));
    std::string out = "KEEP";
    const std::string_view values[] = {"1", "demo"};
    EXPECT_EQ(inferedge::serialize_canonical_record(schema, values, out), AdmissionError::kSchema);
    EXPECT_EQ(out, "KEEP");
}

TEST(AdmissionRecord, ProjectIdAndCompoundIdBoundaries) {
    EXPECT_EQ(inferedge::validate_project_id("a"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_project_id("9abc"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_project_id("a.b-c_d"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_project_id(std::string(128, 'a')), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_project_id(""), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_project_id(std::string(129, 'a')), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_project_id("A"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_project_id("none"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_project_id("unknown"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_project_id(".a"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_project_id("a."), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_project_id("a..b"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_project_id("a b"), AdmissionError::kLexical);

    EXPECT_EQ(inferedge::validate_id_or_none("none"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_id_or_none("a.b"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_id_or_none("unknown"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_id_or_none("None"), AdmissionError::kLexical);

    std::string made = "KEEP";
    EXPECT_EQ(inferedge::make_compound_id("id-", "a", made), AdmissionError::kOk);
    EXPECT_EQ(made, "id-a");
    EXPECT_EQ(inferedge::make_compound_id("id-", std::string(96, 'b'), made), AdmissionError::kOk);
    EXPECT_EQ(made.size(), 99u);

    made = "KEEP";
    EXPECT_EQ(inferedge::make_compound_id("id-", std::string(97, 'b'), made),
              AdmissionError::kLexical);
    EXPECT_EQ(made, "KEEP");
    EXPECT_EQ(inferedge::make_compound_id("id-", "", made), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::make_compound_id("id-", "none", made), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::make_compound_id("no", "ne", made), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::make_compound_id("unk", "nown", made), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::make_compound_id("a..", "b", made), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::make_compound_id(".", "a", made), AdmissionError::kLexical);
    EXPECT_EQ(made, "KEEP");
    EXPECT_EQ(inferedge::make_compound_id(std::string(40, 'a'), std::string(96, 'b'), made),
              AdmissionError::kLexical);
    EXPECT_EQ(made, "KEEP");
}

TEST(AdmissionRecord, IntegerAndListBoundaries) {
    std::uint32_t u32 = 7;
    EXPECT_EQ(inferedge::parse_uint("0", u32), AdmissionError::kOk);
    EXPECT_EQ(u32, 0u);
    EXPECT_EQ(inferedge::parse_uint("4294967295", u32), AdmissionError::kOk);
    EXPECT_EQ(u32, 4294967295u);
    EXPECT_EQ(inferedge::parse_uint("4294967296", u32), AdmissionError::kOverflow);
    EXPECT_EQ(u32, 4294967295u);
    EXPECT_EQ(inferedge::parse_uint("00", u32), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::parse_uint("01", u32), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::parse_uint("-1", u32), AdmissionError::kLexical);
    EXPECT_EQ(u32, 4294967295u);

    std::uint64_t u64 = 7;
    EXPECT_EQ(inferedge::parse_uint64("0", u64), AdmissionError::kOk);
    EXPECT_EQ(u64, 0u);
    EXPECT_EQ(inferedge::parse_uint64("18446744073709551615", u64), AdmissionError::kOk);
    EXPECT_EQ(u64, std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(inferedge::parse_uint64("18446744073709551616", u64), AdmissionError::kOverflow);
    EXPECT_EQ(u64, std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(inferedge::parse_uint64("00", u64), AdmissionError::kLexical);
    EXPECT_EQ(u64, std::numeric_limits<std::uint64_t>::max());

    std::int32_t i32 = 7;
    EXPECT_EQ(inferedge::parse_int32("0", i32), AdmissionError::kOk);
    EXPECT_EQ(i32, 0);
    EXPECT_EQ(inferedge::parse_int32("2147483647", i32), AdmissionError::kOk);
    EXPECT_EQ(i32, 2147483647);
    EXPECT_EQ(inferedge::parse_int32("-2147483648", i32), AdmissionError::kOk);
    EXPECT_EQ(i32, std::numeric_limits<std::int32_t>::min());
    EXPECT_EQ(inferedge::parse_int32("2147483648", i32), AdmissionError::kOverflow);
    EXPECT_EQ(i32, std::numeric_limits<std::int32_t>::min());
    EXPECT_EQ(inferedge::parse_int32("-2147483649", i32), AdmissionError::kOverflow);
    EXPECT_EQ(inferedge::parse_int32("-0", i32), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::parse_int32("+1", i32), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::parse_int32("01", i32), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::parse_int32("-01", i32), AdmissionError::kLexical);
    EXPECT_EQ(i32, std::numeric_limits<std::int32_t>::min());

    Int32List list;
    list.is_none = true;
    EXPECT_EQ(inferedge::parse_int32_list("none", list), AdmissionError::kOk);
    EXPECT_TRUE(list.is_none);
    EXPECT_TRUE(list.values.empty());
    EXPECT_EQ(inferedge::parse_int32_list("-2147483648,0,2147483647", list), AdmissionError::kOk);
    EXPECT_FALSE(list.is_none);
    ASSERT_EQ(list.values.size(), 3u);
    EXPECT_EQ(list.values[0], std::numeric_limits<std::int32_t>::min());
    EXPECT_EQ(list.values[2], 2147483647);

    std::string many;
    for (int index = 0; index < 4096; ++index) {
        if (index != 0) {
            many.push_back(',');
        }
        many.push_back('1');
    }
    EXPECT_EQ(inferedge::parse_int32_list(many, list), AdmissionError::kOk);
    EXPECT_EQ(list.values.size(), 4096u);
    many.push_back(',');
    many.push_back('1');
    const Int32List previous = list;
    EXPECT_EQ(inferedge::parse_int32_list(many, list), AdmissionError::kLexical);
    EXPECT_EQ(list.values.size(), previous.values.size());
    EXPECT_EQ(inferedge::parse_int32_list("1,02", list), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::parse_int32_list("1,", list), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::parse_int32_list("1, 2", list), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::parse_int32_list("", list), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::parse_int32_list("1,2147483648", list), AdmissionError::kOverflow);
    EXPECT_EQ(list.values.size(), previous.values.size());

    Uint64List wide;
    wide.is_none = false;
    wide.values = {9};
    EXPECT_EQ(inferedge::parse_uint64_list("none", wide), AdmissionError::kOk);
    EXPECT_TRUE(wide.is_none);
    EXPECT_EQ(inferedge::parse_uint64_list("0,18446744073709551615", wide), AdmissionError::kOk);
    EXPECT_EQ(wide.values.size(), 2u);
    EXPECT_EQ(inferedge::parse_uint64_list("0,0,0,0,0,0,0,0", wide), AdmissionError::kOk);
    EXPECT_EQ(wide.values.size(), 8u);
    const Uint64List saved = wide;
    EXPECT_EQ(inferedge::parse_uint64_list("0,0,0,0,0,0,0,0,0", wide), AdmissionError::kLexical);
    EXPECT_EQ(wide.values.size(), saved.values.size());
    EXPECT_EQ(inferedge::parse_uint64_list("00", wide), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::parse_uint64_list("1,18446744073709551616", wide),
              AdmissionError::kOverflow);
    EXPECT_EQ(wide.values, saved.values);
}

TEST(AdmissionRecord, DigestLexicalRulesKeepZeroDistinct) {
    EXPECT_EQ(inferedge::validate_sha256(inferedge::kZeroDigest), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_sha256(inferedge::kEmptyByteDigest), AdmissionError::kOk);
    EXPECT_TRUE(inferedge::is_zero_digest(inferedge::kZeroDigest));
    EXPECT_FALSE(inferedge::is_zero_digest(inferedge::kEmptyByteDigest));
    EXPECT_TRUE(inferedge::is_empty_byte_digest(inferedge::kEmptyByteDigest));
    EXPECT_FALSE(inferedge::is_empty_byte_digest(inferedge::kZeroDigest));
    EXPECT_EQ(inferedge::validate_sha256(std::string(63, 'a')), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_sha256(std::string(65, 'a')), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_sha256(std::string(64, 'A')), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_sha256(std::string(64, 'g')), AdmissionError::kLexical);

    EXPECT_EQ(inferedge::validate_digest_or_zero(inferedge::kZeroDigest, true),
              AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_digest_or_zero(inferedge::kZeroDigest, false),
              AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_digest_or_zero(inferedge::kEmptyByteDigest, false),
              AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_digest_or_zero(inferedge::kEmptyByteDigest, true),
              AdmissionError::kOk);
}

TEST(AdmissionRecord, DateAndCalendarAdditionBoundaries) {
    EXPECT_EQ(inferedge::validate_date("2020-01-01"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_date("2100-12-31"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_date("2020-02-29"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_date("2024-02-29"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_date("2021-02-29"), AdmissionError::kInvalidDate);
    EXPECT_EQ(inferedge::validate_date("2100-02-29"), AdmissionError::kInvalidDate);
    EXPECT_EQ(inferedge::validate_date("2019-12-31"), AdmissionError::kInvalidDate);
    EXPECT_EQ(inferedge::validate_date("2101-01-01"), AdmissionError::kInvalidDate);
    EXPECT_EQ(inferedge::validate_date("2020-00-01"), AdmissionError::kInvalidDate);
    EXPECT_EQ(inferedge::validate_date("2020-13-01"), AdmissionError::kInvalidDate);
    EXPECT_EQ(inferedge::validate_date("2020-01-00"), AdmissionError::kInvalidDate);
    EXPECT_EQ(inferedge::validate_date("2020-01-32"), AdmissionError::kInvalidDate);
    EXPECT_EQ(inferedge::validate_date("2020-1-01"), AdmissionError::kInvalidDate);
    EXPECT_EQ(inferedge::validate_date("2020-01-1"), AdmissionError::kInvalidDate);

    std::string out = "KEEP";
    EXPECT_EQ(inferedge::add_calendar_days("2020-01-01", 0, out), AdmissionError::kOk);
    EXPECT_EQ(out, "2020-01-01");
    EXPECT_EQ(inferedge::add_calendar_days("2020-01-01", 31, out), AdmissionError::kOk);
    EXPECT_EQ(out, "2020-02-01");
    EXPECT_EQ(inferedge::add_calendar_days("2020-02-28", 1, out), AdmissionError::kOk);
    EXPECT_EQ(out, "2020-02-29");
    EXPECT_EQ(inferedge::add_calendar_days("2020-02-28", 2, out), AdmissionError::kOk);
    EXPECT_EQ(out, "2020-03-01");
    EXPECT_EQ(inferedge::add_calendar_days("2021-02-28", 1, out), AdmissionError::kOk);
    EXPECT_EQ(out, "2021-03-01");
    EXPECT_EQ(inferedge::add_calendar_days("2024-02-29", 1, out), AdmissionError::kOk);
    EXPECT_EQ(out, "2024-03-01");
    EXPECT_EQ(inferedge::add_calendar_days("2020-01-01", 1520, out), AdmissionError::kOk);
    EXPECT_EQ(out, "2024-02-29");
    EXPECT_EQ(inferedge::add_calendar_days("2020-01-01", 29584, out), AdmissionError::kOk);
    EXPECT_EQ(out, "2100-12-31");
    EXPECT_EQ(inferedge::add_calendar_days("2100-12-31", 0, out), AdmissionError::kOk);
    EXPECT_EQ(out, "2100-12-31");
    EXPECT_EQ(inferedge::add_calendar_days("2100-02-28", 1, out), AdmissionError::kOk);
    EXPECT_EQ(out, "2100-03-01");

    out = "KEEP";
    EXPECT_EQ(inferedge::add_calendar_days("2100-12-31", 1, out), AdmissionError::kInvalidDate);
    EXPECT_EQ(out, "KEEP");
    EXPECT_EQ(inferedge::add_calendar_days("2019-12-31", 1, out), AdmissionError::kInvalidDate);
    EXPECT_EQ(inferedge::add_calendar_days("2021-02-29", 1, out), AdmissionError::kInvalidDate);
    EXPECT_EQ(inferedge::add_calendar_days("2020-01-01", 29585, out), AdmissionError::kInvalidDate);
    EXPECT_EQ(out, "KEEP");
}

TEST(AdmissionRecord, CheckedUint64Addition) {
    std::uint64_t sum = 7;
    EXPECT_EQ(inferedge::add_uint64(0, 0, sum), AdmissionError::kOk);
    EXPECT_EQ(sum, 0u);
    EXPECT_EQ(inferedge::add_uint64(1, 2, sum), AdmissionError::kOk);
    EXPECT_EQ(sum, 3u);
    const auto max = std::numeric_limits<std::uint64_t>::max();
    EXPECT_EQ(inferedge::add_uint64(max - 5, 5, sum), AdmissionError::kOk);
    EXPECT_EQ(sum, max);
    EXPECT_EQ(inferedge::add_uint64(max, 0, sum), AdmissionError::kOk);
    EXPECT_EQ(sum, max);
    sum = 7;
    EXPECT_EQ(inferedge::add_uint64(max, 1, sum), AdmissionError::kOverflow);
    EXPECT_EQ(sum, 7u);
    EXPECT_EQ(inferedge::add_uint64(max - 5, 6, sum), AdmissionError::kOverflow);
    EXPECT_EQ(sum, 7u);
}

TEST(AdmissionRecord, FilenameAndUpstreamNameBoundaries) {
    EXPECT_EQ(inferedge::validate_filename("a"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_filename("A"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_filename("qwen3-0.6b-Q4_K_M.gguf"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_filename(std::string(255, 'a')), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_filename(""), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_filename(std::string(256, 'a')), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_filename(".hidden"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_filename("a/b"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_filename("a b"), AdmissionError::kLexical);

    EXPECT_EQ(inferedge::validate_upstream_name("Qwen/Qwen3-0.6B"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_upstream_name("a"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_upstream_name("a+b"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_upstream_name("a/b"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_upstream_name(std::string(128, 'a')), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_upstream_name(""), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_upstream_name(std::string(129, 'a')), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_upstream_name("/Qwen"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_upstream_name("Qwen/"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_upstream_name("Qwen//A"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_upstream_name("a/./b"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_upstream_name("a/../b"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_upstream_name("."), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_upstream_name(".."), AdmissionError::kLexical);
}

TEST(AdmissionRecord, PhaseYesNoAndErrorNames) {
    EXPECT_EQ(inferedge::validate_yes_no("yes"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_yes_no("no"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_yes_no("Yes"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_yes_no("true"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_yes_no(""), AdmissionError::kLexical);

    EXPECT_EQ(inferedge::validate_phase_id("9-c1"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_phase_id("9-c9"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_phase_id("9-c10"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_phase_id("9-c44"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_phase_id("9-c0"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_phase_id("9-c01"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_phase_id("9-c45"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_phase_id("9-c"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_phase_id("9-C1"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_phase_id("9-c044"), AdmissionError::kLexical);

    const AdmissionError categories[] = {
        AdmissionError::kOk,
        AdmissionError::kSize,
        AdmissionError::kMalformed,
        AdmissionError::kSchema,
        AdmissionError::kDuplicateKey,
        AdmissionError::kUnknownKey,
        AdmissionError::kMissingKey,
        AdmissionError::kLexical,
        AdmissionError::kChecksumFormat,
        AdmissionError::kChecksumMismatch,
        AdmissionError::kOverflow,
        AdmissionError::kInvalidDate,
        AdmissionError::kInternal,
    };
    for (AdmissionError error : categories) {
        const std::string name = inferedge::admission_error_name(error);
        EXPECT_EQ(name.find('/'), std::string::npos);
        EXPECT_EQ(name.find('\\'), std::string::npos);
        EXPECT_FALSE(name.empty());
    }
}

TEST(AdmissionRecord, PatternHexBoundaries) {
    EXPECT_EQ(inferedge::validate_pattern_hex("2a"), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_pattern_hex(std::string(512, '4')), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_pattern_hex(""), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_pattern_hex("a"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_pattern_hex(std::string(513, '4')), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_pattern_hex(std::string(514, '4')), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_pattern_hex("2A"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_pattern_hex("zz"), AdmissionError::kLexical);
    EXPECT_EQ(inferedge::validate_pattern_hex("00"), AdmissionError::kOk);
}

TEST(AdmissionRecord, NameHexUtf8Boundaries) {
    std::string decoded = "KEEP";
    EXPECT_EQ(inferedge::validate_name_hex("41", decoded), AdmissionError::kOk);
    EXPECT_EQ(decoded, "A");
    EXPECT_EQ(inferedge::validate_name_hex("c280", decoded), AdmissionError::kOk);
    EXPECT_EQ(decoded.size(), 2u);
    EXPECT_EQ(inferedge::validate_name_hex("e282ac", decoded), AdmissionError::kOk);
    EXPECT_EQ(decoded.size(), 3u);
    EXPECT_EQ(inferedge::validate_name_hex("f09f9880", decoded), AdmissionError::kOk);
    EXPECT_EQ(decoded.size(), 4u);
    EXPECT_EQ(inferedge::validate_name_hex("ed9fbf", decoded), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_name_hex("f48fbfbf", decoded), AdmissionError::kOk);
    EXPECT_EQ(inferedge::validate_name_hex(std::string(512, '4'), decoded), AdmissionError::kOk);

    const std::string_view rejected[] = {
        "abc",      "AA",       "0g",       "00",   "80",     "c2",     "e0a0",
        "f09080",   "c020",     "c080",     "c1bf", "e08080", "eda080", "edbfbf",
        "f4908080", "f5808080", "f08fbfbf", "",     "a",
    };
    for (std::string_view hex : rejected) {
        decoded = "KEEP";
        EXPECT_EQ(inferedge::validate_name_hex(hex, decoded), AdmissionError::kLexical) << hex;
        EXPECT_EQ(decoded, "KEEP");
    }
    decoded = "KEEP";
    EXPECT_EQ(inferedge::validate_name_hex(std::string(513, '4'), decoded),
              AdmissionError::kLexical);
    EXPECT_EQ(decoded, "KEEP");
    EXPECT_EQ(inferedge::validate_name_hex(std::string(514, '4'), decoded),
              AdmissionError::kLexical);
    EXPECT_EQ(decoded, "KEEP");
}
