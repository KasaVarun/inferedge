#include "inferedge/tensor_canonical.hpp"

#include "inferedge/admission_record.hpp"

#include <limits>
#include <utility>

namespace inferedge {
namespace {

TensorError from_admission(AdmissionError error) {
    switch (error) {
    case AdmissionError::kOk:
        return TensorError::kOk;
    case AdmissionError::kSize:
        return TensorError::kSize;
    case AdmissionError::kMalformed:
        return TensorError::kMalformed;
    case AdmissionError::kSchema:
    case AdmissionError::kDuplicateKey:
    case AdmissionError::kUnknownKey:
    case AdmissionError::kMissingKey:
        return TensorError::kSchema;
    case AdmissionError::kLexical:
        return TensorError::kLexical;
    case AdmissionError::kChecksumFormat:
    case AdmissionError::kChecksumMismatch:
        return TensorError::kChecksum;
    case AdmissionError::kOverflow:
        return TensorError::kOverflow;
    case AdmissionError::kInvalidDate:
    case AdmissionError::kInternal:
        return TensorError::kInternal;
    }
    return TensorError::kInternal;
}

int hex_value(unsigned char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    return -1;
}

TensorError decode_hex_bytes(std::string_view text, std::string& decoded) {
    std::string bytes;
    bytes.resize(text.size() / 2u);
    for (std::size_t index = 0; index < text.size(); index += 2u) {
        const int high = hex_value(static_cast<unsigned char>(text[index]));
        const int low = hex_value(static_cast<unsigned char>(text[index + 1u]));
        if (high < 0 || low < 0) {
            return TensorError::kLexical;
        }
        bytes[index / 2u] = static_cast<char>((high << 4) | low);
    }
    decoded = std::move(bytes);
    return TensorError::kOk;
}

TensorError check_utf8(std::string_view bytes) {
    std::size_t index = 0;
    while (index < bytes.size()) {
        const auto lead = static_cast<unsigned char>(bytes[index]);
        if (lead == 0x00) {
            return TensorError::kInvalidUtf8;
        }
        if (lead <= 0x7F) {
            ++index;
            continue;
        }
        std::size_t need = 0;
        std::uint32_t code = 0;
        std::uint32_t minimum = 0;
        if (lead >= 0xC2 && lead <= 0xDF) {
            need = 2;
            code = lead & 0x1Fu;
            minimum = 0x80u;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            need = 3;
            code = lead & 0x0Fu;
            minimum = 0x800u;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            need = 4;
            code = lead & 0x07u;
            minimum = 0x10000u;
        } else {
            return TensorError::kInvalidUtf8;
        }
        if (index + need > bytes.size()) {
            return TensorError::kInvalidUtf8;
        }
        for (std::size_t extra = 1; extra < need; ++extra) {
            const auto cont = static_cast<unsigned char>(bytes[index + extra]);
            if ((cont & 0xC0u) != 0x80u) {
                return TensorError::kInvalidUtf8;
            }
            code = (code << 6u) | (cont & 0x3Fu);
        }
        if (lead == 0xE0 && static_cast<unsigned char>(bytes[index + 1]) < 0xA0) {
            return TensorError::kInvalidUtf8;
        }
        if (lead == 0xED && static_cast<unsigned char>(bytes[index + 1]) >= 0xA0) {
            return TensorError::kInvalidUtf8;
        }
        if (lead == 0xF0 && static_cast<unsigned char>(bytes[index + 1]) < 0x90) {
            return TensorError::kInvalidUtf8;
        }
        if (lead == 0xF4 && static_cast<unsigned char>(bytes[index + 1]) > 0x8F) {
            return TensorError::kInvalidUtf8;
        }
        if (code < minimum || code > 0x10FFFFu || (code >= 0xD800u && code <= 0xDFFFu)) {
            return TensorError::kInvalidUtf8;
        }
        index += need;
    }
    return TensorError::kOk;
}

bool list_byte(unsigned char c) noexcept {
    return c == '\n' || c == ' ' || (c >= 0x21 && c <= 0x7E && c != '#') || c >= 0x80;
}

TensorError split_lines(std::string_view bytes, std::vector<std::string_view>& lines) {
    if (bytes.size() > kMaxCanonicalRecordBytes) {
        return TensorError::kSize;
    }
    if (bytes.empty() || bytes.back() != '\n') {
        return TensorError::kMalformed;
    }
    for (unsigned char c : bytes) {
        if (!list_byte(c)) {
            return TensorError::kMalformed;
        }
    }
    if (bytes.find("\n\n") != std::string_view::npos) {
        return TensorError::kMalformed;
    }
    std::size_t start = 0;
    while (start < bytes.size()) {
        const std::size_t end = bytes.find('\n', start);
        if (end == std::string_view::npos || end == start) {
            return TensorError::kMalformed;
        }
        lines.push_back(bytes.substr(start, end - start));
        start = end + 1;
    }
    return TensorError::kOk;
}

bool is_checksum_line(std::string_view line) {
    constexpr std::string_view kPrefix{"record_sha256="};
    return line.size() >= kPrefix.size() && line.substr(0, kPrefix.size()) == kPrefix;
}

TensorError check_self_checksum(std::string_view bytes, std::string_view checksum_line,
                                std::string& self_checksum, std::string& full_digest) {
    if (checksum_line.size() != std::string_view{"record_sha256="}.size() + 64) {
        return TensorError::kChecksum;
    }
    const std::string_view hex = checksum_line.substr(std::string_view{"record_sha256="}.size());
    if (validate_sha256(hex) != AdmissionError::kOk) {
        return TensorError::kChecksum;
    }
    if (bytes.size() < checksum_line.size() + 1) {
        return TensorError::kChecksum;
    }
    const std::string_view prefix = bytes.substr(0, bytes.size() - checksum_line.size() - 1);
    const std::string actual = digest_hex(prefix);
    if (actual != hex) {
        return TensorError::kChecksum;
    }
    self_checksum = actual;
    full_digest = digest_hex(bytes);
    return TensorError::kOk;
}

std::vector<std::string_view> split_spaces(std::string_view line, bool& spaced) {
    std::vector<std::string_view> tokens;
    spaced = line.empty() || line.front() == ' ' || line.back() == ' ';
    std::size_t start = 0;
    for (std::size_t index = 0; index <= line.size(); ++index) {
        if (index == line.size() || line[index] == ' ') {
            if (index == start) {
                spaced = true;
            }
            if (index != line.size()) {
                tokens.push_back(line.substr(start, index - start));
                start = index + 1;
            } else if (index != start) {
                tokens.push_back(line.substr(start, index - start));
            }
        }
    }
    return tokens;
}

TensorError parse_role(std::string_view text, TensorRole& role) {
    if (text == "weight") {
        role = TensorRole::kWeight;
        return TensorError::kOk;
    }
    if (text == "non_weight") {
        role = TensorRole::kNonWeight;
        return TensorError::kOk;
    }
    return TensorError::kInvalidRole;
}

bool type_name_ok(std::string_view text) {
    if (text.empty()) {
        return false;
    }
    for (unsigned char c : text) {
        if (c <= 0x20 || c == 0x7F || c == '=' || c == '#') {
            return false;
        }
    }
    return true;
}

TensorError require_project_id(std::string_view text) {
    return from_admission(validate_project_id(text));
}

TensorError require_size(std::string_view text, std::uint32_t& out) {
    return from_admission(parse_uint(text, out));
}

TensorError require_digest(std::string_view text) {
    const AdmissionError status = validate_sha256(text);
    if (status != AdmissionError::kOk) {
        return from_admission(status);
    }
    if (is_zero_digest(text)) {
        return TensorError::kLexical;
    }
    return TensorError::kOk;
}

std::string append_checksum(std::string prefix) {
    const std::string checksum = digest_hex(prefix);
    prefix.append("record_sha256=");
    prefix.append(checksum);
    prefix.push_back('\n');
    return prefix;
}

ClosedSchema schema_keys() {
    return ClosedSchema{{"schema_version", "record_kind", "schema_id", "rule_list_id",
                         "rule_list_file_size", "rule_list_file_sha256", "inventory_id",
                         "inventory_file_size", "inventory_file_sha256", "histogram_id",
                         "histogram_file_size", "histogram_file_sha256", "record_sha256"},
                        "tensor_schema"};
}

TensorError header_line(std::string_view line, std::string_view key, std::string& value) {
    const std::size_t eq = line.find('=');
    if (eq == std::string_view::npos || eq == 0 || line.find(' ') != std::string_view::npos) {
        return TensorError::kMalformed;
    }
    if (line.substr(0, eq) != key) {
        return TensorError::kSchema;
    }
    value.assign(line.substr(eq + 1));
    if (value.empty()) {
        return TensorError::kSchema;
    }
    return TensorError::kOk;
}

} // namespace

const char* tensor_error_name(TensorError error) noexcept {
    switch (error) {
    case TensorError::kOk:
        return "ok";
    case TensorError::kMalformed:
        return "malformed";
    case TensorError::kSize:
        return "size";
    case TensorError::kSchema:
        return "schema";
    case TensorError::kChecksum:
        return "checksum";
    case TensorError::kLexical:
        return "lexical";
    case TensorError::kCountMismatch:
        return "count_mismatch";
    case TensorError::kDuplicate:
        return "duplicate";
    case TensorError::kInvalidRole:
        return "invalid_role";
    case TensorError::kInvalidRank:
        return "invalid_rank";
    case TensorError::kDimensionMismatch:
        return "dimension_mismatch";
    case TensorError::kInvalidUtf8:
        return "invalid_utf8";
    case TensorError::kPattern:
        return "pattern";
    case TensorError::kOverflow:
        return "overflow";
    case TensorError::kHistogramMismatch:
        return "histogram_mismatch";
    case TensorError::kChildIdMismatch:
        return "child_id_mismatch";
    case TensorError::kChildSizeMismatch:
        return "child_size_mismatch";
    case TensorError::kChildDigestMismatch:
        return "child_digest_mismatch";
    case TensorError::kInternal:
        return "internal";
    }
    return "internal";
}

TensorError decode_pattern_hex(std::string_view pattern_hex, std::string& decoded) {
    if (validate_pattern_hex(pattern_hex) != AdmissionError::kOk) {
        return TensorError::kLexical;
    }
    std::string bytes;
    const TensorError status = decode_hex_bytes(pattern_hex, bytes);
    if (status != TensorError::kOk) {
        return status;
    }
    if (bytes.empty() || bytes.size() > 256) {
        return TensorError::kPattern;
    }
    decoded = std::move(bytes);
    return TensorError::kOk;
}

TensorError decode_name_hex(std::string_view name_hex, std::string& decoded) {
    if (validate_pattern_hex(name_hex) != AdmissionError::kOk) {
        return TensorError::kLexical;
    }
    std::string bytes;
    const TensorError decoded_hex = decode_hex_bytes(name_hex, bytes);
    if (decoded_hex != TensorError::kOk) {
        return decoded_hex;
    }
    if (bytes.empty() || bytes.size() > 256) {
        return TensorError::kPattern;
    }
    const TensorError utf8 = check_utf8(bytes);
    if (utf8 != TensorError::kOk) {
        return utf8;
    }
    std::string checked;
    if (validate_name_hex(name_hex, checked) != AdmissionError::kOk || checked != bytes) {
        return TensorError::kInvalidUtf8;
    }
    decoded = std::move(bytes);
    return TensorError::kOk;
}

TensorError match_tensor_name(std::string_view pattern_bytes, std::string_view name_bytes,
                              bool& matched) {
    if (pattern_bytes.empty() || name_bytes.empty() || pattern_bytes.size() > 256 ||
        name_bytes.size() > 256) {
        return TensorError::kPattern;
    }
    const std::size_t n = pattern_bytes.size();
    const std::size_t m = name_bytes.size();
    std::vector<char> previous(m + 1, 0);
    std::vector<char> current(m + 1, 0);
    previous[0] = 1;
    for (std::size_t i = 1; i <= n; ++i) {
        const auto pattern_byte = static_cast<unsigned char>(pattern_bytes[i - 1]);
        if (pattern_byte == 0x2A) {
            current[0] = previous[0];
            for (std::size_t j = 1; j <= m; ++j) {
                current[j] = static_cast<char>(previous[j] != 0 || current[j - 1] != 0);
            }
        } else {
            current[0] = 0;
            for (std::size_t j = 1; j <= m; ++j) {
                current[j] = static_cast<char>(previous[j - 1] != 0 &&
                                               static_cast<unsigned char>(name_bytes[j - 1]) ==
                                                   pattern_byte);
            }
        }
        previous.swap(current);
    }
    matched = previous[m] != 0;
    return TensorError::kOk;
}

TensorError parse_tensor_schema(std::string_view bytes, TensorSchema& out) {
    CanonicalRecord record;
    const TensorError parsed = from_admission(parse_canonical_record(bytes, schema_keys(), record));
    if (parsed != TensorError::kOk) {
        return parsed;
    }
    if (record.values.size() != 13) {
        return TensorError::kInternal;
    }
    TensorSchema schema;
    schema.schema_id = record.values[2];
    schema.rule_list_id = record.values[3];
    schema.inventory_id = record.values[6];
    schema.histogram_id = record.values[9];
    schema.rule_list_file_sha256 = record.values[5];
    schema.inventory_file_sha256 = record.values[8];
    schema.histogram_file_sha256 = record.values[11];
    if (const TensorError id = require_project_id(schema.schema_id); id != TensorError::kOk) {
        return id;
    }
    if (const TensorError id = require_project_id(schema.rule_list_id); id != TensorError::kOk) {
        return id;
    }
    if (const TensorError id = require_project_id(schema.inventory_id); id != TensorError::kOk) {
        return id;
    }
    if (const TensorError id = require_project_id(schema.histogram_id); id != TensorError::kOk) {
        return id;
    }
    if (const TensorError size = require_size(record.values[4], schema.rule_list_file_size);
        size != TensorError::kOk) {
        return size;
    }
    if (const TensorError size = require_size(record.values[7], schema.inventory_file_size);
        size != TensorError::kOk) {
        return size;
    }
    if (const TensorError size = require_size(record.values[10], schema.histogram_file_size);
        size != TensorError::kOk) {
        return size;
    }
    if (const TensorError digest = require_digest(schema.rule_list_file_sha256);
        digest != TensorError::kOk) {
        return digest;
    }
    if (const TensorError digest = require_digest(schema.inventory_file_sha256);
        digest != TensorError::kOk) {
        return digest;
    }
    if (const TensorError digest = require_digest(schema.histogram_file_sha256);
        digest != TensorError::kOk) {
        return digest;
    }
    schema.bytes = record.bytes;
    schema.self_checksum = record.self_checksum;
    schema.full_file_digest = record.full_file_digest;
    out = std::move(schema);
    return TensorError::kOk;
}

TensorError serialize_tensor_schema(const TensorSchema& schema, std::string& out) {
    if (require_project_id(schema.schema_id) != TensorError::kOk ||
        require_project_id(schema.rule_list_id) != TensorError::kOk ||
        require_project_id(schema.inventory_id) != TensorError::kOk ||
        require_project_id(schema.histogram_id) != TensorError::kOk ||
        require_digest(schema.rule_list_file_sha256) != TensorError::kOk ||
        require_digest(schema.inventory_file_sha256) != TensorError::kOk ||
        require_digest(schema.histogram_file_sha256) != TensorError::kOk) {
        return TensorError::kLexical;
    }
    const std::string values[] = {
        "1",
        "tensor_schema",
        schema.schema_id,
        schema.rule_list_id,
        std::to_string(schema.rule_list_file_size),
        schema.rule_list_file_sha256,
        schema.inventory_id,
        std::to_string(schema.inventory_file_size),
        schema.inventory_file_sha256,
        schema.histogram_id,
        std::to_string(schema.histogram_file_size),
        schema.histogram_file_sha256,
    };
    std::string_view views[12];
    for (std::size_t index = 0; index < 12; ++index) {
        views[index] = values[index];
    }
    std::string serialized;
    const TensorError status =
        from_admission(serialize_canonical_record(schema_keys(), views, serialized));
    if (status != TensorError::kOk) {
        return status;
    }
    out = std::move(serialized);
    return TensorError::kOk;
}

TensorError parse_tensor_rule_list(std::string_view bytes, TensorRuleList& out) {
    std::vector<std::string_view> lines;
    if (const TensorError split = split_lines(bytes, lines); split != TensorError::kOk) {
        return split;
    }
    if (lines.size() < 5) {
        return TensorError::kSchema;
    }
    for (std::size_t index = 0; index + 1 < lines.size(); ++index) {
        if (is_checksum_line(lines[index])) {
            return TensorError::kChecksum;
        }
    }
    if (!is_checksum_line(lines.back())) {
        return TensorError::kChecksum;
    }
    std::string version;
    std::string kind;
    std::string list_id;
    std::string count_text;
    if (const TensorError line = header_line(lines[0], "schema_version", version);
        line != TensorError::kOk) {
        return line;
    }
    if (const TensorError line = header_line(lines[1], "record_kind", kind);
        line != TensorError::kOk) {
        return line;
    }
    if (const TensorError line = header_line(lines[2], "list_id", list_id);
        line != TensorError::kOk) {
        return line;
    }
    if (const TensorError line = header_line(lines[3], "rule_count", count_text);
        line != TensorError::kOk) {
        return line;
    }
    if (version != "1" || kind != "tensor_rule_list") {
        return TensorError::kSchema;
    }
    if (const TensorError id = require_project_id(list_id); id != TensorError::kOk) {
        return id;
    }
    std::uint32_t rule_count = 0;
    if (const TensorError count = require_size(count_text, rule_count); count != TensorError::kOk) {
        return count;
    }
    const std::size_t elements = lines.size() - 5;
    if (elements != rule_count) {
        return TensorError::kCountMismatch;
    }
    TensorRuleList parsed;
    parsed.list_id = std::move(list_id);
    parsed.rule_count = rule_count;
    parsed.rules.reserve(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        bool spaced = false;
        const std::vector<std::string_view> tokens = split_spaces(lines[4 + index], spaced);
        if (spaced || tokens.size() != 2) {
            return TensorError::kMalformed;
        }
        TensorRule rule;
        rule.pattern_hex.assign(tokens[0]);
        if (const TensorError pattern = decode_pattern_hex(rule.pattern_hex, rule.pattern_bytes);
            pattern != TensorError::kOk) {
            return pattern == TensorError::kLexical ? TensorError::kPattern : pattern;
        }
        if (const TensorError role = parse_role(tokens[1], rule.role); role != TensorError::kOk) {
            return role;
        }
        for (const TensorRule& existing : parsed.rules) {
            if (existing.pattern_hex == rule.pattern_hex) {
                return TensorError::kDuplicate;
            }
        }
        parsed.rules.push_back(std::move(rule));
    }
    if (const TensorError checksum =
            check_self_checksum(bytes, lines.back(), parsed.self_checksum, parsed.full_file_digest);
        checksum != TensorError::kOk) {
        return checksum;
    }
    parsed.bytes.assign(bytes);
    out = std::move(parsed);
    return TensorError::kOk;
}

TensorError serialize_tensor_rule_list(std::string_view list_id, std::span<const TensorRule> rules,
                                       std::string& out) {
    if (require_project_id(list_id) != TensorError::kOk) {
        return TensorError::kLexical;
    }
    if (rules.size() > std::numeric_limits<std::uint32_t>::max()) {
        return TensorError::kOverflow;
    }
    std::string prefix = "schema_version=1\nrecord_kind=tensor_rule_list\nlist_id=";
    prefix.append(list_id);
    prefix.append("\nrule_count=");
    prefix.append(std::to_string(rules.size()));
    prefix.push_back('\n');
    for (std::size_t index = 0; index < rules.size(); ++index) {
        std::string decoded;
        if (decode_pattern_hex(rules[index].pattern_hex, decoded) != TensorError::kOk ||
            decoded != rules[index].pattern_bytes) {
            return TensorError::kPattern;
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (rules[earlier].pattern_hex == rules[index].pattern_hex) {
                return TensorError::kDuplicate;
            }
        }
        prefix.append(rules[index].pattern_hex);
        prefix.push_back(' ');
        prefix.append(rules[index].role == TensorRole::kWeight ? "weight" : "non_weight");
        prefix.push_back('\n');
    }
    if (prefix.size() > kMaxCanonicalRecordBytes || kMaxCanonicalRecordBytes - prefix.size() < 79) {
        return TensorError::kSize;
    }
    std::string file = append_checksum(std::move(prefix));
    if (file.size() > kMaxCanonicalRecordBytes) {
        return TensorError::kSize;
    }
    out = std::move(file);
    return TensorError::kOk;
}

TensorError parse_tensor_inventory(std::string_view bytes, TensorInventory& out) {
    std::vector<std::string_view> lines;
    if (const TensorError split = split_lines(bytes, lines); split != TensorError::kOk) {
        return split;
    }
    if (lines.size() < 5) {
        return TensorError::kSchema;
    }
    for (std::size_t index = 0; index + 1 < lines.size(); ++index) {
        if (is_checksum_line(lines[index])) {
            return TensorError::kChecksum;
        }
    }
    if (!is_checksum_line(lines.back())) {
        return TensorError::kChecksum;
    }
    std::string version;
    std::string kind;
    std::string inventory_id;
    std::string count_text;
    if (const TensorError line = header_line(lines[0], "schema_version", version);
        line != TensorError::kOk) {
        return line;
    }
    if (const TensorError line = header_line(lines[1], "record_kind", kind);
        line != TensorError::kOk) {
        return line;
    }
    if (const TensorError line = header_line(lines[2], "inventory_id", inventory_id);
        line != TensorError::kOk) {
        return line;
    }
    if (const TensorError line = header_line(lines[3], "tensor_count", count_text);
        line != TensorError::kOk) {
        return line;
    }
    if (version != "1" || kind != "tensor_inventory") {
        return TensorError::kSchema;
    }
    if (const TensorError id = require_project_id(inventory_id); id != TensorError::kOk) {
        return id;
    }
    std::uint32_t tensor_count = 0;
    if (const TensorError count = require_size(count_text, tensor_count);
        count != TensorError::kOk) {
        return count;
    }
    const std::size_t elements = lines.size() - 5;
    if (elements != tensor_count) {
        return TensorError::kCountMismatch;
    }
    TensorInventory parsed;
    parsed.inventory_id = std::move(inventory_id);
    parsed.tensor_count = tensor_count;
    parsed.entries.reserve(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        bool spaced = false;
        const std::vector<std::string_view> tokens = split_spaces(lines[4 + index], spaced);
        if (spaced || tokens.size() != 3) {
            return TensorError::kMalformed;
        }
        TensorInventoryEntry entry;
        entry.name_hex.assign(tokens[0]);
        if (validate_pattern_hex(entry.name_hex) != AdmissionError::kOk) {
            return TensorError::kLexical;
        }
        if (const TensorError name = decode_name_hex(entry.name_hex, entry.name_bytes);
            name != TensorError::kOk) {
            return name;
        }
        if (const TensorError rank = require_size(tokens[1], entry.rank);
            rank != TensorError::kOk) {
            return rank;
        }
        if (entry.rank < 1 || entry.rank > 8) {
            return TensorError::kInvalidRank;
        }
        Uint64List dimensions;
        const AdmissionError dims = parse_uint64_list(tokens[2], dimensions);
        if (dims != AdmissionError::kOk) {
            return from_admission(dims);
        }
        if (dimensions.is_none) {
            return TensorError::kLexical;
        }
        if (dimensions.values.size() != entry.rank) {
            return TensorError::kDimensionMismatch;
        }
        entry.dimensions = std::move(dimensions.values);
        for (const TensorInventoryEntry& existing : parsed.entries) {
            if (existing.name_hex == entry.name_hex || existing.name_bytes == entry.name_bytes) {
                return TensorError::kDuplicate;
            }
        }
        parsed.entries.push_back(std::move(entry));
    }
    if (const TensorError checksum =
            check_self_checksum(bytes, lines.back(), parsed.self_checksum, parsed.full_file_digest);
        checksum != TensorError::kOk) {
        return checksum;
    }
    parsed.bytes.assign(bytes);
    out = std::move(parsed);
    return TensorError::kOk;
}

TensorError serialize_tensor_inventory(std::string_view inventory_id,
                                       std::span<const TensorInventoryEntry> entries,
                                       std::string& out) {
    if (require_project_id(inventory_id) != TensorError::kOk) {
        return TensorError::kLexical;
    }
    std::string prefix = "schema_version=1\nrecord_kind=tensor_inventory\ninventory_id=";
    prefix.append(inventory_id);
    prefix.append("\ntensor_count=");
    prefix.append(std::to_string(entries.size()));
    prefix.push_back('\n');
    for (std::size_t index = 0; index < entries.size(); ++index) {
        std::string decoded;
        if (decode_name_hex(entries[index].name_hex, decoded) != TensorError::kOk ||
            decoded != entries[index].name_bytes) {
            return TensorError::kInvalidUtf8;
        }
        if (entries[index].rank < 1 || entries[index].rank > 8) {
            return TensorError::kInvalidRank;
        }
        if (entries[index].dimensions.size() != entries[index].rank) {
            return TensorError::kDimensionMismatch;
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (entries[earlier].name_hex == entries[index].name_hex ||
                entries[earlier].name_bytes == entries[index].name_bytes) {
                return TensorError::kDuplicate;
            }
        }
        prefix.append(entries[index].name_hex);
        prefix.push_back(' ');
        prefix.append(std::to_string(entries[index].rank));
        prefix.push_back(' ');
        for (std::size_t dim = 0; dim < entries[index].dimensions.size(); ++dim) {
            if (dim != 0) {
                prefix.push_back(',');
            }
            prefix.append(std::to_string(entries[index].dimensions[dim]));
        }
        prefix.push_back('\n');
    }
    if (prefix.size() > kMaxCanonicalRecordBytes || kMaxCanonicalRecordBytes - prefix.size() < 79) {
        return TensorError::kSize;
    }
    std::string file = append_checksum(std::move(prefix));
    if (file.size() > kMaxCanonicalRecordBytes) {
        return TensorError::kSize;
    }
    out = std::move(file);
    return TensorError::kOk;
}

TensorError parse_tensor_histogram(std::string_view bytes, std::uint32_t weight_tensor_count,
                                   TensorHistogram& out) {
    std::vector<std::string_view> lines;
    if (const TensorError split = split_lines(bytes, lines); split != TensorError::kOk) {
        return split;
    }
    if (lines.size() < 5) {
        return TensorError::kSchema;
    }
    for (std::size_t index = 0; index + 1 < lines.size(); ++index) {
        if (is_checksum_line(lines[index])) {
            return TensorError::kChecksum;
        }
    }
    if (!is_checksum_line(lines.back())) {
        return TensorError::kChecksum;
    }
    std::string version;
    std::string kind;
    std::string histogram_id;
    std::string count_text;
    if (const TensorError line = header_line(lines[0], "schema_version", version);
        line != TensorError::kOk) {
        return line;
    }
    if (const TensorError line = header_line(lines[1], "record_kind", kind);
        line != TensorError::kOk) {
        return line;
    }
    if (const TensorError line = header_line(lines[2], "histogram_id", histogram_id);
        line != TensorError::kOk) {
        return line;
    }
    if (const TensorError line = header_line(lines[3], "type_count", count_text);
        line != TensorError::kOk) {
        return line;
    }
    if (version != "1" || kind != "tensor_histogram") {
        return TensorError::kSchema;
    }
    if (const TensorError id = require_project_id(histogram_id); id != TensorError::kOk) {
        return id;
    }
    std::uint32_t type_count = 0;
    if (const TensorError count = require_size(count_text, type_count); count != TensorError::kOk) {
        return count;
    }
    const std::size_t elements = lines.size() - 5;
    if (elements != type_count) {
        return TensorError::kCountMismatch;
    }
    TensorHistogram parsed;
    parsed.histogram_id = std::move(histogram_id);
    parsed.type_count = type_count;
    parsed.weight_tensor_count = weight_tensor_count;
    parsed.entries.reserve(elements);
    std::uint32_t sum = 0;
    for (std::size_t index = 0; index < elements; ++index) {
        bool spaced = false;
        const std::vector<std::string_view> tokens = split_spaces(lines[4 + index], spaced);
        if (spaced || tokens.size() != 2) {
            return TensorError::kMalformed;
        }
        if (!type_name_ok(tokens[0])) {
            return TensorError::kLexical;
        }
        TensorHistogramEntry entry;
        entry.type_name.assign(tokens[0]);
        if (const TensorError count = require_size(tokens[1], entry.count);
            count != TensorError::kOk) {
            return count;
        }
        if (entry.count > std::numeric_limits<std::uint32_t>::max() - sum) {
            return TensorError::kOverflow;
        }
        sum = static_cast<std::uint32_t>(sum + entry.count);
        for (const TensorHistogramEntry& existing : parsed.entries) {
            if (existing.type_name == entry.type_name) {
                return TensorError::kDuplicate;
            }
        }
        parsed.entries.push_back(std::move(entry));
    }
    if (sum != weight_tensor_count) {
        return TensorError::kHistogramMismatch;
    }
    if (const TensorError checksum =
            check_self_checksum(bytes, lines.back(), parsed.self_checksum, parsed.full_file_digest);
        checksum != TensorError::kOk) {
        return checksum;
    }
    parsed.bytes.assign(bytes);
    out = std::move(parsed);
    return TensorError::kOk;
}

TensorError serialize_tensor_histogram(std::string_view histogram_id,
                                       std::span<const TensorHistogramEntry> entries,
                                       std::string& out) {
    if (require_project_id(histogram_id) != TensorError::kOk) {
        return TensorError::kLexical;
    }
    std::string prefix = "schema_version=1\nrecord_kind=tensor_histogram\nhistogram_id=";
    prefix.append(histogram_id);
    prefix.append("\ntype_count=");
    prefix.append(std::to_string(entries.size()));
    prefix.push_back('\n');
    for (std::size_t index = 0; index < entries.size(); ++index) {
        if (!type_name_ok(entries[index].type_name)) {
            return TensorError::kLexical;
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (entries[earlier].type_name == entries[index].type_name) {
                return TensorError::kDuplicate;
            }
        }
        prefix.append(entries[index].type_name);
        prefix.push_back(' ');
        prefix.append(std::to_string(entries[index].count));
        prefix.push_back('\n');
    }
    if (prefix.size() > kMaxCanonicalRecordBytes || kMaxCanonicalRecordBytes - prefix.size() < 79) {
        return TensorError::kSize;
    }
    std::string file = append_checksum(std::move(prefix));
    if (file.size() > kMaxCanonicalRecordBytes) {
        return TensorError::kSize;
    }
    out = std::move(file);
    return TensorError::kOk;
}

TensorError validate_tensor_bundle(std::string_view schema_bytes, std::string_view rule_bytes,
                                   std::string_view inventory_bytes,
                                   std::string_view histogram_bytes,
                                   std::uint32_t weight_tensor_count, TensorBundle& out) {
    TensorBundle bundle;
    if (const TensorError schema = parse_tensor_schema(schema_bytes, bundle.schema);
        schema != TensorError::kOk) {
        return schema;
    }
    if (const TensorError rules = parse_tensor_rule_list(rule_bytes, bundle.rules);
        rules != TensorError::kOk) {
        return rules;
    }
    if (const TensorError inventory = parse_tensor_inventory(inventory_bytes, bundle.inventory);
        inventory != TensorError::kOk) {
        return inventory;
    }
    if (const TensorError histogram =
            parse_tensor_histogram(histogram_bytes, weight_tensor_count, bundle.histogram);
        histogram != TensorError::kOk) {
        return histogram;
    }
    if (bundle.schema.rule_list_id != bundle.rules.list_id ||
        bundle.schema.inventory_id != bundle.inventory.inventory_id ||
        bundle.schema.histogram_id != bundle.histogram.histogram_id) {
        return TensorError::kChildIdMismatch;
    }
    if (bundle.schema.rule_list_file_size != rule_bytes.size() ||
        bundle.schema.inventory_file_size != inventory_bytes.size() ||
        bundle.schema.histogram_file_size != histogram_bytes.size()) {
        return TensorError::kChildSizeMismatch;
    }
    if (bundle.schema.rule_list_file_sha256 != bundle.rules.full_file_digest ||
        bundle.schema.inventory_file_sha256 != bundle.inventory.full_file_digest ||
        bundle.schema.histogram_file_sha256 != bundle.histogram.full_file_digest) {
        return TensorError::kChildDigestMismatch;
    }
    out = std::move(bundle);
    return TensorError::kOk;
}

} // namespace inferedge
