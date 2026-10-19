#include "inferedge/admission_record.hpp"

#include <cstring>
#include <limits>

namespace inferedge {
namespace {

constexpr char kHex[] = "0123456789abcdef";
constexpr std::uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

constexpr std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) noexcept {
    return (value >> bits) | (value << (32u - bits));
}

void sha256_block(std::uint32_t state[8], const std::uint8_t block[64]) noexcept {
    std::uint32_t words[64];
    for (int index = 0; index < 16; ++index) {
        const std::size_t offset = static_cast<std::size_t>(index) * 4u;
        words[index] = (static_cast<std::uint32_t>(block[offset]) << 24u) |
                       (static_cast<std::uint32_t>(block[offset + 1u]) << 16u) |
                       (static_cast<std::uint32_t>(block[offset + 2u]) << 8u) |
                       static_cast<std::uint32_t>(block[offset + 3u]);
    }
    for (int index = 16; index < 64; ++index) {
        const std::uint32_t s0 =
            rotr(words[index - 15], 7u) ^ rotr(words[index - 15], 18u) ^ (words[index - 15] >> 3u);
        const std::uint32_t s1 =
            rotr(words[index - 2], 17u) ^ rotr(words[index - 2], 19u) ^ (words[index - 2] >> 10u);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (int index = 0; index < 64; ++index) {
        const std::uint32_t s1 = rotr(e, 6u) ^ rotr(e, 11u) ^ rotr(e, 25u);
        const std::uint32_t choose = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + s1 + choose + kSha256K[index] + words[index];
        const std::uint32_t s0 = rotr(a, 2u) ^ rotr(a, 13u) ^ rotr(a, 22u);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void sha256(const std::uint8_t* data, std::size_t length, std::uint8_t out[32]) noexcept {
    std::uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    const std::uint64_t bit_length = static_cast<std::uint64_t>(length) * 8u;
    std::size_t offset = 0;
    while (length - offset >= 64u) {
        sha256_block(state, data + offset);
        offset += 64u;
    }

    std::uint8_t block[64] = {};
    const std::size_t remaining = length - offset;
    if (remaining > 0) {
        std::memcpy(block, data + offset, remaining);
    }
    block[remaining] = 0x80u;
    if (remaining >= 56u) {
        sha256_block(state, block);
        std::memset(block, 0, sizeof(block));
    }
    for (int byte = 0; byte < 8; ++byte) {
        block[63 - byte] = static_cast<std::uint8_t>(bit_length >> (8 * byte));
    }
    sha256_block(state, block);

    for (int word = 0; word < 8; ++word) {
        out[word * 4] = static_cast<std::uint8_t>(state[word] >> 24u);
        out[word * 4 + 1] = static_cast<std::uint8_t>(state[word] >> 16u);
        out[word * 4 + 2] = static_cast<std::uint8_t>(state[word] >> 8u);
        out[word * 4 + 3] = static_cast<std::uint8_t>(state[word]);
    }
}

bool is_key_char(unsigned char c, bool first) noexcept {
    if (c >= 'a' && c <= 'z') {
        return true;
    }
    if (first) {
        return false;
    }
    return (c >= '0' && c <= '9') || c == '_' || c == '-';
}

bool valid_key(std::string_view key) noexcept {
    if (key.empty() || key.size() > 64) {
        return false;
    }
    for (std::size_t index = 0; index < key.size(); ++index) {
        if (!is_key_char(static_cast<unsigned char>(key[index]), index == 0)) {
            return false;
        }
    }
    return true;
}

bool contains_key(const std::vector<std::string>& keys, std::string_view key) noexcept {
    for (const std::string& existing : keys) {
        if (existing == key) {
            return true;
        }
    }
    return false;
}

bool is_canonical_byte(unsigned char c) noexcept {
    return c >= 0x21 && c <= 0x7E && c != '#';
}

bool is_canonical_value(std::string_view text) noexcept {
    for (unsigned char c : text) {
        if (!is_canonical_byte(c)) {
            return false;
        }
    }
    return true;
}

bool is_file_byte(unsigned char c) noexcept {
    return c == '\n' || is_canonical_byte(c);
}

int hex_nibble(unsigned char c) noexcept {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    return -1;
}

bool is_lower_hex(std::string_view text, std::size_t min_len, std::size_t max_len) noexcept {
    if (text.size() < min_len || text.size() > max_len || (text.size() % 2u) != 0u) {
        return false;
    }
    for (unsigned char c : text) {
        if (hex_nibble(c) < 0) {
            return false;
        }
    }
    return true;
}

AdmissionError decode_hex(std::string_view text, std::vector<std::uint8_t>& out) {
    if ((text.size() % 2u) != 0u) {
        return AdmissionError::kLexical;
    }
    std::vector<std::uint8_t> bytes;
    bytes.reserve(text.size() / 2u);
    for (std::size_t index = 0; index < text.size(); index += 2u) {
        const int high = hex_nibble(static_cast<unsigned char>(text[index]));
        const int low = hex_nibble(static_cast<unsigned char>(text[index + 1u]));
        if (high < 0 || low < 0) {
            return AdmissionError::kLexical;
        }
        bytes.push_back(static_cast<std::uint8_t>((high << 4) | low));
    }
    out = std::move(bytes);
    return AdmissionError::kOk;
}

bool is_digit(unsigned char c) noexcept {
    return c >= '0' && c <= '9';
}

bool all_digits(std::string_view text) noexcept {
    if (text.empty()) {
        return false;
    }
    for (unsigned char c : text) {
        if (!is_digit(c)) {
            return false;
        }
    }
    return true;
}

bool has_leading_zero(std::string_view digits) noexcept {
    return digits.size() > 1 && digits.front() == '0';
}

AdmissionError parse_u64_digits(std::string_view digits, std::uint64_t& value) {
    if (!all_digits(digits) || has_leading_zero(digits)) {
        return AdmissionError::kLexical;
    }
    if (digits.size() > 20) {
        return AdmissionError::kOverflow;
    }
    if (digits.size() == 20 && digits > std::string_view{"18446744073709551615"}) {
        return AdmissionError::kOverflow;
    }
    std::uint64_t parsed = 0;
    for (unsigned char c : digits) {
        parsed = parsed * 10u + static_cast<std::uint64_t>(c - '0');
    }
    value = parsed;
    return AdmissionError::kOk;
}

bool is_leap_year(int year) noexcept {
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

int days_in_month(int year, int month) noexcept {
    static constexpr int kDays[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && is_leap_year(year)) {
        return 29;
    }
    return kDays[month];
}

bool parse_date_parts(std::string_view text, int& year, int& month, int& day) noexcept {
    if (text.size() != 10 || text[4] != '-' || text[7] != '-') {
        return false;
    }
    for (std::size_t index = 0; index < text.size(); ++index) {
        if (index == 4 || index == 7) {
            continue;
        }
        if (!is_digit(static_cast<unsigned char>(text[index]))) {
            return false;
        }
    }
    year = (text[0] - '0') * 1000 + (text[1] - '0') * 100 + (text[2] - '0') * 10 + (text[3] - '0');
    month = (text[5] - '0') * 10 + (text[6] - '0');
    day = (text[8] - '0') * 10 + (text[9] - '0');
    return true;
}

bool date_in_range(int year, int month, int day) noexcept {
    if (year < 2020 || year > 2100 || month < 1 || month > 12) {
        return false;
    }
    if (day < 1 || day > days_in_month(year, month)) {
        return false;
    }
    return true;
}

int days_from_civil(int year, unsigned month, unsigned day) noexcept {
    year -= month <= 2 ? 1 : 0;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned year_of_era = static_cast<unsigned>(year - era * 400);
    const unsigned day_of_year = (153u * (month + (month > 2 ? -3u : 9u)) + 2u) / 5u + day - 1u;
    const unsigned day_of_era =
        year_of_era * 365u + year_of_era / 4u - year_of_era / 100u + day_of_year;
    return era * 146097 + static_cast<int>(day_of_era) - 719468;
}

void civil_from_days(int z, int& year, unsigned& month, unsigned& day) noexcept {
    z += 719468;
    const int era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned day_of_era = static_cast<unsigned>(z - era * 146097);
    const unsigned year_of_era =
        (day_of_era - day_of_era / 1460u + day_of_era / 36524u - day_of_era / 146096u) / 365u;
    year = static_cast<int>(year_of_era) + era * 400;
    const unsigned day_of_year =
        day_of_era - (365u * year_of_era + year_of_era / 4u - year_of_era / 100u);
    const unsigned month_prime = (5u * day_of_year + 2u) / 153u;
    day = day_of_year - (153u * month_prime + 2u) / 5u + 1u;
    month = month_prime < 10u ? month_prime + 3u : month_prime - 9u;
    year += month <= 2 ? 1 : 0;
}

void append_u4(std::string& out, int value) {
    out.push_back(static_cast<char>('0' + (value / 1000) % 10));
    out.push_back(static_cast<char>('0' + (value / 100) % 10));
    out.push_back(static_cast<char>('0' + (value / 10) % 10));
    out.push_back(static_cast<char>('0' + value % 10));
}

void append_u2(std::string& out, unsigned value) {
    out.push_back(static_cast<char>('0' + (value / 10u) % 10u));
    out.push_back(static_cast<char>('0' + value % 10u));
}

bool project_id_pattern(std::string_view text) noexcept {
    if (text.empty() || text.size() > 128) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>(text.front());
    if (!((first >= 'a' && first <= 'z') || (first >= '0' && first <= '9'))) {
        return false;
    }
    for (unsigned char c : text) {
        const bool ok =
            (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!ok) {
            return false;
        }
    }
    if (text.front() == '.' || text.back() == '.') {
        return false;
    }
    if (text.find("..") != std::string_view::npos) {
        return false;
    }
    return true;
}

bool is_reserved_project_id(std::string_view text) noexcept {
    return text == "none" || text == "unknown";
}

AdmissionError reject_utf8(const std::uint8_t* bytes, std::size_t size, std::string& decoded) {
    std::string temp;
    temp.reserve(size);
    std::size_t index = 0;
    while (index < size) {
        const std::uint8_t lead = bytes[index];
        if (lead == 0x00) {
            return AdmissionError::kLexical;
        }
        if (lead <= 0x7F) {
            temp.push_back(static_cast<char>(lead));
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
            return AdmissionError::kLexical;
        }
        if (index + need > size) {
            return AdmissionError::kLexical;
        }
        for (std::size_t extra = 1; extra < need; ++extra) {
            const std::uint8_t cont = bytes[index + extra];
            if ((cont & 0xC0u) != 0x80u) {
                return AdmissionError::kLexical;
            }
            code = (code << 6u) | (cont & 0x3Fu);
        }
        if (lead == 0xE0 && bytes[index + 1] < 0xA0) {
            return AdmissionError::kLexical;
        }
        if (lead == 0xED && bytes[index + 1] >= 0xA0) {
            return AdmissionError::kLexical;
        }
        if (lead == 0xF0 && bytes[index + 1] < 0x90) {
            return AdmissionError::kLexical;
        }
        if (lead == 0xF4 && bytes[index + 1] > 0x8F) {
            return AdmissionError::kLexical;
        }
        if (code < minimum || code > 0x10FFFFu || (code >= 0xD800u && code <= 0xDFFFu)) {
            return AdmissionError::kLexical;
        }
        for (std::size_t extra = 0; extra < need; ++extra) {
            temp.push_back(static_cast<char>(bytes[index + extra]));
        }
        index += need;
    }
    decoded = std::move(temp);
    return AdmissionError::kOk;
}

std::vector<std::string_view> split_commas(std::string_view text) {
    std::vector<std::string_view> parts;
    std::size_t start = 0;
    for (std::size_t index = 0; index <= text.size(); ++index) {
        if (index == text.size() || text[index] == ',') {
            parts.push_back(text.substr(start, index - start));
            start = index + 1;
        }
    }
    return parts;
}

} // namespace

const char* admission_error_name(AdmissionError error) noexcept {
    switch (error) {
    case AdmissionError::kOk:
        return "ok";
    case AdmissionError::kSize:
        return "size";
    case AdmissionError::kMalformed:
        return "malformed";
    case AdmissionError::kSchema:
        return "schema";
    case AdmissionError::kDuplicateKey:
        return "duplicate_key";
    case AdmissionError::kUnknownKey:
        return "unknown_key";
    case AdmissionError::kMissingKey:
        return "missing_key";
    case AdmissionError::kLexical:
        return "lexical";
    case AdmissionError::kChecksumFormat:
        return "checksum_format";
    case AdmissionError::kChecksumMismatch:
        return "checksum_mismatch";
    case AdmissionError::kOverflow:
        return "overflow";
    case AdmissionError::kInvalidDate:
        return "invalid_date";
    case AdmissionError::kInternal:
        return "internal";
    }
    return "internal";
}

std::string digest_hex(std::string_view bytes) {
    std::uint8_t raw[32];
    sha256(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size(), raw);
    std::string hex(64, '0');
    for (int index = 0; index < 32; ++index) {
        hex[static_cast<std::size_t>(index) * 2u] = kHex[raw[index] >> 4u];
        hex[static_cast<std::size_t>(index) * 2u + 1u] = kHex[raw[index] & 0x0Fu];
    }
    return hex;
}

AdmissionError check_closed_schema(const ClosedSchema& schema) {
    if (schema.keys.size() < 2 || schema.keys.back() != kChecksumKey) {
        return AdmissionError::kSchema;
    }
    bool saw_version = false;
    bool saw_kind = false;
    for (std::size_t index = 0; index < schema.keys.size(); ++index) {
        const std::string& key = schema.keys[index];
        if (!valid_key(key)) {
            return AdmissionError::kSchema;
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (schema.keys[earlier] == key) {
                return AdmissionError::kSchema;
            }
        }
        if (key == "schema_version") {
            saw_version = true;
        }
        if (key == "record_kind") {
            saw_kind = true;
        }
        if (key == kChecksumKey && index + 1 != schema.keys.size()) {
            return AdmissionError::kSchema;
        }
    }
    if (!saw_version) {
        return AdmissionError::kSchema;
    }
    if (!schema.record_kind.empty()) {
        if (!saw_kind || !is_canonical_value(schema.record_kind)) {
            return AdmissionError::kSchema;
        }
    }
    return AdmissionError::kOk;
}

AdmissionError serialize_canonical_record(const ClosedSchema& schema,
                                          std::span<const std::string_view> values,
                                          std::string& out) {
    if (const AdmissionError schema_error = check_closed_schema(schema);
        schema_error != AdmissionError::kOk) {
        return schema_error;
    }
    if (values.size() + 1 != schema.keys.size()) {
        return AdmissionError::kSchema;
    }
    for (std::size_t index = 0; index + 1 < schema.keys.size(); ++index) {
        if (!is_canonical_value(values[index])) {
            return AdmissionError::kMalformed;
        }
        if (schema.keys[index] == "schema_version" && values[index] != "1") {
            return AdmissionError::kSchema;
        }
        if (!schema.record_kind.empty() && schema.keys[index] == "record_kind" &&
            values[index] != schema.record_kind) {
            return AdmissionError::kSchema;
        }
    }

    std::string prefix;
    for (std::size_t index = 0; index + 1 < schema.keys.size(); ++index) {
        prefix.append(schema.keys[index]);
        prefix.push_back('=');
        prefix.append(values[index]);
        prefix.push_back('\n');
    }
    constexpr std::size_t kChecksumLineBytes = 14 + 64 + 1;
    if (prefix.size() > kMaxCanonicalRecordBytes ||
        kMaxCanonicalRecordBytes - prefix.size() < kChecksumLineBytes) {
        return AdmissionError::kSize;
    }
    const std::string checksum = digest_hex(prefix);
    std::string file = prefix;
    file.append(kChecksumKey);
    file.push_back('=');
    file.append(checksum);
    file.push_back('\n');
    if (file.size() > kMaxCanonicalRecordBytes) {
        return AdmissionError::kSize;
    }
    out = std::move(file);
    return AdmissionError::kOk;
}

AdmissionError parse_canonical_record(std::string_view bytes, const ClosedSchema& schema,
                                      CanonicalRecord& out) {
    if (const AdmissionError schema_error = check_closed_schema(schema);
        schema_error != AdmissionError::kOk) {
        return schema_error;
    }
    if (bytes.empty() || bytes.size() > kMaxCanonicalRecordBytes) {
        return AdmissionError::kSize;
    }
    for (unsigned char c : bytes) {
        if (!is_file_byte(c)) {
            return AdmissionError::kMalformed;
        }
    }
    if (bytes.back() != '\n' || (bytes.size() >= 2 && bytes[bytes.size() - 2] == '\n')) {
        return AdmissionError::kMalformed;
    }

    std::vector<std::string> keys;
    std::vector<std::string> values;
    std::size_t cursor = 0;
    while (cursor < bytes.size()) {
        const std::size_t end = bytes.find('\n', cursor);
        if (end == std::string_view::npos || end == cursor) {
            return AdmissionError::kMalformed;
        }
        const std::string_view line = bytes.substr(cursor, end - cursor);
        const std::size_t eq = line.find('=');
        if (eq == std::string_view::npos || eq == 0) {
            return AdmissionError::kMalformed;
        }
        const std::string key{line.substr(0, eq)};
        const std::string value{line.substr(eq + 1)};
        if (contains_key(keys, key)) {
            return AdmissionError::kDuplicateKey;
        }
        keys.push_back(key);
        values.push_back(value);
        cursor = end + 1;
    }

    for (const std::string& key : keys) {
        if (!contains_key(schema.keys, key)) {
            return AdmissionError::kUnknownKey;
        }
    }
    if (keys.size() != schema.keys.size()) {
        return AdmissionError::kMissingKey;
    }
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (keys[index] != schema.keys[index]) {
            return AdmissionError::kSchema;
        }
    }

    std::string version;
    std::string kind;
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (keys[index] == "schema_version") {
            version = values[index];
        }
        if (keys[index] == "record_kind") {
            kind = values[index];
        }
    }
    if (version != "1") {
        return AdmissionError::kSchema;
    }
    if (!schema.record_kind.empty() && kind != schema.record_kind) {
        return AdmissionError::kSchema;
    }
    if (keys.back() != kChecksumKey) {
        return AdmissionError::kInternal;
    }

    const std::string& checksum = values.back();
    if (!is_lower_hex(checksum, 64, 64)) {
        return AdmissionError::kChecksumFormat;
    }
    const std::string checksum_line = std::string{kChecksumKey} + "=" + checksum + "\n";
    if (bytes.size() < checksum_line.size() ||
        bytes.substr(bytes.size() - checksum_line.size()) != checksum_line) {
        return AdmissionError::kInternal;
    }
    const std::string_view prefix = bytes.substr(0, bytes.size() - checksum_line.size());
    if (digest_hex(prefix) != checksum) {
        return AdmissionError::kChecksumMismatch;
    }

    CanonicalRecord accepted;
    accepted.keys = std::move(keys);
    accepted.values = std::move(values);
    accepted.bytes.assign(bytes);
    accepted.self_checksum = checksum;
    accepted.full_file_digest = digest_hex(bytes);
    out = std::move(accepted);
    return AdmissionError::kOk;
}

AdmissionError validate_project_id(std::string_view text) {
    if (!project_id_pattern(text) || is_reserved_project_id(text)) {
        return AdmissionError::kLexical;
    }
    return AdmissionError::kOk;
}

AdmissionError validate_id_or_none(std::string_view text) {
    if (text == "none") {
        return AdmissionError::kOk;
    }
    return validate_project_id(text);
}

AdmissionError make_compound_id(std::string_view prefix, std::string_view component,
                                std::string& out) {
    if (component.size() < 1 || component.size() > 96 ||
        validate_project_id(component) != AdmissionError::kOk) {
        return AdmissionError::kLexical;
    }
    if (prefix.size() > kMaxCanonicalRecordBytes ||
        component.size() > kMaxCanonicalRecordBytes - prefix.size()) {
        return AdmissionError::kLexical;
    }
    const std::string combined = std::string{prefix} + std::string{component};
    if (validate_project_id(combined) != AdmissionError::kOk) {
        return AdmissionError::kLexical;
    }
    out = combined;
    return AdmissionError::kOk;
}

AdmissionError parse_uint(std::string_view text, std::uint32_t& out) {
    std::uint64_t wide = 0;
    const AdmissionError parsed = parse_u64_digits(text, wide);
    if (parsed != AdmissionError::kOk) {
        return parsed;
    }
    if (wide > std::numeric_limits<std::uint32_t>::max()) {
        return AdmissionError::kOverflow;
    }
    out = static_cast<std::uint32_t>(wide);
    return AdmissionError::kOk;
}

AdmissionError parse_uint64(std::string_view text, std::uint64_t& out) {
    std::uint64_t parsed = 0;
    const AdmissionError status = parse_u64_digits(text, parsed);
    if (status != AdmissionError::kOk) {
        return status;
    }
    out = parsed;
    return AdmissionError::kOk;
}

AdmissionError parse_int32(std::string_view text, std::int32_t& out) {
    if (text.empty() || text.front() == '+') {
        return AdmissionError::kLexical;
    }
    const bool negative = text.front() == '-';
    const std::string_view digits = negative ? text.substr(1) : text;
    if (negative && (digits.empty() || digits == "0")) {
        return AdmissionError::kLexical;
    }
    std::uint64_t magnitude = 0;
    const AdmissionError parsed = parse_u64_digits(digits, magnitude);
    if (parsed != AdmissionError::kOk) {
        return parsed;
    }
    if (negative) {
        if (magnitude > 2147483648ull) {
            return AdmissionError::kOverflow;
        }
        if (magnitude == 2147483648ull) {
            out = std::numeric_limits<std::int32_t>::min();
            return AdmissionError::kOk;
        }
        out = -static_cast<std::int32_t>(magnitude);
        return AdmissionError::kOk;
    }
    if (magnitude > 2147483647ull) {
        return AdmissionError::kOverflow;
    }
    out = static_cast<std::int32_t>(magnitude);
    return AdmissionError::kOk;
}

AdmissionError parse_int32_list(std::string_view text, Int32List& out) {
    if (text == "none") {
        out = Int32List{true, {}};
        return AdmissionError::kOk;
    }
    const std::vector<std::string_view> parts = split_commas(text);
    if (parts.empty() || parts.size() > kInt32ListMax) {
        return AdmissionError::kLexical;
    }
    std::vector<std::int32_t> values;
    values.reserve(parts.size());
    for (std::string_view part : parts) {
        std::int32_t value = 0;
        const AdmissionError status = parse_int32(part, value);
        if (status != AdmissionError::kOk) {
            return status;
        }
        values.push_back(value);
    }
    out = Int32List{false, std::move(values)};
    return AdmissionError::kOk;
}

AdmissionError parse_uint64_list(std::string_view text, Uint64List& out) {
    if (text == "none") {
        out = Uint64List{true, {}};
        return AdmissionError::kOk;
    }
    const std::vector<std::string_view> parts = split_commas(text);
    if (parts.empty() || parts.size() > kUint64ListMax) {
        return AdmissionError::kLexical;
    }
    std::vector<std::uint64_t> values;
    values.reserve(parts.size());
    for (std::string_view part : parts) {
        std::uint64_t value = 0;
        const AdmissionError status = parse_uint64(part, value);
        if (status != AdmissionError::kOk) {
            return status;
        }
        values.push_back(value);
    }
    out = Uint64List{false, std::move(values)};
    return AdmissionError::kOk;
}

AdmissionError validate_sha256(std::string_view text) {
    if (!is_lower_hex(text, 64, 64)) {
        return AdmissionError::kLexical;
    }
    return AdmissionError::kOk;
}

bool is_zero_digest(std::string_view text) noexcept {
    return text == kZeroDigest;
}

bool is_empty_byte_digest(std::string_view text) noexcept {
    return text == kEmptyByteDigest;
}

AdmissionError validate_digest_or_zero(std::string_view text, bool zero_eligible) {
    if (const AdmissionError status = validate_sha256(text); status != AdmissionError::kOk) {
        return status;
    }
    if (is_zero_digest(text) && !zero_eligible) {
        return AdmissionError::kLexical;
    }
    return AdmissionError::kOk;
}

AdmissionError validate_date(std::string_view text) {
    int year = 0;
    int month = 0;
    int day = 0;
    if (!parse_date_parts(text, year, month, day) || !date_in_range(year, month, day)) {
        return AdmissionError::kInvalidDate;
    }
    return AdmissionError::kOk;
}

AdmissionError add_calendar_days(std::string_view date, std::uint64_t days, std::string& out) {
    int year = 0;
    int month = 0;
    int day = 0;
    if (!parse_date_parts(date, year, month, day) || !date_in_range(year, month, day)) {
        return AdmissionError::kInvalidDate;
    }
    const int epoch = days_from_civil(2020, 1u, 1u);
    const int index =
        days_from_civil(year, static_cast<unsigned>(month), static_cast<unsigned>(day)) - epoch;
    const int limit = days_from_civil(2100, 12u, 31u) - epoch;
    if (index < 0 || limit < index) {
        return AdmissionError::kInvalidDate;
    }
    const auto remaining = static_cast<std::uint64_t>(limit - index);
    if (days > remaining) {
        return AdmissionError::kInvalidDate;
    }
    int result_year = 0;
    unsigned result_month = 0;
    unsigned result_day = 0;
    civil_from_days(epoch + index + static_cast<int>(days), result_year, result_month, result_day);
    std::string formatted;
    formatted.reserve(10);
    append_u4(formatted, result_year);
    formatted.push_back('-');
    append_u2(formatted, result_month);
    formatted.push_back('-');
    append_u2(formatted, result_day);
    out = std::move(formatted);
    return AdmissionError::kOk;
}

AdmissionError add_uint64(std::uint64_t left, std::uint64_t right, std::uint64_t& sum) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return AdmissionError::kOverflow;
    }
    sum = left + right;
    return AdmissionError::kOk;
}

AdmissionError validate_filename(std::string_view text) {
    if (text.empty() || text.size() > 255) {
        return AdmissionError::kLexical;
    }
    const unsigned char first = static_cast<unsigned char>(text.front());
    if (!((first >= 'A' && first <= 'Z') || (first >= 'a' && first <= 'z') ||
          (first >= '0' && first <= '9'))) {
        return AdmissionError::kLexical;
    }
    for (unsigned char c : text) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                        (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!ok) {
            return AdmissionError::kLexical;
        }
    }
    return AdmissionError::kOk;
}

AdmissionError validate_upstream_name(std::string_view text) {
    if (text.empty() || text.size() > 128 || text.front() == '/' || text.back() == '/') {
        return AdmissionError::kLexical;
    }
    if (text.find("//") != std::string_view::npos) {
        return AdmissionError::kLexical;
    }
    const unsigned char first = static_cast<unsigned char>(text.front());
    if (!((first >= 'A' && first <= 'Z') || (first >= 'a' && first <= 'z') ||
          (first >= '0' && first <= '9'))) {
        return AdmissionError::kLexical;
    }
    for (unsigned char c : text) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                        (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '+' || c == '/' ||
                        c == '-';
        if (!ok) {
            return AdmissionError::kLexical;
        }
    }
    std::size_t start = 0;
    for (std::size_t index = 0; index <= text.size(); ++index) {
        if (index == text.size() || text[index] == '/') {
            const std::string_view segment = text.substr(start, index - start);
            if (segment.empty() || segment == "." || segment == "..") {
                return AdmissionError::kLexical;
            }
            start = index + 1;
        }
    }
    return AdmissionError::kOk;
}

AdmissionError validate_yes_no(std::string_view text) {
    if (text == "yes" || text == "no") {
        return AdmissionError::kOk;
    }
    return AdmissionError::kLexical;
}

AdmissionError validate_phase_id(std::string_view text) {
    constexpr std::string_view kPrefix{"9-c"};
    if (text.size() <= kPrefix.size() || text.substr(0, kPrefix.size()) != kPrefix) {
        return AdmissionError::kLexical;
    }
    const std::string_view number = text.substr(kPrefix.size());
    std::uint32_t value = 0;
    if (parse_uint(number, value) != AdmissionError::kOk || value < 1 || value > 44) {
        return AdmissionError::kLexical;
    }
    return AdmissionError::kOk;
}

AdmissionError validate_pattern_hex(std::string_view text) {
    // Lexical form only. Matching a tensor name, with 0x2A as the only metacharacter,
    // needs the tensor name and belongs to the later tensor phase.
    if (!is_lower_hex(text, 2, 512)) {
        return AdmissionError::kLexical;
    }
    return AdmissionError::kOk;
}

AdmissionError validate_name_hex(std::string_view text, std::string& decoded) {
    if (!is_lower_hex(text, 2, 512)) {
        return AdmissionError::kLexical;
    }
    std::vector<std::uint8_t> bytes;
    if (decode_hex(text, bytes) != AdmissionError::kOk) {
        return AdmissionError::kLexical;
    }
    std::string utf8;
    if (reject_utf8(bytes.data(), bytes.size(), utf8) != AdmissionError::kOk) {
        return AdmissionError::kLexical;
    }
    decoded = std::move(utf8);
    return AdmissionError::kOk;
}

} // namespace inferedge
