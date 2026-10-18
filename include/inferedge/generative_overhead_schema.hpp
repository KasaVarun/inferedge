#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace inferedge {
namespace generative_overhead {

inline constexpr std::string_view kSchemaFamily = "generative_overhead";
inline constexpr int kSchemaVersion = 1;
inline constexpr std::string_view kEvidencePathContract = "generative-overhead-r7-inventory-v1";
inline constexpr std::string_view kFixtureGenerationAlgorithm = "generative_overhead_fixture_gen";
inline constexpr int kFixtureGenerationVersion = 1;

inline constexpr std::string_view kModelId = "qwen3-0.6b";
inline constexpr std::string_view kPackageDirectory =
    "/Users/varunkasa/InferEdgeModels/phase-9-b2a-qwen3-0.6b/package";
inline constexpr std::string_view kGgufFile = "qwen3-0.6b-Q4_K_M.gguf";
inline constexpr std::string_view kGgufSha256 =
    "3f384c3796ca2547c5547be6bca3d487a18807cc1a2c259de6e0e1fa433bb76d";
inline constexpr std::uint64_t kGgufSizeBytes = 484220192ULL;
inline constexpr std::string_view kPackageManifestSha256 =
    "e6ff54460e7e30b3208f76d29eb49a1388e11f2223b478fa4bbf47a48e9ee281";
inline constexpr std::string_view kChatTemplateFile = "qwen3-non-thinking-v1.txt";
inline constexpr std::string_view kChatTemplateSha256 =
    "6cb4fbe89e712bf8e8527aa8fb7d80eb0cad93ae1e236bb4be9c96d71809976a";
inline constexpr std::size_t kChatTemplateBytes = 75;
inline constexpr std::size_t kUserPlaceholderOffset = 17;
inline constexpr std::string_view kUserPlaceholder = "{user}";
inline constexpr std::uint32_t kContextLimit = 4096;
inline constexpr std::uint32_t kPhysicalUbatch = 512;
inline constexpr std::uint32_t kSeqMax = 1;
inline constexpr int kWarmupCountPerWorker = 2;
inline constexpr int kMeasuredPairsPerCell = 20;
inline constexpr int kCellCount = 12;
inline constexpr int kPermutationSeed = 20261017;
inline constexpr int kBootstrapSeed = 20261018;
inline constexpr int kBootstrapReplicates = 10000;
inline constexpr std::int64_t kAbsoluteThresholdNs = 2000000;
inline constexpr int kRelativeThresholdNumerator = 5;
inline constexpr int kRelativeThresholdDenominator = 100;
inline constexpr std::string_view kEmptySha256 =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

inline constexpr std::array<std::string_view, 12> kConstructionCells{
    "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "L01"};
inline constexpr std::array<std::string_view, 12> kGoldenCellOrder{
    "C05", "C02", "C10", "C06", "C04", "C09", "C11", "L01", "C07", "C08", "C01", "C03"};

enum class MeasurementClass { kWarmRequest, kLoad };
enum class CellRole { kPrimary, kSecondary, kDescriptiveLoad };
enum class Implementation { kInferedge, kReference };

struct CellSpec {
    std::string_view id;
    std::string_view fixture_id; // empty for L01
    std::optional<std::uint32_t> max_new_tokens;
    std::optional<std::uint32_t> threads;
    MeasurementClass measurement_class;
    CellRole role;
    bool allow_eos;
    bool allow_maximum_tokens;
    bool allow_load_success;
};

inline constexpr std::array<CellSpec, 12> kCells{{
    {"C01", "P_tiny", 1, 1, MeasurementClass::kWarmRequest, CellRole::kPrimary, false, true, false},
    {"C02", "P_128", 1, 1, MeasurementClass::kWarmRequest, CellRole::kSecondary, false, true,
     false},
    {"C03", "P_512", 1, 1, MeasurementClass::kWarmRequest, CellRole::kSecondary, false, true,
     false},
    {"C04", "P_1024", 1, 1, MeasurementClass::kWarmRequest, CellRole::kSecondary, false, true,
     false},
    {"C05", "P_2048", 1, 1, MeasurementClass::kWarmRequest, CellRole::kSecondary, false, true,
     false},
    {"C06", "P_near", 1, 1, MeasurementClass::kWarmRequest, CellRole::kSecondary, false, true,
     false},
    {"C07", "P_tiny", 16, 1, MeasurementClass::kWarmRequest, CellRole::kSecondary, true, true,
     false},
    {"C08", "P_tiny", 128, 1, MeasurementClass::kWarmRequest, CellRole::kSecondary, true, true,
     false},
    {"C09", "P_tiny", 1, 2, MeasurementClass::kWarmRequest, CellRole::kSecondary, false, true,
     false},
    {"C10", "P_tiny", 1, 4, MeasurementClass::kWarmRequest, CellRole::kSecondary, false, true,
     false},
    {"C11", "P_tiny", 1, 6, MeasurementClass::kWarmRequest, CellRole::kSecondary, false, true,
     false},
    {"L01", "", std::nullopt, std::nullopt, MeasurementClass::kLoad, CellRole::kDescriptiveLoad,
     false, false, true},
}};

inline constexpr const CellSpec* find_cell(std::string_view id) noexcept {
    for (const CellSpec& cell : kCells) {
        if (cell.id == id) {
            return &cell;
        }
    }
    return nullptr;
}

inline constexpr bool pair_index_is_even(int pair_index) noexcept {
    return pair_index % 2 == 0;
}

inline constexpr std::string_view arm_order_for_pair(int pair_index) noexcept {
    return pair_index_is_even(pair_index) ? "ie_then_ref" : "ref_then_ie";
}

inline constexpr std::string_view implementation_for_order_position(int pair_index,
                                                                    int order_position) noexcept {
    const bool ie_first = pair_index_is_even(pair_index);
    if (order_position == 0) {
        return ie_first ? "inferedge" : "reference";
    }
    return ie_first ? "reference" : "inferedge";
}

inline void encode_token_ids_le(const std::int32_t* ids, std::size_t count,
                                std::uint8_t* out) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        const auto value = static_cast<std::uint32_t>(ids[i]);
        out[i * 4U] = static_cast<std::uint8_t>(value);
        out[i * 4U + 1U] = static_cast<std::uint8_t>(value >> 8U);
        out[i * 4U + 2U] = static_cast<std::uint8_t>(value >> 16U);
        out[i * 4U + 3U] = static_cast<std::uint8_t>(value >> 24U);
    }
}

inline bool token_id_in_int32_range(std::int64_t value) noexcept {
    return value >= static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) &&
           value <= static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
}

inline std::string json_escape(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (unsigned char ch : text) {
        switch (ch) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (ch < 0x20U) {
                static constexpr char hex[] = "0123456789abcdef";
                out += "\\u00";
                out += hex[ch >> 4U];
                out += hex[ch & 0x0fU];
            } else {
                out += static_cast<char>(ch);
            }
            break;
        }
    }
    return out;
}

inline bool json_unescape(std::string_view text, std::string& out) {
    out.clear();
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch != '\\') {
            out += ch;
            continue;
        }
        if (i + 1 >= text.size()) {
            return false;
        }
        const char esc = text[++i];
        switch (esc) {
        case '"':
        case '\\':
        case '/':
            out += esc;
            break;
        case 'b':
            out += '\b';
            break;
        case 'f':
            out += '\f';
            break;
        case 'n':
            out += '\n';
            break;
        case 'r':
            out += '\r';
            break;
        case 't':
            out += '\t';
            break;
        case 'u':
            if (i + 4 >= text.size()) {
                return false;
            }
            {
                unsigned value = 0;
                for (int n = 0; n < 4; ++n) {
                    const char hex = text[++i];
                    value <<= 4U;
                    if (hex >= '0' && hex <= '9') {
                        value |= static_cast<unsigned>(hex - '0');
                    } else if (hex >= 'a' && hex <= 'f') {
                        value |= static_cast<unsigned>(hex - 'a' + 10);
                    } else if (hex >= 'A' && hex <= 'F') {
                        value |= static_cast<unsigned>(hex - 'A' + 10);
                    } else {
                        return false;
                    }
                }
                if (value > 0x7fU) {
                    return false;
                }
                out += static_cast<char>(value);
            }
            break;
        default:
            return false;
        }
    }
    return true;
}

inline bool json_extract_string(std::string_view object, std::string_view key, std::string& value) {
    const std::string needle = "\"" + std::string(key) + "\"";
    const auto pos = object.find(needle);
    if (pos == std::string_view::npos) {
        return false;
    }
    auto cursor = pos + needle.size();
    while (cursor < object.size() &&
           (object[cursor] == ' ' || object[cursor] == '\t' || object[cursor] == ':')) {
        if (object[cursor] == ':') {
            ++cursor;
            break;
        }
        ++cursor;
    }
    while (cursor < object.size() && (object[cursor] == ' ' || object[cursor] == '\t')) {
        ++cursor;
    }
    if (cursor >= object.size() || object[cursor] != '"') {
        return false;
    }
    ++cursor;
    std::string raw;
    bool escaped = false;
    for (; cursor < object.size(); ++cursor) {
        const char ch = object[cursor];
        if (escaped) {
            raw += ch;
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            raw += ch;
            escaped = true;
            continue;
        }
        if (ch == '"') {
            return json_unescape(raw, value);
        }
        raw += ch;
    }
    return false;
}

inline bool json_extract_int(std::string_view object, std::string_view key, std::int64_t& value) {
    const std::string needle = "\"" + std::string(key) + "\"";
    const auto pos = object.find(needle);
    if (pos == std::string_view::npos) {
        return false;
    }
    auto cursor = pos + needle.size();
    while (cursor < object.size() && object[cursor] != ':') {
        ++cursor;
    }
    if (cursor >= object.size()) {
        return false;
    }
    ++cursor;
    while (cursor < object.size() && (object[cursor] == ' ' || object[cursor] == '\t')) {
        ++cursor;
    }
    if (cursor >= object.size()) {
        return false;
    }
    const bool negative = object[cursor] == '-';
    if (negative) {
        ++cursor;
    }
    if (cursor >= object.size() || object[cursor] < '0' || object[cursor] > '9') {
        return false;
    }
    std::int64_t parsed = 0;
    for (; cursor < object.size(); ++cursor) {
        const char ch = object[cursor];
        if (ch < '0' || ch > '9') {
            break;
        }
        parsed = parsed * 10 + (ch - '0');
    }
    value = negative ? -parsed : parsed;
    return true;
}

inline std::string json_token_array(const std::vector<std::int32_t>& ids) {
    std::string out = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        out += std::to_string(ids[i]);
    }
    out += ']';
    return out;
}

} // namespace generative_overhead
} // namespace inferedge
