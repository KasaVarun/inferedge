#pragma once

#include <string_view>

namespace inferedge {

// The name is fixed. The version string is injected by CMake from project().
inline constexpr std::string_view project_name() noexcept {
    return "inferedge";
}

inline constexpr std::string_view version() noexcept {
#ifdef INFEREDGE_VERSION
    return INFEREDGE_VERSION;
#else
    return "0.0.0-unknown";
#endif
}

} // namespace inferedge
