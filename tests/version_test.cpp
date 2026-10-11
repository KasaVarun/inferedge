#include "inferedge/version.hpp"

#include <gtest/gtest.h>
#include <string_view>

TEST(VersionTest, ProjectNameIsStable) {
    EXPECT_EQ(inferedge::project_name(), "inferedge");
}

TEST(VersionTest, VersionComesFromTheBuildSystem) {
    // CMake defines INFEREDGE_VERSION from project(VERSION). The fallback
    // string would mean the compile definition was not applied.
    EXPECT_EQ(inferedge::version(), "0.1.0");
    EXPECT_NE(inferedge::version(), "0.0.0-unknown");
}

TEST(VersionTest, VersionHasMajorMinorPatch) {
    const std::string_view value = inferedge::version();
    const std::size_t first_dot = value.find('.');
    const std::size_t second_dot =
        value.find('.', first_dot == std::string_view::npos ? 0 : first_dot + 1);
    EXPECT_NE(first_dot, std::string_view::npos);
    EXPECT_NE(second_dot, std::string_view::npos);
    EXPECT_LT(first_dot, second_dot);
}
