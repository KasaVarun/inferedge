cmake_minimum_required(VERSION 3.24)

project(
    inferedge
    VERSION 0.1.0
    DESCRIPTION "High-performance C++ AI inference runtime and scheduler"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Single-config generators (Ninja, Unix Makefiles) need an explicit build type.
# Multi-config generators (Xcode, Visual Studio) choose the type at build time.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Debug, Release, RelWithDebInfo, or MinSizeRel" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

add_library(inferedge_warnings INTERFACE)
if(MSVC)
    target_compile_options(inferedge_warnings INTERFACE /W4 /permissive-)
else()
    target_compile_options(
        inferedge_warnings
        INTERFACE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wshadow
            -Wnon-virtual-dtor
    )
endif()

add_library(inferedge_core src/cli.cpp)
target_include_directories(inferedge_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(inferedge_core PUBLIC cxx_std_20)
target_compile_definitions(inferedge_core PUBLIC INFEREDGE_VERSION="${PROJECT_VERSION}")
target_link_libraries(inferedge_core PUBLIC inferedge_warnings)

add_executable(inferedge src/main.cpp)
target_link_libraries(inferedge PRIVATE inferedge_core)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
# Avoids a mismatch between the CRT settings of GoogleTest and the tests on MSVC.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(inferedge_tests tests/cli_test.cpp tests/version_test.cpp)
target_link_libraries(inferedge_tests PRIVATE inferedge_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(inferedge_tests)

add_test(NAME inferedge_cli_help COMMAND inferedge --help)
add_test(NAME inferedge_cli_version COMMAND inferedge --version)
set_tests_properties(inferedge_cli_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage: inferedge")
set_tests_properties(inferedge_cli_version PROPERTIES PASS_REGULAR_EXPRESSION "inferedge 0\\.1\\.0")
