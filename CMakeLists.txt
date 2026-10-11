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

add_library(
    inferedge_core
    src/cli.cpp
    src/request.cpp
    src/request_lifecycle.cpp
    src/float_buffer.cpp
)
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
add_executable(
    inferedge_tests
    tests/cli_test.cpp
    tests/version_test.cpp
    tests/request_test.cpp
    tests/request_lifecycle_test.cpp
    tests/float_buffer_test.cpp
    tests/ownership_test.cpp
    tests/native_resource_test.cpp
    tests/stl_container_test.cpp
    tests/template_test.cpp
)
target_link_libraries(inferedge_tests PRIVATE inferedge_core GTest::gtest_main)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntime.cmake")

# The Add fixture is committed. Configuration checks its bytes and does not regenerate it.
set(INFEREDGE_ADD_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/add_f32.onnx")
set(INFEREDGE_ADD_MODEL_SHA256 "37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447")
if(NOT EXISTS "${INFEREDGE_ADD_MODEL}")
    message(FATAL_ERROR "ONNX Add fixture is missing: ${INFEREDGE_ADD_MODEL}")
endif()
file(SHA256 "${INFEREDGE_ADD_MODEL}" inferedge_add_model_sha256)
if(NOT inferedge_add_model_sha256 STREQUAL INFEREDGE_ADD_MODEL_SHA256)
    message(FATAL_ERROR
        "ONNX Add fixture SHA-256 mismatch.\n"
        "  expected: ${INFEREDGE_ADD_MODEL_SHA256}\n"
        "  actual:   ${inferedge_add_model_sha256}\n"
        "Regenerate models/add_f32.onnx with models/generate_add_model.py, then update the expected hash."
    )
endif()

add_executable(
    inferedge_onnx_smoke
    tests/onnx_runtime_smoke_test.cpp
    tests/onnx_add_inference_test.cpp
)
target_link_libraries(
    inferedge_onnx_smoke
    PRIVATE
        inferedge_warnings
        onnxruntime::onnxruntime
        GTest::gtest_main
)
target_compile_definitions(
    inferedge_onnx_smoke
    PRIVATE
        "INFEREDGE_ADD_MODEL_PATH=\"${INFEREDGE_ADD_MODEL}\""
        "INFEREDGE_INVALID_MODEL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/data/not_a_model.bin\""
)
set_target_properties(inferedge_onnx_smoke PROPERTIES
    BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
)

include(GoogleTest)
gtest_discover_tests(inferedge_tests)
gtest_discover_tests(
    inferedge_onnx_smoke
    PROPERTIES ENVIRONMENT "ORT_DISABLE_TELEMETRY=1"
)

add_test(NAME inferedge_cli_help COMMAND inferedge --help)
add_test(NAME inferedge_cli_version COMMAND inferedge --version)
set_tests_properties(inferedge_cli_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage: inferedge")
set_tests_properties(inferedge_cli_version PROPERTIES PASS_REGULAR_EXPRESSION "inferedge 0\\.1\\.0")
