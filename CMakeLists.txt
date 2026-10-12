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
    src/benchmark_statistics.cpp
    src/cli.cpp
    src/request.cpp
    src/request_lifecycle.cpp
    src/float_buffer.cpp
)
target_include_directories(inferedge_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(inferedge_core PUBLIC cxx_std_20)
target_compile_definitions(inferedge_core PUBLIC INFEREDGE_VERSION="${PROJECT_VERSION}")
target_link_libraries(inferedge_core PUBLIC inferedge_warnings)

add_executable(inferedge src/main.cpp src/run_command.cpp)
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
    tests/benchmark_statistics_test.cpp
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

# Committed fixtures. Configuration checks their bytes and does not regenerate them.
set(INFEREDGE_ADD_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/add_f32.onnx")
set(INFEREDGE_ADD_MODEL_SHA256 "37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447")
set(INFEREDGE_INT64_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/int64_echo.onnx")
set(INFEREDGE_INT64_MODEL_SHA256 "4bae136df39291cfccfb7a1b9b0f03186598326f441f755095459ab03805b145")
set(INFEREDGE_RANK2_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/rank2_f32.onnx")
set(INFEREDGE_RANK2_MODEL_SHA256 "193fc307deda81dee0e3b0f5999809a80d92ec46070c66c187b4578daefb5bc2")
set(INFEREDGE_MNIST_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/mnist_mlp.onnx")
set(INFEREDGE_MNIST_MODEL_SHA256 "3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce")
set(INFEREDGE_MNIST_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/mnist")
set(INFEREDGE_MODEL_PATHS
    "${INFEREDGE_ADD_MODEL}" "${INFEREDGE_INT64_MODEL}" "${INFEREDGE_RANK2_MODEL}"
    "${INFEREDGE_MNIST_MODEL}")
set(INFEREDGE_MODEL_HASHES
    "${INFEREDGE_ADD_MODEL_SHA256}" "${INFEREDGE_INT64_MODEL_SHA256}" "${INFEREDGE_RANK2_MODEL_SHA256}"
    "${INFEREDGE_MNIST_MODEL_SHA256}")
set(INFEREDGE_MODEL_LABELS "Add" "int64 Identity" "rank-2 float32" "MNIST MLP")
foreach(model_path expected_hash model_label IN ZIP_LISTS
        INFEREDGE_MODEL_PATHS INFEREDGE_MODEL_HASHES INFEREDGE_MODEL_LABELS)
    if(NOT EXISTS "${model_path}")
        message(FATAL_ERROR "ONNX ${model_label} fixture is missing: ${model_path}")
    endif()
    file(SHA256 "${model_path}" actual_hash)
    if(NOT actual_hash STREQUAL expected_hash)
        message(FATAL_ERROR
            "ONNX ${model_label} fixture SHA-256 mismatch.\n"
            "  expected: ${expected_hash}\n"
            "  actual:   ${actual_hash}\n"
            "Regenerate the model with its script in models/, then update the expected hash."
        )
    endif()
endforeach()

add_library(inferedge_onnx src/inference_engine.cpp)
target_include_directories(inferedge_onnx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(inferedge_onnx PUBLIC cxx_std_20)
target_link_libraries(
    inferedge_onnx
    PUBLIC onnxruntime::onnxruntime
    PRIVATE inferedge_warnings
)

target_link_libraries(inferedge PRIVATE inferedge_onnx)

find_package(Threads REQUIRED)
add_library(inferedge_concurrent src/concurrent_inference.cpp)
target_include_directories(inferedge_concurrent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(inferedge_concurrent PUBLIC cxx_std_20)
target_link_libraries(
    inferedge_concurrent
    PUBLIC inferedge_onnx Threads::Threads
    PRIVATE inferedge_warnings
)

add_library(inferedge_concurrent_benchmark src/concurrent_benchmark.cpp)
target_include_directories(inferedge_concurrent_benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(inferedge_concurrent_benchmark PUBLIC cxx_std_20)
target_link_libraries(
    inferedge_concurrent_benchmark
    PUBLIC inferedge_concurrent inferedge_core Threads::Threads
    PRIVATE inferedge_warnings
)

add_library(inferedge_benchmark src/benchmark_runner.cpp)
target_include_directories(inferedge_benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(inferedge_benchmark PUBLIC cxx_std_20)
target_link_libraries(
    inferedge_benchmark
    PUBLIC inferedge_core inferedge_onnx
    PRIVATE inferedge_warnings
)

set_target_properties(inferedge PROPERTIES
    BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
)

add_library(
    inferedge_benchmark_cli
    src/benchmark_cli.cpp
    src/benchmark_command.cpp
    src/benchmark_json.cpp
)
target_include_directories(inferedge_benchmark_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(inferedge_benchmark_cli PUBLIC cxx_std_20)
target_link_libraries(
    inferedge_benchmark_cli
    PUBLIC inferedge_benchmark
    PRIVATE inferedge_warnings
)
target_compile_definitions(
    inferedge_benchmark_cli
    PRIVATE "INFEREDGE_MNIST_FIXTURE_DIR=\"${INFEREDGE_MNIST_FIXTURE_DIR}\""
)

add_executable(inferedge-bench src/benchmark_main.cpp)
target_link_libraries(inferedge-bench PRIVATE inferedge_benchmark_cli)
set_target_properties(inferedge-bench PROPERTIES
    BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
)

add_library(
    inferedge_concurrent_benchmark_cli
    src/concurrent_benchmark_cli.cpp
    src/concurrent_benchmark_command.cpp
    src/concurrent_benchmark_json.cpp
)
target_include_directories(inferedge_concurrent_benchmark_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(inferedge_concurrent_benchmark_cli PUBLIC cxx_std_20)
target_link_libraries(
    inferedge_concurrent_benchmark_cli
    PUBLIC inferedge_concurrent_benchmark
    PRIVATE inferedge_warnings
)
target_compile_definitions(
    inferedge_concurrent_benchmark_cli
    PRIVATE "INFEREDGE_MNIST_FIXTURE_DIR=\"${INFEREDGE_MNIST_FIXTURE_DIR}\""
)

add_executable(inferedge-bench-concurrent src/concurrent_benchmark_main.cpp)
target_link_libraries(inferedge-bench-concurrent PRIVATE inferedge_concurrent_benchmark_cli)
set_target_properties(inferedge-bench-concurrent PROPERTIES
    BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
)

add_executable(
    inferedge_onnx_smoke
    tests/onnx_runtime_smoke_test.cpp
    tests/onnx_add_inference_test.cpp
    tests/inference_engine_test.cpp
    tests/mnist_inference_test.cpp
    tests/benchmark_runner_test.cpp
    tests/concurrent_inference_test.cpp
    tests/concurrent_benchmark_test.cpp
    tests/concurrent_reliability_test.cpp
)
target_link_libraries(
    inferedge_onnx_smoke
    PRIVATE
        inferedge_warnings
        inferedge_benchmark
        inferedge_concurrent_benchmark
        GTest::gtest_main
)
target_compile_definitions(
    inferedge_onnx_smoke
    PRIVATE
        "INFEREDGE_ADD_MODEL_PATH=\"${INFEREDGE_ADD_MODEL}\""
        "INFEREDGE_INVALID_MODEL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/data/not_a_model.bin\""
        "INFEREDGE_INT64_MODEL_PATH=\"${INFEREDGE_INT64_MODEL}\""
        "INFEREDGE_MNIST_MODEL_PATH=\"${INFEREDGE_MNIST_MODEL}\""
        "INFEREDGE_MNIST_FIXTURE_DIR=\"${INFEREDGE_MNIST_FIXTURE_DIR}\""
)
set_target_properties(inferedge_onnx_smoke PROPERTIES
    BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
)

target_sources(inferedge_tests PRIVATE tests/cli_inference_test.cpp)
target_compile_definitions(
    inferedge_tests
    PRIVATE
        "INFEREDGE_EXECUTABLE=\"$<TARGET_FILE:inferedge>\""
        "INFEREDGE_ADD_MODEL_PATH=\"${INFEREDGE_ADD_MODEL}\""
        "INFEREDGE_INVALID_MODEL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/data/not_a_model.bin\""
        "INFEREDGE_RANK2_MODEL_PATH=\"${INFEREDGE_RANK2_MODEL}\""
)
add_dependencies(inferedge_tests inferedge)

include(GoogleTest)
gtest_discover_tests(
    inferedge_tests
    PROPERTIES ENVIRONMENT "ORT_DISABLE_TELEMETRY=1"
)
gtest_discover_tests(
    inferedge_onnx_smoke
    PROPERTIES
        ENVIRONMENT "ORT_DISABLE_TELEMETRY=1"
        TIMEOUT 120
)

add_executable(
    inferedge_bench_tests
    tests/benchmark_cli_test.cpp
    tests/benchmark_command_test.cpp
    tests/benchmark_json_test.cpp
)
target_link_libraries(
    inferedge_bench_tests
    PRIVATE
        inferedge_warnings
        inferedge_benchmark_cli
        GTest::gtest_main
)
target_compile_definitions(
    inferedge_bench_tests
    PRIVATE
        "INFEREDGE_BENCH_EXECUTABLE=\"$<TARGET_FILE:inferedge-bench>\""
        "INFEREDGE_ADD_MODEL_PATH=\"${INFEREDGE_ADD_MODEL}\""
        "INFEREDGE_MNIST_MODEL_PATH=\"${INFEREDGE_MNIST_MODEL}\""
        "INFEREDGE_RANK2_MODEL_PATH=\"${INFEREDGE_RANK2_MODEL}\""
        "INFEREDGE_INT64_MODEL_PATH=\"${INFEREDGE_INT64_MODEL}\""
        "INFEREDGE_INVALID_MODEL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/data/not_a_model.bin\""
        "INFEREDGE_MNIST_FIXTURE_DIR=\"${INFEREDGE_MNIST_FIXTURE_DIR}\""
)
set_target_properties(inferedge_bench_tests PROPERTIES
    BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
)
add_dependencies(inferedge_bench_tests inferedge-bench)
gtest_discover_tests(
    inferedge_bench_tests
    PROPERTIES ENVIRONMENT "ORT_DISABLE_TELEMETRY=1"
)

add_test(NAME inferedge_cli_help COMMAND inferedge --help)
add_test(NAME inferedge_cli_version COMMAND inferedge --version)
set_tests_properties(inferedge_cli_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage: inferedge")
set_tests_properties(inferedge_cli_version PROPERTIES PASS_REGULAR_EXPRESSION "inferedge 0\\.1\\.0")
add_executable(
    inferedge_bench_concurrent_tests
    tests/concurrent_benchmark_cli_test.cpp
    tests/concurrent_benchmark_json_test.cpp
)
target_link_libraries(
    inferedge_bench_concurrent_tests
    PRIVATE
        inferedge_warnings
        inferedge_concurrent_benchmark_cli
        GTest::gtest_main
)
target_compile_definitions(
    inferedge_bench_concurrent_tests
    PRIVATE
        "INFEREDGE_BENCH_CONCURRENT_EXECUTABLE=\"$<TARGET_FILE:inferedge-bench-concurrent>\""
        "INFEREDGE_BENCH_EXECUTABLE=\"$<TARGET_FILE:inferedge-bench>\""
        "INFEREDGE_ADD_MODEL_PATH=\"${INFEREDGE_ADD_MODEL}\""
        "INFEREDGE_MNIST_MODEL_PATH=\"${INFEREDGE_MNIST_MODEL}\""
        "INFEREDGE_RANK2_MODEL_PATH=\"${INFEREDGE_RANK2_MODEL}\""
        "INFEREDGE_INT64_MODEL_PATH=\"${INFEREDGE_INT64_MODEL}\""
        "INFEREDGE_INVALID_MODEL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/data/not_a_model.bin\""
        "INFEREDGE_MNIST_FIXTURE_DIR=\"${INFEREDGE_MNIST_FIXTURE_DIR}\""
)
set_target_properties(inferedge_bench_concurrent_tests PROPERTIES
    BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
)
add_dependencies(inferedge_bench_concurrent_tests inferedge-bench-concurrent inferedge-bench)
gtest_discover_tests(
    inferedge_bench_concurrent_tests
    PROPERTIES ENVIRONMENT "ORT_DISABLE_TELEMETRY=1"
)

add_test(NAME inferedge_bench_help COMMAND inferedge-bench --help)
add_test(NAME inferedge_bench_version COMMAND inferedge-bench --version)
set_tests_properties(inferedge_bench_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage: inferedge-bench")
set_tests_properties(
    inferedge_bench_version
    PROPERTIES PASS_REGULAR_EXPRESSION "inferedge-bench 0\\.1\\.0"
)
add_test(NAME inferedge_bench_concurrent_help COMMAND inferedge-bench-concurrent --help)
add_test(NAME inferedge_bench_concurrent_version COMMAND inferedge-bench-concurrent --version)
set_tests_properties(
    inferedge_bench_concurrent_help
    PROPERTIES PASS_REGULAR_EXPRESSION "Usage: inferedge-bench-concurrent"
)
set_tests_properties(
    inferedge_bench_concurrent_version
    PROPERTIES PASS_REGULAR_EXPRESSION "inferedge-bench-concurrent 0\\.1\\.0"
)

# Header-only bounded queue. This target does not link ONNX Runtime.
add_executable(inferedge_queue_tests tests/bounded_queue_test.cpp)
target_include_directories(inferedge_queue_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(inferedge_queue_tests PRIVATE cxx_std_20)
target_link_libraries(
    inferedge_queue_tests
    PRIVATE
        inferedge_warnings
        GTest::gtest_main
        Threads::Threads
)
gtest_discover_tests(
    inferedge_queue_tests
    PROPERTIES TIMEOUT 30
)

# Local bounded stress. Default CI does not build or discover this executable.
option(INFEREDGE_ENABLE_LONG_STRESS "Build bounded local concurrency stress tests" OFF)
if(INFEREDGE_ENABLE_LONG_STRESS)
    add_executable(inferedge_concurrent_stress tests/concurrent_stress_test.cpp)
    target_link_libraries(
        inferedge_concurrent_stress
        PRIVATE
            inferedge_warnings
            inferedge_benchmark
            inferedge_concurrent_benchmark
            GTest::gtest_main
    )
    target_compile_definitions(
        inferedge_concurrent_stress
        PRIVATE
            "INFEREDGE_ADD_MODEL_PATH=\"${INFEREDGE_ADD_MODEL}\""
            "INFEREDGE_MNIST_MODEL_PATH=\"${INFEREDGE_MNIST_MODEL}\""
            "INFEREDGE_MNIST_FIXTURE_DIR=\"${INFEREDGE_MNIST_FIXTURE_DIR}\""
    )
    set_target_properties(
        inferedge_concurrent_stress
        PROPERTIES
            BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
            INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
    )
    gtest_discover_tests(
        inferedge_concurrent_stress
        PROPERTIES
            ENVIRONMENT "ORT_DISABLE_TELEMETRY=1"
            TIMEOUT 300
    )
endif()
