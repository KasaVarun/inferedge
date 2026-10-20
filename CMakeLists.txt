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
    src/model_package_manifest.cpp
    src/generate_cli.cpp
    tests/model_package_manifest_test.cpp
    tests/generate_cli_test.cpp
    src/admission_record.cpp
    tests/admission_record_test.cpp
    src/tensor_canonical.cpp
    tests/tensor_canonical_test.cpp
    src/oracle_digest.cpp
    tests/oracle_digest_test.cpp
    src/evidence_store.cpp
    tests/evidence_store_test.cpp
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
set(INFEREDGE_BATCH_ADD_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/batch_add_f32.onnx")
set(INFEREDGE_BATCH_ADD_MODEL_SHA256
    "f33bbe0ad5420419dd2dbb5322f4172e213b854f1124c660b1d732e8f8ce3b23")
set(INFEREDGE_MLP_SMALL_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/mlp_small_f32.onnx")
set(INFEREDGE_MLP_SMALL_MODEL_SHA256
    "bcf02cac789c65cb12785ccfd175827cb21f89574ef298af009b74d7131df1bd")
set(INFEREDGE_MLP_MEDIUM_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/mlp_medium_f32.onnx")
set(INFEREDGE_MLP_MEDIUM_MODEL_SHA256
    "8d43a97b8d44d3d1f601570c8b09b08c90dda4977e6f8b7123eea8c0455024dd")
set(INFEREDGE_GEMM_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/gemm_f32.onnx")
set(INFEREDGE_GEMM_MODEL_SHA256
    "a6203020b870eb04f9fd8e836cf4ccd930a57156a026d2a50163121d83debf32")
set(INFEREDGE_RESNET18_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/resnet18_f32.onnx")
set(INFEREDGE_RESNET18_MODEL_SHA256
    "cfc843f2939dc6065d95d911ec778f57ca5f0f25faa4988a5b5c61011bebd935")
set(INFEREDGE_MLP_SMALL_WEIGHTS "${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/mlp_small_f32.weights")
set(INFEREDGE_MLP_SMALL_WEIGHTS_SHA256
    "b6335064c4e2c89add3ed54452e27dacf3ea7d485be6adbdd24fae1e2e5f1899")
set(INFEREDGE_MLP_MEDIUM_WEIGHTS "${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/mlp_medium_f32.weights")
set(INFEREDGE_MLP_MEDIUM_WEIGHTS_SHA256
    "d3dd2f6148a6684288d4f25b0c426eefde7227468be9120ddd08cd99cbb9059e")
set(INFEREDGE_GEMM_WEIGHTS "${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/gemm_f32.weights")
set(INFEREDGE_GEMM_WEIGHTS_SHA256
    "a335c8b12defcb885f8bada9ef1092655d024d5586fe011a9336ad2d472f82b6")
set(INFEREDGE_MNIST_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/mnist")
set(INFEREDGE_MLP_SMALL_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/mlp_small_f32")
set(INFEREDGE_MLP_MEDIUM_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/mlp_medium_f32")
set(INFEREDGE_GEMM_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/gemm_f32")
set(INFEREDGE_RESNET18_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures/resnet18_f32")
set(INFEREDGE_MODEL_PATHS
    "${INFEREDGE_ADD_MODEL}" "${INFEREDGE_INT64_MODEL}" "${INFEREDGE_RANK2_MODEL}"
    "${INFEREDGE_MNIST_MODEL}" "${INFEREDGE_BATCH_ADD_MODEL}"
    "${INFEREDGE_MLP_SMALL_MODEL}" "${INFEREDGE_MLP_MEDIUM_MODEL}" "${INFEREDGE_GEMM_MODEL}"
    "${INFEREDGE_RESNET18_MODEL}")
set(INFEREDGE_MODEL_HASHES
    "${INFEREDGE_ADD_MODEL_SHA256}" "${INFEREDGE_INT64_MODEL_SHA256}" "${INFEREDGE_RANK2_MODEL_SHA256}"
    "${INFEREDGE_MNIST_MODEL_SHA256}" "${INFEREDGE_BATCH_ADD_MODEL_SHA256}"
    "${INFEREDGE_MLP_SMALL_MODEL_SHA256}" "${INFEREDGE_MLP_MEDIUM_MODEL_SHA256}"
    "${INFEREDGE_GEMM_MODEL_SHA256}" "${INFEREDGE_RESNET18_MODEL_SHA256}")
set(INFEREDGE_MODEL_LABELS
    "Add" "int64 Identity" "rank-2 float32" "MNIST MLP" "batch Add"
    "small MLP" "medium MLP" "GEMM" "ResNet-18")
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

set(INFEREDGE_WEIGHT_PATHS
    "${INFEREDGE_MLP_SMALL_WEIGHTS}" "${INFEREDGE_MLP_MEDIUM_WEIGHTS}" "${INFEREDGE_GEMM_WEIGHTS}")
set(INFEREDGE_WEIGHT_HASHES
    "${INFEREDGE_MLP_SMALL_WEIGHTS_SHA256}" "${INFEREDGE_MLP_MEDIUM_WEIGHTS_SHA256}"
    "${INFEREDGE_GEMM_WEIGHTS_SHA256}")
set(INFEREDGE_WEIGHT_LABELS "small MLP" "medium MLP" "GEMM")
foreach(weight_path expected_hash weight_label IN ZIP_LISTS
        INFEREDGE_WEIGHT_PATHS INFEREDGE_WEIGHT_HASHES INFEREDGE_WEIGHT_LABELS)
    if(NOT EXISTS "${weight_path}")
        message(FATAL_ERROR "Weight sidecar is missing for ${weight_label}: ${weight_path}")
    endif()
    file(SHA256 "${weight_path}" actual_hash)
    if(NOT actual_hash STREQUAL expected_hash)
        message(FATAL_ERROR
            "Weight sidecar SHA-256 mismatch for ${weight_label}.\n"
            "  expected: ${expected_hash}\n"
            "  actual:   ${actual_hash}"
        )
    endif()
endforeach()

set(INFEREDGE_RESNET18_MANIFEST "${INFEREDGE_RESNET18_FIXTURE_DIR}/manifest.txt")
set(INFEREDGE_RESNET18_MANIFEST_SHA256
    "6513989ced2f4105cb12f6abd52fdd009474b094c0b475651c352dde3809888c")
set(INFEREDGE_RESNET18_B01_INPUT "${INFEREDGE_RESNET18_FIXTURE_DIR}/b01.input.f32")
set(INFEREDGE_RESNET18_B01_INPUT_SHA256
    "abfa33dcaa24412ea4d138f43d41b002b1bd98f3c8ddc90a7accc1561db59ebc")
set(INFEREDGE_RESNET18_B01_EXPECTED "${INFEREDGE_RESNET18_FIXTURE_DIR}/b01.expected.f32")
set(INFEREDGE_RESNET18_B01_EXPECTED_SHA256
    "730645e2755e4b2731ec30f444e1168c69e384a9fc92762c774b4b1fcbea655f")
set(INFEREDGE_RESNET18_B02_INPUT "${INFEREDGE_RESNET18_FIXTURE_DIR}/b02.input.f32")
set(INFEREDGE_RESNET18_B02_INPUT_SHA256
    "bbd3fa0c71fc3712617c3fb0ec4c71deebe5b567782132cc2e62e6e177224e23")
set(INFEREDGE_RESNET18_B02_EXPECTED "${INFEREDGE_RESNET18_FIXTURE_DIR}/b02.expected.f32")
set(INFEREDGE_RESNET18_B02_EXPECTED_SHA256
    "a9ccd8c7e84370eadd9c08ceb672bfac03379e8a4ad4fbce120353d30c0fd1b8")
set(INFEREDGE_RESNET18_B04_INPUT "${INFEREDGE_RESNET18_FIXTURE_DIR}/b04.input.f32")
set(INFEREDGE_RESNET18_B04_INPUT_SHA256
    "51d5f71850ec386adfad22a7cd55b8c5143743b0cff73971ced62689075f0976")
set(INFEREDGE_RESNET18_B04_EXPECTED "${INFEREDGE_RESNET18_FIXTURE_DIR}/b04.expected.f32")
set(INFEREDGE_RESNET18_B04_EXPECTED_SHA256
    "2fb614c4301e086f2e61e7184a8751d439db38cf6e92617cec5c763fed137252")
set(INFEREDGE_RESNET18_B08_INPUT "${INFEREDGE_RESNET18_FIXTURE_DIR}/b08.input.f32")
set(INFEREDGE_RESNET18_B08_INPUT_SHA256
    "3b4bcb255b71af58d69875180003eba1f8528da17ff3fbab3a2c4becae33f649")
set(INFEREDGE_RESNET18_B08_EXPECTED "${INFEREDGE_RESNET18_FIXTURE_DIR}/b08.expected.f32")
set(INFEREDGE_RESNET18_B08_EXPECTED_SHA256
    "d48f261eee73c0b27e57fc51523434fbec57672b34d7a3483f8e8a2e41eecabc")
set(INFEREDGE_RESNET18_FIXTURE_PATHS
    "${INFEREDGE_RESNET18_MANIFEST}"
    "${INFEREDGE_RESNET18_B01_INPUT}" "${INFEREDGE_RESNET18_B01_EXPECTED}"
    "${INFEREDGE_RESNET18_B02_INPUT}" "${INFEREDGE_RESNET18_B02_EXPECTED}"
    "${INFEREDGE_RESNET18_B04_INPUT}" "${INFEREDGE_RESNET18_B04_EXPECTED}"
    "${INFEREDGE_RESNET18_B08_INPUT}" "${INFEREDGE_RESNET18_B08_EXPECTED}")
set(INFEREDGE_RESNET18_FIXTURE_HASHES
    "${INFEREDGE_RESNET18_MANIFEST_SHA256}"
    "${INFEREDGE_RESNET18_B01_INPUT_SHA256}" "${INFEREDGE_RESNET18_B01_EXPECTED_SHA256}"
    "${INFEREDGE_RESNET18_B02_INPUT_SHA256}" "${INFEREDGE_RESNET18_B02_EXPECTED_SHA256}"
    "${INFEREDGE_RESNET18_B04_INPUT_SHA256}" "${INFEREDGE_RESNET18_B04_EXPECTED_SHA256}"
    "${INFEREDGE_RESNET18_B08_INPUT_SHA256}" "${INFEREDGE_RESNET18_B08_EXPECTED_SHA256}")
set(INFEREDGE_RESNET18_FIXTURE_LABELS
    "ResNet-18 manifest"
    "ResNet-18 b01 input" "ResNet-18 b01 expected"
    "ResNet-18 b02 input" "ResNet-18 b02 expected"
    "ResNet-18 b04 input" "ResNet-18 b04 expected"
    "ResNet-18 b08 input" "ResNet-18 b08 expected")
foreach(fixture_path expected_hash fixture_label IN ZIP_LISTS
        INFEREDGE_RESNET18_FIXTURE_PATHS INFEREDGE_RESNET18_FIXTURE_HASHES
        INFEREDGE_RESNET18_FIXTURE_LABELS)
    if(NOT EXISTS "${fixture_path}")
        message(FATAL_ERROR "${fixture_label} is missing: ${fixture_path}")
    endif()
    file(SHA256 "${fixture_path}" actual_hash)
    if(NOT actual_hash STREQUAL expected_hash)
        message(FATAL_ERROR
            "${fixture_label} SHA-256 mismatch.\n"
            "  expected: ${expected_hash}\n"
            "  actual:   ${actual_hash}"
        )
    endif()
endforeach()

add_library(
    inferedge_onnx
    src/inference_engine.cpp
    src/tensor_batch.cpp
    src/neural_reference.cpp
    src/resnet18_fixtures.cpp
)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    message(FATAL_ERROR "neural_reference.cpp requires -ffp-contract=off")
endif()
set_source_files_properties(
    src/neural_reference.cpp
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off"
)
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

add_library(inferedge_scheduler src/inference_scheduler.cpp)
target_include_directories(inferedge_scheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(inferedge_scheduler PUBLIC cxx_std_20)
target_link_libraries(
    inferedge_scheduler
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

add_library(inferedge_scheduler_benchmark src/scheduler_benchmark.cpp)
target_include_directories(inferedge_scheduler_benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(inferedge_scheduler_benchmark PUBLIC cxx_std_20)
target_link_libraries(
    inferedge_scheduler_benchmark
    PUBLIC inferedge_scheduler inferedge_core
    PRIVATE inferedge_warnings
)

add_library(
    inferedge_scheduler_benchmark_cli
    src/scheduler_benchmark_cli.cpp
    src/scheduler_benchmark_command.cpp
    src/scheduler_benchmark_json.cpp
)
target_include_directories(
    inferedge_scheduler_benchmark_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(inferedge_scheduler_benchmark_cli PUBLIC cxx_std_20)
target_link_libraries(
    inferedge_scheduler_benchmark_cli
    PUBLIC inferedge_scheduler_benchmark
    PRIVATE inferedge_warnings
)

add_executable(inferedge-bench-scheduler src/scheduler_benchmark_main.cpp)
target_link_libraries(inferedge-bench-scheduler PRIVATE inferedge_scheduler_benchmark_cli)
set_target_properties(inferedge-bench-scheduler PROPERTIES
    BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
)

add_executable(
    inferedge_onnx_smoke
    tests/onnx_runtime_smoke_test.cpp
    tests/onnx_add_inference_test.cpp
    tests/inference_engine_test.cpp
    tests/tensor_batch_test.cpp
    tests/mnist_inference_test.cpp
    tests/neural_reference_test.cpp
    tests/dynamic_workload_inference_test.cpp
    tests/resnet18_inference_test.cpp
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
        inferedge_scheduler_benchmark
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
        "INFEREDGE_BATCH_ADD_MODEL_PATH=\"${INFEREDGE_BATCH_ADD_MODEL}\""
        "INFEREDGE_MLP_SMALL_MODEL_PATH=\"${INFEREDGE_MLP_SMALL_MODEL}\""
        "INFEREDGE_MLP_MEDIUM_MODEL_PATH=\"${INFEREDGE_MLP_MEDIUM_MODEL}\""
        "INFEREDGE_GEMM_MODEL_PATH=\"${INFEREDGE_GEMM_MODEL}\""
        "INFEREDGE_MLP_SMALL_WEIGHTS_PATH=\"${INFEREDGE_MLP_SMALL_WEIGHTS}\""
        "INFEREDGE_MLP_MEDIUM_WEIGHTS_PATH=\"${INFEREDGE_MLP_MEDIUM_WEIGHTS}\""
        "INFEREDGE_GEMM_WEIGHTS_PATH=\"${INFEREDGE_GEMM_WEIGHTS}\""
        "INFEREDGE_MLP_SMALL_FIXTURE_DIR=\"${INFEREDGE_MLP_SMALL_FIXTURE_DIR}\""
        "INFEREDGE_MLP_MEDIUM_FIXTURE_DIR=\"${INFEREDGE_MLP_MEDIUM_FIXTURE_DIR}\""
        "INFEREDGE_GEMM_FIXTURE_DIR=\"${INFEREDGE_GEMM_FIXTURE_DIR}\""
        "INFEREDGE_RESNET18_MODEL_PATH=\"${INFEREDGE_RESNET18_MODEL}\""
        "INFEREDGE_RESNET18_FIXTURE_DIR=\"${INFEREDGE_RESNET18_FIXTURE_DIR}\""
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
add_test(NAME inferedge_bench_scheduler_help COMMAND inferedge-bench-scheduler --help)
add_test(NAME inferedge_bench_scheduler_version COMMAND inferedge-bench-scheduler --version)
set_tests_properties(
    inferedge_bench_scheduler_help
    PROPERTIES PASS_REGULAR_EXPRESSION "Usage: inferedge-bench-scheduler"
)
set_tests_properties(
    inferedge_bench_scheduler_version
    PROPERTIES PASS_REGULAR_EXPRESSION "inferedge-bench-scheduler 0\\.1\\.0"
)

add_executable(inferedge_scheduler_benchmark_tests tests/scheduler_benchmark_test.cpp)
target_link_libraries(
    inferedge_scheduler_benchmark_tests
    PRIVATE
        inferedge_warnings
        inferedge_scheduler_benchmark_cli
        GTest::gtest_main
)
target_compile_definitions(
    inferedge_scheduler_benchmark_tests
    PRIVATE
        "INFEREDGE_BENCH_SCHEDULER_EXECUTABLE=\"$<TARGET_FILE:inferedge-bench-scheduler>\""
        "INFEREDGE_BATCH_ADD_MODEL_PATH=\"${INFEREDGE_BATCH_ADD_MODEL}\""
        "INFEREDGE_INVALID_MODEL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/data/not_a_model.bin\""
        "INFEREDGE_GEMM_MODEL_PATH=\"${INFEREDGE_GEMM_MODEL}\""
        "INFEREDGE_RESNET18_MODEL_PATH=\"${INFEREDGE_RESNET18_MODEL}\""
        "INFEREDGE_RESNET18_FIXTURE_DIR=\"${INFEREDGE_RESNET18_FIXTURE_DIR}\""
)
set_target_properties(
    inferedge_scheduler_benchmark_tests
    PROPERTIES
        BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
        INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
)
add_dependencies(inferedge_scheduler_benchmark_tests inferedge-bench-scheduler)
gtest_discover_tests(
    inferedge_scheduler_benchmark_tests
    PROPERTIES
        ENVIRONMENT "ORT_DISABLE_TELEMETRY=1"
        TIMEOUT 60
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

add_executable(
    inferedge_scheduler_tests
    tests/inference_scheduler_test.cpp
    tests/scheduler_neural_batch_test.cpp
)
target_link_libraries(
    inferedge_scheduler_tests
    PRIVATE
        inferedge_warnings
        inferedge_scheduler
        GTest::gtest_main
)
target_compile_definitions(
    inferedge_scheduler_tests
    PRIVATE
        "INFEREDGE_ADD_MODEL_PATH=\"${INFEREDGE_ADD_MODEL}\""
        "INFEREDGE_INVALID_MODEL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/data/not_a_model.bin\""
        "INFEREDGE_MNIST_MODEL_PATH=\"${INFEREDGE_MNIST_MODEL}\""
        "INFEREDGE_MNIST_FIXTURE_DIR=\"${INFEREDGE_MNIST_FIXTURE_DIR}\""
        "INFEREDGE_BATCH_ADD_MODEL_PATH=\"${INFEREDGE_BATCH_ADD_MODEL}\""
        "INFEREDGE_MLP_SMALL_MODEL_PATH=\"${INFEREDGE_MLP_SMALL_MODEL}\""
        "INFEREDGE_MLP_MEDIUM_MODEL_PATH=\"${INFEREDGE_MLP_MEDIUM_MODEL}\""
        "INFEREDGE_GEMM_MODEL_PATH=\"${INFEREDGE_GEMM_MODEL}\""
        "INFEREDGE_MLP_SMALL_WEIGHTS_PATH=\"${INFEREDGE_MLP_SMALL_WEIGHTS}\""
        "INFEREDGE_MLP_MEDIUM_WEIGHTS_PATH=\"${INFEREDGE_MLP_MEDIUM_WEIGHTS}\""
        "INFEREDGE_GEMM_WEIGHTS_PATH=\"${INFEREDGE_GEMM_WEIGHTS}\""
        "INFEREDGE_RESNET18_MODEL_PATH=\"${INFEREDGE_RESNET18_MODEL}\""
        "INFEREDGE_RESNET18_FIXTURE_DIR=\"${INFEREDGE_RESNET18_FIXTURE_DIR}\""
)
set_target_properties(
    inferedge_scheduler_tests
    PROPERTIES
        BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
        INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
)
gtest_discover_tests(
    inferedge_scheduler_tests
    PROPERTIES
        ENVIRONMENT "ORT_DISABLE_TELEMETRY=1"
        TIMEOUT 60
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

    add_executable(inferedge_scheduler_stress tests/inference_scheduler_stress_test.cpp)
    target_link_libraries(
        inferedge_scheduler_stress
        PRIVATE
            inferedge_warnings
            inferedge_scheduler
            GTest::gtest_main
    )
    target_compile_definitions(
        inferedge_scheduler_stress
        PRIVATE
            "INFEREDGE_ADD_MODEL_PATH=\"${INFEREDGE_ADD_MODEL}\""
            "INFEREDGE_BATCH_ADD_MODEL_PATH=\"${INFEREDGE_BATCH_ADD_MODEL}\""
    )
    set_target_properties(
        inferedge_scheduler_stress
        PROPERTIES
            BUILD_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
            INSTALL_RPATH "${INFEREDGE_ONNXRUNTIME_LIB_DIR}"
    )
    gtest_discover_tests(
        inferedge_scheduler_stress
        PROPERTIES
            ENVIRONMENT "ORT_DISABLE_TELEMETRY=1"
            TIMEOUT 300
    )
endif()

# Optional llama.cpp generative CLI. Default OFF. Configure does not download llama.cpp.
option(INFEREDGE_ENABLE_GENERATIVE
    "Build inferedge-generate against a verified local llama.cpp prefix" OFF)
if(INFEREDGE_ENABLE_GENERATIVE)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/llama_cpp.cmake")
    add_library(
        inferedge_generative
        src/generative_session.cpp
        src/model_package_manifest.cpp
    )
    target_include_directories(inferedge_generative PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(inferedge_generative PUBLIC cxx_std_20)
    target_link_libraries(
        inferedge_generative
        PUBLIC inferedge_warnings
        PRIVATE inferedge_llama_cpp
    )
    add_library(inferedge_generate_cli src/generate_cli.cpp)
    target_include_directories(inferedge_generate_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(inferedge_generate_cli PUBLIC cxx_std_20)
    target_compile_definitions(
        inferedge_generate_cli
        PUBLIC INFEREDGE_VERSION="${PROJECT_VERSION}"
    )
    target_link_libraries(inferedge_generate_cli PUBLIC inferedge_warnings)
    add_executable(inferedge-generate src/generate_main.cpp)
    target_link_libraries(
        inferedge-generate
        PRIVATE inferedge_generate_cli inferedge_generative
    )
    set_target_properties(inferedge-generate PROPERTIES
        BUILD_RPATH "${INFEREDGE_LLAMA_CPP_LIB_DIR}"
    )
    add_executable(
        inferedge_generative_tests
        tests/generative_session_test.cpp
    )
    target_link_libraries(
        inferedge_generative_tests
        PRIVATE
            inferedge_warnings
            inferedge_generative
            inferedge_generate_cli
            GTest::gtest_main
    )
    target_compile_definitions(
        inferedge_generative_tests
        PRIVATE
            "INFEREDGE_GENERATE_EXECUTABLE=\"$<TARGET_FILE:inferedge-generate>\""
    )
    set_target_properties(inferedge_generative_tests PROPERTIES
        BUILD_RPATH "${INFEREDGE_LLAMA_CPP_LIB_DIR}"
    )
    add_dependencies(inferedge_generative_tests inferedge-generate)
    gtest_discover_tests(inferedge_generative_tests)
    add_test(NAME inferedge_generate_help COMMAND inferedge-generate --help)
    add_test(NAME inferedge_generate_version COMMAND inferedge-generate --version)
    set_tests_properties(
        inferedge_generate_help
        PROPERTIES PASS_REGULAR_EXPRESSION "Usage: inferedge-generate"
    )
    set_tests_properties(
        inferedge_generate_version
        PROPERTIES PASS_REGULAR_EXPRESSION "inferedge-generate 0\\.1\\.0"
    )
    add_executable(inferedge-overhead-ie src/generative_overhead_ie_worker.cpp)
    target_link_libraries(
        inferedge-overhead-ie
        PRIVATE inferedge_warnings inferedge_generative inferedge_llama_cpp
    )
    set_target_properties(inferedge-overhead-ie PROPERTIES
        BUILD_RPATH "${INFEREDGE_LLAMA_CPP_LIB_DIR}"
    )
    add_executable(
        inferedge-overhead-ref
        src/generative_overhead_ref_worker.cpp
        src/model_package_manifest.cpp
    )
    target_include_directories(inferedge-overhead-ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(inferedge-overhead-ref PRIVATE cxx_std_20)
    target_link_libraries(
        inferedge-overhead-ref
        PRIVATE inferedge_warnings inferedge_llama_cpp
    )
    set_target_properties(inferedge-overhead-ref PROPERTIES
        BUILD_RPATH "${INFEREDGE_LLAMA_CPP_LIB_DIR}"
    )
    add_test(NAME inferedge_overhead_ie_help COMMAND inferedge-overhead-ie --help)
    add_test(NAME inferedge_overhead_ref_help COMMAND inferedge-overhead-ref --help)
    add_test(NAME inferedge_overhead_ie_version COMMAND inferedge-overhead-ie --version)
    add_test(NAME inferedge_overhead_ref_version COMMAND inferedge-overhead-ref --version)
    set_tests_properties(
        inferedge_overhead_ie_version
        PROPERTIES PASS_REGULAR_EXPRESSION "generative-overhead-r7-inventory-v1"
    )
    set_tests_properties(
        inferedge_overhead_ref_version
        PROPERTIES PASS_REGULAR_EXPRESSION "generative-overhead-r7-inventory-v1"
    )
endif()
