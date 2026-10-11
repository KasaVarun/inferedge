# Agent guide

InferEdge is a C++20 inference runtime. ONNX Runtime 1.31.0 CPU is a pinned configure-time dependency. The executable supports `--help`, `--version`, and one fixed rank-1 float32 model through `InferenceEngine`.

## Current scope

Build the `inferedge` executable, its unit tests, the ONNX Runtime tests, the `inferedge_onnx` library, the committed Add, int64 Identity, and rank-2 float32 fixtures, and Linux CI.

## Do not add yet

- Scheduling, request queues, threading, or dynamic batching
- CUDA, TensorRT, or CoreML execution provider configuration
- HTTP APIs, databases, frontends, or Python bindings
- Benchmarks that do not measure a real implementation
- Dependencies beyond the C++ standard library, CMake, Ninja, GoogleTest fetched by CMake, the pinned ONNX Runtime 1.31.0 CPU archives, and clang-format

`onnx==1.22.0` regenerates `models/add_f32.onnx`, `models/int64_echo.onnx`, and `models/rank2_f32.onnx`. It is not a CMake dependency. Do not install it during configuration.

## Build and test

Configure with Ninja. Keep project warnings as errors. Run `ctest` before considering a change done. Format C++ with the repository `.clang-format` using clang-format 23.1.3.

## Code shape

Keep `main` thin. Put behavior that should be tested in `include/inferedge` and `src`, not in the executable entry point.
