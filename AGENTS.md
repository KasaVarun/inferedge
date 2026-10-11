# Agent guide

InferEdge is a C++20 inference runtime. ONNX Runtime 1.31.0 CPU is a pinned configure-time dependency. The executable supports `--help` and `--version` only.

## Current scope

Build the `inferedge` executable, its unit tests, the ONNX Runtime smoke test, the committed `models/add_f32.onnx` Add fixture executed by that test, and Linux CI.

## Do not add yet

- An inference engine, or model loading outside the committed Add fixture test
- CUDA, TensorRT, or CoreML execution provider configuration
- Scheduling, threading, or dynamic batching
- HTTP APIs, databases, frontends, or Python bindings
- Benchmarks that do not measure a real implementation
- Dependencies beyond the C++ standard library, CMake, Ninja, GoogleTest fetched by CMake, the pinned ONNX Runtime 1.31.0 CPU archives, and clang-format

`onnx==1.22.0` regenerates `models/add_f32.onnx`. It is not a CMake dependency. Do not install it during configuration.

## Build and test

Configure with Ninja. Keep project warnings as errors. Run `ctest` before considering a change done. Format C++ with the repository `.clang-format` using clang-format 23.1.3.

## Code shape

Keep `main` thin. Put behavior that should be tested in `include/inferedge` and `src`, not in the executable entry point.
