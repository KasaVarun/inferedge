# Agent guide

InferEdge is a C++20 inference runtime. This repository is in Phase 0.

## Current scope

Build the `inferedge` executable, its unit tests, and Linux CI. The executable supports `--help` and `--version` only.

## Do not add yet

- ONNX Runtime or any other inference engine
- Scheduling, threading, or dynamic batching
- HTTP APIs, databases, frontends, or Python bindings
- Benchmarks that do not measure a real implementation
- Dependencies beyond the C++ standard library, CMake, Ninja, GoogleTest fetched by CMake, and clang-format

## Build and test

Configure with Ninja. Keep project warnings as errors. Run `ctest` before considering a change done. Format C++ with the repository `.clang-format` using clang-format 23.1.3.

## Code shape

Keep `main` thin. Put behavior that should be tested in `include/inferedge` and `src`, not in the executable entry point.
