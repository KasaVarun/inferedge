# Agent guide

InferEdge is a C++20 inference runtime. ONNX Runtime 1.31.0 CPU is a pinned configure-time dependency. The `inferedge` executable supports `--help`, `--version`, and one fixed rank-1 float32 model through `InferenceEngine`. `inferedge-bench` measures one loaded engine through `run_benchmark`. Its default report is text. `--format json` writes schema version 1 from that report with `std::to_chars`. Do not add a JSON library.

## Current scope

Build the `inferedge` executable, `inferedge-bench`, their unit tests, the ONNX Runtime tests, the `inferedge_onnx` library, the latency statistics calculator, the synchronous benchmark runner, the committed Add, int64 Identity, rank-2 float32, and MNIST MLP fixtures, and Linux CI.

## Do not add yet

- Published speed figures. `inferedge-bench` calls `run_benchmark` for one loaded `InferenceEngine` on the calling thread and can print that report as text or JSON. `summarize_latencies` still only reduces injected durations. JSON schema version 1 does not include raw samples, hardware, build type, or a model hash. `benchmarks/collect_baseline.py` can launch that executable and validate the JSON. `benchmarks/analyze_baseline.py` reduces one completed run set to `aggregate.json`. Neither script infers, and `benchmarks/baselines/` stays untracked until a measured run set is reviewed
- Scheduling, request queues, threading, or dynamic batching
- CUDA, TensorRT, or CoreML execution provider configuration
- HTTP APIs, databases, frontends, or Python bindings
- Dependencies beyond the C++ standard library, CMake, Ninja, GoogleTest fetched by CMake, the pinned ONNX Runtime 1.31.0 CPU archives, and clang-format

`onnx==1.22.0` regenerates `models/add_f32.onnx`, `models/int64_echo.onnx`, and `models/rank2_f32.onnx`. It is not a CMake dependency. Do not install it during configuration.

## Build and test

Configure with Ninja. Keep project warnings as errors. Run `ctest` before considering a change done. Format C++ with the repository `.clang-format` using clang-format 23.1.3.

## Code shape

Keep `main` thin. Put behavior that should be tested in `include/inferedge` and `src`, not in the executable entry point.
