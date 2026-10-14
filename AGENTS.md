# Agent guide

InferEdge is a C++20 inference runtime. ONNX Runtime 1.31.0 CPU is a pinned configure-time dependency. The `inferedge` executable supports `--help`, `--version`, and one fixed rank-1 float32 model through `InferenceEngine`. `inferedge-bench` measures one loaded engine through `run_benchmark`. Its default report is text. `--format json` writes schema version 1 from that report with `std::to_chars`. `inferedge-bench-concurrent` calls `run_concurrent_benchmark`. `--concurrency` is required. `--format json` writes schema version 2 with `std::to_chars`. Schema version 1 is unchanged. Do not add a JSON library.

## Current scope

Build the `inferedge` executable, `inferedge-bench`, `inferedge-bench-concurrent`, their unit tests, the ONNX Runtime tests, the `inferedge_onnx` library, the `inferedge_concurrent` library, the `inferedge_concurrent_benchmark` library, the `inferedge_concurrent_benchmark_cli` library, the latency statistics calculator, the synchronous benchmark runner, the committed Add, int64 Identity, rank-2 float32, and MNIST MLP fixtures, and Linux CI. `inferedge_concurrent` gives each worker its own `InferenceEngine` and checks outputs. `inferedge_concurrent_benchmark` measures those workers inside one synchronized window. `inferedge-bench-concurrent` prepares Add or MNIST, copies one assignment per worker, and calls that runner. `--format json` writes schema version 2. One `InferenceEngine` is still single-threaded. There is no collected concurrent performance baseline. `BoundedFifoQueue` is the waiting-work queue. `inferedge_scheduler` is the Phase 5.2.2 scheduler: one model path, one `InferenceEngine` per worker, drain shutdown, and cancel-pending shutdown. Cancel-pending completes requests still waiting in the queue and leaves an in-flight `run()` with its worker. There is no dynamic batching.

## Scheduler benchmark

`inferedge-bench-scheduler` measures one `InferenceScheduler`. Completion timing is off by default, and `--format json` then writes schema version 3. `benchmarks/collect_scheduler.py` defaults to analysis protocol 2 and publication policy `strict_generator_delivery`. `SchedulerConfig::enable_completion_timing` defaults to false. The schema-3 latency field stays producer-observed: `steady_clock` immediately before `submit()` until the future is observed ready. Historical schema-3 reports remain valid under that definition. Schema versions 1 and 2 are unchanged.

`--completion-timing` turns on `enable_completion_timing` and writes schema version 4. Precision collection is explicit `--analysis-protocol 3` with publication policy `pre_publication_timing`. Each published `SchedulerCompletion` then carries one immutable `pre_publication_time`, a `steady_clock` time point taken immediately before `promise::set_value`. There is no post-publication timestamp and no exact future ready-transition time.

Pre-publication latency runs from immediately before `submit()` to that stamp. It includes scheduler processing and is not pure inference execution time. Pre-publication-to-observation delay runs from the stamp until the collector observes the future ready. It includes `set_value`, readiness propagation, and collector wait, scan, and scheduling delay. It is not pure polling overhead.

Schema 4 adds signed generator-lag percentiles. Strict generator delivery is separate from arrival-timing fidelity. A confirmatory precision-latency comparison requires protocol 3, schema 4, completion timing, predeclared `arrival_fidelity_p95_max_nanoseconds` and `arrival_fidelity_max_lag_nanoseconds`, and a predeclared latency metric of `producer_observed` or `pre_publication`. Missing bounds leave arrival timing `not_certified` and do not establish accurate arrival scheduling. `phase_boundary_fidelity` is `unavailable`.

Protocol 2 and protocol 3 keep paired experimental blocks, within-profile cyclic rotation, a predeclared contrast, and `paired_block_bootstrap_mean` with seed `20261013` and 10000 replicates. An interval needs at least 10 complete counted blocks. The preferred publication count is 20. `winner_declared` and `superiority_claimed` stay false.

Disabled timing still changes the `SchedulerCompletion` layout. Instrumentation overhead is uncalibrated and is not subtracted. No performance gain has been established. Prebuilt ONNX Runtime is not instrumented by the project's sanitizer builds.

## Do not add yet

- Published speed figures. `inferedge-bench` calls `run_benchmark` for one loaded `InferenceEngine` on the calling thread and can print that report as text or JSON. `summarize_latencies` still only reduces injected durations. JSON schema version 1 does not include raw samples, hardware, build type, or a model hash. `benchmarks/collect_baseline.py` can launch that executable and validate the JSON. `benchmarks/analyze_baseline.py` reduces one completed run set to `aggregate.json`. Neither script infers, and `benchmarks/baselines/` stays untracked until a measured run set is reviewed
- Dynamic batching or a shared session pool. Do not call `InferenceEngine::run` concurrently on one engine. Do not add a collected concurrent baseline. `inferedge-bench-concurrent` and JSON schema version 2 exist. They are not a production server, and no concurrency speedup has been measured
- CUDA, TensorRT, or CoreML execution provider configuration
- HTTP APIs, databases, frontends, or Python bindings
- Dependencies beyond the C++ standard library, CMake, Ninja, GoogleTest fetched by CMake, the pinned ONNX Runtime 1.31.0 CPU archives, and clang-format

`onnx==1.22.0` regenerates `models/add_f32.onnx`, `models/int64_echo.onnx`, and `models/rank2_f32.onnx`. It is not a CMake dependency. Do not install it during configuration.

## Build and test

Configure with Ninja. Keep project warnings as errors. Run `ctest` before considering a change done. Format C++ with the repository `.clang-format` using clang-format 23.1.3. `INFEREDGE_ENABLE_LONG_STRESS` defaults to OFF. When it is ON, `inferedge_concurrent_stress` and `inferedge_scheduler_stress` register with a 300-second timeout. Default Linux CI leaves the option OFF. Those tests check completion and accounting. They do not publish speed figures.

## Code shape

Keep `main` thin. Put behavior that should be tested in `include/inferedge` and `src`, not in the executable entry point.
