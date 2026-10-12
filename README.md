# InferEdge

InferEdge is a high-performance C++ inference runtime and scheduler. The repository has a C++20 build, a command-line executable, a benchmark executable, unit tests, formatting, and Linux CI. Configure downloads the pinned ONNX Runtime 1.31.0 CPU library and links it into `inferedge`, `inferedge-bench`, and the ONNX test executables.

`inferedge_onnx` loads one ONNX model into a reusable CPU session. The `inferedge` executable parses a command and calls that session. `inferedge_core` does not link ONNX Runtime.

```text
./inferedge --help
./inferedge --version
```

## Project status

The executable runs one fixed rank-1 float32 model and still answers `--help` and `--version`.

| Present | Not present yet |
| --- | --- |
| CMake + Ninja build | Scheduler, request queue, or dynamic batching |
| `inferedge --model` and `--input` | HTTP API, database, frontend, Python bindings |
| GoogleTest unit tests | CUDA, TensorRT, or CoreML provider setup |
| Synchronous float32 `InferenceEngine` | Concurrent `run()` on one engine |
| clang-format 23.1.3 and Linux GitHub Actions | A production inference API |
| Injected-duration statistics, `run_benchmark`, and `inferedge-bench` text or JSON schema version 1 | A collected concurrent performance baseline |
| Concurrent benchmark library and `inferedge-bench-concurrent` JSON schema version 2, one session per worker | A measured concurrency speedup |

Version: **0.1.0**

## Architecture roadmap

Each phase should stay reviewable on its own. Later phases are intentionally not started.

1. **Phase 0 — Foundation.** Compiler settings, layout, CLI, tests, and CI.
2. **Phase 1 — Core types.** Ownership, status/error reporting, and the data types later phases will share.
3. **Phase 2 — Model loading.** Phase 2.0 links the pinned ONNX Runtime CPU library. Phase 2.1 runs one committed float32 Add graph. Phase 2.2 moves that execution into `InferenceEngine`. Phase 2.3 calls that engine from `inferedge`. Scheduling comes later.
4. **Phase 3 — Execution.** Run one inference and return outputs.
5. **Phase 4 — Scheduling.** Threads and dynamic batching, measured against a real executor.
6. **Phase 5 — Service boundary.** A process API only after the runtime has a stable in-process interface.

## Prerequisites

On this Mac:

| Tool | Used for |
| --- | --- |
| Apple Clang with C++20 | Compiling |
| CMake 3.24 or newer | Generating the build |
| Ninja | Running the build |
| clang-format 23.1.3 | Formatting C++ sources |
| Git | Version control |

Install the formatter if it is missing:

```bash
brew install clang-format
```

Linux CI installs `g++`, `cmake`, `ninja-build`, and the same clang-format version. The workflow runs on `ubuntu-24.04`.

The first configure step downloads GoogleTest 1.18.0 and the ONNX Runtime 1.31.0 CPU archive for the host platform. Both downloads require network access. Neither archive is committed.

## ONNX Runtime

Version **1.31.0**, CPU packages only:

| Host | Archive | SHA-256 |
| --- | --- | --- |
| macOS arm64 | `onnxruntime-osx-arm64-1.31.0.tgz` | `1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc` |
| Linux x86-64 | `onnxruntime-linux-x64-1.31.0.tgz` | `cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57` |

CMake downloads the one archive that matches the host, checks its SHA-256, and extracts it under the build directory. A checksum mismatch or an unsupported operating system and processor stops configuration. The version does not float to a newer release. The CMake files shipped inside the archive point at directories the archive does not contain, so InferEdge defines the imported target `onnxruntime::onnxruntime` from `include/` and `lib/` itself.

Set `INFEREDGE_ONNXRUNTIME_ROOT` to an already extracted package when configure should not download. That directory must contain `include/onnxruntime_cxx_api.h`, the shared library, its runtime soname, and a `VERSION_NUMBER` file that reads `1.31.0`.

`inferedge` and the smoke-test binary record the extracted `lib` directory as their build RPATH. macOS loads `@rpath/libonnxruntime.1.dylib`. Linux loads `libonnxruntime.so.1`.

Official macOS and Linux builds include telemetry. Tests and CI set `ORT_DISABLE_TELEMETRY=1` before the library is initialized. Export the same variable before running `inferedge` or `inferedge_onnx_smoke` outside CTest. This does not rebuild ONNX Runtime with telemetry compiled out.

ONNX Runtime is MIT licensed, copyright Microsoft Corporation. Each archive includes `LICENSE` and `ThirdPartyNotices.txt`. Those notices stay with the extracted package in the build tree.

The smoke test constructs `Ort::Env` and checks that `Ort::GetVersionString()` is `1.31.0`. The same executable loads `models/add_f32.onnx` through `InferenceEngine` and runs it. The inspected macOS library requires macOS 14.0 or newer.

## Add fixture

`models/add_f32.onnx` is a hand-written graph, not a trained model. It has no weights.

| Item | Value |
| --- | --- |
| Graph | `add_f32` |
| Inputs | `A` and `B`, `float32[2]` |
| Output | `C`, `float32[2]` |
| Operation | `C = Add(A, B)` |
| ONNX IR | 10 |
| Opset | 13 |
| Producer | `inferedge` `0.1.0` |
| Generator | `onnx==1.22.0` |
| Size | 203 bytes |
| SHA-256 | `37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447` |

CMake checks that hash with `file(SHA256)` and stops if the file is missing or different. Configuration does not run Python.

The C++ test creates `Ort::Env`, then `Ort::SessionOptions`, then `Ort::Session`. Options select the CPU execution provider, one intra-op thread, and `ORT_DISABLE_ALL`. The test checks the input and output names, `float32` types, and shapes `[2]`. Input tensors borrow `std::vector<float>` storage. `Ort::Session::Run` produces the output, and the test reads that tensor before destroying it.

Two successful cases use exact `float32` values:

| A | B | C |
| --- | --- | --- |
| `[1.0, 2.0]` | `[3.0, 4.0]` | `[4.0, 6.0]` |
| `[-1.5, 0.0]` | `[2.5, -4.0]` | `[1.0, -4.0]` |

The same test expects `Ort::Exception` for a missing file, `tests/data/not_a_model.bin`, a shape of `[3]`, and an `int64` input. Those failures do not stop the process.

To regenerate the fixture on purpose:

```bash
python3 -m venv .venv
.venv/bin/pip install -r models/requirements.txt
.venv/bin/python models/generate_add_model.py
shasum -a 256 models/add_f32.onnx
```

Copy the printed digest into `INFEREDGE_ADD_MODEL_SHA256` in `CMakeLists.txt`. The generator refuses any `onnx` version other than `1.22.0`, passes IR 10 and opset 13 explicitly, and runs `onnx.checker.check_model` before saving. Metadata has no timestamp.

`models/int64_echo.onnx` is a second untrained fixture. It is an Identity of `int64[1]` from `X` to `Y`, IR 10, opset 13, 201 bytes, SHA-256 `4bae136df39291cfccfb7a1b9b0f03186598326f441f755095459ab03805b145`. The engine rejects it because only `float32` tensors are supported. Regenerate it with `models/generate_int64_echo_model.py` and update `INFEREDGE_INT64_MODEL_SHA256`.

`models/rank2_f32.onnx` is an Identity of `float32[1, 2]` from `M` to `N`, IR 10, opset 13, 212 bytes, SHA-256 `193fc307deda81dee0e3b0f5999809a80d92ec46070c66c187b4578daefb5bc2`. The CLI rejects it because the input is not a fixed rank-1 tensor. Regenerate it with `models/generate_rank2_f32_model.py` and update `INFEREDGE_RANK2_MODEL_SHA256`.

`models/mnist_mlp.onnx` is the trained MNIST MLP. CMake checks SHA-256 `3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce` and does not regenerate it. The input is `image`, `float32[784]`. The output is `logits`, `float32[10]`. `tests/mnist_inference_test.cpp` runs that file through `InferenceEngine` and compares it with the ONNX reference fixtures in `models/fixtures/mnist/`.

## Inference engine

`inferedge_onnx` is a static library. The `inferedge` executable links it. `inferedge_core` does not. Public declarations live in `include/inferedge/inference_engine.hpp` and do not include ONNX Runtime headers. `src/inference_engine.cpp` owns the runtime types behind `std::unique_ptr`.

`load_inference_engine` checks that the path is a file, then creates one `Ort::Env` and one `Ort::Session`. Session options stay alive through construction, select the CPU provider, set one intra-op thread, and disable graph optimizations. The logger severity is error, so a routine runtime warning is not written to stderr. The CLI uses stderr for its own `error:` lines. The environment is destroyed after the session. A missing file returns `kModelNotFound`. A file ONNX Runtime cannot load returns `kModelInvalid`. A model input or output that is not a `float32` tensor returns `kUnsupportedType`. On success the returned engine is non-null and `run` reuses that session.

`run` accepts any number of named `float32` tensors. Names are matched to the model, so input order does not matter. The first problem is returned as `InferenceError`: empty name, duplicate name, unexpected name, missing name, negative dimension, overflowing shape product, element-count mismatch, then rank and fixed-dimension mismatch. A negative model dimension is dynamic and accepts a non-negative caller size. A failed `run` leaves the session loaded.

`NamedTensorView::values` is a `std::span<const float>`. It must stay valid for the whole `run` call. ONNX Runtime 1.31 `CreateTensor` takes `float*`, not `const float*`, so `run` copies each input into mutable storage for that call and does not cast away const. The copy is destroyed after `Run` returns. Each output is copied into `NamedTensor::values` before the ONNX Runtime output value is destroyed. Returned tensors own those floats.

`InferenceEngine` is move-only. A moved-from engine has no session. `inputs()` and `outputs()` then return an empty vector, and `run` returns `kExecution`. Those accessors do not dereference an empty implementation.

The engine is not thread-safe. One thread should call a given engine. Per-call input spans and returned vectors are distinct objects, and they share one `Ort::Session`. Concurrent `run` calls on that same engine are not supported.

## Concurrent inference correctness

`inferedge_concurrent` (`include/inferedge/concurrent_inference.hpp`) runs a fixed set of workers. Each worker owns one `InferenceEngine` loaded before any thread starts, and one `std::jthread` calls `run()` only on that engine. Input floats live in that worker's `WorkerRequest`. `NamedTensorView` objects borrow those floats for one call. Expected tensors own their reference values. Add comparisons use zero tolerance. MNIST comparisons use `rtol = 1e-4` and `atol = 1e-5`.

If a model fails to load, the call returns `kModelLoad` with the engine's `InferenceError` and starts no threads. A failure during a call, a wrong output, or an exception inside a worker sets an atomic flag, lets the in-flight `run()` finish, and does not start another call on a worker that has seen the flag. Every created `std::jthread` is joined. The result's `outputs` vector is empty unless every worker completed every requested call. There is no partial success.

`stop_after_starting_workers` aborts thread creation after a chosen count so a partial start cannot wait for a thread that does not exist. No fixed-size barrier is used. `throw_in_worker` makes one worker throw before its first call; the thread function catches it.

This library does not make one `InferenceEngine` thread-safe. It has no request queue, scheduler, or batching. It does not measure latency or throughput, and it does not report a concurrency speedup. `inferedge` and `inferedge-bench` do not link it.

## Concurrent benchmark runner

`run_concurrent_benchmark` lives in `inferedge_concurrent_benchmark` (`include/inferedge/concurrent_benchmark.hpp`). It loads one `InferenceEngine` per worker before starting threads, using the same CPU session as the rest of the runtime: one intra-op thread and graph optimizations disabled. Each worker calls `run()` only on its own engine. Workers do not share input buffers. `inferedge-bench-concurrent` is the command-line wrapper. No concurrent performance baseline has been collected.

Warmup runs on each engine before a shared start gate. Every warmup output is checked, and those durations are discarded. Warmup may be zero. After every worker arrives, the coordinator records one `steady_clock::now()` and releases the gate. A failure before that release cancels the gate and joins every thread that was created. A fixed-size barrier is not used, so a thread that was never created cannot leave the others waiting.

Each measured sample is the clock immediately before `engine.run` and immediately after it returns. That sample includes validation, the input copy, ONNX Runtime execution, and the output copy inside `run`. It excludes the correctness check and destruction of that `InferenceResult`. After the last measured result on a worker is destroyed, that worker records its finish time. The wall interval is the latest of those finish times minus the shared start. It excludes model loading, thread construction, and warmup. It is not the sum of overlapping per-call durations.

`call_summary` is `summarize_latencies` on the pooled per-request samples. Its active-call throughput remains the sample count divided by the sum of those samples, so overlapping calls can make it larger than the wall-clock rate. `aggregate_completed_requests_per_second` is the total completed requests divided by the wall interval in seconds. That is the concurrent throughput figure. It is not server capacity, and it is not the sequential `observed_sequential_throughput_requests_per_second` field. This page does not publish either number and does not claim a speedup.

Numeric comparison matches the sequential runner: `abs(actual - expected) <= absolute_tolerance + relative_tolerance * abs(expected)`. Warmup compares every call. Measured calls always reject an inference error, a missing output, a shape or count mismatch, and a non-finite value. Value comparison uses 1-based iterations, every `correctness_check_interval` iteration, and always the last iteration. Add references use zero tolerance. MNIST references use `rtol = 1e-4` and `atol = 1e-5`. Any failure discards the report. Output tensors are not kept.

Worker count must be from 1 through 64 and must match the assignment list. Warmup and measured counts above 100,000 are rejected, as is a measured count of zero, a check interval of zero, or a worker and measured-iteration product that overflows. `inferedge` and `inferedge-bench` do not link this library. `inferedge-bench-concurrent` does. JSON schema version 1 is unchanged.

## Concurrent benchmark command

`inferedge-bench-concurrent` is a separate executable. It links `inferedge_concurrent_benchmark_cli`, which links `inferedge_concurrent_benchmark`. `inferedge` and `inferedge-bench` do not link that library. The command parses arguments, prepares one workload, loads the model once to check metadata, destroys that probe session, copies the inputs into one owned assignment per worker, and calls `run_concurrent_benchmark`. It does not run the worker loop itself. Fault-injection options on `ConcurrentBenchmarkConfig` stay unset. The default report is text. `--format json` writes schema version 2.

```bash
export ORT_DISABLE_TELEMETRY=1
./build/inferedge-bench-concurrent --help
./build/inferedge-bench-concurrent \
  --model models/add_f32.onnx \
  --workload add \
  --concurrency 2 \
  --warmup 10 \
  --iterations 100
./build/inferedge-bench-concurrent \
  --model models/mnist_mlp.onnx \
  --workload mnist \
  --concurrency 2 \
  --fixture test_00003 \
  --format json
```

`--model`, `--workload`, and `--concurrency` are required. `--concurrency` has no default and is not clamped: it must be an integer from 1 through 64. `--workload` is `add` or `mnist`. `--warmup` defaults to 10 and may be 0. `--iterations` defaults to 100. `--check-interval` defaults to 1. Measured iterations and the check interval must be from 1 through 100000. `--format` is `text` or `json` and defaults to text. `--help`, `-h`, `--version`, and `-V` are standalone and exit 0. With no arguments, the program prints help and exits 0. `--flag value` and `--flag=value` are both accepted. Duplicate flags, unknown flags, and non-numeric counts are usage errors. `std::from_chars` must consume each count. A value that does not fit in a 32-bit unsigned integer is an overflow. A separate argument that starts with `-` is an option, so a negative count has to be written as `--concurrency=-1`, and that form is rejected.

The Add workload uses `A = [1, 2]`, `B = [3, 4]`, and expected `C = [4, 6]`, compared exactly. `--fixture` and `--fixture-dir` are rejected for Add. The MNIST workload reads the committed `inputs/<id>.f32` and `onnx/<id>.f32` files. The default id is `test_00003`. The command checks tensor names, float32 rank-1 shapes, byte length, finite values, and the catalog reference class before the model runs. Comparison uses `rtol = 1e-4` and `atol = 1e-5`. The prepared inputs are copied so each worker owns its own float storage. Those copies stay alive until `run_concurrent_benchmark` returns.

Each worker has its own `InferenceEngine`. Each session uses one intra-op thread and graph optimizations disabled. Inference is CPU only. There is no request queue, no batching, and no scheduler. The text report states that aggregate throughput divides completed requests by the shared wall-clock duration, and active-call throughput divides them by the sum of per-call durations. Neither figure is production server capacity. No concurrency speedup has been measured.

| Exit code | Meaning |
| --- | --- |
| 0 | Help, version, or a completed benchmark |
| 2 | Invalid command line |
| 3 | The model file is missing, invalid, or not a float32 tensor |
| 4 | The fixture directory or file is missing, unreadable, the wrong size, non-finite, the wrong reference class, or a fixture filesystem or I/O exception |
| 5 | Inference failed during warmup or a measured call |
| 6 | Loaded metadata does not match the workload |
| 7 | An output failed the correctness check |
| 8 | A worker failed to start, a worker exception, timing, statistics, JSON serialization, or an unexpected internal error |

Successful text goes to stdout. Errors go to stderr as `error: ...`. A usage error also reprints the help text. A failure does not print a benchmark report. When the runner reports a worker and iteration, the diagnostic includes them. A fixture filesystem exception, such as a symlink loop or a permission error while stating a fixture path, is exit 4 and includes the filesystem reason. An unexpected exception from another step is exit 8 and is reported as `error: internal error: ` followed by the exception text. A JSON failure exits 8, writes the reason to stderr, and leaves stdout empty. The JSON document is assembled completely before it is written.

### JSON schema version 2

`--format json` prints one JSON object and a newline. The document is built in memory and written only after it passes the checks below. Schema version 1 is a different writer and is not used here.

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `2` |
| `tool.name` | `inferedge-bench-concurrent` |
| `tool.version` | The `inferedge` version string |
| `model` | Model path as typed |
| `workload` | `add` or `mnist` |
| `fixture` | JSON `null` for Add. For MNIST, the selected fixture id |
| `backend` | `ONNX Runtime CPU` |
| `configuration.worker_count` | Workers. It matches `--concurrency` |
| `configuration.session_ownership` | `one_inference_engine_per_worker` |
| `configuration.onnx_intra_op_threads_per_session` | `1` |
| `configuration.graph_optimizations` | `disabled` |
| `configuration.warmup_iterations_per_worker` | Unmeasured calls on each worker |
| `configuration.measured_iterations_per_worker` | Measured calls on each worker |
| `configuration.correctness_check_interval` | Numeric check interval |
| `latency_microseconds.sample_count` | Pooled measured samples. It equals the total completed requests |
| `latency_microseconds.minimum`, `mean`, `p50`, `p95`, `p99`, `maximum` | Summary values, in microseconds |
| `latency_microseconds.population_standard_deviation` | Population standard deviation, in microseconds |
| `measurement_wall_microseconds` | Shared wall interval, in microseconds |
| `total_completed_requests` | `worker_count * measured_iterations_per_worker` |
| `per_worker_completed_requests` | One count per worker, each equal to the measured iteration count |
| `aggregate_completed_requests_per_second` | Completed requests divided by the shared wall interval |
| `active_call_requests_per_second` | Completed requests divided by the sum of per-call durations |
| `correctness` | `passed`. JSON is written only after a successful run |

Counts are JSON integers. Durations and both throughput rates are JSON numbers from `std::to_chars` in general format. Raw per-request samples are not serialized. The writer rejects a worker count below 1, a per-worker list whose length or values disagree with the measured work, a sample count that disagrees with the completed requests, a non-positive wall duration, a non-positive throughput, a non-finite metric, a negative latency, percentiles out of order, a negative standard deviation including negative zero, an aggregate rate that does not match the wall interval, and a string that is not valid UTF-8. Quotes, backslashes, and ASCII controls are escaped. An inconsistent report produces no JSON.

## Latency statistics

`summarize_latencies` lives in `inferedge_core` (`include/inferedge/benchmark_statistics.hpp`). It reduces a span of `std::chrono::nanoseconds` supplied by the caller. It does not read a clock, load a model, or call `InferenceEngine::run`. The repository does not publish an inference latency or throughput measurement.

Durations stay in nanoseconds for the minimum, maximum, and total. The mean, P50, P95, and P99 are microseconds as `nanoseconds / 1000.0`, so a fraction of a microsecond remains. Samples are sorted on a copy. The nearest-rank index is `ceil(p * n / 100) - 1`, computed with integer arithmetic. P50 is that element. It is not the average of the two middle samples. The standard deviation is the population deviation, `sqrt(sum((sample - mean)^2) / n)`. One sample has deviation 0. Equal samples have deviation 0.

`active_call_throughput_requests_per_second` is the sample count divided by the sum of the sample durations in seconds. Each sample is one completed call interval. The figure excludes model loading, correctness checks, destruction of results outside those intervals, benchmark loop overhead, queueing, and network transport. It is not end-to-end service throughput.

An empty span, a negative duration, a total of zero, or a sum that overflows the nanosecond integer returns `StatisticsError` and an empty summary. A zero sample is kept when other samples make the total positive. The input span is not reordered.

## Synchronous benchmark runner

`run_benchmark` lives in `inferedge_benchmark` (`include/inferedge/benchmark_runner.hpp`). The caller loads one `InferenceEngine` and passes prepared `NamedTensorView` inputs and `ExpectedTensor` references. The runner does not reload the model. It is not linked into the `inferedge` executable. This page does not publish latency or throughput numbers.

The session under test is the existing CPU session: one intra-op thread and graph optimizations disabled. The runner is single-threaded. One call uses one loaded model.

Warmup calls `run` and checks every output name, shape, finite value, and numeric tolerance. Those durations are discarded. Warmup may be zero. Warmup and measured counts above 100,000 are rejected, as is a measured count of zero or a check interval of zero. Expected outputs are validated before warmup: a missing list, an empty or duplicate name, a negative dimension, an overflowing shape product, an element count that disagrees with the shape, a non-finite reference, or a negative or non-finite tolerance returns an empty report.

The outer `steady_clock` interval starts after warmup and stops after the last `InferenceResult` from a measured call has been destroyed. Inside that loop, each sample is the clock immediately before `engine.run` and immediately after it returns. That sample includes the validation, input copy, ONNX Runtime execution, and output copy inside `run`. It excludes model loading, fixture parsing, warmup, output checks, destruction of the returned result, `summarize_latencies`, and report formatting. The runner does not log inside the timed call.

Every measured call must succeed. Every measured output is checked for count, name, shape, and finite values. Numeric comparison uses 1-based iteration numbers. It runs when the iteration is divisible by `correctness_check_interval`, and it always runs on the last iteration. The default interval is 1. The comparison is `abs(actual - expected) <= absolute_tolerance + relative_tolerance * abs(expected)`. A failed call or a failed check discards the samples.

`call_summary` is `summarize_latencies` on the measured samples. Its active-call throughput is the sample count divided by the sum of those `run` durations. `observed_sequential_throughput_requests_per_second` is the measured iteration count divided by the outer loop duration in seconds. The outer duration includes the measured calls, validation, result destruction, and loop bookkeeping. It excludes warmup, loading, fixture parsing, the statistics call, and printing. Neither figure is server capacity. A zero outer duration is an error. With fewer than 100 samples, P99 is only the nearest-rank element and is often the maximum.

## Benchmark command

`inferedge-bench` is a separate executable. It links `inferedge_benchmark_cli`, which links `inferedge_benchmark`. The `inferedge` executable does not link that library. `inferedge-bench` parses arguments, prepares one workload, calls `load_inference_engine` once, checks the loaded metadata, and calls `run_benchmark`. It does not time `engine.run` itself. The default report is text. `--format json` writes the same completed report as one JSON document.

```bash
export ORT_DISABLE_TELEMETRY=1
./build/inferedge-bench --help
./build/inferedge-bench \
  --model models/add_f32.onnx \
  --workload add \
  --warmup 100 \
  --iterations 1000
./build/inferedge-bench \
  --model models/mnist_mlp.onnx \
  --workload mnist \
  --fixture test_00003 \
  --warmup 100 \
  --iterations 1000
./build/inferedge-bench \
  --model models/add_f32.onnx \
  --workload add \
  --format json \
  > add.json
```

`--model` and `--workload` are required. `--workload` is `add` or `mnist`. `--warmup` defaults to 10 and may be 0. `--iterations` defaults to 100. `--check-interval` defaults to 1. Measured iterations and the check interval must be from 1 through 100000. `--format` is `text` or `json`. Omitting it selects text, which is the report printed today. `--format text` selects that same report. `--format json` and `--format=json` select JSON. An unknown format, an empty format, a missing format value, and a repeated `--format` are usage errors. `--help`, `-h`, `--version`, and `-V` are standalone and exit 0. With no arguments, the program prints help and exits 0. `--flag value` and `--flag=value` are both accepted. Duplicate flags, unknown flags, and non-numeric counts are usage errors. `std::from_chars` must consume each count. A value that does not fit in a 32-bit unsigned integer is an overflow. A separate argument that starts with `-` is an option, so a negative count has to be written as `--iterations=-1`, and that form is rejected.

The Add workload uses `A = [1, 2]`, `B = [3, 4]`, and expected `C = [4, 6]`, with zero tolerance. `--fixture` and `--fixture-dir` are rejected for Add. The MNIST workload reads `inputs/<id>.f32` (784 little-endian float32 values) and `onnx/<id>.f32` (10 little-endian float32 logits) from the fixture directory. The default id is `test_00003`. The 14 ids and expected classes come from `models/fixtures/mnist/manifest.json`. The command does not parse that JSON file and does not recompute SHA-256. It rejects an unknown id, a path in the id, a missing file, a truncated or oversized file, a non-finite value, and a reference argmax that disagrees with the catalog. Comparison uses `rtol = 1e-4` and `atol = 1e-5`.

The default fixture directory is the absolute `models/fixtures/mnist` path recorded when CMake configured the build. A copied `inferedge-bench` does not carry the fixtures with it. Pass `--fixture-dir` to point at that directory. Model compatibility uses `InferenceEngine` metadata, not the filename. Add requires float32 inputs `A[2]` and `B[2]` and float32 output `C[2]`. MNIST requires float32 input `image[784]` and float32 output `logits[10]`.

Use a Release build for measurements you intend to compare. Debug and sanitizer builds are for tests. Record the git revision, ONNX Runtime 1.31.0 CPU, the model path, the fixture id, warmup, iterations, and the check interval. The session is the existing CPU session: one benchmark calling thread, one ONNX intra-op thread, and graph optimizations disabled. Active-call throughput divides completed calls by the sum of measured `run()` durations. Observed sequential throughput divides measured iterations by the outer loop, which includes validation and result destruction. Neither figure is server capacity or a GPU result. This page does not publish a speed number. With fewer than 100 samples, P99 is often the maximum.

| Exit code | Meaning |
| --- | --- |
| 0 | Help, version, or a completed benchmark |
| 2 | Invalid command line |
| 3 | The model file is missing, invalid, or not a float32 tensor |
| 4 | The fixture is missing, the wrong size, non-finite, or has the wrong reference class |
| 5 | Inference failed during warmup or a measured call |
| 6 | Loaded metadata does not match the workload |
| 7 | An output failed the correctness check |
| 8 | The benchmark report, latency statistics, or JSON serialization failed |

Successful text goes to stdout. Errors go to stderr as `error: ...`. A usage error also reprints the help text. A failure does not print the benchmark report. A JSON failure exits 8, writes the reason to stderr, and leaves stdout empty.

### JSON report

`--format json` prints one JSON object and a newline. Stdout has no banner, timing note, or other prose. Shell redirection saves it:

```bash
./build/inferedge-bench \
  --model models/add_f32.onnx \
  --workload add \
  --format json \
  > add.json
```

Schema version 1 has these fields, in this order:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `tool.name` | `inferedge-bench` |
| `tool.version` | The `inferedge` version string |
| `model` | Model path as typed |
| `workload` | `add` or `mnist` |
| `fixture` | JSON `null` for Add. For MNIST, the selected fixture id |
| `backend` | `ONNX Runtime CPU` |
| `configuration.warmup_iterations` | Unmeasured calls |
| `configuration.measured_iterations` | Measured calls |
| `configuration.correctness_check_interval` | Numeric check interval |
| `configuration.calling_threads` | `1`. `run_benchmark` uses the calling thread |
| `configuration.onnx_intra_op_threads` | `1`. The current `InferenceEngine` session calls `SetIntraOpNumThreads(1)` |
| `configuration.graph_optimizations` | `disabled`. That session calls `SetGraphOptimizationLevel(ORT_DISABLE_ALL)` |
| `latency_microseconds.sample_count` | Measured samples. It equals `measured_iterations` |
| `latency_microseconds.minimum` | Shortest measured `run`, converted from nanoseconds to microseconds |
| `latency_microseconds.mean` | Mean from `LatencySummary`, in microseconds |
| `latency_microseconds.p50`, `p95`, `p99` | Nearest-rank percentiles already stored on the summary, in microseconds |
| `latency_microseconds.maximum` | Longest measured `run`, converted from nanoseconds to microseconds |
| `latency_microseconds.population_standard_deviation` | Population standard deviation from the summary, in microseconds |
| `measurement_loop_microseconds` | Outer measured loop, converted from nanoseconds to microseconds |
| `throughput_requests_per_second.active_call` | Sample count divided by the sum of measured `run` durations |
| `throughput_requests_per_second.observed_sequential` | Measured iterations divided by the outer loop duration |
| `correctness` | `passed`. JSON is written only after a successful run |

Counts are JSON integers. Latency, loop duration, and both throughput rates are JSON numbers. The conversion from nanoseconds to microseconds keeps fractional microseconds. Mean, percentiles, standard deviation, and both throughput values are copied from `BenchmarkReport` and `LatencySummary`. The serializer does not run the model, read a clock, or recompute percentiles.

`active_call` excludes validation and loop overhead. `observed_sequential` includes validation and result destruction inside the measured loop. Neither number is server capacity. With a short sample, P99 is often the maximum and can move from run to run.

The fixed thread and optimization fields describe the current engine configuration. They are not a probe of the machine. Schema version 1 does not include CPU frequency, CPU utilization, memory utilization, the git commit, the build type, a model hash, model initialization time, or the raw `call_durations` samples. The summary is the stable contract. A later schema can add samples without changing version 1.

The writer rejects a non-finite metric, a sample count that is zero or different from the measured iteration count, a negative minimum, a maximum below the minimum, percentiles that are out of order, a negative standard deviation (including a tiny negative value or negative zero), a non-positive loop duration, a non-positive throughput, and a string that is not valid UTF-8. A rejected standard deviation is left unchanged and is not rewritten as zero. Quotes, backslashes, and ASCII controls are escaped. Valid non-ASCII UTF-8 is preserved. Invalid UTF-8 is an error, not a replacement character. There is no JSON library dependency.

## Command line

```bash
export ORT_DISABLE_TELEMETRY=1
./build/inferedge --model models/add_f32.onnx --input A=1,2 --input B=3,4
```

```text
Model: models/add_f32.onnx
Backend: ONNX Runtime CPU

Output C:
[4, 6]
```

`--model <path>` and `--model=<path>` select the file. `--input <name>=<values>` and `--input=<name>=<values>` supply one named tensor, and the flag can be repeated. Values are comma-separated finite `float32` numbers in decimal or scientific form. A negative number stays inside that argument, as in `--input A=-1.5,0`. A separate argument that starts with `-` is an option.

`--help`, `-h`, `--version`, and `-V` print to stdout and exit 0. With no arguments, the program prints the same help text and exits 0. Those two requests cannot be combined with any other argument.

This command runs a model only when every input is a fixed rank-1 `float32` tensor. The shape is the dimension stored in the model. A different rank, or a dynamic dimension, is rejected before `run`. The engine checks the element count and reports a missing or unexpected name. Printed numbers are the floats `InferenceEngine::run` returned. `std::to_chars` writes them in general format, so a whole number has no trailing decimal.

| Exit code | Meaning |
| --- | --- |
| 0 | Help, version, or a successful run |
| 2 | Invalid command line |
| 3 | The model file is missing, invalid, or not a float32 tensor |
| 4 | An input does not match the model, or an input shape is not fixed rank-1 |
| 5 | Inference failed while the session was running |

Successful text goes to stdout. Errors go to stderr as `error: ...`. A usage error also reprints the help text. The `Model:` line repeats the path argument as it was typed.

One command accepts at most 64 named inputs and 1,048,576 floats in total, with that same cap on each input. A numeric token is at most 64 characters. The values text of one `--input` is at most 1,048,576 × 65 characters, which is one maximum-length token plus a comma for every allowed value. An input name is at most 1024 characters. `std::from_chars` must consume each token. Overflow and underflow reported by that parser are usage errors. `inf` and `nan` are usage errors.

`parse_cli` does not load a model. `execute_inference` loads it once and calls `run` once. The parsed `vector<float>` stays alive for that call.

## Current limitations

- The CLI runs fixed rank-1 float32 inputs only. It does not accept a dynamic dimension or a rank other than 1.
- There is no scheduler, request queue, thread pool, or batching API.
- `inferedge-bench` does not publish a speed figure. It runs the Add and MNIST workloads only. P99 from a short sample is the nearest-rank element, often the maximum. The default fixture directory is fixed at configure time. JSON schema version 1 omits hardware, build configuration, the model hash, and raw latency samples.
- `inferedge-bench-concurrent` is CPU inference only. Each worker has one independent session, one intra-op thread, and graph optimizations disabled. It has no request queue, batching, scheduler, or production server. No concurrency speedup has been measured, and no concurrent performance baseline has been collected. JSON schema version 2 omits raw samples, hardware, build configuration, and a model hash.
- The engine runs one synchronous `float32` model on CPU, with one intra-op thread and graph optimizations disabled.
- Each `run` copies input floats and output floats. This is not zero-copy execution.
- Concurrent `run()` calls are not supported.
- The committed graphs are hand-written. They are not trained models.
- CUDA, TensorRT, and CoreML providers are not configured.
- Python is used to regenerate fixtures. The C++ build does not install it and does not link a JSON library. An optional test parses a benchmark document with `python3`'s standard-library `json` module and skips that check when `python3` is absent.

## Build

From the repository root:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/inferedge --version
./build/inferedge-bench --help
./build/inferedge-bench-concurrent --help
```

Use `Release` for an optimized build. Ninja is a single-configuration generator, so the build type is chosen at configure time. Reconfigure into a separate directory if you want both trees at once:

```bash
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
```

`Debug` keeps symbols and disables optimization. `Release` enables optimization and defines `NDEBUG`. CMake's defaults for those two configurations are used; this project does not replace them.

Project targets compile as C++20 with extensions off (`-std=c++20`, not `gnu++20` on Clang and GCC). Warnings are errors for InferEdge code: `-Wall -Wextra -Wpedantic -Werror`, plus `-Wshadow` and `-Wnon-virtual-dtor`. MSVC uses `/W4`.

## Test

```bash
ctest --test-dir build --output-on-failure
```

`ctest` runs the project tests, the ONNX Runtime tests (version check, direct Add execution, `InferenceEngine`, and the MNIST MLP fixtures), process tests that spawn `inferedge`, `inferedge-bench`, and `inferedge-bench-concurrent`, and direct checks for `--help` and `--version` on each executable.

To run the binaries directly:

```bash
export ORT_DISABLE_TELEMETRY=1
./build/inferedge_tests
./build/inferedge_onnx_smoke
```

## Formatting

```bash
find include src tests -type f \( -name '*.hpp' -o -name '*.cpp' \) -print0 \
  | xargs -0 clang-format --dry-run --Werror
```

Apply the style with `--dry-run` removed and `-i` added. CI uses clang-format **23.1.3**, matching `.github/workflows/ci.yml`.

## Layout

```text
cmake/               pinned ONNX Runtime download
include/inferedge/   public headers
models/              committed ONNX fixtures and their generators
src/                 library and executable sources
tests/               GoogleTest sources
.github/workflows/   Linux CI
```

`inferedge_core` is a static library. It contains `parse_cli` and does not link ONNX Runtime. The `inferedge` executable links `inferedge_core` and `inferedge_onnx`. `src/run_command.cpp` is compiled into that executable, and its `main` only parses arguments, runs a model when asked, and prints the result. `inferedge-bench` links `inferedge_benchmark_cli` and does not add benchmark code to `inferedge`. `inferedge-bench-concurrent` links `inferedge_concurrent_benchmark_cli`. That library is not linked into `inferedge` or `inferedge-bench`.

## How a C++ build works

### Compilation

The compiler translates one `.cpp` file at a time into an object file. `src/cli.cpp` becomes an object that contains `parse_cli`. `src/main.cpp` becomes an object that contains `main`. A header is not compiled by itself. When a `.cpp` file includes `inferedge/cli.hpp`, the compiler pastes that declaration into the translation unit it is compiling.

### Headers and source files

A header says what other files are allowed to call: names, argument types, and return types. A source file says how the function behaves. `parse_cli` is declared in `include/inferedge/cli.hpp` and defined in `src/cli.cpp`. The tests include the header and link the library, so they exercise the same function `main` calls.

`version.hpp` is header-only because the version is a compile-time string. CMake passes `-DINFEREDGE_VERSION="0.1.0"` from `project(VERSION 0.1.0)`. One version number lives in `CMakeLists.txt`.

### Linking

The linker combines object files and libraries into an executable. It matches the call to `parse_cli` in `main.cpp` with the definition compiled from `cli.cpp`. If the definition were missing, compilation of `main.cpp` could still succeed and the link step would fail with an undefined reference. That split is why a header and a source file are different files.

### What CMake does

CMake does not compile the project. It reads `CMakeLists.txt` and writes a Ninja build file under `build/`. Ninja then runs Clang, the linker, and the test discovery commands. Changing a source file and running `cmake --build build` recompiles only what is out of date.

`FetchContent` downloads GoogleTest during configuration and adds its targets to this build. InferEdge does not vendor a copy of GoogleTest or ONNX Runtime in the repository.

### Why C++20

C++20 is the newest standard this project needs. It is supported by the installed Apple Clang and by current GCC on GitHub-hosted Ubuntu runners. The code uses `std::string_view` and `constexpr` functions that are part of that standard. Newer standards can wait until a language feature earns a place in the runtime.

## License

MIT. See [LICENSE](LICENSE).
