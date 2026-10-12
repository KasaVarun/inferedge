# InferEdge CPU inference baseline

This note records the first measured CPU baseline for `inferedge-bench`. Every figure below is copied from run set `20261011T234924Z-8b26105`. The original JSON files were not rewritten. A sanitized derivative for inspection is in [`published/20261011T234924Z-8b26105/`](published/20261011T234924Z-8b26105/).

## 1. Executive summary

On one Apple silicon Mac, Release build, ONNX Runtime 1.31.0 CPU, one calling thread, and one intra-op thread, the median of five counted run means is **1.550834 microseconds** for the Add model and **7.38004 microseconds** for the MNIST MLP on fixture `test_00003`. The median active-call throughputs are **644814.3386074847** and **135500.62059284232** requests per second. Those rates count one in-process call at a time. They are not server capacity, and this run does not measure a speedup against any other system or GPU.

## 2. Benchmark implementation commit

The collector recorded Git commit `8b26105f5971035a38af6227bc34ca11589327f7`. That is the commit that built `inferedge-bench` and produced these measurements. The original local `aggregate.json` records the same SHA. Its SHA-256 is `dd0b2fb30189b6bc3e2d0a86cd14fc65bd1e85b21c321e772d9a6c6c12f3e5b9`. The sanitized copy [`published/20261011T234924Z-8b26105/aggregate.json`](published/20261011T234924Z-8b26105/aggregate.json) has SHA-256 `8c6643ea5020a01a5f6738618b295080ff274d924186b4340db3bb738b6b9793` because its model paths are repository-relative. The numbers in that file are the original statistics.

## 3. Hardware and OS

| Fact | Recorded value |
| --- | --- |
| Operating system | Darwin |
| macOS version | 27.0.1 |
| Platform | macOS-27.0.1-arm64-arm-64bit-Mach-O |
| CPU architecture | arm64 |
| Hardware model | Mac17,8 |
| Collection start | 2026-10-11T23:49:24Z |

The hardware model is the identifier returned by `sysctl -n hw.model`. The session was a normal interactive macOS login, not a quiescent lab machine.

## 4. Compiler and Release configuration

| Fact | Recorded value |
| --- | --- |
| Compiler | `/usr/bin/c++` |
| Compiler version | Apple clang version 21.0.0 (clang-2100.3.34.2) |
| Compiler target | arm64-apple-darwin27.0.0 |
| CMake | 4.4.4 |
| `CMAKE_BUILD_TYPE` | Release |
| `CMAKE_CXX_FLAGS` | empty |
| `CMAKE_CXX_FLAGS_RELEASE` | `-O3 -DNDEBUG` |
| Executable linker flags | empty |
| Sanitizer instrumentation | absent |
| Tool | `inferedge-bench` 0.1.0 |
| Binary SHA-256 | `871c094371fc8678fe11534f80bb5e5a21adbe51ab4dfd59570b01b2a68c9e39` |

The binary came from the dedicated `build-baseline` directory. Debug and ASan builds were not used.

## 5. ONNX Runtime version and execution settings

ONNX Runtime **1.31.0** CPU. The dynamic dependency is `@rpath/libonnxruntime.1.dylib`, which resolves to `libonnxruntime.1.31.0.dylib` inside that build's pinned package. `ORT_DISABLE_TELEMETRY=1` was set for every process.

Session policy, taken from the engine rather than from a hardware probe:

| Setting | Value |
| --- | --- |
| Backend | ONNX Runtime CPU |
| Calling threads | 1 |
| ONNX intra-op threads | 1 |
| Graph optimizations | disabled |

This is CPU execution. It is not a GPU, CoreML, or TensorRT run.

## 6. Model descriptions and SHA-256 values

| Workload | File | What it does | SHA-256 |
| --- | --- | --- | --- |
| Add | `models/add_f32.onnx` | Rank-1 float32 add. Inputs `A=[1,2]` and `B=[3,4]`, expected `C=[4,6]`. JSON `fixture` is null. | `37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447` |
| MNIST | `models/mnist_mlp.onnx` | MLP with 784 inputs and 10 logits. | `3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce` |

The collector checked both digests against the committed files before any trial.

## 7. MNIST fixture identity

Fixture id `test_00003`. Inputs are `models/fixtures/mnist/inputs/test_00003.f32` (784 little-endian float32 values). Reference logits are `models/fixtures/mnist/onnx/test_00003.f32`. Add does not use a fixture.

## 8. Complete benchmark methodology

Each trial was a new `inferedge-bench` process with `--format json`. Counted runs used warmup 100, measured iterations 1000, and correctness-check interval 1. Five counted runs were kept for each workload. The delay between processes was 5 seconds. There was no adaptive scheduling and no retry.

Order:

```text
pilot-add
pilot-mnist
add-01
mnist-01
add-02
mnist-02
add-03
mnist-03
add-04
mnist-04
add-05
mnist-05
```

A trial was accepted only when the process exited 0, stderr was empty, and the stdout report passed schema version 1 checks, including `correctness` `passed`. All 12 processes met that bar. `failure.json` was not written.

Pilots are stored and are excluded from `aggregate.json` (`pilots_excluded` is true). The cross-run block is the mean or median of the five per-run statistics. `median_of_run_p99_microseconds` is the median of five per-run P99 values. It is not a pooled P99 of 5000 calls, and it is not the average of those five P99 values. `confidence_interval_computed` is false.

## 9. Actual aggregate results

Source: the original `aggregate.json` field `workloads.<name>.cross_run`, also present with the same numbers in [`published/20261011T234924Z-8b26105/aggregate.json`](published/20261011T234924Z-8b26105/aggregate.json). Latency is microseconds. Throughput is requests per second. The relative range is the stored dimensionless ratio `(maximum run mean - minimum run mean) / median run mean`. The percentage is 100 times that stored ratio.

| Metric | Add | MNIST |
| --- | --- | --- |
| Mean of run means | 1.5651514 µs | 7.333332400000001 µs |
| Median of run means | 1.550834 µs | 7.38004 µs |
| Median of run P50 | 1.5 µs | 7.375 µs |
| Median of run P95 | 1.709 µs | 7.625 µs |
| Median of run P99 | 1.917 µs | 11.375 µs |
| Minimum run mean | 1.527538 µs | 6.969299 µs |
| Maximum run mean | 1.651079 µs | 7.498247999999999 µs |
| Relative run-mean range | 0.07966100820590721 (7.966%) | 0.07167291776196322 (7.167%) |
| Median active-call throughput | 644814.3386074847 req/s | 135500.62059284232 req/s |
| Median observed sequential throughput | 625341.9838974439 req/s | 134320.58966738865 req/s |

Each `per_run` value in the aggregate equals the same field in that trial's `runs/<id>.json` report.

## 10. Pilot-run results

Pilots passed the same acceptance checks and are not in the table above.

| Trial | Mean | P50 | P95 | P99 | Active-call | Observed sequential |
| --- | --- | --- | --- | --- | --- | --- |
| pilot-add | 1.051618 µs | 1.042 µs | 1.125 µs | 1.208 µs | 950915.6366665462 req/s | 921340.550500979 req/s |
| pilot-mnist | 7.354077 µs | 7.459 µs | 7.834 µs | 8.542 µs | 135978.99505267624 req/s | 135032.49219343404 req/s |

The pilot-add mean is below every counted Add mean. Each process, including the pilot, still ran 100 warmup iterations before its measured loop. The gap is between processes.

## 11. Run-to-run variability

Counted Add means, in trial order: 1.527538, 1.540106, 1.550834, 1.651079, 1.5562 microseconds. Four of them sit between 1.527538 and 1.5562. `add-04` is the maximum, 1.651079 microseconds.

Counted MNIST means, in trial order: 6.969299, 7.4615, 7.38004, 7.357575, 7.498247999999999 microseconds. The spread from 6.969299 to 7.498247999999999 is the 7.167% relative range above. No counted trial failed, and every stderr file is empty.

## 12. MNIST P99 tail variability

Counted MNIST P99 values, in trial order: 11.166, 15.167, 8.584, 11.375, 11.958 microseconds. The aggregate median of those five numbers is 11.375 microseconds.

`mnist-02` carries the high tail: P99 15.167 microseconds while that same run's P95 is 7.667 microseconds and its mean is 7.4615 microseconds. The mean did not move with the tail. `mnist-03` has the low P99, 8.584 microseconds. These are five separate within-run order statistics. They are not one pooled tail.

## 13. Measurement boundaries

For each measured call, the per-call timer wraps `InferenceEngine::run` on the calling thread. That interval includes input validation, input copy, ONNX Runtime, and output copy inside `run`. The outer measurement loop starts after warmup and stops after the last result is destroyed, so it also includes correctness checks and loop bookkeeping. Check interval 1 compares every measured call after the per-call timer, so those checks affect observed sequential time and do not sit inside the per-call latency.

With 1000 samples, P99 is one nearest-rank order statistic of that process. It is not the maximum, and it is not a percentile of the combined 5000 calls.

## 14. Active-call versus observed throughput

Active-call throughput is the measured iteration count divided by the sum of per-call `run` durations. Observed sequential throughput is the measured iteration count divided by the outer-loop duration. The Add medians are 644814.3386074847 and 625341.9838974439 requests per second. The MNIST medians are 135500.62059284232 and 134320.58966738865 requests per second. Observed is lower on both workloads because the outer loop includes checks and result destruction. The gap is wider for Add, where the call itself is about 1.55 microseconds and the bookkeeping is a larger share of the loop. Neither rate is the capacity of a server, a batch, or a pool of threads.

## 15. Reproduction commands

Run these from the repository root. A new campaign writes a new run-set directory. Its timestamps, latencies, and throughputs will differ from this note.

```bash
export ORT_DISABLE_TELEMETRY=1
cmake -S . -B build-baseline -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-baseline --target inferedge-bench
python3 benchmarks/collect_baseline.py \
  --binary build-baseline/inferedge-bench \
  --expected-commit <reviewed-commit> \
  --preflight-only
python3 benchmarks/collect_baseline.py \
  --binary build-baseline/inferedge-bench \
  --expected-commit <reviewed-commit>
python3 -m benchmarks.analyze_baseline \
  --run-set benchmarks/baselines/<RUN_SET_DIRECTORY>
```

Preflight checks the Release cache, sanitizer absence, the pinned ONNX Runtime 1.31.0 library after symlink resolution, both model SHA-256 values, `inferedge-bench` 0.1.0, and a clean Git tree aside from `models/experiments/`. `--preflight-only` does not start a workload. Defaults are warmup 100, iterations 1000, check interval 1, five counted runs per workload, and a 5 second delay. Pass the commit that should be measured as `--expected-commit`. This campaign used `8b26105f5971035a38af6227bc34ca11589327f7`.

Configure reports `ONNX Runtime 1.31.0 CPU archive verified` when the pinned archive matches. The collector then requires the resolved library file to be the 1.31.0 library inside that build.

## 16. Known limitations

Five runs do not establish statistical significance, and no confidence interval is reported. The machine was in interactive use, so scheduling noise can move a run mean by several percent, as `add-04` and the MNIST P99 spread show. Graph optimizations are disabled, so this is not a tuned ONNX Runtime session. One calling thread and one intra-op thread do not represent a server. Add and MNIST are different workloads: a two-element add versus one 784-input MLP example. No second configuration was measured, so this note does not claim an optimization gain.

The original local run set keeps absolute filesystem paths from the machine that collected it. This document does not repeat those paths. The inspectable copy is [`published/20261011T234924Z-8b26105/`](published/20261011T234924Z-8b26105/), described by [`published/20261011T234924Z-8b26105/PROVENANCE.md`](published/20261011T234924Z-8b26105/PROVENANCE.md). That copy is a derivative: repository-absolute paths were rewritten to repository-relative paths after the collector and the analyzer had already accepted the original run set. Do not edit the original files, and do not pass the published directory to `python3 -m benchmarks.analyze_baseline`. That tool validates a collector run set and writes `aggregate.json` only when the file is absent. This directory already contains the sanitized aggregate. A new measurement must be collected with the committed collector.

## 17. Interpretation for future optimization studies

Use this run as the reference point for a later CPU comparison only when the candidate matches the model SHA-256 values, fixture `test_00003` for MNIST, warmup 100, measured iterations 1000, check interval 1, Release without sanitizers, ONNX Runtime 1.31.0 CPU, one calling thread, one intra-op thread, and disabled graph optimizations. Record the hardware again. Compare the median of run means and the median active-call throughput. Do not compare a median of five P99s with a pooled P99, and do not read either throughput as server capacity. A difference on a later run is a measurement delta. It is not, by itself, evidence of an improvement.

## 18. Sanitized evidence

| File | Role |
| --- | --- |
| [`published/20261011T234924Z-8b26105/PROVENANCE.md`](published/20261011T234924Z-8b26105/PROVENANCE.md) | Run identity, checksums, and the path fields that were rewritten |
| [`published/20261011T234924Z-8b26105/environment.json`](published/20261011T234924Z-8b26105/environment.json) | Hardware, compiler, Release configuration, model checksums, and protocol |
| [`published/20261011T234924Z-8b26105/aggregate.json`](published/20261011T234924Z-8b26105/aggregate.json) | Five counted runs per workload; pilots excluded |
| [`published/20261011T234924Z-8b26105/runs/`](published/20261011T234924Z-8b26105/runs/) | Twelve reports, their stdout copies, empty stderr files, and trial metadata |

The local directory `benchmarks/baselines/20261011T234924Z-8b26105/` is still the authoritative record. The links above point at the sanitized derivative only.
