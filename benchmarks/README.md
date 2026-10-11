# Native inference baselines

`benchmarks/collect_baseline.py` launches the existing `inferedge-bench` executable and stores one schema-version-1 JSON report per process. `benchmarks/analyze_baseline.py` reduces one completed run set. Neither script runs a model.

## Protocol

Use a dedicated Release build. Debug and ASan/UBSan binaries are rejected. The collector reads `CMakeCache.txt` beside the binary, records the compiler and linker flags it finds, and refuses the run when `CMAKE_BUILD_TYPE` is not `Release` or when sanitizer flags or sanitizer symbols are present. It does not assume a particular optimization spelling such as `-O3`.

ONNX Runtime must be the pinned 1.31.0 CPU package selected by that build. On macOS the load command may name `@rpath/libonnxruntime.1.dylib` or the fully versioned `libonnxruntime.1.31.0.dylib`. The collector reads `LC_RPATH`, expands `@executable_path` and `@loader_path`, follows the symlink, and requires the resolved file to be `lib/libonnxruntime.1.31.0.dylib` inside that package. On Linux it accepts `libonnxruntime.so.1` and `libonnxruntime.so.1.31.0` under the same rule. A filename that merely contains `onnxruntime` is not enough, and the Mach-O current-version field is not treated as proof. `--preflight-only` runs these checks and does not start a trial.

Configure and build:

```bash
export ORT_DISABLE_TELEMETRY=1
cmake -S . -B build-baseline -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-baseline --target inferedge-bench
```

The campaign command, after that binary exists:

```bash
python3 benchmarks/collect_baseline.py \
  --binary build-baseline/inferedge-bench
```

Defaults are warmup 100, measured iterations 1000, correctness-check interval 1, five counted runs of each workload, and a 5 second delay between processes. The approved commit is `6c1ba833a512e288b436e06a0bd430b41988518b`. Override `--expected-commit` only for a different reviewed commit.

Each trial is a new process with `ORT_DISABLE_TELEMETRY=1` and `--format json`. Arguments are passed as an array. The shell is not used.

Add:

```text
inferedge-bench --model <absolute add model> --workload add \
  --warmup 100 --iterations 1000 --check-interval 1 --format json
```

MNIST uses fixture `test_00003` and the absolute `models/fixtures/mnist` directory. The input file is `inputs/test_00003.f32` and the reference logits are `onnx/test_00003.f32`. Add has no fixture; its JSON `fixture` is null. Add compares `A=[1,2]` and `B=[3,4]` with expected `C=[4,6]`. MNIST compares a 784-element image with 10 logits.

## Trial order

One uncounted pilot of each workload runs first. Counted trials then alternate so neither workload sits entirely on a cold or warm machine:

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

Pilots stay in the run set. The collector does not write `aggregate.json`. The analyzer reads a completed run set and writes that file. It skips `pilot-add` and `pilot-mnist`.

## Run-set files

The directory name is the UTC timestamp plus the short commit, under `benchmarks/baselines/`. An existing directory is rejected. A second attempt uses a new id.

```text
benchmarks/baselines/<YYYYMMDDTHHMMSSZ-commit>/
  environment.json
  runs/<trial>.stdout
  runs/<trial>.stderr
  runs/<trial>.meta.json
  runs/<trial>.json          # accepted stdout, unchanged
  failure.json               # only when a trial stops the set
```

`environment.json` separates queried host and git facts, the verified CMake and checksum configuration, and the source-defined session policy: one calling thread, one ONNX intra-op thread, graph optimizations disabled, and the ONNX Runtime 1.31.0 CPU provider. A hardware model that `sysctl` cannot return is stored as null.

The collector checks SHA-256:

| File | SHA-256 |
| --- | --- |
| `models/add_f32.onnx` | `37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447` |
| `models/mnist_mlp.onnx` | `3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce` |

It also records the SHA-256 of the `inferedge-bench` binary. Tracked source changes are rejected. `models/experiments/` may stay untracked. Any other untracked path, including an older run set, is rejected. Move an older run set aside by hand. The collector does not delete it.

## Failure

The first failing trial stops the campaign. Exit status must be 0, stderr must be empty, and stdout must be one JSON object plus one newline. Validation then checks schema version 1, the tool name and version, `correctness` `passed`, the workload, fixture, absolute model path, backend, warmup, iteration count, check interval, sample count, thread counts, disabled graph optimizations, finite numbers, latency order, a nonnegative standard deviation, a positive measurement loop, and positive throughput. `NaN` and `Infinity` are rejected. Booleans are not accepted as numbers, and count fields must be JSON integers.

Accepted reports are the process stdout copied unchanged. A rejected trial still keeps `.stdout`, `.stderr`, and `.meta.json`. `failure.json` names the trial and the reason. Earlier accepted `.json` files stay. There is no retry and no aggregate.

Inspect a failure by reading `failure.json`, then that trial's `.stderr`, `.stdout`, and `.meta.json`.

## Analysis

After a run set has `status` `completed`, point the analyzer at that directory. It does not search for a latest run set. Run the command from the repository root:

```bash
python3 -m benchmarks.analyze_baseline \
  --run-set benchmarks/baselines/<YYYYMMDDTHHMMSSZ-commit>
```

`python3 benchmarks/analyze_baseline.py --run-set <directory>` is the same program. It adds the repository root to the module search path itself.

The analyzer reads `environment.json` and these counted reports:

```text
runs/add-01.json through runs/add-05.json
runs/mnist-01.json through runs/mnist-05.json
```

It also reads each counted trial's `runs/<id>.meta.json` and requires `accepted` true, return code 0, and an empty problem list. The manifest must say `completed`, name the run-set directory, record a full git SHA, and carry the model SHA-256 values checked by the collector. Warmup must be 100, measured iterations 1000, check interval 1, and counted runs 5 per workload. Each report is checked with the collector's schema-version-1 validator. A missing report, a `running` or `failed` status, or a rejected report stops analysis before any statistic is written.

`aggregate.json` is schema version 1 for the aggregate and records source benchmark schema version 1. It contains the run-set id, git revision, Release build type, model paths and digests, the experiment configuration taken from the manifest, the five counted trial ids per workload, the per-run statistics copied from those reports, and the cross-run reductions below. `pilots_excluded` is true. `median_of_run_p99_is_pooled_p99` and `confidence_interval_computed` are false.

For each workload the cross-run block is:

| Field | Calculation |
| --- | --- |
| `mean_of_run_means_microseconds` | Arithmetic mean of the five per-run means |
| `median_of_run_means_microseconds` | Median of the five per-run means |
| `median_of_run_p50_microseconds` | Median of the five per-run P50 values |
| `median_of_run_p95_microseconds` | Median of the five per-run P95 values |
| `median_of_run_p99_microseconds` | Median of the five per-run P99 values |
| `minimum_run_mean_microseconds` | Smallest of the five per-run means |
| `maximum_run_mean_microseconds` | Largest of the five per-run means |
| `relative_run_mean_range` | `(maximum run mean - minimum run mean) / median run mean` |
| `median_active_call_throughput_requests_per_second` | Median of the five active-call throughputs |
| `median_observed_sequential_throughput_requests_per_second` | Median of the five observed sequential throughputs |

Latency fields are microseconds. Throughput fields are requests per second. The relative range is a dimensionless ratio. A zero or non-positive median run mean is rejected so the ratio cannot become NaN or infinity. Values are not rounded to a fixed number of decimal places.

`median_of_run_p99_microseconds` is the middle of the five P99 numbers already computed inside each process. It is not the P99 of a pooled list of 5000 calls, and it is not the average of those five P99 numbers. Five runs do not produce a confidence interval. The aggregate states that none was computed.

An existing `aggregate.json` is left untouched and the command returns an error. The full document is built first. The file is published by writing a temporary file in the run set and linking it into place, so a failed analysis does not leave a partial `aggregate.json`. Source reports, metadata, and `environment.json` are not modified.

Do not commit `benchmarks/baselines/` until a measured campaign has been reviewed. The aggregate is still a local artifact.

## What these numbers are

Active-call throughput counts only measured `run` durations. Observed sequential throughput includes validation and result destruction in the measurement loop. Neither number is server capacity. Add and MNIST are different workloads: a two-element add versus a 784-input MLP on one fixture. With 1000 samples, P99 is one order statistic from that process, not a pooled tail across the five runs.

The synthetic tests do not execute the model:

```bash
python3 -m unittest benchmarks.test_collect_baseline benchmarks.test_analyze_baseline
python3 -m py_compile benchmarks/collect_baseline.py benchmarks/analyze_baseline.py \
  benchmarks/test_collect_baseline.py benchmarks/test_analyze_baseline.py
```
