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
python3 -m unittest benchmarks.test_collect_baseline benchmarks.test_analyze_baseline \
  benchmarks.test_collect_concurrent
python3 -m py_compile benchmarks/collect_baseline.py benchmarks/analyze_baseline.py \
  benchmarks/collect_concurrent.py benchmarks/test_collect_baseline.py \
  benchmarks/test_analyze_baseline.py benchmarks/test_collect_concurrent.py
```

## Concurrent scaling collection

`benchmarks/collect_concurrent.py` launches `inferedge-bench-concurrent` and stores one schema-version-2 JSON report per process. It does not launch `inferedge-bench`, does not write `aggregate.json`, and does not run a campaign by being imported. No concurrent analyzer is included. No concurrent scaling run set has been collected, and this repository has no measured concurrent scaling results. The measurement commit is whatever full SHA the operator passes. This file does not name one.

### Matrix

| Workload | Workers | Warmup per worker | Measured per worker | Check interval | Counted processes |
| --- | --- | --- | --- | --- | --- |
| add | 1, 2, 4, 8 | 100 | 1000 | 1 | 5 each |
| mnist | 1, 2, 4, 8 | 100 | 1000 | 1 | 5 each |

Each of the eight configurations also has one uncounted pilot. The campaign is 8 pilots plus 40 counted processes, 48 processes in total. MNIST uses fixture `test_00003`. Add passes no fixture flags. `--concurrency` is present on every command. A failed process stops the campaign. There is no retry and no change of order based on a result. The cooldown is 5 seconds after every process except the first.

### Trial order

Pilots follow the canonical rotation. Counted round `r` rotates the start by `r - 1` positions and stamps run number `r` on each id:

```text
pilot-add-c01
pilot-mnist-c01
pilot-add-c02
pilot-mnist-c02
pilot-add-c04
pilot-mnist-c04
pilot-add-c08
pilot-mnist-c08
add-c01-01
mnist-c01-01
add-c02-01
mnist-c02-01
add-c04-01
mnist-c04-01
add-c08-01
mnist-c08-01
mnist-c01-02
add-c02-02
mnist-c02-02
add-c04-02
mnist-c04-02
add-c08-02
mnist-c08-02
add-c01-02
add-c02-03
mnist-c02-03
add-c04-03
mnist-c04-03
add-c08-03
mnist-c08-03
add-c01-03
mnist-c01-03
mnist-c02-04
add-c04-04
mnist-c04-04
add-c08-04
mnist-c08-04
add-c01-04
mnist-c01-04
add-c02-04
add-c04-05
mnist-c04-05
add-c08-05
mnist-c08-05
add-c01-05
mnist-c01-05
add-c02-05
mnist-c02-05
```

`c01` is one worker. The same configuration is never adjacent to itself. Pilots stay in the run set. A later analyzer is responsible for leaving them out of any aggregate. This collector does not write that aggregate.

### Release preflight

Use a dedicated Release build directory. `CMakeCache.txt` must sit beside `inferedge-bench-concurrent`. `CMAKE_BUILD_TYPE` must be `Release`. `CMAKE_CXX_FLAGS_RELEASE` must request optimization (`-O2`, `-O3`, `-Ofast`, or `/O2`). Sanitizer flags and sanitizer symbols are rejected. The binary's `--version` text must be `inferedge-bench-concurrent 0.1.0`.

ONNX Runtime must be the pinned 1.31.0 CPU package selected by that build. The collector reuses the sequential library resolution: on macOS the load command may name `@rpath/libonnxruntime.1.dylib` or `libonnxruntime.1.31.0.dylib`, and the resolved file must be `lib/libonnxruntime.1.31.0.dylib` inside that package. On Linux it accepts `libonnxruntime.so.1` and `libonnxruntime.so.1.31.0` under the same rule. An ambiguous match, an unrecognized filename, or a library outside the pinned package is rejected.

`--expected-commit` is required. It must be a 40-character lowercase hexadecimal SHA. Preflight requires `git rev-parse HEAD` to equal that SHA. There is no default commit stored in the collector.

Tracked changes are rejected. The concurrent collector allows these untracked roots, and records them in `environment.json`:

- `models/experiments/`
- `benchmarks/baselines/`
- `benchmarks/concurrent_runs/`

Any other untracked path is rejected. The sequential collector's allow-list remains `models/experiments/` only.

The manifest records the operating system, OS version, architecture, Python version, compiler, CMake flags, binary SHA-256, ONNX Runtime version, and the resolved library path. CPU concurrency is queried. On macOS the collector runs `sysctl -n hw.model`, `hw.physicalcpu`, `hw.logicalcpu`, and `hw.ncpu`. `os.cpu_count()` is stored separately. A core count is not derived from a hardware identifier such as `Mac17,8`. A failed query is stored as null plus the error text.

Configure and check without starting a trial:

```bash
export ORT_DISABLE_TELEMETRY=1
cmake -S . -B build-concurrent -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-concurrent --target inferedge-bench-concurrent
python3 benchmarks/collect_concurrent.py \
  --binary build-concurrent/inferedge-bench-concurrent \
  --expected-commit <40-character-measurement-sha> \
  --preflight-only
```

Replace `<40-character-measurement-sha>` with the commit that produced the binary. Collection, once that check has passed:

```bash
python3 benchmarks/collect_concurrent.py \
  --binary build-concurrent/inferedge-bench-concurrent \
  --expected-commit <40-character-measurement-sha>
```

`--preflight-only` exits before any workload process and creates no run-set directory. Each trial is a new process. Arguments are an array. The shell is not used. The child environment sets `ORT_DISABLE_TELEMETRY=1`.

Add:

```text
inferedge-bench-concurrent --model <absolute add model> --workload add \
  --concurrency <1|2|4|8> --warmup 100 --iterations 1000 \
  --check-interval 1 --format json
```

MNIST adds `--fixture test_00003` and `--fixture-dir` set to the absolute `models/fixtures/mnist` directory.

### Checksums

Local files are hashed and compared with these constants. The same constants are compared with `git cat-file` blobs at `--expected-commit`. A hash computed at runtime is not used as its own expected value.

| File | SHA-256 |
| --- | --- |
| `models/add_f32.onnx` | `37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447` |
| `models/mnist_mlp.onnx` | `3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce` |
| `models/fixtures/mnist/inputs/test_00003.f32` | `692f8ff1c2aaa7dd3c5982979516e774e4daf6d8420f1818a68389825ebb84a9` |
| `models/fixtures/mnist/onnx/test_00003.f32` | `9a5daa05f5096f5521deab05c025d3c3691e2587189f176d8699a9521b663a46` |
| `models/fixtures/mnist/manifest.json` | `4e6f95f8626fa89a22f59aed004870b942fcf2d7bb4add2226f9ca4f9f13009a` |

The binary SHA-256 is recorded as well.

### Schema version 2

Stdout must be one JSON object and one trailing newline. `schema_version` must be 2. The tool name is `inferedge-bench-concurrent` and the version is `0.1.0`. `correctness` must be `passed`. The backend is `ONNX Runtime CPU`. Session ownership is `one_inference_engine_per_worker`, intra-op threads per session are 1, and graph optimizations are `disabled`. The workload, absolute model path, fixture (`null` for Add, `test_00003` for MNIST), worker count, warmup, measured iterations, and check interval must match the trial. `total_completed_requests`, every per-worker count, and `latency_microseconds.sample_count` must describe `worker_count * 1000` completed calls, with 1000 on each worker.

`NaN` and `Infinity` are rejected. Booleans are not counts or metrics. Negative latencies, a negative or negative-zero standard deviation, inverted percentiles, a non-positive wall duration, and a non-positive throughput are rejected. Aggregate throughput must match `total_completed_requests / (measurement_wall_microseconds / 1e6)` within a relative tolerance of `1e-6` after JSON parsing. The report is not rewritten. Schema version 1 fields such as `measurement_loop_microseconds` and `throughput_requests_per_second` are unexpected.

### Evidence layout

Run sets go under `benchmarks/concurrent_runs/`, separate from `benchmarks/baselines/`. The directory name is a UTC timestamp plus the seven-character commit prefix. An existing directory is left unchanged and the command fails.

```text
benchmarks/concurrent_runs/<YYYYMMDDTHHMMSSZ-commit>/
  environment.json
  runs/<trial>.stdout
  runs/<trial>.stderr
  runs/<trial>.meta.json
  runs/<trial>.json          # accepted stdout, unchanged
  failure.json               # only when a trial stops the set
```

`environment.json` is `running` while processes are in flight, `completed` only after all 48 trials are accepted, and `failed` when a trial stops the set. `aggregate` in that file stays null.

### Failure

A preflight failure exits nonzero, starts no trial, and creates no run-set directory. A trial failure keeps `.stdout`, `.stderr`, and `.meta.json` for that trial, writes `failure.json`, marks the manifest `failed`, and keeps earlier accepted `.json` files, including pilots. Later trials are not started. An exception while launching a process is recorded the same way. The collector does not create `aggregate.json`.

### Private artifacts

`benchmarks/concurrent_runs/` stays untracked until a later review. The manifest contains absolute paths for the binary, compiler, models, fixture directory, and resolved ONNX Runtime library. Those raw files are not a published result. A later sanitization pass would have to write a separate derivative and leave this directory unchanged. The synthetic tests hash and parse reports. They do not execute a model.
