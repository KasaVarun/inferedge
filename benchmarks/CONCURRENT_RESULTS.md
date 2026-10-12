# InferEdge concurrent scaling results

This note records the first measured concurrent scaling campaign for `inferedge-bench-concurrent`. Every figure below is copied from run set `20261012T032311Z-c07da91`. The original private JSON files were not rewritten. A sanitized derivative for inspection is in [`published/20261012T032311Z-c07da91/`](published/20261012T032311Z-c07da91/).

## 1. Executive summary

On one Apple silicon Mac, Release build, ONNX Runtime 1.31.0 CPU, eight workers achieved **4.868985211597192** times the median aggregate MNIST throughput of one worker in this experiment, and **2.3366525240280547** times the median aggregate Add throughput. Rendered to three significant figures, those ratios are 4.87x and 2.34x. Both ratios use the concurrent worker-count-one result for the same workload as the control.

The eight-worker median aggregate rates are **643425.1321374044** completed MNIST requests per second and **1385401.3334487835** completed Add requests per second. Aggregate throughput divides completed measured requests by the shared wall-clock duration of that process. It is not a per-call rate, and it is not production server capacity.

Both workloads had lower median aggregate throughput at two workers than at one. Median run P99 latency was higher than the one-worker value at every measured concurrency above one. Five counted runs per cell do not support a confidence interval or a claim of statistical significance. Eight workers is the largest concurrency in the matrix. Throughput at eight workers was higher than at four, so this campaign does not show saturation.

## 2. Hardware and software

| Fact | Recorded value |
| --- | --- |
| Operating system | Darwin |
| OS version | 27.0.0 |
| macOS version | 27.0.1 |
| Platform | macOS-27.0.1-arm64-arm-64bit-Mach-O |
| CPU architecture | arm64 |
| Hardware model | Mac17,8 |
| `hw.physicalcpu` | 18 |
| `hw.logicalcpu` | 18 |
| `hw.ncpu` | 18 |
| `os.cpu_count()` | 18 |
| Python | 3.14.7 |
| CMake | 4.4.4 |
| Compiler | `/usr/bin/c++` |
| Compiler version | Apple clang version 21.0.0 (clang-2100.3.34.2) |
| Compiler target | arm64-apple-darwin27.0.0 |
| `CMAKE_BUILD_TYPE` | Release |
| `CMAKE_CXX_FLAGS_RELEASE` | `-O3 -DNDEBUG` |
| Sanitizer instrumentation | absent |
| ONNX Runtime | 1.31.0 CPU |
| Tool | `inferedge-bench-concurrent` 0.1.0 |
| Binary SHA-256 | `614564d20a12d298d0cf453cfe447e32b5645acaf9725bbe6dc3958e7cf37baf` |
| Collection start | 2026-10-12T03:23:11Z |
| Measurement commit | `c07da912da109c09b6e3ea6f9ae8f1a89e97a79e` |

The hardware model is the identifier returned by `sysctl -n hw.model`. The session was a normal interactive macOS login, not a quiescent lab machine. The compiler's recorded `InstalledDir` is `/Library/Developer/CommandLineTools/usr/bin`.

## 3. Inference execution architecture

Each trial is one process. That process creates one `InferenceEngine` per worker. Sessions are not shared. Each session uses the ONNX Runtime CPU provider, one intra-op thread, and graph optimizations disabled (`ORT_DISABLE_ALL`). The collector sets `ORT_DISABLE_TELEMETRY=1`.

There is no request queue, scheduler, or dynamic batch. `InferenceEngine::run` is not called concurrently on one engine. Worker count 1 still uses this concurrent harness: one engine, one worker, and the same aggregate-throughput definition as the larger cells. That one-worker concurrent result is the scaling control.

Aggregate throughput is `total_successful_measured / wall_seconds` for the shared measurement window. Active-call throughput is stored beside it and describes time inside `run`. Scaling ratios do not use the active-call rate.

## 4. Experimental methodology

The matrix is Add and MNIST at worker counts 1, 2, 4, and 8. Each configuration has one pilot and five counted runs. The campaign is 8 pilots plus 40 counted processes, 48 processes in total. Every counted cell in the published aggregate lists exactly those five trial ids.

Each worker warms up for 100 iterations and then measures 1000 iterations. Correctness is checked on every measured iteration. All 48 accepted reports record `correctness: passed`. MNIST uses fixture `test_00003`. Add passes no fixture flags. The cooldown is 5.0 seconds after every process except the first. Trial order is the collector order recorded in `trial_order`. A failed process would have stopped the campaign. This run set has no `failure.json`.

Pilots are validated and kept. Their measurements are excluded from every median, ratio, efficiency, and range in the table below. `pilots_excluded` is true.

The median of five values is the middle value after sorting. Relative throughput range is `(maximum - minimum) / median` of the five aggregate rates in that cell. `median_of_run_p99_is_pooled_p99` is false: the P99 column is the median of five per-process P99 values, not one percentile pooled across processes. `confidence_interval_computed` is false.

## 5. Full eight-row performance table

Medians below are taken from the original `aggregate.json`. Throughput is aggregate completed requests per second. Latencies are microseconds.

| Workload | Workers | Median aggregate throughput | Throughput ratio | Scaling efficiency | Median run mean | Median run P50 | Median run P95 | Median run P99 | Relative throughput range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| add | 1 | 592900.0222337508 | 1.0 | 1.0 | 1.614209 | 1.625 | 1.792 | 2.0 | 0.12797900166516518 |
| add | 2 | 273552.6058074124 | 0.461380663769927 | 0.2306903318849635 | 7.108745 | 6.958 | 9.625 | 10.541 | 0.09731638448947107 |
| add | 4 | 587353.5727543123 | 0.9906452196467421 | 0.24766130491168553 | 6.529506250000001 | 6.416 | 8.125 | 9.791 | 0.44252237800501937 |
| add | 8 | 1385401.3334487835 | 2.3366525240280547 | 0.29208156550350683 | 5.052490625000001 | 4.666 | 8.041 | 11.291 | 0.2973705481083145 |
| mnist | 1 | 132147.68666884885 | 1.0 | 1.0 | 7.500732 | 7.542 | 7.791 | 10.833 | 0.04213274495333564 |
| mnist | 2 | 123198.2259455464 | 0.9322768264136998 | 0.4661384132068499 | 16.022337 | 15.875 | 18.875 | 23.458 | 0.07409562866849889 |
| mnist | 4 | 278469.9302599907 | 2.1072629970270556 | 0.5268157492567639 | 13.99221075 | 13.916 | 16.75 | 18.875 | 0.29990895083715413 |
| mnist | 8 | 643425.1321374044 | 4.868985211597192 | 0.608623151449649 | 11.42514625 | 10.875 | 15.25 | 21.375 | 0.11325592509451522 |

Counted trial ids are `add-c01-01` through `add-c01-05`, and the same pattern for `c02`, `c04`, and `c08`, with `mnist-` for the MNIST cells. The eight pilots are `pilot-add-c01`, `pilot-mnist-c01`, `pilot-add-c02`, `pilot-mnist-c02`, `pilot-add-c04`, `pilot-mnist-c04`, `pilot-add-c08`, and `pilot-mnist-c08`.

## 6. Add throughput scaling analysis

The Add control is worker count 1: median aggregate throughput 592900.0222337508 completed requests per second.

Two workers reached 273552.6058074124, a throughput ratio of 0.461380663769927 and a scaling efficiency of 0.2306903318849635. Four workers reached 587353.5727543123, a ratio of 0.9906452196467421 and an efficiency of 0.24766130491168553. That four-worker median is close to the one-worker median and is not a large gain. Eight workers reached 1385401.3334487835, a ratio of 2.3366525240280547 and an efficiency of 0.29208156550350683. Eight workers achieved 2.34x the median aggregate Add throughput of one worker in this experiment.

Efficiency stays below 1 at every Add concurrency above one. The ratio rises from four workers to eight workers. The campaign did not measure a worker count above eight, so the eight-worker point is the highest Add aggregate throughput in this matrix, not a demonstrated plateau.

The five-run Add ranges, as minimum and maximum aggregate throughput, are 561232.7364810258 to 637111.4894137555 at one worker, 249868.5691326362 to 276489.7196974871 at two, 487472.92087924515 to 747390.0206242276 at four, and 1193132.1523611862 to 1605109.7062388407 at eight.

## 7. MNIST throughput scaling analysis

The MNIST control is worker count 1: median aggregate throughput 132147.68666884885 completed requests per second.

Two workers reached 123198.2259455464, a throughput ratio of 0.9322768264136998 and a scaling efficiency of 0.4661384132068499. Four workers reached 278469.9302599907, a ratio of 2.1072629970270556 and an efficiency of 0.5268157492567639. Eight workers reached 643425.1321374044, a ratio of 4.868985211597192 and an efficiency of 0.608623151449649. Eight workers achieved 4.87x the median aggregate MNIST throughput of one worker in this experiment.

That 4.87x figure is a ratio of median aggregate throughputs. It is not a claim that one MNIST inference became 4.87x faster, and it is not a claim that the neural network itself became 4.87x faster. Per-request latency rose. See section 8.

MNIST efficiency is higher than Add efficiency at 2, 4, and 8 workers, and it remains below 1. Throughput at eight workers is higher than at four. This matrix does not show saturation at eight workers.

The five-run MNIST ranges are 129957.33630606408 to 135525.081084656 at one worker, 120894.00877702593 to 130022.45877930496 at two, 224942.89402867158 to 308458.51865264087 at four, and 625584.4327067428 to 698456.1412760252 at eight.

## 8. Per-request latency trade-offs

Median run-mean latency increased at every concurrency above one, for both workloads.

| Workload | Workers | Median run mean (µs) | Mean-latency ratio |
| --- | ---: | ---: | ---: |
| add | 1 | 1.614209 | 1.0 |
| add | 2 | 7.108745 | 4.403856625752923 |
| add | 4 | 6.529506250000001 | 4.045019108430197 |
| add | 8 | 5.052490625000001 | 3.1300101938472658 |
| mnist | 1 | 7.500732 | 1.0 |
| mnist | 2 | 16.022337 | 2.1361031163358457 |
| mnist | 4 | 13.99221075 | 1.865446032467231 |
| mnist | 8 | 11.42514625 | 1.5232041686064772 |

After the two-worker peak, the median run mean falls at four and eight workers and stays above the one-worker mean. Higher aggregate throughput at eight workers therefore arrives with a slower typical request than the one-worker control.

Active-call throughput is recorded in the aggregate and is not the scaling numerator. At one worker the median active-call rates are 619498.4664315464 Add requests per second and 133320.32126997737 MNIST requests per second. Those rates count time inside `run`. The aggregate rates in section 5 count the shared wall clock.

## 9. P50, P95, and P99 discussion

Median run P50 follows the same shape as the run mean: higher than the one-worker value at 2, 4, and 8, with the largest P50 at two workers.

Median run P95 is also higher than the one-worker value at every larger concurrency. For Add it is 1.792, 9.625, 8.125, and 8.041 microseconds at 1, 2, 4, and 8 workers. For MNIST it is 7.791, 18.875, 16.75, and 15.25.

P99 latency increased at every measured concurrency level above one. The median of the five per-process P99 values is:

| Workload | Workers | Median run P99 (µs) | P99 ratio |
| --- | ---: | ---: | ---: |
| add | 1 | 2.0 | 1.0 |
| add | 2 | 10.541 | 5.2705 |
| add | 4 | 9.791 | 4.8955 |
| add | 8 | 11.291 | 5.6455 |
| mnist | 1 | 10.833 | 1.0 |
| mnist | 2 | 23.458 | 2.1654204744761376 |
| mnist | 4 | 18.875 | 1.7423613034247207 |
| mnist | 8 | 21.375 | 1.9731376350041538 |

The path is not monotone. Add P99 at four workers is below Add P99 at two workers, and MNIST P99 at four workers is below MNIST P99 at two workers. Each of those values is still above its one-worker control. Add's highest median run P99 in this matrix is at eight workers. MNIST's highest is at two workers.

Each P99 is one order statistic from 1000 samples in one process. The published median run P99 is the median of five such statistics. It is not a pooled P99 across processes, and no confidence interval was calculated.

## 10. Run-to-run variability

Relative aggregate-throughput range is `(maximum - minimum) / median` across the five counted runs.

MNIST at one worker is the tightest cell, with relative throughput range 0.04213274495333564 and relative run-mean range 0.041596606837839306. Add at four workers is the widest, with relative throughput range 0.44252237800501937. Its five aggregate rates span 487472.92087924515 to 747390.0206242276. MNIST at four workers is the widest MNIST cell, with relative throughput range 0.29990895083715413.

The other relative throughput ranges are Add 0.12797900166516518, 0.09731638448947107, and 0.2973705481083145 at 1, 2, and 8 workers, and MNIST 0.07409562866849889 and 0.11325592509451522 at 2 and 8 workers.

These ranges describe the five observed runs. They are not confidence intervals.

## 11. Two-worker throughput regressions

Both workloads showed lower median aggregate throughput at two workers than at one worker.

Add fell from 592900.0222337508 to 273552.6058074124 completed requests per second. MNIST fell from 132147.68666884885 to 123198.2259455464. The Add drop is the larger of the two. At the same point, median run-mean latency rose from 1.614209 to 7.108745 microseconds for Add and from 7.500732 to 16.022337 microseconds for MNIST.

Possible causes, stated as hypotheses, include the shared wall-clock window covering worker start and join, interference between two sessions on one machine, and the short Add call making coordination a larger fraction of that window. This campaign did not isolate scheduling, cache behavior, thermal behavior, or CPU contention. None of those mechanisms is established by these results.

Four workers recover Add aggregate throughput to about the one-worker median, and they raise MNIST aggregate throughput above the one-worker median. Eight workers are higher still for both workloads. The two-worker dip is a measured feature of this matrix, not a fitted curve.

## 12. Experimental limitations

- One interactive macOS machine, hardware model Mac17,8, with 18 physical and 18 logical CPUs recorded by `sysctl`. The machine was not reserved as a quiet benchmark host.
- Five counted runs per cell. No confidence interval was calculated, and the results are not a claim of statistical significance.
- Median run P99 is not pooled across processes.
- Worker counts stop at 8. The rise from 4 to 8 does not establish what a larger worker count would do, and it does not establish saturation.
- Aggregate throughput is completed requests divided by shared wall-clock time. It is not a production server capacity, and this binary is not a server.
- The harness gives each worker its own engine. It does not measure a shared session, a queue, or a batch.
- Add and MNIST are different workloads: a two-element add versus a 784-input MLP on fixture `test_00003`.
- The sequential baseline uses a different binary, a different measurement definition, and a different campaign. It is not the denominator of these ratios.

## 13. Reproduction instructions

The recorded campaign used measurement commit `c07da912da109c09b6e3ea6f9ae8f1a89e97a79e` and binary SHA-256 `614564d20a12d298d0cf453cfe447e32b5645acaf9725bbe6dc3958e7cf37baf`. A new collection must pass the full commit that built its own Release binary. It writes a new untracked directory under `benchmarks/concurrent_runs/` and does not replace this run set.

From the repository root, after a Release configure of `inferedge-bench-concurrent`:

```bash
python3 -m benchmarks.collect_concurrent \
  --binary build-concurrent/inferedge-bench-concurrent \
  --expected-commit c07da912da109c09b6e3ea6f9ae8f1a89e97a79e
```

`--preflight-only` checks the binary, models, fixtures, and repository without launching the 48 processes. After a completed collection, reduce that new private directory with:

```bash
python3 -m benchmarks.analyze_concurrent \
  --run-set benchmarks/concurrent_runs/<run-set-id>
```

Do not pass `benchmarks/published/20261012T032311Z-c07da91/` to the analyzer. That directory is a sanitized derivative. It already contains `aggregate.json`, and analysis of this campaign was finished on the original private run set before any path was rewritten.

## 14. Provenance and data integrity

| Fact | Value |
| --- | --- |
| Run-set id | `20261012T032311Z-c07da91` |
| Original `environment.json` SHA-256 | `330a87092ddf31957dc71b14a20e590e16d18aba746fc67583e92289e942d195` |
| Original `aggregate.json` SHA-256 | `767a85f9e6d49b23e5ba8f2950ae4f7cb9111b39476d83731c77b7901dace7f8` |
| Published `environment.json` SHA-256 | `eaa6441168ad88c3b1c06ca9820a832ec051b70f10264e9c240168b273ecaa0c` |
| Published `aggregate.json` SHA-256 | `4ca76583e18a405e40091bfb2e693ab601f3f23013cfca652fb6e4ff66e0aafe` |

The published digests differ from the original digests because approved path fields were rewritten from repository-absolute paths to repository-relative paths. No numeric measurement was changed. The field list, the published file hashes, and the reason the sanitized hashes differ are in [`published/20261012T032311Z-c07da91/PROVENANCE.md`](published/20261012T032311Z-c07da91/PROVENANCE.md).

The private run set remains the authoritative record and stays untracked. This publication is a sanitized derivative, not a new experiment.

## 15. Relationship to sequential baseline

The sequential CPU baseline is a separate campaign, run set `20261011T234924Z-8b26105`, measured with `inferedge-bench` at commit `8b26105f5971035a38af6227bc34ca11589327f7`. Its note is [`BASELINE_RESULTS.md`](BASELINE_RESULTS.md). That campaign reports median active-call throughput of 644814.3386074847 Add requests per second and 135500.62059284232 MNIST requests per second on one calling thread.

Those active-call rates are not the control for this concurrent campaign. Concurrent ratios divide by the concurrent worker-count-one aggregate throughput of the same workload: 592900.0222337508 for Add and 132147.68666884885 for MNIST. The two campaigns use different binaries, different dates, and different throughput definitions. This note does not treat either sequential rate as a concurrency speedup.
