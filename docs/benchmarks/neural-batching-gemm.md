# Neural batching benchmark: gemm_f32 A versus G2

Curated public report for one controlled paired study of configured
batching on InferEdge's neural scheduler path.

Machine-readable companion:
[`neural-batching-gemm-results.json`](neural-batching-gemm-results.json).

This document is a reviewed aggregate extract. It is not a dump of raw
trial stdout/stderr. The local raw run directory used as source evidence
is **not** published on GitHub.

## 1. Executive summary

Under a frozen 1000 RPS open-loop GEMM workload on one Apple Silicon host,
cell **G2** (batching enabled, max batch 4, delay 2 ms) formed real
multi-request batches (mean realized batch size **2.967**). Relative to
unbatched cell **A**, producer-observed P50 end-to-end latency was higher
by a paired mean of **+553.1688 µs** (95% CI **[+483.19375, +640.06988] µs**,
20 paired blocks, seed `20261013`, 10 000 replicates). Successful throughput
was similar between arms (~999.7 successes per total-trial second). No
automatic winner or universal batching ranking is declared.

## 2. Question being tested

For this specific frozen contrast, does configured batching with a 2 ms
maximum batch delay change producer-observed P50 latency relative to an
unbatched baseline, while still forming multi-request batches?

Contrast id: `gemm_f32_A_vs_G2_delay2ms_rps1000`

Sign convention: `G2_minus_A` (`P50_G2 - P50_A`)

## 3. Model and configurations

| Item | Value |
|------|-------|
| Model | `gemm_f32` (features `[batch,512]` → product `[batch,512]`) |
| Workload | `open_loop_constant` |
| Offered rate | 1000 RPS (frozen) |
| Duration / warmup | 1000 ms / 100 requests |
| Schema / protocol | report schema 3 / analysis protocol 2 |
| Reference policy | `precomputed_reference_compare` |

| Cell | Batching | Max batch size | Max batch delay |
|------|----------|----------------|-----------------|
| A (baseline) | disabled | 1 | 0 ms |
| G2 (alternative) | enabled | 4 | 2 ms |

## 4. Environment and artifacts

| Field | Value |
|-------|-------|
| Git commit | `0a8d52b5a9f176a38b65a8b8a2028b87857a1b97` |
| Host | Darwin arm64, hardware model Mac17,8 |
| Compiler | Apple clang 21.0.0 |
| ONNX Runtime | 1.31.0 CPU |
| Build | Release |
| Executable SHA-256 | `acb0af726be4c564e3b829836723946916257d2f92ccce39e07acf50f70fc0d0` |
| ONNX SHA-256 | `a6203020b870eb04f9fd8e836cf4ccd930a57156a026d2a50163121d83debf32` |
| Sidecar SHA-256 | `a335c8b12defcb885f8bada9ef1092655d024d5586fe011a9336ad2d472f82b6` |

Source evidence identifier (local, unpublished):
`neural-gemm-f32-20261014T053518Z-0a8d52b`.

## 5. Measurement methodology

- Primary metric: producer-observed P50 successful end-to-end latency
  (submit start → producer observes completion).
- This endpoint includes completion-observation delay. It is **not** pure
  model inference latency.
- Experimental unit: one paired block (one A trial and one G2 trial).
- Design: one uncounted pilot pair (A then G2), then 20 counted blocks with
  within-profile cyclic rotation (block 1: A then G2; block 2: G2 then A;
  alternate thereafter).
- Statistics: paired-block bootstrap mean of `G2 − A` differences;
  10 000 replicates; seed `20261013`; 95% percentile interval.
- Requests are not resampled as independent experimental units.

## 6. Trial validity

| Check | Result |
|-------|--------|
| Pilots / counted trials / blocks | 2 / 40 / 20 |
| Counted successful requests | 40 000 |
| Correctness mismatches | 0 |
| Generator skips | 0 |
| Admission rejections | 0 |
| Inference failures / timeouts / cancellations | 0 |
| Delivery class (all counted trials) | `STRICT_PRIMARY` |
| Retries / selective exclusions | none |
| G2 mean realized batch size ≥ 2 (every counted G2 trial) | yes |
| Arrival-screen max lag ≤ 5 000 000 ns (every counted trial) | yes |

## 7. Primary statistical finding

| Quantity | Value |
|----------|-------|
| A P50 mean across 20 blocks | 959.26875 µs |
| G2 P50 mean across 20 blocks | 1512.43755 µs |
| Mean paired effect (G2 − A) | **+553.1688 µs** |
| 95% CI | **[+483.19375, +640.06988] µs** |

The interval excludes zero under this protocol. That supports a directional
paired difference for this host/workload/contrast only. It does **not**
authorize a universal “batching is slower” claim.

Per-block P50 pairs and differences are listed in
[`neural-batching-gemm-results.json`](neural-batching-gemm-results.json)
under `paired_blocks`.

## 8. Secondary observations

| Metric | A (mean across blocks) | G2 (mean across blocks) | Paired effect G2 − A |
|--------|------------------------|-------------------------|----------------------|
| Producer-observed P95 | 1278.8104 µs | 2543.62495 µs | +1264.81455 µs |
| Successful throughput | 999.672447… /s | 999.697083… /s | +0.024635… /s |
| Mean realized batch size | 1.0 | 2.967008588540034 | +1.967008588540034 |
| Multi-request fraction (≥2) | 0.0 | 0.99965 | — |
| Engine-run ns / success | 16170.72685 | 17113.97105 | +943.2442 |

Engine-run totals are aggregate descriptive quantities, not isolated
compute latency. Throughput CIs for the paired effect include zero; arm
means are nearly equal.

Generator lag remained under the 5 ms coarse screen on every counted
trial (A max lag range 521 292–1 224 917 ns; G2 798 250–1 529 208 ns).
Arrival fidelity remains `not_certified`.

## 9. Actual batch-formation evidence

Mean realized batch size is computed from sealed histogram bins:

`sum(size × count) / sum(count)` for `size ≥ 1`,
with weighted sum matching `requests_entered_engine_runs` and sealed count
matching `engine_run_count`.

- A: every counted trial sealed as size-1 only.
- G2: mean realized batch **2.9239766…–2.9940120…** across blocks
  (overall mean **2.967008588540034**); multi-request fraction ≈ **0.99965**.

Example nonzero sealed histograms:

| Block | A | G2 |
|------|---|----|
| 1 | size1×1000 | size2×5, size3×330 |
| 10 | size1×1000 | size1×1, size2×15, size3×323 |
| 20 | size1×1000 | size1×1, size2×12, size3×325 |

Configured `max_batch_size = 4` is not used as observed batch size.

## 10. Limitations

- Arrival fidelity was **not** certified; `phase_boundary_fidelity` is
  `unavailable`.
- The 5 000 000 ns maximum-lag screen is a coarse lateness bound, not
  precise scheduling certification.
- Producer-observed latency includes completion-observation effects.
- G2 intentionally permits up to 2 ms of batch-formation wait.
- Queue waiting was not separately instrumented.
- One Apple Silicon host only.
- CPU GEMM (`gemm_f32`), not a production LLM.
- Only this frozen 1000 RPS A/G2 contrast was evaluated in the paired study.
- Do not generalize to all workloads, saturation regimes, or architectures.
- Bootstrap bit-identity is not promised across all software environments.

## 11. Reproduction

Released tooling commit:
[`0a8d52b5a9f176a38b65a8b8a2028b87857a1b97`](https://github.com/KasaVarun/inferedge/commit/0a8d52b5a9f176a38b65a8b8a2028b87857a1b97).

Collector (exact released CLI; substitute a Release
`inferedge-bench-scheduler` binary and repo root):

```bash
python3 -m benchmarks.collect_neural_scheduler \
  --binary <release-binary> \
  --repo-root <repo-root> \
  --expected-commit 0a8d52b5a9f176a38b65a8b8a2028b87857a1b97 \
  --mode reportable \
  --model-id gemm_f32 \
  --cells A,G2 \
  --baseline-cell A \
  --alternative-cell G2 \
  --target-rps 1000 \
  --offered-rate-frozen \
  --counted-runs 20 \
  --duration-ms 1000 \
  --warmup-requests 100 \
  --lag-tolerance-ms 1000 \
  --inter-trial-delay-seconds 0 \
  --max-offered 10000
```

Analyzer:

```bash
python3 -m benchmarks.analyze_neural_scheduler \
  <new-run-directory>
```

Notes:

- The recorded executable SHA-256 identifies the binary used for the
  published provenance. A rebuilt binary must be identified separately.
- Different hosts may produce different latency magnitudes.
- Do not expect bit-identical timing across machines.
- Model regeneration, if needed, follows the repository's pinned
  `onnx==1.22.0` generators and hash pins; normal reproduction should use
  the committed ONNX and sidecar artifacts.

Expected inventory for a successful reportable campaign: 2 pilots +
40 counted trials (20 paired blocks), schema 3, protocol 2, STRICT_PRIMARY
delivery, zero correctness mismatches, G2 mean realized batch ≥ 2, and
arrival screen ≤ 5 000 000 ns on counted trials.

## 12. Future hypotheses (unvalidated)

Separately reviewable work could study adaptive or load-aware batch delay,
reducing unnecessary wait under light load, saturated-throughput and tail
behavior, and finer timing that separates queue wait, compute, and
observation delay. None of those hypotheses are validated by this report.

## Claim policy

Allowed: under these tested conditions, G2 batched for real; producer-observed
P50 was higher for G2 by the reported paired effect; the P50 CI excludes
zero; throughput was similar.

Not claimed: universal batching inferiority or superiority; pure inference
latency; certified arrival precision; isolated queueing delay; production
LLM behavior; a proven root cause beyond the intentional batch-wait design.
