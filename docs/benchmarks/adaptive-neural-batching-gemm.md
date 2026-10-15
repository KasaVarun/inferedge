# Adaptive neural batching benchmark: gemm_f32 A / G2 / G3

Curated public report for one controlled three-arm study of adaptive
early-seal batching on InferEdge's neural scheduler path.

Machine-readable companion:
[`adaptive-neural-batching-gemm-results.json`](adaptive-neural-batching-gemm-results.json).

This document is a reviewed aggregate extract. It is not a dump of raw
trial stdout/stderr. The local raw run directory used as source evidence
is **not** published on GitHub.

Public results format version **2** in the companion JSON is distinct from
runner report schema version **3** and analysis protocol version **2**.

## 1. Executive summary

Under a frozen 1000 RPS open-loop GEMM workload on one Apple Silicon host,
adaptive cell **G3** (batching enabled, max batch 4, delay 2 ms,
`adaptive_batching=true`) sealed mean size-two batches. Relative to fixed
configured-batching cell **G2** (same cap and delay, adaptive off),
producer-observed P50 end-to-end latency was lower by a paired mean of
**−269.6456 µs** (95% CI **[−284.90826375, −257.67249] µs**, 20 three-arm
blocks, seed `20261013`, 10 000 replicates).

Exploratory G3-minus-A producer-observed P50 remained positive
(**+303.4292 µs**; exploratory 95% CI
**[+222.7774550000001, +394.21523125000004] µs**), labeled
**EXPLORATORY; UNADJUSTED FOR MULTIPLE COMPARISONS**. This report does
**not** claim that G3 beats A. Successful throughput under this fixed
offered rate was similar across arms when measured per total-trial second.
No automatic winner or universal adaptive ranking is declared.

## 2. Research question

For this specific frozen contrast, does adaptive early-seal batching (G3)
change producer-observed P50 latency relative to fixed-delay configured
batching (G2) at 1000 RPS on `gemm_f32`, while both arms form multi-request
batches?

Contrast id: `gemm_f32_A_G2_G3_adaptive_rps1000`

Primary sign convention: `G3_minus_G2` (`P50_G3 - P50_G2`)

Exploratory sign convention: `G3_minus_A` (`P50_G3 - P50_A`)

## 3. Scheduling configurations

| Item | Value |
|------|-------|
| Model | `gemm_f32` (features `[batch,512]` → product `[batch,512]`) |
| Workload | `open_loop_constant` |
| Offered rate | 1000 RPS (frozen) |
| Workers | 1 |
| Duration / warmup | 1000 ms / 100 requests |
| Runner schema / analysis protocol | 3 / 2 |
| Reference policy | `precomputed_reference_compare` |

| Cell | Batching | Max batch size | Max batch delay | Adaptive |
|------|----------|----------------|-----------------|----------|
| A | disabled | 1 | 0 ms | false |
| G2 (primary baseline) | enabled | 4 | 2 ms | false |
| G3 (primary alternative) | enabled | 4 | 2 ms | true |

Each cell definition records exactly five identity keys:
`id`, `enable_batching`, `max_batch_size`, `max_batch_delay_ms`, and
`adaptive_batching`. Separately, G3 uses an internal adaptive minimum
target batch of **2** (C++ constant; not a CLI flag and not one of the five
identity keys).

## 4. Hardware and software environment

| Field | Value |
|-------|-------|
| Host | Darwin arm64, hardware model Mac17,8 |
| Compiler | Apple clang 21.0.0 |
| ONNX Runtime | 1.31.0 CPU |
| Build | Release |

## 5. Artifact provenance

| Field | Value |
|-------|-------|
| Collector / analyzer Git commit | `0b2b65d819c8d8b3e39fef9043af962bfed865ac` |
| Binary source Git commit | `c35381c5943330d145e0b1d164368c0abb587266` |
| Executable SHA-256 | `956e5a7ee470c55686addb9ca84c5f438ea33845819a7a7284f40b24db4a94ba` |
| ONNX SHA-256 | `a6203020b870eb04f9fd8e836cf4ccd930a57156a026d2a50163121d83debf32` |
| Sidecar SHA-256 | `a335c8b12defcb885f8bada9ef1092655d024d5586fe011a9336ad2d472f82b6` |

The measured Release `inferedge-bench-scheduler` was built from `c35381c`.
Collection and analysis tooling for this campaign came from `0b2b65d`.
The executable SHA-256 is historical provenance for that binary; a rebuild
may differ.

Source evidence identifier (local, unpublished):
`neural-gemm-f32-20261015T002418Z-0b2b65d`.

## 6. Experimental methodology

- Primary metric: producer-observed P50 successful end-to-end latency
  (submit start → producer observes completion).
- This endpoint includes completion-observation delay. It is **not** pure
  model inference latency.
- Experimental unit: one complete three-arm block (one A, one G2, one G3
  trial).
- Design: three uncounted pilots (A, G2, G3), then 20 counted blocks with
  within-profile cyclic rotation (block 1: A, G2, G3; block 2: G2, G3, A;
  block 3: G3, A, G2; repeat).
- Position counts across 20 counted blocks: A `{0:7, 1:6, 2:7}`,
  G2 `{0:7, 1:7, 2:6}`, G3 `{0:6, 1:7, 2:7}`. Perfect three-position
  balance is impossible with 20 blocks; these counts are disclosed, not
  claimed perfect.
- Primary statistics: paired-block bootstrap mean of `G3 − G2` differences;
  10 000 replicates; seed `20261013`; 95% percentile interval.
- Requests are not resampled as independent experimental units.
- Pilots are excluded from statistics.

## 7. Correctness and validity

| Check | Result |
|-------|--------|
| Pilots / counted trials / blocks | 3 / 60 / 20 |
| Counted successful requests | 60 000 |
| Correctness mismatches | 0 |
| Generator skips | 0 |
| Admission rejections | 0 |
| Inference failures / timeouts / cancellations | 0 |
| Delivery class (all counted trials) | `STRICT_PRIMARY` |
| Retries / selective exclusions | none |
| Adaptive JSON (all trials) | A=false, G2=false, G3=true |
| G2 mean realized batch size ≥ 2 (every counted G2 trial) | yes |
| Arrival-screen max lag ≤ 5 000 000 ns (every counted trial) | yes |

## 8. Primary G3-minus-G2 result

| Quantity | Value |
|----------|-------|
| A P50 mean across 20 blocks | 942.0667 µs |
| G2 P50 mean across 20 blocks | 1515.1415 µs |
| G3 P50 mean across 20 blocks | 1245.4959 µs |
| Mean paired effect (G3 − G2) | **−269.6456 µs** |
| 95% CI | **[−284.90826375, −257.67249] µs** |

The interval excludes zero under this protocol. That supports a directional
paired difference for this host/workload/contrast only. It does **not**
authorize a universal adaptive-batching superiority claim.

Exact per-block P50 values and differences are listed in
[`adaptive-neural-batching-gemm-results.json`](adaptive-neural-batching-gemm-results.json)
under `three_arm_blocks`.

## 9. Exploratory G3-minus-A result

| Quantity | Value |
|----------|-------|
| Mean paired effect (G3 − A) | **+303.4292 µs** |
| Exploratory 95% CI | **[+222.7774550000001, +394.21523125000004] µs** |
| Label | **EXPLORATORY; UNADJUSTED FOR MULTIPLE COMPARISONS** |

G3 remained above A on this exploratory producer-observed P50 contrast.
Do **not** claim that G3 beats A. This contrast is not a co-primary
hypothesis.

## 10. Actual batch formation

Mean realized batch size is computed from sealed histogram bins:

`sum(size × count) / sum(count)` for `size ≥ 1`,
with weighted sum matching `requests_entered_engine_runs` and sealed count
matching `engine_run_count`.

| Cell | Configured max batch | Mean realized batch | Multi-request fraction | Mean engine runs |
|------|----------------------|---------------------|------------------------|------------------|
| A | 1 | 1.0 | 0.0 | 1000 |
| G2 | 4 | 2.967455363504896 | 0.9996 | 337 |
| G3 | 4 | 2.0 | 1.0 | 500 |

- A: every counted trial sealed as size-1 only.
- G2: mean realized batch ranged **2.9239766081871346–2.9940119760479043**
  across counted blocks.
- G3: every counted trial sealed exactly **500** size-2 batches
  (nonzero histogram `{2: 500}`).

Configured `max_batch_size` is not used as the observed batch size.
Sealed-batch histograms do not identify individual adaptive early-seal
events.

## 11. Engine-run trade-offs

Across counted blocks, mean aggregate engine-run nanoseconds per success
were approximately 18036 (A), 16417 (G2), and 22542 (G3). Mean engine-run
counts were 1000 (A), 337 (G2), and 500 (G3). These are descriptive
aggregates, not isolated model-execution latency and not isolated queue
wait.

Successful throughput means (successes per **total-trial** second):
A ≈ 999.820, G2 ≈ 999.709, G3 ≈ 1000.065. Arrival-window throughput means
are separately about 1000.42–1000.46 successes per arrival-window second
and must not be mixed with the total-trial-second denominator.

## 12. Arrival-fidelity limitations

- `arrival_fidelity_status` = `not_certified`
- `phase_boundary_fidelity` = `unavailable`
- Coarse maximum generator-lag screen: **5 000 000 ns**

The screen is a coarse lateness bound. It does not certify precise
request-arrival timing. Producer-observed latency includes
completion-observation effects. No isolated queue-wait timer was
collected. This is a single-host, single-model, single-rate study.

## 13. Reproduction

Tooling commit:
`0b2b65d819c8d8b3e39fef9043af962bfed865ac`.

Binary source commit (historical build identity):
`c35381c5943330d145e0b1d164368c0abb587266`.

Collector (exact released CLI; substitute a Release
`inferedge-bench-scheduler` binary and repo root):

```bash
python3 -m benchmarks.collect_neural_scheduler \
  --binary <release-binary> \
  --repo-root <repo-root> \
  --expected-commit 0b2b65d819c8d8b3e39fef9043af962bfed865ac \
  --mode reportable \
  --model-id gemm_f32 \
  --cells A,G2,G3 \
  --baseline-cell G2 \
  --alternative-cell G3 \
  --target-rps 1000 \
  --offered-rate-frozen \
  --counted-runs 20 \
  --duration-ms 1000 \
  --warmup-requests 100 \
  --max-offered 10000 \
  --lag-tolerance-ms 1000 \
  --inter-trial-delay-seconds 0
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

Expected inventory for a successful reportable campaign: 3 pilots +
60 counted trials (20 three-arm blocks), runner schema 3, analysis
protocol 2, STRICT_PRIMARY delivery, zero correctness mismatches,
G2 mean realized batch ≥ 2, G3 adaptive identity true, and arrival screen
≤ 5 000 000 ns on counted trials.

## 14. Relationship to prior A/G2 study

A separate prior campaign published configured batching A versus G2 at the
same rate and model:

- [`neural-batching-gemm.md`](neural-batching-gemm.md)
- [`neural-batching-gemm-results.json`](neural-batching-gemm-results.json)

That contrast id is `gemm_f32_A_vs_G2_delay2ms_rps1000`. The two campaigns
must not be merged into one paired statistical estimate. Block-level
effects from the older study are not combined with this three-arm study.

## 15. Future research hypotheses (unvalidated)

Separately reviewable work could study arrival-rate-aware scheduling,
adaptive waiting budgets, multi-workload evaluation, saturated-throughput
behavior, and finer timing that separates queue wait, compute, and
observation delay. None of those hypotheses are validated by this report.

## 16. Claim boundaries

Allowed: under these tested conditions, G3 formed mean size-two sealed
batches; G2 formed multi-request batches with mean realized batch size
approximately 2.967; producer-observed P50 was lower for G3 than G2 by the
reported primary paired effect; the primary P50 CI excludes zero;
exploratory G3-minus-A P50 remained positive; successful throughput under
this fixed offered rate was similar per total-trial second.

Not claimed: universal adaptive superiority; that G3 beats A; pure
inference latency; certified arrival precision; isolated queue-wait
reduction; saturated-throughput improvement; production LLM behavior; an
automatic winner or superiority declaration.
