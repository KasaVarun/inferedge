# Qwen3-0.6B generative-overhead benchmark: no directional winner under the frozen C01 protocol

Curated public report for one official protocol-valid generative-overhead
campaign. InferEdge integration is compared with a pinned direct llama.cpp
comparator on one frozen Qwen3-0.6B Q4_K_M package and one tested Apple
Silicon host.

Machine-readable companion:
[`generative-overhead-qwen3-0.6b-results.json`](generative-overhead-qwen3-0.6b-results.json).

This is a curated publication extract derived from independently accepted
official campaign evidence. The raw 922-file evidence root is preserved
separately and is not part of the GitHub publication package.

## 2. Publication status

Curated publication extract from independently accepted official campaign
evidence. The raw 922-file evidence root is preserved separately and is not
part of the GitHub publication package.

Evidence class: `official_campaign`.
Campaign ID: `phase-9-b3d-campaign-001`.
Source commit: `f91cb463fd4026a3dd05af4b0ac7c6f9b3182955`.
Contract: `generative-overhead-r7-inventory-v1`.
`raw_evidence_published_on_github`: `false`.

## 3. Executive summary

In one protocol-valid 20-pair C01 campaign on the frozen Qwen3-0.6B package and tested Apple host, neither implementation satisfied all prespecified practical-threshold and strict 95% bootstrap-interval criteria for a directional winner.

The signed paired-sum `sum_d` is `-3,472,501 ns`. The protocol signed-direction
label for that sign is `inferedge_faster`. That label describes the sign of
`sum_d` only. It is not a winner and is not a superiority result.
`winner_declared` is `false`. `superiority_claimed` is `false`.
`claim_status` is `no_winner_below_threshold_or_ci`. The result does not
establish practical equivalence.

## 4. Research question

For this frozen C01 configuration, does InferEdge integration overhead versus
the pinned direct llama.cpp comparator satisfy every prespecified directional
winner rule: eligible 20-pair campaign, signed direction, exact 20-pair
absolute threshold, integer relative threshold, and a 95% bootstrap interval
that strictly excludes zero in that direction?

## 5. Scope and claim boundary

This package reports one official campaign, one frozen Qwen3-0.6B Q4_K_M
package, one tested Apple Silicon host, and one primary C01 endpoint with 20
paired observations. Secondary cells C02–C11 and L01 are descriptive only.
They do not enter, rescue, average with, or override C01.

The package is not a model-quality, training, online-learning, GPU,
concurrency, energy, Llama-family, other-Qwen, or production-wide latency
result. It is not a production-readiness claim.

## 6. System under test

| Item | Value |
|------|-------|
| InferEdge worker SHA-256 | `1820a52aaceb56477f7c329342e7438224ea02eaf466d582fab622d60d959db9` |
| Reference worker SHA-256 | `44777c42fe2a40d2f44ac554a6da01f406c76550ac0d53d6245981845d8b2b91` |
| Model ID | `qwen3-0.6b` |
| GGUF filename | `qwen3-0.6b-Q4_K_M.gguf` |
| Quantization | `Q4_K_M` |
| GGUF SHA-256 | `3f384c3796ca2547c5547be6bca3d487a18807cc1a2c259de6e0e1fa433bb76d` |
| GGUF size | `484220192` bytes |
| llama.cpp | tag `v0.7.0`, commit `620d2225cc436ab0b001e3200e590e3b286ef9ab` |
| Backend | CPU/native as recorded |
| Context / batch | `n_ctx=4096`, `n_batch=4096`, `n_ubatch=512`, `n_seq_max=1` |
| Host | Darwin arm64, hardware model `Mac17,8` |
| Compiler | Apple clang version 21.0.0 (clang-2100.3.34.2) |
| Build | Release |

## 7. Primary C01 endpoint

| Item | Value |
|------|-------|
| Cell | `C01` |
| Role | `primary` |
| Fixture | `P_tiny` |
| Operation | `warm_request` |
| `max_new_tokens` | `1` |
| Threads | `1` |
| Pair count | `20` |
| Difference definition | InferEdge duration minus reference duration |

## 8. Experimental design

Each counted pair has one InferEdge observation and one reference
observation. Pair order alternates: even pair IDs `00`–`18` are
`ie_then_ref`; odd pair IDs `01`–`19` are `ref_then_ie`. The estimator is
`mean_paired_difference` on the 20 integer differences. Warm-up records are
not published here. The complete 12-cell matrix was protocol-valid; only C01
is the claim endpoint.

## 9. Environment and provenance

Campaign evidence interval (UTC): `2026-10-18T16:35:52.665569Z` through
`2026-10-18T17:22:57.891498Z`. Those times are the official campaign-start
and campaign-end evidence checkpoints.

Collector process boundary (UTC): `2026-10-18T16:35:50.535790Z` through
`2026-10-18T17:22:58.079740Z`. The collector process boundaries enclose the
campaign evidence interval; they are not the campaign-start or campaign-end
checkpoints.

Source commit `f91cb463fd4026a3dd05af4b0ac7c6f9b3182955`. Accepted isolated
analysis of the official inventory reproduced the published aggregate and
summary byte-for-byte. Source-tree CI for that commit:
<https://github.com/KasaVarun/inferedge/actions/runs/39143884593>.

Host description is limited to the recorded Apple hardware model and Darwin
arm64. Local user paths, process IDs, and raw environment captures are not
published.

## 10. Correctness and evidence acceptance

Phase 9-B3E-S independently accepted the official 922-file inventory,
complete matrix, 48 warm-ups, 480 measured records, matching generated
tokens, timing records, and environment gates. Phase 9-B3E-C
independently reproduced C01 integer arithmetic, thresholds, and the frozen
bootstrap. Both audits returned PASS. Rejection codes are `[]`.
`complete_matrix` is `true`.

## 11. Primary C01 result

Exact accepted nanosecond values. Millisecond figures in parentheses are
display conversions only.

| Quantity | Exact value | Display conversion |
|----------|-------------|--------------------|
| Pair count | `20` | — |
| `sum_ie` | `1,490,295,957 ns` | `1490.295957 ms` |
| `sum_ref` | `1,493,768,458 ns` | `1493.768458 ms` |
| `sum_d` | `-3,472,501 ns` | `-3.472501 ms` |
| InferEdge mean | `74,514,797.85 ns` | `74.51479785 ms` |
| Reference mean | `74,688,422.9 ns` | `74.6884229 ms` |
| Mean paired difference | `-173,625.05 ns` | `-0.17362505 ms` |
| Protocol signed-direction label | `inferedge_faster` | sign of `sum_d` only; not a winner |

Descriptive difference statistics (do not affect the winner decision):

| Statistic | Exact value |
|-----------|-------------|
| P25 difference | `-1,591,001 ns` |
| P50 / median difference | `-407,708 ns` |
| P75 difference | `575,333 ns` |
| P95 difference | `3,066,958 ns` |
| IQR | `2,166,334 ns` |
| MAD | `983,041 ns` |

The 20 curated pair durations appear in the companion JSON. Even pairs are
`ie_then_ref`. Odd pairs are `ref_then_ie`. Differences are InferEdge minus
reference.

## 12. Practical-threshold gates

Both frozen gates are required. Eligibility uses integer sums, not rounded
display values.

- Per-pair display absolute threshold: `2,000,000 ns` (`2.0 ms` display).
- Exact 20-pair absolute threshold: `40,000,000 ns` (`40.0 ms` display).
- Applicable faster magnitude: `-sum_d = 3,472,501 ns`.
- Absolute threshold met: `false` (`3,472,501 < 40,000,000`).
- Relative threshold: `5 / 100` (`5%` display). Integer predicate
  `20 * (-sum_d) >= sum_ref` is `69,450,020 >= 1,493,768,458`, which is
  false.
- Relative threshold met: `false`.

## 13. Bootstrap interval

Independent standard-library bootstrap: 10,000 replicates, seed `20261018`,
arithmetic-mean statistic, interpolated quantile at the frozen 95% tails.

- `ci_low_ns`: `-1,060,655.6224999994 ns` (`-1.0606556224999994 ms` display)
- `ci_high_ns`: `730,089.1224999995 ns` (`0.7300891224999995 ms` display)
- Either endpoint equal to zero: no
- CI strictly excludes zero in the observed signed direction: `false`
  (faster direction requires `ci_high_ns < 0`)

## 14. Claim decision

A directional InferEdge-faster winner would require an eligible campaign,
signed direction `inferedge_faster`, both thresholds met, and
`ci_high_ns < 0`. The absolute gate, relative gate, and strict CI gate all
fail. Therefore:

- `winner_declared`: `false`
- `superiority_claimed`: `false`
- `practical_equivalence_claimed`: `false`
- `claim_status`: `no_winner_below_threshold_or_ci`

No-winner means only that the frozen combined directional criteria were not
all satisfied. It does not establish equal runtimes, a zero effect, or that
either implementation won.

## 15. Secondary cells

C02–C11 are descriptive secondary warm-request cells. L01 is descriptive
filesystem-cache-warm package/session load versus a pinned direct load.
None of those cells enters the primary C01 sums, bootstrap, thresholds,
winner decision, or claim status. This initial package does not publish a
secondary-cell ranking and does not select a secondary cell as favorable.

## 16. Limitations

- one host
- one official campaign
- one frozen Qwen3-0.6B Q4_K_M package
- one primary prompt/token/thread configuration (`P_tiny`, `max_new_tokens=1`,
  `threads=1`)
- 20 paired C01 observations
- CPU/native backend configuration as recorded
- no GPU conclusion
- no concurrency conclusion
- no energy conclusion
- no model-quality conclusion
- no training or continual-learning conclusion
- no generalization to other Qwen or Llama models
- no production-wide latency conclusion
- no practical-equivalence conclusion
- raw evidence is preserved locally but is not included in the GitHub
  publication package

## 17. Reproduction and source references

- Source commit: `f91cb463fd4026a3dd05af4b0ac7c6f9b3182955`
- Protocol document: [`../phase-9-b3-generative-overhead-protocol.md`](../phase-9-b3-generative-overhead-protocol.md)
- Collector: `benchmarks/collect_generative.py`
- Analyzer: `benchmarks/analyze_generative.py`
- CI run: <https://github.com/KasaVarun/inferedge/actions/runs/39143884593>
- Official raw evidence root is local and unpublished

## 18. Artifact identities

| Artifact | SHA-256 |
|----------|---------|
| Campaign manifest | `273fc869d45c26118297c36bdc5a37b7e21ff4845b231845af22d43f0cdef371` |
| Aggregate | `93da8a1fa8c97fc12e7c20079210f5484c1a151a8532fdd8450c3cfaa1ed9887` |
| Summary | `a4d020038e79b13ae4c3aa7cf0463cec0602d19c0010083fed4d850bab0eac48` |
| Protocol | `45e54f910d1c64df509d215150475f2ba40dccc1e3bc7f292cfcc5f8cd203651` |
| Collector | `8e289ebadf3f1323cb62ff4cf2eaa379c03bffa89dca214915d702b35edddab7` |
| Analyzer | `deee6b9bfcdc5f5d66845256e60b049deffd07c2ff7dd42d1de42089ef05ff1a` |
| Fixtures | `f4f0ef2640c8f459a8b871e92720053bb2227bf20a210a98e99f98b024cbd61c` |
| GGUF | `3f384c3796ca2547c5547be6bca3d487a18807cc1a2c259de6e0e1fa433bb76d` |
| InferEdge worker | `1820a52aaceb56477f7c329342e7438224ea02eaf466d582fab622d60d959db9` |
| Reference worker | `44777c42fe2a40d2f44ac554a6da01f406c76550ac0d53d6245981845d8b2b91` |

## 19. Machine-readable companion

The companion JSON
[`generative-overhead-qwen3-0.6b-results.json`](generative-overhead-qwen3-0.6b-results.json)
carries publication format version `1`, study ID
`generative_overhead_qwen3_0_6b_c01`, curated C01 pair durations, exact
nanosecond statistics, claim-policy fields, provenance hashes, and the
approved publication sentence. `raw_evidence_published_on_github` is
`false`.

## 20. Explicit non-claims

This package does not declare a directional winner, a production-readiness
result, a model-quality result, a training or continual-learning result, a
GPU or concurrency or energy result, or a conclusion that generalizes
beyond this campaign, model package, and host. The Boolean field
`superiority_claimed` is present only to record `false`.
`practical_equivalence_claimed` is `false`.
