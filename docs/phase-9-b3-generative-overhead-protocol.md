# Phase 9-B3A-R7 generative-overhead benchmark protocol

This document is the sole implementation specification for Phase 9-B3B. It historically supersedes 9-B3A through 9-B3A-R6. No earlier phase is authority. Implementers use only this text.

## Protocol identity

| Field | Value |
|---|---|
| Name | Phase 9-B3A-R7 generative-overhead benchmark protocol |
| Family | `generative_overhead` |
| Protocol document revision | integer `5` |
| JSON schema version | integer `1` |
| Evidence-path contract | `generative-overhead-r7-inventory-v1` |
| Clock | `std::chrono::steady_clock` inside one worker process |

`protocol_document_sha256` is SHA-256 of the exact bytes of `docs/phase-9-b3-generative-overhead-protocol.md`. That digest is the machine-verifiable document identity. The collector computes it from the current file bytes, and the campaign manifest records that exact value. Protocol document revision `5` is a human-readable governance revision. It is not a JSON field and is not added to the 56-field manifest.

Protocol document revision, JSON schema version, and evidence-path contract are distinct. A normative change to this document requires incrementing the protocol document revision, a new `protocol_document_sha256`, and an independent audit before collection. A JSON wire-format change requires JSON schema-version review and likely a schema increment. A contract or methodology change requires contract-ID review and likely a new contract. Unchanged document bytes keep the same revision and SHA. A campaign must reject any `evidence_path_contract` other than `generative-overhead-r7-inventory-v1`.

Revision `1` was the original audited R7 document. Revision `2` incorporates the independently reviewed LPM structural correction and SleepWake mixed-log classification correction. Revision `3` added collector process timeouts, including an unsafe thread-based 15-second `Popen` bound. Revision `4` replaces that process-creation deadline with direct owning-thread process creation and bounded identity liveness. Revision `5` replaces exact historical-list equality with strictly verified pruning of an oldest pre-campaign `pmset` target-event prefix. The campaign-start floor, ioreg triple, and ban on any in-campaign Sleep/Wake/DarkWake are unchanged. Benchmark cells, schedule, in-worker timing boundaries, statistics, manifest schema, inventories, JSON schema version, contract, and claim rules are unchanged.

## Terminology

InferEdge is the integration under test. llama.cpp v0.7.0 is both InferEdge’s pinned backend and the direct reference. Qwen3-0.6B is the model. Q4_K_M is the exact GGUF. One kernel, two paths. InferEdge did not train Qwen.

## Model and dependency identity

| Item | Value |
|---|---|
| Package directory | `/Users/varunkasa/InferEdgeModels/phase-9-b2a-qwen3-0.6b/package` |
| GGUF path | `/Users/varunkasa/InferEdgeModels/phase-9-b2a-qwen3-0.6b/package/qwen3-0.6b-Q4_K_M.gguf` |
| GGUF size | integer `484220192` |
| GGUF SHA-256 | `3f384c3796ca2547c5547be6bca3d487a18807cc1a2c259de6e0e1fa433bb76d` |
| Package manifest SHA-256 | `e6ff54460e7e30b3208f76d29eb49a1388e11f2223b478fa4bbf47a48e9ee281` |
| Template file | `qwen3-non-thinking-v1.txt` (in the package directory) |
| Template size | 75 bytes |
| Template SHA-256 | `6cb4fbe89e712bf8e8527aa8fb7d80eb0cad93ae1e236bb4be9c96d71809976a` |
| `{user}` offset | byte `17` (exactly one occurrence) |
| License SHA-256 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |
| llama.cpp tag | `v0.7.0` |
| llama.cpp commit | `620d2225cc436ab0b001e3200e590e3b286ef9ab` |
| Archive SHA-256 | `55d67acf3ad2ad747a9bbeb75f9f0e173dbcc0d1753a471038a90c819421c5e1` |
| Prefix | `/private/tmp/inferedge-9b1-llama/prefix` |
| Provenance path | `/private/tmp/inferedge-9b1-llama/prefix/share/inferedge/llama-cpp-provenance.txt` |
| Provenance size | integer `252` |
| Provenance SHA-256 | `e983338babf40f60c2bb6ca4a618707da2381c5d97c3231f8bc4c177626c9d59` |
| `libllama.0.7.0.dylib` | `861ca5e181276685823f91fa8c7c6f6d71c2bcf46765779e2b3d66a44925ee70` |
| `libggml.0.27.0.dylib` | `cd15ab8b4dab8f9c21c43ada03a1e55f8e55a2b75ce237296ab912d0125326a4` |
| `libggml-cpu.0.27.0.dylib` | `e6610ea60ab7e564ff6b2661869cf6e0d831a258ae0a7c58a4a13d30b063f539` |
| `libggml-base.0.27.0.dylib` | `55281093079a4c4ddd55166ad8f3aab74af276701797f1eb2c8d805c2ebb318e` |

Reject commit `620d2225b349e2039fa983c5e9cfc86a7e18e340`.

Provenance exact ASCII (LF, no CR):

```
schema_version=1
project=llama.cpp
tag=v0.7.0
commit=620d2225cc436ab0b001e3200e590e3b286ef9ab
source_archive_sha256=55d67acf3ad2ad747a9bbeb75f9f0e173dbcc0d1753a471038a90c819421c5e1
build_type=Release
architecture=arm64
backend=cpu
metal=off
server=off

```

## Cell matrix

| Cell | Fixture | `max_new_tokens` | Threads | `measurement_class` | Role | Success `finish_reason` |
|---|---|---|---|---|---|---|
| C01 | P_tiny | 1 | 1 | `warm_request` | `primary` | `maximum_tokens` only |
| C02 | P_128 | 1 | 1 | `warm_request` | `secondary` | `maximum_tokens` only |
| C03 | P_512 | 1 | 1 | `warm_request` | `secondary` | `maximum_tokens` only |
| C04 | P_1024 | 1 | 1 | `warm_request` | `secondary` | `maximum_tokens` only |
| C05 | P_2048 | 1 | 1 | `warm_request` | `secondary` | `maximum_tokens` only |
| C06 | P_near | 1 | 1 | `warm_request` | `secondary` | `maximum_tokens` only |
| C07 | P_tiny | 16 | 1 | `warm_request` | `secondary` | `eos` or `maximum_tokens`, arms identical |
| C08 | P_tiny | 128 | 1 | `warm_request` | `secondary` | `eos` or `maximum_tokens`, arms identical |
| C09 | P_tiny | 1 | 2 | `warm_request` | `secondary` | `maximum_tokens` only |
| C10 | P_tiny | 1 | 4 | `warm_request` | `secondary` | `maximum_tokens` only |
| C11 | P_tiny | 1 | 6 | `warm_request` | `secondary` | `maximum_tokens` only |
| L01 | JSON `null` | JSON `null` | n/a | `load` | `descriptive_load` | `load_success` only |

Forbidden: extra cells; `P_near` with `max_new_tokens>1`; prompt tokens + `max_new_tokens` > 4096; thread sweeps on long prompts; extra fixtures, threads, quants, models; Llama cells; `llama-cli` as a timed arm; TTFT/prefill/decode-only metrics. Matching arms do not override a forbidden finish. Any other finish invalidates the campaign.

## Ordering and samples

Construction list: `C01,C02,C03,C04,C05,C06,C07,C08,C09,C10,C11,L01`.

```python
import random
def permute_cells(cells, seed=20261017):
    rng = random.Random(seed)
    a = list(cells)
    n = len(a)
    for i in range(n):
        j = i + rng.randrange(n - i)
        a[i], a[j] = a[j], a[i]
    return a
```

Golden visit order: `C05,C02,C10,C06,C04,C09,C11,L01,C07,C08,C01,C03`.

C01–C11: exactly two unmeasured warmups per long-lived worker, then exactly 20 measured pairs.
L01: exactly two unmeasured paired new-process load warmups, then exactly 20 new-process pairs.
`pair_index` 0..19. Pair 0 is even. Even: InferEdge then reference (`arm_order=ie_then_ref`). Odd: reference then InferEdge (`arm_order=ref_then_ie`). `order_position` is 0 then 1.

## Full-campaign rules

A publication-eligible campaign is exactly one `campaign_id`, one evidence root, one `campaign_manifest.json` plus detached digest, and all twelve cells in golden order, each with required warmups and exactly 20 valid measured pairs. All records share the direct identity quintet and the same manifest digest.

Secondary cells cannot replace or rescue C01. No subset publication.

Any crash, worker restart, collector exit, incomplete pair or cell, correctness mismatch, write failure, AC/LPM/sleep failure, identity mismatch, or path-security failure invalidates the entire `campaign_id`. No resume. No one-cell rerun. No splicing. Retry = new ID, new root, new manifest and digest, full matrix from the start. Failed files are diagnostic only.

## Template rendering

Read the 75-byte template. Require the SHA-256 above. Find the unique six bytes `7b 75 73 65 72 7d` (`{user}`) at offset 17. Require exactly one occurrence.

```
rendered_bytes = template_bytes[:pos] + user_bytes + template_bytes[pos+6:]
```

No format engine, replace-all, strip, newline conversion, Unicode normalization, or added newline. Keep the template’s trailing `\n\n`. Official and InferEdge tokenizers receive these exact bytes.

## Fixture generation

Sidecar authority:

- `models/llm/generative_overhead_fixtures.json`
- `models/llm/generative_overhead_fixtures.json.sha256`

`P_tiny` user bytes: ASCII `OK` (`4f 4b`).
For `P_128`, `P_512`, `P_1024`, `P_2048`, `P_near`: `candidate(i) = b"alpha " * i` (six bytes `61 6c 70 68 61 20` per copy). Domain `i = 1..10000` inclusive. No other candidate family.

For each `i`: render as above; official tokenize rendered bytes with `add_special_tokens=false`; InferEdge tokenize with `add_special=true`, `parse_special=true`; require identical token-ID arrays; `count` = that common length. Token-ID mismatch makes that `i` ineligible.

T ∈ {128,512,1024,2048}: eligible if `count ∈ [T-2, T+2]`; choose minimum `(abs(count-T), i)`. Empty eligible set fails closed.
`P_near`: eligible if `count ≤ 4095`; choose maximum `(count, i)`. Empty eligible set fails closed.

Array order: `P_tiny`, `P_128`, `P_512`, `P_1024`, `P_2048`, `P_near`.

Phase 9-B3B constructs the sidecar by tokenization only, then commits it before any dry run. No latency, throughput, or quality may influence bytes.

## Fixture sidecar schema

Serialize with:

```python
json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=False, separators=(",", ":"))
```

UTF-8 encode. Append exactly one LF. Detached `.sha256`: 64 lowercase hex of those exact bytes + one LF.

**Top-level key order and types:**

1. `schema_family` string `generative_overhead`
2. `schema_version` integer `1`
3. `fixture_generation_algorithm` string `generative_overhead_fixture_gen`
4. `fixture_generation_version` integer `1`
5. `model_id` string `qwen3-0.6b`
6. `package_directory` string `/Users/varunkasa/InferEdgeModels/phase-9-b2a-qwen3-0.6b/package`
7. `package_manifest_sha256` `e6ff54460e7e30b3208f76d29eb49a1388e11f2223b478fa4bbf47a48e9ee281`
8. `gguf_file` `qwen3-0.6b-Q4_K_M.gguf`
9. `gguf_sha256` `3f384c3796ca2547c5547be6bca3d487a18807cc1a2c259de6e0e1fa433bb76d`
10. `gguf_size_bytes` integer `484220192`
11. `chat_template_file` `qwen3-non-thinking-v1.txt`
12. `chat_template_sha256` `6cb4fbe89e712bf8e8527aa8fb7d80eb0cad93ae1e236bb4be9c96d71809976a`
13. `tokenizer_identity` `hf:Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca+gguf-vocab`
14. `official_tokenize_flags` `official:add_special_tokens=false`
15. `inferedge_tokenize_flags` `inferedge:add_special=true,parse_special=true`
16. `context_limit` integer `4096`
17. `fixtures` array of 6 objects

**Each fixture object:** `fixture_id` string; `user_text_utf8` string; `user_text_sha256` 64-hex; `rendered_text_utf8` string; `rendered_text_sha256` 64-hex; `official_token_ids` integer array; `inferedge_token_ids` integer array; `expected_token_count` integer.
Then, except `P_tiny`: `selected_i` integer 1..10000.
Then `P_near` only: `context_limit` integer 4096; `max_new_tokens_reserved` integer 1; `derivation_rule` string `max (count, i) among i=1..10000 with count<=4095`; `accepted_count` integer equal to `expected_token_count`.

## Token and text digests

Each token ID is a signed 32-bit two’s-complement little-endian 4-byte value. Concatenate in array order. No length prefix. SHA-256 of that byte string. IDs outside `[−2147483648, 2147483647]` are rejected. Empty list → SHA-256 of zero bytes `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. User/rendered/generated text hashes = SHA-256 of exact UTF-8 bytes, no terminator, no normalization.

## Enums

`record_class`: `warmup` | `measured` | `environment` | `aggregate` | `summary`
`implementation`: `inferedge` | `reference`
`measurement_class`: `warm_request` | `load`
`campaign_class`: `campaign` | `dry_run`
`arm_order`: `ie_then_ref` | `ref_then_ie`
`finish_reason`: `eos` | `maximum_tokens` | `load_success` | `none`
`exclusion_status`: `none`
`cell_role`: `primary` | `secondary` | `descriptive_load`
`direction`: `inferedge_slower` | `inferedge_faster` | `none`
`claim_status`: `inferedge_slower` | `inferedge_faster` | `no_winner_ineligible` | `no_winner_below_threshold_or_ci`

`error_code`:

| InferEdge `GenerationError` | `error_code` |
|---|---|
| `kNone` | `none` |
| `kInvalidPackage` | `invalid_package` |
| `kManifest` | `manifest` |
| `kPathSecurity` | `path_security` |
| `kHashMismatch` | `hash_mismatch` |
| `kUnsupportedModel` | `unsupported_model` |
| `kUnsupportedArchitecture` | `unsupported_architecture` |
| `kUnsupportedQuantization` | `unsupported_quantization` |
| `kLicense` | `license` |
| `kModelLoad` | `model_load` |
| `kTokenization` | `tokenization` |
| `kContextLimit` | `context_limit` |
| `kDecode` | `decode` |
| `kInvalidUtf8` | `invalid_utf8` |
| `kInternal` | `internal` |
| reference null model/context | `model_load` |
| cancel or anything else | `unexpected_error` |

`PackageError` maps through `map_package_error` into this table. `error_message` is diagnostic and never decides eligibility.

`GenerationFinishReason::kEos` → `eos`; `kMaximumTokens` → `maximum_tokens`; L01 success → `load_success`; cancel or failure → `none`.

Success records: `success=true`, `error_code=none`, `error_message=""`, `exclusion_status=none`, `exclusion_reason=""`.
Failed diagnostic records: `success=false`, `finish_reason=none`, non-`none` `error_code`; campaign invalid.

## Rejection codes

Closed set:

```
atomic_publish_failure
campaign_class_invalid
campaign_incomplete
correctness_mismatch
digest_mismatch
duplicate_record
environment_ac_invalid
environment_command_failed
environment_lpm_invalid
environment_parse_invalid
environment_sleepwake_invalid
extra_file
finish_reason_invalid
hardlink_rejected
identity_mismatch
manifest_invalid
matrix_invalid
missing_file
mixed_campaign
pair_order_invalid
path_invalid
record_schema_invalid
sample_count_invalid
symlink_rejected
text_mismatch
timing_invalid
token_mismatch
utf8_invalid
warmup_invalid
worker_restart
```

### Validation phases (this order)

1. Path and filesystem-object validation.
2. Inventory validation.
3. Manifest/digest/schema validation.
4. Direct/transitive identity validation.
5. Environment validation.
6. Warmup validation.
7. Measured-record validation.
8. Cross-arm correctness validation.
9. Matrix/sample/order validation.
10. Analysis and claim validation.

Skip a later check only when it is unsafe or impossible because an earlier executed check failed. For every executed failed rule, add exactly one primary code. Then unique + lexicographic sort → `rejection_codes`. Exception text and walk order never change codes.

### Rule-to-code mapping

**Phase 1**

| Failed rule | Code |
|---|---|
| Absolute / empty / `.` / `..` / `\` / NUL / lexical rewrite / escape from root | `path_invalid` |
| Symlink on root, ancestor, directory, or file | `symlink_rejected` |
| Required file path exists but not `S_ISREG` (directory, FIFO, socket, block/char device, other) | `path_invalid` |
| Required directory path exists but not `S_ISDIR` | `path_invalid` |
| Required file `st_nlink != 1` | `hardlink_rejected` |
| `lstat` fails on a path that must exist | `missing_file` |

**Phase 2**

| Failed rule | Code |
|---|---|
| Required regular file absent | `missing_file` |
| Extra directory not in the legal tree | `extra_file` |
| Extra regular file that is not a legal temp | `extra_file` |
| Second file for the same logical (cell, pair, arm) or (cell, impl, warmup_index) | `duplicate_record` |
| Analyzer start sees `analysis/` present | `campaign_incomplete` |
| Inventory B missing aggregate or summary | `missing_file` |
| Inventory A contains aggregate, summary, or `analysis/` | `extra_file` |
| Wrong zero-padding or illegal relative path that is not a required name | `path_invalid` |

Precedence: required path missing → `missing_file`. Name not required and not a legal temp → `extra_file`. Two required logical keys collide → `duplicate_record`.

**Phase 3**

| Failed rule | Code |
|---|---|
| Manifest JSON key/type/enum/order/value failure | `manifest_invalid` |
| Detached digest file format failure | `record_schema_invalid` |
| Detached digest recompute mismatch | `digest_mismatch` |
| `campaign_class` not `campaign` or `dry_run` | `campaign_class_invalid` |
| Any warmup/measured/environment/aggregate/summary JSON key/type/enum/order failure | `record_schema_invalid` |
| SHA-256 field wrong JSON type or not lowercase 64-hex syntax | `record_schema_invalid` |
| SHA-256 field is valid 64-hex but ≠ SHA-256 of canonical sibling bytes (token LE int32 concat, text UTF-8, env raw file, aggregate bytes, `compiler_version_stdout_utf8.encode("utf-8")`, manifest bytes, fixture JSON bytes) | `digest_mismatch` |

**Phase 4**

| Failed rule | Code |
|---|---|
| Mixed `campaign_id` | `mixed_campaign` |
| Mixed `campaign_manifest_sha256` | `mixed_campaign` |
| Direct quintet ≠ manifest-bound values | `identity_mismatch` |
| Worker binary SHA ≠ manifest `ie_worker_sha256` / `ref_worker_sha256` for that implementation | `identity_mismatch` |

**Phase 5**

| Failed rule | Code |
|---|---|
| Command exit ≠ 0 | `environment_command_failed` |
| AC line / discharging | `environment_ac_invalid` |
| LPM structural or value failure | `environment_lpm_invalid` |
| Sleep/wake / log event / TZ | `environment_sleepwake_invalid` |
| `parsed` recompute mismatch or unparseable required field | `environment_parse_invalid` |
| Environment JSON published without 14 verified raw files | `campaign_incomplete` |

**Phase 6**

| Failed rule | Code |
|---|---|
| Missing warmup; wrong `warmup_index`; failed warmup operation; cross-implementation warmup token/text/finish mismatch; warmup used as measured | `warmup_invalid` |
| Warmup `t0_ns`/`t1_ns`/`duration_ns` wrong type, negative, `t1_ns < t0_ns`, or `duration_ns != t1_ns - t0_ns` | `timing_invalid` |

If both a timing defect and another warmup defect exist, emit both codes.

**Phase 7**

| Failed rule | Code |
|---|---|
| Measured `t0`/`t1`/`duration` type or `t1<t0` or `duration!=t1-t0` | `timing_invalid` |
| Invalid UTF-8 in a UTF-8 field | `utf8_invalid` |
| `pair_index`/`arm_order`/`order_position` disagree with `expanded_trials` | `pair_order_invalid` |
| `finish_reason` violates cell policy | `finish_reason_invalid` |
| Long-lived cell `pid` or `worker_instance_id` changes | `worker_restart` |

**Phase 8**

| Failed rule | Code |
|---|---|
| Prompt or generated token arrays differ | `token_mismatch` |
| Generated UTF-8 bytes or text hash differ | `text_mismatch` |
| Finish reasons differ when both are allowed by cell policy | `correctness_mismatch` |
| Count/UTF-8-validity differ without token/text/finish codes | `correctness_mismatch` |

If tokens differ, add `token_mismatch` (not also `correctness_mismatch` for the same token defect). If tokens match and text differs, `text_mismatch` only. If both differ, add both. Finish-policy violation uses `finish_reason_invalid` even if arms match. In-record digest errors remain `digest_mismatch` and may coexist with `token_mismatch` or `text_mismatch`.

**Phase 9**

| Failed rule | Code |
|---|---|
| Wrong/missing/extra cell or cell parameters | `matrix_invalid` |
| Warmup count ≠ 2 per impl per cell or measured pairs ≠ 20 | `sample_count_invalid` |
| Interrupted lifecycle / missing phase | `campaign_incomplete` |

**Phase 10**

| Failed rule | Code |
|---|---|
| `publish_exclusive` failure | `atomic_publish_failure` |
| Analysis rerun / `analysis/` already existed at analyzer start | `campaign_incomplete` |
| Claim computed on incomplete or ineligible data as if eligible | `campaign_incomplete` |
| Dry-run evidence offered as publication | `campaign_class_invalid` |

## Identity contract

Every warmup, measured, environment, aggregate, and summary JSON object stores:

`campaign_id`, `campaign_manifest_sha256`, `protocol_document_sha256`, `fixtures_sha256`, `source_commit`.

Warmup and measured also store `worker_binary_sha256`. Environment, aggregate, and summary do not store worker hashes. All other identities are bound by the verified exact `campaign_manifest.json` bytes and detached digest.

## Path security

All protocol paths are relative. Reject absolute paths; empty components; `.`; `..`; backslash; NUL; any lexically normalized path ≠ supplied path.

`lstat` the evidence root and every existing component. Do not follow symlinks. Reject any symlink on the root, an ancestor, a directory, or a file.

**File path:** `lstat` succeeds, `stat.S_ISREG(mode)`, not a symlink, `st_nlink == 1`. Reject directory, FIFO, socket, block device, character device, door, or other non-regular object (`path_invalid`). `st_nlink != 1` → `hardlink_rejected`.

**Directory path:** `lstat` succeeds, `stat.S_ISDIR(mode)`, not a symlink.

Lexical path must remain under the exact evidence root.

These rules apply to: manifest, digest, warmup, measured, environment JSON, environment raw files, aggregate, summary, fixture sidecars when verified, and legal temporary files.

## Temporary files

Name: `.{final_basename}.pub.{token}.tmp`
`token` = `uuid.uuid4().hex` (exactly 32 lowercase hexadecimal characters).
Regex: `^\.[^/]+\.pub\.[0-9a-f]{32}\.tmp$`
Must be a regular non-symlink file in an exact legal publication directory listed below. Ignored as evidence only then. Temp in any other directory → `extra_file`. Any other `*.tmp` → `extra_file`.

No temporary-directory grammar. Unexpected directories → `extra_file`. `analysis/` is not a temporary directory.

## Atomic publisher

`publish_exclusive(final_path, exact_bytes)`:

1. Exclusive-create a unique temporary file in the destination directory using the temp grammar.
2. Write all bytes.
3. Flush userspace buffers.
4. `fsync` the temporary file.
5. Re-read; require identical bytes and schema validation when applicable.
6. Create the final path with `os.link` (no-replace, same filesystem). Do not overwrite-rename.
7. Unlink the temporary name only after step 6 succeeds.
8. `fsync` the containing directory.
9. Re-read the final file; require `st_nlink == 1`.

Hard failure (`atomic_publish_failure`, campaign invalid): existing destination; cross-device; link/`fsync`/validation failure. No overwrite fallback.

Used for: manifest, detached digest, warmup, measured, environment JSON, environment raw bins, aggregate, summary, and the B3B fixture sidecar.

## Legal directory tree

Directories are not counted in the 920/922 file totals.

**Inventory A directories** (besides `<evidence_root>`):

```
<evidence_root>/warmup/
<evidence_root>/warmup/C01/inferedge/
<evidence_root>/warmup/C01/reference/
<evidence_root>/warmup/C02/inferedge/
<evidence_root>/warmup/C02/reference/
<evidence_root>/warmup/C03/inferedge/
<evidence_root>/warmup/C03/reference/
<evidence_root>/warmup/C04/inferedge/
<evidence_root>/warmup/C04/reference/
<evidence_root>/warmup/C05/inferedge/
<evidence_root>/warmup/C05/reference/
<evidence_root>/warmup/C06/inferedge/
<evidence_root>/warmup/C06/reference/
<evidence_root>/warmup/C07/inferedge/
<evidence_root>/warmup/C07/reference/
<evidence_root>/warmup/C08/inferedge/
<evidence_root>/warmup/C08/reference/
<evidence_root>/warmup/C09/inferedge/
<evidence_root>/warmup/C09/reference/
<evidence_root>/warmup/C10/inferedge/
<evidence_root>/warmup/C10/reference/
<evidence_root>/warmup/C11/inferedge/
<evidence_root>/warmup/C11/reference/
<evidence_root>/warmup/L01/inferedge/
<evidence_root>/warmup/L01/reference/
<evidence_root>/measured/
<evidence_root>/measured/C01/pair-00/ … pair-19/
<evidence_root>/measured/C02/pair-00/ … pair-19/
<evidence_root>/measured/C03/pair-00/ … pair-19/
<evidence_root>/measured/C04/pair-00/ … pair-19/
<evidence_root>/measured/C05/pair-00/ … pair-19/
<evidence_root>/measured/C06/pair-00/ … pair-19/
<evidence_root>/measured/C07/pair-00/ … pair-19/
<evidence_root>/measured/C08/pair-00/ … pair-19/
<evidence_root>/measured/C09/pair-00/ … pair-19/
<evidence_root>/measured/C10/pair-00/ … pair-19/
<evidence_root>/measured/C11/pair-00/ … pair-19/
<evidence_root>/measured/L01/pair-00/ … pair-19/
<evidence_root>/environment/
<evidence_root>/environment/raw/
<evidence_root>/environment/raw/campaign-start/
<evidence_root>/environment/raw/campaign-end/
<evidence_root>/environment/raw/C01-pre/
<evidence_root>/environment/raw/C01-post/
<evidence_root>/environment/raw/C02-pre/
<evidence_root>/environment/raw/C02-post/
<evidence_root>/environment/raw/C03-pre/
<evidence_root>/environment/raw/C03-post/
<evidence_root>/environment/raw/C04-pre/
<evidence_root>/environment/raw/C04-post/
<evidence_root>/environment/raw/C05-pre/
<evidence_root>/environment/raw/C05-post/
<evidence_root>/environment/raw/C06-pre/
<evidence_root>/environment/raw/C06-post/
<evidence_root>/environment/raw/C07-pre/
<evidence_root>/environment/raw/C07-post/
<evidence_root>/environment/raw/C08-pre/
<evidence_root>/environment/raw/C08-post/
<evidence_root>/environment/raw/C09-pre/
<evidence_root>/environment/raw/C09-post/
<evidence_root>/environment/raw/C10-pre/
<evidence_root>/environment/raw/C10-post/
<evidence_root>/environment/raw/C11-pre/
<evidence_root>/environment/raw/C11-post/
<evidence_root>/environment/raw/L01-pre/
<evidence_root>/environment/raw/L01-post/
```

Each `pair-NN` uses `NN` in `00`..`19`.

**Inventory B addition:** `<evidence_root>/analysis/` (regular directory, not a symlink).

Legal publication directories for temps are exactly those directories that hold required regular files (root for manifest/digest; each warmup impl dir; each pair dir; `environment/`; each `environment/raw/{sample}/`; `analysis/` after exclusive creation).

## Inventory A (analyzer input)

920 regular files:

- `campaign_manifest.json` (1)
- `campaign_manifest.sha256` (1)
- 48 `warmup/{cell}/{implementation}/warmup-{0|1}.json`
- 480 `measured/{cell}/pair-{NN}/{implementation}.json`
- 26 `environment/{sample_id}.json`
- 364 `environment/raw/{sample_id}/{command_id}.{stdout|stderr}.bin`

`2+48+480+26+364=920`. `analysis/` must be absent. No aggregate. No summary. No other non-temp entries. This is the analyzer input.

## Inventory B (final publication package)

922 regular files: every Inventory A file plus `analysis/aggregate.json` and `analysis/summary.json`. The final verifier validates Inventory B read-only and never regenerates analysis files.

## Analyzer start and rerun

Before analysis: Inventory A is valid; `analysis/` is absent. Create `analysis/` with exclusive directory creation. `lstat` must show `S_ISDIR` and not a symlink. `fsync` the evidence root. Publish aggregate. Publish summary. `fsync` `analysis/`. Validate Inventory B.

If `analysis/` already exists (empty, partial, or complete): do not run the analyzer; fail closed (`campaign_incomplete`). Complete Inventory B is handled only by the read-only final verifier. Interrupted create or publish permanently makes that campaign root non-publication. No deletion, overwrite, cleanup, or retry under that `campaign_id`.

## Environment commands

Sample IDs: `campaign-start`, `{cell_id}-pre`, `{cell_id}-post`, `campaign-end` for each of the twelve cells (26 samples).

Command IDs in this order:

| `command_id` | argv |
|---|---|
| `pmset_batt` | `["/usr/bin/pmset","-g","batt"]` |
| `pmset_settings` | `["/usr/bin/pmset","-g"]` |
| `pmset_custom` | `["/usr/bin/pmset","-g","custom"]` |
| `ioreg_power` | `["/usr/sbin/ioreg","-r","-c","IOPMrootDomain","-d","1"]` |
| `pmset_log` | `["/usr/bin/pmset","-g","log"]` |
| `system_profiler_power` | `["/usr/sbin/system_profiler","SPPowerDataType"]` |
| `powermanagement_plist` | `["/usr/bin/plutil","-convert","xml1","-o","-", <resolved_uuid_plist>]` |

UUID discovery:

```python
from pathlib import Path
import re
root = Path("/Library/Preferences")
pat = re.compile(r"^com\.apple\.PowerManagement\.[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\.plist$")
matches = sorted(p for p in root.iterdir() if pat.fullmatch(p.name))
```

Require `len(matches)==1`. That path is the last argv element. Every command exit status must be 0.

Raw paths: `environment/raw/{sample_id}/{command_id}.stdout.bin` and `.stderr.bin`. Always publish both, including empty files.

## Raw-before-JSON (each sample)

1. Run the seven commands in command-ID order.
2. Hold stdout, stderr, and exit statuses in memory.
3. Compute hashes and parsed results.
4. For each command in command-ID order: `publish_exclusive` `.stdout.bin`, then `.stderr.bin`.
5. Re-read and verify all fourteen raw files.
6. Only then `publish_exclusive` the environment JSON.
7. Re-read and verify that JSON.
8. Any failure invalidates the entire campaign.

No environment JSON may be published before all fourteen raw files it references.

## Environment JSON schema

Key order and types:

1. `schema_family` string `generative_overhead`
2. `schema_version` integer `1`
3. `record_class` string `environment`
4. `campaign_id` string
5. `campaign_manifest_sha256` 64-hex
6. `protocol_document_sha256` 64-hex
7. `fixtures_sha256` 64-hex
8. `source_commit` 40-hex
9. `sample_id` string
10. `wall_time_ns` integer (`time.time_ns()`)
11. `commands` array of 7 objects in command-ID order
12. `parsed` object

Each command object: `command_id` string; `argv` array of strings; `exit_status` integer `0`; `stdout_relative_path` string; `stdout_size_bytes` integer ≥0; `stdout_sha256` 64-hex; `stderr_relative_path` string; `stderr_size_bytes` integer ≥0; `stderr_sha256` 64-hex.

`parsed` keys: `ac_line` string; `ac_ok` boolean; `low_power_mode_value` integer; `low_power_mode_off` boolean; `sleepwake_uuid` string; `last_sleep_reason` string; `wake_reason` string; `pmset_log_disqualifying_events` array.

Each event object: `timestamp_text` string; `timezone_text` string; `event_token` string (`Sleep`|`Wake`|`DarkWake`); `aware_unix_second` integer; `raw_line_sha256` 64-hex. Chronological input order. The analyzer recomputes `parsed` from raw bytes.

## AC, LPM, sleep, network

**AC:** `pmset_batt` stdout contains exactly one line `Now drawing from 'AC Power'` and no `Battery Power` or `discharging`.

**LPM:** Before `plistlib.loads`, parse the XML so the root is a dictionary that contains exactly one unambiguous immediate key `AC Power` whose value is exactly one dictionary. That AC dictionary must contain exactly one unambiguous immediate key `LowPowerMode`. A sibling `Battery Power` dictionary, and other top-level power-profile dictionaries, are permitted when they do not duplicate `AC Power`. Other keys inside `AC Power` are permitted when they do not duplicate `LowPowerMode`. Duplicate keys at either of those levels fail closed. Then `plistlib.loads`. Require `root["AC Power"]["LowPowerMode"]` is a Python `int` equal to `0`, not a `bool`. Do not read Low Power Mode from Battery Power or any other profile.

**Sleep/wake:** From `ioreg_power`, exactly one `"SleepWakeUUID" = "<RFC-4122>"`, exactly one `"Last Sleep Reason"`, exactly one `"Wake Reason"`. All three must equal campaign-start values at every later sample.

`pmset_log` classification uses the exact whitespace-separated record-class token after the timestamp and timezone. Do not classify by substring search. A line is date-shaped when characters `[0:19]` match `YYYY-MM-DD HH:MM:SS`. Non-date-shaped lines (headers, footers, `UUID:` / `DriverReason:` continuations, `Sleep/Wakes since boot` summaries, assertion-status prose) are ignored. A date-shaped stamp must parse with `time.strptime(..., "%Y-%m-%d %H:%M:%S")`; an impossible civil date invalidates the campaign. The next whitespace field is the timezone and must match `[+-]HHMM` with hours `00–14` and minutes `00–59`; timezone failure on a date-shaped line invalidates the campaign. The following whitespace field is the record-class token, compared by exact string equality.

Trust-relevant tokens are exactly `Sleep`, `Wake`, and `DarkWake`. `Wake` does not match `WakeDetails` or `WakeTime`. `DarkWake` does not match other wake-related text. `WakeDetails`, `HibernateStats`, `PM`, session tokens, `Assertions`, `Kernel`, `Notification`, `WakeTime`, and every other exact token that is not one of those three are ignored. Assertions or Kernel text mentioning sleep or wake do not become events. Malformed or ambiguous trust-relevant records, duplicate trust-relevant identities (`timestamp_text`, `timezone_text`, `event_token`, `raw_line_sha256`), and trust-relevant timestamp regression invalidate the campaign.

Build `aware_unix_second` as `calendar.timegm(civil_tuple) - offset_seconds` from the stated timezone. Do not use locale-sensitive `mktime`. An event with `aware_unix_second >= floor(campaign_start_ns / 1_000_000_000)` invalidates the campaign, including an event exactly on that start-second boundary and any later end-boundary event. Historical trust-relevant events strictly before that floor are the campaign-start baseline and do not themselves invalidate a new campaign.

Campaign-start extracts the ordered list of trust-relevant event objects (`timestamp_text`, `timezone_text`, `event_token`, `aware_unix_second`, `raw_line_sha256`) and stores that list as the original start baseline and the first accepted rolling baseline. The campaign-start floor and the ioreg triple never change.

A later sample may drop only a contiguous oldest prefix of the current accepted rolling baseline. The later extracted list must be a nonempty exact suffix of that rolling baseline: every retained event must match `timestamp_text`, `timezone_text`, `event_token`, `aware_unix_second`, and `raw_line_sha256` in the same order. One shared retained event is the minimum continuity; the required proof is the entire retained suffix, not a single loose match. An empty later list after a nonempty baseline fails. A complete log reset fails.

The collector updates the rolling baseline to the newly accepted suffix only after that sample validates. A failed sample must not become the next baseline. Previously dropped events must not reappear. The target-event list must not gain a new pre-start event. A longer later target list fails. Dropping the newest retained event, an interior event, noncontiguous events, a reorder, a mutation of timestamp/timezone/token/raw-line hash, or a same-length substitution fails.

A new Sleep, Wake, or DarkWake at or after the campaign-start floor still fails, including after old-prefix pruning and including an event exactly on the start-second boundary. Unrelated appended ignored lines do not change the extracted list and do not invalidate. Do not compare entire raw logs byte-for-byte. macOS may prune oldest `pmset -g log` records; that host rotation is not evidence of an in-campaign sleep/wake unless the suffix or ioreg checks fail.

`pmset_log` stdout must be strict UTF-8 with no NUL and no CR. Reject more than 16777216 bytes, more than 200000 lines, or any line longer than 8192 bytes. These bounds admit the preserved 8.47 MiB capture.

**Network:** Workers use local IPC and local files. This protocol does not claim or gate network isolation. Runtime socket inspection is not a campaign gate.

No CPU-frequency or affinity claim.

## Warmup schema

Path: `warmup/{cell_id}/{implementation}/warmup-{index}.json`
`implementation` ∈ {`inferedge`,`reference`}; `index` ∈ {0,1}.

Key order and types:

1. `schema_family` string `generative_overhead`
2. `schema_version` integer `1`
3. `record_class` string `warmup`
4. `campaign_id` string
5. `campaign_manifest_sha256` 64-hex
6. `protocol_document_sha256` 64-hex
7. `fixtures_sha256` 64-hex
8. `source_commit` 40-hex
9. `cell_id` one of the twelve IDs
10. `implementation` enum
11. `warmup_index` integer 0 or 1
12. `measurement_class` enum (`load` on L01, else `warm_request`)
13. `success` boolean
14. `prompt_token_count_expected` integer ≥0
15. `prompt_token_count_actual` integer ≥0
16. `prompt_token_ids` array of integers in signed int32 range
17. `prompt_token_ids_sha256` 64-hex of LE int32 concat
18. `generated_token_count` integer ≥0
19. `generated_token_ids` array of integers
20. `generated_token_ids_sha256` 64-hex
21. `generated_text_utf8` string
22. `generated_text_sha256` 64-hex of those UTF-8 bytes
23. `finish_reason` enum (cell policy)
24. `utf8_valid` boolean
25. `threads_requested` integer (L01: `1`)
26. `threads_effective` integer
27. `n_ctx` integer
28. `n_batch` integer
29. `n_ubatch` integer
30. `n_seq_max` integer
31. `worker_binary_sha256` 64-hex
32. `worker_instance_id` string
33. `pid` integer
34. `t0_ns` integer ≥0
35. `t1_ns` integer ≥0 and `>= t0_ns`
36. `duration_ns` integer `== t1_ns - t0_ns`
37. `error_code` enum
38. `error_message` string

No `pair_index`. Durations are diagnostic and never enter n=20. L01: empty arrays and text; empty-list digest; `finish_reason=load_success`. Corresponding C01–C11 warmups (same cell, same index) must match tokens, text, finish reason, and UTF-8 validity.

## Measured schema

Path: `measured/{cell_id}/pair-{NN}/{implementation}.json`
`NN` is `pair_index` formatted as two digits `00`..`19`.

| # | Key | JSON type / constraint |
|---|---|---|
| 1 | `schema_family` | string `generative_overhead` |
| 2 | `schema_version` | integer `1` |
| 3 | `record_class` | string `measured` |
| 4 | `campaign_id` | string |
| 5 | `campaign_manifest_sha256` | 64 lowercase hex |
| 6 | `protocol_document_sha256` | 64 lowercase hex |
| 7 | `fixtures_sha256` | 64 lowercase hex |
| 8 | `source_commit` | 40 lowercase hex |
| 9 | `cell_id` | one frozen cell ID |
| 10 | `implementation` | `inferedge` or `reference` |
| 11 | `pair_index` | integer 0..19 |
| 12 | `arm_order` | `ie_then_ref` or `ref_then_ie` |
| 13 | `order_position` | integer 0 or 1 |
| 14 | `measurement_class` | `warm_request` or `load` |
| 15 | `fixture_id` | string; JSON `null` on L01 only |
| 16 | `max_new_tokens` | integer ≥1; JSON `null` on L01 only |
| 17 | `success` | boolean |
| 18 | `prompt_token_count_expected` | integer ≥0 |
| 19 | `prompt_token_count_actual` | integer ≥0 |
| 20 | `prompt_token_ids` | array of signed-int32-range integers; L01 empty |
| 21 | `prompt_token_ids_sha256` | 64-hex of LE int32 concat; L01 empty-list digest |
| 22 | `generated_token_count` | integer ≥0 |
| 23 | `generated_token_ids` | array of integers; L01 empty |
| 24 | `generated_token_ids_sha256` | 64-hex; L01 empty-list digest |
| 25 | `generated_text_utf8` | string; L01 `""` |
| 26 | `generated_text_sha256` | 64-hex of those bytes; L01 SHA-256 of zero bytes |
| 27 | `finish_reason` | enum; cell policy |
| 28 | `utf8_valid` | boolean |
| 29 | `threads_requested` | integer; L01 `1` |
| 30 | `threads_effective` | integer |
| 31 | `n_ctx` | integer |
| 32 | `n_batch` | integer |
| 33 | `n_ubatch` | integer |
| 34 | `n_seq_max` | integer |
| 35 | `worker_binary_sha256` | 64-hex |
| 36 | `worker_instance_id` | string |
| 37 | `pid` | integer |
| 38 | `t0_ns` | integer ≥0 |
| 39 | `t1_ns` | integer ≥0 and `>= t0_ns` |
| 40 | `duration_ns` | integer `== t1_ns - t0_ns` |
| 41 | `error_code` | enum |
| 42 | `error_message` | string; success `""` |
| 43 | `exclusion_status` | `none` |
| 44 | `exclusion_reason` | `""` |
| 45 | `publication_eligible` | boolean, informational, never trusted |

No `pair_id`. No `warmup_index`. No other keys.

## Manifest schema

No self-hash. Serialize with the same `json.dumps` settings + UTF-8 + one LF. `campaign_manifest.sha256` contains 64 lowercase hex of those exact bytes plus one LF. Publish the manifest, then the digest, then verify both before any warmup.

Key order and types:

1. `schema_family` string `generative_overhead`
2. `schema_version` integer `1`
3. `campaign_id` string
4. `campaign_class` `campaign` or `dry_run`
5. `source_commit` 40-hex
6. `protocol_document_sha256` 64-hex
7. `fixtures_sha256` 64-hex
8. `cell_inventory` array of 12 strings `C01`…`C11`,`L01`
9. `golden_cell_order` array of 12 strings `C05,C02,C10,C06,C04,C09,C11,L01,C07,C08,C01,C03`
10. `pair_order_rule` string `even ie_then_ref, odd ref_then_ie`
11. `expanded_trials` array of 480 objects
12. `warmup_count_per_worker` integer `2`
13. `measured_pairs_per_cell` integer `20`
14. `estimator` string `mean_paired_difference`
15. `bootstrap_replicates` integer `10000`
16. `bootstrap_seed` integer `20261018`
17. `confidence_level` number `0.95` (display only)
18. `absolute_threshold_ns` integer `2000000`
19. `relative_threshold_numerator` integer `5`
20. `relative_threshold_denominator` integer `100`
21. `relative_threshold_display` number `0.05` (non-authoritative)
22. `model_package_directory` `/Users/varunkasa/InferEdgeModels/phase-9-b2a-qwen3-0.6b/package`
23. `package_manifest_sha256` `e6ff54460e7e30b3208f76d29eb49a1388e11f2223b478fa4bbf47a48e9ee281`
24. `gguf_path` `/Users/varunkasa/InferEdgeModels/phase-9-b2a-qwen3-0.6b/package/qwen3-0.6b-Q4_K_M.gguf`
25. `gguf_sha256` `3f384c3796ca2547c5547be6bca3d487a18807cc1a2c259de6e0e1fa433bb76d`
26. `gguf_size_bytes` integer `484220192`
27. `chat_template_sha256` `6cb4fbe89e712bf8e8527aa8fb7d80eb0cad93ae1e236bb4be9c96d71809976a`
28. `INFEREDGE_LLAMA_CPP_ROOT` `/private/tmp/inferedge-9b1-llama/prefix`
29. `provenance_path` `/private/tmp/inferedge-9b1-llama/prefix/share/inferedge/llama-cpp-provenance.txt`
30. `provenance_size` integer `252`
31. `provenance_sha256` `e983338babf40f60c2bb6ca4a618707da2381c5d97c3231f8bc4c177626c9d59`
32. `llama_tag` `v0.7.0`
33. `llama_commit` `620d2225cc436ab0b001e3200e590e3b286ef9ab`
34. `llama_archive_sha256` `55d67acf3ad2ad747a9bbeb75f9f0e173dbcc0d1753a471038a90c819421c5e1`
35. `libllama_sha256` `861ca5e181276685823f91fa8c7c6f6d71c2bcf46765779e2b3d66a44925ee70`
36. `libggml_sha256` `cd15ab8b4dab8f9c21c43ada03a1e55f8e55a2b75ce237296ab912d0125326a4`
37. `libggml_cpu_sha256` `e6610ea60ab7e564ff6b2661869cf6e0d831a258ae0a7c58a4a13d30b063f539`
38. `libggml_base_sha256` `55281093079a4c4ddd55166ad8f3aab74af276701797f1eb2c8d805c2ebb318e`
39. `ie_worker_sha256` 64-hex of the InferEdge worker binary
40. `ref_worker_sha256` 64-hex of the reference worker binary
41. `collector_script_sha256` 64-hex of `benchmarks/collect_generative.py`
42. `analyzer_script_sha256` 64-hex of `benchmarks/analyze_generative.py`
43. `host_system` one-line `/usr/bin/uname -s`
44. `host_kernel_release` one-line `/usr/bin/uname -r`
45. `host_architecture` one-line `/usr/bin/uname -m`
46. `host_model` one-line `/usr/sbin/sysctl -n hw.model`
47. `os_product_name` one-line `/usr/bin/sw_vers -productName`
48. `os_product_version` one-line `/usr/bin/sw_vers -productVersion`
49. `os_build_version` one-line `/usr/bin/sw_vers -buildVersion`
50. `compiler_version_first_line` string
51. `compiler_version_stdout_utf8` string
52. `compiler_version_stdout_sha256` 64-hex
53. `environment_policy` `ac_connected,battery_not_discharging,lpm_off_positive,sleepwake_invariant_and_log`
54. `evidence_path_contract` `generative-overhead-r7-inventory-v1`
55. `raw_schema_version` integer `1`
56. `build_type` string `Release`

**Host commands** (uname, sysctl, sw_vers): exit 0; strict UTF-8; exactly one nonempty line; at most one final LF; reject CR, embedded LF, extra output. Strip exactly one trailing LF from that single line.

**Compiler:** invoke the CMake C++ compiler executable with `--version`. Exit 0. Capture complete stdout as raw bytes. Reject invalid UTF-8, any CR, any NUL. Preserve all LF bytes, including a final LF if present. Do not strip or normalize.

```
compiler_version_stdout_utf8 = raw_stdout.decode("utf-8", errors="strict")
require compiler_version_stdout_utf8.encode("utf-8") == raw_stdout
compiler_version_stdout_sha256 = sha256(raw_stdout).hexdigest()
```

Analyzer/final verifier: `sha256(compiler_version_stdout_utf8.encode("utf-8"))` must equal `compiler_version_stdout_sha256`. Split `compiler_version_stdout_utf8` on LF only. `compiler_version_first_line` is the first line whose exact content is nonempty, with no space/tab trim and without an LF terminator. Require at least one nonempty line. Additional lines stay in `compiler_version_stdout_utf8`. The one-line reject rule does not apply to the compiler. No separate compiler evidence file.

**`expanded_trials` object keys in this order:** `cell_id` string; `pair_index` integer 0..19; `arm_order` enum; `order_position` integer 0 or 1; `implementation` enum. Exactly 480 entries in execution order (golden cells × pairs 0–19 × arm order). The analyzer reconstructs this array as structured values and requires equal length, order, keys, types, and values.

## Workers and timing

**C01–C11:** two long-lived processes (InferEdge worker links `inferedge_generative`; reference worker links pinned llama.cpp only and must not link or call `inferedge_generative`). Per cell: start both new workers; verify identity and environment; each loads once outside request timing; each runs warmup-0 and warmup-1; execute twenty interleaved measured pairs with no further warmup; shut down both. `pid` and `worker_instance_id` stay constant. A change is `worker_restart`.

**L01:** new process per arm for warmups and measured pairs. Process exec is outside timing.

Logging: both workers install an identical `llama_log_set` sink before any `t0`: no console or file I/O in `[t0,t1]`; bounded in-memory buffer written after `t1`; no performance-print callback; overflow is `internal`.

**Warm-request timed work.** InferEdge: the complete loaded-session `GenerativeSession::generate()`. Reference independently: apply the same rendered bytes; `llama_set_n_threads(context, threads, threads)`; tokenize probe then write with InferEdge flags; context-budget check; `llama_memory_clear(llama_get_memory(context), true)`; `llama_sampler_init_greedy` and free; full-prompt `llama_decode`; sample/accept until `max_new_tokens` or EOG; exclude EOG from visible text; piece decode; UTF-8 assemble. InferEdge generating-guard and null abort-callback install/clear are integration overhead; the reference does not copy them.

**L01 timed work.** Setup outside timing: logging sink; fully populated parameter structs; resolved GGUF path. Timed:

```
t0 = steady_clock::now();
llama_backend_init();
model = llama_model_load_from_file(gguf_path, model_params);
context = llama_init_from_model(model, context_params);
t1 = steady_clock::now();  // only if both pointers non-null
```

InferEdge: `t0` immediately before `GenerativeSession::load(package_directory)`; `t1` immediately after successful return. That public path includes package validation, hashes, backend init, GGUF load, and context create. Teardown after `t1`. Descriptive; not in the C01 claim. Label: InferEdge package/session load versus minimal direct model/context load under filesystem-cache-warm conditions.

## Collector process timeouts

Revision `1` and revision `2` did not define a collector-side deadline for worker stdio or environment-command capture. Revision `3` added finite request, shutdown, and environment deadlines, and also claimed a 15-second thread-based `Popen` timeout. Revision `4` removes that process-creation timeout. These remaining deadlines reap a hung worker after a handle exists. They are not workload parameters, not measured latency, and not adaptive.

Timeouts are finite, deterministic, identical for InferEdge and reference for equivalent operations, independent of observed backend duration, independent of which backend runs first, independent of golden cell order, and not configurable through campaign CLI, environment variables, or other normal campaign input. An infinite timeout is forbidden.

They sit outside `[t0,t1]`. Worker `t0`/`t1`/`duration_ns` are recorded only from a complete in-process response. Deadline enforcement must not rewrite those fields. A timed-out request is discarded. There is no retry, no partial-line acceptance, no continuation to the next backend or next cell, and no warmup or measured JSON publication for the timed-out operation. The campaign is incomplete and non-resumable. The analyzer remains blocked.

Public abort codes stay in the existing closed set. Collector-internal timeout classes use `environment_command_failed` with a required diagnostic message. The JSON record schema is unchanged.

**Worker process creation.** Both worker executables are prevalidated local regular files: existing path and symlink checks, execute permission, no PATH lookup, no shell, argument-vector only, and the existing `--version` identity preflight. The owning collector thread calls `subprocess.Popen` synchronously and stores the handle immediately. `Popen` success is process creation, not worker liveness. `Popen` raising is fail-closed (`worker launch failed`). The collector does not claim a wall-clock timeout for `Popen` itself and does not launch a thread, daemon, or future to cancel it.

**Worker liveness.** The first bounded liveness and protocol check is the `identity` request. It uses the worker-request deadline. No further request proceeds until identity succeeds. Identity failure terminates, reaps, and closes that session before the exception propagates. The session is owned by the collector from `Popen` return onward.

| Timeout class | Bound (seconds) | Operations | Diagnostic message |
|---|---|---|---|
| Worker request | `300` | one deadline per `identity`, `load`, `warm_request`, or `l01` stdio request, including warmup and measured | `worker request timeout` |
| Worker shutdown | `1` then `1` then `2` | graceful `{"op":"shutdown"}` wait; `SIGTERM` to the process group; `SIGKILL` to the process group; `wait` after each | `worker shutdown timeout` |
| Environment command | `30` | host identity, compiler `--version`, and every environment `subprocess.run` capture | `environment command timeout` |

Worker request `300` is one fixed envelope for both backends and every cell, including C06/`P_near` and L01. It is not a per-cell budget and is not reduced for short prompts.

Justification for the worker-request bound, using protocol workload limits rather than incomplete-campaign samples:

- Maximum official prompt-token count is `4095` (`P_near`); `max_new_tokens` is `1` on that cell
- Maximum `max_new_tokens` on any cell is `128` (C08, `P_tiny`)
- Context capacity is `4096`; `P_near` plus one new token is the supported full-context request
- `n_ubatch=512` implies at most eight prefill microbatches plus decode steps
- Supported package is Qwen3-0.6B Q4_K_M, GGUF size `484220192`, `n_threads=1`, CPU arm64 host class
- Conservative finite allowances inside the same envelope: `50 ms` per prompt token × `4095` = `204.75 s`; `500 ms` per generated token × `128` = `64 s`; GGUF load plus `4096` context create is required to finish inside the same `300 s`
- The remaining margin keeps the deadline large enough for a legitimate `P_near` or L01 operation on the validated host class and small enough to terminate a genuinely hung worker

Deadline check is fail-closed: if `remaining <= 0` before a complete newline-terminated stdout line is received, the request fails. A response that completes with `remaining > 0` passes.

After timeout the collector stops request handling, attempts graceful shutdown if stdin is still writable, sends `SIGTERM` to the process group, sends `SIGKILL` if needed, waits, drains or closes pipes, and must leave no worker zombie or orphan. It must not start the next backend.

**L01 `llama_model_params` (every field assigned):**
`devices=NULL`; `tensor_buft_overrides=NULL`; `n_gpu_layers=0`; `split_mode=LLAMA_SPLIT_MODE_LAYER` (1); `load_mode=LLAMA_LOAD_MODE_AUTO` (−1); `lazy_mode=LLAMA_LAZY_MODE_AUTO` (1); `main_gpu=0`; `tensor_split=NULL`; `progress_callback=NULL`; `progress_callback_user_data=NULL`; `kv_overrides=NULL`; `vocab_only=false`; `check_tensors=false`; `use_extra_bufts=true`; `no_host=false`; `no_alloc=false`; `load_mtp=false`.

**L01 `llama_context_params` (every field assigned):**
`n_ctx=4096`; `n_batch=4096`; `n_ubatch=512`; `n_seq_max=1`; `n_rs_seq=0`; `n_outputs_max=0`; `n_outputs_max_per_seq=1`; `n_threads=1`; `n_threads_batch=1`; `ctx_type=LLAMA_CONTEXT_TYPE_DEFAULT` (0); `rope_scaling_type=LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED` (−1); `pooling_type=LLAMA_POOLING_TYPE_UNSPECIFIED` (−1); `attention_type=LLAMA_ATTENTION_TYPE_UNSPECIFIED` (−1); `flash_attn_type=LLAMA_FLASH_ATTN_TYPE_AUTO` (−1); `rope_freq_base=0.0`; `rope_freq_scale=0.0`; `yarn_ext_factor=−1.0`; `yarn_attn_factor=−1.0`; `yarn_beta_fast=−1.0`; `yarn_beta_slow=−1.0`; `yarn_orig_ctx=0`; `defrag_thold=−1.0`; `cb_eval=NULL`; `cb_eval_user_data=NULL`; `type_k=GGML_TYPE_F16` (1); `type_v=GGML_TYPE_F16` (1); `moe_cache_size=0`; `abort_callback=NULL`; `abort_callback_data=NULL`; `embeddings=false`; `offload_kqv=true`; `no_perf=true`; `op_offload=true`; `swa_full=true`; `kv_unified=false`; `samplers=NULL`; `n_samplers=0`; `ctx_other=NULL`.

## Correctness gate

1. Execute both arms in frozen order.
2. Durably publish both raw records.
3. Validate each record.
4. Compare prompt tokens, generated tokens, counts, finish reason (and cell policy), output bytes, UTF-8.
5. Only then accept the pair.

Mismatch: keep both files; stop; campaign invalid; no retry.

## Bootstrap

Input: 20 integer nanosecond differences `d[i] = IE_i - REF_i`. If length ≠ 20, no interval (`sample_count_invalid`). Do not import ONNX helpers. Do not use a 10-block minimum.

```python
import random

def interpolated_quantile(sorted_values, probability):
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (sorted_values[lower] * (1.0 - fraction)
            + sorted_values[upper] * fraction)

def bootstrap_mean_interval(d):
    assert len(d) == 20
    rng = random.Random(20261018)
    means = []
    for _ in range(10000):
        total = 0.0
        for _draw in range(20):
            total += float(d[rng.randrange(20)])
        means.append(total / 20.0)
    means.sort()
    tail = (1.0 - 0.95) / 2.0
    return interpolated_quantile(means, tail), interpolated_quantile(means, 1.0 - tail)
```

A CI bound equal to 0 fails “strictly excludes zero.”

## Claim arithmetic

`sum_d = Σ(IE − REF)` and `sum_ref = Σ REF` over exactly 20 C01 pairs (integers).

Direction: `sum_d > 0` InferEdge slower; `sum_d < 0` faster; `sum_d == 0` none.

Absolute: slower `sum_d >= 40000000`; faster `-sum_d >= 40000000`.
Relative: slower `20 * sum_d >= sum_ref`; faster `20 * (-sum_d) >= sum_ref`.

Do not use binary `0.05` in predicates. Display means use Python `sum / 20` and `relative_threshold_ns` display uses `sum_ref / 400`, serialized by `json.dumps`; display fields do not decide eligibility.

InferEdge-slower requires: slower direction, both thresholds, CI entirely `> 0`, full campaign valid. Faster is symmetric. Otherwise `winner_declared=false`, `superiority_claimed=false`. Do not declare practical equivalence.

## Descriptive statistics

For sorted twenty-element paired differences `s`:
P25 = `s[4]`; P50 = median = `s[9]`; P75 = `s[14]`; P95 = `s[18]`; IQR = `s[14] - s[4]`.
MAD: `a_i = abs(d_i - s[9])`; sort `a`; `mad_difference_ns = a[9]`. JSON integers. Descriptive only.

## Aggregate `analysis/aggregate.json`

Key order:

1. `schema_family` string `generative_overhead`
2. `schema_version` integer `1`
3. `record_class` `aggregate`
4. `campaign_id` string
5. `campaign_class` enum
6. `campaign_manifest_sha256` 64-hex
7. `protocol_document_sha256` 64-hex
8. `fixtures_sha256` 64-hex
9. `source_commit` 40-hex
10. `eligibility_recomputed` boolean
11. `complete_matrix` boolean
12. `rejection_codes` sorted unique array of closed codes
13. `cells` array in golden order
14. `c01` object
15. `winner_declared` boolean
16. `superiority_claimed` boolean

`cells[]` keys: `cell_id` string; `measurement_class` enum; `role` enum; `pair_count` integer; `inferedge_mean_ns` number or null; `reference_mean_ns` number or null; `mean_difference_ns` number or null; `median_difference_ns` integer or null; `p50_difference_ns` integer or null; `p95_difference_ns` integer or null; `iqr_difference_ns` integer or null; `mad_difference_ns` integer or null.

Eligible: all twelve `pair_count=20`; listed numerics non-null.
Ineligible: `pair_count` = number of complete valid observed pairs; statistics needing 20 pairs = null; winners false.

`c01` keys: `pair_count` integer; `sum_d` integer or null; `sum_ref` integer or null; `mean_difference_ns` number or null; `reference_mean_ns` number or null; `bootstrap_replicates` integer `10000`; `bootstrap_seed` integer `20261018`; `ci_low_ns` number or null; `ci_high_ns` number or null; `absolute_threshold_ns` integer `2000000`; `relative_threshold_numerator` integer `5`; `relative_threshold_denominator` integer `100`; `relative_threshold_ns` number or null; `direction` enum; `absolute_threshold_met` boolean; `relative_threshold_met` boolean; `ci_excludes_zero_strict` boolean; `slower_claim` boolean; `faster_claim` boolean.

No NaN or infinity.

## Summary `analysis/summary.json`

Key order:

1. `schema_family` string `generative_overhead`
2. `schema_version` integer `1`
3. `record_class` `summary`
4. `campaign_id` string
5. `campaign_manifest_sha256` 64-hex
6. `protocol_document_sha256` 64-hex
7. `fixtures_sha256` 64-hex
8. `source_commit` 40-hex
9. `aggregate_sha256` 64-hex of exact `analysis/aggregate.json` bytes
10. `eligibility_recomputed` boolean
11. `complete_matrix` boolean
12. `c01_mean_difference_ns` number or null
13. `c01_ci_low_ns` number or null
14. `c01_ci_high_ns` number or null
15. `c01_absolute_threshold_met` boolean
16. `c01_relative_threshold_met` boolean
17. `winner_declared` boolean
18. `superiority_claimed` boolean
19. `claim_status` enum
20. `secondary_cells_statement` string `secondary cells are descriptive only`
21. `l01_statement` string `L01 is descriptive filesystem-cache-warm package/session load versus minimal direct load`
22. `rejection_codes` the same sorted array as the aggregate

Ineligible: C01 mean/CI null; winners false; `claim_status=no_winner_ineligible`.
Eligible but no directional pass: `claim_status=no_winner_below_threshold_or_ci`.
No claim absent from the aggregate. `campaign_class` is bound through the manifest digest.

## Evidence lifecycle

1. Create a new `campaign_id` and empty evidence root.
2. Verify root and path-security prerequisites.
3. Construct, exclusively publish, and verify `campaign_manifest.json`.
4. Construct, exclusively publish, and verify `campaign_manifest.sha256`.
5. Publish and verify `campaign-start` using raw-before-JSON.
6. For each cell in golden order:
   a. Publish and verify `{cell_id}-pre` using raw-before-JSON.
   b. Create the required workers or processes (L01: new process per arm).
   c. Perform and publish the required warmups.
   d. Perform the 20 pairs in AB/BA order.
   e. For each pair, publish both arm records before correctness comparison.
   f. Apply the correctness and finish-policy gate.
   g. Shut down that cell’s workers or processes.
   h. Publish and verify `{cell_id}-post` using raw-before-JSON.
7. Publish and verify `campaign-end` using raw-before-JSON.
8. Confirm no failure, restart, or interruption occurred.
9. Validate Inventory A.
10. Require `analysis/` absent.
11. Exclusively create `analysis/` and fsync the evidence root.
12. Compute and publish `analysis/aggregate.json`.
13. Compute the SHA-256 of those exact aggregate bytes.
14. Compute and publish `analysis/summary.json`.
15. Validate Inventory B.
16. Run the read-only final verifier.
17. Only after successful final verification may the package be called publication-eligible.

Any missing, repeated, reordered, or interrupted phase makes the campaign non-publication.

## Dry-run firewall

A dry run uses a different `campaign_id`, a different evidence root, and `campaign_class=dry_run`. It is permanently non-publication. The analyzer refuses to merge dry-run and campaign data. Dry-run observations must not change endpoint, cells, pair count, seeds, thresholds, timing, or claim policy. A normative change to this protocol document requires incrementing the protocol document revision, a new `protocol_document_sha256`, and an independent audit before collection. That document revision is not the JSON schema version. Phase 9-B3B must not run a dry run or a publication campaign.

## Claim scope

The only primary claim is measured InferEdge integration overhead versus the pinned direct llama.cpp comparator for C01 on the frozen Qwen package, frozen host, frozen software, and a complete valid campaign, with the interval and integer thresholds.

Prohibited: model-quality improvement; training; online learning; general speedups; other models; Llama-family results; GPU; concurrency; energy; production-wide latency; TTFT-only, prefill-only, or decode-only improvements; independent-kernel superiority; “fastest”; production-ready.

## B3B source allowlist

Authorized source paths only:

- `docs/phase-9-b3-generative-overhead-protocol.md`
- `models/llm/generative_overhead_fixtures.json`
- `models/llm/generative_overhead_fixtures.json.sha256`
- `include/inferedge/generative_overhead_schema.hpp`
- `src/generative_overhead_ie_worker.cpp`
- `src/generative_overhead_ref_worker.cpp`
- `benchmarks/collect_generative.py`
- `benchmarks/analyze_generative.py`
- `benchmarks/test_collect_generative.py`
- `benchmarks/test_analyze_generative.py`
- `CMakeLists.txt`
- `tests/generative_overhead_schema_test.cpp` only if the required schema and linkage tests cannot be placed in the authorized Python tests

No other source file. Schema constants and tests must use `generative-overhead-r7-inventory-v1`. B3B may write the protocol document, generate fixtures by tokenization only, implement schema/workers/collector/analyzer/tests, and wire CMake. B3B may not run model generation, run performance measurements, create a dry-run or campaign evidence root, modify files outside this list, commit, or push unless a later prompt separately authorizes a subset of those actions.
