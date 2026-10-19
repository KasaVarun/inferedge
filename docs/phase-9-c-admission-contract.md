PART I — COMPLETE R23 CONSOLIDATED CONTRACT

# InferEdge Phase 9-C Admission, Multi-Model Benchmarking, and Offline Adaptation Contract

This document is the complete R23 contract.

## 1. Purpose and non-goals

A model is admitted only when an accepted decision for it sits in the head set named by the release pin, the pin is named by the operator configuration, the configuration and the admission-root manifest are named by the authenticated terminal host trust state, and the revocation set does not tombstone that decision. A package manifest, a package-local hash, a package golden file, or a package approval field does not admit a model.

This contract does not add an HTTP service, a database, a frontend, a Python binding, CUDA, TensorRT, CoreML, Metal, or Vulkan. It does not add a shared ONNX session pool or an online learned policy. ONNX `adaptive_batching` is an early seal, not this policy. Campaign 001 is not a paired thread-4 label. Campaign 002 is not authorized. No speed, zero-overhead, shadow-benefit, or invariance result is published by this contract. Weights and GGUF bytes stay outside Git.

Phase 9-C1, after a later independent audit accepts this Part I, may create only `docs/phase-9-c-admission-contract.md` by copying this Part I. Phase 9-C1 does not change runtime code, download weights, execute a model, benchmark, enable telemetry, train, stage, commit, push, dispatch, or deploy.

## 2. Phase map

Each phase from 9-C1 through 9-C44 has one owner. No phase 9-C45 exists.

| Phase | Exact outputs |
|---|---|
| 9-C1 | `docs/phase-9-c-admission-contract.md` |
| 9-C2 | `include/inferedge/admission_record.hpp`, `src/admission_record.cpp`, `tests/admission_record_test.cpp`, `CMakeLists.txt` |
| 9-C3 | `docs/phase-9-c3-capability.md`, `admission/capability/vocabulary.record` |
| 9-C4 | `include/inferedge/tensor_canonical.hpp`, `src/tensor_canonical.cpp`, `tests/tensor_canonical_test.cpp`, `CMakeLists.txt` |
| 9-C5 | `include/inferedge/oracle_digest.hpp`, `src/oracle_digest.cpp`, `tests/oracle_digest_test.cpp`, `CMakeLists.txt` |
| 9-C6 | `include/inferedge/evidence_store.hpp`, `src/evidence_store.cpp`, `tests/evidence_store_test.cpp`, `CMakeLists.txt` |
| 9-C7 | `include/inferedge/ed25519_verify.hpp`, `src/ed25519_verify.cpp`, `tests/ed25519_verify_test.cpp`, `CMakeLists.txt` |
| 9-C8 | `.gitignore` line `benchmarks/generative_phase9c/`, the Class A paths in section 23 or the Class B paths in sections 24 and 25, and `admission/provenance/provenance-<id>.record`. No submission note and no pending decision |
| 9-C9 | `admission/notes/submit-note-<provenance-id>.record`, `admission/signature-records/submit-signature-<provenance-id>.record`, `admission/decisions/pending-<provenance-id>.record`, `benchmarks/generative_phase9c/signatures/submit-sigblob-<provenance-id>.bin`, `admission/signature-indexes/submit-sigindex-<provenance-id>.index` |
| 9-C10 | `tests/provenance_phase_boundary_test.cpp`, `CMakeLists.txt` |
| 9-C11 | `admission/notes/review-provenance-note-<id>.record`, `admission/signature-records/review-provenance-signature-<id>.record`, `admission/decisions/accept-provenance-<id>.record`, `benchmarks/generative_phase9c/signatures/review-provenance-sigblob-<id>.bin`, `admission/signature-indexes/review-provenance-sigindex-<id>.index` |
| 9-C12 | `admission/heads/<head-set-id>.record`, `admission/heads/<head-set-id>.members` |
| 9-C13 | `admission/candidates/candidate-<id>.record`, `admission/candidates/candidate-<id>.index` |
| 9-C14 | `admission/golden/golden-<id>.record`, `admission/golden/cases-<id>.record`, and the case, oracle, and template pairs in section 30 |
| 9-C15 | golden-review family `review-golden-note-<id>`, `review-golden-signature-<id>`, `review-golden-sigblob-<id>`, and `review-golden-sigindex-<id>`, `admission/decisions/pending-golden-<id>.record` |
| 9-C16 | `admission/decisions/accept-golden-<id>.record` |
| 9-C17 | `admission/heads/<head-set-id>.record`, `admission/heads/<head-set-id>.members` |
| 9-C18 | `admission/license/license-<id>.record`, `admission/license/license-<id>.index`, `benchmarks/generative_phase9c/license/<id>.bin` |
| 9-C19 | license-review family `review-license-note-<id>`, `review-license-signature-<id>`, `review-license-sigblob-<id>`, and `review-license-sigindex-<id>`, `admission/decisions/accept-license-<id>.record` |
| 9-C20 | `admission/heads/<head-set-id>.record`, `admission/heads/<head-set-id>.members` |
| 9-C21 | `admission/model-registry/registry-<id>.record` |
| 9-C22 | registry-review family `review-registry-note-<id>`, `review-registry-signature-<id>`, `review-registry-sigblob-<id>`, and `review-registry-sigindex-<id>`, `admission/decisions/accept-registry-<id>.record` |
| 9-C23 | `admission/heads/<head-set-id>.record`, `admission/heads/<head-set-id>.members` |
| 9-C24 | `docs/phase-9-c24-correctness.md` |
| 9-C25 | `docs/phase-9-c25-performance.md` |
| 9-C26 | `docs/phase-9-c26-claims.md` |
| 9-C27 | `admission/invariance/invariance-<id>.record` |
| 9-C28 | `admission/dataset/dataset-<id>.record` |
| 9-C29 | `docs/phase-9-c29-offline-training.md` |
| 9-C30 | `admission/policy/policy-<id>.record`, `admission/policy/action-map-<id>.record` |
| 9-C31 | policy-review family `review-policy-note-<id>`, `review-policy-signature-<id>`, `review-policy-sigblob-<id>`, and `review-policy-sigindex-<id>`, `admission/decisions/accept-policy-<id>.record` |
| 9-C32 | `admission/heads/<head-set-id>.record`, `admission/heads/<head-set-id>.members` |
| 9-C33 | `tests/withdrawal_test.cpp`, the direct and dependent fixture triples under `tests/fixtures/withdrawal/`, `CMakeLists.txt` |
| 9-C34 | `include/inferedge/trust_migration.hpp`, `src/trust_migration.cpp`, `tests/trust_migration_test.cpp`, `CMakeLists.txt`. The export operation in this phase receives `source_trust_admin_signer` and writes the imported lineage-origin family into the section 39 bundle. The import operation verifies that family and publishes those exact bytes to the section 4 trust-root paths before imported `current`. It also writes `import.transaction`. It does not sign the imported origin. It retains all ten section 39 files at their exact paths for the destination lifetime |
| 9-C35 | `tests/activation_record_test.cpp`, the activation fixture triple under `tests/fixtures/activation/`, `CMakeLists.txt` |
| 9-C36 | `include/inferedge/trust_service.hpp`, `src/trust_service.cpp`, `tests/trust_service_test.cpp`, `docs/phase-9-c36-trust-bundle.md`, `CMakeLists.txt`, generation-1 trust, configuration, empty revocation set, first authority registry and envelope, `admission/root/manifest-<id>.record`, and the native lineage-origin family in section 4. `initialize_native_generation` receives `native_bootstrap_signer` and publishes that family before native `current` and active status |
| 9-C37 | `docs/phase-9-c37-rollback.md`, `tests/rollback_test.cpp`, `CMakeLists.txt` |
| 9-C38 | `tests/trust_transition_test.cpp`, `CMakeLists.txt` |
| 9-C39 | `admission/governance/governance-<id>.record`, governance-review family `review-governance-note-<id>`, `review-governance-signature-<id>`, `review-governance-sigblob-<id>`, and `review-governance-sigindex-<id>`, plus governance-content family `governance-signature-<id>`, `governance-sigblob-<id>`, and `governance-sigindex-<id>`, `admission/decisions/accept-governance-<id>.record`, and the trust transition that pins it |
| 9-C40 | `include/inferedge/generative_session.hpp`, `src/generative_session.cpp`, `tests/generative_session_test.cpp`. This phase verifies the section 11 lineage branch. On Branch B it opens exactly `payload.record` and `package.record` under the verified migration id. It does not publish lineage files |
| 9-C41 | `src/generative_session.cpp`, `tests/generative_session_test.cpp` |
| 9-C42 | `docs/phase-9-c42-shadow.md` |
| 9-C43 | `docs/phase-9-c43-activation.md`. Zero or one `admission/activation/activation-<id>.record` when an equal invariance record is already current |
| 9-C44 | `include/inferedge/generative_session.hpp`, `src/generative_session.cpp`, `tests/generative_session_test.cpp` |

`CMakeLists.txt` edits only add the named sources to the existing `inferedge_tests` target, or, for 9-C7, to that same target. Phases 9-C40, 9-C41, and 9-C44 do not edit `CMakeLists.txt`. `src/generate_main.cpp`, `src/generate_cli.cpp`, `src/generative_overhead_ie_worker.cpp`, and `models/llm/README.md` are in no allowlist.

Head sets are ordered. H1 is the accepted provenance decision and precedes acquisition. H2 adds the accepted golden decision and precedes license content. H3 adds the accepted license decision. Registry content cites H3. H4 adds the accepted registry decision. Policy content cites H4. H5 adds the accepted policy decision. A content record cites a head set that already exists. Every head-set file uses `admission/heads/<head-set-id>.record` and `admission/heads/<head-set-id>.members`.

## 3. Root, path, and open rules

The trust service root, the admission root, and the evidence root are three caller-supplied directories. A contract path that begins with `benchmarks/generative_phase9c/` is resolved only under `evidence_root`. A contract path that begins with `admission/` is resolved only under `admission_root`. A contract path that begins with `trust_root/` is resolved only under the caller-supplied trust root; in the path list below, `trust_root/` is that directory. The loader and the importer do not search a working directory, a home directory, or a repository root. They do not list a directory to choose a file. They do not choose a file by name order or modification time. A file this contract does not name has no authority.

A relative path uses `/`, has no empty, `.`, or `..` segment, and does not start with `/`. The implementation opens with `O_NOFOLLOW`, rejects a symlink leaf, and rejects a symlink ancestor. The resolved path stays inside its declared root. Lexical path checks run before the open.

These roots are never read or written by any phase: `benchmarks/baselines/`, `benchmarks/concurrent_runs/`, `models/experiments/`, `benchmarks/scheduler_runs/`, and the campaign 001 evidence root. In this repository the evidence directory is `benchmarks/generative_phase9c/`.

The trust loader and the importer may open only these trust-root paths. Ordinary cold loading opens the four lineage-origin finals under section 11 before it reads `service_status`. It opens `import.transaction` only after section 11 has selected Branch A or Branch B. After Branch B, and only then, it opens exactly `migrations/<migration-id>/payload.record` and `migrations/<migration-id>/package.record`. Migration import may read `service_status`, `migration_high_water`, `nonce/<nonce-id>`, and `import.transaction`. Replay checking opens `nonce/<nonce-id>` by the nonce id from the authenticated payload. No path in this list is both load-gating and forbidden. An unexpected trust-root file is not opened and is not authority. Absence of one named path is only `ENOENT` from `open` with `O_NOFOLLOW` after every ancestor was opened with `O_NOFOLLOW` and was not a symlink. `EACCES`, `EPERM`, `ELOOP`, `EIO`, any other errno, a symlink, a non-regular file, a link count other than 1, unreadable bytes, and malformed bytes are not absence. For `import.transaction`, section 11 applies that result only after the lineage branch is selected. For `lineage-origin.record`, `ENOENT` is not native-origin proof.

| Path | Role |
|---|---|
| `trust_root/key/bootstrap-key-v1` | 32-byte bootstrap key |
| `trust_root/service_status` | operational service status |
| `trust_root/migration_high_water` | operational sequence high water |
| `trust_root/nonce/<nonce-id>` | consumed replay nonce |
| `trust_root/current` | untrusted locator until section 13 finishes; then the terminal trust bytes |
| `trust_root/current.partial` | compare-and-swap temporary for `current` only |
| `trust_root/migration_high_water.partial` | rename temporary for high water |
| `trust_root/service_status.partial` | rename temporary for service status |
| `trust_root/genesis/transition.record` | genesis transition |
| `trust_root/genesis/signature.bin` | genesis raw signature |
| `trust_root/genesis/signature.index` | genesis signature index |
| `trust_root/genesis/verification.record` | genesis verification record |
| `trust_root/transitions/<transition-id>/transition.record` | later transition |
| `trust_root/transitions/<transition-id>/signature.bin` | later raw signature |
| `trust_root/transitions/<transition-id>/signature.index` | later signature index |
| `trust_root/transitions/<transition-id>/verification.record` | later verification record |
| `trust_root/migrations/<migration-id>/package.record` | migration wrapper |
| `trust_root/migrations/<migration-id>/payload.record` | signed migration payload |
| `trust_root/migrations/<migration-id>/signature.bin` | migration raw signature |
| `trust_root/migrations/<migration-id>/signature.index` | migration signature index |
| `trust_root/migrations/<migration-id>/verification.record` | migration verification record |
| `trust_root/migrations/<migration-id>/files.record` | migration file list |
| `trust_root/migrations/<migration-id>/lineage-origin.record` | source-signed imported lineage-origin target |
| `trust_root/migrations/<migration-id>/lineage-origin.signature.bin` | source-signed imported lineage-origin signature |
| `trust_root/migrations/<migration-id>/lineage-origin.signature.index` | imported lineage-origin signature index |
| `trust_root/migrations/<migration-id>/lineage-origin.verification.record` | imported lineage-origin verification record |
| `trust_root/migrations/<abort-id>.abort.record` | migration abort |
| `trust_root/migrations/<receipt-id>.receipt.record` | import receipt |
| `trust_root/import.transaction` | one immutable import record; recovery state before activation and historical provenance after activation; not admission authority |
| `trust_root/lineage-origin.record` | one immutable lineage-origin record; native or imported; not a trust-state digest input |
| `trust_root/lineage-origin.signature.bin` | 64-byte lineage-origin signature |
| `trust_root/lineage-origin.signature.index` | lineage-origin signature index |
| `trust_root/lineage-origin.verification.record` | lineage-origin verification record |
| `trust_root/lineage-origin.record.partial` | rename temporary for the lineage-origin record only |
| `trust_root/lineage-origin.signature.partial` | rename temporary for the lineage-origin signature only |
| `trust_root/lineage-origin.index.partial` | rename temporary for the lineage-origin index only |
| `trust_root/lineage-origin.verification.partial` | rename temporary for the lineage-origin verification record only |
| `trust_root/import-staging/<migration-id>/<trust-relative-path>` | import write-phase temporary whose suffix is one final trust path in this table other than the bootstrap key and other than the eight lineage-origin paths |

`<transition-id>`, `<migration-id>`, `<abort-id>`, `<receipt-id>`, and `<nonce-id>` are `project-id` values. `current.partial`, `service_status.partial`, `migration_high_water.partial`, and `import-staging/` files are not trust authority. While answering a request, `load_admission` opens the four lineage-origin finals and then applies section 11. Branch A and Branch C open no migration-bundle file. Branch B opens `import.transaction` and then exactly `migrations/<migration-id>/payload.record` and `migrations/<migration-id>/package.record`. It does not open the four lineage partials, `import-staging/`, `files.record`, `signature.bin`, `signature.index`, `verification.record`, the four bundle-copy lineage files, abort records, receipt records, or any other migration-bundle file. An open failure is not absence. A name that is not in this table, including a case-folded lineage name, is not opened and is not authority. Sections 11, 13, 14, and 41 open only paths from this table.

Admission opens use only the path grammars later sections name. During the section 41 write phase, and not before it, import may also create `admission_root/import-staging/<migration-id>/<admission-relative-path>` where the suffix is one admission path named by the authenticated closure. That staging tree is not authority. `load_admission` does not open it. Import does not create evidence files.

## 4. Operational trust-service records

`service_status` keys: `schema_version`, `record_kind`, `status`, `record_sha256`. Path: `trust_root/service_status`. Cardinality is one. `status` is `empty`, `active`, `export_frozen`, or `retired`. Producer is 9-C36. Consumer is the loader and section 41. Signature is `none`. It is local operational state, not production admission authority, and it is not part of a trust-file digest. A missing file or an unknown status rejects the load.

`migration_high_water` keys: `schema_version`, `record_kind`, `high_water`, `record_sha256`. Path: `trust_root/migration_high_water`. Cardinality is one. `high_water` is a `uint64`. Producer and consumer are 9-C34. Signature is `none`. It is local operational state. An empty service uses `high_water=0`.

`replay_nonce` keys: `schema_version`, `record_kind`, `nonce_id`, `migration_id`, `record_sha256`. Path: `trust_root/nonce/<nonce-id>`. The payload field `replay_nonce` is that `nonce_id`. Cardinality is one per consumed nonce. Producer and consumer are 9-C34. Signature is `none`. It is local operational state. A repeated nonce rejects import.

`import_transaction` is the immutable record defined in section 41. Path: `trust_root/import.transaction`. Cardinality is one for the life of the destination. Signature is `none`. Producer is 9-C34. Consumers are 9-C34 recovery and the section 11 loader. Before activation it is incomplete-import recovery state. After activation it is historical provenance. It is local operational state, not production admission authority, and it is not a trust-file digest input. `destination_host_id` is a `project-id`, not a digest.

`lineage_origin` keys: `schema_version`, `record_kind`, `lineage_origin_id`, `destination_host_id`, `origin_kind`, `root_trust_id`, `lineage_root_trust_file_size`, `lineage_root_trust_file_sha256`, `migration_id`, `lineage_migration_payload_file_size`, `lineage_migration_payload_file_sha256`, `imported_trust_id`, `lineage_imported_trust_file_size`, `lineage_imported_trust_file_sha256`, `signer_key_source`, `signer_key_id`, `effective_date`, `record_sha256`. Path: `trust_root/lineage-origin.record`. Cardinality is one for the life of the destination. `lineage_origin_id` is the fixed project-id `destination-lineage-v1`. `origin_kind` is `native` or `imported`. `record_kind` is `lineage_origin`. `schema_version` is `1`. The file is mode `0444`, created with `O_CREAT|O_EXCL|O_NOFOLLOW`, and accepted only as a regular file with link count 1. Its signature family is the section 9 lineage-origin family. Producer is 9-C36 for `native` and 9-C34 for `imported`. Consumer is the section 11 loader. It is local operational state, not a trust-file digest input, and not admission authority by itself. A native record uses `migration_id=none`, `imported_trust_id=none`, both import sizes `0`, and both import digests equal to the zero digest. `signer_key_source` is `bootstrap_handle` and `signer_key_id` is `none`. `effective_date` equals the genesis `transition_date`. `root_trust_id`, `lineage_root_trust_file_size`, and `lineage_root_trust_file_sha256` name the generation-1 trust state. An imported record is built on the source before the package wrapper. It uses the payload's `migration_id`, the payload file's size and full-file digest as `lineage_migration_payload_file_size` and `lineage_migration_payload_file_sha256`, and the payload's `trust_id`, `trust_file_size`, and `trust_file_sha256` as `imported_trust_id`, `lineage_imported_trust_file_size`, and `lineage_imported_trust_file_sha256`. `destination_host_id` equals the payload `destination_host_id`. It does not contain a digest of `import.transaction` or of `package.record`. `root_trust_id` and its size and digest name the generation-1 trust state on the imported chain. `signer_key_source` is `blob`. `signer_key_id` is the `trust_admin` entry that verified the migration payload. `effective_date` equals the imported terminal `effective_date`. Any other combination, including `native` with a non-sentinel import field or `imported` with a sentinel import field, fails verification.

`service_status`, `migration_high_water`, `replay_nonce`, `import_transaction`, and `lineage_origin` are not trust-file digest inputs. Cold loading reads `service_status` after the section 3 open and then applies the section 11 transaction rule. Import reads status, high water, the nonce path, and `import.transaction` under section 41 before the write phase. A missing `service_status`, an unknown status, or a symlink rejects the operation that required the file.

## 5. Terminology and lexical classes


A structured file is the serialization in section 6. Its full-file digest is SHA-256 of every byte, including the checksum line and the final LF. Its self-checksum is SHA-256 of every byte before the `record_sha256=` line. That preimage includes the LF of the previous line and excludes the checksum line. A self-checksum is a digest of that prefix. It is not a digest of the file that contains the checksum line.

A raw blob is a regular file of at least 1 byte with a blob index, except a `visible_output` or `token_list_bytes` object that is the empty immediate-end-of-generation payload, whose size is 0. Its digest is SHA-256 of those bytes with nothing removed.

The zero digest is 64 ASCII `0` characters and means absence. The empty-byte digest is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. They are not interchangeable.

Current authorization requires decision state `accepted`, membership in the pinned head set, a matching subject digest, and no tombstone. Effective status `revoked_by_dependency_tombstone` removes current authorization. The historical decision file may still say `accepted`. Historical bytes stay on disk and do not create current authority.

`test_only=yes` marks a fixture. A loader that is not the named unit test rejects it. A fixture never admits a production model.

`project-id` matches `[a-z0-9][a-z0-9._-]{0,127}`, rejects a leading or trailing `.`, rejects any `..` sequence, and rejects `none` and `unknown`. `id-or-none` is `none` or a `project-id`. A compound id is a fixed prefix plus a component. The component is a `project-id` of length 1 through 96, and the whole compound id must itself be a `project-id`. One id identifies one path.

`uint` is decimal 0 through 4294967295 with no leading zero except `0`. `uint64` is decimal 0 through 18446744073709551615 with the same rule. `int32` is -2147483648 through 2147483647, with no leading zero and with `-0` rejected.

`int32-list` is `none` or 1 to 4096 comma-separated `int32` values. `uint64-list` is `none` or 1 to 8 comma-separated `uint64` values. No spaces. `sha256` is 64 lowercase hex digits. `digest-or-zero` is a `sha256` whose zero value is legal only for a zero-eligible field paired with `none`. `date` is Gregorian `YYYY-MM-DD` from `2020-01-01` through `2100-12-31`.

`filename` is 1 to 255 bytes matching `[A-Za-z0-9][A-Za-z0-9._-]{0,254}`. Case is preserved. `qwen3-0.6b-Q4_K_M.gguf` is legal. `upstream-name` is 1 to 128 bytes matching `[A-Za-z0-9][A-Za-z0-9._+/-]{0,127}`, does not start or end with `/`, and has no `//`, `.`, or `..` segment. `Qwen/Qwen3-0.6B` is a name, not a path.

`yes-no` is `yes` or `no`. `phase-id` is `9-c1` through `9-c44`. `pattern_hex` is 2 to 512 even lowercase hex digits. Decoded bytes use `0x2A` as the only metacharacter and match the whole tensor name. `name_hex` is 2 to 512 even lowercase hex digits decoding to shortest-form UTF-8, with no NUL and no overlong sequence.

Checked `uint64` addition computes the sum only when both addends are `uint64` and the mathematical sum is at most 18446744073709551615. Overflow rejects the operation. A filesystem read error rejects the operation.

`probe_result` and `second_result` are `int32` values returned by `llama_tokenize`. `required_token_count` is an `int32` from 1 through 2147483647. `INT32_MIN` is -2147483648. `INT32_MAX` is 2147483647. `migration_sequence` and `duration_ns` are `uint64`. A payload `replay_nonce` is a `project-id`. `applied_threads` is the `uint` 1 or 4.

## 6. Canonical serialization


Keys appear once, in the schema order for that record, as `key=value` plus LF. The last key is `record_sha256`. The file ends on that LF. No CR, space, tab, comment, or blank line is legal. The maximum size is 16384 bytes. `schema_version` is `1`. Duplicate keys, unknown keys, and missing keys reject the file. A symlink rejects the file. List files are structured files. Their element lines are sorted by the stated sort key, duplicates reject the file, and the checksum line follows the last element. Failure of any structured-file check rejects that file and does not replace `current`. Calendar addition converts `YYYY-MM-DD` to days since `2020-01-01`, adds the stated number of days, and converts back.

`telemetry_log` and `deletion_authorization` are the only records with more than one closed key order. The discriminator selects exactly one order from section 37 or section 21. A key that belongs to another order is unknown. A key required by the selected order is missing when absent. Every other record has one key order.

`cleanup_pending` and `cleanup_completion` each have one key order.

## 7. Raw blobs and indexes

Accepted blob classes and maxima are: `publisher_checksum` 1048576, `public_key` exactly 32, `signature_bytes` exactly 64, `source_archive` 8589934592, `converter_binary` 8589934592, `build_recipe` 65536, `parameter_blob` 1048576, `reproduction_output_a` 2147483648, `reproduction_output_b` 2147483648, `license_bytes` 262144, `prompt_body` 65536, `oracle_program` 65536, `template_bytes` 65536, `visible_output` 16777216, `token_list_bytes` 65536, `fixture_bytes` 65536, `telemetry_log_bytes` 1024. `recipe_echo` is not a class. Minimum size is 1 byte except a `visible_output` or `token_list_bytes` object that is the empty immediate-end-of-generation payload.

`parameter_blob` bytes are ASCII lines `name=<project-id>\nvalue=<uint64 or sha256 or project-id>\n`, sorted by name, ending in LF.

`blob_index` keys: `schema_version`, `record_kind`, `blob_id`, `blob_class`, `raw_size`, `raw_sha256`, `storage_path`, `review_note_id`, `review_note_file_size`, `review_note_file_sha256`, `record_sha256`.

`review_note_id=none` and a zero `review_note_file_sha256` are legal only for `public_key`, `signature_bytes`, and `fixture_bytes`. Every other class cites one review note. The note does not cite another note. A blob index is not a signed object. A symlink, an omission of either side of a pair, a duplicate path, or a file outside the closed directory set fails that object. The migration-payload signature index is the one `signature_bytes` index whose `storage_path` is `migrations/<migration-id>/signature.bin` under `trust_root`. The lineage-origin signature index is the one `signature_bytes` index whose `storage_path` is `lineage-origin.signature.bin`, whose `review_note_id` is `none`, whose review size is 0, and whose review digest is the zero digest. Every other `signature_bytes` index uses an evidence-signature path from section 10.

## 8. Digest and size matrix

A semantic digest that names an indexed raw blob equals that blob's `raw_sha256`. The semantic size equals `raw_size`. A mismatch rejects the object. Hex text, a path, the index file, and structured metadata are not the raw bytes. Required equalities: `signature_sha256` equals `raw_sha256` and both sizes are 64; `public_key_sha256` equals `raw_sha256` and `public_key_size` equals `raw_size` equals 32; `actor_key_sha256` equals `raw_sha256` and `actor_key_size` equals `raw_size` equals 32; `checksum_raw_sha256` equals `raw_sha256` and `checksum_size` equals `raw_size`; `parameter_sha256` equals `raw_sha256` and `parameter_size` equals `raw_size`; `gguf_sha256` equals `raw_sha256` and `gguf_size` equals `raw_size`; `archive_sha256` equals `raw_sha256` and `archive_size` equals `raw_size`; `binary_sha256` equals `raw_sha256` and `binary_size` equals `raw_size`; `build_recipe_sha256` equals `raw_sha256` and `build_recipe_size` equals `raw_size`; `output_sha256` equals `raw_sha256` and `output_size` equals `raw_size`; `license_raw_sha256` equals `raw_sha256` and `license_byte_size` equals `raw_size`; `log_raw_sha256` equals `raw_sha256` and `log_size` equals `raw_size`; `input_raw_sha256` equals `raw_sha256` and `input_size` equals `raw_size`; `list_raw_sha256` equals the named source file's raw digest and the line `byte_size` equals that file's raw size; `reserved_object_sha256` equals the staged byte string and `reserved_object_size` equals that string's size; `deletion_entry_sha256` equals the listed file and `deletion_entry_size` equals that file's size; `payload_file_sha256` equals the full migration payload and `payload_file_size` equals that file's size; `tensor_schema_file_sha256` equals the full tensor schema and `tensor_schema_file_size` equals that file's size; `pending_file_sha256` equals the full cleanup-pending file and `pending_file_size` equals that file's size.

The size column names one exact field. The zero-size sentinel is integer 0. It is legal only together with id `none` and the zero digest. Validation order is: (1) lexical class, (2) paired size, (3) self-checksum, (4) full-file digest, (5) raw digest, (6) signature, (7) role, revocation, and verification-context date, (8) tombstone filter, (9) head-set membership. Kind `self` hashes every byte before the target file's `record_sha256=` line. Kind `full` hashes the complete file. Kind `raw` hashes the raw bytes.

The matrix has one row for every `_sha256` field in this contract. `record_sha256` is on every structured schema. It is not zero-eligible.

| Field | Schemas | Target | Kind | Bytes hashed | Size field | Index field | Zero |
|---|---|---|---|---|---|---|---|
| `record_sha256` | every structured schema | same file | self | bytes before its line | parent reference size | none | no |
| `raw_sha256` | `blob_index` | blob file | raw | exact blob bytes | `raw_size` | `blob_id` | no |
| `review_note_file_sha256` | `blob_index` | review note | full | complete note file | `review_note_file_size` | `review_note_id` | yes with `none` |
| `checksum_raw_sha256` | `publisher_authentication`, `provenance` | publisher checksum blob | raw | exact checksum bytes | `checksum_size` | checksum blob id | no |
| `checksum_index_sha256` | `publisher_authentication`, `binding` | checksum blob index | self | index preimage | `checksum_index_size` | `checksum_index_id` | no |
| `public_key_sha256` | `publisher_authentication`, `signature_verification`, authority entry line | 32 raw key bytes | raw | those 32 bytes | `public_key_size` | key blob or bootstrap handle | no |
| `fingerprint_sha256` | `signature_verification` | same 32 raw key bytes | raw | those 32 bytes | `public_key_size` | same key identity | no |
| `signature_file_sha256` | `publisher_authentication`, `trust_transition`, `migration_package` | signature-verification file | full | complete file | `signature_file_size` | `signature_id` or verification path | no |
| `signature_sha256` | `signature_verification`, `migration_package` | 64 signature bytes | raw | those 64 bytes | `signature_size` | signature blob id or signature path | no |
| `signature_index_sha256` | `signature_verification` | signature blob index | self | index preimage | `signature_index_size` | `signature_index_id` | no |
| `signed_object_sha256` | `signature_verification` | object named by `signed_object_kind` | raw or full | raw blob, or complete structured file | `signed_object_size` | `signed_object_id` | no |
| `authentication_file_sha256` | `binding`, `source_manifest`, `provenance` | publisher authentication file | full | complete file | `authentication_file_size` | authentication id | no |
| `artifact_sha256` | `binding` | fetched artifact bytes | raw | exact fetched bytes | `artifact_size` | `artifact_filename` | no |
| `note_file_sha256` | `binding`, `decision` | note file | full | complete note | `note_file_size` | `note_id` | no |
| `note_signature_file_sha256` | `binding`, `decision` | note signature-verification file | full | complete file | `note_signature_file_size` | note signature id | no |
| `binding_file_sha256` | `binding_list`, `provenance` | binding file | full | complete binding | `binding_file_size` | `binding_id` | no |
| `predecessor_registry_file_sha256` | `authority_registry` | previous registry content | full | complete file | `predecessor_registry_file_size` | predecessor id | yes with `none` |
| `entry_list_file_sha256` | `authority_registry` | entry list | full | complete list | `entry_list_file_size` | `entry_list_id` | no |
| `revocation_file_sha256` | `trust_state`, `migration_payload` | revocation set | full | complete set | `revocation_file_size` | revocation id | no |
| `configuration_file_sha256` | `trust_state`, `migration_payload` | operator configuration | full | complete file | `configuration_file_size` | config id | no |
| `authority_registry_file_sha256` | `trust_state`, `trust_transition`, `decision`, `migration_payload` | registry content | full | complete registry | `authority_registry_file_size` | registry id | no |
| `authority_envelope_file_sha256` | `trust_state`, `decision`, `migration_payload` | registry signature-verification file | full | complete envelope | `authority_envelope_file_size` | envelope id | no |
| `bootstrap_key_sha256` | `trust_state`, `trust_transition` | 32 service-held key bytes | raw | those 32 bytes | `bootstrap_key_size` | `bootstrap_key_handle` | no |
| `telemetry_governance_file_sha256` | `trust_state`, `operator_configuration`, `migration_payload` | governance file | full | complete file | `telemetry_governance_file_size` | `telemetry_governance_id` | yes with `none` and mode `disabled` |
| `predecessor_file_sha256` | `trust_state` | previous trust file | full | complete file | `predecessor_file_size` | previous trust id | yes at generation 1 |
| `admission_root_manifest_file_sha256` | `trust_state`, `migration_payload` | admission-root manifest | full | complete manifest | `admission_root_manifest_file_size` | `admission_manifest_id` | no |
| `successor_trust_file_sha256` | `trust_transition` | successor trust file | full | complete file | `successor_trust_file_size` | `successor_trust_id` | no |
| `predecessor_transition_file_sha256` | `trust_transition` | predecessor transition file | full | complete predecessor transition | `predecessor_transition_file_size` | predecessor transition id | yes when predecessor id is `none` |
| `predecessor_successor_trust_file_sha256` | `trust_transition` | predecessor successor trust state | full | complete predecessor trust state | `predecessor_successor_trust_file_size` | predecessor successor trust id | yes when predecessor id is `none` |
| `terminal_transition_file_sha256` | `trust_state` | terminal transition file | full | complete terminal transition | `terminal_transition_file_size` | `terminal_transition_id` | no |
| `key_index_sha256` | `signature_verification`, authority entry line | key index file | full | complete index file | `key_index_size` | `key_index_id` | yes for `bootstrap_handle` |
| `pin_file_sha256` | `operator_configuration` | release pin | full | complete pin | `pin_file_size` | `pin_id` | yes with `none` and `no_model` |
| `activation_decision_file_sha256` | `operator_configuration` | activation decision | full | complete decision | `activation_decision_file_size` | activation decision id | yes with `none` and `activation_mode=inactive` |
| `head_set_file_sha256` | `release_pin`, `registry_entry`, `policy` | head set | full | complete head set | `head_set_file_size` | head-set id | no |
| `member_list_file_sha256` | `head_set` | member list | full | complete list | `member_list_file_size` | member-list id | no |
| `decision_file_sha256` | `head_member_list` | decision file | full | complete decision | `decision_file_size` | decision id | no |
| `accepted_decision_file_sha256` | `revocation_set` | tombstoned accepted decision | full | complete decision | `accepted_decision_file_size` | decision id | no |
| `subject_file_sha256` | `decision`, `review_note` | subject content | full | complete subject | `subject_file_size` | subject id | no |
| `predecessor_decision_file_sha256` | `decision` | predecessor decision | full | complete decision | `predecessor_decision_file_size` | predecessor id | yes with `none` |
| `pending_decision_file_sha256` | `review_note` | pending decision | full | complete decision | `pending_decision_file_size` | pending id | yes for `submit`, `source_binding`, and `withdraw` |
| `actor_key_sha256` | `review_note` | actor 32 raw key bytes | raw | those 32 bytes | `actor_key_size` | actor key blob | no |
| `expected_gguf_sha256` | `provenance` | expected GGUF | raw | exact GGUF bytes | `expected_gguf_size` | candidate blob id | no |
| `binding_list_file_sha256` | `provenance` | binding list | full | complete list | `binding_list_file_size` | binding-list id | no |
| `source_manifest_file_sha256` | `provenance`, `reproduction` | source manifest | full | complete manifest | `source_manifest_file_size` | manifest id | yes for Class A provenance |
| `tensor_schema_file_sha256` | `provenance` | tensor schema | full | complete schema file | `tensor_schema_file_size` | `tensor_schema_id` | yes for Class A with id `none` |
| `provenance_file_sha256` | `candidate_record` | provenance content | full | complete provenance | `provenance_file_size` | provenance id | no |
| `provenance_decision_file_sha256` | `license`, `golden`, `registry_entry` | accepted provenance decision | full | complete decision | `provenance_decision_file_size` | decision id | no |
| `license_raw_sha256` | `license` | license blob | raw | exact license bytes | `license_byte_size` | `license_blob_id` | no |
| `license_index_sha256` | `license` | license blob index | self | index preimage | `license_index_size` | `license_index_id` | no |
| `golden_decision_file_sha256` | `registry_entry` | accepted golden decision | full | complete decision | `golden_decision_file_size` | decision id | no |
| `license_decision_file_sha256` | `registry_entry` | accepted license decision | full | complete decision | `license_decision_file_size` | decision id | no |
| `gguf_sha256` | `registry_entry`, `candidate_record` | candidate GGUF | raw | exact GGUF bytes | `gguf_size` | gguf blob id | no |
| `candidate_gguf_sha256` | `golden` | candidate GGUF | raw | exact GGUF bytes | `candidate_gguf_size` | candidate id | no |
| `candidate_index_sha256` | `candidate_record`, `golden` | candidate blob index | self | index preimage | `candidate_index_size` | candidate index id | no |
| `candidate_record_file_sha256` | `golden`, `activation_authorization` | candidate record | full | complete record | `candidate_record_file_size` | candidate id | no |
| `case_list_file_sha256` | `golden`, `dataset_record` | golden case list | full | complete case list | `case_list_file_size` | case-list id | no |
| `vocabulary_file_sha256` | `golden_case_list` | vocabulary capability file | full | complete file | `vocabulary_file_size` | `vocabulary_id` | no |
| `case_prompt_sha256` | `golden_case_list` | prompt blob | raw | exact prompt bytes | `prompt_size` | prompt blob id | no |
| `case_prompt_index_sha256` | `golden_case_list` | prompt blob index | self | index preimage | `case_prompt_index_size` | prompt index id | no |
| `case_parameter_sha256` | `golden_case_list` | parameter blob | raw | exact parameter bytes | `parameter_size` | parameter blob id | no |
| `case_parameter_index_sha256` | `golden_case_list` | parameter blob index | self | index preimage | `case_parameter_index_size` | parameter index id | no |
| `case_expected_tokens_sha256` | `golden_case_list` | expected token blob | raw | exact token-list bytes | `token_size` | token blob id | no |
| `case_expected_tokens_index_sha256` | `golden_case_list` | token blob index | self | index preimage | `case_expected_tokens_index_size` | token index id | no |
| `case_expected_visible_sha256` | `golden_case_list` | expected visible blob | raw | exact visible bytes | `visible_size` | visible blob id | no |
| `case_expected_visible_index_sha256` | `golden_case_list` | visible blob index | self | index preimage | `case_expected_visible_index_size` | visible index id | no |
| `oracle_index_sha256` | `golden` | oracle blob index | self | index preimage | `oracle_index_size` | oracle index id | no |
| `template_index_sha256` | `golden` | template blob index | self | index preimage | `template_index_size` | template index id | no |
| `rule_list_file_sha256` | `tensor_schema` | tensor rule list | full | complete rule list | `rule_list_file_size` | rule-list id | no |
| `inventory_file_sha256` | `tensor_schema` | tensor inventory | full | complete inventory | `inventory_file_size` | inventory id | no |
| `histogram_file_sha256` | `tensor_schema` | tensor histogram | full | complete histogram | `histogram_file_size` | histogram id | no |
| `file_list_file_sha256` | `source_manifest`, `migration_payload`, `migration_package` | file list | full | complete list | `file_list_file_size` | list id | no |
| `list_raw_sha256` | `source_file_list` | one source file | raw | that file's bytes | line `byte_size` | line blob id | no |
| `converter_file_sha256` | `recipe` | converter record | full | complete converter | `converter_file_size` | converter id | no |
| `parameter_sha256` | `recipe` | parameter blob | raw | exact parameter bytes | `parameter_size` | `parameter_blob_id` | no |
| `parameter_index_sha256` | `recipe` | parameter blob index | self | index preimage | `parameter_index_size` | `parameter_index_id` | no |
| `archive_sha256` | `recipe` | source archive blob | raw | exact archive bytes | `archive_size` | archive blob id | no |
| `archive_index_sha256` | `recipe` | archive blob index | self | index preimage | `archive_index_size` | archive index id | no |
| `binary_sha256` | `recipe` | converter binary blob | raw | exact binary bytes | `binary_size` | binary blob id | no |
| `binary_index_sha256` | `recipe` | binary blob index | self | index preimage | `binary_index_size` | binary index id | no |
| `build_recipe_sha256` | `recipe` | build-recipe blob | raw | exact recipe bytes | `build_recipe_size` | recipe blob id | no |
| `build_recipe_index_sha256` | `recipe` | build-recipe blob index | self | index preimage | `build_recipe_index_size` | recipe index id | no |
| `recipe_file_sha256` | `reproduction` | recipe record | full | complete recipe | `recipe_file_size` | recipe id | no |
| `output_sha256` | `reproduction` | reproduction GGUF | raw | exact output bytes | `output_size` | output blob id | no |
| `output_index_sha256` | `reproduction` | output blob index | self | index preimage | `output_index_size` | output index id | no |
| `evidence_manifest_file_sha256` | `reproduction` | evidence manifest | full | complete manifest | `evidence_manifest_file_size` | manifest id | no |
| `payload_list_file_sha256` | `evidence_manifest` | payload list | full | complete list | `payload_list_file_size` | list id | no |
| `digest_sha256` | `payload_list` | object named by the line path | raw | that file's raw bytes | line `byte_size` | line object id | no |
| `migration_entry_file_sha256` | `migration_file_list` | listed migration file | full or raw | exact listed bytes | line `byte_size` | line relative path | no |
| `payload_file_sha256` | `migration_package`, `import_transaction` | migration payload | full | complete `payload.record` | `payload_file_size` | payload id | no |
| `payload_signature_index_file_sha256` | `migration_package` | migration signature index | full | complete `signature.index` | `payload_signature_index_file_size` | signature index path | no |
| `accepted_run_manifest_file_sha256` | `quarantine_metadata` | run manifest | full | complete manifest | `accepted_run_manifest_file_size` | run id | yes with path `none` |
| `log_raw_sha256` | `run_manifest` | telemetry log | raw | exact log bytes | `log_size` | log blob id | yes when no log exists |
| `trust_file_sha256` | `migration_payload`, `import_receipt`, `import_transaction` | trust file that becomes `current` | full | complete trust file | `trust_file_size` | trust id | no |
| `migration_package_file_sha256` | `migration_abort`, `import_receipt`, `import_transaction` | migration package wrapper | full | complete `package.record` | `migration_package_file_size` | migration id | no |
| `prior_status_file_sha256` | `import_transaction` | pre-import `service_status` | full | those exact bytes | `prior_status_file_size` | none | no |
| `prior_high_water_file_sha256` | `import_transaction` | pre-import `migration_high_water` | full | those exact bytes | `prior_high_water_file_size` | none | no |
| `lineage_root_trust_file_sha256` | `lineage_origin` | generation-1 trust state | full | that trust file | `lineage_root_trust_file_size` | `root_trust_id` | no |
| `lineage_migration_payload_file_sha256` | `lineage_origin` | migration payload, or absence for native | full | `payload.record`, or the zero digest when size is 0 | `lineage_migration_payload_file_size` | `migration_id` | native sentinel |
| `origin_record_file_sha256` | `migration_package` | imported lineage-origin target | full | exact `lineage-origin.record` bytes | `origin_record_file_size` | `destination-lineage-v1` | no |
| `origin_signature_file_sha256` | `migration_package` | imported lineage-origin signature | raw | exact 64 signature bytes | `origin_signature_size` | fixed signature path | no |
| `origin_index_file_sha256` | `migration_package` | imported lineage-origin index | full | exact index bytes | `origin_index_file_size` | fixed index path | no |
| `origin_verification_file_sha256` | `migration_package` | imported lineage-origin verification record | full | exact verification bytes | `origin_verification_file_size` | fixed verification path | no |
| `lineage_imported_trust_file_sha256` | `lineage_origin` | imported trust file, or absence for native | full | those bytes, or the zero digest when size is 0 | `lineage_imported_trust_file_size` | `imported_trust_id` | native sentinel |
| `entry_file_sha256` | `admission_root_manifest` | one pinned admission file | full | complete file | `entry_size` | entry id | no |
| `policy_decision_file_sha256` | `activation_authorization` | accepted policy decision | full | complete decision | `policy_decision_file_size` | decision id | no |
| `invariance_file_sha256` | `activation_authorization`, `dataset_record` | invariance record | full | complete record | `invariance_file_size` | invariance id | no |
| `dataset_file_sha256` | `activation_authorization` | dataset record | full | complete dataset record | `dataset_file_size` | `dataset_id` | no |
| `registry_decision_file_sha256` | `policy` | accepted registry decision | full | complete decision | `registry_decision_file_size` | decision id | no |
| `action_map_file_sha256` | `policy` | action map | full | complete map | `action_map_file_size` | map id | no |
| `rendered_prompt_sha256` | `invariance_record` | rendered prompt bytes | raw | exact rendered bytes | `rendered_size` | none | no |
| `prompt_tokens_sha256` | `invariance_record` | canonical token-list bytes | raw | ASCII `int32-list` plus LF | `prompt_tokens_size` | none | no |
| `generated_tokens_sha256` | `invariance_record` | canonical token-list bytes | raw | those bytes, or empty bytes | `generated_tokens_size` | none | no |
| `visible_output_sha256` | `invariance_record` | visible output bytes | raw | exact visible bytes | `visible_size` | none | no |
| `target_file_sha256` | `deletion_authorization` | deletion target | full | complete target bytes | `target_file_size` | target id | no |
| `deletion_list_file_sha256` | `deletion_authorization`, `cleanup_pending`, `cleanup_completion` | reservation deletion list | full | complete list | `deletion_list_file_size` | deletion list id | no |
| `deletion_entry_sha256` | `reservation_deletion_list` | one listed file | full or raw | exact listed bytes | `deletion_entry_size` | line relative path | no |
| `lock_file_sha256` | `stale_lock_authorization` | evidence lock file | full | complete lock file | `lock_file_size` | lock owner run id | no |
| `authorization_file_sha256` | `deletion_journal`, `cleanup_pending`, `cleanup_completion` | deletion authorization | full | complete authorization | `authorization_file_size` | authorization id | no |
| `pending_file_sha256` | `cleanup_completion` | cleanup pending record | full | complete pending file | `pending_file_size` | authorization id | no |
| `reserved_object_sha256` | `evidence_reservation` | one planned byte string | raw or full | those exact bytes | `reserved_object_size` | staging path | no |
| `policy_file_sha256` | `telemetry_log` | policy content | full | complete policy file | `policy_file_size` | policy id | no |
| `input_raw_sha256` | `withdrawal_expected`, `activation_expected` | fixture blob | raw | exact fixture bytes | `input_size` | `input_blob_id` | no |
| `input_index_sha256` | `withdrawal_expected`, `activation_expected` | fixture blob index | self | index preimage | `input_index_size` | `input_index_id` | no |

Token-list canonical bytes are the ASCII `int32-list` plus one LF. Immediate end of generation uses `generated_tokens=none`, size 0, and `generated_tokens_sha256` equal to the empty-byte digest. An empty visible result uses that same empty-byte digest.

## 9. Signed-target families and trusted dates

`signature_verification` keys: `schema_version`, `record_kind`, `signature_id`, `signed_object_kind`, `signed_object_id`, `signed_object_path`, `signed_object_size`, `signed_object_sha256`, `signature_blob_id`, `signature_index_id`, `signature_index_path`, `signature_index_size`, `signature_index_sha256`, `signature_size`, `signature_sha256`, `signer_key_source`, `signer_key_id`, `public_key_size`, `public_key_sha256`, `key_index_id`, `key_index_size`, `key_index_sha256`, `fingerprint_sha256`, `algorithm`, `verification_result`, `verification_date`, `verifier_id`, `record_sha256`. When `signer_key_source` is `bootstrap_handle`, `signer_key_id` is `none`, `key_index_id` is `none`, `key_index_size` is 0, and `key_index_sha256` is the zero digest. When `signer_key_source` is `blob`, `key_index_id` is the authority `entry_id`, the index path is `admission/keys/<entry-id>.index`, and the raw key path is `benchmarks/generative_phase9c/keys/<entry-id>.bin`. When `signer_key_source` is `publisher_key`, `signer_key_id` is the publisher `public_key_blob_id` and the key path is the Class A or Class B public-key path in section 23 or section 24.

`signed_object_kind` is `raw_blob` or `structured_file`. `algorithm` is `ed25519`. `signature_size` is 64. `public_key_size` is 32. `verification_result` is `verified`. `verifier_id` is `inferedge-ed25519-v1`. `fingerprint_sha256` equals `public_key_sha256`. `signer_key_source` is `blob`, `publisher_key`, or `bootstrap_handle`. For every family except the migration payload, `signed_object_path` and `signature_index_path` are `none`. For the migration payload they are the two paths in section 39.

The Ed25519 preimage is exactly the signed object's bytes. The signature blob, the public key, and the verification file are outside that preimage. `signed_object_id` differs from `signature_id`. A verification record is not a signed object. No content record authorizes itself. No package wrapper approves itself.

`verification_date` is audit metadata. It must equal the `verification_context_date` defined below. It does not select that date and it does not determine validity. A mismatch rejects the verification record.

`verification_context_date` is one date, chosen as follows.

Genesis transition: registry validity checks do not apply. The fixed local bootstrap key is the trust anchor. The genesis verification record's `verification_date` equals the genesis `transition_date`. That equality is metadata. The bootstrap key has no `valid_from`, `valid_until`, or `revoked` field.

Later trust transition: verify the Ed25519 signature over the transition bytes before reading `transition_date` as trusted. The context date is that signed `transition_date`. Require the predecessor registry entry `valid_from <= transition_date <= valid_until` and `revoked=no`. Require `transition_date` to be greater than or equal to the predecessor transition's `transition_date`.

Evidence and admission signatures during current loading, including submission, provenance review, golden review, license review, registry review, policy review, governance review, governance content, activation review, Class A binding, Class B binding, authority-registry envelopes other than the installing check in section 15, deletion authorization, stale-lock authorization, migration abort, import receipt, and withdrawal: the context date is the authenticated terminal trust state's `effective_date`. That field is authenticated because section 12 requires it to equal the terminal transition's signed `transition_date`, and section 13 verifies that transition before the trust state is authoritative. Require `valid_from <= effective_date <= valid_until` and `revoked=no` in the current authenticated registry.

A new transition's own signature uses the later-transition rule above, not the terminal `effective_date` of the predecessor. Evidence signatures carried inside the new publication are checked against the new transition's `transition_date` after that signature verifies, and the published successor's `effective_date` equals that same `transition_date`.

Current revocation: a key with `revoked=yes` in the current authenticated registry cannot authorize current loading, even if the signature was valid on an earlier date. The historical signature bytes remain preserved and do not create current authority.

`publisher_key` is not a registry role. Class A and Class B checksum signatures are verified with the 32-byte publisher key named by the authentication record only after a `source_binding` note, signed by a registry-checked `provenance_reviewer`, covers that authentication file through the binding record's `authentication_file_size` and `authentication_file_sha256`. The publisher key has no registry validity interval. `genesis-v1` and the native lineage-origin record are the signatures that use the local bootstrap key directly. The first authority registry also uses that key under the rule below. An imported lineage-origin record does not use the bootstrap key.

Signed targets are exactly 22 families. Nineteen use storage class `evidence_signature`. One uses storage class `trust_transition_signature`. One uses storage class `migration_bundle_signature`. One uses storage class `lineage_origin_signature`. Trust-state content is not a signed family. A trust state is authenticated only as the successor bytes named by an already verified transition. The migration package wrapper is not a signed family.

| Family | Storage | Target id | Signature id | Blob id | Index id | Required role | Context date |
|---|---|---|---|---|---|---|---|
| Class A checksum | evidence | checksum blob | `checksum-signature-<provenance-id>` | `checksum-sigblob-<provenance-id>` | `checksum-sigindex-<provenance-id>` | `publisher_key` | binding note uses terminal `effective_date`; publisher key has no interval |
| Class B source checksum | evidence | source checksum blob | `source-checksum-signature-<provenance-id>` | `source-checksum-sigblob-<provenance-id>` | `source-checksum-sigindex-<provenance-id>` | `publisher_key` | same as Class A |
| Submission note | evidence | `submit-note-<provenance-id>` | `submit-signature-<provenance-id>` | `submit-sigblob-<provenance-id>` | `submit-sigindex-<provenance-id>` | `provenance_reviewer` | terminal `effective_date` |
| Provenance review | evidence | `review-provenance-note-<id>` | `review-provenance-signature-<id>` | `review-provenance-sigblob-<id>` | `review-provenance-sigindex-<id>` | `provenance_reviewer` | terminal `effective_date` |
| Golden review | evidence | `review-golden-note-<id>` | `review-golden-signature-<id>` | `review-golden-sigblob-<id>` | `review-golden-sigindex-<id>` | `golden_reviewer` | terminal `effective_date` |
| License review | evidence | `review-license-note-<id>` | `review-license-signature-<id>` | `review-license-sigblob-<id>` | `review-license-sigindex-<id>` | `license_reviewer` | terminal `effective_date` |
| Registry review | evidence | `review-registry-note-<id>` | `review-registry-signature-<id>` | `review-registry-sigblob-<id>` | `review-registry-sigindex-<id>` | `registry_reviewer` | terminal `effective_date` |
| Policy review | evidence | `review-policy-note-<id>` | `review-policy-signature-<id>` | `review-policy-sigblob-<id>` | `review-policy-sigindex-<id>` | `policy_reviewer` | terminal `effective_date` |
| Governance review | evidence | `review-governance-note-<id>` | `review-governance-signature-<id>` | `review-governance-sigblob-<id>` | `review-governance-sigindex-<id>` | `telemetry_reviewer` | terminal `effective_date` |
| Governance content | evidence | `governance-<governance-id>` | `governance-signature-<governance-id>` | `governance-sigblob-<governance-id>` | `governance-sigindex-<governance-id>` | `telemetry_reviewer` | terminal `effective_date` |
| Activation review | evidence | `review-activation-note-<activation-id>` | `review-activation-signature-<activation-id>` | `review-activation-sigblob-<activation-id>` | `review-activation-sigindex-<activation-id>` | `policy_reviewer` | terminal `effective_date` |
| Withdrawal note | evidence | `withdraw-note-<subject-id>` | `withdraw-signature-<subject-id>` | `withdraw-sigblob-<subject-id>` | `withdraw-sigindex-<subject-id>` | section 33 role for that subject kind | terminal `effective_date` |
| Class A binding note | evidence | `bind-a-note-<provenance-id>` | `bind-a-signature-<provenance-id>` | `bind-a-sigblob-<provenance-id>` | `bind-a-sigindex-<provenance-id>` | `provenance_reviewer` | terminal `effective_date` |
| Class B binding note | evidence | `bind-b-note-<provenance-id>` | `bind-b-signature-<provenance-id>` | `bind-b-sigblob-<provenance-id>` | `bind-b-sigindex-<provenance-id>` | `provenance_reviewer` | terminal `effective_date` |
| Authority registry | evidence | `registry-<registry-id>` | `registry-envelope-signature-<registry-id>` | `registry-envelope-sigblob-<registry-id>` | `registry-envelope-sigindex-<registry-id>` | `authority_admin`, or bootstrap for the first registry | first registry uses the genesis bootstrap rule below; a later registry uses the new transition's signed `transition_date` after that transition signature verifies |
| Deletion authorization | evidence | `deletion-<authorization-id>` | `deletion-signature-<authorization-id>` | `deletion-sigblob-<authorization-id>` | `deletion-sigindex-<authorization-id>` | `trust_admin` | terminal `effective_date` |
| Stale-lock authorization | evidence | `stale-<authorization-id>` | `stalelock-signature-<authorization-id>` | `stalelock-sigblob-<authorization-id>` | `stalelock-sigindex-<authorization-id>` | `trust_admin` | terminal `effective_date` |
| Migration abort | evidence | `abort-<abort-id>` | `abort-signature-<abort-id>` | `abort-sigblob-<abort-id>` | `abort-sigindex-<abort-id>` | `trust_admin` | terminal `effective_date` of the frozen source |
| Import receipt | evidence | `receipt-<receipt-id>` | `receipt-signature-<receipt-id>` | `receipt-sigblob-<receipt-id>` | `receipt-sigindex-<receipt-id>` | `trust_admin` | imported terminal `effective_date` |
| Migration payload | migration bundle | `payload-<migration-id>` | `migration-signature-<migration-id>` | fixed `signature.bin` | fixed `signature.index` | `trust_admin` | terminal `effective_date` at export, rechecked at import against the imported terminal `effective_date` |
| Trust transition | trust | the transition record | `transition-signature-<transition-id>` | fixed `signature.bin` | fixed `signature.index` | bootstrap for `genesis-v1`, otherwise `trust_admin` | genesis has no registry interval; later transitions use signed `transition_date` |
| Lineage origin | trust lineage | fixed `destination-lineage-v1` | `lineage-origin-signature` | fixed `lineage-origin.signature.bin` | fixed `lineage-origin.signature.index` | bootstrap for `native`; `trust_admin` for `imported` | native uses the genesis `transition_date`; imported uses the imported terminal `effective_date` |

The first authority registry uses the fixed bootstrap key that verified `genesis-v1`. Its context date is the genesis `transition_date`, and that date is metadata only. Registry `valid_from`, `valid_until`, and `revoked` do not apply to the bootstrap key. A later authority registry is installed only by a later trust transition. The installer verifies that new transition's Ed25519 signature first, trusts `transition_date` only after that verification, and uses that new `transition_date` as the registry-signature verification context. The registry signer is an unrevoked `authority_admin` from the authenticated predecessor registry. The check requires `valid_from <= new transition_date <= valid_until`. The new `transition_date` is not earlier than the predecessor transition's `transition_date`. The predecessor transition's date is not the installing verification date.

Evidence-signature family count is 19. Trust-transition signature family count is 1. Migration-bundle signature family count is 1. Lineage-origin signature family count is 1. Total signed-target family count is 22. The raw signature is 64 bytes. `signature_size` equals `raw_size` equals 64. `signature_sha256` equals the index `raw_sha256`. Omission, duplication, or reuse of one id on two paths fails.

## 10. Signature storage

Evidence storage, and only evidence storage, uses `benchmarks/generative_phase9c/signatures/<blob-id>.bin` under `evidence_root`, `admission/signature-indexes/<index-id>.index`, and `admission/signature-records/<signature-id>.record`. Notes use `admission/notes/<note-id>.record`. Those four directories contain only the 19 evidence grammars and the 11 note ids in section 9. Any other file there fails. There is no unnamed blob grammar. The migration payload signature is not stored in those directories.

Trust-transition storage is not in those directories. Genesis files are only `trust_root/genesis/transition.record`, `genesis/signature.bin`, `genesis/signature.index`, and `genesis/verification.record`. Later files are only `trust_root/transitions/<transition-id>/transition.record`, `signature.bin`, `signature.index`, and `verification.record`. The genesis transition id is the fixed project-id `genesis-v1`.

Lineage-origin storage is not in the evidence directories and not in a transition directory. Its only finals are `trust_root/lineage-origin.record`, `trust_root/lineage-origin.signature.bin`, `trust_root/lineage-origin.signature.index`, and `trust_root/lineage-origin.verification.record`. Its only partials are `trust_root/lineage-origin.record.partial`, `trust_root/lineage-origin.signature.partial`, `trust_root/lineage-origin.index.partial`, and `trust_root/lineage-origin.verification.partial`. There is no second name, no history directory, and no caller-chosen path segment. The Ed25519 preimage is the exact `lineage-origin.record` bytes, including `record_sha256`. The signature, index, and verification record are outside that preimage. The index is a `blob_index` with `blob_class=signature_bytes`, `raw_size=64`, and `storage_path=lineage-origin.signature.bin`. The verification record is a `signature_verification` with `signed_object_kind=structured_file`, `signed_object_id=destination-lineage-v1`, `signed_object_path=none`, `signature_id=lineage-origin-signature`, and `verification_result=verified`. Native verification uses `signer_key_source=bootstrap_handle` and the genesis `transition_date`, which is metadata only and is not a registry interval. Imported verification uses `signer_key_source=blob` and the `trust_admin` key that already verified the migration payload. That key is selected from the payload verification record, not from the lineage-origin record. The signature is checked against that key as it was authorized at `effective_date`. A later revocation does not rewrite the family, does not change `origin_kind`, and does not by itself move the destination to the other lineage. Model-admission signatures still follow the current-revocation rule. Publication writes each partial, `fsync`s that partial, renames it onto an absent final with `O_NOFOLLOW`, and `fsync`s the trust root before the next file. The order is record, signature, index, verification. All four files are mode `0444`. A load does not create or rename them. Withdrawal, rollback, export, retirement, staging cleanup, and a later trust transition do not open them for writing.

Migration-bundle storage is not in the evidence directories and not in the transition directories. Its only files are the ten paths in section 39.

## 11. Generation-1 bootstrap

`load_admission(trust_root, admission_root, evidence_root)` receives three caller-supplied roots. A contract path that begins with `benchmarks/generative_phase9c/` is resolved only under `evidence_root`. The loader opens only paths in section 3. It does not search a working directory, a home directory, or a repository root, and it does not list a directory to choose a trust file. It reads `service_status` from section 3 and rejects a missing file or an unknown status before treating the load as successful. Model use requires `status=active`. `empty`, `export_frozen`, and `retired` return before section 13 selects a model. The loader does not scan, and it does not open `import-staging/`.

The loader classifies the lineage-origin family before it classifies `import.transaction` and before it admits a model. Transaction absence is not native-origin proof. `ENOENT` for `lineage-origin.record` is not Branch A.

A truly empty destination has the exact snapshotted `empty` status, absent `current`, and absent `lineage-origin.record`. A model load then returns the empty-service result and admits no model. That exception ends as soon as `current` is present or `service_status` is any record other than that exact `empty` record. Initialization, not the loader, is the only writer of the lineage-origin family.

Branch A, authenticated native origin: all four finals are regular files with link count 1, the record verifies under sections 4 and 6, `origin_kind=native`, the import fields are the exact native sentinels, the signature family verifies under section 9 with the bootstrap key, and `destination_host_id` equals the generation-1 trust state's `host_id`. The section 13 walk contains the root trust tuple named by `root_trust_id`, `lineage_root_trust_file_size`, and `lineage_root_trust_file_sha256`. Every later transition is authorized, generations increase by exactly 1, `revocation_generation` does not decrease, and tombstones stay effective. The loader then opens `import.transaction` and opens no file under `migrations/`. Only `ENOENT` is acceptable. Any present transaction, including a verifying file, a malformed file, a symlink, a non-regular file, a link count other than 1, or an open error, fails closed. The loader does not reinterpret the destination as imported, does not start an import, does not load a model, and does not mutate a file. When the transaction is absent, canonical `active` status and sections 16, 17, and 18 may make `native_service_loadable` true. Native generation 1, native N+1, native N+2, and a later authorized native generation use this same immutable origin. `export_frozen` and `retired` are not loadable.

Branch B, authenticated imported origin: all four finals are regular files with link count 1, the record verifies, `origin_kind=imported`, no import field is a native sentinel, the signature family verifies under section 9 with the recorded `trust_admin` key, and `destination_host_id` equals the record. The loader then opens `import.transaction` and does not treat `ENOENT` as native. It then opens only `trust_root/migrations/<migration-id>/payload.record` and `trust_root/migrations/<migration-id>/package.record`. A missing transaction, a malformed transaction, a symlink, a non-regular file, a link count other than 1, an open error, a missing or mismatched payload, or a missing or mismatched package fails closed. The loader admits no model, does not enter Branch A, does not reconstruct the transaction, and does not modify `current`, status, high water, the nonce, or any lineage file. A transaction that verifies, names the same `migration_id` and imported trust tuple, and binds the package that binds that payload, continues under `import_current_present`, `import_completed`, and imported `service_loadable`. The section 13 walk must contain that imported tuple. Imported N, authorized N+1, and authorized N+2 remain loadable when status is canonical `active` and sections 16, 17, and 18 admit the model. The live digest need not equal the imported digest. `export_frozen` and `retired` can be `import_completed` and are not loadable.

Branch C, origin missing, invalid, or indeterminate: the four finals are not the Branch A family and are not the Branch B family. This includes `ENOENT` on a destination that is not the empty exception, a partial family, a malformed record, a bad self-checksum, an unknown key, a missing key, a duplicate key, noncanonical bytes, a wrong host inside the origin family, a root tuple whose file does not match the origin digest, an unauthorized signer, a swapped signature, a symlink, a directory, a non-regular file, a link count other than 1, an open error, and `origin_kind` disagreeing with its sentinel fields. A transaction mismatch is not Branch C when the imported origin family itself verifies. That mismatch stays in Branch B and fails closed there. The loader admits no model, starts no native load, starts no imported load, starts no fresh import, and resumes nothing. It creates nothing, deletes nothing, and publishes nothing. It returns one deterministic lineage-origin failure. A load never reconstructs the origin. Branch C opens no migration-bundle file.

Request-time sequence. Every load opens and verifies the four fixed lineage-origin finals first and selects Branch A, Branch B, or Branch C before it opens `import.transaction`. Branch A opens no migration-bundle file. Branch C opens no migration-bundle file and fails closed. Branch B opens and verifies `trust_root/import.transaction`, requires its `migration_id` to equal the signed origin `migration_id`, and derives the directory component only from that verified id. It then opens exactly `trust_root/migrations/<migration-id>/payload.record` and `trust_root/migrations/<migration-id>/package.record`. Imported completion and admission continue only after those two files verify. No other request-time open under `migrations/` is permitted.

Path construction uses the origin `migration_id` only after it satisfies the section 5 `project-id` grammar. An empty value, a leading or trailing `.`, any `..` sequence, a slash, a backslash, a NUL, a control character, a percent sign, an uppercase letter, or an absolute path is rejected. The comparison is exact and case-preserving. The implementation appends the fixed names `payload.record` and `package.record`. It does not read a path from a file, accept a caller path, use a glob, list a directory, or normalize the id into a different string.

The open is component-wise from the already validated trust-root descriptor. It opens `migrations` with `O_NOFOLLOW`, then the verified migration-id directory with `O_NOFOLLOW`. Each of those components must be a directory and not a symlink. Each file is opened with `O_RDONLY|O_NOFOLLOW|O_CLOEXEC` and must be a regular file of link count 1. The loader records device, inode, size, and mode, hashes through that descriptor, and `fstat`s again. Device, inode, and size must be unchanged. It does not reopen the path to hash, and it does not read a sibling. Any failure rejects the load and writes nothing.

`payload.record` is parsed under section 6, including the size maximum and the rejection of an unknown, duplicate, or missing key. Its self-checksum, full-file size, and full-file SHA-256 must match `lineage_migration_payload_file_size` and `lineage_migration_payload_file_sha256` and the transaction `payload_file_size` and `payload_file_sha256`. Its `migration_id` and `destination_host_id` must match the origin and the transaction. Its imported trust tuple must match the origin. The origin `effective_date` and `signer_key_id` already verified on the trust-root lineage family are the import authorization. The payload has no signer field, and this load does not open the bundle verification record to recover one. The payload `migration_id`, `destination_host_id`, and imported trust tuple must match that authorized origin. Any disagreement fails closed and writes nothing.

`package.record` is parsed under section 6. Its self-checksum, full-file size, and full-file SHA-256 must match the transaction `migration_package_file_size` and `migration_package_file_sha256`. Its `migration_id` must match the origin and the transaction. Its payload size and digest must equal the opened payload file. Its file-list id, path, size, and digest must equal the payload's file-list fields. Its four origin size and digest fields must equal the already verified trust-root lineage finals: `origin_record_file_size` and `origin_record_file_sha256` for the target, `origin_signature_size` and `origin_signature_file_sha256` for the raw signature, `origin_index_file_size` and `origin_index_file_sha256` for the index, and `origin_verification_file_size` and `origin_verification_file_sha256` for the verification record. No other wrapper is accepted. A package field that disagrees with the transaction or the origin fails closed.

The trust-root lineage family already opened for branch selection is the family the package must match. The loader does not reopen `migrations/<migration-id>/lineage-origin.record`, `lineage-origin.signature.bin`, `lineage-origin.signature.index`, or `lineage-origin.verification.record` during a load. The comparison uses the sizes and the full-file or raw digests defined in section 8, the fixed id `destination-lineage-v1`, the migration id, the destination host, the imported trust tuple, and the payload binding.

The comparison triangle is closed. The signed origin matches the opened payload. The transaction matches that payload and the opened package. The package matches that payload and the trust-root origin family. The origin migration id, the transaction migration id, and the package migration id are equal. The origin host equals the transaction host. The origin imported trust tuple equals the transaction imported trust tuple. The package payload fields equal the origin payload fields. Every size is checked with its SHA-256. The unsigned transaction does not grant authority by itself.

Request-time opens are exactly these. Branch A opens the fixed native lineage-origin family and the ordinary trust, current, and admission files required by sections 13, 16, 17, and 18, and no migration-bundle file. Branch B additionally opens the fixed imported lineage-origin family, `import.transaction`, and the two files above. During a request it does not open `files.record`, `signature.bin`, `signature.index`, `verification.record`, the bundle-copy origin target, the bundle-copy origin signature, the bundle-copy origin index, the bundle-copy origin verification record, an abort record, a receipt, a staging directory, or another migration directory. The loader does not list `migrations/`, search for a package, choose the newest directory, follow a path stored in the wrapper, accept another filename, or fall back to another migration id. An extra sibling directory is not authority. An unknown file does not replace a fixed opened file.

Cold-load results, with no write in any row:

| Case | Result |
|---|---|
| Native origin | Branch A, no bundle open |
| Imported origin and both retained files verify | payload and package checks pass |
| Payload absent | fail closed |
| Package absent | fail closed |
| Payload digest mismatch | fail closed |
| Package digest mismatch | fail closed |
| Package payload field mismatch | fail closed |
| Package origin field mismatch | fail closed |
| Transaction package mismatch | fail closed |
| Transaction payload mismatch | fail closed |
| Wrong migration id | fail closed |
| Symlinked migration directory | fail closed |
| Symlinked payload or package | fail closed |
| Hard-linked payload or package | fail closed |
| `EACCES`, `EPERM`, `EIO`, or any other open or read error | fail closed |
| Extra sibling migration directory | ignored and never selected |
| Missing transaction | Branch B, fail closed, no Branch A |

The origin branch stays bound to the descriptors opened for the lineage family. The transaction check stays bound to its opened descriptor. Payload and package hashes stay bound to their opened descriptors. After each read, `fstat` must show the same device, inode, and size. The loader keeps those descriptors through the admission decision and does not resolve the path again. A concurrent replacement of a directory entry does not change the bytes already verified on the descriptor.



For every load that is not the empty-destination exception, the three branches are exhaustive and mutually exclusive. The loader selects one before admission and does not evaluate `native_service_loadable` and imported `service_loadable` for the same load. A native chain and an imported chain are not spliced. Branch A accepts a walk only when it contains the native origin's root tuple. Branch B accepts a walk only when it contains the imported origin's imported tuple. Deleting `import.transaction` does not select Branch A while an imported origin remains. Deleting the origin family does not select Branch A.

Native initialization belongs to phase 9-C36 and is not a model load and not `import_migration`. The operation is `initialize_native_generation(trust_root, admission_root, evidence_root, native_bootstrap_signer)`, declared in `include/inferedge/trust_service.hpp`. `native_bootstrap_signer` is an opaque signing capability supplied by the caller. It exposes no private-key bytes and no private-key path. Its only signing operation accepts the exact message bytes and returns exactly 64 bytes or a failure. Those 64 bytes are an Ed25519 signature by the private key that corresponds to the 32-byte public key at `trust_root/key/bootstrap-key-v1`. The message is the exact canonical `lineage-origin.record` bytes, including `record_sha256`. No domain-separation prefix is added, matching section 9. The capability is not an on-disk trust root. After the origin family is durable, a load or a later transition does not call it. Loss of the handle does not invalidate a stored family.

A first start requires exact `empty` status, absent `current`, absent `import.transaction`, absent nonce, `migration_high_water` absent or the exact `high_water=0` record, all four lineage finals absent, all four lineage partials absent, a valid section 11 bootstrap and generation-1 trust state, and a usable signer. It constructs the native record, checks `record_sha256`, passes those exact bytes to the signer, and verifies the 64 returned bytes with `bootstrap-key-v1`. It then builds the index and verification record and reverifies the four objects in memory. Only then does it publish. If the handle is missing, the public verification fails, the length is not 64, or the signer fails, the result is `lineage_signer_unavailable` and no file is created. The genesis signature is not reused. The 32-byte public key is not used as a signing seed. No local key is invented.

A resume is not a first start. It requires the same empty status, absent transaction, absent nonce, and baseline high water. `current` stays absent until state N8. The expected native record bytes are derived again from the generation-1 trust state and its `host_id`. Every visible lineage final or partial must be the exact bytes for its place in the N0 through N8 prefix. Any other combination, including an imported `origin_kind`, a symlink, a link count other than 1, or a mixed sentinel, is `lineage_conflict`. The conflicting file stays in place and is not deleted, overwritten, or renamed. `current` and active status are not published, no model loads, and imported initialization does not start. A verifying final family is never replaced.

The publication prefixes are:

| State | Visible files | Signer | Next operation |
|---|---|---|---|
| N0 | no finals and no partials | required before any write | sign in memory, then write the target partial |
| N1 | exact target partial only | required, because the signature is not yet stored | `fsync` the partial if needed, rename it onto the absent target final, `fsync` the trust root |
| N2 | exact target final, no later member | required | write the exact signature partial |
| N3 | exact target final and exact signature partial | not required | rename that partial onto the absent signature final and `fsync` the trust root |
| N4 | exact target and signature finals | not required | write the exact index partial |
| N5 | those finals plus an exact index partial | not required | rename the index partial and `fsync` the trust root |
| N6 | exact target, signature, and index finals | not required | write the exact verification partial |
| N7 | those three finals plus an exact verification partial | not required | rename the verification partial and `fsync` the trust root |
| N8 | all four finals verify | not required | publish `current`, then active status |

Each write uses `O_CREAT|O_EXCL|O_NOFOLLOW` for a partial, mode `0444`, a file `fsync`, a rename onto an absent final, and a trust-root `fsync`. An exact partial is opened with `O_NOFOLLOW`, must be a regular file of link count 1, and must match the expected size and digest. It is renamed, not rewritten. If the signature final or an exact signature partial already verifies, the signer is not called. Ed25519 signing of the same message and key is deterministic for this capability. If a stored signature already verifies, those bytes remain even if a new call would be attempted. If the signature does not yet exist and the signer is unavailable, the result is `lineage_signer_unavailable` and no further file is written. The existing exact prefix stays.

| Crash point | Load | Resume | Prohibited |
|---|---|---|---|
| Before the target partial | empty result | N0, signer required | no file |
| After the target partial `fsync` | empty result | N1, and the signer is required before the rename because the signature is not stored | do not delete the partial |
| After the target rename and directory `fsync` | empty result | N2, signer required | do not replace the target |
| After the signature partial `fsync` | empty result | N3, signer not required | do not rewrite the signature |
| After the signature rename | empty result | N4 | do not replace the signature |
| After the index partial `fsync` | empty result | N5 | do not rewrite the index |
| After the index rename | empty result | N6 | do not replace the index |
| After the verification partial `fsync` | empty result | N7 | do not rewrite the verification record |
| After the verification rename and directory `fsync` | not loadable until active | N8 | do not replace the family |
| Origin complete, `current` absent | not loadable | publish `current` only | do not call the signer |
| `current` present, status still `empty` | not loadable | publish active status | do not rewrite origin or `current` |
| Active status | Branch A may load | no origin write | do not call the signer |

A model load does not create a native origin.


Bounds: at most 64 transitions; at most 16384 bytes for one transition, trust state, signature index, or verification record; exactly 64 signature bytes; exactly 32 public-key bytes; at most 4194304 bytes for every transition file, signature, index, verification record, and successor trust state opened by one load. `current` is read only inside the 16384-byte bound.

An immutable signed transition does not name a future successor. No transition field is a pointer to a later transition.

1. Read exactly `trust_root/key/bootstrap-key-v1`. This read does not use any field of `current` or of any trust state.
2. Require exactly 32 raw bytes. Hexadecimal, Base64, PEM, DER, and path text are not a key.
3. Open only `trust_root/genesis/transition.record`, `genesis/signature.bin`, `genesis/signature.index`, and `genesis/verification.record`.
4. Require the transition id `genesis-v1`, `trust_generation=1`, and predecessor id `none`.
5. Require predecessor size 0 and the zero predecessor digest.
6. Verify the 64-byte signature over the transition bytes with the bootstrap key before trusting `transition_date`.
7. Require the signature index `raw_size=64` and `raw_sha256` equal to the signature bytes.
8. Require the verification record to name this transition, this signature, and this bootstrap key.
9. Do not apply a registry validity interval to the bootstrap key.
10. Require `successor_trust_file_size` and `successor_trust_file_sha256` to match the generation-1 trust state.
11. Require that trust state's `effective_date` to equal the genesis `transition_date`.
12. Require the genesis verification record's `verification_date` to equal that `transition_date`.
13. Do not treat `trust_root/current` as authenticated at this step.

`bootstrap_key_handle` is `bootstrap-key-v1`. The bootstrap key is not a registry entry and is not an evidence file. Service status bytes are not part of the trust-file digest.

## 12. Immutable trust transitions

`trust_transition` keys: `schema_version`, `record_kind`, `transition_id`, `trust_generation`, `transition_date`, `predecessor_transition_id`, `predecessor_transition_path`, `predecessor_transition_file_size`, `predecessor_transition_file_sha256`, `predecessor_successor_trust_id`, `predecessor_successor_trust_file_size`, `predecessor_successor_trust_file_sha256`, `successor_trust_id`, `successor_trust_path`, `successor_trust_file_size`, `successor_trust_file_sha256`, `bootstrap_key_sha256`, `signer_key_id`, `registry_id`, `authority_registry_file_size`, `authority_registry_file_sha256`, `signature_id`, `signature_blob_id`, `signature_index_id`, `signature_file_size`, `signature_file_sha256`, `record_sha256`.

A generation greater than 1 requires `trust_generation` equal to the predecessor generation plus 1. `transition_date` is greater than or equal to the predecessor transition's `transition_date`. The predecessor path, size, and digest must equal the preceding transition file. The predecessor successor id, size, and digest must equal the trust state that preceding transition published. The signer role is `trust_admin` in that predecessor successor's registry, the entry is unrevoked, and the signed `transition_date` is inside that entry's interval. The new successor bytes must match the signed path, size, and digest. That successor's `effective_date` equals this `transition_date`, and it names this transition as its terminal transition. Historical transition bytes are not rewritten.

## 13. Cold restart and forward verification

`current` is an untrusted locator until the last step of this section.

1. Read `trust_root/current` inside the 16384-byte bound. Reject a symlink, a size of 0, and a size above 16384. Apply section 3 before any further open.
2. Treat every field as untrusted. Do not grant authority from any digest in this file. Do not use `verification_date` from any verification file as a context date during this walk.
3. Extract only `terminal_transition_id`, `terminal_transition_path`, and `trust_generation`.
4. If the extracted id is `genesis-v1`, require path `genesis/transition.record` and generation 1. The chain is the genesis transition already verified. Skip the backward walk.
5. Otherwise open the path only when it equals `transitions/<terminal_transition_id>/transition.record` and that id is a `project-id`. Walk backward. Each unverified file contributes only its predecessor id and predecessor path as locators.
6. Reject a cycle, a repeated transition id, a repeated path, a generation that does not decrease by exactly 1 toward 1, more than 64 transitions, more than 4194304 chain bytes, a path that escapes `trust_root`, and a symlink.
7. The walk must end at `genesis-v1` and `genesis/transition.record`.
8. Verify generations 2 and later forward. Verify each signature with the `trust_admin` key authorized by the predecessor's already verified successor before trusting `transition_date` or treating the new transition as authority.
9. Require the signed `transition_date` to be inside that key's interval, `revoked=no`, and not earlier than the predecessor `transition_date`.
10. Verify each new successor trust state before using its registry to authorize the next signer. Require its `effective_date` to equal the transition `transition_date` just verified.
11. Require the final successor bytes to equal `current`.
12. Only then treat `current` as the authenticated terminal trust state. Admission signatures then use that state's `effective_date` as `verification_context_date`.

Untrusted locators grant no authority. The loader does not scan `trust_root` and does not select a newest file.

## 14. Trust publication

`trust_state` keys: `schema_version`, `record_kind`, `trust_id`, `trust_generation`, `effective_date`, `terminal_transition_id`, `terminal_transition_path`, `terminal_transition_file_size`, `terminal_transition_file_sha256`, `revocation_generation`, `revocation_id`, `revocation_file_size`, `revocation_file_sha256`, `config_id`, `configuration_file_size`, `configuration_file_sha256`, `registry_id`, `authority_registry_file_size`, `authority_registry_file_sha256`, `authority_envelope_id`, `authority_envelope_file_size`, `authority_envelope_file_sha256`, `bootstrap_key_handle`, `bootstrap_key_size`, `bootstrap_key_sha256`, `admission_manifest_id`, `admission_root_manifest_file_size`, `admission_root_manifest_file_sha256`, `host_id`, `telemetry_mode`, `telemetry_governance_id`, `telemetry_governance_file_size`, `telemetry_governance_file_sha256`, `predecessor_trust_id`, `predecessor_file_size`, `predecessor_file_sha256`, `record_sha256`.

`bootstrap_key_size` is 32. Generation 1 uses `predecessor_trust_id=none`, predecessor size 0, and the zero `predecessor_file_sha256`. The successor path is `admission/trust/trust-<trust_id>.record` under `admission_root`. After section 13, `trust_root/current` is byte-for-byte identical to that file. Trust-state bytes are not a signed object. `effective_date` equals the terminal transition's signed `transition_date`.

Publication of generation N+1, with N at least 1 and N+1 at most 64:

1. Complete section 13. The authenticated terminal transition and its successor stay immutable.
2. Create a new transition whose predecessor fields name that transition file and that successor trust state, and whose `transition_date` is not earlier than the predecessor `transition_date`.
3. Sign it with an unrevoked `trust_admin` key authorized by that successor's registry. Verify the signature before trusting the new `transition_date`.
4. Create the new successor. Its `terminal_transition_id` names the new transition. Its `effective_date` equals the new `transition_date`. Its `predecessor_trust_id` names the previous successor.
5. Write the four new files only under `trust_root/transitions/<new-transition-id>/`. Do not open predecessor files for writing.
6. Compare-and-swap `current` only while its bytes still equal the old successor digest and the authenticated generation is still N. Write the successor bytes only to `trust_root/current.partial`, `fsync` that file, `fsync` the trust-root directory, and rename `current.partial` onto `current`. The temporary path is the section 3 path and is not authority.
7. A failed swap deletes only `trust_root/current.partial`, the new transition directory, and the unreferenced new successor. Every predecessor file stays unchanged. A crash before rename keeps the old `current`. Files not named by the authenticated chain are not opened and are not authoritative.

Rollback is a new forward transition. It does not rewrite a predecessor, it does not select an older link as `current`, and it does not modify the lineage-origin family.

A destination with `import_completed` true may publish generation N+1, and later N+2, under this section. N is the `trust_generation` of the imported trust state. The new successor does not need to equal the transaction's `trust_file_sha256`. The predecessor must be the authenticated `current`. The transaction is not rewritten, deleted, or consulted as a digest veto. Section 11 then sets `service_loadable` for N+1 or N+2 when status is canonical `active` and the section 13 walk contains the imported trust state. Section 14, not import resume, is the only writer of that new `current`. Import resume must not write `current` after `import_completed`. Withdrawal of a subject, when it changes the terminal trust state, is also one of these forward transitions. A later generation cannot decrease `revocation_generation` or clear a tombstone.

`telemetry_governance` keys: `schema_version`, `record_kind`, `governance_id`, `collection_allowed`, `export_allowed`, `allow_ineligible`, `record_sha256`. `export_allowed` is `no`. Path: `admission/governance/governance-<id>.record`. The governance file is the `governance-signature-<governance-id>` target. The governance review note is a separate signed note. Producer is 9-C39. Consumer is 9-C40. A governance signature uses the terminal `effective_date` after the publishing transition is current, and the transition that pins it uses that transition's `transition_date`.

Before phase 9-C39, trust and configuration contain `telemetry_mode=disabled`, `telemetry_governance_id=none`, `telemetry_governance_file_size=0`, and the zero digest. No collection, storage, export, or background capture is authorized. The governance file is opened only after the terminal trust state is authenticated, and only when mode is `pinned`.

A transition to disabled telemetry sets `telemetry_governance_id=none`, `telemetry_governance_file_size=0`, and the zero digest. Rollback writes a new successor. It copies the current triple or sets that disabled triple. It does not change `disabled` to `pinned` by installing an older trust file. Revocation generation does not decrease.

## 15. Authority registry and revocation

`authority_registry` keys: `schema_version`, `record_kind`, `registry_id`, `predecessor_registry_id`, `predecessor_registry_file_size`, `predecessor_registry_file_sha256`, `entry_list_id`, `entry_list_file_size`, `entry_list_file_sha256`, `entry_count`, `record_sha256`.

`authority_entry_list` keys: `schema_version`, `record_kind`, `list_id`, `entry_count`, then lines, then `record_sha256`. A line is `entry_id role public_key_blob_id public_key_size public_key_sha256 key_index_id key_index_size key_index_sha256 valid_from valid_until revoked`, sorted by `entry_id`. Duplicate ids or key digests fail. `role` is `provenance_reviewer`, `golden_reviewer`, `license_reviewer`, `registry_reviewer`, `policy_reviewer`, `telemetry_reviewer`, `authority_admin`, or `trust_admin`. `revoked` is `yes` or `no`. `public_key_size` is 32.

The first registry uses predecessor `none` and the zero digest. Its envelope uses `bootstrap_handle` and the same 32-byte key that verified `genesis-v1`. Its verification context is the genesis rule in section 9: the genesis `transition_date` is metadata, and the bootstrap key has no validity interval. A later registry is signed by an unrevoked `authority_admin` from the predecessor registry. The installer first verifies the new trust transition. Only then is that transition's signed `transition_date` trusted. That new `transition_date` is the registry-signature verification context. The check requires `valid_from <= new transition_date <= valid_until` and `revoked=no` on the predecessor-registry `authority_admin` entry. The new `transition_date` is not earlier than the predecessor transition's `transition_date`. The predecessor transition's date is not the installing verification date. Content paths are `admission/authority/registry-<id>.record` and `admission/authority/registry-<id>.entries`. The envelope path is only `admission/signature-records/registry-envelope-signature-<id>.record`. `authority_envelope_id` equals that signature id.

For every signature whose required role is one of those eight roles:

1. Use the registry of the trust state allowed to authorize that signature. A transition uses the predecessor successor. A later authority-registry envelope uses the predecessor successor for the installing check and the section 9 new-transition date. Every other family uses the authenticated terminal trust state.
2. Resolve exactly one entry whose `entry_id` equals `signer_key_id`.
3. Require that entry's role to equal the family's required role.
4. Require the raw key and key index to match the entry.
5. Require raw key size 32 and `public_key_size` 32.
6. Require `public_key_sha256` to equal the key blob's `raw_sha256`.
7. Require `revoked=no` in that authorizing registry. `revoked=yes` in the current registry removes current authority and leaves the historical signature bytes in place.
8. Require `valid_from <= verification_context_date <= valid_until`, using the date from section 9, not `verification_date` as an independent input.
9. Require the verification record's `verification_date` to equal that same `verification_context_date`.
10. Reject an unknown, duplicate, expired, not-yet-valid, revoked, or wrong-role entry.
11. A role string inside the signed object grants nothing unless it equals this entry.

## 16. Admission manifest

`admission_root_manifest` keys: `schema_version`, `record_kind`, `manifest_id`, `entry_count`, then lines, then `record_sha256`. A line is `entry_kind entry_id entry_path entry_size entry_file_sha256`, sorted by kind then id. Path: `admission/root/manifest-<id>.record`. `entry_kind` is `operator_configuration`, `release_pin`, `head_set`, `revocation_set`, `authority_registry`, `authority_envelope`, `telemetry_governance`, or `vocabulary_capability`. `trust_state` is not a manifest entry.

The only trust-entry order is: the fixed bootstrap key; the fixed genesis transition; backward locator collection from untrusted `current`; forward verification from genesis; byte equality of the terminal successor with `current`; the authenticated terminal trust state; the manifest; route A; the mode-dependent route-B closure. Governance, registry, configuration, revocation, admission objects, and migration state open only after the terminal trust state is authenticated.

Always required: `operator_configuration`, `revocation_set`, `authority_registry`, `authority_envelope`, and `vocabulary_capability`. Disabled telemetry forbids a governance entry and requires id `none`, size 0, and the zero digest. Pinned telemetry requires exactly one governance entry. `admission_mode=active` requires one pin and one head set. `admission_mode=no_model` forbids the pin and the head set and requires `pin_id=none`, `pin_file_size=0`, and the zero pin digest. A missing required entry, a forbidden entry, a duplicate, an unexpected entry, or a size or digest mismatch fails the load.

## 17. Trust/manifest equality

Before route B, these equalities are required:

| Manifest kind | Must equal |
|---|---|
| `authority_registry` | `registry_id`, path `admission/authority/registry-<registry_id>.record`, `authority_registry_file_size`, `authority_registry_file_sha256` |
| `authority_envelope` | `authority_envelope_id`, path `admission/signature-records/registry-envelope-signature-<authority_envelope_id>.record`, `authority_envelope_file_size`, `authority_envelope_file_sha256` |
| `operator_configuration` | `config_id`, path `admission/config/operator-<config_id>.record`, `configuration_file_size`, `configuration_file_sha256` |
| `revocation_set` | `revocation_id`, path `admission/revocation/revocation-<revocation_generation>.record`, `revocation_file_size`, `revocation_file_sha256` |
| pinned `telemetry_governance` | `telemetry_governance_id`, path `admission/governance/governance-<telemetry_governance_id>.record`, `telemetry_governance_file_size`, `telemetry_governance_file_sha256` |

The manifest cannot substitute a different registry, envelope, configuration, revocation set, or governance record.

## 18. Mode-dependent route loading

Path rules, one each: pin `admission/pins/pin-<pin_id>.record`; head set `admission/heads/<head-set-id>.record`; members `admission/heads/<head-set-id>.members`; decision `admission/decisions/<decision-id>.record`; note `admission/notes/<note-id>.record`; provenance `admission/provenance/provenance-<provenance-id>.record`; Class A binding `admission/publisher/<provenance-id>/binding.record`; Class B binding `admission/classb/<provenance-id>/source-binding.record`; license `admission/license/license-<license-id>.record`; candidate `admission/candidates/candidate-<candidate-id>.record`; golden `admission/golden/golden-<golden-id>.record`; model registry entry `admission/model-registry/registry-<registry_entry_id>.record`; policy `admission/policy/policy-<policy-id>.record`; action map `admission/policy/action-map-<map-id>.record`; dataset `admission/dataset/dataset-<dataset-id>.record`; invariance `admission/invariance/invariance-<invariance-id>.record`; activation `admission/activation/activation-<activation-id>.record`; tensor schema `admission/tensors/schema-<tensor-schema-id>.record`; tensor rule list `admission/tensors/rules-<rule-list-id>.record`; tensor inventory `admission/tensors/inventory-<inventory-id>.record`; tensor histogram `admission/tensors/histogram-<histogram-id>.record`.

Generation 1 is `admission_mode=no_model`, `runtime_mode=disabled`, and `activation_mode=inactive`. It has no pin, no head set, and no admitted model. That load succeeds.

| Edge | Modes | Failure |
|---|---|---|
| pin, head set, members, decisions, notes, provenance, binding, license, candidate, golden, model registry entry | `admission_mode=active` | load fails |
| tensor schema, rule list, inventory, and histogram | Class B provenance opened by an active admission | `provenance_unresolved` |
| policy and action map | `recommend_only`, `shadow`, `activation` | load fails |
| activation content, activation note, dataset, invariance | `activation` | load fails |

`no_model` has none of these edges. A Class A provenance uses `tensor_schema_id=none` and does not open a tensor file. A Class B provenance opens the four tensor files named by section 26. `disabled`/`inactive` with `admission_mode=active` stops after the model-registry row and does not open a policy, action map, dataset, invariance record, or activation record. `recommend_only` and `shadow` add policy and action map. `activation` adds the activation chain. An object forbidden by the mode is not a required edge. A route-B target opens only from an already verified source that names its id, the path rule above, its size, and its digest. The loader does not scan.

Evidence paths under `evidence_root` are `provenance/<id>/`, `classb/<id>/`, `candidates/<id>/gguf/`, `runs/<id>/`, `quarantine/<id>/`, `golden/<id>/`, `license/<id>.bin`, `signatures/<blob-id>.bin` for a section 9 evidence blob id, `keys/<entry-id>.bin`, `partial/<owner_run_id>/`, and `lock/evidence.lock`, `lock/recovery.lock`, and `lock/reservation-<id>.record`. GGUF size is at most 2147483648. Runtime context is 1 through 4096 and at most the trained context. Trained context is 1 through 1000000000.

## 19. Evidence cap and reservations

The cap is 68719476736 bytes. The lock is `benchmarks/generative_phase9c/lock/evidence.lock`, created with `O_CREAT|O_EXCL` and mode `0600`. `evidence_lock` keys: `schema_version`, `record_kind`, `owner_run_id`, `owner_host_id`, `acquired_date`, `record_sha256`. A waiter retries for at most 600 seconds. No unlocked check authorizes a write.

While the exclusive lock is held, `sum_before` is the checked `uint64` sum of `lstat.st_size` for every existing regular file recursively beneath `benchmarks/generative_phase9c/` before the new reservation or payload batch is created. The sum includes the current lock file, old and current reservation files, accepted payloads, raw blobs, indexes stored under that root, manifests, lists, logs, journals, quarantine files, incomplete files, temporary files, deletion-pending files, and every other regular file at any depth. Directories contribute 0. A symlink anywhere beneath the root fails the operation. A socket, device, FIFO, or other non-regular non-directory entry fails the operation. A regular file whose link count is not 1 fails the operation. There is no ignored filename, ignored directory, or hidden-file exception. Overflow fails. A filesystem read error fails.

`evidence_reservation` keys: `schema_version`, `record_kind`, `owner_run_id`, `reserved_bytes`, `path_count`, then lines, then `record_sha256`. A line is `entry <staging_relative_path> <final_relative_path> <reserved_object_size> <reserved_object_sha256> <object_class>`. `object_class` is `published_blob`, `retained_metadata`, or `temporary_metadata`. Lines sort by `staging_relative_path`. Duplicate staging paths, duplicate final paths, or a staging path equal to its final path fail the file. Both paths are relative to `evidence_root` under section 3. `path_count` equals the parsed line count. Path: `benchmarks/generative_phase9c/lock/reservation-<id>.record`. Cardinality is one per in-flight batch. Producer is the evidence writer. Consumer is section 21. Signature is `none`. It is operational evidence, not production admission authority.

`planned_batch_regular_file_bytes` is the maximum total of bytes simultaneously introduced by the current batch, excluding the reservation record. Each line is one byte string and contributes `reserved_object_size` once. The final path is the rename destination of that same byte string and is not a second addend. A copy that retains the staging bytes is forbidden. A metadata file that remains after publication is its own line and is counted once. A metadata file that exists only during the batch and is removed before lock release is its own `temporary_metadata` line and is counted once for the interval in which it exists. `planned_batch_regular_file_bytes` equals the checked sum of the line sizes. That sum is the peak because no line is instantiated twice and publication does not keep a source copy beside the destination.

Before creating a reservation, still holding the lock:

1. Canonically serialize the proposed reservation record in memory and compute `reservation_record_size`.
2. Compute `planned_batch_regular_file_bytes` as the sum of `reserved_object_size` across the lines.
3. Require `reserved_bytes` to equal `reservation_record_size + planned_batch_regular_file_bytes`.
4. Require every staging path to be absent. Require every final path other than `none` to be absent.
5. Require `sum_before + reservation_record_size + planned_batch_regular_file_bytes <= 68719476736`.
6. Create the reservation while still holding the lock.
7. Write each staging file under the same lock with that line's exact size and digest. A write that would exceed `reserved_bytes` fails.
8. Publish each `published_blob` and `retained_metadata` line by an atomic rename of that same file onto its final path. After the rename the staging path does not exist.
9. Remove each `temporary_metadata` staging file before releasing the lock. Its final path is `none` and is not created.
10. Delete the reservation before releasing the lock.
11. Recompute the regular-file sum and require it to be at most 68719476736.
12. On failure, delete only the current batch's still-present staging paths and any final path already renamed from those lines, and delete that reservation, while retaining the lock. Accepted files not named by those lines stay.

Every `staging_relative_path` is `partial/<owner_run_id>/<filename>`, where `<filename>` is a `filename` and `<owner_run_id>` equals the reservation `owner_run_id`. A `published_blob` or `retained_metadata` final path is a relative path outside `partial/`, distinct from its staging path, and not equal to any staging path. A `temporary_metadata` final path is `none`.

`none` is legal only as the final path of a `temporary_metadata` line. Every other final path is a relative path distinct from its staging path. If an operation retains both a source copy and a destination copy, that operation is outside this section and both copies require separate lines whose sizes both count. This section does not define such an operation.

## 20. Stale-lock recovery

A lock is stale only when a signed stale-lock authorization names its path, full-file digest, and owner, and `authorization_date` is at least two calendar days after `acquired_date`. There is no automatic break. The stale-lock signature uses the terminal `effective_date`.

`stale_lock_authorization` keys: `schema_version`, `record_kind`, `authorization_id`, `lock_path`, `lock_owner_run_id`, `lock_file_size`, `lock_file_sha256`, `acquired_date`, `authorization_date`, `test_only`, `record_sha256`. Path: `admission/lock/stale-<id>.record`. `lock_path` is `lock/evidence.lock` or `lock/recovery.lock`, relative to `evidence_root`. Any other path fails the record. `lock_file_sha256` pairs only with `lock_file_size`. `test_only` is `no`. The signer is `trust_admin`. `authorization_date` is at least two days after `acquired_date`. The context date is the terminal `effective_date`. It authorizes removal of that one verified inode and does not admit a model. It does not authorize deletion of an evidence reservation, a partial payload, a temporary file, a metadata file, or the other lock path.

`evidence_lock` is section 19. `recovery_lock` keys: `schema_version`, `record_kind`, `owner_run_id`, `acquired_date`, `record_sha256`. Path: `benchmarks/generative_phase9c/lock/recovery.lock`. Cardinality is one during recovery. Signature is `none`. It is not admission authority. Producer and consumer are the section 21 cleanup writer. A compliant evidence writer never creates or unlinks `recovery.lock`.

A compliant evidence writer unlinks `lock/evidence.lock` only when it created that file, it still holds the creating descriptor, `fstat` device and inode equal that descriptor, and the file's `owner_run_id` is its own. It then unlinks, `fsync`s `lock/`, and closes the descriptor. No compliant writer unlinks a lock it does not own. No compliant writer replaces a lock file in place.

This platform has no atomic unlink-if-inode operation. The cooperative protocol is the exclusive `recovery.lock` plus the descriptor check below. Two recoveries cannot interleave. A compliant writer cannot replace the lock during the sequence because it does not unlink a lock it does not own.

Removal of the lock named by a verified stale-lock authorization:

1. Authenticate the terminal trust state under section 13. Verify the stale-lock authorization, including its signature, role, context date, path, size, and digest. Do not verify a deletion authorization in this section.
2. If `lock_path` is `lock/evidence.lock`, create `recovery.lock` with `O_CREAT|O_EXCL|O_NOFOLLOW` and mode `0600`, containing the `recovery_lock` record. If that path exists, stop and delete nothing else. If `lock_path` is `lock/recovery.lock`, do not create a recovery lock; a separate stale-lock authorization whose digest equals the existing file uses steps 3 through 11 on that path only, does not authorize evidence deletion, and does not acquire `evidence.lock`.
3. Open the authorized `lock_path` with `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`.
4. `fstat` the descriptor. Record device, inode, and byte size. Require a regular file whose link count is 1.
5. Read exactly that size from the descriptor and compute SHA-256. Require the size and digest to equal the authorization. Require the file's `owner_run_id` to equal `lock_owner_run_id`.
6. `fstat` again. Require the same device, inode, and size.
7. Read and hash the descriptor again. Require the same digest.
8. `lstat` the pathname with no symlink follow. Require the same device, inode, and size.
9. Unlink that pathname.
10. `fstat` the still-open descriptor. Require the same device and inode and a link count of 0. If any value differs, do not continue, do not create `evidence.lock`, and do not delete any evidence payload. Close the descriptor. If this attempt created `recovery.lock`, unlink it only after the same device and inode check against the descriptor that created it, then `fsync` `lock/`.
11. On success, `fsync` the parent directory `lock/` and close the old descriptor. If the removed path was `recovery.lock`, stop. Do not create `evidence.lock` and do not delete evidence. If the removed path was `evidence.lock`, create the new `evidence.lock` with `O_CREAT|O_EXCL|O_NOFOLLOW` and mode `0600`, write this cleanup owner's `evidence_lock` record, and `fsync` `lock/`. If that exclusive create fails, unlink `recovery.lock` only when its device and inode still match the descriptor that created it, `fsync` `lock/`, and stop without deleting evidence. On success, unlink `recovery.lock` with that same device and inode check, `fsync` `lock/`, and continue at section 21 while the new `evidence.lock` is held.


No authorization for earlier bytes removes a replacement lock. A mismatch at any step does not unlink.

## 21. Authorized immutable cleanup journals

`deletion_authorization` keys: the two closed orders below, selected by `target_kind`. One file uses one order.

Quarantine order, `target_kind=quarantine_metadata`: `schema_version`, `record_kind`, `authorization_id`, `target_kind`, `target_id`, `target_file_size`, `target_file_sha256`, `authorization_date`, `test_only`, `record_sha256`. Path: `admission/deletion/deletion-<id>.record`. `test_only` is `no`. The signer is `trust_admin`. The context date is the terminal `effective_date`. It authorizes quarantine deletion under section 22 and does not admit a model.

Reservation order, `target_kind=evidence_reservation`: `schema_version`, `record_kind`, `authorization_id`, `target_kind`, `target_id`, `reservation_path`, `target_file_size`, `target_file_sha256`, `owner_run_id`, `deletion_list_id`, `deletion_list_path`, `deletion_list_file_size`, `deletion_list_file_sha256`, `authorization_date`, `test_only`, `record_sha256`. Path: `admission/deletion/deletion-<id>.record`. `reservation_path` is `lock/reservation-<reservation-id>.record` relative to `evidence_root`. `target_file_size` and `target_file_sha256` are that reservation file's full size and full-file digest. `owner_run_id` equals the reservation's `owner_run_id`. `deletion_list_path` is `admission/deletion/reservation-<reservation-id>.files.record`. `test_only` is `no`. The signer is `trust_admin`. The context date is the terminal `effective_date`. It authorizes deletion of the files named by that verified list and does not admit a model. It does not authorize deletion of any path outside that list.

`reservation_deletion_list` keys: `schema_version`, `record_kind`, `list_id`, `reservation_id`, `owner_run_id`, `entry_count`, then lines, then `record_sha256`. A line is `entry <relative_path> <deletion_entry_size> <deletion_entry_sha256> <object_class>`. `object_class` is only `evidence_reservation`, `partial_staging`, or `temporary_metadata`. Lines sort by `relative_path`. Duplicate paths fail. `entry_count` equals the parsed line count. Path: `admission/deletion/reservation-<reservation-id>.files.record`. Producer is 9-C6. Consumer is this section. Signature is `none`. Cardinality is one per reservation cleanup.

Path grammar, rejected before any unlink:

- `evidence_reservation`: the path equals the authorization `reservation_path`, that path is `lock/reservation-<reservation-id>.record`, and the line count of this class is one.
- `partial_staging`: the path is `partial/<owner_run_id>/<filename>` with the same `owner_run_id` as the authorization and the reservation. The reservation contains one line whose `staging_relative_path` equals this path. That reservation line's final path is not this path and is not `none`. The path is not equal to any final path in the reservation.
- `temporary_metadata`: the path is `partial/<owner_run_id>/<filename>` with that same owner. The reservation contains one `temporary_metadata` line whose staging path equals this path and whose final path is `none`.

`<filename>` is a `filename`. The path has exactly those three segments. A path under `candidates/`, `provenance/`, `classb/`, `golden/`, `keys/`, `signatures/`, `license/`, `quarantine/`, `runs/`, or another owner's `partial/` directory is forbidden. An absolute path, a root escape, a symlink, a final published path, and any path that is not the reservation path or a staging path in that authenticated reservation are forbidden. A forbidden line rejects the list and the authorization's use. No unlink is performed.

`cleanup_pending` keys: `schema_version`, `record_kind`, `authorization_id`, `authorization_file_size`, `authorization_file_sha256`, `deletion_list_id`, `deletion_list_file_size`, `deletion_list_file_sha256`, `owner_run_id`, `creation_date`, `state`, `record_sha256`. Path: `admission/deletion/deletion-<authorization-id>.pending.record`. `state` is `deletion_pending`. Signature is `none`. Cardinality is one per authorization. The file is created with `O_CREAT|O_EXCL` and is never overwritten.

`cleanup_completion` keys: `schema_version`, `record_kind`, `authorization_id`, `pending_file_size`, `pending_file_sha256`, `authorization_file_size`, `authorization_file_sha256`, `deletion_list_id`, `deletion_list_file_size`, `deletion_list_file_sha256`, `owner_run_id`, `completion_date`, `state`, `record_sha256`. Path: `admission/deletion/deletion-<authorization-id>.complete.record`. `state` is `deletion_complete`. `pending_file_sha256` is the full-file digest of the pending record. Signature is `none`. Cardinality is one per finished authorization. The file is created with `O_CREAT|O_EXCL` and is never overwritten.

A reservation is stale when `evidence.lock` is absent or its `owner_run_id` differs from the reservation. Staleness does not authorize deletion. The only cleanup procedure is the following. It runs while the new `evidence.lock` is held, after section 20 has removed the previous evidence lock.

1. Verify the reservation-order deletion authorization and the deletion list. Re-read the reservation and require the authorization's size and digest. A grammar failure or a digest mismatch stops the procedure and unlinks nothing.
2. If `deletion-<authorization-id>.complete.record` exists, verify it against the pending record and the authorization. A valid completion means cleanup is finished. A completion that does not verify fails closed and unlinks nothing.
3. If the pending record exists and the completion record does not, do not overwrite the pending record. Require its authorization id, authorization size and digest, deletion-list id, deletion-list size and digest, and owner run id to equal the records supplied in step 1. Verify those signatures again. A different authorization or list fails closed.
4. If neither record exists, create the pending record with those identities, `state=deletion_pending`, and today's `creation_date`. `fsync` the file and its parent directory.
5. For each list line, in order: if the path is absent, it is already deleted. If it exists, require a regular file, link count 1, no symlink, and the line's size and digest, then unlink it and `fsync` the parent. A mismatch fails closed, leaves the pending record, writes no completion record, and unlinks no later path.
6. After every listed path is absent, create the completion record. Do not overwrite an existing file. `fsync` the completion record and its parent directory.
7. A later attempt that finds a valid completion performs no unlink. A later attempt that finds the same pending record repeats steps 3, 5, and 6 only. No journal rewrite is permitted.

## 22. Quarantine

`quarantine_metadata` keys: `schema_version`, `record_kind`, `quarantine_id`, `associated_phase_id`, `associated_run_id`, `directory_id`, `creation_date`, `reason_code`, `accepted_run_manifest_path`, `accepted_run_manifest_file_size`, `accepted_run_manifest_file_sha256`, `retention_start`, `retention_deadline`, `deletion_role`, `state`, `record_sha256`.

The directory is `benchmarks/generative_phase9c/quarantine/<run-id>/`. The metadata file is `quarantine.record` inside it. `quarantine_id` and `associated_run_id` equal `<run-id>`. `directory_id` is `quarantine/<run-id>`. Mode is `0700`. `deletion_role` is `trust_admin`. `state` is `open`. `reason_code` is `checksum_rejected`, `reproduction_mismatch`, `recipe_echo_rejected`, `cap_rejected`, `symlink_rejected`, `signature_failed`, `invariant_mismatch`, or `other_rejected`. This record is quarantine-only. It is not production admission authority. Producer is the evidence writer. Consumer is 9-C6 deletion. Signature is `none`. Cardinality is one per quarantined run.

`recipe_echo` bytes may be stored only here, under the filename `recipe-echo.bin`, and only when `reason_code=recipe_echo_rejected`. They are not an accepted blob class, cannot appear in a payload list, cannot satisfy provenance, and cannot affect admission or publication. Directory size is at most 1073741824 and counts toward the root cap.

If an accepted run manifest already exists, the path and `accepted_run_manifest_file_sha256` name it. Otherwise the path is `none`, the size is 0, and the digest is zero. The record is immutable. `retention_start` equals `creation_date`. `retention_deadline` is 30 days later.

`deletion_journal` keys: `schema_version`, `record_kind`, `quarantine_id`, `authorization_id`, `authorization_file_size`, `authorization_file_sha256`, `state`, `record_sha256`. Path: `benchmarks/generative_phase9c/quarantine/<run-id>/deletion.record`. Cardinality is one during deletion. `state` is `deletion_pending`. Producer is 9-C6. Consumer is recovery. Signature is `none`. It is quarantine-only. Deletion requires `state=open` on the metadata, a civil date on or after the deadline, actor `trust_admin`, a quarantine-order deletion authorization, and the evidence lock. The writer sets `state=deletion_pending` and `fsync`s the journal before any unlink. It then unlinks payloads, the journal, and `quarantine.record`, and removes the directory. Recovery that still sees the journal continues the unlink. Recovery that sees metadata and no journal starts the authorized deletion again. Quarantine bytes never become accepted evidence. The deletion signature uses the terminal `effective_date`.

## 23. Class A admission

The first manual family name is `Qwen/Qwen3-0.6B`. The campaign 001 filename `qwen3-0.6b-Q4_K_M.gguf` is a legal filename and is not an admission. Identity is the upstream name plus `expected_gguf_sha256`. `parameter_count=unknown` is legal only with basis `api_gap`, and it is not a feature or a claim.

`publisher_authentication` keys: `schema_version`, `record_kind`, `authentication_id`, `method`, `checksum_blob_id`, `checksum_size`, `checksum_raw_sha256`, `checksum_index_id`, `checksum_index_size`, `checksum_index_sha256`, `public_key_blob_id`, `public_key_size`, `public_key_sha256`, `signature_id`, `signature_file_size`, `signature_file_sha256`, `record_sha256`.

`method` is `signed_checksum` or `independently_reproducible_conversion`. `public_key_size` is 32. `public_key_sha256` hashes exactly the raw publisher public-key bytes. The public-key index `raw_size` is 32. `public_key_size` equals `raw_size`. `public_key_sha256` equals `raw_sha256`.

Class A is a complete admission method. GNU extraction supplies the lowercase hash and the filename, not a byte size. The operator fetches one artifact, measures its exact byte count, and computes SHA-256 of those same bytes. That digest must equal the GNU line's hash before a binding note is reviewed. The binding stores `artifact_filename`, `artifact_size`, and `artifact_sha256`. Those size and digest fields name one byte string. Provenance copies them to `expected_gguf_size` and `expected_gguf_sha256`. Candidate admission requires both equalities. `source_manifest_id=none` is legal. Class B is not required. `tensor_schema_id=none`, `tensor_schema_file_size=0`, and the zero `tensor_schema_file_sha256` are required on a Class A provenance.

A legal GNU line is 64 lowercase hex digits, then two spaces or one space and an asterisk, then one `filename`, then LF. These results reject extraction and yield `provenance_unresolved`: `leading_backslash`, `embedded_backslash`, `carriage_return`, `blank_line`, `comment`, `uppercase_hex`, `missing_final_lf`, `duplicate_name`, `case_collision`, `multiple_match`, `unsupported_filename`.

Class A uses `method=signed_checksum`. Cardinality is one. `<id>` is the provenance id. Omission, duplication, or an unexpected file yields `provenance_unresolved`.

| Raw path under `evidence_root` | Index path | Class |
|---|---|---|
| `provenance/<id>/checksum.bin` | `admission/publisher/<id>/checksum.index` | `publisher_checksum` |
| `signatures/checksum-sigblob-<id>.bin` | `admission/signature-indexes/checksum-sigindex-<id>.index` | `signature_bytes` |
| `provenance/<id>/public-key.bin` | `admission/publisher/<id>/public-key.index` | `public_key` |

Structured siblings under `admission/publisher/<id>/` are only `authentication.record`, `binding.record`, and `binding-list.record`. The checksum signature and the binding note use only the section 9 ids `checksum-signature-<id>`, `bind-a-note-<id>`, `bind-a-signature-<id>`, `bind-a-sigblob-<id>`, and `bind-a-sigindex-<id>`. Producer is 9-C8. Consumers are 9-C9 and 9-C11. Phase 9-C8 does not create the submission note. A symlink rejects the file.

## 24. Class B paths and schemas

Class B uses `method=independently_reproducible_conversion`. Cardinality is one except the two reproduction records and the two outputs. Producer is 9-C8. Consumers are 9-C9, 9-C10, and 9-C11. Omission of either side of a pair, a duplicate path, a symlink, or a file whose name is not in this closed set yields `provenance_unresolved`. An index cannot be valid without its raw file. These records are production admission material for a Class B provenance and are not used by Class A.

`source_manifest` keys: `schema_version`, `record_kind`, `manifest_id`, `authentication_id`, `authentication_file_size`, `authentication_file_sha256`, `file_list_id`, `file_list_file_size`, `file_list_file_sha256`, `record_sha256`. Path: `admission/classb/<id>/source-manifest.record`. Cardinality is one. Signature is `none`. Authorization effect is none until the provenance decision is accepted. Failure is `provenance_unresolved`.

`source_file_list` keys: `schema_version`, `record_kind`, `list_id`, `file_count`, then lines, then `record_sha256`. A line is `entry <object_kind> <blob_id> <relative_path> <byte_size> <list_raw_sha256>`. `relative_path` is relative to `evidence_root`. Lines sort by the object-kind order in section 24, then by `blob_id`. Duplicate object kinds fail. Duplicate blob ids fail. Duplicate paths fail. An unknown object kind fails. A path that differs from the section 24 row for that object kind fails. Omission of a required kind fails. `file_count` equals the parsed line count and equals 9. Path: `admission/classb/<id>/source-file-list.record`. Cardinality is one. `byte_size` equals that row's raw size. `list_raw_sha256` equals that row's `raw_sha256`. Signature is `none`. Failure is `provenance_unresolved`. A basename is not an identity.

`converter` keys: `schema_version`, `record_kind`, `converter_id`, `converter_name`, `converter_version`, `record_sha256`. `converter_name` is an `upstream-name`. `converter_version` is a `project-id`. Path: `admission/classb/<id>/converter.record`. Cardinality is one. Signature is `none`. Failure is `provenance_unresolved`.

`recipe` keys: `schema_version`, `record_kind`, `recipe_id`, `converter_id`, `converter_file_size`, `converter_file_sha256`, `parameter_blob_id`, `parameter_size`, `parameter_sha256`, `parameter_index_id`, `parameter_index_size`, `parameter_index_sha256`, `archive_blob_id`, `archive_size`, `archive_sha256`, `archive_index_id`, `archive_index_size`, `archive_index_sha256`, `binary_blob_id`, `binary_size`, `binary_sha256`, `binary_index_id`, `binary_index_size`, `binary_index_sha256`, `build_recipe_blob_id`, `build_recipe_size`, `build_recipe_sha256`, `build_recipe_index_id`, `build_recipe_index_size`, `build_recipe_index_sha256`, `record_sha256`. Path: `admission/classb/<id>/recipe.record`. Cardinality is one. `converter_file_sha256` is the full-file digest of `converter.record` and `converter_file_size` is that file's byte size. Each raw alias equals the named blob as in section 8. Signature is `none`. Failure is `provenance_unresolved`.

`<id>` in this table is the provenance id. Each object kind has cardinality one. The source-file line for that kind uses that blob id and that relative path.

| Order | Object kind | Blob id | Relative path under `evidence_root` | Index path |
|---|---|---|---|---|
| 1 | `publisher_checksum` | `source-checksum-<id>` | `classb/<id>/source-checksum.bin` | `admission/classb/<id>/source-checksum.index` |
| 2 | `publisher_signature` | `source-checksum-sigblob-<id>` | `signatures/source-checksum-sigblob-<id>.bin` | `admission/signature-indexes/source-checksum-sigindex-<id>.index` |
| 3 | `publisher_public_key` | `source-public-key-<id>` | `classb/<id>/source-public-key.bin` | `admission/classb/<id>/source-public-key.index` |
| 4 | `source_archive` | `source-archive-<id>` | `classb/<id>/source-archive.bin` | `admission/classb/<id>/source-archive.index` |
| 5 | `converter_binary` | `converter-binary-<id>` | `classb/<id>/converter-binary.bin` | `admission/classb/<id>/converter-binary.index` |
| 6 | `build_recipe` | `build-recipe-<id>` | `classb/<id>/build-recipe.bin` | `admission/classb/<id>/build-recipe.index` |
| 7 | `parameter_blob` | `parameter-<id>` | `classb/<id>/parameter.bin` | `admission/classb/<id>/parameter.index` |
| 8 | `reproduction_output_a` | `output-a-<id>` | `classb/<id>/output-a.gguf` | `admission/classb/<id>/output-a.index` |
| 9 | `reproduction_output_b` | `output-b-<id>` | `classb/<id>/output-b.gguf` | `admission/classb/<id>/output-b.index` |

`publisher_signature` is blob class `signature_bytes` and its size is 64. `publisher_public_key` is blob class `public_key` and its size is 32. Index self-checksums are `checksum_index_sha256`, `archive_index_sha256`, `binary_index_sha256`, `build_recipe_index_sha256`, `parameter_index_sha256`, and `output_index_sha256` for the matching index, each hashing that index's preimage. The source-checksum signature and the binding note use only `source-checksum-signature-<id>`, `bind-b-note-<id>`, `bind-b-signature-<id>`, `bind-b-sigblob-<id>`, and `bind-b-sigindex-<id>`. Structured siblings under `admission/classb/<id>/` are only `source-authentication.record`, `source-binding.record`, `source-binding-list.record`, `source-manifest.record`, `source-file-list.record`, `converter.record`, `recipe.record`, `reproduction-a.record`, `reproduction-b.record`, `evidence-a.manifest`, `evidence-b.manifest`, `payload-a.list`, and `payload-b.list`.

## 25. Reproduction A/B graph

`reproduction` keys: `schema_version`, `record_kind`, `reproduction_id`, `recipe_id`, `recipe_file_size`, `recipe_file_sha256`, `source_manifest_id`, `source_manifest_file_size`, `source_manifest_file_sha256`, `environment_id`, `working_copy_id`, `output_object_kind`, `output_blob_id`, `output_size`, `output_sha256`, `output_index_id`, `output_index_size`, `output_index_sha256`, `evidence_manifest_id`, `evidence_manifest_file_size`, `evidence_manifest_file_sha256`, `record_sha256`. Cardinality is exactly two. The two records share `recipe_id` and `source_manifest_id`, differ in `environment_id` and `working_copy_id`, and agree on `output_size` and `output_sha256`. Signature is `none`. Failure is `provenance_unresolved`. Producer is 9-C8. Consumers are 9-C9, 9-C10, and 9-C11.

`evidence_manifest` keys: `schema_version`, `record_kind`, `manifest_id`, `phase_id`, `payload_list_id`, `payload_list_file_size`, `payload_list_file_sha256`, `record_sha256`. Cardinality is exactly two. Signature is `none`. Failure is `provenance_unresolved`.

`payload_list` keys: `schema_version`, `record_kind`, `list_id`, `object_count`, then lines, then `record_sha256`. A line is `entry <object_kind> <object_id> <relative_path> <byte_size> <digest_class> <digest_sha256>`. `digest_class` is `raw`. Lines sort by the object-kind order below, then by `object_id`. Duplicate kinds, duplicate object ids, and duplicate paths fail. An unknown kind fails. The token `recipe_echo` rejects the list. `object_id`, `relative_path`, `byte_size`, and `digest_sha256` equal the section 24 row for that object kind. Signature is `none`. There is no additional payload raw file. Failure is `provenance_unresolved`.

The letter is part of the id, the path, and the output kind. `<id>` is the provenance id.

| Letter | Reproduction path and id | Output kind and path | Manifest path and id | Payload path and id | Payload kinds, in order | Count |
|---|---|---|---|---|---|---|
| A | `admission/classb/<id>/reproduction-a.record`, `repro-a-<id>` | `reproduction_output_a`, `classb/<id>/output-a.gguf` | `admission/classb/<id>/evidence-a.manifest`, `evid-a-<id>` | `admission/classb/<id>/payload-a.list`, `plist-a-<id>` | `publisher_checksum`, `source_archive`, `converter_binary`, `build_recipe`, `parameter_blob`, `reproduction_output_a` | 6 |
| B | `admission/classb/<id>/reproduction-b.record`, `repro-b-<id>` | `reproduction_output_b`, `classb/<id>/output-b.gguf` | `admission/classb/<id>/evidence-b.manifest`, `evid-b-<id>` | `admission/classb/<id>/payload-b.list`, `plist-b-<id>` | `publisher_checksum`, `source_archive`, `converter_binary`, `build_recipe`, `parameter_blob`, `reproduction_output_b` | 6 |

`reproduction-a.record` contains `evidence_manifest_id=evid-a-<id>` and the full size and digest of `evidence-a.manifest`. That manifest contains `manifest_id=evid-a-<id>`, `payload_list_id=plist-a-<id>`, and the full size and digest of `payload-a.list`. `payload-a.list` contains `list_id=plist-a-<id>` and `object_count=6`. The B files use only the B ids and the B paths. Shared source, converter, and recipe objects appear in both lists with the same section 24 identity. Each list contains only its own reproduction output.

These pairings fail `provenance_unresolved`: an A reproduction naming a B manifest id or digest; a B reproduction naming an A manifest; an A manifest naming a B payload list; a B manifest naming an A payload list; a reused manifest id or list id; equal `environment_id`; equal `working_copy_id`; a missing size or digest; a size or digest mismatch; `object_count` other than 6; an A list that contains `reproduction_output_b`; a B list that contains `reproduction_output_a`.

## 26. Tensor-schema binding

`tensor_schema` keys: `schema_version`, `record_kind`, `schema_id`, `rule_list_id`, `rule_list_file_size`, `rule_list_file_sha256`, `inventory_id`, `inventory_file_size`, `inventory_file_sha256`, `histogram_id`, `histogram_file_size`, `histogram_file_sha256`, `record_sha256`. Path: `admission/tensors/schema-<id>.record`. Producer is 9-C4. Consumer is 9-C11. Signature is `none`.

`tensor_rule_list` keys: `schema_version`, `record_kind`, `list_id`, `rule_count`, lines `pattern_hex role`, `record_sha256`. `role` is `weight` or `non_weight`. Path: `admission/tensors/rules-<id>.record`. Duplicate patterns fail. `rule_count` equals the parsed line count.

`tensor_inventory` keys: `schema_version`, `record_kind`, `inventory_id`, `tensor_count`, lines `name_hex rank dimensions`, `record_sha256`. Rank is 1 through 8. Path: `admission/tensors/inventory-<id>.record`. Duplicate names fail. `tensor_count` equals the parsed line count.

`tensor_histogram` keys: `schema_version`, `record_kind`, `histogram_id`, `type_count`, lines `type_name count`, `record_sha256`. Path: `admission/tensors/histogram-<id>.record`. Counts sum to the weight-tensor count. `type_count` equals the parsed line count.

A Class B provenance carries `tensor_schema_id`, `tensor_schema_file_size`, and `tensor_schema_file_sha256`. The id is a `project-id`. The path is `admission/tensors/schema-<tensor_schema_id>.record`. The size and digest equal that file. That schema's rule-list, inventory, and histogram id, size, and digest equal the three sibling files. All four files are required route-B dependencies when phase 9-C11 reviews that provenance and whenever a later load opens that Class B provenance. Omission or mismatch is `provenance_unresolved`. A Class A provenance uses id `none`, size 0, and the zero digest, and it does not open these files.

The two section 25 reproduction outputs are the only conversion outputs that can satisfy `expected_gguf_sha256`.

## 27. Binding and provenance

`binding` keys: `schema_version`, `record_kind`, `binding_id`, `authentication_id`, `authentication_file_size`, `authentication_file_sha256`, `checksum_index_id`, `checksum_index_size`, `checksum_index_sha256`, `artifact_filename`, `artifact_size`, `artifact_sha256`, `note_id`, `note_file_size`, `note_file_sha256`, `note_signature_id`, `note_signature_file_size`, `note_signature_file_sha256`, `record_sha256`. Class A `binding_id` is `bind-a-record-<provenance-id>`. Class B `binding_id` is `bind-b-record-<provenance-id>`.

`binding_list` keys: `schema_version`, `record_kind`, `list_id`, `binding_count`, lines `binding_id binding_file_size binding_file_sha256` sorted by `binding_id`, then `record_sha256`.

`provenance` keys: `schema_version`, `record_kind`, `provenance_id`, `upstream_name`, `filename`, `method`, `expected_gguf_size`, `expected_gguf_sha256`, `checksum_size`, `checksum_raw_sha256`, `authentication_id`, `authentication_file_size`, `authentication_file_sha256`, `binding_id`, `binding_file_size`, `binding_file_sha256`, `binding_list_id`, `binding_list_file_size`, `binding_list_file_sha256`, `source_manifest_id`, `source_manifest_file_size`, `source_manifest_file_sha256`, `tensor_schema_id`, `tensor_schema_file_size`, `tensor_schema_file_sha256`, `trained_context`, `record_sha256`.

Path: `admission/provenance/provenance-<id>.record`. For Class A, `expected_gguf_size` equals `artifact_size`, `expected_gguf_sha256` equals `artifact_sha256`, `filename` equals `artifact_filename`, `source_manifest_id=none`, the source-manifest size is 0, the source-manifest digest is zero, and the tensor triple is the Class A sentinel in section 26. For Class B, the tensor triple is the section 26 reference and `source_manifest_id` names the section 24 manifest. A mismatched download does not rewrite `expected_gguf_sha256`. Producer is 9-C8. Consumers are 9-C9 and 9-C11. Signature is `none`. The review note is the signed object.

`review_note` keys: `schema_version`, `record_kind`, `note_id`, `statement_code`, `subject_kind`, `subject_id`, `subject_file_size`, `subject_file_sha256`, `pending_decision_id`, `pending_decision_file_size`, `pending_decision_file_sha256`, `actor_key_id`, `actor_key_size`, `actor_key_sha256`, `record_sha256`. `statement_code` is `submit`, `review`, or `withdraw`. `actor_key_size` is 32. `actor_key_sha256` hashes exactly the 32 reviewer public-key bytes selected from the authenticated authority registry entry whose `entry_id` equals `actor_key_id`. The key index `raw_size` is 32. `actor_key_size` equals `raw_size`. `actor_key_sha256` equals `raw_sha256`. A submission note uses `submit`, pending id `none`, pending size 0, and the zero pending digest. A review note whose subject is not `source_binding` uses `review` and names the pending decision. A `source_binding` note uses `statement_code=review`, `subject_kind=source_binding`, `subject_id` equal to the binding id, pending id `none`, pending size 0, and the zero pending digest. `subject_file_size` and `subject_file_sha256` are the binding file's full size and digest. The signer role is `provenance_reviewer`. `signed_object_id` is the note id.

`decision` keys: `schema_version`, `record_kind`, `decision_id`, `subject_kind`, `subject_id`, `subject_file_size`, `subject_file_sha256`, `predecessor_decision_id`, `predecessor_decision_file_size`, `predecessor_decision_file_sha256`, `state`, `note_id`, `note_file_size`, `note_file_sha256`, `note_signature_id`, `note_signature_file_size`, `note_signature_file_sha256`, `registry_id`, `authority_registry_file_size`, `authority_registry_file_sha256`, `authority_envelope_id`, `authority_envelope_file_size`, `authority_envelope_file_sha256`, `decision_date`, `record_sha256`.

Decision `subject_kind` is `provenance`, `license`, `registry_entry`, `policy_pin`, `golden_fixture`, `telemetry_governance`, or `activation_authorization`. A decision cannot use `source_binding`. A decision is not an Ed25519 target. Reviewer `actor_key_sha256` differs from the submitter's. Phase 9-C9 creates only the submission note, its signature triple, and the pending decision. `state` is `pending`, `accepted`, `rejected`, or `withdrawn`.

`license` keys: `schema_version`, `record_kind`, `license_id`, `provenance_id`, `provenance_decision_id`, `provenance_decision_file_size`, `provenance_decision_file_sha256`, `license_blob_id`, `license_byte_size`, `license_raw_sha256`, `license_index_id`, `license_index_size`, `license_index_sha256`, `model_license_compatible`, `llama_acceptance`, `rejection_code`, `record_sha256`.

The raw file is `benchmarks/generative_phase9c/license/<id>.bin`. The index is `admission/license/license-<id>.index`. `model_license_compatible=yes` requires `rejection_code=none`. Otherwise the code is `incompatible`, `missing_acceptance`, `non_osi_unaccepted`, or `unreviewed`. A Llama-family license requires `llama_acceptance=yes` before compatibility `yes`. Software does not make that legal judgment. License content is written only after H2. The license bytes are not a signature. The license-review note is the signed object. Producer is 9-C18. Consumer is 9-C19.

## 28. Candidate and dataset

`candidate_record` keys: `schema_version`, `record_kind`, `candidate_id`, `provenance_id`, `provenance_file_size`, `provenance_file_sha256`, `filename`, `gguf_size`, `gguf_sha256`, `candidate_index_id`, `candidate_index_size`, `candidate_index_sha256`, `storage_path`, `test_only`, `record_sha256`. Path: `admission/candidates/candidate-<id>.record`. Index: `admission/candidates/candidate-<id>.index`. Raw path: `benchmarks/generative_phase9c/candidates/<id>/gguf/<filename>`. `gguf_size` equals provenance `expected_gguf_size`. `gguf_sha256` equals provenance `expected_gguf_sha256`. `test_only` is `no`. Producer is 9-C13. Consumer is 9-C14. Signature is `none`. An unexpected file in that `gguf/` directory fails the candidate.

`dataset_record` keys: `schema_version`, `record_kind`, `dataset_id`, `model_id`, `invariance_id`, `invariance_file_size`, `invariance_file_sha256`, `case_list_id`, `case_list_file_size`, `case_list_file_sha256`, `comparison_result`, `test_only`, `record_sha256`. Path: `admission/dataset/dataset-<id>.record`. `comparison_result` must be `equal`. `test_only` is `no`. The record does not by itself activate thread 4. Producer is 9-C28. Consumer is 9-C44. Signature is `none`.

## 29. Tokenizer probe and write contract

`vocabulary_capability` keys: `schema_version`, `record_kind`, `symbol_present`, `symbol_name`, `symbol_header`, `range_check_enforced`, `record_sha256`. Path: `admission/capability/vocabulary.record`. Committed values are `symbol_present=no`, `symbol_name=none`, `symbol_header=none`, and `range_check_enforced=no`. Phase 9-C3 may record a symbol it reads from the pinned headers. It does not require `llama_vocab_n_tokens`. Range checks start only after `symbol_present=yes` and `range_check_enforced=yes`. On a case list, `vocabulary_id` names this one file, `vocabulary_file_size` is its byte count, and `vocabulary_file_sha256` is SHA-256 of those complete file bytes. Producer is 9-C3. Consumer is every request. Signature is `none`. Cardinality is one.

Context bounds: `n_ctx` and `n_batch` equal the admitted limit, `n_ubatch` is the minimum of that limit and 512, `n_seq_max` is 1, and `n_gpu_layers` is 0. Effective context and `max_new_tokens` are 1 through 4096, and `max_new_tokens` is at most effective context minus `prompt_token_count`. Allocation of the second-call buffer is allowed only when `required_token_count` is within those bounds.

The pinned probe API is `llama_tokenize(const llama_vocab * vocab, const char * text, int32_t text_len, llama_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special)`. Both calls use `add_special=true` and `parse_special=true`. The first raw return is not required to be positive.

First call, size-probe mode: `tokens` is null and `n_tokens_max` is 0. Store the signed return as `probe_result`. Reject `probe_result == INT32_MIN` with `kTokenization`. Require `probe_result < 0`. A zero or positive probe result is `kTokenization`. Compute `required_token_count = -probe_result` by checked `int32` negation, which is defined only after the `INT32_MIN` rejection. Require `1 <= required_token_count <= INT32_MAX`. If that count exceeds the section 29 context and allocation bounds, return `kContextLimit` and do not allocate and do not perform the second call.

Second call: `n_tokens_max` is `required_token_count` and the output buffer has exactly that capacity. Store the signed return as `second_result`. Require `second_result > 0` and `second_result == required_token_count`. A negative, zero, larger, or smaller result is `kTokenization`. `prompt_token_count` is `second_result`.

A negative probe result is the required capacity. A positive second result is the number of tokens written. Phase 9-C40 updates `token_count_from_llama_probe` and `tests/generative_session_test.cpp` so `probe_result=-8` yields `required_token_count=8`, while `INT32_MIN`, 0, and a positive probe result fail. The current helper's acceptance of a positive probe is not this contract. Probe failure writes no adaptive log and does not clear the KV cache.

## 30. Golden and EOG

`golden_case_list` keys: `schema_version`, `record_kind`, `list_id`, `golden_id`, `vocabulary_id`, `vocabulary_file_size`, `vocabulary_file_sha256`, `declared_case_count`, then case lines, then `record_sha256`. Path: `admission/golden/cases-<golden-id>.record`. A case line is `case_id prompt_blob_id prompt_size case_prompt_sha256 prompt_index_id case_prompt_index_size case_prompt_index_sha256 parameter_blob_id parameter_size case_parameter_sha256 parameter_index_id case_parameter_index_size case_parameter_index_sha256 token_blob_id token_size case_expected_tokens_sha256 token_index_id case_expected_tokens_index_size case_expected_tokens_index_sha256 visible_blob_id visible_size case_expected_visible_sha256 visible_index_id case_expected_visible_index_size case_expected_visible_index_sha256 finish_reason eog_observed max_new_tokens`. Lines sort by `case_id`. `declared_case_count` equals the parsed line count. Duplicate ids fail.

Raw pairs, cardinality `declared_case_count`: `prompt-<case-id>.bin`, `params-<case-id>.bin`, `tokens-<case-id>.bin`, and `visible-<case-id>.bin` under `benchmarks/generative_phase9c/golden/<golden-id>/`, with indexes under `admission/golden/<golden-id>/`. Oracle and template are `oracle.bin` and `template.bin` with matching indexes. Any other file in those directories fails. Producer is 9-C14. Consumers are 9-C16 and 9-C28. Signature is `none`.

`golden` keys: `schema_version`, `record_kind`, `golden_id`, `provenance_id`, `provenance_decision_id`, `provenance_decision_file_size`, `provenance_decision_file_sha256`, `case_list_id`, `case_list_file_size`, `case_list_file_sha256`, `oracle_index_id`, `oracle_index_size`, `oracle_index_sha256`, `template_index_id`, `template_index_size`, `template_index_sha256`, `candidate_id`, `candidate_gguf_size`, `candidate_gguf_sha256`, `candidate_index_id`, `candidate_index_size`, `candidate_index_sha256`, `candidate_record_file_size`, `candidate_record_file_sha256`, `record_sha256`.

Path: `admission/golden/golden-<id>.record`. The case token `eog` means the pinned backend classified an end-of-generation token and the public runtime reports `GenerationFinishReason::kEos`. It is not `GenerationFinishReason::kCanceled` and it is not an error. Immediate end of generation uses `finish_reason=eog`, `eog_observed=yes`, token size 0, visible size 0, and both digests equal to SHA-256 of zero bytes. A later end of generation excludes the end-of-generation id. `maximum_tokens` uses a non-empty list and `eog_observed=no`. `canceled` is illegal. Golden acceptance requires the candidate id, GGUF size, GGUF digest, index id, index size, and index digest to equal the opened candidate record and its index, and requires the candidate record's `gguf_size` and `gguf_sha256` to equal provenance `expected_gguf_size` and `expected_gguf_sha256` and the index `raw_size` and `raw_sha256`. Any mismatch blocks acceptance.

Golden prompt accounting uses `prompt_token_count` from the section 29 second call. The raw negative `probe_result` is not a golden field and is not compared with the case token list.


## 31. Benchmark and invariance

`invariance_record` keys: `schema_version`, `record_kind`, `invariance_id`, `model_id`, `rendered_size`, `rendered_prompt_sha256`, `prompt_tokens_size`, `prompt_tokens_sha256`, `generated_tokens_size`, `generated_tokens_sha256`, `visible_size`, `visible_output_sha256`, `finish_reason`, `eog_observed`, `max_new_tokens`, `comparison_result`, `record_sha256`.

Path: `admission/invariance/invariance-<id>.record`. `comparison_result` is `equal` or `mismatch`. Equality requires the same rendered-prompt digest, prompt tokens, generated tokens, finish reason, EOG observation, and visible digest. A mismatch blocks action-4 dataset acceptance, action-4 labels, thread-4 activation, and any invariance or benefit claim. Shadow does not satisfy the gate. Campaign 001 stays excluded. Campaign 002 stays unauthorized. Producer is 9-C27. Consumer is 9-C28 and 9-C44. Signature is `none`.

## 32. Decisions and head sets

`release_pin` keys: `schema_version`, `record_kind`, `pin_id`, `head_set_id`, `head_set_file_size`, `head_set_file_sha256`, `record_sha256`. Path: `admission/pins/pin-<id>.record`.

`head_set` keys: `schema_version`, `record_kind`, `head_set_id`, `member_list_id`, `member_list_file_size`, `member_list_file_sha256`, `member_count`, `record_sha256`. Path: `admission/heads/<head-set-id>.record`.

`head_member_list` keys: `schema_version`, `record_kind`, `list_id`, `member_count`, lines `subject_kind subject_id decision_id decision_file_size decision_file_sha256`, `record_sha256`. Lines sort by subject kind, then subject id. Duplicates fail. `member_count` equals the parsed line count. An empty member list is illegal. Path: `admission/heads/<head-set-id>.members`. This pair is the only head-set path grammar.

`operator_configuration` keys: `schema_version`, `record_kind`, `config_id`, `admission_mode`, `pin_id`, `pin_file_size`, `pin_file_sha256`, `runtime_mode`, `activation_mode`, `telemetry_mode`, `telemetry_governance_id`, `telemetry_governance_file_size`, `telemetry_governance_file_sha256`, `activation_decision_id`, `activation_decision_file_size`, `activation_decision_file_sha256`, `record_sha256`. Path: `admission/config/operator-<id>.record`. `admission_mode` is `active` or `no_model`. The telemetry triple equals the trust state's triple. Section 43 is the only legal runtime and activation pair.

`registry_entry` keys: `schema_version`, `record_kind`, `registry_entry_id`, `provenance_decision_id`, `provenance_decision_file_size`, `provenance_decision_file_sha256`, `golden_decision_id`, `golden_decision_file_size`, `golden_decision_file_sha256`, `license_decision_id`, `license_decision_file_size`, `license_decision_file_sha256`, `head_set_id`, `head_set_file_size`, `head_set_file_sha256`, `gguf_size`, `gguf_sha256`, `record_sha256`. The head set is H3. Path: `admission/model-registry/registry-<id>.record`.

`policy` keys: `schema_version`, `record_kind`, `policy_id`, `registry_decision_id`, `registry_decision_file_size`, `registry_decision_file_sha256`, `head_set_id`, `head_set_file_size`, `head_set_file_sha256`, `action_map_id`, `action_map_file_size`, `action_map_file_sha256`, `record_sha256`. The head set is H4. Path: `admission/policy/policy-<id>.record`.

`revocation_set` keys: `schema_version`, `record_kind`, `revocation_id`, `revocation_generation`, `line_count`, lines, `record_sha256`. A line is `subject_kind subject_id accepted_decision_id accepted_decision_file_size accepted_decision_file_sha256 effective_status`, sorted by kind, subject id, then decision id. Duplicates fail. `line_count` equals the parsed line count. Path: `admission/revocation/revocation-<generation>.record`. Generation 1 may have `line_count=0` and no lines.

## 33. Withdrawal and tombstones

A withdrawal note uses `statement_code=withdraw`. `subject_kind` is the decision subject kind in the table below. `subject_id` is that subject's id. `subject_file_size` and `subject_file_sha256` are that content file's full size and full-file digest. `pending_decision_id` is `none`, pending size is 0, and the pending digest is the zero digest. The transaction writes one new decision with `state=withdrawn` for that subject only. Withdrawal does not modify the lineage-origin family. The new decision's `predecessor_decision_id` is the current accepted decision, and its subject digest equals the note's subject digest.

A withdrawal that is published as a new terminal trust state uses section 14. The historical `import.transaction` does not require the resulting `current` digest to equal the imported `trust_file_sha256`. The note signer role is exactly the role in this table. The registry entry's role equals that role. The signer key is unrevoked and valid at the terminal `effective_date`. The signature family is the withdrawal row in section 9. Paths are `admission/notes/withdraw-note-<subject-id>.record`, `admission/signature-records/withdraw-signature-<subject-id>.record`, `benchmarks/generative_phase9c/signatures/withdraw-sigblob-<subject-id>.bin`, and `admission/signature-indexes/withdraw-sigindex-<subject-id>.index`.

| Decision subject kind | Required withdrawal signer role | Direct result |
|---|---|---|
| `provenance` | `provenance_reviewer` | one `withdrawn` decision |
| `license` | `license_reviewer` | one `withdrawn` decision |
| `golden_fixture` | `golden_reviewer` | one `withdrawn` decision |
| `registry_entry` | `registry_reviewer` | one `withdrawn` decision |
| `policy_pin` | `policy_reviewer` | one `withdrawn` decision |
| `telemetry_governance` | `telemetry_reviewer` | one `withdrawn` decision, then telemetry `disabled` |
| `activation_authorization` | `policy_reviewer` | one `withdrawn` decision, then an inactive pair |

Withdrawal of a source weight, a tokenizer, package bytes, or a source binding is rejected. Those objects lose authorization only when their parent subject is withdrawn: source weight, tokenizer, and source binding through `provenance`; package bytes through `registry_entry`. An adaptation artifact is withdrawn only as `policy_pin`. No other role string is a withdrawal signer.

`effective_status` is `withdrawn` for the requested subject and `revoked_by_dependency_tombstone` for every dependent. The successor revocation generation is the current generation plus 1. `trust_admin` signs the new transition that publishes the revocation set. That transition uses its signed `transition_date`. The withdrawal note uses the new successor's `effective_date`, which equals that `transition_date`.

Provenance closure, to a fixpoint, adds in kind order and then by subject id: every golden whose `candidate_gguf_sha256` equals that provenance's `expected_gguf_sha256`; every license that cites the provenance decision; every registry entry that cites those decisions; every policy that cites those registry decisions; every activation authorization that cites those policy decisions. An empty head set becomes `admission_mode=no_model`, `pin_id=none`, `pin_file_size=0`, and a zero `pin_file_sha256`. A transition that disables telemetry sets `telemetry_governance_id=none`, `telemetry_governance_file_size=0`, and the zero `telemetry_governance_file_sha256`. Rollback cannot lower the revocation generation, install an older registry, or restore a tombstoned subject by scanning. A historical accepted decision that is tombstoned does not authorize a current request.

## 34. Activation authentication

`activation_authorization` keys: `schema_version`, `record_kind`, `activation_id`, `policy_decision_id`, `policy_decision_file_size`, `policy_decision_file_sha256`, `invariance_id`, `invariance_file_size`, `invariance_file_sha256`, `candidate_id`, `candidate_record_file_size`, `candidate_record_file_sha256`, `dataset_id`, `dataset_file_size`, `dataset_file_sha256`, `model_id`, `test_only`, `record_sha256`. Path: `admission/activation/activation-<id>.record`. `test_only` is `no`. The content file is not a signature. Producer is 9-C43. Consumer is 9-C44.

Current authentication requires `review-activation-note-<activation-id>`, its section 9 signature triple, an accepted untombstoned decision that names both the content digest and the note digest, membership of that decision in the current head set, and configuration fields `activation_decision_id`, `activation_decision_file_size`, and `activation_decision_file_sha256` matching that decision. The activation-review signature uses the terminal `effective_date`. `trust_admin` compare-and-swap then publishes `activation`/`active`. A tombstone removes that authorization. Tombstone publication does not by itself select thread 1.

## 35. Action-map selection

`action_map` keys: `schema_version`, `record_kind`, `map_id`, `row_count`, lines `row threshold action`, `record_sha256`. Path: `admission/policy/action-map-<map-id>.record`. `row` is a `uint` starting at 0 and increasing by 1. `threshold` is a canonical `uint64`. `action` is `use_thread_1` or `prefer_thread_4`. Rows are sorted by strictly increasing `threshold`. Duplicate thresholds fail the file. Duplicate row numbers fail the file. `row_count` is a `uint` from 1 through 64. The parsed line count equals `row_count`. `row_count=0` and an empty line list fail the file. Producer is 9-C30. Consumer is 9-C40, 9-C41, and 9-C44. Signature is `none`. The policy-review note is the signed object. A failed map fails the load for `recommend_only`, `shadow`, and `activation`.

The request metric is `request_work_units`. `prompt_token_count` is the positive count returned by the second successful tokenizer call. `max_new_tokens` is the validated request field. Both are converted to checked `uint64`. `request_work_units` is their checked `uint64` sum. Overflow rejects the request before generation and writes no adaptive log.

Selection, after that sum exists:

- Select the row with the greatest `threshold` satisfying `threshold <= request_work_units`.
- Equality selects that row.
- If no threshold satisfies the comparison, the result is `action_map_gap`.
- The gap is not `prefer_thread_4`.

Boundary results:

| Case | Result |
|---|---|
| `request_work_units` below the smallest threshold | `action_map_gap` |
| `request_work_units` equal to a threshold | that row's action |
| `request_work_units` between two thresholds | the row with the greatest threshold still less than or equal to the work units |
| `request_work_units` above the largest threshold | the largest threshold's action |
| `request_work_units` equal to the maximum `uint64` | the same greatest-threshold rule |
| checked addition overflows | reject the request, write no adaptive log, do not select a row |
| `row_count=0` or no lines | the action-map file is rejected and is not a gap |

Section 36 states what each selected result does. Section 44 applies that result on the request path. Section 37 records it.


## 36. Action outcomes


`use_thread_1`: recommend-only recommends thread 1 and applies thread 1; shadow records would-be thread 1 and applies thread 1; activation applies thread 1 and does not require this request's `max_new_tokens` to equal the invariance record.

`prefer_thread_4`: recommend-only recommends thread 4 and applies thread 1; shadow records would-be thread 4 and applies thread 1; activation applies thread 4 only by the section 44 `llama_set_n_threads(ctx, 4, 4)` call after the section 34 chain is current and this request has `comparison_result=equal` and equal `max_new_tokens`. Failure returns `kInvarianceRequired` before KV clear, prefill, and decode, with no silent thread-1 fallback.

`action_map_gap`: recommend-only records no recommendation and applies thread 1; shadow records no would-be action and applies thread 1; activation applies thread 1. The gap is never treated as `prefer_thread_4`.

The configuration pair `activation`/`active` does not by itself set thread 4.

## 37. Telemetry variants and oversize failure

`telemetry_log` keys: the three closed orders below. `runtime_mode` selects the order. One file uses one order. Disabled mode and `no_model` write no log and do not perform the 1024-byte check. The log is written only when telemetry is `pinned`, `collection_allowed=yes`, and the request finished with `status_code=ok`. Path: `benchmarks/generative_phase9c/runs/<run-id>/log.txt`. Index: `admission/runs/log-<run-id>.index`. Class is `telemetry_log_bytes`. Signature is `none`. It is operational evidence. `duration_ns` is a `uint64`. `applied_threads` is 1 or 4. `applied_threads=4` is legal only when section 44's call `llama_set_n_threads(ctx, 4, 4)` completed for that request.

Common prefix, in this order: `schema_version`, `record_kind`, `event`, `status_code`, `runtime_mode`, `applied_threads`, `duration_ns`, `policy_id`, `policy_file_size`, `policy_file_sha256`, `map_result`, `request_work_units`, `selected_threshold`. `event` is `finish`. `status_code` is `ok`. `record_kind` is `telemetry_log`. `map_result` is `use_thread_1`, `prefer_thread_4`, or `action_map_gap`. `request_work_units` is the checked sum from section 35. `selected_threshold` is that row's `uint64` threshold, or `none` for the gap. `applied_threads` is 1 for `use_thread_1` and for `action_map_gap`. `applied_threads` is 4 only for activation with a successful `prefer_thread_4` whose thread-setting call completed.

Recommend-only order adds `recommended_threads` and then `record_sha256`. It does not contain `would_be_threads`. `recommended_threads` is `1` for `use_thread_1`, `4` for `prefer_thread_4`, and `none` only for `action_map_gap`. `runtime_mode` is `recommend_only`. `applied_threads` is 1.

Shadow order adds `would_be_threads` and then `record_sha256`. It does not contain `recommended_threads`. `would_be_threads` is `1` for `use_thread_1`, `4` for `prefer_thread_4`, and `none` only for `action_map_gap`. `runtime_mode` is `shadow`. `applied_threads` is 1.

Activation order ends with `record_sha256` and contains neither `recommended_threads` nor `would_be_threads`. `map_result` and `applied_threads` are the outcome. `runtime_mode` is `activation`.

An unknown key, a duplicate key, an unexpected key, a missing key, or a key out of the selected order rejects the log and rejects the run manifest that names it. There is one encoding of each finished event.

Section 44 restores threads before this section runs. When a log is required, the caller-visible text and token vectors stay empty until restoration, the size check, and publication finish.

1. Build the canonical bytes in memory. Do not truncate and do not drop a field.
2. If the byte count is greater than 1024, create no log file and no run manifest, leave the caller-visible output empty, and return `kTelemetryRejected`.
3. A count of 1024 or less may be published. Create the log with `O_CREAT|O_EXCL|O_NOFOLLOW`, write exactly those bytes, and `fsync` it. Then create the section 38 run manifest the same way.
4. If publication fails after the size check, unlink every file this attempt created under `runs/<run-id>/`, create no manifest, leave the caller-visible output empty, and return `kTelemetryRejected`.
5. Only a published log and manifest, or a mode that does not log, may expose generated output.

Rejected requests and canceled requests write no log, create no manifest, and do not return `kTelemetryRejected`. Disabled telemetry skips this section and returns ordinary output. Phase 9-C40 adds `kTelemetryRejected` to `GenerationError` in the section 45 files. Tests in `tests/generative_session_test.cpp` cover a 1024-byte record, a 1025-byte record, a rejected request, a canceled request, and one valid record of each of the three orders.

## 38. Fixtures and run records

`withdrawal_expected` keys: `schema_version`, `record_kind`, `fixture_id`, `fixture_role`, `input_blob_id`, `input_size`, `input_raw_sha256`, `input_index_id`, `input_index_size`, `input_index_sha256`, `expected_effective_status`, `test_only`, `record_sha256`. `fixture_role` is `direct` or `dependent`. `expected_effective_status` is `withdrawn` for direct and `revoked_by_dependency_tombstone` for dependent. `test_only` is `yes`. Paths: `tests/fixtures/withdrawal/direct.record` and `tests/fixtures/withdrawal/dependent.record`. Raw fixtures: `tests/fixtures/withdrawal/direct.bin` is exactly `provenance provenance-fixture decision-fixture withdrawn\n`, and `dependent.bin` is exactly `golden_fixture golden-fixture decision-fixture revoked_by_dependency_tombstone\n`. Indexes: `tests/fixtures/withdrawal/direct.index` and `dependent.index`, class `fixture_bytes`. Producer is 9-C33. Consumer is `tests/withdrawal_test.cpp`. Signature is `none`. These files are test-only. They are not accepted evidence and not quarantine. A production loader rejects `test_only=yes`.

`activation_expected` keys: `schema_version`, `record_kind`, `fixture_id`, `input_blob_id`, `input_size`, `input_raw_sha256`, `input_index_id`, `input_index_size`, `input_index_sha256`, `expected_runtime_mode`, `expected_activation_mode`, `test_only`, `record_sha256`. Path: `tests/fixtures/activation/activation.record`. Raw path: `tests/fixtures/activation/activation.bin`, exactly `runtime_mode=activation\nactivation_mode=active\n`. Index: `tests/fixtures/activation/activation.index`, class `fixture_bytes`. `expected_runtime_mode` is `activation`. `expected_activation_mode` is `active`. `test_only` is `yes`. Producer is 9-C35. Consumer is `tests/activation_record_test.cpp`. The fixture does not authorize production activation.

`run_manifest` keys: `schema_version`, `record_kind`, `manifest_id`, `run_id`, `phase_id`, `log_blob_id`, `log_size`, `log_raw_sha256`, `record_sha256`. Path: `benchmarks/generative_phase9c/runs/<run-id>/run.manifest`. The log path and index are the section 37 pair. When no log exists, `log_blob_id=none`, `log_size=0`, and `log_raw_sha256` is the zero digest. When telemetry is disabled, no manifest and no log are created. Producer is 9-C40, 9-C41, or 9-C44. Consumer is operator audit. Signature is `none`. This is operational evidence, not production admission authority. Cardinality is one per accepted run that collected a log.

Phase 9-C44 tests start from valid `activation`/`active` and include the section 35 boundary table: below the smallest threshold records a gap and applies thread 1; equality selects that row; a value between two thresholds selects the greatest threshold still less than or equal to the work units; a value above the largest threshold selects that largest row; overflow rejects the request and writes no log. An equal-invariance request whose selected result is `prefer_thread_4` uses thread 4. The same configuration with `use_thread_1` or `action_map_gap` uses thread 1. After the activation tombstone is published, and before compare-and-swap, the next request returns the admission error and does not clear the KV cache, prefill, decode, or use thread 1. Only after compare-and-swap publishes `disabled`/`inactive`, `recommend_only`/`inactive`, or `shadow`/`inactive` does a new request use thread 1 under that pair. The tombstoned authorization cannot select thread 4.

Phase 9-C40 tests the section 29 probe table and the section 37 size table. Phase 9-C44 tests the section 44 thread-call order: modes that stay at thread 1 record only `(1, 1)` before tokenization; successful activation `prefer_thread_4` records `(1, 1)`, then `(4, 4)` before `llama_memory_clear`, then `(1, 1)` inside the funnel before the result is returned. Prefill status 2, any other prefill failure, token-decode status 2, any other decode failure, a cancel flag after `(4, 4)`, a 1025-byte telemetry record, and a publication failure each record `(4, 4)` then `(1, 1)` and return with empty caller-visible output. `applied_threads=4` is absent unless the `(4, 4)` call completed and the request result is success. `GenerativeSessionTestAccess` observes those argument pairs. Its production function pointer is null, so production always calls `llama_set_n_threads`. A test-only non-null hook that declines restoration returns `kThreadRestoreFailed` and sets the session restore flag.

## 39. Migration bundle

The migration bootstrap bundle is ten files under `trust_root/migrations/<migration-id>/`. The directory contains only these names. Any other name, a duplicate name, or a symlink fails the bundle.

| File | Role |
|---|---|
| `package.record` | unsigned wrapper |
| `payload.record` | Ed25519 signed target |
| `signature.bin` | 64 raw signature bytes |
| `signature.index` | blob index of `signature.bin` |
| `verification.record` | verification record for the payload |
| `files.record` | payload-pinned runtime file list |
| `lineage-origin.record` | source-signed imported origin target |
| `lineage-origin.signature.bin` | 64-byte source signature |
| `lineage-origin.signature.index` | origin signature index |
| `lineage-origin.verification.record` | origin verification record |

`migration_payload` keys: `schema_version`, `record_kind`, `migration_id`, `migration_sequence`, `source_host_id`, `destination_host_id`, `trust_generation`, `trust_id`, `trust_file_size`, `trust_file_sha256`, `registry_id`, `authority_registry_file_size`, `authority_registry_file_sha256`, `authority_envelope_id`, `authority_envelope_file_size`, `authority_envelope_file_sha256`, `config_id`, `configuration_file_size`, `configuration_file_sha256`, `revocation_id`, `revocation_file_size`, `revocation_file_sha256`, `telemetry_mode`, `telemetry_governance_id`, `telemetry_governance_file_size`, `telemetry_governance_file_sha256`, `admission_manifest_id`, `admission_root_manifest_file_size`, `admission_root_manifest_file_sha256`, `file_list_id`, `file_list_path`, `file_list_file_size`, `file_list_file_sha256`, `export_date`, `replay_nonce`, `record_sha256`.

Payload path: `trust_root/migrations/<migration-id>/payload.record`. This file is the signed target. It does not contain a signature id, a signature digest, an index digest, a verification digest, or a wrapper digest. `file_list_path` is `migrations/<migration-id>/files.record`. The signer role is `trust_admin`. The context date is the terminal `effective_date` at export.

Raw signature path: `trust_root/migrations/<migration-id>/signature.bin`. Its size is 64. Index path: `trust_root/migrations/<migration-id>/signature.index`. It is a `blob_index` with `blob_class=signature_bytes`, `raw_size=64`, `raw_sha256` equal to the signature bytes, `storage_path=migrations/<migration-id>/signature.bin`, and `review_note_id=none`. Verification path: `trust_root/migrations/<migration-id>/verification.record`. It is a `signature_verification` record. `signature_id` is `migration-signature-<migration-id>`. `signed_object_kind=structured_file`, `signed_object_id=payload-<migration-id>`, `signed_object_path=migrations/<migration-id>/payload.record`, `signed_object_size` and `signed_object_sha256` are the payload's full size and full-file digest, `signature_blob_id` is `migration-sigbin-<migration-id>`, `signature_index_id` is `migration-sigindex-<migration-id>`, `signature_size=64`, `signature_sha256` equals the raw signature, `signature_index_path=migrations/<migration-id>/signature.index`, and `signer_key_id` is the `trust_admin` entry. Those ids name this bundle. They do not create a second signature file or a second index file. The verification record does not contain the wrapper digest or the file-list digest. The signature file does not contain a digest field.

`migration_package` keys: `schema_version`, `record_kind`, `migration_id`, `payload_id`, `payload_path`, `payload_file_size`, `payload_file_sha256`, `signature_path`, `signature_size`, `signature_sha256`, `signature_index_path`, `payload_signature_index_file_size`, `payload_signature_index_file_sha256`, `verification_path`, `signature_file_size`, `signature_file_sha256`, `file_list_id`, `file_list_path`, `file_list_file_size`, `file_list_file_sha256`, `origin_record_path`, `origin_record_file_size`, `origin_record_file_sha256`, `origin_signature_path`, `origin_signature_size`, `origin_signature_file_sha256`, `origin_index_path`, `origin_index_file_size`, `origin_index_file_sha256`, `origin_verification_path`, `origin_verification_file_size`, `origin_verification_file_sha256`, `record_sha256`.

Package path: `trust_root/migrations/<migration-id>/package.record`. The wrapper is not a signed target and does not authorize admission. It is accepted only when all of the following hold:

1. `payload_path`, `signature_path`, `signature_index_path`, `verification_path`, and `file_list_path` are the five companion paths in the table above.
2. Each size and digest equals the named file.
3. The index `raw_size` is 64 and its `raw_sha256` equals the raw signature.
4. The verification record's signed-object path, size, and digest equal the payload.
5. Ed25519 verifies the payload bytes under the section 9 migration-payload rule.
6. The payload's file-list id, path, size, and digest equal the wrapper's file-list id, path, size, and digest.
7. `migration_id` is the same on the wrapper, the payload, the verification record, and the file list.
8. `origin_record_path`, `origin_signature_path`, `origin_index_path`, and `origin_verification_path` are the four lineage names in the table. Each size and digest equals that file. The origin record's payload size and digest equal `payload.record`. Its imported trust tuple equals the payload trust tuple. Its `destination_host_id` and `migration_id` equal the payload. Ed25519 verifies the origin record under the same `trust_admin` key and `effective_date` as the payload. `revoked=no` holds at that date. A missing, extra, or duplicate origin companion fails the bundle.

Hash dependencies are only these edges. `payload.record` contains `file_list_file_sha256` of `files.record` and its own prefix self-checksum. `signature.bin` contains no digest field; the Ed25519 operation covers the payload bytes and the resulting 64 bytes do not re-enter the payload. `signature.index` contains `raw_sha256` of `signature.bin`. `verification.record` contains the payload full-file digest and the signature raw digest. `package.record` contains the full-file or raw digests of the payload, the signature, the index, the verification record, the file list, `lineage-origin.record`, `lineage-origin.signature.bin`, `lineage-origin.signature.index`, and `lineage-origin.verification.record`. No edge runs from the payload, the signature, the index, the verification record, or the file list back to `package.record`. The imported origin record contains the payload full-file digest and does not contain a package digest or an `import.transaction` digest. Its signature, index, and verification record follow the section 10 lineage graph. `package.record` contains the digests of those four origin files and nothing in those four files contains the package digest. Export order is: runtime `files.record`; payload target; payload signature family signed by `source_trust_admin_signer`; origin target; origin signature by that same signer; origin index; origin verification record; unsigned package wrapper. `source_trust_admin_signer` is an opaque capability supplied to `export_migration` in `include/inferedge/trust_migration.hpp`. It exposes no private-key bytes and no private-key path. The destination never calls it. No object stores a digest of a byte string that stores a digest of that same object. The graph is acyclic.

## 40. Migration closure

`migration_file_list` keys: `schema_version`, `record_kind`, `list_id`, `migration_id`, `entry_count`, then lines, then `record_sha256`. Path: `trust_root/migrations/<migration-id>/files.record`. A line is `entry <root_kind> <relative_path> <byte_size> <migration_entry_file_sha256>`. `root_kind` is `trust`, `admission`, or `evidence`. Lines sort by root kind, then relative path. A duplicate path, a duplicate object, an absolute path, a root escape, or a symlink fails. `entry_count` equals the parsed count. `migration_entry_file_sha256` hashes the exact listed bytes. `byte_size` is that file's size.

The ten bootstrap paths are mandatory import inputs and are not file-list entries. Listing any of them fails export. Those paths are `migrations/<migration-id>/package.record`, `migrations/<migration-id>/payload.record`, `migrations/<migration-id>/signature.bin`, `migrations/<migration-id>/signature.index`, `migrations/<migration-id>/verification.record`, `migrations/<migration-id>/files.record`, `migrations/<migration-id>/lineage-origin.record`, `migrations/<migration-id>/lineage-origin.signature.bin`, `migrations/<migration-id>/lineage-origin.signature.index`, and `migrations/<migration-id>/lineage-origin.verification.record`.

For the exported mode, the file list contains every runtime file below that the exported mode uses, and no other authoritative file. A missing dependency fails export. An unexpected authoritative file fails export. Destination verification uses the ten bootstrap files first, then only this closure, plus the already provisioned 32-byte bootstrap anchor.

Trust root, `root_kind=trust`:

- `genesis/transition.record`
- `genesis/signature.bin`
- `genesis/signature.index`
- `genesis/verification.record`
- for every later transition in the authenticated chain, `transitions/<transition-id>/transition.record`, `signature.bin`, `signature.index`, and `verification.record`

Admission root, `root_kind=admission`:

- every successor trust state `admission/trust/trust-<trust_id>.record`, including the terminal trust state whose bytes are copied to `current` after verification
- the admission-root manifest
- the operator configuration, revocation set, authority registry content, authority entry list `admission/authority/registry-<id>.entries`, authority envelope, and vocabulary capability
- the release pin and head-set record and member list when admission mode is `active`
- telemetry governance when pinned
- every decision, review note, note signature index, and note verification record required by the exported mode
- every provenance, binding, binding list, license, candidate, golden, case list, model-registry entry, policy, action map, dataset, invariance, and activation object required by section 18 for that mode
- every Class B provenance's tensor schema, tensor rule list, tensor inventory, and tensor histogram
- every blob index named by those records, including key indexes `admission/keys/<entry-id>.index` and signature indexes
- Class B structured siblings and indexes when the exported provenance uses Class B

Evidence root, `root_kind=evidence`, already present at the destination and not copied by import:

- every signer public-key raw blob `benchmarks/generative_phase9c/keys/<entry-id>.bin` named by a listed entry
- every registry signature blob and every note, deletion, and stale-lock signature blob named by a listed verification record
- every Class A or Class B raw blob, license blob, candidate GGUF, and golden raw blob named by a listed record

Abort records and receipt records are absent from this pre-import closure. The bootstrap key is absent from this closure. `service_status`, `migration_high_water`, `replay_nonce`, `current`, and `import.transaction` are local and are not list entries. After verification, the destination writes `current` as a byte copy of the listed terminal trust state. All ten migration-bundle files remain at their exact paths for the destination lifetime, mode `0444`, regular files, link count 1, not symlinks, and byte-identical to the verified import. They are `files.record`, `payload.record`, `signature.bin`, `signature.index`, `verification.record`, `lineage-origin.record`, `lineage-origin.signature.bin`, `lineage-origin.signature.index`, `lineage-origin.verification.record`, and `package.record`, each under `trust_root/migrations/<migration-id>/`. They are not current admission authority. Staging cleanup, import cleanup, evidence cleanup, reservation cleanup, quarantine cleanup, export, withdrawal, rollback, retirement, a later current transition, telemetry cleanup, model removal, and load recovery do not delete or replace them. No deletion-authorization grammar names them. If one is absent or altered outside these operations, a later load fails closed and does not reconstruct it. `load_admission` opens only the two Branch B files named in section 11.

Class B closure includes both payload lists defined in section 25. `payload-a.list` and `payload-b.list` are separate files. Each is closed at object count 6.

## 41. Import pre-write gates and lifecycle

`migration_abort` keys: `schema_version`, `record_kind`, `abort_id`, `migration_id`, `migration_package_file_size`, `migration_package_file_sha256`, `record_sha256`. Path: `trust_root/migrations/<abort-id>.abort.record`. Cardinality is one per abort. Producer is 9-C34. Consumer is the frozen source. Signature is the abort family in section 9, signed by `trust_admin`, with context date the frozen source's terminal `effective_date`. `migration_package_file_sha256` is the wrapper's full-file digest. It is operational trust state, not production model admission. It does not exist in the pre-import closure. Abort returns a frozen source to `active` only while the destination is `empty`.

`import_receipt` keys: `schema_version`, `record_kind`, `receipt_id`, `migration_id`, `migration_package_file_size`, `migration_package_file_sha256`, `trust_id`, `trust_file_size`, `trust_file_sha256`, `record_sha256`. Path: `trust_root/migrations/<receipt-id>.receipt.record` on the destination. Cardinality is one per successful import. Producer is 9-C34 on the destination. Consumer is the source retirement step. Signature is the receipt family in section 9, signed by `trust_admin`, with context date the imported terminal `effective_date`. It binds the wrapper digest to the imported package and `trust_file_sha256` to the imported terminal trust state. It does not exist in the pre-import closure. After a receipt, the source may become `retired`.

`import_migration(destination_trust_root, destination_admission_root, destination_evidence_root, migration_id, destination_host_id)` opens only section 3 paths, plus the admission and evidence paths the authenticated closure names, using `O_RDONLY|O_NOFOLLOW` until the write phase. The pre-write order is exactly:

1. Read the ten section 39 bootstrap-bundle files read-only.
2. Validate their paths, ids, sizes, and digests, including wrapper agreement and the section 39 origin-companion checks. A missing origin file, a bad origin signature, a wrong signer role, a revoked signer, a wrong `effective_date`, a wrong host, a wrong payload digest, or a wrong imported trust digest returns failure here.
3. Verify Ed25519 over the payload bytes using the 32-byte key named by the verification record. The key path is the section 9 evidence key path for `signer_key_id`, the key size is 32, and the key digest equals `public_key_sha256`. This step does not yet apply the registry date. The section 9 role, `revoked=no`, and `effective_date` checks run at step 9 after the chain is authenticated. The payload signature does not replace the step 8 local key equality.
4. Read and validate `files.record` read-only. Do not treat it as authority until the signature and the later chain checks succeed.
5. Read `destination_trust_root/key/bootstrap-key-v1`.
6. Require exactly 32 raw bytes.
7. Verify the listed genesis transition with that key.
8. Require SHA-256 of those 32 bytes to equal the authenticated genesis `bootstrap_key_sha256` and the `bootstrap_key_sha256` of every listed trust state.
9. Verify the trust chain and the entire source closure read-only under sections 13 and 40.
10. Read `service_status`, `migration_high_water`, `import.transaction` when that path exists, and `nonce/<replay_nonce>` when that path exists. Read every nonce file only by the payload nonce id. A fresh import, with `import.transaction` absent, requires the exact `empty` status record, `migration_sequence` greater than `high_water`, an absent nonce path, and an absent `current`. The function argument `destination_host_id`, the payload `destination_host_id`, and the imported trust state's `host_id` must be the same `project-id`. A resume is permitted only by the identity rule below. Any identity mismatch returns `migration_conflict`. No write happens in this step.

Only after step 10 succeeds, and only after the snapshot below is recorded in memory, may the importer create any file. A failure at steps 1 through 10, including bootstrap inequality or an origin-companion failure, returns failure, creates nothing, deletes nothing, and does not alter service status, high water, the nonce store, `current`, `import.transaction`, or any lineage file.

Import replaces no existing generation. It is allowed only into an empty destination: `service_status` verifies with `status=empty`, `current` is absent, and `import.transaction` is absent. `export_frozen`, `retired`, and `active` reject the import before any write. A present `current` while `import.transaction` is absent rejects the import before any write. Resume of the same transaction, defined below, is the only later call that may observe `current` or a nonce.

The pre-import snapshot is read-only and is taken at step 10:

- `service_status` is present. Its exact file size and full-file SHA-256 are `prior_status_file_size` and `prior_status_file_sha256`. `status` is `empty`.
- `current` is absent. The snapshot token is `prior_current=absent`.
- `migration_high_water` is present. Its exact file size and full-file SHA-256 are `prior_high_water_file_size` and `prior_high_water_file_sha256`. Its `high_water` value is `prior_high_water`. `migration_sequence` is greater than `prior_high_water`.
- `nonce/<replay_nonce>` is absent.
- `import.transaction` is absent.
- `import-staging/<migration-id>/` under both the trust root and the admission root is absent, or it is handled by the staging rules below only after this same snapshot has been checked on a resume that already has a matching transaction. A fresh import requires both staging directories to be absent. Those absence checks apply only to a fresh import. Resume does not take a new snapshot and does not require the nonce or `current` to be absent. Resume uses the stored transaction fields.

`import_transaction` keys: `schema_version`, `record_kind`, `transaction_id`, `migration_id`, `nonce_id`, `destination_host_id`, `migration_package_file_size`, `migration_package_file_sha256`, `payload_file_size`, `payload_file_sha256`, `prior_status_file_size`, `prior_status_file_sha256`, `prior_current`, `prior_high_water`, `prior_high_water_file_size`, `prior_high_water_file_sha256`, `trust_id`, `trust_file_size`, `trust_file_sha256`, `new_high_water`, `state`, `record_sha256`. `record_kind` is `import_transaction`. `transaction_id` equals `migration_id`. `nonce_id` equals the payload `replay_nonce`. `destination_host_id` is the payload `destination_host_id` and the function argument, compared as exact `project-id` bytes, not as a digest. `prior_current` is `absent`. `new_high_water` equals `migration_sequence` and is greater than `prior_high_water`. `trust_id`, `trust_file_size`, and `trust_file_sha256` name the terminal trust-state bytes that will become `current`. `state` is `intent_recorded` and is never rewritten. Path: `trust_root/import.transaction`. The file is created with `O_CREAT|O_EXCL|O_NOFOLLOW` and mode `0444`. Signature is `none`. Producer is 9-C34. Consumers are 9-C34 recovery and section 11. It does not admit a model. Progress after this record is derived from observed files, not from a second write to this record.

The chosen retry model is idempotent resume of this same authenticated migration. Resume returns `migration_conflict`, creates nothing, deletes nothing, replaces nothing, and publishes neither `current` nor `service_status` unless every comparison below holds. A passing resume continues at the first incomplete observable step. The nonce is not deleted, and an exact matching nonce does not forbid that resume.

Resume identity, all exact:

- `transaction_id` equals `migration_id` and equals the supplied migration id.
- `nonce_id` equals the payload `replay_nonce`.
- `destination_host_id` equals the function argument, the payload `destination_host_id`, and the imported trust state's `host_id`.
- `migration_package_file_size` and `migration_package_file_sha256` equal the supplied wrapper.
- `payload_file_size` and `payload_file_sha256` equal the supplied payload.
- `prior_status_file_size` and `prior_status_file_sha256` equal the stored snapshot.
- `prior_current` is `absent`.
- `prior_high_water`, `prior_high_water_file_size`, and `prior_high_water_file_sha256` equal the stored snapshot.
- `trust_id`, `trust_file_size`, and `trust_file_sha256` equal the authenticated terminal trust state that this import publishes.
- `new_high_water` equals the payload `migration_sequence` and is greater than `prior_high_water`.
- The live bootstrap key's SHA-256 equals the authenticated genesis `bootstrap_key_sha256` and the `bootstrap_key_sha256` of every listed trust state.
- The payload `registry_id`, registry size, and registry digest, and the verification record's `signer_key_id`, are the identities that passed step 9. Equal payload bytes already bind those registry fields; resume still parses them and rejects a disagreement.
- `record_sha256` matches the section 6 self-checksum.
- `state` is `intent_recorded`.

A second payload, a second wrapper, or a second host id under this transaction is `migration_conflict`. One destination accepts one empty-destination import. A fresh import is rejected whenever `service_status` is not the snapshotted `empty` record. After `import_completed`, a same-identity call is Case A: it does not write `current`, high water, status, or the nonce, and it may run only step 16. A different migration id, wrapper, payload, nonce, or host returns `migration_conflict`. A different migration cannot replace `import.transaction`, cannot rewrite it, and cannot reuse its nonce. Section 14 forward transitions, section 33 withdrawal transitions, section 46 rollback-as-forward-transition, and an authorized export remain available.

Observable predicates use only visible files and the section 13 walk. They are not synonyms. Recovery does not record whether a pre-crash `fsync` returned. An exact final means that publication is present. An absent closure final with an exact validated staged source may be published again only inside Case B. That permission does not publish `current`. Steps 9 through 11 run only in Case B0. `import_completed` false is never by itself permission to write `current`, `current.partial`, high water, status, or the nonce. A mismatched final or operational partial is `migration_conflict`. The live process may call a publication durable only after its own required `fsync` returns.

1. `intent_recorded`: `import.transaction` is a regular file with link count 1 and its bytes verify.
2. `closure_published`: every closure final named by the verified file list has its listed size and full-file digest.
3. `nonce_recorded`: `nonce/<nonce_id>` is a regular file with link count 1, verifies, and names this `migration_id`.
4. `high_water_published`: `migration_high_water` is a regular file with link count 1 and its bytes are the exact `new_high_water` record.
5. `import_current_present`: this is only the initial-publication predicate. It is true when the transaction verifies, `current` is a regular file with link count 1, those bytes are the imported trust state named by `trust_id`, `trust_file_size`, and `trust_file_sha256`, that chain validates under section 13, and `service_status` is still the exact snapshotted `empty` record or the exact canonical `active` record with no later section 14 successor yet published. It decides whether the initial `current` step is present and whether the first active status may be published. It is not required after a later authorized forward transition, and a descendant does not make it true.
6. `import_completed`: this is the cold-restart historical fact. It is true only when the section 4 imported lineage-origin family verifies and matches this transaction's migration id and imported trust tuple, and the section 11 payload and package checks succeed, including the closed origin, transaction, and package comparison; a missing or mismatched payload or package makes this predicate false and makes the load fail closed, and that failure is not permission to rewrite `current` unless the Case B0 rules in this section independently hold; the transaction is present, is a regular file with link count 1, and verifies, including `destination_host_id`; the exact nonce file verifies; `migration_high_water` is a regular file with link count 1 and is the exact `new_high_water` record; `service_status` is canonical `active`, `export_frozen`, or `retired`; `current` is present; `current` passes section 13; the backward walk contains the imported trust state named by `trust_id`, `trust_file_size`, and `trust_file_sha256`; every transition from that state to live `current` is authorized; each generation is exactly one greater than its predecessor; no later `revocation_generation` is smaller; tombstones stay effective; and no link is missing, malformed, or unauthorized. It stays true for live N, authorized N+1, authorized N+2, a withdrawal forward transition, a revocation forward transition, and a rollback that is a new forward transition. It does not require live `current` to equal N.
7. Imported `service_loadable`: true only in section 11 Branch B, and only when `import_completed` is true, `service_status` is the exact canonical `active` record, and sections 13, 16, 17, and 18 admit the model. It is the only model-load gate inside Branch B. It is not evaluated in Branch A or Branch C. It is not `native_service_loadable`.
8. `native_service_loadable`: true only in section 11 Branch A, and only when the native lineage-origin family verifies, `destination_host_id` equals the bound generation-1 trust state's `host_id`, the section 13 walk contains that root tuple, `import.transaction` is absent, `service_status` is the exact canonical `active` record, `current` is present and is a regular file with link count 1, every later native transition is authorized, each generation is exactly one greater than its predecessor, no later `revocation_generation` is smaller, tombstones stay effective, and sections 16, 17, and 18 admit the model. Ending at `genesis-v1` is not sufficient. It does not require `import_completed` or imported `service_loadable`. It applies to native generation 1 and to authorized native descendants, including native N+1, native N+2, and later generations. It is a logical predicate. The lineage-origin record is the durable discriminator, and this predicate is not itself an on-disk record.

No later operation changes `migration_high_water`. After `import_completed`, the import algorithm does not rewrite it. A value below `new_high_water`, a value above it, different bytes, a symlink, or a link count other than 1 is not `import_completed` and is not an incomplete import.

Status classification: when `import.transaction` is absent, exact `empty` is not an import and is not loadable, and canonical `active` is potentially `native_service_loadable`. When a verifying transaction is present, exact `empty` is an incomplete import and is not loadable; canonical `active` is an imported completion when `import_completed` is true and is then potentially loadable through imported `service_loadable`; `export_frozen` and `retired` are imported completions and are not loadable when `import_completed` is true. Any other status bytes are a conflict. A later authorized status transition does not make a completed import incomplete.

Same-identity `import_migration` and crash resume classify before any write. A fresh import is permitted only when `import.transaction` is absent, status is the exact snapshotted `empty` record, `current` is absent, and the fresh-import preconditions in this section hold. A fresh import is not Case C. Its first write is step 1. After that transaction verifies, every later call classifies again before any further write.

- Case A, completed import, `import_completed` true, which includes a valid matching imported lineage-origin family: not an incomplete resume. Do not write `current`, `current.partial`, high water, status, the nonce, a closure final, or any lineage-origin file. Do not restore N over N+1 or N+2. Step 16 may remove only expected staging names whose finals already match. Unexpected or mismatched staging is `migration_conflict` and is not deleted. Steps 9 through 11 do not run.
- Case B, valid incomplete import: the transaction verifies, including link count 1 and the full resume identity and `destination_host_id`; status is the exact snapshotted canonical `empty` record; `import_completed` is false; the nonce, high water, closure, and staging state satisfy the incomplete-import state machines in this section; and the lineage-origin family is either absent before step 8L or is the exact valid imported family after that step. `current` is Case B0 or Case B1. Any other `current` is Case C. A native origin on this destination is Case C.
- Case B0, `current` absent: this is the only case that may run steps 9 through 11. Before those steps, require the exact snapshotted `empty` status, a verifying transaction, the full resume identity, the correct destination host, the exact permitted nonce state, the exact permitted high-water state, the exact closure finals, the exact valid imported lineage-origin family, no unrelated `current`, no mismatched `current`, no unexpected `current` file type, and no conflicting `current.partial`. Then create or validate `current.partial` containing the exact imported trust-state bytes, `fsync` that partial, `fsync` the trust root, atomically rename it onto absent `current`, `fsync` the trust root, re-read `current`, and require those exact imported bytes. Only that successful re-read sets `import_current_present` true. Only then may active-status publication start. A retry of steps 9 through 11 is permitted only after a new classification that is still Case B0.
- Case B1, the exact imported `current` is already present: do not create `current.partial`, do not rename `current`, and do not rewrite `current`. Validate `current` again and derive `import_current_present` true. If status is still the exact snapshotted `empty` record, the next publication is the exact `active` record through steps 12 through 14. Steps 9 through 11 do not run.
- Case C, every other observable state: `migration_conflict`. Create nothing, delete nothing, replace nothing, and publish nothing. Do not create or rename `current.partial`. Do not write `current`, high water, status, the nonce, a closure final, or staging. Case C includes canonical `active` status with an unrelated `current`, `active` status with a broken descendant walk, exact `empty` status with any `current` other than Case B0 absence or the Case B1 imported bytes, `export_frozen` or `retired` without `import_completed`, `current` from another root, the same trust id with different bytes, a missing parent transition, a decreased `revocation_generation`, a cleared tombstone, a mismatched transaction identity, a high-water conflict, a nonce conflict, a mismatched or unexpected partial, an unknown status, a symlink, a disallowed hard link, a malformed operational file, a wrong `origin_kind`, a conflicting lineage family, a missing lineage family after `current` or a non-empty status exists, an origin and transaction mismatch, and an origin whose imported tuple is not the transaction tuple. Steps 9 through 11 are prohibited. `import_completed` false is not permission to write `current`. Status, high water, the nonce, closure, and staging stay unchanged. The previously authorized step-16 staging cleanup does not run in Case C.

Once `import_completed` is true, import resume must never write `current`, even when the live digest differs from the imported digest. Only section 14 may change `current` after that. A same-identity call that observes valid N+1 or N+2 is Case A and must not rename `current.partial` or regenerate N.

Unrelated-current attack. Canonical `active` status, a `current` from another otherwise valid trust root, and a verifying transaction that names imported N, with `import_completed` false, are Case C. The result is `migration_conflict`. The call creates no `current.partial`, renames no `current`, writes no status and no high water, and deletes nothing. A later load stays fail-closed. No implementation restores N over that unrelated `current`.

Broken-descendant attack. Canonical `active` status, a `current` that claims descent from imported N, one missing, malformed, or unauthorized transition link, and a verifying transaction are not `import_completed`. They are Case C. Import resume writes nothing and does not repair the chain by republishing N. The load admits nothing.

Empty-status current matrix, when the transaction verifies and the other incomplete-import conditions are in view. Exact `empty` status with `current` absent and every other Case B condition is Case B0. Exact `empty` status with the exact imported N and every other Case B condition is Case B1. Any other `current`, a malformed `current`, a symlink `current`, a `current` whose link count is not 1, and a mismatched `current.partial` are Case C. There is no path that repairs a non-matching `current` by replacing it.

Write order. Each step starts only after the preceding step's required fsync has returned. The first write in the list is the first write permitted after step 10.

1. Create `import.transaction` with the snapshot and the new identities. `fsync` the file. `fsync` the trust root, the parent of `import.transaction`. This must precede every other write so a crash has one intent. A crash before this step leaves every operational file untouched.
2. Create `trust_root/import-staging/<migration-id>/` and `admission_root/import-staging/<migration-id>/` with mode `0700`. Reject a symlink. `fsync` each parent. The directory exists before any staged byte is trusted to a rename.
3. Write each closure file to a `*.partial` name inside the matching staging directory. `fsync` that partial. Rename it onto the staged name. `fsync` the staging directory. A staged byte is not published until its own fsync has returned.
4. `fsync` every staged file and both staging directories again before any final rename. The rename source is durable before the directory entry changes.
5. For each closure final, in file-list order, require the final path to be absent, rename the staged file onto it, and `fsync` that final's parent directory. Immutable files are published before the nonce so a consumed nonce refers to bytes already on disk. Do not rename the bootstrap key, `current`, `service_status`, `migration_high_water`, or the nonce.
6. `fsync` every parent that received a rename in step 5. The closure is durable before `current` is created.
7. Create `nonce/<nonce_id>` with `O_CREAT|O_EXCL|O_NOFOLLOW` and mode `0444`. `fsync` the file. `fsync` `trust_root/nonce/`. The nonce is recorded before high water advances, and it is not removed afterward.
8. Write `migration_high_water.partial` with `high_water=new_high_water`. `fsync` the partial. `fsync` the trust root. Rename it onto `migration_high_water`. `fsync` the trust root. High water moves only forward, and it moves before the lineage-origin family, while status is still `empty`.
8L. Publish the imported lineage-origin family before any `current` byte. This step runs only in Case B, while `current` is absent and status is exact `empty`. Copy the four already verified bundle files byte for byte. Do not sign, do not call `source_trust_admin_signer`, and do not rebuild the record. Publish record, signature, index, and verification, each through its own partial, file `fsync`, rename onto an absent final, and trust-root `fsync`, in that order. Re-read all four finals and require the section 9 verification to succeed. A final that already matches is left in place. A mismatched final or mismatched partial is `migration_conflict` and is not overwritten or deleted. A valid final family is never replaced. Steps 9 through 11 do not start until this family verifies.
9. This step runs only in Case B0, and only after step 8L verifies, including a fresh import whose steps 1 through 8 have left `current` absent and status the exact `empty` record. Write `current.partial` with the exact terminal trust-state bytes. `fsync` the partial. `fsync` the trust root.
10. Rename `current.partial` onto `current`. `current` was absent, so this creates it.
11. `fsync` the trust root. `current` is durable only after this fsync. Status is still the pre-import `empty` file.
12. Write `service_status.partial` with `status=active`. `fsync` the partial. `fsync` the trust root. This starts only after step 11.
13. Rename `service_status.partial` onto `service_status`.
14. `fsync` the trust root. `status=active` is durable only after this fsync, and only after step 11. This is the activation publication.
15. The exact active record is now visible with exact imported `current`, so `import_completed` is true and section 11 Branch B may find imported `service_loadable`. Do not rewrite `import.transaction`. Import resume must not write `current` after this point.
16. Remove staging files whose names are closure paths for this transaction and whose finals already match. `fsync` each parent after the unlinks, then remove the empty staging directories and `fsync` their parents. An unexpected name is not removed. Failure of this cleanup does not change status or `current`.

`service_status=active` is step 13, after durable `current` at step 11. No step publishes active status earlier.

Staging rules for `import-staging/<migration-id>/`:

- Absent on a fresh import: step 2 creates it.
- Absent on resume: step 2 creates it if finals are not yet complete; otherwise skip to the next incomplete progress value.
- Present, and every present expected file matches the file list: keep those files and write only names that are still missing.
- Present but incomplete: write only missing expected names. Do not rewrite a name whose digest already matches.
- An expected staging file whose digest does not match, while its final is absent, and only while the call is Case B: unlink that one regular file after `O_NOFOLLOW`, link count 1, and the same device and inode check used in section 20, then write it again. `fsync` the file and the directory. Case C does not unlink it.
- An expected final already matches: leave the final. A stale staging copy of that name may be removed in step 16.
- An expected final exists and does not match: stop. Delete nothing else. Do not publish status.
- An unexpected name, a symlink, a non-regular file, or a directory whose migration id differs from `transaction_id`: stop. Delete nothing. Do not publish a later operational file.
- Present after `import_completed`: this is Case A. Imported `service_loadable` follows section 11 Branch B. Step 16 may remove expected names only. Unexpected names stay unopened and are not authority. Cleanup must not write `current`. Case C does not run this cleanup.

Handled failure after the first write uses these classes and no other cleanup:

| Class | Failure result |
|---|---|
| Pre-import `service_status` bytes, before step 13 | The live file is still those bytes. Do not replace it |
| `service_status` after step 14 | Leave `active`. Do not restore `empty` |
| `service_status.partial` | If its bytes are the exact active record and the live status file is the exact snapshotted empty record, resume may rename it. Any other partial is `migration_conflict` and is left in place |
| Pre-import high-water bytes, before step 8 | Leave them |
| High water after step 8 | Leave `new_high_water`. Do not write `prior_high_water` back |
| `migration_high_water.partial` | If its bytes are the exact `new_high_water` record and the live file is the exact prior record, resume may rename it. Any other partial is `migration_conflict` and is left in place |
| `current` before step 10 | It remains absent |
| `current` after step 11 | Leave the new bytes. Do not remove them |
| `current.partial` | In Case B0 only, if its bytes are the exact imported trust state and `current` is absent, resume may rename it. Any other partial is `migration_conflict` and is left in place. Case A, Case B1, and Case C do not rename it |
| Nonce | Leave it once created |
| Verifying `import.transaction` | Leave it. Do not rewrite it and do not delete it. An unverifiable file at that path is removed only by the step-1 crash repair |
| Closure finals already renamed | Leave them. They are not authority until Branch B imported `service_loadable` |
| Expected staging files | Keep a matching file. Replace only a mismatched expected staging file whose final is absent, by the rule above |
| Unexpected staging names | Leave them unopened. Delete nothing |

Steps 9 through 11 run only after classification as Case B0. Case B1 skips those steps. Case A skips those steps. Case C forbids those steps. `import_completed` false is never sufficient permission to write `current`. A failure of step 13 or 14 repeats steps 12 through 14 only in Case B1, or in Case B0 after `import_current_present` is true, and only while status is still the snapshotted empty record. That retry does not change a matching imported `current`. There is no sentence that both deletes and preserves the same operational file. A crash-table action that writes `current` or `current.partial` is legal only in Case B0. A crash-table action that writes status is legal only after `import_current_present` is true and status is still the snapshotted empty record. Case C does not use those writes.

Crash recovery reads the transaction, recomputes derived progress, and repeats the pre-write signature, registry, and bootstrap checks before any further write. The same bundle only may continue.

| Crash after | Status observed | `current` | High water | Nonce | Loads | Same migration | Other migration | Recovery action |
|---|---|---|---|---|---|---|---|---|
| Before step 1 | prior `empty` | absent | prior | absent | no | start fresh | allowed if this snapshot still holds | No file was created |
| Step 1 file write, before its fsync | prior `empty` | absent | prior | absent | no | resume if the file verifies; otherwise a fresh import after the repair in this cell | no while a transaction file remains | If the file verifies, continue at step 2. If it does not verify, unlink only `import.transaction` when `current` is absent, the nonce is absent, both staging directories are absent, and status still verifies as `empty`, using the section 20 device, inode, and link-count checks, then `fsync` the trust root. If any later file is present, fail closed and delete nothing |
| Step 1 durable | `empty` | absent | prior | absent | no | resume | no | Continue at step 2 |
| Staging directory create | `empty` | absent | prior | absent | no | resume | no | Create a missing real directory. A symlink stops and deletes nothing |
| Staged-file write | `empty` | absent | prior | absent | no | resume | no | Keep matching staged files. Replace only a mismatched expected staging file whose final is absent |
| All staged files durable | `empty` | absent | prior | absent | no | resume | no | Continue at step 5 |
| During step 5 | `empty` | absent | prior | absent | no | resume | no | Leave matching finals. Continue with finals that are still absent. A mismatched final stops and deletes nothing |
| Before step 7 | `empty` | absent | prior | absent | no | resume | no | Publish the nonce if closure finals match |
| After step 7 | `empty` | absent | prior or new | present and matching | no | resume | no | Do not delete the nonce. Continue at the first missing later step |
| Before step 8, nonce present | `empty` | absent | prior | matching | no | resume | no | Publish high water |
| After step 8 | `empty` | absent | `new_high_water` | matching | no | resume | no | Do not lower high water. Continue at step 8L |
| During step 8L, exact record partial, finals absent | `empty` | absent | new | matching | no | resume | no | Rename only that exact partial |
| During step 8L, mismatched partial or mismatched final | `empty` | absent | new | matching | no | no | no | `migration_conflict`. Leave the files. Do not overwrite a final |
| After step 8L, four finals verify | `empty` | absent | new | matching | no | resume | no | Do not replace the family. Continue at step 9 |
| During step 9 | `empty` | absent | new | matching | no | resume | no | If `current.partial` is the exact imported trust state, resume may rename it. A mismatched partial is left in place and the result is `migration_conflict` |
| After step 9 fsync, before step 10 | `empty` | absent | new | matching | no | resume | no | Rename the matching partial, then fsync the trust root |
| After step 10, before step 11 | `empty` | new bytes if the rename is visible; otherwise absent | new | matching | no | resume | no | If `current` matches, fsync the trust root. If it is not visible, repeat the rename from a matching partial or rewrite the partial |
| After step 11, before step 12 | `empty` | matching | new | matching | no | resume | no | Continue at step 12. Do not change `current` |
| After step 12 fsync, before step 13 | `empty` | matching | new | matching | no | resume | no | Rename the matching status partial |
| After step 13, before step 14 | `empty` if the rename is not visible; otherwise `active` | matching | new | matching | only if the observed status is `active` and section 11 passes | resume if status is still `empty` | no | If status is `active` and `current` is the exact imported trust state, fsync the trust root and treat `import_completed` as true. If status is still prior, repeat step 13 |
| After step 14, before step 16 | `active` | matching | new | matching | yes | no further publish | no | Leave operational files. Run step 16 |
| During step 16 | `active` | matching | new | matching | yes | no | no | Unlink only expected staging names whose finals match |

During incomplete import, `active` status with a missing or non-matching imported `current` is `migration_conflict`. It serves nothing and authorizes no deleting repair. That pair is not produced by this order. After `import_completed`, N+1 or N+2 is not incomplete merely because its digest differs from the imported digest, and Case A must not write `current`.

A second import cannot begin. Status is no longer `empty` after activation, and `import.transaction` is never replaced. High water is never written to a smaller value. Revocation generation is the generation in the published trust state and is not lowered by recovery. Tombstoned subjects in that trust state stay tombstoned. Resume does not install a different payload.

High-water cases, using the live `migration_high_water` file:

The live `migration_high_water` file is accepted only as a regular file with link count 1. A symlink or any other link count is `migration_conflict` and deletes nothing. This is the same link-count rule as `current` and the nonce. 1. Exact stored `prior_high_water` value and the exact snapshotted bytes, size, and digest, link count 1: resume may publish `new_high_water` after every earlier gate has passed.
2. Exact `new_high_water` value and the exact expected bytes, size, and digest, link count 1: high-water publication is present. Continue.
3. Numeric value less than `prior_high_water`: `migration_conflict`.
4. Numeric value strictly between `prior_high_water` and `new_high_water`: `migration_conflict`.
5. Numeric value greater than `new_high_water`: `migration_conflict`.
6. The same numeric prior or new value with different bytes, size, or digest: `migration_conflict`.
7. A missing high-water file: `migration_conflict`.
8. Any other `migration_high_water.partial`: `migration_conflict`. The one permitted partial is the exact `new_high_water` record while the live file is still case 1.

Each conflict deletes nothing, does not lower high water, does not replace `current`, does not publish status, and does not start another migration.

Nonce cases. The only permitted states are an absent `nonce/<nonce_id>` before nonce publication, and, after that publication, one regular file with link count 1 whose parsed record names this `nonce_id` and this `migration_id`. `migration_conflict`, with nothing deleted, covers a nonce id bound to another migration, another nonce id for this migration, a malformed nonce, mismatched nonce bytes, a symlink, a link count other than 1, an unexpected nonce partial, a nonce present while `import.transaction` is absent, and a transaction present with a different nonce.

Current cases during incomplete import. The only permitted states are an absent `current` before its publication, and, after that publication, the exact imported trust-state bytes with link count 1. `migration_conflict`, with nothing deleted, covers an older current, a different valid current, a malformed current, the same trust id with different bytes, a symlink, a link count other than 1, a mismatched `current.partial`, and `active` status without that exact current. After `import_completed`, `current` may change only through section 14. Import resume must not write it. Later bytes are not required to equal the historical imported digest. The section 13 chain must still contain the imported trust state.

Status cases during incomplete import. The live file must be the exact snapshotted `empty` record until the exact imported `current` is present. Activation may then publish only the exact section 4 record with `status=active`. `migration_conflict`, with nothing deleted, covers `export_frozen`, `retired`, an unknown or malformed status, `active` before `import_current_present`, the same status value with unexpected bytes, and a status partial that is not the exact active record. After `import_completed`, `export_frozen` and `retired` are completed and not loadable. They are not incomplete import, and import resume must not write status or `current`.

The loader rules in section 11 apply at every row. Status other than `active` does not answer loads. `current` is a locator until section 13, and it is not consulted for model use while status is `empty`. Staging files and closure files of an import that is not `service_loadable` are not selected by scanning and are not admission authority. `service_status` does not by itself admit a model.

A frozen, retired, or empty service does not answer loads. Import does not create generation 1. A different bundle that reuses a recorded nonce fails. Partial cleanup is not used for a pre-write failure.

Phase 9-C34 tests this section in `tests/trust_migration_test.cpp`, including `destination_host_id` mismatch, each high-water case, each nonce rejection, and refusal of a second import. Phase 9-C40 tests the loader outcomes in `tests/generative_session_test.cpp`. Both use the existing allowlists. No new source file is required. Phase 9-C40 covers the native and imported load results: transaction absent, canonical `active`, and valid native generation 1 loads; transaction absent and valid native N+1 loads; transaction absent and valid native N+2 loads; transaction absent and a broken native chain does not load; a present malformed transaction does not fall back to native loading; a transaction symlink does not fall back to native loading; an unreadable transaction does not fall back to native loading; imported N loads; imported N+1 loads; imported N+2 loads. Phase 9-C34 covers classification and write safety: a same-identity `import_migration` after N+1 does not write `current`; the same call after N+2 does not write `current`; Case B0 publishes the exact imported `current`; Case B1 does not rewrite the exact imported `current`; active unrelated `current` is Case C and writes nothing; a broken descendant is Case C and writes nothing; exact `empty` status with a different `current` is Case C and writes nothing; `import_completed` false without Case B does not permit a `current` write; Case C does not create `current.partial`; Case C does not rename `current`; Case C leaves status, high water, the nonce, closure, and staging unchanged. The same tests still cover a missing N-to-N+1 link, an N+1 chain that does not contain the imported trust state, a lower `revocation_generation`, a cleared tombstone, an export-frozen descendant that is `import_completed` and not imported `service_loadable`, a retired descendant that is `import_completed` and not imported `service_loadable`, empty status with the exact imported `current` that is not `import_completed` and not imported `service_loadable`, active status with that exact `current` that is `import_completed`, active status with an authenticated descendant that is `import_completed`, a malformed transaction that admits nothing, leftover staging cleanup after N+1 that does not write `current`, `destination_host_id` mismatch, each incomplete-import high-water case, and each nonce rejection. Phase 9-C36 tests native lineage publication in `tests/trust_service_test.cpp`. Phase 9-C34 tests imported lineage publication, crash recovery, and immutability in `tests/trust_migration_test.cpp`. Phase 9-C40 tests load results in `tests/generative_session_test.cpp`. The lineage results are: an empty destination with no origin admits no model; native origin plus native N loads; native origin plus native N+1 loads; native origin plus native N+2 loads; native origin plus a present transaction fails closed; imported origin plus a matching transaction plus imported N loads; the same plus N+1 loads; the same plus N+2 loads; imported origin plus a missing transaction fails closed and does not enter Branch A; imported origin plus a malformed transaction fails closed; a missing origin with imported `current` fails closed; deleting the transaction does not produce a native load; deleting the origin does not produce a native load; a wrong host fails closed; a wrong transaction digest fails closed; a wrong imported trust digest fails closed; an unauthorized signature fails closed; a symlink origin fails closed; a hard-linked origin fails closed; an exact partial origin family may continue only inside the authorized initialization; a conflicting final origin writes nothing; a same-identity import after N+1 changes neither origin nor `current`; a native forward transition does not change the origin; withdrawal, rollback, and export do not change the origin. Phase 9-C36 also tests the native signer and the N0 through N8 prefixes in `tests/trust_service_test.cpp`: a correct handle signs; a missing handle before N0 writes nothing; a handle whose signature does not verify under `bootstrap-key-v1` writes nothing; a length other than 64 fails; an invalid signature fails; the genesis signature is not accepted for the origin record; no test supplies a private-key path; resume at N3 or later does not call the signer; resume at N0 or N2 without a signer writes nothing further; each N0 through N8 prefix resumes at that state; a mismatched target, signature, index, or verification partial stays and writes nothing; a conflicting final stays; mixed native and imported fields conflict; a symlink partial and a hard-linked partial conflict; N8 with `current` absent may publish `current`; `current` present with empty status may publish active status; active status before N8 conflicts. Phase 9-C34 tests source signing and destination refusal in `tests/trust_migration_test.cpp`: a correct `source_trust_admin_signer` produces the family; a missing family fails before a destination write; a wrong role, a revoked signer, a wrong `effective_date`, a wrong host, a wrong payload digest, and a wrong imported trust digest fail before a destination write; the destination operation has no signing call; an origin file omitted from the wrapper fails; an extra or duplicate origin companion fails. Phase 9-C40 tests the request-time opens in `tests/generative_session_test.cpp`: an imported cold load opens exactly `payload.record` and `package.record`; a native cold load opens no bundle file; a missing payload fails closed; a missing package fails closed; a payload hash mismatch fails closed; a package hash mismatch fails closed; a package payload-field mismatch fails closed; a package origin-target mismatch fails closed; a package origin-signature mismatch fails closed; a package origin-index mismatch fails closed; a package origin-verification mismatch fails closed; a transaction payload mismatch fails closed; a transaction package mismatch fails closed; a wrong migration id fails closed; a path-traversal migration id fails closed; a symlink migration directory fails closed; a symlink payload fails closed; a symlink package fails closed; a hard-linked payload fails closed; a hard-linked package fails closed; an input or output error fails closed; an extra sibling migration directory is not selected; the loader does not list `migrations/`; a missing transaction stays in Branch B and does not enter Branch A; imported N+1 and imported N+2 load when the retained payload and package still match. Phase 9-C34 tests that staging cleanup, import cleanup, export, withdrawal, rollback, and retirement do not delete or replace any of the ten retained bundle files.


Restart classification, using only visible files:

| Visible state | `import_current_present` | `import_completed` | Imported `service_loadable` | Import call |
|---|---|---|---|---|
| No transaction, exact empty status | false | false | false | fresh import may start |
| Transaction verifies, closure incomplete, exact empty status | false | false | false | Case B resume |
| Nonce and exact `new_high_water` present, `current` absent, exact empty status | false | false | false | Case B0 may publish imported `current` |
| Exact imported `current`, exact empty status | true | false | false | Case B1 may publish active status; do not rewrite `current`; no model load |
| Exact imported `current`, exact active status, walk is N | true | true | true after section 11 | Case A; do not write `current` |
| Authorized N+1, exact active status, walk contains N | false | true | true after section 11 | Case A; do not restore N |
| Authorized N+2, exact active status, walk contains N | false | true | true after section 11 | Case A; do not restore N |
| Export-frozen, walk contains N, exact `new_high_water` | false | true | false | Case A; do not write status or `current` |
| Retired, walk contains N, exact `new_high_water` | false | true | false | Case A; do not write status or `current` |
| Active status, walk does not contain N | false | false | false | Case C; write nothing; do not republish N |
| Active status, unrelated chain | false | false | false | Case C; write nothing; do not restore N |
| Exact empty status, `current` other than absent or exact imported N | false | false | false | Case C; do not write `current` |
| Transaction present but not a verifying file | false | false | false | Branch C; no native fallback; no resume |
| Leftover expected staging after N+1 | false | true | follows status | Case A cleanup only; do not write `current` |

N+1 is not incomplete solely because its digest differs from imported N. A transaction that section 11 classifies as absent is not an import resume in this table. Canonical `active` status and a valid native chain are evaluated with `native_service_loadable` in Branch A. A present malformed, symlinked, unreadable, or otherwise non-verifying transaction is Branch C.


## 42. Offline adaptation and claim firewall

Phase 9-C29 documents offline training on an accepted dataset whose invariance result is `equal`. This contract does not start that phase, does not train, and does not adapt a live session. A trained artifact enters only as a `policy_pin` after H4 and an independent review. The runtime has no path that updates weights from requests. No speed, invariance, or benefit claim is published here. Campaign 001 is not an input. Campaign 002 is not a phase.

## 43. State matrix

The only legal runtime pairs are `disabled`/`inactive`, `recommend_only`/`inactive`, `shadow`/`inactive`, and `activation`/`active`. Generation 1 is `admission_mode=no_model`, `runtime_mode=disabled`, and `activation_mode=inactive`. `no_model` may pair only with an inactive runtime pair. `activation_mode` never overrides `runtime_mode`. Phase 9-C40's successor is `recommend_only`/`inactive`. Phase 9-C41's successor is `shadow`/`inactive`. Phase 9-C44's successor is `activation`/`active` only when section 34 is current.

| State | Pin and head set | Governance | Route B beyond the admitted model |
|---|---|---|---|
| `no_model`, telemetry disabled | forbidden; pin id `none`, size 0, zero digest | forbidden; id `none`, size 0, zero digest | none |
| `no_model`, telemetry pinned | forbidden, same zero pin | exactly one | none |
| active `disabled`/`inactive`, telemetry disabled | one each | forbidden, zero triple | model closure only |
| active `disabled`/`inactive`, telemetry pinned | one each | exactly one | model closure only |
| `recommend_only`/`inactive` | one each | the trust triple | policy and action map |
| `shadow`/`inactive` | one each | the trust triple | policy and action map |
| `activation`/`active` | one each | the trust triple | policy, action map, activation content, activation note, dataset, invariance |

## 44. Exact request control flow and thread calls

The pinned thread API is `void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch)`. The first argument is the loaded context. A completed call is success. A null context at a required call is `kThreadRestoreFailed` when the call is a restoration, and `kModelLoad` when no context is loaded.

Session-private `thread_restore_required` starts false. After a completed `(4, 4)` call it becomes true until a completed `(1, 1)` restoration clears it.

1. If `thread_restore_required` is true, call `llama_set_n_threads(ctx, 1, 1)` before any other work. Return from that void call on a non-null context clears the flag. A null context leaves the flag true, returns `kThreadRestoreFailed`, writes no log, withholds output, and does not tokenize.
2. Run sections 11, 13, 4, 16, 17, and 18. Read `service_status` from section 3. Authenticate the terminal trust state before the manifest and the mode-dependent route.
3. Reject an illegal runtime and activation pair from section 43 before opening a forbidden object.
4. If `admission_mode` is `no_model`, return `kPolicyUnavailable` before candidate loading, rendering, action-map resolution, tokenization, KV clear, prefill, and decode. Write no adaptive log.
5. For `admission_mode=active`, use only the closure required by the runtime mode. `disabled` does not open a policy or an action map.
6. Reject a structurally invalid request. `GenerationRequest::threads` must be 1 through 256 and is not an action.
7. If cancellation is already set, finish as canceled and write no log.
8. Call `llama_set_n_threads(ctx, 1, 1)` before rendering and tokenization.
9. Render the prompt and enforce the rendered-size bound.
10. Perform section 29. Probe failure or a second-call mismatch returns `kTokenization` and writes no log. A count that does not fit the context budget returns `kContextLimit` and writes no log.
11. Apply the capability range check only when `symbol_present=yes` and `range_check_enforced=yes`.
12. If `runtime_mode` is `disabled`, do not resolve an action map and do not compute `request_work_units`. Leave both thread counts at 1. Missing policy is success for this mode. Continue at step 20. Write no adaptive telemetry.
13. Convert `prompt_token_count` and `max_new_tokens` to checked `uint64` and add them. Overflow returns `kAdmissionRejected` before step 20 and writes no adaptive log.
14. If `recommend_only` or `shadow` lacks a current policy or action map, return `kPolicyUnavailable` before step 20. Both leave thread counts at 1.
15. Recommend-only runs section 35 and records the section 36 recommendation through the section 37 recommend-only order. Applied threads stay 1. There is no `(4, 4)` call.
16. Shadow runs section 35, records the section 36 would-be result through the section 37 shadow order, leaves applied threads at 1, and does not change the generated bytes. There is no `(4, 4)` call.
17. Activation requires the section 34 chain. A missing policy or map returns `kPolicyUnavailable`. A failed chain, tombstone, candidate, or dataset returns `kAdmissionRejected`. Section 35 selects the result. `use_thread_1` and `action_map_gap` leave both thread counts at 1. `prefer_thread_4` failure returns `kInvarianceRequired` before any `(4, 4)` call, before the second cancellation check, and before KV clear. There is no silent thread-1 fallback while the pair remains `activation`/`active`.
18. For activation `prefer_thread_4`, after every admission, authorization, tombstone, dataset, invariance, and request-parameter check has passed, call `llama_set_n_threads(ctx, 4, 4)`. This call is before the second cancellation check, before `llama_memory_clear`, before prefill, and before decode. When that void call returns, set `thread_restore_required` immediately. A null context at this call is `kModelLoad`, the flag stays false, and no `(4, 4)` call was completed.
19. From this step until the funnel in step 23 has run, do not return to the caller. The funnel may return `kThreadRestoreFailed` while the flag remains true if the context is null. Store one pending result. Keep generated text and token vectors in that private result. Caller-visible output stays empty.
20. Check cancellation. If it is set, store a canceled pending result. Do not prefill or decode. Enter the funnel.
21. If no pending result is stored, call `llama_memory_clear` on the memory handle from `llama_get_memory`. A null handle stores `kDecode` and enters the funnel. The pinned clear call is void. Return from that void call means the clear finished. There is no separate boolean failure. Then prefill and decode.
22. `llama_decode` status map, for prefill and for every later token decode:
    - `0` is success. Continue. No telemetry is written for this status alone.
    - `2` is cancellation. Store a canceled pending result. KV state from this request is not reused. Output stays withheld. No telemetry is written. Restoration is required. Enter the funnel.
    - Any other `int32` is failure. Store `kDecode`. KV state from this request is not reused. Output stays withheld. No telemetry is written. Restoration is required. Enter the funnel.
    A cancel flag observed after step 18, a context-limit failure discovered after step 18, `kInvalidUtf8`, `kInternal`, a null sampler, and any other runtime failure use the same funnel. Each stores its own pending error, withholds output, and writes no success telemetry.
23. After step 18, the funnel is the only return to the caller. Steps 1 through 17 may still return while the flag is false. If the funnel is entered with the flag false, do not call `llama_set_n_threads` again; threads are already `(1, 1)`, and continue at step 24. If the flag is true, call `llama_set_n_threads(ctx, 1, 1)` once. A null context stores nothing from the void API: the pending result becomes `kThreadRestoreFailed` even if another pending error was stored, the flag stays true, output stays empty, and no telemetry is written. The next request begins at step 1. A non-null context whose void call has returned has completed restoration. Set the flag false only after that return. The context thread counts are then `(1, 1)`.
24. After the flag is false, inspect the pending result. Cancellation and every failure return empty caller-visible text and empty caller-visible token vectors. They write no success telemetry and no run manifest. A success result stays private.
25. Only a success pending result with `runtime_mode` other than `disabled` may run section 37. `runtime_mode=disabled` writes no adaptive log and no run manifest. Serialize telemetry, apply the 1024-byte check, and publish the log and manifest before any caller-visible copy. A size rejection or a publication failure returns `kTelemetryRejected`, leaves caller-visible output empty, and leaves threads at `(1, 1)`. Disabled telemetry also skips section 37.
26. Copy the private text and token vectors to the caller-visible result only after step 25 succeeds or telemetry is not required. `applied_threads=4` is recorded only when step 18 completed and this success result is the one being published.

The pinned `llama_set_n_threads` call returns `void`. Completion is return from the call on a non-null context. `kThreadRestoreFailed` means the context pointer was null when restoration was required. It does not mean the void call returned an error code. No request inherits `(4, 4)` after the flag has been cleared. A request that leaves the flag true because the context was null does not expose output.

Phase 9-C40 adds `GenerativeSession::load_admission(trust_root, admission_root, evidence_root)`, the section 29 probe rule, and the errors `kAdmissionRejected`, `kPolicyUnavailable`, and `kTelemetryRejected`. Phase 9-C41 adds shadow. Phase 9-C44 adds `kInvarianceRequired`, `kThreadRestoreFailed`, the `(4, 4)` call, and this funnel. The funnel is implemented in `src/generative_session.cpp`.

## 45. Exact source allowlists

Phase 9-C40 and phase 9-C44 may change only `include/inferedge/generative_session.hpp`, `src/generative_session.cpp`, and `tests/generative_session_test.cpp`. Phase 9-C41 may change only the `.cpp` and the test. `CMakeLists.txt` already compiles those sources when `INFEREDGE_ENABLE_GENERATIVE` is on. The CLI, the overhead workers, and `models/llm/README.md` are outside those allowlists.

Phase 9-C40 and phase 9-C44 add the new `GenerationError` enumerators in `include/inferedge/generative_session.hpp` only. They do not add a source file and they do not edit `CMakeLists.txt`.


## 46. Failure and rollback

A failed lexical, digest, signature, role, date, path, symlink, cap, mode-pair, or revocation check leaves `current` in place. Operational rollback is a new forward transition under section 14. It may select an older pin only when every subject in that pin's head set is free of tombstones. It does not lower `revocation_generation`, install an older authority registry, or enable telemetry implicitly. The no-model result uses `pin_id=none`, `pin_file_size=0`, and the zero pin digest. After import activation, this forward rollback is not compared for equality with the historical imported-current digest. It still cannot lower `revocation_generation` or clear a tombstone.

## 47. Phase 9-C1 authorization

After an independent audit accepts this Part I, phase 9-C1 may create exactly `docs/phase-9-c-admission-contract.md` and copy this Part I into it. This contract does not create that file.

## 48. Roadmap through Phase 9-C44

Section 2 is the roadmap. Correctness precedes performance. Invariance precedes dataset acceptance, training documentation, and activation. Governance precedes any log. Recommend-only precedes shadow. Shadow precedes activation. Withdrawal fixtures and rollback tests precede activation. Campaign 001 is not an input. Campaign 002 is not a phase.


