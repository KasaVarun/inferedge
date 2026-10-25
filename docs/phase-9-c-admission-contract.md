PART I — COMPLETE R23 CONSOLIDATED CONTRACT

# InferEdge Phase 9-C Admission, Multi-Model Benchmarking, and Offline Adaptation Contract

This document is the complete R23 contract. Erratum E1 adds the Phase 9-C7 evidence-store integration rights in section 2. The pre-erratum bytes remain historical evidence at SHA-256 8404752f60416282fde0d12c085caf0ddede3a4acfd1e6cd8d146a1a77b46f8e. Erratum E1 adds no phase 9-C45 and does not change sections 19 through 22.

## 1. Purpose and non-goals

A model is admitted only when an accepted decision for it sits in the head set named by the release pin, the pin is named by the operator configuration, the configuration and the admission-root manifest are named by the authenticated terminal host trust state, and the revocation set does not tombstone that decision. A package manifest, a package-local hash, a package golden file, or a package approval field does not admit a model.

This contract does not add an HTTP service, a database, a frontend, a Python binding, CUDA, TensorRT, CoreML, Metal, or Vulkan. It does not add a shared ONNX session pool or an online learned policy. ONNX `adaptive_batching` is an early seal, not this policy. Campaign 001 is not a paired thread-4 label. Campaign 002 is not authorized. No speed, zero-overhead, shadow-benefit, or invariance result is published by this contract. Weights and GGUF bytes stay outside Git.

Phase 9-C1, after a later independent audit accepts this Part I, may create only `docs/phase-9-c-admission-contract.md` by copying this Part I. Phase 9-C1 does not change runtime code, download weights, execute a model, benchmark, enable telemetry, train, stage, commit, push, dispatch, or deploy.

## 2. Phase map

Each phase from 9-C1 through 9-C44 has one introducing owner. No phase 9-C45 exists. Where section 2 lists one file on two rows, the earlier phase introduces it and the later phase may edit it only for the purpose stated for that later phase.

| Phase | Exact outputs |
|---|---|
| 9-C1 | `docs/phase-9-c-admission-contract.md` |
| 9-C2 | `include/inferedge/admission_record.hpp`, `src/admission_record.cpp`, `tests/admission_record_test.cpp`, `CMakeLists.txt` |
| 9-C3 | `docs/phase-9-c3-capability.md`, `admission/capability/vocabulary.record` |
| 9-C4 | `include/inferedge/tensor_canonical.hpp`, `src/tensor_canonical.cpp`, `tests/tensor_canonical_test.cpp`, `CMakeLists.txt` |
| 9-C5 | `include/inferedge/oracle_digest.hpp`, `src/oracle_digest.cpp`, `tests/oracle_digest_test.cpp`, `CMakeLists.txt` |
| 9-C6 | `include/inferedge/evidence_store.hpp`, `src/evidence_store.cpp`, `tests/evidence_store_test.cpp`, `CMakeLists.txt` |
| 9-C7 | `include/inferedge/ed25519_verify.hpp`, `src/ed25519_verify.cpp`, `tests/ed25519_verify_test.cpp`, `src/evidence_store.cpp`, `tests/evidence_store_test.cpp`, the `AuthenticatedCapability::authenticate` declaration in `include/inferedge/evidence_store.hpp`, `CMakeLists.txt` |
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
| 9-C33 | `include/inferedge/trust_transition.hpp`, `src/trust_transition.cpp`, `tests/trust_transition_publication_test.cpp`, `tests/withdrawal_test.cpp`, the direct and dependent fixture triples under `tests/fixtures/withdrawal/`, `CMakeLists.txt`. This phase defines `publish_verified_transition`, `publish_verified_status`, `publish_withdrawal`, and `publish_ordinary_transition` exactly once. It does not define import, native bootstrap, rollback, governance, activation, export freeze, or retirement |
| 9-C34 | `include/inferedge/trust_migration.hpp`, `src/trust_migration.cpp`, `tests/trust_migration_test.cpp`, `CMakeLists.txt`. This phase defines `publish_imported_initialization`, `publish_export_freeze`, and `publish_retirement` exactly once. The export operation receives `source_trust_admin_signer` and writes the imported lineage-origin family into the section 39 bundle. The import operation verifies that family and publishes those exact bytes to the section 4 trust-root paths before imported `current`. It also writes `import.transaction` and `import-destination.record`. It does not sign the imported origin and does not rewrite the imported successor. It retains all ten section 39 files at their exact paths for the destination lifetime |
| 9-C35 | `tests/activation_record_test.cpp`, the activation fixture triple under `tests/fixtures/activation/`, `CMakeLists.txt` |
| 9-C36 | `include/inferedge/trust_service.hpp`, `src/trust_service.cpp`, `tests/trust_service_test.cpp`, `docs/phase-9-c36-trust-bundle.md`, `CMakeLists.txt`, generation-1 trust, configuration, empty revocation set, first authority registry and envelope, `admission/root/manifest-<id>.record`, and the native lineage-origin family in section 4. `publish_native_initialization` receives `native_bootstrap_signer` and publishes the genesis family, then the native lineage-origin family, then native `current`, then active status |
| 9-C37 | `include/inferedge/trust_rollback.hpp`, `src/trust_rollback.cpp`, `docs/phase-9-c37-rollback.md`, `tests/rollback_test.cpp`, `CMakeLists.txt`. `publish_rollback` is defined in `src/trust_rollback.cpp`. `tests/rollback_test.cpp` calls that function and is not a friend |
| 9-C38 | `tests/trust_transition_test.cpp`, `CMakeLists.txt`. Tests call `publish_ordinary_transition`. This phase defines no writer |
| 9-C39 | `include/inferedge/trust_governance.hpp`, `src/trust_governance.cpp`, `tests/trust_governance_test.cpp` and `CMakeLists.txt`, then `admission/governance/governance-<id>.record`, governance-review family `review-governance-note-<id>`, `review-governance-signature-<id>`, `review-governance-sigblob-<id>`, and `review-governance-sigindex-<id>`, plus governance-content family `governance-signature-<id>`, `governance-sigblob-<id>`, and `governance-sigindex-<id>`, `admission/decisions/accept-governance-<id>.record`, and the trust transition that pins it. `publish_governance_pin` is defined in `src/trust_governance.cpp` |
| 9-C40 | `include/inferedge/generative_session.hpp`, `src/generative_session.cpp`, `tests/generative_session_test.cpp`. This phase verifies the section 11 lineage branch. On Branch B it opens exactly `payload.record` and `package.record` under the verified migration id. It does not publish lineage files |
| 9-C41 | `src/generative_session.cpp`, `tests/generative_session_test.cpp` |
| 9-C42 | `docs/phase-9-c42-shadow.md` |
| 9-C43 | `include/inferedge/trust_activation.hpp`, `src/trust_activation.cpp`, `tests/trust_activation_publication_test.cpp` and `docs/phase-9-c43-activation.md`, plus `CMakeLists.txt`. `publish_activation` is defined in `src/trust_activation.cpp`. Zero or one `admission/activation/activation-<id>.record` when an equal invariance record is already current |
| 9-C44 | `include/inferedge/generative_session.hpp`, `src/generative_session.cpp`, `tests/generative_session_test.cpp` |

An edit to `CMakeLists.txt` adds only sources and tests named by the same section 2 row to the existing `inferedge_tests` target. Phase 9-C7 uses that same target. Phases 9-C40, 9-C41, and 9-C44 do not edit `CMakeLists.txt`. The absence of `CMakeLists.txt` from the Phase 9-C39 and Phase 9-C43 rows before Erratum E3 is a contract defect: those phases must add their own publication source and test, and no other phase may do it. Erratum E3 names `CMakeLists.txt` in those two rows. Each edits `CMakeLists.txt` only to add the source and test named in its own row, exactly once. `src/generate_main.cpp`, `src/generate_cli.cpp`, `src/generative_overhead_ie_worker.cpp`, and `models/llm/README.md` are in no allowlist.

Phase 9-C6 is a source-and-test foundation. It introduces the evidence store and leaves `AuthenticatedCapability::authenticate` fail-closed: it returns unavailable and stores no capability. Phase 9-C7 is a source-and-test verifier integration. At these phases `src/evidence_store.cpp` and `src/ed25519_verify.cpp` are compiled only into `inferedge_tests`. They are not linked into `inferedge`, `inferedge_core`, or the benchmark, scheduler, concurrent, or generative executables. These phases do not enable authenticated recovery, cleanup, or deletion in a production executable. A later phase may do that only when its own section 2 row names that production target and both sources. This erratum does not create that phase. Phase 9-C7 implements Ed25519 verification in `src/ed25519_verify.cpp`, which owns the one strong verifier function, and may edit `src/evidence_store.cpp` so `authenticate` calls that function directly. The function returns a verification decision, not a capability, and it cannot assign the internal stamp. The decision carries the authorization digest, signature-family id, terminal trust id, effective date, registry-entry id, signer-key id, key-valid-from, key-valid-until, role, revocation result, verification date, operation, canonical authorization path, and target id. A bare acceptance, or a decision missing any of those fields, is malformed and stores no capability. `src/evidence_store.cpp` compares every returned identity with its own opened and read inputs, the verified authorization, the authenticated trust state, and the caller arguments. A mismatch stores no capability. Only then may it assign the internal stamp. Capability construction, private state, evidence-root and admission-root capture, and destructive recovery, cleanup, and deletion remain in that file. There is no weak symbol, callback, registration, macro, dynamic load, symbol interposition, test verifier definition, alternate verifier object, public verifier subclass, or external capability factory. `inferedge_tests` links `src/ed25519_verify.cpp` exactly once together with `src/evidence_store.cpp`. Omitting that definition fails the link as an unresolved symbol. A second strong definition fails the link as a duplicate symbol. There is no runtime result for a missing verifier definition; that absence is a negative link test. A rejected signature is a runtime rejected decision. A missing or unreadable signed input is a runtime unavailable decision. A build that omits the official verifier object and substitutes another object is outside this controlled target and is forbidden by the CMake rule. Phase 9-C7 may edit `tests/evidence_store_test.cpp` to replace expectations that currently require `kUnavailable` where a real accepted decision must succeed, and to add signed-fixture coverage for accepted deletion authorization, accepted stale-lock authorization, malformed signature, wrong signature family, wrong signer, wrong role, revoked signer, a not-yet-valid key, an expired key, wrong verification date, `test_only=yes`, altered authorization bytes, altered canonical path, altered target, trust-current replacement, evidence-root replacement, admission-root replacement, cleanup retry, quarantine unexpected-content preservation, successful quarantine deletion, and successful reservation cleanup. Tests for a rejected decision and for an unavailable signed input remain. Tests do not define the verifier, assign the stamp, construct private state, use a callback, use a test friend, use an authority macro, or install global authority. Phase 9-C7 may edit only the `authenticate` declaration, adding `std::string_view trust_root` immediately before `evidence_root`. `CapabilityResult::release_capability` and every other public evidence-store declaration stay unchanged. Before the stamp, sections 9, 10, 13, and 15 apply in full to the deletion-authorization and stale-lock families. Every requirement in those sections applies, including requirements not restated in this paragraph. The following checks are included and do not reduce those sections: the exact authorization bytes are the signed preimage; the record kind and signature family are the section 9 family; `test_only=no`; the signature record, index, and exact 64-byte blob are the section 10 objects and match their blob identities; the key blob identity and exact 32-byte public key match the one registry entry; the section 13 backward walk and forward verification end in byte equality of `current` and the terminal successor; the trust id and effective date come from that authenticated state; the entry role is `trust_admin`; `valid_from <= effective_date <= valid_until`; `revoked=no`; the verification date equals that effective date; and the operation, canonical authorization path, target identity, section 3 path and no-follow checks, and full-file digest and self-checksum rules all hold. Caller strings are comparisons, not proof. The bound terminal trust id and effective date must equal the caller's `terminal_trust_state_id` and `terminal_effective_date`. The signer-key id, key-valid-from, and key-valid-until must equal the caller's `signer_key_id`, `key_valid_from`, and `key_valid_until`. There is no caller role or revocation string. The requested operation, canonical authorization path, and target identity are bound only when they match the verified authorization. `trust_root/current` is opened by the section 3 component-wise no-follow walk, without a directory scan. The same descriptor is retained and the original byte sequence is retained. Immediately before the stamp that descriptor is `fstat`ed: device, inode, regular-file type, and size must be unchanged. It is then rewound and reread in full. The reread must return the complete retained sequence. A short read, a read error, or any byte difference stores no capability. The complete reread must also equal the authenticated terminal successor. The capability does not store trust-root device, inode, or kind, and later destructive operations do not reopen the trust root. It does store the evidence-root and admission-root device, inode, and directory kind taken from the descriptors opened for those roots. Those identities are rechecked before mutation. The live evidence-lock descriptor used for mutation must match the bound evidence-root identity. An admission-root replacement is detected before mutation. Failure stores no capability. Sections 20 through 22 and descriptor-relative `O_NOFOLLOW` opens remain in force. Phase 9-C7 edits `CMakeLists.txt` only to add `src/ed25519_verify.cpp` and `tests/ed25519_verify_test.cpp` to `inferedge_tests`. Erratum E3 corrects that sequencing. Phase 9-C7 compiles into `inferedge_tests`, exactly once, only `src/evidence_store.cpp`, `src/ed25519_verify.cpp`, `tests/evidence_store_test.cpp`, and `tests/ed25519_verify_test.cpp`. Phase 9-C7 does not reference `src/trust_transition.cpp`, `src/trust_migration.cpp`, `src/trust_service.cpp`, `src/trust_rollback.cpp`, `src/trust_governance.cpp`, or `src/trust_activation.cpp`. Phase 9-C7 does not define `INFEREDGE_TRUST_PUBLICATION_TEST_SEAM`, does not create those future headers, sources, or tests, does not implement publication writers, and does not claim completion of P0-P9, S0-S3, or I0-I3. Section 14 remains normative text for the owning publication phases. Phase 9-C33 creates `src/trust_transition.cpp` and its owned tests and adds each exactly once to `inferedge_tests`. Phase 9-C33 introduces `INFEREDGE_TRUST_PUBLICATION_TEST_SEAM` as a private definition of `inferedge_tests` only, and it implements and tests the transition and status publication machinery owned by its row, including the applicable P and S states. Phase 9-C34 creates `src/trust_migration.cpp` and `tests/trust_migration_test.cpp` and adds each exactly once. It implements and tests the import, export, and retirement ownership in its row, including the applicable I states. Phase 9-C36 creates `src/trust_service.cpp` and `tests/trust_service_test.cpp` and adds each exactly once. Phase 9-C37 creates `src/trust_rollback.cpp` and `tests/rollback_test.cpp` and adds each exactly once. Phase 9-C39 creates `src/trust_governance.cpp` and `tests/trust_governance_test.cpp` and adds each exactly once. Phase 9-C43 creates `src/trust_activation.cpp` and `tests/trust_activation_publication_test.cpp` and adds each exactly once. A phase adds to `inferedge_tests` only sources and tests named in its own section 2 row. After introduction, each trust-publication source stays compiled exactly once in `inferedge_tests`. No earlier phase names a future source. No phase repeats a source already present. The private test seam begins at Phase 9-C33 and stays private to `inferedge_tests`. None of those publication sources is linked into `inferedge`, `inferedge_core`, or a benchmark, scheduler, concurrent, or generative executable or library. No production target writes trust state because the test target contains those sources. This erratum adds no phase 9-C45.

Head sets are ordered. H1 is the accepted provenance decision and precedes acquisition. H2 adds the accepted golden decision and precedes license content. H3 adds the accepted license decision. Registry content cites H3. H4 adds the accepted registry decision. Policy content cites H4. H5 adds the accepted policy decision. A content record cites a head set that already exists. Every head-set file uses `admission/heads/<head-set-id>.record` and `admission/heads/<head-set-id>.members`.

## 3. Root, path, and open rules

The trust service root, the admission root, and the evidence root are three caller-supplied directories. A contract path that begins with `benchmarks/generative_phase9c/` is resolved only under `evidence_root`. A contract path that begins with `admission/` is resolved only under `admission_root`. A contract path that begins with `trust_root/` is resolved only under the caller-supplied trust root; in the path list below, `trust_root/` is that directory. The loader and the importer do not search a working directory, a home directory, or a repository root. They do not list a directory to choose a file. They do not choose a file by name order or modification time. A file this contract does not name has no authority. Publication writers classify the opened trust-root directory before any publication write. Darwin classification reads `f_fstypename` and accepts only `apfs`. Linux classification compares `static_cast<uint32_t>(f_type)` with `0xEF53` and `0x01021994`. Values `0x794c7630`, `0x6969`, `0xFF534D42`, and `0x65735546` are unsupported. Every other `f_type` is unsupported. A failed `fstatfs`, or an empty Darwin name, is indeterminate. The host image name is not an input. `flock` returning `EINVAL` after successful classification is `kKernelLockInvariant`. `struct statfs` is not named by a publication header.

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
| `trust_root/publication-root.record` | durable device, inode, lexical path, and host binding of this trust root |
| `trust_root/status-authorizations` | directory of immutable per-operation status authorizations; not scanned for authority |
| `trust_root/status-authorizations/<status-operation-id>.record` | immutable authorization of one status operation |
| `trust_root/import-destination.record` | destination import identity; not a byte of the source successor |
| `trust_root/export-authorizations` | directory of immutable export authorizations; not scanned for authority |
| `trust_root/export-authorizations/<export-operation-id>.record` | export authorization record |
| `trust_root/export-authorizations/<export-operation-id>.signature.bin` | export authorization raw signature |
| `trust_root/export-authorizations/<export-operation-id>.signature.index` | export authorization signature index |
| `trust_root/export-authorizations/<export-operation-id>.verification.record` | export authorization verification record |
| `trust_root/genesis/transition.payload` | genesis trust-transition signed payload |
| `trust_root/genesis/transition.record` | genesis transition |
| `trust_root/genesis/signature.bin` | genesis raw signature |
| `trust_root/genesis/signature.index` | genesis signature index |
| `trust_root/genesis/verification.record` | genesis verification record |
| `trust_root/transitions/<transition-id>/transition.payload` | later trust-transition signed payload |
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

`<transition-id>`, `<migration-id>`, `<abort-id>`, `<receipt-id>`, `<nonce-id>`, `<status-operation-id>`, and `<export-operation-id>` are `project-id` values. `current.partial`, `service_status.partial`, `migration_high_water.partial`, and `import-staging/` files are not trust authority. While answering a request, `load_admission` opens the four lineage-origin finals and then applies section 11. Branch A and Branch C open no migration-bundle file. Branch B opens `import.transaction` and then exactly `migrations/<migration-id>/payload.record` and `migrations/<migration-id>/package.record`. It does not open the four lineage partials, `import-staging/`, `files.record`, `signature.bin`, `signature.index`, `verification.record`, the four bundle-copy lineage files, abort records, receipt records, or any other migration-bundle file. An open failure is not absence. A name that is not in this table, including a case-folded lineage name, is not opened and is not authority. Sections 11, 13, 14, and 41 open only paths from this table.

Admission opens use only the path grammars later sections name. During the section 41 write phase, and not before it, import may also create `admission_root/import-staging/<migration-id>/<admission-relative-path>` where the suffix is one admission path named by the authenticated closure. That staging tree is not authority. `load_admission` does not open it. Import does not create evidence files.

## 4. Operational trust-service records

`service_status` keys: `schema_version`, `record_kind`, `status`, `record_sha256`. Path: `trust_root/service_status`. Cardinality is one. `status` is `empty`, `active`, `export_frozen`, or `retired`. Producer is 9-C36 for the initial empty record and for native active status. Producer is 9-C34 for imported active status, for export freeze, and for retirement. Consumer is the loader and section 41. Signature is `none`. It is local operational state, not production admission authority, and it is not part of a trust-file digest. A missing file or an unknown status rejects the load.

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

`review_note_id=none` and a zero `review_note_file_sha256` are legal only for `public_key`, `signature_bytes`, and `fixture_bytes`. Every other class cites one review note. The note does not cite another note. A blob index is not a signed object. A symlink, an omission of either side of a pair, a duplicate path, or a file outside the closed directory set fails that object. The migration-payload signature index is the one `signature_bytes` index whose `storage_path` is `migrations/<migration-id>/signature.bin` under `trust_root`. The lineage-origin signature index is the one `signature_bytes` index whose `storage_path` is `lineage-origin.signature.bin`, whose `review_note_id` is `none`, whose review size is 0, and whose review digest is the zero digest. The trust-transition signature index is the `signature_bytes` index whose `storage_path` is `signature.bin` in `trust_root/genesis` or in `trust_root/transitions/<transition-id>`, whose `review_note_id` is `none`, whose review size is 0, and whose review digest is the zero digest. It is not an evidence-signature path. Every other `signature_bytes` index uses an evidence-signature path from section 10.

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
| `signed_payload_sha256` | `trust_transition` | trust-transition signed payload | full | complete payload file, including its checksum line and final LF | `signed_payload_size` | `signed_payload_id` | no |
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
| `binding_record_sha256` | `trust_state`, `trust_transition`, `status_authorization` | binding record named by `binding_record_kind` | full | finished binding record, including its self-checksum line and final LF; not a filesystem path | `binding_record_size` | `trust_id`, `transition_id`, or `status_operation_id` for the containing record | no |
| `portable_plan_sha256` | `status_authorization`, `import_destination_binding`, `export_authorization` | portable publication plan | full | finished portable plan record, including its self-checksum line and final LF; not a filesystem path | `portable_plan_size` | `status_operation_id`, `migration_id`, or `export_operation_id` | no |
| `writer_bound_sha256` | `status_authorization` | writer-bound identity | full | finished writer-bound record, including its self-checksum line and final LF; not a filesystem path | `writer_bound_size` | `status_operation_id` | no |
| `publication_authorization_file_sha256` | `transition_operation_binding`, `status_operation_binding` | publication authorization file | full | complete authorization file | `publication_authorization_file_size` | `publication_authorization_id` | yes with `none` only for `native_initialization` on a transition binding and for `native_initialization` or `imported_initialization` on a status binding; no for every other kind |
| `publication_authorization_preimage_sha256` | `transition_operation_binding`, `status_operation_binding` | signed authorization preimage | full | exact bytes presented to Ed25519 | `publication_authorization_file_size` | `publication_authorization_id` | yes with `none` on the same kinds as `publication_authorization_file_sha256`; no otherwise |
| `signer_publication_key_sha256` | `transition_operation_binding`, `status_operation_binding`, `export_authorization` | 32 raw key bytes, or absence | raw | those 32 bytes when the size is 32; no bytes when the size is 0 | `signer_publication_key_size` | `signer_key_id` | yes with `none` and size 0 only for native and imported status bindings; no when the size is 32 |
| `source_binding_record_sha256` | `import_destination_binding`, `portable_publication_plan` | source transition-operation-binding record | full | finished source binding record | `source_binding_record_size` | source trust id | no on `import_destination_binding`; exact zero on a portable plan whose kind is not `imported_initialization` |
| `source_successor_file_sha256` | `import_destination_binding` | source successor file | full | complete copied successor | `source_successor_file_size` | `source_successor_trust_id` | no |
| `source_transition_file_sha256` | `import_destination_binding` | source transition file | full | complete source transition | `source_transition_file_size` | `source_transition_id` | no |
| `source_current_file_sha256` | `status_operation_binding`, `export_authorization` | current trust file | full | complete `current` | `source_current_file_size` | `source_current_id` | no |
| `requested_status_prefix_sha256` | `status_operation_binding` | new service_status prefix | self | bytes before the new `record_sha256` line | `requested_status_prefix_size` | `status_operation_id` | no |
| `admission_root_manifest_file_sha256` | `trust_state`, `migration_payload` | admission-root manifest | full | complete manifest | `admission_root_manifest_file_size` | `admission_manifest_id` | no |
| `successor_trust_file_sha256` | `trust_transition` | successor trust file | full | complete file | `successor_trust_file_size` | `successor_trust_id` | no |
| `predecessor_transition_file_sha256` | `trust_transition` | predecessor transition file | full | complete predecessor transition | `predecessor_transition_file_size` | predecessor transition id | yes when predecessor id is `none` |
| `predecessor_successor_trust_file_sha256` | `trust_transition` | predecessor successor trust state | full | complete predecessor trust state | `predecessor_successor_trust_file_size` | predecessor successor trust id | yes when predecessor id is `none` |
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
| `payload_file_sha256` | `migration_package`, `import_transaction`, `import_destination_binding` | migration payload | full | complete `payload.record` | `payload_file_size` | payload id | no |
| `payload_signature_index_file_sha256` | `migration_package` | migration signature index | full | complete `signature.index` | `payload_signature_index_file_size` | signature index path | no |
| `accepted_run_manifest_file_sha256` | `quarantine_metadata` | run manifest | full | complete manifest | `accepted_run_manifest_file_size` | run id | yes with path `none` |
| `log_raw_sha256` | `run_manifest` | telemetry log | raw | exact log bytes | `log_size` | log blob id | yes when no log exists |
| `trust_file_sha256` | `migration_payload`, `import_receipt`, `import_transaction` | trust file that becomes `current` | full | complete trust file | `trust_file_size` | trust id | no |
| `migration_package_file_sha256` | `migration_abort`, `import_receipt`, `import_transaction`, `export_authorization` | migration package wrapper | full | complete `package.record` | `migration_package_file_size` | migration id | no |
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

`signed_payload_sha256` is the full-file digest of `transition.payload`. It is not zero-eligible. On `trust_transition`, `signature_file_sha256` remains the complete verification record and is not an Ed25519 input. The payload does not contain `signature_file_sha256`. `signature_index_sha256` remains kind `self`: the index bytes before `record_sha256=`, not the full index file.

Token-list canonical bytes are the ASCII `int32-list` plus one LF. Immediate end of generation uses `generated_tokens=none`, size 0, and `generated_tokens_sha256` equal to the empty-byte digest. An empty visible result uses that same empty-byte digest.

## 9. Signed-target families and trusted dates

`signature_verification` keys: `schema_version`, `record_kind`, `signature_id`, `signed_object_kind`, `signed_object_id`, `signed_object_path`, `signed_object_size`, `signed_object_sha256`, `signature_blob_id`, `signature_index_id`, `signature_index_path`, `signature_index_size`, `signature_index_sha256`, `signature_size`, `signature_sha256`, `signer_key_source`, `signer_key_id`, `public_key_size`, `public_key_sha256`, `key_index_id`, `key_index_size`, `key_index_sha256`, `fingerprint_sha256`, `algorithm`, `verification_result`, `verification_date`, `verifier_id`, `record_sha256`. When `signer_key_source` is `bootstrap_handle`, `signer_key_id` is `none`, `key_index_id` is `none`, `key_index_size` is 0, and `key_index_sha256` is the zero digest. When `signer_key_source` is `blob`, `key_index_id` is the authority `entry_id`, the index path is `admission/keys/<entry-id>.index`, and the raw key path is `benchmarks/generative_phase9c/keys/<entry-id>.bin`. When `signer_key_source` is `publisher_key`, `signer_key_id` is the publisher `public_key_blob_id` and the key path is the Class A or Class B public-key path in section 23 or section 24.

`signed_object_kind` is `raw_blob` or `structured_file`. `algorithm` is `ed25519`. `signature_size` is 64. `public_key_size` is 32. `verification_result` is `verified`. `verifier_id` is `inferedge-ed25519-v1`. `fingerprint_sha256` equals `public_key_sha256`. `signer_key_source` is `blob`, `publisher_key`, or `bootstrap_handle`. For every family except the migration payload, `signed_object_path` and `signature_index_path` are `none`. For the migration payload they are the two paths in section 39.

For every signed family except the trust transition, the Ed25519 preimage is exactly the signed object's bytes. The signature blob, the public key, and the verification file are outside that preimage. For the trust transition, the signed object is the finished `trust_transition_signed_payload` file `transition.payload`. The Ed25519 preimage is the 48-byte ASCII line `INFEREDGE-ED25519-V1 TRUST-TRANSITION-PAYLOAD 1` plus LF, immediately followed by every byte of that payload file, including its `record_sha256` line and final LF. The transition record, the signature blob, the public key, the index, and the verification file are outside that preimage. A verifier concatenates the constant and the file. It does not omit `record_sha256`, replace a field with the zero digest, or sign a prefix of `transition.record`. `signed_object_id` differs from `signature_id`. A verification record is not a signed object. No content record authorizes itself. No package wrapper approves itself.

`verification_date` is audit metadata. It must equal the `verification_context_date` defined below. It does not select that date and it does not determine validity. A mismatch rejects the verification record.

`verification_context_date` is one date, chosen as follows.

Genesis transition: registry validity checks do not apply. The fixed local bootstrap key is the trust anchor. The genesis verification record's `verification_date` equals the genesis `transition_date`. That equality is metadata. The bootstrap key has no `valid_from`, `valid_until`, or `revoked` field.

Later trust transition: verify the Ed25519 signature over the domain-separated `transition.payload` bytes before reading `transition_date` as trusted. The context date is that signed `transition_date`. Require the predecessor registry entry `valid_from <= transition_date <= valid_until` and `revoked=no`. Require `transition_date` to be greater than or equal to the predecessor transition's `transition_date`.

Evidence and admission signatures during current loading, including submission, provenance review, golden review, license review, registry review, policy review, governance review, governance content, activation review, Class A binding, Class B binding, authority-registry envelopes other than the installing check in section 15, deletion authorization, stale-lock authorization, migration abort, import receipt, and withdrawal: the context date is the authenticated terminal trust state's `effective_date`. That field is authenticated because section 12 requires it to equal the terminal transition's signed `transition_date`, and section 13 verifies that transition before the trust state is authoritative. Require `valid_from <= effective_date <= valid_until` and `revoked=no` in the current authenticated registry.

A new transition's own signature uses the later-transition rule above, not the terminal `effective_date` of the predecessor. Evidence signatures carried inside the new publication are checked against the new transition's `transition_date` after that signature verifies, and the published successor's `effective_date` equals that same `transition_date`.

Current revocation: a key with `revoked=yes` in the current authenticated registry cannot authorize current loading, even if the signature was valid on an earlier date. The historical signature bytes remain preserved and do not create current authority.

`publisher_key` is not a registry role. Class A and Class B checksum signatures are verified with the 32-byte publisher key named by the authentication record only after a `source_binding` note, signed by a registry-checked `provenance_reviewer`, covers that authentication file through the binding record's `authentication_file_size` and `authentication_file_sha256`. The publisher key has no registry validity interval. `genesis-v1` and the native lineage-origin record are the signatures that use the local bootstrap key directly. The first authority registry also uses that key under the rule below. An imported lineage-origin record does not use the bootstrap key.

Signed targets are exactly 23 families. Nineteen use storage class `evidence_signature`. One uses storage class `trust_transition_signature`. One uses storage class `migration_bundle_signature`. One uses storage class `lineage_origin_signature`. One uses storage class `trust_status_signature`. Trust-state content is not a signed family. A trust state is authenticated only as the successor bytes named by an already verified transition. The migration package wrapper is not a signed family.

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
| Trust transition | trust | `transition-payload-<transition-id>` | `transition-signature-<transition-id>` | `transition-sigblob-<transition-id>` | `transition-sigindex-<transition-id>` | bootstrap for `genesis-v1`, otherwise `trust_admin` | genesis has no registry interval; later transitions use the signed payload `transition_date` |
| Lineage origin | trust lineage | fixed `destination-lineage-v1` | `lineage-origin-signature` | fixed `lineage-origin.signature.bin` | fixed `lineage-origin.signature.index` | bootstrap for `native`; `trust_admin` for `imported` | native uses the genesis `transition_date`; imported uses the imported terminal `effective_date` |
| Export authorization | trust status | `export-authorization-<export-operation-id>` | `export-authorization-signature-<export-operation-id>` | `<export-operation-id>.signature.bin` | `<export-operation-id>.signature.index` | `trust_admin` | source terminal `effective_date` |

The first authority registry uses the fixed bootstrap key that verified `genesis-v1`. Its context date is the genesis `transition_date`, and that date is metadata only. Registry `valid_from`, `valid_until`, and `revoked` do not apply to the bootstrap key. A later authority registry is installed only by a later trust transition. The installer verifies that new transition's Ed25519 signature first, trusts `transition_date` only after that verification, and uses that new `transition_date` as the registry-signature verification context. The registry signer is an unrevoked `authority_admin` from the authenticated predecessor registry. The check requires `valid_from <= new transition_date <= valid_until`. The new `transition_date` is not earlier than the predecessor transition's `transition_date`. The predecessor transition's date is not the installing verification date.

Evidence-signature family count is 19. Trust-transition signature family count is 1. Migration-bundle signature family count is 1. Lineage-origin signature family count is 1. Trust-status signature family count is 1. Total signed-target family count is 23. The raw signature is 64 bytes. `signature_size` equals `raw_size` equals 64. `signature_sha256` equals the index `raw_sha256`. Omission, duplication, or reuse of one id on two paths fails closed.

Trust-transition result-code precedence. The first matching row wins. Later rows are not reported. A matching row forbids any further filesystem mutation. Row 24 does not delete an incomplete rename. If no row matches, the already authorized mutation may proceed.

| Order | Condition | Code | Mutation |
|---|---|---|---|
| 1 | Open, read, fstat, fsync, or rename error that is not a classified identity error | `kIo` | forbidden |
| 2 | Path escape, absolute path, empty segment, `..`, or duplicate path | `kPath` | forbidden |
| 3 | Symlink | `kSymlink` | forbidden |
| 4 | Directory where a file is required, or file where a directory is required | `kType` | forbidden |
| 5 | Other non-regular file | `kSpecial` | forbidden |
| 6 | Owner mismatch | `kOwner` | forbidden |
| 7 | Mode other than `0444` | `kForbidden` | forbidden |
| 8 | Link count other than 1 | `kLink` | forbidden |
| 9 | Structured file, preimage, or chain bound exceeded | `kCap` | forbidden |
| 10 | Required file absent | `kUnavailable` | forbidden |
| 11 | Two candidates for one id | `kDuplicate` | forbidden |
| 12 | Size field or short read disagrees with the opened bytes | `kSize` | forbidden |
| 13 | Illegal integer, digest, date, or id spelling | `kLexical` | forbidden |
| 14 | Missing, extra, reordered, or unknown field; wrong record kind; CRLF; NUL; empty value; trailing bytes; generation outside 1 through 64 | `kSchema` | forbidden |
| 15 | Self-checksum mismatch | `kChecksum` | forbidden |
| 16 | Digest mismatch, or a zero digest in a field that is not zero-eligible | `kDigest` | forbidden |
| 17 | Parsed records do not name each other | `kState` | forbidden |
| 18 | Generation jump, rollback, or unchanged generation | `kState` | forbidden |
| 19 | `transition_date` earlier than the predecessor date, or outside the signer interval | `kDate` | forbidden |
| 20 | Revoked key, wrong role, revocation-generation decrease, or revocation-identity change | `kAuthorization` | forbidden |
| 21 | Ed25519 failure, wrong key, or legacy message | `kAuthorization` | forbidden |
| 22 | Verified payload does not name the requested operation | `kAuthorization` | forbidden |
| 23 | Device, inode, or byte replacement, including same bytes on a new inode | `kRace` | forbidden |
| 24 | fsync or rename failure after authorization | `kIo` | no further mutation and no deletion |


## 10. Signature storage

Evidence storage, and only evidence storage, uses `benchmarks/generative_phase9c/signatures/<blob-id>.bin` under `evidence_root`, `admission/signature-indexes/<index-id>.index`, and `admission/signature-records/<signature-id>.record`. Notes use `admission/notes/<note-id>.record`. Those four directories contain only the 19 evidence grammars and the 11 note ids in section 9. Any other file there fails. There is no unnamed blob grammar. The migration payload signature is not stored in those directories.

Trust-transition storage is not in those directories. Genesis files are only `trust_root/genesis/transition.payload`, `genesis/transition.record`, `genesis/signature.bin`, `genesis/signature.index`, and `genesis/verification.record`. Later files are only `trust_root/transitions/<transition-id>/transition.payload`, `transition.record`, `signature.bin`, `signature.index`, and `verification.record`. The genesis transition id is the fixed project-id `genesis-v1`. The signed object is `transition.payload`. `transition.record` is the completed transition metadata record. It names the payload and the verification record. It is not the signed object. The transition-operation-binding is the logical record defined in section 14. It is not a file and it is not `transition.record`.

Lineage-origin storage is not in the evidence directories and not in a transition directory. Its only finals are `trust_root/lineage-origin.record`, `trust_root/lineage-origin.signature.bin`, `trust_root/lineage-origin.signature.index`, and `trust_root/lineage-origin.verification.record`. Its only partials are `trust_root/lineage-origin.record.partial`, `trust_root/lineage-origin.signature.partial`, `trust_root/lineage-origin.index.partial`, and `trust_root/lineage-origin.verification.partial`. There is no second name, no history directory, and no caller-chosen path segment. The Ed25519 preimage is the exact `lineage-origin.record` bytes, including `record_sha256`. The signature, index, and verification record are outside that preimage. The index is a `blob_index` with `blob_class=signature_bytes`, `raw_size=64`, and `storage_path=lineage-origin.signature.bin`. The verification record is a `signature_verification` with `signed_object_kind=structured_file`, `signed_object_id=destination-lineage-v1`, `signed_object_path=none`, `signature_id=lineage-origin-signature`, and `verification_result=verified`. Native verification uses `signer_key_source=bootstrap_handle` and the genesis `transition_date`, which is metadata only and is not a registry interval. Imported verification uses `signer_key_source=blob` and the `trust_admin` key that already verified the migration payload. That key is selected from the payload verification record, not from the lineage-origin record. The signature is checked against that key as it was authorized at `effective_date`. A later revocation does not rewrite the family, does not change `origin_kind`, and does not by itself move the destination to the other lineage. Model-admission signatures still follow the current-revocation rule. Publication writes each partial, `fsync`s that partial, renames it onto an absent final with `O_NOFOLLOW`, and `fsync`s the trust root before the next file. The order is record, signature, index, verification. All four files are mode `0444`. A load does not create or rename them. Withdrawal, rollback, export, retirement, staging cleanup, and a later trust transition do not open them for writing. Export-authorization storage is the four paths in section 3 under `trust_root/export-authorizations/`. Status-authorization storage is `trust_root/status-authorizations/<status-operation-id>.record`. Neither storage is an evidence directory. The 19 evidence grammars are unchanged. Export files and status-authorization files are mode `0444`, regular, link count 1, and are not symlinks.

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

Native initialization belongs to phase 9-C36 and is not a model load and not `import_migration`. The operation is `publish_native_initialization`, declared in `include/inferedge/trust_service.hpp` and defined in `src/trust_service.cpp`. `native_bootstrap_signer` is an opaque signing capability supplied by the caller. It exposes no private-key bytes and no private-key path. Its only signing operation accepts the exact message bytes and returns exactly 64 bytes or a failure. Those 64 bytes are an Ed25519 signature by the private key that corresponds to the 32-byte public key at `trust_root/key/bootstrap-key-v1`. The lineage-origin message is the exact canonical `lineage-origin.record` bytes, including its `record_sha256`. No domain-separation prefix is added to that lineage-origin message. The genesis trust-transition message is not `transition.record`. It is the section 9 preimage: the 48-byte domain line `INFEREDGE-ED25519-V1 TRUST-TRANSITION-PAYLOAD 1` plus LF, followed by the exact canonical `transition.payload` bytes, including that payload's `record_sha256`. The capability is not an on-disk trust root. After the origin family is durable, a load or a later transition does not call it. Loss of the handle does not invalidate a stored family.

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


Bounds: at most 64 transitions; at most 16384 bytes for one transition, one trust-transition signed payload, one trust state, one signature index, or one verification record, and at most 16432 bytes for one trust-transition Ed25519 input; exactly 64 signature bytes; exactly 32 public-key bytes; at most 4194304 bytes for every transition file, signature, index, verification record, and successor trust state opened by one load. `current` is read only inside the 16384-byte bound.

An immutable signed transition does not name a future successor. No transition field is a pointer to a later transition.

1. Read exactly `trust_root/key/bootstrap-key-v1`. This read does not use any field of `current` or of any trust state.
2. Require exactly 32 raw bytes. Hexadecimal, Base64, PEM, DER, and path text are not a key.
3. Open only `trust_root/genesis/transition.payload`, `genesis/transition.record`, `genesis/signature.bin`, `genesis/signature.index`, and `genesis/verification.record`.
4. Require the transition id `genesis-v1`, `trust_generation=1`, and predecessor id `none`.
5. Require predecessor size 0 and the zero predecessor digest.
6. Verify the 64-byte signature over the domain-separated `transition.payload` bytes with the bootstrap key before trusting `transition_date`. Do not treat `transition.record` as the signed message.
7. Require the signature index `raw_size=64` and `raw_sha256` equal to the signature bytes.
8. Require the verification record to name this transition, this signature, and this bootstrap key.
9. Do not apply a registry validity interval to the bootstrap key.
10. Require `successor_trust_file_size` and `successor_trust_file_sha256` to match the generation-1 trust state.
11. Require that trust state's `effective_date` to equal the genesis `transition_date`.
12. Require the genesis verification record's `verification_date` to equal that `transition_date`.
13. Do not treat `trust_root/current` as authenticated at this step.

`bootstrap_key_handle` is `bootstrap-key-v1`. The bootstrap key is not a registry entry and is not an evidence file. Service status bytes are not part of the trust-file digest.

## 12. Immutable trust transitions

`trust_transition` keys: `schema_version`, `record_kind`, `transition_id`, `trust_generation`, `transition_date`, `predecessor_transition_id`, `predecessor_transition_path`, `predecessor_transition_file_size`, `predecessor_transition_file_sha256`, `predecessor_successor_trust_id`, `predecessor_successor_trust_file_size`, `predecessor_successor_trust_file_sha256`, `successor_trust_id`, `successor_trust_path`, `successor_trust_file_size`, `successor_trust_file_sha256`, `bootstrap_key_sha256`, `signer_key_id`, `registry_id`, `authority_registry_file_size`, `authority_registry_file_sha256`, `signed_payload_id`, `signed_payload_path`, `signed_payload_size`, `signed_payload_sha256`, `signature_id`, `signature_blob_id`, `signature_index_id`, `signature_file_size`, `signature_file_sha256`, `publication_kind`, `binding_record_kind`, `binding_record_size`, `binding_record_sha256`, `record_sha256`. `trust_transition_signed_payload` keys: `schema_version`, `record_kind`, `transition_id`, `trust_generation`, `transition_date`, `predecessor_transition_id`, `predecessor_transition_path`, `predecessor_transition_file_size`, `predecessor_transition_file_sha256`, `predecessor_successor_trust_id`, `predecessor_successor_trust_file_size`, `predecessor_successor_trust_file_sha256`, `successor_trust_id`, `successor_trust_path`, `successor_trust_file_size`, `successor_trust_file_sha256`, `bootstrap_key_sha256`, `signer_key_id`, `registry_id`, `authority_registry_file_size`, `authority_registry_file_sha256`, `publication_kind`, `binding_record_kind`, `binding_record_size`, `binding_record_sha256`, `record_sha256`. The payload file is `transition.payload`. Its Ed25519 preimage is the section 9 domain line plus the exact payload file. `signed_payload_id` is `transition-payload-<transition-id>`. `signature_id` is `transition-signature-<transition-id>`. `signature_blob_id` is `transition-sigblob-<transition-id>`. `signature_index_id` is `transition-sigindex-<transition-id>`. The transition copies every payload key from `transition_id` through `authority_registry_file_sha256`, and the publication and binding keys, and it does not copy the payload `record_sha256`. `signature_file_size` and `signature_file_sha256` name the complete verification record. They are outside the Ed25519 preimage. The transition-operation-binding remains the logical record in section 14 and is not `transition.record`.

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
8. Verify generations 2 and later forward. Verify each Ed25519 signature over that transition's domain-separated `transition.payload` with the `trust_admin` key authorized by the predecessor's already verified successor before trusting `transition_date` or treating the new transition as authority. Reject a signature whose message is `transition.record`, a prefix of that record, or a zero-substituted field.

Immediate identity revalidation. Immediately before every filesystem mutation authorized by a verified trust transition, including stale-lock cleanup, reservation cleanup, quarantine deletion, retry, and current-pointer publication, the caller re-resolves each identity-dependent path with openat on the retained trust-root directory descriptor and O_NOFOLLOW. The opened object must be a regular file, mode `0444`, link count 1, with the same device, inode, and size captured during verification. A reread or full-file digest must equal the verified bytes. Equal bytes on a different device or inode return `kRace`. A failed revalidation performs no target mutation.
9. Require the signed `transition_date` to be inside that key's interval, `revoked=no`, and not earlier than the predecessor `transition_date`.
10. Verify each new successor trust state before using its registry to authorize the next signer. Require its `effective_date` to equal the transition `transition_date` just verified.
11. Require the final successor bytes to equal `current`.
12. Only then treat `current` as the authenticated terminal trust state. Admission signatures then use that state's `effective_date` as `verification_context_date`.

Untrusted locators grant no authority. The loader does not scan `trust_root` and does not select a newest file.

## 14. Trust publication

`trust_state` keys, in this order: `schema_version`, `record_kind`, `trust_id`, `trust_generation`, `effective_date`, `terminal_transition_id`, `terminal_transition_path`, `revocation_generation`, `revocation_id`, `revocation_file_size`, `revocation_file_sha256`, `config_id`, `configuration_file_size`, `configuration_file_sha256`, `registry_id`, `authority_registry_file_size`, `authority_registry_file_sha256`, `authority_envelope_id`, `authority_envelope_file_size`, `authority_envelope_file_sha256`, `bootstrap_key_handle`, `bootstrap_key_size`, `bootstrap_key_sha256`, `admission_manifest_id`, `admission_root_manifest_file_size`, `admission_root_manifest_file_sha256`, `host_id`, `publication_kind`, `publication_authorization_id`, `publication_authorization_path`, `publication_authorization_file_size`, `publication_authorization_file_sha256`, `publication_authorization_preimage_sha256`, `signer_key_id`, `signer_publication_key_size`, `signer_publication_key_sha256`, `signature_family_id`, `lineage_kind`, `binding_record_kind`, `binding_record_size`, `binding_record_sha256`, `telemetry_mode`, `telemetry_governance_id`, `telemetry_governance_file_size`, `telemetry_governance_file_sha256`, `predecessor_trust_id`, `predecessor_file_size`, `predecessor_file_sha256`, `record_sha256`.

`record_kind` is `trust_state`. `schema_version` is `1`. `bootstrap_key_size` is 32. Generation 1 uses predecessor id `none`, predecessor size 0, and the zero predecessor digest. The successor path is `admission/trust/trust-<trust_id>.record` under `admission_root`. After section 13, `trust_root/current` is byte-for-byte identical to that file. Trust-state bytes are not a signed object. `effective_date` equals the terminal transition's signed `transition_date`. `binding_record_kind` is `transition_operation_binding`. `binding_record_sha256` is the full-file SHA-256 of that binding record. A trust state that contains `terminal_transition_file_size` or `terminal_transition_file_sha256` is rejected.

A `trust_state` rejects every key in this prohibited list: `terminal_transition_file_size`, `terminal_transition_file_sha256`, `expected_status`, `requested_status`, `export_operation_id`, `receipt_id`, `status_operation_id`, `device`, `inode`, `canonical_path`, `writer_bound_sha256`, `root_device`, `root_inode`, `canonical_root_path`, `destination_nonce_id`, `package_id`. Publication kinds `imported_initialization`, `export_freeze`, and `retirement` are prohibited as `trust_state.publication_kind`.

The transition-operation-binding record is not a file. `record_kind` is `transition_operation_binding`. Its keys, in this order, are: `schema_version`, `record_kind`, `publication_kind`, `publication_authorization_id`, `publication_authorization_path`, `publication_authorization_file_size`, `publication_authorization_file_sha256`, `publication_authorization_preimage_sha256`, `signer_key_id`, `signer_publication_key_size`, `signer_publication_key_sha256`, `signature_family_id`, `trust_id`, `trust_generation`, `effective_date`, `terminal_transition_id`, `terminal_transition_path`, `predecessor_trust_id`, `predecessor_file_size`, `predecessor_file_sha256`, `host_id`, `lineage_kind`, `revocation_generation`, `revocation_id`, `revocation_file_size`, `revocation_file_sha256`, `config_id`, `configuration_file_size`, `configuration_file_sha256`, `registry_id`, `authority_registry_file_size`, `authority_registry_file_sha256`, `authority_envelope_id`, `authority_envelope_file_size`, `authority_envelope_file_sha256`, `bootstrap_key_handle`, `bootstrap_key_size`, `bootstrap_key_sha256`, `admission_manifest_id`, `admission_root_manifest_file_size`, `admission_root_manifest_file_sha256`, `record_sha256`. It excludes `binding_record_sha256`, `binding_record_size`, the successor full-file digest, the transition full-file digest, `portable_plan_size`, `portable_plan_sha256`, `writer_bound_size`, `writer_bound_sha256`, `device`, `inode`, and `canonical_path`. `binding_record_size` is the byte length of the finished binding record. `binding_record_sha256` is SHA-256 of every byte of that finished record, including its self-checksum line and final LF. The successor and the transition serialize this same key list from the same values and therefore the same bytes. The transition stores that digest and does not store a second formula.

Transition-operation-binding field rules:

| Field | withdrawal | ordinary | revocation | rollback | native_initialization | governance_pin | activation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `publication_kind` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `publication_authorization_id` | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | required concrete value | required concrete value |
| `publication_authorization_path` | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | required concrete value | required concrete value |
| `publication_authorization_file_size` | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | required concrete value | required concrete value |
| `publication_authorization_file_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | required concrete value | required concrete value |
| `publication_authorization_preimage_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | required concrete value | required concrete value |
| `signer_key_id` | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | required concrete value | required concrete value |
| `signer_publication_key_size` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `signer_publication_key_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `signature_family_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `lineage_kind` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `predecessor_trust_id` | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | required concrete value | required concrete value |
| `predecessor_file_size` | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | required concrete value | required concrete value |
| `predecessor_file_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | required concrete value | required concrete value |
| `trust_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `effective_date` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `terminal_transition_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `terminal_transition_path` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `host_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `revocation_generation` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `revocation_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `revocation_file_size` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `revocation_file_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `config_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `configuration_file_size` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `configuration_file_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `registry_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `authority_registry_file_size` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `authority_registry_file_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `authority_envelope_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `authority_envelope_file_size` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `authority_envelope_file_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `bootstrap_key_handle` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `bootstrap_key_size` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `bootstrap_key_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `admission_manifest_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `admission_root_manifest_file_size` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `admission_root_manifest_file_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `trust_generation` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |

Constraint for each cell above, in the same field order and the same kind order:

`publication_kind`: `withdrawal` = token `withdrawal`; `ordinary` = token `ordinary`; `revocation` = token `revocation`; `rollback` = token `rollback`; `native_initialization` = token `native_initialization`; `governance_pin` = token `governance_pin`; `activation` = token `activation`.
`publication_authorization_id`: `withdrawal` = existing signed object bytes; zero digest prohibited; `ordinary` = existing signed object bytes; zero digest prohibited; `revocation` = existing signed object bytes; zero digest prohibited; `rollback` = existing signed object bytes; zero digest prohibited; `native_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `governance_pin` = existing signed object bytes; zero digest prohibited; `activation` = existing signed object bytes; zero digest prohibited.
`publication_authorization_path`: `withdrawal` = existing signed object bytes; zero digest prohibited; `ordinary` = existing signed object bytes; zero digest prohibited; `revocation` = existing signed object bytes; zero digest prohibited; `rollback` = existing signed object bytes; zero digest prohibited; `native_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `governance_pin` = existing signed object bytes; zero digest prohibited; `activation` = existing signed object bytes; zero digest prohibited.
`publication_authorization_file_size`: `withdrawal` = existing signed object bytes; zero digest prohibited; `ordinary` = existing signed object bytes; zero digest prohibited; `revocation` = existing signed object bytes; zero digest prohibited; `rollback` = existing signed object bytes; zero digest prohibited; `native_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `governance_pin` = existing signed object bytes; zero digest prohibited; `activation` = existing signed object bytes; zero digest prohibited.
`publication_authorization_file_sha256`: `withdrawal` = existing signed object bytes; zero digest prohibited; `ordinary` = existing signed object bytes; zero digest prohibited; `revocation` = existing signed object bytes; zero digest prohibited; `rollback` = existing signed object bytes; zero digest prohibited; `native_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `governance_pin` = existing signed object bytes; zero digest prohibited; `activation` = existing signed object bytes; zero digest prohibited.
`publication_authorization_preimage_sha256`: `withdrawal` = existing signed object bytes; zero digest prohibited; `ordinary` = existing signed object bytes; zero digest prohibited; `revocation` = existing signed object bytes; zero digest prohibited; `rollback` = existing signed object bytes; zero digest prohibited; `native_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `governance_pin` = existing signed object bytes; zero digest prohibited; `activation` = existing signed object bytes; zero digest prohibited.
`signer_key_id`: `withdrawal` = one project-id; `none` prohibited; `ordinary` = one project-id; `none` prohibited; `revocation` = one project-id; `none` prohibited; `rollback` = one project-id; `none` prohibited; `native_initialization` = signer_key_id `none`; `governance_pin` = one project-id; `none` prohibited; `activation` = one project-id; `none` prohibited.
`signer_publication_key_size`: `withdrawal` = integer `32`; `ordinary` = integer `32`; `revocation` = integer `32`; `rollback` = integer `32`; `native_initialization` = integer `32`; `governance_pin` = integer `32`; `activation` = integer `32`.
`signer_publication_key_sha256`: `withdrawal` = raw SHA-256 of the 32-byte public key; zero digest prohibited; `ordinary` = raw SHA-256 of the 32-byte public key; zero digest prohibited; `revocation` = raw SHA-256 of the 32-byte public key; zero digest prohibited; `rollback` = raw SHA-256 of the 32-byte public key; zero digest prohibited; `native_initialization` = raw SHA-256 of the 32-byte public key; zero digest prohibited; `governance_pin` = raw SHA-256 of the 32-byte public key; zero digest prohibited; `activation` = raw SHA-256 of the 32-byte public key; zero digest prohibited.
`signature_family_id`: `withdrawal` = token `trust_transition`; `ordinary` = token `trust_transition`; `revocation` = token `trust_transition`; `rollback` = token `trust_transition`; `native_initialization` = token `trust_transition`; `governance_pin` = token `trust_transition`; `activation` = token `trust_transition`.
`lineage_kind`: `withdrawal` = token equal to the predecessor trust state's `lineage_kind`; `ordinary` = token equal to the predecessor trust state's `lineage_kind`; `revocation` = token equal to the predecessor trust state's `lineage_kind`; `rollback` = token equal to the predecessor trust state's `lineage_kind`; `native_initialization` = token `native`; `governance_pin` = token equal to the predecessor trust state's `lineage_kind`; `activation` = token equal to the predecessor trust state's `lineage_kind`.
`predecessor_trust_id`: `withdrawal` = the predecessor successor id, size, and full-file digest; `ordinary` = the predecessor successor id, size, and full-file digest; `revocation` = the predecessor successor id, size, and full-file digest; `rollback` = the predecessor successor id, size, and full-file digest; `native_initialization` = predecessor id `none`, size `0`, digest 64 ASCII `0` characters; `governance_pin` = the predecessor successor id, size, and full-file digest; `activation` = the predecessor successor id, size, and full-file digest.
`predecessor_file_size`: `withdrawal` = the predecessor successor id, size, and full-file digest; `ordinary` = the predecessor successor id, size, and full-file digest; `revocation` = the predecessor successor id, size, and full-file digest; `rollback` = the predecessor successor id, size, and full-file digest; `native_initialization` = predecessor id `none`, size `0`, digest 64 ASCII `0` characters; `governance_pin` = the predecessor successor id, size, and full-file digest; `activation` = the predecessor successor id, size, and full-file digest.
`predecessor_file_sha256`: `withdrawal` = the predecessor successor id, size, and full-file digest; `ordinary` = the predecessor successor id, size, and full-file digest; `revocation` = the predecessor successor id, size, and full-file digest; `rollback` = the predecessor successor id, size, and full-file digest; `native_initialization` = predecessor id `none`, size `0`, digest 64 ASCII `0` characters; `governance_pin` = the predecessor successor id, size, and full-file digest; `activation` = the predecessor successor id, size, and full-file digest.
`trust_id`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`effective_date`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`terminal_transition_id`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`terminal_transition_path`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`host_id`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`revocation_generation`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`revocation_id`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`revocation_file_size`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`revocation_file_sha256`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`config_id`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`configuration_file_size`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`configuration_file_sha256`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`registry_id`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`authority_registry_file_size`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`authority_registry_file_sha256`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`authority_envelope_id`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`authority_envelope_file_size`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`authority_envelope_file_sha256`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`bootstrap_key_handle`: `withdrawal` = token `bootstrap-key-v1`; `ordinary` = token `bootstrap-key-v1`; `revocation` = token `bootstrap-key-v1`; `rollback` = token `bootstrap-key-v1`; `native_initialization` = token `bootstrap-key-v1`; `governance_pin` = token `bootstrap-key-v1`; `activation` = token `bootstrap-key-v1`.
`bootstrap_key_size`: `withdrawal` = integer `32`; `ordinary` = integer `32`; `revocation` = integer `32`; `rollback` = integer `32`; `native_initialization` = integer `32`; `governance_pin` = integer `32`; `activation` = integer `32`.
`bootstrap_key_sha256`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`admission_manifest_id`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`admission_root_manifest_file_size`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`admission_root_manifest_file_sha256`: `withdrawal` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `ordinary` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `revocation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `rollback` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `native_initialization` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `governance_pin` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited; `activation` = the generation-1 or successor value required by sections 11 and 15; zero digest prohibited.
`trust_generation`: `withdrawal` = predecessor generation plus 1; `ordinary` = predecessor generation plus 1; `revocation` = predecessor generation plus 1; `rollback` = predecessor generation plus 1; `native_initialization` = integer `1`; `governance_pin` = predecessor generation plus 1; `activation` = predecessor generation plus 1.

Every `trust_state` field that is also a binding key uses the same cell. `telemetry_mode`, `telemetry_governance_id`, `telemetry_governance_file_size`, and `telemetry_governance_file_sha256` are trust-state keys and are not binding keys. Their values remain the sentences in this section that begin `Before phase 9-C39` and `A transition to disabled telemetry`, and the section 46 rollback sentence. Tombstones, pin identity, runtime mode, and admission mode are not binding keys. Section 33, section 34, and section 46 govern them.

The status-operation-binding record is not a file. `record_kind` is `status_operation_binding`. Its keys, in this order, are: `schema_version`, `record_kind`, `status_operation_id`, `publication_kind`, `expected_status`, `requested_status`, `expected_status_file_size`, `expected_status_file_sha256`, `requested_status_prefix_size`, `requested_status_prefix_sha256`, `source_current_id`, `source_current_file_size`, `source_current_file_sha256`, `publication_authorization_id`, `publication_authorization_path`, `publication_authorization_file_size`, `publication_authorization_file_sha256`, `publication_authorization_preimage_sha256`, `signer_key_id`, `signer_publication_key_size`, `signer_publication_key_sha256`, `signature_family_id`, `host_id`, `portable_plan_sha256`, `record_sha256`. It excludes `device`, `inode`, `canonical_path`, `writer_bound_sha256`, successor bytes, and transition full-file digests. Kinds `withdrawal`, `ordinary`, `revocation`, `rollback`, `governance_pin`, and `activation` are prohibited from serializing this record.

Status-operation-binding field rules:

| Field | native_initialization | imported_initialization | export_freeze | retirement |
| --- | --- | --- | --- | --- |
| `publication_kind` | required concrete value | required concrete value | required concrete value | required concrete value |
| `expected_status` | required concrete value | required concrete value | required concrete value | required concrete value |
| `requested_status` | required concrete value | required concrete value | required concrete value | required concrete value |
| `publication_authorization_id` | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `publication_authorization_path` | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `publication_authorization_file_size` | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `publication_authorization_file_sha256` | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `publication_authorization_preimage_sha256` | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `signer_key_id` | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `signer_publication_key_size` | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `signer_publication_key_sha256` | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `signature_family_id` | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `status_operation_id` | required concrete value | required concrete value | required concrete value | required concrete value |
| `source_current_id` | required concrete value | required concrete value | required concrete value | required concrete value |
| `source_current_file_size` | required concrete value | required concrete value | required concrete value | required concrete value |
| `source_current_file_sha256` | required concrete value | required concrete value | required concrete value | required concrete value |
| `expected_status_file_size` | required concrete value | required concrete value | required concrete value | required concrete value |
| `expected_status_file_sha256` | required concrete value | required concrete value | required concrete value | required concrete value |
| `requested_status_prefix_size` | required concrete value | required concrete value | required concrete value | required concrete value |
| `requested_status_prefix_sha256` | required concrete value | required concrete value | required concrete value | required concrete value |
| `host_id` | required concrete value | required concrete value | required concrete value | required concrete value |
| `portable_plan_sha256` | required concrete value | required concrete value | required concrete value | required concrete value |

Constraint for each cell above, in the same field order and the same kind order:

`publication_kind`: `native_initialization` = token `native_initialization`; `imported_initialization` = token `imported_initialization`; `export_freeze` = token `export_freeze`; `retirement` = token `retirement`.
`expected_status`: `native_initialization` = token `empty`; `imported_initialization` = token `empty`; `export_freeze` = token `active`; `retirement` = token `export_frozen`.
`requested_status`: `native_initialization` = token `active`; `imported_initialization` = token `active`; `export_freeze` = token `export_frozen`; `retirement` = token `retired`.
`publication_authorization_id`: `native_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `imported_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `export_freeze` = the export authorization record; zero digest prohibited; `retirement` = the destination receipt record; zero digest prohibited.
`publication_authorization_path`: `native_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `imported_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `export_freeze` = the export authorization record; zero digest prohibited; `retirement` = the destination receipt record; zero digest prohibited.
`publication_authorization_file_size`: `native_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `imported_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `export_freeze` = the export authorization record; zero digest prohibited; `retirement` = the destination receipt record; zero digest prohibited.
`publication_authorization_file_sha256`: `native_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `imported_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `export_freeze` = the export authorization record; zero digest prohibited; `retirement` = the destination receipt record; zero digest prohibited.
`publication_authorization_preimage_sha256`: `native_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `imported_initialization` = id `none`, path `none`, size `0`, file digest 64 ASCII `0` characters, preimage digest 64 ASCII `0` characters; `export_freeze` = the export authorization record; zero digest prohibited; `retirement` = the destination receipt record; zero digest prohibited.
`signer_key_id`: `native_initialization` = signer_key_id `none`, key size `0`, key digest 64 ASCII `0` characters, signature_family_id `none`; `imported_initialization` = signer_key_id `none`, key size `0`, key digest 64 ASCII `0` characters, signature_family_id `none`; `export_freeze` = trust_admin project-id, key size `32`, concrete key digest, signature_family_id `export_authorization`; `retirement` = trust_admin project-id, key size `32`, concrete key digest, signature_family_id `import_receipt`.
`signer_publication_key_size`: `native_initialization` = signer_key_id `none`, key size `0`, key digest 64 ASCII `0` characters, signature_family_id `none`; `imported_initialization` = signer_key_id `none`, key size `0`, key digest 64 ASCII `0` characters, signature_family_id `none`; `export_freeze` = trust_admin project-id, key size `32`, concrete key digest, signature_family_id `export_authorization`; `retirement` = trust_admin project-id, key size `32`, concrete key digest, signature_family_id `import_receipt`.
`signer_publication_key_sha256`: `native_initialization` = signer_key_id `none`, key size `0`, key digest 64 ASCII `0` characters, signature_family_id `none`; `imported_initialization` = signer_key_id `none`, key size `0`, key digest 64 ASCII `0` characters, signature_family_id `none`; `export_freeze` = trust_admin project-id, key size `32`, concrete key digest, signature_family_id `export_authorization`; `retirement` = trust_admin project-id, key size `32`, concrete key digest, signature_family_id `import_receipt`.
`signature_family_id`: `native_initialization` = signer_key_id `none`, key size `0`, key digest 64 ASCII `0` characters, signature_family_id `none`; `imported_initialization` = signer_key_id `none`, key size `0`, key digest 64 ASCII `0` characters, signature_family_id `none`; `export_freeze` = trust_admin project-id, key size `32`, concrete key digest, signature_family_id `export_authorization`; `retirement` = trust_admin project-id, key size `32`, concrete key digest, signature_family_id `import_receipt`.
`status_operation_id`: `native_initialization` = one project-id; `none` prohibited; `imported_initialization` = one project-id; `none` prohibited; `export_freeze` = one project-id; `none` prohibited; `retirement` = one project-id; `none` prohibited.
`source_current_id`: `native_initialization` = the trust id named by current; `imported_initialization` = the trust id named by current; `export_freeze` = the trust id named by current; `retirement` = the trust id named by current.
`source_current_file_size`: `native_initialization` = byte size of current; `imported_initialization` = byte size of current; `export_freeze` = byte size of current; `retirement` = byte size of current.
`source_current_file_sha256`: `native_initialization` = full-file SHA-256 of current; zero digest prohibited; `imported_initialization` = full-file SHA-256 of current; zero digest prohibited; `export_freeze` = full-file SHA-256 of current; zero digest prohibited; `retirement` = full-file SHA-256 of current; zero digest prohibited.
`expected_status_file_size`: `native_initialization` = byte size of the expected service_status file; `imported_initialization` = byte size of the expected service_status file; `export_freeze` = byte size of the expected service_status file; `retirement` = byte size of the expected service_status file.
`expected_status_file_sha256`: `native_initialization` = full-file SHA-256 of that file; zero digest prohibited; `imported_initialization` = full-file SHA-256 of that file; zero digest prohibited; `export_freeze` = full-file SHA-256 of that file; zero digest prohibited; `retirement` = full-file SHA-256 of that file; zero digest prohibited.
`requested_status_prefix_size`: `native_initialization` = byte size of the new service_status prefix; `imported_initialization` = byte size of the new service_status prefix; `export_freeze` = byte size of the new service_status prefix; `retirement` = byte size of the new service_status prefix.
`requested_status_prefix_sha256`: `native_initialization` = section 6 self-checksum of that prefix; zero digest prohibited; `imported_initialization` = section 6 self-checksum of that prefix; zero digest prohibited; `export_freeze` = section 6 self-checksum of that prefix; zero digest prohibited; `retirement` = section 6 self-checksum of that prefix; zero digest prohibited.
`host_id`: `native_initialization` = the host project-id of the root being published; `imported_initialization` = the host project-id of the root being published; `export_freeze` = the host project-id of the root being published; `retirement` = the host project-id of the root being published.
`portable_plan_sha256`: `native_initialization` = full-file digest of the portable plan record; zero digest prohibited; `imported_initialization` = full-file digest of the portable plan record; zero digest prohibited; `export_freeze` = full-file digest of the portable plan record; zero digest prohibited; `retirement` = full-file digest of the portable plan record; zero digest prohibited.

The portable-publication-plan record is not a file. `record_kind` is `portable_publication_plan`. Its keys, in this order, are: `schema_version`, `record_kind`, `publication_kind`, `status_operation_id`, `export_operation_id`, `trust_id`, `transition_id`, `host_id`, `requested_root_path`, `new_binding_record_size`, `new_binding_record_sha256`, `source_binding_record_size`, `source_binding_record_sha256`, `successor_file_size`, `successor_file_sha256`, `expected_status`, `requested_status`, `record_sha256`. It contains no `device`, no `inode`, and no writer-bound digest. The friend seals these bytes before the writer opens the root. The writer does not modify the sealed bytes.

Portable-plan field rules:

| Field | withdrawal | ordinary | revocation | rollback | native_initialization | imported_initialization | export_freeze | retirement | governance_pin | activation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `publication_kind` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `status_operation_id` | exact sentinel | exact sentinel | exact sentinel | exact sentinel | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | exact sentinel |
| `export_operation_id` | exact sentinel | exact sentinel | exact sentinel | exact sentinel | exact sentinel | exact sentinel | required concrete value | exact sentinel | exact sentinel | exact sentinel |
| `trust_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `transition_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `host_id` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `requested_root_path` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value |
| `new_binding_record_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `source_binding_record_sha256` | exact sentinel | exact sentinel | exact sentinel | exact sentinel | exact sentinel | required concrete value | exact sentinel | exact sentinel | exact sentinel | exact sentinel |
| `successor_file_sha256` | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | exact sentinel | required concrete value | required concrete value |
| `expected_status` | exact sentinel | exact sentinel | exact sentinel | exact sentinel | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | exact sentinel |
| `requested_status` | exact sentinel | exact sentinel | exact sentinel | exact sentinel | required concrete value | required concrete value | required concrete value | required concrete value | exact sentinel | exact sentinel |

Constraint for each cell above, in the same field order and the same kind order:

`publication_kind`: `withdrawal` = token `withdrawal`; `ordinary` = token `ordinary`; `revocation` = token `revocation`; `rollback` = token `rollback`; `native_initialization` = token `native_initialization`; `imported_initialization` = token `imported_initialization`; `export_freeze` = token `export_freeze`; `retirement` = token `retirement`; `governance_pin` = token `governance_pin`; `activation` = token `activation`.
`status_operation_id`: `withdrawal` = token `none`; `ordinary` = token `none`; `revocation` = token `none`; `rollback` = token `none`; `native_initialization` = one project-id; `none` prohibited; `imported_initialization` = one project-id; `none` prohibited; `export_freeze` = one project-id; `none` prohibited; `retirement` = one project-id; `none` prohibited; `governance_pin` = token `none`; `activation` = token `none`.
`export_operation_id`: `withdrawal` = token `none`; `ordinary` = token `none`; `revocation` = token `none`; `rollback` = token `none`; `native_initialization` = token `none`; `imported_initialization` = token `none`; `export_freeze` = one project-id; `none` prohibited; `retirement` = token `none`; `governance_pin` = token `none`; `activation` = token `none`.
`trust_id`: `withdrawal` = the successor trust id, or the source current trust id for export and retirement; `ordinary` = the successor trust id, or the source current trust id for export and retirement; `revocation` = the successor trust id, or the source current trust id for export and retirement; `rollback` = the successor trust id, or the source current trust id for export and retirement; `native_initialization` = the successor trust id, or the source current trust id for export and retirement; `imported_initialization` = the successor trust id, or the source current trust id for export and retirement; `export_freeze` = the successor trust id, or the source current trust id for export and retirement; `retirement` = the successor trust id, or the source current trust id for export and retirement; `governance_pin` = the successor trust id, or the source current trust id for export and retirement; `activation` = the successor trust id, or the source current trust id for export and retirement.
`transition_id`: `withdrawal` = the new transition id; `ordinary` = the new transition id; `revocation` = the new transition id; `rollback` = the new transition id; `native_initialization` = the new transition id; `imported_initialization` = the source transition id already stored in the copied successor; `export_freeze` = token `none`; `retirement` = token `none`; `governance_pin` = the new transition id; `activation` = the new transition id.
`host_id`: `withdrawal` = destination host for imported_initialization; otherwise the host of the root being published; `ordinary` = destination host for imported_initialization; otherwise the host of the root being published; `revocation` = destination host for imported_initialization; otherwise the host of the root being published; `rollback` = destination host for imported_initialization; otherwise the host of the root being published; `native_initialization` = destination host for imported_initialization; otherwise the host of the root being published; `imported_initialization` = destination host for imported_initialization; otherwise the host of the root being published; `export_freeze` = destination host for imported_initialization; otherwise the host of the root being published; `retirement` = destination host for imported_initialization; otherwise the host of the root being published; `governance_pin` = destination host for imported_initialization; otherwise the host of the root being published; `activation` = destination host for imported_initialization; otherwise the host of the root being published.
`requested_root_path`: `withdrawal` = absolute lexical path, length 1 to 1024, beginning with `/`, no NUL, no LF, no `//`, and no `.` or `..` segment; `ordinary` = absolute lexical path, length 1 to 1024, beginning with `/`, no NUL, no LF, no `//`, and no `.` or `..` segment; `revocation` = absolute lexical path, length 1 to 1024, beginning with `/`, no NUL, no LF, no `//`, and no `.` or `..` segment; `rollback` = absolute lexical path, length 1 to 1024, beginning with `/`, no NUL, no LF, no `//`, and no `.` or `..` segment; `native_initialization` = absolute lexical path, length 1 to 1024, beginning with `/`, no NUL, no LF, no `//`, and no `.` or `..` segment; `imported_initialization` = absolute lexical path, length 1 to 1024, beginning with `/`, no NUL, no LF, no `//`, and no `.` or `..` segment; `export_freeze` = absolute lexical path, length 1 to 1024, beginning with `/`, no NUL, no LF, no `//`, and no `.` or `..` segment; `retirement` = absolute lexical path, length 1 to 1024, beginning with `/`, no NUL, no LF, no `//`, and no `.` or `..` segment; `governance_pin` = absolute lexical path, length 1 to 1024, beginning with `/`, no NUL, no LF, no `//`, and no `.` or `..` segment; `activation` = absolute lexical path, length 1 to 1024, beginning with `/`, no NUL, no LF, no `//`, and no `.` or `..` segment.
`new_binding_record_sha256`: `withdrawal` = digest of the new transition_operation_binding; zero digest prohibited; `ordinary` = digest of the new transition_operation_binding; zero digest prohibited; `revocation` = digest of the new transition_operation_binding; zero digest prohibited; `rollback` = digest of the new transition_operation_binding; zero digest prohibited; `native_initialization` = digest of the new transition_operation_binding; zero digest prohibited; `imported_initialization` = size `0` and digest 64 ASCII `0` characters; `export_freeze` = size `0` and digest 64 ASCII `0` characters; `retirement` = size `0` and digest 64 ASCII `0` characters; `governance_pin` = digest of the new transition_operation_binding; zero digest prohibited; `activation` = digest of the new transition_operation_binding; zero digest prohibited.
`source_binding_record_sha256`: `withdrawal` = size `0` and digest 64 ASCII `0` characters; `ordinary` = size `0` and digest 64 ASCII `0` characters; `revocation` = size `0` and digest 64 ASCII `0` characters; `rollback` = size `0` and digest 64 ASCII `0` characters; `native_initialization` = size `0` and digest 64 ASCII `0` characters; `imported_initialization` = the source successor binding_record_sha256; zero digest prohibited; `export_freeze` = size `0` and digest 64 ASCII `0` characters; `retirement` = size `0` and digest 64 ASCII `0` characters; `governance_pin` = size `0` and digest 64 ASCII `0` characters; `activation` = size `0` and digest 64 ASCII `0` characters.
`successor_file_sha256`: `withdrawal` = size and full-file digest of the new successor bytes; zero digest prohibited; `ordinary` = size and full-file digest of the new successor bytes; zero digest prohibited; `revocation` = size and full-file digest of the new successor bytes; zero digest prohibited; `rollback` = size and full-file digest of the new successor bytes; zero digest prohibited; `native_initialization` = size and full-file digest of the new successor bytes; zero digest prohibited; `imported_initialization` = size and full-file digest of the copied source successor; zero digest prohibited; `export_freeze` = size `0` and digest 64 ASCII `0` characters; `retirement` = size `0` and digest 64 ASCII `0` characters; `governance_pin` = size and full-file digest of the new successor bytes; zero digest prohibited; `activation` = size and full-file digest of the new successor bytes; zero digest prohibited.
`expected_status`: `withdrawal` = token `none`; `ordinary` = token `none`; `revocation` = token `none`; `rollback` = token `none`; `native_initialization` = token `empty`; `imported_initialization` = token `empty`; `export_freeze` = token `active`; `retirement` = token `export_frozen`; `governance_pin` = token `none`; `activation` = token `none`.
`requested_status`: `withdrawal` = token `none`; `ordinary` = token `none`; `revocation` = token `none`; `rollback` = token `none`; `native_initialization` = token `active`; `imported_initialization` = token `active`; `export_freeze` = token `export_frozen`; `retirement` = token `retired`; `governance_pin` = token `none`; `activation` = token `none`.

The writer-bound-identity record is not a file. `record_kind` is `writer_bound_identity`. Its keys, in this order, are: `schema_version`, `record_kind`, `portable_plan_sha256`, `device`, `inode`, `canonical_path`, `host_id`, `record_sha256`. `device` and `inode` are the `uint64` values of `st_dev` and `st_ino` from `fstat` of the opened trust-root directory. If either value is not representable as `uint64`, the result is `kRootRejected` and no publication file is written. `canonical_path` is the caller path after the lexical check, not a symlink-resolved path. `writer_bound_sha256` is SHA-256 of every byte of this finished record. The sealed plan does not store this record.

`publication_root` keys, in this order: `schema_version`, `record_kind`, `device`, `inode`, `canonical_path`, `host_id`, `record_sha256`. Path: `trust_root/publication-root.record`. `record_kind` is `publication_root`. Cardinality is one for the life of the directory. Signature is `none`. It is local operational state and is not authority for a trust transition. Producer is 9-C33. The file is mode `0444`, created with `O_CREAT|O_EXCL|O_NOFOLLOW`, accepted only as a regular file with link count 1. It is not a symlink and not a hard link. It is immutable: the writer never rewrites it and never deletes it.

`status_authorization` keys, in this order: `schema_version`, `record_kind`, `status_operation_id`, `publication_kind`, `binding_record_kind`, `binding_record_size`, `binding_record_sha256`, `expected_status`, `requested_status`, `portable_plan_size`, `portable_plan_sha256`, `writer_bound_size`, `writer_bound_sha256`, `record_sha256`. Path: `trust_root/status-authorizations/<status-operation-id>.record`. `record_kind` is `status_authorization`. `status-operation-id` is a project-id and is not `none` and not `unknown`. Cardinality is one file per status-operation-id. Distinct operations use distinct ids, so native or imported activation, export freeze, and retirement each keep their own file. Signature is `none`. Producer is 9-C36 for native activation and 9-C34 for imported activation, export freeze, and retirement. A completed earlier file does not occupy the later operation's path.

`import_destination_binding` keys, in this order: `schema_version`, `record_kind`, `migration_id`, `destination_transaction_id`, `destination_nonce_id`, `package_id`, `migration_package_file_size`, `migration_package_file_sha256`, `payload_id`, `payload_file_size`, `payload_file_sha256`, `high_water`, `source_successor_trust_id`, `source_successor_file_size`, `source_successor_file_sha256`, `source_transition_id`, `source_transition_file_size`, `source_transition_file_sha256`, `source_binding_record_size`, `source_binding_record_sha256`, `destination_host_id`, `requested_trust_generation`, `requested_current_trust_id`, `status_operation_id`, `portable_plan_size`, `portable_plan_sha256`, `record_sha256`. Path: `trust_root/import-destination.record`. `record_kind` is `import_destination_binding`. Cardinality is one for the life of the destination. Signature is `none`. Producer is 9-C34. Mode `0444`, `O_CREAT|O_EXCL|O_NOFOLLOW`, regular file, link count 1. Every key is a required concrete value. Zero digests are prohibited. The record contains no `device` and no `inode`. It is not a byte inside the source successor and it is not a byte inside the source transition.

`export_authorization` keys, in this order: `schema_version`, `record_kind`, `export_operation_id`, `status_operation_id`, `package_id`, `migration_package_file_size`, `migration_package_file_sha256`, `source_current_id`, `source_current_file_size`, `source_current_file_sha256`, `expected_status`, `requested_status`, `signer_key_id`, `signer_publication_key_size`, `signer_publication_key_sha256`, `signature_family_id`, `host_id`, `registry_id`, `authority_registry_file_size`, `authority_registry_file_sha256`, `effective_date`, `portable_plan_size`, `portable_plan_sha256`, `record_sha256`. Path: `trust_root/export-authorizations/<export-operation-id>.record`. `record_kind` is `export_authorization`. `export-operation-id` is a project-id and is not `none` and not `unknown`. Cardinality is one file per export-operation-id. `expected_status` is `active`. `requested_status` is `export_frozen`. `signature_family_id` is `export_authorization`. `signer_publication_key_size` is `32`. The signer key digest is concrete and the zero digest is prohibited. `effective_date` is the source terminal trust state's `effective_date`. The registry entry role is `trust_admin`, `revoked=no`, and `valid_from <= effective_date <= valid_until`. The Ed25519 preimage is the exact finished record bytes, including `record_sha256`. The signature blob, index, and verification record are outside that preimage. Paths, in the same directory, are `<export-operation-id>.signature.bin`, `<export-operation-id>.signature.index`, and `<export-operation-id>.verification.record`. Each is mode `0444`, created with `O_CREAT|O_EXCL|O_NOFOLLOW`, regular, link count 1, `fsync`, and parent `fsync` before the next file. The order is record, signature, index, verification. Producer is 9-C34. A second call that finds those four exact files for the same export-operation-id returns `kOk` with `bytes_written` false. A file at that id with different bytes returns `kConflict`, writes nothing, and deletes nothing. The record contains no `device` and no `inode`.


`binding_record_kind` is the required concrete value `transition_operation_binding` on every successor in the seven transition kinds. `binding_record_size` is the required concrete byte length of the finished binding record and is not zero. `binding_record_sha256` is the required concrete full-file digest of that record and the zero digest is prohibited. Status publication kinds are prohibited in `trust_state.binding_record_kind`.

### Root binding

Stage A seals the portable plan. The friend checks the phase predicate, serializes the portable plan, and stores `portable_plan_sha256`. That object is immutable. It has no device and no inode.

Stage B begins when the writer opens the trust-root directory with `O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`. It calls `fstat` and `fstatfs`, classifies the filesystem, and only then takes the process mutex and the nonblocking exclusive `flock`. No publication file is written before classification and the lock. If `publication-root.record` is absent, the writer creates it from this `fstat` and the lexical path, `fsync`s the file, and `fsync`s the directory. If it exists, `device`, `inode`, `canonical_path`, and `host_id` must equal this open. A mismatch returns `kConflict`, writes nothing else, and deletes nothing. The writer then serializes `writer_bound_identity` and computes `writer_bound_sha256`. It does not write those bytes into the sealed plan, the successor, the transition, the import-destination record, or the export-authorization record. A retry recomputes both digests. The portable digest must equal the sealed plan. The writer-bound digest must equal the digest implied by the current `publication-root.record` and that portable digest. Imported source bytes are not an input to `device` or `inode`.

### P0-P9

Genesis files are only `trust_root/genesis/transition.payload`, `genesis/transition.record`, `genesis/signature.bin`, `genesis/signature.index`, and `genesis/verification.record`. Later transition files are only `trust_root/transitions/<transition-id>/` with those five basenames. The successor file is only `admission/trust/trust-<trust_id>.record`. Each new file is created with `O_CREAT|O_EXCL|O_NOFOLLOW`, mode `0444`, link count 1, a full write, `fsync`, `fstat`, rewind, and reread. A short write is `kConflict` when a foreign byte is already durable and `kPrefixMismatch` when the byte string differs from this operation's exact prefix. Nothing is deleted. The writer does not scan a directory to choose a file.

| State | Required observation |
|---|---|
| P0 | Successor absent. Transition directory absent. `current.partial` absent. `current` still names the predecessor, or `current` is absent when the predecessor current is absent. `publication-root.record` may already be present. |
| P1 | Exact successor bytes. Recomputed transition-operation-binding digest matches those bytes. Transition directory absent. |
| P2 | P1 plus the family directory mode `0700` and `ENOENT` for `transition.payload`, `signature.bin`, `signature.index`, `verification.record`, and `transition.record`. An unexpected name in that directory is `kConflict`. |
| P3 | P2 plus exact `transition.payload`. `signature.bin` absent. `transition.record` absent. |
| P4 | P3 plus exact `signature.bin`. Index absent. |
| P5 | P4 plus exact `signature.index`. Verification absent. |
| P6 | P5 plus exact `verification.record`. `transition.record` absent. |
| P7 | Exact five-file family. Directory mode `0555`. `current.partial` absent. `current` still the predecessor, or absent for genesis and for native initialization before N8. |
| P8 | P7 family plus exact `current.partial` containing the successor bytes. `current` still the predecessor, or absent when the plan says the predecessor current is absent. |
| P9 | `current` byte-identical to the successor. Parent `fsync` has returned on this call. Final reread matches. `current.partial` absent. |

Advances: P0 to P1 writes the successor only when `SuccessorDurability` is `kMustCreate`. P1 to P2 creates the directory at `0700`. P2 to P3 writes `transition.payload`. P3 to P4 writes `signature.bin` only when that file is absent. P4 to P5 writes the index. P5 to P6 writes the verification record. P6 to P7 writes `transition.record`, changes the directory to `0555`, and `fsync`s it. P7 to P8 writes `current.partial`. P8 to P9 renames that partial onto `current`, `fsync`s the trust root, and rereads `current`. The signer is called only when the signature file and an exact signature partial are both absent. The message is the section 9 domain line plus the already durable `transition.payload`. One call must return 64 bytes that verify under the plan's public key. A mismatch at any state returns `kPrefixMismatch` when the path belongs to this successor id and the binding differs, and `kConflict` when the file is a different object. The writer compares the recomputed portable digest and the recomputed writer-bound digest with the sealed plan and the publication-root record at every state.

At P0 the fully checked plan may create the successor. At P1 and P2 the writer reads the successor and recomputes the transition-operation-binding digest from the successor fields. A request whose kind, authorization, signer key, public-key digest, signature family, host id, or lineage fields differ from those bytes returns `kPrefixMismatch` and writes nothing further.

After P9, a present `current.partial` returns `kConflict`, writes nothing, and deletes nothing, whether or not its bytes equal the successor. `kOk` at P9 requires `current.partial` to be absent. Presence of `publication-root.record` does not move the state out of P0.

Trust-transition mandatory tests. Each row is one case and has one result code.

| Setup | Operation | Result | Durable state | Retry | Deletion or overwrite |
|---|---|---|---|---|---|
| Exact generation-1 family and domain-separated payload | authenticate | `kOk` | unchanged | yes | no |
| Same payload bytes checked by an Ed25519 implementation other than src/ed25519_verify.cpp | compare verifier input | `kOk` | unchanged | yes | no |
| One payload byte flipped before sealing | authenticate | `kAuthorization` | unchanged | no | no |
| Domain line differs by one byte | authenticate | `kAuthorization` | unchanged | no | no |
| schema_version other than 1 | authenticate | `kSchema` | unchanged | no | no |
| Registry id differs between payload and transition | authenticate | `kState` | unchanged | no | no |
| Transition id differs between payload and transition | authenticate | `kState` | unchanged | no | no |
| Signed publication_kind is valid and differs from the requested operation | authorize operation | `kAuthorization` | unchanged | no | no |
| Transition and payload semantic mismatch: publication_kind fields differ | authenticate | `kState` | unchanged | no | no |
| Generation is not the predecessor generation plus 1 | authenticate | `kState` | unchanged | no | no |
| Binding bytes change and binding_record_sha256 no longer matches | authenticate | `kDigest` | unchanged | no | no |
| Successor revocation_generation is lower than the predecessor | authenticate | `kAuthorization` | unchanged | no | no |
| Generation-2 successor decreases revocation_generation | authenticate | `kAuthorization` | unchanged | no | no |
| signed_payload_sha256 differs from the payload file | authenticate | `kDigest` | unchanged | no | no |
| Payload-size mismatch: signed_payload_size differs from the payload file length | authenticate | `kSize` | unchanged | no | no |
| signature_id differs between transition and verification record | authenticate | `kState` | unchanged | no | no |
| signature_blob_id differs from the index blob_id | authenticate | `kState` | unchanged | no | no |
| signer_key_id does not name the opened key | authenticate | `kState` | unchanged | no | no |
| signature_index_id differs between transition and verification record | authenticate | `kState` | unchanged | no | no |
| signature_sha256 differs from the 64 signature bytes | authenticate | `kDigest` | unchanged | no | no |
| Signature file is 63 bytes | authenticate | `kSize` | unchanged | no | no |
| Signature file is 65 bytes | authenticate | `kSize` | unchanged | no | no |
| Public key is not the key that signed the payload | authenticate | `kAuthorization` | unchanged | no | no |
| A required family file is absent | authenticate | `kUnavailable` | unchanged | no | no |
| Two files claim the same transition id | authenticate | `kDuplicate` | unchanged | no | no |
| Payload contains an extra key | authenticate | `kSchema` | unchanged | no | no |
| Payload omits a required key | authenticate | `kSchema` | unchanged | no | no |
| Payload keys are reordered | authenticate | `kSchema` | unchanged | no | no |
| Payload uses CRLF | authenticate | `kSchema` | unchanged | no | no |
| Payload has a trailing byte after the final LF | authenticate | `kSchema` | unchanged | no | no |
| Payload contains an embedded NUL | authenticate | `kSchema` | unchanged | no | no |
| A family path is a symlink | authenticate | `kSymlink` | unchanged | no | no |
| Payload, key, or signature has link count other than 1 | authenticate | `kLink` | unchanged | no | no |
| A family file mode is not 0444 | authenticate | `kForbidden` | unchanged | no | no |
| A directory is substituted for a family file | authenticate | `kType` | unchanged | no | no |
| Same bytes on a different inode before mutation | cleanup | `kRace` | unchanged | yes | no |
| A target is replaced after revalidation fstat and before the mutation call | cleanup | `kRace` | unchanged | yes | no |
| A valid signature names a different transition id | cleanup | `kAuthorization` | unchanged | no | no |
| Signature verifies an R2 prefix and not the domain-separated payload | authenticate | `kAuthorization` | unchanged | no | no |
| signature_file_sha256 is 64 ASCII 0 characters | authenticate | `kDigest` | unchanged | no | no |
| Crash before P0: death before the successor file is created | resume the same publication | `kOk` | P9 | yes | no |
| Crash at P0: death after no successor bytes are durable | resume the same publication | `kOk` | P9 | yes | no |
| Crash at P1: death after the exact successor fsync and before the family directory exists | resume the same publication | `kOk` | P9 | yes | no |
| Crash at P2: death after the empty five-name directory fsync | resume the same publication | `kOk` | P9 | yes | no |
| Crash at P3: death after the exact payload fsync and before signature.bin exists | resume the same publication | `kOk` | P9 | yes | no |
| Crash at P4: death after the exact signature.bin fsync and before the index exists | resume the same publication | `kOk` | P9 | yes | no |
| Crash at P5: death after the exact index fsync and before the verification record exists | resume the same publication | `kOk` | P9 | yes | no |
| Crash at P6: death after the exact verification record fsync and before transition.record exists | resume the same publication | `kOk` | P9 | yes | no |
| Crash at P7: death after the exact five-file family and directory mode 0555 | resume the same publication | `kOk` | P9 | yes | no |
| Crash at P8: death after exact current.partial fsync and before current is replaced | resume the same publication | `kOk` | P9 | yes | no |
| Crash at P9 or post-publication retry: exact current already matches the successor | retry the same publication | `kOk` | P9 | yes | no |
| Partial family with payload only, matching P3 | resume publication | `kOk` | P9 | yes | no |
| Partial family with payload plus signature, matching P4 | resume publication | `kOk` | P9 | yes | no |
| Partial family through index, matching P5 | resume publication | `kOk` | P9 | yes | no |
| Partial family through verification record, matching P6 | resume publication | `kOk` | P9 | yes | no |
| Missing one member from an otherwise complete family: signature.bin absent while the index is present | resume publication | `kConflict` | unchanged | no | no |
| Unexpected sixth member in the family directory | resume publication | `kConflict` | unchanged | no | no |

Kind `imported_initialization` does not use the P-machine. `SuccessorDurability` is `kAlreadyExact`. The writer opens the successor read-only and does not create it, truncate it, or sign it.

### Imported-current states

Imported initialization does not create or sign the imported successor or the imported transition. `publish_verified_transition` on kind `imported_initialization` uses only these states.

| State | Required observation |
|---|---|
| I0 | The source successor is already durable at `admission/trust/trust-<id>.record` and its bytes equal the plan's successor bytes. The binding recomputed from those bytes equals the successor's `binding_record_sha256` and the plan's `source_binding_record_sha256`. The source transition's `successor_trust_file_sha256` equals that file. `import-destination.record` matches the plan. `current` absent. `current.partial` absent. Lineage family verifies. Status is the exact snapshotted `empty` record. Destination transaction, nonce, package, host, and root fields are not required to occur inside the successor. |
| I1 | I0 plus exact `current.partial` containing those unchanged successor bytes. |
| I2 | `current` byte-identical to the successor. Partial absent. Trust-root `fsync` returned on this call. Reread matches. Status still `empty`. |
| I3 | I2 plus canonical `active` status and the plan-specific `status-authorizations/<status-operation-id>.record`. |

I0 to I1 writes `current.partial` only. I1 to I2 renames it onto `current`, `fsync`s the trust root, and rereads. I2 to I3 calls `publish_verified_status` for canonical `active` under the plan's status-operation-id. A crash after the rename is I2 once this call's `fsync` and reread succeed. Case A, including an authorized descendant N+1 or N+2 of the same import identity, returns `kOk` with `bytes_written` false and does not enter I1. Case B0 is the only case that may advance I0 to I2. Case B1 does not rewrite `current` and may advance only from I2 to I3 while status is still the snapshotted `empty` record. Case C returns `kConflict` and writes nothing. A source binding that does not recompute, or a successor that differs from the plan, returns `kConflict` and writes nothing.

A destination with `import_completed` true may publish generation N+1, and later N+2, under the P0-P9 rules. N is the `trust_generation` of the imported trust state. The new successor does not need to equal the transaction's `trust_file_sha256`. The predecessor must be the authenticated `current`. The transaction is not rewritten, deleted, or consulted as a digest veto. Section 11 then sets `service_loadable` for N+1 or N+2 when status is canonical `active` and the section 13 walk contains the imported trust state. Section 14, not import resume, is the only writer of that new `current`. Import resume must not write `current` after `import_completed`. Withdrawal of a subject, when it changes the terminal trust state, is also one of these forward transitions. A later generation cannot decrease `revocation_generation` or clear a tombstone.

### Native order

Kind `native_initialization` runs the P-machine for `genesis-v1` through P7 while `current` stays absent, then the section 11 states N0 through N8 for the lineage-origin family, then P7 to P9 for `current`, then `publish_verified_status` for canonical `active` at the plan's status-operation-id. The genesis signer message is the section 9 domain-separated `transition.payload`. The lineage-origin signer message remains the exact `lineage-origin.record` bytes. Each signer call occurs only while that signature file is absent. An imported lineage at this destination returns `kConflict` before a write. The status-operation-id used here is not reused for export freeze or retirement.

### Status states

`publish_verified_status` opens the root stored in the plan, classifies it, and takes the process mutex and directory `flock` for that call. It does not reuse a root retained by another call. The authorization path is exactly `trust_root/status-authorizations/<status-operation-id>.record` from the sealed plan. The writer opens that directory with `openat` and `O_DIRECTORY|O_NOFOLLOW` on the trust-root descriptor. If the directory is absent, `mkdirat` uses mode `0755`, then `openat`, `fchmod` `0755`, and an `fsync` of the trust root. A symlink is `kConflict`. Directory link count is not an authority input. The writer does not call `readdir` and does not select a sibling by name order or modification time. An unrelated sibling file is not opened.

| State | Required observation |
|---|---|
| S0 | Live status equals the plan's expected bytes. The plan-specific authorization path is `ENOENT` from `openat` with `O_NOFOLLOW`. `service_status.partial` absent. |
| S1 | That authorization file is exact for this portable plan and for the writer-bound digest recomputed on this call. Live status still the expected bytes. Partial absent. Link count is 1. |
| S2 | S1 plus exact `service_status.partial` containing the requested bytes. |
| S3 | Live status equals the requested bytes. Partial absent. Authorization file still exact. Parent `fsync` returned on this call. Reread matches. |
| S3P | Live status equals the requested bytes and the authorization file is exact, while `service_status.partial` exists, for any byte string in that partial. |

S0 to S1 creates the authorization file with `O_CREAT|O_EXCL|O_NOFOLLOW`, mode `0444`, link count 1, full write, `fsync`, and parent `fsync`. The written bytes are the section 4 status-authorization record. `writer_bound_sha256` is computed after the open and is not taken from the sealed plan. S1 to S2 writes the partial. S2 to S3 renames it onto `service_status`, `fsync`s the trust root, and rereads. `EEXIST` on the authorization path opens that file read-only. Matching portable fields, matching recomputed writer-bound digest, and expected live status are S1. Matching fields and requested live status with the partial absent are S3. Different bytes are `kConflict`. Export freeze is legal only from exact canonical `active` to exact canonical `export_frozen`. Retirement is legal only from exact canonical `export_frozen` to exact canonical `retired`. A bare status token is not a plan and is not accepted by `publish_verified_status`. A repeat at S3 whose portable plan and recomputed writer-bound digest match returns `kOk` with `bytes_written` false.

S3P returns `kConflict`. It is not `kOk`. The partial stays in place. `service_status` stays in place. Every status-authorization file stays in place. The call writes nothing and deletes nothing. It does not start native initialization, imported initialization, export freeze, or retirement. `tests/trust_transition_publication_test.cpp` contains a case that creates `service_status.partial` after S3 and requires this result and these unchanged bytes.

### Withdrawal

`publish_withdrawal` returns `kAuthorizationRejected` before the plan constructor unless the subject kind and role pair is one of: `provenance` with `provenance_reviewer`, `license` with `license_reviewer`, `golden_fixture` with `golden_reviewer`, `registry_entry` with `registry_reviewer`, `policy_pin` with `policy_reviewer`, `telemetry_governance` with `telemetry_reviewer`, `activation_authorization` with `policy_reviewer`. The registry entry role equals that role. The key is unrevoked at `effective_date`. The note path is `admission/notes/withdraw-note-<subject-id>.record` and the section 9 withdrawal family verifies. Successor revocation generation equals the predecessor revocation generation plus 1. The successor tombstone list contains every predecessor tombstone. The lineage-origin family is unchanged. Only then does the function construct `VerifiedTransitionPlan` and call `publish_verified_transition`.

### Rollback

`publish_rollback` returns `kAuthorizationRejected` before the plan constructor unless successor generation equals predecessor generation plus 1, successor revocation generation is not lower, the successor tombstone list contains every predecessor tombstone, the successor registry identity equals the predecessor registry identity, a changed pin lists only subjects absent from the successor tombstones, telemetry is the predecessor triple or the disabled zero triple, and `admission_mode=no_model` uses pin id `none`, size 0, and the zero digest. Only then does it construct `VerifiedTransitionPlan` and call `publish_verified_transition`. `tests/rollback_test.cpp` is not a friend.

### Governance pin

`publish_governance_pin` returns `kAuthorizationRejected` before the plan constructor unless the governance content identity matches, the governance-review family verifies with role `telemetry_reviewer`, the accepted decision names the content digest and the review-note digest, the decision id is a member of `head_set_members`, the decision id is absent from the predecessor tombstones, and the successor telemetry triple equals the request triple. Only then does it construct `VerifiedTransitionPlan` with kind `governance_pin` and call `publish_verified_transition`.

### Activation

`publish_activation` returns `kAuthorizationRejected` before the plan constructor unless `requested_runtime_mode` is `activation`, `requested_activation_mode` is `active`, the activation-review family verifies with role `policy_reviewer`, the accepted decision id is in `head_set_members` and absent from the predecessor tombstones, and the policy decision, candidate, dataset, invariance, and model id in the request equal the successor fields. Only then does it construct `VerifiedTransitionPlan` with kind `activation` and call `publish_verified_transition`. This is the section 34 compare-and-swap.

### Export freeze

`publish_export_freeze` returns `kAuthorizationRejected` before the status-plan constructor unless the source current matches, the live status bytes are the canonical `active` record, the export-authorization family verifies under `trust_admin` at `effective_date`, the public-key digest matches the registry entry, and the package identity matches the authorization record. Only then does the function construct `VerifiedStatusPlan` and call `publish_verified_status` for the plan's own status-operation-id.

### Retirement

`publish_retirement` returns `kAuthorizationRejected` before the status-plan constructor unless the source status is the exact canonical `export_frozen` record, the receipt file on the destination trust root is `migrations/<receipt-id>.receipt.record`, the destination admission root contains `admission/signature-records/receipt-signature-<receipt-id>.record` and `admission/signature-indexes/receipt-sigindex-<receipt-id>.index`, the destination evidence root contains `benchmarks/generative_phase9c/signatures/receipt-sigblob-<receipt-id>.bin`, those four objects verify under `trust_admin`, and the receipt binds the stated migration package and trust file. Retirement from `active`, `empty`, or `retired` is rejected. The status write uses the retirement status-operation-id and does not open the activation or export-freeze authorization files. A later call that finds S3 for the same operation identity returns `kOk` with `bytes_written` false.

### Ordinary transitions

`publish_ordinary_transition` accepts only kind `ordinary` or kind `revocation`. It returns `kAuthorizationRejected` before the plan constructor when the signer role is not `trust_admin`, the signature does not verify, the generation is not the predecessor generation plus 1, the revocation generation decreases, or a predecessor tombstone is absent from the successor. It then calls `publish_verified_transition`. Phase 9-C38 does not define another writer.

### Result policy

| Token | Policy |
|---|---|
| `kOk` | The operation is at its authenticated completion. The first call that reaches completion returns `kOk` and may have written. An exact later call with the same portable plan, the same recomputed writer-bound digest, matching files, and no `current.partial` returns `kOk` with `bytes_written` false and writes nothing. S3P is not `kOk`. A different identity on the same bytes is not `kOk`. |
| `kBusyMutex` | Mutex is busy. No write. Same request may retry. No poison. No restart. |
| `kBusyKernelLock` | `flock` returned `EWOULDBLOCK` or `EAGAIN`. No write. Retry is allowed. No poison. |
| `kReentered` | Owner is this thread. Detected before `try_lock`. No write. This thread does not retry until the owner clears. No poison. |
| `kRootRejected` | Path is rejected or is not the plan path. No write. A corrected path may be submitted. No poison. |
| `kFilesystemUnsupported` | Classification is unsupported. No write. The same root is not retried. A different supported root may be submitted. No poison. |
| `kFilesystemIndeterminate` | `fstatfs` failed. No write. Retry is allowed. No poison. |
| `kInterrupted` | `flock` acquisition returned `EINTR`. No write. Retry is allowed. No poison. |
| `kAllocationFailed` | `std::bad_alloc` was caught. No write. Retry is allowed. No poison. |
| `kKernelLockInvariant` | `flock` returned `EBADF`, `EINVAL`, or any errno other than `EAGAIN`, `EWOULDBLOCK`, `EINTR`, `ENOLCK`, and `EIO`. Poison before mutex release. No retry. Restart and reread. |
| `kKernelLockResource` | `flock` returned `ENOLCK`. No write. Retry is allowed. No poison. |
| `kKernelLockIo` | `flock` returned `EIO` during acquisition. No write. Retry is allowed. No poison. |
| `kConflict` | A foreign file, an illegal partial, or S3P occupies a path. No further write. No deletion. No retry of a write. Reread. No poison. |
| `kAuthorizationRejected` | The phase predicate failed before the plan constructor, or the constructor was not reached. No publication byte from this call. A corrected request may be submitted. No poison. |
| `kPrefixMismatch` | This operation's path has a durable prefix for a different identity. No further write. No deletion. Reread. No poison. |
| `kReleaseFailedBeforeDurable` | `LOCK_UN` failed with an errno other than `EBADF` before completion reread. Poison. No retry. Restart and reread. |
| `kOutcomeUncertain` | Completion reread succeeded and a later non-`EBADF` release failed. Poison. No retry. Restart and reread. Do not start a second successor. |
| `kReleaseInvariant` | `LOCK_UN` returned `EBADF`. Poison before mutex release. No retry. Restart and reread. |
| `kDescriptorInvariant` | `close` returned `EBADF` or `EINTR`. One attempt only. Poison before mutex release. No retry of the integer. Restart and reread. |
| `kPoisoned` | The latch was already set. No write. No retry. Restart and reread. |

`EBADF` is not a successful close and is not a successful unlock. `close` is attempted once. `LOCK_UN` is attempted once. An abandoned guard sets poison while holding the mutex, attempts `LOCK_UN` once if the lock was acquired, attempts `close` once, retains both errnos, clears the owner, and unlocks only after the poison store is visible. It does not throw. `finish` runs that release once. A second `finish` returns the stored outcome.

Same-thread reentry is an acquire load of `std::atomic<std::thread::id>` before `try_lock`. The empty owner is a default-constructed `std::thread::id`. Poison is `std::atomic<bool>` stored with release while the mutex is held. Readers take no mutex and no `flock`. Publication functions do not call `fork`, `posix_spawn`, or `exec`.

### Crash and recovery

| Observation | Result | Write | Retry | Restart |
|---|---|---|---|---|
| Process death after any `fsync` and before the next rename | resume at the state whose files match | only the next defined advance | same operation | yes |
| `publication-root.record` mismatches the open | `kConflict` | none | no | reread |
| Authorization path occupied by different bytes | `kConflict` | none | no | reread |
| S3P | `kConflict` | none | no | reread |
| Exact S3 or exact P9 or exact I3 | `kOk` and `bytes_written` false | none | observation only | no |

### Test seam and platform tests

`PublicationHooks` uses `FilesystemClass`, `FlockHookResult`, and `CloseHookResult`. It does not use `struct statfs`. `set_publication_hooks_for_test` exists only when `INFEREDGE_TRUST_PUBLICATION_TEST_SEAM` is defined. It returns true only on the first call in the process, and only before any publication function has been entered. A later call returns false and leaves the hooks unchanged. The setter is called from the test's initial thread. It is not called concurrently. Hooks cannot construct a plan. Poison and release-failure tests run in a child process created by the test harness with `posix_spawn`. The child exits. The parent does not reuse the child's descriptors. A forced `close` error does not skip the one real `close` attempt. The child's descriptors end when the child exits.

The Linux positive test `lstat`s `/dev/shm`, requires a real directory that is not a symlink, opens it, and requires tmpfs magic `0x01021994`. It copies the bytes of `/dev/shm/inferedge-trust-pub-XXXXXX` including the terminating NUL into `std::array<char, sizeof("/dev/shm/inferedge-trust-pub-XXXXXX")>` and passes `template_bytes.data()` to `mkdtemp`. A null return fails the test. The test writes only inside that returned child. Cleanup removes only that child. Cleanup failure fails the test and reports that path. The test does not recursively delete `/dev/shm`. If `/dev/shm` is absent, not a directory, a symlink, or not tmpfs, the test fails.

The Darwin positive test uses the same mutable-array rule with the template `/private/tmp/inferedge-trust-pub-XXXXXX` and requires classification `apfs`. A null `mkdtemp` return fails the test. Cleanup failure fails the test and reports the retained path.

### Declarations

The following headers are the normative declarations. They are contract text in this section. This artifact does not create the files in the repository.

`include/inferedge/trust_transition.hpp`

```cpp
#ifndef INFEREDGE_TRUST_TRANSITION_HPP
#define INFEREDGE_TRUST_TRANSITION_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace inferedge {

struct ProjectId {
  std::string value;
};

struct IdOrNone {
  bool is_none = true;
  ProjectId id;
};

struct CivilDate {
  std::uint16_t year = 0;
  std::uint8_t month = 0;
  std::uint8_t day = 0;
};

struct Sha256Digest {
  std::array<std::uint8_t, 32> bytes{};
};

enum class TrustPublicationResult : std::uint8_t {
  kOk = 1,
  kBusyMutex = 2,
  kBusyKernelLock = 3,
  kReentered = 4,
  kRootRejected = 5,
  kFilesystemUnsupported = 6,
  kFilesystemIndeterminate = 7,
  kInterrupted = 8,
  kAllocationFailed = 9,
  kKernelLockInvariant = 10,
  kKernelLockResource = 11,
  kKernelLockIo = 12,
  kConflict = 13,
  kAuthorizationRejected = 14,
  kPrefixMismatch = 15,
  kReleaseFailedBeforeDurable = 16,
  kOutcomeUncertain = 17,
  kReleaseInvariant = 18,
  kDescriptorInvariant = 19,
  kPoisoned = 20
};

enum class TrustPublicationKind : std::uint8_t {
  kWithdrawal = 1,
  kOrdinary = 2,
  kRevocation = 3,
  kRollback = 4,
  kNativeInitialization = 5,
  kImportedInitialization = 6,
  kExportFreeze = 7,
  kRetirement = 8,
  kGovernancePin = 9,
  kActivation = 10
};

enum class TrustPublicationStage : std::uint8_t {
  kNone = 0,
  kRootUnbound = 1,
  kRootBound = 2,
  kP0 = 3,
  kP1 = 4,
  kP2 = 5,
  kP3 = 6,
  kP4 = 7,
  kP5 = 8,
  kP6 = 9,
  kP7 = 10,
  kP8 = 11,
  kP9 = 12,
  kN0 = 13,
  kN1 = 14,
  kN2 = 15,
  kN3 = 16,
  kN4 = 17,
  kN5 = 18,
  kN6 = 19,
  kN7 = 20,
  kN8 = 21,
  kI0 = 22,
  kI1 = 23,
  kI2 = 24,
  kI3 = 25,
  kS0 = 26,
  kS1 = 27,
  kS2 = 28,
  kS3 = 29
};

enum class SignatureFamily : std::uint8_t {
  kTrustTransition = 1,
  kLineageOrigin = 2,
  kWithdrawal = 3,
  kReceipt = 4,
  kExportAuthorization = 5,
  kGovernanceContent = 6,
  kGovernanceReview = 7,
  kActivationReview = 8
};

enum class LineageKind : std::uint8_t { kNone = 0, kNative = 1, kImported = 2 };

enum class ServiceStatus : std::uint8_t {
  kEmpty = 1,
  kActive = 2,
  kExportFrozen = 3,
  kRetired = 4
};

enum class WithdrawalSubjectKind : std::uint8_t {
  kProvenance = 1,
  kLicense = 2,
  kGoldenFixture = 3,
  kRegistryEntry = 4,
  kPolicyPin = 5,
  kTelemetryGovernance = 6,
  kActivationAuthorization = 7
};

enum class TrustRole : std::uint8_t {
  kProvenanceReviewer = 1,
  kLicenseReviewer = 2,
  kGoldenReviewer = 3,
  kRegistryReviewer = 4,
  kPolicyReviewer = 5,
  kTelemetryReviewer = 6,
  kTrustAdmin = 7
};

enum class RuntimeMode : std::uint8_t {
  kDisabled = 1,
  kRecommendOnly = 2,
  kShadow = 3,
  kActivation = 4
};

enum class ActivationMode : std::uint8_t { kInactive = 1, kActive = 2 };

enum class AdmissionMode : std::uint8_t { kNoModel = 1, kActive = 2 };

enum class PredecessorCurrent : std::uint8_t { kAbsent = 1, kExactBytes = 2 };
enum class SuccessorDurability : std::uint8_t { kMustCreate = 1, kAlreadyExact = 2 };

enum class FilesystemClass : std::uint8_t {
  kApfs = 1,
  kExtFamily = 2,
  kTmpfs = 3,
  kUnsupported = 4,
  kIndeterminate = 5
};

struct FileIdentity {
  IdOrNone id;
  std::string canonical_relative_path;
  std::uint64_t size_bytes = 0;
  Sha256Digest sha256{};
};

struct WriterBoundIdentity {
  Sha256Digest portable_plan_sha256{};
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
  std::string canonical_path;
  ProjectId host_id;
  Sha256Digest writer_bound_sha256{};
};

struct PortablePublicationPlan {
  TrustPublicationKind kind = TrustPublicationKind::kOrdinary;
  IdOrNone status_operation_id;
  IdOrNone export_operation_id;
  ProjectId trust_id;
  IdOrNone transition_id;
  ProjectId host_id;
  std::string requested_root_path;
  std::string binding_record_kind;
  Sha256Digest new_binding_record_sha256{};
  std::uint64_t new_binding_record_size = 0;
  Sha256Digest source_binding_record_sha256{};
  std::uint64_t source_binding_record_size = 0;
  std::uint64_t successor_file_size = 0;
  Sha256Digest successor_file_sha256{};
  std::optional<ServiceStatus> expected_status;
  std::optional<ServiceStatus> requested_status;
  Sha256Digest portable_plan_sha256{};
};

struct ReleaseDiagnostic {
  bool lock_un_attempted = false;
  int lock_un_errno = 0;
  bool close_attempted = false;
  int close_errno = 0;
  bool poisoned = false;
};

struct TrustPublicationOutcome {
  TrustPublicationResult result = TrustPublicationResult::kPoisoned;
  TrustPublicationStage stage = TrustPublicationStage::kNone;
  ReleaseDiagnostic release{};
  Sha256Digest operation_identity{};
  bool bytes_written = false;
  bool durable_files_may_exist = false;
};

struct FlockHookResult {
  int rc = -1;
  int err_no = 0;
};

struct CloseHookResult {
  int rc = -1;
  int err_no = 0;
};

struct PublicationHooks {
  FilesystemClass (*classify)(std::uint64_t device, std::uint64_t inode) = nullptr;
  FlockHookResult (*flock_op)(int fd, int operation) = nullptr;
  CloseHookResult (*close_fd)(int fd) = nullptr;
};

struct WithdrawalRequest {
  std::string trust_root_path;
  std::string admission_root_path;
  std::string evidence_root_path;
  ProjectId host_id;
  WithdrawalSubjectKind subject_kind = WithdrawalSubjectKind::kProvenance;
  ProjectId subject_id;
  TrustRole signer_role = TrustRole::kProvenanceReviewer;
  ProjectId signer_key_id;
  CivilDate effective_date{};
  std::array<std::uint8_t, 32> public_key{};
  FileIdentity note;
  std::string note_bytes;
  std::array<std::uint8_t, 64> note_signature{};
  FileIdentity note_signature_record;
  FileIdentity note_signature_index;
  FileIdentity note_verification;
  FileIdentity predecessor;
  std::uint32_t predecessor_generation = 0;
  std::uint64_t predecessor_revocation_generation = 0;
  std::vector<ProjectId> predecessor_tombstones;
  FileIdentity successor;
  std::uint32_t successor_generation = 0;
  std::uint64_t successor_revocation_generation = 0;
  std::vector<ProjectId> successor_tombstones;
  FileIdentity transition;
  FileIdentity lineage_record;
  FileIdentity registry_entry;
  std::string successor_bytes;
  std::string transition_record_bytes;
  std::array<std::uint8_t, 64> transition_signature{};
  std::string signature_index_bytes;
  std::string verification_record_bytes;
  std::string predecessor_current_bytes;
  std::string authorization_preimage_bytes;
};

struct OrdinaryTransitionRequest {
  std::string trust_root_path;
  std::string admission_root_path;
  ProjectId host_id;
  TrustPublicationKind kind = TrustPublicationKind::kOrdinary;
  FileIdentity authorization;
  std::string authorization_preimage_bytes;
  ProjectId signer_key_id;
  TrustRole signer_role = TrustRole::kTrustAdmin;
  CivilDate effective_date{};
  std::array<std::uint8_t, 32> public_key{};
  FileIdentity predecessor;
  std::uint32_t predecessor_generation = 0;
  std::uint64_t predecessor_revocation_generation = 0;
  std::vector<ProjectId> predecessor_tombstones;
  FileIdentity predecessor_registry;
  FileIdentity successor;
  std::uint32_t successor_generation = 0;
  std::uint64_t successor_revocation_generation = 0;
  std::vector<ProjectId> successor_tombstones;
  FileIdentity transition;
  LineageKind lineage_kind = LineageKind::kNone;
  std::string successor_bytes;
  std::string transition_record_bytes;
  std::array<std::uint8_t, 64> transition_signature{};
  std::string signature_index_bytes;
  std::string verification_record_bytes;
  std::string predecessor_current_bytes;
};

struct ImportedInitializationRequest;
struct ExportFreezeRequest;
struct RetirementRequest;
struct NativeInitializationRequest;
struct RollbackRequest;
struct GovernancePinRequest;
struct ActivationPublicationRequest;

class VerifiedTransitionPlan;
class VerifiedStatusPlan;

TrustPublicationOutcome publish_verified_transition(VerifiedTransitionPlan&& plan) noexcept;
TrustPublicationOutcome publish_verified_status(VerifiedStatusPlan&& plan) noexcept;
TrustPublicationOutcome publish_withdrawal(const WithdrawalRequest&) noexcept;
TrustPublicationOutcome publish_ordinary_transition(const OrdinaryTransitionRequest&) noexcept;
TrustPublicationOutcome publish_imported_initialization(const ImportedInitializationRequest&) noexcept;
TrustPublicationOutcome publish_export_freeze(const ExportFreezeRequest&) noexcept;
TrustPublicationOutcome publish_retirement(const RetirementRequest&) noexcept;
TrustPublicationOutcome publish_native_initialization(const NativeInitializationRequest&) noexcept;
TrustPublicationOutcome publish_rollback(const RollbackRequest&) noexcept;
TrustPublicationOutcome publish_governance_pin(const GovernancePinRequest&) noexcept;
TrustPublicationOutcome publish_activation(const ActivationPublicationRequest&) noexcept;

std::optional<TrustPublicationKind> parse_trust_publication_kind(std::string_view) noexcept;
std::string_view canonical_trust_publication_kind(TrustPublicationKind) noexcept;
std::optional<SignatureFamily> parse_signature_family(std::string_view) noexcept;
std::string_view canonical_signature_family(SignatureFamily) noexcept;
std::optional<LineageKind> parse_lineage_kind(std::string_view) noexcept;
std::string_view canonical_lineage_kind(LineageKind) noexcept;
std::optional<ServiceStatus> parse_service_status(std::string_view) noexcept;
std::string_view canonical_service_status(ServiceStatus) noexcept;
std::optional<WithdrawalSubjectKind> parse_withdrawal_subject_kind(std::string_view) noexcept;
std::string_view canonical_withdrawal_subject_kind(WithdrawalSubjectKind) noexcept;
std::optional<TrustRole> parse_trust_role(std::string_view) noexcept;
std::string_view canonical_trust_role(TrustRole) noexcept;
std::optional<RuntimeMode> parse_runtime_mode(std::string_view) noexcept;
std::string_view canonical_runtime_mode(RuntimeMode) noexcept;
std::optional<ActivationMode> parse_activation_mode(std::string_view) noexcept;
std::string_view canonical_activation_mode(ActivationMode) noexcept;
std::optional<AdmissionMode> parse_admission_mode(std::string_view) noexcept;
std::string_view canonical_admission_mode(AdmissionMode) noexcept;
bool project_id_ok(std::string_view) noexcept;
std::string canonical_sha256(const Sha256Digest&) noexcept;
bool sha256_from_hex(std::string_view, Sha256Digest&) noexcept;

class VerifiedTransitionPlan {
 public:
  VerifiedTransitionPlan() = delete;
  VerifiedTransitionPlan(const VerifiedTransitionPlan&) = delete;
  VerifiedTransitionPlan& operator=(const VerifiedTransitionPlan&) = delete;
  VerifiedTransitionPlan& operator=(VerifiedTransitionPlan&&) = delete;

 private:
  struct Data {
    PortablePublicationPlan portable;
    std::string successor_bytes;
    std::string transition_record_bytes;
    std::array<std::uint8_t, 64> signature_blob{};
    std::string signature_index_bytes;
    std::string verification_record_bytes;
    std::string authorization_preimage_bytes;
    std::array<std::uint8_t, 32> public_key{};
    std::string lineage_record_bytes;
    std::array<std::uint8_t, 64> lineage_signature_blob{};
    std::string lineage_index_bytes;
    std::string lineage_verification_bytes;
    PredecessorCurrent predecessor_current = PredecessorCurrent::kExactBytes;
    std::string predecessor_current_bytes;
    SuccessorDurability successor_durability = SuccessorDurability::kMustCreate;
  };
  Data data_;
  explicit VerifiedTransitionPlan(Data data) noexcept : data_(std::move(data)) {}
  VerifiedTransitionPlan(VerifiedTransitionPlan&&) noexcept = default;
  ~VerifiedTransitionPlan() = default;

  friend TrustPublicationOutcome publish_verified_transition(VerifiedTransitionPlan&&) noexcept;
  friend TrustPublicationOutcome publish_withdrawal(const WithdrawalRequest&) noexcept;
  friend TrustPublicationOutcome publish_ordinary_transition(const OrdinaryTransitionRequest&) noexcept;
  friend TrustPublicationOutcome publish_imported_initialization(
      const ImportedInitializationRequest&) noexcept;
  friend TrustPublicationOutcome publish_native_initialization(
      const NativeInitializationRequest&) noexcept;
  friend TrustPublicationOutcome publish_rollback(const RollbackRequest&) noexcept;
  friend TrustPublicationOutcome publish_governance_pin(const GovernancePinRequest&) noexcept;
  friend TrustPublicationOutcome publish_activation(const ActivationPublicationRequest&) noexcept;
};

class VerifiedStatusPlan {
 public:
  VerifiedStatusPlan() = delete;
  VerifiedStatusPlan(const VerifiedStatusPlan&) = delete;
  VerifiedStatusPlan& operator=(const VerifiedStatusPlan&) = delete;
  VerifiedStatusPlan& operator=(VerifiedStatusPlan&&) = delete;

 private:
  struct Data {
    PortablePublicationPlan portable;
    std::string expected_status_bytes;
    std::string requested_status_bytes;
    std::string status_path;
    std::string status_partial_path;
    std::string authorization_preimage_bytes;
    std::string signature_record_bytes;
    std::array<std::uint8_t, 64> signature_blob{};
    std::string signature_index_bytes;
    std::string verification_record_bytes;
    std::array<std::uint8_t, 32> public_key{};
    std::string current_bytes;
    std::string portable_authorization_prefix;
  };
  Data data_;
  explicit VerifiedStatusPlan(Data data) noexcept : data_(std::move(data)) {}
  VerifiedStatusPlan(VerifiedStatusPlan&&) noexcept = default;
  ~VerifiedStatusPlan() = default;

  friend TrustPublicationOutcome publish_verified_status(VerifiedStatusPlan&&) noexcept;
  friend TrustPublicationOutcome publish_export_freeze(const ExportFreezeRequest&) noexcept;
  friend TrustPublicationOutcome publish_retirement(const RetirementRequest&) noexcept;
  friend TrustPublicationOutcome publish_native_initialization(
      const NativeInitializationRequest&) noexcept;
  friend TrustPublicationOutcome publish_imported_initialization(
      const ImportedInitializationRequest&) noexcept;
};

#if defined(INFEREDGE_TRUST_PUBLICATION_TEST_SEAM)
bool set_publication_hooks_for_test(const PublicationHooks& hooks) noexcept;
#endif


static_assert(!std::is_default_constructible_v<VerifiedTransitionPlan>);
static_assert(!std::is_copy_constructible_v<VerifiedTransitionPlan>);
static_assert(!std::is_move_constructible_v<VerifiedTransitionPlan>);
static_assert(!std::is_move_assignable_v<VerifiedTransitionPlan>);
static_assert(!std::is_destructible_v<VerifiedTransitionPlan>);
static_assert(!std::is_default_constructible_v<VerifiedStatusPlan>);
static_assert(!std::is_copy_constructible_v<VerifiedStatusPlan>);
static_assert(!std::is_move_constructible_v<VerifiedStatusPlan>);
static_assert(!std::is_destructible_v<VerifiedStatusPlan>);

}  // namespace inferedge

#endif

```

`include/inferedge/trust_rollback.hpp`

```cpp
#ifndef INFEREDGE_TRUST_ROLLBACK_HPP
#define INFEREDGE_TRUST_ROLLBACK_HPP

#include "trust_transition.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace inferedge {

struct RollbackRequest {
  std::string trust_root_path;
  std::string admission_root_path;
  ProjectId host_id;
  FileIdentity authorization;
  std::string authorization_preimage_bytes;
  std::array<std::uint8_t, 64> transition_signature{};
  std::array<std::uint8_t, 32> public_key{};
  ProjectId signer_key_id;
  TrustRole signer_role = TrustRole::kTrustAdmin;
  CivilDate effective_date{};
  FileIdentity predecessor;
  std::uint32_t predecessor_generation = 0;
  std::uint64_t predecessor_revocation_generation = 0;
  std::vector<ProjectId> predecessor_tombstones;
  FileIdentity successor;
  std::uint32_t successor_generation = 0;
  std::uint64_t successor_revocation_generation = 0;
  std::vector<ProjectId> successor_tombstones;
  FileIdentity transition;
  FileIdentity predecessor_registry;
  FileIdentity successor_registry;
  FileIdentity predecessor_pin;
  FileIdentity successor_pin;
  std::vector<ProjectId> successor_pin_subjects;
  RuntimeMode predecessor_runtime_mode = RuntimeMode::kDisabled;
  RuntimeMode successor_runtime_mode = RuntimeMode::kDisabled;
  AdmissionMode successor_admission_mode = AdmissionMode::kNoModel;
  ProjectId predecessor_telemetry_governance_id;
  ProjectId successor_telemetry_governance_id;
  std::uint64_t successor_telemetry_size = 0;
  Sha256Digest successor_telemetry_sha256{};
  std::string successor_bytes;
  std::string transition_record_bytes;
  std::string signature_index_bytes;
  std::string verification_record_bytes;
  std::string predecessor_current_bytes;
};

}  // namespace inferedge

#endif

```

`include/inferedge/trust_governance.hpp`

```cpp
#ifndef INFEREDGE_TRUST_GOVERNANCE_HPP
#define INFEREDGE_TRUST_GOVERNANCE_HPP

#include "trust_transition.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace inferedge {

struct GovernancePinRequest {
  std::string trust_root_path;
  std::string admission_root_path;
  std::string evidence_root_path;
  ProjectId host_id;
  ProjectId governance_id;
  FileIdentity governance_content;
  FileIdentity review_note;
  FileIdentity review_signature_record;
  std::array<std::uint8_t, 64> review_signature_blob{};
  FileIdentity review_signature_index;
  FileIdentity review_verification;
  FileIdentity accepted_decision;
  FileIdentity head_set;
  std::vector<ProjectId> head_set_members;
  ProjectId telemetry_governance_id;
  std::uint64_t telemetry_governance_file_size = 0;
  Sha256Digest telemetry_governance_file_sha256{};
  ProjectId signer_key_id;
  TrustRole signer_role = TrustRole::kTelemetryReviewer;
  CivilDate effective_date{};
  std::array<std::uint8_t, 32> public_key{};
  FileIdentity predecessor;
  std::uint32_t predecessor_generation = 0;
  std::uint64_t predecessor_revocation_generation = 0;
  std::vector<ProjectId> predecessor_tombstones;
  FileIdentity successor;
  std::uint32_t successor_generation = 0;
  FileIdentity transition;
  std::array<std::uint8_t, 64> transition_signature{};
  std::string successor_bytes;
  std::string transition_record_bytes;
  std::string signature_index_bytes;
  std::string verification_record_bytes;
  std::string predecessor_current_bytes;
  std::string authorization_preimage_bytes;
};

}  // namespace inferedge

#endif

```

`include/inferedge/trust_activation.hpp`

```cpp
#ifndef INFEREDGE_TRUST_ACTIVATION_HPP
#define INFEREDGE_TRUST_ACTIVATION_HPP

#include "trust_transition.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace inferedge {

struct ActivationPublicationRequest {
  std::string trust_root_path;
  std::string admission_root_path;
  std::string evidence_root_path;
  ProjectId host_id;
  FileIdentity activation_authorization;
  FileIdentity accepted_decision;
  FileIdentity policy_decision;
  FileIdentity candidate;
  FileIdentity dataset_file;
  FileIdentity invariance_file;
  ProjectId model_id;
  FileIdentity head_set;
  std::vector<ProjectId> head_set_members;
  RuntimeMode requested_runtime_mode = RuntimeMode::kActivation;
  ActivationMode requested_activation_mode = ActivationMode::kActive;
  FileIdentity review_note;
  FileIdentity review_signature_record;
  std::array<std::uint8_t, 64> review_signature_blob{};
  FileIdentity review_signature_index;
  FileIdentity review_verification;
  ProjectId signer_key_id;
  TrustRole signer_role = TrustRole::kPolicyReviewer;
  CivilDate effective_date{};
  std::array<std::uint8_t, 32> public_key{};
  FileIdentity predecessor;
  std::uint32_t predecessor_generation = 0;
  std::uint64_t predecessor_revocation_generation = 0;
  std::vector<ProjectId> predecessor_tombstones;
  FileIdentity successor;
  std::uint32_t successor_generation = 0;
  FileIdentity transition;
  std::array<std::uint8_t, 64> transition_signature{};
  std::string successor_bytes;
  std::string transition_record_bytes;
  std::string signature_index_bytes;
  std::string verification_record_bytes;
  std::string predecessor_current_bytes;
  std::string authorization_preimage_bytes;
};

}  // namespace inferedge

#endif

```

`include/inferedge/trust_migration.hpp`

```cpp
#ifndef INFEREDGE_TRUST_MIGRATION_HPP
#define INFEREDGE_TRUST_MIGRATION_HPP

#include "trust_transition.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace inferedge {

struct ImportedInitializationRequest {
  std::string destination_trust_root_path;
  std::string destination_admission_root_path;
  std::string destination_evidence_root_path;
  ProjectId destination_host_id;
  ProjectId migration_id;
  ProjectId transaction_id;
  ProjectId nonce_id;
  FileIdentity package_file;
  FileIdentity payload_file;
  FileIdentity transaction_file;
  FileIdentity imported_successor;
  FileIdentity imported_transition;
  FileIdentity lineage_record;
  FileIdentity lineage_signature;
  FileIdentity lineage_index;
  FileIdentity lineage_verification;
  std::uint64_t prior_high_water = 0;
  std::uint64_t new_high_water = 0;
  FileIdentity prior_high_water_file;
  FileIdentity new_high_water_file;
  FileIdentity prior_status_file;
  std::string imported_successor_bytes;
  std::string empty_status_bytes;
  std::string active_status_bytes;
  std::array<std::uint8_t, 32> live_bootstrap_key{};
  Sha256Digest bootstrap_key_sha256{};
  ProjectId status_operation_id;
  FileIdentity source_binding;
  std::string import_destination_record_bytes;
};

struct ExportFreezeRequest {
  std::string source_trust_root_path;
  std::string source_admission_root_path;
  std::string source_evidence_root_path;
  ProjectId source_host_id;
  FileIdentity source_current;
  std::string source_current_bytes;
  std::string expected_active_status_bytes;
  FileIdentity expected_status_file;
  ProjectId export_operation_id;
  std::string authorization_record_bytes;
  FileIdentity authorization_record;
  std::string authorization_preimage_bytes;
  std::string signature_record_bytes;
  FileIdentity signature_record;
  std::array<std::uint8_t, 64> signature_blob{};
  FileIdentity signature_blob_file;
  std::string signature_index_bytes;
  FileIdentity signature_index;
  std::string verification_record_bytes;
  FileIdentity verification_record;
  ProjectId registry_entry_id;
  ProjectId signer_key_id;
  TrustRole signer_role = TrustRole::kTrustAdmin;
  CivilDate effective_date{};
  std::array<std::uint8_t, 32> public_key{};
  Sha256Digest public_key_sha256{};
  FileIdentity package_file;
  std::string export_frozen_status_bytes;
  ProjectId status_operation_id;
};

struct RetirementRequest {
  std::string source_trust_root_path;
  std::string source_admission_root_path;
  std::string source_evidence_root_path;
  ProjectId source_host_id;
  std::string destination_trust_root_path;
  std::string destination_admission_root_path;
  std::string destination_evidence_root_path;
  ProjectId destination_host_id;
  FileIdentity source_current;
  std::string source_current_bytes;
  std::string expected_export_frozen_status_bytes;
  FileIdentity expected_status_file;
  ProjectId migration_id;
  ProjectId transaction_id;
  FileIdentity completed_export;
  ProjectId receipt_id;
  std::string receipt_record_bytes;
  FileIdentity receipt_file;
  std::string receipt_preimage_bytes;
  std::string signature_record_bytes;
  FileIdentity signature_record;
  std::array<std::uint8_t, 64> signature_blob{};
  FileIdentity signature_blob_file;
  std::string signature_index_bytes;
  FileIdentity signature_index;
  std::string verification_record_bytes;
  FileIdentity verification_record;
  ProjectId signer_key_id;
  TrustRole signer_role = TrustRole::kTrustAdmin;
  CivilDate effective_date{};
  std::array<std::uint8_t, 32> public_key{};
  Sha256Digest public_key_sha256{};
  std::string retired_status_bytes;
  ProjectId status_operation_id;
};

}  // namespace inferedge

#endif

```

`include/inferedge/trust_service.hpp`

```cpp
#ifndef INFEREDGE_TRUST_SERVICE_HPP
#define INFEREDGE_TRUST_SERVICE_HPP

#include "trust_transition.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace inferedge {

class NativeBootstrapSigner {
 public:
  virtual ~NativeBootstrapSigner() = default;
  virtual bool sign(std::span<const std::uint8_t> message,
                    std::array<std::uint8_t, 64>& out) noexcept = 0;
};

struct NativeInitializationRequest {
  std::string trust_root_path;
  std::string admission_root_path;
  std::string evidence_root_path;
  ProjectId host_id;
  NativeBootstrapSigner* signer = nullptr;
  std::array<std::uint8_t, 32> bootstrap_public_key{};
  Sha256Digest bootstrap_key_sha256{};
  FileIdentity bootstrap_key_file;
  FileIdentity generation1_successor;
  std::uint32_t generation = 1;
  CivilDate effective_date{};
  FileIdentity genesis_transition;
  std::string generation1_bytes;
  std::string genesis_transition_bytes;
  std::string genesis_index_bytes;
  std::string genesis_verification_bytes;
  std::array<std::uint8_t, 64> genesis_signature{};
  FileIdentity configuration_file;
  FileIdentity registry_file;
  FileIdentity revocation_file;
  FileIdentity manifest_file;
  std::string lineage_record_bytes;
  std::string lineage_index_bytes;
  std::string lineage_verification_bytes;
  std::array<std::uint8_t, 64> lineage_signature{};
  std::string empty_status_bytes;
  std::string active_status_bytes;
  ProjectId status_operation_id;
};

}  // namespace inferedge

#endif

```

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

- `genesis/transition.payload`
- `genesis/transition.record`
- `genesis/signature.bin`
- `genesis/signature.index`
- `genesis/verification.record`
- for every later transition in the authenticated chain, `transitions/<transition-id>/transition.payload`, `transition.record`, `signature.bin`, `signature.index`, and `verification.record`

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

Section 2 is the roadmap. Erratum E2 section 14 is contract text only. This artifact does not authorize applying it, creating publication sources, staging, or release. Correctness precedes performance. Invariance precedes dataset acceptance, training documentation, and activation. Governance precedes any log. Recommend-only precedes shadow. Shadow precedes activation. Withdrawal fixtures and rollback tests precede activation. Campaign 001 is not an input. Campaign 002 is not a phase.


