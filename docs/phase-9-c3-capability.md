# Phase 9-C3 Vocabulary Capability

Status: committed capability vocabulary for the Phase 9-C contract.

## Scope and non-goals

Phase 9-C3 records one vocabulary-capability file. The file states that no vocabulary-size symbol is committed and that no vocabulary range check is enabled.

This phase does not admit a model, load a model, enable telemetry, authorize activation, execute tokenization, download weights, or publish a benchmark or performance claim. It does not place the record in an admission-root manifest, and it does not implement runtime consumption of the record. Tensor canonicalization is not part of this phase. Phase 9-C4 remains a separate phase.

## Authority

The authoritative contract is `docs/phase-9-c-admission-contract.md`.

The record follows section 2 for phase ownership, section 5 for `yes-no` and `id-or-none`, section 6 for canonical `key=value` serialization, section 8 for the distinction between a self-checksum and a full-file digest, section 16 for the always-required manifest entry, section 29 for the vocabulary schema and the two-call tokenizer probe, and section 46 for fail-closed rejection.

## Record

Path: `admission/capability/vocabulary.record`.

Closed key order:

1. `schema_version`
2. `record_kind`
3. `symbol_present`
4. `symbol_name`
5. `symbol_header`
6. `range_check_enforced`
7. `record_sha256`

Committed values:

- `schema_version=1`
- `record_kind=vocabulary_capability`
- `symbol_present=no`
- `symbol_name=none`
- `symbol_header=none`
- `range_check_enforced=no`

`symbol_present=no` means Phase 9-C3 does not commit a vocabulary-size symbol. `llama_vocab_n_tokens` is not required.

`symbol_name=none` means the record names no symbol.

`symbol_header=none` means the record names no header.

`range_check_enforced=no` means no vocabulary range check is enabled. A range check starts only when a future authorized record has both `symbol_present=yes` and `range_check_enforced=yes`.

## Tokenizer probe

This record does not disable the section 29 two-call `llama_tokenize` probe. Both calls use `add_special=true` and `parse_special=true`. The first call is the size probe. The second call writes the tokens. Probe failure remains a tokenization failure under section 29. The capability range check in the request control flow applies only when both `symbol_present` and `range_check_enforced` are `yes`.

## Future golden case lists

On a future golden case list, `vocabulary_id` names this one capability file. `vocabulary_file_size` is the full file size, 208 bytes. `vocabulary_file_sha256` is the full-file digest of those 208 bytes. This phase does not create a golden case list.

## Future admission-root manifests

Section 16 makes `vocabulary_capability` an always-required admission-root manifest entry. A missing required entry fails the load. This phase does not create that manifest and does not claim the record is already listed in one.

## Identity

Producer: Phase 9-C3.

Consumer: every request.

Signature: none.

Cardinality: one.

## Digests

`record_sha256` is the self-checksum. It is SHA-256 of every byte before the `record_sha256=` line, including the LF that ends the `range_check_enforced` line, and excluding the checksum line.

The full-file digest is SHA-256 of all 208 file bytes, including the checksum line and its final LF. The self-checksum is not the full-file digest.

- Self-checksum: `8c272b11ee24e622a49c54e09d33b380f9b904a865a1c1f3d2287fbda12d8569`
- Full-file SHA-256: `00652eaa0040219bc8d7d1eedc370840e2fabf76a962be5bb0676420100bb687`
- Size: 208 bytes
- Lines: 7

## Fail-closed behavior

A lexical, digest, size, or schema failure rejects the file and leaves current state in place. An altered byte, a wrong kind, a checksum mismatch, or a disagreement between the recorded size and the full-file digest fails closed. Enabling a range check requires a future record in which both `symbol_present` and `range_check_enforced` are `yes`. This file does not enable that check.
