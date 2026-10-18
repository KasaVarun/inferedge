#!/usr/bin/env python3
"""Mechanics tests for the R7 generative-overhead analyzer."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from benchmarks.analyze_generative import (
    CONSTRUCTION_CELLS,
    EMPTY_SHA256,
    EVIDENCE_PATH_CONTRACT,
    EXPECTED_CELL_IDS,
    MEASURED_KEYS,
    MEASURED_PAIRS,
    REJECTION_CODES,
    WARMUP_KEYS,
    analyze_inventory_a,
    bootstrap_mean_interval,
    build_aggregate,
    build_summary,
    claim_from_c01,
    classify_digest_field,
    compare_arms,
    descriptive_stats,
    load_json_object,
    matrix_is_complete,
    reject_existing_analysis,
    sibling_digest,
    unique_sorted,
    validate_measured_record,
    validate_warmup_record,
    verify_readonly,
)
from benchmarks.collect_generative import (
    canonical_json_bytes,
    inventory_a_relative_paths,
    inventory_b_relative_paths,
    token_ids_sha256,
)


def identity_fields() -> dict:
    return {
        "schema_family": "generative_overhead",
        "schema_version": 1,
        "campaign_id": "camp-1",
        "campaign_manifest_sha256": "a" * 64,
        "protocol_document_sha256": "b" * 64,
        "fixtures_sha256": "c" * 64,
        "source_commit": "d" * 40,
    }


def warmup_record(**overrides: object) -> dict:
    ids = [1, 2]
    text = "x"
    record = {
        **identity_fields(),
        "record_class": "warmup",
        "cell_id": "C01",
        "implementation": "inferedge",
        "warmup_index": 0,
        "measurement_class": "warm_request",
        "success": True,
        "prompt_token_count_expected": 2,
        "prompt_token_count_actual": 2,
        "prompt_token_ids": ids,
        "prompt_token_ids_sha256": token_ids_sha256(ids),
        "generated_token_count": 1,
        "generated_token_ids": [3],
        "generated_token_ids_sha256": token_ids_sha256([3]),
        "generated_text_utf8": text,
        "generated_text_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
        "finish_reason": "maximum_tokens",
        "utf8_valid": True,
        "threads_requested": 1,
        "threads_effective": 1,
        "n_ctx": 4096,
        "n_batch": 4096,
        "n_ubatch": 512,
        "n_seq_max": 1,
        "worker_binary_sha256": "e" * 64,
        "worker_instance_id": "ie-1",
        "pid": 10,
        "t0_ns": 1,
        "t1_ns": 3,
        "duration_ns": 2,
        "error_code": "none",
        "error_message": "",
    }
    record.update(overrides)
    ordered = {key: record[key] for key in WARMUP_KEYS}
    return ordered


def measured_record(**overrides: object) -> dict:
    ids = [1, 2]
    text = "x"
    record = {
        **identity_fields(),
        "record_class": "measured",
        "cell_id": "C01",
        "implementation": "inferedge",
        "pair_index": 0,
        "arm_order": "ie_then_ref",
        "order_position": 0,
        "measurement_class": "warm_request",
        "fixture_id": "P_tiny",
        "max_new_tokens": 1,
        "success": True,
        "prompt_token_count_expected": 2,
        "prompt_token_count_actual": 2,
        "prompt_token_ids": ids,
        "prompt_token_ids_sha256": token_ids_sha256(ids),
        "generated_token_count": 1,
        "generated_token_ids": [3],
        "generated_token_ids_sha256": token_ids_sha256([3]),
        "generated_text_utf8": text,
        "generated_text_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
        "finish_reason": "maximum_tokens",
        "utf8_valid": True,
        "threads_requested": 1,
        "threads_effective": 1,
        "n_ctx": 4096,
        "n_batch": 4096,
        "n_ubatch": 512,
        "n_seq_max": 1,
        "worker_binary_sha256": "e" * 64,
        "worker_instance_id": "ie-1",
        "pid": 10,
        "t0_ns": 10,
        "t1_ns": 20,
        "duration_ns": 10,
        "error_code": "none",
        "error_message": "",
        "exclusion_status": "none",
        "exclusion_reason": "",
        "publication_eligible": True,
    }
    record.update(overrides)
    return {key: record[key] for key in MEASURED_KEYS}


class SchemaAndMappingTests(unittest.TestCase):
    def test_contract_and_codes(self) -> None:
        self.assertEqual(EVIDENCE_PATH_CONTRACT, "generative-overhead-r7-inventory-v1")
        self.assertEqual(len(REJECTION_CODES), len(set(REJECTION_CODES)))
        self.assertEqual(REJECTION_CODES, sorted(REJECTION_CODES))
        self.assertEqual(REJECTION_CODES.count("warmup_invalid"), 1)
        self.assertNotIn("generative-overhead-r6-inventory-v1", REJECTION_CODES)

    def test_warmup_timing_invalid(self) -> None:
        codes = validate_warmup_record(warmup_record(duration_ns=99))
        self.assertIn("timing_invalid", codes)
        self.assertNotIn("warmup_invalid", [c for c in codes if c == "timing_invalid"])

    def test_malformed_digest_schema(self) -> None:
        self.assertEqual(classify_digest_field(12, "a" * 64), "record_schema_invalid")
        self.assertEqual(classify_digest_field("ZZ", "a" * 64), "record_schema_invalid")

    def test_valid_hex_wrong_sibling(self) -> None:
        expected = token_ids_sha256([1])
        self.assertEqual(classify_digest_field("a" * 64, expected), "digest_mismatch")

    def test_empty_digest(self) -> None:
        self.assertEqual(token_ids_sha256([]), EMPTY_SHA256)

    def test_independent_token_text_codes(self) -> None:
        left = measured_record()
        right = measured_record(
            implementation="reference",
            order_position=1,
            prompt_token_ids=[9],
            prompt_token_ids_sha256=token_ids_sha256([9]),
            generated_text_utf8="y",
            generated_text_sha256=__import__("hashlib").sha256(b"y").hexdigest(),
        )
        codes = compare_arms(left, right)
        self.assertIn("token_mismatch", codes)
        self.assertIn("text_mismatch", codes)

    def test_unique_sorted(self) -> None:
        self.assertEqual(unique_sorted(["text_mismatch", "token_mismatch", "token_mismatch"]), ["text_mismatch", "token_mismatch"])

    def test_finish_policy(self) -> None:
        codes = validate_measured_record(measured_record(finish_reason="eos"))
        self.assertIn("finish_reason_invalid", codes)
        c07 = measured_record(cell_id="C07", finish_reason="eos")
        self.assertNotIn("finish_reason_invalid", validate_measured_record(c07))

    def test_pair_order(self) -> None:
        trial = {
            "cell_id": "C01",
            "pair_index": 0,
            "arm_order": "ie_then_ref",
            "order_position": 0,
            "implementation": "inferedge",
        }
        self.assertEqual(validate_measured_record(measured_record(), trial), [])
        bad = validate_measured_record(measured_record(pair_index=1), trial)
        self.assertIn("pair_order_invalid", bad)

    def test_wrong_field_order(self) -> None:
        record = warmup_record()
        record["zzz"] = 1
        self.assertIn("record_schema_invalid", validate_warmup_record(record))


class InventoryAndRerunTests(unittest.TestCase):
    def test_inventory_counts(self) -> None:
        self.assertEqual(len(inventory_a_relative_paths()), 920)
        self.assertEqual(len(inventory_b_relative_paths()), 922)

    def test_missing_extra_and_analysis_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codes = analyze_inventory_a(root)
            self.assertIn("missing_file", codes)
            (root / "unexpected.txt").write_text("x", encoding="utf-8")
            codes = analyze_inventory_a(root)
            self.assertIn("extra_file", codes)
            os.mkdir(root / "analysis")
            self.assertEqual(reject_existing_analysis(root), ["campaign_incomplete"])

    def test_wrong_zero_padding_is_extra_or_path(self) -> None:
        self.assertIn("measured/C01/pair-00/inferedge.json", inventory_a_relative_paths())
        self.assertNotIn("measured/C01/pair-0/inferedge.json", inventory_a_relative_paths())


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_determinism_and_zero_endpoint(self) -> None:
        diffs = [0] * 20
        low, high = bootstrap_mean_interval(diffs)
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 0.0)
        again = bootstrap_mean_interval(diffs)
        self.assertEqual((low, high), again)

    def test_integer_thresholds(self) -> None:
        ie = [3_000_000] * 20
        ref = [100_000] * 20
        claim = claim_from_c01(ie, ref)
        self.assertTrue(claim["absolute_threshold_met"])
        self.assertTrue(claim["relative_threshold_met"])
        self.assertEqual(claim["direction"], "inferedge_slower")
        boundary = claim_from_c01([2_100_000] * 20, [100_000] * 20)
        self.assertTrue(boundary["absolute_threshold_met"])
        below = claim_from_c01([2_099_999] * 20, [100_000] * 20)
        self.assertFalse(below["absolute_threshold_met"])

    def test_descriptive_indices(self) -> None:
        values = list(range(20))
        stats = descriptive_stats(values)
        self.assertEqual(stats["p25_difference_ns"], 4)
        self.assertEqual(stats["p50_difference_ns"], 9)
        self.assertEqual(stats["p75_difference_ns"], 14)
        self.assertEqual(stats["p95_difference_ns"], 18)

    def test_aggregate_summary_nulls_and_dry_run(self) -> None:
        identities = {
            "campaign_class": "dry_run",
            "campaign_manifest_sha256": "a" * 64,
            "protocol_document_sha256": "b" * 64,
            "fixtures_sha256": "c" * 64,
            "source_commit": "d" * 40,
        }
        aggregate = build_aggregate("dry", identities, {}, {"pair_count": 0}, ["campaign_class_invalid"])
        payload = canonical_json_bytes(aggregate)
        summary = build_summary(aggregate, payload)
        self.assertEqual(summary["claim_status"], "no_winner_ineligible")
        self.assertIsNone(summary["c01_mean_difference_ns"])
        self.assertEqual(summary["aggregate_sha256"], __import__("hashlib").sha256(payload).hexdigest())
        self.assertFalse(aggregate["winner_declared"])

    def test_ci_endpoint_zero_not_strict(self) -> None:
        claim = claim_from_c01([5] * 20, [5] * 20)
        self.assertFalse(claim["ci_excludes_zero_strict"])
        self.assertFalse(claim["winner_declared"])


class ExactMatrixEligibilityTests(unittest.TestCase):
    def _stats(self, cells: list[str], pair_count: int = 20) -> dict:
        return {cell: {"pair_count": pair_count} for cell in cells}

    def _identities(self, campaign_class: str = "campaign") -> dict:
        return {
            "campaign_class": campaign_class,
            "campaign_manifest_sha256": "a" * 64,
            "protocol_document_sha256": "b" * 64,
            "fixtures_sha256": "c" * 64,
            "source_commit": "d" * 40,
        }

    def _c01(self) -> dict:
        ie = [3_000_000] * 20
        ref = [100_000] * 20
        claim = claim_from_c01(ie, ref)
        claim["pair_count"] = 20
        return claim

    def test_exact_twelve_cells_complete(self) -> None:
        stats = self._stats(list(CONSTRUCTION_CELLS))
        self.assertTrue(matrix_is_complete(stats, []))
        aggregate = build_aggregate("full", self._identities(), stats, self._c01(), [])
        self.assertTrue(aggregate["complete_matrix"])
        self.assertTrue(aggregate["winner_declared"])
        self.assertEqual(aggregate["c01"]["direction"], "inferedge_slower")

    def test_c01_only_incomplete(self) -> None:
        stats = self._stats(["C01"])
        self.assertFalse(matrix_is_complete(stats, []))
        aggregate = build_aggregate("c01", self._identities(), stats, self._c01(), [])
        self.assertFalse(aggregate["complete_matrix"])
        self.assertFalse(aggregate["winner_declared"])

    def test_each_missing_cell_incomplete(self) -> None:
        for missing in CONSTRUCTION_CELLS:
            cells = [cell for cell in CONSTRUCTION_CELLS if cell != missing]
            stats = self._stats(cells)
            self.assertFalse(matrix_is_complete(stats, []), missing)

    def test_eleven_thirteen_unknown_and_pair_counts(self) -> None:
        eleven = self._stats(list(CONSTRUCTION_CELLS)[:-1])
        self.assertFalse(matrix_is_complete(eleven, []))
        thirteen = self._stats(list(CONSTRUCTION_CELLS) + ["C12"])
        self.assertFalse(matrix_is_complete(thirteen, []))
        unknown = self._stats(["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "CXX"])
        self.assertFalse(matrix_is_complete(unknown, []))
        nineteen = self._stats(list(CONSTRUCTION_CELLS))
        nineteen["C03"] = {"pair_count": 19}
        self.assertFalse(matrix_is_complete(nineteen, []))
        twenty_one = self._stats(list(CONSTRUCTION_CELLS))
        twenty_one["C03"] = {"pair_count": 21}
        self.assertFalse(matrix_is_complete(twenty_one, []))

    def test_duplicate_and_oversized_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dup.json"
            path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
            with self.assertRaises(Exception) as caught:
                load_json_object(path)
            self.assertIn("record_schema_invalid", caught.exception.codes)
            huge = Path(tmp) / "huge.json"
            huge.write_bytes(b'{"a":"' + (b"x" * 2_000_001) + b'"}')
            with self.assertRaises(Exception) as caught:
                load_json_object(huge)
            self.assertIn("record_schema_invalid", caught.exception.codes)

    def test_blocking_codes_and_partial_and_dry_run(self) -> None:
        stats = self._stats(list(CONSTRUCTION_CELLS))
        aggregate = build_aggregate("blocked", self._identities(), stats, self._c01(), ["timing_invalid"])
        self.assertFalse(aggregate["complete_matrix"])
        self.assertFalse(aggregate["winner_declared"])
        self.assertFalse(aggregate["superiority_claimed"])
        summary = build_summary(aggregate, canonical_json_bytes(aggregate))
        self.assertEqual(summary["claim_status"], "no_winner_ineligible")
        partial = build_aggregate("partial", self._identities(), {"C01": {"pair_count": 20}}, {"pair_count": 20}, [])
        self.assertFalse(partial["winner_declared"])
        dry = build_aggregate("dry", self._identities("dry_run"), stats, self._c01(), [])
        self.assertFalse(dry["winner_declared"])
        self.assertIn("campaign_class_invalid", dry["rejection_codes"])

    def test_valid_full_matrix_keeps_statistics(self) -> None:
        stats = {}
        for cell in EXPECTED_CELL_IDS:
            stats[cell] = {
                "pair_count": MEASURED_PAIRS,
                "inferedge_mean_ns": 3_000_000.0,
                "reference_mean_ns": 100_000.0,
                "mean_difference_ns": 2_900_000.0,
                "median_difference_ns": 2_900_000,
                "p50_difference_ns": 2_900_000,
                "p95_difference_ns": 2_900_000,
                "iqr_difference_ns": 0,
                "mad_difference_ns": 0,
            }
        aggregate = build_aggregate("stats", self._identities(), stats, self._c01(), [])
        self.assertTrue(aggregate["complete_matrix"])
        self.assertEqual(len(aggregate["cells"]), 12)
        self.assertEqual(aggregate["c01"]["sum_d"], 20 * 2_900_000)


class MappingTableTests(unittest.TestCase):
    def test_required_mapping_codes_exist(self) -> None:
        required = {
            "atomic_publish_failure",
            "campaign_class_invalid",
            "campaign_incomplete",
            "correctness_mismatch",
            "digest_mismatch",
            "duplicate_record",
            "environment_ac_invalid",
            "environment_command_failed",
            "environment_lpm_invalid",
            "environment_parse_invalid",
            "environment_sleepwake_invalid",
            "extra_file",
            "finish_reason_invalid",
            "hardlink_rejected",
            "identity_mismatch",
            "manifest_invalid",
            "matrix_invalid",
            "missing_file",
            "mixed_campaign",
            "pair_order_invalid",
            "path_invalid",
            "record_schema_invalid",
            "sample_count_invalid",
            "symlink_rejected",
            "timing_invalid",
            "token_mismatch",
            "text_mismatch",
            "utf8_invalid",
            "warmup_invalid",
            "worker_restart",
        }
        self.assertEqual(set(REJECTION_CODES), required)


if __name__ == "__main__":
    unittest.main()
