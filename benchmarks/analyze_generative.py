#!/usr/bin/env python3
"""Phase 9-B3A-R7 generative-overhead analyzer.

No model or tokenizer dependency. Import and --help do not analyze a campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import stat
import sys
from pathlib import Path
from typing import Any

from benchmarks.collect_generative import (
    ABSOLUTE_THRESHOLD_NS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CELL_SPECS,
    COMMAND_IDS,
    CONSTRUCTION_CELLS,
    EMPTY_SHA256,
    EVIDENCE_PATH_CONTRACT,
    GOLDEN_CELL_ORDER,
    MEASURED_PAIRS,
    RELATIVE_THRESHOLD_DENOMINATOR,
    RELATIVE_THRESHOLD_NUMERATOR,
    SCHEMA_FAMILY,
    SCHEMA_VERSION,
    WARMUP_COUNT,
    canonical_json_bytes,
    encode_token_ids_le,
    environment_sample_ids,
    expanded_trials,
    inventory_a_relative_paths,
    inventory_b_relative_paths,
    lstat_no_follow,
    publish_exclusive,
    reject_relative_path,
    require_directory,
    require_regular_file,
    sha256_bytes,
    token_ids_sha256,
)

REJECTION_CODES = [
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
    "text_mismatch",
    "timing_invalid",
    "token_mismatch",
    "utf8_invalid",
    "warmup_invalid",
    "worker_restart",
]

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
TEMP_NAME_RE = re.compile(r"^\.[^/]+\.pub\.[0-9a-f]{32}\.tmp$")
MAX_TRUSTED_JSON_BYTES = 2_000_000
EXPECTED_CELL_IDS = tuple(CONSTRUCTION_CELLS)

WARMUP_KEYS = [
    "schema_family",
    "schema_version",
    "record_class",
    "campaign_id",
    "campaign_manifest_sha256",
    "protocol_document_sha256",
    "fixtures_sha256",
    "source_commit",
    "cell_id",
    "implementation",
    "warmup_index",
    "measurement_class",
    "success",
    "prompt_token_count_expected",
    "prompt_token_count_actual",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "generated_token_count",
    "generated_token_ids",
    "generated_token_ids_sha256",
    "generated_text_utf8",
    "generated_text_sha256",
    "finish_reason",
    "utf8_valid",
    "threads_requested",
    "threads_effective",
    "n_ctx",
    "n_batch",
    "n_ubatch",
    "n_seq_max",
    "worker_binary_sha256",
    "worker_instance_id",
    "pid",
    "t0_ns",
    "t1_ns",
    "duration_ns",
    "error_code",
    "error_message",
]

MEASURED_KEYS = [
    "schema_family",
    "schema_version",
    "record_class",
    "campaign_id",
    "campaign_manifest_sha256",
    "protocol_document_sha256",
    "fixtures_sha256",
    "source_commit",
    "cell_id",
    "implementation",
    "pair_index",
    "arm_order",
    "order_position",
    "measurement_class",
    "fixture_id",
    "max_new_tokens",
    "success",
    "prompt_token_count_expected",
    "prompt_token_count_actual",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "generated_token_count",
    "generated_token_ids",
    "generated_token_ids_sha256",
    "generated_text_utf8",
    "generated_text_sha256",
    "finish_reason",
    "utf8_valid",
    "threads_requested",
    "threads_effective",
    "n_ctx",
    "n_batch",
    "n_ubatch",
    "n_seq_max",
    "worker_binary_sha256",
    "worker_instance_id",
    "pid",
    "t0_ns",
    "t1_ns",
    "duration_ns",
    "error_code",
    "error_message",
    "exclusion_status",
    "exclusion_reason",
    "publication_eligible",
]


class AnalysisError(Exception):
    def __init__(self, codes: list[str], message: str) -> None:
        super().__init__(message)
        self.codes = unique_sorted(codes)
        self.message = message


def unique_sorted(codes: list[str]) -> list[str]:
    return sorted(set(codes))


def load_json_object(path: Path, max_bytes: int = MAX_TRUSTED_JSON_BYTES) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise AnalysisError(["record_schema_invalid"], "oversized json")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnalysisError(["utf8_invalid"], "json is not utf-8") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise AnalysisError(["record_schema_invalid"], f"duplicate key {key}")
            document[key] = value
        return document

    try:
        document = json.loads(text, object_pairs_hook=reject_duplicates)
    except AnalysisError:
        raise
    except json.JSONDecodeError as exc:
        raise AnalysisError(["record_schema_invalid"], "json decode failed") from exc
    if not isinstance(document, dict):
        raise AnalysisError(["record_schema_invalid"], "json is not an object")
    return document


def matrix_is_complete(cell_stats: dict[str, dict[str, Any]], codes: list[str]) -> bool:
    if codes:
        return False
    expected = set(EXPECTED_CELL_IDS)
    actual = set(cell_stats)
    if actual != expected:
        return False
    return all(cell_stats[cell_id].get("pair_count") == MEASURED_PAIRS for cell_id in EXPECTED_CELL_IDS)


def object_key_order(path: Path) -> list[str]:
    document = load_json_object(path)
    return list(document.keys())


def is_hex64(value: object) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def sibling_digest(kind: str, record: dict[str, Any]) -> str | None:
    if kind == "prompt_token_ids_sha256":
        return token_ids_sha256([int(x) for x in record["prompt_token_ids"]])
    if kind == "generated_token_ids_sha256":
        return token_ids_sha256([int(x) for x in record["generated_token_ids"]])
    if kind == "generated_text_sha256":
        return sha256_bytes(record["generated_text_utf8"].encode("utf-8"))
    return None


def classify_digest_field(value: object, expected: str | None) -> str | None:
    if not isinstance(value, str):
        return "record_schema_invalid"
    if HEX64.fullmatch(value) is None:
        return "record_schema_invalid"
    if expected is not None and value != expected:
        return "digest_mismatch"
    return None


def check_timing(record: dict[str, Any]) -> str | None:
    for key in ("t0_ns", "t1_ns", "duration_ns"):
        if type(record.get(key)) is not int:
            return "timing_invalid"
        if record[key] < 0:
            return "timing_invalid"
    if record["t1_ns"] < record["t0_ns"]:
        return "timing_invalid"
    if record["duration_ns"] != record["t1_ns"] - record["t0_ns"]:
        return "timing_invalid"
    return None


def interpolated_quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def bootstrap_mean_interval(differences: list[int]) -> tuple[float, float]:
    if len(differences) != 20:
        raise AnalysisError(["sample_count_invalid"], "bootstrap requires n=20")
    rng = random.Random(BOOTSTRAP_SEED)
    means: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        total = 0.0
        for _draw in range(20):
            total += float(differences[rng.randrange(20)])
        means.append(total / 20.0)
    means.sort()
    tail = (1.0 - 0.95) / 2.0
    return interpolated_quantile(means, tail), interpolated_quantile(means, 1.0 - tail)


def descriptive_stats(differences: list[int]) -> dict[str, int]:
    ordered = sorted(differences)
    median = ordered[9]
    absdevs = sorted(abs(value - median) for value in differences)
    return {
        "p25_difference_ns": ordered[4],
        "median_difference_ns": median,
        "p50_difference_ns": ordered[9],
        "p75_difference_ns": ordered[14],
        "p95_difference_ns": ordered[18],
        "iqr_difference_ns": ordered[14] - ordered[4],
        "mad_difference_ns": absdevs[9],
    }


def claim_from_c01(ie: list[int], ref: list[int]) -> dict[str, Any]:
    if len(ie) != len(ref):
        raise AnalysisError(["sample_count_invalid"], "paired lengths differ")
    diffs = [left - right for left, right in zip(ie, ref)]
    sum_d = sum(diffs)
    sum_ref = sum(ref)
    ci_low, ci_high = bootstrap_mean_interval(diffs)
    if sum_d > 0:
        direction = "inferedge_slower"
        absolute = sum_d >= 40_000_000
        relative = 20 * sum_d >= sum_ref
    elif sum_d < 0:
        direction = "inferedge_faster"
        absolute = -sum_d >= 40_000_000
        relative = 20 * (-sum_d) >= sum_ref
    else:
        direction = "none"
        absolute = False
        relative = False
    ci_ok = ci_low > 0 if direction == "inferedge_slower" else ci_high < 0 if direction == "inferedge_faster" else False
    if ci_low == 0 or ci_high == 0:
        ci_ok = False
    slower = direction == "inferedge_slower" and absolute and relative and ci_ok
    faster = direction == "inferedge_faster" and absolute and relative and ci_ok
    return {
        "sum_d": sum_d,
        "sum_ref": sum_ref,
        "mean_difference_ns": sum(diffs) / 20,
        "reference_mean_ns": sum_ref / 20,
        "ci_low_ns": ci_low,
        "ci_high_ns": ci_high,
        "direction": direction,
        "absolute_threshold_met": absolute,
        "relative_threshold_met": relative,
        "ci_excludes_zero_strict": ci_ok,
        "slower_claim": slower,
        "faster_claim": faster,
        "winner_declared": slower or faster,
        "stats": descriptive_stats(diffs),
    }


def walk_regular_files(root: Path) -> list[str]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            found.append(rel)
    return found


def validate_paths_and_inventory(root: Path, expected: list[str], analysis_must_absent: bool) -> list[str]:
    codes: list[str] = []
    root_code = require_directory(root)
    if root_code:
        return [root_code]
    if analysis_must_absent and (root / "analysis").exists():
        codes.append("campaign_incomplete")
    seen_logical: dict[str, int] = {}
    present = walk_regular_files(root)
    expected_set = set(expected)
    for rel in present:
        path_code = reject_relative_path(rel)
        if path_code:
            codes.append(path_code)
            continue
        full = root / rel
        name = Path(rel).name
        if TEMP_NAME_RE.fullmatch(name):
            file_code = require_regular_file(full)
            if file_code:
                codes.append(file_code)
            continue
        if rel not in expected_set:
            codes.append("extra_file")
            continue
        file_code = require_regular_file(full)
        if file_code:
            codes.append(file_code)
        if rel.startswith("measured/"):
            logical = "/".join(Path(rel).parts[:3])
            seen_logical[logical] = seen_logical.get(logical, 0) + 1
        if rel.startswith("warmup/"):
            logical = rel
            seen_logical[logical] = seen_logical.get(logical, 0) + 1
    for rel in expected:
        if rel not in present:
            codes.append("missing_file")
    if any(count > 2 and key.startswith("measured/") for key, count in seen_logical.items()):
        codes.append("duplicate_record")
    return unique_sorted(codes)


def finish_allowed(cell_id: str, finish: str) -> bool:
    return finish in CELL_SPECS[cell_id]["finish"]


def validate_record_schema(record: dict[str, Any], keys: list[str], record_class: str) -> list[str]:
    codes: list[str] = []
    if list(record.keys()) != keys:
        codes.append("record_schema_invalid")
        return codes
    if record.get("schema_family") != SCHEMA_FAMILY or record.get("schema_version") != SCHEMA_VERSION:
        codes.append("record_schema_invalid")
    if record.get("record_class") != record_class:
        codes.append("record_schema_invalid")
    if not is_hex64(record.get("campaign_manifest_sha256")):
        codes.append("record_schema_invalid")
    if not is_hex64(record.get("protocol_document_sha256")):
        codes.append("record_schema_invalid")
    if not is_hex64(record.get("fixtures_sha256")):
        codes.append("record_schema_invalid")
    if not isinstance(record.get("source_commit"), str) or HEX40.fullmatch(record["source_commit"]) is None:
        codes.append("record_schema_invalid")
    return unique_sorted(codes)


def validate_warmup_record(record: dict[str, Any]) -> list[str]:
    codes = validate_record_schema(record, WARMUP_KEYS, "warmup")
    timing = check_timing(record)
    if timing:
        codes.append(timing)
    if record.get("warmup_index") not in (0, 1):
        codes.append("warmup_invalid")
    if record.get("success") is not True:
        codes.append("warmup_invalid")
    for field in ("prompt_token_ids_sha256", "generated_token_ids_sha256", "generated_text_sha256"):
        digest_code = classify_digest_field(record.get(field), sibling_digest(field, record) if "prompt_token_ids" in record else None)
        if digest_code:
            codes.append(digest_code)
    return unique_sorted(codes)


def validate_measured_record(record: dict[str, Any], trial: dict[str, Any] | None = None) -> list[str]:
    codes = validate_record_schema(record, MEASURED_KEYS, "measured")
    timing = check_timing(record)
    if timing:
        codes.append(timing)
    cell_id = record.get("cell_id")
    if cell_id not in CELL_SPECS:
        codes.append("matrix_invalid")
    elif not finish_allowed(cell_id, record.get("finish_reason", "")):
        codes.append("finish_reason_invalid")
    if trial is not None:
        if (
            record.get("pair_index") != trial["pair_index"]
            or record.get("arm_order") != trial["arm_order"]
            or record.get("order_position") != trial["order_position"]
            or record.get("implementation") != trial["implementation"]
        ):
            codes.append("pair_order_invalid")
    for field in ("prompt_token_ids_sha256", "generated_token_ids_sha256", "generated_text_sha256"):
        if field in record and field.replace("_sha256", "") in record:
            digest_code = classify_digest_field(record.get(field), sibling_digest(field, record))
            if digest_code:
                codes.append(digest_code)
    return unique_sorted(codes)


def compare_arms(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    if left.get("prompt_token_ids") != right.get("prompt_token_ids") or left.get(
        "generated_token_ids"
    ) != right.get("generated_token_ids"):
        codes.append("token_mismatch")
    if left.get("generated_text_utf8") != right.get("generated_text_utf8") or left.get(
        "generated_text_sha256"
    ) != right.get("generated_text_sha256"):
        codes.append("text_mismatch")
    if left.get("finish_reason") != right.get("finish_reason"):
        if "token_mismatch" not in codes and "text_mismatch" not in codes:
            codes.append("correctness_mismatch")
    if (
        left.get("prompt_token_count_actual") != right.get("prompt_token_count_actual")
        or left.get("utf8_valid") != right.get("utf8_valid")
    ) and not codes:
        codes.append("correctness_mismatch")
    return unique_sorted(codes)


def analyze_inventory_a(root: Path) -> list[str]:
    return validate_paths_and_inventory(root, inventory_a_relative_paths(), True)


def analyze_inventory_b(root: Path) -> list[str]:
    return validate_paths_and_inventory(root, inventory_b_relative_paths(), False)


def reject_existing_analysis(root: Path) -> list[str]:
    if (root / "analysis").exists():
        return ["campaign_incomplete"]
    return []


def build_aggregate(campaign_id: str, identities: dict[str, str], cell_stats: dict[str, dict], c01: dict[str, Any], codes: list[str]) -> dict[str, Any]:
    if identities.get("campaign_class") == "dry_run" and "campaign_class_invalid" not in codes:
        codes = list(codes) + ["campaign_class_invalid"]
    if set(cell_stats) != set(EXPECTED_CELL_IDS) or any(
        cell_stats.get(cell_id, {}).get("pair_count") != MEASURED_PAIRS for cell_id in EXPECTED_CELL_IDS
    ):
        if "matrix_invalid" not in codes and "sample_count_invalid" not in codes:
            codes = list(codes) + ["sample_count_invalid"]
    eligible = matrix_is_complete(cell_stats, codes)
    cells = []
    for cell_id in GOLDEN_CELL_ORDER:
        spec = CELL_SPECS[cell_id]
        stats = cell_stats.get(cell_id, {"pair_count": 0})
        cells.append(
            {
                "cell_id": cell_id,
                "measurement_class": spec["measurement_class"],
                "role": spec["role"],
                "pair_count": stats.get("pair_count", 0),
                "inferedge_mean_ns": stats.get("inferedge_mean_ns"),
                "reference_mean_ns": stats.get("reference_mean_ns"),
                "mean_difference_ns": stats.get("mean_difference_ns"),
                "median_difference_ns": stats.get("median_difference_ns"),
                "p50_difference_ns": stats.get("p50_difference_ns"),
                "p95_difference_ns": stats.get("p95_difference_ns"),
                "iqr_difference_ns": stats.get("iqr_difference_ns"),
                "mad_difference_ns": stats.get("mad_difference_ns"),
            }
        )
    winner = bool(eligible and c01.get("winner_declared"))
    return {
        "schema_family": SCHEMA_FAMILY,
        "schema_version": SCHEMA_VERSION,
        "record_class": "aggregate",
        "campaign_id": campaign_id,
        "campaign_class": identities.get("campaign_class", "campaign"),
        "campaign_manifest_sha256": identities["campaign_manifest_sha256"],
        "protocol_document_sha256": identities["protocol_document_sha256"],
        "fixtures_sha256": identities["fixtures_sha256"],
        "source_commit": identities["source_commit"],
        "eligibility_recomputed": True,
        "complete_matrix": eligible,
        "rejection_codes": unique_sorted(codes),
        "cells": cells,
        "c01": {
            "pair_count": c01.get("pair_count", 0),
            "sum_d": c01.get("sum_d"),
            "sum_ref": c01.get("sum_ref"),
            "mean_difference_ns": c01.get("mean_difference_ns"),
            "reference_mean_ns": c01.get("reference_mean_ns"),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci_low_ns": c01.get("ci_low_ns"),
            "ci_high_ns": c01.get("ci_high_ns"),
            "absolute_threshold_ns": ABSOLUTE_THRESHOLD_NS,
            "relative_threshold_numerator": RELATIVE_THRESHOLD_NUMERATOR,
            "relative_threshold_denominator": RELATIVE_THRESHOLD_DENOMINATOR,
            "relative_threshold_ns": None if c01.get("sum_ref") is None else c01["sum_ref"] / 400,
            "direction": c01.get("direction", "none"),
            "absolute_threshold_met": bool(c01.get("absolute_threshold_met")),
            "relative_threshold_met": bool(c01.get("relative_threshold_met")),
            "ci_excludes_zero_strict": bool(c01.get("ci_excludes_zero_strict")),
            "slower_claim": bool(eligible and c01.get("slower_claim")),
            "faster_claim": bool(eligible and c01.get("faster_claim")),
        },
        "winner_declared": winner,
        "superiority_claimed": winner,
    }


def build_summary(aggregate: dict[str, Any], aggregate_bytes: bytes) -> dict[str, Any]:
    c01 = aggregate["c01"]
    eligible = aggregate["complete_matrix"] and not aggregate["rejection_codes"]
    if not eligible:
        status = "no_winner_ineligible"
    elif c01["slower_claim"]:
        status = "inferedge_slower"
    elif c01["faster_claim"]:
        status = "inferedge_faster"
    else:
        status = "no_winner_below_threshold_or_ci"
    return {
        "schema_family": SCHEMA_FAMILY,
        "schema_version": SCHEMA_VERSION,
        "record_class": "summary",
        "campaign_id": aggregate["campaign_id"],
        "campaign_manifest_sha256": aggregate["campaign_manifest_sha256"],
        "protocol_document_sha256": aggregate["protocol_document_sha256"],
        "fixtures_sha256": aggregate["fixtures_sha256"],
        "source_commit": aggregate["source_commit"],
        "aggregate_sha256": sha256_bytes(aggregate_bytes),
        "eligibility_recomputed": True,
        "complete_matrix": aggregate["complete_matrix"],
        "c01_mean_difference_ns": None if not eligible else c01["mean_difference_ns"],
        "c01_ci_low_ns": None if not eligible else c01["ci_low_ns"],
        "c01_ci_high_ns": None if not eligible else c01["ci_high_ns"],
        "c01_absolute_threshold_met": bool(c01["absolute_threshold_met"]),
        "c01_relative_threshold_met": bool(c01["relative_threshold_met"]),
        "winner_declared": aggregate["winner_declared"],
        "superiority_claimed": aggregate["superiority_claimed"],
        "claim_status": status,
        "secondary_cells_statement": "secondary cells are descriptive only",
        "l01_statement": "L01 is descriptive filesystem-cache-warm package/session load versus minimal direct load",
        "rejection_codes": list(aggregate["rejection_codes"]),
    }


def run_analyze(root: Path) -> dict[str, Any]:
    codes = reject_existing_analysis(root)
    if codes:
        raise AnalysisError(codes, "analysis directory exists")
    codes.extend(analyze_inventory_a(root))
    identities = {
        "campaign_class": "campaign",
        "campaign_id": "unknown",
        "campaign_manifest_sha256": "0" * 64,
        "protocol_document_sha256": "0" * 64,
        "fixtures_sha256": "0" * 64,
        "source_commit": "0" * 40,
    }
    cell_stats: dict[str, dict] = {}
    c01: dict[str, Any] = {"pair_count": 0}
    manifest_path = root / "campaign_manifest.json"
    if manifest_path.is_file() and "missing_file" not in codes:
        manifest = load_json_object(manifest_path)
        identities["campaign_class"] = manifest.get("campaign_class", "campaign")
        identities["campaign_id"] = manifest.get("campaign_id", "unknown")
        identities["campaign_manifest_sha256"] = sha256_bytes(manifest_path.read_bytes())
        identities["protocol_document_sha256"] = str(manifest.get("protocol_document_sha256", "0" * 64))
        identities["fixtures_sha256"] = str(manifest.get("fixtures_sha256", "0" * 64))
        identities["source_commit"] = str(manifest.get("source_commit", "0" * 40))
        if identities["campaign_class"] == "dry_run":
            codes.append("campaign_class_invalid")
        digest_path = root / "campaign_manifest.sha256"
        if digest_path.is_file():
            listed = digest_path.read_text(encoding="ascii").strip()
            if listed != identities["campaign_manifest_sha256"]:
                codes.append("digest_mismatch")
        trials = expanded_trials()
        ie_c01: list[int] = []
        ref_c01: list[int] = []
        for trial in trials:
            rel = (
                f"measured/{trial['cell_id']}/pair-{trial['pair_index']:02d}/"
                f"{trial['implementation']}.json"
            )
            path = root / rel
            if not path.is_file():
                continue
            record = load_json_object(path)
            codes.extend(validate_measured_record(record, trial))
            cell_id = trial["cell_id"]
            cell_stats.setdefault(cell_id, {"pair_count": 0, "ie": [], "ref": []})
            if record.get("success") is True and trial["implementation"] == "inferedge":
                cell_stats[cell_id].setdefault("ie", []).append(int(record["duration_ns"]))
            if record.get("success") is True and trial["implementation"] == "reference":
                cell_stats[cell_id].setdefault("ref", []).append(int(record["duration_ns"]))
        for cell_id in EXPECTED_CELL_IDS:
            stats = cell_stats.setdefault(cell_id, {"ie": [], "ref": []})
            ie_samples = stats.get("ie", [])
            ref_samples = stats.get("ref", [])
            paired = min(len(ie_samples), len(ref_samples))
            stats["pair_count"] = paired
            if paired == MEASURED_PAIRS:
                diffs = [left - right for left, right in zip(ie_samples, ref_samples)]
                ordered = descriptive_stats(diffs)
                stats.update(
                    {
                        "inferedge_mean_ns": sum(ie_samples) / MEASURED_PAIRS,
                        "reference_mean_ns": sum(ref_samples) / MEASURED_PAIRS,
                        "mean_difference_ns": sum(diffs) / MEASURED_PAIRS,
                        **ordered,
                    }
                )
            else:
                codes.append("sample_count_invalid")
        ie_c01 = cell_stats.get("C01", {}).get("ie", [])
        ref_c01 = cell_stats.get("C01", {}).get("ref", [])
        if len(ie_c01) == MEASURED_PAIRS and len(ref_c01) == MEASURED_PAIRS:
            c01 = claim_from_c01(ie_c01, ref_c01)
            c01["pair_count"] = MEASURED_PAIRS
        else:
            c01 = {"pair_count": min(len(ie_c01), len(ref_c01))}
            codes.append("sample_count_invalid")
        for stats in cell_stats.values():
            stats.pop("ie", None)
            stats.pop("ref", None)
    if codes:
        codes = unique_sorted(codes)
    os.mkdir(root / "analysis")
    aggregate = build_aggregate(identities.get("campaign_id", "unknown"), identities, cell_stats, c01, codes)
    payload = canonical_json_bytes(aggregate)
    publish_exclusive(root / "analysis" / "aggregate.json", payload)
    summary = build_summary(aggregate, payload)
    publish_exclusive(root / "analysis" / "summary.json", canonical_json_bytes(summary))
    return {"aggregate": aggregate, "summary": summary, "rejection_codes": unique_sorted(codes)}


def verify_readonly(root: Path) -> list[str]:
    return analyze_inventory_b(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze_generative")
    parser.add_argument("--mode", choices=("analyze", "verify"), required=False)
    parser.add_argument("--evidence-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode is None:
        parser.print_help()
        return 0
    if args.evidence_root is None:
        raise AnalysisError(["campaign_incomplete"], "analyze/verify require --evidence-root")
    root = Path(args.evidence_root)
    if args.mode == "analyze":
        run_analyze(root)
        return 0
    codes = verify_readonly(root)
    if codes:
        raise AnalysisError(codes, "final verification failed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AnalysisError as exc:
        print(",".join(exc.codes) + ": " + exc.message, file=sys.stderr)
        sys.exit(1)
