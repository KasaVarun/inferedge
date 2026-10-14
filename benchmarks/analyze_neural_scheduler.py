#!/usr/bin/env python3
"""Analyze one neural_scheduler_batching run set.

Confirmatory contrasts are allowlisted: gemm_f32 C minus A, and the frozen
A versus G2 study (G2 minus A). This module never declares a winner. It
refuses Add or overhead experiment identities.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.analyze_scheduler import (
    paired_block_bootstrap_mean,
    paired_metric_effect,
)
from benchmarks.collect_baseline import ONNXRUNTIME_VERSION, CollectorError
from benchmarks.collect_neural_scheduler import (
    AG2_OFFERED_RATE,
    ANALYSIS_PROTOCOL_VERSION,
    ARRIVAL_FIDELITY_STATUS_NOT_CERTIFIED,
    ARRIVAL_SCREEN_MAX_LAG_NS,
    ARRIVAL_SCREEN_POLICY_VERSION,
    CONTRAST_ID_AC,
    CONTRAST_ID_AG2,
    CONTRAST_SIGN_CONVENTION,
    CONTRAST_SIGN_G2_MINUS_A,
    EXPERIMENT_KIND,
    PHASE_BOUNDARY_FIDELITY_UNAVAILABLE,
    PERSISTENT_BATCHING_MIN_MEAN,
    REFERENCE_CHECKING_POLICY,
    REPORTABLE_ALTERNATIVE,
    REPORTABLE_BASELINE,
    REPORTABLE_CELLS,
    REPORTABLE_CONTRAST_AG2,
    REPORTABLE_MODEL_ID,
    model_contract,
    normalize_cell_definition,
    rejection_total,
    validate_trial,
)
from benchmarks.collect_scheduler import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    MINIMUM_INTERVAL_BLOCKS,
    PREFERRED_COUNTED_RUNS,
    SCHEMA_VERSION,
    write_json,
)


class AnalyzerError(CollectorError):
    pass


# Exact (git_commit, benchmark_binary_sha256) pairs authorized for legacy
# pre-adaptive report revalidation. Cross-product matching is forbidden.
LEGACY_PRE_ADAPTIVE_IDENTITY_PAIRS = frozenset(
    {
        (
            "0a8d52b5a9f176a38b65a8b8a2028b87857a1b97",
            "acb0af726be4c564e3b829836723946916257d2f92ccce39e07acf50f70fc0d0",
        ),
    }
)

LEGACY_ADAPTIVE_IDENTITY_LIMITATIONS = (
    "adaptive_batching inferred false under pre-adaptive report contract",
    "Identity is supported by recorded collection metadata, not cryptographic attestation.",
)


def _load_json(path: pathlib.Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalyzerError(f"could not read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise AnalyzerError(f"{path} is not a JSON object")
    return document


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(report: dict, *path: str) -> float | None:
    cursor: object = report
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    if not _finite(cursor):
        return None
    return float(cursor)


def _mean_batch_size(report: dict) -> float | None:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return None
    histogram = metrics.get("batch_size_histogram")
    if not isinstance(histogram, list) or not histogram:
        return None
    runs = metrics.get("engine_run_count")
    entered = metrics.get("requests_entered_engine_runs")
    if type(runs) is not int or runs <= 0 or type(entered) is not int:
        return None
    return float(entered) / float(runs)


def _engine_per_success(report: dict) -> float | None:
    total = _number(report, "metrics", "total_engine_run_nanoseconds")
    succeeded = report.get("succeeded")
    if total is None or type(succeeded) is not int or succeeded <= 0:
        return None
    return total / float(succeeded)


PRIMARY_METRIC = "p50_e2e_latency"
METRIC_EXTRACTORS = (
    (PRIMARY_METRIC, "microseconds", lambda report: _number(report, "successful_e2e_latency", "p50_microseconds")),
    ("p95_e2e_latency", "microseconds", lambda report: _number(report, "successful_e2e_latency", "p95_microseconds")),
    (
        "successful_throughput",
        "successes_per_total_trial_second",
        lambda report: _number(report, "throughput", "successful_per_total_trial_second"),
    ),
    ("mean_realized_batch_size", "requests_per_engine_run", _mean_batch_size),
    (
        "total_engine_run_nanoseconds",
        "nanoseconds",
        lambda report: _number(report, "metrics", "total_engine_run_nanoseconds"),
    ),
    ("engine_run_nanoseconds_per_success", "nanoseconds_per_success_descriptive", _engine_per_success),
    (
        "generator_lag_minimum_nanoseconds",
        "nanoseconds",
        lambda report: _number(report, "generator_lag", "minimum_nanoseconds"),
    ),
    (
        "generator_lag_maximum_nanoseconds",
        "nanoseconds",
        lambda report: _number(report, "generator_lag", "maximum_nanoseconds"),
    ),
    (
        "generator_lag_mean_nanoseconds",
        "nanoseconds",
        lambda report: _number(report, "generator_lag", "mean_nanoseconds"),
    ),
)


FAIRNESS_SHARED = (
    "git_commit",
    "benchmark_binary_sha256",
    "model_id",
    "model_sha256",
    "weight_sidecar_sha256",
    "reference_checking_policy",
    "offered_rate",
    "profile_id",
    "input_width",
    "output_width",
    "operating_system",
    "cpu_architecture",
    "platform",
    "hardware_model",
    "onnxruntime_version",
    "compiler_version",
    "cmake_build_type",
    "schema_version",
    "analysis_protocol_version",
)


def _require_text(record: dict, key: str) -> str:
    value = record.get(key)
    if type(value) is not str or not value:
        raise AnalyzerError(f"{key} is missing")
    return value


def _effect_definition(sign: str) -> str:
    if sign == CONTRAST_SIGN_CONVENTION:
        return "difference = C_metric - A_metric"
    if sign == CONTRAST_SIGN_G2_MINUS_A:
        return "difference = G2_metric - A_metric"
    raise AnalyzerError(f"unsupported contrast sign convention {sign!r}")


def _cell_map_from_definitions(definitions: object, expected_cells: tuple[str, ...]) -> dict[str, dict]:
    if not isinstance(definitions, list) or not definitions:
        raise AnalyzerError("cell_definitions are missing")
    resolved: dict[str, dict] = {}
    for raw in definitions:
        if not isinstance(raw, dict):
            raise AnalyzerError("cell definition is not an object")
        try:
            cell = normalize_cell_definition(raw)
        except CollectorError as exc:
            raise AnalyzerError(str(exc)) from exc
        cell_id = cell["id"]
        if cell_id in resolved:
            raise AnalyzerError(f"duplicate cell definition for {cell_id}")
        resolved[cell_id] = cell
    for cell_id in expected_cells:
        if cell_id not in resolved:
            raise AnalyzerError(f"cell definition for {cell_id} is missing")
    unexpected = [cell_id for cell_id in resolved if cell_id not in expected_cells]
    if unexpected:
        raise AnalyzerError(f"unexpected cell definition(s): {', '.join(unexpected)}")
    if list(resolved.keys()) != list(expected_cells):
        # Presence is required; order of the definitions list may differ from cells.
        pass
    return resolved


def _require_cell_matches(cell: dict, expected: dict, label: str) -> None:
    actual = normalize_cell_definition(cell)
    want = normalize_cell_definition(expected)
    if actual != want:
        raise AnalyzerError(f"{label} cell definition does not match the allowlisted contrast")


def _resolve_confirmatory_contrast(protocol: dict) -> dict:
    cells = tuple(protocol.get("cells") or ())
    contrast_id = protocol.get("contrast_id")
    sign = protocol.get("contrast_sign_convention")
    if cells == REPORTABLE_CELLS:
        if contrast_id not in {None, CONTRAST_ID_AC}:
            raise AnalyzerError(f"unknown contrast_id {contrast_id!r} for A/C evidence")
        if sign != CONTRAST_SIGN_CONVENTION:
            raise AnalyzerError("contrast sign convention is not C_minus_A")
        if protocol.get("baseline_cell") != REPORTABLE_BASELINE:
            raise AnalyzerError("baseline cell must be A")
        if protocol.get("alternative_cell") != REPORTABLE_ALTERNATIVE:
            raise AnalyzerError("alternative cell must be C")
        if protocol.get("arrival_fidelity_status") is not None:
            raise AnalyzerError("A/C contrast must not carry an A/G2 arrival policy")
        if protocol.get("arrival_screen_max_lag_ns") is not None:
            raise AnalyzerError("A/C contrast must not carry an A/G2 arrival screen")
        if protocol.get("persistent_batching_cell") is not None:
            raise AnalyzerError("A/C contrast must not require G2 persistent batching")
        expected = {
            "A": {"id": "A", "enable_batching": False, "max_batch_size": 1, "max_batch_delay_ms": 0},
            "C": {"id": "C", "enable_batching": True, "max_batch_size": 4, "max_batch_delay_ms": 0},
        }
        cell_map = _cell_map_from_definitions(protocol.get("cell_definitions"), REPORTABLE_CELLS)
        for cell_id, want in expected.items():
            _require_cell_matches(cell_map[cell_id], want, cell_id)
        return {
            "contrast_id": CONTRAST_ID_AC,
            "cells": REPORTABLE_CELLS,
            "baseline_cell": REPORTABLE_BASELINE,
            "alternative_cell": REPORTABLE_ALTERNATIVE,
            "contrast_sign_convention": CONTRAST_SIGN_CONVENTION,
            "effect_definition": _effect_definition(CONTRAST_SIGN_CONVENTION),
            "required_counted_runs": None,
            "fixed_offered_rate": None,
            "arrival_screen_max_lag_ns": None,
            "persistent_batching_cell": None,
            "persistent_batching_min_mean": None,
            "cell_map": cell_map,
            "expected_cells": expected,
        }
    if cells == REPORTABLE_CONTRAST_AG2["cells"]:
        if contrast_id != CONTRAST_ID_AG2:
            raise AnalyzerError(
                "A/G2 confirmatory evidence requires contrast_id "
                f"{CONTRAST_ID_AG2!r}"
            )
        if sign != CONTRAST_SIGN_G2_MINUS_A:
            raise AnalyzerError("contrast sign convention is not G2_minus_A")
        if protocol.get("baseline_cell") != "A":
            raise AnalyzerError("baseline cell must be A")
        if protocol.get("alternative_cell") != "G2":
            raise AnalyzerError("alternative cell must be G2")
        if protocol.get("arrival_fidelity_status") != ARRIVAL_FIDELITY_STATUS_NOT_CERTIFIED:
            raise AnalyzerError("arrival_fidelity_status must be not_certified")
        if protocol.get("phase_boundary_fidelity") != PHASE_BOUNDARY_FIDELITY_UNAVAILABLE:
            raise AnalyzerError("phase_boundary_fidelity must be unavailable")
        if protocol.get("arrival_screen_max_lag_ns") != ARRIVAL_SCREEN_MAX_LAG_NS:
            raise AnalyzerError("arrival_screen_max_lag_ns must be 5000000")
        if protocol.get("arrival_screen_policy_version") != ARRIVAL_SCREEN_POLICY_VERSION:
            raise AnalyzerError("arrival_screen_policy_version mismatch")
        if protocol.get("persistent_batching_cell") != "G2":
            raise AnalyzerError("persistent_batching_cell must be G2")
        if protocol.get("persistent_batching_min_mean") != PERSISTENT_BATCHING_MIN_MEAN:
            raise AnalyzerError("persistent_batching_min_mean must be 2.0")
        offered = protocol.get("offered_rate")
        if not isinstance(offered, (int, float)) or float(offered) != AG2_OFFERED_RATE:
            raise AnalyzerError(f"A/G2 offered rate must be {AG2_OFFERED_RATE:g}")
        expected = REPORTABLE_CONTRAST_AG2["expected_cells"]
        cell_map = _cell_map_from_definitions(
            protocol.get("cell_definitions"), REPORTABLE_CONTRAST_AG2["cells"]
        )
        for cell_id, want in expected.items():
            _require_cell_matches(cell_map[cell_id], want, cell_id)
        return {
            "contrast_id": CONTRAST_ID_AG2,
            "cells": REPORTABLE_CONTRAST_AG2["cells"],
            "baseline_cell": "A",
            "alternative_cell": "G2",
            "contrast_sign_convention": CONTRAST_SIGN_G2_MINUS_A,
            "effect_definition": _effect_definition(CONTRAST_SIGN_G2_MINUS_A),
            "required_counted_runs": PREFERRED_COUNTED_RUNS,
            "fixed_offered_rate": AG2_OFFERED_RATE,
            "arrival_screen_max_lag_ns": ARRIVAL_SCREEN_MAX_LAG_NS,
            "persistent_batching_cell": "G2",
            "persistent_batching_min_mean": PERSISTENT_BATCHING_MIN_MEAN,
            "cell_map": cell_map,
            "expected_cells": expected,
        }
    if "G1" in cells:
        raise AnalyzerError("G1 is calibration-only and is never confirmatory")
    raise AnalyzerError(f"unknown confirmatory cell pair {cells!r}")


def _check_environment(environment: dict) -> tuple[dict, dict | None]:
    if environment.get("experiment_kind") != EXPERIMENT_KIND:
        raise AnalyzerError("refusing a run that is not neural_scheduler_batching")
    protocol = environment.get("collection_protocol")
    if not isinstance(protocol, dict):
        raise AnalyzerError("collection protocol is missing")
    if protocol.get("experiment_kind") != EXPERIMENT_KIND:
        raise AnalyzerError("collection protocol experiment_kind mismatch")
    if protocol.get("analysis_protocol_version") != ANALYSIS_PROTOCOL_VERSION:
        raise AnalyzerError("analysis protocol is not 2")
    if protocol.get("reference_checking_policy") != REFERENCE_CHECKING_POLICY:
        raise AnalyzerError("reference checking policy is not precomputed_reference_compare")
    if protocol.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise AnalyzerError("bootstrap seed is not 20261013")
    if protocol.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES:
        raise AnalyzerError("bootstrap replicates are not 10000")
    if protocol.get("confidence_level") != CONFIDENCE_LEVEL:
        raise AnalyzerError("confidence level is not 0.95")
    verified = environment.get("verified_configuration")
    if not isinstance(verified, dict):
        raise AnalyzerError("verified_configuration is missing")
    queried = environment.get("queried")
    if not isinstance(queried, dict):
        raise AnalyzerError("queried identity is missing")
    if verified.get("cmake_build_type") != "Release":
        raise AnalyzerError("the benchmark binary is not a Release build")
    if queried.get("onnxruntime_version") != ONNXRUNTIME_VERSION:
        raise AnalyzerError(
            f"onnxruntime_version is {queried.get('onnxruntime_version')!r}, "
            f"expected {ONNXRUNTIME_VERSION}"
        )
    model_id = protocol.get("model_id")
    if type(model_id) is not str:
        raise AnalyzerError("model_id is missing")
    contract = model_contract(model_id)
    if verified.get("model_sha256") != contract["model_sha256"]:
        raise AnalyzerError("ONNX hash does not match the pinned neural model")
    if verified.get("weight_sidecar_sha256") != contract["sidecar_sha256"]:
        raise AnalyzerError("sidecar hash does not match the pinned neural model")
    if verified.get("model_id") != model_id:
        raise AnalyzerError("verified model_id does not match the protocol")
    if verified.get("reference_checking_policy") != REFERENCE_CHECKING_POLICY:
        raise AnalyzerError("verified reference policy mismatch")
    if protocol.get("model_sha256") != contract["model_sha256"]:
        raise AnalyzerError("protocol model hash mismatch")
    if protocol.get("weight_sidecar_sha256") != contract["sidecar_sha256"]:
        raise AnalyzerError("protocol sidecar hash mismatch")
    evidence_class = protocol.get("evidence_class")
    mode = protocol.get("collection_mode")
    if evidence_class not in {"correctness", "calibration", "confirmatory"}:
        raise AnalyzerError("evidence class is missing or unsupported")
    if mode == "smoke" and evidence_class != "correctness":
        raise AnalyzerError("smoke evidence must stay classified as correctness")
    if mode == "calibration" and evidence_class != "calibration":
        raise AnalyzerError("calibration evidence class mismatch")
    if mode == "reportable" and evidence_class != "confirmatory":
        raise AnalyzerError("reportable evidence class mismatch")
    counted = protocol.get("counted_runs_per_cell")
    if type(counted) is not int or counted < 0:
        raise AnalyzerError("counted_runs_per_cell is missing")
    if evidence_class == "correctness" and counted != 0:
        raise AnalyzerError("a smoke pair cannot be promoted into counted blocks")
    contrast = None
    if evidence_class == "confirmatory":
        if model_id != REPORTABLE_MODEL_ID:
            raise AnalyzerError("confirmatory analysis supports only gemm_f32")
        if protocol.get("offered_rate_frozen") is not True:
            raise AnalyzerError("confirmatory analysis requires a frozen offered rate")
        if protocol.get("zero_admission_rejection_required") is not True:
            raise AnalyzerError("confirmatory analysis requires the zero-rejection gate")
        contrast = _resolve_confirmatory_contrast(protocol)
        if contrast["required_counted_runs"] is None:
            if counted < MINIMUM_INTERVAL_BLOCKS:
                raise AnalyzerError(
                    f"confirmatory analysis needs at least {MINIMUM_INTERVAL_BLOCKS} counted blocks"
                )
        elif counted != contrast["required_counted_runs"]:
            raise AnalyzerError(
                "A/G2 confirmatory analysis requires exactly "
                f"{contrast['required_counted_runs']} counted blocks"
            )
        source_policy = environment.get("source_defined_policy")
        if isinstance(source_policy, dict):
            if source_policy.get("contrast_sign_convention") != contrast["contrast_sign_convention"]:
                raise AnalyzerError("source_defined_policy contrast sign mismatch")
            if contrast["contrast_id"] == CONTRAST_ID_AG2:
                if source_policy.get("contrast_id") != CONTRAST_ID_AG2:
                    raise AnalyzerError("source_defined_policy contrast_id mismatch")
                if source_policy.get("arrival_screen_max_lag_ns") != ARRIVAL_SCREEN_MAX_LAG_NS:
                    raise AnalyzerError("source_defined_policy arrival screen mismatch")
    elif protocol.get("contrast_id") == CONTRAST_ID_AG2:
        raise AnalyzerError("A/G2 contrast_id is confirmatory-only")
    return protocol, contrast


def _fairness_record(environment: dict, meta: dict, report: dict) -> dict:
    protocol = environment["collection_protocol"]
    queried = environment["queried"]
    verified = environment["verified_configuration"]
    return {
        "git_commit": queried["git_commit"],
        "benchmark_binary_sha256": queried["benchmark_binary_sha256"],
        "model_id": protocol["model_id"],
        "model_sha256": verified["model_sha256"],
        "weight_sidecar_sha256": verified["weight_sidecar_sha256"],
        "reference_checking_policy": verified["reference_checking_policy"],
        "offered_rate": protocol["offered_rate"],
        "profile_id": meta["profile_id"],
        "input_width": verified["input_width"],
        "output_width": verified["output_width"],
        "operating_system": queried.get("operating_system"),
        "cpu_architecture": queried.get("cpu_architecture"),
        "platform": queried.get("platform"),
        "hardware_model": queried.get("hardware_model"),
        "onnxruntime_version": queried.get("onnxruntime_version"),
        "compiler_version": queried.get("compiler_version"),
        "cmake_build_type": verified.get("cmake_build_type"),
        "schema_version": report.get("schema_version"),
        "analysis_protocol_version": protocol.get("analysis_protocol_version"),
        "enable_batching": report["scheduler"]["enable_batching"],
        "max_batch_size": report["scheduler"]["max_batch_size"],
        "max_batch_delay_ms": report["scheduler"]["max_batch_delay_ms"],
    }


def _check_plan(plan: list[dict], counted_runs: int, cells: list[str], profile_id: str) -> None:
    if not isinstance(plan, list) or not plan:
        raise AnalyzerError("execution plan is empty")
    pilots = [trial for trial in plan if trial.get("pilot") is True]
    counted = [trial for trial in plan if trial.get("pilot") is not True]
    if len(pilots) != len(cells):
        raise AnalyzerError("pilot count does not match the cell set")
    pilot_cells = [trial.get("cell_id") for trial in pilots]
    if pilot_cells != cells:
        raise AnalyzerError("pilot order does not match the cell order")
    if counted_runs == 0:
        if counted:
            raise AnalyzerError("smoke evidence must not contain counted trials")
        return
    if len(counted) != counted_runs * len(cells):
        raise AnalyzerError("counted trial count does not match the plan")
    for block_index in range(1, counted_runs + 1):
        rows = [trial for trial in counted if trial.get("block_index") == block_index]
        if len(rows) != len(cells):
            raise AnalyzerError(f"block {block_index} does not contain both arms")
        cell_ids = [trial.get("cell_id") for trial in rows]
        if len(set(cell_ids)) != len(cells):
            raise AnalyzerError(f"block {block_index} has a duplicate or missing arm")
        offset = block_index - 1
        expected = [cells[(position + offset) % len(cells)] for position in range(len(cells))]
        if cell_ids != expected:
            raise AnalyzerError(f"block {block_index} rotation is incorrect")
        for trial in rows:
            if trial.get("profile_id") != profile_id:
                raise AnalyzerError("profile_id changed inside the plan")
            if trial.get("pilot") is True:
                raise AnalyzerError("a counted trial is marked as pilot")


def _load_trials(run_set: pathlib.Path, plan: list[dict]) -> dict[str, dict]:
    found = {}
    for trial in plan:
        trial_id = str(trial["trial_id"])
        meta_path = run_set / "runs" / f"{trial_id}.meta.json"
        if not meta_path.is_file():
            raise AnalyzerError(f"{trial_id} metadata is missing")
        found[trial_id] = _load_json(meta_path)
    return found


def _resolve_adaptive_identity(
    *,
    queried: dict,
    loaded_reports: dict[str, dict],
    legacy_pre_adaptive_identity: bool,
) -> dict:
    """Return strict-new or authorized-legacy adaptive identity settings."""
    presence = []
    true_values = []
    for trial_id, document in loaded_reports.items():
        scheduler = document.get("scheduler")
        if not isinstance(scheduler, dict):
            raise AnalyzerError(f"{trial_id} scheduler object is missing")
        has_field = "adaptive_batching" in scheduler
        presence.append(has_field)
        if has_field and scheduler.get("adaptive_batching") is True:
            true_values.append(trial_id)
    if presence and any(presence) and not all(presence):
        raise AnalyzerError("mixed adaptive_batching field presence across trial reports")
    all_missing = bool(presence) and not any(presence)

    if not legacy_pre_adaptive_identity:
        if all_missing:
            raise AnalyzerError(
                "new-runner adaptive identity requires scheduler.adaptive_batching "
                "(pass --legacy-pre-adaptive-identity only for authorized pre-R1 evidence)"
            )
        return {
            "adaptive_identity_status": "strict_new",
            "require_adaptive_batching_field": True,
            "adaptive_batching": False,
            "limitations": [],
        }

    if true_values:
        raise AnalyzerError(
            "legacy adaptive identity forbids adaptive_batching true "
            f"in {', '.join(true_values)}"
        )
    if not all_missing:
        raise AnalyzerError(
            "legacy adaptive identity requires homogeneous missing scheduler.adaptive_batching"
        )
    pair = (queried.get("git_commit"), queried.get("benchmark_binary_sha256"))
    if pair not in LEGACY_PRE_ADAPTIVE_IDENTITY_PAIRS:
        raise AnalyzerError(
            "legacy adaptive identity pair is not allowlisted "
            f"(commit={pair[0]!r}, binary_sha256={pair[1]!r})"
        )
    return {
        "adaptive_identity_status": "legacy_authorized",
        "require_adaptive_batching_field": False,
        "adaptive_batching": False,
        "limitations": list(LEGACY_ADAPTIVE_IDENTITY_LIMITATIONS),
    }


def analyze(
    run_set: pathlib.Path, *, legacy_pre_adaptive_identity: bool = False
) -> pathlib.Path:
    run_set = pathlib.Path(run_set)
    aggregate_path = run_set / "aggregate.json"
    if aggregate_path.exists():
        raise AnalyzerError(f"refusing to overwrite existing evidence: {aggregate_path}")
    if (run_set / "failure.json").exists():
        raise AnalyzerError("refusing to analyze a failed partial campaign")
    environment = _load_json(run_set / "environment.json")
    if environment.get("status") != "completed":
        raise AnalyzerError("environment status is not completed")
    protocol, contrast = _check_environment(environment)
    plan_document = _load_json(run_set / "execution_plan.json")
    plan = plan_document.get("trials")
    if not isinstance(plan, list):
        raise AnalyzerError("execution plan is missing")
    cells = list(protocol["cells"])
    profile_id = protocol["profiles"][0] if protocol.get("profiles") else None
    if type(profile_id) is not str:
        raise AnalyzerError("traffic profile is missing")
    counted = int(protocol["counted_runs_per_cell"])
    _check_plan(plan, counted, cells, profile_id)
    if protocol.get("trial_order") != [str(trial["trial_id"]) for trial in plan]:
        raise AnalyzerError("trial order does not match the execution plan")
    metas = _load_trials(run_set, plan)
    queried = environment["queried"]
    verified = environment["verified_configuration"]
    for key in (
        "git_commit",
        "benchmark_binary_path",
        "benchmark_binary_sha256",
        "compiler_path",
        "compiler_version",
        "operating_system",
        "cpu_architecture",
    ):
        _require_text(queried, key)
    model = pathlib.Path(verified["model_path"])
    binary = pathlib.Path(queried["benchmark_binary_path"])
    if contrast is not None:
        cell_map = contrast["cell_map"]
        sign = contrast["contrast_sign_convention"]
        effect_definition = contrast["effect_definition"]
        contrast_id = contrast["contrast_id"]
        arrival_screen = contrast["arrival_screen_max_lag_ns"]
        persistent_cell = contrast["persistent_batching_cell"]
        persistent_min = contrast["persistent_batching_min_mean"]
    else:
        cell_map = _cell_map_from_definitions(protocol.get("cell_definitions"), tuple(cells))
        sign = protocol.get("contrast_sign_convention") or CONTRAST_SIGN_CONVENTION
        effect_definition = (
            _effect_definition(sign)
            if sign in {CONTRAST_SIGN_CONVENTION, CONTRAST_SIGN_G2_MINUS_A}
            else "difference = alternative_metric - baseline_metric"
        )
        contrast_id = protocol.get("contrast_id")
        arrival_screen = None
        persistent_cell = None
        persistent_min = None
    profile = protocol["profile_definitions"][0]
    loaded_texts: dict[str, str] = {}
    loaded_reports: dict[str, dict] = {}
    problems: list[str] = []
    for trial in plan:
        trial_id = str(trial["trial_id"])
        report_path = run_set / "runs" / f"{trial_id}.json"
        if not report_path.is_file():
            problems.append(f"{trial_id} report is missing")
            continue
        text = report_path.read_text(encoding="utf-8")
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            problems.append(f"{trial_id} report is not valid JSON: {exc}")
            continue
        if not isinstance(document, dict):
            problems.append(f"{trial_id} report is not a JSON object")
            continue
        loaded_texts[trial_id] = text
        loaded_reports[trial_id] = document
    if problems:
        raise AnalyzerError("; ".join(problems))
    identity = _resolve_adaptive_identity(
        queried=queried,
        loaded_reports=loaded_reports,
        legacy_pre_adaptive_identity=legacy_pre_adaptive_identity,
    )
    reports: dict[str, dict] = {}
    fairness: list[dict] = []
    for trial in plan:
        trial_id = str(trial["trial_id"])
        meta = metas[trial_id]
        for field in ("cell_id", "profile_id", "pilot", "block_index", "position_in_block", "execution_index"):
            if meta.get(field) != trial.get(field):
                problems.append(f"{trial_id} {field} does not match the plan")
        if meta.get("experiment_kind") != EXPERIMENT_KIND:
            problems.append(f"{trial_id} experiment_kind mismatch")
        if meta.get("model_id") != protocol["model_id"]:
            problems.append(f"{trial_id} model_id mismatch")
        if meta.get("benchmark_binary_sha256") != queried["benchmark_binary_sha256"]:
            problems.append(f"{trial_id} benchmark_binary_sha256 differs")
        if meta.get("model_sha256") != verified["model_sha256"]:
            problems.append(f"{trial_id} model_sha256 differs")
        if meta.get("weight_sidecar_sha256") != verified["weight_sidecar_sha256"]:
            problems.append(f"{trial_id} weight_sidecar_sha256 differs")
        if meta.get("reference_checking_policy") != REFERENCE_CHECKING_POLICY:
            problems.append(f"{trial_id} reference policy differs")
        if meta.get("git_commit") != queried["git_commit"]:
            problems.append(f"{trial_id} git_commit differs")
        if meta.get("offered_rate") != protocol["offered_rate"]:
            problems.append(f"{trial_id} offered_rate differs")
        if meta.get("collector_valid") is not True or meta.get("execution_timeout") is True:
            problems.append(f"{trial_id} is not a valid completed trial")
        if meta.get("command") != trial.get("command"):
            problems.append(f"{trial_id} executed command diverges from the execution plan")
        cell_id = str(trial["cell_id"])
        if cell_id not in cell_map:
            problems.append(f"{trial_id} unknown cell {cell_id}")
            continue
        cell = cell_map[cell_id]
        expected = {
            "model_path": str(model),
            "workload": profile["workload"],
            "worker_count": protocol["measurement"]["worker_count"],
            "queue_capacity": protocol["measurement"]["queue_capacity"],
            "enable_batching": cell["enable_batching"],
            "max_batch_size": cell["max_batch_size"],
            "max_batch_delay_ms": cell["max_batch_delay_ms"],
            "adaptive_batching": identity["adaptive_batching"],
            "require_adaptive_batching_field": identity["require_adaptive_batching_field"],
            "warmup_requests": protocol["measurement"]["warmup_requests"],
            "check_interval": protocol["measurement"]["check_interval"],
            "duration_ms": protocol["measurement"]["duration_ms"],
            "lag_tolerance_ms": protocol["measurement"]["lag_tolerance_ms"],
            "comparison_class": "primary",
            "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
            "target_rps": profile.get("target_rps"),
        }
        counted_trial = not bool(trial["pilot"])
        apply_arrival = arrival_screen is not None and counted_trial
        apply_persistent = (
            persistent_cell is not None and counted_trial and cell_id == persistent_cell
        )
        document, _status, arm_problems = validate_trial(
            loaded_texts[trial_id],
            expected=expected,
            mode=str(protocol["collection_mode"]),
            require_zero_rejections=bool(protocol.get("zero_admission_rejection_required")),
            arrival_screen_max_lag_ns=int(arrival_screen) if apply_arrival else None,
            require_persistent_batching=apply_persistent,
            persistent_batching_min_mean=(
                float(persistent_min) if apply_persistent else None
            ),
        )
        if arm_problems or document is None:
            problems.append(f"{trial_id} " + "; ".join(arm_problems or ["invalid report"]))
            continue
        if document.get("schema_version") != SCHEMA_VERSION:
            problems.append(f"{trial_id} schema_version is not 3")
        if rejection_total(document) != 0 and protocol.get("zero_admission_rejection_required"):
            problems.append(f"{trial_id} has scheduler admission rejections")
        scheduler = document.get("scheduler")
        if isinstance(scheduler, dict):
            reported = {
                "id": cell_id,
                "enable_batching": scheduler.get("enable_batching"),
                "max_batch_size": scheduler.get("max_batch_size"),
                "max_batch_delay_ms": scheduler.get("max_batch_delay_ms"),
            }
            if normalize_cell_definition(reported) != cell:
                problems.append(
                    f"{trial_id} runner scheduler configuration diverges from cell_definitions"
                )
        reports[trial_id] = document
        if not trial["pilot"]:
            fairness.append(_fairness_record(environment, meta, document))
    if problems:
        raise AnalyzerError("; ".join(problems))
    if fairness:
        first = fairness[0]
        for other in fairness[1:]:
            for key in FAIRNESS_SHARED:
                if first.get(key) != other.get(key):
                    raise AnalyzerError(f"fairness mismatch on {key}")
    evidence_class = protocol["evidence_class"]
    pairs = []
    if counted > 0:
        baseline = str(protocol["baseline_cell"])
        alternative = str(protocol["alternative_cell"])
        for block_index in range(1, counted + 1):
            rows = [
                trial
                for trial in plan
                if trial.get("block_index") == block_index and not trial.get("pilot")
            ]
            left = next(trial for trial in rows if trial["cell_id"] == baseline)
            right = next(trial for trial in rows if trial["cell_id"] == alternative)
            pairs.append(
                (
                    block_index,
                    reports[str(left["trial_id"])],
                    reports[str(right["trial_id"])],
                )
            )
    if contrast is not None and contrast["required_counted_runs"] is not None:
        allow_interval = (
            evidence_class == "confirmatory" and counted == contrast["required_counted_runs"]
        )
    else:
        allow_interval = evidence_class == "confirmatory" and counted >= MINIMUM_INTERVAL_BLOCKS
    metrics = []
    for name, units, extractor in METRIC_EXTRACTORS:
        if counted == 0:
            break
        baseline_values = []
        alternative_values = []
        indexes = []
        for block_index, baseline_report, alternative_report in pairs:
            left = extractor(baseline_report)
            right = extractor(alternative_report)
            if left is None or right is None:
                raise AnalyzerError(f"{name} is missing or nonfinite in block {block_index}")
            baseline_values.append(left)
            alternative_values.append(right)
            indexes.append(block_index)
        effect = paired_metric_effect(
            baseline_values,
            alternative_values,
            units=units,
            allow_interval=allow_interval,
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
            level=CONFIDENCE_LEVEL,
        )
        effect["metric"] = name
        effect["blocks"] = [
            {
                "block_index": indexes[index],
                "baseline": baseline_values[index],
                "alternative": alternative_values[index],
                "difference": alternative_values[index] - baseline_values[index],
            }
            for index in range(len(indexes))
        ]
        effect["effect_definition"] = effect_definition
        effect["winner_declared"] = False
        effect["superiority_claimed"] = False
        metrics.append(effect)
    check = None
    if any(metric["metric"] == PRIMARY_METRIC and metric["confidence_interval_computed"] for metric in metrics):
        latency = next(metric for metric in metrics if metric["metric"] == PRIMARY_METRIC)
        differences = [block["difference"] for block in latency["blocks"]]
        check = paired_block_bootstrap_mean(
            differences,
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
            level=CONFIDENCE_LEVEL,
        )
        if check != latency["confidence_interval"]:
            raise AnalyzerError("paired bootstrap did not reproduce the latency interval")
    baseline_cell = str(protocol.get("baseline_cell"))
    alternative_cell = str(protocol.get("alternative_cell"))
    histograms = []
    for block_index, baseline_report, alternative_report in pairs:
        histograms.append(
            {
                "block_index": block_index,
                baseline_cell: baseline_report["metrics"]["batch_size_histogram"],
                alternative_cell: alternative_report["metrics"]["batch_size_histogram"],
            }
        )
    limitations = [
        "Producer-observed latency includes open-loop observation delay.",
        "Reference preparation occurs before submit_start and can affect generator lag.",
        "The 1000 ms lag tolerance is not arrival-fidelity certification.",
        "Engine-run duration per success is an aggregate descriptive quantity.",
        "No automatic batching superiority claim is made.",
        "Results are specific to the recorded model, binary, host, and profile.",
    ]
    if contrast_id == CONTRAST_ID_AG2:
        limitations.extend(
            [
                "arrival_fidelity_status remains not_certified for the A/G2 study.",
                "phase_boundary_fidelity remains unavailable.",
                "The 5 ms coarse generator-lag screen is not precise arrival certification.",
                "Persistent batching gate uses mean realized batch size, not configured max.",
            ]
        )
    aggregate = {
        "experiment_kind": EXPERIMENT_KIND,
        "status": "completed",
        "evidence_class": evidence_class,
        "collection_mode": protocol["collection_mode"],
        "model_id": protocol["model_id"],
        "model_sha256": verified["model_sha256"],
        "weight_sidecar_sha256": verified["weight_sidecar_sha256"],
        "reference_checking_policy": REFERENCE_CHECKING_POLICY,
        "schema_version": SCHEMA_VERSION,
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "contrast_id": contrast_id,
        "contrast_sign_convention": sign,
        "baseline_cell": protocol.get("baseline_cell"),
        "alternative_cell": protocol.get("alternative_cell"),
        "primary_metric": PRIMARY_METRIC,
        "effect_definition": effect_definition,
        "winner_declared": False,
        "superiority_claimed": False,
        "experimental_unit": "paired_block",
        "pilots_excluded": True,
        "counted_runs_per_cell": counted,
        "preferred_counted_runs": PREFERRED_COUNTED_RUNS,
        "minimum_interval_blocks": MINIMUM_INTERVAL_BLOCKS,
        "offered_rate": protocol["offered_rate"],
        "offered_rate_frozen": protocol.get("offered_rate_frozen"),
        "traffic_profile": profile_id,
        "arrival_fidelity_status": protocol.get("arrival_fidelity_status"),
        "phase_boundary_fidelity": protocol.get("phase_boundary_fidelity"),
        "arrival_screen_max_lag_ns": protocol.get("arrival_screen_max_lag_ns"),
        "arrival_screen_policy_version": protocol.get("arrival_screen_policy_version"),
        "bootstrap": {
            "method": "paired_block_bootstrap_mean",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "level": CONFIDENCE_LEVEL,
            "minimum_blocks": MINIMUM_INTERVAL_BLOCKS,
            "computed": allow_interval,
        },
        "metrics": metrics,
        "batch_size_histograms": histograms,
        "bootstrap_reproduction": check,
        "limitations": limitations + list(identity["limitations"]),
        "adaptive_identity_status": identity["adaptive_identity_status"],
        "queried": {
            "git_commit": queried["git_commit"],
            "benchmark_binary_sha256": queried["benchmark_binary_sha256"],
            "compiler_path": queried["compiler_path"],
            "compiler_version": queried["compiler_version"],
            "cmake_build_type": verified["cmake_build_type"],
            "onnxruntime_version": queried["onnxruntime_version"],
            "operating_system": queried["operating_system"],
            "cpu_architecture": queried["cpu_architecture"],
            "platform": queried.get("platform"),
            "hardware_model": queried.get("hardware_model"),
        },
        "binary_path": str(binary),
        "model_path": str(model),
    }
    write_json(aggregate_path, aggregate)
    return aggregate_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze one neural scheduler run set.")
    parser.add_argument("run_set")
    parser.add_argument(
        "--legacy-pre-adaptive-identity",
        action="store_true",
        help=(
            "Authorize revalidation of pre-R1 reports that omit scheduler.adaptive_batching. "
            "Requires an allowlisted (git_commit, benchmark_binary_sha256) pair and "
            "homogeneous missing fields. Default analysis stays strict."
        ),
    )
    args = parser.parse_args(argv)
    try:
        analyze(
            pathlib.Path(args.run_set),
            legacy_pre_adaptive_identity=bool(args.legacy_pre_adaptive_identity),
        )
    except AnalyzerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
