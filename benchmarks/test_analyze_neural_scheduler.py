#!/usr/bin/env python3
"""Deterministic tests for neural scheduler analysis."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from benchmarks.analyze_neural_scheduler import (
    LEGACY_PRE_ADAPTIVE_IDENTITY_PAIRS,
    AnalyzerError,
    analyze,
)
from benchmarks.analyze_scheduler import (
    AnalyzerError as AddAnalyzerError,
)
from benchmarks.analyze_scheduler import analyze as analyze_add
from benchmarks.analyze_scheduler import paired_block_bootstrap_mean
from benchmarks.collect_neural_scheduler import (
    AG2_OFFERED_RATE,
    AG2G3_OFFERED_RATE,
    ANALYSIS_PROTOCOL_VERSION,
    ARRIVAL_FIDELITY_STATUS_NOT_CERTIFIED,
    ARRIVAL_SCREEN_MAX_LAG_NS,
    ARRIVAL_SCREEN_POLICY_VERSION,
    CONTRAST_ID_AG2,
    CONTRAST_ID_AG2G3,
    CONTRAST_SIGN_CONVENTION,
    CONTRAST_SIGN_G2_MINUS_A,
    CONTRAST_SIGN_G3_MINUS_A,
    CONTRAST_SIGN_G3_MINUS_G2,
    EXPERIMENT_KIND,
    NEURAL_MODELS,
    PHASE_BOUNDARY_FIDELITY_UNAVAILABLE,
    PERSISTENT_BATCHING_MIN_MEAN,
    REFERENCE_CHECKING_POLICY,
    THREE_ARM_POSITION_COUNTS_20,
    bind_commands,
    build_plan,
    open_loop_profile,
    resolve_neural_cell,
    serialize_cell_definitions,
)
from benchmarks.collect_scheduler import (
    BINARY_NAME,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CELL_BY_ID,
    CONFIDENCE_LEVEL,
    SCHEMA_VERSION,
)
from benchmarks.test_collect_scheduler import COMMIT, dump_report, sample_report

BINARY = pathlib.Path("/tmp") / BINARY_NAME
MODEL = pathlib.Path("/models/gemm_f32.onnx")
BINARY_SHA = "ab" * 32
GEMM = NEURAL_MODELS["gemm_f32"]
PROFILE = open_loop_profile("neural_open", 90.0)
AG2_PROFILE = open_loop_profile("neural_open", AG2_OFFERED_RATE)
AG2G3_PROFILE = open_loop_profile("neural_open", AG2G3_OFFERED_RATE)


def _cell(cell_id: str) -> dict:
    if cell_id == "G3":
        return resolve_neural_cell("G3", allow_g3=True)
    return resolve_neural_cell(cell_id)


def _report(
    cell_id: str,
    block: int,
    *,
    lag_max: int = 1000,
    mean_batch: float | None = None,
) -> dict:
    cell = _cell(cell_id)
    document = sample_report(model=str(MODEL))
    document["generator"]["warmup_requests"] = 100
    document["scheduler"]["enable_batching"] = cell["enable_batching"]
    document["scheduler"]["max_batch_size"] = cell["max_batch_size"]
    document["scheduler"]["max_batch_delay_ms"] = cell["max_batch_delay_ms"]
    document["scheduler"]["adaptive_batching"] = bool(cell.get("adaptive_batching", False))
    if cell_id == "G3":
        document["scheduler"]["adaptive_batching"] = True
    # Fixed offsets yield deterministic primary/secondary differences.
    offset = 0.0
    if cell_id in {"C", "G2"}:
        offset = 5.0
    elif cell_id == "G3":
        offset = 2.0
    document["successful_e2e_latency"]["p50_microseconds"] = 10.0 + block + offset
    document["successful_e2e_latency"]["p95_microseconds"] = 20.0 + block
    document["generator_lag"]["maximum_nanoseconds"] = lag_max
    if mean_batch is not None:
        runs = 50
        entered = int(round(mean_batch * runs))
        document["metrics"]["engine_run_count"] = runs
        document["metrics"]["requests_entered_engine_runs"] = entered
        histogram = [0] * 33
        if abs(mean_batch - 2.0) < 1e-12:
            histogram[2] = runs
        elif abs(mean_batch - 1.0) < 1e-12:
            histogram[1] = runs
        else:
            # Construct a valid sealed histogram near the requested mean.
            histogram[1] = max(0, 2 * runs - entered)
            histogram[2] = runs - histogram[1]
        document["metrics"]["batch_size_histogram"] = histogram
    return document


def _measurement() -> dict:
    return {
        "worker_count": 1,
        "queue_capacity": 64,
        "duration_ms": 1000,
        "max_offered": 100000,
        "max_outstanding": 64,
        "completion_timeout_ms": 5000,
        "warmup_requests": 100,
        "lag_tolerance_ms": 1000,
        "check_interval": 1,
    }


def _write_run(
    root: pathlib.Path,
    counted_blocks: int,
    *,
    mode: str | None = None,
    evidence_class: str | None = None,
    offered_rate_frozen: bool = True,
    model_id: str = "gemm_f32",
    contrast: str = "ac",
    run_name: str = "neural-run",
    lag_max: int = 1000,
    g2_mean_batch: float = 2.0,
    g3_mean_batch: float | None = None,
) -> pathlib.Path:
    run_set = root / run_name
    runs = run_set / "runs"
    runs.mkdir(parents=True)
    if contrast == "ac":
        cell_ids = ["A", "C"]
        profile = PROFILE
        sign = CONTRAST_SIGN_CONVENTION
        baseline = "A"
        alternative = "C"
        contrast_fields: dict = {}
        five_key = False
    elif contrast == "ag2":
        cell_ids = ["A", "G2"]
        profile = AG2_PROFILE
        sign = CONTRAST_SIGN_G2_MINUS_A
        baseline = "A"
        alternative = "G2"
        contrast_fields = {
            "contrast_id": CONTRAST_ID_AG2,
            "arrival_fidelity_status": ARRIVAL_FIDELITY_STATUS_NOT_CERTIFIED,
            "phase_boundary_fidelity": PHASE_BOUNDARY_FIDELITY_UNAVAILABLE,
            "arrival_screen_max_lag_ns": ARRIVAL_SCREEN_MAX_LAG_NS,
            "arrival_screen_policy_version": ARRIVAL_SCREEN_POLICY_VERSION,
            "persistent_batching_cell": "G2",
            "persistent_batching_min_mean": PERSISTENT_BATCHING_MIN_MEAN,
        }
        five_key = False
    elif contrast == "ag2g3":
        cell_ids = ["A", "G2", "G3"]
        profile = AG2G3_PROFILE
        sign = CONTRAST_SIGN_G3_MINUS_G2
        baseline = "G2"
        alternative = "G3"
        contrast_fields = {
            "contrast_id": CONTRAST_ID_AG2G3,
            "arrival_fidelity_status": ARRIVAL_FIDELITY_STATUS_NOT_CERTIFIED,
            "phase_boundary_fidelity": PHASE_BOUNDARY_FIDELITY_UNAVAILABLE,
            "arrival_screen_max_lag_ns": ARRIVAL_SCREEN_MAX_LAG_NS,
            "arrival_screen_policy_version": ARRIVAL_SCREEN_POLICY_VERSION,
            "persistent_batching_cell": "G2",
            "persistent_batching_min_mean": PERSISTENT_BATCHING_MIN_MEAN,
            "unbatched_cell": "A",
            "secondary_baseline_cell": "A",
            "secondary_alternative_cell": "G3",
            "secondary_contrast_sign_convention": CONTRAST_SIGN_G3_MINUS_A,
            "secondary_comparison_class": "exploratory",
            "secondary_multiple_comparisons_adjustment": "none_unadjusted",
            "require_five_key_cell_definitions": True,
            "legacy_adaptive_identity_prohibited": True,
            "rotation_position_counts": {
                cell_id: {str(position): count for position, count in sorted(positions.items())}
                for cell_id, positions in sorted(THREE_ARM_POSITION_COUNTS_20.items())
            },
        }
        five_key = True
    else:
        raise AssertionError(contrast)
    cells = [_cell(cell_id) for cell_id in cell_ids]
    profiles = [profile]
    cell_map = {cell["id"]: cell for cell in cells}
    plan = bind_commands(
        build_plan(cells, profiles, counted_blocks),
        binary=BINARY,
        model=MODEL,
        model_id=model_id,
        cells=cell_map,
        profiles={profile["id"]: profile},
        measurement=_measurement(),
    )
    if mode is None:
        mode = "smoke" if counted_blocks == 0 else "reportable"
    if evidence_class is None:
        evidence_class = {
            "smoke": "correctness",
            "calibration": "calibration",
            "reportable": "confirmatory",
        }[mode]
    contract = NEURAL_MODELS[model_id]
    for trial in plan:
        cell_id = str(trial["cell_id"])
        block = 0 if trial["block_index"] is None else int(trial["block_index"])
        mean_batch = None
        if not trial["pilot"]:
            if cell_id == "G2":
                mean_batch = g2_mean_batch
            elif cell_id == "G3" and g3_mean_batch is not None:
                mean_batch = g3_mean_batch
        report = _report(cell_id, block, lag_max=lag_max, mean_batch=mean_batch)
        trial_id = str(trial["trial_id"])
        (runs / f"{trial_id}.json").write_text(dump_report(report), encoding="utf-8")
        meta = {
            "trial_id": trial_id,
            "experiment_kind": EXPERIMENT_KIND,
            "model_id": model_id,
            "cell_id": cell_id,
            "profile_id": trial["profile_id"],
            "pilot": trial["pilot"],
            "run_number": trial.get("run_number"),
            "block_index": trial["block_index"],
            "position_in_block": trial["position_in_block"],
            "execution_index": trial["execution_index"],
            "collector_valid": True,
            "execution_timeout": False,
            "benchmark_binary_sha256": BINARY_SHA,
            "model_sha256": contract["model_sha256"],
            "weight_sidecar_sha256": contract["sidecar_sha256"],
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
            "git_commit": COMMIT,
            "offered_rate": profile["target_rps"],
            "command": trial["command"],
        }
        (runs / f"{trial_id}.meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    protocol = {
        "experiment_kind": EXPERIMENT_KIND,
        "evidence_class": evidence_class,
        "collection_mode": mode,
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "publication_policy": "strict_generator_delivery",
        "comparison_class": "primary" if mode == "reportable" else "exploratory",
        "model_id": model_id,
        "model_sha256": contract["model_sha256"],
        "weight_sidecar_sha256": contract["sidecar_sha256"],
        "weight_sidecar_path": "/models/fixtures/gemm_f32.weights",
        "input_width": contract["input_width"],
        "output_width": contract["output_width"],
        "reference_checking_policy": REFERENCE_CHECKING_POLICY,
        "contrast_sign_convention": sign,
        "calibration_status": "not_calibrated",
        "offered_rate_frozen": offered_rate_frozen,
        "offered_rate": profile["target_rps"],
        "traffic_profile": profile["id"],
        "cells": cell_ids,
        "profiles": [profile["id"]],
        "cell_definitions": serialize_cell_definitions(cells, include_adaptive=five_key),
        "profile_definitions": profiles,
        "measurement": _measurement(),
        "counted_runs_per_cell": counted_blocks,
        "counted_run_target": counted_blocks,
        "trial_order": [str(trial["trial_id"]) for trial in plan],
        "baseline_cell": baseline,
        "alternative_cell": alternative,
        "contrast_source": "predeclared",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": CONFIDENCE_LEVEL,
        "zero_admission_rejection_required": mode == "reportable",
        "require_strict_generator_delivery": mode in {"smoke", "reportable"},
        **contrast_fields,
    }
    environment = {
        "status": "completed",
        "experiment_kind": EXPERIMENT_KIND,
        "queried": {
            "git_commit": COMMIT,
            "benchmark_binary_path": str(BINARY),
            "benchmark_binary_sha256": BINARY_SHA,
            "compiler_path": "/usr/bin/c++",
            "compiler_version": "c++ synthetic\n",
            "cmake_flag_text": "-O3 -DNDEBUG",
            "onnxruntime_version": "1.31.0",
            "onnxruntime_resolved_library": "/tmp/libonnxruntime.dylib",
            "operating_system": "Darwin",
            "os_version": "24.0.0",
            "platform": "macOS-15.0-arm64-arm-64bit",
            "cpu_architecture": "arm64",
            "logical_cpu_count": 8,
            "hardware_model": "synthetic",
            "hardware_model_error": None,
            "hw_physicalcpu": 8,
            "ort_disable_telemetry": "1",
        },
        "verified_configuration": {
            "cmake_build_type": "Release",
            "model_path": str(MODEL),
            "model_sha256": contract["model_sha256"],
            "model_id": model_id,
            "weight_sidecar_path": "/models/fixtures/gemm_f32.weights",
            "weight_sidecar_sha256": contract["sidecar_sha256"],
            "input_width": contract["input_width"],
            "output_width": contract["output_width"],
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
        },
        "source_defined_policy": {
            "experiment_kind": EXPERIMENT_KIND,
            "contrast_sign_convention": sign,
            **contrast_fields,
        },
        "collection_protocol": protocol,
    }
    (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
    (run_set / "execution_plan.json").write_text(json.dumps({"trials": plan}) + "\n", encoding="utf-8")
    return run_set


def _mutate_report(run_set: pathlib.Path, trial_id: str, mutator) -> None:
    path = run_set / "runs" / f"{trial_id}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    mutator(document)
    path.write_text(dump_report(document), encoding="utf-8")


def _mutate_meta(run_set: pathlib.Path, trial_id: str, **fields) -> None:
    path = run_set / "runs" / f"{trial_id}.meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(fields)
    path.write_text(json.dumps(meta) + "\n", encoding="utf-8")


def _mutate_environment(run_set: pathlib.Path, mutator) -> None:
    path = run_set / "environment.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    mutator(document)
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")


class NeuralAnalyzeHappyPathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_set = _write_run(pathlib.Path(self.temporary.name), 10)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_paired_blocks_use_c_minus_a_and_fixed_seed(self) -> None:
        path = analyze(self.run_set)
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["experiment_kind"], EXPERIMENT_KIND)
        self.assertFalse(document["winner_declared"])
        self.assertFalse(document["superiority_claimed"])
        self.assertEqual(document["contrast_sign_convention"], "C_minus_A")
        self.assertEqual(document["schema_version"], SCHEMA_VERSION)
        latency = next(metric for metric in document["metrics"] if metric["metric"] == "p50_e2e_latency")
        self.assertEqual(latency["units"], "microseconds")
        self.assertAlmostEqual(latency["effect_estimate"], 5.0)
        self.assertTrue(all(block["difference"] == 5.0 for block in latency["blocks"]))
        self.assertTrue(latency["confidence_interval_computed"])
        differences = [5.0] * 10
        reproduced = paired_block_bootstrap_mean(
            differences,
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
            level=CONFIDENCE_LEVEL,
        )
        self.assertEqual(document["bootstrap_reproduction"], reproduced)
        self.assertEqual(document["bootstrap"]["seed"], 20261013)

    def test_pilot_is_excluded_and_rotation_is_checked(self) -> None:
        analyze(self.run_set)
        plan = json.loads((self.run_set / "execution_plan.json").read_text(encoding="utf-8"))["trials"]
        pilots = [trial for trial in plan if trial["pilot"]]
        self.assertEqual([trial["cell_id"] for trial in pilots], ["A", "C"])
        counted = [trial for trial in plan if not trial["pilot"]]
        self.assertEqual(counted[0]["cell_id"], "A")
        self.assertEqual(counted[1]["cell_id"], "C")
        self.assertEqual(counted[2]["cell_id"], "C")
        self.assertEqual(counted[3]["cell_id"], "A")


class NeuralAnalyzeRejectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_set = _write_run(pathlib.Path(self.temporary.name), 10)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_wrong_experiment_kind(self) -> None:
        _mutate_environment(self.run_set, lambda doc: doc.update(experiment_kind="batching"))
        with self.assertRaises(AnalyzerError):
            analyze(self.run_set)

    def test_wrong_model_hash(self) -> None:
        def mutate(doc: dict) -> None:
            doc["verified_configuration"]["model_sha256"] = "00" * 32
            doc["collection_protocol"]["model_sha256"] = "00" * 32

        _mutate_environment(self.run_set, mutate)
        with self.assertRaises(AnalyzerError):
            analyze(self.run_set)

    def test_wrong_sidecar_hash(self) -> None:
        def mutate(doc: dict) -> None:
            doc["verified_configuration"]["weight_sidecar_sha256"] = "11" * 32
            doc["collection_protocol"]["weight_sidecar_sha256"] = "11" * 32

        _mutate_environment(self.run_set, mutate)
        with self.assertRaises(AnalyzerError):
            analyze(self.run_set)

    def test_wrong_reference_policy(self) -> None:
        def mutate(doc: dict) -> None:
            doc["verified_configuration"]["reference_checking_policy"] = "post_observe"
            doc["collection_protocol"]["reference_checking_policy"] = "post_observe"

        _mutate_environment(self.run_set, mutate)
        with self.assertRaises(AnalyzerError):
            analyze(self.run_set)

    def test_mixed_binary_hash(self) -> None:
        _mutate_meta(self.run_set, "A-neural_open-01", benchmark_binary_sha256="cd" * 32)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("benchmark_binary_sha256", str(raised.exception))

    def test_mixed_offered_rate(self) -> None:
        _mutate_meta(self.run_set, "C-neural_open-01", offered_rate=123.0)
        with self.assertRaises(AnalyzerError):
            analyze(self.run_set)

    def test_generator_skip_is_rejected(self) -> None:
        def mutate(document: dict) -> None:
            document["submit_attempts"] = 99
            document["generator_skips_lag"] = 1
            document["accepted"] = 99
            document["succeeded"] = 99
            document["successful_e2e_latency"]["sample_count"] = 99
            document["metrics"]["engine_run_count"] = 99
            document["metrics"]["requests_entered_engine_runs"] = 99
            document["metrics"]["batch_size_histogram"] = [0, 99] + [0] * 31
            document["correctness_checked"] = 99
            document["generator_lag"]["count"] = 99

        _mutate_report(self.run_set, "A-neural_open-01", mutate)
        with self.assertRaises(AnalyzerError):
            analyze(self.run_set)

    def test_admission_rejection_is_rejected(self) -> None:
        def mutate(document: dict) -> None:
            document["rejections"]["full"] = 1
            document["accepted"] = 99
            document["succeeded"] = 99
            document["successful_e2e_latency"]["sample_count"] = 99
            document["metrics"]["engine_run_count"] = 99
            document["metrics"]["requests_entered_engine_runs"] = 99
            document["metrics"]["batch_size_histogram"] = [0, 99] + [0] * 31
            document["correctness_checked"] = 99

        _mutate_report(self.run_set, "C-neural_open-01", mutate)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("rejection", str(raised.exception).lower())

    def test_correctness_mismatch_is_rejected(self) -> None:
        _mutate_report(self.run_set, "A-neural_open-02", lambda doc: doc.update(correctness_mismatches=1))
        with self.assertRaises(AnalyzerError):
            analyze(self.run_set)

    def test_insufficient_blocks_and_unfrozen_rate(self) -> None:
        small = _write_run(pathlib.Path(self.temporary.name) / "small", 5)
        with self.assertRaises(AnalyzerError):
            analyze(small)
        unfrozen = _write_run(
            pathlib.Path(self.temporary.name) / "unfrozen",
            10,
            offered_rate_frozen=False,
        )
        with self.assertRaises(AnalyzerError):
            analyze(unfrozen)

    def test_partial_campaign_and_wrong_schema(self) -> None:
        (self.run_set / "failure.json").write_text(json.dumps({"status": "failed"}) + "\n", encoding="utf-8")
        with self.assertRaises(AnalyzerError):
            analyze(self.run_set)
        (self.run_set / "failure.json").unlink()
        _mutate_report(self.run_set, "A-neural_open-03", lambda doc: doc.update(schema_version=4))
        with self.assertRaises(AnalyzerError):
            analyze(self.run_set)

    def test_no_automatic_winner_and_no_request_pseudoreplication(self) -> None:
        document = json.loads(analyze(self.run_set).read_text(encoding="utf-8"))
        self.assertFalse(document["winner_declared"])
        self.assertEqual(document["experimental_unit"], "paired_block")
        self.assertEqual(document["bootstrap"]["method"], "paired_block_bootstrap_mean")
        latency = next(metric for metric in document["metrics"] if metric["metric"] == "p50_e2e_latency")
        self.assertEqual(latency["paired_block_count"], 10)


class AG2AnalyzeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_ag2_uses_g2_minus_a_and_no_winner(self) -> None:
        run_set = _write_run(self.root, 20, contrast="ag2", run_name="ag2-ok")
        document = json.loads(analyze(run_set).read_text(encoding="utf-8"))
        self.assertEqual(document["contrast_id"], CONTRAST_ID_AG2)
        self.assertEqual(document["contrast_sign_convention"], CONTRAST_SIGN_G2_MINUS_A)
        self.assertEqual(document["effect_definition"], "difference = G2_metric - A_metric")
        self.assertEqual(document["arrival_fidelity_status"], "not_certified")
        self.assertEqual(document["phase_boundary_fidelity"], "unavailable")
        self.assertEqual(document["arrival_screen_max_lag_ns"], ARRIVAL_SCREEN_MAX_LAG_NS)
        self.assertFalse(document["winner_declared"])
        self.assertFalse(document["superiority_claimed"])
        latency = next(metric for metric in document["metrics"] if metric["metric"] == "p50_e2e_latency")
        self.assertAlmostEqual(latency["effect_estimate"], 5.0)
        self.assertTrue(all(block["difference"] == 5.0 for block in latency["blocks"]))
        self.assertEqual(latency["paired_block_count"], 20)

    def test_wrong_contrast_id_rejected(self) -> None:
        run_set = _write_run(self.root, 20, contrast="ag2", run_name="bad-id")

        def mutate(doc: dict) -> None:
            doc["collection_protocol"]["contrast_id"] = "unknown_contrast"
            doc["source_defined_policy"]["contrast_id"] = "unknown_contrast"

        _mutate_environment(run_set, mutate)
        with self.assertRaises(AnalyzerError):
            analyze(run_set)

    def test_g1_never_confirmatory(self) -> None:
        run_set = _write_run(self.root, 20, contrast="ag2", run_name="g1-refuse")

        def mutate(doc: dict) -> None:
            doc["collection_protocol"]["cells"] = ["A", "G1"]
            doc["collection_protocol"]["alternative_cell"] = "G1"
            doc["collection_protocol"]["cell_definitions"] = [_cell("A"), _cell("G1")]

        _mutate_environment(run_set, mutate)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(run_set)
        self.assertIn("G1", str(raised.exception))

    def test_g2_wrong_delay_rejected(self) -> None:
        run_set = _write_run(self.root, 20, contrast="ag2", run_name="bad-delay")

        def mutate(doc: dict) -> None:
            for cell in doc["collection_protocol"]["cell_definitions"]:
                if cell["id"] == "G2":
                    cell["max_batch_delay_ms"] = 1

        _mutate_environment(run_set, mutate)
        with self.assertRaises(AnalyzerError):
            analyze(run_set)

    def test_g2_wrong_batch_cap_rejected(self) -> None:
        run_set = _write_run(self.root, 20, contrast="ag2", run_name="bad-cap")

        def mutate(doc: dict) -> None:
            for cell in doc["collection_protocol"]["cell_definitions"]:
                if cell["id"] == "G2":
                    cell["max_batch_size"] = 8

        _mutate_environment(run_set, mutate)
        with self.assertRaises(AnalyzerError):
            analyze(run_set)

    def test_wrong_offered_rate_rejected(self) -> None:
        run_set = _write_run(self.root, 20, contrast="ag2", run_name="bad-rate")

        def mutate(doc: dict) -> None:
            doc["collection_protocol"]["offered_rate"] = 500.0

        _mutate_environment(run_set, mutate)
        with self.assertRaises(AnalyzerError):
            analyze(run_set)

    def test_missing_or_altered_arrival_policy_rejected(self) -> None:
        missing = _write_run(self.root, 20, contrast="ag2", run_name="no-arrival")

        def drop(doc: dict) -> None:
            del doc["collection_protocol"]["arrival_fidelity_status"]
            del doc["source_defined_policy"]["arrival_fidelity_status"]

        _mutate_environment(missing, drop)
        with self.assertRaises(AnalyzerError):
            analyze(missing)
        altered = _write_run(self.root, 20, contrast="ag2", run_name="alt-arrival")

        def change(doc: dict) -> None:
            doc["collection_protocol"]["arrival_screen_max_lag_ns"] = 1_000_000
            doc["source_defined_policy"]["arrival_screen_max_lag_ns"] = 1_000_000

        _mutate_environment(altered, change)
        with self.assertRaises(AnalyzerError):
            analyze(altered)

    def test_lag_above_and_equal_screen(self) -> None:
        equal = _write_run(
            self.root,
            20,
            contrast="ag2",
            run_name="lag-eq",
            lag_max=ARRIVAL_SCREEN_MAX_LAG_NS,
        )
        analyze(equal)
        over = _write_run(
            self.root,
            20,
            contrast="ag2",
            run_name="lag-over",
            lag_max=ARRIVAL_SCREEN_MAX_LAG_NS + 1,
        )
        with self.assertRaises(AnalyzerError) as raised:
            analyze(over)
        self.assertIn("arrival screen", str(raised.exception))

    def test_g2_mean_batch_gate(self) -> None:
        ok = _write_run(
            self.root, 20, contrast="ag2", run_name="batch-ok", g2_mean_batch=2.0
        )
        analyze(ok)
        low = _write_run(
            self.root, 20, contrast="ag2", run_name="batch-low", g2_mean_batch=1.5
        )
        # Force mean < 2 on a counted G2 trial while keeping histogram accounting valid.
        def mutate(document: dict) -> None:
            document["metrics"]["engine_run_count"] = 80
            document["metrics"]["requests_entered_engine_runs"] = 100
            document["metrics"]["batch_size_histogram"] = [0, 60, 20] + [0] * 30

        _mutate_report(low, "G2-neural_open-01", mutate)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(low)
        self.assertIn("persistent-batching", str(raised.exception))

    def test_incomplete_ag2_campaign_rejected(self) -> None:
        run_set = _write_run(self.root, 10, contrast="ag2", run_name="partial-ag2")
        with self.assertRaises(AnalyzerError) as raised:
            analyze(run_set)
        self.assertIn("20", str(raised.exception))

    def test_wrong_block_order_and_duplicate_arm(self) -> None:
        run_set = _write_run(self.root, 20, contrast="ag2", run_name="bad-order")
        plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
        counted = [trial for trial in plan["trials"] if not trial["pilot"]]
        counted[0]["cell_id"], counted[1]["cell_id"] = counted[1]["cell_id"], counted[0]["cell_id"]
        (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")

        def sync(doc: dict) -> None:
            doc["collection_protocol"]["trial_order"] = [
                str(trial["trial_id"]) for trial in plan["trials"]
            ]

        _mutate_environment(run_set, sync)
        with self.assertRaises(AnalyzerError):
            analyze(run_set)

    def test_ac_backward_compatible(self) -> None:
        run_set = _write_run(self.root, 10, contrast="ac", run_name="ac-compat")
        document = json.loads(analyze(run_set).read_text(encoding="utf-8"))
        self.assertEqual(document["contrast_sign_convention"], "C_minus_A")
        self.assertEqual(document["effect_definition"], "difference = C_metric - A_metric")


class AG2G3AnalyzeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_20_block_primary_and_secondary(self) -> None:
        run_set = _write_run(self.root, 20, contrast="ag2g3", run_name="ag2g3-ok")
        document = json.loads(analyze(run_set).read_text(encoding="utf-8"))
        self.assertEqual(document["contrast_id"], CONTRAST_ID_AG2G3)
        self.assertEqual(document["contrast_sign_convention"], CONTRAST_SIGN_G3_MINUS_G2)
        self.assertEqual(document["effect_definition"], "difference = G3_metric - G2_metric")
        self.assertEqual(document["secondary_contrast_sign_convention"], CONTRAST_SIGN_G3_MINUS_A)
        self.assertEqual(document["experimental_unit"], "three_arm_paired_block")
        self.assertFalse(document["winner_declared"])
        self.assertFalse(document["superiority_claimed"])
        latency = next(metric for metric in document["metrics"] if metric["metric"] == "p50_e2e_latency")
        self.assertEqual(latency["contrast_role"], "primary")
        self.assertTrue(all(block["difference"] == -3.0 for block in latency["blocks"]))
        self.assertEqual(latency["paired_block_count"], 20)
        self.assertTrue(latency["confidence_interval_computed"])
        differences = [-3.0] * 20
        reproduced = paired_block_bootstrap_mean(
            differences,
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
            level=CONFIDENCE_LEVEL,
        )
        self.assertEqual(document["bootstrap_reproduction"], reproduced)
        secondary = document["secondary_metrics"][0]
        self.assertEqual(secondary["comparison_class"], "exploratory")
        self.assertIn("EXPLORATORY", secondary["interval_label"])
        self.assertIn("UNADJUSTED", secondary["interval_label"])
        self.assertTrue(all(block["difference"] == 2.0 for block in secondary["blocks"]))
        self.assertEqual(
            document["rotation_position_counts"]["A"],
            {"0": 7, "1": 6, "2": 7},
        )

    def test_unknown_and_missing_g3_rejected(self) -> None:
        run_set = _write_run(self.root, 20, contrast="ag2g3", run_name="bad-id")

        def bad_id(doc: dict) -> None:
            doc["collection_protocol"]["contrast_id"] = "unknown"
            doc["source_defined_policy"]["contrast_id"] = "unknown"

        _mutate_environment(run_set, bad_id)
        with self.assertRaises(AnalyzerError):
            analyze(run_set)
        missing = _write_run(self.root, 20, contrast="ag2", run_name="no-g3")

        def force_three(doc: dict) -> None:
            doc["collection_protocol"]["cells"] = ["A", "G2", "G3"]
            doc["collection_protocol"]["contrast_id"] = CONTRAST_ID_AG2G3

        _mutate_environment(missing, force_three)
        with self.assertRaises(AnalyzerError):
            analyze(missing)

    def test_adaptive_identity_gates(self) -> None:
        false_g3 = _write_run(self.root, 20, contrast="ag2g3", run_name="g3-false")

        def make_false(document: dict) -> None:
            document["scheduler"]["adaptive_batching"] = False

        _mutate_report(false_g3, "G3-neural_open-01", make_false)
        with self.assertRaises(AnalyzerError) as raised_false:
            analyze(false_g3)
        self.assertIn("adaptive_batching", str(raised_false.exception))
        missing = _write_run(self.root, 20, contrast="ag2g3", run_name="g3-missing")

        def drop(document: dict) -> None:
            del document["scheduler"]["adaptive_batching"]

        _mutate_report(missing, "G3-neural_open-01", drop)
        with self.assertRaises(AnalyzerError):
            analyze(missing)
        true_g2 = _write_run(self.root, 20, contrast="ag2g3", run_name="g2-true")

        def make_true(document: dict) -> None:
            document["scheduler"]["adaptive_batching"] = True

        _mutate_report(true_g2, "G2-neural_open-01", make_true)
        with self.assertRaises(AnalyzerError):
            analyze(true_g2)

    def test_batch_delay_cap_hash_and_rate_gates(self) -> None:
        bad_delay = _write_run(self.root, 20, contrast="ag2g3", run_name="bad-delay")

        def delay(doc: dict) -> None:
            for cell in doc["collection_protocol"]["cell_definitions"]:
                if cell["id"] == "G2":
                    cell["max_batch_delay_ms"] = 1

        _mutate_environment(bad_delay, delay)
        with self.assertRaises(AnalyzerError):
            analyze(bad_delay)
        bad_cap = _write_run(self.root, 20, contrast="ag2g3", run_name="bad-cap")

        def cap(doc: dict) -> None:
            for cell in doc["collection_protocol"]["cell_definitions"]:
                if cell["id"] == "G3":
                    cell["max_batch_size"] = 8

        _mutate_environment(bad_cap, cap)
        with self.assertRaises(AnalyzerError):
            analyze(bad_cap)
        bad_hash = _write_run(self.root, 20, contrast="ag2g3", run_name="bad-bin")

        def bin_hash(doc: dict) -> None:
            doc["queried"]["benchmark_binary_sha256"] = "ff" * 32

        _mutate_environment(bad_hash, bin_hash)
        with self.assertRaises(AnalyzerError):
            analyze(bad_hash)
        bad_model = _write_run(self.root, 20, contrast="ag2g3", run_name="bad-model")

        def model_hash(doc: dict) -> None:
            doc["verified_configuration"]["model_sha256"] = "00" * 32
            doc["collection_protocol"]["model_sha256"] = "00" * 32

        _mutate_environment(bad_model, model_hash)
        with self.assertRaises(AnalyzerError):
            analyze(bad_model)
        bad_side = _write_run(self.root, 20, contrast="ag2g3", run_name="bad-side")

        def side_hash(doc: dict) -> None:
            doc["verified_configuration"]["weight_sidecar_sha256"] = "11" * 32
            doc["collection_protocol"]["weight_sidecar_sha256"] = "11" * 32

        _mutate_environment(bad_side, side_hash)
        with self.assertRaises(AnalyzerError):
            analyze(bad_side)
        bad_rate = _write_run(self.root, 20, contrast="ag2g3", run_name="bad-rate")

        def rate(doc: dict) -> None:
            doc["collection_protocol"]["offered_rate"] = 500.0

        _mutate_environment(bad_rate, rate)
        with self.assertRaises(AnalyzerError):
            analyze(bad_rate)

    def test_arrival_and_batch_gates(self) -> None:
        bad_arrival = _write_run(self.root, 20, contrast="ag2g3", run_name="bad-arrival")

        def arrival(doc: dict) -> None:
            doc["collection_protocol"]["arrival_screen_max_lag_ns"] = 1_000_000

        _mutate_environment(bad_arrival, arrival)
        with self.assertRaises(AnalyzerError):
            analyze(bad_arrival)
        lag = _write_run(
            self.root,
            20,
            contrast="ag2g3",
            run_name="lag-over",
            lag_max=ARRIVAL_SCREEN_MAX_LAG_NS + 1,
        )
        with self.assertRaises(AnalyzerError) as raised:
            analyze(lag)
        self.assertIn("arrival screen", str(raised.exception))
        g2_low = _write_run(
            self.root, 20, contrast="ag2g3", run_name="g2-low", g2_mean_batch=1.5
        )

        def mutate(document: dict) -> None:
            document["metrics"]["engine_run_count"] = 80
            document["metrics"]["requests_entered_engine_runs"] = 100
            document["metrics"]["batch_size_histogram"] = [0, 60, 20] + [0] * 30

        _mutate_report(g2_low, "G2-neural_open-01", mutate)
        with self.assertRaises(AnalyzerError) as g2_raised:
            analyze(g2_low)
        self.assertIn("persistent-batching", str(g2_raised.exception))
        g3_low = _write_run(
            self.root,
            20,
            contrast="ag2g3",
            run_name="g3-low-ok",
            g3_mean_batch=1.5,
        )
        document = json.loads(analyze(g3_low).read_text(encoding="utf-8"))
        self.assertEqual(document["contrast_id"], CONTRAST_ID_AG2G3)
        self.assertFalse(document["winner_declared"])

    def test_order_duplicate_partial_and_legacy(self) -> None:
        bad_order = _write_run(self.root, 20, contrast="ag2g3", run_name="bad-order")
        plan = json.loads((bad_order / "execution_plan.json").read_text(encoding="utf-8"))
        counted = [trial for trial in plan["trials"] if not trial["pilot"]]
        counted[0]["cell_id"], counted[1]["cell_id"] = counted[1]["cell_id"], counted[0]["cell_id"]
        (bad_order / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")

        def sync(doc: dict) -> None:
            doc["collection_protocol"]["trial_order"] = [
                str(trial["trial_id"]) for trial in plan["trials"]
            ]

        _mutate_environment(bad_order, sync)
        with self.assertRaises(AnalyzerError):
            analyze(bad_order)
        partial = _write_run(self.root, 10, contrast="ag2g3", run_name="partial")
        with self.assertRaises(AnalyzerError) as raised:
            analyze(partial)
        self.assertIn("20", str(raised.exception))
        legacy = _write_run(self.root, 20, contrast="ag2g3", run_name="legacy-forbid")
        with self.assertRaises(AnalyzerError) as legacy_raised:
            analyze(legacy, legacy_pre_adaptive_identity=True)
        self.assertIn("prohibited", str(legacy_raised.exception))

    def test_historical_contrasts_still_valid(self) -> None:
        ac = _write_run(self.root, 10, contrast="ac", run_name="hist-ac")
        ag2 = _write_run(self.root, 20, contrast="ag2", run_name="hist-ag2")
        self.assertEqual(
            json.loads(analyze(ac).read_text(encoding="utf-8"))["contrast_sign_convention"],
            "C_minus_A",
        )
        self.assertEqual(
            json.loads(analyze(ag2).read_text(encoding="utf-8"))["contrast_id"],
            CONTRAST_ID_AG2,
        )


class AddAnalyzerSafetyTest(unittest.TestCase):
    def test_existing_add_analyzer_rejects_neural_evidence(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        try:
            # Use an Add-shaped plan identity with neural model hashes so the
            # existing analyzer reaches its natural model-hash refusal.
            from benchmarks.collect_scheduler import MODEL_SHA256 as ADD_SHA
            from benchmarks.collect_scheduler import PROFILE_BY_ID, execution_plan

            root = pathlib.Path(temporary.name)
            run_set = root / "neural-add-refuse"
            runs = run_set / "runs"
            runs.mkdir(parents=True)
            cells = [CELL_BY_ID["A"], CELL_BY_ID["C"]]
            profiles = [PROFILE_BY_ID["low_open"]]
            plan = execution_plan(cells, profiles, 10, 2)
            for trial in plan:
                trial_id = str(trial["trial_id"])
                cell_id = str(trial["cell_id"])
                report = _report(cell_id, int(trial["block_index"] or 0))
                report["model"] = str(MODEL)
                (runs / f"{trial_id}.json").write_text(dump_report(report), encoding="utf-8")
                meta = {
                    "trial_id": trial_id,
                    "configuration_id": cell_id,
                    "workload_id": "low_open",
                    "pilot": trial["pilot"],
                    "run_number": trial["run_number"],
                    "collector_valid": True,
                    "execution_timeout": False,
                    "benchmark_binary_sha256": BINARY_SHA,
                    "model_sha256": GEMM["model_sha256"],
                    "git_commit": COMMIT,
                    "command": ["placeholder"],
                }
                (runs / f"{trial_id}.meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
            environment = {
                "status": "completed",
                "run_set_id": run_set.name,
                "aggregate": None,
                "experiment_kind": EXPERIMENT_KIND,
                "queried": {
                    "git_commit": COMMIT,
                    "benchmark_binary_path": str(BINARY),
                    "benchmark_binary_sha256": BINARY_SHA,
                    "benchmark_tool_version": "0.1.0",
                    "compiler_path": "/usr/bin/c++",
                    "compiler_version": "c++ synthetic\n",
                    "cmake_flag_text": "-O3 -DNDEBUG",
                    "onnxruntime_version": "1.31.0",
                    "operating_system": "Darwin",
                    "cpu_architecture": "arm64",
                    "platform": "macOS",
                    "hardware_model": "synthetic",
                },
                "verified_configuration": {
                    "cmake_build_type": "Release",
                    "model_path": str(MODEL),
                    "model_sha256": GEMM["model_sha256"],
                },
                "collection_protocol": {
                    "analysis_protocol_version": 2,
                    "publication_policy": "strict_generator_delivery",
                    "comparison_class": "primary",
                    "workload_family": "open_loop",
                    "cells": ["A", "C"],
                    "profiles": ["low_open"],
                    "measurement": _measurement(),
                    "counted_runs_per_cell": 10,
                    "counted_run_target": 10,
                    "trial_order": [str(trial["trial_id"]) for trial in plan],
                    "baseline_cell": "A",
                    "alternative_cell": "C",
                    "contrast_source": "predeclared",
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "confidence_level": CONFIDENCE_LEVEL,
                    "block_protocol": "within_profile_cyclic_rotation",
                },
            }
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            (run_set / "execution_plan.json").write_text(
                json.dumps({"trials": plan}) + "\n", encoding="utf-8"
            )
            self.assertNotEqual(GEMM["model_sha256"], ADD_SHA)
            with self.assertRaises(AddAnalyzerError) as raised:
                analyze_add(run_set)
            self.assertIn("pinned batch_add fixture", str(raised.exception))
            self.assertFalse((run_set / "aggregate.json").exists())
        finally:
            temporary.cleanup()


LEGACY_COMMIT, LEGACY_BINARY = next(iter(LEGACY_PRE_ADAPTIVE_IDENTITY_PAIRS))


def _trial_report_paths(run_set: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in (run_set / "runs").glob("*.json")
        if not path.name.endswith(".meta.json")
    )


def _strip_adaptive_field(run_set: pathlib.Path) -> None:
    for path in _trial_report_paths(run_set):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["scheduler"].pop("adaptive_batching", None)
        path.write_text(dump_report(document), encoding="utf-8")


def _set_legacy_identity(run_set: pathlib.Path, commit: str, binary_sha: str) -> None:
    def sync(environment: dict) -> None:
        environment["queried"]["git_commit"] = commit
        environment["queried"]["benchmark_binary_sha256"] = binary_sha

    _mutate_environment(run_set, sync)
    for path in (run_set / "runs").glob("*.meta.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["git_commit"] = commit
        meta["benchmark_binary_sha256"] = binary_sha
        path.write_text(json.dumps(meta) + "\n", encoding="utf-8")


class AdaptiveIdentityAnalyzerTest(unittest.TestCase):
    def test_strict_new_records_status(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        try:
            run_set = _write_run(pathlib.Path(temporary.name), 20, contrast="ag2")
            aggregate = analyze(run_set)
            document = json.loads(aggregate.read_text(encoding="utf-8"))
            self.assertEqual(document["adaptive_identity_status"], "strict_new")
            self.assertFalse(
                any("pre-adaptive report contract" in item for item in document["limitations"])
            )
        finally:
            temporary.cleanup()

    def test_missing_field_rejected_without_legacy_flag(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        try:
            run_set = _write_run(pathlib.Path(temporary.name), 20, contrast="ag2")
            _strip_adaptive_field(run_set)
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("adaptive_batching", str(raised.exception))
        finally:
            temporary.cleanup()

    def test_legacy_approved_identity_accepted(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        try:
            run_set = _write_run(pathlib.Path(temporary.name), 20, contrast="ag2")
            _set_legacy_identity(run_set, LEGACY_COMMIT, LEGACY_BINARY)
            _strip_adaptive_field(run_set)
            aggregate = analyze(run_set, legacy_pre_adaptive_identity=True)
            document = json.loads(aggregate.read_text(encoding="utf-8"))
            self.assertEqual(document["adaptive_identity_status"], "legacy_authorized")
            self.assertTrue(
                any("pre-adaptive report contract" in item for item in document["limitations"])
            )
            self.assertTrue(
                any("not cryptographic attestation" in item for item in document["limitations"])
            )
        finally:
            temporary.cleanup()

    def test_legacy_unapproved_identity_rejected(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        try:
            run_set = _write_run(pathlib.Path(temporary.name), 20, contrast="ag2")
            _strip_adaptive_field(run_set)
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set, legacy_pre_adaptive_identity=True)
            self.assertIn("not allowlisted", str(raised.exception))
        finally:
            temporary.cleanup()

    def test_legacy_cross_matched_pair_rejected(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        try:
            run_set = _write_run(pathlib.Path(temporary.name), 20, contrast="ag2")
            _set_legacy_identity(run_set, LEGACY_COMMIT, "ff" * 32)
            _strip_adaptive_field(run_set)
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set, legacy_pre_adaptive_identity=True)
            self.assertIn("not allowlisted", str(raised.exception))
        finally:
            temporary.cleanup()

    def test_legacy_mixed_field_presence_rejected(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        try:
            run_set = _write_run(pathlib.Path(temporary.name), 20, contrast="ag2")
            _set_legacy_identity(run_set, LEGACY_COMMIT, LEGACY_BINARY)
            reports = _trial_report_paths(run_set)
            document = json.loads(reports[0].read_text(encoding="utf-8"))
            document["scheduler"].pop("adaptive_batching", None)
            reports[0].write_text(dump_report(document), encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set, legacy_pre_adaptive_identity=True)
            self.assertIn("mixed adaptive_batching", str(raised.exception))
        finally:
            temporary.cleanup()

    def test_no_g3_cell_in_resolver(self) -> None:
        with self.assertRaises(Exception):
            resolve_neural_cell("G3")


if __name__ == "__main__":
    unittest.main()
