#!/usr/bin/env python3
"""Deterministic tests for neural scheduler analysis."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from benchmarks.analyze_neural_scheduler import AnalyzerError, analyze
from benchmarks.analyze_scheduler import (
    AnalyzerError as AddAnalyzerError,
)
from benchmarks.analyze_scheduler import analyze as analyze_add
from benchmarks.analyze_scheduler import paired_block_bootstrap_mean
from benchmarks.collect_neural_scheduler import (
    ANALYSIS_PROTOCOL_VERSION,
    CONTRAST_SIGN_CONVENTION,
    EXPERIMENT_KIND,
    NEURAL_MODELS,
    REFERENCE_CHECKING_POLICY,
    bind_commands,
    build_plan,
    open_loop_profile,
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


def _report(cell_id: str, block: int) -> dict:
    cell = CELL_BY_ID[cell_id]
    document = sample_report(model=str(MODEL))
    document["generator"]["warmup_requests"] = 100
    document["scheduler"]["enable_batching"] = cell["enable_batching"]
    document["scheduler"]["max_batch_size"] = cell["max_batch_size"]
    document["scheduler"]["max_batch_delay_ms"] = cell["max_batch_delay_ms"]
    # A baseline stays lower; C is higher by a fixed offset so C-A is positive.
    document["successful_e2e_latency"]["p50_microseconds"] = 10.0 + block + (5.0 if cell_id == "C" else 0.0)
    document["successful_e2e_latency"]["p95_microseconds"] = 20.0 + block
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
) -> pathlib.Path:
    run_set = root / "neural-run"
    runs = run_set / "runs"
    runs.mkdir(parents=True)
    cells = [CELL_BY_ID["A"], CELL_BY_ID["C"]]
    profiles = [PROFILE]
    plan = bind_commands(
        build_plan(cells, profiles, counted_blocks),
        binary=BINARY,
        model=MODEL,
        model_id=model_id,
        cells={"A": CELL_BY_ID["A"], "C": CELL_BY_ID["C"]},
        profiles={"neural_open": PROFILE},
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
        report = _report(cell_id, block)
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
            "offered_rate": PROFILE["target_rps"],
            "command": trial["command"],
        }
        (runs / f"{trial_id}.meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
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
        "collection_protocol": {
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
            "contrast_sign_convention": CONTRAST_SIGN_CONVENTION,
            "calibration_status": "not_calibrated",
            "offered_rate_frozen": offered_rate_frozen,
            "offered_rate": PROFILE["target_rps"],
            "traffic_profile": PROFILE["id"],
            "cells": ["A", "C"],
            "profiles": [PROFILE["id"]],
            "cell_definitions": cells,
            "profile_definitions": profiles,
            "measurement": _measurement(),
            "counted_runs_per_cell": counted_blocks,
            "counted_run_target": counted_blocks,
            "trial_order": [str(trial["trial_id"]) for trial in plan],
            "baseline_cell": "A",
            "alternative_cell": "C",
            "contrast_source": "predeclared",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": CONFIDENCE_LEVEL,
            "zero_admission_rejection_required": mode == "reportable",
            "require_strict_generator_delivery": mode in {"smoke", "reportable"},
        },
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


if __name__ == "__main__":
    unittest.main()
