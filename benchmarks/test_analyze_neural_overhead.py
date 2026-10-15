#!/usr/bin/env python3
"""Deterministic tests for neural batch-decision timing overhead analysis."""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import unittest

from benchmarks.analyze_neural_overhead import AnalyzerError, analyze
from benchmarks.analyze_scheduler import BOOTSTRAP_SEED as BATCHING_SEED
from benchmarks.analyze_scheduler import paired_block_bootstrap_mean
from benchmarks.collect_neural_overhead import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    DURATION_MS,
    EVENT_CAPACITY,
    EXPERIMENT_KIND,
    EXPLORATORY_COUNTED_BLOCKS,
    FORMAL_COUNTED_BLOCKS,
    INTENDED_VARIABLE,
    MODEL_ID,
    OVERHEAD_PROTOCOL_MARKER_MEANING,
    TARGET_RPS,
    bind_commands,
    build_plan,
    policy_cell,
    shared_launch,
)
from benchmarks.collect_neural_scheduler import NEURAL_MODELS, REFERENCE_CHECKING_POLICY
from benchmarks.collect_scheduler import BINARY_NAME
from benchmarks.test_collect_neural_overhead import _arm_report
from benchmarks.test_collect_scheduler import COMMIT, dump_report

BINARY = pathlib.Path("/tmp") / BINARY_NAME
MODEL = pathlib.Path("/models/gemm_f32.onnx")
BINARY_SHA = "ab" * 32
GEMM = NEURAL_MODELS["gemm_f32"]
SIDECAR_SHA = GEMM["sidecar_sha256"]
MODEL_SHA = GEMM["model_sha256"]


def _report(arm: str, block: int, cell: dict | None = None) -> dict:
    cell = cell or policy_cell("G2")
    document = _arm_report(arm, cell)
    document["successful_e2e_latency"]["p50_microseconds"] = 2.0 + block
    document["successful_e2e_latency"]["p95_microseconds"] = 20.0
    document["successful_e2e_latency"]["p99_microseconds"] = 21.0
    if arm == "on":
        document["successful_e2e_latency"]["p50_microseconds"] = 2.0 + block + (block + 1)
    return document


def _write_run(
    root: pathlib.Path,
    counted_blocks: int,
    *,
    cell_id: str = "G2",
    mode: str | None = None,
) -> pathlib.Path:
    run_set = root / "neural-overhead-run"
    runs = run_set / "runs"
    runs.mkdir(parents=True)
    cell = policy_cell(cell_id)
    plan = bind_commands(build_plan(counted_blocks, cell_id), BINARY, MODEL, cell)
    if mode is None:
        if counted_blocks == 0:
            mode = "smoke"
        elif counted_blocks == EXPLORATORY_COUNTED_BLOCKS:
            mode = "exploratory"
        elif counted_blocks == FORMAL_COUNTED_BLOCKS:
            mode = "formal"
        else:
            mode = "exploratory"
    evidence_class = {
        "smoke": "correctness",
        "exploratory": "exploratory",
        "formal": "counted_overhead",
    }[mode]
    for trial in plan:
        arm = str(trial["arm"])
        block = 0 if trial["block_index"] is None else int(trial["block_index"])
        report = _report(arm, block, cell)
        trial_id = str(trial["trial_id"])
        (runs / f"{trial_id}.json").write_text(dump_report(report), encoding="utf-8")
        meta = {
            "trial_id": trial_id,
            "arm": arm,
            "pilot": trial["pilot"],
            "block_index": trial["block_index"],
            "position_in_block": trial["position_in_block"],
            "execution_index": trial["execution_index"],
            "run_index": trial["run_index"],
            "policy_cell": cell_id,
            "cell_id": cell_id,
            "collector_valid": True,
            "execution_timeout": False,
            "benchmark_binary_sha256": BINARY_SHA,
            "model_sha256": MODEL_SHA,
            "weight_sidecar_sha256": SIDECAR_SHA,
            "git_commit": COMMIT,
            "command": trial["command"],
            "report_path": str(runs / f"{trial_id}.json"),
        }
        (runs / f"{trial_id}.meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    environment = {
        "experiment_kind": EXPERIMENT_KIND,
        "intended_variable": INTENDED_VARIABLE,
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
            "model_sha256": MODEL_SHA,
            "weight_sidecar_sha256": SIDECAR_SHA,
            "model_id": MODEL_ID,
            "policy_cell": cell_id,
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
        },
        "collection_protocol": {
            "experiment_kind": EXPERIMENT_KIND,
            "intended_variable": INTENDED_VARIABLE,
            "analysis_protocol_version": 0,
            "analysis_protocol_version_meaning": OVERHEAD_PROTOCOL_MARKER_MEANING,
            "collection_mode": mode,
            "evidence_class": evidence_class,
            "shared_launch": shared_launch(cell),
            "counted_blocks": counted_blocks,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": CONFIDENCE_LEVEL,
            "policy_cell": cell_id,
            "model_id": MODEL_ID,
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
            "arrival_fidelity_status": "not_certified",
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


class FairnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_set = _write_run(pathlib.Path(self.temporary.name), 3)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_wrong_binary_sha_rejected(self) -> None:
        _mutate_meta(self.run_set, "neural-overhead-b00-on", benchmark_binary_sha256="cd" * 32)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("benchmark_binary_sha256", str(raised.exception))

    def test_wrong_model_sha_rejected(self) -> None:
        _mutate_meta(self.run_set, "neural-overhead-b00-off", model_sha256="ef" * 32)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("model_sha256", str(raised.exception))

    def test_wrong_sidecar_sha_rejected(self) -> None:
        _mutate_meta(self.run_set, "neural-overhead-b00-on", weight_sidecar_sha256="11" * 32)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("weight_sidecar_sha256", str(raised.exception))

    def test_wrong_policy_rejected(self) -> None:
        def mutate(document: dict) -> None:
            document["scheduler"]["arrival_aware_batching"] = True

        _mutate_report(self.run_set, "neural-overhead-b00-on", mutate)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertTrue(
            "arrival_aware" in str(raised.exception) or "policy" in str(raised.exception).lower()
            or "Gen C" in str(raised.exception)
            or "identity" in str(raised.exception)
        )

    def test_wrong_schema_rejected(self) -> None:
        off = json.loads(
            (self.run_set / "runs" / "neural-overhead-b00-off.json").read_text(encoding="utf-8")
        )
        (self.run_set / "runs" / "neural-overhead-b00-on.json").write_text(
            dump_report(off), encoding="utf-8"
        )
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("schema", str(raised.exception))


class InventoryTest(unittest.TestCase):
    def test_missing_paired_arm_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)
            plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
            plan["trials"] = [
                trial for trial in plan["trials"] if trial["trial_id"] != "neural-overhead-b00-on"
            ]
            for index, trial in enumerate(plan["trials"]):
                trial["execution_index"] = index
            (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("exactly one OFF and one ON", str(raised.exception))

    def test_duplicate_arm_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)
            plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
            for trial in plan["trials"]:
                if trial["trial_id"] == "neural-overhead-b00-on":
                    trial["arm"] = "off"
            (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("one OFF and one ON", str(raised.exception))

    def test_wrong_block_index_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["collection_protocol"]["counted_blocks"] = 4
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertTrue(
                "three" in str(raised.exception)
                or "missing block" in str(raised.exception)
                or "exactly three" in str(raised.exception)
            )

    def test_wrong_arm_order_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)
            plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
            counted = [trial for trial in plan["trials"] if trial["block_index"] == 0]
            counted.sort(key=lambda trial: trial["position_in_block"])
            counted[0]["arm"], counted[1]["arm"] = counted[1]["arm"], counted[0]["arm"]
            (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("order", str(raised.exception))

    def test_mixed_policy_run_set_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3, cell_id="G2")
            plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
            for trial in plan["trials"]:
                if trial["trial_id"] == "neural-overhead-b01-on":
                    trial["policy_cell"] = "G4"
                    trial["cell_id"] = "G4"
            (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertTrue(
                "policy" in str(raised.exception).lower() or "G4" in str(raised.exception)
            )


class EffectTest(unittest.TestCase):
    def test_exploratory_descriptive_no_ci(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = json.loads(
                analyze(_write_run(pathlib.Path(directory), 3)).read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate["counted_blocks"], 3)
            self.assertEqual(aggregate["evidence_class"], "exploratory")
            self.assertFalse(aggregate["bootstrap"]["computed"])
            latency = next(
                item for item in aggregate["shared_metrics"] if item["metric"] == "p50_e2e_latency"
            )
            self.assertEqual(latency["paired_block_count"], 3)
            self.assertFalse(latency["confidence_interval_computed"])
            self.assertIsNone(latency["confidence_interval"])
            self.assertFalse(aggregate["winner_declared"])
            self.assertFalse(aggregate["zero_overhead_claimed"])
            self.assertFalse(aggregate["g4_superiority_inferred"])
            self.assertEqual(aggregate["arrival_fidelity"]["status"], "not_certified")
            self.assertTrue(aggregate["on_only_observations"])
            self.assertIn("seal_reason_frequencies", aggregate["on_only_observations"][0])

    def test_smoke_no_ci(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = json.loads(
                analyze(_write_run(pathlib.Path(directory), 0)).read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate["shared_metrics"], [])
            self.assertFalse(aggregate["bootstrap"]["computed"])
            self.assertEqual(aggregate["evidence_class"], "correctness")

    def test_formal_paired_bootstrap_no_request_pseudoreplication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            run_set = _write_run(root, FORMAL_COUNTED_BLOCKS, mode="formal")
            first = json.loads(analyze(run_set).read_text(encoding="utf-8"))
            copy = root / "copy"
            shutil.copytree(run_set, copy)
            (copy / "aggregate.json").unlink()
            second = json.loads(analyze(copy).read_text(encoding="utf-8"))
            latency = next(
                item for item in first["shared_metrics"] if item["metric"] == "p50_e2e_latency"
            )
            repeated = next(
                item for item in second["shared_metrics"] if item["metric"] == "p50_e2e_latency"
            )
            self.assertEqual(latency["confidence_interval"], repeated["confidence_interval"])
            self.assertTrue(latency["confidence_interval_computed"])
            self.assertEqual(latency["paired_block_count"], 20)
            differences = [block["difference"] for block in latency["blocks"]]
            self.assertEqual(len(differences), 20)
            expected = paired_block_bootstrap_mean(
                differences,
                seed=BOOTSTRAP_SEED,
                replicates=BOOTSTRAP_REPLICATES,
                level=CONFIDENCE_LEVEL,
            )
            other = paired_block_bootstrap_mean(
                differences,
                seed=BATCHING_SEED,
                replicates=BOOTSTRAP_REPLICATES,
                level=CONFIDENCE_LEVEL,
            )
            self.assertEqual(latency["confidence_interval"], expected)
            self.assertNotEqual(expected, other)
            self.assertEqual(BOOTSTRAP_SEED, 20261015)
            self.assertFalse(first["bootstrap"]["request_level_bootstrap"])
            self.assertFalse(first["winner_declared"])
            self.assertFalse(first["g4_superiority_inferred"])
            self.assertFalse(first["zero_overhead_claimed"])

    def test_interval_containing_zero_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), FORMAL_COUNTED_BLOCKS, mode="formal")
            for block in range(FORMAL_COUNTED_BLOCKS):
                def one(document: dict, expected: float = 2.0 + block) -> None:
                    document["successful_e2e_latency"]["p50_microseconds"] = expected

                _mutate_report(run_set, f"neural-overhead-b{block:02d}-off", one)
                _mutate_report(run_set, f"neural-overhead-b{block:02d}-on", one)
            aggregate = json.loads(analyze(run_set).read_text(encoding="utf-8"))
            latency = next(
                item for item in aggregate["shared_metrics"] if item["metric"] == "p50_e2e_latency"
            )
            self.assertEqual(latency["mean_paired_difference"], 0.0)
            self.assertEqual(latency["interpretation"], "inconclusive")
            self.assertFalse(aggregate["zero_overhead_claimed"])

    def test_g4_run_set_does_not_infer_superiority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = json.loads(
                analyze(_write_run(pathlib.Path(directory), 3, cell_id="G4")).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(aggregate["policy_cell"], "G4")
            self.assertFalse(aggregate["g4_superiority_inferred"])
            self.assertFalse(aggregate["superiority_claimed"])
            self.assertFalse(aggregate["bootstrap"]["computed"])


class IsolationTest(unittest.TestCase):
    def test_wrong_experiment_kind_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["experiment_kind"] = "neural_scheduler_batching"
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("batch_decision_timing_overhead", str(raised.exception))

    def test_completion_timing_overhead_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["experiment_kind"] = "completion_timing_overhead"
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("batch_decision_timing_overhead", str(raised.exception))

    def test_protocol_two_marker_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["collection_protocol"]["analysis_protocol_version"] = 2
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("sentinel", str(raised.exception))

    def test_neural_scheduler_analyzer_rejects_this_kind(self) -> None:
        from benchmarks.analyze_neural_scheduler import AnalyzerError as NeuralAnalyzerError
        from benchmarks.analyze_neural_scheduler import analyze as analyze_neural

        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["status"] = "completed"
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            with self.assertRaises(NeuralAnalyzerError) as raised:
                analyze_neural(run_set)
            self.assertIn("neural_scheduler_batching", str(raised.exception))

    def test_add_overhead_analyzer_rejects_this_kind(self) -> None:
        from benchmarks.analyze_overhead import AnalyzerError as AddAnalyzerError
        from benchmarks.analyze_overhead import analyze as analyze_add

        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)
            with self.assertRaises(AddAnalyzerError) as raised:
                analyze_add(run_set)
            self.assertIn("completion_timing_overhead", str(raised.exception))

    def test_batching_analyzer_rejects_protocol_sentinel(self) -> None:
        from benchmarks.analyze_scheduler import AnalyzerError as BatchAnalyzerError
        from benchmarks.analyze_scheduler import analyze as analyze_batch

        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)
            with self.assertRaises(BatchAnalyzerError) as raised:
                analyze_batch(run_set)
            self.assertTrue(
                "not supported" in str(raised.exception)
                or "sentinel" in str(raised.exception)
                or "protocol" in str(raised.exception)
            )

    def test_smoke_cannot_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["collection_protocol"]["collection_mode"] = "smoke"
            environment["collection_protocol"]["evidence_class"] = "correctness"
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("promoted", str(raised.exception))

    def test_trace_overflow_hard_stop_in_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 3)

            def mutate(document: dict) -> None:
                document["batch_decision"]["trace_overflow"] = True
                document["batch_decision"]["timing_data_valid"] = False
                document["batch_decision"]["events_dropped"] = 1
                document["batch_decision"]["sealed_batches_total"] = 101

            _mutate_report(run_set, "neural-overhead-b00-on", mutate)
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertTrue(
                "overflow" in str(raised.exception) or "dropped" in str(raised.exception)
            )

    def test_frozen_workload_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = json.loads(
                analyze(_write_run(pathlib.Path(directory), 3)).read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate["model_id"], "gemm_f32")
            self.assertEqual(aggregate["policy_cell"], "G2")
            self.assertEqual(aggregate["experiment_kind"], EXPERIMENT_KIND)
            self.assertEqual(aggregate["primary_endpoint"], "p50_e2e_latency")
            self.assertEqual(TARGET_RPS, 100.0)
            self.assertEqual(DURATION_MS, 5000)
            self.assertEqual(EVENT_CAPACITY, 1024)


if __name__ == "__main__":
    unittest.main()
