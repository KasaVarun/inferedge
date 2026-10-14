#!/usr/bin/env python3
"""Deterministic tests for paired completion-timing overhead analysis."""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import unittest

from benchmarks.analyze_overhead import AnalyzerError, analyze
from benchmarks.analyze_scheduler import BOOTSTRAP_SEED as BATCHING_SEED
from benchmarks.analyze_scheduler import paired_block_bootstrap_mean
from benchmarks.collect_overhead import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    EXPERIMENT_KIND,
    INTENDED_VARIABLE,
    OVERHEAD_PROTOCOL_MARKER_MEANING,
    bind_commands,
    build_plan,
    shared_launch,
)
from benchmarks.collect_scheduler import BINARY_NAME, MODEL_SHA256
from benchmarks.test_collect_scheduler import COMMIT, dump_report, sample_report, schema4_report

BINARY = pathlib.Path("/tmp") / BINARY_NAME
MODEL = pathlib.Path("/models/batch_add_f32.onnx")
BINARY_SHA = "ab" * 32


def _report(arm: str, block: int) -> dict:
    document = sample_report(model=str(MODEL))
    document["generator"]["warmup_requests"] = 100
    document["successful_e2e_latency"]["p50_microseconds"] = 2.0 + block
    document["successful_e2e_latency"]["p95_microseconds"] = 20.0
    document["successful_e2e_latency"]["p99_microseconds"] = 21.0
    if arm == "on":
        document = schema4_report(document)
        document["successful_e2e_latency"]["p50_microseconds"] = 2.0 + block + (block + 1)
    return document


def _write_run(root: pathlib.Path, counted_blocks: int) -> pathlib.Path:
    run_set = root / "overhead-run"
    runs = run_set / "runs"
    runs.mkdir(parents=True)
    plan = bind_commands(build_plan(counted_blocks), BINARY, MODEL)
    mode = "smoke" if counted_blocks == 0 else "exploratory"
    evidence_class = "correctness" if counted_blocks == 0 else "exploratory"
    for trial in plan:
        arm = str(trial["arm"])
        block = 0 if trial["block_index"] is None else int(trial["block_index"])
        report = _report(arm, block)
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
            "collector_valid": True,
            "execution_timeout": False,
            "benchmark_binary_sha256": BINARY_SHA,
            "model_sha256": MODEL_SHA256,
            "git_commit": COMMIT,
            "command": trial["command"],
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
            "cpu_architecture": "arm64",
            "hardware_model": "synthetic",
            "ort_disable_telemetry": "1",
        },
        "verified_configuration": {
            "cmake_build_type": "Release",
            "model_path": str(MODEL),
            "model_sha256": MODEL_SHA256,
        },
        "collection_protocol": {
            "experiment_kind": EXPERIMENT_KIND,
            "analysis_protocol_version": 0,
            "analysis_protocol_version_meaning": OVERHEAD_PROTOCOL_MARKER_MEANING,
            "collection_mode": mode,
            "evidence_class": evidence_class,
            "shared_launch": shared_launch(),
            "counted_blocks": counted_blocks,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": CONFIDENCE_LEVEL,
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
        self.run_set = _write_run(pathlib.Path(self.temporary.name), 1)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_different_binary_hash_is_rejected(self) -> None:
        _mutate_meta(self.run_set, "overhead-b00-on", benchmark_binary_sha256="cd" * 32)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("benchmark_binary_sha256", str(raised.exception))

    def test_different_model_hash_is_rejected(self) -> None:
        _mutate_meta(self.run_set, "overhead-b00-off", model_sha256="ef" * 32)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("model_sha256", str(raised.exception))

    def test_different_workload_is_rejected(self) -> None:
        _mutate_report(self.run_set, "overhead-b00-on", lambda document: document.update(workload="closed_loop"))
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("workload", str(raised.exception))

    def test_different_target_rate_is_rejected(self) -> None:
        def mutate(document: dict) -> None:
            document["burst_phases"][0]["requests_per_second"] = 200.0

        _mutate_report(self.run_set, "overhead-b00-on", mutate)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("target_rps", str(raised.exception))

    def test_different_scheduled_span_is_rejected(self) -> None:
        _mutate_report(
            self.run_set,
            "overhead-b00-off",
            lambda document: document.update(scheduled_span_nanoseconds=980_000_000),
        )
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("scheduled_span_nanoseconds", str(raised.exception))

    def test_different_observation_quantum_is_rejected(self) -> None:
        _mutate_report(
            self.run_set,
            "overhead-b00-on",
            lambda document: document.update(observation_quantum_nanoseconds=2_000_000),
        )
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("quantum", str(raised.exception))

    def test_mixed_batching_settings_are_rejected(self) -> None:
        def mutate(document: dict) -> None:
            document["scheduler"]["enable_batching"] = True

        _mutate_report(self.run_set, "overhead-b00-on", mutate)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("enable_batching", str(raised.exception))

    def test_both_arms_off_is_rejected(self) -> None:
        off = json.loads((self.run_set / "runs" / "overhead-b00-off.json").read_text(encoding="utf-8"))
        (self.run_set / "runs" / "overhead-b00-on.json").write_text(dump_report(off), encoding="utf-8")
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("schema", str(raised.exception))

    def test_both_arms_on_is_rejected(self) -> None:
        on = json.loads((self.run_set / "runs" / "overhead-b00-on.json").read_text(encoding="utf-8"))
        (self.run_set / "runs" / "overhead-b00-off.json").write_text(dump_report(on), encoding="utf-8")
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("schema", str(raised.exception))

    def test_wrong_execution_index_is_rejected(self) -> None:
        _mutate_meta(self.run_set, "overhead-b00-on", execution_index=9)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("execution_index", str(raised.exception))

    def test_wrong_pilot_flag_is_rejected(self) -> None:
        _mutate_meta(self.run_set, "overhead-pilot-off", pilot=False)
        with self.assertRaises(AnalyzerError) as raised:
            analyze(self.run_set)
        self.assertIn("pilot", str(raised.exception))


class InventoryTest(unittest.TestCase):
    def test_missing_arm_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
            plan["trials"] = [trial for trial in plan["trials"] if trial["trial_id"] != "overhead-b00-on"]
            (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("exactly one OFF and one ON", str(raised.exception))

    def test_duplicate_arm_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
            for trial in plan["trials"]:
                if trial["trial_id"] == "overhead-b00-on":
                    trial["arm"] = "off"
            (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("one OFF and one ON", str(raised.exception))

    def test_duplicate_block_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
            for trial in plan["trials"]:
                if trial["block_index"] == 0:
                    trial["block_index"] = 0
            extra = json.loads(json.dumps(plan["trials"][-1]))
            extra["trial_id"] = "overhead-b00-extra"
            extra["execution_index"] = len(plan["trials"])
            plan["trials"].append(extra)
            (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("exactly one OFF and one ON", str(raised.exception))

    def test_missing_block_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["collection_protocol"]["counted_blocks"] = 2
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("missing block", str(raised.exception))

    def test_wrong_counted_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
            counted = [trial for trial in plan["trials"] if trial["block_index"] == 0]
            counted.sort(key=lambda trial: trial["position_in_block"])
            counted[0]["arm"], counted[1]["arm"] = counted[1]["arm"], counted[0]["arm"]
            (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("order", str(raised.exception))


class EffectTest(unittest.TestCase):
    def test_pilot_is_excluded_and_one_pair_has_no_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)

            def far(document: dict) -> None:
                document["successful_e2e_latency"]["p50_microseconds"] = 500.0

            _mutate_report(run_set, "overhead-pilot-off", far)
            _mutate_report(run_set, "overhead-pilot-on", far)
            aggregate = json.loads(analyze(run_set).read_text(encoding="utf-8"))
            latency = next(item for item in aggregate["shared_metrics"] if item["metric"] == "p50_e2e_latency")
            self.assertEqual(latency["paired_block_count"], 1)
            self.assertEqual(latency["blocks"][0]["block_index"], 0)
            self.assertEqual(latency["blocks"][0]["difference"], 1.0)
            self.assertFalse(latency["confidence_interval_computed"])
            self.assertIsNone(latency["confidence_interval"])
            self.assertFalse(aggregate["winner_declared"])
            self.assertFalse(aggregate["bootstrap"]["computed"])
            self.assertEqual(aggregate["on_only_observations"][0]["off_counterpart"], None)
            self.assertNotIn("pre_publication_latency", aggregate["shared_metrics"][0])

    def test_negative_paired_effect_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)

            def lower(document: dict) -> None:
                document["successful_e2e_latency"]["p50_microseconds"] = 1.0

            _mutate_report(run_set, "overhead-b00-on", lower)
            aggregate = json.loads(analyze(run_set).read_text(encoding="utf-8"))
            latency = next(item for item in aggregate["shared_metrics"] if item["metric"] == "p50_e2e_latency")
            self.assertLess(latency["mean_paired_difference"], 0.0)
            self.assertEqual(latency["median_paired_difference"], latency["mean_paired_difference"])
            self.assertFalse(aggregate["winner_declared"])
            self.assertFalse(aggregate["superiority_claimed"])
            self.assertFalse(latency["superiority_claimed"])

    def test_zero_baseline_leaves_the_relative_effect_undefined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)

            def zero(document: dict) -> None:
                document["throughput"]["successful_attempts_per_arrival_window_second"] = 0.0

            _mutate_report(run_set, "overhead-b00-off", zero)
            aggregate = json.loads(analyze(run_set).read_text(encoding="utf-8"))
            rate = next(item for item in aggregate["shared_metrics"] if item["metric"] == "successful_throughput")
            self.assertIsNone(rate["relative_effect_if_defined"])
            self.assertGreater(rate["mean_paired_difference"], 0.0)

    def test_negative_baseline_is_not_used_as_a_percentage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)

            def negative(document: dict) -> None:
                document["generator_lag"]["minimum_nanoseconds"] = -5

            _mutate_report(run_set, "overhead-b00-off", negative)
            aggregate = json.loads(analyze(run_set).read_text(encoding="utf-8"))
            lag = next(item for item in aggregate["shared_metrics"] if item["metric"] == "generator_lag_minimum")
            self.assertEqual(lag["blocks"][0]["off"], -5.0)
            self.assertIsNone(lag["relative_effect_if_defined"])
            self.assertEqual(lag["mean_paired_difference"], 5.0)

    def test_nonfinite_metric_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)

            def blank(document: dict) -> None:
                document["successful_e2e_latency"]["mean_microseconds"] = None

            _mutate_report(run_set, "overhead-b00-on", blank)
            with self.assertRaises(AnalyzerError):
                analyze(run_set)
            self.assertFalse((run_set / "aggregate.json").exists())

    def test_seeded_bootstrap_is_deterministic_at_ten_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            run_set = _write_run(root, 10)
            first = json.loads(analyze(run_set).read_text(encoding="utf-8"))
            copy = root / "copy"
            shutil.copytree(run_set, copy)
            (copy / "aggregate.json").unlink()
            second = json.loads(analyze(copy).read_text(encoding="utf-8"))
            latency = next(item for item in first["shared_metrics"] if item["metric"] == "p50_e2e_latency")
            repeated = next(item for item in second["shared_metrics"] if item["metric"] == "p50_e2e_latency")
            self.assertEqual(latency["confidence_interval"], repeated["confidence_interval"])
            self.assertTrue(latency["confidence_interval_computed"])
            differences = [block["difference"] for block in latency["blocks"]]
            self.assertEqual(len(differences), 10)
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
            self.assertFalse(first["winner_declared"])
            self.assertFalse(first["superiority_claimed"])
            self.assertFalse(first["zero_overhead_claimed"])

    def test_nine_blocks_do_not_receive_an_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = json.loads(analyze(_write_run(pathlib.Path(directory), 9)).read_text(encoding="utf-8"))
            self.assertFalse(aggregate["bootstrap"]["computed"])
            self.assertTrue(all(not item["confidence_interval_computed"] for item in aggregate["shared_metrics"]))
            self.assertFalse(aggregate["winner_declared"])

    def test_smoke_pair_has_no_interval_and_no_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = json.loads(analyze(_write_run(pathlib.Path(directory), 0)).read_text(encoding="utf-8"))
            self.assertEqual(aggregate["shared_metrics"], [])
            self.assertEqual(aggregate["on_only_observations"], [])
            self.assertFalse(aggregate["bootstrap"]["computed"])
            self.assertFalse(aggregate["winner_declared"])
            self.assertEqual(aggregate["evidence_class"], "correctness")
            self.assertIn("not a general overhead estimate", " ".join(aggregate["limitations"]))

    def test_second_aggregate_write_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            analyze(run_set)
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("overwrite", str(raised.exception))


class AuditGapTest(unittest.TestCase):
    def test_batching_analyzer_rejects_the_overhead_sentinel(self) -> None:
        from benchmarks.analyze_scheduler import AnalyzerError as BatchAnalyzerError
        from benchmarks.analyze_scheduler import analyze as analyze_batch

        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            with self.assertRaises(BatchAnalyzerError) as raised:
                analyze_batch(run_set)
            self.assertIn("not supported", str(raised.exception))

    def test_wrong_experiment_kind_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["experiment_kind"] = "batching"
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("completion_timing_overhead", str(raised.exception))

    def test_protocol_two_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["collection_protocol"]["analysis_protocol_version"] = 2
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("sentinel", str(raised.exception))

    def test_wrong_ort_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["queried"]["onnxruntime_version"] = "1.30.0"
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("onnxruntime_version", str(raised.exception))

    def test_smoke_pair_cannot_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["collection_protocol"]["collection_mode"] = "smoke"
            environment["collection_protocol"]["evidence_class"] = "correctness"
            (run_set / "environment.json").write_text(json.dumps(environment) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("promoted", str(raised.exception))

    def test_missing_pilot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
            plan["trials"] = [trial for trial in plan["trials"] if not trial["pilot"]]
            (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("pilot", str(raised.exception))

    def test_duplicate_pilot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
            extra = json.loads(json.dumps(plan["trials"][0]))
            extra["trial_id"] = "overhead-pilot-extra"
            extra["execution_index"] = len(plan["trials"])
            plan["trials"].append(extra)
            (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("exactly one", str(raised.exception))

    def test_plan_command_divergence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
            for trial in plan["trials"]:
                if trial["trial_id"] == "overhead-b00-off":
                    trial["command"] = [token if token != "100" else "1" for token in trial["command"]]
            (run_set / "execution_plan.json").write_text(json.dumps(plan) + "\n", encoding="utf-8")
            _mutate_meta(
                run_set,
                "overhead-b00-off",
                command=next(trial["command"] for trial in plan["trials"] if trial["trial_id"] == "overhead-b00-off"),
            )
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("diverges", str(raised.exception))

    def test_different_scheduler_cell_is_rejected(self) -> None:
        def mutate(document: dict) -> None:
            document["scheduler"]["max_batch_size"] = 8

        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            _mutate_report(run_set, "overhead-b00-on", mutate)
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("max_batch_size", str(raised.exception))

    def test_unequal_histograms_are_reported(self) -> None:
        def mutate(document: dict) -> None:
            document["metrics"]["engine_run_count"] = 50
            document["metrics"]["requests_entered_engine_runs"] = 100
            document["metrics"]["batch_size_histogram"] = [0, 0, 50] + [0] * 30

        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            _mutate_report(run_set, "overhead-b00-on", mutate)
            aggregate = json.loads(analyze(run_set).read_text(encoding="utf-8"))
            self.assertFalse(aggregate["batch_size_histograms"][0]["equal"])
            self.assertFalse(aggregate["winner_declared"])

    def test_interval_containing_zero_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 10)
            for block in range(10):
                def one(document: dict, expected: float = 2.0 + block) -> None:
                    document["successful_e2e_latency"]["p50_microseconds"] = expected

                _mutate_report(run_set, f"overhead-b{block:02d}-off", one)
                _mutate_report(run_set, f"overhead-b{block:02d}-on", one)
            aggregate = json.loads(analyze(run_set).read_text(encoding="utf-8"))
            latency = next(item for item in aggregate["shared_metrics"] if item["metric"] == "p50_e2e_latency")
            self.assertEqual(latency["mean_paired_difference"], 0.0)
            self.assertEqual(latency["interpretation"], "inconclusive")
            self.assertFalse(latency["comparison_supported"])
            self.assertFalse(aggregate["zero_overhead_claimed"])

    def test_missing_metric_is_not_dropped(self) -> None:
        def mutate(document: dict) -> None:
            del document["metrics"]["engine_run_count"]

        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            _mutate_report(run_set, "overhead-b00-off", mutate)
            with self.assertRaises(AnalyzerError):
                analyze(run_set)
            self.assertFalse((run_set / "aggregate.json").exists())

    def test_metadata_filename_must_match_the_trial_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_set = _write_run(pathlib.Path(directory), 1)
            source = run_set / "runs" / "overhead-b00-off.meta.json"
            source.rename(run_set / "runs" / "renamed.meta.json")
            with self.assertRaises(AnalyzerError) as raised:
                analyze(run_set)
            self.assertIn("does not match trial id", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
