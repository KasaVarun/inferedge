#!/usr/bin/env python3
"""Synthetic tests for concurrent scaling analysis. They do not run a model."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.analyze_concurrent import (  # noqa: E402
    AnalyzerError,
    analyze_and_write,
    build_aggregate,
    main,
)
from benchmarks.collect_baseline import public_manifest  # noqa: E402
from benchmarks.collect_concurrent import (  # noqa: E402
    ADD_MODEL_SHA256,
    AGGREGATE_RELATIVE_TOLERANCE,
    MEASURED_ITERATIONS_PER_WORKER,
    MNIST_FIXTURE_MANIFEST_SHA256,
    MNIST_INPUT_SHA256,
    MNIST_MODEL_SHA256,
    MNIST_REFERENCE_SHA256,
    ORDERING_TOLERANCE,
    execution_plan,
    trial_order,
)

RUN_SET_ID = "20261012T000000Z-abcdef0"
COMMIT = "a" * 40
ADD_MODEL = "/models/add_f32.onnx"
MNIST_MODEL = "/models/mnist_mlp.onnx"
ACTIVE_CALL = 7.0
PILOT_MEAN = 1_000_000.0

ADD_MEANS = {
    1: [10, 11, 12, 13, 30],
    2: [20, 22, 24, 26, 48],
    4: [30, 33, 36, 39, 60],
    8: [40, 44, 48, 52, 70],
}
ADD_P99 = {
    1: [40, 41, 42, 43, 80],
    2: [80, 82, 84, 86, 160],
    4: [120, 123, 126, 129, 240],
    8: [160, 164, 168, 172, 320],
}
ADD_AGGREGATE = {
    1: [100, 110, 120, 130, 200],
    2: [200, 220, 240, 260, 400],
    4: [400, 440, 480, 520, 800],
    8: [400, 440, 480, 520, 560],
}


def _series(table: dict[int, list[float]], workload: str, worker_count: int, index: int) -> float:
    value = float(table[worker_count][index])
    if workload == "mnist" and table is ADD_AGGREGATE:
        return value / 10.0
    if workload == "mnist":
        return value + 1000.0
    return value


def report_for(trial: dict[str, object]) -> dict:
    workload = str(trial["workload"])
    worker_count = int(trial["worker_count"])  # type: ignore[arg-type]
    total = worker_count * MEASURED_ITERATIONS_PER_WORKER
    if trial["pilot"] is True:
        mean = PILOT_MEAN
        p99 = PILOT_MEAN
        aggregate = float(total) * 1_000_000.0
        wall = 1.0
    else:
        index = int(trial["run_number"]) - 1  # type: ignore[arg-type]
        mean = _series(ADD_MEANS, workload, worker_count, index)
        p99 = _series(ADD_P99, workload, worker_count, index)
        aggregate = _series(ADD_AGGREGATE, workload, worker_count, index)
        wall = total * 1_000_000.0 / aggregate
    return {
        "schema_version": 2,
        "tool": {"name": "inferedge-bench-concurrent", "version": "0.1.0"},
        "model": ADD_MODEL if workload == "add" else MNIST_MODEL,
        "workload": workload,
        "fixture": None if workload == "add" else "test_00003",
        "backend": "ONNX Runtime CPU",
        "configuration": {
            "worker_count": worker_count,
            "session_ownership": "one_inference_engine_per_worker",
            "onnx_intra_op_threads_per_session": 1,
            "graph_optimizations": "disabled",
            "warmup_iterations_per_worker": 100,
            "measured_iterations_per_worker": 1000,
            "correctness_check_interval": 1,
        },
        "latency_microseconds": {
            "sample_count": total,
            "minimum": 1.0 if mean > 0 else 0.0,
            "mean": mean,
            "p50": mean,
            "p95": (mean + p99) / 2.0,
            "p99": p99,
            "maximum": p99 + 1.0,
            "population_standard_deviation": 0.25 if mean > 0 else 0.0,
        },
        "measurement_wall_microseconds": wall,
        "total_completed_requests": total,
        "per_worker_completed_requests": [MEASURED_ITERATIONS_PER_WORKER] * worker_count,
        "aggregate_completed_requests_per_second": aggregate,
        "active_call_requests_per_second": ACTIVE_CALL,
        "correctness": "passed",
    }


def metadata_for(trial: dict[str, object], run_set: pathlib.Path) -> dict:
    trial_id = str(trial["trial_id"])
    runs = run_set / "runs"
    return {
        "trial_id": trial_id,
        "workload": trial["workload"],
        "worker_count": trial["worker_count"],
        "pilot": trial["pilot"],
        "accepted": True,
        "command": ["inferedge-bench-concurrent", "--format", "json"],
        "started_utc": "2026-10-12T00:00:00Z",
        "finished_utc": "2026-10-12T00:00:01Z",
        "returncode": 0,
        "stdout_path": str(runs / f"{trial_id}.stdout"),
        "stderr_path": str(runs / f"{trial_id}.stderr"),
        "problems": [],
    }


def environment(status: str = "completed") -> dict:
    plan = execution_plan()
    pilots = [str(trial["trial_id"]) for trial in plan if trial["pilot"] is True]
    preflight = {
        "queried": {
            "git_commit": COMMIT,
            "benchmark_tool_version": "0.1.0",
        },
        "verified_configuration": {
            "cmake_build_type": "Release",
            "sanitizer_instrumentation": "absent",
            "onnxruntime_version": "1.31.0",
            "add_model_path": ADD_MODEL,
            "add_model_sha256": ADD_MODEL_SHA256,
            "mnist_model_path": MNIST_MODEL,
            "mnist_model_sha256": MNIST_MODEL_SHA256,
            "mnist_fixture_id": "test_00003",
            "mnist_fixture_inputs_sha256": MNIST_INPUT_SHA256,
            "mnist_fixture_reference_sha256": MNIST_REFERENCE_SHA256,
            "mnist_fixture_manifest_sha256": MNIST_FIXTURE_MANIFEST_SHA256,
        },
        "source_defined_policy": {
            "backend": "ONNX Runtime CPU",
            "session_ownership": "one_inference_engine_per_worker",
            "onnx_intra_op_threads_per_session": 1,
            "graph_optimizations": "disabled",
        },
        "collection_protocol": {
            "workloads": ["add", "mnist"],
            "worker_counts": [1, 2, 4, 8],
            "warmup_iterations_per_worker": 100,
            "measured_iterations_per_worker": 1000,
            "correctness_check_interval": 1,
            "counted_runs_per_configuration": 5,
            "pilot_count": 8,
            "counted_trial_count": 40,
            "process_count": 48,
            "pilots": pilots,
            "trial_order": trial_order(),
            "inter_trial_delay_seconds": 5.0,
            "environment_variable": "ORT_DISABLE_TELEMETRY=1",
            "ordering_tolerance": ORDERING_TOLERANCE,
            "aggregate_relative_tolerance": AGGREGATE_RELATIVE_TOLERANCE,
        },
    }
    return public_manifest(preflight, RUN_SET_ID, status)


def dump_report(document: dict) -> str:
    return json.dumps(document) + "\n"


class Campaign:
    def __init__(self, root: pathlib.Path) -> None:
        self.run_set = root / RUN_SET_ID
        self.runs = self.run_set / "runs"
        self.manifest = environment()
        self.documents = {str(trial["trial_id"]): report_for(trial) for trial in execution_plan()}
        self.metas = {
            str(trial["trial_id"]): metadata_for(trial, self.run_set) for trial in execution_plan()
        }
        self.run_set.mkdir()
        self.runs.mkdir()
        self.rewrite_environment()
        for trial_id in self.documents:
            self.rewrite_trial(trial_id)
            self.rewrite_meta(trial_id)
            (self.runs / f"{trial_id}.stderr").write_bytes(b"")

    def rewrite_environment(self) -> None:
        text = json.dumps(self.manifest, indent=2) + "\n"
        (self.run_set / "environment.json").write_text(text, encoding="utf-8")

    def rewrite_trial(self, trial_id: str) -> None:
        text = dump_report(self.documents[trial_id])
        (self.runs / f"{trial_id}.stdout").write_text(text, encoding="utf-8")
        (self.runs / f"{trial_id}.json").write_text(text, encoding="utf-8")

    def rewrite_meta(self, trial_id: str) -> None:
        text = json.dumps(self.metas[trial_id], indent=2) + "\n"
        (self.runs / f"{trial_id}.meta.json").write_text(text, encoding="utf-8")

    def replace(self, trial_id: str, **overrides: object) -> None:
        document = self.documents[trial_id]
        for key, value in overrides.items():
            current = document.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                current.update(value)
            else:
                document[key] = value
        self.rewrite_trial(trial_id)

    def source_digest(self) -> dict[str, str]:
        digest = {}
        for path in sorted(self.run_set.rglob("*")):
            if path.is_file() and path.name != "aggregate.json" and not path.name.startswith(".aggregate."):
                digest[str(path.relative_to(self.run_set))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest


def tree_digest(root: pathlib.Path) -> dict[str, str] | None:
    if not root.exists():
        return None
    digest = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def worker_result(document: dict, workload: str, worker_count: int) -> dict:
    rows = [
        row
        for row in document["workloads"][workload]["worker_results"]
        if row["worker_count"] == worker_count
    ]
    if len(rows) != 1:
        raise AssertionError(f"{workload} worker count {worker_count} appeared {len(rows)} times")
    return rows[0]


class AnalyzeConcurrentTest(unittest.TestCase):
    def make(self) -> Campaign:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Campaign(pathlib.Path(temporary.name))

    def assert_rejected(self, text: str, mutate) -> None:
        built = self.make()
        mutate(built)
        before = built.source_digest()
        with self.assertRaises(AnalyzerError) as caught:
            analyze_and_write(built.run_set)
        self.assertIn(text, str(caught.exception))
        self.assertFalse((built.run_set / "aggregate.json").exists())
        self.assertEqual(list(built.run_set.glob(".aggregate.json.*.tmp")), [])
        self.assertEqual(built.source_digest(), before)

    def test_help_mentions_run_set(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "benchmarks.analyze_concurrent", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--run-set", completed.stdout)
        self.assertIn("not launch a benchmark", completed.stdout)

    def test_module_invocation_from_repository_root(self) -> None:
        built = self.make()
        before_baselines = tree_digest(REPO_ROOT / "benchmarks" / "baselines")
        before_experiments = tree_digest(REPO_ROOT / "models" / "experiments")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.analyze_concurrent",
                "--run-set",
                str(built.run_set),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((built.run_set / "aggregate.json").is_file())
        self.assertIn(str(built.run_set / "aggregate.json"), completed.stdout)
        self.assertEqual(tree_digest(REPO_ROOT / "benchmarks" / "baselines"), before_baselines)
        self.assertEqual(tree_digest(REPO_ROOT / "models" / "experiments"), before_experiments)

    def test_direct_script_invocation(self) -> None:
        built = self.make()
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "benchmarks" / "analyze_concurrent.py"),
                "--run-set",
                str(built.run_set),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((built.run_set / "aggregate.json").is_file())

    def test_valid_completed_campaign(self) -> None:
        built = self.make()
        destination = analyze_and_write(built.run_set)
        document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertTrue(destination.read_text(encoding="utf-8").endswith("\n"))
        self.assertEqual(document["aggregate_schema_version"], 2)
        self.assertEqual(document["source_benchmark_schema_version"], 2)
        self.assertEqual(document["run_set_id"], RUN_SET_ID)
        self.assertEqual(document["git_commit"], COMMIT)
        self.assertEqual(document["build_type"], "Release")
        self.assertTrue(document["pilots_excluded"])
        self.assertFalse(document["median_of_run_p99_is_pooled_p99"])
        self.assertFalse(document["confidence_interval_computed"])
        self.assertEqual(document["counted_runs_per_configuration"], 5)
        self.assertEqual(document["models"]["add"]["sha256"], ADD_MODEL_SHA256)
        self.assertEqual(document["models"]["mnist"]["sha256"], MNIST_MODEL_SHA256)
        self.assertIsNone(document["models"]["add"]["fixture"])
        self.assertEqual(document["models"]["mnist"]["fixture"], "test_00003")
        self.assertEqual(document["experiment_configuration"]["worker_counts"], [1, 2, 4, 8])
        self.assertEqual(document["experiment_configuration"]["warmup_iterations_per_worker"], 100)
        self.assertEqual(document["experiment_configuration"]["measured_iterations_per_worker"], 1000)
        self.assertEqual(document["experiment_configuration"]["control_worker_count"], 1)
        self.assertIn("not a pooled percentile", document["definitions"]["median_of_run_p99"])
        self.assertIn("do not establish statistical significance", document["definitions"]["confidence_interval"])
        self.assertIn("not the scaling numerator", document["definitions"]["throughput_ratio"])
        self.assertIn("Sequential active-call throughput is not a speedup baseline", document["definitions"]["throughput_ratio"])
        self.assertEqual(list(built.run_set.glob(".aggregate.json.*.tmp")), [])

    def test_exact_48_trial_identities(self) -> None:
        document = build_aggregate(self.make().run_set)
        order = document["trial_order"]
        self.assertEqual(order, trial_order())
        self.assertEqual(len(order), 48)
        self.assertEqual(len(set(order)), 48)
        self.assertNotEqual(sorted(order), order)
        self.assertEqual(order[0], "pilot-add-c01")
        self.assertEqual(order[7], "pilot-mnist-c08")
        self.assertEqual(order[8], "add-c01-01")
        self.assertEqual(order[9], "mnist-c01-01")
        self.assertEqual(order[16], "mnist-c01-02")
        self.assertEqual(order[-1], "mnist-c02-05")
        pilots = [name for name in order if name.startswith("pilot-")]
        counted = [name for name in order if not name.startswith("pilot-")]
        self.assertEqual(len(pilots), 8)
        self.assertEqual(len(counted), 40)
        self.assertEqual(document["pilot_trial_ids"], pilots)
        self.assertEqual(document["counted_trial_ids"], counted)
        self.assertEqual(document["counted_trial_ids"][1], "mnist-c01-01")
        add_one = worker_result(document, "add", 1)
        self.assertEqual(
            add_one["counted_trial_ids"],
            ["add-c01-01", "add-c01-02", "add-c01-03", "add-c01-04", "add-c01-05"],
        )
        for workload in ("add", "mnist"):
            for worker_count in (1, 2, 4, 8):
                ids = worker_result(document, workload, worker_count)["counted_trial_ids"]
                self.assertEqual(len(ids), 5)
                self.assertTrue(all(not trial_id.startswith("pilot-") for trial_id in ids))

    def test_missing_run_set(self) -> None:
        missing = pathlib.Path(tempfile.mkdtemp()) / "absent"
        with self.assertRaises(AnalyzerError) as caught:
            analyze_and_write(missing)
        self.assertIn("does not exist", str(caught.exception))

    def test_missing_environment(self) -> None:
        self.assert_rejected(
            "environment.json is missing",
            lambda built: (built.run_set / "environment.json").unlink(),
        )

    def test_status_running(self) -> None:
        def mutate(built: Campaign) -> None:
            built.manifest["status"] = "running"
            built.rewrite_environment()

        self.assert_rejected("running", mutate)

    def test_status_failed(self) -> None:
        def mutate(built: Campaign) -> None:
            built.manifest["status"] = "failed"
            built.rewrite_environment()

        self.assert_rejected("failed", mutate)

    def test_failure_json_present(self) -> None:
        def mutate(built: Campaign) -> None:
            (built.run_set / "failure.json").write_text('{"status": "failed"}\n', encoding="utf-8")

        self.assert_rejected("failure.json is present", mutate)

    def test_missing_counted_report(self) -> None:
        self.assert_rejected(
            "missing counted report: add-c02-01.json",
            lambda built: (built.runs / "add-c02-01.json").unlink(),
        )

    def test_missing_pilot_report(self) -> None:
        self.assert_rejected(
            "missing pilot report: pilot-mnist-c04.json",
            lambda built: (built.runs / "pilot-mnist-c04.json").unlink(),
        )

    def test_duplicate_trial_id(self) -> None:
        def mutate(built: Campaign) -> None:
            built.metas["add-c01-02"]["trial_id"] = "add-c01-01"
            built.rewrite_meta("add-c01-02")

        self.assert_rejected("duplicate trial identity: add-c01-01", mutate)

    def test_unexpected_trial_id(self) -> None:
        def mutate(built: Campaign) -> None:
            (built.runs / "add-c03-01.json").write_text("{}\n", encoding="utf-8")

        self.assert_rejected("unexpected trial artifact: add-c03-01.json", mutate)

    def test_incorrect_trial_order(self) -> None:
        def mutate(built: Campaign) -> None:
            order = built.manifest["collection_protocol"]["trial_order"]
            order[8], order[9] = order[9], order[8]
            built.rewrite_environment()

        self.assert_rejected("trial_order", mutate)

    def test_schema_version_1_rejection(self) -> None:
        self.assert_rejected("schema_version is 1, expected 2", lambda built: built.replace("add-c01-01", schema_version=1))

    def test_metadata_acceptance_false(self) -> None:
        def mutate(built: Campaign) -> None:
            built.metas["add-c01-01"]["accepted"] = False
            built.rewrite_meta("add-c01-01")

        self.assert_rejected("not an accepted measurement", mutate)

    def test_nonzero_return_code(self) -> None:
        def mutate(built: Campaign) -> None:
            built.metas["mnist-c08-05"]["returncode"] = 3
            built.rewrite_meta("mnist-c08-05")

        self.assert_rejected("return code is not 0", mutate)

    def test_metadata_problems_nonempty(self) -> None:
        def mutate(built: Campaign) -> None:
            built.metas["add-c04-02"]["problems"] = ["drift"]
            built.rewrite_meta("add-c04-02")

        self.assert_rejected("metadata records problems", mutate)

    def test_nonempty_stderr(self) -> None:
        self.assert_rejected(
            "stderr is not empty",
            lambda built: (built.runs / "add-c01-01.stderr").write_bytes(b"warn\n"),
        )

    def test_stdout_json_mismatch(self) -> None:
        self.assert_rejected(
            "stdout does not match the accepted report",
            lambda built: (built.runs / "add-c01-01.stdout").write_text("{}\n", encoding="utf-8"),
        )

    def test_wrong_worker_count(self) -> None:
        self.assert_rejected(
            "worker_count is 2, expected 1",
            lambda built: built.replace("add-c01-01", configuration={"worker_count": 2}),
        )

    def test_wrong_workload(self) -> None:
        self.assert_rejected(
            "workload is 'mnist', expected 'add'",
            lambda built: built.replace("add-c01-01", workload="mnist"),
        )

    def test_wrong_model_path(self) -> None:
        self.assert_rejected(
            "model is '/tmp/other.onnx'",
            lambda built: built.replace("add-c01-01", model="/tmp/other.onnx"),
        )

    def test_wrong_fixture(self) -> None:
        self.assert_rejected(
            "fixture is 'test_00000'",
            lambda built: built.replace("mnist-c01-01", fixture="test_00000"),
        )

    def test_wrong_sample_count(self) -> None:
        self.assert_rejected(
            "sample_count is 999, expected 1000",
            lambda built: built.replace("add-c01-01", latency_microseconds={"sample_count": 999}),
        )

    def test_non_finite_metrics(self) -> None:
        self.assert_rejected(
            "NaN",
            lambda built: built.replace("add-c01-01", latency_microseconds={"mean": float("nan")}),
        )

    def test_negative_latency(self) -> None:
        self.assert_rejected(
            "minimum latency is negative",
            lambda built: built.replace("add-c01-01", latency_microseconds={"minimum": -1.0}),
        )

    def test_negative_zero_standard_deviation(self) -> None:
        self.assert_rejected(
            "population standard deviation is negative",
            lambda built: built.replace(
                "add-c01-01",
                latency_microseconds={"population_standard_deviation": -0.0},
            ),
        )

    def test_invalid_percentile_order(self) -> None:
        self.assert_rejected(
            "out of order",
            lambda built: built.replace("add-c01-01", latency_microseconds={"p50": 90, "p95": 1}),
        )

    def test_zero_wall_duration(self) -> None:
        self.assert_rejected(
            "measurement wall duration is not positive",
            lambda built: built.replace("add-c01-01", measurement_wall_microseconds=0),
        )

    def test_zero_aggregate_throughput(self) -> None:
        self.assert_rejected(
            "aggregate throughput is not positive",
            lambda built: built.replace("add-c01-01", aggregate_completed_requests_per_second=0),
        )

    def test_missing_one_of_five_counted_trials(self) -> None:
        self.assert_rejected(
            "missing counted report: add-c08-05.json",
            lambda built: (built.runs / "add-c08-05.json").unlink(),
        )

    def test_pilots_excluded_from_reductions(self) -> None:
        built = self.make()
        document = build_aggregate(built.run_set)
        add_one = worker_result(document, "add", 1)
        means = [row["mean_microseconds"] for row in add_one["per_run"]]
        aggregates = [row["aggregate_completed_requests_per_second"] for row in add_one["per_run"]]
        self.assertEqual(means, [10, 11, 12, 13, 30])
        self.assertNotIn(PILOT_MEAN, means)
        self.assertNotIn(1_000_000_000.0, aggregates)
        self.assertEqual(add_one["cross_run"]["median_of_run_means_microseconds"], 12)
        self.assertEqual(add_one["cross_run"]["median_aggregate_completed_requests_per_second"], 120)
        for trial_id in document["pilot_trial_ids"]:
            self.assertTrue((built.runs / f"{trial_id}.json").is_file())
            self.assertNotIn(trial_id, add_one["counted_trial_ids"])

    def test_median_aggregate_throughput(self) -> None:
        document = build_aggregate(self.make().run_set)
        add_one = worker_result(document, "add", 1)["cross_run"]
        self.assertEqual(add_one["median_aggregate_completed_requests_per_second"], 120)
        self.assertEqual(add_one["minimum_aggregate_completed_requests_per_second"], 100)
        self.assertEqual(add_one["maximum_aggregate_completed_requests_per_second"], 200)
        self.assertNotEqual(
            add_one["median_aggregate_completed_requests_per_second"],
            (100 + 110 + 120 + 130 + 200) / 5,
        )
        self.assertEqual(
            worker_result(document, "mnist", 1)["cross_run"][
                "median_aggregate_completed_requests_per_second"
            ],
            12,
        )

    def test_median_run_p99(self) -> None:
        stats = worker_result(build_aggregate(self.make().run_set), "add", 1)["cross_run"]
        self.assertEqual(stats["median_of_run_p99_microseconds"], 42)
        self.assertNotEqual(stats["median_of_run_p99_microseconds"], (40 + 41 + 42 + 43 + 80) / 5)
        self.assertEqual(stats["median_of_run_p50_microseconds"], 12)
        self.assertEqual(stats["median_of_run_p95_microseconds"], 27)

    def test_throughput_ratios(self) -> None:
        document = build_aggregate(self.make().run_set)
        expected = {1: 1.0, 2: 2.0, 4: 4.0, 8: 4.0}
        for worker_count, ratio in expected.items():
            self.assertEqual(
                worker_result(document, "add", worker_count)["scaling"]["throughput_ratio"],
                ratio,
            )
            self.assertEqual(
                worker_result(document, "mnist", worker_count)["scaling"]["throughput_ratio"],
                ratio,
            )

    def test_scaling_efficiency(self) -> None:
        document = build_aggregate(self.make().run_set)
        expected = {1: 1.0, 2: 1.0, 4: 1.0, 8: 0.5}
        for worker_count, efficiency in expected.items():
            self.assertEqual(
                worker_result(document, "add", worker_count)["scaling"]["scaling_efficiency"],
                efficiency,
            )

    def test_mean_latency_ratios(self) -> None:
        document = build_aggregate(self.make().run_set)
        expected = {1: 1.0, 2: 2.0, 4: 3.0, 8: 4.0}
        for worker_count, ratio in expected.items():
            self.assertEqual(
                worker_result(document, "add", worker_count)["scaling"]["latency_mean_ratio"],
                ratio,
            )
        self.assertEqual(
            worker_result(document, "add", 1)["cross_run"]["median_of_run_means_microseconds"],
            12,
        )
        self.assertEqual(
            worker_result(document, "add", 1)["cross_run"]["mean_of_run_means_microseconds"],
            76 / 5,
        )

    def test_p99_latency_ratios(self) -> None:
        document = build_aggregate(self.make().run_set)
        expected = {1: 1.0, 2: 2.0, 4: 3.0, 8: 4.0}
        for worker_count, ratio in expected.items():
            self.assertEqual(
                worker_result(document, "add", worker_count)["scaling"]["latency_p99_ratio"],
                ratio,
            )

    def test_relative_throughput_range(self) -> None:
        stats = worker_result(build_aggregate(self.make().run_set), "add", 1)["cross_run"]
        self.assertEqual(stats["relative_aggregate_throughput_range"], (200 - 100) / 120)
        self.assertEqual(stats["relative_run_mean_range"], (30 - 10) / 12)
        self.assertNotEqual(stats["relative_run_mean_range"], (30 - 10) / (76 / 5))

    def test_existing_aggregate_rejection(self) -> None:
        built = self.make()
        destination = built.run_set / "aggregate.json"
        destination.write_text("SENTINEL\n", encoding="utf-8")
        before = built.source_digest()
        with self.assertRaises(AnalyzerError) as caught:
            analyze_and_write(built.run_set)
        self.assertIn("already exists", str(caught.exception))
        self.assertEqual(destination.read_text(encoding="utf-8"), "SENTINEL\n")
        self.assertEqual(built.source_digest(), before)
        self.assertEqual(list(built.run_set.glob(".aggregate.json.*.tmp")), [])

    def test_atomic_publish_failure(self) -> None:
        built = self.make()
        before = built.source_digest()
        with mock.patch("benchmarks.analyze_concurrent.os.link", side_effect=OSError("disk full")):
            with self.assertRaises(AnalyzerError) as caught:
                analyze_and_write(built.run_set)
        self.assertIn("failed to publish", str(caught.exception))
        self.assertFalse((built.run_set / "aggregate.json").exists())
        self.assertEqual(list(built.run_set.glob(".aggregate.json.*.tmp")), [])
        self.assertEqual(built.source_digest(), before)

    def test_no_source_report_modification(self) -> None:
        built = self.make()
        before = built.source_digest()
        environment_bytes = (built.run_set / "environment.json").read_bytes()
        report_bytes = (built.runs / "add-c01-01.json").read_bytes()
        stdout_bytes = (built.runs / "add-c01-01.stdout").read_bytes()
        analyze_and_write(built.run_set)
        self.assertEqual(built.source_digest(), before)
        self.assertEqual((built.run_set / "environment.json").read_bytes(), environment_bytes)
        self.assertEqual((built.runs / "add-c01-01.json").read_bytes(), report_bytes)
        self.assertEqual((built.runs / "add-c01-01.stdout").read_bytes(), stdout_bytes)
        self.assertEqual(report_bytes, stdout_bytes)
        self.assertTrue((built.run_set / "aggregate.json").is_file())

    def test_active_call_is_not_the_scaling_numerator(self) -> None:
        document = build_aggregate(self.make().run_set)
        add_one = worker_result(document, "add", 1)
        add_two = worker_result(document, "add", 2)
        self.assertEqual(add_one["cross_run"]["median_active_call_requests_per_second"], ACTIVE_CALL)
        self.assertEqual(add_two["cross_run"]["median_active_call_requests_per_second"], ACTIVE_CALL)
        self.assertEqual(add_two["scaling"]["throughput_ratio"], 2.0)
        self.assertNotEqual(add_two["scaling"]["throughput_ratio"], 1.0)

    def test_worker_count_one_invariants(self) -> None:
        document = build_aggregate(self.make().run_set)
        for workload in ("add", "mnist"):
            scaling = worker_result(document, workload, 1)["scaling"]
            self.assertEqual(scaling["throughput_ratio"], 1.0)
            self.assertEqual(scaling["scaling_efficiency"], 1.0)
            self.assertEqual(scaling["latency_mean_ratio"], 1.0)
            self.assertEqual(scaling["latency_p99_ratio"], 1.0)

    def test_zero_median_mean_denominator(self) -> None:
        def mutate(built: Campaign) -> None:
            for run_number in range(1, 6):
                built.replace(
                    f"add-c01-{run_number:02d}",
                    latency_microseconds={
                        "minimum": 0.0,
                        "mean": 0.0,
                        "p50": 0.0,
                        "p95": 0.0,
                        "p99": 0.0,
                        "maximum": 0.0,
                        "population_standard_deviation": 0.0,
                    },
                )

        self.assert_rejected("denominator is not positive", mutate)

    def test_metadata_worker_count_mismatch(self) -> None:
        def mutate(built: Campaign) -> None:
            built.metas["add-c01-01"]["worker_count"] = 4
            built.rewrite_meta("add-c01-01")

        self.assert_rejected("worker_count metadata", mutate)

    def test_workloads_stay_independent(self) -> None:
        document = build_aggregate(self.make().run_set)
        add_mean = worker_result(document, "add", 1)["cross_run"]["median_of_run_means_microseconds"]
        mnist_mean = worker_result(document, "mnist", 1)["cross_run"]["median_of_run_means_microseconds"]
        self.assertEqual(add_mean, 12)
        self.assertEqual(mnist_mean, 1012)
        self.assertNotEqual(add_mean, mnist_mean)

    def test_main_reports_analyzer_error(self) -> None:
        built = self.make()
        (built.run_set / "environment.json").unlink()
        stderr = __import__("io").StringIO()
        with mock.patch("sys.stderr", stderr):
            code = main(["--run-set", str(built.run_set)])
        self.assertEqual(code, 1)
        self.assertIn("error:", stderr.getvalue())
        self.assertIn("environment.json is missing", stderr.getvalue())
        self.assertFalse((built.run_set / "aggregate.json").exists())

    def test_environment_aggregate_field_must_stay_null(self) -> None:
        def mutate(built: Campaign) -> None:
            built.manifest["aggregate"] = {"status": "estimated"}
            built.rewrite_environment()

        self.assert_rejected("aggregate field is not null", mutate)

    def test_invalid_pilot_report_is_rejected(self) -> None:
        self.assert_rejected(
            "pilot-add-c01: schema_version is 1, expected 2",
            lambda built: built.replace("pilot-add-c01", schema_version=1),
        )

    def test_wrong_warmup_protocol_is_rejected(self) -> None:
        def mutate(built: Campaign) -> None:
            built.manifest["collection_protocol"]["warmup_iterations_per_worker"] = 10
            built.rewrite_environment()

        self.assert_rejected("warmup_iterations_per_worker", mutate)

    def test_link_race_does_not_overwrite(self) -> None:
        built = self.make()
        destination = built.run_set / "aggregate.json"

        def publish_existing(source, dest) -> None:
            pathlib.Path(dest).write_text("SENTINEL\n", encoding="utf-8")
            raise FileExistsError(dest)

        before = built.source_digest()
        with mock.patch("benchmarks.analyze_concurrent.os.link", side_effect=publish_existing):
            with self.assertRaises(AnalyzerError) as caught:
                analyze_and_write(built.run_set)
        self.assertIn("already exists", str(caught.exception))
        self.assertEqual(destination.read_text(encoding="utf-8"), "SENTINEL\n")
        self.assertEqual(list(built.run_set.glob(".aggregate.json.*.tmp")), [])
        self.assertEqual(built.source_digest(), before)


if __name__ == "__main__":
    unittest.main()
