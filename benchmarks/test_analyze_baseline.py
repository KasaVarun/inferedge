#!/usr/bin/env python3
"""Synthetic tests for cross-run baseline analysis. They do not run inferedge-bench."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.analyze_baseline import (  # noqa: E402
    AnalyzerError,
    analyze_and_write,
    build_aggregate,
)
from benchmarks.collect_baseline import (  # noqa: E402
    ADD_MODEL_SHA256,
    MNIST_MODEL_SHA256,
    public_manifest,
    trial_ids,
)

RUN_SET_ID = "20261012T000000Z-abcdef0"
COMMIT = "6c1ba833a512e288b436e06a0bd430b41988518b"
ADD_MODEL = "/models/add_f32.onnx"
MNIST_MODEL = "/models/mnist_mlp.onnx"
ADD_MEANS = [10, 12, 11, 13, 14]
ADD_P50 = [20, 22, 21, 23, 24]
ADD_P95 = [30, 32, 31, 33, 34]
ADD_P99 = [40, 41, 42, 43, 80]
ADD_ACTIVE = [100, 110, 120, 130, 200]
ADD_OBSERVED = [50, 60, 70, 80, 200]


def report(workload: str, index: int, **overrides: object) -> dict:
    offset = 0 if workload == "add" else 100
    mean = ADD_MEANS[index] + offset
    p50 = ADD_P50[index] + offset
    p95 = ADD_P95[index] + offset
    p99 = ADD_P99[index] + offset
    document = {
        "schema_version": 1,
        "tool": {"name": "inferedge-bench", "version": "0.1.0"},
        "model": ADD_MODEL if workload == "add" else MNIST_MODEL,
        "workload": workload,
        "fixture": None if workload == "add" else "test_00003",
        "backend": "ONNX Runtime CPU",
        "configuration": {
            "warmup_iterations": 100,
            "measured_iterations": 1000,
            "correctness_check_interval": 1,
            "calling_threads": 1,
            "onnx_intra_op_threads": 1,
            "graph_optimizations": "disabled",
        },
        "latency_microseconds": {
            "sample_count": 1000,
            "minimum": 1.0,
            "mean": mean,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "maximum": p99 + 1,
            "population_standard_deviation": 0.25,
        },
        "measurement_loop_microseconds": 500.0,
        "throughput_requests_per_second": {
            "active_call": ADD_ACTIVE[index] + offset,
            "observed_sequential": ADD_OBSERVED[index] + offset,
        },
        "correctness": "passed",
    }
    for key, value in overrides.items():
        if key in document and isinstance(document[key], dict) and isinstance(value, dict):
            document[key].update(value)
        else:
            document[key] = value
    return document


def dump_report(document: dict) -> str:
    return json.dumps(document) + "\n"


def environment(status: str = "completed") -> dict:
    preflight = {
        "queried": {
            "git_commit": COMMIT,
            "git_status_porcelain": "?? models/experiments/\n",
            "approved_untracked_paths": ["models/experiments"],
            "started_utc": "2026-10-12T00:00:00Z",
            "operating_system": "Darwin",
            "platform": "macOS-test",
            "macos_version": None,
            "hardware_model": None,
            "hardware_model_error": None,
            "cpu_architecture": "arm64",
            "compiler_path": "/usr/bin/c++",
            "compiler_version": "c++ test\n",
            "cmake_version": "cmake version 3.24.0\n",
            "benchmark_binary_path": "/build-baseline/inferedge-bench",
            "benchmark_binary_sha256": "ab" * 32,
            "benchmark_tool_version": "0.1.0",
            "onnxruntime_link_output": "libonnxruntime.1.31.0.dylib\n",
        },
        "verified_configuration": {
            "cmake_build_type": "Release",
            "cmake_cxx_flags": "",
            "cmake_cxx_flags_release": "",
            "cmake_exe_linker_flags": "",
            "cmake_exe_linker_flags_release": "",
            "sanitizer_instrumentation": "absent",
            "onnxruntime_version": "1.31.0",
            "add_model_path": ADD_MODEL,
            "add_model_sha256": ADD_MODEL_SHA256,
            "mnist_model_path": MNIST_MODEL,
            "mnist_model_sha256": MNIST_MODEL_SHA256,
            "mnist_fixture_directory": "/fixtures/mnist",
            "mnist_fixture_id": "test_00003",
        },
        "source_defined_policy": {
            "backend": "ONNX Runtime CPU",
            "calling_threads": 1,
            "onnx_intra_op_threads": 1,
            "graph_optimizations": "disabled",
            "onnxruntime_version_pin": "1.31.0",
            "allowed_untracked_roots": ["models/experiments"],
            "note": "source policy",
        },
        "collection_protocol": {
            "workloads": ["add", "mnist"],
            "warmup_iterations": 100,
            "measured_iterations": 1000,
            "correctness_check_interval": 1,
            "counted_runs_per_workload": 5,
            "pilots": ["pilot-add", "pilot-mnist"],
            "trial_order": trial_ids(5),
            "inter_trial_delay_seconds": 5.0,
            "environment_variable": "ORT_DISABLE_TELEMETRY=1",
        },
    }
    return public_manifest(preflight, RUN_SET_ID, status)


def metadata(trial_id: str) -> dict:
    return {
        "trial_id": trial_id,
        "accepted": True,
        "command": ["inferedge-bench", "--workload", trial_id.split("-")[0], "--format", "json"],
        "started_utc": "2026-10-12T00:00:00Z",
        "finished_utc": "2026-10-12T00:00:01Z",
        "returncode": 0,
        "stdout_path": f"/tmp/{trial_id}.stdout",
        "stderr_path": f"/tmp/{trial_id}.stderr",
        "problems": [],
    }


class RunSetBuilder:
    def __init__(self, root: pathlib.Path) -> None:
        self.run_set = root / RUN_SET_ID
        self.runs = self.run_set / "runs"
        self.runs.mkdir(parents=True)
        self.reports: dict[str, str] = {}
        self.metas: dict[str, dict] = {}
        for workload in ("add", "mnist"):
            for index in range(5):
                trial_id = f"{workload}-{index + 1:02d}"
                self.reports[trial_id] = dump_report(report(workload, index))
                self.metas[trial_id] = metadata(trial_id)
        for trial_id, workload in (("pilot-add", "add"), ("pilot-mnist", "mnist")):
            pilot = report(workload, 0)
            pilot["latency_microseconds"]["mean"] = 1000000
            pilot["latency_microseconds"]["maximum"] = 1000001
            self.reports[trial_id] = dump_report(pilot)
            self.metas[trial_id] = metadata(trial_id)
        self.manifest = environment()
        self.write()

    def write(self) -> None:
        (self.run_set / "environment.json").write_text(
            json.dumps(self.manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        for trial_id, text in self.reports.items():
            (self.runs / f"{trial_id}.json").write_text(text, encoding="utf-8")
        for trial_id, meta in self.metas.items():
            (self.runs / f"{trial_id}.meta.json").write_text(
                json.dumps(meta, indent=2) + "\n",
                encoding="utf-8",
            )

    def source_digest(self) -> dict[str, str]:
        digest = {}
        for path in sorted(self.run_set.rglob("*")):
            if path.is_file() and path.name != "aggregate.json" and not path.name.startswith(".aggregate."):
                digest[str(path.relative_to(self.run_set))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest


def add_cross_run(run_set: pathlib.Path) -> dict:
    return build_aggregate(run_set)["workloads"]["add"]["cross_run"]


class AnalyzeBaselineTest(unittest.TestCase):
    def make(self) -> RunSetBuilder:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return RunSetBuilder(pathlib.Path(temporary.name))

    def test_complete_valid_run_set(self) -> None:
        built = self.make()
        destination = analyze_and_write(built.run_set)
        document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(document["aggregate_schema_version"], 1)
        self.assertEqual(document["source_benchmark_schema_version"], 1)
        self.assertEqual(document["run_set_id"], RUN_SET_ID)
        self.assertEqual(document["git_commit"], COMMIT)
        self.assertEqual(document["build_type"], "Release")
        self.assertEqual(document["models"]["add"]["sha256"], ADD_MODEL_SHA256)
        self.assertEqual(document["models"]["mnist"]["sha256"], MNIST_MODEL_SHA256)
        self.assertEqual(document["models"]["mnist"]["fixture"], "test_00003")
        self.assertIsNone(document["models"]["add"]["fixture"])
        self.assertEqual(document["counted_runs_per_workload"], 5)
        self.assertTrue(document["pilots_excluded"])
        self.assertFalse(document["median_of_run_p99_is_pooled_p99"])
        self.assertFalse(document["confidence_interval_computed"])
        self.assertEqual(document["experiment_configuration"]["measured_iterations"], 1000)
        self.assertEqual(document["workloads"]["mnist"]["cross_run"]["median_of_run_means_microseconds"], 112)
        self.assertNotIn("pilot-add", document["workloads"]["add"]["counted_trial_ids"])

    def test_missing_environment_manifest(self) -> None:
        built = self.make()
        (built.run_set / "environment.json").unlink()
        with self.assertRaises(AnalyzerError) as caught:
            build_aggregate(built.run_set)
        self.assertIn("environment.json is missing", str(caught.exception))

    def test_collection_status_running(self) -> None:
        built = self.make()
        built.manifest["status"] = "running"
        built.write()
        with self.assertRaises(AnalyzerError) as caught:
            build_aggregate(built.run_set)
        self.assertIn("running", str(caught.exception))

    def test_collection_status_failed(self) -> None:
        built = self.make()
        built.manifest["status"] = "failed"
        built.write()
        with self.assertRaises(AnalyzerError) as caught:
            build_aggregate(built.run_set)
        self.assertIn("failed", str(caught.exception))

    def test_missing_add_report(self) -> None:
        built = self.make()
        (built.runs / "add-03.json").unlink()
        with self.assertRaises(AnalyzerError) as caught:
            build_aggregate(built.run_set)
        self.assertIn("missing counted report: add-03.json", str(caught.exception))
        self.assertFalse((built.run_set / "aggregate.json").exists())

    def test_missing_mnist_report(self) -> None:
        built = self.make()
        (built.runs / "mnist-04.json").unlink()
        with self.assertRaises(AnalyzerError) as caught:
            build_aggregate(built.run_set)
        self.assertIn("missing counted report: mnist-04.json", str(caught.exception))

    def test_duplicate_trial_identity(self) -> None:
        built = self.make()
        built.metas["add-02"]["trial_id"] = "add-01"
        built.write()
        with self.assertRaises(AnalyzerError) as caught:
            build_aggregate(built.run_set)
        self.assertIn("duplicate trial identity: add-01", str(caught.exception))

    def test_wrong_schema_version(self) -> None:
        self._replace("add-01", schema_version=2)
        self._assert_problem("schema_version")

    def test_wrong_workload(self) -> None:
        self._replace("add-01", workload="bert")
        self._assert_problem("workload")

    def test_wrong_fixture(self) -> None:
        self._replace("mnist-01", fixture="test_00000")
        self._assert_problem("fixture")

    def test_wrong_model_path(self) -> None:
        self._replace("add-01", model="/tmp/other.onnx")
        self._assert_problem("model")

    def test_unexpected_iteration_count(self) -> None:
        self._replace("add-01", configuration={"measured_iterations": 10})
        self._assert_problem("measured_iterations")

    def test_non_finite_metrics(self) -> None:
        built = self.make()
        text = built.reports["add-01"].replace('"mean": 10', '"mean": NaN')
        (built.runs / "add-01.json").write_text(text, encoding="utf-8")
        with self.assertRaises(AnalyzerError) as caught:
            build_aggregate(built.run_set)
        self.assertTrue("NaN" in str(caught.exception) or "not valid JSON" in str(caught.exception))

    def test_negative_standard_deviation(self) -> None:
        self._replace("add-01", latency_microseconds={"population_standard_deviation": -1.0})
        self._assert_problem("standard deviation")

    def test_negative_zero_standard_deviation(self) -> None:
        self._replace("add-01", latency_microseconds={"population_standard_deviation": -0.0})
        self._assert_problem("standard deviation")

    def test_wrong_benchmark_backend(self) -> None:
        self._replace("mnist-02", backend="CUDA")
        self._assert_problem("backend")

    def test_incorrect_percentile_ordering(self) -> None:
        self._replace("add-04", latency_microseconds={"p50": 90, "p95": 1})
        self._assert_problem("out of order")

    def test_mean_of_run_means(self) -> None:
        self.assertEqual(add_cross_run(self.make().run_set)["mean_of_run_means_microseconds"], 12)

    def test_median_of_run_means(self) -> None:
        self.assertEqual(add_cross_run(self.make().run_set)["median_of_run_means_microseconds"], 12)

    def test_median_of_run_p50(self) -> None:
        self.assertEqual(add_cross_run(self.make().run_set)["median_of_run_p50_microseconds"], 22)

    def test_median_of_run_p95(self) -> None:
        self.assertEqual(add_cross_run(self.make().run_set)["median_of_run_p95_microseconds"], 32)

    def test_median_of_run_p99(self) -> None:
        stats = add_cross_run(self.make().run_set)
        self.assertEqual(stats["median_of_run_p99_microseconds"], 42)
        self.assertNotEqual(stats["median_of_run_p99_microseconds"], (40 + 41 + 42 + 43 + 80) / 5)

    def test_median_active_call_throughput(self) -> None:
        value = add_cross_run(self.make().run_set)["median_active_call_throughput_requests_per_second"]
        self.assertEqual(value, 120)
        self.assertNotEqual(value, (100 + 110 + 120 + 130 + 200) / 5)

    def test_median_observed_throughput(self) -> None:
        value = add_cross_run(self.make().run_set)[
            "median_observed_sequential_throughput_requests_per_second"
        ]
        self.assertEqual(value, 70)

    def test_minimum_and_maximum_run_means(self) -> None:
        stats = add_cross_run(self.make().run_set)
        self.assertEqual(stats["minimum_run_mean_microseconds"], 10)
        self.assertEqual(stats["maximum_run_mean_microseconds"], 14)

    def test_relative_variation(self) -> None:
        self.assertEqual(add_cross_run(self.make().run_set)["relative_run_mean_range"], 4 / 12)

    def test_existing_aggregate_protection(self) -> None:
        built = self.make()
        destination = built.run_set / "aggregate.json"
        destination.write_text("SENTINEL\n", encoding="utf-8")
        with self.assertRaises(AnalyzerError) as caught:
            analyze_and_write(built.run_set)
        self.assertIn("already exists", str(caught.exception))
        self.assertEqual(destination.read_text(encoding="utf-8"), "SENTINEL\n")

    def test_atomic_output_behavior(self) -> None:
        built = self.make()
        with mock.patch("benchmarks.analyze_baseline.os.link", side_effect=OSError("disk full")):
            with self.assertRaises(AnalyzerError) as caught:
                analyze_and_write(built.run_set)
        self.assertIn("failed to publish", str(caught.exception))
        self.assertFalse((built.run_set / "aggregate.json").exists())
        self.assertEqual(list(built.run_set.glob(".aggregate.json.*.tmp")), [])

        destination = analyze_and_write(built.run_set)
        document = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(document["workloads"]["add"]["cross_run"]["mean_of_run_means_microseconds"], 12)
        self.assertEqual(list(built.run_set.glob(".aggregate.json.*.tmp")), [])
        self.assertTrue(destination.read_text(encoding="utf-8").endswith("\n"))

    def test_pilot_exclusion(self) -> None:
        built = self.make()
        document = build_aggregate(built.run_set)
        self.assertEqual(document["pilot_trial_ids"], ["pilot-add", "pilot-mnist"])
        self.assertEqual(
            [row["trial_id"] for row in document["workloads"]["add"]["per_run"]],
            ["add-01", "add-02", "add-03", "add-04", "add-05"],
        )
        self.assertEqual(document["workloads"]["add"]["cross_run"]["median_of_run_means_microseconds"], 12)
        self.assertNotIn(1000000, [row["mean_microseconds"] for row in document["workloads"]["add"]["per_run"]])

    def test_no_source_report_modification(self) -> None:
        built = self.make()
        before = built.source_digest()
        analyze_and_write(built.run_set)
        self.assertEqual(built.source_digest(), before)
        self.assertTrue((built.run_set / "aggregate.json").is_file())

    def test_zero_relative_variation_denominator(self) -> None:
        built = self.make()
        for index in range(5):
            document = report("add", index)
            document["latency_microseconds"].update(
                {"minimum": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "maximum": 0}
            )
            built.reports[f"add-{index + 1:02d}"] = dump_report(document)
        built.write()
        with self.assertRaises(AnalyzerError) as caught:
            build_aggregate(built.run_set)
        self.assertIn("denominator is not positive", str(caught.exception))
        self.assertFalse((built.run_set / "aggregate.json").exists())

    def _replace(self, trial_id: str, **overrides: object) -> None:
        built = self.make()
        workload = trial_id.split("-")[0]
        index = int(trial_id.split("-")[1]) - 1
        document = report(workload, index)
        for key, value in overrides.items():
            if key in document and isinstance(document[key], dict) and isinstance(value, dict):
                document[key].update(value)
            else:
                document[key] = value
        built.reports[trial_id] = dump_report(document)
        built.write()
        self.built = built

    def _assert_problem(self, needle: str) -> None:
        with self.assertRaises(AnalyzerError) as caught:
            build_aggregate(self.built.run_set)
        self.assertIn(needle, str(caught.exception))
        self.assertFalse((self.built.run_set / "aggregate.json").exists())


if __name__ == "__main__":
    unittest.main()
