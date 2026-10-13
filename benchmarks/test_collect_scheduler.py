#!/usr/bin/env python3
"""Deterministic tests for the schema-3 scheduler collector."""

from __future__ import annotations

import json
import os
import pathlib
import platform
import shutil
import tempfile
import unittest

from benchmarks.collect_baseline import ProcessResult, git_status_problems
from benchmarks.collect_scheduler import (
    ALLOWED_UNTRACKED,
    BINARY_NAME,
    CELL_BY_ID,
    MINIMUM_SUBMIT_FRACTION,
    MODEL_RELATIVE,
    MODEL_SHA256,
    PROFILE_BY_ID,
    TOOL_NAME,
    TOOL_VERSION,
    CollectorError,
    SupervisedResult,
    classify_report,
    collect,
    execution_plan,
    preflight,
    supervise_process,
    trial_order,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMIT = "9a77f685c1d9d5091763caccc48ded1a4d08786b"


def dump_report(document: dict) -> str:
    return json.dumps(document) + "\n"


def expected_launch(**overrides) -> dict:
    value = {
        "model_path": "/models/batch_add_f32.onnx",
        "workload": "open_loop_constant",
        "worker_count": 1,
        "queue_capacity": 64,
        "enable_batching": False,
        "max_batch_size": 1,
        "max_batch_delay_ms": 0,
        "warmup_requests": 0,
        "check_interval": 1,
        "duration_ms": 1000,
        "lag_tolerance_ms": 1000,
        "comparison_class": "primary",
    }
    value.update(overrides)
    return value


def sample_report(**overrides) -> dict:
    histogram = [0, 100] + [0] * 31
    document = {
        "schema_version": 3,
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "valid": True,
        "invalid_reasons": [],
        "model": "/models/batch_add_f32.onnx",
        "workload": "open_loop_constant",
        "e2e_latency_definition": (
            "steady_clock immediately before submit() until the future is observed ready"
        ),
        "completion_observation": "open_loop_poll",
        "observation_quantum_nanoseconds": 1_000_000,
        "scheduler": {
            "worker_count": 1,
            "queue_capacity": 64,
            "enable_batching": False,
            "max_batch_size": 1,
            "max_batch_delay_ms": 0,
            "enable_metrics": True,
        },
        "generator": {
            "lag_definition": "actual_submit_start_minus_scheduled_arrival",
            "lag_policy": "stop_and_record",
            "outstanding_policy": "skip_opportunity",
            "warmup_requests": 0,
            "warmup_separate_scheduler": True,
            "check_interval": 1,
        },
        "configured_duration_nanoseconds": 1_000_000_000,
        "scheduled_span_nanoseconds": 990_000_000,
        "arrival_stop_reason": "configured_duration",
        "arrival_window_nanoseconds": 1_000_000_000,
        "completion_nanoseconds": 1_000_000,
        "total_trial_nanoseconds": 1_001_000_000,
        "completion_timeout_bound": "future_observation_only",
        "completion_timeout_note": "observation only",
        "outstanding_cap_note": "cap skips are not rejections",
        "scheduled_opportunities": 100,
        "submit_attempts": 100,
        "generator_skips_outstanding": 0,
        "generator_skips_lag": 0,
        "accepted": 100,
        "rejections": {"full": 0, "closed": 0, "not_running": 0},
        "succeeded": 100,
        "failed_inference": 0,
        "failed_worker": 0,
        "failed_scheduler": 0,
        "canceled": 0,
        "timeouts": 0,
        "generator_lag": {
            "count": 100,
            "minimum_nanoseconds": 0,
            "maximum_nanoseconds": 1000,
            "mean_nanoseconds": 10.0,
        },
        "successful_e2e_latency": {
            "sample_count": 100,
            "minimum_nanoseconds": 1000,
            "maximum_nanoseconds": 5000,
            "mean_microseconds": 2.0,
            "p50_microseconds": 2.0,
            "p95_microseconds": 4.0,
            "p99_microseconds": 4.5,
            "percentiles_reported": True,
            "precise_tail": True,
        },
        "throughput": {
            "arrival_window_seconds": 1.0,
            "completion_seconds": 0.001,
            "total_trial_seconds": 1.001,
            "offered_opportunities_per_arrival_window_second": 100.0,
            "submit_attempts_per_arrival_window_second": 100.0,
            "accepted_per_arrival_window_second": 100.0,
            "successful_attempts_per_arrival_window_second": 100.0,
            "successful_per_total_trial_second": 99.9,
            "successful_attempts_per_arrival_window_second_note": (
                "offered-cohort rate; not steady-state completion capacity"
            ),
            "successful_per_total_trial_second_note": (
                "measured successes divided by the arrival window plus the completion drain"
            ),
        },
        "metrics": {
            "enabled": True,
            "finalized": True,
            "engine_run_count": 100,
            "requests_entered_engine_runs": 100,
            "batch_size_histogram": histogram,
            "total_assembly_nanoseconds": 0,
            "total_engine_run_nanoseconds": 1000,
            "total_split_nanoseconds": 0,
            "histogram_saturated": False,
        },
        "correctness_checked": 100,
        "correctness_mismatches": 0,
        "error_count": 0,
        "errors": [],
        "burst_phases": [
            {
                "index": 0,
                "requests_per_second": 50.0,
                "duration_nanoseconds": 1_000_000_000,
                "scheduled_opportunities": 100,
            }
        ],
        "queue_depth": 0,
        "executing": 0,
        "live_workers": 0,
        "exceptional_promise_backstop": "destructor completions are absent from scheduler counters",
    }
    document.update(overrides)
    return document


def problems_for(document: dict, **expected_overrides) -> list[str]:
    _report, status = classify_report(
        dump_report(document),
        expected=expected_launch(**expected_overrides),
    )
    return list(status["problems"]) + list(status["publication_problems"])


class ReportValidationTest(unittest.TestCase):
    def test_valid_delivery_fraction_is_publication_eligible(self) -> None:
        _report, status = classify_report(dump_report(sample_report()), expected=expected_launch())
        self.assertEqual(status["problems"], [])
        self.assertTrue(status["runner_valid"])
        self.assertTrue(status["collector_valid"])
        self.assertTrue(status["publication_eligible"])
        self.assertTrue(status["steady_state_eligible"])
        self.assertEqual(status["delivery_fraction"], 1.0)
        self.assertGreaterEqual(status["delivery_fraction"], MINIMUM_SUBMIT_FRACTION)

    def test_wrong_schema_is_rejected(self) -> None:
        document = sample_report(schema_version=2)
        self.assertTrue(any("schema_version" in item for item in problems_for(document)))

    def test_wrong_count_type_is_rejected(self) -> None:
        document = sample_report(succeeded=100.5)
        self.assertTrue(any("succeeded" in item for item in problems_for(document)))

    def test_boolean_is_not_an_integer_count(self) -> None:
        document = sample_report(succeeded=True)
        self.assertTrue(any("succeeded" in item for item in problems_for(document)))

    def test_submit_fraction_at_the_publication_boundary(self) -> None:
        document = sample_report(submit_attempts=99, generator_skips_outstanding=1, accepted=99, succeeded=99)
        document["successful_e2e_latency"]["sample_count"] = 99
        document["successful_e2e_latency"]["precise_tail"] = False
        document["metrics"]["engine_run_count"] = 99
        document["metrics"]["requests_entered_engine_runs"] = 99
        document["metrics"]["batch_size_histogram"] = [0, 99] + [0] * 31
        document["correctness_checked"] = 99
        _report, status = classify_report(dump_report(document), expected=expected_launch())
        self.assertTrue(status["publication_eligible"])
        self.assertEqual(status["delivery_fraction"], 0.99)

    def test_nonfinite_value_is_rejected(self) -> None:
        text = dump_report(sample_report()).replace("10.0", "Infinity", 1)
        _report, status = classify_report(text, expected=expected_launch())
        self.assertTrue(status["problems"])
        self.assertFalse(status["collector_valid"])

    def test_accounting_mismatch_is_rejected(self) -> None:
        document = sample_report(accepted=99)
        self.assertTrue(any("accepted" in item for item in problems_for(document)))

    def test_histogram_mismatch_is_rejected(self) -> None:
        document = sample_report()
        document["metrics"]["engine_run_count"] = 90
        self.assertTrue(any("engine_run_count" in item for item in problems_for(document)))

    def test_lag_stop_is_not_collector_valid(self) -> None:
        document = sample_report(
            valid=False,
            invalid_reasons=["generator_lag"],
            arrival_stop_reason="generator_lag",
            generator_skips_lag=100,
            submit_attempts=0,
            accepted=0,
            succeeded=0,
        )
        document["successful_e2e_latency"]["sample_count"] = 0
        document["successful_e2e_latency"]["percentiles_reported"] = False
        document["metrics"]["engine_run_count"] = 0
        document["metrics"]["requests_entered_engine_runs"] = 0
        document["metrics"]["batch_size_histogram"] = [0] * 33
        document["burst_phases"][0]["scheduled_opportunities"] = 100
        document["correctness_checked"] = 0
        found = problems_for(document)
        self.assertTrue(any("runner valid" in item or "lag" in item for item in found))

    def test_outstanding_suppression_above_one_percent_is_not_publishable(self) -> None:
        document = sample_report(submit_attempts=98, generator_skips_outstanding=2, accepted=98, succeeded=98)
        document["successful_e2e_latency"]["sample_count"] = 98
        document["metrics"]["engine_run_count"] = 98
        document["metrics"]["requests_entered_engine_runs"] = 98
        document["metrics"]["batch_size_histogram"] = [0, 98] + [0] * 31
        document["correctness_checked"] = 98
        _report, status = classify_report(dump_report(document), expected=expected_launch())
        self.assertTrue(status["runner_valid"])
        self.assertTrue(status["collector_valid"])
        self.assertFalse(status["publication_eligible"])
        self.assertTrue(any("submit fraction" in item for item in status["publication_problems"]))

    def test_early_max_offered_is_bounded_not_steady_state(self) -> None:
        document = sample_report(arrival_stop_reason="max_offered", scheduled_span_nanoseconds=1000)
        _report, status = classify_report(dump_report(document), expected=expected_launch())
        self.assertTrue(status["collector_valid"])
        self.assertTrue(status["publication_eligible"])
        self.assertFalse(status["steady_state_eligible"])
        self.assertEqual(status["window_class"], "bounded_max_offered")

    def test_wrong_observation_method_is_rejected(self) -> None:
        document = sample_report(completion_observation="closed_loop_producer_wait")
        self.assertTrue(any("completion_observation" in item for item in problems_for(document)))


class PlanTest(unittest.TestCase):
    def test_rotated_order_and_five_counted_trials(self) -> None:
        cells = [CELL_BY_ID["A"], CELL_BY_ID["B"]]
        profiles = [PROFILE_BY_ID["low_open"]]
        order = trial_order(cells, profiles, 5)
        self.assertEqual(
            order,
            [
                "pilot-A-low_open",
                "pilot-B-low_open",
                "A-low_open-01",
                "B-low_open-01",
                "B-low_open-02",
                "A-low_open-02",
                "A-low_open-03",
                "B-low_open-03",
                "B-low_open-04",
                "A-low_open-04",
                "A-low_open-05",
                "B-low_open-05",
            ],
        )
        counted = [trial for trial in execution_plan(cells, profiles, 5) if not trial["pilot"]]
        self.assertEqual(len(counted), 10)
        self.assertTrue(all(trial["run_number"] in range(1, 6) for trial in counted))

    def test_open_and_closed_profiles_cannot_share_a_plan_family(self) -> None:
        from benchmarks.collect_scheduler import require_one_family

        with self.assertRaises(CollectorError):
            require_one_family([PROFILE_BY_ID["low_open"], PROFILE_BY_ID["closed_optional"]])


class GitPolicyTest(unittest.TestCase):
    def test_private_roots_are_allowed_and_other_untracked_paths_are_not(self) -> None:
        porcelain = "\n".join(
            [
                "?? benchmarks/baselines/",
                "?? benchmarks/concurrent_runs/",
                "?? benchmarks/scheduler_runs/",
                "?? models/experiments/",
                "?? notes/unexpected.txt",
            ]
        )
        problems = git_status_problems(
            porcelain,
            expected_commit=COMMIT,
            actual_commit=COMMIT,
            allowed_untracked=ALLOWED_UNTRACKED,
        )
        self.assertEqual(problems, ["unexpected untracked path: notes/unexpected.txt"])

    def test_dirty_tracked_worktree_is_rejected(self) -> None:
        problems = git_status_problems(
            " M CMakeLists.txt\n",
            expected_commit=COMMIT,
            actual_commit=COMMIT,
            allowed_untracked=ALLOWED_UNTRACKED,
        )
        self.assertTrue(any("tracked worktree change" in item for item in problems))

    def test_dirty_index_is_rejected(self) -> None:
        problems = git_status_problems(
            "M  CMakeLists.txt\n",
            expected_commit=COMMIT,
            actual_commit=COMMIT,
            allowed_untracked=ALLOWED_UNTRACKED,
        )
        self.assertTrue(any("tracked worktree change" in item for item in problems))


class World:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.build_dir = root / "build-release"
        self.runtime = root / "onnxruntime-cpu"
        self.binary = self.build_dir / BINARY_NAME
        self.compiler = "/usr/bin/c++"
        self.head = COMMIT
        self.porcelain = ""
        self.build_type = "Release"
        self.cxx_flags = ""
        self.release_flags = "-O3 -DNDEBUG"
        self.nm_output = ""
        self.version_stdout = f"{TOOL_NAME} {TOOL_VERSION}\n"
        self.fail_timeout = False
        self.bad_model = False
        self.trial_calls: list[list[str]] = []

    def build(self) -> None:
        model = self.repo / MODEL_RELATIVE
        model.parent.mkdir(parents=True, exist_ok=True)
        source = REPO_ROOT / MODEL_RELATIVE
        if self.bad_model:
            model.write_bytes(b"not-the-model")
        else:
            shutil.copyfile(source, model)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.binary.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.binary.chmod(0o755)
        library = self.runtime / "lib"
        library.mkdir(parents=True, exist_ok=True)
        (self.runtime / "VERSION_NUMBER").write_text("1.31.0\n", encoding="utf-8")
        name = (
            "libonnxruntime.1.31.0.dylib"
            if platform.system() == "Darwin"
            else "libonnxruntime.so.1.31.0"
        )
        versioned = library / name
        versioned.write_bytes(b"ort")
        (self.binary.parent / "CMakeCache.txt").write_text(
            "".join(
                f"{key}:STRING={value}\n"
                for key, value in {
                    "CMAKE_BUILD_TYPE": self.build_type,
                    "CMAKE_CXX_COMPILER": self.compiler,
                    "CMAKE_CXX_FLAGS": self.cxx_flags,
                    "CMAKE_CXX_FLAGS_RELEASE": self.release_flags,
                    "CMAKE_EXE_LINKER_FLAGS": "",
                    "CMAKE_EXE_LINKER_FLAGS_RELEASE": "",
                    "INFEREDGE_ONNXRUNTIME_ROOT": str(self.runtime),
                }.items()
            ),
            encoding="utf-8",
        )
        self.link_output = (
            f"{BINARY_NAME}:\n\t@rpath/{name} (compatibility version 1.0.0, current version 1.31.0)\n"
        )
        self.rpath_output = f"          cmd LC_RPATH\n         path {library} (offset 12)\n"
        self.ldd_output = f"\t{name} => {versioned} (0x1000)\n"

    def blob(self, _repo: pathlib.Path, _commit: str, relative: str) -> str:
        if relative != MODEL_RELATIVE:
            raise CollectorError(relative)
        if self.bad_model:
            return "0" * 64
        return MODEL_SHA256

    def runner(self, args, cwd=None, env=None, timeout_seconds=None):
        del cwd, timeout_seconds
        command = args[0]
        if command == "git":
            if "rev-parse" in args:
                return ProcessResult(0, self.head + "\n", "")
            if "status" in args:
                return ProcessResult(0, self.porcelain, "")
            raise AssertionError(args)
        if command == "nm":
            return ProcessResult(0, self.nm_output, "")
        if command == "otool" and args[1] == "-L":
            return ProcessResult(0, self.link_output, "")
        if command == "otool" and args[1] == "-l":
            return ProcessResult(0, self.rpath_output, "")
        if command == "ldd":
            return ProcessResult(0, self.ldd_output, "")
        if command == "sysctl":
            values = {"hw.model": "synthetic", "hw.physicalcpu": "8", "hw.logicalcpu": "8", "hw.ncpu": "8"}
            return ProcessResult(0, values[args[-1]] + "\n", "")
        if command == "sw_vers":
            return ProcessResult(0, "15.0\n", "")
        if command == "nproc":
            return ProcessResult(0, "8\n", "")
        if command == "cmake":
            return ProcessResult(0, "cmake version 3.28.3\n", "")
        if command == self.compiler:
            return ProcessResult(0, "c++ (synthetic) 13.3.0\n", "")
        if pathlib.Path(command).name == BINARY_NAME and args[1:] == ["--version"]:
            return ProcessResult(0, self.version_stdout, "")
        if pathlib.Path(command).name == BINARY_NAME and "--format" in args:
            if env is None or env.get("ORT_DISABLE_TELEMETRY") != "1":
                raise AssertionError("telemetry was not disabled")
            self.trial_calls.append(list(args))
            if self.fail_timeout:
                return SupervisedResult(-1, dump_report(sample_report()), "", timed_out=True)
            model = args[args.index("--model") + 1]
            workload = args[args.index("--workload") + 1]
            duration_ms = int(args[args.index("--duration-ms") + 1])
            warmup = int(args[args.index("--warmup") + 1])
            document = sample_report(model=model, workload=workload)
            document["configured_duration_nanoseconds"] = duration_ms * 1_000_000
            document["arrival_window_nanoseconds"] = duration_ms * 1_000_000
            document["total_trial_nanoseconds"] = duration_ms * 1_000_000 + document["completion_nanoseconds"]
            document["generator"]["warmup_requests"] = warmup
            if document["burst_phases"]:
                document["burst_phases"][0]["duration_nanoseconds"] = duration_ms * 1_000_000
            document["scheduler"]["enable_batching"] = "--enable-batching" in args
            document["scheduler"]["max_batch_size"] = int(args[args.index("--max-batch-size") + 1])
            document["scheduler"]["max_batch_delay_ms"] = int(
                args[args.index("--max-batch-delay-ms") + 1]
            )
            if workload == "closed_loop":
                document["completion_observation"] = "closed_loop_producer_wait"
                document["observation_quantum_nanoseconds"] = 0
                document["burst_phases"] = []
                document["scheduled_span_nanoseconds"] = 0
            return ProcessResult(0, dump_report(document), "")
        raise AssertionError(args)

    def argv(self, **overrides) -> list[str]:
        output = self.root / "scheduler-output"
        values = {
            "--binary": str(self.binary),
            "--repo-root": str(self.repo),
            "--expected-commit": COMMIT,
            "--cells": "A",
            "--profiles": "low_open",
            "--output-root": str(output),
            "--counted-runs": "5",
            "--inter-trial-delay-seconds": "0",
            "--warmup-requests": "0",
        }
        for key, value in overrides.items():
            values["--" + key.replace("_", "-")] = value
        argv = []
        for key, value in values.items():
            argv.extend([key, value])
        return argv


class PreflightAndCollectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.world = World(pathlib.Path(self.temporary.name))
        self.world.build()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_release_preflight_passes(self) -> None:
        checked = preflight(
            repo_root=self.world.repo,
            binary=self.world.binary,
            expected_commit=COMMIT,
            cells=[CELL_BY_ID["A"]],
            profiles=[PROFILE_BY_ID["low_open"]],
            measurement={
                "worker_count": 1,
                "queue_capacity": 64,
                "duration_ms": 1000,
                "max_offered": 100000,
                "max_outstanding": 64,
                "completion_timeout_ms": 5000,
                "warmup_requests": 0,
                "lag_tolerance_ms": 1000,
                "check_interval": 1,
            },
            counted_runs=5,
            delay_seconds=0,
            comparison_class="primary",
            runner=self.world.runner,
            blob_sha256=self.world.blob,
        )
        self.assertEqual(checked["verified_configuration"]["cmake_build_type"], "Release")
        self.assertEqual(checked["verified_configuration"]["sanitizer_instrumentation"], "absent")
        self.assertEqual(checked["verified_configuration"]["model_sha256"], MODEL_SHA256)
        self.assertEqual(len(checked["queried"]["benchmark_binary_sha256"]), 64)
        self.assertEqual(checked["queried"]["git_commit"], COMMIT)

    def test_debug_and_sanitizer_builds_are_rejected(self) -> None:
        self.world.build_type = "Debug"
        self.world.build()
        with self.assertRaises(CollectorError):
            self.test_release_preflight_passes()
        self.world.build_type = "Release"
        self.world.cxx_flags = "-fsanitize=address"
        self.world.build()
        with self.assertRaises(CollectorError):
            self.test_release_preflight_passes()

    def test_wrong_model_hash_is_rejected(self) -> None:
        self.world.bad_model = True
        self.world.build()
        with self.assertRaises(CollectorError) as caught:
            self.test_release_preflight_passes()
        self.assertIn("SHA-256", str(caught.exception))

    def test_wrong_binary_name_is_rejected(self) -> None:
        foreign = self.world.build_dir / "inferedge-bench"
        foreign.write_bytes(self.world.binary.read_bytes())
        foreign.chmod(0o755)
        with self.assertRaises(CollectorError):
            preflight(
                repo_root=self.world.repo,
                binary=foreign,
                expected_commit=COMMIT,
                cells=[CELL_BY_ID["A"]],
                profiles=[PROFILE_BY_ID["low_open"]],
                measurement=self._measurement(),
                counted_runs=5,
                delay_seconds=0,
                comparison_class="primary",
                runner=self.world.runner,
                blob_sha256=self.world.blob,
            )

    def test_collection_keeps_pilots_out_of_the_counted_names(self) -> None:
        from benchmarks.collect_scheduler import build_parser

        args = build_parser().parse_args(self.world.argv())
        run_set = collect(args, runner=self.world.runner, sleep=lambda _seconds: None, blob_sha256=self.world.blob)
        names = sorted(path.name for path in (run_set / "runs").glob("*.json") if not path.name.endswith(".meta.json"))
        self.assertIn("pilot-A-low_open.json", names)
        self.assertEqual(sum(name.startswith("A-low_open-") for name in names), 5)
        self.assertFalse((run_set / "aggregate.json").exists())
        self.assertEqual(self.world.trial_calls[0][self.world.trial_calls[0].index("--model") + 1].endswith(MODEL_RELATIVE), True)

    def test_process_timeout_preserves_failure_without_a_successful_report(self) -> None:
        from benchmarks.collect_scheduler import build_parser

        self.world.fail_timeout = True
        args = build_parser().parse_args(self.world.argv(counted_runs="1"))
        with self.assertRaises(CollectorError):
            collect(args, runner=self.world.runner, sleep=lambda _seconds: None, blob_sha256=self.world.blob)
        run_sets = list((self.world.root / "scheduler-output").iterdir())
        self.assertEqual(len(run_sets), 1)
        self.assertTrue((run_sets[0] / "failure.json").is_file())
        self.assertFalse((run_sets[0] / "runs" / "pilot-A-low_open.json").exists())
        meta = json.loads((run_sets[0] / "runs" / "pilot-A-low_open.meta.json").read_text(encoding="utf-8"))
        self.assertTrue(meta["execution_timeout"])
        self.assertFalse(meta["publication_eligible"])

    def test_nonzero_exit_stops_the_set(self) -> None:
        from benchmarks.collect_scheduler import build_parser

        def failing(args, cwd=None, env=None, timeout_seconds=None):
            if pathlib.Path(args[0]).name == BINARY_NAME and "--format" in args:
                return ProcessResult(1, "", "runtime failure")
            return self.world.runner(args, cwd=cwd, env=env, timeout_seconds=timeout_seconds)

        args = build_parser().parse_args(self.world.argv(counted_runs="1"))
        with self.assertRaises(CollectorError):
            collect(args, runner=failing, sleep=lambda _seconds: None, blob_sha256=self.world.blob)

    def test_existing_run_set_is_not_overwritten(self) -> None:
        from benchmarks.collect_baseline import create_run_set

        parent = self.world.root / "scheduler-output"
        created = create_run_set(parent, "20261013T000000Z-9a77f68")
        (created / "runs" / "keep.txt").write_text("preserve\n", encoding="utf-8")
        with self.assertRaises(CollectorError):
            create_run_set(parent, "20261013T000000Z-9a77f68")
        self.assertEqual((created / "runs" / "keep.txt").read_text(encoding="utf-8"), "preserve\n")

    def _measurement(self) -> dict:
        return {
            "worker_count": 1,
            "queue_capacity": 64,
            "duration_ms": 1000,
            "max_offered": 100000,
            "max_outstanding": 64,
            "completion_timeout_ms": 5000,
            "warmup_requests": 0,
            "lag_tolerance_ms": 1000,
            "check_interval": 1,
        }


class ProcessSupervisionTest(unittest.TestCase):
    def test_wall_clock_timeout_reaps_the_child(self) -> None:
        result = supervise_process(
            [os.environ.get("PYTHON", "python3"), "-c", "import time; time.sleep(30)"],
            cwd=None,
            env=os.environ.copy(),
            timeout_seconds=0.3,
        )
        self.assertTrue(result.timed_out)
        self.assertNotEqual(result.returncode, 0)


class SmokeCollectionTest(unittest.TestCase):
    def test_tiny_fake_trial_uses_a_temporary_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            world = World(pathlib.Path(temporary))
            world.build()
            from benchmarks.collect_scheduler import build_parser

            args = build_parser().parse_args(
                world.argv(cells="A", profiles="low_open", counted_runs="1", duration_ms="50", max_offered="4", max_outstanding="4")
            )
            run_set = collect(
                args,
                runner=world.runner,
                sleep=lambda _seconds: None,
                blob_sha256=world.blob,
            )
            self.assertTrue(run_set.resolve().is_relative_to(pathlib.Path(temporary).resolve()))
            self.assertNotIn("benchmarks/baselines", str(run_set))
            self.assertNotIn("benchmarks/concurrent_runs", str(run_set))
            self.assertNotIn("models/experiments", str(run_set))
            self.assertEqual(len(world.trial_calls), 2)


if __name__ == "__main__":
    unittest.main()
