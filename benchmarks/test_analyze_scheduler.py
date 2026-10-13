#!/usr/bin/env python3
"""Deterministic tests for the schema-3 scheduler analyzer."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from benchmarks.analyze_scheduler import (
    AnalyzerError,
    analyze,
    assert_fair_pair,
    fairness_record,
    reduce_group,
)
from benchmarks.collect_scheduler import (
    CELL_BY_ID,
    PROFILE_BY_ID,
    build_parser,
    collect,
)
from benchmarks.test_collect_scheduler import COMMIT, World, sample_report


class ReductionTest(unittest.TestCase):
    def test_median_variability_and_no_confidence_interval(self) -> None:
        reports = []
        for throughput in (10.0, 12.0, 11.0, 13.0, 9.0):
            document = sample_report()
            document["throughput"]["successful_attempts_per_arrival_window_second"] = throughput
            document["successful_e2e_latency"]["p50_microseconds"] = throughput
            document["successful_e2e_latency"]["p95_microseconds"] = throughput + 1
            document["successful_e2e_latency"]["precise_tail"] = True
            document["successful_e2e_latency"]["sample_count"] = 100
            reports.append(document)
        statuses = [{"window_class": "configured_duration"} for _report in reports]
        summary = reduce_group(reports, statuses)
        self.assertEqual(summary["counted_runs"], 5)
        self.assertEqual(summary["median_successful_attempts_per_arrival_window_second"], 11.0)
        self.assertEqual(summary["median_of_run_p95_microseconds"], 12.0)
        self.assertEqual(summary["median_of_run_p99_microseconds"], 4.5)
        self.assertEqual(summary["p99_reduction"], "median_of_run_p99")
        self.assertIsNone(summary["p99_omitted_reason"])
        self.assertFalse(summary["confidence_interval_computed"])
        self.assertFalse(summary["winner_declared"])
        self.assertNotIn("global_p99", summary)
        self.assertNotIn("pooled_p99", summary)
        self.assertIsNotNone(summary["run_to_run_throughput_pstdev"])

    def test_small_sample_p99_is_omitted(self) -> None:
        document = sample_report()
        document["successful_e2e_latency"]["precise_tail"] = False
        document["successful_e2e_latency"]["sample_count"] = 4
        document["succeeded"] = 4
        document["accepted"] = 4
        document["submit_attempts"] = 4
        document["scheduled_opportunities"] = 4
        document["burst_phases"][0]["scheduled_opportunities"] = 4
        document["metrics"]["engine_run_count"] = 4
        document["metrics"]["requests_entered_engine_runs"] = 4
        document["metrics"]["batch_size_histogram"] = [0, 4] + [0] * 31
        document["correctness_checked"] = 4
        summary = reduce_group([document], [{"window_class": "configured_duration"}])
        self.assertIsNone(summary["median_of_run_p99_microseconds"])
        self.assertIn("precise_tail", summary["p99_omitted_reason"])
        self.assertFalse(summary["confidence_interval_computed"])

    def test_mismatched_cells_are_not_compared(self) -> None:
        environment = {
            "queried": {
                "git_commit": COMMIT,
                "benchmark_binary_sha256": "a" * 64,
                "onnxruntime_resolved_library": "/ort/libonnxruntime.so.1.31.0",
            },
            "verified_configuration": {
                "model_sha256": "b" * 64,
                "sanitizer_instrumentation": "absent",
            },
            "collection_protocol": {
                "workload_family": "open_loop",
                "environment_variable": "ORT_DISABLE_TELEMETRY=1",
                "measurement": {
                    "duration_ms": 1000,
                    "max_offered": 100,
                    "max_outstanding": 64,
                    "lag_tolerance_ms": 1000,
                    "completion_timeout_ms": 5000,
                },
            },
        }
        left_report = sample_report()
        right_report = sample_report()
        right_report["scheduler"]["worker_count"] = 2
        left = fairness_record(environment, left_report, {"workload_id": "low_open"})
        right = fairness_record(environment, right_report, {"workload_id": "low_open"})
        with self.assertRaises(AnalyzerError) as caught:
            assert_fair_pair(left, right)
        self.assertIn("worker_count", str(caught.exception))

    def test_different_arrival_rates_are_not_a_fair_pair(self) -> None:
        environment = {
            "queried": {
                "git_commit": COMMIT,
                "benchmark_binary_sha256": "a" * 64,
                "onnxruntime_resolved_library": "/ort/libonnxruntime.so.1.31.0",
                "onnxruntime_version": "1.31.0",
                "compiler_version": "c++ 13.3.0\n",
                "cmake_flag_text": "CMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG",
                "operating_system": "Darwin",
                "cpu_architecture": "arm64",
                "platform": "macOS",
                "hardware_model": "synthetic",
            },
            "verified_configuration": {
                "model_sha256": "b" * 64,
                "sanitizer_instrumentation": "absent",
            },
            "collection_protocol": {
                "workload_family": "open_loop",
                "environment_variable": "ORT_DISABLE_TELEMETRY=1",
                "measurement": {
                    "duration_ms": 1000,
                    "max_offered": 100,
                    "max_outstanding": 64,
                    "lag_tolerance_ms": 1000,
                    "completion_timeout_ms": 5000,
                },
            },
        }
        low = fairness_record(environment, sample_report(), {"workload_id": "low_open"})
        high = fairness_record(environment, sample_report(), {"workload_id": "high_open"})
        with self.assertRaises(AnalyzerError) as caught:
            assert_fair_pair(low, high)
        self.assertIn("profile_id", str(caught.exception))
        self.assertIn("target_offered_rate", str(caught.exception))

    def test_corrupt_recorded_command_is_rejected(self) -> None:
        from benchmarks.analyze_scheduler import require_recorded_command

        environment = {
            "collection_protocol": {
                "measurement": {
                    "worker_count": 1,
                    "queue_capacity": 64,
                    "duration_ms": 1000,
                    "max_offered": 100000,
                    "max_outstanding": 64,
                    "lag_tolerance_ms": 1000,
                    "completion_timeout_ms": 5000,
                    "warmup_requests": 0,
                    "check_interval": 1,
                }
            }
        }
        meta = {
            "trial_id": "A-low_open-01",
            "workload_id": "low_open",
            "command": [
                "inferedge-bench-scheduler",
                "--workers", "1",
                "--queue-capacity", "64",
                "--workload", "open_loop_constant",
                "--duration-ms", "1000",
                "--max-offered", "100000",
                "--max-outstanding", "64",
                "--lag-tolerance-ms", "1000",
                "--completion-timeout-ms", "5000",
                "--warmup", "0",
                "--check-interval", "1",
                "--target-rps", "1000",
            ],
        }
        with self.assertRaises(AnalyzerError) as caught:
            require_recorded_command(meta, environment)
        self.assertIn("--target-rps", str(caught.exception))


class AggregateFileTest(unittest.TestCase):
    def test_five_counted_trials_exclude_the_pilot_and_do_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            world = World(pathlib.Path(temporary))
            world.build()
            args = build_parser().parse_args(world.argv())
            run_set = collect(
                args,
                runner=world.runner,
                sleep=lambda _seconds: None,
                blob_sha256=world.blob,
            )
            before = {
                path.name: path.read_bytes()
                for path in (run_set / "runs").iterdir()
                if path.is_file()
            }
            document = analyze(run_set)
            after = {
                path.name: path.read_bytes()
                for path in (run_set / "runs").iterdir()
                if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual(document["counted_runs_per_cell"], 5)
            self.assertEqual(len(document["steady_state_groups"]), 1)
            self.assertEqual(document["steady_state_groups"][0]["counted_runs"], 5)
            self.assertTrue(document["pilots_excluded"])
            self.assertFalse(document["confidence_interval_computed"])
            self.assertFalse(document["winner_declared"])
            encoded = json.dumps(document)
            self.assertNotIn("\"global_p99\"", encoded)
            self.assertNotIn("\"pooled_p99\"", encoded)
            with self.assertRaises(AnalyzerError):
                analyze(run_set)
            self.assertIn(PROFILE_BY_ID["low_open"]["id"], document["steady_state_groups"][0]["workload_id"])
            self.assertEqual(document["steady_state_groups"][0]["configuration_id"], CELL_BY_ID["A"]["id"])


if __name__ == "__main__":
    unittest.main()
