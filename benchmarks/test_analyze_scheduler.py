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
    fairness_differences,
    fairness_record,
    reduce_group,
)
from benchmarks.collect_scheduler import (
    CELL_BY_ID,
    PROFILE_BY_ID,
    build_parser,
    collect,
)
from benchmarks.test_collect_scheduler import COMMIT, World, sample_report, schema4_report


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


class Protocol2StatisticsTest(unittest.TestCase):
    def test_known_positive_negative_and_zero_effects(self) -> None:
        from benchmarks.analyze_scheduler import paired_metric_effect

        positive = paired_metric_effect([10.0, 10.0], [11.0, 13.0], units="requests per second", allow_interval=False)
        self.assertEqual(positive["effect_estimate"], 2.0)
        self.assertEqual(positive["median_paired_difference"], 2.0)
        self.assertEqual(positive["relative_effect_if_defined"], 20.0)
        self.assertFalse(positive["comparison_supported"])
        self.assertFalse(positive["superiority_claimed"])
        negative = paired_metric_effect([5.0, 5.0], [3.0, 3.0], units="microseconds", allow_interval=False)
        self.assertEqual(negative["effect_estimate"], -2.0)
        zero = paired_metric_effect([4.0, 4.0], [4.0, 4.0], units="microseconds", allow_interval=False)
        self.assertEqual(zero["effect_estimate"], 0.0)
        self.assertEqual(zero["median_paired_difference"], 0.0)
        self.assertEqual(positive["relative_effect_definition"], "mean of per-block 100 * difference / baseline")
        negative_baseline = paired_metric_effect([-2.0], [2.0], units="microseconds", allow_interval=False)
        self.assertIsNone(negative_baseline["relative_effect_if_defined"])
        self.assertEqual(negative_baseline["effect_estimate"], 4.0)

    def test_zero_baseline_does_not_invent_a_percentage(self) -> None:
        from benchmarks.analyze_scheduler import paired_metric_effect

        result = paired_metric_effect([0.0, 0.0], [5.0, 5.0], units="requests per second", allow_interval=False)
        self.assertIsNone(result["relative_effect_if_defined"])
        self.assertEqual(result["effect_estimate"], 5.0)
        self.assertTrue(any("undefined" in note for note in result["limitation_notes"]))

    def test_missing_metric_and_undefined_batch_size(self) -> None:
        from benchmarks.analyze_scheduler import _mean_batch_size, paired_metric_effect

        missing = paired_metric_effect([1.0, None], [2.0, 2.0], units="microseconds", allow_interval=False)
        self.assertIsNone(missing["effect_estimate"])
        self.assertFalse(missing["confidence_interval_computed"])
        document = sample_report()
        document["metrics"]["engine_run_count"] = 0
        document["metrics"]["requests_entered_engine_runs"] = 0
        self.assertIsNone(_mean_batch_size(document))

    def test_bootstrap_seed_is_deterministic_and_intervals_have_the_required_meaning(self) -> None:
        from benchmarks.analyze_scheduler import paired_block_bootstrap_mean, paired_metric_effect

        values = [float(index) for index in range(1, 11)]
        first = paired_block_bootstrap_mean(values, seed=20261013, replicates=10000)
        second = paired_block_bootstrap_mean(values, seed=20261013, replicates=10000)
        self.assertEqual(first, second)
        one_draw = paired_block_bootstrap_mean(values, seed=20261013, replicates=1)
        other_draw = paired_block_bootstrap_mean(values, seed=20261014, replicates=1)
        self.assertNotEqual(one_draw, other_draw)
        self.assertIsNone(paired_block_bootstrap_mean([1.0, 2.0, 3.0]))
        includes = paired_metric_effect([0.0] * 10, [0.0] * 10, units="requests per second", allow_interval=True)
        self.assertTrue(includes["confidence_interval_computed"])
        self.assertLessEqual(includes["confidence_interval"]["low"], 0.0)
        self.assertGreaterEqual(includes["confidence_interval"]["high"], 0.0)
        self.assertFalse(includes["comparison_supported"])
        self.assertTrue(any("includes zero" in note for note in includes["limitation_notes"]))
        excludes = paired_metric_effect([1.0] * 10, [3.0] * 10, units="requests per second", allow_interval=True)
        self.assertEqual(excludes["effect_estimate"], 2.0)
        self.assertGreater(excludes["confidence_interval"]["low"], 0.0)
        self.assertTrue(excludes["comparison_supported"])
        self.assertFalse(excludes["superiority_claimed"])
        self.assertTrue(any("does not mean the alternative is universally faster" in note for note in excludes["limitation_notes"]))
        short = paired_metric_effect([1.0] * 5, [2.0] * 5, units="requests per second", allow_interval=True)
        self.assertIsNone(short["confidence_interval"])
        self.assertFalse(short["comparison_supported"])
        self.assertEqual(short["effect_estimate"], 1.0)

    def test_plan_identity_rejects_span_count_and_accepts_a_different_arrival_window(self) -> None:
        from benchmarks.analyze_scheduler import plan_identity_differences

        left = sample_report()
        right = sample_report()
        status = {"window_class": "configured_duration"}
        right["scheduled_span_nanoseconds"] = left["scheduled_span_nanoseconds"] - 1
        self.assertIn("scheduled_span_nanoseconds", plan_identity_differences(left, right, status, status))
        right = sample_report()
        right["scheduled_opportunities"] = 90
        self.assertIn("scheduled_opportunities", plan_identity_differences(left, right, status, status))
        right = sample_report(arrival_window_nanoseconds=1_000_000_100)
        self.assertEqual(plan_identity_differences(left, right, status, status), [])
        longer = sample_report(arrival_window_nanoseconds=3_000_000_000)
        from benchmarks.collect_scheduler import classify_report

        _report, outside = classify_report(
            json.dumps(longer) + "\n",
            expected={
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
                "analysis_protocol_version": 2,
            },
        )
        self.assertNotEqual(outside["delivery_class"], "STRICT_PRIMARY")


def _collect(root: pathlib.Path, **overrides):
    world = World(root)
    world.build()
    args = build_parser().parse_args(world.argv(**overrides))
    run_set = collect(args, runner=world.runner, sleep=lambda _seconds: None, blob_sha256=world.blob)
    return world, run_set


def _rewrite(run_set: pathlib.Path, cell: str, profile: str, mutate) -> None:
    for path in (run_set / "runs").glob(f"{cell}-{profile}-*.json"):
        if path.name.endswith(".meta.json"):
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        mutate(document)
        path.write_text(json.dumps(document) + "\n", encoding="utf-8")


class Protocol2AggregateTest(unittest.TestCase):
    def test_five_blocks_stay_descriptive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _world, run_set = _collect(pathlib.Path(temporary), cells="A,B", profiles="low_open", counted_runs="5")
            document = analyze(run_set)
            self.assertEqual(document["analysis_protocol_version"], 2)
            self.assertEqual(document["counted_runs_per_cell"], 5)
            self.assertEqual(len(document["contrasts"]), 1)
            metric = document["contrasts"][0]["metrics"]["successful_offered_cohort_rate"]
            self.assertEqual(metric["effect_estimate"], 0.0)
            self.assertEqual(metric["paired_block_count"], 5)
            self.assertIsNone(metric["confidence_interval"])
            self.assertFalse(metric["comparison_supported"])
            self.assertFalse(document["confidence_interval_computed"])
            self.assertFalse(document["winner_declared"])
            self.assertFalse(document["comparison_supported"])

    def test_ten_blocks_can_exclude_zero_without_declaring_a_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _world, run_set = _collect(pathlib.Path(temporary), cells="A,B", profiles="low_open", counted_runs="10")

            def bump(document: dict) -> None:
                document["throughput"]["successful_attempts_per_arrival_window_second"] += 5
                document["successful_e2e_latency"]["p95_microseconds"] += 5

            _rewrite(run_set, "B", "low_open", bump)
            document = analyze(run_set)
            metric = document["contrasts"][0]["metrics"]["successful_offered_cohort_rate"]
            self.assertEqual(metric["paired_block_count"], 10)
            self.assertEqual(metric["effect_estimate"], 5.0)
            self.assertTrue(metric["confidence_interval_computed"])
            self.assertGreater(metric["confidence_interval"]["low"], 0.0)
            self.assertTrue(metric["comparison_supported"])
            self.assertFalse(metric["superiority_claimed"])
            self.assertFalse(document["winner_declared"])
            self.assertFalse(document["superiority_claimed"])
            self.assertFalse(document["comparison_supported"])
            encoded = json.dumps(document)
            self.assertNotIn("\"pooled_p99\"", encoded)
            self.assertNotIn("\"global_p99\"", encoded)
            self.assertNotIn("universally faster", encoded.lower().split("does not mean the alternative is universally faster")[0])

    def test_twenty_blocks_use_the_preferred_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _world, run_set = _collect(pathlib.Path(temporary), cells="A,B", profiles="low_open", counted_runs="20")
            document = analyze(run_set)
            self.assertEqual(document["preferred_counted_runs"], 20)
            self.assertEqual(document["contrasts"][0]["paired_block_count"], 20)
            self.assertTrue(document["confidence_interval_computed"])
            self.assertFalse(document["winner_declared"])

    def test_missing_duplicate_and_extra_blocks_do_not_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _world, missing = _collect(root / "missing", cells="A,B", profiles="low_open", counted_runs="2")
            (missing / "runs" / "B-low_open-02.json").unlink()
            with self.assertRaises(AnalyzerError):
                analyze(missing)
            self.assertFalse((missing / "aggregate.json").exists())
            _world, duplicated = _collect(root / "duplicated", cells="A,B", profiles="low_open", counted_runs="2")
            meta_path = duplicated / "runs" / "B-low_open-02.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["block_index"] = 1
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            with self.assertRaises(AnalyzerError) as duplicate:
                analyze(duplicated)
            self.assertIn("block", str(duplicate.exception))
            self.assertFalse((duplicated / "aggregate.json").exists())
            _world, extra = _collect(root / "extra", cells="A,B", profiles="low_open", counted_runs="2")
            (extra / "runs" / "extra.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(AnalyzerError):
                analyze(extra)
            self.assertFalse((extra / "aggregate.json").exists())

    def test_mismatched_run_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _world, run_set = _collect(pathlib.Path(temporary), cells="A", profiles="low_open", counted_runs="5")
            environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            environment["collection_protocol"]["counted_runs_per_cell"] = 10
            (run_set / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
            with self.assertRaises(AnalyzerError):
                analyze(run_set)

    def test_protocol2_without_a_version_is_not_downgraded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _world, run_set = _collect(pathlib.Path(temporary), cells="A", profiles="low_open", counted_runs="1")
            environment_path = run_set / "environment.json"
            before = {
                path.name: path.read_bytes()
                for path in (run_set / "runs").iterdir()
                if path.is_file()
            }
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            environment["collection_protocol"].pop("analysis_protocol_version")
            environment_path.write_text(json.dumps(environment), encoding="utf-8")
            with self.assertRaises(AnalyzerError) as caught:
                analyze(run_set)
            self.assertIn("refusing to downgrade", str(caught.exception))
            self.assertFalse((run_set / "aggregate.json").exists())
            after = {
                path.name: path.read_bytes()
                for path in (run_set / "runs").iterdir()
                if path.is_file()
            }
            self.assertEqual(before, after)

    def test_protocol1_evidence_keeps_the_original_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _world, run_set = _collect(pathlib.Path(temporary), cells="A", profiles="low_open", counted_runs="5")
            environment_path = run_set / "environment.json"
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            environment["collection_protocol"].pop("analysis_protocol_version")
            environment["collection_protocol"].pop("publication_policy", None)
            environment["source_defined_policy"].pop("publication_policy", None)
            environment_path.write_text(json.dumps(environment), encoding="utf-8")
            document = analyze(run_set)
            self.assertEqual(document["analysis_protocol_version"], 1)
            self.assertEqual(document["minimum_submit_fraction"], 0.99)
            self.assertFalse(document["confidence_interval_computed"])
            self.assertFalse(document["winner_declared"])
            self.assertNotIn("contrasts", document)

    def test_different_spans_are_not_paired_and_windows_inside_the_bound_are(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _world, blocked = _collect(root / "span", cells="A,B", profiles="low_open", counted_runs="1")

            def shorten(document: dict) -> None:
                document["scheduled_span_nanoseconds"] -= 1

            _rewrite(blocked, "B", "low_open", shorten)
            with self.assertRaises(AnalyzerError) as caught:
                analyze(blocked)
            self.assertIn("scheduled_span_nanoseconds", str(caught.exception))
            _world, allowed = _collect(root / "window", cells="A,B", profiles="low_open", counted_runs="1")

            def shift(document: dict) -> None:
                document["arrival_window_nanoseconds"] += 1000
                document["total_trial_nanoseconds"] += 1000

            _rewrite(allowed, "B", "low_open", shift)
            document = analyze(allowed)
            windows = document["contrasts"][0]["arrival_window_nanoseconds"]
            self.assertNotEqual(windows["baseline"], windows["alternative"])
            self.assertIsNone(document["contrasts"][0]["metrics"]["successful_offered_cohort_rate"]["confidence_interval"])

    def test_cross_profile_pairs_stay_separate_and_extra_cells_need_a_predeclared_contrast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _world, paired = _collect(root / "profiles", cells="A,B", profiles="low_open,high_open", counted_runs="1")

            def low(document: dict) -> None:
                document["throughput"]["successful_attempts_per_arrival_window_second"] += 1

            def high(document: dict) -> None:
                document["throughput"]["successful_attempts_per_arrival_window_second"] += 10

            _rewrite(paired, "B", "low_open", low)
            _rewrite(paired, "B", "high_open", high)
            document = analyze(paired)
            effects = {
                contrast["profile_id"]: contrast["metrics"]["successful_offered_cohort_rate"]["effect_estimate"]
                for contrast in document["contrasts"]
            }
            self.assertEqual(effects, {"low_open": 1.0, "high_open": 10.0})
            _world, many = _collect(
                root / "many",
                cells="A,B,C",
                profiles="low_open",
                counted_runs="1",
                baseline_cell="A",
                alternative_cell="B",
            )
            declared = analyze(many)
            self.assertEqual(len(declared["contrasts"]), 1)
            self.assertEqual(declared["contrasts"][0]["alternative_cell"], "B")
            self.assertFalse(declared["winner_declared"])
            _world, undeclared = _collect(root / "plain", cells="A,B,C", profiles="low_open", counted_runs="1")
            plain = analyze(undeclared)
            self.assertEqual(plain["contrasts"], [])
            self.assertFalse(plain["comparison_supported"])

    def test_fairness_still_rejects_runtime_duration_and_observation_mismatches(self) -> None:
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
            "verified_configuration": {"model_sha256": "b" * 64, "sanitizer_instrumentation": "absent"},
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
        left = fairness_record(environment, sample_report(), {"workload_id": "low_open"})
        runtime = dict(left)
        runtime["onnxruntime_version"] = "1.30.0"
        with self.assertRaises(AnalyzerError):
            assert_fair_pair(left, runtime)
        duration = dict(left)
        duration["duration_ms"] = 2000
        with self.assertRaises(AnalyzerError):
            assert_fair_pair(left, duration)
        observation = dict(left)
        observation["completion_observation"] = "closed_loop_producer_wait"
        with self.assertRaises(AnalyzerError):
            assert_fair_pair(left, observation)


class Protocol3AnalysisTest(unittest.TestCase):
    def test_predeclared_metric_is_distinct_and_does_not_declare_a_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            world = World(pathlib.Path(temporary))
            world.build()
            args = build_parser().parse_args(
                world.argv(
                    cells="A,B",
                    counted_runs="1",
                    analysis_protocol="3",
                    latency_metric="pre_publication",
                    arrival_fidelity_p95_max_ns="1000",
                    arrival_fidelity_max_lag_ns="1000",
                )
            )
            run_set = collect(
                args, runner=world.runner, sleep=lambda _seconds: None, blob_sha256=world.blob
            )
            document = analyze(run_set)
        self.assertEqual(document["analysis_protocol_version"], 3)
        self.assertEqual(document["source_schema_version"], 4)
        self.assertEqual(document["publication_policy"], "pre_publication_timing")
        self.assertEqual(document["latency_metric"], "pre_publication")
        self.assertEqual(document["arrival_timing_fidelity"], "passed")
        self.assertFalse(document["winner_declared"])
        self.assertFalse(document["superiority_claimed"])
        metrics = document["contrasts"][0]["metrics"]
        self.assertIn("p50_pre_publication_latency", metrics)
        self.assertIn("p50_e2e_latency", metrics)
        self.assertNotEqual(
            metrics["p50_pre_publication_latency"]["units"], metrics["p50_e2e_latency"]["units"]
        )
        delay = metrics["p50_pre_publication_to_observation_delay"]
        self.assertFalse(delay["confidence_interval_computed"])
        self.assertIn("set_value", delay["units"])
        self.assertIn("phase_boundary_fidelity is unavailable", " ".join(document["limitations"]))
        self.assertFalse(document["comparison_supported"])
        self.assertFalse(document["instrumentation_overhead_calibrated"])

    def test_p99_is_omitted_when_the_tail_is_not_precise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            world = World(pathlib.Path(temporary))
            world.build()
            world.force_imprecise_tail = True
            args = build_parser().parse_args(
                world.argv(
                    cells="A,B",
                    counted_runs="1",
                    analysis_protocol="3",
                    latency_metric="producer_observed",
                    arrival_fidelity_p95_max_ns="1000",
                    arrival_fidelity_max_lag_ns="1000",
                )
            )
            run_set = collect(
                args, runner=world.runner, sleep=lambda _seconds: None, blob_sha256=world.blob
            )
            document = analyze(run_set)
        reason = document["contrasts"][0]["metrics"]["p99_pre_publication_latency"][
            "p99_omitted_reason"
        ]
        self.assertIn("precise_tail", reason)
        self.assertFalse(document["winner_declared"])

    def test_clock_period_mismatch_is_not_a_fair_pair(self) -> None:
        environment = {
            "queried": {
                "git_commit": COMMIT,
                "benchmark_binary_sha256": "a" * 64,
                "onnxruntime_resolved_library": "/ort/libonnxruntime.so.1.31.0",
            },
            "verified_configuration": {"model_sha256": "b" * 64, "sanitizer_instrumentation": "absent"},
            "collection_protocol": {
                "workload_family": "open_loop",
                "environment_variable": "ORT_DISABLE_TELEMETRY=1",
                "analysis_protocol_version": 3,
                "latency_metric": "pre_publication",
                "arrival_fidelity_p95_max_nanoseconds": 1000,
                "arrival_fidelity_max_lag_nanoseconds": 1000,
                "measurement": {
                    "duration_ms": 1000,
                    "max_offered": 100,
                    "max_outstanding": 64,
                    "lag_tolerance_ms": 1000,
                    "completion_timeout_ms": 5000,
                },
            },
        }
        left = fairness_record(environment, schema4_report(), {"workload_id": "low_open"})
        other = schema4_report()
        other["steady_clock_period_denominator"] = 1000
        right = fairness_record(environment, other, {"workload_id": "low_open"})
        self.assertIn("steady_clock_period_denominator", fairness_differences(left, right))
        self.assertEqual(left["enable_completion_timing"], True)
        self.assertEqual(left["latency_metric"], "pre_publication")


if __name__ == "__main__":
    unittest.main()
