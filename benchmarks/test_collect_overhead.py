#!/usr/bin/env python3
"""Deterministic tests for the completion-timing overhead collector."""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import sys
import tempfile
import unittest

from benchmarks.analyze_overhead import analyze
from benchmarks.analyze_scheduler import analysis_protocol_version
from benchmarks.collect_overhead import (
    CELL_ID,
    EXPERIMENT_KIND,
    OverheadError,
    arm_order,
    build_plan,
    collect,
    command_for_arm,
    commands_differ_only_by_timing,
    counted_blocks_for,
    validate_arm,
)
from benchmarks.collect_scheduler import (
    BINARY_NAME,
    MODEL_SHA256,
    PRECISION_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SupervisedResult,
    supervise_process,
)
from benchmarks.test_collect_scheduler import COMMIT, World, dump_report, sample_report, schema4_report

MODEL = pathlib.Path("/models/batch_add_f32.onnx")


def _arm_report(arm: str, **overrides) -> dict:
    document = sample_report(model=str(MODEL))
    document["generator"]["warmup_requests"] = 100
    if arm == "on":
        document = schema4_report(document)
    document.update(overrides)
    return document


def _scale(document: dict, count: int) -> None:
    document["accepted"] = count
    document["succeeded"] = count
    document["successful_e2e_latency"]["sample_count"] = count
    document["metrics"]["engine_run_count"] = count
    document["metrics"]["requests_entered_engine_runs"] = count
    document["metrics"]["batch_size_histogram"] = [0, count] + [0] * 31
    document["correctness_checked"] = count
    document["generator_lag"]["count"] = count
    for name in ("pre_publication_latency", "pre_publication_to_observation_delay"):
        if name in document:
            document[name]["sample_count"] = count
            document[name]["precise_tail"] = count >= 100


class CommandAndPlanTest(unittest.TestCase):
    def test_off_on_command_delta_is_only_completion_timing(self) -> None:
        binary = pathlib.Path("/tmp") / BINARY_NAME
        off = command_for_arm(binary, MODEL, "off")
        on = command_for_arm(binary, MODEL, "on")
        self.assertNotIn("--completion-timing", off)
        self.assertEqual(on.count("--completion-timing"), 1)
        self.assertTrue(commands_differ_only_by_timing(off, on))
        self.assertIn("--target-rps", off)
        self.assertEqual(off[off.index("--target-rps") + 1], "50")
        self.assertNotIn("--enable-batching", off)
        self.assertEqual(off[off.index("--workers") + 1], "1")
        self.assertEqual(off[off.index("--queue-capacity") + 1], "64")
        self.assertEqual(off[off.index("--max-batch-size") + 1], "1")
        self.assertEqual(off[off.index("--max-batch-delay-ms") + 1], "0")
        self.assertEqual(off[off.index("--duration-ms") + 1], "1000")
        self.assertEqual(off[off.index("--warmup") + 1], "100")

    def test_block_index_starts_at_zero_and_alternates(self) -> None:
        self.assertEqual(arm_order(0), ("off", "on"))
        self.assertEqual(arm_order(1), ("on", "off"))
        plan = build_plan(2)
        self.assertEqual([(trial["arm"], trial["pilot"]) for trial in plan[:2]], [("off", True), ("on", True)])
        counted = [trial for trial in plan if not trial["pilot"]]
        self.assertEqual([trial["block_index"] for trial in counted], [0, 0, 1, 1])
        self.assertEqual([trial["arm"] for trial in counted], ["off", "on", "on", "off"])
        self.assertEqual([trial["execution_index"] for trial in plan], list(range(6)))

    def test_smoke_is_one_pair_and_reportable_is_twenty_blocks(self) -> None:
        self.assertEqual(counted_blocks_for("smoke", None), 0)
        self.assertEqual(len(build_plan(0)), 2)
        self.assertEqual(counted_blocks_for("reportable", None), 20)
        self.assertEqual(counted_blocks_for("exploratory", 3), 3)
        with self.assertRaises(OverheadError):
            counted_blocks_for("smoke", 1)
        with self.assertRaises(OverheadError):
            counted_blocks_for("reportable", 10)


class ValidationTest(unittest.TestCase):
    def test_schema3_off_validation_passes(self) -> None:
        _document, status, problems = validate_arm(dump_report(_arm_report("off")), "off", MODEL)
        self.assertEqual(problems, [])
        self.assertTrue(status["collector_valid"])
        self.assertEqual(status["delivery_class"], "STRICT_PRIMARY")
        self.assertEqual(status["analysis_protocol_version"], 2)

    def test_schema4_on_validation_passes(self) -> None:
        document, status, problems = validate_arm(dump_report(_arm_report("on")), "on", MODEL)
        self.assertEqual(problems, [])
        self.assertEqual(document["schema_version"], PRECISION_SCHEMA_VERSION)
        self.assertTrue(document["scheduler"]["enable_completion_timing"])
        self.assertEqual(status["analysis_protocol_version"], 3)
        self.assertEqual(status["delivery_class"], "STRICT_PRIMARY")
        self.assertEqual(status["arrival_timing_fidelity"], "not_certified")

    def test_strict_delivery_requires_full_submission(self) -> None:
        document = _arm_report("off")
        self.assertEqual(document["submit_attempts"], document["scheduled_opportunities"])
        self.assertEqual(document["generator_skips_outstanding"], 0)
        self.assertEqual(document["generator_skips_lag"], 0)
        self.assertEqual(document["arrival_stop_reason"], "configured_duration")
        _document, _status, problems = validate_arm(dump_report(document), "off", MODEL)
        self.assertEqual(problems, [])

    def test_one_generator_skip_is_invalid(self) -> None:
        document = _arm_report("off")
        document["submit_attempts"] = 99
        document["generator_skips_lag"] = 1
        _scale(document, 99)
        _document, status, problems = validate_arm(dump_report(document), "off", MODEL)
        self.assertTrue(problems)
        self.assertNotEqual(status["delivery_class"], "STRICT_PRIMARY")

    def test_rejection_with_full_generator_submission_stays_valid(self) -> None:
        document = _arm_report("off")
        document["rejections"]["full"] = 10
        _scale(document, 90)
        document["generator_lag"]["count"] = document["submit_attempts"]
        _document, status, problems = validate_arm(dump_report(document), "off", MODEL)
        self.assertEqual(problems, [])
        self.assertEqual(status["delivery_class"], "STRICT_PRIMARY")
        self.assertEqual(document["submit_attempts"], document["scheduled_opportunities"])

    def test_malformed_timing_fields_are_invalid(self) -> None:
        document = _arm_report("on")
        document["pre_publication_latency"]["sample_count"] = document["succeeded"] - 1
        _document, _status, problems = validate_arm(dump_report(document), "on", MODEL)
        self.assertTrue(any("pre_publication_latency" in item for item in problems))

    def test_wrong_schema_for_an_arm_is_invalid(self) -> None:
        _document, _status, off_problems = validate_arm(dump_report(_arm_report("on")), "off", MODEL)
        _document, _status, on_problems = validate_arm(dump_report(_arm_report("off")), "on", MODEL)
        self.assertTrue(any("schema_version" in item for item in off_problems))
        self.assertTrue(any("schema_version" in item for item in on_problems))
        self.assertEqual(SCHEMA_VERSION, 3)
        self.assertEqual(PRECISION_SCHEMA_VERSION, 4)


class CollectionSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.world = World(pathlib.Path(self.temporary.name))
        self.world.build()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self, mode: str, counted: str | None = None) -> list[str]:
        argv = [
            "--binary",
            str(self.world.binary),
            "--repo-root",
            str(self.world.repo),
            "--expected-commit",
            COMMIT,
            "--output-root",
            str(self.world.root / "overhead-output"),
            "--mode",
            mode,
        ]
        if counted is not None:
            argv.extend(["--counted-runs", counted])
        return argv

    def test_smoke_collects_one_pilot_pair_and_excludes_it_from_the_interval(self) -> None:
        from benchmarks.collect_overhead import build_parser

        args = build_parser().parse_args(self._args("smoke"))
        run_set = collect(args, runner=self.world.runner, blob_sha256=self.world.blob)
        self.assertEqual(len(self.world.trial_calls), 2)
        self.assertTrue(commands_differ_only_by_timing(self.world.trial_calls[0], self.world.trial_calls[1]))
        self.assertIn("--completion-timing", self.world.trial_calls[1])
        environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
        self.assertEqual(environment["experiment_kind"], EXPERIMENT_KIND)
        self.assertEqual(environment["collection_protocol"]["cell_id"], CELL_ID)
        self.assertEqual(environment["collection_protocol"]["counted_blocks"], 0)
        self.assertEqual(environment["collection_protocol"]["evidence_class"], "correctness")
        self.assertEqual(environment["collection_protocol"]["analysis_protocol_version"], 0)
        self.assertIn("sentinel only", environment["collection_protocol"]["analysis_protocol_version_meaning"])
        saved_commands = [trial["command"] for trial in environment["plan"]]
        self.assertEqual(saved_commands, self.world.trial_calls)
        with self.assertRaises(Exception):
            analysis_protocol_version(environment)
        aggregate = json.loads(analyze(run_set).read_text(encoding="utf-8"))
        self.assertEqual(aggregate["shared_metrics"], [])
        self.assertFalse(aggregate["bootstrap"]["computed"])
        self.assertFalse(aggregate["winner_declared"])
        self.assertFalse(aggregate["superiority_claimed"])
        self.assertFalse((run_set / "failure.json").exists())

    def test_timeout_preserves_partial_output_and_does_not_retry(self) -> None:
        from benchmarks.collect_overhead import build_parser

        self.world.fail_timeout = True
        args = build_parser().parse_args(self._args("smoke"))
        with self.assertRaises(OverheadError) as raised:
            collect(args, runner=self.world.runner, blob_sha256=self.world.blob)
        run_set = raised.exception.run_set
        self.assertIsNotNone(run_set)
        failure = json.loads((run_set / "failure.json").read_text(encoding="utf-8"))
        self.assertFalse(failure["retried"])
        self.assertTrue(failure["execution_timeout"])
        self.assertTrue(failure["preserved_earlier_trials"])
        trial_id = failure["trial_id"]
        self.assertTrue((run_set / "runs" / f"{trial_id}.stdout").is_file())
        self.assertFalse((run_set / "runs" / f"{trial_id}.json").exists())
        self.assertFalse((run_set / "aggregate.json").exists())
        self.assertEqual(len(self.world.trial_calls), 1)

    def test_second_run_does_not_overwrite(self) -> None:
        from benchmarks.collect_overhead import build_parser

        moment = datetime.datetime(2026, 10, 14, 12, 0, tzinfo=datetime.timezone.utc)
        args = build_parser().parse_args(self._args("smoke"))
        first = collect(args, runner=self.world.runner, blob_sha256=self.world.blob, clock=lambda: moment)
        stdout = next((first / "runs").glob("*.stdout")).read_text(encoding="utf-8")
        with self.assertRaises(OverheadError):
            collect(args, runner=self.world.runner, blob_sha256=self.world.blob, clock=lambda: moment)
        self.assertEqual(next((first / "runs").glob("*.stdout")).read_text(encoding="utf-8"), stdout)
        self.assertFalse((first / "aggregate.json").exists())

    def test_process_timeout_reaps_the_child(self) -> None:
        directory = pathlib.Path(self.temporary.name)
        pid_path = directory / "child.pid"
        script = (
            "import os, pathlib, sys, time\n"
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        result = supervise_process(
            [sys.executable, "-c", script, str(pid_path)],
            cwd=directory,
            env={},
            timeout_seconds=0.4,
        )
        self.assertTrue(result.timed_out)
        pid = int(pid_path.read_text(encoding="utf-8"))
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_timed_out_stdout_is_not_accepted_and_stderr_is_preserved(self) -> None:
        from benchmarks.collect_overhead import bind_commands, build_plan, execute_plan

        run_set = self.world.root / "timeout-evidence"
        (run_set / "runs").mkdir(parents=True)
        plan = bind_commands(build_plan(0), self.world.binary, self.world.repo / "models/batch_add_f32.onnx")
        checked = {
            "commit": COMMIT,
            "queried": {"benchmark_binary_sha256": "ab" * 32},
            "verified_configuration": {"model_sha256": MODEL_SHA256},
        }

        def runner(command, cwd=None, env=None, timeout_seconds=None):
            del command, cwd, env, timeout_seconds
            return SupervisedResult(-1, "partial-stdout", "partial-stderr", timed_out=True)

        accepted = execute_plan(
            run_set=run_set,
            binary=self.world.binary,
            repo_root=self.world.repo,
            model=self.world.repo / "models/batch_add_f32.onnx",
            plan=plan,
            checked=checked,
            runner=runner,
            sleep=None,
        )
        self.assertFalse(accepted)
        failure = json.loads((run_set / "failure.json").read_text(encoding="utf-8"))
        trial_id = failure["trial_id"]
        self.assertFalse(failure["retried"])
        self.assertTrue(failure["execution_timeout"])
        self.assertEqual((run_set / "runs" / f"{trial_id}.stdout").read_text(encoding="utf-8"), "partial-stdout")
        self.assertEqual((run_set / "runs" / f"{trial_id}.stderr").read_text(encoding="utf-8"), "partial-stderr")
        self.assertFalse((run_set / "runs" / f"{trial_id}.json").exists())

    def test_failed_counted_arm_stops_the_study_and_keeps_earlier_trials(self) -> None:
        from benchmarks.collect_overhead import build_parser

        def runner(command, cwd=None, env=None, timeout_seconds=None):
            result = self.world.runner(command, cwd=cwd, env=env, timeout_seconds=timeout_seconds)
            if len(self.world.trial_calls) == 3:
                return SupervisedResult(1, result.stdout, "counted arm failed")
            return result

        args = build_parser().parse_args(self._args("exploratory", "1"))
        with self.assertRaises(OverheadError) as raised:
            collect(args, runner=runner, blob_sha256=self.world.blob)
        run_set = raised.exception.run_set
        failure = json.loads((run_set / "failure.json").read_text(encoding="utf-8"))
        self.assertFalse(failure["retried"])
        self.assertEqual(failure["trial_id"], "overhead-b00-off")
        self.assertTrue((run_set / "runs" / "overhead-pilot-off.json").is_file())
        self.assertTrue((run_set / "runs" / "overhead-pilot-on.json").is_file())
        self.assertFalse((run_set / "runs" / "overhead-b00-off.json").exists())
        self.assertTrue((run_set / "runs" / "overhead-b00-off.stderr").is_file())
        self.assertFalse((run_set / "aggregate.json").exists())
        self.assertEqual(len(self.world.trial_calls), 3)
