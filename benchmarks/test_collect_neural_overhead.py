#!/usr/bin/env python3
"""Deterministic tests for the neural batch-decision timing overhead collector."""

from __future__ import annotations

import datetime
import json
import pathlib
import tempfile
import unittest

from benchmarks.analyze_neural_overhead import analyze
from benchmarks.analyze_scheduler import analysis_protocol_version
from benchmarks.collect_neural_overhead import (
    ALLOWED_POLICY_CELLS,
    BOOTSTRAP_SEED,
    DURATION_MS,
    EVENT_CAPACITY,
    EXPERIMENT_KIND,
    EXPLORATORY_COUNTED_BLOCKS,
    FORMAL_COUNTED_BLOCKS,
    INTENDED_VARIABLE,
    MODEL_ID,
    OVERHEAD_PROTOCOL_MARKER,
    TARGET_RPS,
    NeuralOverheadError,
    arm_order,
    bind_commands,
    build_plan,
    build_parser,
    collect,
    command_for_arm,
    commands_differ_only_by_batch_decision,
    counted_blocks_for,
    evidence_class_for,
    measurement_for,
    policy_cell,
    shared_launch,
    validate_arm,
)
from benchmarks.collect_neural_scheduler import (
    NEURAL_MODELS,
    REFERENCE_CHECKING_POLICY,
    model_contract,
)
from benchmarks.collect_scheduler import (
    BATCH_DECISION_SCHEMA_VERSION,
    BINARY_NAME,
    SCHEMA_VERSION,
    SupervisedResult,
)
from benchmarks.test_collect_neural_scheduler import NeuralWorld
from benchmarks.test_collect_scheduler import (
    COMMIT,
    ProcessResult,
    dump_report,
    sample_report,
    schema5_batch_decision,
    schema5_event,
    schema5_report,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GEMM = NEURAL_MODELS["gemm_f32"]
MODEL = pathlib.Path("/models/gemm_f32.onnx")


def _g2() -> dict:
    return policy_cell("G2")


def _g4() -> dict:
    return policy_cell("G4")


def _scale(document: dict, count: int) -> None:
    document["accepted"] = count
    document["succeeded"] = count
    document["submit_attempts"] = count
    document["scheduled_opportunities"] = count
    document["successful_e2e_latency"]["sample_count"] = count
    document["successful_e2e_latency"]["precise_tail"] = count >= 100
    document["metrics"]["engine_run_count"] = count
    document["metrics"]["requests_entered_engine_runs"] = count
    document["metrics"]["batch_size_histogram"] = [0, count] + [0] * 31
    document["correctness_checked"] = count
    document["generator_lag"]["count"] = count
    if document.get("burst_phases"):
        document["burst_phases"][0]["scheduled_opportunities"] = count


def _arm_report(arm: str, cell: dict | None = None, **overrides) -> dict:
    cell = cell or _g2()
    duration_ns = DURATION_MS * 1_000_000
    if arm == "on":
        events = [
            schema5_event(
                batch_id=index + 1,
                open_ns=index * 20,
                seal_ns=index * 20 + 10,
                batch_size=1,
                seal_reason="deadline_expired",
            )
            for index in range(100)
        ]
        document = schema5_report()
        document["batch_decision"] = schema5_batch_decision(events=events, sealed=100)
        document["scheduler"]["enable_batch_decision_timing"] = True
        document["scheduler"]["batch_decision_event_capacity"] = EVENT_CAPACITY
    else:
        document = sample_report()
    document["model"] = str(MODEL)
    document["configured_duration_nanoseconds"] = duration_ns
    document["arrival_window_nanoseconds"] = duration_ns
    document["scheduled_span_nanoseconds"] = duration_ns - 10_000_000
    document["total_trial_nanoseconds"] = duration_ns + document["completion_nanoseconds"]
    document["generator"]["warmup_requests"] = 100
    document["scheduler"]["enable_batching"] = True
    document["scheduler"]["max_batch_size"] = 4
    document["scheduler"]["max_batch_delay_ms"] = 2
    document["scheduler"]["adaptive_batching"] = False
    document["scheduler"]["arrival_aware_batching"] = bool(cell["arrival_aware_batching"])
    document["scheduler"]["enable_metrics"] = True
    document["burst_phases"] = [
        {
            "index": 0,
            "requests_per_second": TARGET_RPS,
            "duration_nanoseconds": duration_ns,
            "scheduled_opportunities": 100,
        }
    ]
    _scale(document, 100)
    document.update(overrides)
    return document


class NeuralOverheadWorld(NeuralWorld):
    def __init__(self, root: pathlib.Path, cell_id: str = "G2") -> None:
        super().__init__(root, model_id=MODEL_ID)
        self.cell_id = cell_id
        self.cell = policy_cell(cell_id)

    def runner(self, args, cwd=None, env=None, timeout_seconds=None):
        if pathlib.Path(args[0]).name == BINARY_NAME and "--format" in args:
            if env is None or env.get("ORT_DISABLE_TELEMETRY") != "1":
                raise AssertionError("telemetry was not disabled")
            self.trial_calls.append(list(args))
            if self.fail_timeout:
                return SupervisedResult(-1, dump_report(sample_report()), "", timed_out=True)
            arm = "on" if "--batch-decision-timing" in args else "off"
            model = args[args.index("--model") + 1]
            document = _arm_report(arm, self.cell, model=model)
            if "--completion-timing" in args:
                raise AssertionError("completion timing must stay off")
            if arm == "on":
                if args[args.index("--batch-decision-event-capacity") + 1] != str(EVENT_CAPACITY):
                    raise AssertionError("capacity")
            if "--arrival-aware-batching" in args:
                document["scheduler"]["arrival_aware_batching"] = True
            else:
                document["scheduler"]["arrival_aware_batching"] = False
            if "--target-rps" in args:
                document["burst_phases"][0]["requests_per_second"] = float(
                    args[args.index("--target-rps") + 1]
                )
            return ProcessResult(0, dump_report(document), "")
        return super().runner(args, cwd=cwd, env=env, timeout_seconds=timeout_seconds)


class IdentityAndPlanTest(unittest.TestCase):
    def test_experiment_kind_and_protocol_sentinel(self) -> None:
        self.assertEqual(EXPERIMENT_KIND, "batch_decision_timing_overhead")
        self.assertEqual(INTENDED_VARIABLE, "enable_batch_decision_timing")
        self.assertEqual(OVERHEAD_PROTOCOL_MARKER, 0)
        self.assertEqual(BOOTSTRAP_SEED, 20261015)
        self.assertEqual(MODEL_ID, "gemm_f32")
        self.assertEqual(REFERENCE_CHECKING_POLICY, "precomputed_reference_compare")

    def test_g2_and_g4_policy_contracts(self) -> None:
        g2 = _g2()
        g4 = _g4()
        self.assertEqual(ALLOWED_POLICY_CELLS, frozenset({"G2", "G4"}))
        self.assertEqual(g2["enable_batching"], True)
        self.assertEqual(g2["max_batch_size"], 4)
        self.assertEqual(g2["max_batch_delay_ms"], 2)
        self.assertIs(g2["adaptive_batching"], False)
        self.assertIs(g2["arrival_aware_batching"], False)
        self.assertEqual(g4["enable_batching"], True)
        self.assertEqual(g4["max_batch_size"], 4)
        self.assertEqual(g4["max_batch_delay_ms"], 2)
        self.assertIs(g4["adaptive_batching"], False)
        self.assertIs(g4["arrival_aware_batching"], True)
        launch_g2 = shared_launch(g2)
        launch_g4 = shared_launch(g4)
        self.assertFalse(launch_g2["arrival_aware_batching"])
        self.assertTrue(launch_g4["arrival_aware_batching"])
        self.assertEqual(launch_g2["target_rps"], TARGET_RPS)
        self.assertEqual(launch_g2["duration_ms"], DURATION_MS)
        self.assertFalse(launch_g2["completion_timing"])

    def test_mixed_policy_is_not_a_single_run_set(self) -> None:
        with self.assertRaises(NeuralOverheadError):
            policy_cell("G3")
        with self.assertRaises(NeuralOverheadError):
            policy_cell("A")

    def test_smoke_exploratory_formal_plans(self) -> None:
        self.assertEqual(counted_blocks_for("smoke", None), 0)
        self.assertEqual(counted_blocks_for("exploratory", None), EXPLORATORY_COUNTED_BLOCKS)
        self.assertEqual(counted_blocks_for("formal", None), FORMAL_COUNTED_BLOCKS)
        self.assertEqual(evidence_class_for("smoke", 0), "correctness")
        self.assertEqual(evidence_class_for("exploratory", 3), "exploratory")
        self.assertEqual(evidence_class_for("formal", 20), "counted_overhead")
        smoke = build_plan(0, "G2")
        exploratory = build_plan(3, "G2")
        formal = build_plan(20, "G2")
        self.assertEqual(len(smoke), 2)
        self.assertEqual(len(exploratory), 8)
        self.assertEqual(len(formal), 42)
        self.assertEqual([trial["arm"] for trial in smoke], ["off", "on"])
        self.assertTrue(all(trial["pilot"] for trial in smoke))
        counted = [trial for trial in exploratory if not trial["pilot"]]
        self.assertEqual([trial["arm"] for trial in counted], ["off", "on", "on", "off", "off", "on"])
        self.assertEqual(arm_order(0), ("off", "on"))
        self.assertEqual(arm_order(1), ("on", "off"))
        with self.assertRaises(NeuralOverheadError):
            counted_blocks_for("smoke", 1)
        with self.assertRaises(NeuralOverheadError):
            counted_blocks_for("exploratory", 2)
        with self.assertRaises(NeuralOverheadError):
            counted_blocks_for("formal", 10)


class CommandParityTest(unittest.TestCase):
    def test_off_on_command_delta_is_only_batch_decision_flags(self) -> None:
        binary = pathlib.Path("/tmp") / BINARY_NAME
        cell = _g2()
        off = command_for_arm(binary, MODEL, cell, "off")
        on = command_for_arm(binary, MODEL, cell, "on")
        self.assertNotIn("--batch-decision-timing", off)
        self.assertNotIn("--batch-decision-event-capacity", off)
        self.assertNotIn("--completion-timing", off)
        self.assertNotIn("--completion-timing", on)
        self.assertEqual(on.count("--batch-decision-timing"), 1)
        self.assertEqual(on[on.index("--batch-decision-event-capacity") + 1], "1024")
        self.assertTrue(commands_differ_only_by_batch_decision(off, on))
        self.assertIn("--model-id", off)
        self.assertEqual(off[off.index("--model-id") + 1], "gemm_f32")
        self.assertEqual(off[off.index("--target-rps") + 1], "100")
        self.assertEqual(off[off.index("--duration-ms") + 1], "5000")
        self.assertEqual(off[off.index("--workers") + 1], "1")
        self.assertEqual(off[off.index("--max-batch-size") + 1], "4")
        self.assertEqual(off[off.index("--max-batch-delay-ms") + 1], "2")
        self.assertIn("--enable-batching", off)
        self.assertNotIn("--arrival-aware-batching", off)
        g4 = _g4()
        g4_off = command_for_arm(binary, MODEL, g4, "off")
        g4_on = command_for_arm(binary, MODEL, g4, "on")
        self.assertIn("--arrival-aware-batching", g4_off)
        self.assertTrue(commands_differ_only_by_batch_decision(g4_off, g4_on))
        self.assertFalse(commands_differ_only_by_batch_decision(off, g4_on))

    def test_measurement_for_arms(self) -> None:
        off = measurement_for("off")
        on = measurement_for("on")
        self.assertNotIn("enable_batch_decision_timing", off)
        self.assertTrue(on["enable_batch_decision_timing"])
        self.assertEqual(on["batch_decision_event_capacity"], 1024)


class ValidationTest(unittest.TestCase):
    def test_schema3_off_and_schema5_on(self) -> None:
        cell = _g2()
        off_doc, off_status, off_problems = validate_arm(
            dump_report(_arm_report("off", cell)), "off", MODEL, cell
        )
        on_doc, on_status, on_problems = validate_arm(
            dump_report(_arm_report("on", cell)), "on", MODEL, cell
        )
        self.assertEqual(off_problems, [])
        self.assertEqual(on_problems, [])
        self.assertEqual(off_doc["schema_version"], SCHEMA_VERSION)
        self.assertEqual(on_doc["schema_version"], BATCH_DECISION_SCHEMA_VERSION)
        self.assertNotIn("enable_batch_decision_timing", off_doc["scheduler"])
        self.assertNotIn("batch_decision", off_doc)
        self.assertTrue(on_doc["scheduler"]["enable_batch_decision_timing"])
        self.assertEqual(on_doc["scheduler"]["batch_decision_event_capacity"], 1024)
        self.assertEqual(off_status["delivery_class"], "STRICT_PRIMARY")
        self.assertEqual(on_status["delivery_class"], "STRICT_PRIMARY")

    def test_g4_on_accepts_arrival_aware_seal_reason(self) -> None:
        cell = _g4()
        document = _arm_report("on", cell)
        document["batch_decision"]["events"][0]["seal_reason"] = "g4_arrival_aware"
        _doc, status, problems = validate_arm(dump_report(document), "on", MODEL, cell)
        self.assertEqual(problems, [])
        self.assertEqual(status["delivery_class"], "STRICT_PRIMARY")

    def test_wrong_schema_for_an_arm_is_invalid(self) -> None:
        cell = _g2()
        _doc, _status, off_problems = validate_arm(
            dump_report(_arm_report("on", cell)), "off", MODEL, cell
        )
        _doc, _status, on_problems = validate_arm(
            dump_report(_arm_report("off", cell)), "on", MODEL, cell
        )
        self.assertTrue(any("schema_version" in item for item in off_problems))
        self.assertTrue(any("schema_version" in item for item in on_problems))

    def test_delivery_hard_stop(self) -> None:
        cell = _g2()
        document = _arm_report("off", cell)
        document["submit_attempts"] = 99
        document["generator_skips_lag"] = 1
        _scale(document, 99)
        document["submit_attempts"] = 99
        document["scheduled_opportunities"] = 100
        document["generator_skips_lag"] = 1
        _doc, status, problems = validate_arm(dump_report(document), "off", MODEL, cell)
        self.assertTrue(problems)
        self.assertNotEqual(status["delivery_class"], "STRICT_PRIMARY")

    def test_trace_overflow_and_dropped_events_hard_stop(self) -> None:
        cell = _g2()
        overflow = _arm_report("on", cell)
        overflow["batch_decision"]["trace_overflow"] = True
        overflow["batch_decision"]["timing_data_valid"] = False
        overflow["batch_decision"]["events_dropped"] = 1
        overflow["batch_decision"]["sealed_batches_total"] = 101
        _doc, _status, problems = validate_arm(dump_report(overflow), "on", MODEL, cell)
        self.assertTrue(any("overflow" in item for item in problems))
        dropped = _arm_report("on", cell)
        dropped["batch_decision"]["events_dropped"] = 1
        dropped["batch_decision"]["timing_data_valid"] = False
        dropped["batch_decision"]["sealed_batches_total"] = 101
        _doc, _status, problems = validate_arm(dump_report(dropped), "on", MODEL, cell)
        self.assertTrue(any("dropped" in item for item in problems))

    def test_engine_run_mismatch_hard_stop(self) -> None:
        cell = _g2()
        document = _arm_report("on", cell)
        document["metrics"]["engine_run_count"] = 50
        _doc, _status, problems = validate_arm(dump_report(document), "on", MODEL, cell)
        self.assertTrue(any("engine_run_count" in item for item in problems))

    def test_forbidden_g2_seal_reason(self) -> None:
        cell = _g2()
        document = _arm_report("on", cell)
        document["batch_decision"]["events"][0]["seal_reason"] = "g4_arrival_aware"
        _doc, _status, problems = validate_arm(dump_report(document), "on", MODEL, cell)
        self.assertTrue(any("forbidden" in item for item in problems))

    def test_wrong_policy_identity_rejected(self) -> None:
        cell = _g2()
        document = _arm_report("off", cell)
        document["scheduler"]["arrival_aware_batching"] = True
        _doc, _status, problems = validate_arm(dump_report(document), "off", MODEL, cell)
        self.assertTrue(problems)


class CollectionSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.world = NeuralOverheadWorld(pathlib.Path(self.temporary.name), "G2")
        self.world.build()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self, mode: str, counted: str | None = None, cell: str = "G2") -> list[str]:
        argv = [
            "--binary",
            str(self.world.binary),
            "--repo-root",
            str(self.world.repo),
            "--expected-commit",
            COMMIT,
            "--output-root",
            str(self.world.root / "neural-overhead-output"),
            "--mode",
            mode,
            "--policy-cell",
            cell,
        ]
        if counted is not None:
            argv.extend(["--counted-runs", counted])
        return argv

    def test_smoke_collects_two_trials_and_excludes_pilots(self) -> None:
        args = build_parser().parse_args(self._args("smoke"))
        run_set = collect(args, runner=self.world.runner)
        self.assertEqual(len(self.world.trial_calls), 2)
        self.assertTrue(
            commands_differ_only_by_batch_decision(
                self.world.trial_calls[0], self.world.trial_calls[1]
            )
        )
        environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
        self.assertEqual(environment["experiment_kind"], EXPERIMENT_KIND)
        self.assertEqual(environment["collection_protocol"]["policy_cell"], "G2")
        self.assertEqual(environment["collection_protocol"]["counted_blocks"], 0)
        self.assertEqual(environment["collection_protocol"]["evidence_class"], "correctness")
        self.assertEqual(environment["collection_protocol"]["analysis_protocol_version"], 0)
        self.assertEqual(environment["collection_protocol"]["bootstrap_seed"], 20261015)
        self.assertEqual(
            environment["verified_configuration"]["model_sha256"], GEMM["model_sha256"]
        )
        self.assertEqual(
            environment["verified_configuration"]["weight_sidecar_sha256"],
            GEMM["sidecar_sha256"],
        )
        with self.assertRaises(Exception):
            analysis_protocol_version(environment)
        aggregate = json.loads(analyze(run_set).read_text(encoding="utf-8"))
        self.assertEqual(aggregate["shared_metrics"], [])
        self.assertFalse(aggregate["bootstrap"]["computed"])
        self.assertFalse(aggregate["winner_declared"])
        self.assertFalse(aggregate["g4_superiority_inferred"])

    def test_exploratory_eight_trial_plan_no_retry(self) -> None:
        args = build_parser().parse_args(self._args("exploratory"))
        run_set = collect(args, runner=self.world.runner)
        self.assertEqual(len(self.world.trial_calls), 8)
        plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))
        self.assertEqual(len(plan["trials"]), 8)
        aggregate = json.loads(analyze(run_set).read_text(encoding="utf-8"))
        self.assertEqual(aggregate["counted_blocks"], 3)
        self.assertFalse(aggregate["bootstrap"]["computed"])
        latency = next(item for item in aggregate["shared_metrics"] if item["metric"] == "p50_e2e_latency")
        self.assertEqual(latency["paired_block_count"], 3)
        self.assertFalse(latency["confidence_interval_computed"])
        self.assertFalse(aggregate["winner_declared"])

    def test_failed_arm_stops_without_retry(self) -> None:
        def runner(command, cwd=None, env=None, timeout_seconds=None):
            result = self.world.runner(command, cwd=cwd, env=env, timeout_seconds=timeout_seconds)
            if len(self.world.trial_calls) == 1:
                return SupervisedResult(1, result.stdout, "first arm failed")
            return result

        args = build_parser().parse_args(self._args("smoke"))
        with self.assertRaises(NeuralOverheadError) as raised:
            collect(args, runner=runner)
        run_set = raised.exception.run_set
        failure = json.loads((run_set / "failure.json").read_text(encoding="utf-8"))
        self.assertFalse(failure["retried"])
        self.assertFalse(failure["replaced"])
        self.assertTrue(failure["preserved_earlier_trials"])
        self.assertEqual(len(self.world.trial_calls), 1)
        self.assertFalse((run_set / "aggregate.json").exists())

    def test_second_run_does_not_overwrite(self) -> None:
        moment = datetime.datetime(2026, 10, 15, 12, 0, tzinfo=datetime.timezone.utc)
        args = build_parser().parse_args(self._args("smoke"))
        first = collect(args, runner=self.world.runner, clock=lambda: moment)
        stdout = next((first / "runs").glob("*.stdout")).read_text(encoding="utf-8")
        with self.assertRaises(NeuralOverheadError):
            collect(args, runner=self.world.runner, clock=lambda: moment)
        self.assertEqual(next((first / "runs").glob("*.stdout")).read_text(encoding="utf-8"), stdout)

    def test_formal_plan_is_forty_two_trials_without_execution(self) -> None:
        plan = bind_commands(
            build_plan(FORMAL_COUNTED_BLOCKS, "G2"),
            self.world.binary,
            self.world.repo / GEMM["model_relative"],
            _g2(),
        )
        self.assertEqual(len(plan), 42)
        self.assertEqual(plan[0]["arm"], "off")
        self.assertEqual(plan[1]["arm"], "on")
        self.assertTrue(plan[0]["pilot"])
        counted = [trial for trial in plan if not trial["pilot"]]
        self.assertEqual(len(counted), 40)
        self.assertEqual([trial["arm"] for trial in counted[:4]], ["off", "on", "on", "off"])

    def test_artifacts_match_committed_gemm(self) -> None:
        contract = model_contract("gemm_f32")
        self.assertEqual(contract["model_sha256"], GEMM["model_sha256"])
        source = REPO_ROOT / contract["model_relative"]
        self.assertTrue(source.is_file())
        self.assertTrue((REPO_ROOT / contract["sidecar_relative"]).is_file())


if __name__ == "__main__":
    unittest.main()
