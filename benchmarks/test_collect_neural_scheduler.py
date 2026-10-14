#!/usr/bin/env python3
"""Deterministic tests for the neural scheduler collector."""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import unittest

from benchmarks.collect_neural_scheduler import (
    CALIBRATION_ONLY_NEURAL_CELLS,
    CALIBRATION_VARIANT_NONZERO_BATCH_DELAY_V1,
    EXPERIMENT_KIND,
    NEURAL_LOCAL_CELLS,
    NEURAL_LOCAL_CELL_BY_ID,
    NEURAL_MODELS,
    REFERENCE_CHECKING_POLICY,
    REPORTABLE_MODEL_ID,
    NeuralCollectorError,
    bind_commands,
    build_plan,
    build_parser,
    collect,
    counted_runs_for,
    evidence_class_for,
    model_contract,
    neural_scheduler_command,
    open_loop_profile,
    resolve_neural_cell,
    resolve_neural_cells,
    validate_reportable_args,
    validate_trial,
    verify_neural_artifacts,
)
from benchmarks.collect_scheduler import BINARY_NAME, CELLS, CELL_BY_ID, SCHEMA_VERSION, scheduler_command
from benchmarks.test_collect_scheduler import COMMIT, World, dump_report, sample_report

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GEMM = NEURAL_MODELS["gemm_f32"]


class NeuralWorld(World):
    def __init__(self, root: pathlib.Path, model_id: str = "gemm_f32") -> None:
        super().__init__(root)
        self.model_id = model_id
        self.contract = model_contract(model_id)

    def build(self) -> None:
        super().build()
        model = self.repo / self.contract["model_relative"]
        model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / self.contract["model_relative"], model)
        sidecar = self.repo / self.contract["sidecar_relative"]
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / self.contract["sidecar_relative"], sidecar)

    def blob(self, _repo: pathlib.Path, _commit: str, relative: str) -> str:
        if relative == self.contract["model_relative"]:
            return self.contract["model_sha256"]
        if relative == self.contract["sidecar_relative"]:
            return self.contract["sidecar_sha256"]
        raise AssertionError(relative)

    def runner(self, args, cwd=None, env=None, timeout_seconds=None):
        if pathlib.Path(args[0]).name == BINARY_NAME and "--format" in args:
            if "--model-id" not in args:
                raise AssertionError("neural trials must pass --model-id")
            model_id = args[args.index("--model-id") + 1]
            if model_id != self.model_id:
                raise AssertionError(model_id)
        return super().runner(args, cwd=cwd, env=env, timeout_seconds=timeout_seconds)


def _gemm_report(model_path: str, cell_id: str = "A", **overrides) -> dict:
    cell = CELL_BY_ID[cell_id]
    document = sample_report(
        model=model_path,
        scheduler={
            "worker_count": 1,
            "queue_capacity": 64,
            "enable_batching": cell["enable_batching"],
            "max_batch_size": cell["max_batch_size"],
            "max_batch_delay_ms": cell["max_batch_delay_ms"],
            "enable_metrics": True,
        },
    )
    document["generator"]["warmup_requests"] = 100
    document.update(overrides)
    return document


class ContractTest(unittest.TestCase):
    def test_gemm_pins_and_widths(self) -> None:
        contract = model_contract("gemm_f32")
        self.assertEqual(contract["model_sha256"], GEMM["model_sha256"])
        self.assertEqual(contract["sidecar_sha256"], GEMM["sidecar_sha256"])
        self.assertEqual(contract["input_width"], 512)
        self.assertEqual(contract["output_width"], 512)
        self.assertEqual(REFERENCE_CHECKING_POLICY, "precomputed_reference_compare")

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(NeuralCollectorError):
            model_contract("batch_add_f32")

    def test_modes_and_counts(self) -> None:
        self.assertEqual(evidence_class_for("smoke"), "correctness")
        self.assertEqual(evidence_class_for("calibration"), "calibration")
        self.assertEqual(evidence_class_for("reportable"), "confirmatory")
        self.assertEqual(counted_runs_for("smoke", None), 0)
        self.assertEqual(counted_runs_for("reportable", None), 20)
        with self.assertRaises(NeuralCollectorError):
            counted_runs_for("reportable", 5)
        with self.assertRaises(NeuralCollectorError):
            counted_runs_for("smoke", 1)


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


class NeuralLocalCellTest(unittest.TestCase):
    def test_g1_g2_exact_definitions(self) -> None:
        self.assertEqual(
            NEURAL_LOCAL_CELL_BY_ID["G1"],
            {"id": "G1", "enable_batching": True, "max_batch_size": 4, "max_batch_delay_ms": 1},
        )
        self.assertEqual(
            NEURAL_LOCAL_CELL_BY_ID["G2"],
            {"id": "G2", "enable_batching": True, "max_batch_size": 4, "max_batch_delay_ms": 2},
        )
        self.assertEqual(CALIBRATION_ONLY_NEURAL_CELLS, frozenset({"G1", "G2"}))
        self.assertEqual(len(NEURAL_LOCAL_CELLS), 2)

    def test_g1_g2_distinct_from_historical_g_and_h(self) -> None:
        self.assertNotIn("G1", CELL_BY_ID)
        self.assertNotIn("G2", CELL_BY_ID)
        self.assertEqual(CELL_BY_ID["G"]["max_batch_size"], 8)
        self.assertEqual(CELL_BY_ID["G"]["max_batch_delay_ms"], 1)
        self.assertEqual(CELL_BY_ID["H"]["max_batch_size"], 16)
        self.assertEqual(CELL_BY_ID["H"]["max_batch_delay_ms"], 1)
        self.assertNotEqual(NEURAL_LOCAL_CELL_BY_ID["G1"], CELL_BY_ID["G"])
        self.assertNotEqual(NEURAL_LOCAL_CELL_BY_ID["G2"], CELL_BY_ID["H"])

    def test_shared_add_registry_unchanged(self) -> None:
        self.assertEqual(
            [cell["id"] for cell in CELLS],
            ["A", "B", "C", "D", "E", "F", "G", "H"],
        )
        self.assertEqual(set(CELL_BY_ID), {"A", "B", "C", "D", "E", "F", "G", "H"})
        shared_a = CELL_BY_ID["A"]
        resolved_a = resolve_neural_cell("A")
        self.assertEqual(resolved_a, shared_a)
        self.assertIsNot(resolved_a, shared_a)
        resolved_a["max_batch_size"] = 99
        self.assertEqual(CELL_BY_ID["A"]["max_batch_size"], 1)

    def test_resolve_copies_and_rejects_unknown_or_duplicate(self) -> None:
        g1 = resolve_neural_cell("G1")
        self.assertIsNot(g1, NEURAL_LOCAL_CELL_BY_ID["G1"])
        g1["max_batch_delay_ms"] = 99
        self.assertEqual(NEURAL_LOCAL_CELL_BY_ID["G1"]["max_batch_delay_ms"], 1)
        with self.assertRaises(NeuralCollectorError):
            resolve_neural_cell("Z9")
        with self.assertRaises(NeuralCollectorError) as raised:
            resolve_neural_cells(["A", "G1", "A"], "calibration")
        self.assertIn("duplicate", str(raised.exception))
        with self.assertRaises(NeuralCollectorError):
            resolve_neural_cells([], "calibration")

    def test_calibration_only_mode_gate(self) -> None:
        cells = resolve_neural_cells(["A", "G1", "G2"], "calibration")
        self.assertEqual([cell["id"] for cell in cells], ["A", "G1", "G2"])
        with self.assertRaises(NeuralCollectorError) as smoke:
            resolve_neural_cells(["A", "G1"], "smoke")
        self.assertIn("calibration-only", str(smoke.exception))
        with self.assertRaises(NeuralCollectorError) as reportable:
            resolve_neural_cells(["A", "G2"], "reportable")
        self.assertIn("calibration-only", str(reportable.exception))


class CommandAndPlanTest(unittest.TestCase):
    def test_command_inserts_model_id_and_not_add_fallback(self) -> None:
        binary = pathlib.Path("/tmp") / BINARY_NAME
        model = pathlib.Path("/models/gemm_f32.onnx")
        profile = open_loop_profile("neural_open", 120.0)
        command = neural_scheduler_command(
            binary,
            model,
            "gemm_f32",
            CELL_BY_ID["C"],
            profile,
            _measurement(),
        )
        self.assertEqual(command[command.index("--model-id") + 1], "gemm_f32")
        self.assertEqual(command[command.index("--model") + 1], str(model))
        self.assertIn("--enable-batching", command)
        self.assertEqual(command[command.index("--max-batch-size") + 1], "4")
        self.assertEqual(command[command.index("--target-rps") + 1], "120")
        self.assertNotIn("batch_add_f32", " ".join(command))

    def test_a_g1_g2_runner_argv(self) -> None:
        binary = pathlib.Path("/tmp") / BINARY_NAME
        model = pathlib.Path("/models/gemm_f32.onnx")
        profile = open_loop_profile("neural_open", 500.0)
        measurement = _measurement()
        for cell_id, enable, size, delay in (
            ("A", False, "1", "0"),
            ("G1", True, "4", "1"),
            ("G2", True, "4", "2"),
        ):
            cell = resolve_neural_cell(cell_id)
            command = neural_scheduler_command(
                binary, model, "gemm_f32", cell, profile, measurement
            )
            self.assertEqual(command[command.index("--model-id") + 1], "gemm_f32")
            self.assertEqual(command[command.index("--model") + 1], str(model))
            self.assertEqual(command[command.index("--max-batch-size") + 1], size)
            self.assertEqual(command[command.index("--max-batch-delay-ms") + 1], delay)
            if enable:
                self.assertIn("--enable-batching", command)
            else:
                self.assertNotIn("--enable-batching", command)

    def test_add_scheduler_command_unchanged_for_shared_cells(self) -> None:
        binary = pathlib.Path("/tmp") / BINARY_NAME
        model = pathlib.Path("/models/batch_add_f32.onnx")
        profile = open_loop_profile("neural_open", 50.0)
        add_command = scheduler_command(binary, model, CELL_BY_ID["A"], profile, _measurement())
        self.assertNotIn("--model-id", add_command)
        self.assertNotIn("--enable-batching", add_command)
        self.assertEqual(add_command[add_command.index("--max-batch-delay-ms") + 1], "0")

    def test_plan_rotates_a_then_c(self) -> None:
        cells = [CELL_BY_ID["A"], CELL_BY_ID["C"]]
        profiles = [open_loop_profile("neural_open", 80.0)]
        smoke = build_plan(cells, profiles, 0)
        self.assertEqual([trial["cell_id"] for trial in smoke], ["A", "C"])
        self.assertTrue(all(trial["pilot"] for trial in smoke))
        plan = build_plan(cells, profiles, 2)
        counted = [trial for trial in plan if not trial["pilot"]]
        self.assertEqual([trial["cell_id"] for trial in counted], ["A", "C", "C", "A"])
        self.assertEqual([trial["block_index"] for trial in counted], [1, 1, 2, 2])

    def test_six_trial_plan_for_a_g1_g2(self) -> None:
        cells = resolve_neural_cells(["A", "G1", "G2"], "calibration")
        profiles = [open_loop_profile("neural_open", 500.0)]
        plan = build_plan(cells, profiles, 1)
        self.assertEqual(len(plan), 6)
        self.assertEqual(
            [trial["trial_id"] for trial in plan],
            [
                "pilot-A-neural_open",
                "pilot-G1-neural_open",
                "pilot-G2-neural_open",
                "A-neural_open-01",
                "G1-neural_open-01",
                "G2-neural_open-01",
            ],
        )
        self.assertEqual(
            [trial["cell_id"] for trial in plan],
            ["A", "G1", "G2", "A", "G1", "G2"],
        )
        self.assertEqual([trial["pilot"] for trial in plan], [True, True, True, False, False, False])
        counted = [trial for trial in plan if not trial["pilot"]]
        self.assertEqual([trial["block_index"] for trial in counted], [1, 1, 1])
        self.assertEqual([trial["position_in_block"] for trial in counted], [0, 1, 2])


class ValidationGateTest(unittest.TestCase):
    def test_reportable_requires_frozen_rate_and_gemm(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--binary",
                "/tmp/x",
                "--repo-root",
                ".",
                "--expected-commit",
                COMMIT,
                "--mode",
                "reportable",
                "--model-id",
                "mlp_small_f32",
                "--cells",
                "A,C",
                "--target-rps",
                "100",
                "--offered-rate-frozen",
            ]
        )
        with self.assertRaises(NeuralCollectorError):
            validate_reportable_args(args)
        args.model_id = REPORTABLE_MODEL_ID
        args.offered_rate_frozen = False
        with self.assertRaises(NeuralCollectorError):
            validate_reportable_args(args)

    def test_reportable_rejects_g1_g2_cells_argument(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--binary",
                "/tmp/x",
                "--repo-root",
                ".",
                "--expected-commit",
                COMMIT,
                "--mode",
                "reportable",
                "--model-id",
                REPORTABLE_MODEL_ID,
                "--cells",
                "A,G1,G2",
                "--target-rps",
                "100",
                "--offered-rate-frozen",
            ]
        )
        with self.assertRaises(NeuralCollectorError) as raised:
            validate_reportable_args(args)
        self.assertIn("A,C", str(raised.exception))

    def test_expected_launch_detects_wrong_reported_delay(self) -> None:
        document = sample_report(
            model="/models/gemm_f32.onnx",
            scheduler={
                "worker_count": 1,
                "queue_capacity": 64,
                "enable_batching": True,
                "max_batch_size": 4,
                "max_batch_delay_ms": 2,
                "enable_metrics": True,
            },
        )
        document["generator"]["warmup_requests"] = 100
        expected = {
            "model_path": "/models/gemm_f32.onnx",
            "workload": "open_loop_constant",
            "worker_count": 1,
            "queue_capacity": 64,
            "enable_batching": True,
            "max_batch_size": 4,
            "max_batch_delay_ms": 1,
            "warmup_requests": 100,
            "check_interval": 1,
            "duration_ms": 1000,
            "lag_tolerance_ms": 1000,
            "comparison_class": "primary",
            "analysis_protocol_version": 2,
            "target_rps": 500.0,
        }
        _document, _status, problems = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=False,
        )
        self.assertTrue(
            any("max_batch_delay_ms" in item and "expected 1" in item for item in problems)
        )

    def test_zero_rejection_gate(self) -> None:
        document = _gemm_report("/models/gemm_f32.onnx", "A")
        expected = {
            "model_path": "/models/gemm_f32.onnx",
            "workload": "open_loop_constant",
            "worker_count": 1,
            "queue_capacity": 64,
            "enable_batching": False,
            "max_batch_size": 1,
            "max_batch_delay_ms": 0,
            "warmup_requests": 100,
            "check_interval": 1,
            "duration_ms": 1000,
            "lag_tolerance_ms": 1000,
            "comparison_class": "primary",
            "analysis_protocol_version": 2,
            "target_rps": 50.0,
        }
        _document, status, problems = validate_trial(
            dump_report(document),
            expected=expected,
            mode="reportable",
            require_zero_rejections=True,
        )
        self.assertEqual(problems, [])
        self.assertEqual(status["delivery_class"], "STRICT_PRIMARY")
        document["rejections"]["full"] = 1
        document["accepted"] = 99
        document["succeeded"] = 99
        document["successful_e2e_latency"]["sample_count"] = 99
        document["metrics"]["engine_run_count"] = 99
        document["metrics"]["requests_entered_engine_runs"] = 99
        document["metrics"]["batch_size_histogram"] = [0, 99] + [0] * 31
        document["correctness_checked"] = 99
        document["generator_lag"]["count"] = 100
        _document, _status, problems = validate_trial(
            dump_report(document),
            expected=expected,
            mode="reportable",
            require_zero_rejections=True,
        )
        self.assertTrue(any("admission rejection" in item for item in problems))

    def test_generator_skip_fails_strict_delivery(self) -> None:
        document = _gemm_report("/models/gemm_f32.onnx")
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
        _document, status, problems = validate_trial(
            dump_report(document),
            expected={
                "model_path": "/models/gemm_f32.onnx",
                "workload": "open_loop_constant",
                "worker_count": 1,
                "queue_capacity": 64,
                "enable_batching": False,
                "max_batch_size": 1,
                "max_batch_delay_ms": 0,
                "warmup_requests": 100,
                "check_interval": 1,
                "duration_ms": 1000,
                "lag_tolerance_ms": 1000,
                "comparison_class": "primary",
                "analysis_protocol_version": 2,
                "target_rps": 50.0,
            },
            mode="reportable",
            require_zero_rejections=True,
        )
        self.assertTrue(problems)
        self.assertNotEqual(status["delivery_class"], "STRICT_PRIMARY")


class ArtifactAndCollectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.world = NeuralWorld(pathlib.Path(self.temporary.name))
        self.world.build()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self, mode: str, **extra: str) -> list[str]:
        argv = [
            "--binary",
            str(self.world.binary),
            "--repo-root",
            str(self.world.repo),
            "--expected-commit",
            COMMIT,
            "--output-root",
            str(self.world.root / "neural-output"),
            "--mode",
            mode,
            "--model-id",
            self.world.model_id,
            "--cells",
            "A,C",
            "--inter-trial-delay-seconds",
            "0",
            "--warmup-requests",
            "100",
        ]
        for key, value in extra.items():
            argv.extend(["--" + key.replace("_", "-"), value])
        return argv

    def test_verify_artifacts_and_wrong_sidecar(self) -> None:
        artifacts = verify_neural_artifacts(self.world.repo, "gemm_f32")
        self.assertEqual(artifacts["model_sha256"], GEMM["model_sha256"])
        sidecar = self.world.repo / GEMM["sidecar_relative"]
        sidecar.write_bytes(b"corrupted")
        with self.assertRaises(NeuralCollectorError) as raised:
            verify_neural_artifacts(self.world.repo, "gemm_f32")
        self.assertIn("sidecar SHA-256", str(raised.exception))

    def test_smoke_collects_pilot_pair_with_model_id(self) -> None:
        args = build_parser().parse_args(self._args("smoke"))
        run_set = collect(args, runner=self.world.runner)
        self.assertEqual(len(self.world.trial_calls), 2)
        for command in self.world.trial_calls:
            self.assertEqual(command[command.index("--model-id") + 1], "gemm_f32")
            self.assertTrue(command[command.index("--model") + 1].endswith("gemm_f32.onnx"))
        environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
        self.assertEqual(environment["experiment_kind"], EXPERIMENT_KIND)
        self.assertEqual(environment["collection_protocol"]["evidence_class"], "correctness")
        self.assertEqual(
            environment["verified_configuration"]["reference_checking_policy"],
            REFERENCE_CHECKING_POLICY,
        )
        self.assertEqual(environment["verified_configuration"]["model_sha256"], GEMM["model_sha256"])
        self.assertEqual(
            environment["verified_configuration"]["weight_sidecar_sha256"],
            GEMM["sidecar_sha256"],
        )
        self.assertTrue(run_set.name.startswith("neural-gemm-f32-"))
        self.assertEqual(SCHEMA_VERSION, 3)

    def test_reportable_refuses_unfrozen_rate(self) -> None:
        args = build_parser().parse_args(
            self._args("reportable", counted_runs="10", target_rps="90")
        )
        with self.assertRaises(NeuralCollectorError) as raised:
            collect(args, runner=self.world.runner)
        self.assertIn("offered-rate-frozen", str(raised.exception))

    def test_smoke_rejects_g1_g2(self) -> None:
        args = build_parser().parse_args(self._args("smoke", cells="A,G1,G2"))
        with self.assertRaises(NeuralCollectorError) as raised:
            collect(args, runner=self.world.runner)
        self.assertIn("calibration-only", str(raised.exception))
        self.assertEqual(self.world.trial_calls, [])

    def test_reportable_rejects_g1_g2(self) -> None:
        argv = self._args("reportable", cells="A,G1,G2", counted_runs="10", target_rps="90")
        argv.append("--offered-rate-frozen")
        args = build_parser().parse_args(argv)
        with self.assertRaises(NeuralCollectorError) as raised:
            collect(args, runner=self.world.runner)
        message = str(raised.exception)
        self.assertTrue("A,C" in message or "calibration-only" in message)
        self.assertEqual(self.world.trial_calls, [])

    def test_reportable_a_c_accepted_with_frozen_rate_preflight_path(self) -> None:
        argv = self._args("reportable", counted_runs="10", target_rps="90")
        argv.append("--offered-rate-frozen")
        args = build_parser().parse_args(argv)
        validate_reportable_args(args)
        self.assertEqual(args.cells, "A,C")
        self.assertEqual(args.model_id, REPORTABLE_MODEL_ID)
        self.assertTrue(args.offered_rate_frozen)

    def test_calibration_a_g1_g2_records_variant_and_delays(self) -> None:
        argv = self._args(
            "calibration",
            cells="A,G1,G2",
            counted_runs="1",
            target_rps="500",
            baseline_cell="A",
            alternative_cell="G1",
        )
        args = build_parser().parse_args(argv)
        run_set = collect(args, runner=self.world.runner)
        self.assertEqual(len(self.world.trial_calls), 6)
        environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
        protocol = environment["collection_protocol"]
        self.assertEqual(protocol["evidence_class"], "calibration")
        self.assertEqual(protocol["experiment_kind"], EXPERIMENT_KIND)
        self.assertEqual(protocol["reference_checking_policy"], REFERENCE_CHECKING_POLICY)
        self.assertEqual(protocol["calibration_variant"], CALIBRATION_VARIANT_NONZERO_BATCH_DELAY_V1)
        self.assertEqual(
            environment["source_defined_policy"]["calibration_variant"],
            CALIBRATION_VARIANT_NONZERO_BATCH_DELAY_V1,
        )
        definitions = {cell["id"]: cell for cell in protocol["cell_definitions"]}
        self.assertEqual(definitions["A"]["max_batch_delay_ms"], 0)
        self.assertFalse(definitions["A"]["enable_batching"])
        self.assertEqual(definitions["G1"]["max_batch_delay_ms"], 1)
        self.assertEqual(definitions["G2"]["max_batch_delay_ms"], 2)
        self.assertEqual(definitions["G1"]["max_batch_size"], 4)
        self.assertEqual(definitions["G2"]["max_batch_size"], 4)
        plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))["trials"]
        self.assertEqual(len(plan), 6)
        self.assertEqual(
            [trial["trial_id"] for trial in plan],
            [
                "pilot-A-neural_open",
                "pilot-G1-neural_open",
                "pilot-G2-neural_open",
                "A-neural_open-01",
                "G1-neural_open-01",
                "G2-neural_open-01",
            ],
        )
        by_id = {trial["trial_id"]: trial for trial in plan}
        self.assertNotIn("--enable-batching", by_id["A-neural_open-01"]["command"])
        self.assertEqual(
            by_id["G1-neural_open-01"]["command"][
                by_id["G1-neural_open-01"]["command"].index("--max-batch-delay-ms") + 1
            ],
            "1",
        )
        self.assertEqual(
            by_id["G2-neural_open-01"]["command"][
                by_id["G2-neural_open-01"]["command"].index("--max-batch-delay-ms") + 1
            ],
            "2",
        )
        self.assertIn("--enable-batching", by_id["G1-neural_open-01"]["command"])
        self.assertIn("--enable-batching", by_id["G2-neural_open-01"]["command"])
        meta_g1 = json.loads(
            (run_set / "runs" / "G1-neural_open-01.meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            meta_g1["command"][meta_g1["command"].index("--max-batch-delay-ms") + 1], "1"
        )
        self.assertTrue(meta_g1["collector_valid"])
        g1_cmd = self.world.trial_calls[4]
        g2_cmd = self.world.trial_calls[5]
        self.assertEqual(g1_cmd[g1_cmd.index("--max-batch-delay-ms") + 1], "1")
        self.assertEqual(g2_cmd[g2_cmd.index("--max-batch-delay-ms") + 1], "2")
        self.assertEqual(g1_cmd[g1_cmd.index("--model-id") + 1], "gemm_f32")

    def test_wrong_commit_fails_before_launch(self) -> None:
        args = build_parser().parse_args(self._args("smoke", expected_commit="0" * 40))
        with self.assertRaises(NeuralCollectorError):
            collect(args, runner=self.world.runner)
        self.assertEqual(self.world.trial_calls, [])

    def test_failure_preserves_evidence_without_retry(self) -> None:
        original = self.world.runner

        def flaky(args, cwd=None, env=None, timeout_seconds=None):
            result = original(args, cwd=cwd, env=env, timeout_seconds=timeout_seconds)
            if pathlib.Path(args[0]).name == BINARY_NAME and "--format" in args:
                if len(self.world.trial_calls) >= 1:
                    from benchmarks.collect_scheduler import SupervisedResult

                    return SupervisedResult(1, result.stdout, "boom\n", timed_out=False)
            return result

        args = build_parser().parse_args(self._args("smoke"))
        with self.assertRaises(NeuralCollectorError) as raised:
            collect(args, runner=flaky)
        run_set = raised.exception.run_set
        assert run_set is not None
        self.assertTrue((run_set / "failure.json").is_file())
        self.assertTrue((run_set / "runs" / "pilot-A-low_open.stdout").is_file())
        failure = json.loads((run_set / "failure.json").read_text(encoding="utf-8"))
        self.assertFalse(failure["retried"])
        self.assertTrue(failure["preserved_earlier_trials"])

    def test_exclusive_create_refuses_overwrite(self) -> None:
        args = build_parser().parse_args(self._args("smoke"))
        first = collect(args, runner=self.world.runner)

        def fixed_clock():
            import datetime

            return datetime.datetime(2026, 10, 14, 12, 0, 0, tzinfo=datetime.timezone.utc)

        self.world.trial_calls.clear()
        with self.assertRaises(NeuralCollectorError) as raised:
            collect(args, runner=self.world.runner, clock=fixed_clock)
            collect(args, runner=self.world.runner, clock=fixed_clock)
        self.assertIn("exists", str(raised.exception).lower())
        self.assertTrue(first.is_dir())


class BindCommandTest(unittest.TestCase):
    def test_bound_plan_embeds_exact_argv(self) -> None:
        binary = pathlib.Path("/tmp") / BINARY_NAME
        model = pathlib.Path("/models/gemm_f32.onnx")
        cells = {"A": CELL_BY_ID["A"], "C": CELL_BY_ID["C"]}
        profiles = {"neural_open": open_loop_profile("neural_open", 70.0)}
        plan = build_plan([cells["A"], cells["C"]], [profiles["neural_open"]], 0)
        measurement = {
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
        bound = bind_commands(
            plan,
            binary=binary,
            model=model,
            model_id="gemm_f32",
            cells=cells,
            profiles=profiles,
            measurement=measurement,
        )
        self.assertEqual(bound[0]["command"][bound[0]["command"].index("--model-id") + 1], "gemm_f32")
        self.assertNotIn("--enable-batching", bound[0]["command"])
        self.assertIn("--enable-batching", bound[1]["command"])


if __name__ == "__main__":
    unittest.main()
