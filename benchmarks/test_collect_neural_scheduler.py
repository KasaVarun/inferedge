#!/usr/bin/env python3
"""Deterministic tests for the neural scheduler collector."""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import unittest

from benchmarks.collect_neural_scheduler import (
    ADAPTIVE_EARLY_SEAL_CELLS,
    ADAPTIVE_EARLY_SEAL_RATE,
    AG2_OFFERED_RATE,
    AG2G3_OFFERED_RATE,
    ARRIVAL_AWARE_BATCHING_CELLS,
    ARRIVAL_AWARE_BATCHING_RATE,
    ARRIVAL_SCREEN_MAX_LAG_NS,
    CALIBRATION_ONLY_NEURAL_CELLS,
    CALIBRATION_VARIANT_ADAPTIVE_EARLY_SEAL_V1,
    CALIBRATION_VARIANT_ARRIVAL_AWARE_BATCHING_V1,
    CALIBRATION_VARIANT_NONZERO_BATCH_DELAY_V1,
    CELL_DEFINITION_ADAPTIVE_KEY,
    CELL_DEFINITION_ARRIVAL_KEY,
    CELL_DEFINITION_KEYS,
    CONTRAST_ID_AG2,
    CONTRAST_ID_AG2G3,
    CONTRAST_SIGN_G2_MINUS_A,
    CONTRAST_SIGN_G3_MINUS_A,
    CONTRAST_SIGN_G3_MINUS_G2,
    EXPERIMENT_KIND,
    NEURAL_LOCAL_CELLS,
    NEURAL_LOCAL_CELL_BY_ID,
    NEURAL_MODELS,
    REFERENCE_CHECKING_POLICY,
    REPORTABLE_CONTRAST_AC,
    REPORTABLE_CONTRAST_AG2,
    REPORTABLE_CONTRAST_AG2G3,
    REPORTABLE_MODEL_ID,
    THREE_ARM_POSITION_COUNTS_20,
    NeuralCollectorError,
    adaptive_batching_for_cell,
    arrival_aware_batching_for_cell,
    arrival_screen_problems,
    bind_commands,
    build_plan,
    build_parser,
    collect,
    counted_runs_for,
    evidence_class_for,
    model_contract,
    multi_request_batch_fraction,
    multi_size_batch_fraction,
    neural_scheduler_command,
    normalize_cell_definition,
    open_loop_profile,
    realized_mean_batch_size,
    require_rotation_position_counts,
    resolve_calibration_variant,
    resolve_neural_cell,
    resolve_neural_cells,
    resolve_reportable_contrast,
    serialize_cell_definitions,
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
            "adaptive_batching": False,
            "arrival_aware_batching": False,
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
    def test_g1_g2_g3_exact_definitions(self) -> None:
        self.assertEqual(
            NEURAL_LOCAL_CELL_BY_ID["G1"],
            {"id": "G1", "enable_batching": True, "max_batch_size": 4, "max_batch_delay_ms": 1},
        )
        self.assertEqual(
            NEURAL_LOCAL_CELL_BY_ID["G2"],
            {"id": "G2", "enable_batching": True, "max_batch_size": 4, "max_batch_delay_ms": 2},
        )
        self.assertEqual(
            NEURAL_LOCAL_CELL_BY_ID["G3"],
            {
                "id": "G3",
                "enable_batching": True,
                "max_batch_size": 4,
                "max_batch_delay_ms": 2,
                "adaptive_batching": True,
            },
        )
        self.assertEqual(
            NEURAL_LOCAL_CELL_BY_ID["G4"],
            {
                "id": "G4",
                "enable_batching": True,
                "max_batch_size": 4,
                "max_batch_delay_ms": 2,
                "adaptive_batching": False,
                "arrival_aware_batching": True,
            },
        )
        self.assertEqual(CALIBRATION_ONLY_NEURAL_CELLS, frozenset({"G1", "G2", "G3", "G4"}))
        self.assertEqual(len(NEURAL_LOCAL_CELLS), 4)
        self.assertNotIn("G3", CELL_BY_ID)
        self.assertNotIn("G4", CELL_BY_ID)

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
        self.assertIn("not permitted", str(smoke.exception))
        with self.assertRaises(NeuralCollectorError) as reportable:
            resolve_neural_cells(["A", "G2"], "reportable")
        self.assertIn("not permitted", str(reportable.exception))
        allowed = resolve_neural_cells(
            ["A", "G2"], "reportable", reportable_contrast=REPORTABLE_CONTRAST_AG2
        )
        self.assertEqual([cell["id"] for cell in allowed], ["A", "G2"])
        with self.assertRaises(NeuralCollectorError):
            resolve_neural_cells(
                ["A", "G1"], "reportable", reportable_contrast=REPORTABLE_CONTRAST_AG2
            )
        with self.assertRaises(NeuralCollectorError):
            resolve_neural_cells(
                ["A", "G3"], "reportable", reportable_contrast=REPORTABLE_CONTRAST_AG2
            )
        with self.assertRaises(NeuralCollectorError) as bare_g3:
            resolve_neural_cell("G3")
        self.assertIn("calibration-only", str(bare_g3.exception))
        g3 = resolve_neural_cells(["A", "G2", "G3"], "calibration")
        self.assertEqual([cell["id"] for cell in g3], ["A", "G2", "G3"])
        self.assertTrue(g3[2]["adaptive_batching"])
        three = resolve_neural_cells(
            ["A", "G2", "G3"], "reportable", reportable_contrast=REPORTABLE_CONTRAST_AG2G3
        )
        self.assertEqual([cell["id"] for cell in three], ["A", "G2", "G3"])
        with self.assertRaises(NeuralCollectorError):
            resolve_neural_cells(
                ["A", "G2", "G3"], "smoke", reportable_contrast=REPORTABLE_CONTRAST_AG2G3
            )


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

    def test_expected_launch_requires_adaptive_false(self) -> None:
        from benchmarks.collect_neural_scheduler import _expected_launch

        checked = {
            "paths": {"model": pathlib.Path("/models/gemm_f32.onnx")},
            "collection_protocol": {
                "measurement": {
                    "worker_count": 1,
                    "queue_capacity": 64,
                    "warmup_requests": 100,
                    "check_interval": 1,
                    "duration_ms": 1000,
                    "lag_tolerance_ms": 1000,
                }
            },
        }
        expected = _expected_launch(
            checked, resolve_neural_cell("G2"), open_loop_profile("neural_open", 1000.0)
        )
        self.assertTrue(expected["require_adaptive_batching_field"])
        self.assertTrue(expected["require_arrival_aware_batching_field"])
        self.assertFalse(expected["adaptive_batching"])
        self.assertFalse(expected["arrival_aware_batching"])
        g3_expected = _expected_launch(
            checked,
            resolve_neural_cell("G3", allow_g3=True),
            open_loop_profile("neural_open", 1000.0),
        )
        self.assertTrue(g3_expected["require_adaptive_batching_field"])
        self.assertTrue(g3_expected["require_arrival_aware_batching_field"])
        self.assertTrue(g3_expected["adaptive_batching"])
        self.assertFalse(g3_expected["arrival_aware_batching"])
        g4_expected = _expected_launch(
            checked,
            resolve_neural_cell("G4", allow_g4=True),
            open_loop_profile("neural_open", 1000.0),
        )
        self.assertFalse(g4_expected["adaptive_batching"])
        self.assertTrue(g4_expected["arrival_aware_batching"])

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
            self.assertNotIn("--adaptive-batching", command)
            if enable:
                self.assertIn("--enable-batching", command)
            else:
                self.assertNotIn("--enable-batching", command)
        g3 = resolve_neural_cell("G3", allow_g3=True)
        g3_command = neural_scheduler_command(
            binary, model, "gemm_f32", g3, profile, measurement
        )
        self.assertIn("--enable-batching", g3_command)
        self.assertIn("--adaptive-batching", g3_command)
        self.assertNotIn("--arrival-aware-batching", g3_command)
        self.assertEqual(g3_command[g3_command.index("--max-batch-size") + 1], "4")
        self.assertEqual(g3_command[g3_command.index("--max-batch-delay-ms") + 1], "2")
        g4 = resolve_neural_cell("G4", allow_g4=True)
        g4_command = neural_scheduler_command(
            binary, model, "gemm_f32", g4, profile, measurement
        )
        self.assertIn("--enable-batching", g4_command)
        self.assertIn("--arrival-aware-batching", g4_command)
        self.assertNotIn("--adaptive-batching", g4_command)
        self.assertEqual(g4_command[g4_command.index("--max-batch-size") + 1], "4")
        self.assertEqual(g4_command[g4_command.index("--max-batch-delay-ms") + 1], "2")

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
        message = str(raised.exception)
        self.assertTrue(
            "A,C" in message or "A,G2" in message or "A,G2,G3" in message
        )

    def test_expected_launch_detects_wrong_reported_delay(self) -> None:
        document = sample_report(
            model="/models/gemm_f32.onnx",
            scheduler={
                "worker_count": 1,
                "queue_capacity": 64,
                "enable_batching": True,
                "max_batch_size": 4,
                "max_batch_delay_ms": 2,
                "adaptive_batching": False,
                "arrival_aware_batching": False,
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
            "adaptive_batching": False,
            "arrival_aware_batching": False,
            "require_adaptive_batching_field": True,
            "require_arrival_aware_batching_field": True,
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
        self.assertIn("not permitted", str(raised.exception))
        self.assertEqual(self.world.trial_calls, [])

    def test_smoke_rejects_g2(self) -> None:
        args = build_parser().parse_args(self._args("smoke", cells="A,G2"))
        with self.assertRaises(NeuralCollectorError) as raised:
            collect(args, runner=self.world.runner)
        self.assertIn("not permitted", str(raised.exception))
        self.assertEqual(self.world.trial_calls, [])

    def test_reportable_rejects_g1_g2(self) -> None:
        argv = self._args("reportable", cells="A,G1,G2", counted_runs="10", target_rps="90")
        argv.append("--offered-rate-frozen")
        args = build_parser().parse_args(argv)
        with self.assertRaises(NeuralCollectorError) as raised:
            collect(args, runner=self.world.runner)
        message = str(raised.exception)
        self.assertTrue(
            "A,C" in message
            or "A,G2" in message
            or "A,G2,G3" in message
            or "not permitted" in message
        )
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
        for cell in protocol["cell_definitions"]:
            self.assertEqual(set(cell), set(CELL_DEFINITION_KEYS))
            self.assertNotIn("adaptive_batching", cell)
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


class AG2ContrastCollectorTest(unittest.TestCase):
    def test_exact_ag2_contrast_accepted(self) -> None:
        contract = resolve_reportable_contrast(
            model_id=REPORTABLE_MODEL_ID,
            cell_ids=["A", "G2"],
            baseline_cell="A",
            alternative_cell="G2",
            target_rps=AG2_OFFERED_RATE,
            counted_runs=20,
            offered_rate_frozen=True,
        )
        self.assertEqual(contract["contrast_id"], CONTRAST_ID_AG2)
        self.assertEqual(contract["contrast_sign_convention"], CONTRAST_SIGN_G2_MINUS_A)
        self.assertEqual(contract["arrival_screen_max_lag_ns"], ARRIVAL_SCREEN_MAX_LAG_NS)
        self.assertEqual(contract["persistent_batching_min_mean"], 2.0)

    def test_wrong_contrast_cells_and_rate_rejected(self) -> None:
        with self.assertRaises(NeuralCollectorError):
            resolve_reportable_contrast(
                model_id=REPORTABLE_MODEL_ID,
                cell_ids=["A", "G1"],
                baseline_cell="A",
                alternative_cell="G1",
                target_rps=AG2_OFFERED_RATE,
                counted_runs=20,
                offered_rate_frozen=True,
            )
        with self.assertRaises(NeuralCollectorError) as rate:
            resolve_reportable_contrast(
                model_id=REPORTABLE_MODEL_ID,
                cell_ids=["A", "G2"],
                baseline_cell="A",
                alternative_cell="G2",
                target_rps=500.0,
                counted_runs=20,
                offered_rate_frozen=True,
            )
        self.assertIn("1000", str(rate.exception))
        with self.assertRaises(NeuralCollectorError) as blocks:
            resolve_reportable_contrast(
                model_id=REPORTABLE_MODEL_ID,
                cell_ids=["A", "G2"],
                baseline_cell="A",
                alternative_cell="G2",
                target_rps=AG2_OFFERED_RATE,
                counted_runs=10,
                offered_rate_frozen=True,
            )
        self.assertIn("20", str(blocks.exception))

    def test_g2_wrong_delay_or_cap_rejected_in_definitions(self) -> None:
        wrong_delay = dict(REPORTABLE_CONTRAST_AG2["expected_cells"]["G2"])
        wrong_delay["max_batch_delay_ms"] = 1
        self.assertNotEqual(wrong_delay, REPORTABLE_CONTRAST_AG2["expected_cells"]["G2"])
        wrong_cap = dict(REPORTABLE_CONTRAST_AG2["expected_cells"]["G2"])
        wrong_cap["max_batch_size"] = 8
        self.assertEqual(NEURAL_LOCAL_CELL_BY_ID["G2"]["max_batch_delay_ms"], 2)
        self.assertEqual(NEURAL_LOCAL_CELL_BY_ID["G2"]["max_batch_size"], 4)

    def test_arrival_screen_inclusive_bound(self) -> None:
        document = sample_report()
        document["generator_lag"]["maximum_nanoseconds"] = ARRIVAL_SCREEN_MAX_LAG_NS
        self.assertEqual(arrival_screen_problems(document, ARRIVAL_SCREEN_MAX_LAG_NS), [])
        document["generator_lag"]["maximum_nanoseconds"] = ARRIVAL_SCREEN_MAX_LAG_NS + 1
        problems = arrival_screen_problems(document, ARRIVAL_SCREEN_MAX_LAG_NS)
        self.assertTrue(problems)
        missing = sample_report()
        del missing["generator_lag"]
        self.assertTrue(arrival_screen_problems(missing, ARRIVAL_SCREEN_MAX_LAG_NS))

    def test_persistent_batching_mean_gate(self) -> None:
        document = sample_report()
        document["metrics"]["engine_run_count"] = 50
        document["metrics"]["requests_entered_engine_runs"] = 100
        document["metrics"]["batch_size_histogram"] = [0, 0, 50] + [0] * 30
        mean, problems = realized_mean_batch_size(document)
        self.assertEqual(problems, [])
        self.assertEqual(mean, 2.0)
        _document, _status, ok = validate_trial(
            dump_report(document),
            expected={
                "model_path": document["model"],
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
                "target_rps": 50.0,
            },
            mode="reportable",
            require_zero_rejections=True,
            require_persistent_batching=True,
            persistent_batching_min_mean=2.0,
        )
        self.assertEqual(ok, [])
        document["metrics"]["engine_run_count"] = 80
        document["metrics"]["requests_entered_engine_runs"] = 100
        document["metrics"]["batch_size_histogram"] = [0, 60, 20] + [0] * 30
        mean_low, _ = realized_mean_batch_size(document)
        self.assertLess(mean_low, 2.0)
        _document, _status, rejected = validate_trial(
            dump_report(document),
            expected={
                "model_path": document["model"],
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
                "target_rps": 50.0,
            },
            mode="reportable",
            require_zero_rejections=True,
            require_persistent_batching=True,
            persistent_batching_min_mean=2.0,
        )
        self.assertTrue(any("persistent-batching" in item for item in rejected))

    def test_validate_reportable_args_ag2_path(self) -> None:
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
                "A,G2",
                "--target-rps",
                "1000",
                "--counted-runs",
                "20",
                "--offered-rate-frozen",
            ]
        )
        contract = validate_reportable_args(args)
        self.assertEqual(contract["contrast_id"], CONTRAST_ID_AG2)

    def test_ac_contrast_still_valid(self) -> None:
        contract = resolve_reportable_contrast(
            model_id=REPORTABLE_MODEL_ID,
            cell_ids=["A", "C"],
            baseline_cell=None,
            alternative_cell=None,
            target_rps=90.0,
            counted_runs=10,
            offered_rate_frozen=True,
        )
        self.assertEqual(contract["contrast_id"], "gemm_f32_A_vs_C")
        self.assertEqual(contract["contrast_sign_convention"], "C_minus_A")


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


class G3AdaptiveCalibrationTest(unittest.TestCase):
    def test_adaptive_helper_and_normalization(self) -> None:
        self.assertFalse(adaptive_batching_for_cell(resolve_neural_cell("A")))
        self.assertFalse(adaptive_batching_for_cell(resolve_neural_cell("G2")))
        self.assertFalse(adaptive_batching_for_cell(CELL_BY_ID["C"]))
        self.assertTrue(adaptive_batching_for_cell(resolve_neural_cell("G3", allow_g3=True)))
        with self.assertRaises(NeuralCollectorError):
            adaptive_batching_for_cell(
                {
                    "id": "G3",
                    "enable_batching": True,
                    "max_batch_size": 4,
                    "max_batch_delay_ms": 2,
                }
            )
        with self.assertRaises(NeuralCollectorError):
            adaptive_batching_for_cell(
                {
                    "id": "G3",
                    "enable_batching": True,
                    "max_batch_size": 4,
                    "max_batch_delay_ms": 2,
                    "adaptive_batching": False,
                }
            )
        with self.assertRaises(NeuralCollectorError):
            adaptive_batching_for_cell(
                {
                    "id": "G3",
                    "enable_batching": True,
                    "max_batch_size": 4,
                    "max_batch_delay_ms": 2,
                    "adaptive_batching": "true",
                }
            )
        with self.assertRaises(NeuralCollectorError):
            adaptive_batching_for_cell(
                {
                    "id": "G2",
                    "enable_batching": True,
                    "max_batch_size": 4,
                    "max_batch_delay_ms": 2,
                    "adaptive_batching": True,
                }
            )
        normalized_a = normalize_cell_definition(resolve_neural_cell("A"))
        self.assertIs(normalized_a["adaptive_batching"], False)
        self.assertIs(normalized_a["arrival_aware_batching"], False)
        normalized_g3 = normalize_cell_definition(resolve_neural_cell("G3", allow_g3=True))
        self.assertIs(normalized_g3["adaptive_batching"], True)
        self.assertIs(normalized_g3["arrival_aware_batching"], False)

    def test_dual_serialization_four_vs_five_key(self) -> None:
        cells = resolve_neural_cells(["A", "G2", "G3"], "calibration")
        four = serialize_cell_definitions(cells, include_adaptive=False)
        five = serialize_cell_definitions(cells, include_adaptive=True)
        for cell in four:
            self.assertEqual(set(cell), set(CELL_DEFINITION_KEYS))
            self.assertNotIn(CELL_DEFINITION_ARRIVAL_KEY, cell)
        self.assertEqual(
            five[0],
            {
                "id": "A",
                "enable_batching": False,
                "max_batch_size": 1,
                "max_batch_delay_ms": 0,
                "adaptive_batching": False,
            },
        )
        self.assertEqual(
            five[1],
            {
                "id": "G2",
                "enable_batching": True,
                "max_batch_size": 4,
                "max_batch_delay_ms": 2,
                "adaptive_batching": False,
            },
        )
        self.assertEqual(
            five[2],
            {
                "id": "G3",
                "enable_batching": True,
                "max_batch_size": 4,
                "max_batch_delay_ms": 2,
                "adaptive_batching": True,
            },
        )
        for cell in five:
            self.assertNotIn(CELL_DEFINITION_ARRIVAL_KEY, cell)
        reportable = serialize_cell_definitions(
            [resolve_neural_cell("A"), resolve_neural_cell("C")], include_adaptive=False
        )
        for cell in reportable:
            self.assertEqual(set(cell), set(CELL_DEFINITION_KEYS))
        self.assertEqual(REPORTABLE_CONTRAST_AC["expected_cells"]["A"].keys(), set(CELL_DEFINITION_KEYS))
        self.assertEqual(REPORTABLE_CONTRAST_AG2["expected_cells"]["G2"].keys(), set(CELL_DEFINITION_KEYS))

    def test_variant_resolution_frozen_plan(self) -> None:
        cells = resolve_neural_cells(list(ADAPTIVE_EARLY_SEAL_CELLS), "calibration")
        profile = open_loop_profile("neural_open", ADAPTIVE_EARLY_SEAL_RATE)
        measurement = _measurement()
        variant = resolve_calibration_variant(
            mode="calibration",
            cells=cells,
            model_id=REPORTABLE_MODEL_ID,
            profile=profile,
            measurement=measurement,
            counted_runs=1,
        )
        self.assertEqual(variant, CALIBRATION_VARIANT_ADAPTIVE_EARLY_SEAL_V1)
        with self.assertRaises(NeuralCollectorError):
            resolve_calibration_variant(
                mode="calibration",
                cells=cells,
                model_id=REPORTABLE_MODEL_ID,
                profile=open_loop_profile("neural_open", 500.0),
                measurement=measurement,
                counted_runs=1,
            )
        with self.assertRaises(NeuralCollectorError):
            resolve_calibration_variant(
                mode="calibration",
                cells=resolve_neural_cells(["A", "G3"], "calibration"),
                model_id=REPORTABLE_MODEL_ID,
                profile=profile,
                measurement=measurement,
                counted_runs=1,
            )

    def test_six_trial_plan_a_g2_g3(self) -> None:
        cells = resolve_neural_cells(["A", "G2", "G3"], "calibration")
        plan = build_plan(cells, [open_loop_profile("neural_open", 1000.0)], 1)
        self.assertEqual(len(plan), 6)
        self.assertEqual(
            [trial["trial_id"] for trial in plan],
            [
                "pilot-A-neural_open",
                "pilot-G2-neural_open",
                "pilot-G3-neural_open",
                "A-neural_open-01",
                "G2-neural_open-01",
                "G3-neural_open-01",
            ],
        )
        self.assertEqual([trial["pilot"] for trial in plan], [True, True, True, False, False, False])

    def test_g3_runner_identity_validation(self) -> None:
        from benchmarks.collect_neural_scheduler import _expected_launch

        checked = {
            "paths": {"model": pathlib.Path("/models/gemm_f32.onnx")},
            "collection_protocol": {
                "measurement": {
                    "worker_count": 1,
                    "queue_capacity": 64,
                    "warmup_requests": 100,
                    "check_interval": 1,
                    "duration_ms": 1000,
                    "lag_tolerance_ms": 1000,
                }
            },
        }
        cell = resolve_neural_cell("G3", allow_g3=True)
        expected = _expected_launch(checked, cell, open_loop_profile("neural_open", 1000.0))
        document = sample_report(model="/models/gemm_f32.onnx")
        document["scheduler"] = {
            "worker_count": 1,
            "queue_capacity": 64,
            "enable_batching": True,
            "max_batch_size": 4,
            "max_batch_delay_ms": 2,
            "adaptive_batching": True,
            "arrival_aware_batching": False,
            "enable_metrics": True,
        }
        document["generator"]["warmup_requests"] = 100
        _doc, status, problems = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertEqual(problems, [])
        self.assertEqual(status["delivery_class"], "STRICT_PRIMARY")

        document["scheduler"]["adaptive_batching"] = False
        _doc, _status, rejected_false = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertTrue(any("adaptive_batching" in item for item in rejected_false))

        del document["scheduler"]["adaptive_batching"]
        _doc, _status, rejected_missing = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertTrue(any("adaptive_batching" in item for item in rejected_missing))

        document["scheduler"]["adaptive_batching"] = True
        document["scheduler"]["arrival_aware_batching"] = False
        document["scheduler"]["max_batch_delay_ms"] = 1
        _doc, _status, rejected_delay = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertTrue(any("max_batch_delay_ms" in item for item in rejected_delay))

        document["scheduler"]["max_batch_delay_ms"] = 2
        document["scheduler"]["max_batch_size"] = 8
        _doc, _status, rejected_cap = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertTrue(any("max_batch_size" in item for item in rejected_cap))

        document["scheduler"]["max_batch_size"] = 4
        document["scheduler"]["unexpected_key"] = True
        _doc, _status, rejected_extra = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertTrue(any("unexpected_key" in item for item in rejected_extra))

    def test_mean_batch_below_two_is_diagnostic_only(self) -> None:
        document = sample_report(model="/models/gemm_f32.onnx")
        document["metrics"]["batch_size_histogram"] = [0, 10] + [0] * 31
        document["metrics"]["engine_run_count"] = 10
        document["metrics"]["requests_entered_engine_runs"] = 10
        mean, problems = realized_mean_batch_size(document)
        self.assertEqual(problems, [])
        self.assertEqual(mean, 1.0)
        fraction, fraction_problems = multi_request_batch_fraction(document)
        self.assertEqual(fraction_problems, [])
        self.assertEqual(fraction, 0.0)
        # Plumbing validation does not gate on mean >= 2.
        from benchmarks.collect_neural_scheduler import _expected_launch

        checked = {
            "paths": {"model": pathlib.Path("/models/gemm_f32.onnx")},
            "collection_protocol": {
                "measurement": {
                    "worker_count": 1,
                    "queue_capacity": 64,
                    "warmup_requests": 100,
                    "check_interval": 1,
                    "duration_ms": 1000,
                    "lag_tolerance_ms": 1000,
                }
            },
        }
        cell = resolve_neural_cell("G3", allow_g3=True)
        expected = _expected_launch(checked, cell, open_loop_profile("neural_open", 1000.0))
        document["scheduler"] = {
            "worker_count": 1,
            "queue_capacity": 64,
            "enable_batching": True,
            "max_batch_size": 4,
            "max_batch_delay_ms": 2,
            "adaptive_batching": True,
            "arrival_aware_batching": False,
            "enable_metrics": True,
        }
        document["generator"]["warmup_requests"] = 100
        _doc, status, problems = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertEqual(problems, [])
        self.assertEqual(status["delivery_class"], "STRICT_PRIMARY")
        self.assertLess(mean, 2.0)

    def test_no_g3_confirmatory_contrast(self) -> None:
        self.assertNotIn("G3", REPORTABLE_CONTRAST_AG2["allow_neural_local_cells"])
        self.assertNotIn("G3", REPORTABLE_CONTRAST_AC["allow_neural_local_cells"])
        with self.assertRaises(NeuralCollectorError):
            resolve_reportable_contrast(
                model_id=REPORTABLE_MODEL_ID,
                cell_ids=["A", "G3"],
                baseline_cell="A",
                alternative_cell="G3",
                target_rps=1000.0,
                counted_runs=20,
                offered_rate_frozen=True,
            )


class G3AdaptiveCollectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.world = NeuralWorld(pathlib.Path(self.temporary.name))
        self.world.build()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self, mode: str, **extra: str) -> list[str]:
        values = {
            "binary": str(self.world.binary),
            "repo-root": str(self.world.repo),
            "expected-commit": COMMIT,
            "mode": mode,
            "model-id": "gemm_f32",
            "cells": "A,G2,G3",
            "target-rps": "1000",
            "counted-runs": "1",
            "output-root": str(pathlib.Path(self.temporary.name) / "out"),
        }
        values.update({key.replace("_", "-"): value for key, value in extra.items()})
        argv: list[str] = []
        for key, value in values.items():
            argv.extend([f"--{key}", value])
        return argv

    def test_g3_calibration_accepted_with_five_key_metadata(self) -> None:
        args = build_parser().parse_args(self._args("calibration"))
        run_set = collect(args, runner=self.world.runner)
        self.assertEqual(len(self.world.trial_calls), 6)
        environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
        protocol = environment["collection_protocol"]
        self.assertEqual(protocol["calibration_variant"], CALIBRATION_VARIANT_ADAPTIVE_EARLY_SEAL_V1)
        self.assertTrue(protocol["zero_admission_rejection_required"])
        self.assertTrue(protocol["require_strict_generator_delivery"])
        definitions = {cell["id"]: cell for cell in protocol["cell_definitions"]}
        self.assertIs(definitions["A"]["adaptive_batching"], False)
        self.assertIs(definitions["G2"]["adaptive_batching"], False)
        self.assertIs(definitions["G3"]["adaptive_batching"], True)
        for cell in protocol["cell_definitions"]:
            self.assertIn("adaptive_batching", cell)
            self.assertNotIn(CELL_DEFINITION_ARRIVAL_KEY, cell)
            self.assertEqual(len(cell), 5)
        plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))["trials"]
        by_id = {trial["trial_id"]: trial for trial in plan}
        self.assertNotIn("--adaptive-batching", by_id["A-neural_open-01"]["command"])
        self.assertNotIn("--adaptive-batching", by_id["G2-neural_open-01"]["command"])
        self.assertIn("--adaptive-batching", by_id["G3-neural_open-01"]["command"])
        self.assertNotIn("--arrival-aware-batching", by_id["G3-neural_open-01"]["command"])
        self.assertIn("--enable-batching", by_id["G3-neural_open-01"]["command"])
        g3_launch = self.world.trial_calls[5]
        self.assertIn("--adaptive-batching", g3_launch)
        meta = json.loads((run_set / "runs" / "G3-neural_open-01.meta.json").read_text(encoding="utf-8"))
        self.assertTrue(meta["collector_valid"])

    def test_g3_smoke_and_reportable_rejected(self) -> None:
        smoke_argv = [
            "--binary",
            str(self.world.binary),
            "--repo-root",
            str(self.world.repo),
            "--expected-commit",
            COMMIT,
            "--mode",
            "smoke",
            "--model-id",
            "gemm_f32",
            "--cells",
            "A,G3",
            "--output-root",
            str(pathlib.Path(self.temporary.name) / "smoke-g3"),
        ]
        with self.assertRaises(NeuralCollectorError) as smoke:
            collect(build_parser().parse_args(smoke_argv), runner=self.world.runner)
        self.assertIn("not permitted", str(smoke.exception))
        argv = self._args("reportable", cells="A,G3", counted_runs="20", target_rps="1000")
        argv.append("--offered-rate-frozen")
        with self.assertRaises(NeuralCollectorError):
            collect(build_parser().parse_args(argv), runner=self.world.runner)
        self.assertEqual(self.world.trial_calls, [])

    def test_hard_invalid_delivery_stops_without_soft_promote(self) -> None:
        original = self.world.runner

        def weak_delivery(args, cwd=None, env=None, timeout_seconds=None):
            result = original(args, cwd=cwd, env=env, timeout_seconds=timeout_seconds)
            if pathlib.Path(args[0]).name != BINARY_NAME or "--format" not in args:
                return result
            document = json.loads(result.stdout)
            document["submit_attempts"] = document["scheduled_opportunities"] - 1
            document["generator_skips_outstanding"] = 1
            document["accepted"] = document["submit_attempts"]
            document["succeeded"] = document["accepted"]
            document["successful_e2e_latency"]["sample_count"] = document["succeeded"]
            document["metrics"]["engine_run_count"] = document["succeeded"]
            document["metrics"]["requests_entered_engine_runs"] = document["succeeded"]
            document["metrics"]["batch_size_histogram"] = [0, document["succeeded"]] + [0] * 31
            document["correctness_checked"] = document["succeeded"]
            document["generator_lag"]["count"] = document["submit_attempts"]
            from benchmarks.collect_baseline import ProcessResult

            return ProcessResult(0, dump_report(document), "")

        with self.assertRaises(NeuralCollectorError) as raised:
            collect(
                build_parser().parse_args(self._args("calibration")),
                runner=weak_delivery,
            )
        run_set = raised.exception.run_set
        assert run_set is not None
        failure = json.loads((run_set / "failure.json").read_text(encoding="utf-8"))
        self.assertFalse(failure["accepted"])
        meta = json.loads(
            (run_set / "runs" / f"{failure['trial_id']}.meta.json").read_text(encoding="utf-8")
        )
        self.assertFalse(meta["collector_valid"])
        self.assertNotIn("calibration_infeasible_markers", meta)

    def test_reportable_ac_four_key_metadata_unchanged(self) -> None:
        argv = [
            "--binary",
            str(self.world.binary),
            "--repo-root",
            str(self.world.repo),
            "--expected-commit",
            COMMIT,
            "--mode",
            "smoke",
            "--model-id",
            "gemm_f32",
            "--cells",
            "A,C",
            "--output-root",
            str(pathlib.Path(self.temporary.name) / "smoke-out"),
        ]
        run_set = collect(build_parser().parse_args(argv), runner=self.world.runner)
        protocol = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))[
            "collection_protocol"
        ]
        for cell in protocol["cell_definitions"]:
            self.assertEqual(set(cell), set(CELL_DEFINITION_KEYS))
            self.assertNotIn("adaptive_batching", cell)


class AG2G3ContrastCollectorTest(unittest.TestCase):
    def test_exact_ag2g3_contrast_accepted(self) -> None:
        contract = resolve_reportable_contrast(
            model_id=REPORTABLE_MODEL_ID,
            cell_ids=["A", "G2", "G3"],
            baseline_cell="G2",
            alternative_cell="G3",
            target_rps=AG2G3_OFFERED_RATE,
            counted_runs=20,
            offered_rate_frozen=True,
        )
        self.assertEqual(contract["contrast_id"], CONTRAST_ID_AG2G3)
        self.assertEqual(contract["contrast_sign_convention"], CONTRAST_SIGN_G3_MINUS_G2)
        self.assertEqual(
            contract["secondary_contrast_sign_convention"], CONTRAST_SIGN_G3_MINUS_A
        )
        self.assertTrue(contract["require_five_key_cell_definitions"])
        self.assertEqual(contract["arrival_screen_max_lag_ns"], ARRIVAL_SCREEN_MAX_LAG_NS)
        self.assertEqual(contract["expected_position_counts"], THREE_ARM_POSITION_COUNTS_20)

    def test_wrong_combination_rate_and_blocks_rejected(self) -> None:
        with self.assertRaises(NeuralCollectorError):
            resolve_reportable_contrast(
                model_id=REPORTABLE_MODEL_ID,
                cell_ids=["A", "G3"],
                baseline_cell="A",
                alternative_cell="G3",
                target_rps=AG2G3_OFFERED_RATE,
                counted_runs=20,
                offered_rate_frozen=True,
            )
        with self.assertRaises(NeuralCollectorError) as rate:
            resolve_reportable_contrast(
                model_id=REPORTABLE_MODEL_ID,
                cell_ids=["A", "G2", "G3"],
                baseline_cell="G2",
                alternative_cell="G3",
                target_rps=500.0,
                counted_runs=20,
                offered_rate_frozen=True,
            )
        self.assertIn("1000", str(rate.exception))
        with self.assertRaises(NeuralCollectorError) as blocks:
            resolve_reportable_contrast(
                model_id=REPORTABLE_MODEL_ID,
                cell_ids=["A", "G2", "G3"],
                baseline_cell="G2",
                alternative_cell="G3",
                target_rps=AG2G3_OFFERED_RATE,
                counted_runs=10,
                offered_rate_frozen=True,
            )
        self.assertIn("20", str(blocks.exception))
        with self.assertRaises(NeuralCollectorError):
            resolve_reportable_contrast(
                model_id=REPORTABLE_MODEL_ID,
                cell_ids=["A", "G2", "G3"],
                baseline_cell="A",
                alternative_cell="G3",
                target_rps=AG2G3_OFFERED_RATE,
                counted_runs=20,
                offered_rate_frozen=True,
            )

    def test_five_key_new_and_four_key_old(self) -> None:
        five = serialize_cell_definitions(
            [
                resolve_neural_cell("A"),
                resolve_neural_cell("G2"),
                resolve_neural_cell("G3", allow_g3=True),
            ],
            include_adaptive=True,
        )
        self.assertEqual(len(five), 3)
        for cell in five:
            self.assertIn("adaptive_batching", cell)
        self.assertIs(five[2]["adaptive_batching"], True)
        four = serialize_cell_definitions(
            [resolve_neural_cell("A"), resolve_neural_cell("G2")],
            include_adaptive=False,
        )
        for cell in four:
            self.assertNotIn("adaptive_batching", cell)
        ac = serialize_cell_definitions(
            [dict(REPORTABLE_CONTRAST_AC["expected_cells"]["A"]), dict(REPORTABLE_CONTRAST_AC["expected_cells"]["C"])],
            include_adaptive=False,
        )
        for cell in ac:
            self.assertEqual(set(cell), set(CELL_DEFINITION_KEYS))

    def test_command_flags_and_63_trial_rotation(self) -> None:
        binary = pathlib.Path("/tmp") / BINARY_NAME
        model = pathlib.Path("/models/gemm_f32.onnx")
        cells = {
            "A": resolve_neural_cell("A"),
            "G2": resolve_neural_cell("G2"),
            "G3": resolve_neural_cell("G3", allow_g3=True),
        }
        profile = open_loop_profile("neural_open", AG2G3_OFFERED_RATE)
        measurement = {
            "worker_count": 1,
            "queue_capacity": 64,
            "duration_ms": 1000,
            "max_offered": 10000,
            "max_outstanding": 64,
            "completion_timeout_ms": 5000,
            "warmup_requests": 100,
            "lag_tolerance_ms": 1000,
            "check_interval": 1,
        }
        plan = bind_commands(
            build_plan([cells["A"], cells["G2"], cells["G3"]], [profile], 20),
            binary=binary,
            model=model,
            model_id="gemm_f32",
            cells=cells,
            profiles={profile["id"]: profile},
            measurement=measurement,
        )
        self.assertEqual(len(plan), 63)
        pilots = [trial for trial in plan if trial["pilot"]]
        counted = [trial for trial in plan if not trial["pilot"]]
        self.assertEqual([trial["cell_id"] for trial in pilots], ["A", "G2", "G3"])
        self.assertEqual(len(counted), 60)
        self.assertEqual(
            [trial["cell_id"] for trial in counted if trial["block_index"] == 1],
            ["A", "G2", "G3"],
        )
        self.assertEqual(
            [trial["cell_id"] for trial in counted if trial["block_index"] == 2],
            ["G2", "G3", "A"],
        )
        self.assertEqual(
            [trial["cell_id"] for trial in counted if trial["block_index"] == 3],
            ["G3", "A", "G2"],
        )
        counts = require_rotation_position_counts(plan, THREE_ARM_POSITION_COUNTS_20)
        self.assertEqual(counts, THREE_ARM_POSITION_COUNTS_20)
        by_id = {trial["trial_id"]: trial for trial in plan}
        self.assertNotIn("--adaptive-batching", by_id["A-neural_open-01"]["command"])
        self.assertNotIn("--enable-batching", by_id["A-neural_open-01"]["command"])
        self.assertIn("--enable-batching", by_id["G2-neural_open-01"]["command"])
        self.assertNotIn("--adaptive-batching", by_id["G2-neural_open-01"]["command"])
        self.assertIn("--enable-batching", by_id["G3-neural_open-01"]["command"])
        self.assertIn("--adaptive-batching", by_id["G3-neural_open-01"]["command"])

    def test_g3_smoke_and_unrelated_reportable_rejected(self) -> None:
        with self.assertRaises(NeuralCollectorError):
            resolve_neural_cells(["A", "G2", "G3"], "smoke")
        with self.assertRaises(NeuralCollectorError):
            resolve_neural_cells(
                ["A", "G2", "G3"], "reportable", reportable_contrast=REPORTABLE_CONTRAST_AG2
            )
        with self.assertRaises(NeuralCollectorError):
            resolve_neural_cells(
                ["A", "G1", "G2"], "reportable", reportable_contrast=REPORTABLE_CONTRAST_AG2G3
            )

    def test_no_shared_add_registry_changes(self) -> None:
        self.assertNotIn("G3", CELL_BY_ID)
        self.assertNotIn("G2", CELL_BY_ID)
        self.assertNotIn("G4", CELL_BY_ID)
        self.assertEqual(set(REPORTABLE_CONTRAST_AG2G3["allow_neural_local_cells"]), {"G2", "G3"})
        self.assertNotIn("G4", REPORTABLE_CONTRAST_AG2G3["allow_neural_local_cells"])
        self.assertNotIn("G4", REPORTABLE_CONTRAST_AG2["allow_neural_local_cells"])
        self.assertNotIn("G4", REPORTABLE_CONTRAST_AC["allow_neural_local_cells"])

    def test_validate_reportable_args_ag2g3_path(self) -> None:
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
                "A,G2,G3",
                "--baseline-cell",
                "G2",
                "--alternative-cell",
                "G3",
                "--target-rps",
                "1000",
                "--counted-runs",
                "20",
                "--offered-rate-frozen",
            ]
        )
        contract = validate_reportable_args(args)
        self.assertEqual(contract["contrast_id"], CONTRAST_ID_AG2G3)

    def test_wrong_rotation_rejected(self) -> None:
        cells = [
            resolve_neural_cell("A"),
            resolve_neural_cell("G2"),
            resolve_neural_cell("G3", allow_g3=True),
        ]
        plan = build_plan(cells, [open_loop_profile("neural_open", 1000.0)], 20)
        plan[3]["cell_id"], plan[4]["cell_id"] = plan[4]["cell_id"], plan[3]["cell_id"]
        with self.assertRaises(NeuralCollectorError):
            require_rotation_position_counts(plan, THREE_ARM_POSITION_COUNTS_20)


def _measurement_g4() -> dict:
    return {
        "worker_count": 1,
        "queue_capacity": 64,
        "duration_ms": 1000,
        "max_offered": 10000,
        "max_outstanding": 64,
        "completion_timeout_ms": 5000,
        "warmup_requests": 100,
        "lag_tolerance_ms": 1000,
        "check_interval": 1,
    }


class G4ArrivalAwareCalibrationTest(unittest.TestCase):
    def test_g4_exact_cell_and_helpers(self) -> None:
        cell = resolve_neural_cell("G4", allow_g4=True)
        self.assertEqual(
            cell,
            {
                "id": "G4",
                "enable_batching": True,
                "max_batch_size": 4,
                "max_batch_delay_ms": 2,
                "adaptive_batching": False,
                "arrival_aware_batching": True,
            },
        )
        self.assertFalse(adaptive_batching_for_cell(cell))
        self.assertTrue(arrival_aware_batching_for_cell(cell))
        normalized = normalize_cell_definition(cell)
        self.assertIs(normalized[CELL_DEFINITION_ADAPTIVE_KEY], False)
        self.assertIs(normalized[CELL_DEFINITION_ARRIVAL_KEY], True)
        with self.assertRaises(NeuralCollectorError) as bare:
            resolve_neural_cell("G4")
        self.assertIn("calibration-only", str(bare.exception))

    def test_g4_mode_gates(self) -> None:
        cells = resolve_neural_cells(["A", "G2", "G3", "G4"], "calibration")
        self.assertEqual([cell["id"] for cell in cells], ["A", "G2", "G3", "G4"])
        with self.assertRaises(NeuralCollectorError) as smoke:
            resolve_neural_cells(["A", "G4"], "smoke")
        self.assertIn("not permitted", str(smoke.exception))
        with self.assertRaises(NeuralCollectorError):
            resolve_neural_cells(
                ["A", "G2", "G3", "G4"],
                "reportable",
                reportable_contrast=REPORTABLE_CONTRAST_AG2G3,
            )
        with self.assertRaises(NeuralCollectorError):
            resolve_reportable_contrast(
                model_id=REPORTABLE_MODEL_ID,
                cell_ids=["A", "G4"],
                baseline_cell="A",
                alternative_cell="G4",
                target_rps=1000.0,
                counted_runs=20,
                offered_rate_frozen=True,
            )

    def test_variant_resolution_and_precedence(self) -> None:
        cells = resolve_neural_cells(list(ARRIVAL_AWARE_BATCHING_CELLS), "calibration")
        profile = open_loop_profile("neural_open", ARRIVAL_AWARE_BATCHING_RATE)
        measurement = _measurement_g4()
        variant = resolve_calibration_variant(
            mode="calibration",
            cells=cells,
            model_id=REPORTABLE_MODEL_ID,
            profile=profile,
            measurement=measurement,
            counted_runs=1,
        )
        self.assertEqual(variant, CALIBRATION_VARIANT_ARRIVAL_AWARE_BATCHING_V1)
        g3_only = resolve_neural_cells(list(ADAPTIVE_EARLY_SEAL_CELLS), "calibration")
        self.assertEqual(
            resolve_calibration_variant(
                mode="calibration",
                cells=g3_only,
                model_id=REPORTABLE_MODEL_ID,
                profile=open_loop_profile("neural_open", ADAPTIVE_EARLY_SEAL_RATE),
                measurement=measurement,
                counted_runs=1,
            ),
            CALIBRATION_VARIANT_ADAPTIVE_EARLY_SEAL_V1,
        )
        with self.assertRaises(NeuralCollectorError) as wrong_set:
            resolve_calibration_variant(
                mode="calibration",
                cells=resolve_neural_cells(["A", "G4"], "calibration"),
                model_id=REPORTABLE_MODEL_ID,
                profile=profile,
                measurement=measurement,
                counted_runs=1,
            )
        self.assertIn("A,G2,G3,G4", str(wrong_set.exception))
        with self.assertRaises(NeuralCollectorError):
            resolve_calibration_variant(
                mode="calibration",
                cells=cells,
                model_id=REPORTABLE_MODEL_ID,
                profile=open_loop_profile("neural_open", 500.0),
                measurement=measurement,
                counted_runs=1,
            )
        with self.assertRaises(NeuralCollectorError):
            resolve_calibration_variant(
                mode="calibration",
                cells=cells,
                model_id=REPORTABLE_MODEL_ID,
                profile=profile,
                measurement=measurement,
                counted_runs=2,
            )

    def test_six_key_and_historical_serialization(self) -> None:
        cells = resolve_neural_cells(list(ARRIVAL_AWARE_BATCHING_CELLS), "calibration")
        six = serialize_cell_definitions(
            cells, include_adaptive=True, include_arrival_aware=True
        )
        self.assertEqual(len(six), 4)
        expected = {
            "A": (False, False),
            "G2": (False, False),
            "G3": (True, False),
            "G4": (False, True),
        }
        for cell in six:
            self.assertEqual(
                set(cell),
                set(CELL_DEFINITION_KEYS)
                | {CELL_DEFINITION_ADAPTIVE_KEY, CELL_DEFINITION_ARRIVAL_KEY},
            )
            adaptive, arrival = expected[cell["id"]]
            self.assertIs(cell["adaptive_batching"], adaptive)
            self.assertIs(cell["arrival_aware_batching"], arrival)
        five = serialize_cell_definitions(
            resolve_neural_cells(["A", "G2", "G3"], "calibration"),
            include_adaptive=True,
        )
        for cell in five:
            self.assertEqual(len(cell), 5)
            self.assertNotIn(CELL_DEFINITION_ARRIVAL_KEY, cell)
        four_ac = serialize_cell_definitions(
            [resolve_neural_cell("A"), resolve_neural_cell("C")],
            include_adaptive=False,
        )
        for cell in four_ac:
            self.assertEqual(set(cell), set(CELL_DEFINITION_KEYS))
        four_ag2 = serialize_cell_definitions(
            [resolve_neural_cell("A"), resolve_neural_cell("G2")],
            include_adaptive=False,
        )
        for cell in four_ag2:
            self.assertEqual(set(cell), set(CELL_DEFINITION_KEYS))

    def test_policy_rejection_matrix(self) -> None:
        base_g4 = {
            "id": "G4",
            "enable_batching": True,
            "max_batch_size": 4,
            "max_batch_delay_ms": 2,
            "adaptive_batching": False,
            "arrival_aware_batching": True,
        }
        missing = dict(base_g4)
        del missing["arrival_aware_batching"]
        with self.assertRaises(NeuralCollectorError):
            normalize_cell_definition(missing)
        false_arrival = dict(base_g4)
        false_arrival["arrival_aware_batching"] = False
        with self.assertRaises(NeuralCollectorError):
            normalize_cell_definition(false_arrival)
        adaptive_true = dict(base_g4)
        adaptive_true["adaptive_batching"] = True
        with self.assertRaises(NeuralCollectorError):
            normalize_cell_definition(adaptive_true)
        g3_arrival = {
            "id": "G3",
            "enable_batching": True,
            "max_batch_size": 4,
            "max_batch_delay_ms": 2,
            "adaptive_batching": True,
            "arrival_aware_batching": True,
        }
        with self.assertRaises(NeuralCollectorError):
            normalize_cell_definition(g3_arrival)
        both_true = {
            "id": "G2",
            "enable_batching": True,
            "max_batch_size": 4,
            "max_batch_delay_ms": 2,
            "adaptive_batching": True,
            "arrival_aware_batching": True,
        }
        with self.assertRaises(NeuralCollectorError):
            normalize_cell_definition(both_true)
        bad_delay = dict(base_g4)
        bad_delay["max_batch_delay_ms"] = 1
        with self.assertRaises(NeuralCollectorError):
            normalize_cell_definition(bad_delay)
        bad_cap = dict(base_g4)
        bad_cap["max_batch_size"] = 8
        with self.assertRaises(NeuralCollectorError):
            normalize_cell_definition(bad_cap)
        unknown = dict(base_g4)
        unknown["unexpected"] = True
        with self.assertRaises(NeuralCollectorError):
            normalize_cell_definition(unknown)

    def test_eight_trial_plan(self) -> None:
        cells = resolve_neural_cells(list(ARRIVAL_AWARE_BATCHING_CELLS), "calibration")
        plan = build_plan(cells, [open_loop_profile("neural_open", 1000.0)], 1)
        self.assertEqual(len(plan), 8)
        self.assertEqual(
            [trial["trial_id"] for trial in plan],
            [
                "pilot-A-neural_open",
                "pilot-G2-neural_open",
                "pilot-G3-neural_open",
                "pilot-G4-neural_open",
                "A-neural_open-01",
                "G2-neural_open-01",
                "G3-neural_open-01",
                "G4-neural_open-01",
            ],
        )
        self.assertEqual(
            [trial["pilot"] for trial in plan],
            [True, True, True, True, False, False, False, False],
        )

    def test_g4_runner_identity_validation(self) -> None:
        from benchmarks.collect_neural_scheduler import _expected_launch

        checked = {
            "paths": {"model": pathlib.Path("/models/gemm_f32.onnx")},
            "collection_protocol": {
                "measurement": {
                    "worker_count": 1,
                    "queue_capacity": 64,
                    "warmup_requests": 100,
                    "check_interval": 1,
                    "duration_ms": 1000,
                    "lag_tolerance_ms": 1000,
                }
            },
        }
        cell = resolve_neural_cell("G4", allow_g4=True)
        expected = _expected_launch(checked, cell, open_loop_profile("neural_open", 1000.0))
        document = sample_report(model="/models/gemm_f32.onnx")
        document["scheduler"] = {
            "worker_count": 1,
            "queue_capacity": 64,
            "enable_batching": True,
            "max_batch_size": 4,
            "max_batch_delay_ms": 2,
            "adaptive_batching": False,
            "arrival_aware_batching": True,
            "enable_metrics": True,
        }
        document["generator"]["warmup_requests"] = 100
        _doc, status, problems = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertEqual(problems, [])
        self.assertEqual(status["delivery_class"], "STRICT_PRIMARY")

        document["scheduler"]["arrival_aware_batching"] = False
        _doc, _status, rejected_false = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertTrue(any("arrival_aware_batching" in item for item in rejected_false))

        del document["scheduler"]["arrival_aware_batching"]
        document["scheduler"]["arrival_aware_batching"] = True
        document["scheduler"]["adaptive_batching"] = True
        _doc, _status, rejected_both = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertTrue(any("adaptive_batching" in item for item in rejected_both))

        document["scheduler"]["adaptive_batching"] = False
        del document["scheduler"]["arrival_aware_batching"]
        _doc, _status, rejected_missing = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertTrue(any("arrival_aware_batching" in item for item in rejected_missing))

        document["scheduler"]["arrival_aware_batching"] = True
        document["scheduler"]["max_batch_delay_ms"] = 1
        _doc, _status, rejected_delay = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertTrue(any("max_batch_delay_ms" in item for item in rejected_delay))

        document["scheduler"]["max_batch_delay_ms"] = 2
        document["scheduler"]["max_batch_size"] = 8
        _doc, _status, rejected_cap = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertTrue(any("max_batch_size" in item for item in rejected_cap))

        document["scheduler"]["max_batch_size"] = 4
        document["scheduler"]["unexpected_key"] = True
        _doc, _status, rejected_extra = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertTrue(any("unexpected_key" in item for item in rejected_extra))

    def test_mean_batch_below_two_diagnostic_only(self) -> None:
        document = sample_report(model="/models/gemm_f32.onnx")
        document["metrics"]["batch_size_histogram"] = [0, 10] + [0] * 31
        document["metrics"]["engine_run_count"] = 10
        document["metrics"]["requests_entered_engine_runs"] = 10
        mean, problems = realized_mean_batch_size(document)
        self.assertEqual(problems, [])
        self.assertEqual(mean, 1.0)
        request_fraction, request_problems = multi_request_batch_fraction(document)
        self.assertEqual(request_problems, [])
        self.assertEqual(request_fraction, 0.0)
        batch_fraction, batch_problems = multi_size_batch_fraction(document)
        self.assertEqual(batch_problems, [])
        self.assertEqual(batch_fraction, 0.0)
        from benchmarks.collect_neural_scheduler import _expected_launch

        checked = {
            "paths": {"model": pathlib.Path("/models/gemm_f32.onnx")},
            "collection_protocol": {
                "measurement": {
                    "worker_count": 1,
                    "queue_capacity": 64,
                    "warmup_requests": 100,
                    "check_interval": 1,
                    "duration_ms": 1000,
                    "lag_tolerance_ms": 1000,
                }
            },
        }
        cell = resolve_neural_cell("G4", allow_g4=True)
        expected = _expected_launch(checked, cell, open_loop_profile("neural_open", 1000.0))
        document["scheduler"] = {
            "worker_count": 1,
            "queue_capacity": 64,
            "enable_batching": True,
            "max_batch_size": 4,
            "max_batch_delay_ms": 2,
            "adaptive_batching": False,
            "arrival_aware_batching": True,
            "enable_metrics": True,
        }
        document["generator"]["warmup_requests"] = 100
        _doc, status, problems = validate_trial(
            dump_report(document),
            expected=expected,
            mode="calibration",
            require_zero_rejections=True,
            require_strict_generator_delivery=True,
        )
        self.assertEqual(problems, [])
        self.assertEqual(status["delivery_class"], "STRICT_PRIMARY")
        self.assertLess(mean, 2.0)

    def test_no_g4_confirmatory_contrast(self) -> None:
        for contrast in (
            REPORTABLE_CONTRAST_AC,
            REPORTABLE_CONTRAST_AG2,
            REPORTABLE_CONTRAST_AG2G3,
        ):
            self.assertNotIn("G4", contrast["allow_neural_local_cells"])
            self.assertNotIn("G4", contrast["cells"])
        self.assertNotIn("G4", CELL_BY_ID)


class G4ArrivalAwareCollectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.world = NeuralWorld(pathlib.Path(self.temporary.name))
        self.world.build()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self, mode: str, **extra: str) -> list[str]:
        values = {
            "binary": str(self.world.binary),
            "repo-root": str(self.world.repo),
            "expected-commit": COMMIT,
            "mode": mode,
            "model-id": "gemm_f32",
            "cells": "A,G2,G3,G4",
            "baseline-cell": "A",
            "alternative-cell": "G4",
            "target-rps": "1000",
            "counted-runs": "1",
            "output-root": str(pathlib.Path(self.temporary.name) / "out"),
        }
        values.update({key.replace("_", "-"): value for key, value in extra.items()})
        argv: list[str] = []
        for key, value in values.items():
            argv.extend([f"--{key}", value])
        return argv

    def test_g4_calibration_accepted_with_six_key_metadata(self) -> None:
        args = build_parser().parse_args(self._args("calibration"))
        run_set = collect(args, runner=self.world.runner)
        self.assertEqual(len(self.world.trial_calls), 8)
        environment = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
        protocol = environment["collection_protocol"]
        self.assertEqual(
            protocol["calibration_variant"], CALIBRATION_VARIANT_ARRIVAL_AWARE_BATCHING_V1
        )
        self.assertTrue(protocol["zero_admission_rejection_required"])
        self.assertTrue(protocol["require_strict_generator_delivery"])
        definitions = {cell["id"]: cell for cell in protocol["cell_definitions"]}
        self.assertEqual(
            (definitions["A"]["adaptive_batching"], definitions["A"]["arrival_aware_batching"]),
            (False, False),
        )
        self.assertEqual(
            (definitions["G2"]["adaptive_batching"], definitions["G2"]["arrival_aware_batching"]),
            (False, False),
        )
        self.assertEqual(
            (definitions["G3"]["adaptive_batching"], definitions["G3"]["arrival_aware_batching"]),
            (True, False),
        )
        self.assertEqual(
            (definitions["G4"]["adaptive_batching"], definitions["G4"]["arrival_aware_batching"]),
            (False, True),
        )
        for cell in protocol["cell_definitions"]:
            self.assertEqual(len(cell), 6)
        plan = json.loads((run_set / "execution_plan.json").read_text(encoding="utf-8"))["trials"]
        by_id = {trial["trial_id"]: trial for trial in plan}
        self.assertNotIn("--adaptive-batching", by_id["A-neural_open-01"]["command"])
        self.assertNotIn("--arrival-aware-batching", by_id["A-neural_open-01"]["command"])
        self.assertNotIn("--adaptive-batching", by_id["G2-neural_open-01"]["command"])
        self.assertNotIn("--arrival-aware-batching", by_id["G2-neural_open-01"]["command"])
        self.assertIn("--adaptive-batching", by_id["G3-neural_open-01"]["command"])
        self.assertNotIn("--arrival-aware-batching", by_id["G3-neural_open-01"]["command"])
        self.assertIn("--arrival-aware-batching", by_id["G4-neural_open-01"]["command"])
        self.assertNotIn("--adaptive-batching", by_id["G4-neural_open-01"]["command"])
        self.assertIn("--enable-batching", by_id["G4-neural_open-01"]["command"])
        g4_launch = self.world.trial_calls[7]
        self.assertIn("--arrival-aware-batching", g4_launch)
        self.assertNotIn("--adaptive-batching", g4_launch)
        meta = json.loads((run_set / "runs" / "G4-neural_open-01.meta.json").read_text(encoding="utf-8"))
        self.assertTrue(meta["collector_valid"])
        self.assertIn("realized_mean_batch_size", meta)

    def test_g4_smoke_and_reportable_rejected(self) -> None:
        smoke_argv = [
            "--binary",
            str(self.world.binary),
            "--repo-root",
            str(self.world.repo),
            "--expected-commit",
            COMMIT,
            "--mode",
            "smoke",
            "--model-id",
            "gemm_f32",
            "--cells",
            "A,G4",
            "--output-root",
            str(pathlib.Path(self.temporary.name) / "smoke-g4"),
        ]
        with self.assertRaises(NeuralCollectorError) as smoke:
            collect(build_parser().parse_args(smoke_argv), runner=self.world.runner)
        self.assertIn("not permitted", str(smoke.exception))
        argv = self._args("reportable", cells="A,G4", counted_runs="20", target_rps="1000")
        argv.append("--offered-rate-frozen")
        with self.assertRaises(NeuralCollectorError):
            collect(build_parser().parse_args(argv), runner=self.world.runner)
        self.assertEqual(self.world.trial_calls, [])

    def test_hard_invalid_delivery_stops_without_soft_promote(self) -> None:
        original = self.world.runner

        def weak_delivery(args, cwd=None, env=None, timeout_seconds=None):
            result = original(args, cwd=cwd, env=env, timeout_seconds=timeout_seconds)
            if pathlib.Path(args[0]).name != BINARY_NAME or "--format" not in args:
                return result
            document = json.loads(result.stdout)
            document["submit_attempts"] = document["scheduled_opportunities"] - 1
            document["generator_skips_outstanding"] = 1
            document["accepted"] = document["submit_attempts"]
            document["succeeded"] = document["accepted"]
            document["successful_e2e_latency"]["sample_count"] = document["succeeded"]
            document["metrics"]["engine_run_count"] = document["succeeded"]
            document["metrics"]["requests_entered_engine_runs"] = document["succeeded"]
            document["metrics"]["batch_size_histogram"] = [0, document["succeeded"]] + [0] * 31
            document["correctness_checked"] = document["succeeded"]
            document["generator_lag"]["count"] = document["submit_attempts"]
            from benchmarks.collect_baseline import ProcessResult

            return ProcessResult(0, dump_report(document), "")

        with self.assertRaises(NeuralCollectorError) as raised:
            collect(
                build_parser().parse_args(self._args("calibration")),
                runner=weak_delivery,
            )
        run_set = raised.exception.run_set
        assert run_set is not None
        failure = json.loads((run_set / "failure.json").read_text(encoding="utf-8"))
        self.assertFalse(failure["accepted"])
        meta = json.loads(
            (run_set / "runs" / f"{failure['trial_id']}.meta.json").read_text(encoding="utf-8")
        )
        self.assertFalse(meta["collector_valid"])
        self.assertNotIn("calibration_infeasible_markers", meta)


if __name__ == "__main__":
    unittest.main()
