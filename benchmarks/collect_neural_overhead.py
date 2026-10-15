#!/usr/bin/env python3
"""Collect a paired neural batch-decision timing overhead run.

OFF omits batch-decision timing and must emit schema 3.
ON passes --batch-decision-timing with capacity 1024 and must emit schema 5.
One scheduler policy (G2 or G4) per run set. Completion timing stays off.
experiment_kind is batch_decision_timing_overhead with protocol sentinel 0.
This module does not execute formal studies by itself and never declares a winner.
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import platform
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.collect_baseline import (
    ONNXRUNTIME_VERSION,
    assess_build_cache,
    cache_flag_text,
    compiler_and_cmake,
    create_run_set,
    git_status_problems,
    locate_build_dir,
    locate_pinned_runtime,
    make_run_set_id,
    parse_cmake_cache,
    public_manifest,
    query_text,
    sanitizer_symbol_problems,
    sha256_file,
    symbol_tool_args,
    utc_now,
    verify_linked_onnx_runtime,
)
from benchmarks.collect_concurrent import hardware_metadata, release_optimization_problems
from benchmarks.collect_neural_scheduler import (
    REFERENCE_CHECKING_POLICY,
    adaptive_batching_for_cell,
    arrival_aware_batching_for_cell,
    batch_open_wait_diagnostics,
    instrumented_trace_problems,
    neural_scheduler_command,
    normalize_cell_definition,
    open_loop_profile,
    resolve_neural_cell,
    serialize_cell_definitions,
    verify_neural_artifacts,
)
from benchmarks.collect_scheduler import (
    ALLOWED_UNTRACKED,
    BACKEND,
    BATCH_DECISION_SCHEMA_VERSION,
    BINARY_NAME,
    CollectorError,
    GRAPH_OPTIMIZATIONS,
    MINIMUM_INTERVAL_BLOCKS,
    SCHEMA_VERSION,
    SCHEMA5_EXTRA_KEYS,
    TOOL_VERSION,
    classify_report,
    link_tool_args,
    replace_json,
    require_commit_sha,
    supervise_process,
    version_text_problems,
    wall_timeout_seconds,
    write_exclusive_text,
    write_json,
)

EXPERIMENT_KIND = "batch_decision_timing_overhead"
INTENDED_VARIABLE = "enable_batch_decision_timing"
PUBLICATION_GATE = "strict_generator_delivery_per_arm"
MODEL_ID = "gemm_f32"
TARGET_RPS = 100.0
DURATION_MS = 5000
PROFILE_ID = "neural_overhead_open"
OUTPUT_ROOT_RELATIVE = "benchmarks/scheduler_runs"
ALLOWED_POLICY_CELLS = frozenset({"G2", "G4"})
SMOKE_COUNTED_BLOCKS = 0
EXPLORATORY_COUNTED_BLOCKS = 3
FORMAL_COUNTED_BLOCKS = 20
BOOTSTRAP_SEED = 20261015
BOOTSTRAP_REPLICATES = 10000
CONFIDENCE_LEVEL = 0.95
EVENT_CAPACITY = 1024
OVERHEAD_PROTOCOL_MARKER = 0
OVERHEAD_PROTOCOL_MARKER_MEANING = (
    "sentinel only; not analysis protocol 1, 2, or 3; "
    "experiment_kind batch_decision_timing_overhead is the identity"
)
ARRIVAL_FIDELITY_STATUS = "not_certified"


class NeuralOverheadError(CollectorError):
    def __init__(self, message: str, run_set: pathlib.Path | None = None) -> None:
        super().__init__(message)
        self.run_set = run_set


def policy_cell(cell_id: str) -> dict:
    if cell_id not in ALLOWED_POLICY_CELLS:
        raise NeuralOverheadError(
            f"policy cell must be one of {sorted(ALLOWED_POLICY_CELLS)}; got {cell_id!r}"
        )
    allow_g4 = cell_id == "G4"
    cell = resolve_neural_cell(cell_id, allow_g3=False, allow_g4=allow_g4)
    return normalize_cell_definition(cell)


def shared_measurement() -> dict:
    return {
        "worker_count": 1,
        "queue_capacity": 64,
        "duration_ms": DURATION_MS,
        "max_offered": 10000,
        "max_outstanding": 64,
        "completion_timeout_ms": 5000,
        "warmup_requests": 100,
        "lag_tolerance_ms": 1000,
        "check_interval": 1,
    }


def measurement_for(arm: str) -> dict:
    measurement = shared_measurement()
    if arm == "on":
        measurement["enable_batch_decision_timing"] = True
        measurement["batch_decision_event_capacity"] = EVENT_CAPACITY
    elif arm != "off":
        raise NeuralOverheadError(f"unknown arm {arm!r}")
    return measurement


def shared_launch(cell: dict) -> dict:
    if cell["id"] not in ALLOWED_POLICY_CELLS:
        raise NeuralOverheadError(f"unsupported policy cell {cell['id']!r}")
    adaptive = adaptive_batching_for_cell(cell)
    arrival = arrival_aware_batching_for_cell(cell)
    if adaptive:
        raise NeuralOverheadError("overhead study forbids adaptive_batching true")
    if cell["id"] == "G2" and arrival:
        raise NeuralOverheadError("G2 forbids arrival_aware_batching true")
    if cell["id"] == "G4" and not arrival:
        raise NeuralOverheadError("G4 requires arrival_aware_batching true")
    if (
        cell["enable_batching"] is not True
        or int(cell["max_batch_size"]) != 4
        or int(cell["max_batch_delay_ms"]) != 2
    ):
        raise NeuralOverheadError(
            f"{cell['id']} must keep batching enabled with max_batch_size=4 and delay=2"
        )
    return {
        "model_id": MODEL_ID,
        "policy_cell": cell["id"],
        "profile_id": PROFILE_ID,
        "workload": "open_loop_constant",
        "target_rps": TARGET_RPS,
        "enable_batching": True,
        "max_batch_size": 4,
        "max_batch_delay_ms": 2,
        "adaptive_batching": False,
        "arrival_aware_batching": bool(arrival),
        "worker_count": 1,
        "queue_capacity": 64,
        "warmup_requests": 100,
        "duration_ms": DURATION_MS,
        "max_offered": 10000,
        "max_outstanding": 64,
        "lag_tolerance_ms": 1000,
        "completion_timeout_ms": 5000,
        "check_interval": 1,
        "completion_timing": False,
        "reference_checking_policy": REFERENCE_CHECKING_POLICY,
    }


def counted_blocks_for(mode: str, counted_runs: int | None) -> int:
    if mode == "smoke":
        if counted_runs not in {None, 0}:
            raise NeuralOverheadError("smoke runs one uncounted OFF/ON pair and no counted blocks")
        return SMOKE_COUNTED_BLOCKS
    if mode == "exploratory":
        if counted_runs not in {None, EXPLORATORY_COUNTED_BLOCKS}:
            raise NeuralOverheadError(
                f"exploratory mode uses exactly {EXPLORATORY_COUNTED_BLOCKS} counted blocks"
            )
        return EXPLORATORY_COUNTED_BLOCKS
    if mode == "formal":
        if counted_runs not in {None, FORMAL_COUNTED_BLOCKS}:
            raise NeuralOverheadError(
                f"formal mode uses exactly {FORMAL_COUNTED_BLOCKS} counted blocks"
            )
        return FORMAL_COUNTED_BLOCKS
    raise NeuralOverheadError(f"unknown collection mode {mode!r}")


def evidence_class_for(mode: str, counted_blocks: int) -> str:
    if mode == "smoke":
        if counted_blocks != 0:
            raise NeuralOverheadError("a smoke pair cannot be promoted into counted blocks")
        return "correctness"
    if counted_blocks <= 0:
        raise NeuralOverheadError("a counted study needs at least one OFF/ON block")
    if mode == "formal":
        if counted_blocks != FORMAL_COUNTED_BLOCKS:
            raise NeuralOverheadError("formal mode requires twenty counted blocks")
        return "counted_overhead"
    if mode == "exploratory":
        if counted_blocks != EXPLORATORY_COUNTED_BLOCKS:
            raise NeuralOverheadError("exploratory mode requires three counted blocks")
        return "exploratory"
    raise NeuralOverheadError(f"unknown collection mode {mode!r}")


def arm_order(block_index: int) -> tuple[str, str]:
    if block_index % 2 == 0:
        return ("off", "on")
    return ("on", "off")


def build_plan(counted_blocks: int, cell_id: str) -> list[dict]:
    if counted_blocks < 0 or counted_blocks > FORMAL_COUNTED_BLOCKS:
        raise NeuralOverheadError(
            f"counted blocks {counted_blocks} are outside 0..{FORMAL_COUNTED_BLOCKS}"
        )
    if cell_id not in ALLOWED_POLICY_CELLS:
        raise NeuralOverheadError(f"unsupported policy cell {cell_id!r}")
    trials: list[dict] = []
    execution_index = 0

    def add(
        arm: str,
        *,
        pilot: bool,
        block_index: int | None,
        position: int,
        order_label: str,
    ) -> None:
        nonlocal execution_index
        label = "pilot" if pilot else f"b{block_index:02d}"
        trials.append(
            {
                "trial_id": f"neural-overhead-{label}-{arm}",
                "arm": arm,
                "pilot": pilot,
                "block_index": block_index,
                "position_in_block": position,
                "execution_index": execution_index,
                "run_index": None if pilot else block_index,
                "cell_id": cell_id,
                "policy_cell": cell_id,
                "profile_id": PROFILE_ID,
                "arm_order": order_label,
            }
        )
        execution_index += 1

    for position, arm in enumerate(("off", "on")):
        add(arm, pilot=True, block_index=None, position=position, order_label="off_then_on")
    for block_index in range(counted_blocks):
        order = arm_order(block_index)
        order_label = "_then_".join(order)
        for position, arm in enumerate(order):
            add(
                arm,
                pilot=False,
                block_index=block_index,
                position=position,
                order_label=order_label,
            )
    return trials


def command_for_arm(
    binary: pathlib.Path,
    model: pathlib.Path,
    cell: dict,
    arm: str,
) -> list[str]:
    profile = open_loop_profile(PROFILE_ID, TARGET_RPS)
    return neural_scheduler_command(
        binary,
        model,
        MODEL_ID,
        cell,
        profile,
        measurement_for(arm),
    )


def commands_differ_only_by_batch_decision(off: list[str], on: list[str]) -> bool:
    if "--batch-decision-timing" in off or "--batch-decision-event-capacity" in off:
        return False
    if "--completion-timing" in off or "--completion-timing" in on:
        return False
    if on.count("--batch-decision-timing") != 1:
        return False
    if "--batch-decision-event-capacity" not in on:
        return False
    capacity_index = on.index("--batch-decision-event-capacity")
    if capacity_index + 1 >= len(on) or on[capacity_index + 1] != str(EVENT_CAPACITY):
        return False
    cleaned: list[str] = []
    skip_next = False
    for token in on:
        if skip_next:
            skip_next = False
            continue
        if token == "--batch-decision-timing":
            continue
        if token == "--batch-decision-event-capacity":
            skip_next = True
            continue
        cleaned.append(token)
    return cleaned == off


def bind_commands(
    plan: list[dict],
    binary: pathlib.Path,
    model: pathlib.Path,
    cell: dict,
) -> list[dict]:
    bound = []
    for trial in plan:
        command = command_for_arm(binary, model, cell, str(trial["arm"]))
        bound.append({**trial, "command": command})
    off = command_for_arm(binary, model, cell, "off")
    on = command_for_arm(binary, model, cell, "on")
    if not commands_differ_only_by_batch_decision(off, on):
        raise NeuralOverheadError(
            "OFF and ON commands differ by more than batch-decision timing flags"
        )
    return bound


def expected_for(model: pathlib.Path, cell: dict, arm: str) -> dict:
    launch = shared_launch(cell)
    measurement = shared_measurement()
    expected = {
        "model_path": str(model),
        "workload": launch["workload"],
        "worker_count": measurement["worker_count"],
        "queue_capacity": measurement["queue_capacity"],
        "enable_batching": True,
        "max_batch_size": 4,
        "max_batch_delay_ms": 2,
        "adaptive_batching": False,
        "arrival_aware_batching": launch["arrival_aware_batching"],
        "require_adaptive_batching_field": True,
        "require_arrival_aware_batching_field": True,
        "warmup_requests": measurement["warmup_requests"],
        "check_interval": measurement["check_interval"],
        "duration_ms": measurement["duration_ms"],
        "lag_tolerance_ms": measurement["lag_tolerance_ms"],
        "comparison_class": "primary",
        "analysis_protocol_version": 2,
        "target_rps": TARGET_RPS,
    }
    if arm == "on":
        expected["enable_batch_decision_timing"] = True
        expected["batch_decision_event_capacity"] = EVENT_CAPACITY
    elif arm != "off":
        raise NeuralOverheadError(f"unknown arm {arm!r}")
    return expected


def validate_arm(
    text: str,
    arm: str,
    model: pathlib.Path,
    cell: dict,
) -> tuple[dict | None, dict, list[str]]:
    expected = expected_for(model, cell, arm)
    document, status = classify_report(text, expected=expected)
    problems: list[str] = []
    if status.get("collector_valid") is not True:
        problems.extend(str(item) for item in status.get("problems", []))
        if not problems:
            problems.append("collector rejected the trial")
    if status.get("delivery_class") != "STRICT_PRIMARY":
        problems.append(
            f"delivery_class is {status.get('delivery_class')!r}, expected STRICT_PRIMARY"
        )
    if status.get("runner_valid") is not True:
        problems.append("runner valid flag is not true")
    if document is None:
        return None, status, problems
    want_schema = BATCH_DECISION_SCHEMA_VERSION if arm == "on" else SCHEMA_VERSION
    if document.get("schema_version") != want_schema:
        problems.append(
            f"{arm} schema_version is {document.get('schema_version')!r}, expected {want_schema}"
        )
    scheduler = document.get("scheduler")
    if not isinstance(scheduler, dict):
        problems.append("scheduler object is missing")
        return document, status, problems
    if arm == "off":
        if "enable_batch_decision_timing" in scheduler:
            problems.append("OFF arm recorded enable_batch_decision_timing")
        if "batch_decision_event_capacity" in scheduler:
            problems.append("OFF arm recorded batch_decision_event_capacity")
        if "batch_decision" in document:
            problems.append("OFF arm contains batch_decision")
        for key in SCHEMA5_EXTRA_KEYS:
            if key in document:
                problems.append(f"OFF arm contains unexpected top-level field {key}")
    else:
        if scheduler.get("enable_batch_decision_timing") is not True:
            problems.append("ON arm did not record enable_batch_decision_timing")
        if scheduler.get("batch_decision_event_capacity") != EVENT_CAPACITY:
            problems.append("ON arm batch_decision_event_capacity is not 1024")
        if scheduler.get("enable_completion_timing") not in {False, None}:
            problems.append("ON arm must keep completion timing off")
        problems.extend(instrumented_trace_problems(document, cell, expected))
    if document.get("arrival_stop_reason") != "configured_duration":
        problems.append("arrival stop is not configured_duration")
    if document.get("correctness_mismatches") != 0:
        problems.append("correctness mismatches are not zero")
    if document.get("failed_inference") != 0:
        problems.append("failed_inference is not zero")
    if document.get("failed_scheduler") != 0:
        problems.append("failed_scheduler is not zero")
    if document.get("failed_worker") != 0:
        problems.append("failed_worker is not zero")
    if document.get("timeouts") != 0:
        problems.append("timeouts are not zero")
    if document.get("canceled") != 0:
        problems.append("canceled is not zero")
    if document.get("generator_skips_outstanding") != 0:
        problems.append("generator_skips_outstanding is not zero")
    if document.get("generator_skips_lag") != 0:
        problems.append("generator_skips_lag is not zero")
    rejections = document.get("rejections")
    if isinstance(rejections, dict):
        if sum(int(rejections.get(name, 0) or 0) for name in ("full", "closed", "not_running")) != 0:
            problems.append("scheduler admission rejections are not zero")
    if document.get("submit_attempts") != document.get("scheduled_opportunities"):
        problems.append("submit_attempts does not equal scheduled_opportunities")
    if document.get("accepted") != document.get("submit_attempts"):
        problems.append("accepted does not equal submit_attempts")
    if document.get("succeeded") != document.get("accepted"):
        problems.append("succeeded does not equal accepted")
    metrics = document.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("finalized") is not True:
        problems.append("metrics are not finalized")
    if isinstance(metrics, dict) and metrics.get("histogram_saturated") is True:
        problems.append("histogram is saturated")
    if (
        scheduler.get("adaptive_batching") is not False
        or scheduler.get("arrival_aware_batching") is not shared_launch(cell)["arrival_aware_batching"]
    ):
        problems.append("scheduler Gen C policy identity does not match the frozen cell")
    return document, status, problems


def _write_failure(run_set: pathlib.Path, payload: dict) -> None:
    path = run_set / "failure.json"
    if path.exists():
        return
    write_json(path, payload)


def overhead_preflight(
    *,
    repo_root: pathlib.Path,
    binary: pathlib.Path,
    expected_commit: str,
    cell: dict,
    mode: str,
    counted_blocks: int,
    runner,
) -> dict:
    commit = require_commit_sha(expected_commit)
    cell = normalize_cell_definition(cell)
    launch = shared_launch(cell)
    measurement = shared_measurement()
    profile = open_loop_profile(PROFILE_ID, TARGET_RPS)
    if not binary.is_file():
        raise NeuralOverheadError(f"benchmark binary does not exist: {binary}")
    if not os.access(binary, os.X_OK):
        raise NeuralOverheadError(f"benchmark binary is not executable: {binary}")
    if binary.resolve().name != BINARY_NAME:
        raise NeuralOverheadError(
            f"benchmark binary must be named {BINARY_NAME}, got {binary.resolve().name}"
        )
    build_dir = locate_build_dir(binary)
    cache = parse_cmake_cache((build_dir / "CMakeCache.txt").read_text(encoding="utf-8"))
    problems = assess_build_cache(cache)
    problems.extend(release_optimization_problems(cache))
    if problems:
        raise NeuralOverheadError("; ".join(problems))
    symbols = query_text(runner, symbol_tool_args(binary))
    if symbols.returncode != 0:
        raise NeuralOverheadError(
            "could not inspect the benchmark binary for sanitizer symbols: "
            f"{symbols.stderr.strip() or symbols.stdout.strip()}"
        )
    symbol_problems = sanitizer_symbol_problems(symbols.stdout + symbols.stderr)
    if symbol_problems:
        raise NeuralOverheadError("; ".join(symbol_problems))
    runtime_root = locate_pinned_runtime(cache, build_dir)
    linked = query_text(runner, link_tool_args(binary))
    if linked.returncode != 0 and not linked.stdout.strip():
        raise NeuralOverheadError(
            "could not identify the ONNX Runtime library linked into the benchmark binary: "
            f"{linked.stderr.strip() or linked.stdout.strip()}"
        )
    load_command_output = ""
    if platform.system() == "Darwin":
        load_commands = query_text(runner, ["otool", "-l", str(binary)])
        if load_commands.returncode != 0:
            raise NeuralOverheadError(
                "could not read Mach-O load commands for the benchmark binary: "
                f"{load_commands.stderr.strip() or load_commands.stdout.strip()}"
            )
        load_command_output = load_commands.stdout
    resolved_library = verify_linked_onnx_runtime(
        binary=binary,
        runtime_root=runtime_root,
        link_output=linked.stdout + linked.stderr,
        load_command_output=load_command_output,
    )
    onnx_version = (runtime_root / "VERSION_NUMBER").read_text(encoding="utf-8").strip()
    if onnx_version != ONNXRUNTIME_VERSION:
        raise NeuralOverheadError(
            f"ONNX Runtime VERSION_NUMBER is {onnx_version!r}, expected {ONNXRUNTIME_VERSION!r}"
        )
    version = query_text(runner, [str(binary), "--version"])
    version_problems = version_text_problems(version.stdout, version.stderr, version.returncode)
    if version_problems:
        raise NeuralOverheadError("; ".join(version_problems))
    artifacts = verify_neural_artifacts(repo_root, MODEL_ID)
    head = query_text(runner, ["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    status = query_text(runner, ["git", "-C", str(repo_root), "status", "--porcelain"])
    if head.returncode != 0 or status.returncode != 0:
        raise NeuralOverheadError(
            "git verification failed: " + (head.stderr.strip() or status.stderr.strip())
        )
    actual_commit = head.stdout.strip()
    git_problems = git_status_problems(
        status.stdout,
        expected_commit=commit,
        actual_commit=actual_commit,
        allowed_untracked=ALLOWED_UNTRACKED,
    )
    if git_problems:
        raise NeuralOverheadError("; ".join(git_problems))
    host = hardware_metadata(runner)
    tools = compiler_and_cmake(runner, cache["CMAKE_CXX_COMPILER"])
    cell_definitions = serialize_cell_definitions(
        [cell], include_adaptive=True, include_arrival_aware=True
    )
    if len(cell_definitions) != 1 or len(cell_definitions[0]) != 6:
        raise NeuralOverheadError("policy cell definitions must be six-key Gen C records")
    evidence_class = evidence_class_for(mode, counted_blocks)
    return {
        "queried": {
            "queried_at_utc": utc_now(),
            "git_commit": actual_commit,
            "compiler_path": tools["compiler_path"],
            "compiler_version": tools["compiler_version"],
            "cmake_version": tools["cmake_version"],
            "cmake_flag_text": cache_flag_text(cache),
            "benchmark_binary_path": str(binary.resolve()),
            "benchmark_binary_sha256": sha256_file(binary),
            "benchmark_tool_version": TOOL_VERSION,
            "onnxruntime_version": onnx_version,
            "onnxruntime_resolved_library": str(resolved_library),
            "onnxruntime_link_output": linked.stdout,
            "operating_system": host["system"],
            "os_version": host["os_version"],
            "platform": host["platform"],
            "cpu_architecture": host["machine"],
            "hardware_model": host["hardware_model"],
            "hardware_model_error": host["hardware_model_error"],
            "logical_cpu_count": host["logical_cpu_count"],
            "logical_cpu_count_source": host["logical_cpu_count_source"],
            "hw_physicalcpu": host["hw_physicalcpu"],
            "hw_physicalcpu_error": host["hw_physicalcpu_error"],
            "hw_logicalcpu": host["hw_logicalcpu"],
            "hw_logicalcpu_error": host["hw_logicalcpu_error"],
            "nproc_cpu_count": host["nproc_cpu_count"],
            "nproc_cpu_count_error": host["nproc_cpu_count_error"],
            "python_version": host["python_version"],
            "ort_disable_telemetry": "1",
        },
        "verified_configuration": {
            "cmake_build_type": cache["CMAKE_BUILD_TYPE"],
            "sanitizer_instrumentation": "absent",
            "onnxruntime_version": onnx_version,
            "model_path": str(artifacts["model_path"]),
            "model_sha256": artifacts["model_sha256"],
            "model_relative": artifacts["model_relative"],
            "model_id": MODEL_ID,
            "weight_sidecar_path": str(artifacts["weight_sidecar_path"]),
            "weight_sidecar_relative": artifacts["weight_sidecar_relative"],
            "weight_sidecar_sha256": artifacts["weight_sidecar_sha256"],
            "input_name": artifacts["input_name"],
            "output_name": artifacts["output_name"],
            "input_width": artifacts["input_width"],
            "output_width": artifacts["output_width"],
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
            "policy_cell": cell["id"],
        },
        "source_defined_policy": {
            "backend": BACKEND,
            "graph_optimizations": GRAPH_OPTIMIZATIONS,
            "onnxruntime_version_pin": ONNXRUNTIME_VERSION,
            "schema_version": None,
            "off_schema_version": SCHEMA_VERSION,
            "on_schema_version": BATCH_DECISION_SCHEMA_VERSION,
            "analysis_protocol_version": OVERHEAD_PROTOCOL_MARKER,
            "analysis_protocol_version_meaning": OVERHEAD_PROTOCOL_MARKER_MEANING,
            "publication_policy": PUBLICATION_GATE,
            "experiment_kind": EXPERIMENT_KIND,
            "intended_variable": INTENDED_VARIABLE,
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
            "arrival_fidelity_status": ARRIVAL_FIDELITY_STATUS,
            "allowed_untracked_roots": list(ALLOWED_UNTRACKED),
            "rates_are_calibrated_capacity": False,
        },
        "collection_protocol": {
            "experiment_kind": EXPERIMENT_KIND,
            "intended_variable": INTENDED_VARIABLE,
            "evidence_class": evidence_class,
            "collection_mode": mode,
            "analysis_protocol_version": OVERHEAD_PROTOCOL_MARKER,
            "analysis_protocol_version_meaning": OVERHEAD_PROTOCOL_MARKER_MEANING,
            "publication_policy": PUBLICATION_GATE,
            "comparison_class": "exploratory",
            "model_id": MODEL_ID,
            "model_sha256": artifacts["model_sha256"],
            "weight_sidecar_sha256": artifacts["weight_sidecar_sha256"],
            "weight_sidecar_path": str(artifacts["weight_sidecar_path"]),
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
            "policy_cell": cell["id"],
            "cell_definitions": cell_definitions,
            "shared_launch": launch,
            "measurement": measurement,
            "profile_definitions": [profile],
            "counted_blocks": counted_blocks,
            "pilot_pair": "off_then_on",
            "counted_rotation": "block_index starts at 0; even off-then-on; odd on-then-off",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": CONFIDENCE_LEVEL,
            "minimum_interval_blocks": MINIMUM_INTERVAL_BLOCKS,
            "preferred_counted_blocks": FORMAL_COUNTED_BLOCKS,
            "offered_rate": TARGET_RPS,
            "traffic_profile": PROFILE_ID,
            "arrival_fidelity_status": ARRIVAL_FIDELITY_STATUS,
            "arrival_fidelity_note": (
                "STRICT_PRIMARY is delivery only. Schema 3 has no lag percentiles. "
                "Schema 5 batch timing has no accepted-enqueue stamps. "
                "Arrival fidelity remains not_certified."
            ),
            "environment_variable": "ORT_DISABLE_TELEMETRY=1",
            "external_wall_timeout_seconds": wall_timeout_seconds(measurement),
            "zero_admission_rejection_required": True,
            "require_strict_generator_delivery": True,
        },
        "paths": {
            "model": artifacts["model_path"],
            "sidecar": artifacts["weight_sidecar_path"],
        },
        "commit": actual_commit,
        "cell": cell,
        "profile": profile,
    }


def _overlay_identity(
    manifest: dict,
    plan: list[dict],
    counted_blocks: int,
    mode: str,
    cell: dict,
) -> dict:
    policy = dict(manifest["source_defined_policy"])
    protocol = dict(manifest["collection_protocol"])
    policy["publication_policy"] = PUBLICATION_GATE
    policy["experiment_kind"] = EXPERIMENT_KIND
    policy["intended_variable"] = INTENDED_VARIABLE
    policy["off_schema_version"] = SCHEMA_VERSION
    policy["on_schema_version"] = BATCH_DECISION_SCHEMA_VERSION
    policy["schema_version"] = None
    policy["analysis_protocol_version"] = OVERHEAD_PROTOCOL_MARKER
    policy["analysis_protocol_version_meaning"] = OVERHEAD_PROTOCOL_MARKER_MEANING
    policy["limitations"] = [
        "The contrast is the incremental effect of enabling batch-decision timing.",
        "Producer-observed P50 is not isolated CPU instrumentation cost.",
        "Instrumentation ON may perturb batch formation and seal decisions.",
        "Arrival fidelity remains not_certified.",
        "Results do not generalize across policies, loads, or machines.",
        "An interval that includes zero is inconclusive, not a zero-overhead proof.",
    ]
    protocol["publication_policy"] = PUBLICATION_GATE
    protocol["experiment_kind"] = EXPERIMENT_KIND
    protocol["intended_variable"] = INTENDED_VARIABLE
    protocol["analysis_protocol_version"] = OVERHEAD_PROTOCOL_MARKER
    protocol["analysis_protocol_version_meaning"] = OVERHEAD_PROTOCOL_MARKER_MEANING
    protocol["collection_mode"] = mode
    protocol["evidence_class"] = evidence_class_for(mode, counted_blocks)
    protocol["counted_blocks"] = counted_blocks
    protocol["shared_launch"] = shared_launch(cell)
    protocol["policy_cell"] = cell["id"]
    protocol["pilot_pair"] = "off_then_on"
    protocol["counted_rotation"] = "block_index starts at 0; even off-then-on; odd on-then-off"
    protocol["bootstrap_seed"] = BOOTSTRAP_SEED
    protocol["bootstrap_replicates"] = BOOTSTRAP_REPLICATES
    protocol["confidence_level"] = CONFIDENCE_LEVEL
    protocol["arrival_fidelity_status"] = ARRIVAL_FIDELITY_STATUS
    document = dict(manifest)
    document["experiment_kind"] = EXPERIMENT_KIND
    document["intended_variable"] = INTENDED_VARIABLE
    document["source_defined_policy"] = policy
    document["collection_protocol"] = protocol
    document["plan"] = plan
    document["allowed_untracked_roots"] = list(ALLOWED_UNTRACKED)
    return document


def execute_plan(
    *,
    run_set: pathlib.Path,
    binary: pathlib.Path,
    repo_root: pathlib.Path,
    model: pathlib.Path,
    cell: dict,
    plan: list[dict],
    checked: dict,
    runner,
) -> bool:
    runs = run_set / "runs"
    measurement = shared_measurement()
    timeout_seconds = wall_timeout_seconds(measurement)
    for trial in plan:
        trial_id = str(trial["trial_id"])
        arm = str(trial["arm"])
        command = list(trial["command"])
        expected_command = command_for_arm(binary, model, cell, arm)
        if command != expected_command:
            raise NeuralOverheadError(
                f"{trial_id} plan command diverges from the {arm} arm command",
                run_set,
            )
        env = {"ORT_DISABLE_TELEMETRY": "1"}
        try:
            result = runner(command, cwd=repo_root, env=env, timeout_seconds=timeout_seconds)
        except TypeError:
            result = runner(command, cwd=repo_root, env=env)
        stdout_path = runs / f"{trial_id}.stdout"
        stderr_path = runs / f"{trial_id}.stderr"
        report_path = runs / f"{trial_id}.json"
        timed_out = bool(getattr(result, "timed_out", False))
        problems: list[str] = []
        document = None
        status: dict = {
            "runner_valid": False,
            "collector_valid": False,
            "delivery_class": None,
        }
        try:
            write_exclusive_text(stdout_path, result.stdout)
            write_exclusive_text(stderr_path, result.stderr)
        except (OSError, CollectorError) as exc:
            problems.append(f"could not preserve trial output: {exc}")
        if timed_out:
            problems.append("process exceeded the external wall-clock timeout")
        elif result.returncode != 0:
            problems.append(f"exit status {result.returncode}")
        elif result.stderr != "":
            problems.append("stderr is not empty")
        else:
            document, status, arm_problems = validate_arm(result.stdout, arm, model, cell)
            problems.extend(arm_problems)
            if not problems and document is not None:
                try:
                    write_exclusive_text(report_path, result.stdout)
                except (OSError, CollectorError) as exc:
                    problems.append(f"could not preserve trial report: {exc}")
        wait = None
        timing_valid = None
        if document is not None and arm == "on" and not problems:
            snap = document.get("batch_decision")
            events = snap.get("events") if isinstance(snap, dict) else None
            if isinstance(events, list):
                wait = batch_open_wait_diagnostics(events)
            if isinstance(snap, dict):
                timing_valid = snap.get("timing_data_valid") is True
        meta = {
            "trial_id": trial_id,
            "experiment_kind": EXPERIMENT_KIND,
            "intended_variable": INTENDED_VARIABLE,
            "arm": arm,
            "arm_order": trial.get("arm_order"),
            "pilot": bool(trial["pilot"]),
            "block_index": trial["block_index"],
            "position_in_block": trial["position_in_block"],
            "execution_index": trial["execution_index"],
            "run_index": trial["run_index"],
            "cell_id": cell["id"],
            "policy_cell": cell["id"],
            "profile_id": PROFILE_ID,
            "model_id": MODEL_ID,
            "schema_version": None if document is None else document.get("schema_version"),
            "command": command,
            "returncode": result.returncode,
            "execution_timeout": timed_out,
            "wall_timeout_seconds": timeout_seconds,
            "runner_valid": status.get("runner_valid"),
            "collector_valid": status.get("collector_valid") is True and not problems,
            "delivery_class": status.get("delivery_class"),
            "arrival_fidelity_status": ARRIVAL_FIDELITY_STATUS,
            "timing_data_valid": timing_valid,
            "batch_open_wait": wait,
            "problems": problems,
            "benchmark_binary_sha256": checked["queried"]["benchmark_binary_sha256"],
            "model_sha256": checked["verified_configuration"]["model_sha256"],
            "weight_sidecar_sha256": checked["verified_configuration"]["weight_sidecar_sha256"],
            "git_commit": checked["commit"],
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
            "retried": False,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "report_path": str(report_path) if report_path.is_file() else None,
        }
        write_json(runs / f"{trial_id}.meta.json", meta)
        if problems:
            _write_failure(
                run_set,
                {
                    "status": "failed",
                    "accepted": False,
                    "trial_id": trial_id,
                    "arm": arm,
                    "policy_cell": cell["id"],
                    "pilot": bool(trial["pilot"]),
                    "reason": "; ".join(problems),
                    "execution_timeout": timed_out,
                    "preserved_earlier_trials": True,
                    "retried": False,
                    "replaced": False,
                    "aggregate": None,
                },
            )
            return False
    return True


def collect(
    args: argparse.Namespace,
    runner=None,
    clock=None,
) -> pathlib.Path:
    repo_root = pathlib.Path(args.repo_root).resolve()
    binary = pathlib.Path(args.binary).resolve()
    cell = policy_cell(str(args.policy_cell))
    counted = counted_blocks_for(args.mode, args.counted_runs)
    measurement = shared_measurement()
    if runner is None:

        def runner(command, cwd=None, env=None, timeout_seconds=None):
            return supervise_process(
                command,
                cwd=cwd,
                env=env,
                timeout_seconds=wall_timeout_seconds(measurement)
                if timeout_seconds is None
                else timeout_seconds,
            )

    def query_runner(command, cwd=None, env=None):
        try:
            return runner(command, cwd=cwd, env=env, timeout_seconds=30)
        except TypeError:
            return runner(command, cwd=cwd, env=env)

    checked = overhead_preflight(
        repo_root=repo_root,
        binary=binary,
        expected_commit=args.expected_commit,
        cell=cell,
        mode=args.mode,
        counted_blocks=counted,
        runner=query_runner,
    )
    model = checked["paths"]["model"]
    plan = bind_commands(build_plan(counted, cell["id"]), binary, model, cell)
    moment = datetime.datetime.now(datetime.timezone.utc) if clock is None else clock()
    run_set_id = "neural-overhead-" + make_run_set_id(moment, checked["commit"])
    parent = (
        pathlib.Path(args.output_root).resolve()
        if args.output_root
        else repo_root / OUTPUT_ROOT_RELATIVE
    )
    try:
        run_set = create_run_set(parent, run_set_id)
    except CollectorError as exc:
        raise NeuralOverheadError(str(exc)) from exc
    manifest = _overlay_identity(
        public_manifest(checked, run_set_id, "running"),
        plan,
        counted,
        args.mode,
        cell,
    )
    try:
        write_json(run_set / "environment.json", manifest)
        write_json(run_set / "execution_plan.json", {"trials": plan})
        accepted = execute_plan(
            run_set=run_set,
            binary=binary,
            repo_root=repo_root,
            model=model,
            cell=cell,
            plan=plan,
            checked=checked,
            runner=runner,
        )
    except NeuralOverheadError:
        raise
    except CollectorError as exc:
        raise NeuralOverheadError(str(exc), run_set) from exc
    except Exception as exc:
        raise NeuralOverheadError(f"neural overhead collection error: {exc}", run_set) from exc
    final = _overlay_identity(
        public_manifest(checked, run_set_id, "completed" if accepted else "failed"),
        plan,
        counted,
        args.mode,
        cell,
    )
    replace_json(run_set / "environment.json", final)
    if not accepted:
        raise NeuralOverheadError(
            f"neural overhead collection stopped; see {run_set / 'failure.json'}",
            run_set,
        )
    return run_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a gemm_f32 G2/G4 batch-decision timing overhead run under "
            "benchmarks/scheduler_runs. Smoke is one uncounted pair. "
            "Exploratory uses three counted blocks. Formal planning is supported "
            "but requires separate execution authorization. "
            "This does not analyze or publish a speed claim."
        )
    )
    parser.add_argument("--binary", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--mode", choices=("smoke", "exploratory", "formal"), required=True)
    parser.add_argument(
        "--policy-cell",
        choices=sorted(ALLOWED_POLICY_CELLS),
        required=True,
        help="Exactly one scheduler policy per run set (G2 or G4).",
    )
    parser.add_argument("--counted-runs", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        collect(args)
    except NeuralOverheadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
