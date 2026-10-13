#!/usr/bin/env python3
"""Collect schema-version-3 reports from inferedge-bench-scheduler.

This module does not run a neural network itself. It checks a Release build,
the pinned batch-add model, and the repository, then starts one supervised
inferedge-bench-scheduler process per trial. New collections use analysis
protocol 2. A strict primary open-loop trial must submit every scheduled
opportunity. Scheduler rejections remain admission outcomes. Protocol 1 is
applied only when existing evidence has no protocol version or records
version 1. The collector does not write aggregate.json and does not retry a
failed counted trial.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import platform
import signal
import subprocess
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.collect_baseline import (
    BACKEND,
    GRAPH_OPTIMIZATIONS,
    ONNXRUNTIME_VERSION,
    CollectorError,
    ProcessResult,
    approved_untracked_paths,
    assess_build_cache,
    cache_flag_text,
    compiler_and_cmake,
    create_run_set,
    git_status_problems,
    json_integer,
    json_number,
    link_tool_args,
    locate_pinned_runtime,
    make_run_set_id,
    parse_cmake_cache,
    parse_single_document,
    public_manifest,
    query_text,
    require_exact_keys,
    sanitizer_symbol_problems,
    sha256_file,
    symbol_tool_args,
    utc_now,
    verify_linked_onnx_runtime,
)
from benchmarks.collect_concurrent import hardware_metadata, release_optimization_problems

TOOL_NAME = "inferedge-bench-scheduler"
TOOL_VERSION = "0.1.0"
BINARY_NAME = "inferedge-bench-scheduler"
SCHEMA_VERSION = 3
MODEL_RELATIVE = "models/batch_add_f32.onnx"
MODEL_SHA256 = "f33bbe0ad5420419dd2dbb5322f4172e213b854f1124c660b1d732e8f8ce3b23"
OUTPUT_ROOT_RELATIVE = "benchmarks/scheduler_runs"
HISTOGRAM_BINS = 33
UINT64_MAX = 18446744073709551615
OPEN_LOOP_QUANTUM_NS = 1_000_000
MINIMUM_SUBMIT_FRACTION = 0.99
ANALYSIS_PROTOCOL_VERSION = 2
PUBLICATION_POLICY = "strict_generator_delivery"
PROTOCOL1_PUBLICATION_POLICY = "submit_fraction_0_99"
DEFAULT_COUNTED_RUNS = 5
PROTOCOL2_DEFAULT_COUNTED_RUNS = 10
PREFERRED_COUNTED_RUNS = 20
MINIMUM_INTERVAL_BLOCKS = 10
MAX_COUNTED_RUNS = 30
BOOTSTRAP_SEED = 20261013
BOOTSTRAP_REPLICATES = 10000
CONFIDENCE_LEVEL = 0.95
STATISTICAL_METHOD = "paired_block_bootstrap_mean"
BLOCK_PROTOCOL = "within_profile_cyclic_rotation"
TIMING_QUALITY = "submitted_lag_max_within_configured_tolerance"
TIMING_QUALITY_NOTE = (
    "Maximum submitted lag is within the configured tolerance. "
    "The schema records count, minimum, maximum, and mean only. "
    "This is not a percentile arrival distribution and does not certify "
    "microsecond-level scheduling accuracy."
)
DEFAULT_DELAY_SECONDS = 5.0
OBSERVATION_OPEN = "open_loop_poll"
OBSERVATION_CLOSED = "closed_loop_producer_wait"
E2E_DEFINITION = (
    "steady_clock immediately before submit() until the future is observed ready"
)
ARRIVAL_WINDOW_NOTE = "offered-cohort rate; not steady-state completion capacity"
TOTAL_TRIAL_NOTE = "measured successes divided by the arrival window plus the completion drain"
COMPLETION_TIMEOUT_BOUND = "future_observation_only"

ALLOWED_UNTRACKED = (
    "models/experiments",
    "benchmarks/baselines",
    "benchmarks/concurrent_runs",
    OUTPUT_ROOT_RELATIVE,
)

CELLS = (
    {"id": "A", "enable_batching": False, "max_batch_size": 1, "max_batch_delay_ms": 0},
    {"id": "B", "enable_batching": True, "max_batch_size": 1, "max_batch_delay_ms": 0},
    {"id": "C", "enable_batching": True, "max_batch_size": 4, "max_batch_delay_ms": 0},
    {"id": "D", "enable_batching": True, "max_batch_size": 8, "max_batch_delay_ms": 0},
    {"id": "E", "enable_batching": True, "max_batch_size": 16, "max_batch_delay_ms": 0},
    {"id": "F", "enable_batching": True, "max_batch_size": 32, "max_batch_delay_ms": 0},
    {"id": "G", "enable_batching": True, "max_batch_size": 8, "max_batch_delay_ms": 1},
    {"id": "H", "enable_batching": True, "max_batch_size": 16, "max_batch_delay_ms": 1},
)
CELL_BY_ID = {cell["id"]: cell for cell in CELLS}

PROFILES = (
    {
        "id": "low_open",
        "workload": "open_loop_constant",
        "family": "open_loop",
        "target_rps": 50.0,
        "label": "declared low offered rate; not a calibrated capacity",
    },
    {
        "id": "moderate_open",
        "workload": "open_loop_constant",
        "family": "open_loop",
        "target_rps": 200.0,
        "label": "declared moderate offered rate; not a calibrated capacity",
    },
    {
        "id": "high_open",
        "workload": "open_loop_constant",
        "family": "open_loop",
        "target_rps": 1000.0,
        "label": "declared high offered rate; not a calibrated capacity",
    },
    {
        "id": "burst_open",
        "workload": "open_loop_burst",
        "family": "open_loop",
        "burst_high_rps": 500.0,
        "burst_low_rps": 50.0,
        "burst_high_ms": 200,
        "burst_low_ms": 200,
        "label": "declared burst rates; not a calibrated capacity",
    },
    {
        "id": "closed_optional",
        "workload": "closed_loop",
        "family": "closed_loop",
        "producer_count": 1,
        "label": "optional closed-loop comparison; not an open-loop arrival process",
    },
)
PROFILE_BY_ID = {profile["id"]: profile for profile in PROFILES}

DEFAULT_MEASUREMENT = {
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

TOP_LEVEL_KEYS = (
    "schema_version",
    "tool",
    "tool_version",
    "valid",
    "invalid_reasons",
    "model",
    "workload",
    "e2e_latency_definition",
    "completion_observation",
    "observation_quantum_nanoseconds",
    "scheduler",
    "generator",
    "configured_duration_nanoseconds",
    "scheduled_span_nanoseconds",
    "arrival_stop_reason",
    "arrival_window_nanoseconds",
    "completion_nanoseconds",
    "total_trial_nanoseconds",
    "completion_timeout_bound",
    "completion_timeout_note",
    "outstanding_cap_note",
    "scheduled_opportunities",
    "submit_attempts",
    "generator_skips_outstanding",
    "generator_skips_lag",
    "accepted",
    "rejections",
    "succeeded",
    "failed_inference",
    "failed_worker",
    "failed_scheduler",
    "canceled",
    "timeouts",
    "generator_lag",
    "successful_e2e_latency",
    "throughput",
    "metrics",
    "correctness_checked",
    "correctness_mismatches",
    "error_count",
    "errors",
    "burst_phases",
    "queue_depth",
    "executing",
    "live_workers",
    "exceptional_promise_backstop",
)
SCHEDULER_KEYS = (
    "worker_count",
    "queue_capacity",
    "enable_batching",
    "max_batch_size",
    "max_batch_delay_ms",
    "enable_metrics",
)
GENERATOR_KEYS = (
    "lag_definition",
    "lag_policy",
    "outstanding_policy",
    "warmup_requests",
    "warmup_separate_scheduler",
    "check_interval",
)
REJECTION_KEYS = ("full", "closed", "not_running")
LAG_KEYS = ("count", "minimum_nanoseconds", "maximum_nanoseconds", "mean_nanoseconds")
LATENCY_KEYS = (
    "sample_count",
    "minimum_nanoseconds",
    "maximum_nanoseconds",
    "mean_microseconds",
    "p50_microseconds",
    "p95_microseconds",
    "p99_microseconds",
    "percentiles_reported",
    "precise_tail",
)
THROUGHPUT_KEYS = (
    "arrival_window_seconds",
    "completion_seconds",
    "total_trial_seconds",
    "offered_opportunities_per_arrival_window_second",
    "submit_attempts_per_arrival_window_second",
    "accepted_per_arrival_window_second",
    "successful_attempts_per_arrival_window_second",
    "successful_per_total_trial_second",
    "successful_attempts_per_arrival_window_second_note",
    "successful_per_total_trial_second_note",
)
METRICS_KEYS = (
    "enabled",
    "finalized",
    "engine_run_count",
    "requests_entered_engine_runs",
    "batch_size_histogram",
    "total_assembly_nanoseconds",
    "total_engine_run_nanoseconds",
    "total_split_nanoseconds",
    "histogram_saturated",
)
COUNT_FIELDS = (
    "scheduled_opportunities",
    "submit_attempts",
    "generator_skips_outstanding",
    "generator_skips_lag",
    "accepted",
    "succeeded",
    "failed_inference",
    "failed_worker",
    "failed_scheduler",
    "canceled",
    "timeouts",
    "correctness_checked",
    "correctness_mismatches",
    "error_count",
    "queue_depth",
    "executing",
    "live_workers",
)
DURATION_FIELDS = (
    "configured_duration_nanoseconds",
    "scheduled_span_nanoseconds",
    "arrival_window_nanoseconds",
    "completion_nanoseconds",
    "total_trial_nanoseconds",
    "observation_quantum_nanoseconds",
)

LIMITATIONS = (
    "The model is the tiny dynamic batch_add_f32 graph. Scheduler overhead may dominate.",
    "ONNX Runtime is the pinned CPU build. This collector does not run a GPU provider.",
    "Open-loop completion observation polls future readiness at 1 ms. "
    "p50, p95, and p99 include that observation delay and are not pure engine latency "
    "or precise service latency.",
    "Test hooks remain compiled into inferedge-bench-scheduler and stay disabled on this CLI.",
    "Scheduler metrics are enabled for every collected trial.",
    "Five counted runs do not establish statistical significance. "
    "No confidence interval is computed.",
    "The host may have competing load.",
    "Linux opt-in stress tests are not part of this collection.",
    "completion_timeout_ms bounds future observation only. "
    "This collector adds an external process wall-clock timeout because shutdown(kDrain) "
    "can block inside an InferenceEngine::run that has already started.",
    "A C++ trial can be runner-valid while the outstanding cap suppresses fewer than "
    "half of the schedule. Primary publication requires a 99 percent submit fraction.",
)

PROTOCOL_2_LIMITATIONS = (
    "The model is the tiny dynamic batch_add_f32 graph. Scheduler overhead may dominate.",
    "ONNX Runtime is the pinned CPU build. This collector does not run a GPU provider.",
    "Open-loop completion observation polls future readiness at 1 ms. "
    "p50, p95, and p99 include that observation delay and are not pure engine latency "
    "or precise service latency.",
    "Test hooks remain compiled into inferedge-bench-scheduler and stay disabled on this CLI.",
    "Scheduler metrics are enabled for every collected trial.",
    "Analysis protocol 2 requires 100 percent generator delivery for a strict primary "
    "open-loop trial. Scheduler rejections are delivered submissions, not generator skips.",
    "The generator lag summary is submitted_lag_max_within_configured_tolerance. "
    "It is not a percentile arrival distribution and does not certify microsecond-level accuracy. "
    "Full generator delivery is not arrival-timing fidelity. "
    "The default lag tolerance is 1000 ms, so zero skips do not certify the target rate, "
    "zero jitter, or a calibrated arrival process.",
    "A confidence interval requires at least 10 complete paired blocks. "
    "Ten blocks are not a power guarantee. "
    "Twenty counted blocks are the preferred publication count. "
    "Fewer blocks stay descriptive and do not produce an interval.",
    "Only one predeclared contrast per profile can carry a confirmatory interval. "
    "Other cells stay descriptive. No multiplicity correction is applied.",
    "A paired-block bootstrap interval describes the mean paired difference under this protocol. "
    "It is not a claim of universal performance superiority.",
    "Cell order rotates inside each profile. Rotation does not remove host contention, "
    "thermal drift, or clock drift.",
    "Requests inside one trial are not independent experimental blocks.",
    "Completion drain is a measured outcome and may differ across batching cells.",
    "The host may have competing load.",
    "Linux opt-in stress tests are not part of this collection.",
    "completion_timeout_ms bounds future observation only. "
    "This collector adds an external process wall-clock timeout because shutdown(kDrain) "
    "can block inside an InferenceEngine::run that has already started.",
    "Declared rates are not calibrated capacity.",
)


class SupervisedResult(ProcessResult):
    def __init__(
        self,
        returncode: int,
        stdout: str = "",
        stderr: str = "",
        *,
        timed_out: bool = False,
    ) -> None:
        super().__init__(returncode, stdout, stderr)
        self.timed_out = timed_out


def require_commit_sha(commit: str) -> str:
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise CollectorError(
            "--expected-commit must be a 40-character lowercase hexadecimal Git SHA, "
            f"got {commit!r}"
        )
    return commit


def cell_by_id(cell_id: str) -> dict:
    try:
        return CELL_BY_ID[cell_id]
    except KeyError as exc:
        known = ", ".join(CELL_BY_ID)
        raise CollectorError(f"unknown cell {cell_id!r}; known cells are {known}") from exc


def profile_by_id(profile_id: str) -> dict:
    try:
        return PROFILE_BY_ID[profile_id]
    except KeyError as exc:
        known = ", ".join(PROFILE_BY_ID)
        raise CollectorError(f"unknown profile {profile_id!r}; known profiles are {known}") from exc


def parse_id_list(text: str, lookup) -> list[dict]:
    if text.strip() == "":
        raise CollectorError("an explicit id list is required")
    selected = []
    seen = set()
    for raw in text.split(","):
        item_id = raw.strip()
        if item_id == "":
            raise CollectorError(f"empty id in {text!r}")
        if item_id in seen:
            raise CollectorError(f"duplicate id {item_id}")
        seen.add(item_id)
        selected.append(lookup(item_id))
    return selected


def require_one_family(profiles: list[dict]) -> str:
    families = {profile["family"] for profile in profiles}
    if len(families) != 1:
        raise CollectorError(
            "open-loop and closed-loop profiles cannot be collected into one run set"
        )
    return families.pop()


def execution_pairs(cells: list[dict], profiles: list[dict]) -> list[tuple[dict, dict]]:
    return [(cell, profile) for cell in cells for profile in profiles]


def format_trial(cell_id: str, profile_id: str, run_number: int | None) -> str:
    if run_number is None:
        return f"pilot-{cell_id}-{profile_id}"
    return f"{cell_id}-{profile_id}-{run_number:02d}"


def require_counted_runs(counted_runs: int) -> int:
    if type(counted_runs) is not int or isinstance(counted_runs, bool):
        raise CollectorError("counted runs must be an integer")
    if counted_runs < 1 or counted_runs > MAX_COUNTED_RUNS:
        raise CollectorError(
            f"counted runs must be from 1 through {MAX_COUNTED_RUNS}, got {counted_runs}"
        )
    return counted_runs


def execution_plan(
    cells: list[dict],
    profiles: list[dict],
    counted_runs: int,
    protocol: int = 1,
) -> list[dict[str, object]]:
    """One uncounted pilot per pair, then rotated counted rounds.

    Protocol 1 rotates the full cell-by-profile list. Protocol 2 keeps each
    counted block inside one workload profile and rotates cells inside that block.
    """
    require_counted_runs(counted_runs)
    if protocol == 1:
        plan = _protocol1_plan(cells, profiles, counted_runs)
    elif protocol == 2:
        plan = _protocol2_plan(cells, profiles, counted_runs)
    else:
        raise CollectorError(f"unsupported analysis protocol {protocol}")
    identities = [str(trial["trial_id"]) for trial in plan]
    if len(identities) != len(set(identities)):
        raise CollectorError("execution plan contains a duplicate trial id")
    for index, trial in enumerate(plan):
        trial["execution_index"] = index
        trial["counted_run_target"] = counted_runs
        trial["configuration_id"] = trial["cell_id"]
    return plan


def _protocol1_plan(
    cells: list[dict],
    profiles: list[dict],
    counted_runs: int,
) -> list[dict[str, object]]:
    pairs = execution_pairs(cells, profiles)
    plan: list[dict[str, object]] = []
    for cell, profile in pairs:
        plan.append(_trial(cell, profile, None))
    width = len(pairs)
    for run_number in range(1, counted_runs + 1):
        offset = run_number - 1
        for slot in range(width):
            cell, profile = pairs[(slot + offset) % width]
            plan.append(_trial(cell, profile, run_number))
    return plan


def _protocol2_plan(
    cells: list[dict],
    profiles: list[dict],
    counted_runs: int,
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for profile in profiles:
        for cell in cells:
            plan.append(_trial(cell, profile, None))
    width = len(cells)
    for block_index in range(1, counted_runs + 1):
        offset = block_index - 1
        for profile in profiles:
            for position in range(width):
                cell = cells[(position + offset) % width]
                plan.append(
                    _trial(
                        cell,
                        profile,
                        block_index,
                        block_index=block_index,
                        position_in_block=position,
                    )
                )
    return plan


def _trial(
    cell: dict,
    profile: dict,
    run_number: int | None,
    block_index: int | None = None,
    position_in_block: int | None = None,
) -> dict[str, object]:
    return {
        "trial_id": format_trial(str(cell["id"]), str(profile["id"]), run_number),
        "cell_id": cell["id"],
        "profile_id": profile["id"],
        "pilot": run_number is None,
        "run_number": run_number,
        "block_index": block_index,
        "position_in_block": position_in_block,
    }


def trial_order(
    cells: list[dict],
    profiles: list[dict],
    counted_runs: int,
    protocol: int = 1,
) -> list[str]:
    return [
        str(trial["trial_id"])
        for trial in execution_plan(cells, profiles, counted_runs, protocol)
    ]


def resolve_contrast(
    cells: list[dict],
    baseline_cell: str | None,
    alternative_cell: str | None,
) -> dict[str, object]:
    """Infer a contrast only when exactly two cells were selected."""
    selected = [str(cell["id"]) for cell in cells]
    if baseline_cell is None and alternative_cell is None:
        if len(selected) == 2:
            return {
                "baseline_cell": selected[0],
                "alternative_cell": selected[1],
                "contrast_source": "inferred_two_cell_order",
            }
        return {
            "baseline_cell": None,
            "alternative_cell": None,
            "contrast_source": "descriptive_only",
        }
    if baseline_cell is None or alternative_cell is None:
        raise CollectorError("baseline_cell and alternative_cell must both be set")
    if baseline_cell == alternative_cell:
        raise CollectorError("baseline_cell and alternative_cell must differ")
    if baseline_cell not in selected or alternative_cell not in selected:
        raise CollectorError("the predeclared contrast must use selected cells")
    return {
        "baseline_cell": baseline_cell,
        "alternative_cell": alternative_cell,
        "contrast_source": "predeclared",
    }


def wall_timeout_seconds(measurement: dict) -> float:
    """External process bound. It is not the runner's completion_timeout_ms."""
    seconds = (
        (measurement["duration_ms"] + measurement["completion_timeout_ms"]) / 1000.0
        + 30.0
        + measurement["warmup_requests"] * 0.05
    )
    if not math.isfinite(seconds) or seconds <= 0:
        raise CollectorError("external wall-clock timeout is not a positive finite number")
    return seconds


def measurement_from_args(args: argparse.Namespace) -> dict:
    measurement = dict(DEFAULT_MEASUREMENT)
    for name in (
        "duration_ms",
        "max_offered",
        "max_outstanding",
        "completion_timeout_ms",
        "warmup_requests",
        "lag_tolerance_ms",
        "check_interval",
    ):
        value = getattr(args, name)
        if value is not None:
            measurement[name] = value
    if measurement["max_outstanding"] > measurement["max_offered"]:
        raise CollectorError("max_outstanding cannot exceed max_offered")
    for name, value in measurement.items():
        if type(value) is not int or value < 0:
            raise CollectorError(f"{name} must be a nonnegative integer")
    if measurement["worker_count"] < 1 or measurement["queue_capacity"] < 1:
        raise CollectorError("worker_count and queue_capacity must be positive")
    if measurement["duration_ms"] < 1 or measurement["completion_timeout_ms"] < 1:
        raise CollectorError("duration_ms and completion_timeout_ms must be positive")
    if measurement["max_offered"] < 1 or measurement["check_interval"] < 1:
        raise CollectorError("max_offered and check_interval must be positive")
    return measurement


def _rate_token(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        raise CollectorError(f"arrival rate {value!r} is not a positive finite number")
    if float(value).is_integer():
        return str(int(value))
    return format(value, ".6f").rstrip("0").rstrip(".")


def scheduler_command(
    binary: pathlib.Path,
    model: pathlib.Path,
    cell: dict,
    profile: dict,
    measurement: dict,
) -> list[str]:
    if binary.name != BINARY_NAME:
        raise CollectorError(f"refusing to launch {binary.name}; expected {BINARY_NAME}")
    command = [
        str(binary),
        "--model",
        str(model),
        "--workers",
        str(measurement["worker_count"]),
        "--queue-capacity",
        str(measurement["queue_capacity"]),
        "--workload",
        str(profile["workload"]),
        "--max-offered",
        str(measurement["max_offered"]),
        "--max-outstanding",
        str(measurement["max_outstanding"]),
        "--duration-ms",
        str(measurement["duration_ms"]),
        "--completion-timeout-ms",
        str(measurement["completion_timeout_ms"]),
        "--warmup",
        str(measurement["warmup_requests"]),
        "--lag-tolerance-ms",
        str(measurement["lag_tolerance_ms"]),
        "--check-interval",
        str(measurement["check_interval"]),
        "--max-batch-size",
        str(cell["max_batch_size"]),
        "--max-batch-delay-ms",
        str(cell["max_batch_delay_ms"]),
        "--format",
        "json",
    ]
    if cell["enable_batching"]:
        command.append("--enable-batching")
    workload = profile["workload"]
    if workload == "open_loop_constant":
        command.extend(["--target-rps", _rate_token(float(profile["target_rps"]))])
    elif workload == "open_loop_burst":
        command.extend(
            [
                "--burst-high-rps",
                _rate_token(float(profile["burst_high_rps"])),
                "--burst-low-rps",
                _rate_token(float(profile["burst_low_rps"])),
                "--burst-high-ms",
                str(profile["burst_high_ms"]),
                "--burst-low-ms",
                str(profile["burst_low_ms"]),
            ]
        )
    elif workload == "closed_loop":
        command.extend(["--producer-count", str(profile["producer_count"])])
    else:
        raise CollectorError(f"unsupported workload {workload}")
    return command


def terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def supervise_process(
    args: list[str],
    *,
    cwd: pathlib.Path | None,
    env: dict[str, str] | None,
    timeout_seconds: float,
) -> SupervisedResult:
    """Run one argument array. A wall-clock timeout reaps the process group."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise CollectorError("process timeout must be positive and finite")
    try:
        process = subprocess.Popen(
            args,
            cwd=None if cwd is None else str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        return SupervisedResult(127, "", f"failed to execute {args[0]}: {exc}")
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        stdout = ""
        stderr = ""
        if process.stdout is not None:
            stdout = process.stdout.read() or ""
        if process.stderr is not None:
            stderr = process.stderr.read() or ""
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        return SupervisedResult(
            process.returncode if process.returncode is not None else -1,
            stdout,
            stderr,
            timed_out=True,
        )
    return SupervisedResult(process.returncode if process.returncode is not None else -1, stdout, stderr)


def version_text_problems(stdout: str, stderr: str, returncode: int) -> list[str]:
    expected = f"{TOOL_NAME} {TOOL_VERSION}\n"
    problems = []
    if returncode != 0:
        problems.append(f"{TOOL_NAME} --version exited {returncode}")
    if stderr:
        problems.append(f"{TOOL_NAME} --version wrote stderr")
    if stdout != expected:
        problems.append(f"{TOOL_NAME} --version returned {stdout!r}, expected {expected!r}")
    return problems


def git_blob_sha256(repo_root: pathlib.Path, commit: str, relative: str) -> str:
    if relative != MODEL_RELATIVE:
        raise CollectorError(f"refusing to read an unpinned Git path: {relative}")
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "blob", f"{commit}:{relative}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CollectorError(
            f"git blob {commit}:{relative} could not be read: {detail or completed.returncode}"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def verify_model(repo_root: pathlib.Path, commit: str, blob_sha256) -> pathlib.Path:
    path = repo_root / MODEL_RELATIVE
    if not path.is_file():
        raise CollectorError(f"pinned model is missing: {path}")
    actual = sha256_file(path)
    if actual != MODEL_SHA256:
        raise CollectorError(
            f"SHA-256 mismatch for {MODEL_RELATIVE}: expected {MODEL_SHA256}, actual {actual}"
        )
    committed = blob_sha256(repo_root, commit, MODEL_RELATIVE)
    if committed != MODEL_SHA256:
        raise CollectorError(
            "committed SHA-256 mismatch for "
            f"{commit}:{MODEL_RELATIVE}: expected {MODEL_SHA256}, actual {committed}"
        )
    return path


def locate_build_dir(binary: pathlib.Path) -> pathlib.Path:
    cache = binary.parent / "CMakeCache.txt"
    if not cache.is_file():
        raise CollectorError(
            "CMakeCache.txt was not found beside the benchmark binary at "
            f"{cache}. The binary must be the inferedge-bench-scheduler produced in its "
            "CMake build directory, not a copy"
        )
    return binary.parent


def _nonnegative_int(value: object, label: str, problems: list[str]) -> int | None:
    number = json_integer(value, label, problems)
    if number is None:
        return None
    if number < 0:
        problems.append(f"{label} is negative")
        return None
    if number > UINT64_MAX:
        problems.append(f"{label} exceeds uint64")
        return None
    return number


def _object(value: object, label: str, problems: list[str]) -> dict | None:
    if not isinstance(value, dict):
        problems.append(f"{label} is not an object")
        return None
    return value


def _bool(value: object, label: str, problems: list[str]) -> bool | None:
    if type(value) is not bool:
        problems.append(f"{label} is not a JSON boolean")
        return None
    return value


def analysis_protocol_of(expected: dict) -> int:
    value = expected.get("analysis_protocol_version", 1)
    if value not in {1, 2}:
        raise CollectorError(f"analysis_protocol_version {value!r} is not 1 or 2")
    return int(value)


def strict_arrival_window_problems(
    configured: int,
    span: int,
    arrival: int,
    tolerance_ns: int,
) -> list[str]:
    """Full-duration bounds. Completion drain is intentionally not compared."""
    problems = []
    if arrival < span:
        problems.append("arrival window is shorter than the scheduled span")
    if tolerance_ns < 0 or configured < 0:
        problems.append("configured duration or lag tolerance is negative")
        return problems
    if configured > UINT64_MAX - tolerance_ns:
        problems.append("configured duration plus lag tolerance overflows uint64")
        return problems
    if arrival > configured + tolerance_ns:
        problems.append("arrival window exceeds configured duration plus lag tolerance")
    return problems


def _fraction_status(document: dict) -> dict[str, float | None]:
    scheduled = document.get("scheduled_opportunities")
    attempts = document.get("submit_attempts")
    accepted = document.get("accepted")
    delivery = None
    rejection = None
    acceptance = None
    completion = None
    succeeded = document.get("succeeded")
    if type(scheduled) is int and scheduled > 0 and type(attempts) is int:
        delivery = attempts / scheduled
    rejections = document.get("rejections")
    if type(attempts) is int and attempts > 0 and isinstance(rejections, dict):
        if all(type(rejections.get(key)) is int for key in REJECTION_KEYS):
            rejected = rejections["full"] + rejections["closed"] + rejections["not_running"]
            rejection = rejected / attempts
            if type(accepted) is int:
                acceptance = accepted / attempts
    if type(accepted) is int and accepted > 0 and type(succeeded) is int:
        completion = succeeded / accepted
    return {
        "delivery_fraction": delivery,
        "submission_fraction": delivery,
        "scheduler_rejection_fraction": rejection,
        "scheduler_acceptance_fraction": acceptance,
        "successful_completion_fraction": completion,
    }


def _histogram_saturated(document: dict) -> bool:
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        return False
    if metrics.get("histogram_saturated") is True:
        return True
    histogram = metrics.get("batch_size_histogram")
    return isinstance(histogram, list) and any(value == UINT64_MAX for value in histogram)


def _strict_lag_problems(document: dict, expected: dict) -> list[str]:
    lag = document["generator_lag"]
    problems = []
    count = lag["count"]
    minimum = lag["minimum_nanoseconds"]
    maximum = lag["maximum_nanoseconds"]
    mean = lag["mean_nanoseconds"]
    if count != document["submit_attempts"]:
        problems.append("generator lag count does not equal submit_attempts")
    if isinstance(mean, bool) or type(mean) not in {int, float}:
        problems.append("generator lag mean is not a number")
    elif not math.isfinite(float(mean)):
        problems.append("generator lag mean is not finite")
    elif type(minimum) is int and type(maximum) is int and not minimum <= float(mean) <= maximum:
        problems.append("generator lag mean is outside the recorded minimum and maximum")
    tolerance = expected["lag_tolerance_ms"] * 1_000_000
    if type(maximum) is int and maximum > tolerance:
        problems.append("recorded generator lag exceeds the configured tolerance")
    return problems


def _protocol2_delivery(document: dict, expected: dict) -> tuple[str, list[str], str | None]:
    """Classify one collector-valid trial. Precedence is fixed top to bottom."""
    saturated = _histogram_saturated(document)
    if document["workload"] == "closed_loop":
        if saturated:
            return "CLOSED_LOOP_INVALID", ["histogram counters are saturated"], None
        return "CLOSED_LOOP_VALID", [], None
    if saturated:
        return "INVALID", ["histogram counters are saturated"], None
    scheduled = document["scheduled_opportunities"]
    attempts = document["submit_attempts"]
    if scheduled <= 0:
        return "INVALID", ["open-loop schedule is empty"], None
    if (
        attempts != scheduled
        or document["generator_skips_outstanding"] != 0
        or document["generator_skips_lag"] != 0
    ):
        return "EXPLORATORY_OVERLOAD", ["generator delivery is below 100 percent"], None
    lag_problems = _strict_lag_problems(document, expected)
    if lag_problems:
        return "INVALID", lag_problems, None
    reason = document["arrival_stop_reason"]
    if reason == "max_offered":
        return (
            "BOUNDED_COMPLETE",
            ["bounded opportunity horizon is not strict primary"],
            TIMING_QUALITY,
        )
    if reason != "configured_duration":
        return "INVALID", ["arrival stop is not a full-duration completion"], None
    window_problems = strict_arrival_window_problems(
        document["configured_duration_nanoseconds"],
        document["scheduled_span_nanoseconds"],
        document["arrival_window_nanoseconds"],
        expected["lag_tolerance_ms"] * 1_000_000,
    )
    if window_problems:
        return "INVALID", window_problems, None
    return "STRICT_PRIMARY", [], TIMING_QUALITY


def classify_report(
    text: str,
    *,
    expected: dict,
) -> tuple[dict | None, dict[str, object]]:
    """Split runner validity, collector validity, publication, and delivery class.

    Missing analysis_protocol_version keeps the protocol-1 contract. Protocol 2
    does not reinterpret that evidence.
    """
    document, problems = parse_single_document(text)
    status: dict[str, object] = {
        "runner_valid": False,
        "collector_valid": False,
        "publication_eligible": False,
        "steady_state_eligible": False,
        "delivery_fraction": None,
        "scheduler_rejection_fraction": None,
        "scheduler_acceptance_fraction": None,
        "window_class": None,
        "delivery_class": None,
        "timing_quality": None,
        "timing_quality_note": None,
        "analysis_protocol_version": 1,
        "problems": problems,
        "publication_problems": [],
    }
    if document is None:
        status["delivery_class"] = "INVALID"
        return None, status
    protocol = analysis_protocol_of(expected)
    status["analysis_protocol_version"] = protocol
    _check_document(document, expected, problems)
    publication: list[str] = []
    collector_problems = [item for item in problems if item]
    if document.get("valid") is not True:
        collector_problems.append("runner valid flag is not true")
    status["runner_valid"] = document.get("valid") is True
    status["problems"] = collector_problems
    if collector_problems:
        if protocol == 2:
            status["delivery_class"] = "INVALID"
        return document, status
    status["collector_valid"] = True
    status.update(_fraction_status(document))
    reason = document.get("arrival_stop_reason")
    if reason == "max_offered":
        status["window_class"] = "bounded_max_offered"
    elif reason == "configured_duration":
        status["window_class"] = "configured_duration"
    else:
        status["window_class"] = "generator_lag"
    if protocol == 1:
        _publication_rules(document, expected, publication)
        eligible = not publication and expected["comparison_class"] == "primary"
        if expected["comparison_class"] != "primary":
            publication.append("comparison_class is not primary")
        status["publication_problems"] = publication
        status["publication_eligible"] = eligible
        status["steady_state_eligible"] = eligible and reason == "configured_duration"
        return document, status
    delivery_class, delivery_problems, timing = _protocol2_delivery(document, expected)
    status["delivery_class"] = delivery_class
    status["timing_quality"] = timing
    if timing is not None:
        status["timing_quality_note"] = TIMING_QUALITY_NOTE
    publication.extend(delivery_problems)
    if expected["comparison_class"] != "primary":
        publication.append("comparison_class is not primary")
    eligible = expected["comparison_class"] == "primary" and delivery_class == "STRICT_PRIMARY"
    status["publication_problems"] = publication
    status["publication_eligible"] = eligible
    status["steady_state_eligible"] = eligible
    return document, status


def _check_document(document: dict, expected: dict, problems: list[str]) -> None:
    require_exact_keys(document, TOP_LEVEL_KEYS, "report", problems)
    if problems:
        return
    schema = json_integer(document["schema_version"], "schema_version", problems)
    if schema is not None and schema != SCHEMA_VERSION:
        problems.append(f"schema_version is {schema}, expected {SCHEMA_VERSION}")
    if document["tool"] != TOOL_NAME:
        problems.append(f"tool is {document['tool']!r}, expected {TOOL_NAME!r}")
    if document["tool_version"] != TOOL_VERSION:
        problems.append(f"tool_version is {document['tool_version']!r}")
    if type(document["valid"]) is not bool:
        problems.append("valid is not a JSON boolean")
    if not isinstance(document["invalid_reasons"], list) or any(
        type(item) is not str for item in document["invalid_reasons"]
    ):
        problems.append("invalid_reasons is not a list of strings")
    if document["model"] != expected["model_path"]:
        problems.append("model path does not match the launched model")
    if document["workload"] != expected["workload"]:
        problems.append(
            f"workload is {document['workload']!r}, expected {expected['workload']!r}"
        )
    if document["e2e_latency_definition"] != E2E_DEFINITION:
        problems.append("e2e_latency_definition does not match the runner contract")
    _check_observation(document, expected["workload"], problems)
    if document["completion_timeout_bound"] != COMPLETION_TIMEOUT_BOUND:
        problems.append("completion_timeout_bound is not future_observation_only")
    if type(document["completion_timeout_note"]) is not str:
        problems.append("completion_timeout_note is not a string")
    if type(document["outstanding_cap_note"]) is not str:
        problems.append("outstanding_cap_note is not a string")
    if type(document["exceptional_promise_backstop"]) is not str:
        problems.append("exceptional_promise_backstop is not a string")
    if document["arrival_stop_reason"] not in {
        "configured_duration",
        "max_offered",
        "generator_lag",
    }:
        problems.append(f"arrival_stop_reason is {document['arrival_stop_reason']!r}")
    for label in DURATION_FIELDS:
        _nonnegative_int(document[label], label, problems)
    for label in COUNT_FIELDS:
        _nonnegative_int(document[label], label, problems)
    _check_scheduler(document, expected, problems)
    _check_generator(document, expected, problems)
    _check_rejections(document, problems)
    _check_lag(document, expected, problems)
    _check_latency(document, problems)
    _check_throughput(document, problems)
    _check_metrics(document, problems)
    _check_errors(document, problems)
    _check_phases(document, problems)
    _check_accounting(document, problems)
    _check_windows(document, expected, problems)


def _check_observation(document: dict, workload: str, problems: list[str]) -> None:
    observation = document["completion_observation"]
    quantum = document["observation_quantum_nanoseconds"]
    if workload == "closed_loop":
        if observation != OBSERVATION_CLOSED:
            problems.append(f"completion_observation is {observation!r}, expected closed-loop wait")
        if type(quantum) is int and quantum != 0:
            problems.append("closed-loop observation quantum is not 0")
        return
    if observation != OBSERVATION_OPEN:
        problems.append(f"completion_observation is {observation!r}, expected open_loop_poll")
    if type(quantum) is int and quantum != OPEN_LOOP_QUANTUM_NS:
        problems.append("open-loop observation quantum is not 1000000 nanoseconds")


def _check_scheduler(document: dict, expected: dict, problems: list[str]) -> None:
    scheduler = _object(document["scheduler"], "scheduler", problems)
    if scheduler is None:
        return
    require_exact_keys(scheduler, SCHEDULER_KEYS, "scheduler", problems)
    if any(key not in scheduler for key in SCHEDULER_KEYS):
        return
    for label, expected_value in (
        ("worker_count", expected["worker_count"]),
        ("queue_capacity", expected["queue_capacity"]),
        ("max_batch_size", expected["max_batch_size"]),
        ("max_batch_delay_ms", expected["max_batch_delay_ms"]),
    ):
        number = _nonnegative_int(scheduler[label], f"scheduler.{label}", problems)
        if number is not None and number != expected_value:
            problems.append(f"scheduler.{label} is {number}, expected {expected_value}")
    batching = _bool(scheduler["enable_batching"], "scheduler.enable_batching", problems)
    if batching is not None and batching != expected["enable_batching"]:
        problems.append("scheduler.enable_batching does not match the cell")
    metrics = _bool(scheduler["enable_metrics"], "scheduler.enable_metrics", problems)
    if metrics is not None and metrics is not True:
        problems.append("scheduler metrics are disabled")


def _check_generator(document: dict, expected: dict, problems: list[str]) -> None:
    generator = _object(document["generator"], "generator", problems)
    if generator is None:
        return
    require_exact_keys(generator, GENERATOR_KEYS, "generator", problems)
    if any(key not in generator for key in GENERATOR_KEYS):
        return
    if generator["lag_definition"] != "actual_submit_start_minus_scheduled_arrival":
        problems.append("generator lag definition is unexpected")
    if generator["lag_policy"] != "stop_and_record":
        problems.append("generator lag policy is unexpected")
    if generator["outstanding_policy"] != "skip_opportunity":
        problems.append("generator outstanding policy is unexpected")
    warmup = _nonnegative_int(generator["warmup_requests"], "generator.warmup_requests", problems)
    if warmup is not None and warmup != expected["warmup_requests"]:
        problems.append("warmup_requests does not match the launch")
    separate = _bool(
        generator["warmup_separate_scheduler"], "generator.warmup_separate_scheduler", problems
    )
    if separate is not None and separate is not True:
        problems.append("warmup did not use a separate scheduler")
    interval = _nonnegative_int(generator["check_interval"], "generator.check_interval", problems)
    if interval is not None and interval != expected["check_interval"]:
        problems.append("check_interval does not match the launch")


def _check_rejections(document: dict, problems: list[str]) -> None:
    rejections = _object(document["rejections"], "rejections", problems)
    if rejections is None:
        return
    require_exact_keys(rejections, REJECTION_KEYS, "rejections", problems)
    for key in REJECTION_KEYS:
        if key in rejections:
            _nonnegative_int(rejections[key], f"rejections.{key}", problems)


def _check_lag(document: dict, expected: dict, problems: list[str]) -> None:
    lag = _object(document["generator_lag"], "generator_lag", problems)
    if lag is None:
        return
    require_exact_keys(lag, LAG_KEYS, "generator_lag", problems)
    if any(key not in lag for key in LAG_KEYS):
        return
    count = _nonnegative_int(lag["count"], "generator_lag.count", problems)
    minimum = json_integer(lag["minimum_nanoseconds"], "generator_lag.minimum_nanoseconds", problems)
    maximum = json_integer(lag["maximum_nanoseconds"], "generator_lag.maximum_nanoseconds", problems)
    mean = json_number(lag["mean_nanoseconds"], "generator_lag.mean_nanoseconds", problems)
    if count == 0:
        return
    if minimum is not None and maximum is not None and minimum > maximum:
        problems.append("generator lag minimum exceeds maximum")
    del expected


def _check_latency(document: dict, problems: list[str]) -> None:
    latency = _object(document["successful_e2e_latency"], "successful_e2e_latency", problems)
    if latency is None:
        return
    require_exact_keys(latency, LATENCY_KEYS, "successful_e2e_latency", problems)
    if any(key not in latency for key in LATENCY_KEYS):
        return
    sample_count = _nonnegative_int(latency["sample_count"], "latency.sample_count", problems)
    for label in ("minimum_nanoseconds", "maximum_nanoseconds"):
        number = json_integer(latency[label], f"latency.{label}", problems)
        if number is not None and number < 0:
            problems.append(f"latency.{label} is negative")
    for label in ("mean_microseconds", "p50_microseconds", "p95_microseconds", "p99_microseconds"):
        number = json_number(latency[label], f"latency.{label}", problems)
        if number is not None and number < 0:
            problems.append(f"latency.{label} is negative")
    reported = _bool(latency["percentiles_reported"], "latency.percentiles_reported", problems)
    _bool(latency["precise_tail"], "latency.precise_tail", problems)
    succeeded = document.get("succeeded")
    if type(sample_count) is int and type(succeeded) is int and sample_count != succeeded:
        problems.append("successful latency sample_count does not equal succeeded")
    if reported is False and type(sample_count) is int and sample_count > 0:
        problems.append("successful samples exist but percentiles_reported is false")


def _check_throughput(document: dict, problems: list[str]) -> None:
    throughput = _object(document["throughput"], "throughput", problems)
    if throughput is None:
        return
    require_exact_keys(throughput, THROUGHPUT_KEYS, "throughput", problems)
    if any(key not in throughput for key in THROUGHPUT_KEYS):
        return
    for label in THROUGHPUT_KEYS:
        if label.endswith("_note"):
            continue
        number = json_number(throughput[label], f"throughput.{label}", problems)
        if number is not None and number < 0:
            problems.append(f"throughput.{label} is negative")
    if throughput["successful_attempts_per_arrival_window_second_note"] != ARRIVAL_WINDOW_NOTE:
        problems.append("arrival-window throughput note was changed")
    if throughput["successful_per_total_trial_second_note"] != TOTAL_TRIAL_NOTE:
        problems.append("total-trial throughput note was changed")
    arrival = throughput["arrival_window_seconds"]
    total = throughput["total_trial_seconds"]
    if type(arrival) in {int, float} and not isinstance(arrival, bool) and float(arrival) <= 0:
        problems.append("arrival_window_seconds is not positive")
    if type(total) in {int, float} and not isinstance(total, bool) and float(total) <= 0:
        problems.append("total_trial_seconds is not positive")


def _check_metrics(document: dict, problems: list[str]) -> None:
    metrics = _object(document["metrics"], "metrics", problems)
    if metrics is None:
        return
    require_exact_keys(metrics, METRICS_KEYS, "metrics", problems)
    if any(key not in metrics for key in METRICS_KEYS):
        return
    enabled = _bool(metrics["enabled"], "metrics.enabled", problems)
    finalized = _bool(metrics["finalized"], "metrics.finalized", problems)
    saturated = _bool(metrics["histogram_saturated"], "metrics.histogram_saturated", problems)
    runs = _nonnegative_int(metrics["engine_run_count"], "metrics.engine_run_count", problems)
    entered = _nonnegative_int(
        metrics["requests_entered_engine_runs"],
        "metrics.requests_entered_engine_runs",
        problems,
    )
    for label in (
        "total_assembly_nanoseconds",
        "total_engine_run_nanoseconds",
        "total_split_nanoseconds",
    ):
        _nonnegative_int(metrics[label], f"metrics.{label}", problems)
    histogram = metrics["batch_size_histogram"]
    if not isinstance(histogram, list) or len(histogram) != HISTOGRAM_BINS:
        problems.append(f"batch_size_histogram does not have {HISTOGRAM_BINS} bins")
        return
    bins: list[int] = []
    saturated_count = saturated is True
    for index, value in enumerate(histogram):
        number = _nonnegative_int(value, f"histogram[{index}]", problems)
        if number is None:
            return
        if number == UINT64_MAX:
            saturated_count = True
        bins.append(number)
    if bins[0] != 0:
        problems.append("histogram[0] is not 0")
    if enabled is not True or finalized is not True:
        problems.append("metrics are not enabled and finalized")
    if saturated_count:
        return
    if runs is None or entered is None:
        return
    if sum(bins[1:]) != runs:
        problems.append("histogram bin sum does not equal engine_run_count")
    weighted = sum(index * bins[index] for index in range(1, HISTOGRAM_BINS))
    if weighted != entered:
        problems.append("weighted histogram does not equal requests_entered_engine_runs")


def _check_errors(document: dict, problems: list[str]) -> None:
    errors = document["errors"]
    error_count = document["error_count"]
    if not isinstance(errors, list):
        problems.append("errors is not a list")
        return
    if len(errors) > 8:
        problems.append("errors list is unbounded above 8 records")
    if type(error_count) is int and len(errors) > error_count:
        problems.append("errors list is longer than error_count")
    for index, item in enumerate(errors):
        if not isinstance(item, dict):
            problems.append(f"errors[{index}] is not an object")
            continue
        require_exact_keys(item, ("attempt_id", "kind", "message"), f"errors[{index}]", problems)
    if document.get("valid") is True and type(error_count) is int and error_count != 0:
        problems.append("a valid trial records errors")
    if any(name != 0 for name in ("queue_depth", "executing", "live_workers") if type(document.get(name)) is int and document[name] != 0):
        problems.append("scheduler is not quiescent")
    if document.get("correctness_mismatches") not in {0, None} and type(
        document.get("correctness_mismatches")
    ) is int and document["correctness_mismatches"] != 0:
        problems.append("correctness mismatches are present")


def _check_phases(document: dict, problems: list[str]) -> None:
    phases = document["burst_phases"]
    if not isinstance(phases, list):
        problems.append("burst_phases is not a list")
        return
    total = 0
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            problems.append(f"burst_phases[{index}] is not an object")
            continue
        require_exact_keys(
            phase,
            ("index", "requests_per_second", "duration_nanoseconds", "scheduled_opportunities"),
            f"burst_phases[{index}]",
            problems,
        )
        if "scheduled_opportunities" in phase:
            count = _nonnegative_int(
                phase["scheduled_opportunities"],
                f"burst_phases[{index}].scheduled_opportunities",
                problems,
            )
            if count is not None:
                total += count
        if "requests_per_second" in phase:
            json_number(
                phase["requests_per_second"],
                f"burst_phases[{index}].requests_per_second",
                problems,
            )
        if "duration_nanoseconds" in phase:
            _nonnegative_int(
                phase["duration_nanoseconds"],
                f"burst_phases[{index}].duration_nanoseconds",
                problems,
            )
    scheduled = document.get("scheduled_opportunities")
    if document.get("workload") != "closed_loop" and type(scheduled) is int and total != scheduled:
        problems.append("burst phase scheduled counts do not equal scheduled_opportunities")
    if document.get("workload") == "closed_loop" and phases:
        problems.append("closed-loop report contains open-loop burst phases")


def _check_accounting(document: dict, problems: list[str]) -> None:
    needed = (
        "scheduled_opportunities",
        "submit_attempts",
        "generator_skips_outstanding",
        "generator_skips_lag",
        "accepted",
        "succeeded",
        "failed_inference",
        "failed_worker",
        "failed_scheduler",
        "canceled",
        "timeouts",
    )
    if any(type(document.get(name)) is not int for name in needed):
        return
    rejections = document.get("rejections")
    if not isinstance(rejections, dict) or any(
        type(rejections.get(name)) is not int for name in REJECTION_KEYS
    ):
        return
    scheduled = document["scheduled_opportunities"]
    attempts = document["submit_attempts"]
    skips = document["generator_skips_outstanding"] + document["generator_skips_lag"]
    if scheduled != attempts + skips:
        problems.append("scheduled opportunities do not equal attempts plus generator skips")
    rejection_total = rejections["full"] + rejections["closed"] + rejections["not_running"]
    if attempts != document["accepted"] + rejection_total:
        problems.append("attempts do not equal accepted plus rejections")
    terminals = (
        document["succeeded"]
        + document["failed_inference"]
        + document["failed_worker"]
        + document["failed_scheduler"]
        + document["canceled"]
    )
    if document["timeouts"] != 0:
        problems.append("timeouts are not zero")
    elif document["accepted"] != terminals:
        problems.append("accepted requests do not equal terminal statuses")
    if document["generator_skips_lag"] != 0 and document.get("valid") is True:
        problems.append("generator lag stop is marked runner-valid")


def _check_windows(document: dict, expected: dict, problems: list[str]) -> None:
    configured = document.get("configured_duration_nanoseconds")
    if type(configured) is int and configured != expected["duration_ms"] * 1_000_000:
        problems.append("configured duration does not match the launch")
    arrival = document.get("arrival_window_nanoseconds")
    completion = document.get("completion_nanoseconds")
    total = document.get("total_trial_nanoseconds")
    if type(arrival) is int and type(completion) is int and type(total) is int:
        if arrival <= 0 or total <= 0:
            problems.append("measurement window is not positive")
        elif total != arrival + completion:
            problems.append("total trial duration is not arrival window plus completion")
    if document.get("workload") == "closed_loop" and document.get("scheduled_span_nanoseconds") not in {
        0,
        None,
    }:
        if document.get("scheduled_span_nanoseconds") != 0:
            problems.append("closed-loop scheduled span is not 0")


def _publication_rules(document: dict, expected: dict, publication: list[str]) -> None:
    metrics = document["metrics"]
    if metrics.get("histogram_saturated") is True or any(
        value == UINT64_MAX for value in metrics["batch_size_histogram"]
    ):
        publication.append("histogram counters are saturated")
    if document["generator_skips_lag"] != 0 or document["arrival_stop_reason"] == "generator_lag":
        publication.append("generator lag stopped the schedule")
    lag = document["generator_lag"]
    tolerance = expected["lag_tolerance_ms"] * 1_000_000
    if lag["count"] and lag["maximum_nanoseconds"] > tolerance:
        publication.append("recorded generator lag exceeds the configured tolerance")
    if document["workload"] != "closed_loop":
        scheduled = document["scheduled_opportunities"]
        if scheduled <= 0:
            publication.append("open-loop schedule is empty")
        else:
            fraction = document["submit_attempts"] / scheduled
            if fraction < MINIMUM_SUBMIT_FRACTION:
                publication.append(
                    "submit fraction "
                    f"{fraction:.6f} is below the publication threshold {MINIMUM_SUBMIT_FRACTION}"
                )


def write_exclusive_text(path: pathlib.Path, text: str) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise CollectorError(f"refusing to overwrite existing evidence: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_json(path: pathlib.Path, value: dict) -> None:
    write_exclusive_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def replace_json(path: pathlib.Path, value: dict) -> None:
    """Write the replacement first. The previous file stays if that write fails."""
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def preflight(
    *,
    repo_root: pathlib.Path,
    binary: pathlib.Path,
    expected_commit: str,
    cells: list[dict],
    profiles: list[dict],
    measurement: dict,
    counted_runs: int,
    delay_seconds: float,
    comparison_class: str,
    analysis_protocol_version: int = ANALYSIS_PROTOCOL_VERSION,
    baseline_cell: str | None = None,
    alternative_cell: str | None = None,
    runner=query_text,
    blob_sha256=git_blob_sha256,
) -> dict:
    commit = require_commit_sha(expected_commit)
    if comparison_class not in {"primary", "exploratory"}:
        raise CollectorError("comparison_class must be primary or exploratory")
    if analysis_protocol_version != ANALYSIS_PROTOCOL_VERSION:
        raise CollectorError(
            "new collections use analysis protocol 2; protocol 1 is reserved for existing evidence"
        )
    require_counted_runs(counted_runs)
    contrast = resolve_contrast(cells, baseline_cell, alternative_cell)
    family = require_one_family(profiles)
    if not binary.is_file():
        raise CollectorError(f"benchmark binary does not exist: {binary}")
    if not os.access(binary, os.X_OK):
        raise CollectorError(f"benchmark binary is not executable: {binary}")
    if binary.resolve().name != BINARY_NAME:
        raise CollectorError(
            f"benchmark binary must be named {BINARY_NAME}, got {binary.resolve().name}"
        )
    build_dir = locate_build_dir(binary)
    cache = parse_cmake_cache((build_dir / "CMakeCache.txt").read_text(encoding="utf-8"))
    problems = assess_build_cache(cache)
    problems.extend(release_optimization_problems(cache))
    if problems:
        raise CollectorError("; ".join(problems))
    symbols = query_text(runner, symbol_tool_args(binary))
    if symbols.returncode != 0:
        raise CollectorError(
            "could not inspect the benchmark binary for sanitizer symbols: "
            f"{symbols.stderr.strip() or symbols.stdout.strip()}"
        )
    symbol_problems = sanitizer_symbol_problems(symbols.stdout + symbols.stderr)
    if symbol_problems:
        raise CollectorError("; ".join(symbol_problems))
    runtime_root = locate_pinned_runtime(cache, build_dir)
    linked = query_text(runner, link_tool_args(binary))
    if linked.returncode != 0 and not linked.stdout.strip():
        raise CollectorError(
            "could not identify the ONNX Runtime library linked into the benchmark binary: "
            f"{linked.stderr.strip() or linked.stdout.strip()}"
        )
    load_command_output = ""
    if platform.system() == "Darwin":
        load_commands = query_text(runner, ["otool", "-l", str(binary)])
        if load_commands.returncode != 0:
            raise CollectorError(
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
        raise CollectorError(
            f"ONNX Runtime VERSION_NUMBER is {onnx_version!r}, expected {ONNXRUNTIME_VERSION!r}"
        )
    version = query_text(runner, [str(binary), "--version"])
    version_problems = version_text_problems(version.stdout, version.stderr, version.returncode)
    if version_problems:
        raise CollectorError("; ".join(version_problems))
    model = verify_model(repo_root, commit, blob_sha256)
    head = query_text(runner, ["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    status = query_text(runner, ["git", "-C", str(repo_root), "status", "--porcelain"])
    if head.returncode != 0 or status.returncode != 0:
        raise CollectorError(
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
        raise CollectorError("; ".join(git_problems))
    host = hardware_metadata(runner)
    tools = compiler_and_cmake(runner, cache["CMAKE_CXX_COMPILER"])
    plan = execution_plan(cells, profiles, counted_runs, analysis_protocol_version)
    pilots = [str(trial["trial_id"]) for trial in plan if trial["pilot"]]
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
            "model_path": str(model),
            "model_sha256": MODEL_SHA256,
            "model_relative": MODEL_RELATIVE,
        },
        "source_defined_policy": {
            "backend": BACKEND,
            "graph_optimizations": GRAPH_OPTIMIZATIONS,
            "onnxruntime_version_pin": ONNXRUNTIME_VERSION,
            "schema_version": SCHEMA_VERSION,
            "analysis_protocol_version": analysis_protocol_version,
            "publication_policy": PUBLICATION_POLICY,
            "protocol_1_publication_policy": PROTOCOL1_PUBLICATION_POLICY,
            "protocol_1_minimum_submit_fraction": MINIMUM_SUBMIT_FRACTION,
            "minimum_submit_fraction": 1.0,
            "minimum_submit_fraction_scope": "protocol 2 strict open-loop generator delivery",
            "timing_quality": TIMING_QUALITY,
            "timing_quality_note": TIMING_QUALITY_NOTE,
            "calibration_status": "not_calibrated",
            "allowed_untracked_roots": list(ALLOWED_UNTRACKED),
            "limitations": list(PROTOCOL_2_LIMITATIONS),
            "rates_are_calibrated_capacity": False,
        },
        "collection_protocol": {
            "analysis_protocol_version": analysis_protocol_version,
            "publication_policy": PUBLICATION_POLICY,
            "comparison_class": comparison_class,
            "workload_family": family,
            "cells": [cell["id"] for cell in cells],
            "profiles": [profile["id"] for profile in profiles],
            "cell_definitions": cells,
            "profile_definitions": profiles,
            "measurement": measurement,
            "counted_runs_per_cell": counted_runs,
            "counted_run_target": counted_runs,
            "preferred_counted_runs": PREFERRED_COUNTED_RUNS,
            "minimum_interval_blocks": MINIMUM_INTERVAL_BLOCKS,
            "pilot_count": len(pilots),
            "counted_trial_count": counted_runs * len(cells) * len(profiles),
            "pilots": pilots,
            "trial_order": [str(trial["trial_id"]) for trial in plan],
            "block_protocol": BLOCK_PROTOCOL,
            "baseline_cell": contrast["baseline_cell"],
            "alternative_cell": contrast["alternative_cell"],
            "contrast_source": contrast["contrast_source"],
            "statistical_method": STATISTICAL_METHOD,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": CONFIDENCE_LEVEL,
            "inter_trial_delay_seconds": delay_seconds,
            "environment_variable": "ORT_DISABLE_TELEMETRY=1",
            "external_wall_timeout_seconds": wall_timeout_seconds(measurement),
            "completion_timeout_is_process_deadline": False,
        },
        "paths": {"model": model},
        "commit": actual_commit,
        "plan": plan,
    }


def _expected_launch(checked: dict, cell: dict, profile: dict) -> dict:
    measurement = checked["collection_protocol"]["measurement"]
    return {
        "model_path": str(checked["paths"]["model"]),
        "workload": profile["workload"],
        "worker_count": measurement["worker_count"],
        "queue_capacity": measurement["queue_capacity"],
        "enable_batching": cell["enable_batching"],
        "max_batch_size": cell["max_batch_size"],
        "max_batch_delay_ms": cell["max_batch_delay_ms"],
        "warmup_requests": measurement["warmup_requests"],
        "check_interval": measurement["check_interval"],
        "duration_ms": measurement["duration_ms"],
        "lag_tolerance_ms": measurement["lag_tolerance_ms"],
        "comparison_class": checked["collection_protocol"]["comparison_class"],
        "analysis_protocol_version": checked["collection_protocol"]["analysis_protocol_version"],
    }


def _write_failure(run_set: pathlib.Path, payload: dict) -> None:
    path = run_set / "failure.json"
    if path.exists():
        return
    write_json(path, payload)


def execute_trials(
    *,
    run_set: pathlib.Path,
    binary: pathlib.Path,
    repo_root: pathlib.Path,
    checked: dict,
    runner,
    sleep,
) -> bool:
    runs = run_set / "runs"
    measurement = checked["collection_protocol"]["measurement"]
    timeout_seconds = wall_timeout_seconds(measurement)
    model = checked["paths"]["model"]
    for index, trial in enumerate(checked["plan"]):
        if index > 0 and checked["collection_protocol"]["inter_trial_delay_seconds"] > 0:
            sleep(checked["collection_protocol"]["inter_trial_delay_seconds"])
        cell = cell_by_id(str(trial["cell_id"]))
        profile = profile_by_id(str(trial["profile_id"]))
        trial_id = str(trial["trial_id"])
        command = scheduler_command(binary, model, cell, profile, measurement)
        env = dict(os.environ)
        env["ORT_DISABLE_TELEMETRY"] = "1"
        started = utc_now()
        try:
            result = runner(
                command,
                cwd=repo_root,
                env=env,
                timeout_seconds=timeout_seconds,
            )
        except TypeError:
            result = runner(command, cwd=repo_root, env=env)
        except Exception as exc:
            result = SupervisedResult(127, "", f"failed to execute trial {trial_id}: {exc}")
        finished = utc_now()
        stdout_path = runs / f"{trial_id}.stdout"
        stderr_path = runs / f"{trial_id}.stderr"
        report_path = runs / f"{trial_id}.json"
        timed_out = bool(getattr(result, "timed_out", False))
        problems: list[str] = []
        publication: list[str] = []
        classification: dict[str, object] = {
            "runner_valid": False,
            "collector_valid": False,
            "publication_eligible": False,
            "steady_state_eligible": False,
            "delivery_fraction": None,
            "submission_fraction": None,
            "scheduler_rejection_fraction": None,
            "scheduler_acceptance_fraction": None,
            "successful_completion_fraction": None,
            "window_class": None,
            "delivery_class": None,
            "timing_quality": None,
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
            _document, classification = classify_report(
                result.stdout,
                expected=_expected_launch(checked, cell, profile),
            )
            problems.extend(list(classification["problems"]))
            publication.extend(list(classification["publication_problems"]))
            primary = checked["collection_protocol"]["comparison_class"] == "primary"
            if primary and publication:
                problems.extend(publication)
            try:
                write_exclusive_text(report_path, result.stdout)
            except (OSError, CollectorError) as exc:
                problems.append(f"could not preserve trial report: {exc}")
        if problems and classification.get("delivery_class") is None:
            classification["delivery_class"] = "INVALID"
        meta = {
            "trial_id": trial_id,
            "configuration_id": cell["id"],
            "workload_id": profile["id"],
            "workload": profile["workload"],
            "run_index": trial["run_number"],
            "block_index": trial.get("block_index"),
            "position_in_block": trial.get("position_in_block"),
            "execution_index": trial.get("execution_index"),
            "counted_run_target": trial.get("counted_run_target"),
            "pilot": bool(trial["pilot"]),
            "analysis_protocol_version": checked["collection_protocol"]["analysis_protocol_version"],
            "publication_policy": checked["collection_protocol"]["publication_policy"],
            "comparison_class": checked["collection_protocol"]["comparison_class"],
            "returncode": result.returncode,
            "execution_timeout": timed_out,
            "wall_timeout_seconds": timeout_seconds,
            "model_sha256": MODEL_SHA256,
            "benchmark_binary_sha256": checked["queried"]["benchmark_binary_sha256"],
            "git_commit": checked["commit"],
            "runner_valid": classification["runner_valid"],
            "collector_valid": classification["collector_valid"],
            "publication_eligible": classification["publication_eligible"] and not problems,
            "steady_state_eligible": classification["steady_state_eligible"] and not problems,
            "window_class": classification["window_class"],
            "delivery_class": classification["delivery_class"],
            "timing_quality": classification["timing_quality"],
            "timing_quality_note": classification.get("timing_quality_note"),
            "delivery_fraction": classification["delivery_fraction"],
            "submission_fraction": classification.get("submission_fraction"),
            "scheduler_rejection_fraction": classification["scheduler_rejection_fraction"],
            "scheduler_acceptance_fraction": classification["scheduler_acceptance_fraction"],
            "successful_completion_fraction": classification.get("successful_completion_fraction"),
            "minimum_submit_fraction": 1.0,
            "publication_problems": publication,
            "problems": problems,
            "command": command,
            "started_utc": started,
            "finished_utc": finished,
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
    sleep=time.sleep,
    blob_sha256=git_blob_sha256,
    clock=None,
) -> pathlib.Path:
    repo_root = pathlib.Path(args.repo_root).resolve()
    binary = pathlib.Path(args.binary).resolve()
    cells = parse_id_list(args.cells, cell_by_id)
    profiles = parse_id_list(args.profiles, profile_by_id)
    measurement = measurement_from_args(args)
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

    checked = preflight(
        repo_root=repo_root,
        binary=binary,
        expected_commit=args.expected_commit,
        cells=cells,
        profiles=profiles,
        measurement=measurement,
        counted_runs=args.counted_runs,
        delay_seconds=args.inter_trial_delay_seconds,
        comparison_class=args.comparison_class,
        baseline_cell=args.baseline_cell,
        alternative_cell=args.alternative_cell,
        runner=query_runner,
        blob_sha256=blob_sha256,
    )
    moment = datetime.datetime.now(datetime.timezone.utc) if clock is None else clock()
    run_set_id = make_run_set_id(moment, checked["commit"])
    parent = pathlib.Path(args.output_root).resolve() if args.output_root else (
        repo_root / OUTPUT_ROOT_RELATIVE
    )
    run_set = create_run_set(parent, run_set_id)
    manifest = public_manifest(checked, run_set_id, "running")
    try:
        write_json(run_set / "environment.json", manifest)
        accepted = execute_trials(
            run_set=run_set,
            binary=binary,
            repo_root=repo_root,
            checked=checked,
            runner=runner,
            sleep=sleep,
        )
        if (run_set / "aggregate.json").exists():
            accepted = False
            _write_failure(
                run_set,
                {
                    "status": "failed",
                    "accepted": False,
                    "trial_id": None,
                    "reason": "collector must not write aggregate.json",
                    "preserved_earlier_trials": True,
                    "aggregate": None,
                },
            )
        final = public_manifest(checked, run_set_id, "completed" if accepted else "failed")
        replace_json(run_set / "environment.json", final)
    except CollectorError:
        raise
    except Exception as exc:
        raise CollectorError(f"collection error: {exc}") from exc
    if not accepted:
        raise CollectorError(f"collection stopped; see {run_set / 'failure.json'}")
    return run_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect inferedge-bench-scheduler schema-3 trials into a new "
            "benchmarks/scheduler_runs directory. This does not analyze or publish them."
        )
    )
    parser.add_argument("--binary", required=True)
    parser.add_argument("--repo-root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--cells", required=True, help="Comma-separated cell ids, for example A,B")
    parser.add_argument(
        "--profiles",
        required=True,
        help="Comma-separated profile ids. Do not mix open-loop and closed-loop.",
    )
    parser.add_argument("--comparison-class", choices=("primary", "exploratory"), default="primary")
    parser.add_argument("--baseline-cell", default=None)
    parser.add_argument("--alternative-cell", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--counted-runs",
        type=int,
        default=PROTOCOL2_DEFAULT_COUNTED_RUNS,
        help=(
            "Counted blocks per cell, from 1 through 30. "
            "Protocol 2 defaults to 10, the minimum that can support an interval. "
            "The preferred publication count is 20. Smoke tests should pass 1."
        ),
    )
    parser.add_argument("--inter-trial-delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--duration-ms", type=int)
    parser.add_argument("--max-offered", type=int)
    parser.add_argument("--max-outstanding", type=int)
    parser.add_argument("--completion-timeout-ms", type=int)
    parser.add_argument("--warmup-requests", type=int)
    parser.add_argument("--lag-tolerance-ms", type=int)
    parser.add_argument("--check-interval", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None, runner=None, sleep=time.sleep, blob_sha256=git_blob_sha256) -> int:
    args = build_parser().parse_args(argv)
    try:
        require_commit_sha(args.expected_commit)
        if args.preflight_only:
            cells = parse_id_list(args.cells, cell_by_id)
            profiles = parse_id_list(args.profiles, profile_by_id)
            measurement = measurement_from_args(args)

            def query_runner(command, cwd=None, env=None):
                if runner is None:
                    return supervise_process(command, cwd=cwd, env=env, timeout_seconds=30)
                try:
                    return runner(command, cwd=cwd, env=env, timeout_seconds=30)
                except TypeError:
                    return runner(command, cwd=cwd, env=env)

            checked = preflight(
                repo_root=pathlib.Path(args.repo_root).resolve(),
                binary=pathlib.Path(args.binary).resolve(),
                expected_commit=args.expected_commit,
                cells=cells,
                profiles=profiles,
                measurement=measurement,
                counted_runs=args.counted_runs,
                delay_seconds=args.inter_trial_delay_seconds,
                comparison_class=args.comparison_class,
                baseline_cell=args.baseline_cell,
                alternative_cell=args.alternative_cell,
                runner=query_runner,
                blob_sha256=blob_sha256,
            )
            print("preflight passed")
            print(checked["queried"]["onnxruntime_resolved_library"])
            return 0
        run_set = collect(args, runner=runner, sleep=sleep, blob_sha256=blob_sha256)
    except CollectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(run_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
