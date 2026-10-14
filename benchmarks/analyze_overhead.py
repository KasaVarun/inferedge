#!/usr/bin/env python3
"""Paired analysis for one completion-timing overhead run set.

The experimental unit is the counted OFF/ON block. Pilots are excluded.
difference = ON metric - OFF metric. This module never declares a winner.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.analyze_scheduler import (
    MINIMUM_INTERVAL_BLOCKS,
    paired_block_bootstrap_mean,
    paired_metric_effect,
)
from benchmarks.collect_baseline import ONNXRUNTIME_VERSION
from benchmarks.collect_overhead import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CELL_ID,
    CONFIDENCE_LEVEL,
    EXPERIMENT_KIND,
    INTENDED_VARIABLE,
    OVERHEAD_PROTOCOL_MARKER,
    OVERHEAD_PROTOCOL_MARKER_MEANING,
    PROFILE_ID,
    arm_order,
    command_for_arm,
    commands_differ_only_by_timing,
    shared_launch,
    validate_arm,
)
from benchmarks.collect_scheduler import (
    OPEN_LOOP_QUANTUM_NS,
    PRECISION_SCHEMA_VERSION,
    PROFILE_BY_ID,
    SCHEMA4_EXTRA_KEYS,
    SCHEMA_VERSION,
    CollectorError,
    write_json,
)

TARGET_RPS = float(PROFILE_BY_ID[PROFILE_ID]["target_rps"])

LIMITATIONS = [
    "The study isolates the incremental cost of enabling timing in the current binary.",
    "Both arms already contain the larger SchedulerCompletion layout.",
    "The study does not measure layout changes relative to the earlier binary.",
    "The first workload is the tiny Add model, unbatched, at 50 offered requests per second.",
    "That workload may be too light to reveal meaningful timing overhead.",
    "Host scheduling variation, thermal drift, CPU frequency changes, and background load remain.",
    "Open-loop futures are observed by 1 ms polling.",
    "Schema 3 has no generator-lag percentile distribution, so arrival fidelity is not jointly certified.",
    "The default 1000 ms generator tolerance does not establish precise arrival scheduling.",
    "Burst phase fidelity is not claimed. There is no universal microsecond gate.",
    "Instrumentation may alter batch grouping in a future batched experiment.",
    "This is not a general overhead estimate and it does not extrapolate to other workloads or machines.",
    "An interval that includes zero is inconclusive. It does not establish zero performance cost.",
    "An interval that excludes zero reports only that paired difference for this measured metric.",
]


class AnalyzerError(CollectorError):
    pass


def _load_json(path: pathlib.Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalyzerError(f"could not read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalyzerError(f"{path.name} is not a JSON object")
    return value


def _finite(value: object) -> bool:
    if isinstance(value, bool) or type(value) not in {int, float}:
        return False
    return value == value and value not in {float("inf"), float("-inf")}


def _number(report: dict, *path: str) -> float | None:
    value: object = report
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if not _finite(value):
        return None
    return float(value)


def _rejection_fraction(report: dict) -> float | None:
    attempts = _number(report, "submit_attempts")
    rejections = report.get("rejections")
    if attempts is None or attempts == 0 or not isinstance(rejections, dict):
        return None
    total = 0.0
    for name in ("full", "closed", "not_running"):
        part = rejections.get(name)
        if not _finite(part):
            return None
        total += float(part)
    return total / attempts


def _delivery_fraction(report: dict) -> float | None:
    scheduled = _number(report, "scheduled_opportunities")
    attempts = _number(report, "submit_attempts")
    if scheduled is None or scheduled == 0 or attempts is None:
        return None
    return attempts / scheduled


SHARED_METRICS: list[tuple[str, str, object]] = [
    ("p50_e2e_latency", "microseconds", lambda report: _number(report, "successful_e2e_latency", "p50_microseconds")),
    ("mean_e2e_latency", "microseconds", lambda report: _number(report, "successful_e2e_latency", "mean_microseconds")),
    (
        "successful_throughput",
        "successful requests per arrival-window second",
        lambda report: _number(report, "throughput", "successful_attempts_per_arrival_window_second"),
    ),
    (
        "scheduler_rejection_fraction",
        "fraction of submit_attempts",
        _rejection_fraction,
    ),
    (
        "generator_delivery_fraction",
        "submit_attempts / scheduled_opportunities",
        _delivery_fraction,
    ),
    ("engine_run_count", "count", lambda report: _number(report, "metrics", "engine_run_count")),
    ("succeeded", "count", lambda report: _number(report, "succeeded")),
    ("scheduled_opportunities", "count", lambda report: _number(report, "scheduled_opportunities")),
    ("submit_attempts", "count", lambda report: _number(report, "submit_attempts")),
    ("arrival_window_duration", "nanoseconds", lambda report: _number(report, "arrival_window_nanoseconds")),
    ("generator_lag_count", "count", lambda report: _number(report, "generator_lag", "count")),
    (
        "generator_lag_minimum",
        "nanoseconds",
        lambda report: _number(report, "generator_lag", "minimum_nanoseconds"),
    ),
    (
        "generator_lag_maximum",
        "nanoseconds",
        lambda report: _number(report, "generator_lag", "maximum_nanoseconds"),
    ),
    ("generator_lag_mean", "nanoseconds", lambda report: _number(report, "generator_lag", "mean_nanoseconds")),
    ("accepted", "count", lambda report: _number(report, "accepted")),
    ("failed_inference", "count", lambda report: _number(report, "failed_inference")),
    ("failed_worker", "count", lambda report: _number(report, "failed_worker")),
    ("failed_scheduler", "count", lambda report: _number(report, "failed_scheduler")),
    ("canceled", "count", lambda report: _number(report, "canceled")),
    ("timeouts", "count", lambda report: _number(report, "timeouts")),
    ("rejections_full", "count", lambda report: _number(report, "rejections", "full")),
    ("rejections_closed", "count", lambda report: _number(report, "rejections", "closed")),
    ("rejections_not_running", "count", lambda report: _number(report, "rejections", "not_running")),
]


def _shared_identity(report: dict) -> dict:
    scheduler = report.get("scheduler") if isinstance(report.get("scheduler"), dict) else {}
    generator = report.get("generator") if isinstance(report.get("generator"), dict) else {}
    phases = report.get("burst_phases")
    phase = phases[0] if isinstance(phases, list) and phases and isinstance(phases[0], dict) else {}
    identity = {
        "model": report.get("model"),
        "workload": report.get("workload"),
        "e2e_latency_definition": report.get("e2e_latency_definition"),
        "completion_observation": report.get("completion_observation"),
        "observation_quantum_nanoseconds": report.get("observation_quantum_nanoseconds"),
        "configured_duration_nanoseconds": report.get("configured_duration_nanoseconds"),
        "scheduled_span_nanoseconds": report.get("scheduled_span_nanoseconds"),
        "arrival_stop_reason": report.get("arrival_stop_reason"),
        "completion_timeout_bound": report.get("completion_timeout_bound"),
        "scheduled_opportunities": report.get("scheduled_opportunities"),
        "worker_count": scheduler.get("worker_count"),
        "queue_capacity": scheduler.get("queue_capacity"),
        "enable_batching": scheduler.get("enable_batching"),
        "max_batch_size": scheduler.get("max_batch_size"),
        "max_batch_delay_ms": scheduler.get("max_batch_delay_ms"),
        "enable_metrics": scheduler.get("enable_metrics"),
        "lag_definition": generator.get("lag_definition"),
        "lag_policy": generator.get("lag_policy"),
        "outstanding_policy": generator.get("outstanding_policy"),
        "warmup_requests": generator.get("warmup_requests"),
        "check_interval": generator.get("check_interval"),
        "target_rps": phase.get("requests_per_second"),
        "phase_duration_nanoseconds": phase.get("duration_nanoseconds"),
    }
    return identity


def _cell_problems(identity: dict) -> list[str]:
    expected = {
        "workload": "open_loop_constant",
        "observation_quantum_nanoseconds": OPEN_LOOP_QUANTUM_NS,
        "worker_count": 1,
        "queue_capacity": 64,
        "enable_batching": False,
        "max_batch_size": 1,
        "max_batch_delay_ms": 0,
        "target_rps": TARGET_RPS,
    }
    return [f"{name} is {identity.get(name)!r}, expected {value!r}" for name, value in expected.items() if identity.get(name) != value]


def _identity_problems(off: dict, on: dict) -> list[str]:
    left = _shared_identity(off)
    right = _shared_identity(on)
    problems = [f"shared setting {name} differs" for name in left if left[name] != right[name]]
    problems.extend(f"off {item}" for item in _cell_problems(left))
    problems.extend(f"on {item}" for item in _cell_problems(right))
    if off.get("schema_version") != SCHEMA_VERSION or on.get("schema_version") != PRECISION_SCHEMA_VERSION:
        problems.append(
            f"schema pair is {off.get('schema_version')!r}/{on.get('schema_version')!r}, expected 3/4"
        )
    off_timing = off.get("scheduler", {}).get("enable_completion_timing", False) if isinstance(off.get("scheduler"), dict) else False
    on_timing = on.get("scheduler", {}).get("enable_completion_timing", False) if isinstance(on.get("scheduler"), dict) else False
    if off_timing is not False or on_timing is not True:
        problems.append("enable_completion_timing is not the only intended scheduler difference")
    extras = [name for name in SCHEMA4_EXTRA_KEYS if name in off]
    if extras:
        problems.append("OFF report contains schema-4 fields: " + ", ".join(extras))
    return problems


def _on_only(report: dict, block_index: int) -> dict:
    lag = report["generator_lag"]
    return {
        "block_index": block_index,
        "pre_publication_latency": report["pre_publication_latency"],
        "pre_publication_to_observation_delay": report["pre_publication_to_observation_delay"],
        "generator_lag_p50_nanoseconds": lag["p50_nanoseconds"],
        "generator_lag_p95_nanoseconds": lag["p95_nanoseconds"],
        "generator_lag_p99_nanoseconds": lag["p99_nanoseconds"],
        "timing_outcome_counts": report["timing_outcome_counts"],
        "steady_clock_period_numerator": report["steady_clock_period_numerator"],
        "steady_clock_period_denominator": report["steady_clock_period_denominator"],
        "steady_clock_period_note": report["steady_clock_period_note"],
        "phase_boundary_fidelity": report["phase_boundary_fidelity"],
        "off_counterpart": None,
    }


def _histogram_record(off: dict, on: dict, block_index: int) -> dict:
    left = off["metrics"]["batch_size_histogram"]
    right = on["metrics"]["batch_size_histogram"]
    return {
        "block_index": block_index,
        "units": "count per batch-size bin",
        "off": left,
        "on": right,
        "equal": left == right,
    }


def _metric_result(name: str, units: str, off_values: list[float], on_values: list[float], block_indexes: list[int], allow_interval: bool) -> dict:
    effect = paired_metric_effect(
        off_values,
        on_values,
        units=units,
        allow_interval=allow_interval,
        seed=BOOTSTRAP_SEED,
        replicates=BOOTSTRAP_REPLICATES,
        level=CONFIDENCE_LEVEL,
    )
    blocks = []
    for index, off_value, on_value in zip(block_indexes, off_values, on_values):
        blocks.append(
            {
                "block_index": index,
                "off": off_value,
                "on": on_value,
                "difference": on_value - off_value,
            }
        )
    interval = effect["confidence_interval"]
    if interval is None:
        interpretation = "interval not computed"
    elif float(interval["low"]) > 0.0 or float(interval["high"]) < 0.0:
        interpretation = "paired difference for this metric on this workload and machine only"
    else:
        interpretation = "inconclusive"
    effect["metric"] = name
    effect["blocks"] = blocks
    effect["mean_paired_difference"] = effect["effect_estimate"]
    effect["interpretation"] = interpretation
    effect["winner_declared"] = False
    effect["superiority_claimed"] = False
    effect["zero_overhead_claimed"] = False
    if interpretation == "inconclusive":
        effect["limitation_notes"] = list(effect["limitation_notes"]) + [
            "The interval includes zero. That is inconclusive and is not a zero-cost result."
        ]
    return effect


def _check_plan(plan: list[dict], counted_blocks: int) -> None:
    if len(plan) < 2 or any(not isinstance(trial, dict) for trial in plan):
        raise AnalyzerError("the run is missing the uncounted OFF/ON pilot pair")
    pilots = [trial for trial in plan if trial.get("pilot") is True]
    pilot_ids = [trial.get("trial_id") for trial in pilots]
    if pilot_ids != ["overhead-pilot-off", "overhead-pilot-on"]:
        raise AnalyzerError("the run must contain exactly one uncounted OFF/ON pilot pair")
    if [plan[0].get("trial_id"), plan[1].get("trial_id")] != ["overhead-pilot-off", "overhead-pilot-on"]:
        raise AnalyzerError("the pilot pair must be the first two trials")
    if [plan[0].get("arm"), plan[1].get("arm")] != ["off", "on"]:
        raise AnalyzerError("the pilot pair is not OFF then ON")
    if any(trial.get("block_index") is not None for trial in pilots):
        raise AnalyzerError("a pilot cannot carry a counted block index")
    counted = [trial for trial in plan if not trial["pilot"]]
    indexes = [trial["execution_index"] for trial in plan]
    if indexes != list(range(len(plan))):
        raise AnalyzerError("execution_index is not a contiguous rotation starting at 0")
    seen: set[int] = set()
    for block_index in range(counted_blocks):
        rows = [trial for trial in counted if trial["block_index"] == block_index]
        if not rows:
            raise AnalyzerError(f"missing block {block_index}")
        if len(rows) != 2:
            raise AnalyzerError(f"block {block_index} does not contain exactly one OFF and one ON")
        rows.sort(key=lambda trial: trial["position_in_block"])
        arms = [trial["arm"] for trial in rows]
        if arms.count("off") != 1 or arms.count("on") != 1:
            raise AnalyzerError(f"block {block_index} arms are {arms}, expected one OFF and one ON")
        if tuple(arms) != arm_order(block_index):
            raise AnalyzerError(
                f"block {block_index} order is {arms}, expected {list(arm_order(block_index))}"
            )
        seen.add(block_index)
    if len(counted) != counted_blocks * 2 or seen != set(range(counted_blocks)):
        raise AnalyzerError("duplicate block or counted trial count does not match counted_blocks")


def _require_text(record: dict, key: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or value == "":
        raise AnalyzerError(f"{key} is missing")


def _load_trials(run_set: pathlib.Path, plan: list[dict]) -> dict[str, dict]:
    runs = run_set / "runs"
    found: dict[str, dict] = {}
    for path in sorted(runs.glob("*.meta.json")):
        meta = _load_json(path)
        trial_id = meta.get("trial_id")
        if not isinstance(trial_id, str) or trial_id in found:
            raise AnalyzerError(f"duplicate or unreadable trial metadata: {path.name}")
        if path.name != f"{trial_id}.meta.json":
            raise AnalyzerError(f"{path.name} does not match trial id {trial_id}")
        found[trial_id] = meta
    expected = [str(trial["trial_id"]) for trial in plan]
    if sorted(found) != sorted(expected):
        raise AnalyzerError("trial metadata does not match the execution plan")
    return found


def analyze(run_set: pathlib.Path) -> pathlib.Path:
    run_set = pathlib.Path(run_set)
    aggregate_path = run_set / "aggregate.json"
    if aggregate_path.exists():
        raise AnalyzerError(f"refusing to overwrite existing evidence: {aggregate_path}")
    environment = _load_json(run_set / "environment.json")
    if environment.get("experiment_kind") != EXPERIMENT_KIND:
        raise AnalyzerError("refusing a run that is not completion_timing_overhead")
    if environment.get("intended_variable") != INTENDED_VARIABLE:
        raise AnalyzerError("intended variable is not enable_completion_timing")
    protocol = environment.get("collection_protocol")
    if not isinstance(protocol, dict):
        raise AnalyzerError("collection protocol is missing")
    if protocol.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise AnalyzerError("bootstrap seed is not 20261014")
    if protocol.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES:
        raise AnalyzerError("bootstrap replicates are not 10000")
    if protocol.get("confidence_level") != CONFIDENCE_LEVEL:
        raise AnalyzerError("confidence level is not 0.95")
    if protocol.get("analysis_protocol_version") != OVERHEAD_PROTOCOL_MARKER:
        raise AnalyzerError(
            "analysis_protocol_version must stay the overhead sentinel 0; "
            "it is not analysis protocol 1, 2, or 3"
        )
    if protocol.get("analysis_protocol_version_meaning") != OVERHEAD_PROTOCOL_MARKER_MEANING:
        raise AnalyzerError("the overhead protocol marker is not documented on this run")
    if protocol.get("shared_launch") != shared_launch():
        raise AnalyzerError("shared launch settings differ from the cell-A low_open contract")
    evidence_class = protocol.get("evidence_class")
    mode = protocol.get("collection_mode")
    if evidence_class not in {"correctness", "exploratory", "counted_overhead"}:
        raise AnalyzerError("evidence class is missing")
    if mode == "smoke" or evidence_class == "correctness":
        if mode != "smoke" or evidence_class != "correctness":
            raise AnalyzerError("smoke evidence must stay classified as correctness")
    counted_blocks = protocol.get("counted_blocks")
    if type(counted_blocks) is not int or counted_blocks < 0:
        raise AnalyzerError("counted_blocks is missing")
    if evidence_class == "correctness" and counted_blocks != 0:
        raise AnalyzerError("a smoke pair cannot be promoted into counted blocks")
    if evidence_class == "counted_overhead" and counted_blocks != 20:
        raise AnalyzerError("a reportable study uses 20 counted blocks")
    if evidence_class == "exploratory" and not 1 <= counted_blocks <= 30:
        raise AnalyzerError("an exploratory study needs 1 through 30 counted blocks")
    plan_document = _load_json(run_set / "execution_plan.json")
    plan = plan_document.get("trials")
    if not isinstance(plan, list):
        raise AnalyzerError("execution plan is missing")
    _check_plan(plan, counted_blocks)
    metas = _load_trials(run_set, plan)
    queried = environment["queried"]
    verified = environment["verified_configuration"]
    for key in (
        "git_commit",
        "benchmark_binary_path",
        "benchmark_binary_sha256",
        "compiler_path",
        "compiler_version",
        "cmake_flag_text",
        "onnxruntime_resolved_library",
        "operating_system",
        "cpu_architecture",
        "hardware_model",
    ):
        _require_text(queried, key)
    _require_text(verified, "model_path")
    _require_text(verified, "model_sha256")
    if verified.get("cmake_build_type") != "Release":
        raise AnalyzerError("the benchmark binary is not a Release build")
    if queried.get("onnxruntime_version") != ONNXRUNTIME_VERSION:
        raise AnalyzerError(
            f"onnxruntime_version is {queried.get('onnxruntime_version')!r}, "
            f"expected {ONNXRUNTIME_VERSION}"
        )
    if queried.get("ort_disable_telemetry") != "1":
        raise AnalyzerError("ORT telemetry was not disabled")
    model = pathlib.Path(verified["model_path"])
    binary = pathlib.Path(queried["benchmark_binary_path"])
    reports: dict[str, dict] = {}
    problems: list[str] = []
    for trial in plan:
        trial_id = str(trial["trial_id"])
        meta = metas[trial_id]
        for field in ("arm", "pilot", "block_index", "position_in_block", "execution_index", "run_index"):
            if meta.get(field) != trial.get(field):
                problems.append(f"{trial_id} {field} does not match the plan")
        if meta.get("benchmark_binary_sha256") != queried["benchmark_binary_sha256"]:
            problems.append(f"{trial_id} benchmark_binary_sha256 differs")
        if meta.get("model_sha256") != verified["model_sha256"]:
            problems.append(f"{trial_id} model_sha256 differs")
        if meta.get("git_commit") != queried["git_commit"]:
            problems.append(f"{trial_id} git_commit differs")
        if meta.get("collector_valid") is not True or meta.get("execution_timeout") is True:
            problems.append(f"{trial_id} is not a valid completed trial")
        expected_command = command_for_arm(binary, model, str(trial["arm"]))
        if trial.get("command") != expected_command:
            problems.append(f"{trial_id} plan command diverges from the {trial['arm']} arm")
        if meta.get("command") != trial.get("command"):
            problems.append(f"{trial_id} executed command diverges from the execution plan")
        report_name = meta.get("report_path")
        if report_name is not None and pathlib.Path(str(report_name)).name != f"{trial_id}.json":
            problems.append(f"{trial_id} report path does not match the trial id")
        report_path = run_set / "runs" / f"{trial_id}.json"
        if not report_path.is_file():
            problems.append(f"{trial_id} report is missing")
            continue
        text = report_path.read_text(encoding="utf-8")
        document, _status, arm_problems = validate_arm(text, str(trial["arm"]), model)
        if arm_problems or document is None:
            problems.append(f"{trial_id} " + "; ".join(arm_problems or ["invalid report"]))
            continue
        reports[trial_id] = document
    if problems:
        raise AnalyzerError("; ".join(problems))
    off_command = command_for_arm(binary, model, "off")
    on_command = command_for_arm(binary, model, "on")
    if not commands_differ_only_by_timing(off_command, on_command):
        raise AnalyzerError("OFF and ON commands differ by more than --completion-timing")
    pairs = []
    for block_index in range(counted_blocks):
        rows = [trial for trial in plan if trial["block_index"] == block_index and not trial["pilot"]]
        off = next(trial for trial in rows if trial["arm"] == "off")
        on = next(trial for trial in rows if trial["arm"] == "on")
        off_report = reports[str(off["trial_id"])]
        on_report = reports[str(on["trial_id"])]
        identity = _identity_problems(off_report, on_report)
        if identity:
            raise AnalyzerError(f"block {block_index}: " + "; ".join(identity))
        pairs.append((block_index, off_report, on_report))
    allow_interval = counted_blocks >= MINIMUM_INTERVAL_BLOCKS
    metrics = []
    for name, units, extractor in SHARED_METRICS:
        off_values = []
        on_values = []
        indexes = []
        for block_index, off_report, on_report in pairs:
            off_value = extractor(off_report)
            on_value = extractor(on_report)
            if off_value is None or on_value is None:
                raise AnalyzerError(f"{name} is missing or nonfinite in block {block_index}")
            off_values.append(off_value)
            on_values.append(on_value)
            indexes.append(block_index)
        if off_values:
            metrics.append(_metric_result(name, units, off_values, on_values, indexes, allow_interval))
    check = None
    if any(metric["metric"] == "p50_e2e_latency" and metric["confidence_interval_computed"] for metric in metrics):
        latency = next(metric for metric in metrics if metric["metric"] == "p50_e2e_latency")
        differences = [block["difference"] for block in latency["blocks"]]
        check = paired_block_bootstrap_mean(
            differences,
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
            level=CONFIDENCE_LEVEL,
        )
        if check != latency["confidence_interval"]:
            raise AnalyzerError("paired bootstrap did not reproduce the latency interval")
    aggregate = {
        "experiment_kind": EXPERIMENT_KIND,
        "intended_variable": INTENDED_VARIABLE,
        "cell_id": CELL_ID,
        "profile_id": PROFILE_ID,
        "status": "completed",
        "winner_declared": False,
        "superiority_claimed": False,
        "zero_overhead_claimed": False,
        "general_overhead_estimate": False,
        "experimental_unit": "paired_block",
        "effect_definition": "difference = on_metric - off_metric",
        "relative_effect_definition": (
            "mean of per-block 100 * difference / off_value, only when every off value is strictly positive"
        ),
        "bootstrap": {
            "method": "paired_block_bootstrap_mean",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "level": CONFIDENCE_LEVEL,
            "minimum_blocks": MINIMUM_INTERVAL_BLOCKS,
            "computed": allow_interval and evidence_class != "correctness",
        },
        "evidence_class": evidence_class,
        "analysis_protocol_version": OVERHEAD_PROTOCOL_MARKER,
        "analysis_protocol_version_meaning": OVERHEAD_PROTOCOL_MARKER_MEANING,
        "counted_blocks": counted_blocks,
        "pilots_excluded": True,
        "block_index_origin": 0,
        "pilot_order": "off_then_on",
        "counted_rotation": "even blocks off_then_on; odd blocks on_then_off",
        "arrival_fidelity": {
            "common_gate": "strict generator delivery and shared lag count, minimum, maximum, and mean",
            "certified_percentile_fidelity": False,
            "reason": (
                "Schema 3 does not contain lag p50, p95, or p99. "
                "The 1000 ms tolerance is not precise arrival scheduling."
            ),
            "phase_boundary_fidelity": "unavailable",
        },
        "shared_metrics": metrics,
        "batch_size_histograms": [
            _histogram_record(off_report, on_report, block_index) for block_index, off_report, on_report in pairs
        ],
        "on_only_observations": [_on_only(on_report, block_index) for block_index, _off, on_report in pairs],
        "bootstrap_reproduction": check,
        "limitations": LIMITATIONS,
        "queried": {
            "git_commit": queried["git_commit"],
            "benchmark_binary_sha256": queried["benchmark_binary_sha256"],
            "compiler_path": queried["compiler_path"],
            "compiler_version": queried["compiler_version"],
            "cmake_build_type": verified["cmake_build_type"],
            "cmake_flag_text": queried["cmake_flag_text"],
            "onnxruntime_version": queried["onnxruntime_version"],
            "onnxruntime_resolved_library": queried["onnxruntime_resolved_library"],
            "operating_system": queried["operating_system"],
            "cpu_architecture": queried["cpu_architecture"],
            "hardware_model": queried["hardware_model"],
            "model_sha256": verified["model_sha256"],
            "ort_disable_telemetry": queried["ort_disable_telemetry"],
        },
    }
    write_json(aggregate_path, aggregate)
    return aggregate_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze one completion-timing overhead run set.")
    parser.add_argument("run_set")
    args = parser.parse_args(argv)
    try:
        analyze(pathlib.Path(args.run_set))
    except AnalyzerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
