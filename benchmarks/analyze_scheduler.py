#!/usr/bin/env python3
"""Reduce one completed scheduler run set to aggregate.json.

This module does not launch inferedge-bench-scheduler and does not run a model.
Pilot trials stay on disk and are excluded from the reductions. A median of
per-run percentiles is not a pooled percentile. No confidence interval is
computed. The analyzer does not modify raw trial files and does not overwrite
an existing aggregate.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.collect_scheduler import (
    CELL_BY_ID,
    DEFAULT_COUNTED_RUNS,
    LIMITATIONS,
    MINIMUM_SUBMIT_FRACTION,
    MODEL_SHA256,
    PROFILE_BY_ID,
    TOOL_NAME,
    TOOL_VERSION,
    classify_report,
    execution_plan,
    profile_by_id,
)

AGGREGATE_SCHEMA_VERSION = 3
P99_SAMPLE_FLOOR = 100
FAIRNESS_SHARED = (
    "git_commit",
    "benchmark_binary_sha256",
    "model_sha256",
    "operating_system",
    "cpu_architecture",
    "platform",
    "hardware_model",
    "onnxruntime_version",
    "onnxruntime_resolved_library",
    "compiler_version",
    "release_flags",
    "sanitizer_instrumentation",
    "test_hook_cli",
    "workload_family",
    "profile_id",
    "workload",
    "target_offered_rate",
    "burst_schedule",
    "producer_count",
    "duration_ms",
    "configured_duration_nanoseconds",
    "max_offered",
    "max_outstanding",
    "lag_tolerance_ms",
    "completion_timeout_ms",
    "warmup_requests",
    "check_interval",
    "worker_count",
    "queue_capacity",
    "telemetry",
    "completion_observation",
    "observation_quantum_nanoseconds",
)
CELL_VARIABLES = ("enable_batching", "max_batch_size", "max_batch_delay_ms")


class AnalyzerError(Exception):
    """The run set cannot be reduced. No aggregate is published."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze one completed inferedge-bench-scheduler run set."
    )
    parser.add_argument("--run-set", required=True)
    return parser


def _object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise AnalyzerError(f"{label} is not a JSON object")
    return value


def load_environment(run_set: pathlib.Path) -> dict:
    path = run_set / "environment.json"
    if not path.is_file():
        raise AnalyzerError(f"environment.json is missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalyzerError(f"environment.json is not valid JSON: {exc.msg}") from exc
    return _object(document, "environment.json")


def require_completed(environment: dict, run_set: pathlib.Path) -> None:
    if environment.get("status") != "completed":
        raise AnalyzerError(f"collection status is {environment.get('status')!r}, expected 'completed'")
    if (run_set / "failure.json").exists():
        raise AnalyzerError("failure.json is present")
    if environment.get("aggregate") is not None:
        raise AnalyzerError("environment.json aggregate field is not null")
    if environment.get("run_set_id") != run_set.name:
        raise AnalyzerError("run-set id does not match the directory name")


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _pstdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.pstdev(values))


def _command_value(command: list, flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag)
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def require_recorded_command(meta: dict, environment: dict) -> None:
    """The saved argv must match the profile and measurement being compared."""
    command = meta.get("command")
    if not isinstance(command, list) or not command:
        raise AnalyzerError(f"{meta.get('trial_id')} is missing its recorded command")
    profile = PROFILE_BY_ID[meta["workload_id"]]
    measurement = environment["collection_protocol"]["measurement"]
    expected = {
        "--workers": str(measurement["worker_count"]),
        "--queue-capacity": str(measurement["queue_capacity"]),
        "--workload": profile["workload"],
        "--duration-ms": str(measurement["duration_ms"]),
        "--max-offered": str(measurement["max_offered"]),
        "--max-outstanding": str(measurement["max_outstanding"]),
        "--lag-tolerance-ms": str(measurement["lag_tolerance_ms"]),
        "--completion-timeout-ms": str(measurement["completion_timeout_ms"]),
        "--warmup": str(measurement["warmup_requests"]),
        "--check-interval": str(measurement["check_interval"]),
    }
    if profile["workload"] == "open_loop_constant":
        expected["--target-rps"] = str(int(profile["target_rps"])) if float(profile["target_rps"]).is_integer() else str(profile["target_rps"])
    elif profile["workload"] == "open_loop_burst":
        expected["--burst-high-rps"] = str(int(profile["burst_high_rps"]))
        expected["--burst-low-rps"] = str(int(profile["burst_low_rps"]))
        expected["--burst-high-ms"] = str(profile["burst_high_ms"])
        expected["--burst-low-ms"] = str(profile["burst_low_ms"])
    else:
        expected["--producer-count"] = str(profile["producer_count"])
    for flag, value in expected.items():
        if _command_value(command, flag) != value:
            raise AnalyzerError(
                f"{meta.get('trial_id')} recorded {flag} as {_command_value(command, flag)!r}, expected {value!r}"
            )


def fairness_record(environment: dict, report: dict, meta: dict) -> dict:
    protocol = environment["collection_protocol"]
    measurement = protocol["measurement"]
    queried = environment["queried"]
    verified = environment["verified_configuration"]
    profile = PROFILE_BY_ID[meta["workload_id"]]
    if profile["workload"] == "open_loop_constant":
        offered = profile["target_rps"]
        burst = None
        producers = 1
    elif profile["workload"] == "open_loop_burst":
        offered = None
        burst = {
            "burst_high_rps": profile["burst_high_rps"],
            "burst_low_rps": profile["burst_low_rps"],
            "burst_high_ms": profile["burst_high_ms"],
            "burst_low_ms": profile["burst_low_ms"],
        }
        producers = 1
    else:
        offered = None
        burst = None
        producers = profile["producer_count"]
    return {
        "git_commit": queried["git_commit"],
        "benchmark_binary_sha256": queried["benchmark_binary_sha256"],
        "model_sha256": verified["model_sha256"],
        "operating_system": queried.get("operating_system"),
        "cpu_architecture": queried.get("cpu_architecture"),
        "platform": queried.get("platform"),
        "hardware_model": queried.get("hardware_model"),
        "onnxruntime_version": queried.get("onnxruntime_version"),
        "onnxruntime_resolved_library": queried.get("onnxruntime_resolved_library"),
        "compiler_version": queried.get("compiler_version"),
        "release_flags": queried.get("cmake_flag_text"),
        "sanitizer_instrumentation": verified.get("sanitizer_instrumentation"),
        "test_hook_cli": "disabled",
        "workload_family": protocol["workload_family"],
        "profile_id": meta["workload_id"],
        "workload": report["workload"],
        "target_offered_rate": offered,
        "burst_schedule": burst,
        "producer_count": producers,
        "duration_ms": measurement["duration_ms"],
        "configured_duration_nanoseconds": report["configured_duration_nanoseconds"],
        "max_offered": measurement["max_offered"],
        "max_outstanding": measurement["max_outstanding"],
        "lag_tolerance_ms": measurement["lag_tolerance_ms"],
        "completion_timeout_ms": measurement["completion_timeout_ms"],
        "warmup_requests": report["generator"]["warmup_requests"],
        "check_interval": report["generator"]["check_interval"],
        "worker_count": report["scheduler"]["worker_count"],
        "queue_capacity": report["scheduler"]["queue_capacity"],
        "telemetry": protocol.get("environment_variable"),
        "completion_observation": report["completion_observation"],
        "observation_quantum_nanoseconds": report["observation_quantum_nanoseconds"],
        "enable_batching": report["scheduler"]["enable_batching"],
        "max_batch_size": report["scheduler"]["max_batch_size"],
        "max_batch_delay_ms": report["scheduler"]["max_batch_delay_ms"],
    }


def fairness_differences(left: dict, right: dict) -> list[str]:
    """Differences other than the batching cell variables."""
    differences = []
    for key in FAIRNESS_SHARED:
        if left.get(key) != right.get(key):
            differences.append(key)
    return differences


def assert_fair_pair(left: dict, right: dict) -> None:
    differences = fairness_differences(left, right)
    if differences:
        raise AnalyzerError(
            "refusing to compare mismatched cells: " + ", ".join(differences)
        )


def _expected_for(environment: dict, meta: dict) -> dict:
    protocol = environment["collection_protocol"]
    measurement = protocol["measurement"]
    cell = CELL_BY_ID[meta["configuration_id"]]
    profile = profile_by_id(meta["workload_id"])
    return {
        "model_path": environment["verified_configuration"]["model_path"],
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
        "comparison_class": protocol["comparison_class"],
    }


def _load_trial(run_set: pathlib.Path, environment: dict, trial: dict) -> tuple[dict, dict]:
    trial_id = str(trial["trial_id"])
    report_path = run_set / "runs" / f"{trial_id}.json"
    meta_path = run_set / "runs" / f"{trial_id}.meta.json"
    if not report_path.is_file() or not meta_path.is_file():
        raise AnalyzerError(f"trial evidence is missing: {trial_id}")
    meta = _object(json.loads(meta_path.read_text(encoding="utf-8")), trial_id)
    text = report_path.read_text(encoding="utf-8")
    report, status = classify_report(text, expected=_expected_for(environment, meta))
    if report is None or not status["collector_valid"]:
        raise AnalyzerError(f"{trial_id} is not collector-valid")
    if meta.get("pilot") is not trial["pilot"]:
        raise AnalyzerError(f"{trial_id} pilot flag does not match the plan")
    if environment["collection_protocol"]["comparison_class"] == "primary" and not trial["pilot"]:
        if not status["publication_eligible"]:
            raise AnalyzerError(f"{trial_id} is not publication-eligible")
    if meta.get("git_commit") != environment["queried"]["git_commit"]:
        raise AnalyzerError(f"{trial_id} git commit does not match the run set")
    if meta.get("model_sha256") != MODEL_SHA256:
        raise AnalyzerError(f"{trial_id} model hash does not match the pinned fixture")
    if meta.get("benchmark_binary_sha256") != environment["queried"]["benchmark_binary_sha256"]:
        raise AnalyzerError(f"{trial_id} binary hash does not match the run set")
    require_recorded_command(meta, environment)
    return report, status


def _rejection_rate(report: dict) -> float:
    attempts = report["submit_attempts"]
    if attempts == 0:
        return 0.0
    rejections = report["rejections"]
    total = rejections["full"] + rejections["closed"] + rejections["not_running"]
    return total / attempts


def _skip_fraction(report: dict) -> float:
    scheduled = report["scheduled_opportunities"]
    if scheduled == 0:
        return 0.0
    return (report["generator_skips_outstanding"] + report["generator_skips_lag"]) / scheduled


def _mean_batch_size(report: dict) -> float | None:
    metrics = report["metrics"]
    runs = metrics["engine_run_count"]
    if runs == 0:
        return None
    return metrics["requests_entered_engine_runs"] / runs


def _histogram_median(reports: list[dict]) -> list[float]:
    width = len(reports[0]["metrics"]["batch_size_histogram"])
    medians = []
    for index in range(width):
        medians.append(
            _median([report["metrics"]["batch_size_histogram"][index] for report in reports])
        )
    return medians


def reduce_group(reports: list[dict], statuses: list[dict]) -> dict:
    throughput = [
        report["throughput"]["successful_attempts_per_arrival_window_second"] for report in reports
    ]
    total_trial = [report["throughput"]["successful_per_total_trial_second"] for report in reports]
    p50_values = []
    p95_values = []
    p99_values = []
    p99_supported = True
    for report in reports:
        latency = report["successful_e2e_latency"]
        if not latency["percentiles_reported"]:
            p99_supported = False
            continue
        p50_values.append(latency["p50_microseconds"])
        p95_values.append(latency["p95_microseconds"])
        if latency["precise_tail"] is True and latency["sample_count"] >= P99_SAMPLE_FLOOR:
            p99_values.append(latency["p99_microseconds"])
        else:
            p99_supported = False
    batch_means = [_mean_batch_size(report) for report in reports]
    submit_fractions = []
    for report in reports:
        scheduled = report["scheduled_opportunities"]
        submit_fractions.append(report["submit_attempts"] / scheduled if scheduled else 0.0)
    return {
        "counted_runs": len(reports),
        "latency_sample_count_per_run": [
            report["successful_e2e_latency"]["sample_count"] for report in reports
        ],
        "completion_observation": reports[0]["completion_observation"],
        "observation_quantum_nanoseconds": reports[0]["observation_quantum_nanoseconds"],
        "latency_includes_observation_delay": True,
        "not_pure_engine_latency": True,
        "median_successful_attempts_per_arrival_window_second": _median(throughput),
        "successful_attempts_per_arrival_window_second_note": reports[0]["throughput"][
            "successful_attempts_per_arrival_window_second_note"
        ],
        "median_successful_per_total_trial_second": _median(total_trial),
        "successful_per_total_trial_second_note": reports[0]["throughput"][
            "successful_per_total_trial_second_note"
        ],
        "not_steady_state_capacity": True,
        "run_to_run_throughput_pstdev": _pstdev(throughput),
        "run_to_run_p95_pstdev": _pstdev(p95_values) if len(p95_values) == len(reports) else None,
        "median_of_run_p50_microseconds": _median(p50_values) if len(p50_values) == len(reports) else None,
        "median_of_run_p95_microseconds": _median(p95_values) if len(p95_values) == len(reports) else None,
        "median_of_run_p99_microseconds": _median(p99_values) if p99_supported else None,
        "p99_reduction": "median_of_run_p99",
        "p99_omitted_reason": None
        if p99_supported
        else "at least one run has precise_tail false or fewer than 100 successful samples",
        "median_scheduler_rejection_rate": _median([_rejection_rate(report) for report in reports]),
        "median_generator_skip_fraction": _median([_skip_fraction(report) for report in reports]),
        "median_submit_fraction": _median(submit_fractions),
        "median_engine_run_count": _median(
            [report["metrics"]["engine_run_count"] for report in reports]
        ),
        "median_observed_mean_batch_size": _median(batch_means) if None not in batch_means else None,
        "median_batch_size_histogram": _histogram_median(reports),
        "histogram_index_equals_batch_size": True,
        "arrival_stop_reasons": [report["arrival_stop_reason"] for report in reports],
        "window_classes": [status["window_class"] for status in statuses],
        "confidence_interval_computed": False,
        "winner_declared": False,
    }


def analyze(run_set: pathlib.Path) -> dict:
    if not run_set.is_dir():
        raise AnalyzerError(f"run-set directory does not exist: {run_set}")
    aggregate_path = run_set / "aggregate.json"
    if aggregate_path.exists():
        raise AnalyzerError(f"refusing to overwrite existing aggregate: {aggregate_path}")
    environment = load_environment(run_set)
    require_completed(environment, run_set)
    protocol = _object(environment.get("collection_protocol"), "collection_protocol")
    if protocol.get("counted_runs_per_cell") != DEFAULT_COUNTED_RUNS:
        raise AnalyzerError("primary analysis requires 5 counted runs per cell")
    cells = [CELL_BY_ID[cell_id] for cell_id in protocol["cells"]]
    profiles = [PROFILE_BY_ID[profile_id] for profile_id in protocol["profiles"]]
    plan = execution_plan(cells, profiles, DEFAULT_COUNTED_RUNS)
    if protocol.get("trial_order") != [trial["trial_id"] for trial in plan]:
        raise AnalyzerError("trial order does not match the rotated collection plan")
    if environment["verified_configuration"]["model_sha256"] != MODEL_SHA256:
        raise AnalyzerError("run set model hash is not the pinned batch_add fixture")
    if environment["queried"]["benchmark_tool_version"] != TOOL_VERSION:
        raise AnalyzerError("benchmark tool version does not match the collector")
    loaded = []
    for trial in plan:
        report, status = _load_trial(run_set, environment, trial)
        loaded.append((trial, report, status))
    families = {item[1]["workload"] == "closed_loop" for item in loaded}
    if len(families) != 1:
        raise AnalyzerError("open-loop and closed-loop trials are in one run set")
    groups: dict[tuple[str, str], list[tuple[dict, dict, dict]]] = {}
    for trial, report, status in loaded:
        if trial["pilot"]:
            continue
        key = (str(trial["cell_id"]), str(trial["profile_id"]))
        groups.setdefault(key, []).append((trial, report, status))
    steady = []
    bounded = []
    for (cell_id, profile_id), items in groups.items():
        if len(items) != DEFAULT_COUNTED_RUNS:
            raise AnalyzerError(f"{cell_id}/{profile_id} does not have 5 counted runs")
        reports = [report for _trial, report, _status in items]
        statuses = [status for _trial, _report, status in items]
        observations = {report["completion_observation"] for report in reports}
        if len(observations) != 1:
            raise AnalyzerError(f"{cell_id}/{profile_id} mixes completion observation methods")
        windows = {status["window_class"] for status in statuses}
        if len(windows) != 1:
            raise AnalyzerError(f"{cell_id}/{profile_id} mixes measurement-window classes")
        summary = reduce_group(reports, statuses)
        summary["configuration_id"] = cell_id
        summary["workload_id"] = profile_id
        summary["cell"] = {key: CELL_BY_ID[cell_id][key] for key in ("id", *CELL_VARIABLES)}
        summary["profile"] = PROFILE_BY_ID[profile_id]
        if all(status["steady_state_eligible"] for status in statuses):
            steady.append(summary)
        else:
            summary["steady_state"] = False
            summary["presentation"] = (
                "bounded opportunity horizon; not a full-duration steady-state experiment"
            )
            bounded.append(summary)
    fairness = [
        fairness_record(environment, report, {"workload_id": trial["profile_id"]})
        for trial, report, _status in loaded
        if not trial["pilot"]
    ]
    if protocol["comparison_class"] == "primary":
        by_profile: dict[str, list[dict]] = {}
        for record in fairness:
            by_profile.setdefault(str(record["profile_id"]), []).append(record)
        for profile_id, records in by_profile.items():
            for other in records[1:]:
                differences = fairness_differences(records[0], other)
                if differences:
                    raise AnalyzerError(
                        f"profile {profile_id} is not a fair comparison set: " + ", ".join(differences)
                    )
    document = {
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "source_schema_version": 3,
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "run_set_id": run_set.name,
        "comparison_class": protocol["comparison_class"],
        "pilots_excluded": True,
        "counted_runs_per_cell": DEFAULT_COUNTED_RUNS,
        "confidence_interval_computed": False,
        "winner_declared": False,
        "minimum_submit_fraction": MINIMUM_SUBMIT_FRACTION,
        "steady_state_groups": steady,
        "bounded_opportunity_groups": bounded,
        "primary_comparisons_are_not_mixed_with_exploratory": True,
        "limitations": list(LIMITATIONS),
        "model_sha256": MODEL_SHA256,
        "git_commit": environment["queried"]["git_commit"],
    }
    if protocol["comparison_class"] != "primary":
        document["steady_state_groups"] = []
        document["exploratory_groups"] = steady + bounded
        document["bounded_opportunity_groups"] = []
        document["publication_eligible"] = False
    else:
        document["exploratory_groups"] = []
        document["publication_eligible"] = True
    _write_aggregate(aggregate_path, document)
    return document


def _write_aggregate(path: pathlib.Path, document: dict) -> None:
    payload = json.dumps(document, indent=2) + "\n"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise AnalyzerError(f"refusing to overwrite existing aggregate: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = analyze(pathlib.Path(args.run_set).resolve())
    except AnalyzerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(document["run_set_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
