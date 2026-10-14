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
import math
import os
import pathlib
import random
import statistics
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.collect_scheduler import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CELL_BY_ID,
    CONFIDENCE_LEVEL,
    DEFAULT_COUNTED_RUNS,
    LIMITATIONS,
    MAX_COUNTED_RUNS,
    MINIMUM_INTERVAL_BLOCKS,
    MINIMUM_SUBMIT_FRACTION,
    PUBLICATION_POLICY_3,
    MODEL_SHA256,
    PREFERRED_COUNTED_RUNS,
    PROFILE_BY_ID,
    PROTOCOL1_PUBLICATION_POLICY,
    PROTOCOL_2_LIMITATIONS,
    PUBLICATION_POLICY,
    STATISTICAL_METHOD,
    TIMING_QUALITY,
    TIMING_QUALITY_NOTE,
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
    "schema_version",
    "analysis_protocol_version",
    "enable_completion_timing",
    "latency_metric",
    "arrival_fidelity_p95_max_nanoseconds",
    "arrival_fidelity_max_lag_nanoseconds",
    "steady_clock_period_numerator",
    "steady_clock_period_denominator",
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
        "scheduled_opportunities": report["scheduled_opportunities"],
        "scheduled_span_nanoseconds": report["scheduled_span_nanoseconds"],
        "arrival_stop_reason": report["arrival_stop_reason"],
        "enable_batching": report["scheduler"]["enable_batching"],
        "max_batch_size": report["scheduler"]["max_batch_size"],
        "max_batch_delay_ms": report["scheduler"]["max_batch_delay_ms"],
        "schema_version": report.get("schema_version"),
        "analysis_protocol_version": protocol.get("analysis_protocol_version"),
        "enable_completion_timing": report.get("scheduler", {}).get("enable_completion_timing", False),
        "latency_metric": protocol.get("latency_metric"),
        "arrival_fidelity_p95_max_nanoseconds": protocol.get("arrival_fidelity_p95_max_nanoseconds"),
        "arrival_fidelity_max_lag_nanoseconds": protocol.get("arrival_fidelity_max_lag_nanoseconds"),
        "steady_clock_period_numerator": report.get("steady_clock_period_numerator"),
        "steady_clock_period_denominator": report.get("steady_clock_period_denominator"),
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
    expected = {
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
    version = protocol.get("analysis_protocol_version")
    if version is not None:
        expected["analysis_protocol_version"] = version
    expected["latency_metric"] = protocol.get("latency_metric")
    expected["arrival_fidelity_p95_max_nanoseconds"] = protocol.get(
        "arrival_fidelity_p95_max_nanoseconds"
    )
    expected["arrival_fidelity_max_lag_nanoseconds"] = protocol.get(
        "arrival_fidelity_max_lag_nanoseconds"
    )
    expected["arrival_rate_relative_tolerance"] = protocol.get("arrival_rate_relative_tolerance")
    if profile.get("target_rps") is not None:
        expected["target_rps"] = profile["target_rps"]
    return expected


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
    protocol_version = environment["collection_protocol"].get("analysis_protocol_version", 1)
    if environment["collection_protocol"]["comparison_class"] == "primary" and not trial["pilot"]:
        if protocol_version in {2, 3}:
            required = (
                "CLOSED_LOOP_VALID" if report["workload"] == "closed_loop" else "STRICT_PRIMARY"
            )
            if status.get("delivery_class") != required:
                raise AnalyzerError(
                    f"{trial_id} delivery class is {status.get('delivery_class')}, expected {required}"
                )
        elif not status["publication_eligible"]:
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


def _analyze_protocol1(run_set: pathlib.Path) -> dict:
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
        "analysis_protocol_version": 1,
        "publication_policy": PROTOCOL1_PUBLICATION_POLICY,
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


NARROW_CLAIM = (
    "comparison_supported means this metric's paired-block interval excludes zero. "
    "It does not mean the alternative is universally faster, better, or statistically superior."
)
BOOTSTRAP_LIMITATIONS = (
    "Blocks are treated as approximately exchangeable under the fixed rotation.",
    "Rotation does not remove host contention, thermal drift, clock drift, or CPU frequency changes.",
    "Ten blocks are not a guarantee of statistical power.",
    "The percentile bootstrap is not a correction for systematic bias.",
    "Requests within one trial are not independent experimental blocks.",
    "The interval describes a run-level mean paired difference on this protocol, binary, and host.",
    "It is not evidence of universal performance.",
    "Practical importance is separate from whether the interval excludes zero.",
    "Generator lag is recorded only as count, minimum, maximum, and mean. "
    "submitted_lag_max_within_configured_tolerance is not a percentile arrival distribution.",
    "Small-sample intervals can be unstable.",
)


def _policy_values(environment: dict) -> list:
    protocol = environment.get("collection_protocol")
    source = environment.get("source_defined_policy")
    policies = []
    if isinstance(protocol, dict):
        policies.append(protocol.get("publication_policy"))
    if isinstance(source, dict):
        policies.append(source.get("publication_policy"))
    return policies


def _marks_protocol3(environment: dict) -> bool:
    return PUBLICATION_POLICY_3 in _policy_values(environment)


def _marks_protocol2(environment: dict) -> bool:
    return PUBLICATION_POLICY in _policy_values(environment)


def analysis_protocol_version(environment: dict) -> int:
    """Missing version is protocol 1 only when the run set does not claim protocol 2."""
    protocol = environment.get("collection_protocol")
    version = protocol.get("analysis_protocol_version") if isinstance(protocol, dict) else None
    if _marks_protocol3(environment) and version != 3:
        raise AnalyzerError(
            "protocol-3 publication policy is recorded but analysis_protocol_version is "
            f"{version!r}; refusing to downgrade the publication policy"
        )
    if _marks_protocol2(environment) and version != 2:
        raise AnalyzerError(
            "protocol-2 publication policy is recorded but analysis_protocol_version is "
            f"{version!r}; refusing to downgrade the publication policy"
        )
    if version is None:
        return 1
    if version not in {1, 2, 3}:
        raise AnalyzerError(f"analysis_protocol_version {version!r} is not supported")
    return int(version)


def _finite(value: object) -> bool:
    if isinstance(value, bool) or type(value) not in {int, float}:
        return False
    return math.isfinite(float(value))


def _interpolated_quantile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def paired_block_bootstrap_mean(
    differences: list[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    level: float = CONFIDENCE_LEVEL,
) -> dict | None:
    """Percentile interval for the mean of paired block differences.

    Each replicate resamples blocks with replacement. Requests are not resamples.
    """
    if len(differences) < MINIMUM_INTERVAL_BLOCKS:
        return None
    if not 0.0 < level < 1.0:
        raise AnalyzerError("confidence level must be between 0 and 1")
    if replicates < 1:
        raise AnalyzerError("bootstrap replicates must be positive")
    generator = random.Random(seed)
    width = len(differences)
    means: list[float] = []
    for _replicate in range(replicates):
        total = 0.0
        for _draw in range(width):
            total += differences[generator.randrange(width)]
        means.append(total / width)
    means.sort()
    tail = (1.0 - level) / 2.0
    return {
        "low": _interpolated_quantile(means, tail),
        "high": _interpolated_quantile(means, 1.0 - tail),
        "level": level,
        "method": "percentile_linear_interpolation",
    }


def _interval_excludes_zero(interval: dict) -> bool:
    return float(interval["high"]) < 0.0 or float(interval["low"]) > 0.0


def paired_metric_effect(
    baseline: list[float | None],
    alternative: list[float | None],
    *,
    units: str,
    allow_interval: bool,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    level: float = CONFIDENCE_LEVEL,
) -> dict:
    notes = [NARROW_CLAIM, *BOOTSTRAP_LIMITATIONS]
    result = {
        "effect_estimate": None,
        "median_paired_difference": None,
        "relative_effect_if_defined": None,
        "confidence_interval": None,
        "confidence_interval_computed": False,
        "comparison_supported": False,
        "superiority_claimed": False,
        "paired_block_count": 0,
        "units": units,
        "interval_unavailable_reason": None,
        "limitation_notes": notes,
    }
    if len(baseline) != len(alternative) or not baseline:
        result["interval_unavailable_reason"] = "the paired blocks are incomplete"
        return result
    if any(not _finite(value) for value in baseline + alternative):
        result["interval_unavailable_reason"] = "a paired sample is missing or nonfinite"
        return result
    differences = [float(alternative[index]) - float(baseline[index]) for index in range(len(baseline))]
    mean_difference = sum(differences) / len(differences)
    result["effect_estimate"] = mean_difference
    result["median_paired_difference"] = float(statistics.median(differences))
    result["paired_block_count"] = len(differences)
    if all(float(value) > 0.0 for value in baseline):
        result["relative_effect_if_defined"] = sum(
            100.0 * differences[index] / float(baseline[index]) for index in range(len(baseline))
        ) / len(baseline)
        result["relative_effect_definition"] = "mean of per-block 100 * difference / baseline"
    else:
        result["relative_effect_if_defined"] = None
        result["relative_effect_definition"] = "mean of per-block 100 * difference / baseline"
        notes.append(
            "The relative percentage is undefined because a baseline measurement is not positive."
        )
    if not allow_interval:
        result["interval_unavailable_reason"] = (
            f"an interval requires at least {MINIMUM_INTERVAL_BLOCKS} strict full-duration blocks"
        )
        return result
    interval = paired_block_bootstrap_mean(
        differences,
        seed=seed,
        replicates=replicates,
        level=level,
    )
    result["confidence_interval"] = interval
    result["confidence_interval_computed"] = interval is not None
    if interval is None:
        result["interval_unavailable_reason"] = (
            f"fewer than {MINIMUM_INTERVAL_BLOCKS} complete blocks"
        )
        return result
    if _interval_excludes_zero(interval):
        result["comparison_supported"] = True
    else:
        notes.append("The confidence interval includes zero.")
    return result


def _delivery_fraction(report: dict) -> float | None:
    scheduled = report["scheduled_opportunities"]
    if scheduled == 0:
        return None
    return report["submit_attempts"] / scheduled


def _successful_completion_fraction(report: dict) -> float | None:
    """Succeeded divided by accepted. Rejections are outside this denominator."""
    accepted = report["accepted"]
    if accepted == 0:
        return None
    return report["succeeded"] / accepted


def _rejection_fraction(report: dict) -> float | None:
    attempts = report["submit_attempts"]
    if attempts == 0:
        return None
    return _rejection_rate(report)


def _latency_value(report: dict, key: str) -> float | None:
    latency = report["successful_e2e_latency"]
    if latency["percentiles_reported"] is not True:
        return None
    value = latency[key]
    if not _finite(value):
        return None
    return float(value)


def _p99_value(report: dict) -> float | None:
    latency = report["successful_e2e_latency"]
    if latency["precise_tail"] is not True or latency["sample_count"] < P99_SAMPLE_FLOOR:
        return None
    return _latency_value(report, "p99_microseconds")


def plan_identity_differences(
    left_report: dict,
    right_report: dict,
    left_status: dict,
    right_status: dict,
) -> list[str]:
    differences = []
    if left_report["scheduled_opportunities"] != right_report["scheduled_opportunities"]:
        differences.append("scheduled_opportunities")
    if left_report["scheduled_span_nanoseconds"] != right_report["scheduled_span_nanoseconds"]:
        differences.append("scheduled_span_nanoseconds")
    if left_status.get("window_class") != right_status.get("window_class"):
        differences.append("window_class")
    if left_report["arrival_stop_reason"] != right_report["arrival_stop_reason"]:
        differences.append("arrival_stop_reason")
    if left_report["arrival_stop_reason"] != "configured_duration":
        differences.append("full_duration")
    return differences


def _timing_value(report: dict, object_name: str, key: str) -> float | None:
    summary = report[object_name]
    if summary["percentiles_reported"] is not True:
        return None
    value = summary[key]
    if not _finite(value):
        return None
    return float(value)


def _timing_p99(report: dict, object_name: str) -> float | None:
    summary = report[object_name]
    if summary["precise_tail"] is not True or summary["sample_count"] < P99_SAMPLE_FLOOR:
        return None
    return _timing_value(report, object_name, "p99_nanoseconds")


def _paired_metric_catalog(protocol_version: int = 2) -> list[tuple[str, str, object]]:
    catalog = [
        (
            "successful_offered_cohort_rate",
            "successful requests per arrival-window second",
            lambda report: report["throughput"]["successful_attempts_per_arrival_window_second"],
        ),
        (
            "successful_total_trial_rate",
            "successful requests per total-trial second",
            lambda report: report["throughput"]["successful_per_total_trial_second"],
        ),
        (
            "p50_e2e_latency",
            "microseconds",
            lambda report: _latency_value(report, "p50_microseconds"),
        ),
        (
            "p95_e2e_latency",
            "microseconds",
            lambda report: _latency_value(report, "p95_microseconds"),
        ),
        (
            "scheduler_rejection_fraction",
            "rejections divided by submit_attempts",
            _rejection_fraction,
        ),
        (
            "successful_completion_fraction",
            "succeeded divided by accepted",
            _successful_completion_fraction,
        ),
        (
            "generator_delivery_fraction",
            "submit_attempts divided by scheduled_opportunities",
            _delivery_fraction,
        ),
        (
            "observed_mean_batch_size",
            "requests per engine run",
            _mean_batch_size,
        ),
        (
            "engine_run_count",
            "engine runs",
            lambda report: report["metrics"]["engine_run_count"],
        ),
    ]
    if protocol_version == 3:
        catalog.extend(
            [
                (
                    "p50_pre_publication_latency",
                    "nanoseconds",
                    lambda report: _timing_value(
                        report, "pre_publication_latency", "p50_nanoseconds"
                    ),
                ),
                (
                    "p95_pre_publication_latency",
                    "nanoseconds",
                    lambda report: _timing_value(
                        report, "pre_publication_latency", "p95_nanoseconds"
                    ),
                ),
                (
                    "p50_pre_publication_to_observation_delay",
                    "nanoseconds; includes set_value and collector delay",
                    lambda report: _timing_value(
                        report, "pre_publication_to_observation_delay", "p50_nanoseconds"
                    ),
                ),
            ]
        )
    return catalog


def _verify_block_inventory(plan: list[dict], loaded: list[tuple[dict, dict, dict, dict]]) -> None:
    meta_by_id = {str(trial["trial_id"]): meta for trial, _report, _status, meta in loaded}
    seen: dict[tuple[object, object, object], str] = {}
    for trial in plan:
        trial_id = str(trial["trial_id"])
        meta = meta_by_id.get(trial_id)
        if meta is None:
            raise AnalyzerError(f"missing block evidence for {trial_id}")
        if bool(meta.get("pilot")) != bool(trial["pilot"]):
            raise AnalyzerError(f"{trial_id} pilot flag does not match the block plan")
        if meta.get("run_index") != trial["run_number"]:
            raise AnalyzerError(f"{trial_id} run_index does not match the plan")
        if meta.get("execution_index") != trial["execution_index"]:
            raise AnalyzerError(f"{trial_id} execution_index does not match the plan")
        if meta.get("counted_run_target") != trial["counted_run_target"]:
            raise AnalyzerError(f"{trial_id} counted_run_target does not match the plan")
        if trial["pilot"]:
            continue
        if meta.get("block_index") != trial["block_index"]:
            raise AnalyzerError(f"{trial_id} block_index does not match the plan")
        if meta.get("position_in_block") != trial["position_in_block"]:
            raise AnalyzerError(f"{trial_id} position_in_block does not match the plan")
        if meta.get("configuration_id") != trial["cell_id"]:
            raise AnalyzerError(f"{trial_id} configuration_id does not match the plan")
        if meta.get("workload_id") != trial["profile_id"]:
            raise AnalyzerError(f"{trial_id} profile_id does not match the plan")
        key = (trial["profile_id"], trial["block_index"], trial["cell_id"])
        if key in seen:
            raise AnalyzerError(f"duplicated block identity {key[0]}/{key[1]}/{key[2]}")
        seen[key] = trial_id
    expected = {
        (trial["profile_id"], trial["block_index"], trial["cell_id"])
        for trial in plan
        if not trial["pilot"]
    }
    if set(seen) != expected:
        missing = expected - set(seen)
        raise AnalyzerError(f"missing paired block: {sorted(missing)!r}")


def _build_contrast(
    *,
    profile: dict,
    baseline: str,
    alternative: str,
    counted_runs: int,
    loaded: list[tuple[dict, dict, dict, dict]],
    allow_interval: bool,
    seed: int,
    replicates: int,
    level: float,
    protocol_version: int = 2,
    latency_metric: str | None = None,
    fidelity_passed: bool = True,
) -> dict:
    profile_id = profile["id"]
    by_block: dict[int, dict[str, tuple[dict, dict]]] = {}
    for trial, report, status, _meta in loaded:
        if trial["pilot"] or trial["profile_id"] != profile_id:
            continue
        if trial["cell_id"] not in {baseline, alternative}:
            continue
        slot = by_block.setdefault(int(trial["block_index"]), {})
        if trial["cell_id"] in slot:
            raise AnalyzerError(
                f"duplicated cell {trial['cell_id']} in {profile_id} block {trial['block_index']}"
            )
        slot[str(trial["cell_id"])] = (report, status)
    baseline_reports = []
    alternative_reports = []
    for block_index in range(1, counted_runs + 1):
        slot = by_block.get(block_index)
        if slot is None or baseline not in slot or alternative not in slot:
            raise AnalyzerError(f"incomplete paired block {profile_id}/{block_index}")
        left_report, left_status = slot[baseline]
        right_report, right_status = slot[alternative]
        if profile["family"] == "open_loop":
            differences = plan_identity_differences(
                left_report, right_report, left_status, right_status
            )
            if differences:
                raise AnalyzerError(
                    f"profile {profile_id} block {block_index} is not a matched plan: "
                    + ", ".join(differences)
                )
            if left_status.get("delivery_class") != "STRICT_PRIMARY":
                raise AnalyzerError(f"baseline block {block_index} is not strict primary")
            if right_status.get("delivery_class") != "STRICT_PRIMARY":
                raise AnalyzerError(f"alternative block {block_index} is not strict primary")
        baseline_reports.append(left_report)
        alternative_reports.append(right_report)
    metrics = {}
    open_loop = allow_interval and profile["family"] == "open_loop"

    def metric_interval(name: str) -> bool:
        if protocol_version != 3:
            return open_loop
        if "observation_delay" in name:
            return False
        producer = name in {"p50_e2e_latency", "p95_e2e_latency", "p99_e2e_latency"}
        pre_publication = "pre_publication_latency" in name
        if not producer and not pre_publication:
            return open_loop
        if not fidelity_passed or latency_metric is None:
            return False
        if latency_metric == "producer_observed":
            return open_loop and producer
        if latency_metric == "pre_publication":
            return open_loop and pre_publication
        return False

    for name, units, extractor in _paired_metric_catalog(protocol_version):
        metrics[name] = paired_metric_effect(
            [extractor(report) for report in baseline_reports],
            [extractor(report) for report in alternative_reports],
            units=units,
            allow_interval=metric_interval(name),
            seed=seed,
            replicates=replicates,
            level=level,
        )
    p99_baseline = [_p99_value(report) for report in baseline_reports]
    p99_alternative = [_p99_value(report) for report in alternative_reports]
    if any(value is None for value in p99_baseline + p99_alternative):
        metrics["p99_e2e_latency"] = {
            "effect_estimate": None,
            "confidence_interval": None,
            "confidence_interval_computed": False,
            "comparison_supported": False,
            "superiority_claimed": False,
            "p99_omitted_reason": (
                "at least one run has precise_tail false or fewer than 100 successful samples"
            ),
            "limitation_notes": [NARROW_CLAIM],
        }
    else:
        metrics["p99_e2e_latency"] = paired_metric_effect(
            p99_baseline,
            p99_alternative,
            units="microseconds",
            allow_interval=metric_interval("p99_e2e_latency"),
            seed=seed,
            replicates=replicates,
            level=level,
        )
    if protocol_version == 3:
        pre_p99_baseline = [
            _timing_p99(report, "pre_publication_latency") for report in baseline_reports
        ]
        pre_p99_alternative = [
            _timing_p99(report, "pre_publication_latency") for report in alternative_reports
        ]
        if any(value is None for value in pre_p99_baseline + pre_p99_alternative):
            metrics["p99_pre_publication_latency"] = {
                "effect_estimate": None,
                "confidence_interval": None,
                "confidence_interval_computed": False,
                "comparison_supported": False,
                "superiority_claimed": False,
                "p99_omitted_reason": (
                    "at least one run has precise_tail false or fewer than 100 successful samples"
                ),
                "limitation_notes": [NARROW_CLAIM],
            }
        else:
            metrics["p99_pre_publication_latency"] = paired_metric_effect(
                pre_p99_baseline,
                pre_p99_alternative,
                units="nanoseconds",
                allow_interval=metric_interval("p99_pre_publication_latency"),
                seed=seed,
                replicates=replicates,
                level=level,
            )
        metrics["p99_e2e_latency"]["p99_reduction"] = "paired_difference_of_run_p99"
    return {
        "profile_id": profile_id,
        "workload": profile["workload"],
        "baseline_cell": baseline,
        "alternative_cell": alternative,
        "paired_block_count": counted_runs,
        "winner_declared": False,
        "superiority_claimed": False,
        "metrics": metrics,
        "arrival_window_nanoseconds": {
            "baseline": [report["arrival_window_nanoseconds"] for report in baseline_reports],
            "alternative": [report["arrival_window_nanoseconds"] for report in alternative_reports],
        },
        "completion_nanoseconds": {
            "baseline": [report["completion_nanoseconds"] for report in baseline_reports],
            "alternative": [report["completion_nanoseconds"] for report in alternative_reports],
        },
        "fraction_definitions": {
            "submission_fraction": "submit_attempts / scheduled_opportunities",
            "scheduler_acceptance_fraction": "accepted / submit_attempts",
            "scheduler_rejection_fraction": "(full + closed + not_running) / submit_attempts",
            "successful_completion_fraction": "succeeded / accepted",
        },
        "generator_delivery_fraction": metrics["generator_delivery_fraction"]["effect_estimate"],
        "scheduler_rejection_fraction": metrics["scheduler_rejection_fraction"]["effect_estimate"],
        "successful_completion_fraction": metrics["successful_completion_fraction"]["effect_estimate"],
    }


def _analyze_protocol2(run_set: pathlib.Path) -> dict:
    if not run_set.is_dir():
        raise AnalyzerError(f"run-set directory does not exist: {run_set}")
    aggregate_path = run_set / "aggregate.json"
    if aggregate_path.exists():
        raise AnalyzerError(f"refusing to overwrite existing aggregate: {aggregate_path}")
    environment = load_environment(run_set)
    require_completed(environment, run_set)
    protocol = _object(environment.get("collection_protocol"), "collection_protocol")
    counted_runs = protocol.get("counted_runs_per_cell")
    if type(counted_runs) is not int or counted_runs < 1 or counted_runs > MAX_COUNTED_RUNS:
        raise AnalyzerError("protocol 2 counted runs must be an integer from 1 through 30")
    if protocol.get("counted_run_target") != counted_runs:
        raise AnalyzerError("counted_run_target does not match counted_runs_per_cell")
    cells = [CELL_BY_ID[cell_id] for cell_id in protocol["cells"]]
    profiles = [PROFILE_BY_ID[profile_id] for profile_id in protocol["profiles"]]
    plan = execution_plan(cells, profiles, counted_runs, 2)
    if protocol.get("trial_order") != [trial["trial_id"] for trial in plan]:
        raise AnalyzerError("trial order does not match the protocol 2 block plan")
    if environment["verified_configuration"]["model_sha256"] != MODEL_SHA256:
        raise AnalyzerError("run set model hash is not the pinned batch_add fixture")
    if environment["queried"]["benchmark_tool_version"] != TOOL_VERSION:
        raise AnalyzerError("benchmark tool version does not match the collector")
    report_names = {
        path.name[: -len(".json")]
        for path in (run_set / "runs").glob("*.json")
        if not path.name.endswith(".meta.json")
    }
    expected_names = {str(trial["trial_id"]) for trial in plan}
    if report_names != expected_names:
        raise AnalyzerError("trial reports do not match the protocol 2 plan")
    loaded = []
    for trial in plan:
        report, status = _load_trial(run_set, environment, trial)
        meta_path = run_set / "runs" / f"{trial['trial_id']}.meta.json"
        meta = _object(json.loads(meta_path.read_text(encoding="utf-8")), str(trial["trial_id"]))
        loaded.append((trial, report, status, meta))
    _verify_block_inventory(plan, loaded)
    families = {item[1]["workload"] == "closed_loop" for item in loaded}
    if len(families) != 1:
        raise AnalyzerError("open-loop and closed-loop trials are in one run set")
    groups: dict[tuple[str, str], list[tuple[dict, dict, dict]]] = {}
    for trial, report, status, _meta in loaded:
        if trial["pilot"]:
            continue
        key = (str(trial["cell_id"]), str(trial["profile_id"]))
        groups.setdefault(key, []).append((trial, report, status))
    steady = []
    bounded = []
    closed_groups = []
    for (cell_id, profile_id), items in groups.items():
        if len(items) != counted_runs:
            raise AnalyzerError(f"{cell_id}/{profile_id} does not have {counted_runs} counted runs")
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
        if reports[0]["workload"] == "closed_loop":
            summary["steady_state"] = False
            summary["presentation"] = "closed-loop trials are not an open-loop arrival process"
            closed_groups.append(summary)
        elif all(status["steady_state_eligible"] for status in statuses):
            steady.append(summary)
        else:
            summary["steady_state"] = False
            summary["presentation"] = (
                "bounded opportunity horizon; not a full-duration steady-state experiment"
            )
            bounded.append(summary)
    fairness = [
        fairness_record(environment, report, {"workload_id": trial["profile_id"]})
        for trial, report, _status, _meta in loaded
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
    protocol_number = analysis_protocol_version(environment)
    baseline = protocol.get("baseline_cell")
    alternative = protocol.get("alternative_cell")
    contrast_source = protocol.get("contrast_source")
    contrasts = []
    interval_allowed = (
        protocol["comparison_class"] == "primary"
        and counted_runs >= MINIMUM_INTERVAL_BLOCKS
        and contrast_source in {"predeclared", "inferred_two_cell_order"}
        and baseline in CELL_BY_ID
        and alternative in CELL_BY_ID
        and not bounded
        and not closed_groups
    )
    if contrast_source == "descriptive_only" or baseline is None or alternative is None:
        interval_allowed = False
    elif len(protocol["cells"]) > 2 and contrast_source != "predeclared":
        interval_allowed = False
    if (
        protocol["comparison_class"] == "primary"
        and baseline is not None
        and alternative is not None
        and contrast_source != "descriptive_only"
    ):
        seed = int(protocol.get("bootstrap_seed", BOOTSTRAP_SEED))
        replicates = int(protocol.get("bootstrap_replicates", BOOTSTRAP_REPLICATES))
        level = float(protocol.get("confidence_level", CONFIDENCE_LEVEL))
        fidelity_passed = protocol_number != 3 or all(
            status.get("arrival_timing_fidelity") == "passed"
            for trial, _report, status, _meta in loaded
            if not trial["pilot"]
        )
        for profile in profiles:
            contrasts.append(
                _build_contrast(
                    profile=profile,
                    baseline=str(baseline),
                    alternative=str(alternative),
                    counted_runs=counted_runs,
                    loaded=loaded,
                    allow_interval=interval_allowed,
                    seed=seed,
                    replicates=replicates,
                    level=level,
                    protocol_version=protocol_number,
                    latency_metric=protocol.get("latency_metric"),
                    fidelity_passed=fidelity_passed,
                )
            )
    interval_computed = any(
        metric.get("confidence_interval_computed") is True
        for contrast in contrasts
        for metric in contrast["metrics"].values()
    )
    protocol_number = analysis_protocol_version(environment)
    document = {
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "source_schema_version": 4 if protocol_number == 3 else 3,
        "analysis_protocol_version": protocol_number,
        "publication_policy": PUBLICATION_POLICY_3 if protocol_number == 3 else PUBLICATION_POLICY,
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "run_set_id": run_set.name,
        "comparison_class": protocol["comparison_class"],
        "pilots_excluded": True,
        "counted_runs_per_cell": counted_runs,
        "counted_run_target": protocol.get("counted_run_target", counted_runs),
        "preferred_counted_runs": PREFERRED_COUNTED_RUNS,
        "minimum_interval_blocks": MINIMUM_INTERVAL_BLOCKS,
        "block_protocol": protocol.get("block_protocol"),
        "baseline_cell": baseline,
        "alternative_cell": alternative,
        "contrast_source": contrast_source,
        "statistical_method": STATISTICAL_METHOD,
        "bootstrap_seed": protocol.get("bootstrap_seed", BOOTSTRAP_SEED),
        "bootstrap_replicates": protocol.get("bootstrap_replicates", BOOTSTRAP_REPLICATES),
        "confidence_level": protocol.get("confidence_level", CONFIDENCE_LEVEL),
        "timing_quality": TIMING_QUALITY,
        "timing_quality_note": TIMING_QUALITY_NOTE,
        "lag_tolerance_ms": protocol["measurement"]["lag_tolerance_ms"],
        "arrival_timing_fidelity": "not_certified",
        "full_generator_delivery_is_not_arrival_timing_fidelity": True,
        "calibration_status": "not_calibrated",
        "confidence_interval_computed": interval_computed,
        "comparison_supported": False,
        "superiority_claimed": False,
        "winner_declared": False,
        "strict_generator_delivery": 1.0,
        "steady_state_groups": steady,
        "bounded_opportunity_groups": bounded,
        "closed_loop_groups": closed_groups,
        "contrasts": contrasts,
        "primary_comparisons_are_not_mixed_with_exploratory": True,
        "limitations": list(PROTOCOL_2_LIMITATIONS) + list(BOOTSTRAP_LIMITATIONS),
        "model_sha256": MODEL_SHA256,
        "git_commit": environment["queried"]["git_commit"],
    }
    if protocol["comparison_class"] != "primary":
        document["steady_state_groups"] = []
        document["exploratory_groups"] = steady + bounded + closed_groups
        document["bounded_opportunity_groups"] = []
        document["closed_loop_groups"] = []
        document["publication_eligible"] = False
        document["contrasts"] = []
    else:
        document["exploratory_groups"] = []
        document["publication_eligible"] = not bounded and not closed_groups and bool(steady)
    if protocol_number == 3:
        certified = all(
            status.get("arrival_timing_fidelity") == "passed"
            for trial, _report, status, _meta in loaded
            if not trial["pilot"]
        )
        document["latency_metric"] = protocol.get("latency_metric")
        document["arrival_timing_fidelity"] = "passed" if certified else "not_certified"
        document["phase_boundary_fidelity"] = "unavailable"
        document["instrumentation_overhead_calibrated"] = False
        document["limitations"] = list(document["limitations"]) + [
            "phase_boundary_fidelity is unavailable; signed lag percentiles do not certify burst waveform fidelity",
            "instrumentation overhead is not calibrated and is not subtracted from latency",
            "pre_publication_to_observation_delay includes set_value and collector delay; it is not pure polling overhead",
            "pre_publication_latency is not pure engine execution time",
        ]
    _write_aggregate(aggregate_path, document)
    return document


def analyze(run_set: pathlib.Path) -> dict:
    if not run_set.is_dir():
        raise AnalyzerError(f"run-set directory does not exist: {run_set}")
    environment = load_environment(run_set)
    version = analysis_protocol_version(environment)
    if version == 1:
        return _analyze_protocol1(run_set)
    return _analyze_protocol2(run_set)


def _write_aggregate(path: pathlib.Path, document: dict) -> None:
    payload = json.dumps(document, indent=2, allow_nan=False) + "\n"
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
