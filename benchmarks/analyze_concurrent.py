#!/usr/bin/env python3
"""Reduce one completed concurrent run set to aggregate.json.

This module does not launch inferedge-bench-concurrent and does not run a
model. It reads the files collect_concurrent.py already wrote, checks every
pilot and counted trial, and publishes one aggregate. Pilot measurements stay
on disk and are excluded from the reductions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import sys

# `python3 benchmarks/analyze_concurrent.py` puts this directory on sys.path.
# The package import needs the repository root. `python3 -m` already provides it.
if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.collect_concurrent import (
    ADD_MODEL_SHA256,
    AGGREGATE_RELATIVE_TOLERANCE,
    APPROVED_WORKER_COUNTS,
    BACKEND,
    CANONICAL_CONFIGURATIONS,
    CORRECTNESS_CHECK_INTERVAL,
    COUNTED_RUNS_PER_CONFIGURATION,
    GRAPH_OPTIMIZATIONS,
    INTER_TRIAL_DELAY_SECONDS,
    MEASURED_ITERATIONS_PER_WORKER,
    MNIST_FIXTURE_ID,
    MNIST_FIXTURE_MANIFEST_SHA256,
    MNIST_INPUT_SHA256,
    MNIST_MODEL_SHA256,
    MNIST_REFERENCE_SHA256,
    ONNX_INTRA_OP_THREADS_PER_SESSION,
    ONNXRUNTIME_VERSION,
    ORDERING_TOLERANCE,
    SESSION_OWNERSHIP,
    TOOL_VERSION,
    WARMUP_ITERATIONS_PER_WORKER,
    execution_plan,
    trial_order,
    validate_concurrent_report,
)

AGGREGATE_SCHEMA_VERSION = 2
SOURCE_SCHEMA_VERSION = 2
EVIDENCE_SUFFIXES = (".stdout", ".stderr", ".meta.json", ".json")


class AnalyzerError(Exception):
    """The run set cannot be reduced. No aggregate is published."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one completed inferedge-bench-concurrent run set. "
            "This command does not launch a benchmark."
        )
    )
    parser.add_argument(
        "--run-set",
        required=True,
        help="Path to one benchmarks/concurrent_runs/<run-set-id> directory",
    )
    return parser


def _object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise AnalyzerError(f"{label} is not a JSON object")
    return value


def _exact(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise AnalyzerError(f"{label} is {value!r}, expected {expected!r}")


def require_plan() -> list[dict[str, object]]:
    """Use the collector plan. Reject a protocol that is no longer the approved campaign."""
    plan = execution_plan()
    identities = [str(trial["trial_id"]) for trial in plan]
    pilots = [trial for trial in plan if trial["pilot"] is True]
    counted = [trial for trial in plan if trial["pilot"] is False]
    if (
        len(plan) != 48
        or len(identities) != len(set(identities))
        or identities != trial_order()
        or len(pilots) != 8
        or len(counted) != 40
        or COUNTED_RUNS_PER_CONFIGURATION != 5
        or WARMUP_ITERATIONS_PER_WORKER != 100
        or MEASURED_ITERATIONS_PER_WORKER != 1000
        or CORRECTNESS_CHECK_INTERVAL != 1
        or INTER_TRIAL_DELAY_SECONDS != 5.0
        or tuple(APPROVED_WORKER_COUNTS) != (1, 2, 4, 8)
        or COUNTED_RUNS_PER_CONFIGURATION * len(CANONICAL_CONFIGURATIONS) != 40
    ):
        raise AnalyzerError("collector execution plan is not the approved 48-trial protocol")
    if any(str(trial["trial_id"]).startswith("pilot-") for trial in counted):
        raise AnalyzerError("a counted trial id uses the pilot prefix")
    if any(not str(trial["trial_id"]).startswith("pilot-") for trial in pilots):
        raise AnalyzerError("a pilot trial id is missing the pilot prefix")
    return plan


def load_environment(run_set: pathlib.Path) -> dict:
    if not run_set.is_dir():
        raise AnalyzerError(f"run-set directory does not exist: {run_set}")
    path = run_set / "environment.json"
    if not path.is_file():
        raise AnalyzerError(f"environment.json is missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AnalyzerError(f"environment.json could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AnalyzerError(f"environment.json is not valid JSON: {exc.msg}") from exc
    return _object(document, "environment.json")


def require_completed(environment: dict, run_set: pathlib.Path) -> None:
    status = environment.get("status")
    if status != "completed":
        raise AnalyzerError(f"collection status is {status!r}, expected 'completed'")
    failure = run_set / "failure.json"
    if failure.exists():
        raise AnalyzerError(f"failure.json is present: {failure}")
    if environment.get("aggregate") is not None:
        raise AnalyzerError(
            "environment.json aggregate field is not null. "
            "The collector leaves it null; analysis writes aggregate.json"
        )
    run_set_id = environment.get("run_set_id")
    if not isinstance(run_set_id, str) or run_set_id == "":
        raise AnalyzerError("run-set id is missing from environment.json")
    if run_set_id != run_set.name:
        raise AnalyzerError(
            f"run-set id is {run_set_id!r}, expected the directory name {run_set.name!r}"
        )


def require_protocol(environment: dict, plan: list[dict[str, object]]) -> dict:
    protocol = _object(environment.get("collection_protocol"), "collection_protocol")
    pilots = [str(trial["trial_id"]) for trial in plan if trial["pilot"] is True]
    _exact(protocol.get("workloads"), ["add", "mnist"], "collection_protocol.workloads")
    _exact(
        protocol.get("worker_counts"),
        list(APPROVED_WORKER_COUNTS),
        "collection_protocol.worker_counts",
    )
    _exact(
        protocol.get("warmup_iterations_per_worker"),
        WARMUP_ITERATIONS_PER_WORKER,
        "warmup_iterations_per_worker",
    )
    _exact(
        protocol.get("measured_iterations_per_worker"),
        MEASURED_ITERATIONS_PER_WORKER,
        "measured_iterations_per_worker",
    )
    _exact(
        protocol.get("correctness_check_interval"),
        CORRECTNESS_CHECK_INTERVAL,
        "correctness_check_interval",
    )
    _exact(
        protocol.get("counted_runs_per_configuration"),
        COUNTED_RUNS_PER_CONFIGURATION,
        "counted_runs_per_configuration",
    )
    _exact(protocol.get("pilot_count"), 8, "collection_protocol.pilot_count")
    _exact(protocol.get("counted_trial_count"), 40, "collection_protocol.counted_trial_count")
    _exact(protocol.get("process_count"), 48, "collection_protocol.process_count")
    _exact(protocol.get("pilots"), pilots, "collection_protocol.pilots")
    _exact(protocol.get("trial_order"), trial_order(), "collection_protocol.trial_order")
    if len(protocol["trial_order"]) != len(set(protocol["trial_order"])):
        raise AnalyzerError("collection_protocol.trial_order contains a duplicate trial identity")
    _exact(
        protocol.get("environment_variable"),
        "ORT_DISABLE_TELEMETRY=1",
        "collection_protocol.environment_variable",
    )
    _exact(
        protocol.get("inter_trial_delay_seconds"),
        INTER_TRIAL_DELAY_SECONDS,
        "inter_trial_delay_seconds",
    )
    _exact(protocol.get("ordering_tolerance"), ORDERING_TOLERANCE, "ordering_tolerance")
    _exact(
        protocol.get("aggregate_relative_tolerance"),
        AGGREGATE_RELATIVE_TOLERANCE,
        "aggregate_relative_tolerance",
    )
    return protocol


def require_provenance(environment: dict) -> tuple[dict, dict, dict]:
    queried = _object(environment.get("queried"), "queried")
    verified = _object(environment.get("verified_configuration"), "verified_configuration")
    policy = _object(environment.get("source_defined_policy"), "source_defined_policy")
    commit = queried.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise AnalyzerError("approved git revision metadata is missing or is not a full SHA")
    _exact(queried.get("benchmark_tool_version"), TOOL_VERSION, "benchmark_tool_version")
    _exact(verified.get("cmake_build_type"), "Release", "cmake_build_type")
    _exact(verified.get("sanitizer_instrumentation"), "absent", "sanitizer_instrumentation")
    _exact(verified.get("onnxruntime_version"), ONNXRUNTIME_VERSION, "onnxruntime_version")
    _exact(verified.get("add_model_sha256"), ADD_MODEL_SHA256, "add_model_sha256")
    _exact(verified.get("mnist_model_sha256"), MNIST_MODEL_SHA256, "mnist_model_sha256")
    _exact(verified.get("mnist_fixture_id"), MNIST_FIXTURE_ID, "mnist_fixture_id")
    _exact(
        verified.get("mnist_fixture_inputs_sha256"),
        MNIST_INPUT_SHA256,
        "mnist_fixture_inputs_sha256",
    )
    _exact(
        verified.get("mnist_fixture_reference_sha256"),
        MNIST_REFERENCE_SHA256,
        "mnist_fixture_reference_sha256",
    )
    _exact(
        verified.get("mnist_fixture_manifest_sha256"),
        MNIST_FIXTURE_MANIFEST_SHA256,
        "mnist_fixture_manifest_sha256",
    )
    for label in ("add_model_path", "mnist_model_path"):
        path = verified.get(label)
        if not isinstance(path, str) or path == "":
            raise AnalyzerError(f"{label} is missing")
    _exact(policy.get("backend"), BACKEND, "source_defined_policy.backend")
    _exact(policy.get("session_ownership"), SESSION_OWNERSHIP, "source_defined_policy.session_ownership")
    _exact(
        policy.get("onnx_intra_op_threads_per_session"),
        ONNX_INTRA_OP_THREADS_PER_SESSION,
        "source_defined_policy.onnx_intra_op_threads_per_session",
    )
    _exact(
        policy.get("graph_optimizations"),
        GRAPH_OPTIMIZATIONS,
        "source_defined_policy.graph_optimizations",
    )
    return queried, verified, policy


def require_evidence_files(run_set: pathlib.Path, plan: list[dict[str, object]]) -> None:
    runs = run_set / "runs"
    if not runs.is_dir():
        raise AnalyzerError(f"runs directory is missing: {runs}")
    expected = {
        f"{trial['trial_id']}{suffix}" for trial in plan for suffix in EVIDENCE_SUFFIXES
    }
    found = set()
    for path in runs.iterdir():
        if not path.is_file():
            raise AnalyzerError(f"unexpected trial artifact: {path.name}")
        found.add(path.name)
    missing = sorted(expected - found)
    if missing:
        name = missing[0]
        kind = "pilot" if name.startswith("pilot-") else "counted"
        raise AnalyzerError(f"missing {kind} report: {name}")
    extra = sorted(found - expected)
    if extra:
        raise AnalyzerError(f"unexpected trial artifact: {extra[0]}")


def load_metadata(
    run_set: pathlib.Path,
    trial: dict[str, object],
    seen: list[str],
) -> None:
    trial_id = str(trial["trial_id"])
    path = run_set / "runs" / f"{trial_id}.meta.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AnalyzerError(f"trial metadata for {trial_id} could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AnalyzerError(f"trial metadata for {trial_id} is not valid JSON: {exc.msg}") from exc
    meta = _object(meta, f"trial metadata for {trial_id}")
    claimed = meta.get("trial_id")
    if claimed in seen:
        raise AnalyzerError(f"duplicate trial identity: {claimed}")
    if claimed != trial_id:
        raise AnalyzerError(f"trial identity mismatch for {trial_id}: metadata says {claimed!r}")
    seen.append(str(claimed))
    if meta.get("workload") != trial["workload"]:
        raise AnalyzerError(
            f"trial {trial_id} workload metadata is {meta.get('workload')!r}, "
            f"expected {trial['workload']!r}"
        )
    if type(meta.get("worker_count")) is not int or meta.get("worker_count") != trial["worker_count"]:
        raise AnalyzerError(
            f"trial {trial_id} worker_count metadata is {meta.get('worker_count')!r}, "
            f"expected {trial['worker_count']!r}"
        )
    if meta.get("pilot") is not trial["pilot"]:
        raise AnalyzerError(f"trial {trial_id} pilot flag does not match the execution plan")
    if meta.get("accepted") is not True:
        raise AnalyzerError(f"trial {trial_id} is not an accepted measurement")
    if type(meta.get("returncode")) is not int or meta.get("returncode") != 0:
        raise AnalyzerError(f"trial {trial_id} return code is not 0")
    if meta.get("problems") != []:
        raise AnalyzerError(f"trial {trial_id} metadata records problems")


def load_trial_report(
    run_set: pathlib.Path,
    trial: dict[str, object],
    model_path: str,
) -> dict:
    trial_id = str(trial["trial_id"])
    workload = str(trial["workload"])
    worker_count = int(trial["worker_count"])  # type: ignore[arg-type]
    runs = run_set / "runs"
    try:
        stdout_bytes = (runs / f"{trial_id}.stdout").read_bytes()
        report_bytes = (runs / f"{trial_id}.json").read_bytes()
        stderr_bytes = (runs / f"{trial_id}.stderr").read_bytes()
    except OSError as exc:
        raise AnalyzerError(f"{trial_id}: could not read trial evidence: {exc}") from exc
    if stderr_bytes != b"":
        raise AnalyzerError(f"{trial_id}: stderr is not empty")
    if stdout_bytes != report_bytes:
        raise AnalyzerError(f"{trial_id}: stdout does not match the accepted report")
    try:
        text = report_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnalyzerError(f"{trial_id}: accepted report is not UTF-8") from exc
    problems = validate_concurrent_report(
        text,
        workload=workload,
        model_path=model_path,
        worker_count=worker_count,
    )
    if problems:
        raise AnalyzerError(f"{trial_id}: {'; '.join(problems)}")
    document = json.loads(text)
    if document.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise AnalyzerError(f"{trial_id}: schema_version is not {SOURCE_SCHEMA_VERSION}")
    return document


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalyzerError(f"{label} is not a JSON number")
    number = float(value)
    if not math.isfinite(number) or (math.copysign(1.0, number) < 0 and number == 0):
        raise AnalyzerError(f"{label} is not a finite number")
    return number


def finite_nonnegative(value: object, label: str) -> float:
    number = finite_number(value, label)
    if number < 0:
        raise AnalyzerError(f"{label} is negative")
    return number


def finite_positive(value: object, label: str) -> float:
    number = finite_nonnegative(value, label)
    if number <= 0:
        raise AnalyzerError(f"{label} is not positive")
    return number


def positive_median(values: list[float], label: str) -> float:
    if len(values) != COUNTED_RUNS_PER_CONFIGURATION:
        raise AnalyzerError(f"{label} does not have {COUNTED_RUNS_PER_CONFIGURATION} counted runs")
    median = float(statistics.median(values))
    if not math.isfinite(median) or median <= 0 or math.copysign(1.0, median) < 0:
        raise AnalyzerError(f"{label} denominator is not positive: {median}")
    return median


def relative_range(values: list[float], median: float, label: str) -> float:
    if not math.isfinite(median) or median <= 0 or math.copysign(1.0, median) < 0:
        raise AnalyzerError(f"{label} denominator is not positive: {median}")
    ratio = (max(values) - min(values)) / median
    if not math.isfinite(ratio) or ratio < 0 or math.copysign(1.0, ratio) < 0:
        raise AnalyzerError(f"{label} is not a finite nonnegative number")
    return ratio


def divide(numerator: float, denominator: float, label: str) -> float:
    if (
        not math.isfinite(denominator)
        or denominator <= 0
        or math.copysign(1.0, denominator) < 0
    ):
        raise AnalyzerError(f"{label} denominator is not positive: {denominator}")
    if not math.isfinite(numerator) or numerator < 0 or math.copysign(1.0, numerator) < 0:
        raise AnalyzerError(f"{label} numerator is not a finite nonnegative number")
    value = numerator / denominator
    if not math.isfinite(value) or value < 0 or math.copysign(1.0, value) < 0:
        raise AnalyzerError(f"{label} is not a finite nonnegative number")
    return value


def cross_run_statistics(documents: list[dict]) -> tuple[dict, list[dict]]:
    per_run = []
    for document in documents:
        latency = document["latency_microseconds"]
        per_run.append(
            {
                "mean_microseconds": finite_nonnegative(latency["mean"], "mean"),
                "p50_microseconds": finite_nonnegative(latency["p50"], "p50"),
                "p95_microseconds": finite_nonnegative(latency["p95"], "p95"),
                "p99_microseconds": finite_nonnegative(latency["p99"], "p99"),
                "aggregate_completed_requests_per_second": finite_positive(
                    document["aggregate_completed_requests_per_second"],
                    "aggregate_completed_requests_per_second",
                ),
                "active_call_requests_per_second": finite_positive(
                    document["active_call_requests_per_second"],
                    "active_call_requests_per_second",
                ),
            }
        )
    means = [row["mean_microseconds"] for row in per_run]
    aggregates = [row["aggregate_completed_requests_per_second"] for row in per_run]
    median_mean = positive_median(means, "median of run means")
    median_aggregate = positive_median(aggregates, "median aggregate throughput")
    summary = {
        "mean_of_run_means_microseconds": statistics.fmean(means),
        "median_of_run_means_microseconds": median_mean,
        "median_of_run_p50_microseconds": float(
            statistics.median(row["p50_microseconds"] for row in per_run)
        ),
        "median_of_run_p95_microseconds": float(
            statistics.median(row["p95_microseconds"] for row in per_run)
        ),
        "median_of_run_p99_microseconds": float(
            statistics.median(row["p99_microseconds"] for row in per_run)
        ),
        "minimum_run_mean_microseconds": min(means),
        "maximum_run_mean_microseconds": max(means),
        "relative_run_mean_range": relative_range(means, median_mean, "relative_run_mean_range"),
        "median_aggregate_completed_requests_per_second": median_aggregate,
        "minimum_aggregate_completed_requests_per_second": min(aggregates),
        "maximum_aggregate_completed_requests_per_second": max(aggregates),
        "relative_aggregate_throughput_range": relative_range(
            aggregates,
            median_aggregate,
            "relative_aggregate_throughput_range",
        ),
        "median_active_call_requests_per_second": float(
            statistics.median(row["active_call_requests_per_second"] for row in per_run)
        ),
    }
    for label, value in summary.items():
        if not math.isfinite(value) or value < 0 or math.copysign(1.0, value) < 0:
            raise AnalyzerError(f"{label} is not a finite nonnegative number")
    return summary, per_run


def scaling_metrics(worker_count: int, cell: dict, control: dict) -> dict:
    metrics = {
        "throughput_ratio": divide(
            cell["median_aggregate_completed_requests_per_second"],
            control["median_aggregate_completed_requests_per_second"],
            "throughput_ratio",
        ),
        "latency_mean_ratio": divide(
            cell["median_of_run_means_microseconds"],
            control["median_of_run_means_microseconds"],
            "latency_mean_ratio",
        ),
        "latency_p99_ratio": divide(
            cell["median_of_run_p99_microseconds"],
            control["median_of_run_p99_microseconds"],
            "latency_p99_ratio",
        ),
    }
    metrics["scaling_efficiency"] = divide(
        metrics["throughput_ratio"],
        float(worker_count),
        "scaling_efficiency",
    )
    if worker_count == 1:
        for label, value in metrics.items():
            if value != 1.0:
                raise AnalyzerError(f"worker count 1 {label} is {value}, expected 1")
    return metrics


def reject_nonfinite(value: object, label: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value) or (math.copysign(1.0, value) < 0 and value == 0):
            raise AnalyzerError(f"{label} is not a finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            reject_nonfinite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_nonfinite(item, f"{label}[{index}]")


def _model_path(verified: dict, workload: str) -> str:
    if workload == "add":
        return str(verified["add_model_path"])
    return str(verified["mnist_model_path"])


def build_aggregate(run_set: pathlib.Path) -> dict:
    plan = require_plan()
    environment = load_environment(run_set)
    require_completed(environment, run_set)
    protocol = require_protocol(environment, plan)
    queried, verified, _policy = require_provenance(environment)
    require_evidence_files(run_set, plan)

    seen: list[str] = []
    loaded: dict[str, dict] = {}
    for trial in plan:
        load_metadata(run_set, trial, seen)
        loaded[str(trial["trial_id"])] = load_trial_report(
            run_set,
            trial,
            _model_path(verified, str(trial["workload"])),
        )
    if seen != trial_order():
        raise AnalyzerError("loaded trial identities do not match the execution plan")

    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for trial in plan:
        if trial["pilot"] is True:
            continue
        key = (str(trial["workload"]), int(trial["worker_count"]))  # type: ignore[arg-type]
        grouped.setdefault(key, []).append(trial)

    workloads: dict[str, dict] = {}
    for workload in ("add", "mnist"):
        worker_results = []
        control_summary: dict | None = None
        for worker_count in APPROVED_WORKER_COUNTS:
            rows = grouped.get((workload, worker_count), [])
            rows.sort(key=lambda trial: int(trial["run_number"]))  # type: ignore[arg-type]
            run_numbers = [int(trial["run_number"]) for trial in rows]  # type: ignore[arg-type]
            if run_numbers != [1, 2, 3, 4, 5]:
                raise AnalyzerError(
                    f"{workload} worker count {worker_count} does not have counted runs 1 through 5"
                )
            if any(trial["pilot"] is True or str(trial["trial_id"]).startswith("pilot-") for trial in rows):
                raise AnalyzerError(f"{workload} worker count {worker_count} includes a pilot trial")
            documents = [loaded[str(trial["trial_id"])] for trial in rows]
            summary, per_run = cross_run_statistics(documents)
            per_run = [
                {"trial_id": str(trial["trial_id"]), **row} for trial, row in zip(rows, per_run)
            ]
            if worker_count == 1:
                control_summary = summary
            if control_summary is None:
                raise AnalyzerError(f"{workload} is missing the worker count 1 control")
            worker_results.append(
                {
                    "worker_count": worker_count,
                    "counted_trial_ids": [str(trial["trial_id"]) for trial in rows],
                    "per_run": per_run,
                    "cross_run": summary,
                    "scaling": scaling_metrics(worker_count, summary, control_summary),
                }
            )
        workloads[workload] = {"worker_results": worker_results}

    pilot_ids = [str(trial["trial_id"]) for trial in plan if trial["pilot"] is True]
    counted_ids = [str(trial["trial_id"]) for trial in plan if trial["pilot"] is False]
    document = {
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "source_benchmark_schema_version": SOURCE_SCHEMA_VERSION,
        "run_set_id": environment["run_set_id"],
        "git_commit": queried["git_commit"],
        "build_type": verified["cmake_build_type"],
        "pilots_excluded": True,
        "pilot_trial_ids": pilot_ids,
        "counted_trial_ids": counted_ids,
        "trial_order": list(protocol["trial_order"]),
        "counted_runs_per_configuration": COUNTED_RUNS_PER_CONFIGURATION,
        "models": {
            "add": {
                "path": verified["add_model_path"],
                "sha256": verified["add_model_sha256"],
                "fixture": None,
            },
            "mnist": {
                "path": verified["mnist_model_path"],
                "sha256": verified["mnist_model_sha256"],
                "fixture": MNIST_FIXTURE_ID,
            },
        },
        "experiment_configuration": {
            "workloads": protocol["workloads"],
            "worker_counts": protocol["worker_counts"],
            "warmup_iterations_per_worker": protocol["warmup_iterations_per_worker"],
            "measured_iterations_per_worker": protocol["measured_iterations_per_worker"],
            "correctness_check_interval": protocol["correctness_check_interval"],
            "counted_runs_per_configuration": protocol["counted_runs_per_configuration"],
            "session_ownership": SESSION_OWNERSHIP,
            "onnx_intra_op_threads_per_session": ONNX_INTRA_OP_THREADS_PER_SESSION,
            "graph_optimizations": GRAPH_OPTIMIZATIONS,
            "backend": BACKEND,
            "onnxruntime_version": verified["onnxruntime_version"],
            "cmake_build_type": verified["cmake_build_type"],
            "sanitizer_instrumentation": verified["sanitizer_instrumentation"],
            "inter_trial_delay_seconds": protocol["inter_trial_delay_seconds"],
            "environment_variable": protocol["environment_variable"],
            "control_worker_count": 1,
        },
        "units": {
            "latency": "microseconds",
            "throughput": "requests_per_second",
            "ratios": "dimensionless",
            "relative_run_mean_range": "dimensionless_ratio",
            "relative_aggregate_throughput_range": "dimensionless_ratio",
        },
        "definitions": {
            "within_run": (
                "Each report's mean, p50, p95, and p99 describe the measured calls pooled "
                "inside that one concurrent process. P99 is that process's nearest-rank "
                "order statistic."
            ),
            "median_of_run_p99": (
                "median_of_run_p99_microseconds is the median of the five per-process P99 "
                "values. It is not a pooled percentile of the combined samples. Schema "
                "version 2 does not contain raw samples, so a pooled P99 is not computed."
            ),
            "relative_run_mean_range": (
                "(maximum run mean - minimum run mean) / median of the run means"
            ),
            "relative_aggregate_throughput_range": (
                "(maximum aggregate throughput - minimum aggregate throughput) / "
                "median aggregate throughput"
            ),
            "throughput_ratio": (
                "throughput_ratio(N) is the median aggregate completed requests per second "
                "at worker count N divided by the same median at worker count 1 for that "
                "workload. Active-call throughput is recorded separately and is not the "
                "scaling numerator. Sequential active-call throughput is not a speedup baseline."
            ),
            "scaling_efficiency": (
                "scaling_efficiency(N) is throughput_ratio(N) divided by N. "
                "Linear scaling is not assumed."
            ),
            "latency_ratios": (
                "latency_mean_ratio(N) and latency_p99_ratio(N) divide the worker-count-N "
                "median of run means, or the median of per-process P99 values, by the "
                "worker-count-1 value for the same workload."
            ),
            "confidence_interval": (
                "Five counted runs do not establish statistical significance. "
                "No confidence interval is computed."
            ),
            "pilots": (
                "Pilot trials are validated with the same evidence checks and then excluded "
                "from latency reductions, throughput reductions, scaling ratios, scaling "
                "efficiency, and run-to-run variation."
            ),
        },
        "median_of_run_p99_is_pooled_p99": False,
        "confidence_interval_computed": False,
        "workloads": workloads,
    }
    reject_nonfinite(document, "aggregate")
    return document


def publish_aggregate(run_set: pathlib.Path, document: dict) -> pathlib.Path:
    dest = run_set / "aggregate.json"
    if dest.exists():
        raise AnalyzerError(f"aggregate.json already exists: {dest}")
    try:
        payload = json.dumps(document, indent=2, allow_nan=False) + "\n"
    except ValueError as exc:
        raise AnalyzerError(f"aggregate contains a non-finite number: {exc}") from exc
    temporary = run_set / f".aggregate.json.{os.getpid()}.tmp"
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.link(temporary, dest)
    except FileExistsError as exc:
        raise AnalyzerError(f"aggregate.json already exists: {dest}") from exc
    except OSError as exc:
        raise AnalyzerError(f"failed to publish aggregate.json: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return dest


def analyze_and_write(run_set: pathlib.Path) -> pathlib.Path:
    destination = run_set / "aggregate.json"
    if destination.exists():
        raise AnalyzerError(f"aggregate.json already exists: {destination}")
    document = build_aggregate(run_set)
    return publish_aggregate(run_set, document)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        destination = analyze_and_write(pathlib.Path(args.run_set))
    except AnalyzerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
