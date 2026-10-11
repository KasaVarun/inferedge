#!/usr/bin/env python3
"""Reduce one completed inferedge-bench run set to aggregate.json.

This module does not launch inferedge-bench and does not run a model. It reads
the files collect_baseline.py already wrote, checks that the counted trials
agree with that manifest, and publishes one aggregate. Pilot trials stay on
disk and are excluded from the statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import sys

# `python3 benchmarks/analyze_baseline.py` puts this directory on sys.path.
# The package import needs the repository root. `python3 -m` already provides it.
if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.collect_baseline import (
    ADD_MODEL_SHA256,
    BACKEND,
    GRAPH_OPTIMIZATIONS,
    MNIST_FIXTURE_ID,
    MNIST_MODEL_SHA256,
    ONNXRUNTIME_VERSION,
    TOOL_NAME,
    TOOL_VERSION,
    trial_ids,
    validate_report,
)

AGGREGATE_SCHEMA_VERSION = 1
SOURCE_SCHEMA_VERSION = 1
APPROVED_COUNTED_RUNS = 5
APPROVED_WARMUP = 100
APPROVED_ITERATIONS = 1000
APPROVED_CHECK_INTERVAL = 1
PILOT_IDS = ("pilot-add", "pilot-mnist")


class AnalyzerError(Exception):
    """The run set cannot be reduced. No aggregate is published."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze one completed inferedge-bench run set."
    )
    parser.add_argument(
        "--run-set",
        required=True,
        help="Path to one benchmarks/baselines/<run-set-id> directory",
    )
    return parser


def _object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise AnalyzerError(f"{label} is not a JSON object")
    return value


def _exact(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise AnalyzerError(f"{label} is {value!r}, expected {expected!r}")


def load_environment(run_set: pathlib.Path) -> dict:
    if not run_set.is_dir():
        raise AnalyzerError(f"run-set directory does not exist: {run_set}")
    path = run_set / "environment.json"
    if not path.is_file():
        raise AnalyzerError(f"environment.json is missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalyzerError(f"environment.json is not valid JSON: {exc.msg}") from exc
    return _object(document, "environment.json")


def require_completed(environment: dict, run_set: pathlib.Path) -> None:
    status = environment.get("status")
    if status != "completed":
        raise AnalyzerError(f"collection status is {status!r}, expected 'completed'")
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


def require_protocol(environment: dict) -> dict:
    protocol = _object(environment.get("collection_protocol"), "collection_protocol")
    _exact(protocol.get("workloads"), ["add", "mnist"], "collection_protocol.workloads")
    _exact(protocol.get("warmup_iterations"), APPROVED_WARMUP, "warmup_iterations")
    _exact(
        protocol.get("measured_iterations"),
        APPROVED_ITERATIONS,
        "measured_iterations",
    )
    _exact(
        protocol.get("correctness_check_interval"),
        APPROVED_CHECK_INTERVAL,
        "correctness_check_interval",
    )
    _exact(
        protocol.get("counted_runs_per_workload"),
        APPROVED_COUNTED_RUNS,
        "counted_runs_per_workload",
    )
    _exact(protocol.get("pilots"), list(PILOT_IDS), "collection_protocol.pilots")
    expected_order = trial_ids(APPROVED_COUNTED_RUNS)
    _exact(protocol.get("trial_order"), expected_order, "collection_protocol.trial_order")
    if len(protocol["trial_order"]) != len(set(protocol["trial_order"])):
        raise AnalyzerError("collection_protocol.trial_order contains a duplicate trial identity")
    _exact(
        protocol.get("environment_variable"),
        "ORT_DISABLE_TELEMETRY=1",
        "collection_protocol.environment_variable",
    )
    delay = protocol.get("inter_trial_delay_seconds")
    if isinstance(delay, bool) or not isinstance(delay, (int, float)):
        raise AnalyzerError("inter_trial_delay_seconds is not a JSON number")
    if not math.isfinite(float(delay)) or float(delay) < 0:
        raise AnalyzerError("inter_trial_delay_seconds is not a nonnegative finite number")
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
    for label in ("add_model_path", "mnist_model_path"):
        path = verified.get(label)
        if not isinstance(path, str) or path == "":
            raise AnalyzerError(f"{label} is missing")
    _exact(policy.get("backend"), BACKEND, "source_defined_policy.backend")
    _exact(policy.get("calling_threads"), 1, "source_defined_policy.calling_threads")
    _exact(policy.get("onnx_intra_op_threads"), 1, "source_defined_policy.onnx_intra_op_threads")
    _exact(policy.get("graph_optimizations"), GRAPH_OPTIMIZATIONS, "source_defined_policy.graph_optimizations")
    return queried, verified, policy


def counted_ids() -> dict[str, list[str]]:
    names = trial_ids(APPROVED_COUNTED_RUNS)
    return {
        "add": [name for name in names if name.startswith("add-")],
        "mnist": [name for name in names if name.startswith("mnist-")],
    }


def is_counted_report_name(name: str) -> bool:
    if not name.endswith(".json") or name.endswith(".meta.json"):
        return False
    workload, separator, number = name[:-5].partition("-")
    return workload in {"add", "mnist"} and separator == "-" and len(number) == 2 and number.isdigit()


def require_counted_files(run_set: pathlib.Path, expected: dict[str, list[str]]) -> None:
    runs = run_set / "runs"
    if not runs.is_dir():
        raise AnalyzerError(f"runs directory is missing: {runs}")
    expected_names = {f"{trial_id}.json" for ids in expected.values() for trial_id in ids}
    found_names = {path.name for path in runs.iterdir() if path.is_file() and is_counted_report_name(path.name)}
    missing = sorted(expected_names - found_names)
    if missing:
        raise AnalyzerError(f"missing counted report: {missing[0]}")
    extra = sorted(found_names - expected_names)
    if extra:
        raise AnalyzerError(f"unexpected counted report: {extra[0]}")


def load_metadata(run_set: pathlib.Path, trial_id: str, seen: list[str]) -> None:
    path = run_set / "runs" / f"{trial_id}.meta.json"
    if not path.is_file():
        raise AnalyzerError(f"trial metadata is missing: {trial_id}")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalyzerError(f"trial metadata for {trial_id} is not valid JSON: {exc.msg}") from exc
    meta = _object(meta, f"trial metadata for {trial_id}")
    claimed = meta.get("trial_id")
    if claimed in seen:
        raise AnalyzerError(f"duplicate trial identity: {claimed}")
    if claimed != trial_id:
        raise AnalyzerError(f"trial identity mismatch for {trial_id}: metadata says {claimed!r}")
    seen.append(claimed)
    if meta.get("accepted") is not True:
        raise AnalyzerError(f"trial {trial_id} is not an accepted measurement")
    if type(meta.get("returncode")) is not int or meta.get("returncode") != 0:
        raise AnalyzerError(f"trial {trial_id} return code is not 0")
    if meta.get("problems") != []:
        raise AnalyzerError(f"trial {trial_id} metadata records problems")


def load_counted_report(
    run_set: pathlib.Path,
    trial_id: str,
    workload: str,
    model_path: str,
) -> dict:
    path = run_set / "runs" / f"{trial_id}.json"
    text = path.read_text(encoding="utf-8")
    problems = validate_report(
        text,
        workload=workload,
        model_path=model_path,
        warmup=APPROVED_WARMUP,
        iterations=APPROVED_ITERATIONS,
        check_interval=APPROVED_CHECK_INTERVAL,
        tool_version=TOOL_VERSION,
    )
    if problems:
        raise AnalyzerError(f"{trial_id}: {'; '.join(problems)}")
    document = json.loads(text)
    if document.get("tool", {}).get("name") != TOOL_NAME:
        raise AnalyzerError(f"{trial_id}: tool.name is not {TOOL_NAME}")
    return document


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalyzerError(f"{label} is not a JSON number")
    number = float(value)
    if not math.isfinite(number) or (math.copysign(1.0, number) < 0 and number == 0):
        raise AnalyzerError(f"{label} is not a finite number")
    return number


def cross_run_statistics(documents: list[dict]) -> tuple[dict, list[dict]]:
    per_run = []
    for document in documents:
        latency = document["latency_microseconds"]
        throughput = document["throughput_requests_per_second"]
        per_run.append(
            {
                "mean_microseconds": finite_number(latency["mean"], "mean"),
                "p50_microseconds": finite_number(latency["p50"], "p50"),
                "p95_microseconds": finite_number(latency["p95"], "p95"),
                "p99_microseconds": finite_number(latency["p99"], "p99"),
                "active_call_requests_per_second": finite_number(
                    throughput["active_call"], "active_call"
                ),
                "observed_sequential_requests_per_second": finite_number(
                    throughput["observed_sequential"], "observed_sequential"
                ),
            }
        )
    means = [row["mean_microseconds"] for row in per_run]
    median_mean = float(statistics.median(means))
    if not math.isfinite(median_mean) or median_mean <= 0 or math.copysign(1.0, median_mean) < 0:
        raise AnalyzerError(
            "relative variation denominator is not positive: "
            f"median of run means is {median_mean}"
        )
    minimum = min(means)
    maximum = max(means)
    relative = (maximum - minimum) / median_mean
    summary = {
        "mean_of_run_means_microseconds": statistics.fmean(means),
        "median_of_run_means_microseconds": median_mean,
        "median_of_run_p50_microseconds": float(statistics.median(row["p50_microseconds"] for row in per_run)),
        "median_of_run_p95_microseconds": float(statistics.median(row["p95_microseconds"] for row in per_run)),
        "median_of_run_p99_microseconds": float(statistics.median(row["p99_microseconds"] for row in per_run)),
        "minimum_run_mean_microseconds": minimum,
        "maximum_run_mean_microseconds": maximum,
        "relative_run_mean_range": relative,
        "median_active_call_throughput_requests_per_second": float(
            statistics.median(row["active_call_requests_per_second"] for row in per_run)
        ),
        "median_observed_sequential_throughput_requests_per_second": float(
            statistics.median(row["observed_sequential_requests_per_second"] for row in per_run)
        ),
    }
    for label, value in summary.items():
        if not math.isfinite(value) or math.copysign(1.0, value) < 0:
            raise AnalyzerError(f"{label} is not a finite nonnegative number")
    return summary, per_run


def reject_nonfinite(value: object, label: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value) or math.copysign(1.0, value) < 0 and value == 0:
            raise AnalyzerError(f"{label} is not a finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            reject_nonfinite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_nonfinite(item, f"{label}[{index}]")


def build_aggregate(run_set: pathlib.Path) -> dict:
    environment = load_environment(run_set)
    require_completed(environment, run_set)
    protocol = require_protocol(environment)
    queried, verified, _policy = require_provenance(environment)
    identities = counted_ids()
    require_counted_files(run_set, identities)
    seen: list[str] = []
    reports: dict[str, list[dict]] = {}
    for workload, trial_list in identities.items():
        model_path = verified["add_model_path"] if workload == "add" else verified["mnist_model_path"]
        loaded = []
        for trial_id in trial_list:
            load_metadata(run_set, trial_id, seen)
            loaded.append(load_counted_report(run_set, trial_id, workload, model_path))
        reports[workload] = loaded

    workloads = {}
    for workload, trial_list in identities.items():
        summary, per_run = cross_run_statistics(reports[workload])
        per_run = [{"trial_id": trial_id, **row} for trial_id, row in zip(trial_list, per_run)]
        workloads[workload] = {
            "counted_trial_ids": trial_list,
            "per_run": per_run,
            "cross_run": summary,
        }

    document = {
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "source_benchmark_schema_version": SOURCE_SCHEMA_VERSION,
        "run_set_id": environment["run_set_id"],
        "git_commit": queried["git_commit"],
        "build_type": verified["cmake_build_type"],
        "pilots_excluded": True,
        "pilot_trial_ids": list(PILOT_IDS),
        "counted_runs_per_workload": APPROVED_COUNTED_RUNS,
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
            "warmup_iterations": protocol["warmup_iterations"],
            "measured_iterations": protocol["measured_iterations"],
            "correctness_check_interval": protocol["correctness_check_interval"],
            "calling_threads": 1,
            "onnx_intra_op_threads": 1,
            "graph_optimizations": GRAPH_OPTIMIZATIONS,
            "backend": BACKEND,
            "onnxruntime_version": verified["onnxruntime_version"],
            "cmake_build_type": verified["cmake_build_type"],
            "sanitizer_instrumentation": verified["sanitizer_instrumentation"],
            "counted_runs_per_workload": protocol["counted_runs_per_workload"],
            "inter_trial_delay_seconds": protocol["inter_trial_delay_seconds"],
            "environment_variable": protocol["environment_variable"],
        },
        "units": {
            "latency": "microseconds",
            "throughput": "requests_per_second",
            "relative_run_mean_range": "dimensionless_ratio",
        },
        "definitions": {
            "within_run": (
                "Each counted report's mean, p50, p95, and p99 describe the measured "
                "calls of that one process. P99 is that process's nearest-rank order statistic."
            ),
            "cross_run": (
                "Cross-run values reduce the five per-run statistics. "
                "median_of_run_p99_microseconds is the median of the five per-run P99 values. "
                "It is not a pooled P99 of the combined samples, and it is not the average "
                "of the five P99 values."
            ),
            "relative_run_mean_range": (
                "(maximum run mean - minimum run mean) / median of the run means"
            ),
            "confidence_interval": "Five runs do not produce a confidence interval. None is reported.",
            "pilots": "pilot-add and pilot-mnist are excluded from every statistic.",
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
