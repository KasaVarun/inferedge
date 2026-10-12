#!/usr/bin/env python3
"""Collect schema-version-2 reports from inferedge-bench-concurrent.

This module checks a Release build and the repository, then starts one fresh
inferedge-bench-concurrent process per trial. It does not launch inferedge-bench,
does not write aggregate.json, and does not retry a failed trial.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import math
import os
import pathlib
import platform
import subprocess
import sys
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.collect_baseline import (
    BACKEND,
    GRAPH_OPTIMIZATIONS,
    ONNXRUNTIME_VERSION,
    ORDERING_TOLERANCE,
    CollectorError,
    ProcessResult,
    approved_untracked_paths,
    assess_build_cache,
    compiler_and_cmake,
    create_run_set,
    git_status_problems,
    json_integer,
    json_number,
    link_tool_args,
    locate_pinned_runtime,
    make_run_set_id,
    negative_sign,
    ordered_metric,
    parse_cmake_cache,
    parse_single_document,
    public_manifest,
    query_text,
    require_exact_keys,
    sanitizer_symbol_problems,
    sha256_file,
    subprocess_runner,
    symbol_tool_args,
    utc_now,
    verify_linked_onnx_runtime,
    write_json,
)

TOOL_NAME = "inferedge-bench-concurrent"
TOOL_VERSION = "0.1.0"
SESSION_OWNERSHIP = "one_inference_engine_per_worker"
ONNX_INTRA_OP_THREADS_PER_SESSION = 1
MNIST_FIXTURE_ID = "test_00003"
WARMUP_ITERATIONS_PER_WORKER = 100
MEASURED_ITERATIONS_PER_WORKER = 1000
CORRECTNESS_CHECK_INTERVAL = 1
COUNTED_RUNS_PER_CONFIGURATION = 5
INTER_TRIAL_DELAY_SECONDS = 5.0
AGGREGATE_RELATIVE_TOLERANCE = 1e-6
BINARY_NAME = "inferedge-bench-concurrent"

# These digests are the protocol. Preflight compares files with them.
# They are not replaced by a hash computed at runtime.
ADD_MODEL_SHA256 = "37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447"
MNIST_MODEL_SHA256 = "3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce"
MNIST_INPUT_SHA256 = "692f8ff1c2aaa7dd3c5982979516e774e4daf6d8420f1818a68389825ebb84a9"
MNIST_REFERENCE_SHA256 = "9a5daa05f5096f5521deab05c025d3c3691e2587189f176d8699a9521b663a46"
MNIST_FIXTURE_MANIFEST_SHA256 = "4e6f95f8626fa89a22f59aed004870b942fcf2d7bb4add2226f9ca4f9f13009a"

PINNED_FILES = (
    ("models/add_f32.onnx", ADD_MODEL_SHA256),
    ("models/mnist_mlp.onnx", MNIST_MODEL_SHA256),
    ("models/fixtures/mnist/inputs/test_00003.f32", MNIST_INPUT_SHA256),
    ("models/fixtures/mnist/onnx/test_00003.f32", MNIST_REFERENCE_SHA256),
    ("models/fixtures/mnist/manifest.json", MNIST_FIXTURE_MANIFEST_SHA256),
)

ALLOWED_UNTRACKED = (
    "models/experiments",
    "benchmarks/baselines",
    "benchmarks/concurrent_runs",
)
APPROVED_WORKER_COUNTS = (1, 2, 4, 8)
CANONICAL_CONFIGURATIONS = (
    ("add", 1),
    ("mnist", 1),
    ("add", 2),
    ("mnist", 2),
    ("add", 4),
    ("mnist", 4),
    ("add", 8),
    ("mnist", 8),
)

TOP_LEVEL_KEYS = (
    "schema_version",
    "tool",
    "model",
    "workload",
    "fixture",
    "backend",
    "configuration",
    "latency_microseconds",
    "measurement_wall_microseconds",
    "total_completed_requests",
    "per_worker_completed_requests",
    "aggregate_completed_requests_per_second",
    "active_call_requests_per_second",
    "correctness",
)
CONFIGURATION_KEYS = (
    "worker_count",
    "session_ownership",
    "onnx_intra_op_threads_per_session",
    "graph_optimizations",
    "warmup_iterations_per_worker",
    "measured_iterations_per_worker",
    "correctness_check_interval",
)
LATENCY_KEYS = (
    "sample_count",
    "minimum",
    "mean",
    "p50",
    "p95",
    "p99",
    "maximum",
    "population_standard_deviation",
)
SEQUENTIAL_ONLY_KEYS = (
    "measurement_loop_microseconds",
    "throughput_requests_per_second",
    "calling_threads",
    "onnx_intra_op_threads",
)


def require_commit_sha(commit: str) -> str:
    """Accept one full lowercase Git commit SHA. There is no default commit."""
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise CollectorError(
            "--expected-commit must be a 40-character lowercase hexadecimal Git SHA, "
            f"got {commit!r}"
        )
    return commit


def format_trial(workload: str, concurrency: int, run_number: int | None) -> str:
    tag = f"c{concurrency:02d}"
    if run_number is None:
        return f"pilot-{workload}-{tag}"
    return f"{workload}-{tag}-{run_number:02d}"


def execution_plan() -> list[dict[str, object]]:
    """Eight pilots, then five rotated rounds. The same configuration is never adjacent."""
    plan: list[dict[str, object]] = []
    for workload, concurrency in CANONICAL_CONFIGURATIONS:
        plan.append(
            {
                "trial_id": format_trial(workload, concurrency, None),
                "workload": workload,
                "worker_count": concurrency,
                "pilot": True,
                "run_number": None,
            }
        )
    width = len(CANONICAL_CONFIGURATIONS)
    for run_number in range(1, COUNTED_RUNS_PER_CONFIGURATION + 1):
        offset = run_number - 1
        for slot in range(width):
            workload, concurrency = CANONICAL_CONFIGURATIONS[(slot + offset) % width]
            plan.append(
                {
                    "trial_id": format_trial(workload, concurrency, run_number),
                    "workload": workload,
                    "worker_count": concurrency,
                    "pilot": False,
                    "run_number": run_number,
                }
            )
    return plan


def trial_order() -> list[str]:
    return [str(trial["trial_id"]) for trial in execution_plan()]


def release_optimization_problems(cache: dict[str, str]) -> list[str]:
    """Release must request optimization. The exact -O spelling is the cache's."""
    flags = cache.get("CMAKE_CXX_FLAGS_RELEASE", "")
    lowered = flags.lower()
    if any(token in lowered for token in ("-o0", "-og", "/od")):
        return [f"CMAKE_CXX_FLAGS_RELEASE disables optimization: {flags}"]
    if not any(token in flags for token in ("-O2", "-O3", "-Ofast", "/O2")):
        return [
            "CMAKE_CXX_FLAGS_RELEASE does not enable an optimized Release configuration: "
            f"{flags or 'empty'}"
        ]
    return []


def locate_build_dir(binary: pathlib.Path) -> pathlib.Path:
    cache = binary.parent / "CMakeCache.txt"
    if not cache.is_file():
        raise CollectorError(
            "CMakeCache.txt was not found beside the benchmark binary at "
            f"{cache}. The binary must be the inferedge-bench-concurrent produced in its "
            "CMake build directory, not a copy"
        )
    return binary.parent


def git_blob_sha256(repo_root: pathlib.Path, commit: str, relative: str) -> str:
    """SHA-256 of one committed blob. Bytes stay binary; a shell is not used."""
    pinned = {path for path, _digest in PINNED_FILES}
    if relative not in pinned:
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


def verify_pinned_files(
    repo_root: pathlib.Path,
    commit: str,
    blob_sha256,
) -> None:
    for relative, expected in PINNED_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise CollectorError(f"pinned file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise CollectorError(
                f"SHA-256 mismatch for {relative}: expected {expected}, actual {actual}"
            )
        committed = blob_sha256(repo_root, commit, relative)
        if committed != expected:
            raise CollectorError(
                "committed SHA-256 mismatch for "
                f"{commit}:{relative}: expected {expected}, actual {committed}"
            )


def _query_integer(runner, args: list[str], label: str) -> tuple[int | None, str | None]:
    result = query_text(runner, args)
    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip() or f"exit {result.returncode}"
        return None, f"{label} failed: {detail}"
    text = result.stdout.strip()
    if not text.isdecimal():
        return None, f"{label} returned {text!r}"
    return int(text), None


def hardware_metadata(runner) -> dict[str, object]:
    """Record queried host facts. Core counts come from CPU queries, not hw.model."""
    logical = os.cpu_count()
    metadata: dict[str, object] = {
        "system": platform.system(),
        "platform": platform.platform(),
        "os_version": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "mac_version": platform.mac_ver()[0] or None,
        "hardware_model": None,
        "hardware_model_error": None,
        "logical_cpu_count": logical,
        "logical_cpu_count_source": "os.cpu_count",
        "logical_cpu_count_error": None if logical is not None else "os.cpu_count() returned None",
        "hw_physicalcpu": None,
        "hw_physicalcpu_error": None,
        "hw_logicalcpu": None,
        "hw_logicalcpu_error": None,
        "hw_ncpu": None,
        "hw_ncpu_error": None,
        "nproc_cpu_count": None,
        "nproc_cpu_count_error": None,
    }
    if platform.system() == "Darwin":
        model = query_text(runner, ["sysctl", "-n", "hw.model"])
        if model.returncode == 0 and model.stdout.strip():
            metadata["hardware_model"] = model.stdout.strip()
        else:
            detail = model.stderr.strip() or f"exit {model.returncode}"
            metadata["hardware_model_error"] = f"sysctl -n hw.model failed: {detail}"
        for key, name in (
            ("hw_physicalcpu", "hw.physicalcpu"),
            ("hw_logicalcpu", "hw.logicalcpu"),
            ("hw_ncpu", "hw.ncpu"),
        ):
            value, error = _query_integer(runner, ["sysctl", "-n", name], f"sysctl -n {name}")
            metadata[key] = value
            metadata[key + "_error"] = error
        if not metadata["mac_version"]:
            sw = query_text(runner, ["sw_vers", "-productVersion"])
            if sw.returncode == 0 and sw.stdout.strip():
                metadata["mac_version"] = sw.stdout.strip()
        metadata["nproc_cpu_count_error"] = "nproc is queried only on non-Darwin hosts"
        return metadata

    metadata["hardware_model_error"] = "hw.model is queried only on Darwin"
    metadata["hw_physicalcpu_error"] = "hw.physicalcpu is queried only on Darwin"
    metadata["hw_logicalcpu_error"] = "hw.logicalcpu is queried only on Darwin"
    metadata["hw_ncpu_error"] = "hw.ncpu is queried only on Darwin"
    value, error = _query_integer(runner, ["nproc"], "nproc")
    metadata["nproc_cpu_count"] = value
    metadata["nproc_cpu_count_error"] = error
    return metadata


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


def concurrent_command(
    binary: pathlib.Path,
    *,
    workload: str,
    worker_count: int,
    add_model: pathlib.Path,
    mnist_model: pathlib.Path,
    fixture_dir: pathlib.Path,
) -> list[str]:
    if binary.name != BINARY_NAME:
        raise CollectorError(
            f"refusing to launch {binary.name}; the collector only launches {BINARY_NAME}"
        )
    if worker_count not in APPROVED_WORKER_COUNTS:
        raise CollectorError(f"worker count {worker_count} is outside 1, 2, 4, 8")
    if workload == "add":
        model = add_model
        fixture_args: list[str] = []
    elif workload == "mnist":
        model = mnist_model
        fixture_args = ["--fixture", MNIST_FIXTURE_ID, "--fixture-dir", str(fixture_dir)]
    else:
        raise CollectorError(f"unknown workload: {workload}")
    return [
        str(binary),
        "--model",
        str(model),
        "--workload",
        workload,
        "--concurrency",
        str(worker_count),
        "--warmup",
        str(WARMUP_ITERATIONS_PER_WORKER),
        "--iterations",
        str(MEASURED_ITERATIONS_PER_WORKER),
        "--check-interval",
        str(CORRECTNESS_CHECK_INTERVAL),
        "--format",
        "json",
        *fixture_args,
    ]


def validate_concurrent_report(
    text: str,
    *,
    workload: str,
    model_path: str,
    worker_count: int,
) -> list[str]:
    """Validate one schema-version-2 document. Schema version 1 is rejected."""
    document, problems = parse_single_document(text)
    if document is None:
        return problems
    require_exact_keys(document, TOP_LEVEL_KEYS, "report", problems)
    for key in SEQUENTIAL_ONLY_KEYS:
        if key in document:
            problems.append(f"report has sequential-only field {key}")
    schema = json_integer(document.get("schema_version"), "schema_version", problems)
    if schema is not None and schema != 2:
        problems.append(f"schema_version is {schema}, expected 2")
    tool = document.get("tool")
    if not isinstance(tool, dict):
        problems.append("tool is not an object")
    else:
        require_exact_keys(tool, ("name", "version"), "tool", problems)
        if tool.get("name") != TOOL_NAME:
            problems.append(f"tool.name is {tool.get('name')!r}, expected {TOOL_NAME!r}")
        if tool.get("version") != TOOL_VERSION:
            problems.append(f"tool.version is {tool.get('version')!r}, expected {TOOL_VERSION!r}")
    if document.get("correctness") != "passed":
        problems.append(f"correctness is {document.get('correctness')!r}, expected 'passed'")
    if document.get("workload") != workload:
        problems.append(f"workload is {document.get('workload')!r}, expected {workload!r}")
    expected_fixture = None if workload == "add" else MNIST_FIXTURE_ID
    if document.get("fixture") != expected_fixture:
        problems.append(f"fixture is {document.get('fixture')!r}, expected {expected_fixture!r}")
    if document.get("model") != model_path:
        problems.append(f"model is {document.get('model')!r}, expected {model_path!r}")
    if document.get("backend") != BACKEND:
        problems.append(f"backend is {document.get('backend')!r}, expected {BACKEND!r}")

    configuration = document.get("configuration")
    if not isinstance(configuration, dict):
        problems.append("configuration is not an object")
        configuration = {}
    else:
        require_exact_keys(configuration, CONFIGURATION_KEYS, "configuration", problems)
        for key in ("calling_threads", "onnx_intra_op_threads"):
            if key in configuration:
                problems.append(f"configuration has sequential-only field {key}")
    found_workers = json_integer(configuration.get("worker_count"), "worker_count", problems)
    if found_workers is not None and found_workers != worker_count:
        problems.append(f"worker_count is {found_workers}, expected {worker_count}")
    if configuration.get("session_ownership") != SESSION_OWNERSHIP:
        problems.append(
            "session_ownership is "
            f"{configuration.get('session_ownership')!r}, expected {SESSION_OWNERSHIP!r}"
        )
    intra = json_integer(
        configuration.get("onnx_intra_op_threads_per_session"),
        "onnx_intra_op_threads_per_session",
        problems,
    )
    if intra is not None and intra != ONNX_INTRA_OP_THREADS_PER_SESSION:
        problems.append(
            "onnx_intra_op_threads_per_session is "
            f"{intra}, expected {ONNX_INTRA_OP_THREADS_PER_SESSION}"
        )
    if configuration.get("graph_optimizations") != GRAPH_OPTIMIZATIONS:
        problems.append(
            "graph_optimizations is "
            f"{configuration.get('graph_optimizations')!r}, expected {GRAPH_OPTIMIZATIONS!r}"
        )
    warmup = json_integer(
        configuration.get("warmup_iterations_per_worker"),
        "warmup_iterations_per_worker",
        problems,
    )
    if warmup is not None and warmup != WARMUP_ITERATIONS_PER_WORKER:
        problems.append(
            f"warmup_iterations_per_worker is {warmup}, expected {WARMUP_ITERATIONS_PER_WORKER}"
        )
    measured = json_integer(
        configuration.get("measured_iterations_per_worker"),
        "measured_iterations_per_worker",
        problems,
    )
    if measured is not None and measured != MEASURED_ITERATIONS_PER_WORKER:
        problems.append(
            "measured_iterations_per_worker is "
            f"{measured}, expected {MEASURED_ITERATIONS_PER_WORKER}"
        )
    interval = json_integer(
        configuration.get("correctness_check_interval"),
        "correctness_check_interval",
        problems,
    )
    if interval is not None and interval != CORRECTNESS_CHECK_INTERVAL:
        problems.append(
            f"correctness_check_interval is {interval}, expected {CORRECTNESS_CHECK_INTERVAL}"
        )

    expected_total = worker_count * MEASURED_ITERATIONS_PER_WORKER
    total = json_integer(
        document.get("total_completed_requests"),
        "total_completed_requests",
        problems,
    )
    if total is not None and total != expected_total:
        problems.append(f"total_completed_requests is {total}, expected {expected_total}")
    per_worker = document.get("per_worker_completed_requests")
    counts: list[int] = []
    if not isinstance(per_worker, list):
        problems.append("per_worker_completed_requests is not an array")
    else:
        if len(per_worker) != worker_count:
            problems.append(
                f"per_worker_completed_requests has {len(per_worker)} entries, expected {worker_count}"
            )
        for index, value in enumerate(per_worker):
            found = json_integer(value, f"per_worker_completed_requests[{index}]", problems)
            if found is None:
                continue
            counts.append(found)
            if found != MEASURED_ITERATIONS_PER_WORKER:
                problems.append(
                    f"per_worker_completed_requests[{index}] is {found}, "
                    f"expected {MEASURED_ITERATIONS_PER_WORKER}"
                )
        if len(counts) == len(per_worker) and sum(counts) != expected_total:
            problems.append(
                f"per-worker completed requests sum to {sum(counts)}, expected {expected_total}"
            )

    latency = document.get("latency_microseconds")
    if not isinstance(latency, dict):
        problems.append("latency_microseconds is not an object")
        latency = {}
    else:
        require_exact_keys(latency, LATENCY_KEYS, "latency_microseconds", problems)
    sample_count = json_integer(latency.get("sample_count"), "sample_count", problems)
    if sample_count is not None and sample_count != expected_total:
        problems.append(f"sample_count is {sample_count}, expected {expected_total}")
    metrics: dict[str, float | None] = {}
    for key in LATENCY_KEYS:
        if key == "sample_count":
            continue
        metrics[key] = json_number(latency.get(key), key, problems)
    for key in ("minimum", "mean", "p50", "p95", "p99", "maximum"):
        value = metrics.get(key)
        if value is not None and (value < 0 or negative_sign(value)):
            problems.append(f"{key} latency is negative")
    if all(metrics[key] is not None for key in ("minimum", "p50", "p95", "p99", "maximum")):
        chain = (("minimum", "p50"), ("p50", "p95"), ("p95", "p99"), ("p99", "maximum"))
        for earlier, later in chain:
            if not ordered_metric(metrics[earlier], metrics[later]):
                problems.append(f"latency metrics are out of order at {earlier} and {later}")
                break
    deviation = metrics.get("population_standard_deviation")
    if deviation is not None and negative_sign(deviation):
        problems.append("population standard deviation is negative")

    wall = json_number(
        document.get("measurement_wall_microseconds"),
        "measurement_wall_microseconds",
        problems,
    )
    if wall is not None and (wall <= 0 or negative_sign(wall)):
        problems.append("measurement wall duration is not positive")
    aggregate = json_number(
        document.get("aggregate_completed_requests_per_second"),
        "aggregate_completed_requests_per_second",
        problems,
    )
    if aggregate is not None and (aggregate <= 0 or negative_sign(aggregate)):
        problems.append("aggregate throughput is not positive")
    active = json_number(
        document.get("active_call_requests_per_second"),
        "active_call_requests_per_second",
        problems,
    )
    if active is not None and (active <= 0 or negative_sign(active)):
        problems.append("active-call throughput is not positive")
    if (
        wall is not None
        and wall > 0
        and not negative_sign(wall)
        and total is not None
        and total > 0
        and aggregate is not None
        and math.isfinite(aggregate)
    ):
        expected_aggregate = total / (wall / 1_000_000.0)
        scale = max(abs(expected_aggregate), abs(aggregate), 1.0)
        if (
            not math.isfinite(expected_aggregate)
            or abs(aggregate - expected_aggregate) > scale * AGGREGATE_RELATIVE_TOLERANCE
        ):
            problems.append("aggregate throughput does not match the wall duration")
    return problems


def preflight(
    *,
    repo_root: pathlib.Path,
    binary: pathlib.Path,
    expected_commit: str,
    runner=subprocess_runner,
    blob_sha256=git_blob_sha256,
) -> dict:
    commit = require_commit_sha(expected_commit)
    if not binary.is_file():
        raise CollectorError(f"benchmark binary does not exist: {binary}")
    if not os.access(binary, os.X_OK):
        raise CollectorError(f"benchmark binary is not executable: {binary}")
    if binary.resolve().name != BINARY_NAME:
        raise CollectorError(
            "benchmark binary must be named inferedge-bench-concurrent, "
            f"got {binary.resolve().name}"
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

    version = query_text(runner, [str(binary), "--version"])
    version_problems = version_text_problems(version.stdout, version.stderr, version.returncode)
    if version_problems:
        raise CollectorError("; ".join(version_problems))

    verify_pinned_files(repo_root, commit, blob_sha256)
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

    add_model = (repo_root / "models" / "add_f32.onnx").resolve()
    mnist_model = (repo_root / "models" / "mnist_mlp.onnx").resolve()
    fixture_dir = (repo_root / "models" / "fixtures" / "mnist").resolve()
    tools = compiler_and_cmake(runner, cache["CMAKE_CXX_COMPILER"])
    host = hardware_metadata(runner)
    pilots = [trial["trial_id"] for trial in execution_plan() if trial["pilot"]]
    return {
        "queried": {
            "git_commit": actual_commit,
            "git_status_porcelain": status.stdout,
            "approved_untracked_paths": approved_untracked_paths(
                status.stdout,
                allowed_untracked=ALLOWED_UNTRACKED,
            ),
            "started_utc": utc_now(),
            "operating_system": host["system"],
            "os_version": host["os_version"],
            "platform": host["platform"],
            "macos_version": host["mac_version"],
            "hardware_model": host["hardware_model"],
            "hardware_model_error": host["hardware_model_error"],
            "cpu_architecture": host["machine"],
            "python_version": host["python_version"],
            "logical_cpu_count": host["logical_cpu_count"],
            "logical_cpu_count_source": host["logical_cpu_count_source"],
            "logical_cpu_count_error": host["logical_cpu_count_error"],
            "hw_physicalcpu": host["hw_physicalcpu"],
            "hw_physicalcpu_error": host["hw_physicalcpu_error"],
            "hw_logicalcpu": host["hw_logicalcpu"],
            "hw_logicalcpu_error": host["hw_logicalcpu_error"],
            "hw_ncpu": host["hw_ncpu"],
            "hw_ncpu_error": host["hw_ncpu_error"],
            "nproc_cpu_count": host["nproc_cpu_count"],
            "nproc_cpu_count_error": host["nproc_cpu_count_error"],
            "compiler_path": tools["compiler_path"],
            "compiler_version": tools["compiler_version"],
            "cmake_version": tools["cmake_version"],
            "benchmark_binary_path": str(binary.resolve()),
            "benchmark_binary_sha256": sha256_file(binary),
            "benchmark_tool_version": TOOL_VERSION,
            "onnxruntime_link_output": linked.stdout,
            "onnxruntime_resolved_library": str(resolved_library),
        },
        "verified_configuration": {
            "cmake_build_type": cache["CMAKE_BUILD_TYPE"],
            "cmake_cxx_flags": cache["CMAKE_CXX_FLAGS"],
            "cmake_cxx_flags_release": cache["CMAKE_CXX_FLAGS_RELEASE"],
            "cmake_exe_linker_flags": cache["CMAKE_EXE_LINKER_FLAGS"],
            "cmake_exe_linker_flags_release": cache["CMAKE_EXE_LINKER_FLAGS_RELEASE"],
            "sanitizer_instrumentation": "absent",
            "onnxruntime_version": onnx_version,
            "add_model_path": str(add_model),
            "add_model_sha256": ADD_MODEL_SHA256,
            "mnist_model_path": str(mnist_model),
            "mnist_model_sha256": MNIST_MODEL_SHA256,
            "mnist_fixture_directory": str(fixture_dir),
            "mnist_fixture_id": MNIST_FIXTURE_ID,
            "mnist_fixture_inputs_sha256": MNIST_INPUT_SHA256,
            "mnist_fixture_reference_sha256": MNIST_REFERENCE_SHA256,
            "mnist_fixture_manifest_sha256": MNIST_FIXTURE_MANIFEST_SHA256,
        },
        "source_defined_policy": {
            "backend": BACKEND,
            "session_ownership": SESSION_OWNERSHIP,
            "onnx_intra_op_threads_per_session": ONNX_INTRA_OP_THREADS_PER_SESSION,
            "graph_optimizations": GRAPH_OPTIMIZATIONS,
            "onnxruntime_version_pin": ONNXRUNTIME_VERSION,
            "allowed_untracked_roots": list(ALLOWED_UNTRACKED),
            "note": (
                "Each worker has its own InferenceEngine, one intra-op thread, "
                "ORT_DISABLE_ALL, and the CPU provider. Aggregate throughput uses the "
                "shared wall interval. These values are source policy, not a hardware probe."
            ),
        },
        "collection_protocol": {
            "workloads": ["add", "mnist"],
            "worker_counts": list(APPROVED_WORKER_COUNTS),
            "warmup_iterations_per_worker": WARMUP_ITERATIONS_PER_WORKER,
            "measured_iterations_per_worker": MEASURED_ITERATIONS_PER_WORKER,
            "correctness_check_interval": CORRECTNESS_CHECK_INTERVAL,
            "counted_runs_per_configuration": COUNTED_RUNS_PER_CONFIGURATION,
            "pilot_count": len(pilots),
            "counted_trial_count": COUNTED_RUNS_PER_CONFIGURATION * len(CANONICAL_CONFIGURATIONS),
            "process_count": len(execution_plan()),
            "pilots": pilots,
            "trial_order": trial_order(),
            "inter_trial_delay_seconds": INTER_TRIAL_DELAY_SECONDS,
            "environment_variable": "ORT_DISABLE_TELEMETRY=1",
            "ordering_tolerance": ORDERING_TOLERANCE,
            "aggregate_relative_tolerance": AGGREGATE_RELATIVE_TOLERANCE,
        },
        "paths": {
            "add_model": add_model,
            "mnist_model": mnist_model,
            "fixture_dir": fixture_dir,
        },
        "commit": actual_commit,
    }


def _write_failure(
    run_set: pathlib.Path,
    *,
    trial_id: str,
    reason: str,
    command: list[str],
    started: str,
    finished: str,
    returncode: int,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
) -> None:
    write_json(
        run_set / "failure.json",
        {
            "status": "failed",
            "accepted": False,
            "trial_id": trial_id,
            "reason": reason,
            "command": command,
            "started_utc": started,
            "finished_utc": finished,
            "returncode": returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "preserved_earlier_trials": True,
            "aggregate": None,
        },
    )


def execute_trials(
    *,
    run_set: pathlib.Path,
    binary: pathlib.Path,
    repo_root: pathlib.Path,
    preflight_result: dict,
    runner=subprocess_runner,
    sleep=time.sleep,
) -> bool:
    paths = preflight_result["paths"]
    runs = run_set / "runs"
    for index, trial in enumerate(execution_plan()):
        trial_id = str(trial["trial_id"])
        workload = str(trial["workload"])
        worker_count = int(trial["worker_count"])
        started = utc_now()
        command: list[str] = []
        try:
            if index > 0 and INTER_TRIAL_DELAY_SECONDS > 0:
                sleep(INTER_TRIAL_DELAY_SECONDS)
            command = concurrent_command(
                binary,
                workload=workload,
                worker_count=worker_count,
                add_model=paths["add_model"],
                mnist_model=paths["mnist_model"],
                fixture_dir=paths["fixture_dir"],
            )
            env = dict(os.environ)
            env["ORT_DISABLE_TELEMETRY"] = "1"
            result = runner(command, cwd=repo_root, env=env)
        except Exception as exc:
            result = ProcessResult(127, "", f"failed to execute trial {trial_id}: {exc}")
        finished = utc_now()
        stdout_path = runs / f"{trial_id}.stdout"
        stderr_path = runs / f"{trial_id}.stderr"
        report_path = runs / f"{trial_id}.json"
        output_problems: list[str] = []
        try:
            stdout_path.write_text(result.stdout, encoding="utf-8")
            stderr_path.write_text(result.stderr, encoding="utf-8")
        except OSError as exc:
            output_problems.append(f"could not preserve trial output: {exc}")
        model_path = str(paths["add_model"] if workload == "add" else paths["mnist_model"])
        problems: list[str] = []
        if result.returncode != 0:
            problems.append(f"exit status {result.returncode}")
        if result.stderr != "":
            problems.append("stderr is not empty")
        if not problems and not output_problems:
            problems.extend(
                validate_concurrent_report(
                    result.stdout,
                    workload=workload,
                    model_path=model_path,
                    worker_count=worker_count,
                )
            )
        if not problems and not output_problems:
            try:
                report_path.write_text(result.stdout, encoding="utf-8")
            except OSError as exc:
                output_problems.append(f"could not preserve accepted report: {exc}")
        problems.extend(output_problems)
        meta = {
            "trial_id": trial_id,
            "workload": workload,
            "worker_count": worker_count,
            "pilot": bool(trial["pilot"]),
            "accepted": not problems,
            "command": command,
            "started_utc": started,
            "finished_utc": finished,
            "returncode": result.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "problems": problems,
        }
        write_json(runs / f"{trial_id}.meta.json", meta)
        if problems:
            _write_failure(
                run_set,
                trial_id=trial_id,
                reason="; ".join(problems),
                command=command,
                started=started,
                finished=finished,
                returncode=result.returncode,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            return False
    return True


def _mark_failed(run_set: pathlib.Path, checked: dict, run_set_id: str, reason: str) -> None:
    try:
        write_json(run_set / "environment.json", public_manifest(checked, run_set_id, "failed"))
        failure_path = run_set / "failure.json"
        if not failure_path.exists():
            write_json(
                failure_path,
                {
                    "status": "failed",
                    "accepted": False,
                    "trial_id": None,
                    "reason": reason,
                    "preserved_earlier_trials": True,
                    "aggregate": None,
                },
            )
    except OSError:
        return


def open_run_set(parent: pathlib.Path, run_set_id: str) -> pathlib.Path:
    """Create one new run set. An existing directory is left unchanged."""
    try:
        return create_run_set(parent, run_set_id)
    except CollectorError:
        raise
    except OSError as exc:
        created = parent / run_set_id
        if created.is_dir():
            try:
                write_json(
                    created / "failure.json",
                    {
                        "status": "failed",
                        "accepted": False,
                        "trial_id": None,
                        "reason": f"could not finish creating the run set: {exc}",
                        "preserved_earlier_trials": True,
                        "aggregate": None,
                    },
                )
            except OSError:
                pass
        raise CollectorError(f"could not create the run-set directory: {exc}") from exc


def collect(
    args: argparse.Namespace,
    runner=subprocess_runner,
    sleep=time.sleep,
    blob_sha256=git_blob_sha256,
) -> pathlib.Path:
    repo_root = pathlib.Path(args.repo_root).resolve()
    binary = pathlib.Path(args.binary).resolve()
    checked = preflight(
        repo_root=repo_root,
        binary=binary,
        expected_commit=args.expected_commit,
        runner=runner,
        blob_sha256=blob_sha256,
    )
    run_set_id = make_run_set_id(datetime.datetime.now(datetime.timezone.utc), checked["commit"])
    run_set = open_run_set(repo_root / "benchmarks" / "concurrent_runs", run_set_id)
    try:
        write_json(run_set / "environment.json", public_manifest(checked, run_set_id, "running"))
        accepted = execute_trials(
            run_set=run_set,
            binary=binary,
            repo_root=repo_root,
            preflight_result=checked,
            runner=runner,
            sleep=sleep,
        )
        if accepted and (run_set / "aggregate.json").exists():
            accepted = False
            write_json(
                run_set / "failure.json",
                {
                    "status": "failed",
                    "accepted": False,
                    "trial_id": None,
                    "reason": "collector must not write aggregate.json",
                    "preserved_earlier_trials": True,
                    "aggregate": None,
                },
            )
        status = "completed" if accepted else "failed"
        write_json(run_set / "environment.json", public_manifest(checked, run_set_id, status))
    except CollectorError:
        _mark_failed(run_set, checked, run_set_id, "collection stopped")
        raise
    except Exception as exc:
        _mark_failed(run_set, checked, run_set_id, f"collection error: {exc}")
        raise CollectorError(f"collection error: {exc}") from exc
    if not accepted:
        raise CollectorError(f"collection stopped; see {run_set / 'failure.json'}")
    return run_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect inferedge-bench-concurrent JSON trials into a new concurrent run-set directory."
        )
    )
    parser.add_argument(
        "--binary",
        required=True,
        help="Path to the Release inferedge-bench-concurrent binary",
    )
    parser.add_argument(
        "--repo-root",
        default=str(pathlib.Path(__file__).resolve().parents[1]),
        help="Repository root. Defaults to the parent of benchmarks/",
    )
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="Full 40-character lowercase Git SHA that HEAD and the built binary must match",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check the Release binary, models, fixtures, and repository, then exit",
    )
    return parser


def main(
    argv: list[str] | None = None,
    runner=subprocess_runner,
    sleep=time.sleep,
    blob_sha256=git_blob_sha256,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        require_commit_sha(args.expected_commit)
        if args.preflight_only:
            checked = preflight(
                repo_root=pathlib.Path(args.repo_root).resolve(),
                binary=pathlib.Path(args.binary).resolve(),
                expected_commit=args.expected_commit,
                runner=runner,
                blob_sha256=blob_sha256,
            )
            print("preflight passed")
            print(checked["queried"]["onnxruntime_resolved_library"])
            return 0
        run_set = collect(args, runner=runner, sleep=sleep, blob_sha256=blob_sha256)
    except CollectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(run_set)
    return 0


if __name__ == "__main__":
    sys.exit(main())
