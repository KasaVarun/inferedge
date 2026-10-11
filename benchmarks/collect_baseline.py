#!/usr/bin/env python3
"""Collect schema-version-1 reports from the native inferedge-bench executable.

This module does not run a neural network itself. It checks the build and the
repository, then starts one inferedge-bench process per trial. It does not
write aggregate.json. A failed trial stops the run set and leaves the files
already written in place.
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
import subprocess
import sys
import time

EXPECTED_COMMIT = "6c1ba833a512e288b436e06a0bd430b41988518b"
TOOL_NAME = "inferedge-bench"
TOOL_VERSION = "0.1.0"
ONNXRUNTIME_VERSION = "1.31.0"
BACKEND = "ONNX Runtime CPU"
ADD_MODEL_SHA256 = "37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447"
MNIST_MODEL_SHA256 = "3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce"
MNIST_FIXTURE_ID = "test_00003"
CALLING_THREADS = 1
ONNX_INTRA_OP_THREADS = 1
GRAPH_OPTIMIZATIONS = "disabled"
HISTORICAL_UNTRACKED = "models/experiments"
ORDERING_TOLERANCE = 1e-9

REQUIRED_CACHE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_CXX_COMPILER",
    "CMAKE_CXX_FLAGS",
    "CMAKE_CXX_FLAGS_RELEASE",
    "CMAKE_EXE_LINKER_FLAGS",
    "CMAKE_EXE_LINKER_FLAGS_RELEASE",
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
    "measurement_loop_microseconds",
    "throughput_requests_per_second",
    "correctness",
)
CONFIGURATION_KEYS = (
    "warmup_iterations",
    "measured_iterations",
    "correctness_check_interval",
    "calling_threads",
    "onnx_intra_op_threads",
    "graph_optimizations",
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
THROUGHPUT_KEYS = ("active_call", "observed_sequential")


class CollectorError(Exception):
    """The run set must not continue."""


class ProcessResult:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def subprocess_runner(
    args: list[str],
    *,
    cwd: pathlib.Path | None = None,
    env: dict[str, str] | None = None,
) -> ProcessResult:
    """Run one command without a shell. The caller supplies the argument array."""
    try:
        completed = subprocess.run(
            args,
            cwd=None if cwd is None else str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, UnicodeError) as exc:
        return ProcessResult(127, "", f"failed to execute {args[0]}: {exc}")
    return ProcessResult(completed.returncode, completed.stdout, completed.stderr)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_cmake_cache(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        left, value = line.split("=", 1)
        if ":" not in left:
            continue
        name = left.split(":", 1)[0]
        values[name] = value
    return values


def cache_flag_text(cache: dict[str, str]) -> str:
    parts = []
    for name, value in cache.items():
        if "FLAGS" in name or name.endswith("_LINKER_FLAGS"):
            parts.append(f"{name}={value}")
    return "\n".join(parts)


def sanitizer_flag_problems(cache: dict[str, str]) -> list[str]:
    problems = []
    for name, value in cache.items():
        if "FLAGS" not in name:
            continue
        lowered = value.lower()
        if "fsanitize" in lowered or "libasan" in lowered or "libubsan" in lowered:
            problems.append(f"{name} enables sanitizer instrumentation: {value}")
    return problems


def assess_build_cache(cache: dict[str, str]) -> list[str]:
    """Release is required. Optimization spellings are recorded, not assumed."""
    problems = []
    for key in REQUIRED_CACHE_KEYS:
        if key not in cache:
            problems.append(f"CMakeCache.txt has no {key}; the build flags cannot be verified")
    if problems:
        return problems
    build_type = cache["CMAKE_BUILD_TYPE"]
    if build_type != "Release":
        problems.append(
            f"CMAKE_BUILD_TYPE is {build_type or 'empty'}, expected Release. "
            "Debug, RelWithDebInfo, MinSizeRel, and multi-config generators are rejected"
        )
    problems.extend(sanitizer_flag_problems(cache))
    if not cache["CMAKE_CXX_COMPILER"]:
        problems.append("CMAKE_CXX_COMPILER is empty")
    return problems


def sanitizer_symbol_problems(nm_output: str) -> list[str]:
    markers = ("__asan", "___asan", "__ubsan", "asan_report", "ubsan_")
    problems = []
    for line in nm_output.splitlines():
        if any(marker in line for marker in markers):
            problems.append(f"benchmark binary contains sanitizer symbol: {line.strip()}")
            break
    return problems


def onnx_link_problems(link_output: str) -> list[str]:
    accepted = ("libonnxruntime.1.31.0.dylib", "libonnxruntime.so.1.31.0")
    if any(name in link_output for name in accepted):
        return []
    return [
        "benchmark binary does not link the pinned ONNX Runtime 1.31.0 CPU library "
        "(expected libonnxruntime.1.31.0.dylib or libonnxruntime.so.1.31.0)"
    ]


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


def make_run_set_id(moment: datetime.datetime, commit: str) -> str:
    if len(commit) < 7 or any(character not in "0123456789abcdef" for character in commit[:7]):
        raise CollectorError(f"git commit is not a hex SHA: {commit}")
    stamp = moment.astimezone(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{commit[:7]}"


def normalize_status_path(path: str) -> str:
    cleaned = path.strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].replace("\\\\", "\\").replace('\\"', '"')
    return cleaned.rstrip("/")


def path_is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def git_status_problems(
    porcelain: str,
    *,
    expected_commit: str,
    actual_commit: str,
    allowed_untracked: tuple[str, ...] = (HISTORICAL_UNTRACKED,),
) -> list[str]:
    problems = []
    if actual_commit != expected_commit:
        problems.append(f"HEAD is {actual_commit}, expected {expected_commit}")
    for raw_line in porcelain.splitlines():
        if not raw_line.strip():
            continue
        if not raw_line.startswith("?? "):
            problems.append(f"tracked worktree change: {raw_line}")
            continue
        path = normalize_status_path(raw_line[3:])
        if any(path_is_under(path, allowed) for allowed in allowed_untracked):
            continue
        problems.append(f"unexpected untracked path: {path}")
    return problems


def approved_untracked_paths(
    porcelain: str,
    allowed_untracked: tuple[str, ...] = (HISTORICAL_UNTRACKED,),
) -> list[str]:
    paths = []
    for raw_line in porcelain.splitlines():
        if not raw_line.startswith("?? "):
            continue
        path = normalize_status_path(raw_line[3:])
        if any(path_is_under(path, allowed) for allowed in allowed_untracked):
            paths.append(path)
    return paths


def trial_ids(counted_runs: int) -> list[str]:
    if counted_runs < 1:
        raise CollectorError("counted runs must be at least 1")
    names = ["pilot-add", "pilot-mnist"]
    for number in range(1, counted_runs + 1):
        names.append(f"add-{number:02d}")
        names.append(f"mnist-{number:02d}")
    return names


def workload_for_trial(trial_id: str) -> str:
    if trial_id == "pilot-add" or trial_id.startswith("add-"):
        return "add"
    if trial_id == "pilot-mnist" or trial_id.startswith("mnist-"):
        return "mnist"
    raise CollectorError(f"unknown trial id: {trial_id}")


def bench_command(
    binary: pathlib.Path,
    *,
    workload: str,
    add_model: pathlib.Path,
    mnist_model: pathlib.Path,
    fixture_dir: pathlib.Path,
    warmup: int,
    iterations: int,
    check_interval: int,
) -> list[str]:
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
        *fixture_args,
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--check-interval",
        str(check_interval),
        "--format",
        "json",
    ]


def reject_json_constant(token: str) -> None:
    raise json.JSONDecodeError(f"nonstandard number {token}", token, 0)


def parse_single_document(text: str) -> tuple[dict | None, list[str]]:
    if text == "":
        return None, ["stdout is empty"]
    if not text.endswith("\n"):
        return None, ["stdout is missing the trailing newline"]
    if text.endswith("\n\n"):
        return None, ["stdout has more than one trailing newline"]
    body = text[:-1]
    if not body.startswith("{"):
        return None, ["stdout has leading text before the JSON object"]
    decoder = json.JSONDecoder(parse_constant=reject_json_constant)
    try:
        value, end = decoder.raw_decode(body)
    except json.JSONDecodeError as exc:
        return None, [f"stdout is not valid JSON: {exc.msg}"]
    if body[end:] != "":
        return None, ["stdout has extra data after the JSON object"]
    if not isinstance(value, dict):
        return None, ["stdout JSON is not an object"]
    return value, []


def require_exact_keys(value: dict, keys: tuple[str, ...], label: str, problems: list[str]) -> None:
    missing = [key for key in keys if key not in value]
    extra = [key for key in value if key not in keys]
    if missing:
        problems.append(f"{label} is missing {', '.join(missing)}")
    if extra:
        problems.append(f"{label} has unexpected {', '.join(extra)}")


def json_integer(value: object, label: str, problems: list[str]) -> int | None:
    if type(value) is not int:
        problems.append(f"{label} is not a JSON integer")
        return None
    return value


def json_number(value: object, label: str, problems: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        problems.append(f"{label} is not a JSON number")
        return None
    number = float(value)
    if not math.isfinite(number):
        problems.append(f"{label} is not finite")
        return None
    return number


def negative_sign(value: float) -> bool:
    return math.copysign(1.0, value) < 0


def ordered_metric(earlier: float, later: float) -> bool:
    """Same 1e-9 allowance the C++ writer uses for equivalent representations."""
    if earlier <= later:
        return True
    scale = max(abs(earlier), abs(later), 1.0)
    return (earlier - later) <= scale * ORDERING_TOLERANCE


def validate_report(
    text: str,
    *,
    workload: str,
    model_path: str,
    warmup: int,
    iterations: int,
    check_interval: int,
    tool_version: str = TOOL_VERSION,
) -> list[str]:
    document, problems = parse_single_document(text)
    if document is None:
        return problems
    require_exact_keys(document, TOP_LEVEL_KEYS, "report", problems)
    schema = json_integer(document.get("schema_version"), "schema_version", problems)
    if schema is not None and schema != 1:
        problems.append(f"schema_version is {schema}, expected 1")
    tool = document.get("tool")
    if not isinstance(tool, dict):
        problems.append("tool is not an object")
    else:
        require_exact_keys(tool, ("name", "version"), "tool", problems)
        if tool.get("name") != TOOL_NAME:
            problems.append(f"tool.name is {tool.get('name')!r}, expected {TOOL_NAME!r}")
        if tool.get("version") != tool_version:
            problems.append(f"tool.version is {tool.get('version')!r}, expected {tool_version!r}")
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
    expected_counts = {
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "correctness_check_interval": check_interval,
        "calling_threads": CALLING_THREADS,
        "onnx_intra_op_threads": ONNX_INTRA_OP_THREADS,
    }
    for key, expected in expected_counts.items():
        found = json_integer(configuration.get(key), key, problems)
        if found is not None and found != expected:
            problems.append(f"{key} is {found}, expected {expected}")
    if configuration.get("graph_optimizations") != GRAPH_OPTIMIZATIONS:
        problems.append(
            "graph_optimizations is "
            f"{configuration.get('graph_optimizations')!r}, expected {GRAPH_OPTIMIZATIONS!r}"
        )

    latency = document.get("latency_microseconds")
    if not isinstance(latency, dict):
        problems.append("latency_microseconds is not an object")
        latency = {}
    else:
        require_exact_keys(latency, LATENCY_KEYS, "latency_microseconds", problems)
    sample_count = json_integer(latency.get("sample_count"), "sample_count", problems)
    if sample_count is not None and sample_count != iterations:
        problems.append(f"sample_count is {sample_count}, expected {iterations}")
    metrics = {}
    for key in LATENCY_KEYS:
        if key == "sample_count":
            continue
        metrics[key] = json_number(latency.get(key), key, problems)
    if all(metrics[key] is not None for key in ("minimum", "p50", "p95", "p99", "maximum")):
        chain = (
            ("minimum", "p50"),
            ("p50", "p95"),
            ("p95", "p99"),
            ("p99", "maximum"),
        )
        for earlier, later in chain:
            if not ordered_metric(metrics[earlier], metrics[later]):
                problems.append(f"latency metrics are out of order at {earlier} and {later}")
                break
    deviation = metrics.get("population_standard_deviation")
    if deviation is not None and negative_sign(deviation):
        problems.append("population standard deviation is negative")
    loop = json_number(
        document.get("measurement_loop_microseconds"),
        "measurement_loop_microseconds",
        problems,
    )
    if loop is not None and (loop <= 0 or negative_sign(loop)):
        problems.append("measurement loop duration is not positive")

    throughput = document.get("throughput_requests_per_second")
    if not isinstance(throughput, dict):
        problems.append("throughput_requests_per_second is not an object")
        throughput = {}
    else:
        require_exact_keys(throughput, THROUGHPUT_KEYS, "throughput_requests_per_second", problems)
    for key in THROUGHPUT_KEYS:
        value = json_number(throughput.get(key), key, problems)
        if value is not None and (value <= 0 or negative_sign(value)):
            problems.append(f"{key} throughput is not positive")
    return problems


def locate_build_dir(binary: pathlib.Path) -> pathlib.Path:
    cache = binary.parent / "CMakeCache.txt"
    if not cache.is_file():
        raise CollectorError(
            "CMakeCache.txt was not found beside the benchmark binary at "
            f"{cache}. The binary must be the inferedge-bench produced in its "
            "CMake build directory, not a copy"
        )
    return binary.parent


def read_version_number(cache: dict[str, str], build_dir: pathlib.Path) -> str:
    root = cache.get("INFEREDGE_ONNXRUNTIME_ROOT", "")
    candidates: list[pathlib.Path] = []
    if root:
        candidates.append(pathlib.Path(root) / "VERSION_NUMBER")
    else:
        extracted = build_dir / "_deps" / "onnxruntime"
        if extracted.is_dir():
            candidates.extend(sorted(extracted.glob("*/VERSION_NUMBER")))
    files = [path for path in candidates if path.is_file()]
    if len(files) != 1:
        raise CollectorError(
            "expected one ONNX Runtime VERSION_NUMBER file under the build "
            f"directory, found {len(files)}. Refusing to guess the runtime version"
        )
    text = files[0].read_text(encoding="utf-8").strip()
    if text != ONNXRUNTIME_VERSION:
        raise CollectorError(
            f"ONNX Runtime VERSION_NUMBER is {text!r}, expected {ONNXRUNTIME_VERSION!r}"
        )
    return text


def query_text(runner, args: list[str], cwd: pathlib.Path | None = None) -> ProcessResult:
    try:
        return runner(args, cwd=cwd, env=None)
    except TypeError:
        try:
            return runner(args)
        except OSError as exc:
            return ProcessResult(127, "", f"failed to execute {args[0]}: {exc}")
    except OSError as exc:
        return ProcessResult(127, "", f"failed to execute {args[0]}: {exc}")


def hardware_metadata(runner) -> dict[str, object]:
    """Record queried host facts. A missing hardware model stays null."""
    metadata: dict[str, object] = {
        "system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mac_version": platform.mac_ver()[0] or None,
        "hardware_model": None,
        "hardware_model_error": None,
    }
    if platform.system() != "Darwin":
        metadata["hardware_model_error"] = "hw.model is queried only on Darwin"
        return metadata
    result = query_text(runner, ["sysctl", "-n", "hw.model"])
    if result.returncode == 0 and result.stdout.strip():
        metadata["hardware_model"] = result.stdout.strip()
    else:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        metadata["hardware_model_error"] = f"sysctl -n hw.model failed: {detail}"
    if not metadata["mac_version"]:
        sw = query_text(runner, ["sw_vers", "-productVersion"])
        if sw.returncode == 0 and sw.stdout.strip():
            metadata["mac_version"] = sw.stdout.strip()
    return metadata


def compiler_and_cmake(runner, compiler: str) -> dict[str, str]:
    compiler_result = query_text(runner, [compiler, "--version"])
    if compiler_result.returncode != 0:
        raise CollectorError(
            f"compiler {compiler} --version failed: {compiler_result.stderr.strip()}"
        )
    cmake_result = query_text(runner, ["cmake", "--version"])
    if cmake_result.returncode != 0:
        raise CollectorError(f"cmake --version failed: {cmake_result.stderr.strip()}")
    return {
        "compiler_path": compiler,
        "compiler_version": compiler_result.stdout,
        "cmake_version": cmake_result.stdout,
    }


def link_tool_args(binary: pathlib.Path) -> list[str]:
    if platform.system() == "Darwin":
        return ["otool", "-L", str(binary)]
    return ["ldd", str(binary)]


def symbol_tool_args(binary: pathlib.Path) -> list[str]:
    if platform.system() == "Darwin":
        return ["nm", "-gU", str(binary)]
    return ["nm", "-g", str(binary)]


def preflight(
    *,
    repo_root: pathlib.Path,
    binary: pathlib.Path,
    expected_commit: str,
    warmup: int,
    iterations: int,
    check_interval: int,
    counted_runs: int,
    delay_seconds: float,
    runner=subprocess_runner,
) -> dict:
    if not binary.is_file():
        raise CollectorError(f"benchmark binary does not exist: {binary}")
    if not os.access(binary, os.X_OK):
        raise CollectorError(f"benchmark binary is not executable: {binary}")
    build_dir = locate_build_dir(binary)
    cache = parse_cmake_cache((build_dir / "CMakeCache.txt").read_text(encoding="utf-8"))
    problems = assess_build_cache(cache)
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

    linked = query_text(runner, link_tool_args(binary))
    if linked.returncode != 0:
        raise CollectorError(
            "could not identify the ONNX Runtime library linked into the benchmark binary: "
            f"{linked.stderr.strip() or linked.stdout.strip()}"
        )
    link_problems = onnx_link_problems(linked.stdout + linked.stderr)
    if link_problems:
        raise CollectorError("; ".join(link_problems))
    onnx_version = read_version_number(cache, build_dir)

    version = query_text(runner, [str(binary), "--version"])
    version_problems = version_text_problems(version.stdout, version.stderr, version.returncode)
    if version_problems:
        raise CollectorError("; ".join(version_problems))

    add_model = (repo_root / "models" / "add_f32.onnx").resolve()
    mnist_model = (repo_root / "models" / "mnist_mlp.onnx").resolve()
    fixture_dir = (repo_root / "models" / "fixtures" / "mnist").resolve()
    for path, expected in ((add_model, ADD_MODEL_SHA256), (mnist_model, MNIST_MODEL_SHA256)):
        if not path.is_file():
            raise CollectorError(f"model file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise CollectorError(
                f"SHA-256 mismatch for {path}: expected {expected}, actual {actual}"
            )
    for relative in (
        pathlib.Path("inputs") / f"{MNIST_FIXTURE_ID}.f32",
        pathlib.Path("onnx") / f"{MNIST_FIXTURE_ID}.f32",
    ):
        fixture_path = fixture_dir / relative
        if not fixture_path.is_file():
            raise CollectorError(f"MNIST fixture file is missing: {fixture_path}")

    head = query_text(runner, ["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    status = query_text(runner, ["git", "-C", str(repo_root), "status", "--porcelain"])
    if head.returncode != 0 or status.returncode != 0:
        raise CollectorError(
            "git verification failed: " + (head.stderr.strip() or status.stderr.strip())
        )
    commit = head.stdout.strip()
    git_problems = git_status_problems(
        status.stdout,
        expected_commit=expected_commit,
        actual_commit=commit,
    )
    if git_problems:
        raise CollectorError("; ".join(git_problems))

    tools = compiler_and_cmake(runner, cache["CMAKE_CXX_COMPILER"])
    host = hardware_metadata(runner)
    return {
        "queried": {
            "git_commit": commit,
            "git_status_porcelain": status.stdout,
            "approved_untracked_paths": approved_untracked_paths(status.stdout),
            "started_utc": utc_now(),
            "operating_system": host["system"],
            "platform": host["platform"],
            "macos_version": host["mac_version"],
            "hardware_model": host["hardware_model"],
            "hardware_model_error": host["hardware_model_error"],
            "cpu_architecture": host["machine"],
            "compiler_path": tools["compiler_path"],
            "compiler_version": tools["compiler_version"],
            "cmake_version": tools["cmake_version"],
            "benchmark_binary_path": str(binary.resolve()),
            "benchmark_binary_sha256": sha256_file(binary),
            "benchmark_tool_version": TOOL_VERSION,
            "onnxruntime_link_output": linked.stdout,
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
        },
        "source_defined_policy": {
            "backend": BACKEND,
            "calling_threads": CALLING_THREADS,
            "onnx_intra_op_threads": ONNX_INTRA_OP_THREADS,
            "graph_optimizations": GRAPH_OPTIMIZATIONS,
            "onnxruntime_version_pin": ONNXRUNTIME_VERSION,
            "allowed_untracked_roots": [HISTORICAL_UNTRACKED],
            "note": (
                "InferenceEngine sets one intra-op thread, ORT_DISABLE_ALL, and the "
                "CPU provider. run_benchmark uses the calling thread. These values "
                "are source policy, not a hardware probe."
            ),
        },
        "collection_protocol": {
            "workloads": ["add", "mnist"],
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "correctness_check_interval": check_interval,
            "counted_runs_per_workload": counted_runs,
            "pilots": ["pilot-add", "pilot-mnist"],
            "trial_order": trial_ids(counted_runs),
            "inter_trial_delay_seconds": delay_seconds,
            "environment_variable": "ORT_DISABLE_TELEMETRY=1",
        },
        "paths": {
            "add_model": add_model,
            "mnist_model": mnist_model,
            "fixture_dir": fixture_dir,
        },
        "commit": commit,
    }


def write_json(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def public_manifest(preflight_result: dict, run_set_id: str, status: str) -> dict:
    return {
        "run_set_id": run_set_id,
        "status": status,
        "aggregate": None,
        "queried": preflight_result["queried"],
        "verified_configuration": preflight_result["verified_configuration"],
        "source_defined_policy": preflight_result["source_defined_policy"],
        "collection_protocol": preflight_result["collection_protocol"],
    }


def create_run_set(baselines_dir: pathlib.Path, run_set_id: str) -> pathlib.Path:
    baselines_dir.mkdir(parents=True, exist_ok=True)
    path = baselines_dir / run_set_id
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise CollectorError(f"run-set directory already exists: {path}") from exc
    (path / "runs").mkdir()
    return path


def execute_trials(
    *,
    run_set: pathlib.Path,
    binary: pathlib.Path,
    repo_root: pathlib.Path,
    preflight_result: dict,
    counted_runs: int,
    warmup: int,
    iterations: int,
    check_interval: int,
    delay_seconds: float,
    runner=subprocess_runner,
    sleep=time.sleep,
) -> bool:
    paths = preflight_result["paths"]
    runs = run_set / "runs"
    for index, trial_id in enumerate(trial_ids(counted_runs)):
        if index > 0 and delay_seconds > 0:
            sleep(delay_seconds)
        workload = workload_for_trial(trial_id)
        command = bench_command(
            binary,
            workload=workload,
            add_model=paths["add_model"],
            mnist_model=paths["mnist_model"],
            fixture_dir=paths["fixture_dir"],
            warmup=warmup,
            iterations=iterations,
            check_interval=check_interval,
        )
        env = dict(os.environ)
        env["ORT_DISABLE_TELEMETRY"] = "1"
        started = utc_now()
        result = runner(command, cwd=repo_root, env=env)
        finished = utc_now()
        stdout_path = runs / f"{trial_id}.stdout"
        stderr_path = runs / f"{trial_id}.stderr"
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        model_path = str(paths["add_model"] if workload == "add" else paths["mnist_model"])
        problems: list[str] = []
        if result.returncode != 0:
            problems.append(f"exit status {result.returncode}")
        elif result.stderr != "":
            problems.append("stderr is not empty")
        else:
            problems.extend(
                validate_report(
                    result.stdout,
                    workload=workload,
                    model_path=model_path,
                    warmup=warmup,
                    iterations=iterations,
                    check_interval=check_interval,
                )
            )
        meta = {
            "trial_id": trial_id,
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
            write_json(
                run_set / "failure.json",
                {
                    "status": "failed",
                    "accepted": False,
                    "trial_id": trial_id,
                    "reason": "; ".join(problems),
                    "command": command,
                    "started_utc": started,
                    "finished_utc": finished,
                    "returncode": result.returncode,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "preserved_earlier_trials": True,
                    "aggregate": None,
                },
            )
            return False
        (runs / f"{trial_id}.json").write_text(result.stdout, encoding="utf-8")
    return True


def collect(args: argparse.Namespace, runner=subprocess_runner, sleep=time.sleep) -> pathlib.Path:
    repo_root = pathlib.Path(args.repo_root).resolve()
    binary = pathlib.Path(args.binary).resolve()
    checked = preflight(
        repo_root=repo_root,
        binary=binary,
        expected_commit=args.expected_commit,
        warmup=args.warmup,
        iterations=args.iterations,
        check_interval=args.check_interval,
        counted_runs=args.runs,
        delay_seconds=args.delay_seconds,
        runner=runner,
    )
    run_set_id = make_run_set_id(datetime.datetime.now(datetime.timezone.utc), checked["commit"])
    run_set = create_run_set(repo_root / "benchmarks" / "baselines", run_set_id)
    write_json(run_set / "environment.json", public_manifest(checked, run_set_id, "running"))
    accepted = execute_trials(
        run_set=run_set,
        binary=binary,
        repo_root=repo_root,
        preflight_result=checked,
        counted_runs=args.runs,
        warmup=args.warmup,
        iterations=args.iterations,
        check_interval=args.check_interval,
        delay_seconds=args.delay_seconds,
        runner=runner,
        sleep=sleep,
    )
    status = "completed" if accepted else "failed"
    write_json(run_set / "environment.json", public_manifest(checked, run_set_id, status))
    if not accepted:
        raise CollectorError(f"collection stopped; see {run_set / 'failure.json'}")
    return run_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect inferedge-bench JSON trials into a new run-set directory."
    )
    parser.add_argument("--binary", required=True, help="Path to the Release inferedge-bench")
    parser.add_argument(
        "--repo-root",
        default=str(pathlib.Path(__file__).resolve().parents[1]),
        help="Repository root. Defaults to the parent of benchmarks/",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--check-interval", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5, help="Counted runs per workload")
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    parser.add_argument("--expected-commit", default=EXPECTED_COMMIT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_set = collect(args)
    except CollectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(run_set)
    return 0


if __name__ == "__main__":
    sys.exit(main())
