#!/usr/bin/env python3
"""Synthetic tests for the concurrent collector. They do not run inference."""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.collect_baseline import (  # noqa: E402
    HISTORICAL_UNTRACKED,
    validate_report,
    verify_linked_onnx_runtime,
)
from benchmarks.collect_baseline import (
    git_status_problems as sequential_git_status_problems,
)
from benchmarks.collect_baseline import (
    trial_ids as sequential_trial_ids,
)
from benchmarks.collect_concurrent import (  # noqa: E402
    ADD_MODEL_SHA256,
    ALLOWED_UNTRACKED,
    BINARY_NAME,
    MNIST_FIXTURE_MANIFEST_SHA256,
    MNIST_INPUT_SHA256,
    MNIST_MODEL_SHA256,
    MNIST_REFERENCE_SHA256,
    PINNED_FILES,
    TOOL_NAME,
    TOOL_VERSION,
    CollectorError,
    ProcessResult,
    build_parser,
    collect,
    concurrent_command,
    create_run_set,
    execution_plan,
    git_blob_sha256,
    main,
    preflight,
    release_optimization_problems,
    subprocess_runner,
    trial_order,
    validate_concurrent_report,
    verify_pinned_files,
)

PINNED_ORT_LIBRARIES = frozenset(
    {
        "libonnxruntime.1.31.0.dylib",
        "libonnxruntime.so.1.31.0",
    }
)
COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
WARMUP = 100
MEASURED = 1000
INTERVAL = 1

EXPECTED_ORDER = [
    "pilot-add-c01",
    "pilot-mnist-c01",
    "pilot-add-c02",
    "pilot-mnist-c02",
    "pilot-add-c04",
    "pilot-mnist-c04",
    "pilot-add-c08",
    "pilot-mnist-c08",
    "add-c01-01",
    "mnist-c01-01",
    "add-c02-01",
    "mnist-c02-01",
    "add-c04-01",
    "mnist-c04-01",
    "add-c08-01",
    "mnist-c08-01",
    "mnist-c01-02",
    "add-c02-02",
    "mnist-c02-02",
    "add-c04-02",
    "mnist-c04-02",
    "add-c08-02",
    "mnist-c08-02",
    "add-c01-02",
    "add-c02-03",
    "mnist-c02-03",
    "add-c04-03",
    "mnist-c04-03",
    "add-c08-03",
    "mnist-c08-03",
    "add-c01-03",
    "mnist-c01-03",
    "mnist-c02-04",
    "add-c04-04",
    "mnist-c04-04",
    "add-c08-04",
    "mnist-c08-04",
    "add-c01-04",
    "mnist-c01-04",
    "add-c02-04",
    "add-c04-05",
    "mnist-c04-05",
    "add-c08-05",
    "mnist-c08-05",
    "add-c01-05",
    "mnist-c01-05",
    "add-c02-05",
    "mnist-c02-05",
]


def sample_report(
    workload: str = "add",
    worker_count: int = 1,
    model_path: str = "/models/add_f32.onnx",
    **overrides: object,
) -> dict:
    total = worker_count * MEASURED
    wall_us = 1_000_000.0
    document = {
        "schema_version": 2,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "model": model_path,
        "workload": workload,
        "fixture": None if workload == "add" else "test_00003",
        "backend": "ONNX Runtime CPU",
        "configuration": {
            "worker_count": worker_count,
            "session_ownership": "one_inference_engine_per_worker",
            "onnx_intra_op_threads_per_session": 1,
            "graph_optimizations": "disabled",
            "warmup_iterations_per_worker": WARMUP,
            "measured_iterations_per_worker": MEASURED,
            "correctness_check_interval": INTERVAL,
        },
        "latency_microseconds": {
            "sample_count": total,
            "minimum": 1.0,
            "mean": 2.0,
            "p50": 2.0,
            "p95": 3.0,
            "p99": 4.0,
            "maximum": 5.0,
            "population_standard_deviation": 0.5,
        },
        "measurement_wall_microseconds": wall_us,
        "total_completed_requests": total,
        "per_worker_completed_requests": [MEASURED] * worker_count,
        "aggregate_completed_requests_per_second": total / (wall_us / 1_000_000.0),
        "active_call_requests_per_second": 50.0,
        "correctness": "passed",
    }
    for key, value in overrides.items():
        if key in document and isinstance(document[key], dict) and isinstance(value, dict):
            document[key].update(value)
        else:
            document[key] = value
    return document


def dump_report(document: dict) -> str:
    return json.dumps(document) + "\n"


def expect(text: str, workload: str = "add", worker_count: int = 1, model_path: str = "/models/add_f32.onnx") -> list[str]:
    return validate_concurrent_report(
        text,
        workload=workload,
        model_path=model_path,
        worker_count=worker_count,
    )


def configuration_key(trial_id: str) -> tuple[str, str]:
    body = trial_id[len("pilot-") :] if trial_id.startswith("pilot-") else trial_id
    workload, concurrency, *_rest = body.split("-")
    return workload, concurrency


class Stage:
    """A temporary Release tree and repository. The benchmark process is never real."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.build_dir = root / "build-concurrent"
        self.runtime = root / "onnxruntime-cpu"
        self.binary = self.build_dir / BINARY_NAME
        self.compiler = "/usr/bin/c++"
        self.head = COMMIT
        self.porcelain = ""
        self.version_stdout = f"{TOOL_NAME} {TOOL_VERSION}\n"
        self.ort_version = "1.31.0"
        self.build_type = "Release"
        self.cxx_flags = ""
        self.release_flags = "-O3 -DNDEBUG"
        self.nm_output = ""
        self.overrides: dict[str, bytes] = {}
        self.bad_blob = False
        self.foreign = False
        self.cpu_fail = False
        self.executable = True
        self.write_cache = True
        self.fail_trial: int | None = None
        self.fail_returncode = 0
        self.fail_stdout = ""
        self.fail_stderr = ""
        self.raise_on_trial: int | None = None
        self.trial_calls: list[list[str]] = []
        self.sleeps: list[float] = []
        self.library_name = (
            "libonnxruntime.1.31.0.dylib"
            if platform.system() == "Darwin"
            else "libonnxruntime.so.1.31.0"
        )

    def build(self) -> None:
        for relative, _digest in PINNED_FILES:
            destination = self.repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative in self.overrides:
                destination.write_bytes(self.overrides[relative])
            else:
                shutil.copyfile(REPO_ROOT / relative, destination)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.binary.write_bytes(b"#!/bin/sh\nexit 0\n")
        if self.executable:
            self.binary.chmod(0o755)
        else:
            self.binary.chmod(0o644)
        library = self.runtime / "lib"
        library.mkdir(parents=True)
        (self.runtime / "VERSION_NUMBER").write_text(self.ort_version + "\n", encoding="utf-8")
        versioned = library / self.library_name
        versioned.write_bytes(b"ort")
        foreign_dir = self.root / "foreign"
        foreign_dir.mkdir()
        foreign = foreign_dir / self.library_name
        foreign.write_bytes(b"foreign")
        if self.write_cache:
            (self.binary.parent / "CMakeCache.txt").write_text(self._cache_text(), encoding="utf-8")
        resolved = foreign if self.foreign else versioned
        rpath = foreign.parent if self.foreign else library
        self.link_output = (
            f"{BINARY_NAME}:\n"
            f"\t@rpath/{self.library_name} (compatibility version 1.0.0, current version 1.31.0)\n"
        )
        self.rpath_output = "          cmd LC_RPATH\n" f"         path {rpath} (offset 12)\n"
        self.ldd_output = f"\t{self.library_name} => {resolved} (0x1000)\n"

    def _cache_text(self) -> str:
        rows = {
            "CMAKE_BUILD_TYPE": self.build_type,
            "CMAKE_CXX_COMPILER": self.compiler,
            "CMAKE_CXX_FLAGS": self.cxx_flags,
            "CMAKE_CXX_FLAGS_RELEASE": self.release_flags,
            "CMAKE_EXE_LINKER_FLAGS": "",
            "CMAKE_EXE_LINKER_FLAGS_RELEASE": "",
            "INFEREDGE_ONNXRUNTIME_ROOT": str(self.runtime),
        }
        return "".join(f"{key}:STRING={value}\n" for key, value in rows.items())

    def blob(self, _repo: pathlib.Path, _commit: str, relative: str) -> str:
        if self.bad_blob:
            return "0" * 64
        return dict(PINNED_FILES)[relative]

    def args(self, *, preflight_only: bool = False) -> list[str]:
        argv = [
            "--binary",
            str(self.binary),
            "--repo-root",
            str(self.repo),
            "--expected-commit",
            COMMIT,
        ]
        if preflight_only:
            argv.append("--preflight-only")
        return argv

    def runner(self, args, cwd=None, env=None):
        del cwd
        if not isinstance(args, list):
            raise AssertionError("commands must be argument lists")
        command = args[0]
        if command == "git":
            if "rev-parse" in args:
                return ProcessResult(0, self.head + "\n", "")
            if "status" in args:
                return ProcessResult(0, self.porcelain, "")
            raise AssertionError(args)
        if command == "nm":
            return ProcessResult(0, self.nm_output, "")
        if command == "otool" and args[1] == "-L":
            return ProcessResult(0, self.link_output, "")
        if command == "otool" and args[1] == "-l":
            return ProcessResult(0, self.rpath_output, "")
        if command == "ldd":
            return ProcessResult(0, self.ldd_output, "")
        if command == "sysctl":
            if self.cpu_fail:
                return ProcessResult(1, "", "sysctl unavailable")
            values = {
                "hw.model": "Mac17,8",
                "hw.physicalcpu": "8",
                "hw.logicalcpu": "10",
                "hw.ncpu": "8",
            }
            return ProcessResult(0, values[args[-1]] + "\n", "")
        if command == "sw_vers":
            return ProcessResult(0, "15.0\n", "")
        if command == "nproc":
            if self.cpu_fail:
                return ProcessResult(1, "", "nproc unavailable")
            return ProcessResult(0, "8\n", "")
        if command == "cmake":
            return ProcessResult(0, "cmake version 3.24.0\n", "")
        if command == self.compiler:
            return ProcessResult(0, "c++ (synthetic) 1.0\n", "")
        if pathlib.Path(command).resolve() == self.binary.resolve() and args[1:] == ["--version"]:
            return ProcessResult(0, self.version_stdout, "")
        if pathlib.Path(command).resolve() == self.binary.resolve() and "--workload" in args:
            if env is None or env.get("ORT_DISABLE_TELEMETRY") != "1":
                raise AssertionError("ORT_DISABLE_TELEMETRY was not set")
            self.trial_calls.append(list(args))
            index = len(self.trial_calls) - 1
            if index == 0 or index == len(EXPECTED_ORDER) - 1:
                run_sets = list((self.repo / "benchmarks" / "concurrent_runs").iterdir())
                manifest = json.loads((run_sets[0] / "environment.json").read_text(encoding="utf-8"))
                if manifest["status"] != "running":
                    raise AssertionError(manifest["status"])
            if self.raise_on_trial == index:
                raise RuntimeError("boom")
            if self.fail_trial == index:
                return ProcessResult(self.fail_returncode, self.fail_stdout, self.fail_stderr)
            workload = args[args.index("--workload") + 1]
            workers = int(args[args.index("--concurrency") + 1])
            model = args[args.index("--model") + 1]
            return ProcessResult(0, dump_report(sample_report(workload, workers, model)), "")
        raise AssertionError(args)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def run_preflight(self) -> dict:
        return preflight(
            repo_root=self.repo,
            binary=self.binary,
            expected_commit=COMMIT,
            runner=self.runner,
            blob_sha256=self.blob,
        )


class ReportValidationTest(unittest.TestCase):
    def test_valid_add_and_mnist_reports(self) -> None:
        self.assertEqual(expect(dump_report(sample_report())), [])
        mnist = sample_report("mnist", 4, "/models/mnist_mlp.onnx")
        self.assertEqual(expect(dump_report(mnist), "mnist", 4, "/models/mnist_mlp.onnx"), [])

    def test_schema_version_1_is_rejected(self) -> None:
        document = sample_report()
        document["schema_version"] = 1
        document["measurement_loop_microseconds"] = 10.0
        document["throughput_requests_per_second"] = {"active_call": 1.0, "observed_sequential": 1.0}
        problems = expect(dump_report(document))
        self.assertTrue(any("schema_version is 1, expected 2" in item for item in problems))
        self.assertTrue(any("measurement_loop_microseconds" in item for item in problems))
        self.assertTrue(any("throughput_requests_per_second" in item for item in problems))

    def test_sequential_validator_rejects_schema_2(self) -> None:
        problems = validate_report(
            dump_report(sample_report()),
            workload="add",
            model_path="/models/add_f32.onnx",
            warmup=WARMUP,
            iterations=MEASURED,
            check_interval=INTERVAL,
        )
        self.assertTrue(problems)

    def test_wrong_workload_and_session_policy(self) -> None:
        document = sample_report()
        document["workload"] = "bert"
        document["configuration"]["session_ownership"] = "shared_session"
        problems = expect(dump_report(document))
        self.assertTrue(any("workload is 'bert'" in item for item in problems))
        self.assertTrue(any("session_ownership" in item for item in problems))

    def test_wrong_worker_count(self) -> None:
        document = sample_report(configuration={"worker_count": 2})
        problems = expect(dump_report(document))
        self.assertTrue(any("worker_count is 2, expected 1" in item for item in problems))

    def test_wrong_request_count(self) -> None:
        document = sample_report()
        document["total_completed_requests"] = 1001
        problems = expect(dump_report(document))
        self.assertTrue(any("total_completed_requests is 1001, expected 1000" in item for item in problems))

    def test_wrong_per_worker_counts(self) -> None:
        document = sample_report(worker_count=2)
        document["per_worker_completed_requests"] = [MEASURED, 500]
        problems = expect(dump_report(document), worker_count=2)
        self.assertTrue(any("per_worker_completed_requests[1] is 500" in item for item in problems))

    def test_wrong_per_worker_length(self) -> None:
        document = sample_report(worker_count=2)
        document["per_worker_completed_requests"] = [MEASURED]
        problems = expect(dump_report(document), worker_count=2)
        self.assertTrue(any("has 1 entries, expected 2" in item for item in problems))

    def test_wrong_sample_count(self) -> None:
        document = sample_report(latency_microseconds={"sample_count": 999})
        problems = expect(dump_report(document))
        self.assertTrue(any("sample_count is 999, expected 1000" in item for item in problems))

    def test_non_finite_metrics(self) -> None:
        text = dump_report(sample_report()).replace('"mean": 2.0', '"mean": NaN')
        problems = expect(text)
        self.assertTrue(problems)
        self.assertTrue("NaN" in problems[0] or "not valid JSON" in problems[0])
        infinity = dump_report(sample_report()).replace('"maximum": 5.0', '"maximum": Infinity')
        self.assertTrue(expect(infinity))

    def test_negative_zero_standard_deviation(self) -> None:
        document = sample_report(latency_microseconds={"population_standard_deviation": -0.0})
        problems = expect(dump_report(document))
        self.assertTrue(any("standard deviation is negative" in item for item in problems))

    def test_boolean_is_not_a_count(self) -> None:
        document = sample_report(configuration={"worker_count": True})
        problems = expect(dump_report(document))
        self.assertTrue(any("worker_count is not a JSON integer" in item for item in problems))

    def test_negative_wall_and_aggregate_mismatch(self) -> None:
        negative = sample_report()
        negative["measurement_wall_microseconds"] = -5.0
        self.assertTrue(any("wall duration" in item for item in expect(dump_report(negative))))
        mismatched = sample_report()
        mismatched["aggregate_completed_requests_per_second"] = 1.0
        self.assertTrue(any("does not match the wall duration" in item for item in expect(dump_report(mismatched))))

    def test_percentile_order_and_small_tolerance(self) -> None:
        inverted = sample_report(latency_microseconds={"p95": 4.0, "p99": 3.0})
        self.assertTrue(any("out of order" in item for item in expect(dump_report(inverted))))
        close = sample_report(latency_microseconds={"minimum": 2.0, "p50": 2.0 - 1e-12})
        self.assertEqual(expect(dump_report(close)), [])

    def test_stdout_framing(self) -> None:
        body = dump_report(sample_report())
        self.assertIn("stdout is empty", expect("")[0])
        self.assertIn("trailing newline", expect(body[:-1])[0])
        self.assertIn("more than one trailing newline", expect(body + "\n")[0])
        self.assertIn("leading text", expect("banner\n" + body)[0])
        self.assertIn("extra data", expect(body + body)[0])


class ProtocolTest(unittest.TestCase):
    def test_exact_48_trial_order(self) -> None:
        order = trial_order()
        self.assertEqual(order, EXPECTED_ORDER)
        self.assertEqual(len(order), 48)
        self.assertEqual(len(set(order)), 48)
        keys = [configuration_key(trial_id) for trial_id in order]
        self.assertTrue(all(left != right for left, right in zip(keys, keys[1:])))
        counted = [trial for trial in execution_plan() if not trial["pilot"]]
        self.assertEqual(len(counted), 40)
        self.assertEqual(sum(1 for trial in execution_plan() if trial["pilot"]), 8)
        for workload in ("add", "mnist"):
            for workers in (1, 2, 4, 8):
                matches = [
                    trial
                    for trial in counted
                    if trial["workload"] == workload and trial["worker_count"] == workers
                ]
                self.assertEqual(len(matches), 5)

    def test_explicit_concurrency_and_no_sequential_binary(self) -> None:
        binary = pathlib.Path("/opt") / BINARY_NAME
        add = concurrent_command(
            binary,
            workload="add",
            worker_count=4,
            add_model=pathlib.Path("/models/add_f32.onnx"),
            mnist_model=pathlib.Path("/models/mnist_mlp.onnx"),
            fixture_dir=pathlib.Path("/fixtures/mnist"),
        )
        mnist = concurrent_command(
            binary,
            workload="mnist",
            worker_count=8,
            add_model=pathlib.Path("/models/add_f32.onnx"),
            mnist_model=pathlib.Path("/models/mnist_mlp.onnx"),
            fixture_dir=pathlib.Path("/fixtures/mnist"),
        )
        self.assertEqual(add[add.index("--concurrency") + 1], "4")
        self.assertEqual(mnist[mnist.index("--concurrency") + 1], "8")
        self.assertNotIn("--fixture", add)
        self.assertEqual(mnist[mnist.index("--fixture") + 1], "test_00003")
        self.assertIsInstance(add, list)
        with self.assertRaises(CollectorError) as caught:
            concurrent_command(
                pathlib.Path("/opt/inferedge-bench"),
                workload="add",
                worker_count=1,
                add_model=pathlib.Path("/models/add_f32.onnx"),
                mnist_model=pathlib.Path("/models/mnist_mlp.onnx"),
                fixture_dir=pathlib.Path("/fixtures/mnist"),
            )
        self.assertIn("refusing to launch inferedge-bench", str(caught.exception))

    def test_runner_and_git_blob_do_not_use_a_shell(self) -> None:
        with mock.patch("benchmarks.collect_baseline.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            subprocess_runner([str(pathlib.Path("/opt") / BINARY_NAME), "--version"], cwd=None, env={})
        self.assertIsInstance(run.call_args.args[0], list)
        self.assertFalse(run.call_args.kwargs.get("shell", False))
        with mock.patch("benchmarks.collect_concurrent.subprocess.run") as git_run:
            git_run.return_value = mock.Mock(returncode=0, stdout=b"abc", stderr=b"")
            digest = git_blob_sha256(pathlib.Path("/repo"), COMMIT, "models/add_f32.onnx")
        self.assertEqual(digest, hashlib.sha256(b"abc").hexdigest())
        self.assertEqual(git_run.call_args.args[0][0], "git")
        self.assertFalse(git_run.call_args.kwargs.get("shell", False))
        self.assertNotIn("shell=True", str(git_run.call_args))

    def test_expected_commit_is_required_and_must_be_a_full_sha(self) -> None:
        with self.assertRaises(SystemExit):
            with redirect_stderr(StringIO()):
                build_parser().parse_args(["--binary", "inferedge-bench-concurrent"])
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            stderr = StringIO()
            with redirect_stderr(stderr):
                code = main(
                    ["--binary", "missing", "--repo-root", str(root), "--expected-commit", "abc"],
                )
            self.assertEqual(code, 1)
            self.assertIn("40-character", stderr.getvalue())
            self.assertFalse((root / "benchmarks" / "concurrent_runs").exists())

    def test_sequential_allow_list_is_unchanged(self) -> None:
        self.assertEqual(HISTORICAL_UNTRACKED, "models/experiments")
        self.assertEqual(sequential_trial_ids(5)[0], "pilot-add")
        problems = sequential_git_status_problems(
            "?? benchmarks/concurrent_runs/example\n",
            expected_commit=COMMIT,
            actual_commit=COMMIT,
        )
        self.assertTrue(any("unexpected untracked path" in item for item in problems))

    def test_pinned_hashes_match_the_repository_blobs(self) -> None:
        self.assertEqual(
            [path for path, _digest in PINNED_FILES],
            [
                "models/add_f32.onnx",
                "models/mnist_mlp.onnx",
                "models/fixtures/mnist/inputs/test_00003.f32",
                "models/fixtures/mnist/onnx/test_00003.f32",
                "models/fixtures/mnist/manifest.json",
            ],
        )
        self.assertEqual(ADD_MODEL_SHA256, "37eb1b1b1adee439d2d14b91a0b217bd15ab809610d1271ea821d56e6400a447")
        self.assertEqual(MNIST_MODEL_SHA256, "3473a4b2673fe4353c2b3a04a7d432717a4dfcdc42fc217565c99743542d8cce")
        self.assertEqual(MNIST_INPUT_SHA256, "692f8ff1c2aaa7dd3c5982979516e774e4daf6d8420f1818a68389825ebb84a9")
        self.assertEqual(MNIST_REFERENCE_SHA256, "9a5daa05f5096f5521deab05c025d3c3691e2587189f176d8699a9521b663a46")
        self.assertEqual(
            MNIST_FIXTURE_MANIFEST_SHA256,
            "4e6f95f8626fa89a22f59aed004870b942fcf2d7bb4add2226f9ca4f9f13009a",
        )
        head = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(head.returncode, 0)
        verify_pinned_files(REPO_ROOT, head.stdout.strip(), git_blob_sha256)

    def test_git_blob_hash_follows_the_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = pathlib.Path(temporary)
            target = repo / "models" / "add_f32.onnx"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"committed-bytes")
            template = repo / "empty-template"
            template.mkdir()
            self._git(repo, ["init", "-b", "main", f"--template={template}"])
            self._git(repo, ["add", "models/add_f32.onnx"])
            self._git(
                repo,
                [
                    "-c",
                    "user.email=collector-test@example.com",
                    "-c",
                    "user.name=collector-test",
                    "commit",
                    "-m",
                    "pin",
                ],
            )
            head = self._git(repo, ["rev-parse", "HEAD"]).stdout.strip()
            committed = git_blob_sha256(repo, head, "models/add_f32.onnx")
            self.assertEqual(committed, hashlib.sha256(b"committed-bytes").hexdigest())
            target.write_bytes(b"worktree-bytes")
            self.assertEqual(git_blob_sha256(repo, head, "models/add_f32.onnx"), committed)

    def _git(self, repo: pathlib.Path, args: list[str]) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed


class PreflightTest(unittest.TestCase):
    def test_valid_release_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Stage(pathlib.Path(temporary))
            stage.build()
            runtime_root = stage.runtime.resolve()
            checked = stage.run_preflight()
        self.assertEqual(checked["verified_configuration"]["cmake_build_type"], "Release")
        self.assertEqual(checked["verified_configuration"]["cmake_cxx_flags_release"], "-O3 -DNDEBUG")
        self.assertEqual(checked["verified_configuration"]["sanitizer_instrumentation"], "absent")
        self.assertEqual(checked["verified_configuration"]["onnxruntime_version"], "1.31.0")
        self.assertEqual(checked["verified_configuration"]["add_model_sha256"], ADD_MODEL_SHA256)
        self.assertEqual(checked["verified_configuration"]["mnist_model_sha256"], MNIST_MODEL_SHA256)
        self.assertEqual(checked["verified_configuration"]["mnist_fixture_inputs_sha256"], MNIST_INPUT_SHA256)
        self.assertEqual(
            checked["verified_configuration"]["mnist_fixture_reference_sha256"],
            MNIST_REFERENCE_SHA256,
        )
        self.assertEqual(
            checked["verified_configuration"]["mnist_fixture_manifest_sha256"],
            MNIST_FIXTURE_MANIFEST_SHA256,
        )
        self.assertEqual(checked["queried"]["benchmark_tool_version"], "0.1.0")
        self.assertEqual(len(checked["queried"]["benchmark_binary_sha256"]), 64)
        resolved = pathlib.Path(checked["queried"]["onnxruntime_resolved_library"])
        self.assertIn(resolved.name, PINNED_ORT_LIBRARIES)
        self.assertEqual(resolved, (runtime_root / "lib" / resolved.name).resolve())
        self.assertEqual(checked["source_defined_policy"]["allowed_untracked_roots"], list(ALLOWED_UNTRACKED))
        self.assertEqual(checked["collection_protocol"]["trial_order"], EXPECTED_ORDER)
        self.assertEqual(checked["collection_protocol"]["inter_trial_delay_seconds"], 5.0)
        self.assertEqual(checked["queried"]["logical_cpu_count_source"], "os.cpu_count")
        if platform.system() == "Darwin":
            self.assertEqual(checked["queried"]["hardware_model"], "Mac17,8")
            self.assertEqual(checked["queried"]["hw_physicalcpu"], 8)
            self.assertEqual(checked["queried"]["hw_logicalcpu"], 10)
            self.assertEqual(checked["queried"]["hw_ncpu"], 8)
            self.assertNotEqual(checked["queried"]["hw_logicalcpu"], 17)
        else:
            self.assertIsNone(checked["queried"]["hardware_model"])
            self.assertIn("Darwin", str(checked["queried"]["hardware_model_error"]))
            self.assertEqual(checked["queried"]["nproc_cpu_count"], 8)

    def test_versioned_libraries_resolve_only_inside_the_pinned_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            macos = self._pinned_package(base / "macos", "libonnxruntime.1.31.0.dylib")
            macos_library = macos / "lib" / "libonnxruntime.1.31.0.dylib"
            resolved = verify_linked_onnx_runtime(
                binary=pathlib.Path("/tmp") / BINARY_NAME,
                runtime_root=macos,
                link_output=(
                    "inferedge-bench-concurrent:\n"
                    "\t@rpath/libonnxruntime.1.31.0.dylib "
                    "(compatibility version 1.0.0, current version 1.31.0)\n"
                ),
                load_command_output=self._rpath(macos / "lib"),
            )
            self.assertEqual(resolved.name, "libonnxruntime.1.31.0.dylib")
            self.assertEqual(resolved, macos_library.resolve())

            linux = self._pinned_package(base / "linux", "libonnxruntime.so.1.31.0")
            linux_library = linux / "lib" / "libonnxruntime.so.1.31.0"
            resolved = verify_linked_onnx_runtime(
                binary=pathlib.Path("/tmp") / BINARY_NAME,
                runtime_root=linux,
                link_output=f"\tlibonnxruntime.so.1.31.0 => {linux_library} (0x1000)\n",
            )
            self.assertEqual(resolved.name, "libonnxruntime.so.1.31.0")
            self.assertEqual(resolved, linux_library.resolve())

            with self.assertRaises(CollectorError) as caught:
                verify_linked_onnx_runtime(
                    binary=pathlib.Path("/tmp") / BINARY_NAME,
                    runtime_root=macos,
                    link_output=(
                        "inferedge-bench-concurrent:\n"
                        "\t@rpath/libonnxruntime.dylib "
                        "(compatibility version 1.0.0, current version 1.31.0)\n"
                    ),
                    load_command_output=self._rpath(macos / "lib"),
                )
            self.assertIn("unrecognized ONNX Runtime dependency", str(caught.exception))

            foreign = base / "other" / "libonnxruntime.1.31.0.dylib"
            foreign.parent.mkdir()
            foreign.write_bytes(b"foreign")
            with self.assertRaises(CollectorError) as caught:
                verify_linked_onnx_runtime(
                    binary=pathlib.Path("/tmp") / BINARY_NAME,
                    runtime_root=macos,
                    link_output=(
                        "inferedge-bench-concurrent:\n"
                        f"\t{foreign} (compatibility version 1.0.0, current version 1.31.0)\n"
                    ),
                )
            self.assertIn("outside the pinned installation", str(caught.exception))

    def _pinned_package(self, root: pathlib.Path, filename: str) -> pathlib.Path:
        library = root / "lib"
        library.mkdir(parents=True)
        (root / "VERSION_NUMBER").write_text("1.31.0\n", encoding="utf-8")
        (library / filename).write_bytes(b"ort")
        return root

    def _rpath(self, path: pathlib.Path) -> str:
        return "          cmd LC_RPATH\n" f"         path {path} (offset 12)\n"

    def test_wrong_binary_missing_cache_and_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            missing = root / "missing" / BINARY_NAME
            with self.assertRaises(CollectorError) as caught:
                preflight(
                    repo_root=root,
                    binary=missing,
                    expected_commit=COMMIT,
                    runner=lambda *args, **kwargs: ProcessResult(0),
                    blob_sha256=lambda *_args: "",
                )
            self.assertIn("does not exist", str(caught.exception))

            stage = Stage(root / "stage")
            stage.executable = False
            stage.build()
            with self.assertRaises(CollectorError) as caught:
                stage.run_preflight()
            self.assertIn("not executable", str(caught.exception))

            copied = Stage(root / "copy")
            copied.write_cache = False
            copied.build()
            with self.assertRaises(CollectorError) as caught:
                copied.run_preflight()
            self.assertIn("CMakeCache.txt", str(caught.exception))

            named = Stage(root / "named")
            named.binary = named.build_dir / "inferedge-bench"
            named.build()
            with self.assertRaises(CollectorError) as caught:
                named.run_preflight()
            self.assertIn("must be named inferedge-bench-concurrent", str(caught.exception))

    def test_wrong_binary_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Stage(pathlib.Path(temporary))
            stage.version_stdout = "inferedge-bench 0.1.0\n"
            stage.build()
            with self.assertRaises(CollectorError) as caught:
                stage.run_preflight()
            self.assertIn("inferedge-bench 0.1.0", str(caught.exception))

    def test_wrong_git_commit_and_dirty_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            wrong = Stage(root / "commit")
            wrong.head = OTHER_COMMIT
            wrong.build()
            with self.assertRaises(CollectorError) as caught:
                wrong.run_preflight()
            self.assertIn(f"HEAD is {OTHER_COMMIT}", str(caught.exception))

            dirty = Stage(root / "dirty")
            dirty.porcelain = " M src/concurrent_benchmark_command.cpp\n"
            dirty.build()
            with self.assertRaises(CollectorError) as caught:
                dirty.run_preflight()
            self.assertIn("tracked worktree change", str(caught.exception))

    def test_untracked_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            unexpected = Stage(root / "unexpected")
            unexpected.porcelain = "?? notes.txt\n"
            unexpected.build()
            with self.assertRaises(CollectorError) as caught:
                unexpected.run_preflight()
            self.assertIn("unexpected untracked path: notes.txt", str(caught.exception))

            approved = Stage(root / "approved")
            approved.porcelain = (
                "?? models/experiments/local.pt\n"
                "?? benchmarks/baselines/old/\n"
                "?? benchmarks/concurrent_runs/previous/\n"
            )
            approved.build()
            checked = approved.run_preflight()
            self.assertEqual(len(checked["queried"]["approved_untracked_paths"]), 3)

    def test_wrong_onnx_runtime_version_and_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            version = Stage(root / "version")
            version.ort_version = "1.22.0"
            version.build()
            with self.assertRaises(CollectorError) as caught:
                version.run_preflight()
            self.assertIn("1.22.0", str(caught.exception))

            foreign = Stage(root / "foreign")
            foreign.foreign = True
            foreign.build()
            with self.assertRaises(CollectorError) as caught:
                foreign.run_preflight()
            self.assertIn("outside the pinned installation", str(caught.exception))

    def test_wrong_model_and_fixture_hashes(self) -> None:
        cases = {
            "models/add_f32.onnx": "models/add_f32.onnx",
            "models/mnist_mlp.onnx": "models/mnist_mlp.onnx",
            "models/fixtures/mnist/inputs/test_00003.f32": "models/fixtures/mnist/inputs/test_00003.f32",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for name, relative in cases.items():
                stage = Stage(root / name.replace("/", "_"))
                stage.overrides[relative] = b"wrong-bytes"
                stage.build()
                with self.assertRaises(CollectorError) as caught:
                    stage.run_preflight()
                self.assertIn(f"SHA-256 mismatch for {relative}", str(caught.exception))

            committed = Stage(root / "committed")
            committed.bad_blob = True
            committed.build()
            with self.assertRaises(CollectorError) as caught:
                committed.run_preflight()
            self.assertIn("committed SHA-256 mismatch", str(caught.exception))

    def test_sanitizer_and_unoptimized_release_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            sanitized = Stage(root / "sanitized")
            sanitized.cxx_flags = "-fsanitize=address,undefined"
            sanitized.build()
            with self.assertRaises(CollectorError) as caught:
                sanitized.run_preflight()
            self.assertIn("sanitizer", str(caught.exception))

            symbols = Stage(root / "symbols")
            symbols.nm_output = "000000 T ___asan_report\n"
            symbols.build()
            with self.assertRaises(CollectorError) as caught:
                symbols.run_preflight()
            self.assertIn("sanitizer symbol", str(caught.exception))

            slow = Stage(root / "slow")
            slow.release_flags = "-O0"
            slow.build()
            with self.assertRaises(CollectorError) as caught:
                slow.run_preflight()
            self.assertIn("disables optimization", str(caught.exception))
        self.assertTrue(release_optimization_problems({"CMAKE_CXX_FLAGS_RELEASE": ""}))

    def test_failed_cpu_query_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Stage(pathlib.Path(temporary))
            stage.cpu_fail = True
            stage.build()
            checked = stage.run_preflight()
        if platform.system() == "Darwin":
            self.assertIsNone(checked["queried"]["hw_logicalcpu"])
            self.assertIn("failed", str(checked["queried"]["hw_logicalcpu_error"]))
            self.assertIsNone(checked["queried"]["hardware_model"])
        else:
            self.assertIsNone(checked["queried"]["nproc_cpu_count"])
            self.assertIn("failed", str(checked["queried"]["nproc_cpu_count_error"]))

    def test_preflight_only_creates_no_run_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Stage(pathlib.Path(temporary))
            stage.build()
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(stage.args(preflight_only=True), runner=stage.runner, blob_sha256=stage.blob)
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertIn("preflight passed", stdout.getvalue())
            self.assertFalse((stage.repo / "benchmarks" / "concurrent_runs").exists())
            self.assertEqual(stage.trial_calls, [])

            stage.version_stdout = "inferedge-bench 0.1.0\n"
            failed = StringIO()
            with redirect_stderr(failed):
                failed_code = main(
                    stage.args(preflight_only=True),
                    runner=stage.runner,
                    blob_sha256=stage.blob,
                )
            self.assertEqual(failed_code, 1)
            self.assertIn("error:", failed.getvalue())
            self.assertFalse((stage.repo / "benchmarks" / "concurrent_runs").exists())
            self.assertEqual(stage.trial_calls, [])


class CollectionTest(unittest.TestCase):
    def test_completed_campaign_preserves_order_and_writes_no_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Stage(pathlib.Path(temporary))
            stage.build()
            run_set = collect(
                build_parser().parse_args(stage.args()),
                runner=stage.runner,
                sleep=stage.sleep,
                blob_sha256=stage.blob,
            )
            manifest = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertIsNone(manifest["aggregate"])
            self.assertEqual(manifest["collection_protocol"]["trial_order"], EXPECTED_ORDER)
            self.assertFalse((run_set / "aggregate.json").exists())
            self.assertFalse((run_set / "failure.json").exists())
            accepted = {
                path.name
                for path in (run_set / "runs").glob("*.json")
                if path.name.count(".") == 1
            }
            self.assertEqual(accepted, {f"{trial_id}.json" for trial_id in EXPECTED_ORDER})
            observed = [
                (
                    args[args.index("--workload") + 1],
                    int(args[args.index("--concurrency") + 1]),
                )
                for args in stage.trial_calls
            ]
            expected_pairs = [
                (workload, int(tag[1:]))
                for trial_id in EXPECTED_ORDER
                for workload, tag in [configuration_key(trial_id)]
            ]
            self.assertEqual(observed, expected_pairs)
            self.assertEqual(len(stage.sleeps), 47)
            self.assertEqual(set(stage.sleeps), {5.0})
            self.assertEqual(len(stage.trial_calls), 48)
            self.assertEqual(
                pathlib.Path(stage.trial_calls[0][0]).resolve(),
                stage.binary.resolve(),
            )
            self.assertIn("--concurrency", stage.trial_calls[0])
            self.assertNotIn("sh", stage.trial_calls[0])
            self.assertNotIn("bash", stage.trial_calls[0])
            pilot = json.loads((run_set / "runs" / "pilot-add-c01.meta.json").read_text(encoding="utf-8"))
            self.assertTrue(pilot["accepted"])
            self.assertTrue(pilot["pilot"])
            self.assertEqual(pilot["worker_count"], 1)
            self.assertTrue((run_set / "runs" / "pilot-mnist-c08.json").is_file())

    def test_nonzero_exit_nonempty_stderr_and_exception_preserve_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            exited = Stage(root / "exit")
            exited.fail_trial = 1
            exited.fail_returncode = 5
            exited.fail_stderr = ""
            exited.build()
            with self.assertRaises(CollectorError):
                collect(
                    build_parser().parse_args(exited.args()),
                    runner=exited.runner,
                    sleep=exited.sleep,
                    blob_sha256=exited.blob,
                )
            run_set = next((exited.repo / "benchmarks" / "concurrent_runs").iterdir())
            manifest = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse((run_set / "aggregate.json").exists())
            self.assertTrue((run_set / "runs" / "pilot-add-c01.json").is_file())
            self.assertFalse((run_set / "runs" / "pilot-mnist-c01.json").exists())
            self.assertTrue((run_set / "runs" / "pilot-mnist-c01.stdout").is_file())
            self.assertTrue((run_set / "runs" / "pilot-mnist-c01.stderr").is_file())
            self.assertFalse((run_set / "runs" / "pilot-add-c02.stdout").exists())
            failure = json.loads((run_set / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["trial_id"], "pilot-mnist-c01")
            self.assertIn("exit status 5", failure["reason"])
            self.assertIsNone(failure["aggregate"])
            self.assertTrue(failure["preserved_earlier_trials"])
            self.assertEqual(len(exited.trial_calls), 2)

            noisy = Stage(root / "stderr")
            noisy.fail_trial = 0
            noisy.fail_returncode = 0
            noisy.fail_stderr = "error: model file is missing\n"
            noisy.fail_stdout = dump_report(sample_report())
            noisy.build()
            with self.assertRaises(CollectorError):
                collect(
                    build_parser().parse_args(noisy.args()),
                    runner=noisy.runner,
                    sleep=noisy.sleep,
                    blob_sha256=noisy.blob,
                )
            noisy_set = next((noisy.repo / "benchmarks" / "concurrent_runs").iterdir())
            stderr = (noisy_set / "runs" / "pilot-add-c01.stderr").read_text(encoding="utf-8")
            self.assertEqual(stderr, "error: model file is missing\n")
            self.assertFalse((noisy_set / "runs" / "pilot-add-c01.json").exists())
            self.assertFalse((noisy_set / "aggregate.json").exists())
            noisy_manifest = json.loads((noisy_set / "environment.json").read_text(encoding="utf-8"))
            self.assertEqual(noisy_manifest["status"], "failed")

            exploded = Stage(root / "explode")
            exploded.raise_on_trial = 1
            exploded.build()
            with self.assertRaises(CollectorError):
                collect(
                    build_parser().parse_args(exploded.args()),
                    runner=exploded.runner,
                    sleep=exploded.sleep,
                    blob_sha256=exploded.blob,
                )
            exploded_set = next((exploded.repo / "benchmarks" / "concurrent_runs").iterdir())
            exploded_stderr = (exploded_set / "runs" / "pilot-mnist-c01.stderr").read_text(encoding="utf-8")
            self.assertIn("boom", exploded_stderr)
            self.assertTrue((exploded_set / "runs" / "pilot-add-c01.json").is_file())
            self.assertEqual(len(exploded.trial_calls), 2)

    def test_existing_run_set_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = pathlib.Path(temporary)
            first = create_run_set(parent, "20261012T030405Z-aaaaaaa")
            (first / "runs" / "kept.txt").write_text("keep\n", encoding="utf-8")
            with self.assertRaises(CollectorError) as caught:
                create_run_set(parent, "20261012T030405Z-aaaaaaa")
            self.assertIn("already exists", str(caught.exception))
            self.assertEqual((first / "runs" / "kept.txt").read_text(encoding="utf-8"), "keep\n")

            stage = Stage(parent / "campaign")
            stage.build()
            fixed = datetime.datetime(2026, 10, 12, 3, 4, 5, tzinfo=datetime.timezone.utc)

            class Frozen(datetime.datetime):
                @classmethod
                def now(cls, tz=None):
                    return fixed

            with mock.patch("benchmarks.collect_concurrent.datetime.datetime", Frozen):
                run_set = collect(
                    build_parser().parse_args(stage.args()),
                    runner=stage.runner,
                    sleep=stage.sleep,
                    blob_sha256=stage.blob,
                )
                with self.assertRaises(CollectorError) as caught:
                    collect(
                        build_parser().parse_args(stage.args()),
                        runner=stage.runner,
                        sleep=stage.sleep,
                        blob_sha256=stage.blob,
                    )
            self.assertIn("already exists", str(caught.exception))
            manifest = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(stage.trial_calls), 48)
            self.assertFalse((stage.repo / "benchmarks" / "baselines").exists())

    def test_report_write_failure_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Stage(pathlib.Path(temporary))
            stage.build()
            original = pathlib.Path.write_text

            def fail_accepted_report(self, data, *args, **kwargs):
                if self.name == "pilot-mnist-c01.json":
                    raise OSError("disk full")
                return original(self, data, *args, **kwargs)

            with mock.patch.object(pathlib.Path, "write_text", fail_accepted_report):
                with self.assertRaises(CollectorError):
                    collect(
                        build_parser().parse_args(stage.args()),
                        runner=stage.runner,
                        sleep=stage.sleep,
                        blob_sha256=stage.blob,
                    )
            run_set = next((stage.repo / "benchmarks" / "concurrent_runs").iterdir())
            manifest = json.loads((run_set / "environment.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue((run_set / "runs" / "pilot-add-c01.json").is_file())
            self.assertFalse((run_set / "runs" / "pilot-mnist-c01.json").exists())
            self.assertTrue((run_set / "runs" / "pilot-mnist-c01.stdout").is_file())
            meta = json.loads(
                (run_set / "runs" / "pilot-mnist-c01.meta.json").read_text(encoding="utf-8")
            )
            self.assertFalse(meta["accepted"])
            self.assertIn("could not preserve accepted report", " ".join(meta["problems"]))
            failure = json.loads((run_set / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["trial_id"], "pilot-mnist-c01")
            self.assertFalse((run_set / "aggregate.json").exists())
            self.assertFalse((run_set / "runs" / "pilot-add-c02.stdout").exists())
            self.assertEqual(len(stage.trial_calls), 2)

    def test_incomplete_run_directory_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            preserved = root / "preserved-run"
            preserved.mkdir()
            (preserved / "kept.txt").write_text("keep\n", encoding="utf-8")
            stage = Stage(root / "campaign")
            stage.build()
            original = pathlib.Path.mkdir

            def fail_runs(self, mode=0o777, parents=False, exist_ok=False):
                if self.name == "runs":
                    raise OSError("cannot create runs")
                return original(self, mode, parents=parents, exist_ok=exist_ok)

            stderr = StringIO()
            with mock.patch.object(pathlib.Path, "mkdir", fail_runs):
                with redirect_stderr(stderr):
                    code = main(stage.args(), runner=stage.runner, sleep=stage.sleep, blob_sha256=stage.blob)
            self.assertEqual(code, 1)
            self.assertIn("could not create the run-set directory", stderr.getvalue())
            self.assertEqual((preserved / "kept.txt").read_text(encoding="utf-8"), "keep\n")
            created = next((stage.repo / "benchmarks" / "concurrent_runs").iterdir())
            failure = json.loads((created / "failure.json").read_text(encoding="utf-8"))
            self.assertIn("cannot create runs", failure["reason"])
            self.assertFalse((created / "environment.json").exists())
            self.assertFalse((created / "aggregate.json").exists())
            self.assertEqual(stage.trial_calls, [])


if __name__ == "__main__":
    unittest.main()
