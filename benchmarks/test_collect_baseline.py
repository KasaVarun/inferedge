#!/usr/bin/env python3
"""Synthetic tests for the baseline collector. They do not run inferedge-bench."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.collect_baseline import (  # noqa: E402
    EXPECTED_COMMIT,
    CollectorError,
    ProcessResult,
    approved_untracked_paths,
    assess_build_cache,
    bench_command,
    build_parser,
    create_run_set,
    execute_trials,
    git_status_problems,
    subprocess_runner,
    trial_ids,
    validate_report,
    verify_linked_onnx_runtime,
)

ADD_MODEL = "/models/add_f32.onnx"
MNIST_MODEL = "/models/mnist_mlp.onnx"
WARMUP = 100
ITERATIONS = 1000
INTERVAL = 1


def sample_report(workload: str = "add", **overrides: object) -> dict:
    fixture = None if workload == "add" else "test_00003"
    model = ADD_MODEL if workload == "add" else MNIST_MODEL
    document = {
        "schema_version": 1,
        "tool": {"name": "inferedge-bench", "version": "0.1.0"},
        "model": model,
        "workload": workload,
        "fixture": fixture,
        "backend": "ONNX Runtime CPU",
        "configuration": {
            "warmup_iterations": WARMUP,
            "measured_iterations": ITERATIONS,
            "correctness_check_interval": INTERVAL,
            "calling_threads": 1,
            "onnx_intra_op_threads": 1,
            "graph_optimizations": "disabled",
        },
        "latency_microseconds": {
            "sample_count": ITERATIONS,
            "minimum": 1.0,
            "mean": 2.0,
            "p50": 2.0,
            "p95": 3.0,
            "p99": 4.0,
            "maximum": 5.0,
            "population_standard_deviation": 0.5,
        },
        "measurement_loop_microseconds": 10.0,
        "throughput_requests_per_second": {
            "active_call": 100.0,
            "observed_sequential": 80.0,
        },
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


def expect(text: str, workload: str = "add") -> list[str]:
    model = ADD_MODEL if workload == "add" else MNIST_MODEL
    return validate_report(
        text,
        workload=workload,
        model_path=model,
        warmup=WARMUP,
        iterations=ITERATIONS,
        check_interval=INTERVAL,
    )


class ReportValidationTest(unittest.TestCase):
    def test_valid_add_report(self) -> None:
        self.assertEqual(expect(dump_report(sample_report("add"))), [])

    def test_valid_mnist_report(self) -> None:
        self.assertEqual(expect(dump_report(sample_report("mnist")), "mnist"), [])

    def test_malformed_json(self) -> None:
        problems = expect("{not-json\n")
        self.assertTrue(problems)
        self.assertIn("not valid JSON", problems[0])

    def test_leading_banner_is_rejected(self) -> None:
        text = "InferEdge benchmark\n" + dump_report(sample_report())
        self.assertIn("leading text", expect(text)[0])

    def test_empty_stdout(self) -> None:
        self.assertIn("stdout is empty", expect(""))

    def test_multiple_json_objects(self) -> None:
        text = dump_report(sample_report()) + dump_report(sample_report())
        self.assertIn("extra data", expect(text)[0])

    def test_extra_text_after_json(self) -> None:
        text = dump_report(sample_report())[:-1] + " trailing\n"
        self.assertIn("extra data", expect(text)[0])

    def test_missing_trailing_newline(self) -> None:
        self.assertIn("trailing newline", expect(dump_report(sample_report())[:-1])[0])

    def test_wrong_schema_version(self) -> None:
        document = sample_report()
        document["schema_version"] = 2
        self.assertTrue(any("schema_version" in item for item in expect(dump_report(document))))

    def test_missing_fields(self) -> None:
        document = sample_report()
        del document["correctness"]
        self.assertTrue(any("missing" in item for item in expect(dump_report(document))))

    def test_wrong_workload(self) -> None:
        document = sample_report()
        document["workload"] = "bert"
        self.assertTrue(any("workload" in item for item in expect(dump_report(document))))

    def test_wrong_fixture(self) -> None:
        document = sample_report("mnist")
        document["fixture"] = "test_00000"
        problems = expect(dump_report(document), "mnist")
        self.assertTrue(any("fixture" in item for item in problems))

    def test_wrong_model_path(self) -> None:
        document = sample_report()
        document["model"] = "/tmp/other.onnx"
        self.assertTrue(any("model" in item for item in expect(dump_report(document))))

    def test_wrong_iteration_count(self) -> None:
        document = sample_report(configuration={"measured_iterations": 10})
        self.assertTrue(any("measured_iterations" in item for item in expect(dump_report(document))))

    def test_wrong_sample_count(self) -> None:
        document = sample_report(latency_microseconds={"sample_count": 999})
        self.assertTrue(any("sample_count" in item for item in expect(dump_report(document))))

    def test_non_finite_metrics(self) -> None:
        text = dump_report(sample_report()).replace('"mean": 2.0', '"mean": NaN')
        problems = expect(text)
        self.assertTrue(problems)
        self.assertTrue("NaN" in problems[0] or "not valid JSON" in problems[0])

    def test_boolean_numeric_field(self) -> None:
        document = sample_report(latency_microseconds={"mean": True})
        self.assertTrue(any("mean" in item and "number" in item for item in expect(dump_report(document))))

    def test_negative_standard_deviation(self) -> None:
        document = sample_report(latency_microseconds={"population_standard_deviation": -1e-12})
        problems = expect(dump_report(document))
        self.assertTrue(any("standard deviation" in item for item in problems))

    def test_negative_zero_standard_deviation(self) -> None:
        document = sample_report(latency_microseconds={"population_standard_deviation": -0.0})
        problems = expect(dump_report(document))
        self.assertTrue(any("standard deviation" in item for item in problems))

    def test_invalid_latency_ordering(self) -> None:
        document = sample_report(latency_microseconds={"p50": 5.0, "p95": 1.0})
        self.assertTrue(any("out of order" in item for item in expect(dump_report(document))))

    def test_zero_throughput(self) -> None:
        document = sample_report(throughput_requests_per_second={"active_call": 0})
        self.assertTrue(any("active_call" in item for item in expect(dump_report(document))))


class CollectionProtocolTest(unittest.TestCase):
    def test_nonzero_child_process_exit(self) -> None:
        self._run_failure(returncode=5, stderr="", stdout=dump_report(sample_report()))

    def test_nonempty_stderr(self) -> None:
        self._run_failure(returncode=0, stderr="error: model file is missing\n", stdout="")

    def test_duplicate_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = create_run_set(root, "20261012T000000Z-6c1ba83")
            self.assertTrue(first.is_dir())
            with self.assertRaises(Exception) as caught:
                create_run_set(root, "20261012T000000Z-6c1ba83")
            self.assertIn("already exists", str(caught.exception))

    def test_tracked_source_changes(self) -> None:
        problems = git_status_problems(
            " M src/benchmark_command.cpp\n",
            expected_commit=EXPECTED_COMMIT,
            actual_commit=EXPECTED_COMMIT,
        )
        self.assertTrue(any("tracked worktree change" in item for item in problems))

    def test_unexpected_untracked_files(self) -> None:
        problems = git_status_problems(
            "?? notes.txt\n",
            expected_commit=EXPECTED_COMMIT,
            actual_commit=EXPECTED_COMMIT,
        )
        self.assertTrue(any("unexpected untracked path: notes.txt" in item for item in problems))

    def test_approved_historical_experiments(self) -> None:
        porcelain = "?? models/experiments/\n?? models/experiments/mnist_mlp_5epochs.pt\n"
        self.assertEqual(
            approved_untracked_paths(porcelain),
            ["models/experiments", "models/experiments/mnist_mlp_5epochs.pt"],
        )
        self.assertEqual(
            git_status_problems(
                porcelain,
                expected_commit=EXPECTED_COMMIT,
                actual_commit=EXPECTED_COMMIT,
            ),
            [],
        )

    def test_binary_mismatch_or_wrong_build(self) -> None:
        debug = self._release_cache()
        debug["CMAKE_BUILD_TYPE"] = "Debug"
        self.assertTrue(any("Debug" in item for item in assess_build_cache(debug)))

        sanitized = self._release_cache()
        sanitized["CMAKE_CXX_FLAGS"] = "-fsanitize=address,undefined"
        self.assertTrue(any("sanitizer" in item for item in assess_build_cache(sanitized)))

        release = self._release_cache()
        release["CMAKE_CXX_FLAGS_RELEASE"] = ""
        self.assertEqual(assess_build_cache(release), [])
        self.assertNotIn("-O3", release["CMAKE_CXX_FLAGS_RELEASE"])

    def test_subprocess_argument_construction(self) -> None:
        binary = pathlib.Path("/opt/inferedge-bench")
        add = bench_command(
            binary,
            workload="add",
            add_model=pathlib.Path(ADD_MODEL),
            mnist_model=pathlib.Path(MNIST_MODEL),
            fixture_dir=pathlib.Path("/fixtures/mnist"),
            warmup=WARMUP,
            iterations=ITERATIONS,
            check_interval=INTERVAL,
        )
        mnist = bench_command(
            binary,
            workload="mnist",
            add_model=pathlib.Path(ADD_MODEL),
            mnist_model=pathlib.Path(MNIST_MODEL),
            fixture_dir=pathlib.Path("/fixtures/mnist"),
            warmup=WARMUP,
            iterations=ITERATIONS,
            check_interval=INTERVAL,
        )
        self.assertEqual(
            add,
            [
                "/opt/inferedge-bench",
                "--model",
                ADD_MODEL,
                "--workload",
                "add",
                "--warmup",
                "100",
                "--iterations",
                "1000",
                "--check-interval",
                "1",
                "--format",
                "json",
            ],
        )
        self.assertEqual(
            mnist,
            [
                "/opt/inferedge-bench",
                "--model",
                MNIST_MODEL,
                "--workload",
                "mnist",
                "--fixture",
                "test_00003",
                "--fixture-dir",
                "/fixtures/mnist",
                "--warmup",
                "100",
                "--iterations",
                "1000",
                "--check-interval",
                "1",
                "--format",
                "json",
            ],
        )
        self.assertIsInstance(add, list)
        self.assertNotIn("--fixture", add)

    def test_trial_ordering(self) -> None:
        args = build_parser().parse_args(["--binary", "inferedge-bench"])
        self.assertEqual(args.warmup, 100)
        self.assertEqual(args.iterations, 1000)
        self.assertEqual(args.check_interval, 1)
        self.assertEqual(args.runs, 5)
        self.assertEqual(args.delay_seconds, 5.0)
        self.assertEqual(args.expected_commit, EXPECTED_COMMIT)
        self.assertEqual(
            trial_ids(5),
            [
                "pilot-add",
                "pilot-mnist",
                "add-01",
                "mnist-01",
                "add-02",
                "mnist-02",
                "add-03",
                "mnist-03",
                "add-04",
                "mnist-04",
                "add-05",
                "mnist-05",
            ],
        )

    def test_no_aggregate_on_incomplete_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_set = create_run_set(pathlib.Path(temporary), "run-set")
            calls = []

            def runner(args, cwd=None, env=None):
                calls.append((list(args), env))
                if args[args.index("--workload") + 1] == "add":
                    return ProcessResult(0, dump_report(sample_report("add")), "")
                return ProcessResult(8, "", "error: benchmark report failed\n")

            accepted = execute_trials(
                run_set=run_set,
                binary=pathlib.Path("/opt/inferedge-bench"),
                repo_root=pathlib.Path("/repo"),
                preflight_result=self._paths(),
                counted_runs=5,
                warmup=WARMUP,
                iterations=ITERATIONS,
                check_interval=INTERVAL,
                delay_seconds=0,
                runner=runner,
                sleep=lambda _seconds: None,
            )
            self.assertFalse(accepted)
            self.assertFalse((run_set / "aggregate.json").exists())
            self.assertTrue((run_set / "failure.json").is_file())
            self.assertTrue((run_set / "runs" / "pilot-add.json").is_file())
            self.assertFalse((run_set / "runs" / "pilot-mnist.json").exists())
            self.assertTrue((run_set / "runs" / "pilot-mnist.stdout").is_file())
            self.assertTrue((run_set / "runs" / "pilot-mnist.stderr").is_file())
            failure = json.loads((run_set / "failure.json").read_text(encoding="utf-8"))
            self.assertFalse(failure["accepted"])
            self.assertIsNone(failure["aggregate"])
            self.assertEqual(calls[0][1]["ORT_DISABLE_TELEMETRY"], "1")
            self.assertIsInstance(calls[0][0], list)

    def test_runner_does_not_use_a_shell(self) -> None:
        with mock.patch("benchmarks.collect_baseline.subprocess.run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            subprocess_runner(["/opt/inferedge-bench", "--version"], cwd=None, env={"A": "b"})
        self.assertEqual(run.call_args.args[0], ["/opt/inferedge-bench", "--version"])
        self.assertFalse(run.call_args.kwargs.get("shell", False))

    def _run_failure(self, *, returncode: int, stderr: str, stdout: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_set = create_run_set(pathlib.Path(temporary), "run-set")

            def runner(args, cwd=None, env=None):
                return ProcessResult(returncode, stdout, stderr)

            accepted = execute_trials(
                run_set=run_set,
                binary=pathlib.Path("/opt/inferedge-bench"),
                repo_root=pathlib.Path("/repo"),
                preflight_result=self._paths(),
                counted_runs=1,
                warmup=WARMUP,
                iterations=ITERATIONS,
                check_interval=INTERVAL,
                delay_seconds=0,
                runner=runner,
                sleep=lambda _seconds: None,
            )
            self.assertFalse(accepted)
            self.assertFalse((run_set / "aggregate.json").exists())
            self.assertFalse((run_set / "runs" / "pilot-add.json").exists())
            self.assertTrue((run_set / "runs" / "pilot-add.stdout").is_file())
            self.assertEqual((run_set / "runs" / "pilot-add.stderr").read_text(encoding="utf-8"), stderr)

    def _paths(self) -> dict:
        return {
            "paths": {
                "add_model": pathlib.Path(ADD_MODEL),
                "mnist_model": pathlib.Path(MNIST_MODEL),
                "fixture_dir": pathlib.Path("/fixtures/mnist"),
            }
        }

    def _release_cache(self) -> dict[str, str]:
        return {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_CXX_COMPILER": "/usr/bin/c++",
            "CMAKE_CXX_FLAGS": "",
            "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
            "CMAKE_EXE_LINKER_FLAGS": "",
            "CMAKE_EXE_LINKER_FLAGS_RELEASE": "",
        }


class OnnxLinkageTest(unittest.TestCase):
    def test_macos_versioned_dylib_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versioned = self._runtime(root, "libonnxruntime.1.31.0.dylib")
            resolved = self._verify_macho(root, "@rpath/libonnxruntime.1.31.0.dylib", [root / "lib"])
            self.assertEqual(resolved, versioned.resolve())

    def test_macos_abi_symlink_resolves_to_pinned_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versioned = self._runtime(root, "libonnxruntime.1.31.0.dylib")
            (root / "lib" / "libonnxruntime.1.dylib").symlink_to("libonnxruntime.1.31.0.dylib")
            resolved = self._verify_macho(root, "@rpath/libonnxruntime.1.dylib", [root / "lib"])
            self.assertEqual(resolved, versioned.resolve())

    def test_wrong_runtime_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self._runtime(root, "libonnxruntime.1.31.0.dylib", version="1.22.0")
            with self.assertRaises(CollectorError) as caught:
                self._verify_macho(root, "@rpath/libonnxruntime.1.31.0.dylib", [root / "lib"])
            self.assertIn("1.22.0", str(caught.exception))

    def test_missing_symlink_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "lib").mkdir()
            (root / "VERSION_NUMBER").write_text("1.31.0\n", encoding="utf-8")
            (root / "lib" / "libonnxruntime.1.dylib").symlink_to("libonnxruntime.1.31.0.dylib")
            with self.assertRaises(CollectorError) as caught:
                self._verify_macho(root, "@rpath/libonnxruntime.1.dylib", [root / "lib"])
            self.assertIn("symlink target is missing", str(caught.exception))

    def test_library_outside_pinned_installation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            root = base / "pinned"
            self._runtime(root, "libonnxruntime.1.31.0.dylib")
            foreign = base / "other" / "libonnxruntime.1.31.0.dylib"
            foreign.parent.mkdir()
            foreign.write_bytes(b"foreign")
            (root / "lib" / "libonnxruntime.1.dylib").symlink_to(foreign)
            with self.assertRaises(CollectorError) as caught:
                self._verify_macho(root, "@rpath/libonnxruntime.1.dylib", [root / "lib"])
            self.assertIn("outside the pinned installation", str(caught.exception))

    def test_ambiguous_resolution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self._runtime(root, "libonnxruntime.1.31.0.dylib")
            first = root / "rpath-a"
            second = root / "rpath-b"
            first.mkdir()
            second.mkdir()
            (first / "libonnxruntime.1.dylib").write_bytes(b"a")
            (second / "libonnxruntime.1.dylib").write_bytes(b"b")
            with self.assertRaises(CollectorError) as caught:
                self._verify_macho(
                    root,
                    "@rpath/libonnxruntime.1.dylib",
                    [first, second],
                )
            self.assertIn("ambiguous", str(caught.exception))

    def test_unrecognized_dylib_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self._runtime(root, "libonnxruntime.1.31.0.dylib")
            link = (
                "inferedge-bench:\n"
                "\t@rpath/libonnxruntime.dylib (compatibility version 1.0.0, current version 1.31.0)\n"
            )
            with self.assertRaises(CollectorError) as caught:
                verify_linked_onnx_runtime(
                    binary=pathlib.Path("/tmp/inferedge-bench"),
                    runtime_root=root,
                    link_output=link,
                    load_command_output=self._rpaths([root / "lib"]),
                )
            self.assertIn("unrecognized ONNX Runtime dependency", str(caught.exception))

    def test_linux_versioned_library_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versioned = self._runtime(root, "libonnxruntime.so.1.31.0")
            link = f"\tlibonnxruntime.so.1.31.0 => {versioned} (0x1000)\n"
            resolved = verify_linked_onnx_runtime(
                binary=pathlib.Path("/tmp/inferedge-bench"),
                runtime_root=root,
                link_output=link,
            )
            self.assertEqual(resolved, versioned.resolve())

    def test_linux_soname_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            versioned = self._runtime(root, "libonnxruntime.so.1.31.0")
            soname = root / "lib" / "libonnxruntime.so.1"
            soname.symlink_to("libonnxruntime.so.1.31.0")
            link = f"\tlibonnxruntime.so.1 => {soname} (0x1000)\n"
            resolved = verify_linked_onnx_runtime(
                binary=pathlib.Path("/tmp/inferedge-bench"),
                runtime_root=root,
                link_output=link,
            )
            self.assertEqual(resolved, versioned.resolve())

    def test_missing_onnx_runtime_dependency_is_rejected(self) -> None:
        link = (
            "inferedge-bench:\n"
            "\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1.31.0)\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self._runtime(root, "libonnxruntime.1.31.0.dylib")
            with self.assertRaises(CollectorError) as caught:
                verify_linked_onnx_runtime(
                    binary=pathlib.Path("/tmp/inferedge-bench"),
                    runtime_root=root,
                    link_output=link,
                )
            self.assertIn("does not link ONNX Runtime", str(caught.exception))

    def test_unresolved_dependency_has_an_explicit_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self._runtime(root, "libonnxruntime.1.31.0.dylib")
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(CollectorError) as caught:
                self._verify_macho(root, "@rpath/libonnxruntime.1.dylib", [empty])
            message = str(caught.exception)
            self.assertIn("unable to resolve @rpath/libonnxruntime.1.dylib", message)
            self.assertIn(str(empty / "libonnxruntime.1.dylib"), message)

    def _runtime(self, root: pathlib.Path, filename: str, version: str = "1.31.0") -> pathlib.Path:
        library = root / "lib"
        library.mkdir(parents=True)
        (root / "VERSION_NUMBER").write_text(version + "\n", encoding="utf-8")
        versioned = library / filename
        versioned.write_bytes(b"ort")
        return versioned

    def _rpaths(self, paths: list[pathlib.Path]) -> str:
        return "".join(
            "          cmd LC_RPATH\n"
            "      cmdsize 16\n"
            f"         path {path} (offset 12)\n"
            for path in paths
        )

    def _verify_macho(
        self,
        root: pathlib.Path,
        install_name: str,
        rpaths: list[pathlib.Path],
    ) -> pathlib.Path:
        link = (
            "inferedge-bench:\n"
            f"\t{install_name} (compatibility version 1.0.0, current version 1.31.0)\n"
        )
        return verify_linked_onnx_runtime(
            binary=pathlib.Path("/tmp/inferedge-bench"),
            runtime_root=root,
            link_output=link,
            load_command_output=self._rpaths(rpaths),
        )


if __name__ == "__main__":
    unittest.main()
