#!/usr/bin/env python3
"""Collect a paired completion-timing overhead run.

OFF omits --completion-timing and must emit schema 3.
ON passes --completion-timing and must emit schema 4.
The first study is cell A, unbatched, at the low_open rate.
This module does not run the reportable 20-block study by itself
and does not change protocols 1, 2, or 3.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.collect_baseline import create_run_set, make_run_set_id, public_manifest
from benchmarks.collect_scheduler import (
    ALLOWED_UNTRACKED,
    CELL_BY_ID,
    DEFAULT_MEASUREMENT,
    PROFILE_BY_ID,
    SCHEMA_VERSION,
    PRECISION_SCHEMA_VERSION,
    CollectorError,
    classify_report,
    git_blob_sha256,
    preflight,
    scheduler_command,
    supervise_process,
    wall_timeout_seconds,
    replace_json,
    write_exclusive_text,
    write_json,
)

EXPERIMENT_KIND = "completion_timing_overhead"
INTENDED_VARIABLE = "enable_completion_timing"
PUBLICATION_GATE = "strict_generator_delivery_per_arm"
CELL_ID = "A"
PROFILE_ID = "low_open"
OUTPUT_ROOT_RELATIVE = "benchmarks/scheduler_runs"
REPORTABLE_COUNTED_BLOCKS = 20
MAX_COUNTED_BLOCKS = 30
BOOTSTRAP_SEED = 20261014
BOOTSTRAP_REPLICATES = 10000
CONFIDENCE_LEVEL = 0.95
# Sentinel stored in analysis_protocol_version so this run cannot be read as
# batching protocol 1, 2, or 3. It is not itself a published analysis protocol.
# The identity is experiment_kind.
OVERHEAD_PROTOCOL_MARKER = 0
OVERHEAD_PROTOCOL_MARKER_MEANING = (
    "sentinel only; not analysis protocol 1, 2, or 3; "
    "experiment_kind completion_timing_overhead is the identity"
)
# Counted block_index starts at 0. Even blocks run OFF then ON.
# Odd blocks run ON then OFF. The single pilot pair is uncounted and
# always runs OFF then ON. It does not use the counted rotation.


class OverheadError(CollectorError):
    def __init__(self, message: str, run_set: pathlib.Path | None = None) -> None:
        super().__init__(message)
        self.run_set = run_set


def counted_blocks_for(mode: str, counted_runs: int | None) -> int:
    if mode == "smoke":
        if counted_runs not in {None, 0}:
            raise OverheadError("smoke runs one uncounted OFF/ON pair and no counted blocks")
        return 0
    if mode == "reportable":
        if counted_runs not in {None, REPORTABLE_COUNTED_BLOCKS}:
            raise OverheadError("a reportable study uses 20 counted blocks")
        return REPORTABLE_COUNTED_BLOCKS
    if mode != "exploratory":
        raise OverheadError(f"unknown collection mode {mode!r}")
    if counted_runs is None or not 1 <= counted_runs <= MAX_COUNTED_BLOCKS:
        raise OverheadError("exploratory counted blocks must be from 1 through 30")
    return counted_runs


def arm_order(block_index: int) -> tuple[str, str]:
    """Counted rotation. block_index starts at zero."""
    if block_index % 2 == 0:
        return ("off", "on")
    return ("on", "off")


def build_plan(counted_blocks: int) -> list[dict]:
    if counted_blocks < 0 or counted_blocks > MAX_COUNTED_BLOCKS:
        raise OverheadError(f"counted blocks {counted_blocks} are outside 0..{MAX_COUNTED_BLOCKS}")
    trials: list[dict] = []
    execution_index = 0

    def add(arm: str, *, pilot: bool, block_index: int | None, position: int) -> None:
        nonlocal execution_index
        label = "pilot" if pilot else f"b{block_index:02d}"
        trials.append(
            {
                "trial_id": f"overhead-{label}-{arm}",
                "arm": arm,
                "pilot": pilot,
                "block_index": block_index,
                "position_in_block": position,
                "execution_index": execution_index,
                "run_index": None if pilot else block_index,
                "cell_id": CELL_ID,
                "profile_id": PROFILE_ID,
            }
        )
        execution_index += 1

    for position, arm in enumerate(("off", "on")):
        add(arm, pilot=True, block_index=None, position=position)
    for block_index in range(counted_blocks):
        for position, arm in enumerate(arm_order(block_index)):
            add(arm, pilot=False, block_index=block_index, position=position)
    return trials


def shared_launch() -> dict:
    """Settings shared by both arms. Completion timing is not part of this launch."""
    cell = CELL_BY_ID[CELL_ID]
    profile = PROFILE_BY_ID[PROFILE_ID]
    if (
        cell["enable_batching"] is not False
        or cell["max_batch_size"] != 1
        or cell["max_batch_delay_ms"] != 0
    ):
        raise OverheadError("cell A must stay unbatched with max batch size 1 and delay 0")
    if profile["workload"] != "open_loop_constant" or float(profile["target_rps"]) != 50.0:
        raise OverheadError("low_open must stay an open loop at 50 requests per second")
    measurement = DEFAULT_MEASUREMENT
    return {
        "cell_id": CELL_ID,
        "profile_id": PROFILE_ID,
        "workload": profile["workload"],
        "target_rps": profile["target_rps"],
        "enable_batching": False,
        "max_batch_size": cell["max_batch_size"],
        "max_batch_delay_ms": cell["max_batch_delay_ms"],
        "worker_count": measurement["worker_count"],
        "queue_capacity": measurement["queue_capacity"],
        "warmup_requests": measurement["warmup_requests"],
        "duration_ms": measurement["duration_ms"],
        "max_offered": measurement["max_offered"],
        "max_outstanding": measurement["max_outstanding"],
        "lag_tolerance_ms": measurement["lag_tolerance_ms"],
        "completion_timeout_ms": measurement["completion_timeout_ms"],
        "check_interval": measurement["check_interval"],
        "completion_timing": False,
    }


def evidence_class_for(mode: str, counted_blocks: int) -> str:
    if mode == "smoke":
        if counted_blocks != 0:
            raise OverheadError("a smoke pair cannot be promoted into counted blocks")
        return "correctness"
    if counted_blocks <= 0:
        raise OverheadError("a counted study needs at least one OFF/ON block")
    if mode == "reportable":
        return "counted_overhead"
    return "exploratory"


def bind_commands(plan: list[dict], binary: pathlib.Path, model: pathlib.Path) -> list[dict]:
    """Copy the plan and store the exact argv each child process will run."""
    bound = []
    for trial in plan:
        command = command_for_arm(binary, model, str(trial["arm"]))
        bound.append({**trial, "command": command})
    off = command_for_arm(binary, model, "off")
    on = command_for_arm(binary, model, "on")
    if not commands_differ_only_by_timing(off, on):
        raise OverheadError("OFF and ON commands differ by more than --completion-timing")
    return bound


def measurement_for(arm: str) -> dict:
    measurement = dict(DEFAULT_MEASUREMENT)
    if arm == "on":
        measurement["completion_timing"] = True
    elif arm != "off":
        raise OverheadError(f"unknown arm {arm!r}")
    return measurement


def command_for_arm(binary: pathlib.Path, model: pathlib.Path, arm: str) -> list[str]:
    return scheduler_command(
        binary,
        model,
        CELL_BY_ID[CELL_ID],
        PROFILE_BY_ID[PROFILE_ID],
        measurement_for(arm),
    )


def commands_differ_only_by_timing(off: list[str], on: list[str]) -> bool:
    if "--completion-timing" in off or on.count("--completion-timing") != 1:
        return False
    return [token for token in on if token != "--completion-timing"] == off


def expected_for(model: pathlib.Path, arm: str) -> dict:
    """Schema-3 or schema-4 classifier input.

    comparison_class primary selects the existing strict-delivery classifier.
    It does not record this run as a batching protocol.
    """
    launch = shared_launch()
    measurement = DEFAULT_MEASUREMENT
    return {
        "model_path": str(model),
        "workload": launch["workload"],
        "worker_count": measurement["worker_count"],
        "queue_capacity": measurement["queue_capacity"],
        "enable_batching": launch["enable_batching"],
        "max_batch_size": launch["max_batch_size"],
        "max_batch_delay_ms": launch["max_batch_delay_ms"],
        # New overhead evidence requires explicit adaptive identity.
        # Cell A remains non-adaptive; the runner must report false.
        "adaptive_batching": False,
        "require_adaptive_batching_field": True,
        "warmup_requests": measurement["warmup_requests"],
        "check_interval": measurement["check_interval"],
        "duration_ms": measurement["duration_ms"],
        "lag_tolerance_ms": measurement["lag_tolerance_ms"],
        "comparison_class": "primary",
        "analysis_protocol_version": 3 if arm == "on" else 2,
        "target_rps": PROFILE_BY_ID[PROFILE_ID]["target_rps"],
    }


def validate_arm(text: str, arm: str, model: pathlib.Path) -> tuple[dict | None, dict, list[str]]:
    """Schema 3 for OFF and schema 4 for ON, then the shared strict-delivery gate."""
    document, status = classify_report(text, expected=expected_for(model, arm))
    problems: list[str] = []
    if status.get("collector_valid") is not True:
        problems.extend(str(item) for item in status.get("problems", []))
        if not problems:
            problems.append("collector rejected the trial")
    if status.get("delivery_class") != "STRICT_PRIMARY":
        problems.append(
            f"delivery_class is {status.get('delivery_class')!r}, expected STRICT_PRIMARY"
        )
    if status.get("runner_valid") is not True:
        problems.append("runner valid flag is not true")
    if document is None:
        return None, status, problems
    expected_schema = PRECISION_SCHEMA_VERSION if arm == "on" else SCHEMA_VERSION
    if document.get("schema_version") != expected_schema:
        problems.append(
            f"{arm} schema_version is {document.get('schema_version')!r}, expected {expected_schema}"
        )
    timing = document.get("scheduler", {}).get("enable_completion_timing", False)
    if arm == "on" and timing is not True:
        problems.append("ON arm did not record enable_completion_timing")
    if arm == "off" and timing not in {False, None}:
        problems.append("OFF arm recorded completion timing")
    if document.get("correctness_mismatches") != 0:
        problems.append("correctness mismatches are not zero")
    if document.get("metrics", {}).get("histogram_saturated") is True:
        problems.append("histogram is saturated")
    return document, status, problems


def _write_failure(run_set: pathlib.Path, payload: dict) -> None:
    path = run_set / "failure.json"
    if path.exists():
        return
    write_json(path, payload)


def _overlay_identity(manifest: dict, plan: list[dict], counted_blocks: int, mode: str) -> dict:
    policy = dict(manifest["source_defined_policy"])
    protocol = dict(manifest["collection_protocol"])
    policy["publication_policy"] = PUBLICATION_GATE
    policy["experiment_kind"] = EXPERIMENT_KIND
    policy["intended_variable"] = INTENDED_VARIABLE
    policy["off_schema_version"] = SCHEMA_VERSION
    policy["on_schema_version"] = PRECISION_SCHEMA_VERSION
    policy["schema_version"] = None
    policy["analysis_protocol_version"] = OVERHEAD_PROTOCOL_MARKER
    policy["analysis_protocol_version_meaning"] = OVERHEAD_PROTOCOL_MARKER_MEANING
    policy["limitations"] = [
        "The contrast is the incremental cost of enabling completion timing in the current binary.",
        "Both arms already contain the larger SchedulerCompletion layout.",
        "This is not a measurement against the binary from before that layout.",
        "The first workload is the tiny Add model, unbatched, at 50 offered requests per second.",
        "That load may be too light to reveal meaningful timing overhead.",
        "Host scheduling, thermal drift, CPU frequency, and background load remain.",
        "Open-loop observation polls at 1 ms.",
        "Schema 3 does not report a generator-lag percentile distribution.",
        "The 1000 ms generator tolerance is not precise arrival scheduling.",
        "phase_boundary_fidelity is unavailable and burst phase fidelity is not claimed.",
        "Enabling timing may change batch grouping in a future batched experiment.",
        "These results are not a general overhead estimate.",
    ]
    protocol["publication_policy"] = PUBLICATION_GATE
    protocol["experiment_kind"] = EXPERIMENT_KIND
    protocol["intended_variable"] = INTENDED_VARIABLE
    protocol["analysis_protocol_version"] = OVERHEAD_PROTOCOL_MARKER
    protocol["analysis_protocol_version_meaning"] = OVERHEAD_PROTOCOL_MARKER_MEANING
    protocol["collection_mode"] = mode
    protocol["evidence_class"] = evidence_class_for(mode, counted_blocks)
    protocol["counted_blocks"] = counted_blocks
    protocol["shared_launch"] = shared_launch()
    protocol["discarded_batching_plan"] = True
    protocol["pilot_pair"] = "off_then_on"
    protocol["counted_rotation"] = "block_index starts at 0; even off-then-on; odd on-then-off"
    protocol["bootstrap_seed"] = BOOTSTRAP_SEED
    protocol["bootstrap_replicates"] = BOOTSTRAP_REPLICATES
    protocol["confidence_level"] = CONFIDENCE_LEVEL
    protocol["cell_id"] = CELL_ID
    protocol["profile_id"] = PROFILE_ID
    protocol["arrival_fidelity_note"] = (
        "Both arms must meet strict generator delivery. Schema 3 has no lag percentiles, "
        "so the arms do not share a certified arrival-percentile bound. "
        "The 1000 ms lag tolerance is not precise arrival scheduling. "
        "phase_boundary_fidelity is unavailable."
    )
    manifest = dict(manifest)
    manifest["experiment_kind"] = EXPERIMENT_KIND
    manifest["intended_variable"] = INTENDED_VARIABLE
    manifest["source_defined_policy"] = policy
    manifest["collection_protocol"] = protocol
    manifest["plan"] = plan
    manifest["allowed_untracked_roots"] = list(ALLOWED_UNTRACKED)
    return manifest


def execute_plan(
    *,
    run_set: pathlib.Path,
    binary: pathlib.Path,
    repo_root: pathlib.Path,
    model: pathlib.Path,
    plan: list[dict],
    checked: dict,
    runner,
    sleep,
) -> bool:
    del sleep
    runs = run_set / "runs"
    measurement = DEFAULT_MEASUREMENT
    timeout_seconds = wall_timeout_seconds(measurement)
    for trial in plan:
        trial_id = str(trial["trial_id"])
        arm = str(trial["arm"])
        command = list(trial["command"])
        expected_command = command_for_arm(binary, model, arm)
        if command != expected_command:
            raise OverheadError(
                f"{trial_id} plan command diverges from the {arm} arm command",
                run_set,
            )
        env = {"ORT_DISABLE_TELEMETRY": "1"}
        try:
            result = runner(command, cwd=repo_root, env=env, timeout_seconds=timeout_seconds)
        except TypeError:
            result = runner(command, cwd=repo_root, env=env)
        stdout_path = runs / f"{trial_id}.stdout"
        stderr_path = runs / f"{trial_id}.stderr"
        report_path = runs / f"{trial_id}.json"
        timed_out = bool(getattr(result, "timed_out", False))
        problems: list[str] = []
        document = None
        status: dict = {
            "runner_valid": False,
            "collector_valid": False,
            "delivery_class": None,
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
            document, status, arm_problems = validate_arm(result.stdout, arm, model)
            problems.extend(arm_problems)
            if not problems and document is not None:
                try:
                    write_exclusive_text(report_path, result.stdout)
                except (OSError, CollectorError) as exc:
                    problems.append(f"could not preserve trial report: {exc}")
        meta = {
            "trial_id": trial_id,
            "experiment_kind": EXPERIMENT_KIND,
            "intended_variable": INTENDED_VARIABLE,
            "arm": arm,
            "pilot": bool(trial["pilot"]),
            "block_index": trial["block_index"],
            "position_in_block": trial["position_in_block"],
            "execution_index": trial["execution_index"],
            "run_index": trial["run_index"],
            "cell_id": CELL_ID,
            "profile_id": PROFILE_ID,
            "schema_version": None if document is None else document.get("schema_version"),
            "command": command,
            "returncode": result.returncode,
            "execution_timeout": timed_out,
            "wall_timeout_seconds": timeout_seconds,
            "runner_valid": status.get("runner_valid"),
            "collector_valid": status.get("collector_valid") is True and not problems,
            "delivery_class": status.get("delivery_class"),
            "problems": problems,
            "benchmark_binary_sha256": checked["queried"]["benchmark_binary_sha256"],
            "model_sha256": checked["verified_configuration"]["model_sha256"],
            "git_commit": checked["commit"],
            "retried": False,
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
                    "arm": arm,
                    "pilot": bool(trial["pilot"]),
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
    sleep=None,
    blob_sha256=git_blob_sha256,
    clock=None,
) -> pathlib.Path:
    repo_root = pathlib.Path(args.repo_root).resolve()
    binary = pathlib.Path(args.binary).resolve()
    counted = counted_blocks_for(args.mode, args.counted_runs)
    cell = CELL_BY_ID[CELL_ID]
    profile = PROFILE_BY_ID[PROFILE_ID]
    measurement = dict(DEFAULT_MEASUREMENT)
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
        cells=[cell],
        profiles=[profile],
        measurement=dict(measurement),
        counted_runs=1,
        delay_seconds=0.0,
        comparison_class="exploratory",
        analysis_protocol_version=2,
        runner=query_runner,
        blob_sha256=blob_sha256,
    )
    plan = bind_commands(build_plan(counted), binary, checked["paths"]["model"])
    moment = datetime.datetime.now(datetime.timezone.utc) if clock is None else clock()
    run_set_id = "overhead-" + make_run_set_id(moment, checked["commit"])
    parent = (
        pathlib.Path(args.output_root).resolve()
        if args.output_root
        else repo_root / OUTPUT_ROOT_RELATIVE
    )
    try:
        run_set = create_run_set(parent, run_set_id)
    except CollectorError as exc:
        raise OverheadError(str(exc)) from exc
    manifest = _overlay_identity(
        public_manifest(checked, run_set_id, "running"),
        plan,
        counted,
        args.mode,
    )
    try:
        write_json(run_set / "environment.json", manifest)
        write_json(run_set / "execution_plan.json", {"trials": plan})
        accepted = execute_plan(
            run_set=run_set,
            binary=binary,
            repo_root=repo_root,
            model=checked["paths"]["model"],
            plan=plan,
            checked=checked,
            runner=runner,
            sleep=sleep,
        )
    except OverheadError:
        raise
    except CollectorError as exc:
        raise OverheadError(str(exc), run_set) from exc
    except Exception as exc:
        raise OverheadError(f"overhead collection error: {exc}", run_set) from exc
    final = _overlay_identity(
        public_manifest(checked, run_set_id, "completed" if accepted else "failed"),
        plan,
        counted,
        args.mode,
    )
    replace_json(run_set / "environment.json", final)
    if not accepted:
        raise OverheadError(f"overhead collection stopped; see {run_set / 'failure.json'}", run_set)
    return run_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a cell-A low_open completion-timing overhead run under "
            "benchmarks/scheduler_runs. Smoke is one uncounted pair. "
            "This does not analyze or publish a speed claim."
        )
    )
    parser.add_argument("--binary", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--mode", choices=("smoke", "exploratory", "reportable"), required=True)
    parser.add_argument("--counted-runs", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        collect(args)
    except OverheadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
