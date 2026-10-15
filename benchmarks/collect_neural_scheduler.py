#!/usr/bin/env python3
"""Collect schema-3 neural scheduler trials for InferEdge.

This module is additive. It does not teach the historical Add collector about
neural models. Reportable mode allowlists gemm_f32 A versus C (C_minus_A), the
frozen A versus G2 contrast (G2_minus_A at 1000 RPS), and the three-arm
A/G2/G3 adaptive contrast (G3_minus_G2 primary at 1000 RPS). Offered rate must
be frozen before reportable mode. This collector does not run a real campaign by
itself.
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
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.collect_baseline import (
    BACKEND,
    GRAPH_OPTIMIZATIONS,
    ONNXRUNTIME_VERSION,
    CollectorError,
    assess_build_cache,
    cache_flag_text,
    compiler_and_cmake,
    create_run_set,
    git_status_problems,
    link_tool_args,
    locate_pinned_runtime,
    make_run_set_id,
    parse_cmake_cache,
    public_manifest,
    query_text,
    sanitizer_symbol_problems,
    sha256_file,
    symbol_tool_args,
    utc_now,
    verify_linked_onnx_runtime,
)
from benchmarks.collect_concurrent import hardware_metadata, release_optimization_problems
from benchmarks.collect_scheduler import (
    ALLOWED_UNTRACKED,
    BATCH_DECISION_SCHEMA_VERSION,
    BINARY_NAME,
    BLOCK_PROTOCOL,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CELL_BY_ID,
    CONFIDENCE_LEVEL,
    DEFAULT_MEASUREMENT,
    MINIMUM_INTERVAL_BLOCKS,
    PREFERRED_COUNTED_RUNS,
    PUBLICATION_POLICY,
    SCHEMA_VERSION,
    STATISTICAL_METHOD,
    TOOL_VERSION,
    classify_report,
    expected_schema_version,
    execution_plan,
    locate_build_dir,
    require_commit_sha,
    require_counted_runs,
    require_one_family,
    resolve_contrast,
    scheduler_command,
    supervise_process,
    version_text_problems,
    wall_timeout_seconds,
    write_exclusive_text,
    write_json,
    replace_json,
)

EXPERIMENT_KIND = "neural_scheduler_batching"
REFERENCE_CHECKING_POLICY = "precomputed_reference_compare"
CONTRAST_SIGN_CONVENTION = "C_minus_A"
OUTPUT_ROOT_RELATIVE = "benchmarks/scheduler_runs"
ANALYSIS_PROTOCOL_VERSION = 2
MAX_COUNTED_RUNS = 30
DEFAULT_DELAY_SECONDS = 0.0
REPORTABLE_MODEL_ID = "gemm_f32"
REPORTABLE_CELLS = ("A", "C")
REPORTABLE_BASELINE = "A"
REPORTABLE_ALTERNATIVE = "C"

# Neural-local delay / adaptive / arrival-aware probes. Not part of the historical
# Add CELL_BY_ID registry. Distinct from shared cell G (max_batch_size 8, delay 1)
# and H (max 16, delay 1).
# G3 is minimum-target early-sealing (C++ kAdaptiveMinTargetBatch=2); no CLI for the target.
# G4 is arrival-aware early seal (EWMA alpha=1/8); no estimator tuning CLI.
NEURAL_LOCAL_CELLS = (
    {"id": "G1", "enable_batching": True, "max_batch_size": 4, "max_batch_delay_ms": 1},
    {"id": "G2", "enable_batching": True, "max_batch_size": 4, "max_batch_delay_ms": 2},
    {
        "id": "G3",
        "enable_batching": True,
        "max_batch_size": 4,
        "max_batch_delay_ms": 2,
        "adaptive_batching": True,
    },
    {
        "id": "G4",
        "enable_batching": True,
        "max_batch_size": 4,
        "max_batch_delay_ms": 2,
        "adaptive_batching": False,
        "arrival_aware_batching": True,
    },
)
NEURAL_LOCAL_CELL_BY_ID = {cell["id"]: cell for cell in NEURAL_LOCAL_CELLS}
# G1 stays calibration-only. G2 is calibration-only except the frozen A/G2
# reportable contrast. G3 is calibration-only except the frozen A/G2/G3
# three-arm reportable contrast. G4 is calibration-only with no reportable contrast.
CALIBRATION_ONLY_NEURAL_CELLS = frozenset(NEURAL_LOCAL_CELL_BY_ID)
CALIBRATION_VARIANT_NONZERO_BATCH_DELAY_V1 = "nonzero_batch_delay_v1"
CALIBRATION_VARIANT_ADAPTIVE_EARLY_SEAL_V1 = "adaptive_early_seal_v1"
CALIBRATION_VARIANT_ARRIVAL_AWARE_BATCHING_V1 = "arrival_aware_batching_v1"
CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_V1 = "arrival_aware_sparse_v1"
CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_INSTRUMENTED_V1 = (
    "arrival_aware_sparse_instrumented_v1"
)
ADAPTIVE_EARLY_SEAL_CELLS = ("A", "G2", "G3")
ARRIVAL_AWARE_BATCHING_CELLS = ("A", "G2", "G3", "G4")
ADAPTIVE_EARLY_SEAL_RATE = 1000.0
ADAPTIVE_EARLY_SEAL_COUNTED_RUNS = 1
ARRIVAL_AWARE_BATCHING_RATE = 1000.0
ARRIVAL_AWARE_BATCHING_COUNTED_RUNS = 1
ARRIVAL_AWARE_BATCHING_DURATION_MS = 1000
ARRIVAL_AWARE_SPARSE_RATES = frozenset({50.0, 100.0})
ARRIVAL_AWARE_SPARSE_DURATION_MS = 5000
ARRIVAL_AWARE_SPARSE_COUNTED_RUNS = 1
ARRIVAL_AWARE_SPARSE_INSTRUMENTED_RATE = 100.0
ARRIVAL_AWARE_SPARSE_INSTRUMENTED_DURATION_MS = 5000
ARRIVAL_AWARE_SPARSE_INSTRUMENTED_COUNTED_RUNS = 1
ARRIVAL_AWARE_SPARSE_INSTRUMENTED_CAPACITY = 1024
STRICT_CALIBRATION_VARIANTS = frozenset(
    {
        CALIBRATION_VARIANT_ADAPTIVE_EARLY_SEAL_V1,
        CALIBRATION_VARIANT_ARRIVAL_AWARE_BATCHING_V1,
        CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_V1,
        CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_INSTRUMENTED_V1,
    }
)
SEAL_REASONS_FORBIDDEN_G2 = frozenset({"g3_minimum_target", "g4_arrival_aware"})
SEAL_REASONS_FORBIDDEN_G3 = frozenset({"g4_arrival_aware"})
SEAL_REASONS_FORBIDDEN_G4 = frozenset({"g3_minimum_target"})
CELL_DEFINITION_KEYS = ("id", "enable_batching", "max_batch_size", "max_batch_delay_ms")
CELL_DEFINITION_ADAPTIVE_KEY = "adaptive_batching"
CELL_DEFINITION_ARRIVAL_KEY = "arrival_aware_batching"
CELL_DEFINITION_ALLOWED_KEYS = CELL_DEFINITION_KEYS + (
    CELL_DEFINITION_ADAPTIVE_KEY,
    CELL_DEFINITION_ARRIVAL_KEY,
)

CONTRAST_ID_AC = "gemm_f32_A_vs_C"
CONTRAST_ID_AG2 = "gemm_f32_A_vs_G2_delay2ms_rps1000"
CONTRAST_ID_AG2G3 = "gemm_f32_A_G2_G3_adaptive_rps1000"
CONTRAST_SIGN_G2_MINUS_A = "G2_minus_A"
CONTRAST_SIGN_G3_MINUS_G2 = "G3_minus_G2"
CONTRAST_SIGN_G3_MINUS_A = "G3_minus_A"
ARRIVAL_FIDELITY_STATUS_NOT_CERTIFIED = "not_certified"
PHASE_BOUNDARY_FIDELITY_UNAVAILABLE = "unavailable"
ARRIVAL_SCREEN_POLICY_VERSION = "coarse_max_lag_v1"
ARRIVAL_SCREEN_MAX_LAG_NS = 5_000_000
PERSISTENT_BATCHING_MIN_MEAN = 2.0
AG2_OFFERED_RATE = 1000.0
AG2G3_OFFERED_RATE = 1000.0
# Twenty counted blocks under within_profile_cyclic_rotation on (A,G2,G3).
# Perfect three-position balance is impossible; record and enforce these counts.
THREE_ARM_COUNTED_BLOCKS = PREFERRED_COUNTED_RUNS
THREE_ARM_POSITION_COUNTS_20 = {
    "A": {0: 7, 1: 6, 2: 7},
    "G2": {0: 7, 1: 7, 2: 6},
    "G3": {0: 6, 1: 7, 2: 7},
}

REPORTABLE_CONTRAST_AC = {
    "contrast_id": CONTRAST_ID_AC,
    "model_id": REPORTABLE_MODEL_ID,
    "cells": REPORTABLE_CELLS,
    "baseline_cell": REPORTABLE_BASELINE,
    "alternative_cell": REPORTABLE_ALTERNATIVE,
    "contrast_sign_convention": CONTRAST_SIGN_CONVENTION,
    "required_counted_runs": None,
    "fixed_offered_rate": None,
    "arrival_fidelity_status": None,
    "phase_boundary_fidelity": None,
    "arrival_screen_max_lag_ns": None,
    "arrival_screen_policy_version": None,
    "persistent_batching_cell": None,
    "persistent_batching_min_mean": None,
    "allow_neural_local_cells": frozenset(),
    "require_five_key_cell_definitions": False,
    "expected_cells": {
        "A": {"id": "A", "enable_batching": False, "max_batch_size": 1, "max_batch_delay_ms": 0},
        "C": {"id": "C", "enable_batching": True, "max_batch_size": 4, "max_batch_delay_ms": 0},
    },
}

REPORTABLE_CONTRAST_AG2 = {
    "contrast_id": CONTRAST_ID_AG2,
    "model_id": REPORTABLE_MODEL_ID,
    "cells": ("A", "G2"),
    "baseline_cell": "A",
    "alternative_cell": "G2",
    "contrast_sign_convention": CONTRAST_SIGN_G2_MINUS_A,
    "required_counted_runs": PREFERRED_COUNTED_RUNS,
    "fixed_offered_rate": AG2_OFFERED_RATE,
    "arrival_fidelity_status": ARRIVAL_FIDELITY_STATUS_NOT_CERTIFIED,
    "phase_boundary_fidelity": PHASE_BOUNDARY_FIDELITY_UNAVAILABLE,
    "arrival_screen_max_lag_ns": ARRIVAL_SCREEN_MAX_LAG_NS,
    "arrival_screen_policy_version": ARRIVAL_SCREEN_POLICY_VERSION,
    "persistent_batching_cell": "G2",
    "persistent_batching_min_mean": PERSISTENT_BATCHING_MIN_MEAN,
    "allow_neural_local_cells": frozenset({"G2"}),
    "require_five_key_cell_definitions": False,
    "expected_cells": {
        "A": {"id": "A", "enable_batching": False, "max_batch_size": 1, "max_batch_delay_ms": 0},
        "G2": {"id": "G2", "enable_batching": True, "max_batch_size": 4, "max_batch_delay_ms": 2},
    },
}

REPORTABLE_CONTRAST_AG2G3 = {
    "contrast_id": CONTRAST_ID_AG2G3,
    "model_id": REPORTABLE_MODEL_ID,
    "cells": ("A", "G2", "G3"),
    "baseline_cell": "G2",
    "alternative_cell": "G3",
    "contrast_sign_convention": CONTRAST_SIGN_G3_MINUS_G2,
    "secondary_baseline_cell": "A",
    "secondary_alternative_cell": "G3",
    "secondary_contrast_sign_convention": CONTRAST_SIGN_G3_MINUS_A,
    "secondary_comparison_class": "exploratory",
    "secondary_multiple_comparisons_adjustment": "none_unadjusted",
    "required_counted_runs": THREE_ARM_COUNTED_BLOCKS,
    "fixed_offered_rate": AG2G3_OFFERED_RATE,
    "arrival_fidelity_status": ARRIVAL_FIDELITY_STATUS_NOT_CERTIFIED,
    "phase_boundary_fidelity": PHASE_BOUNDARY_FIDELITY_UNAVAILABLE,
    "arrival_screen_max_lag_ns": ARRIVAL_SCREEN_MAX_LAG_NS,
    "arrival_screen_policy_version": ARRIVAL_SCREEN_POLICY_VERSION,
    "persistent_batching_cell": "G2",
    "persistent_batching_min_mean": PERSISTENT_BATCHING_MIN_MEAN,
    "unbatched_cell": "A",
    "allow_neural_local_cells": frozenset({"G2", "G3"}),
    "require_five_key_cell_definitions": True,
    "legacy_adaptive_identity_prohibited": True,
    "expected_position_counts": THREE_ARM_POSITION_COUNTS_20,
    "expected_cells": {
        "A": {
            "id": "A",
            "enable_batching": False,
            "max_batch_size": 1,
            "max_batch_delay_ms": 0,
            "adaptive_batching": False,
        },
        "G2": {
            "id": "G2",
            "enable_batching": True,
            "max_batch_size": 4,
            "max_batch_delay_ms": 2,
            "adaptive_batching": False,
        },
        "G3": {
            "id": "G3",
            "enable_batching": True,
            "max_batch_size": 4,
            "max_batch_delay_ms": 2,
            "adaptive_batching": True,
        },
    },
}

REPORTABLE_CONTRASTS_BY_ID = {
    CONTRAST_ID_AC: REPORTABLE_CONTRAST_AC,
    CONTRAST_ID_AG2: REPORTABLE_CONTRAST_AG2,
    CONTRAST_ID_AG2G3: REPORTABLE_CONTRAST_AG2G3,
}

NEURAL_MODELS = {
    "mlp_small_f32": {
        "model_relative": "models/mlp_small_f32.onnx",
        "model_sha256": "bcf02cac789c65cb12785ccfd175827cb21f89574ef298af009b74d7131df1bd",
        "sidecar_relative": "models/fixtures/mlp_small_f32.weights",
        "sidecar_sha256": "b6335064c4e2c89add3ed54452e27dacf3ea7d485be6adbdd24fae1e2e5f1899",
        "input_name": "features",
        "output_name": "logits",
        "input_width": 32,
        "output_width": 8,
    },
    "mlp_medium_f32": {
        "model_relative": "models/mlp_medium_f32.onnx",
        "model_sha256": "8d43a97b8d44d3d1f601570c8b09b08c90dda4977e6f8b7123eea8c0455024dd",
        "sidecar_relative": "models/fixtures/mlp_medium_f32.weights",
        "sidecar_sha256": "d3dd2f6148a6684288d4f25b0c426eefde7227468be9120ddd08cd99cbb9059e",
        "input_name": "features",
        "output_name": "logits",
        "input_width": 256,
        "output_width": 16,
    },
    "gemm_f32": {
        "model_relative": "models/gemm_f32.onnx",
        "model_sha256": "a6203020b870eb04f9fd8e836cf4ccd930a57156a026d2a50163121d83debf32",
        "sidecar_relative": "models/fixtures/gemm_f32.weights",
        "sidecar_sha256": "a335c8b12defcb885f8bada9ef1092655d024d5586fe011a9336ad2d472f82b6",
        "input_name": "features",
        "output_name": "product",
        "input_width": 512,
        "output_width": 512,
    },
}


class NeuralCollectorError(CollectorError):
    def __init__(self, message: str, run_set: pathlib.Path | None = None) -> None:
        super().__init__(message)
        self.run_set = run_set


def model_contract(model_id: str) -> dict:
    try:
        return dict(NEURAL_MODELS[model_id])
    except KeyError as exc:
        known = ", ".join(sorted(NEURAL_MODELS))
        raise NeuralCollectorError(f"unsupported model_id {model_id!r}; known: {known}") from exc


def resolve_neural_cell(
    cell_id: str, *, allow_g3: bool = False, allow_g4: bool = False
) -> dict:
    """Resolve one cell id to an independent dict copy.

    G1/G2/G3/G4 come from neural-local definitions. All other ids come from the
    shared Add CELL_BY_ID registry. The shared registry is never mutated.

    Bare resolve_neural_cell("G3") / ("G4") is rejected so those cells cannot be
    treated as general. Callers must resolve them through resolve_neural_cells
    with mode='calibration' (G4), or calibration / allowlisted A/G2/G3 reportable
    (G3).
    """
    if cell_id == "G4" and not allow_g4:
        raise NeuralCollectorError(
            "G4 is calibration-only unless resolved via resolve_neural_cells "
            "for calibration mode"
        )
    if cell_id == "G3" and not allow_g3:
        raise NeuralCollectorError(
            "G3 is calibration-only unless resolved via resolve_neural_cells "
            "for calibration or the allowlisted A/G2/G3 reportable contrast"
        )
    if cell_id in NEURAL_LOCAL_CELL_BY_ID:
        return dict(NEURAL_LOCAL_CELL_BY_ID[cell_id])
    if cell_id in CELL_BY_ID:
        return dict(CELL_BY_ID[cell_id])
    known_shared = ", ".join(CELL_BY_ID)
    known_local = ", ".join(NEURAL_LOCAL_CELL_BY_ID)
    raise NeuralCollectorError(
        f"unknown cell {cell_id!r}; shared cells are {known_shared}; "
        f"neural-local calibration cells are {known_local}"
    )


def resolve_neural_cells(
    cell_ids: list[str],
    mode: str,
    *,
    reportable_contrast: dict | None = None,
) -> list[dict]:
    """Resolve ordered cell ids for the neural collector.

    G1 remains calibration-only. G2 is calibration-only except the frozen A/G2
    reportable contrast. G3 is calibration-only except the frozen A/G2/G3
    reportable contrast. G4 is calibration-only with no reportable contrast.
    Smoke never accepts G3 or G4. Duplicates are rejected.
    """
    if not cell_ids:
        raise NeuralCollectorError("cells are required")
    seen: set[str] = set()
    duplicates: list[str] = []
    for cell_id in cell_ids:
        if cell_id in seen:
            duplicates.append(cell_id)
        seen.add(cell_id)
    if duplicates:
        raise NeuralCollectorError(f"duplicate cell id(s): {', '.join(duplicates)}")
    local_requested = [cell_id for cell_id in cell_ids if cell_id in CALIBRATION_ONLY_NEURAL_CELLS]
    allowed_local: frozenset[str] = frozenset()
    if local_requested:
        if mode == "calibration":
            allowed_local = CALIBRATION_ONLY_NEURAL_CELLS
        elif mode == "reportable" and reportable_contrast is not None:
            allowed_local = frozenset(reportable_contrast.get("allow_neural_local_cells") or ())
        disallowed = [cell_id for cell_id in local_requested if cell_id not in allowed_local]
        if disallowed:
            raise NeuralCollectorError(
                f"neural-local cells {', '.join(disallowed)} are not permitted in mode {mode!r}"
            )
    allow_g3 = mode == "calibration" or "G3" in allowed_local
    allow_g4 = mode == "calibration"
    return [
        resolve_neural_cell(cell_id, allow_g3=allow_g3, allow_g4=allow_g4)
        for cell_id in cell_ids
    ]


def _require_json_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise NeuralCollectorError(f"{label} must be a JSON boolean")
    return value


def adaptive_batching_for_cell(cell: dict) -> bool:
    """Return the effective adaptive_batching policy for a neural cell.

    G3 must explicitly declare adaptive_batching=true. Historical A/C/G1/G2/G4
    default to false when the key is absent. Non-boolean values are rejected.
    G4 cannot set adaptive_batching=true.
    """
    if not isinstance(cell, dict):
        raise NeuralCollectorError("cell definition is not an object")
    cell_id = str(cell.get("id", ""))
    if cell_id == "G3":
        if CELL_DEFINITION_ADAPTIVE_KEY not in cell:
            raise NeuralCollectorError("G3 requires adaptive_batching=true")
        value = _require_json_bool(cell[CELL_DEFINITION_ADAPTIVE_KEY], "adaptive_batching")
        if value is not True:
            raise NeuralCollectorError("G3 requires adaptive_batching=true")
        if cell.get("enable_batching") is not True:
            raise NeuralCollectorError("G3 requires enable_batching=true")
        if int(cell.get("max_batch_size", -1)) != 4:
            raise NeuralCollectorError("G3 requires max_batch_size=4")
        if int(cell.get("max_batch_delay_ms", -1)) != 2:
            raise NeuralCollectorError("G3 requires max_batch_delay_ms=2")
        return True
    if CELL_DEFINITION_ADAPTIVE_KEY in cell:
        value = _require_json_bool(cell[CELL_DEFINITION_ADAPTIVE_KEY], "adaptive_batching")
        if value is True:
            raise NeuralCollectorError(
                f"cell {cell_id!r} cannot set adaptive_batching=true (G3 only)"
            )
        return False
    return False


def arrival_aware_batching_for_cell(cell: dict) -> bool:
    """Return the effective arrival_aware_batching policy for a neural cell.

    G4 must explicitly declare arrival_aware_batching=true with adaptive false.
    G3 and historical cells default to false when the key is absent. G3 cannot
    set arrival_aware_batching=true. Non-boolean values are rejected.
    """
    if not isinstance(cell, dict):
        raise NeuralCollectorError("cell definition is not an object")
    cell_id = str(cell.get("id", ""))
    if cell_id == "G4":
        if CELL_DEFINITION_ARRIVAL_KEY not in cell:
            raise NeuralCollectorError("G4 requires arrival_aware_batching=true")
        value = _require_json_bool(cell[CELL_DEFINITION_ARRIVAL_KEY], "arrival_aware_batching")
        if value is not True:
            raise NeuralCollectorError("G4 requires arrival_aware_batching=true")
        if cell.get("enable_batching") is not True:
            raise NeuralCollectorError("G4 requires enable_batching=true")
        if int(cell.get("max_batch_size", -1)) != 4:
            raise NeuralCollectorError("G4 requires max_batch_size=4")
        if int(cell.get("max_batch_delay_ms", -1)) != 2:
            raise NeuralCollectorError("G4 requires max_batch_delay_ms=2")
        if CELL_DEFINITION_ADAPTIVE_KEY in cell:
            adaptive = _require_json_bool(
                cell[CELL_DEFINITION_ADAPTIVE_KEY], "adaptive_batching"
            )
            if adaptive is True:
                raise NeuralCollectorError("G4 requires adaptive_batching=false")
        return True
    if cell_id == "G3":
        if CELL_DEFINITION_ARRIVAL_KEY in cell:
            value = _require_json_bool(
                cell[CELL_DEFINITION_ARRIVAL_KEY], "arrival_aware_batching"
            )
            if value is True:
                raise NeuralCollectorError("G3 cannot set arrival_aware_batching=true")
            return False
        return False
    if CELL_DEFINITION_ARRIVAL_KEY in cell:
        value = _require_json_bool(cell[CELL_DEFINITION_ARRIVAL_KEY], "arrival_aware_batching")
        if value is True:
            raise NeuralCollectorError(
                f"cell {cell_id!r} cannot set arrival_aware_batching=true (G4 only)"
            )
        return False
    return False


def normalize_cell_definition(cell: dict) -> dict:
    """Normalize a cell definition for comparison and in-memory use.

    Base four keys are always required. adaptive_batching and
    arrival_aware_batching are optional for historical cells (effective false),
    mandatory true for G3 and G4 respectively. Unknown keys are rejected. The
    normalized form always carries both policy booleans so analyzer imports
    comparing four-key or five-key historical definitions still match after
    defaulting missing fields to false.
    """
    missing = [key for key in CELL_DEFINITION_KEYS if key not in cell]
    if missing:
        raise NeuralCollectorError(f"cell definition missing fields: {', '.join(missing)}")
    unexpected = [key for key in cell if key not in CELL_DEFINITION_ALLOWED_KEYS]
    if unexpected:
        raise NeuralCollectorError(f"cell definition has unexpected fields: {', '.join(unexpected)}")
    base = {
        "id": str(cell["id"]),
        "enable_batching": bool(cell["enable_batching"]),
        "max_batch_size": int(cell["max_batch_size"]),
        "max_batch_delay_ms": int(cell["max_batch_delay_ms"]),
    }
    probe = dict(base)
    if CELL_DEFINITION_ADAPTIVE_KEY in cell:
        probe[CELL_DEFINITION_ADAPTIVE_KEY] = cell[CELL_DEFINITION_ADAPTIVE_KEY]
    if CELL_DEFINITION_ARRIVAL_KEY in cell:
        probe[CELL_DEFINITION_ARRIVAL_KEY] = cell[CELL_DEFINITION_ARRIVAL_KEY]
    adaptive = adaptive_batching_for_cell(probe)
    arrival = arrival_aware_batching_for_cell(probe)
    if adaptive and arrival:
        raise NeuralCollectorError(
            "adaptive_batching and arrival_aware_batching are mutually exclusive"
        )
    return {
        **base,
        CELL_DEFINITION_ADAPTIVE_KEY: adaptive,
        CELL_DEFINITION_ARRIVAL_KEY: arrival,
    }


def serialize_cell_definitions(
    cells: list[dict],
    *,
    include_adaptive: bool,
    include_arrival_aware: bool = False,
) -> list[dict]:
    """Serialize cell_definitions for environment metadata.

    Historical A/C and A/G2 reportable keep four-key on-disk shapes.
    adaptive_early_seal_v1 calibration and the A/G2/G3 reportable contrast
    require five-key definitions for every cell. arrival_aware_batching_v1
    and arrival_aware_sparse_v1 require six-key definitions for every cell.
    Format is chosen by the caller from the validated calibration / reportable
    variant, not inferred from arbitrary cell dictionary keys.
    """
    if include_arrival_aware and not include_adaptive:
        raise NeuralCollectorError(
            "six-key cell_definitions require include_adaptive with include_arrival_aware"
        )
    serialized: list[dict] = []
    for cell in cells:
        normalized = normalize_cell_definition(cell)
        entry = {
            "id": normalized["id"],
            "enable_batching": normalized["enable_batching"],
            "max_batch_size": normalized["max_batch_size"],
            "max_batch_delay_ms": normalized["max_batch_delay_ms"],
        }
        if include_adaptive:
            entry[CELL_DEFINITION_ADAPTIVE_KEY] = bool(normalized[CELL_DEFINITION_ADAPTIVE_KEY])
        if include_arrival_aware:
            entry[CELL_DEFINITION_ARRIVAL_KEY] = bool(normalized[CELL_DEFINITION_ARRIVAL_KEY])
        serialized.append(entry)
    return serialized


def _require_finite_number(value: object, label: str) -> float:
    """Return a finite float; reject bools and non-numeric values."""
    if type(value) is bool or type(value) not in (int, float):
        raise NeuralCollectorError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise NeuralCollectorError(f"{label} must be a finite number")
    return number


def _require_exact_int(value: object, label: str, expected: int) -> int:
    """Return an exact integer match; reject bools (bool is a subclass of int)."""
    if type(value) is not int:
        raise NeuralCollectorError(f"{label} must be an integer")
    if value != expected:
        raise NeuralCollectorError(f"{label} must be {expected}")
    return value


def resolve_calibration_variant(
    *,
    mode: str,
    cells: list[dict],
    model_id: str,
    profile: dict,
    measurement: dict,
    counted_runs: int,
) -> str | None:
    """Return a named calibration variant or None.

    G4 cell set A,G2,G3,G4 dispatches exactly:
    - arrival_aware_batching_v1: gemm_f32, 1000 RPS, duration_ms=1000, workers=1,
      counted_runs=1, instrumentation OFF (E7 plumbing contract)
    - arrival_aware_sparse_v1: gemm_f32, target_rps in {50,100}, duration_ms=5000,
      workers=1, counted_runs=1, instrumentation OFF
    - arrival_aware_sparse_instrumented_v1: gemm_f32, 100 RPS, duration_ms=5000,
      workers=1, counted_runs=1, schema 5, capacity 1024, instrumentation ON
    Instrumentation ON is rejected for E7 and for 50 RPS sparse. Capacity without
    timing and timing without capacity are rejected. G3 adaptive_early_seal_v1
    remains gemm_f32 A,G2,G3 at 1000 RPS / 1000 ms. Other neural-local delay
    calibrations keep nonzero_batch_delay_v1.
    """
    if mode != "calibration":
        return None
    cell_ids = tuple(str(cell["id"]) for cell in cells)
    if "G4" in cell_ids or cell_ids == ARRIVAL_AWARE_BATCHING_CELLS:
        if cell_ids != ARRIVAL_AWARE_BATCHING_CELLS:
            raise NeuralCollectorError(
                "G4 arrival-aware calibration requires cells A,G2,G3,G4 in that order; "
                f"got {','.join(cell_ids)!r}"
            )
        if model_id != REPORTABLE_MODEL_ID:
            raise NeuralCollectorError(
                f"G4 arrival-aware calibration requires model_id {REPORTABLE_MODEL_ID}"
            )
        rate = _require_finite_number(profile.get("target_rps"), "target_rps")
        _require_exact_int(measurement.get("worker_count", -1), "worker_count", 1)
        duration = measurement.get("duration_ms")
        if type(duration) is not int:
            raise NeuralCollectorError("duration_ms must be an integer")
        if type(counted_runs) is not int:
            raise NeuralCollectorError("counted_runs must be an integer")
        timing_on = measurement.get("enable_batch_decision_timing") is True
        capacity = measurement.get("batch_decision_event_capacity", 0)
        if capacity is None:
            capacity = 0
        if type(capacity) is not int or isinstance(capacity, bool):
            raise NeuralCollectorError("batch_decision_event_capacity must be an integer")
        if timing_on and capacity == 0:
            raise NeuralCollectorError(
                "enable_batch_decision_timing requires batch_decision_event_capacity"
            )
        if not timing_on and capacity != 0:
            raise NeuralCollectorError(
                "batch_decision_event_capacity requires enable_batch_decision_timing"
            )
        if (
            rate == ARRIVAL_AWARE_BATCHING_RATE
            and duration == ARRIVAL_AWARE_BATCHING_DURATION_MS
            and counted_runs == ARRIVAL_AWARE_BATCHING_COUNTED_RUNS
        ):
            if timing_on:
                raise NeuralCollectorError(
                    f"{CALIBRATION_VARIANT_ARRIVAL_AWARE_BATCHING_V1} does not support "
                    "batch-decision timing instrumentation"
                )
            for cell in cells:
                normalize_cell_definition(cell)
            return CALIBRATION_VARIANT_ARRIVAL_AWARE_BATCHING_V1
        if (
            rate in ARRIVAL_AWARE_SPARSE_RATES
            and duration == ARRIVAL_AWARE_SPARSE_DURATION_MS
            and counted_runs == ARRIVAL_AWARE_SPARSE_COUNTED_RUNS
        ):
            if timing_on:
                if rate != ARRIVAL_AWARE_SPARSE_INSTRUMENTED_RATE:
                    raise NeuralCollectorError(
                        f"{CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_INSTRUMENTED_V1} "
                        f"requires target_rps={ARRIVAL_AWARE_SPARSE_INSTRUMENTED_RATE:g}; "
                        f"got {rate:g}"
                    )
                if capacity != ARRIVAL_AWARE_SPARSE_INSTRUMENTED_CAPACITY:
                    raise NeuralCollectorError(
                        f"{CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_INSTRUMENTED_V1} "
                        f"requires batch_decision_event_capacity="
                        f"{ARRIVAL_AWARE_SPARSE_INSTRUMENTED_CAPACITY}"
                    )
                for cell in cells:
                    normalize_cell_definition(cell)
                return CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_INSTRUMENTED_V1
            for cell in cells:
                normalize_cell_definition(cell)
            return CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_V1
        raise NeuralCollectorError(
            "G4 arrival-aware calibration requires either "
            f"{CALIBRATION_VARIANT_ARRIVAL_AWARE_BATCHING_V1} "
            f"(target_rps={ARRIVAL_AWARE_BATCHING_RATE:g}, "
            f"duration_ms={ARRIVAL_AWARE_BATCHING_DURATION_MS}), "
            f"{CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_V1} "
            f"(target_rps in {{{', '.join(f'{item:g}' for item in sorted(ARRIVAL_AWARE_SPARSE_RATES))}}}, "
            f"duration_ms={ARRIVAL_AWARE_SPARSE_DURATION_MS}), or "
            f"{CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_INSTRUMENTED_V1} "
            f"(target_rps={ARRIVAL_AWARE_SPARSE_INSTRUMENTED_RATE:g}, "
            f"duration_ms={ARRIVAL_AWARE_SPARSE_INSTRUMENTED_DURATION_MS}, "
            f"capacity={ARRIVAL_AWARE_SPARSE_INSTRUMENTED_CAPACITY}); "
            f"got target_rps={rate:g}, duration_ms={duration}, counted_runs={counted_runs}, "
            f"instrumentation={'on' if timing_on else 'off'}"
        )
    if "G3" in cell_ids or cell_ids == ADAPTIVE_EARLY_SEAL_CELLS:
        if measurement.get("enable_batch_decision_timing") is True:
            raise NeuralCollectorError(
                "batch-decision timing is only supported for "
                f"{CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_INSTRUMENTED_V1}"
            )
        if cell_ids != ADAPTIVE_EARLY_SEAL_CELLS:
            raise NeuralCollectorError(
                "adaptive_early_seal_v1 requires cells A,G2,G3 in that order; "
                f"got {','.join(cell_ids)!r}"
            )
        if model_id != REPORTABLE_MODEL_ID:
            raise NeuralCollectorError(
                f"adaptive_early_seal_v1 requires model_id {REPORTABLE_MODEL_ID}"
            )
        rate = profile.get("target_rps")
        if not isinstance(rate, (int, float)) or float(rate) != ADAPTIVE_EARLY_SEAL_RATE:
            raise NeuralCollectorError(
                f"adaptive_early_seal_v1 requires target_rps {ADAPTIVE_EARLY_SEAL_RATE:g}"
            )
        if int(measurement.get("worker_count", -1)) != 1:
            raise NeuralCollectorError("adaptive_early_seal_v1 requires worker_count=1")
        if int(measurement.get("duration_ms", -1)) != 1000:
            raise NeuralCollectorError("adaptive_early_seal_v1 requires duration_ms=1000")
        if counted_runs != ADAPTIVE_EARLY_SEAL_COUNTED_RUNS:
            raise NeuralCollectorError(
                f"adaptive_early_seal_v1 requires counted_runs="
                f"{ADAPTIVE_EARLY_SEAL_COUNTED_RUNS}"
            )
        for cell in cells:
            normalize_cell_definition(cell)
        return CALIBRATION_VARIANT_ADAPTIVE_EARLY_SEAL_V1
    if measurement.get("enable_batch_decision_timing") is True:
        raise NeuralCollectorError(
            "batch-decision timing is only supported for "
            f"{CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_INSTRUMENTED_V1}"
        )
    if (
        type(measurement.get("batch_decision_event_capacity", 0)) is int
        and not isinstance(measurement.get("batch_decision_event_capacity", 0), bool)
        and int(measurement.get("batch_decision_event_capacity", 0)) != 0
    ):
        raise NeuralCollectorError(
            "batch_decision_event_capacity requires enable_batch_decision_timing"
        )
    uses_nonzero_delay = any(str(cell["id"]) in CALIBRATION_ONLY_NEURAL_CELLS for cell in cells)
    if uses_nonzero_delay:
        return CALIBRATION_VARIANT_NONZERO_BATCH_DELAY_V1
    return None


def realized_mean_batch_size(document: dict) -> tuple[float | None, list[str]]:
    problems: list[str] = []
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        return None, ["metrics object is missing"]
    histogram = metrics.get("batch_size_histogram")
    if not isinstance(histogram, list) or not histogram:
        return None, ["batch_size_histogram is missing"]
    if histogram[0] != 0:
        problems.append("batch_size_histogram bin 0 must be zero")
    sealed = 0
    weighted = 0
    for size, count in enumerate(histogram):
        if type(count) is not int or count < 0:
            problems.append(f"batch_size_histogram[{size}] is invalid")
            continue
        if size == 0:
            continue
        sealed += count
        weighted += size * count
    runs = metrics.get("engine_run_count")
    entered = metrics.get("requests_entered_engine_runs")
    if type(runs) is not int or runs <= 0:
        problems.append("engine_run_count is missing or zero")
    elif sealed != runs:
        problems.append("batch histogram sealed count does not match engine_run_count")
    if type(entered) is not int or entered < 0:
        problems.append("requests_entered_engine_runs is missing")
    elif weighted != entered:
        problems.append("batch histogram weighted sum does not match requests_entered_engine_runs")
    if sealed <= 0:
        problems.append("no sealed batches are recorded")
        return None, problems
    return float(weighted) / float(sealed), problems


def multi_request_batch_fraction(document: dict) -> tuple[float | None, list[str]]:
    """Fraction of requests processed in sealed batches of size >= 2.

    Diagnostic only. Does not certify early-seal events.
    """
    mean, problems = realized_mean_batch_size(document)
    if problems and mean is None:
        return None, problems
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        return None, ["metrics object is missing"]
    histogram = metrics.get("batch_size_histogram")
    entered = metrics.get("requests_entered_engine_runs")
    if type(entered) is not int or entered <= 0:
        return None, ["requests_entered_engine_runs is missing or zero"]
    multi = 0
    for size, count in enumerate(histogram):
        if size >= 2 and type(count) is int and count > 0:
            multi += size * count
    return float(multi) / float(entered), problems


def multi_size_batch_fraction(document: dict) -> tuple[float | None, list[str]]:
    """Fraction of sealed batches with size >= 2.

    Diagnostic only. Does not certify early-seal events.
    """
    mean, problems = realized_mean_batch_size(document)
    if problems and mean is None:
        return None, problems
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        return None, ["metrics object is missing"]
    histogram = metrics.get("batch_size_histogram")
    if not isinstance(histogram, list) or not histogram:
        return None, ["batch_size_histogram is missing"]
    sealed = 0
    multi = 0
    for size, count in enumerate(histogram):
        if size == 0:
            continue
        if type(count) is not int or count < 0:
            return None, [f"batch_size_histogram[{size}] is invalid"]
        sealed += count
        if size >= 2:
            multi += count
    if sealed <= 0:
        return None, ["no sealed batches are recorded"]
    return float(multi) / float(sealed), problems


def arrival_screen_problems(document: dict, max_lag_ns: int) -> list[str]:
    lag = document.get("generator_lag")
    if not isinstance(lag, dict):
        return ["generator_lag is missing for the arrival screen"]
    maximum = lag.get("maximum_nanoseconds")
    if type(maximum) is not int:
        return ["generator_lag.maximum_nanoseconds is missing or not an integer"]
    if maximum > max_lag_ns:
        return [
            f"generator_lag.maximum_nanoseconds is {maximum}, "
            f"exceeds arrival screen {max_lag_ns} (inclusive)"
        ]
    return []


def coarse_max_lag_v1(document: dict, max_lag_ns: int = ARRIVAL_SCREEN_MAX_LAG_NS) -> bool | None:
    """Derived non-gating coarse lag screen.

    True when generator_lag.maximum_nanoseconds <= max_lag_ns. None when the
    lag summary is unavailable. Not an arrival-fidelity certificate.
    """
    lag = document.get("generator_lag")
    if not isinstance(lag, dict):
        return None
    maximum = lag.get("maximum_nanoseconds")
    if type(maximum) is not int:
        return None
    return maximum <= int(max_lag_ns)


def counted_rotation_position_counts(plan: list[dict]) -> dict[str, dict[int, int]]:
    """Count position_in_block occurrences per cell for counted trials only."""
    counts: dict[str, dict[int, int]] = {}
    for trial in plan:
        if trial.get("pilot") is True:
            continue
        cell_id = str(trial["cell_id"])
        position = trial.get("position_in_block")
        if type(position) is not int:
            raise NeuralCollectorError(
                f"{trial.get('trial_id')} counted trial is missing position_in_block"
            )
        cell_counts = counts.setdefault(cell_id, {})
        cell_counts[position] = cell_counts.get(position, 0) + 1
    return counts


def require_rotation_position_counts(
    plan: list[dict], expected: dict[str, dict[int, int]]
) -> dict[str, dict[int, int]]:
    actual = counted_rotation_position_counts(plan)
    if actual != expected:
        raise NeuralCollectorError(
            "rotation position counts diverge from the frozen three-arm contract: "
            f"actual={actual!r}, expected={expected!r}"
        )
    return actual


def unbatched_size_one_problems(document: dict) -> list[str]:
    """Require sealed histogram accounting consistent with unbatched size-1 runs."""
    mean, problems = realized_mean_batch_size(document)
    if mean is None:
        return problems or ["mean realized batch size could not be computed"]
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        return ["metrics object is missing"]
    histogram = metrics.get("batch_size_histogram")
    if not isinstance(histogram, list):
        return ["batch_size_histogram is missing"]
    for size, count in enumerate(histogram):
        if size == 1:
            continue
        if type(count) is int and count != 0:
            problems.append(
                f"unbatched cell has sealed batches of size {size} (count {count})"
            )
    if abs(float(mean) - 1.0) > 1e-12:
        problems.append(f"unbatched mean realized batch size is {mean}, expected 1.0")
    return problems


def resolve_reportable_contrast(
    *,
    model_id: str,
    cell_ids: list[str],
    baseline_cell: str | None,
    alternative_cell: str | None,
    target_rps: float | None,
    counted_runs: int | None,
    offered_rate_frozen: bool,
) -> dict:
    if model_id != REPORTABLE_MODEL_ID:
        raise NeuralCollectorError(
            f"reportable mode supports only {REPORTABLE_MODEL_ID}, got {model_id!r}"
        )
    if not offered_rate_frozen:
        raise NeuralCollectorError(
            "reportable mode requires --offered-rate-frozen after feasibility calibration"
        )
    if target_rps is None:
        raise NeuralCollectorError(
            "reportable mode requires an explicit --target-rps; refusing to invent a rate"
        )
    cells = tuple(cell_ids)
    if cells == REPORTABLE_CELLS:
        baseline = baseline_cell or REPORTABLE_BASELINE
        alternative = alternative_cell or REPORTABLE_ALTERNATIVE
        if baseline != REPORTABLE_BASELINE or alternative != REPORTABLE_ALTERNATIVE:
            raise NeuralCollectorError("A/C reportable contrast requires baseline A and alternative C")
        if counted_runs is not None and counted_runs < MINIMUM_INTERVAL_BLOCKS:
            raise NeuralCollectorError(
                f"reportable A/C needs at least {MINIMUM_INTERVAL_BLOCKS} counted blocks"
            )
        contract = dict(REPORTABLE_CONTRAST_AC)
        contract["resolved_offered_rate"] = float(target_rps)
        return contract
    if cells == REPORTABLE_CONTRAST_AG2["cells"]:
        baseline = baseline_cell or "A"
        alternative = alternative_cell or "G2"
        if baseline != "A" or alternative != "G2":
            raise NeuralCollectorError(
                "A/G2 reportable contrast requires baseline A and alternative G2"
            )
        if float(target_rps) != AG2_OFFERED_RATE:
            raise NeuralCollectorError(
                f"A/G2 reportable contrast requires --target-rps {AG2_OFFERED_RATE:g}"
            )
        if counted_runs is None:
            counted_runs = PREFERRED_COUNTED_RUNS
        if counted_runs != PREFERRED_COUNTED_RUNS:
            raise NeuralCollectorError(
                f"A/G2 reportable contrast requires exactly {PREFERRED_COUNTED_RUNS} counted blocks"
            )
        contract = dict(REPORTABLE_CONTRAST_AG2)
        contract["resolved_offered_rate"] = AG2_OFFERED_RATE
        return contract
    if cells == REPORTABLE_CONTRAST_AG2G3["cells"]:
        baseline = baseline_cell or "G2"
        alternative = alternative_cell or "G3"
        if baseline != "G2" or alternative != "G3":
            raise NeuralCollectorError(
                "A/G2/G3 reportable contrast requires primary baseline G2 and alternative G3"
            )
        if float(target_rps) != AG2G3_OFFERED_RATE:
            raise NeuralCollectorError(
                f"A/G2/G3 reportable contrast requires --target-rps {AG2G3_OFFERED_RATE:g}"
            )
        if counted_runs is None:
            counted_runs = THREE_ARM_COUNTED_BLOCKS
        if counted_runs != THREE_ARM_COUNTED_BLOCKS:
            raise NeuralCollectorError(
                f"A/G2/G3 reportable contrast requires exactly "
                f"{THREE_ARM_COUNTED_BLOCKS} counted blocks"
            )
        contract = dict(REPORTABLE_CONTRAST_AG2G3)
        contract["resolved_offered_rate"] = AG2G3_OFFERED_RATE
        return contract
    raise NeuralCollectorError(
        "reportable mode supports only cells A,C, the frozen A,G2 contrast, "
        f"or the frozen A,G2,G3 contrast; got {','.join(cell_ids)!r}"
    )


def evidence_class_for(mode: str) -> str:
    if mode == "smoke":
        return "correctness"
    if mode == "calibration":
        return "calibration"
    if mode == "reportable":
        return "confirmatory"
    raise NeuralCollectorError(f"unsupported collection mode {mode!r}")


def counted_runs_for(mode: str, counted_runs: int | None) -> int:
    if mode == "smoke":
        if counted_runs not in {None, 0}:
            raise NeuralCollectorError("smoke uses only the uncounted pilot pair")
        return 0
    if counted_runs is None:
        if mode == "reportable":
            return PREFERRED_COUNTED_RUNS
        raise NeuralCollectorError("calibration requires --counted-runs")
    if type(counted_runs) is not int or counted_runs < 1 or counted_runs > MAX_COUNTED_RUNS:
        raise NeuralCollectorError("counted runs must be an integer from 1 through 30")
    if mode == "reportable" and counted_runs < MINIMUM_INTERVAL_BLOCKS:
        raise NeuralCollectorError(
            f"reportable mode needs at least {MINIMUM_INTERVAL_BLOCKS} counted blocks"
        )
    return counted_runs


def open_loop_profile(profile_id: str, target_rps: float) -> dict:
    if not math.isfinite(target_rps) or target_rps <= 0:
        raise NeuralCollectorError("target_rps must be a positive finite number")
    return {
        "id": profile_id,
        "workload": "open_loop_constant",
        "family": "open_loop",
        "target_rps": float(target_rps),
        "label": "neural open-loop constant; rate is explicit metadata, not a capacity claim",
    }


def neural_scheduler_command(
    binary: pathlib.Path,
    model: pathlib.Path,
    model_id: str,
    cell: dict,
    profile: dict,
    measurement: dict,
) -> list[str]:
    command = scheduler_command(binary, model, cell, profile, measurement)
    if "--model-id" in command:
        raise NeuralCollectorError("Add scheduler_command unexpectedly emitted --model-id")
    model_index = command.index("--model")
    command[model_index:model_index] = ["--model-id", model_id]
    adaptive = adaptive_batching_for_cell(cell)
    arrival = arrival_aware_batching_for_cell(cell)
    if adaptive and arrival:
        raise NeuralCollectorError(
            "adaptive_batching and arrival_aware_batching are mutually exclusive"
        )
    if adaptive:
        command.append("--adaptive-batching")
    if arrival:
        command.append("--arrival-aware-batching")
    if measurement.get("enable_batch_decision_timing") is True:
        capacity = measurement.get("batch_decision_event_capacity")
        if type(capacity) is not int or isinstance(capacity, bool) or capacity < 1:
            raise NeuralCollectorError(
                "batch_decision_event_capacity must be a positive integer when "
                "enable_batch_decision_timing is true"
            )
        command.append("--batch-decision-timing")
        command.extend(["--batch-decision-event-capacity", str(capacity)])
    return command


def file_digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_neural_artifacts(repo_root: pathlib.Path, model_id: str) -> dict:
    contract = model_contract(model_id)
    model_path = repo_root / contract["model_relative"]
    sidecar_path = repo_root / contract["sidecar_relative"]
    if not model_path.is_file():
        raise NeuralCollectorError(f"neural model is missing: {model_path}")
    if not sidecar_path.is_file():
        raise NeuralCollectorError(f"weight sidecar is missing: {sidecar_path}")
    model_digest = file_digest(model_path)
    sidecar_digest = file_digest(sidecar_path)
    if model_digest != contract["model_sha256"]:
        raise NeuralCollectorError(
            f"ONNX SHA-256 mismatch for {model_id}: expected {contract['model_sha256']}, "
            f"actual {model_digest}"
        )
    if sidecar_digest != contract["sidecar_sha256"]:
        raise NeuralCollectorError(
            f"sidecar SHA-256 mismatch for {model_id}: expected {contract['sidecar_sha256']}, "
            f"actual {sidecar_digest}"
        )
    return {
        "model_id": model_id,
        "model_path": model_path.resolve(),
        "model_relative": contract["model_relative"],
        "model_sha256": model_digest,
        "weight_sidecar_path": sidecar_path.resolve(),
        "weight_sidecar_relative": contract["sidecar_relative"],
        "weight_sidecar_sha256": sidecar_digest,
        "input_name": contract["input_name"],
        "output_name": contract["output_name"],
        "input_width": contract["input_width"],
        "output_width": contract["output_width"],
        "reference_checking_policy": REFERENCE_CHECKING_POLICY,
    }


def build_plan(cells: list[dict], profiles: list[dict], counted_runs: int) -> list[dict]:
    if counted_runs == 0:
        plan = []
        for cell in cells:
            for profile in profiles:
                plan.append(
                    {
                        "trial_id": f"pilot-{cell['id']}-{profile['id']}",
                        "cell_id": cell["id"],
                        "profile_id": profile["id"],
                        "pilot": True,
                        "run_number": None,
                        "block_index": None,
                        "position_in_block": None,
                        "configuration_id": cell["id"],
                    }
                )
        for index, trial in enumerate(plan):
            trial["execution_index"] = index
            trial["counted_run_target"] = 0
        return plan
    return execution_plan(cells, profiles, counted_runs, ANALYSIS_PROTOCOL_VERSION)


def bind_commands(
    plan: list[dict],
    *,
    binary: pathlib.Path,
    model: pathlib.Path,
    model_id: str,
    cells: dict[str, dict],
    profiles: dict[str, dict],
    measurement: dict,
) -> list[dict]:
    bound = []
    for trial in plan:
        cell = cells[str(trial["cell_id"])]
        profile = profiles[str(trial["profile_id"])]
        command = neural_scheduler_command(binary, model, model_id, cell, profile, measurement)
        bound.append({**trial, "command": command})
    return bound


def rejection_total(document: dict) -> int:
    rejections = document.get("rejections")
    if not isinstance(rejections, dict):
        return -1
    total = 0
    for key in ("full", "closed", "not_running"):
        value = rejections.get(key)
        if type(value) is not int or value < 0:
            return -1
        total += value
    return total


def nearest_rank_percentile(sorted_values: list[int], percentile: float) -> int | None:
    """Nearest-rank percentile with 1-based rank = ceil(p/100 * n)."""
    count = len(sorted_values)
    if count == 0:
        return None
    if not math.isfinite(percentile) or percentile <= 0 or percentile > 100:
        raise NeuralCollectorError("percentile must be in (0, 100]")
    rank = int(math.ceil(percentile / 100.0 * count))
    rank = max(1, min(count, rank))
    return sorted_values[rank - 1]


def batch_open_wait_diagnostics(events: list[dict]) -> dict:
    """Diagnostic open-wait summary. Not confirmatory inference."""
    if not events:
        return {
            "status": "zero_event",
            "count": 0,
            "unit": "nanoseconds",
            "seal_reason_frequencies": {},
        }
    waits: list[int] = []
    frequencies: dict[str, int] = {}
    for event in events:
        open_ns = event["batch_open_ns"]
        seal_ns = event["batch_seal_ns"]
        waits.append(int(seal_ns) - int(open_ns))
        reason = str(event["seal_reason"])
        frequencies[reason] = frequencies.get(reason, 0) + 1
    ordered = sorted(waits)
    return {
        "status": "complete",
        "count": len(ordered),
        "unit": "nanoseconds",
        "minimum_nanoseconds": ordered[0],
        "median_nanoseconds": nearest_rank_percentile(ordered, 50),
        "p95_nanoseconds": nearest_rank_percentile(ordered, 95),
        "maximum_nanoseconds": ordered[-1],
        "mean_nanoseconds": float(sum(ordered)) / float(len(ordered)),
        "seal_reason_frequencies": frequencies,
    }


def instrumented_trace_problems(document: dict, cell: dict, expected: dict) -> list[str]:
    """Neural-local strict schema-5 accounting and policy seal-reason gates."""
    problems: list[str] = []
    snap = document.get("batch_decision")
    if not isinstance(snap, dict):
        return ["batch_decision object is missing"]
    if snap.get("events_dropped") != 0:
        problems.append("instrumented calibration forbids dropped batch-decision events")
    if snap.get("trace_overflow") is not False:
        problems.append("instrumented calibration forbids batch-decision trace overflow")
    if snap.get("timing_data_valid") is not True:
        problems.append("instrumented calibration requires timing_data_valid=true")
    if snap.get("buffer_capacity_per_worker") != ARRIVAL_AWARE_SPARSE_INSTRUMENTED_CAPACITY:
        problems.append(
            "instrumented calibration requires buffer_capacity_per_worker="
            f"{ARRIVAL_AWARE_SPARSE_INSTRUMENTED_CAPACITY}"
        )
    events = snap.get("events")
    if not isinstance(events, list):
        problems.append("batch_decision.events is not a list")
        return problems
    cell_id = str(cell.get("id"))
    enable_batching = bool(cell.get("enable_batching"))
    sealed = snap.get("sealed_batches_total")
    if not enable_batching:
        if sealed != 0 or events or snap.get("events_recorded") != 0:
            problems.append("unbatched cell A must emit a zero-event batch-decision trace")
        return problems
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        problems.append("metrics object is missing")
        return problems
    engine_runs = metrics.get("engine_run_count")
    if type(engine_runs) is not int or engine_runs < 0:
        problems.append("engine_run_count is missing")
    elif sealed != engine_runs:
        problems.append(
            "sealed_batches_total must equal metrics.engine_run_count for batching cells"
        )
    mean, mean_problems = realized_mean_batch_size(document)
    if mean_problems and mean is None:
        problems.extend(mean_problems)
    forbidden: frozenset[str]
    if cell_id == "G2":
        forbidden = SEAL_REASONS_FORBIDDEN_G2
    elif cell_id == "G3":
        forbidden = SEAL_REASONS_FORBIDDEN_G3
    elif cell_id == "G4":
        forbidden = SEAL_REASONS_FORBIDDEN_G4
    else:
        forbidden = frozenset()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        reason = event.get("seal_reason")
        if type(reason) is str and reason in forbidden:
            problems.append(
                f"batch_decision.events[{index}].seal_reason {reason!r} is forbidden for cell {cell_id}"
            )
    del expected
    return problems


def validate_trial(
    text: str,
    *,
    expected: dict,
    mode: str,
    require_zero_rejections: bool,
    arrival_screen_max_lag_ns: int | None = None,
    require_persistent_batching: bool = False,
    persistent_batching_min_mean: float | None = None,
    require_unbatched_size_one: bool = False,
    require_strict_generator_delivery: bool | None = None,
    cell: dict | None = None,
) -> tuple[dict | None, dict, list[str]]:
    document, status = classify_report(text, expected=expected)
    problems: list[str] = []
    if status.get("collector_valid") is not True:
        problems.extend(str(item) for item in status.get("problems", []))
        if not problems:
            problems.append("collector rejected the trial")
    if document is None:
        return None, status, problems
    want_schema = expected_schema_version(expected)
    if document.get("schema_version") != want_schema:
        problems.append(
            f"schema_version is {document.get('schema_version')!r}, expected {want_schema}"
        )
    if (
        want_schema == BATCH_DECISION_SCHEMA_VERSION
        and cell is not None
        and expected.get("enable_batch_decision_timing") is True
    ):
        problems.extend(instrumented_trace_problems(document, cell, expected))
    if document.get("valid") is not True:
        problems.append("runner valid flag is not true")
    if document.get("correctness_mismatches") != 0:
        problems.append("correctness mismatches are not zero")
    if type(document.get("correctness_checked")) is not int or document["correctness_checked"] < 1:
        if mode == "reportable":
            problems.append("correctness_checked is missing or zero")
    if document.get("failed_inference") not in {0, None} and document.get("failed_inference") != 0:
        problems.append("failed_inference is not zero")
    if document.get("failed_worker") != 0:
        problems.append("failed_worker is not zero")
    if document.get("failed_scheduler") != 0:
        problems.append("failed_scheduler is not zero")
    if document.get("canceled") != 0:
        problems.append("canceled is not zero")
    if document.get("timeouts") != 0:
        problems.append("timeouts are not zero")
    metrics = document.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("finalized") is not True:
        problems.append("metrics are not finalized")
    if isinstance(metrics, dict) and metrics.get("histogram_saturated") is True:
        problems.append("histogram is saturated")
    strict_delivery = (
        mode in {"reportable", "smoke"}
        if require_strict_generator_delivery is None
        else bool(require_strict_generator_delivery)
    )
    if strict_delivery and status.get("delivery_class") != "STRICT_PRIMARY":
        problems.append(
            f"delivery_class is {status.get('delivery_class')!r}, expected STRICT_PRIMARY"
        )
    if require_zero_rejections:
        total = rejection_total(document)
        if total != 0:
            problems.append(f"scheduler admission rejections are {total}, expected 0")
        if document.get("accepted") != document.get("submit_attempts"):
            problems.append("accepted does not equal submit_attempts")
        if document.get("succeeded") != document.get("accepted"):
            problems.append("succeeded does not equal accepted")
        if type(document.get("scheduled_opportunities")) is not int or document[
            "scheduled_opportunities"
        ] <= 0:
            problems.append("scheduled_opportunities must be positive")
        if document.get("submit_attempts") != document.get("scheduled_opportunities"):
            problems.append("submit_attempts does not equal scheduled_opportunities")
        if document.get("generator_skips_outstanding") not in {0, None} and document.get(
            "generator_skips_outstanding"
        ) != 0:
            problems.append("generator_skips_outstanding is not zero")
        if document.get("generator_skips_lag") not in {0, None} and document.get(
            "generator_skips_lag"
        ) != 0:
            problems.append("generator_skips_lag is not zero")
    if arrival_screen_max_lag_ns is not None:
        problems.extend(arrival_screen_problems(document, int(arrival_screen_max_lag_ns)))
    if require_persistent_batching:
        minimum = (
            PERSISTENT_BATCHING_MIN_MEAN
            if persistent_batching_min_mean is None
            else float(persistent_batching_min_mean)
        )
        mean, histogram_problems = realized_mean_batch_size(document)
        problems.extend(histogram_problems)
        if mean is None:
            problems.append("mean realized batch size could not be computed")
        elif mean < minimum:
            problems.append(
                f"mean realized batch size is {mean}, below persistent-batching gate {minimum}"
            )
    if require_unbatched_size_one:
        problems.extend(unbatched_size_one_problems(document))
    return document, status, problems


def adaptive_identity_consistency_problems(
    *,
    cell: dict,
    command: list[str],
    expected: dict,
    document: dict,
) -> list[str]:
    """Require cell, command, expected launch, and runner JSON policy identity agree."""
    problems: list[str] = []
    try:
        configured_adaptive = adaptive_batching_for_cell(cell)
        configured_arrival = arrival_aware_batching_for_cell(cell)
    except NeuralCollectorError as exc:
        return [str(exc)]
    if configured_adaptive and configured_arrival:
        problems.append("adaptive_batching and arrival_aware_batching are mutually exclusive")
    adaptive_flag = "--adaptive-batching" in command
    arrival_flag = "--arrival-aware-batching" in command
    if adaptive_flag is not configured_adaptive:
        problems.append(
            "command adaptive flag "
            f"{'present' if adaptive_flag else 'absent'} disagrees with cell "
            f"adaptive_batching={configured_adaptive}"
        )
    if arrival_flag is not configured_arrival:
        problems.append(
            "command arrival-aware flag "
            f"{'present' if arrival_flag else 'absent'} disagrees with cell "
            f"arrival_aware_batching={configured_arrival}"
        )
    expected_adaptive = expected.get("adaptive_batching")
    if type(expected_adaptive) is not bool or expected_adaptive is not configured_adaptive:
        problems.append(
            f"expected launch adaptive_batching is {expected_adaptive!r}, "
            f"disagrees with cell adaptive_batching={configured_adaptive}"
        )
    expected_arrival = expected.get("arrival_aware_batching")
    if type(expected_arrival) is not bool or expected_arrival is not configured_arrival:
        problems.append(
            f"expected launch arrival_aware_batching is {expected_arrival!r}, "
            f"disagrees with cell arrival_aware_batching={configured_arrival}"
        )
    scheduler = document.get("scheduler")
    if not isinstance(scheduler, dict):
        problems.append("scheduler object is missing for adaptive identity")
        return problems
    if CELL_DEFINITION_ADAPTIVE_KEY not in scheduler:
        problems.append("scheduler.adaptive_batching is missing")
        return problems
    reported = scheduler[CELL_DEFINITION_ADAPTIVE_KEY]
    if type(reported) is not bool:
        problems.append("scheduler.adaptive_batching is not a JSON boolean")
    elif reported is not configured_adaptive:
        problems.append(
            f"scheduler.adaptive_batching is {reported}, expected {configured_adaptive}"
        )
    if CELL_DEFINITION_ARRIVAL_KEY not in scheduler:
        problems.append("scheduler.arrival_aware_batching is missing")
    else:
        reported_arrival = scheduler[CELL_DEFINITION_ARRIVAL_KEY]
        if type(reported_arrival) is not bool:
            problems.append("scheduler.arrival_aware_batching is not a JSON boolean")
        elif reported_arrival is not configured_arrival:
            problems.append(
                f"scheduler.arrival_aware_batching is {reported_arrival}, "
                f"expected {configured_arrival}"
            )
    return problems


def _write_failure(run_set: pathlib.Path, payload: dict) -> None:
    path = run_set / "failure.json"
    if path.exists():
        return
    write_json(path, payload)


def neural_preflight(
    *,
    repo_root: pathlib.Path,
    binary: pathlib.Path,
    expected_commit: str,
    model_id: str,
    cells: list[dict],
    profiles: list[dict],
    measurement: dict,
    counted_runs: int,
    mode: str,
    offered_rate_frozen: bool,
    delay_seconds: float,
    baseline_cell: str,
    alternative_cell: str,
    runner,
    reportable_contrast: dict | None = None,
) -> dict:
    commit = require_commit_sha(expected_commit)
    if counted_runs > 0:
        require_counted_runs(counted_runs)
    family = require_one_family(profiles)
    if family != "open_loop":
        raise NeuralCollectorError("neural collector supports only open_loop profiles")
    cells = [normalize_cell_definition(cell) for cell in cells]
    profile = profiles[0]
    calibration_variant = resolve_calibration_variant(
        mode=mode,
        cells=cells,
        model_id=model_id,
        profile=profile,
        measurement=measurement,
        counted_runs=counted_runs,
    )
    include_arrival_aware_definitions = calibration_variant in {
        CALIBRATION_VARIANT_ARRIVAL_AWARE_BATCHING_V1,
        CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_V1,
        CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_INSTRUMENTED_V1,
    }
    include_adaptive_definitions = (
        calibration_variant == CALIBRATION_VARIANT_ADAPTIVE_EARLY_SEAL_V1
        or include_arrival_aware_definitions
    )
    if reportable_contrast is not None and reportable_contrast.get(
        "require_five_key_cell_definitions"
    ):
        include_adaptive_definitions = True
    cell_definitions = serialize_cell_definitions(
        cells,
        include_adaptive=include_adaptive_definitions,
        include_arrival_aware=include_arrival_aware_definitions,
    )
    if reportable_contrast is not None:
        expected_cells = reportable_contrast.get("expected_cells")
        if isinstance(expected_cells, dict):
            for cell in cells:
                expected = expected_cells.get(cell["id"])
                if expected is None:
                    raise NeuralCollectorError(
                        f"reportable contrast does not allow cell {cell['id']!r}"
                    )
                if cell != normalize_cell_definition(expected):
                    raise NeuralCollectorError(
                        f"cell {cell['id']!r} does not match the frozen contrast definition"
                    )
        if baseline_cell != reportable_contrast["baseline_cell"]:
            raise NeuralCollectorError("baseline cell does not match the frozen contrast")
        if alternative_cell != reportable_contrast["alternative_cell"]:
            raise NeuralCollectorError("alternative cell does not match the frozen contrast")
        if [cell["id"] for cell in cells] != list(reportable_contrast["cells"]):
            raise NeuralCollectorError("cell order does not match the frozen contrast")
    contrast = resolve_contrast(cells, baseline_cell, alternative_cell)
    if not binary.is_file():
        raise NeuralCollectorError(f"benchmark binary does not exist: {binary}")
    if not os.access(binary, os.X_OK):
        raise NeuralCollectorError(f"benchmark binary is not executable: {binary}")
    if binary.resolve().name != BINARY_NAME:
        raise NeuralCollectorError(
            f"benchmark binary must be named {BINARY_NAME}, got {binary.resolve().name}"
        )
    build_dir = locate_build_dir(binary)
    cache = parse_cmake_cache((build_dir / "CMakeCache.txt").read_text(encoding="utf-8"))
    problems = assess_build_cache(cache)
    problems.extend(release_optimization_problems(cache))
    if problems:
        raise NeuralCollectorError("; ".join(problems))
    symbols = query_text(runner, symbol_tool_args(binary))
    if symbols.returncode != 0:
        raise NeuralCollectorError(
            "could not inspect the benchmark binary for sanitizer symbols: "
            f"{symbols.stderr.strip() or symbols.stdout.strip()}"
        )
    symbol_problems = sanitizer_symbol_problems(symbols.stdout + symbols.stderr)
    if symbol_problems:
        raise NeuralCollectorError("; ".join(symbol_problems))
    runtime_root = locate_pinned_runtime(cache, build_dir)
    linked = query_text(runner, link_tool_args(binary))
    if linked.returncode != 0 and not linked.stdout.strip():
        raise NeuralCollectorError(
            "could not identify the ONNX Runtime library linked into the benchmark binary: "
            f"{linked.stderr.strip() or linked.stdout.strip()}"
        )
    load_command_output = ""
    if platform.system() == "Darwin":
        load_commands = query_text(runner, ["otool", "-l", str(binary)])
        if load_commands.returncode != 0:
            raise NeuralCollectorError(
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
    if onnx_version != ONNXRUNTIME_VERSION:
        raise NeuralCollectorError(
            f"ONNX Runtime VERSION_NUMBER is {onnx_version!r}, expected {ONNXRUNTIME_VERSION!r}"
        )
    version = query_text(runner, [str(binary), "--version"])
    version_problems = version_text_problems(version.stdout, version.stderr, version.returncode)
    if version_problems:
        raise NeuralCollectorError("; ".join(version_problems))
    artifacts = verify_neural_artifacts(repo_root, model_id)
    head = query_text(runner, ["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    status = query_text(runner, ["git", "-C", str(repo_root), "status", "--porcelain"])
    if head.returncode != 0 or status.returncode != 0:
        raise NeuralCollectorError(
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
        raise NeuralCollectorError("; ".join(git_problems))
    host = hardware_metadata(runner)
    tools = compiler_and_cmake(runner, cache["CMAKE_CXX_COMPILER"])
    plan = build_plan(cells, profiles, counted_runs)
    pilots = [str(trial["trial_id"]) for trial in plan if trial["pilot"]]
    rotation_position_counts = None
    if reportable_contrast is not None:
        sign = str(reportable_contrast["contrast_sign_convention"])
        contrast_id = str(reportable_contrast["contrast_id"])
        arrival_status = reportable_contrast.get("arrival_fidelity_status")
        phase_boundary = reportable_contrast.get("phase_boundary_fidelity")
        arrival_screen = reportable_contrast.get("arrival_screen_max_lag_ns")
        arrival_screen_version = reportable_contrast.get("arrival_screen_policy_version")
        persistent_cell = reportable_contrast.get("persistent_batching_cell")
        persistent_min = reportable_contrast.get("persistent_batching_min_mean")
        expected_positions = reportable_contrast.get("expected_position_counts")
        if isinstance(expected_positions, dict):
            rotation_position_counts = require_rotation_position_counts(plan, expected_positions)
    else:
        sign = CONTRAST_SIGN_CONVENTION
        contrast_id = None
        arrival_status = None
        phase_boundary = None
        arrival_screen = None
        arrival_screen_version = None
        persistent_cell = None
        persistent_min = None
    contrast_identity: dict = {}
    if contrast_id is not None:
        contrast_identity["contrast_id"] = contrast_id
    if reportable_contrast is not None:
        for key in (
            "secondary_baseline_cell",
            "secondary_alternative_cell",
            "secondary_contrast_sign_convention",
            "secondary_comparison_class",
            "secondary_multiple_comparisons_adjustment",
            "unbatched_cell",
            "legacy_adaptive_identity_prohibited",
            "require_five_key_cell_definitions",
        ):
            if key in reportable_contrast and reportable_contrast[key] is not None:
                contrast_identity[key] = reportable_contrast[key]
    if arrival_status is not None:
        contrast_identity["arrival_fidelity_status"] = arrival_status
        contrast_identity["phase_boundary_fidelity"] = phase_boundary
        contrast_identity["arrival_screen_max_lag_ns"] = arrival_screen
        contrast_identity["arrival_screen_policy_version"] = arrival_screen_version
        contrast_identity["arrival_screen_note"] = (
            "Coarse generator-lag screen only; not precise arrival-fidelity certification. "
            "Inclusive bound on generator_lag.maximum_nanoseconds."
        )
    if persistent_cell is not None:
        contrast_identity["persistent_batching_cell"] = persistent_cell
        contrast_identity["persistent_batching_min_mean"] = persistent_min
        contrast_identity["persistent_batching_note"] = (
            "Mean realized batch size from sealed histograms; never the configured max. "
            "G3 mean batch size remains diagnostic under the three-arm contrast."
        )
    if rotation_position_counts is not None:
        contrast_identity["rotation_position_counts"] = {
            cell_id: {str(position): count for position, count in sorted(positions.items())}
            for cell_id, positions in sorted(rotation_position_counts.items())
        }
        contrast_identity["rotation_balance_note"] = (
            "Twenty blocks cannot provide perfect three-position balance; "
            "recorded counts are enforced and are not claimed to be exact balance."
        )
    return {
        "queried": {
            "queried_at_utc": utc_now(),
            "git_commit": actual_commit,
            "compiler_path": tools["compiler_path"],
            "compiler_version": tools["compiler_version"],
            "cmake_version": tools["cmake_version"],
            "cmake_flag_text": cache_flag_text(cache),
            "benchmark_binary_path": str(binary.resolve()),
            "benchmark_binary_sha256": sha256_file(binary),
            "benchmark_tool_version": TOOL_VERSION,
            "onnxruntime_version": onnx_version,
            "onnxruntime_resolved_library": str(resolved_library),
            "onnxruntime_link_output": linked.stdout,
            "operating_system": host["system"],
            "os_version": host["os_version"],
            "platform": host["platform"],
            "cpu_architecture": host["machine"],
            "hardware_model": host["hardware_model"],
            "hardware_model_error": host["hardware_model_error"],
            "logical_cpu_count": host["logical_cpu_count"],
            "logical_cpu_count_source": host["logical_cpu_count_source"],
            "hw_physicalcpu": host["hw_physicalcpu"],
            "hw_physicalcpu_error": host["hw_physicalcpu_error"],
            "hw_logicalcpu": host["hw_logicalcpu"],
            "hw_logicalcpu_error": host["hw_logicalcpu_error"],
            "nproc_cpu_count": host["nproc_cpu_count"],
            "nproc_cpu_count_error": host["nproc_cpu_count_error"],
            "python_version": host["python_version"],
            "ort_disable_telemetry": "1",
        },
        "verified_configuration": {
            "cmake_build_type": cache["CMAKE_BUILD_TYPE"],
            "sanitizer_instrumentation": "absent",
            "onnxruntime_version": onnx_version,
            "model_path": str(artifacts["model_path"]),
            "model_sha256": artifacts["model_sha256"],
            "model_relative": artifacts["model_relative"],
            "model_id": model_id,
            "weight_sidecar_path": str(artifacts["weight_sidecar_path"]),
            "weight_sidecar_relative": artifacts["weight_sidecar_relative"],
            "weight_sidecar_sha256": artifacts["weight_sidecar_sha256"],
            "input_name": artifacts["input_name"],
            "output_name": artifacts["output_name"],
            "input_width": artifacts["input_width"],
            "output_width": artifacts["output_width"],
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
        },
        "source_defined_policy": {
            "backend": BACKEND,
            "graph_optimizations": GRAPH_OPTIMIZATIONS,
            "onnxruntime_version_pin": ONNXRUNTIME_VERSION,
            "schema_version": (
                BATCH_DECISION_SCHEMA_VERSION
                if calibration_variant
                == CALIBRATION_VARIANT_ARRIVAL_AWARE_SPARSE_INSTRUMENTED_V1
                else SCHEMA_VERSION
            ),
            "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
            "publication_policy": PUBLICATION_POLICY,
            "experiment_kind": EXPERIMENT_KIND,
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
            "contrast_sign_convention": sign,
            "calibration_status": "not_calibrated" if mode != "calibration" else "calibration_mode",
            "offered_rate_frozen": offered_rate_frozen,
            "allowed_untracked_roots": list(ALLOWED_UNTRACKED),
            "rates_are_calibrated_capacity": False,
            **contrast_identity,
            **(
                {"calibration_variant": calibration_variant}
                if calibration_variant is not None
                else {}
            ),
        },
        "collection_protocol": {
            "experiment_kind": EXPERIMENT_KIND,
            "evidence_class": evidence_class_for(mode),
            "collection_mode": mode,
            "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
            "publication_policy": PUBLICATION_POLICY,
            "comparison_class": "primary" if mode == "reportable" else "exploratory",
            "workload_family": family,
            "model_id": model_id,
            "model_sha256": artifacts["model_sha256"],
            "weight_sidecar_sha256": artifacts["weight_sidecar_sha256"],
            "weight_sidecar_path": str(artifacts["weight_sidecar_path"]),
            "input_width": artifacts["input_width"],
            "output_width": artifacts["output_width"],
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
            "contrast_sign_convention": sign,
            "calibration_status": "not_calibrated" if mode != "calibration" else "calibration_mode",
            "offered_rate_frozen": offered_rate_frozen,
            "offered_rate": float(profile["target_rps"]),
            "traffic_profile": profile["id"],
            "cells": [cell["id"] for cell in cells],
            "profiles": [profile["id"] for profile in profiles],
            "cell_definitions": cell_definitions,
            "profile_definitions": profiles,
            "measurement": measurement,
            "counted_runs_per_cell": counted_runs,
            "counted_run_target": counted_runs,
            "preferred_counted_runs": PREFERRED_COUNTED_RUNS,
            "minimum_interval_blocks": MINIMUM_INTERVAL_BLOCKS,
            "pilot_count": len(pilots),
            "counted_trial_count": counted_runs * len(cells) * len(profiles),
            "pilots": pilots,
            "trial_order": [str(trial["trial_id"]) for trial in plan],
            "block_protocol": BLOCK_PROTOCOL,
            "baseline_cell": contrast["baseline_cell"],
            "alternative_cell": contrast["alternative_cell"],
            "contrast_source": contrast["contrast_source"],
            "statistical_method": STATISTICAL_METHOD,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": CONFIDENCE_LEVEL,
            "inter_trial_delay_seconds": delay_seconds,
            "environment_variable": "ORT_DISABLE_TELEMETRY=1",
            "external_wall_timeout_seconds": wall_timeout_seconds(measurement),
            "zero_admission_rejection_required": mode == "reportable"
            or calibration_variant in STRICT_CALIBRATION_VARIANTS,
            "require_strict_generator_delivery": mode in {"smoke", "reportable"}
            or calibration_variant in STRICT_CALIBRATION_VARIANTS,
            **contrast_identity,
            **(
                {"calibration_variant": calibration_variant}
                if calibration_variant is not None
                else {}
            ),
        },
        "paths": {
            "model": artifacts["model_path"],
            "sidecar": artifacts["weight_sidecar_path"],
        },
        "commit": actual_commit,
        "plan": plan,
    }


def _overlay_identity(manifest: dict, plan: list[dict], checked: dict) -> dict:
    document = dict(manifest)
    document["experiment_kind"] = EXPERIMENT_KIND
    document["collection_protocol"] = dict(checked["collection_protocol"])
    document["source_defined_policy"] = dict(checked["source_defined_policy"])
    document["verified_configuration"] = dict(checked["verified_configuration"])
    document["plan"] = plan
    document["allowed_untracked_roots"] = list(ALLOWED_UNTRACKED)
    return document


def _expected_launch(checked: dict, cell: dict, profile: dict) -> dict:
    measurement = checked["collection_protocol"]["measurement"]
    expected = {
        "model_path": str(checked["paths"]["model"]),
        "workload": profile["workload"],
        "worker_count": measurement["worker_count"],
        "queue_capacity": measurement["queue_capacity"],
        "enable_batching": cell["enable_batching"],
        "max_batch_size": cell["max_batch_size"],
        "max_batch_delay_ms": cell["max_batch_delay_ms"],
        "adaptive_batching": adaptive_batching_for_cell(cell),
        "arrival_aware_batching": arrival_aware_batching_for_cell(cell),
        "require_adaptive_batching_field": True,
        "require_arrival_aware_batching_field": True,
        "warmup_requests": measurement["warmup_requests"],
        "check_interval": measurement["check_interval"],
        "duration_ms": measurement["duration_ms"],
        "lag_tolerance_ms": measurement["lag_tolerance_ms"],
        "comparison_class": "primary",
        "analysis_protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "target_rps": profile.get("target_rps"),
    }
    if measurement.get("enable_batch_decision_timing") is True:
        expected["enable_batch_decision_timing"] = True
        expected["batch_decision_event_capacity"] = int(
            measurement["batch_decision_event_capacity"]
        )
    return expected


def execute_plan(
    *,
    run_set: pathlib.Path,
    binary: pathlib.Path,
    repo_root: pathlib.Path,
    checked: dict,
    cells: dict[str, dict],
    profiles: dict[str, dict],
    plan: list[dict],
    runner,
    sleep,
) -> bool:
    del sleep
    runs = run_set / "runs"
    measurement = checked["collection_protocol"]["measurement"]
    timeout_seconds = wall_timeout_seconds(measurement)
    mode = checked["collection_protocol"]["collection_mode"]
    require_zero = bool(checked["collection_protocol"]["zero_admission_rejection_required"])
    require_strict = bool(
        checked["collection_protocol"].get("require_strict_generator_delivery")
    )
    calibration_variant = checked["collection_protocol"].get("calibration_variant")
    hard_policy_calibration = calibration_variant in STRICT_CALIBRATION_VARIANTS
    model = checked["paths"]["model"]
    model_id = checked["collection_protocol"]["model_id"]
    recorded_cells = {
        str(item["id"]): item
        for item in checked["collection_protocol"].get("cell_definitions", [])
    }
    for trial in plan:
        trial_id = str(trial["trial_id"])
        cell = cells[str(trial["cell_id"])]
        profile = profiles[str(trial["profile_id"])]
        command = list(trial["command"])
        expected_command = neural_scheduler_command(
            binary, model, model_id, cell, profile, measurement
        )
        if command != expected_command:
            raise NeuralCollectorError(
                f"{trial_id} plan command diverges from the reconstructed command",
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
            protocol = checked["collection_protocol"]
            apply_arrival_screen = (
                mode == "reportable"
                and protocol.get("arrival_screen_max_lag_ns") is not None
                and not bool(trial["pilot"])
            )
            apply_persistent = (
                mode == "reportable"
                and protocol.get("persistent_batching_cell") is not None
                and str(trial["cell_id"]) == str(protocol["persistent_batching_cell"])
                and not bool(trial["pilot"])
            )
            expected = _expected_launch(checked, cell, profile)
            document, status, trial_problems = validate_trial(
                result.stdout,
                expected=expected,
                mode=mode,
                require_zero_rejections=require_zero,
                arrival_screen_max_lag_ns=(
                    int(protocol["arrival_screen_max_lag_ns"]) if apply_arrival_screen else None
                ),
                require_persistent_batching=apply_persistent,
                persistent_batching_min_mean=(
                    float(protocol["persistent_batching_min_mean"]) if apply_persistent else None
                ),
                require_strict_generator_delivery=require_strict,
                cell=cell,
            )
            problems.extend(trial_problems)
            if (
                document is not None
                and not problems
                and expected.get("enable_batch_decision_timing") is True
            ):
                snap = document.get("batch_decision")
                events = snap.get("events") if isinstance(snap, dict) else None
                if isinstance(events, list):
                    status = dict(status)
                    status["batch_open_wait"] = batch_open_wait_diagnostics(events)
                    if isinstance(snap, dict):
                        status["timing_data_valid"] = snap.get("timing_data_valid") is True
                status = dict(status)
                status["arrival_fidelity_status"] = ARRIVAL_FIDELITY_STATUS_NOT_CERTIFIED
            if document is not None and hard_policy_calibration:
                recorded = recorded_cells.get(str(trial["cell_id"]))
                if recorded is None:
                    problems.append(
                        f"{trial_id} missing recorded cell_definitions entry"
                    )
                else:
                    try:
                        if adaptive_batching_for_cell(recorded) != adaptive_batching_for_cell(
                            cell
                        ):
                            problems.append(
                                f"{trial_id} recorded cell adaptive_batching disagrees "
                                "with execution cell"
                            )
                        if arrival_aware_batching_for_cell(
                            recorded
                        ) != arrival_aware_batching_for_cell(cell):
                            problems.append(
                                f"{trial_id} recorded cell arrival_aware_batching disagrees "
                                "with execution cell"
                            )
                    except NeuralCollectorError as exc:
                        problems.append(str(exc))
                problems.extend(
                    adaptive_identity_consistency_problems(
                        cell=cell,
                        command=command,
                        expected=expected,
                        document=document,
                    )
                )
                # Diagnostic only: surface batch-formation and coarse lag without failing.
                mean, mean_problems = realized_mean_batch_size(document)
                if not mean_problems and mean is not None:
                    status = dict(status)
                    status["realized_mean_batch_size"] = mean
                    request_fraction, request_problems = multi_request_batch_fraction(document)
                    if not request_problems and request_fraction is not None:
                        status["multi_request_batch_fraction"] = request_fraction
                    batch_fraction, batch_problems = multi_size_batch_fraction(document)
                    if not batch_problems and batch_fraction is not None:
                        status["multi_size_batch_fraction"] = batch_fraction
                lag_screen = coarse_max_lag_v1(document)
                if lag_screen is not None:
                    status = dict(status)
                    status["coarse_max_lag_v1"] = lag_screen
            if (
                mode == "calibration"
                and document is not None
                and not hard_policy_calibration
            ):
                # Keep historical delay-calibration evidence when delivery is not strict.
                soft = [
                    item
                    for item in problems
                    if "delivery_class" in item or "admission rejection" in item
                ]
                hard = [item for item in problems if item not in soft]
                if not hard and soft:
                    problems = []
                    status = dict(status)
                    status["calibration_infeasible_markers"] = soft
            if not problems and document is not None:
                try:
                    write_exclusive_text(report_path, result.stdout)
                except (OSError, CollectorError) as exc:
                    problems.append(f"could not preserve trial report: {exc}")
        meta = {
            "trial_id": trial_id,
            "experiment_kind": EXPERIMENT_KIND,
            "model_id": model_id,
            "cell_id": trial["cell_id"],
            "profile_id": trial["profile_id"],
            "pilot": bool(trial["pilot"]),
            "run_number": trial.get("run_number"),
            "block_index": trial.get("block_index"),
            "position_in_block": trial.get("position_in_block"),
            "execution_index": trial["execution_index"],
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
            "weight_sidecar_sha256": checked["verified_configuration"]["weight_sidecar_sha256"],
            "reference_checking_policy": REFERENCE_CHECKING_POLICY,
            "git_commit": checked["commit"],
            "offered_rate": checked["collection_protocol"]["offered_rate"],
            "retried": False,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "report_path": str(report_path) if report_path.is_file() else None,
        }
        if "calibration_infeasible_markers" in status:
            meta["calibration_infeasible_markers"] = status["calibration_infeasible_markers"]
        for diagnostic_key in (
            "realized_mean_batch_size",
            "multi_request_batch_fraction",
            "multi_size_batch_fraction",
            "coarse_max_lag_v1",
            "batch_open_wait",
            "timing_data_valid",
            "arrival_fidelity_status",
        ):
            if diagnostic_key in status:
                meta[diagnostic_key] = status[diagnostic_key]
        write_json(runs / f"{trial_id}.meta.json", meta)
        if problems:
            _write_failure(
                run_set,
                {
                    "status": "failed",
                    "accepted": False,
                    "trial_id": trial_id,
                    "cell_id": trial["cell_id"],
                    "pilot": bool(trial["pilot"]),
                    "reason": "; ".join(problems),
                    "execution_timeout": timed_out,
                    "preserved_earlier_trials": True,
                    "retried": False,
                    "replaced": False,
                    "aggregate": None,
                    "experiment_kind": EXPERIMENT_KIND,
                },
            )
            return False
    return True


def validate_reportable_args(args: argparse.Namespace) -> dict:
    cell_ids = [item.strip() for item in args.cells.split(",") if item.strip()]
    return resolve_reportable_contrast(
        model_id=args.model_id,
        cell_ids=cell_ids,
        baseline_cell=args.baseline_cell,
        alternative_cell=args.alternative_cell,
        target_rps=args.target_rps,
        counted_runs=args.counted_runs,
        offered_rate_frozen=bool(args.offered_rate_frozen),
    )


def collect(
    args: argparse.Namespace,
    runner=None,
    sleep=None,
    clock=None,
) -> pathlib.Path:
    repo_root = pathlib.Path(args.repo_root).resolve()
    binary = pathlib.Path(args.binary).resolve()
    mode = args.mode
    reportable_contrast = None
    if mode == "reportable":
        reportable_contrast = validate_reportable_args(args)
    model_id = args.model_id
    model_contract(model_id)
    counted = counted_runs_for(mode, args.counted_runs)
    if reportable_contrast is not None and reportable_contrast.get("required_counted_runs") is not None:
        counted = int(reportable_contrast["required_counted_runs"])
    cell_ids = [item.strip() for item in args.cells.split(",") if item.strip()]
    cells = resolve_neural_cells(cell_ids, mode, reportable_contrast=reportable_contrast)
    if args.target_rps is None:
        if mode == "smoke":
            target_rps = 50.0
            profile_id = "low_open"
        else:
            raise NeuralCollectorError("--target-rps is required outside smoke defaults")
    else:
        target_rps = float(args.target_rps)
        profile_id = args.profile_id or "neural_open"
    if reportable_contrast is not None and reportable_contrast.get("fixed_offered_rate") is not None:
        target_rps = float(reportable_contrast["fixed_offered_rate"])
    profile = open_loop_profile(profile_id, target_rps)
    profiles = [profile]
    measurement = dict(DEFAULT_MEASUREMENT)
    for name in (
        "duration_ms",
        "max_offered",
        "max_outstanding",
        "completion_timeout_ms",
        "warmup_requests",
        "lag_tolerance_ms",
        "check_interval",
    ):
        value = getattr(args, name, None)
        if value is not None:
            measurement[name] = value
    timing_requested = bool(getattr(args, "batch_decision_timing", False))
    capacity_arg = getattr(args, "batch_decision_event_capacity", None)
    if timing_requested and capacity_arg is None:
        raise NeuralCollectorError(
            "--batch-decision-timing requires --batch-decision-event-capacity"
        )
    if capacity_arg is not None and not timing_requested:
        raise NeuralCollectorError(
            "--batch-decision-event-capacity requires --batch-decision-timing"
        )
    if timing_requested:
        if type(capacity_arg) is not int or isinstance(capacity_arg, bool) or capacity_arg < 1:
            raise NeuralCollectorError(
                "--batch-decision-event-capacity must be a positive integer"
            )
        measurement["enable_batch_decision_timing"] = True
        measurement["batch_decision_event_capacity"] = int(capacity_arg)
    offered_rate_frozen = bool(args.offered_rate_frozen)
    if mode == "reportable" and not offered_rate_frozen:
        raise NeuralCollectorError("reportable mode requires a frozen offered rate")
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

    if reportable_contrast is not None:
        baseline = str(reportable_contrast["baseline_cell"])
        alternative = str(reportable_contrast["alternative_cell"])
    else:
        baseline = args.baseline_cell or cells[0]["id"]
        alternative = args.alternative_cell or cells[-1]["id"]
    checked = neural_preflight(
        repo_root=repo_root,
        binary=binary,
        expected_commit=args.expected_commit,
        model_id=model_id,
        cells=cells,
        profiles=profiles,
        measurement=measurement,
        counted_runs=counted,
        mode=mode,
        offered_rate_frozen=offered_rate_frozen,
        delay_seconds=float(args.inter_trial_delay_seconds),
        baseline_cell=baseline,
        alternative_cell=alternative,
        runner=query_runner,
        reportable_contrast=reportable_contrast,
    )
    cell_map = {cell["id"]: cell for cell in cells}
    profile_map = {profile["id"]: profile for profile in profiles}
    plan = bind_commands(
        checked["plan"],
        binary=binary,
        model=checked["paths"]["model"],
        model_id=model_id,
        cells=cell_map,
        profiles=profile_map,
        measurement=measurement,
    )
    checked["plan"] = plan
    moment = datetime.datetime.now(datetime.timezone.utc) if clock is None else clock()
    prefix = f"neural-{model_id.replace('_', '-')}-"
    run_set_id = prefix + make_run_set_id(moment, checked["commit"])
    parent = (
        pathlib.Path(args.output_root).resolve()
        if args.output_root
        else repo_root / OUTPUT_ROOT_RELATIVE
    )
    try:
        run_set = create_run_set(parent, run_set_id)
    except CollectorError as exc:
        raise NeuralCollectorError(str(exc)) from exc
    manifest = _overlay_identity(public_manifest(checked, run_set_id, "running"), plan, checked)
    try:
        write_json(run_set / "environment.json", manifest)
        write_json(run_set / "execution_plan.json", {"trials": plan})
        accepted = execute_plan(
            run_set=run_set,
            binary=binary,
            repo_root=repo_root,
            checked=checked,
            cells=cell_map,
            profiles=profile_map,
            plan=plan,
            runner=runner,
            sleep=sleep or (lambda _seconds: None),
        )
    except NeuralCollectorError:
        raise
    except CollectorError as exc:
        raise NeuralCollectorError(str(exc), run_set) from exc
    except Exception as exc:
        raise NeuralCollectorError(f"neural collection error: {exc}", run_set) from exc
    final = _overlay_identity(
        public_manifest(checked, run_set_id, "completed" if accepted else "failed"),
        plan,
        checked,
    )
    replace_json(run_set / "environment.json", final)
    if not accepted:
        raise NeuralCollectorError(
            f"neural collection stopped; see {run_set / 'failure.json'}", run_set
        )
    return run_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect neural scheduler evidence under benchmarks/scheduler_runs. "
            "Reportable mode allowlists gemm_f32 A versus C, or the frozen A versus G2 "
            "contrast at 1000 RPS, and requires a frozen offered rate. "
            "This does not analyze or publish a speed claim."
        )
    )
    parser.add_argument("--binary", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--mode", choices=("smoke", "calibration", "reportable"), required=True)
    parser.add_argument("--model-id", required=True, choices=sorted(NEURAL_MODELS))
    parser.add_argument("--cells", default="A,C")
    parser.add_argument("--baseline-cell", default=None)
    parser.add_argument("--alternative-cell", default=None)
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--target-rps", type=float, default=None)
    parser.add_argument("--offered-rate-frozen", action="store_true")
    parser.add_argument("--counted-runs", type=int)
    parser.add_argument("--inter-trial-delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--duration-ms", type=int)
    parser.add_argument("--max-offered", type=int)
    parser.add_argument("--max-outstanding", type=int)
    parser.add_argument("--completion-timeout-ms", type=int)
    parser.add_argument("--warmup-requests", type=int)
    parser.add_argument("--lag-tolerance-ms", type=int)
    parser.add_argument("--check-interval", type=int)
    parser.add_argument(
        "--batch-decision-timing",
        action="store_true",
        help=(
            "Opt in to schema-5 sealed-batch instrumentation for the neural "
            "arrival_aware_sparse_instrumented_v1 calibration only."
        ),
    )
    parser.add_argument(
        "--batch-decision-event-capacity",
        type=int,
        default=None,
        help=(
            "Per-worker batch-decision event capacity. Required with "
            "--batch-decision-timing; rejected alone."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.preflight_only:
            # Keep preflight-only as a dry validation path for local operators.
            class _Args:
                pass

            dry = _Args()
            dry.__dict__.update(vars(args))
            dry.output_root = args.output_root or str(pathlib.Path(args.repo_root) / "tmp-unused")
            # Validate arguments without writing evidence by running through collect gates.
            if args.mode == "reportable":
                validate_reportable_args(args)
            model_contract(args.model_id)
            counted_runs_for(args.mode, args.counted_runs)
            print(json.dumps({"ok": True, "experiment_kind": EXPERIMENT_KIND}))
            return 0
        collect(args)
    except NeuralCollectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
