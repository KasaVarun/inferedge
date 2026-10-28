#!/usr/bin/env python3
"""Phase 9-B3A-R7 generative-overhead collector.

Importing this module or invoking --help does not start a campaign.
Collection requires --mode collect and every required argument.
"""

from __future__ import annotations

import argparse
import calendar
import errno
import hashlib
import json
import math
import os
import random
import re
import select
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

SCHEMA_FAMILY = "generative_overhead"
SCHEMA_VERSION = 1
EVIDENCE_PATH_CONTRACT = "generative-overhead-r7-inventory-v1"
FIXTURE_GENERATION_ALGORITHM = "generative_overhead_fixture_gen"
FIXTURE_GENERATION_VERSION = 1
MODEL_ID = "qwen3-0.6b"
PACKAGE_DIRECTORY = "/Users/varunkasa/InferEdgeModels/phase-9-b2a-qwen3-0.6b/package"
PACKAGE_MANIFEST_SHA256 = "e6ff54460e7e30b3208f76d29eb49a1388e11f2223b478fa4bbf47a48e9ee281"
GGUF_FILE = "qwen3-0.6b-Q4_K_M.gguf"
GGUF_SHA256 = "3f384c3796ca2547c5547be6bca3d487a18807cc1a2c259de6e0e1fa433bb76d"
GGUF_SIZE_BYTES = 484220192
CHAT_TEMPLATE_FILE = "qwen3-non-thinking-v1.txt"
CHAT_TEMPLATE_SHA256 = "6cb4fbe89e712bf8e8527aa8fb7d80eb0cad93ae1e236bb4be9c96d71809976a"
CHAT_TEMPLATE_BYTES = 75
USER_PLACEHOLDER = b"{user}"
USER_PLACEHOLDER_OFFSET = 17
TOKENIZER_IDENTITY = "hf:Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca+gguf-vocab"
OFFICIAL_TOKENIZE_FLAGS = "official:add_special_tokens=false"
INFEREDGE_TOKENIZE_FLAGS = "inferedge:add_special=true,parse_special=true"
CONTEXT_LIMIT = 4096
LICENSE_SHA256 = "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
LLAMA_TAG = "v0.7.0"
LLAMA_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
LLAMA_ARCHIVE_SHA256 = "55d67acf3ad2ad747a9bbeb75f9f0e173dbcc0d1753a471038a90c819421c5e1"
REJECTED_LLAMA_COMMIT = "620d2225b349e2039fa983c5e9cfc86a7e18e340"
LLAMA_PREFIX = "/private/tmp/inferedge-9b1-llama/prefix"
PROVENANCE_PATH = "/private/tmp/inferedge-9b1-llama/prefix/share/inferedge/llama-cpp-provenance.txt"
PROVENANCE_SIZE = 252
PROVENANCE_SHA256 = "e983338babf40f60c2bb6ca4a618707da2381c5d97c3231f8bc4c177626c9d59"
LIBLLAMA_SHA256 = "861ca5e181276685823f91fa8c7c6f6d71c2bcf46765779e2b3d66a44925ee70"
LIBGGML_SHA256 = "cd15ab8b4dab8f9c21c43ada03a1e55f8e55a2b75ce237296ab912d0125326a4"
LIBGGML_CPU_SHA256 = "e6610ea60ab7e564ff6b2661869cf6e0d831a258ae0a7c58a4a13d30b063f539"
LIBGGML_BASE_SHA256 = "55281093079a4c4ddd55166ad8f3aab74af276701797f1eb2c8d805c2ebb318e"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
PERMUTATION_SEED = 20261017
BOOTSTRAP_SEED = 20261018
BOOTSTRAP_REPLICATES = 10000
ABSOLUTE_THRESHOLD_NS = 2000000
RELATIVE_THRESHOLD_NUMERATOR = 5
RELATIVE_THRESHOLD_DENOMINATOR = 100
WARMUP_COUNT = 2
MEASURED_PAIRS = 20
P_TINY_USER = b"OK"
CONSTRUCTION_CELLS = [
    "C01",
    "C02",
    "C03",
    "C04",
    "C05",
    "C06",
    "C07",
    "C08",
    "C09",
    "C10",
    "C11",
    "L01",
]
GOLDEN_CELL_ORDER = [
    "C05",
    "C02",
    "C10",
    "C06",
    "C04",
    "C09",
    "C11",
    "L01",
    "C07",
    "C08",
    "C01",
    "C03",
]
CELL_SPECS = {
    "C01": {"fixture_id": "P_tiny", "max_new_tokens": 1, "threads": 1, "measurement_class": "warm_request", "role": "primary", "finish": ("maximum_tokens",)},
    "C02": {"fixture_id": "P_128", "max_new_tokens": 1, "threads": 1, "measurement_class": "warm_request", "role": "secondary", "finish": ("maximum_tokens",)},
    "C03": {"fixture_id": "P_512", "max_new_tokens": 1, "threads": 1, "measurement_class": "warm_request", "role": "secondary", "finish": ("maximum_tokens",)},
    "C04": {"fixture_id": "P_1024", "max_new_tokens": 1, "threads": 1, "measurement_class": "warm_request", "role": "secondary", "finish": ("maximum_tokens",)},
    "C05": {"fixture_id": "P_2048", "max_new_tokens": 1, "threads": 1, "measurement_class": "warm_request", "role": "secondary", "finish": ("maximum_tokens",)},
    "C06": {"fixture_id": "P_near", "max_new_tokens": 1, "threads": 1, "measurement_class": "warm_request", "role": "secondary", "finish": ("maximum_tokens",)},
    "C07": {"fixture_id": "P_tiny", "max_new_tokens": 16, "threads": 1, "measurement_class": "warm_request", "role": "secondary", "finish": ("eos", "maximum_tokens")},
    "C08": {"fixture_id": "P_tiny", "max_new_tokens": 128, "threads": 1, "measurement_class": "warm_request", "role": "secondary", "finish": ("eos", "maximum_tokens")},
    "C09": {"fixture_id": "P_tiny", "max_new_tokens": 1, "threads": 2, "measurement_class": "warm_request", "role": "secondary", "finish": ("maximum_tokens",)},
    "C10": {"fixture_id": "P_tiny", "max_new_tokens": 1, "threads": 4, "measurement_class": "warm_request", "role": "secondary", "finish": ("maximum_tokens",)},
    "C11": {"fixture_id": "P_tiny", "max_new_tokens": 1, "threads": 6, "measurement_class": "warm_request", "role": "secondary", "finish": ("maximum_tokens",)},
    "L01": {"fixture_id": None, "max_new_tokens": None, "threads": None, "measurement_class": "load", "role": "descriptive_load", "finish": ("load_success",)},
}
COMMAND_IDS = (
    "pmset_batt",
    "pmset_settings",
    "pmset_custom",
    "ioreg_power",
    "pmset_log",
    "system_profiler_power",
    "powermanagement_plist",
)
TEMP_NAME_RE = re.compile(r"^\.[^/]+\.pub\.[0-9a-f]{32}\.tmp$")
UUID_PLIST_RE = re.compile(
    r"^com\.apple\.PowerManagement\.[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\.plist$"
)
MAX_LPM_XML_BYTES = 262144
MAX_PMSET_LOG_BYTES = 16_777_216
MAX_PMSET_LOG_LINE_BYTES = 8192
MAX_PMSET_LOG_LINES = 200_000
PMSET_TARGET_TOKENS = frozenset({"Sleep", "Wake", "DarkWake"})
PMSET_DATE_SHAPED_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
PMSET_TIMEZONE_RE = re.compile(r"^([+-])(\d{2})(\d{2})$")
PLIST_BACKUP_SUFFIXES = (".old", ".backup", ".bak", ".tmp", ".orig")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
PROTECTED_PATH_SEGMENTS = frozenset(
    {"baselines", "concurrent_runs", "scheduler_runs", "experiments"}
)
MAX_TRUSTED_JSON_BYTES = 2_000_000
MAX_WORKER_LINE_BYTES = 1_048_576
MAX_WORKER_STDERR_BYTES = 1_048_576
WORKER_REQUEST_TIMEOUT_S = 300.0
WORKER_SHUTDOWN_GRACE_S = 1.0
WORKER_SHUTDOWN_TERM_S = 1.0
WORKER_SHUTDOWN_KILL_S = 2.0
REAP_SLICE_S = 0.05
ENVIRONMENT_COMMAND_TIMEOUT_S = 30.0
WORKER_OP_TIMEOUT_S = WORKER_REQUEST_TIMEOUT_S
PAIR_ORDER_RULE = "even ie_then_ref, odd ref_then_ie"
ENVIRONMENT_POLICY = "ac_connected,battery_not_discharging,lpm_off_positive,sleepwake_invariant_and_log"
WORKER_RESPONSE_KEYS = (
    "ok",
    "error_code",
    "error_message",
    "finish_reason",
    "prompt_token_ids",
    "generated_token_ids",
    "generated_text_utf8",
    "prompt_token_count",
    "generated_token_count",
    "threads_requested",
    "threads_effective",
    "n_ctx",
    "n_batch",
    "n_ubatch",
    "n_seq_max",
    "worker_instance_id",
    "pid",
    "t0_ns",
    "t1_ns",
    "duration_ns",
)
WARMUP_KEYS = [
    "schema_family",
    "schema_version",
    "record_class",
    "campaign_id",
    "campaign_manifest_sha256",
    "protocol_document_sha256",
    "fixtures_sha256",
    "source_commit",
    "cell_id",
    "implementation",
    "warmup_index",
    "measurement_class",
    "success",
    "prompt_token_count_expected",
    "prompt_token_count_actual",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "generated_token_count",
    "generated_token_ids",
    "generated_token_ids_sha256",
    "generated_text_utf8",
    "generated_text_sha256",
    "finish_reason",
    "utf8_valid",
    "threads_requested",
    "threads_effective",
    "n_ctx",
    "n_batch",
    "n_ubatch",
    "n_seq_max",
    "worker_binary_sha256",
    "worker_instance_id",
    "pid",
    "t0_ns",
    "t1_ns",
    "duration_ns",
    "error_code",
    "error_message",
]
MEASURED_KEYS = [
    "schema_family",
    "schema_version",
    "record_class",
    "campaign_id",
    "campaign_manifest_sha256",
    "protocol_document_sha256",
    "fixtures_sha256",
    "source_commit",
    "cell_id",
    "implementation",
    "pair_index",
    "arm_order",
    "order_position",
    "measurement_class",
    "fixture_id",
    "max_new_tokens",
    "success",
    "prompt_token_count_expected",
    "prompt_token_count_actual",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "generated_token_count",
    "generated_token_ids",
    "generated_token_ids_sha256",
    "generated_text_utf8",
    "generated_text_sha256",
    "finish_reason",
    "utf8_valid",
    "threads_requested",
    "threads_effective",
    "n_ctx",
    "n_batch",
    "n_ubatch",
    "n_seq_max",
    "worker_binary_sha256",
    "worker_instance_id",
    "pid",
    "t0_ns",
    "t1_ns",
    "duration_ns",
    "error_code",
    "error_message",
    "exclusion_status",
    "exclusion_reason",
    "publication_eligible",
]
IDENTITY_RESPONSE_KEYS = (
    "ok",
    "implementation",
    "worker_instance_id",
    "pid",
    "evidence_path_contract",
)

# Test hooks.
fsync_file = os.fsync
os_link = os.link
uuid4_hex = lambda: uuid.uuid4().hex


class CollectorError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def canonical_json_bytes(value: object) -> bytes:
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=False, separators=(",", ":"))
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detached_digest_bytes(payload: bytes) -> bytes:
    return (sha256_bytes(payload) + "\n").encode("ascii")


def encode_token_ids_le(ids: list[int]) -> bytes:
    out = bytearray()
    for token in ids:
        if token < -2147483648 or token > 2147483647:
            raise CollectorError("record_schema_invalid", "token id outside int32")
        out.extend(int(token).to_bytes(4, "little", signed=True))
    return bytes(out)


def token_ids_sha256(ids: list[int]) -> str:
    return sha256_bytes(encode_token_ids_le(ids))


def permute_cells(cells: list[str], seed: int = PERMUTATION_SEED) -> list[str]:
    rng = random.Random(seed)
    array = list(cells)
    n = len(array)
    for i in range(n):
        j = i + rng.randrange(n - i)
        array[i], array[j] = array[j], array[i]
    return array


def expanded_trials(cell_order: list[str] | None = None, measured_pairs: int | None = None) -> list[dict]:
    trials = []
    order = GOLDEN_CELL_ORDER if cell_order is None else list(cell_order)
    pairs = MEASURED_PAIRS if measured_pairs is None else measured_pairs
    for cell_id in order:
        for pair_index in range(pairs):
            arm_order = "ie_then_ref" if pair_index % 2 == 0 else "ref_then_ie"
            first = "inferedge" if arm_order == "ie_then_ref" else "reference"
            second = "reference" if first == "inferedge" else "inferedge"
            trials.append(
                {
                    "cell_id": cell_id,
                    "pair_index": pair_index,
                    "arm_order": arm_order,
                    "order_position": 0,
                    "implementation": first,
                }
            )
            trials.append(
                {
                    "cell_id": cell_id,
                    "pair_index": pair_index,
                    "arm_order": arm_order,
                    "order_position": 1,
                    "implementation": second,
                }
            )
    return trials


def environment_sample_ids(cell_order: list[str] | None = None) -> list[str]:
    samples = ["campaign-start"]
    order = GOLDEN_CELL_ORDER if cell_order is None else list(cell_order)
    for cell_id in order:
        samples.append(f"{cell_id}-pre")
        samples.append(f"{cell_id}-post")
    samples.append("campaign-end")
    return samples


def inventory_a_relative_paths(
    cells: list[str] | None = None,
    measured_pairs: int | None = None,
    warmup_count: int | None = None,
    cell_order: list[str] | None = None,
) -> list[str]:
    paths = ["campaign_manifest.json", "campaign_manifest.sha256"]
    cell_ids = CONSTRUCTION_CELLS if cells is None else list(cells)
    pairs = MEASURED_PAIRS if measured_pairs is None else measured_pairs
    warmups = WARMUP_COUNT if warmup_count is None else warmup_count
    for cell_id in cell_ids:
        for impl in ("inferedge", "reference"):
            for index in range(warmups):
                paths.append(f"warmup/{cell_id}/{impl}/warmup-{index}.json")
    for cell_id in cell_ids:
        for pair_index in range(pairs):
            pair = f"{pair_index:02d}"
            for impl in ("inferedge", "reference"):
                paths.append(f"measured/{cell_id}/pair-{pair}/{impl}.json")
    for sample_id in environment_sample_ids(cell_order):
        paths.append(f"environment/{sample_id}.json")
        for command_id in COMMAND_IDS:
            paths.append(f"environment/raw/{sample_id}/{command_id}.stdout.bin")
            paths.append(f"environment/raw/{sample_id}/{command_id}.stderr.bin")
    return paths


def inventory_b_relative_paths() -> list[str]:
    return inventory_a_relative_paths() + ["analysis/aggregate.json", "analysis/summary.json"]


def reject_relative_path(relative: str) -> str | None:
    if relative.startswith("/") or relative == "" or "\\" in relative or "\x00" in relative:
        return "path_invalid"
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return "path_invalid"
    if "/".join(parts) != relative:
        return "path_invalid"
    return None


def lstat_no_follow(path: Path):
    return os.lstat(path)


def path_is_symlink(path: Path) -> bool:
    try:
        return stat.S_ISLNK(lstat_no_follow(path).st_mode)
    except FileNotFoundError:
        return False


def require_regular_file(path: Path) -> str | None:
    try:
        info = lstat_no_follow(path)
    except FileNotFoundError:
        return "missing_file"
    if stat.S_ISLNK(info.st_mode):
        return "symlink_rejected"
    if not stat.S_ISREG(info.st_mode):
        return "path_invalid"
    if info.st_nlink != 1:
        return "hardlink_rejected"
    return None


def require_directory(path: Path) -> str | None:
    try:
        info = lstat_no_follow(path)
    except FileNotFoundError:
        return "missing_file"
    if stat.S_ISLNK(info.st_mode):
        return "symlink_rejected"
    if not stat.S_ISDIR(info.st_mode):
        return "path_invalid"
    return None


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        fsync_file(fd)
    finally:
        os.close(fd)


def publish_exclusive(final_path: Path, exact_bytes: bytes) -> None:
    parent = final_path.parent
    code = require_directory(parent)
    if code is not None:
        raise CollectorError(code, f"publication directory invalid: {parent}")
    if reject_relative_path(final_path.name) and final_path.name.startswith("."):
        pass
    token = uuid4_hex()
    if len(token) != 32 or any(ch not in "0123456789abcdef" for ch in token):
        raise CollectorError("atomic_publish_failure", "temp token is not 32 hex")
    temp_name = f".{final_path.name}.pub.{token}.tmp"
    if TEMP_NAME_RE.fullmatch(temp_name) is None:
        raise CollectorError("atomic_publish_failure", "temp grammar failed")
    temp_path = parent / temp_name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temp_path, flags, 0o644)
    try:
        written = 0
        while written < len(exact_bytes):
            written += os.write(fd, exact_bytes[written:])
        os.fsync  # keep name for tests
        fsync_file(fd)
    finally:
        os.close(fd)
    reread = temp_path.read_bytes()
    if reread != exact_bytes:
        raise CollectorError("atomic_publish_failure", "temp reread mismatch")
    try:
        os_link(temp_path, final_path)
    except OSError as exc:
        raise CollectorError("atomic_publish_failure", f"exclusive link failed: {exc}") from exc
    try:
        os.unlink(temp_path)
    except OSError:
        pass
    fsync_directory(parent)
    final_info = lstat_no_follow(final_path)
    if not stat.S_ISREG(final_info.st_mode) or final_info.st_nlink != 1:
        raise CollectorError("atomic_publish_failure", "final file is not a unique regular file")
    if final_path.read_bytes() != exact_bytes:
        raise CollectorError("atomic_publish_failure", "final reread mismatch")


def render_template(template_bytes: bytes, user_bytes: bytes) -> bytes:
    if len(template_bytes) != CHAT_TEMPLATE_BYTES:
        raise CollectorError("identity_mismatch", "template size")
    if sha256_bytes(template_bytes) != CHAT_TEMPLATE_SHA256:
        raise CollectorError("identity_mismatch", "template hash")
    pos = template_bytes.find(USER_PLACEHOLDER)
    if pos != USER_PLACEHOLDER_OFFSET or template_bytes.find(USER_PLACEHOLDER, pos + 1) != -1:
        raise CollectorError("identity_mismatch", "template placeholder")
    return template_bytes[:pos] + user_bytes + template_bytes[pos + 6 :]


def candidate_user_bytes(index: int) -> bytes:
    return b"alpha " * index


def official_tokenize(tokenizer, rendered: bytes) -> list[int]:
    return list(tokenizer.encode(rendered.decode("utf-8"), add_special_tokens=False))


def inferedge_tokenize_worker(worker: Path, gguf: Path, rendered: bytes) -> list[int]:
    with InferedgeTokenizer(worker, gguf) as tokenizer:
        return tokenizer.tokenize(rendered)


class InferedgeTokenizer:
    def __init__(self, worker: Path, gguf: Path) -> None:
        self.process = subprocess.Popen(
            [str(worker), "--tokenize-only", "--gguf", str(gguf)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def tokenize(self, rendered: bytes) -> list[int]:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(canonical_json_bytes({"text": rendered.decode("utf-8")}))
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        document = json.loads(line.decode("utf-8"))
        if not document.get("ok"):
            raise CollectorError("tokenization", document.get("error_message", "tokenize failed"))
        return [int(token) for token in document["token_ids"]]

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=30)

    def __enter__(self) -> "InferedgeTokenizer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def select_fixtures(official_counts: list[tuple[int, int, list[int]]]) -> dict[str, dict]:
    # official_counts: (i, count, ids) for eligible matching tokenizations
    selected: dict[str, dict] = {}
    for target in (128, 512, 1024, 2048):
        eligible = [
            (abs(count - target), i, count, ids)
            for i, count, ids in official_counts
            if target - 2 <= count <= target + 2
        ]
        if not eligible:
            raise CollectorError("identity_mismatch", f"no eligible fixture for {target}")
        eligible.sort()
        _delta, index, count, ids = eligible[0]
        selected[f"P_{target}"] = {"selected_i": index, "count": count, "ids": ids}
    near = [(count, i, ids) for i, count, ids in official_counts if count <= 4095]
    if not near:
        raise CollectorError("identity_mismatch", "no eligible P_near")
    near.sort()
    count, index, ids = near[-1]
    selected["P_near"] = {"selected_i": index, "count": count, "ids": ids}
    return selected


def fixture_object(fixture_id: str, user: bytes, rendered: bytes, ids: list[int], extra: dict) -> dict:
    document = {
        "fixture_id": fixture_id,
        "user_text_utf8": user.decode("utf-8"),
        "user_text_sha256": sha256_bytes(user),
        "rendered_text_utf8": rendered.decode("utf-8"),
        "rendered_text_sha256": sha256_bytes(rendered),
        "official_token_ids": ids,
        "inferedge_token_ids": list(ids),
        "expected_token_count": len(ids),
    }
    document.update(extra)
    return document


def generate_fixtures(
    official_tokenizer_dir: Path,
    inferedge_worker: Path,
    output_json: Path,
    output_digest: Path,
) -> dict:
    package = Path(PACKAGE_DIRECTORY)
    template = (package / CHAT_TEMPLATE_FILE).read_bytes()
    gguf = package / GGUF_FILE
    if gguf.stat().st_size != GGUF_SIZE_BYTES or sha256_file(gguf) != GGUF_SHA256:
        raise CollectorError("identity_mismatch", "GGUF identity")
    if sha256_file(package / "manifest.txt") != PACKAGE_MANIFEST_SHA256:
        raise CollectorError("identity_mismatch", "package manifest")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise CollectorError("identity_mismatch", "official tokenizer unavailable") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(official_tokenizer_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    tiny_rendered = render_template(template, P_TINY_USER)
    tiny_official = official_tokenize(tokenizer, tiny_rendered)
    with InferedgeTokenizer(inferedge_worker, gguf) as ie_tokenizer:
        tiny_ie = ie_tokenizer.tokenize(tiny_rendered)
        if tiny_official != tiny_ie:
            raise CollectorError("token_mismatch", "P_tiny token mismatch")
        matches: list[tuple[int, int, list[int]]] = []
        for index in range(1, 10001):
            user = candidate_user_bytes(index)
            rendered = render_template(template, user)
            official_ids = official_tokenize(tokenizer, rendered)
            inferedge_ids = ie_tokenizer.tokenize(rendered)
            if official_ids != inferedge_ids:
                continue
            matches.append((index, len(official_ids), official_ids))
    chosen = select_fixtures(matches)
    fixtures = [
        fixture_object("P_tiny", P_TINY_USER, tiny_rendered, tiny_official, {}),
        fixture_object(
            "P_128",
            candidate_user_bytes(chosen["P_128"]["selected_i"]),
            render_template(template, candidate_user_bytes(chosen["P_128"]["selected_i"])),
            chosen["P_128"]["ids"],
            {"selected_i": chosen["P_128"]["selected_i"]},
        ),
        fixture_object(
            "P_512",
            candidate_user_bytes(chosen["P_512"]["selected_i"]),
            render_template(template, candidate_user_bytes(chosen["P_512"]["selected_i"])),
            chosen["P_512"]["ids"],
            {"selected_i": chosen["P_512"]["selected_i"]},
        ),
        fixture_object(
            "P_1024",
            candidate_user_bytes(chosen["P_1024"]["selected_i"]),
            render_template(template, candidate_user_bytes(chosen["P_1024"]["selected_i"])),
            chosen["P_1024"]["ids"],
            {"selected_i": chosen["P_1024"]["selected_i"]},
        ),
        fixture_object(
            "P_2048",
            candidate_user_bytes(chosen["P_2048"]["selected_i"]),
            render_template(template, candidate_user_bytes(chosen["P_2048"]["selected_i"])),
            chosen["P_2048"]["ids"],
            {"selected_i": chosen["P_2048"]["selected_i"]},
        ),
        fixture_object(
            "P_near",
            candidate_user_bytes(chosen["P_near"]["selected_i"]),
            render_template(template, candidate_user_bytes(chosen["P_near"]["selected_i"])),
            chosen["P_near"]["ids"],
            {
                "selected_i": chosen["P_near"]["selected_i"],
                "context_limit": CONTEXT_LIMIT,
                "max_new_tokens_reserved": 1,
                "derivation_rule": "max (count, i) among i=1..10000 with count<=4095",
                "accepted_count": chosen["P_near"]["count"],
            },
        ),
    ]
    sidecar = {
        "schema_family": SCHEMA_FAMILY,
        "schema_version": SCHEMA_VERSION,
        "fixture_generation_algorithm": FIXTURE_GENERATION_ALGORITHM,
        "fixture_generation_version": FIXTURE_GENERATION_VERSION,
        "model_id": MODEL_ID,
        "package_directory": PACKAGE_DIRECTORY,
        "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
        "gguf_file": GGUF_FILE,
        "gguf_sha256": GGUF_SHA256,
        "gguf_size_bytes": GGUF_SIZE_BYTES,
        "chat_template_file": CHAT_TEMPLATE_FILE,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "tokenizer_identity": TOKENIZER_IDENTITY,
        "official_tokenize_flags": OFFICIAL_TOKENIZE_FLAGS,
        "inferedge_tokenize_flags": INFEREDGE_TOKENIZE_FLAGS,
        "context_limit": CONTEXT_LIMIT,
        "fixtures": fixtures,
    }
    payload = canonical_json_bytes(sidecar)
    publish_exclusive(output_json, payload)
    publish_exclusive(output_digest, detached_digest_bytes(payload))
    return sidecar


def capture_one_line(argv: list[str]) -> str:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=ENVIRONMENT_COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise CollectorError("environment_command_failed", "environment command timeout") from exc
    if completed.returncode != 0:
        raise CollectorError("environment_command_failed", "host command failed")
    raw = completed.stdout
    text = raw.decode("utf-8")
    if b"\r" in raw or b"\x00" in raw:
        raise CollectorError("utf8_invalid", "host command contained CR/NUL")
    if text.endswith("\n"):
        text = text[:-1]
    if "\n" in text or text == "":
        raise CollectorError("environment_parse_invalid", "host command is not one line")
    return text


def capture_compiler(compiler: Path) -> tuple[str, str, str]:
    try:
        completed = subprocess.run(
            [str(compiler), "--version"],
            capture_output=True,
            check=False,
            timeout=ENVIRONMENT_COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise CollectorError("environment_command_failed", "environment command timeout") from exc
    if completed.returncode != 0:
        raise CollectorError("environment_command_failed", "compiler --version failed")
    raw = completed.stdout
    if b"\r" in raw or b"\x00" in raw:
        raise CollectorError("utf8_invalid", "compiler stdout contained CR/NUL")
    text = raw.decode("utf-8")
    if text.encode("utf-8") != raw:
        raise CollectorError("utf8_invalid", "compiler stdout is not strict UTF-8")
    first = ""
    for line in text.split("\n"):
        if line != "":
            first = line
            break
    if first == "":
        raise CollectorError("environment_parse_invalid", "compiler stdout has no nonempty line")
    return first, text, sha256_bytes(raw)


def discover_powermanagement_plist(preferences_root: Path | None = None) -> Path:
    root = Path("/Library/Preferences") if preferences_root is None else preferences_root
    try:
        root_info = os.lstat(root)
    except FileNotFoundError as exc:
        raise CollectorError("environment_lpm_invalid", "preferences root missing") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise CollectorError("environment_lpm_invalid", "preferences root invalid")
    matches: list[Path] = []
    try:
        names = sorted(entry.name for entry in os.scandir(root))
    except OSError as exc:
        raise CollectorError("environment_lpm_invalid", "preferences root unreadable") from exc
    for name in names:
        if any(name.endswith(suffix) for suffix in PLIST_BACKUP_SUFFIXES):
            continue
        if not UUID_PLIST_RE.fullmatch(name):
            continue
        path = root / name
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise CollectorError("environment_lpm_invalid", "authoritative plist is a symlink")
        if not stat.S_ISREG(info.st_mode):
            raise CollectorError("environment_lpm_invalid", "authoritative plist is not a regular file")
        matches.append(path)
    if len(matches) == 0:
        raise CollectorError("environment_lpm_invalid", "no authoritative PowerManagement plist")
    if len(matches) != 1:
        raise CollectorError("environment_lpm_invalid", "UUID PowerManagement plist is not unique")
    return matches[0]


def command_argv(command_id: str, plist: Path) -> list[str]:
    mapping = {
        "pmset_batt": ["/usr/bin/pmset", "-g", "batt"],
        "pmset_settings": ["/usr/bin/pmset", "-g"],
        "pmset_custom": ["/usr/bin/pmset", "-g", "custom"],
        "ioreg_power": ["/usr/sbin/ioreg", "-r", "-c", "IOPMrootDomain", "-d", "1"],
        "pmset_log": ["/usr/bin/pmset", "-g", "log"],
        "system_profiler_power": ["/usr/sbin/system_profiler", "SPPowerDataType"],
        "powermanagement_plist": ["/usr/bin/plutil", "-convert", "xml1", "-o", "-", str(plist)],
    }
    return mapping[command_id]


def parse_ac(stdout: bytes) -> tuple[str, bool]:
    text = stdout.decode("utf-8")
    lines = [line for line in text.splitlines() if line == "Now drawing from 'AC Power'"]
    if len(lines) != 1 or "Battery Power" in text or "discharging" in text:
        raise CollectorError("environment_ac_invalid", "AC parse failed")
    return lines[0], True


def _plist_dict_map(element: object, context: str) -> dict[str, object]:
    tag = getattr(element, "tag", "")
    if tag != "dict":
        raise CollectorError("environment_lpm_invalid", f"{context} is not a dictionary")
    children = list(element)
    if len(children) % 2 != 0:
        raise CollectorError("environment_lpm_invalid", f"{context} has unpaired keys")
    mapping: dict[str, object] = {}
    for index in range(0, len(children), 2):
        key_el = children[index]
        value_el = children[index + 1]
        if getattr(key_el, "tag", "") != "key" or not isinstance(key_el.text, str) or key_el.text == "":
            raise CollectorError("environment_lpm_invalid", f"{context} has an invalid key")
        name = key_el.text
        if name in mapping:
            raise CollectorError("environment_lpm_invalid", f"duplicate key {name}")
        mapping[name] = value_el
    return mapping


def parse_lpm_xml(xml_text: str) -> int:
    import plistlib
    import xml.etree.ElementTree as ET

    if not isinstance(xml_text, str) or xml_text == "":
        raise CollectorError("environment_lpm_invalid", "empty lpm xml")
    raw = xml_text.encode("utf-8")
    if len(raw) > MAX_LPM_XML_BYTES:
        raise CollectorError("environment_lpm_invalid", "oversized lpm xml")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise CollectorError("environment_lpm_invalid", "malformed lpm xml") from exc
    if root.tag != "plist":
        raise CollectorError("environment_lpm_invalid", "plist root")
    dicts = [child for child in list(root) if child.tag == "dict"]
    if len(dicts) != 1:
        raise CollectorError("environment_lpm_invalid", "plist dict")
    top = _plist_dict_map(dicts[0], "plist root")
    if "AC Power" not in top:
        raise CollectorError("environment_lpm_invalid", "AC Power missing")
    ac_el = top["AC Power"]
    ac_map = _plist_dict_map(ac_el, "AC Power")
    if "LowPowerMode" not in ac_map:
        raise CollectorError("environment_lpm_invalid", "LowPowerMode missing")
    try:
        loaded = plistlib.loads(raw)
    except Exception as exc:
        raise CollectorError("environment_lpm_invalid", "plistlib failed") from exc
    if not isinstance(loaded, dict) or "AC Power" not in loaded:
        raise CollectorError("environment_lpm_invalid", "loaded AC Power missing")
    ac_loaded = loaded["AC Power"]
    if not isinstance(ac_loaded, dict) or "LowPowerMode" not in ac_loaded:
        raise CollectorError("environment_lpm_invalid", "loaded LowPowerMode missing")
    value = ac_loaded["LowPowerMode"]
    if type(value) is not int or value != 0:
        raise CollectorError("environment_lpm_invalid", "LowPowerMode is not integer zero")
    return value


def parse_sleepwake(ioreg_stdout: bytes) -> tuple[str, str, str]:
    text = ioreg_stdout.decode("utf-8")
    uuid_matches = re.findall(r'"SleepWakeUUID" = "([0-9A-Fa-f-]{36})"', text)
    sleep_matches = re.findall(r'"Last Sleep Reason" = "([^"]*)"', text)
    wake_matches = re.findall(r'"Wake Reason" = "([^"]*)"', text)
    if len(uuid_matches) != 1 or len(sleep_matches) != 1 or len(wake_matches) != 1:
        raise CollectorError("environment_sleepwake_invalid", "SleepWake fields are not unique")
    return uuid_matches[0], sleep_matches[0], wake_matches[0]


def parse_pmset_timezone(token: str) -> int:
    matched = PMSET_TIMEZONE_RE.fullmatch(token)
    if matched is None:
        raise CollectorError("environment_sleepwake_invalid", "pmset log timezone")
    hours = int(matched.group(2))
    minutes = int(matched.group(3))
    if hours > 14 or minutes > 59:
        raise CollectorError("environment_sleepwake_invalid", "pmset log timezone")
    sign = 1 if matched.group(1) == "+" else -1
    return sign * (hours * 3600 + minutes * 60)


def compare_pmset_log_baseline(baseline: list[dict], observed: list[dict]) -> None:
    if observed == baseline:
        return
    if baseline and not observed:
        raise CollectorError("environment_sleepwake_invalid", "pmset log empty overlap")
    if observed and not baseline:
        raise CollectorError("environment_sleepwake_invalid", "pmset log new events")
    if len(observed) < len(baseline):
        if observed == baseline[-len(observed) :]:
            return
        if observed == baseline[: len(observed)]:
            raise CollectorError("environment_sleepwake_invalid", "pmset log newest dropped")
        raise CollectorError("environment_sleepwake_invalid", "pmset log interior deletion")
    if len(observed) > len(baseline):
        if observed[-len(baseline) :] == baseline:
            raise CollectorError("environment_sleepwake_invalid", "pmset log reappeared events")
        raise CollectorError("environment_sleepwake_invalid", "pmset log new events")
    raise CollectorError("environment_sleepwake_invalid", "pmset log reordered")


def parse_pmset_log_events(stdout: bytes, campaign_start_ns: int) -> list[dict]:
    if len(stdout) > MAX_PMSET_LOG_BYTES:
        raise CollectorError("environment_sleepwake_invalid", "pmset log oversized")
    if b"\x00" in stdout or b"\r" in stdout:
        raise CollectorError("environment_sleepwake_invalid", "pmset log binary")
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CollectorError("environment_sleepwake_invalid", "pmset log utf-8") from exc
    lines = text.splitlines()
    if len(lines) > MAX_PMSET_LOG_LINES:
        raise CollectorError("environment_sleepwake_invalid", "pmset log line count")
    events: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    last_aware: int | None = None
    start_floor = campaign_start_ns // 1_000_000_000
    for line in lines:
        encoded = line.encode("utf-8")
        if len(encoded) > MAX_PMSET_LOG_LINE_BYTES:
            raise CollectorError("environment_sleepwake_invalid", "pmset log line length")
        if len(line) < 19:
            continue
        stamp = line[0:19]
        if not PMSET_DATE_SHAPED_RE.fullmatch(stamp):
            continue
        try:
            local = time.strptime(stamp, "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:
            raise CollectorError("environment_sleepwake_invalid", "pmset log impossible date") from exc
        rest = line[19:].lstrip()
        parts = rest.split()
        if len(parts) < 2:
            raise CollectorError("environment_sleepwake_invalid", "pmset log timezone")
        timezone = parts[0]
        offset = parse_pmset_timezone(timezone)
        event = parts[1]
        if event not in PMSET_TARGET_TOKENS:
            continue
        aware = calendar.timegm(local) - offset
        if last_aware is not None and aware < last_aware:
            raise CollectorError("environment_sleepwake_invalid", "pmset log timestamp regression")
        last_aware = aware
        if aware >= start_floor:
            raise CollectorError("environment_sleepwake_invalid", "sleep/wake during campaign")
        digest = sha256_bytes(encoded)
        identity = (stamp, timezone, event, digest)
        if identity in seen:
            raise CollectorError("environment_sleepwake_invalid", "pmset log duplicate event")
        seen.add(identity)
        events.append(
            {
                "timestamp_text": stamp,
                "timezone_text": timezone,
                "event_token": event,
                "aware_unix_second": aware,
                "raw_line_sha256": digest,
            }
        )
    return events


def validate_contract(value: str) -> None:
    if value != EVIDENCE_PATH_CONTRACT:
        raise CollectorError("manifest_invalid", "evidence_path_contract")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="collect_generative")
    parser.add_argument(
        "--mode",
        choices=("collect", "generate-fixtures", "mechanics-self-check"),
        required=False,
    )
    parser.add_argument("--campaign-id")
    parser.add_argument("--evidence-root")
    parser.add_argument("--campaign-class", choices=("campaign", "dry_run"))
    parser.add_argument("--source-commit")
    parser.add_argument("--protocol-document")
    parser.add_argument("--fixtures")
    parser.add_argument("--ie-worker")
    parser.add_argument("--ref-worker")
    parser.add_argument("--compiler")
    parser.add_argument("--official-tokenizer-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--output-digest")
    parser.add_argument("--contract")
    parser.add_argument("--model-package")
    parser.add_argument("--fixtures-sha256")
    parser.add_argument("--build-type", default="Release")
    return parser


def require_collect_args(args: argparse.Namespace) -> None:
    required = [
        args.mode,
        args.campaign_id,
        args.evidence_root,
        args.campaign_class,
        args.source_commit,
        args.protocol_document,
        args.fixtures,
        args.ie_worker,
        args.ref_worker,
        args.compiler,
        args.contract,
        args.model_package,
    ]
    if args.mode != "collect" or any(value is None for value in required):
        raise CollectorError("campaign_incomplete", "collect requires every explicit argument")


def mechanics_self_check() -> int:
    if permute_cells(CONSTRUCTION_CELLS) != GOLDEN_CELL_ORDER:
        raise CollectorError("matrix_invalid", "golden order")
    if len(expanded_trials()) != 480:
        raise CollectorError("sample_count_invalid", "expanded trials")
    if len(inventory_a_relative_paths()) != 920:
        raise CollectorError("campaign_incomplete", "inventory A")
    if len(inventory_b_relative_paths()) != 922:
        raise CollectorError("campaign_incomplete", "inventory B")
    validate_contract(EVIDENCE_PATH_CONTRACT)
    print("mechanics-self-check ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode is None:
        parser.print_help()
        return 0
    if args.mode == "mechanics-self-check":
        return mechanics_self_check()
    if args.mode == "generate-fixtures":
        if args.official_tokenizer_dir is None or args.ie_worker is None:
            raise CollectorError("campaign_incomplete", "fixture generation arguments")
        output_json = Path(args.output_json or "models/llm/generative_overhead_fixtures.json")
        output_digest = Path(args.output_digest or str(output_json) + ".sha256")
        generate_fixtures(Path(args.official_tokenizer_dir), Path(args.ie_worker), output_json, output_digest)
        return 0
    if args.mode == "collect":
        require_collect_args(args)
        run_collect(args)
        return 0
    parser.print_help()
    return 0


class CollectHooks:
    """Test-only injection. Production CLI never constructs this object."""

    def __init__(self) -> None:
        self.cell_order: list[str] | None = None
        self.measured_pairs: int | None = None
        self.warmup_count: int | None = None
        self.skip_package_hash: bool = False
        self.worker_factory: Callable[[str, Path, Path], "WorkerSession"] | None = None
        self.command_runner: Callable[[list[str]], tuple[bytes, bytes, int]] | None = None
        self.host_fields: dict[str, str] | None = None
        self.environment_raw: dict[str, tuple[bytes, bytes]] | None = None
        self.plist_path: Path | None = None
        self.strict_full_fixtures: bool = True
        self.clock_ns: Callable[[], int] | None = None
        self.after_root_created: Callable[[Path], None] | None = None
        self.before_publish: Callable[[Path], None] | None = None
        self.interrupt_after: str | None = None


def load_json_strict(data: bytes, max_bytes: int = MAX_TRUSTED_JSON_BYTES) -> dict[str, Any]:
    if len(data) > max_bytes:
        raise CollectorError("record_schema_invalid", "oversized json")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CollectorError("utf8_invalid", "json is not utf-8") from exc

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise CollectorError("record_schema_invalid", f"duplicate key {key}")
            document[key] = value
        return document

    try:
        document = json.loads(text, object_pairs_hook=reject_duplicates)
    except CollectorError:
        raise
    except json.JSONDecodeError as exc:
        raise CollectorError("record_schema_invalid", "json decode failed") from exc
    if not isinstance(document, dict):
        raise CollectorError("record_schema_invalid", "json is not an object")
    return document


def lexical_parts(path: Path) -> list[str]:
    return [part for part in path.parts if part not in {"/", ""}]


def reject_protected_lexical_path(path: Path) -> None:
    if any(part in PROTECTED_PATH_SEGMENTS for part in lexical_parts(path)):
        raise CollectorError("path_invalid", "protected evidence root")


def _is_system_volume_alias(path: Path) -> bool:
    return path in {Path("/var"), Path("/tmp"), Path("/etc")}


def reject_symlink_ancestors(path: Path) -> None:
    current = path
    seen: set[str] = set()
    while True:
        key = str(current)
        if key in seen:
            break
        seen.add(key)
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            current = current.parent
            if current == current.parent and not current.exists():
                break
            continue
        if stat.S_ISLNK(info.st_mode) and not _is_system_volume_alias(current):
            raise CollectorError("symlink_rejected", f"symlinked path component: {current}")
        if current.parent == current:
            break
        current = current.parent


def require_launchable_worker(path: Path) -> None:
    reject_protected_lexical_path(path)
    if not path.is_absolute():
        raise CollectorError("path_invalid", "worker path must be absolute")
    reject_symlink_ancestors(path)
    reason = require_regular_file(path)
    if reason == "missing_file":
        raise CollectorError("missing_file", "missing worker executable")
    if reason is not None:
        raise CollectorError("path_invalid", reason)
    if not os.access(path, os.X_OK):
        raise CollectorError("path_invalid", "worker is not executable")


def require_nonexistent_or_empty_root(root: Path) -> None:
    if not root.exists():
        return
    info = os.lstat(root)
    if stat.S_ISLNK(info.st_mode):
        raise CollectorError("symlink_rejected", "evidence root is a symlink")
    if not stat.S_ISDIR(info.st_mode):
        raise CollectorError("path_invalid", "evidence root exists and is not a directory")
    if any(root.iterdir()):
        raise CollectorError("campaign_incomplete", "evidence root exists and is not empty")
    raise CollectorError("campaign_incomplete", "evidence root already exists")


def walk_regular_files(root: Path) -> list[str]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            full = Path(dirpath) / name
            found.append(full.relative_to(root).as_posix())
    return found


def verify_inventory_a(root: Path, expected: list[str]) -> dict[str, dict[str, Any]]:
    present = walk_regular_files(root)
    expected_set = set(expected)
    present_set = set(present)
    if present_set != expected_set:
        extra = sorted(present_set - expected_set)
        missing = sorted(expected_set - present_set)
        if extra:
            raise CollectorError("extra_file", f"inventory extra: {extra[:3]}")
        raise CollectorError("missing_file", f"inventory missing: {missing[:3]}")
    observed: dict[str, dict[str, Any]] = {}
    for rel in expected:
        full = root / rel
        code = require_regular_file(full)
        if code is not None:
            raise CollectorError(code, f"inventory file invalid: {rel}")
        payload = full.read_bytes()
        observed[rel] = {"size_bytes": len(payload), "sha256": sha256_bytes(payload)}
    if (root / "analysis").exists():
        raise CollectorError("campaign_incomplete", "analysis directory exists before collection complete")
    return observed


def require_int_timing(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise CollectorError("timing_invalid", f"{name} is boolean")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CollectorError("timing_invalid", f"{name} is nonfinite")
        raise CollectorError("timing_invalid", f"{name} is not integer")
    if type(value) is not int:
        raise CollectorError("timing_invalid", f"{name} is not integer")
    if value < 0:
        raise CollectorError("timing_invalid", f"{name} is negative")
    if value > 2**63 - 1:
        raise CollectorError("timing_invalid", f"{name} overflow")
    return value


def validate_generation_response(document: dict[str, Any]) -> None:
    if set(document) != set(WORKER_RESPONSE_KEYS):
        extra = set(document) - set(WORKER_RESPONSE_KEYS)
        missing = set(WORKER_RESPONSE_KEYS) - set(document)
        if extra:
            raise CollectorError("record_schema_invalid", f"unknown response key {sorted(extra)}")
        raise CollectorError("record_schema_invalid", f"missing response key {sorted(missing)}")
    t0 = require_int_timing(document["t0_ns"], "t0_ns")
    t1 = require_int_timing(document["t1_ns"], "t1_ns")
    duration = require_int_timing(document["duration_ns"], "duration_ns")
    if t1 < t0 or duration != t1 - t0:
        raise CollectorError("timing_invalid", "impossible timing relationship")
    if document.get("ok") is not True:
        raise CollectorError("record_schema_invalid", "worker response is not ok")
    if type(document.get("pid")) is not int:
        raise CollectorError("record_schema_invalid", "pid")
    for key in ("prompt_token_ids", "generated_token_ids"):
        if not isinstance(document[key], list) or any(type(item) is not int for item in document[key]):
            raise CollectorError("record_schema_invalid", key)


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


def _signal_owned(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        os.kill(process.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def reap_owned_process(
    process: subprocess.Popen[bytes],
    *,
    grace_s: float,
    term_s: float,
    kill_s: float,
    write_shutdown: bool,
    on_wait: Callable[[], None] | None = None,
    close_pipes: bool = True,
) -> bool:
    def wait_hook() -> None:
        if on_wait is not None:
            on_wait()

    def settled() -> bool:
        return process.poll() is not None and not _process_group_exists(process.pid)

    def phase(budget_s: float) -> None:
        deadline = time.monotonic() + budget_s
        while True:
            wait_hook()
            if settled():
                return
            now = time.monotonic()
            if now >= deadline:
                return
            remaining = min(REAP_SLICE_S, deadline - now)
            if process.poll() is None:
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass
            else:
                time.sleep(min(0.01, remaining))

    wait_hook()
    if write_shutdown and process.poll() is None and process.stdin is not None:
        try:
            process.stdin.write(b'{"op":"shutdown"}\n')
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
    phase(grace_s)
    if process.poll() is None or _process_group_exists(process.pid):
        wait_hook()
        _signal_owned(process, signal.SIGTERM)
        phase(term_s)
    if process.poll() is None or _process_group_exists(process.pid):
        wait_hook()
        _signal_owned(process, signal.SIGKILL)
        phase(kill_s)
    reaped = process.poll() is not None and not _process_group_exists(process.pid)
    if reaped and close_pipes:
        _close_process_pipes(process)
    return reaped


class WorkerSession:
    popen_impl = staticmethod(subprocess.Popen)

    def __init__(
        self,
        executable: Path,
        argv: list[str],
        implementation: str,
        timeout_s: float | None = None,
        request_timeout_s: float | None = None,
        skip_executable_check: bool = False,
        shutdown_grace_s: float = WORKER_SHUTDOWN_GRACE_S,
        shutdown_term_s: float = WORKER_SHUTDOWN_TERM_S,
        shutdown_kill_s: float = WORKER_SHUTDOWN_KILL_S,
    ) -> None:
        self.executable = executable
        self.argv = argv
        self.implementation = implementation
        if timeout_s is not None:
            request_timeout_s = timeout_s
        self.request_timeout_s = WORKER_REQUEST_TIMEOUT_S if request_timeout_s is None else request_timeout_s
        self.timeout_s = self.request_timeout_s
        self.shutdown_grace_s = shutdown_grace_s
        self.shutdown_term_s = shutdown_term_s
        self.shutdown_kill_s = shutdown_kill_s
        self.process: subprocess.Popen[bytes] | None = None
        self.identity: dict[str, Any] = {}
        self.stderr_bytes = bytearray()
        self.cleanup_error: CollectorError | None = None
        self.requests_sent = 0
        self.stderr_total = 0
        self.stderr_overflow = False
        self.stderr_overflow_reported = False
        self.stderr_nonblocking = False
        self.stderr_at_eof = False
        self.stderr_read_error: CollectorError | None = None
        self.stderr_inherited_open = False
        self.pipes_closed = False
        try:
            if not skip_executable_check:
                require_launchable_worker(executable)
            try:
                self.process = self.popen_impl(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
            except OSError as exc:
                raise CollectorError("environment_command_failed", f"worker launch failed: {exc}") from exc
            if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
                raise CollectorError("environment_command_failed", "worker pipes missing")
            try:
                os.set_blocking(self.process.stderr.fileno(), False)
            except OSError as exc:
                raise CollectorError(
                    "environment_command_failed",
                    f"worker stderr nonblocking failed: {exc}",
                ) from exc
            self.stderr_nonblocking = True
        except BaseException:
            self.close()
            raise

    def pending_empty(self) -> bool:
        return self.process is None or self.process.poll() is not None

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.stderr_overflow and not self.stderr_overflow_reported:
            self._consume_stderr(classify=True)
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise CollectorError("environment_command_failed", "worker is not running")
        self.requests_sent += 1
        line = canonical_json_bytes(payload)
        try:
            self.process.stdin.write(line)
            self.process.stdin.flush()
        except BrokenPipeError:
            self._consume_stderr(classify=True)
            self._note_exit()
            raise CollectorError("environment_command_failed", "broken pipe")
        raw = self._read_one_line(self.request_timeout_s, "worker request timeout")
        self._consume_stderr(classify=True)
        self._reject_extra_stdout()
        document = load_json_strict(raw, MAX_WORKER_LINE_BYTES)
        return document

    def _read_one_line(self, timeout_s: float, timeout_message: str) -> bytes:
        assert self.process is not None and self.process.stdout is not None
        stdout_fd = self.process.stdout.fileno()
        buf = bytearray()
        deadline = time.monotonic() + timeout_s
        while True:
            self._consume_stderr(classify=True)
            if b"\n" in buf:
                line, rest = buf.split(b"\n", 1)
                self._consume_stderr(classify=True)
                if self.stderr_overflow:
                    if not self.stderr_overflow_reported:
                        self.stderr_overflow_reported = True
                    raise CollectorError("record_schema_invalid", "oversized stderr")
                if rest:
                    raise CollectorError("record_schema_invalid", "extra response")
                try:
                    line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise CollectorError("utf8_invalid", "invalid UTF-8") from exc
                return bytes(line)
            if time.monotonic() >= deadline:
                self._consume_stderr(classify=True)
                if self.stderr_overflow:
                    if not self.stderr_overflow_reported:
                        self.stderr_overflow_reported = True
                    raise CollectorError("record_schema_invalid", "oversized stderr")
                raise CollectorError("environment_command_failed", timeout_message)
            watch = [stdout_fd]
            stderr_fd: int | None = None
            if (
                not self.stderr_at_eof
                and self.stderr_nonblocking
                and self.process.stderr is not None
            ):
                try:
                    stderr_fd = self.process.stderr.fileno()
                except ValueError:
                    stderr_fd = None
            if stderr_fd is not None:
                watch.append(stderr_fd)
            remaining = deadline - time.monotonic()
            if remaining < 0:
                remaining = 0
            try:
                readable, _, _ = select.select(watch, [], [], remaining)
            except InterruptedError:
                continue
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                raise
            if stderr_fd is not None and stderr_fd in readable:
                self._consume_stderr(classify=True)
            if stdout_fd in readable:
                if self.stderr_overflow:
                    if not self.stderr_overflow_reported:
                        self.stderr_overflow_reported = True
                    raise CollectorError("record_schema_invalid", "oversized stderr")
                chunk = os.read(stdout_fd, 4096)
                if not chunk:
                    self._classify_stdout_eof(deadline)
                buf.extend(chunk)
                if len(buf) > MAX_WORKER_LINE_BYTES:
                    self._consume_stderr(classify=True)
                    if self.stderr_overflow:
                        if not self.stderr_overflow_reported:
                            self.stderr_overflow_reported = True
                        raise CollectorError("record_schema_invalid", "oversized stderr")
                    raise CollectorError("record_schema_invalid", "oversized stdout")
                continue
            if stderr_fd is None or stderr_fd not in readable:
                self._consume_stderr(classify=True)
                if self.stderr_overflow:
                    if not self.stderr_overflow_reported:
                        self.stderr_overflow_reported = True
                    raise CollectorError("record_schema_invalid", "oversized stderr")
                raise CollectorError("environment_command_failed", timeout_message)

    def _reject_extra_stdout(self) -> None:
        if self.stderr_overflow:
            if not self.stderr_overflow_reported:
                self.stderr_overflow_reported = True
            raise CollectorError("record_schema_invalid", "oversized stderr")
        assert self.process is not None and self.process.stdout is not None
        fd = self.process.stdout.fileno()
        readable, _, _ = select.select([fd], [], [], 0)
        if readable:
            extra = os.read(fd, 1)
            if extra:
                self._consume_stderr(classify=True)
                if self.stderr_overflow:
                    if not self.stderr_overflow_reported:
                        self.stderr_overflow_reported = True
                    raise CollectorError("record_schema_invalid", "oversized stderr")
                raise CollectorError("record_schema_invalid", "extra response")

    def _classify_stdout_eof(self, deadline: float) -> None:
        while True:
            self._consume_stderr(classify=True)
            if self.stderr_overflow:
                if not self.stderr_overflow_reported:
                    self.stderr_overflow_reported = True
                raise CollectorError("record_schema_invalid", "oversized stderr")
            if self.stderr_read_error is not None:
                raise self.stderr_read_error
            abnormal = self._abnormal_exit_error()
            if abnormal is not None:
                raise abnormal
            if self.process is not None and self.process.poll() == 0:
                raise CollectorError("environment_command_failed", "unexpected EOF")
            now = time.monotonic()
            if now >= deadline or self.process is None:
                raise CollectorError("environment_command_failed", "unexpected EOF")
            remaining = deadline - now
            stderr_fd: int | None = None
            if not self.stderr_at_eof and self.stderr_nonblocking:
                stderr_fd = self._stderr_fd()
            if stderr_fd is not None:
                try:
                    select.select([stderr_fd], [], [], remaining)
                except InterruptedError:
                    continue
                except OSError as exc:
                    if exc.errno == errno.EINTR:
                        continue
                    raise
                continue
            try:
                self.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

    def _abnormal_exit_error(self) -> CollectorError | None:
        if self.process is None:
            return None
        code = self.process.poll()
        if code is None:
            return None
        if code < 0:
            return CollectorError("environment_command_failed", f"worker signal exit {-code}")
        if code != 0:
            return CollectorError("environment_command_failed", f"worker nonzero exit {code}")
        return None

    def _note_exit(self) -> None:
        self._consume_stderr(classify=True)
        abnormal = self._abnormal_exit_error()
        if abnormal is not None:
            raise abnormal

    def _stderr_fd(self) -> int | None:
        if self.process is None or self.process.stderr is None:
            return None
        try:
            return self.process.stderr.fileno()
        except ValueError:
            return None

    def _consume_stderr(self, *, classify: bool) -> None:
        if not self.stderr_nonblocking or self.stderr_at_eof or self.process is None:
            return
        fd = self._stderr_fd()
        if fd is None:
            return
        while True:
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                return
            except InterruptedError:
                continue
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    return
                if exc.errno == errno.EINTR:
                    continue
                if self.stderr_read_error is None:
                    self.stderr_read_error = CollectorError(
                        "environment_command_failed",
                        f"worker stderr read failed: {exc.errno}",
                    )
                if classify and not self.stderr_overflow:
                    raise self.stderr_read_error
                return
            if not chunk:
                self.stderr_at_eof = True
                return
            self.stderr_total += len(chunk)
            room = MAX_WORKER_STDERR_BYTES - len(self.stderr_bytes)
            if room > 0:
                self.stderr_bytes.extend(chunk[:room])
            if self.stderr_total > MAX_WORKER_STDERR_BYTES:
                self.stderr_overflow = True
                if classify and not self.stderr_overflow_reported:
                    self.stderr_overflow_reported = True
                    raise CollectorError("record_schema_invalid", "oversized stderr")

    def _authoritative_drain(self, reaped: bool) -> None:
        self._consume_stderr(classify=False)
        if self.stderr_at_eof or not reaped:
            return
        fd = self._stderr_fd()
        if fd is None:
            return
        deadline = time.monotonic() + self.shutdown_kill_s
        while not self.stderr_at_eof:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                _readable, _, _ = select.select([fd], [], [], remaining)
            except InterruptedError:
                continue
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                if self.stderr_read_error is None:
                    self.stderr_read_error = CollectorError(
                        "environment_command_failed",
                        f"worker stderr read failed: {exc.errno}",
                    )
                return
            self._consume_stderr(classify=False)
        if not self.stderr_at_eof:
            self.stderr_inherited_open = True

    def _classify_close(self, reaped: bool, active: bool) -> None:
        if self.stderr_overflow and not self.stderr_overflow_reported:
            if not reaped and self.cleanup_error is None:
                self.cleanup_error = CollectorError("environment_command_failed", "worker shutdown timeout")
            if not active:
                self.stderr_overflow_reported = True
                raise CollectorError("record_schema_invalid", "oversized stderr")
            return
        if self.stderr_overflow_reported:
            if not reaped and self.cleanup_error is None:
                self.cleanup_error = CollectorError("environment_command_failed", "worker shutdown timeout")
            return
        if active:
            if self.cleanup_error is None:
                if self.stderr_read_error is not None:
                    self.cleanup_error = self.stderr_read_error
                elif self.stderr_inherited_open:
                    self.cleanup_error = CollectorError(
                        "environment_command_failed",
                        "worker stderr still open",
                    )
                elif not reaped:
                    self.cleanup_error = CollectorError(
                        "environment_command_failed",
                        "worker shutdown timeout",
                    )
                else:
                    self.cleanup_error = self._abnormal_exit_error()
            return
        if self.stderr_read_error is not None:
            raise self.stderr_read_error
        if self.stderr_inherited_open:
            raise CollectorError("environment_command_failed", "worker stderr still open")
        if not reaped:
            self.cleanup_error = CollectorError("environment_command_failed", "worker shutdown timeout")
            raise self.cleanup_error
        abnormal = self._abnormal_exit_error()
        if abnormal is not None:
            raise abnormal

    def _close_owned_pipes(self) -> None:
        if self.pipes_closed:
            return
        if self.process is not None:
            _close_process_pipes(self.process)
        self.pipes_closed = True
        self.process = None

    def close(self) -> None:
        if self.process is None:
            return
        active = sys.exc_info()[1] is not None
        try:
            if not self.stderr_nonblocking:
                reap_owned_process(
                    self.process,
                    grace_s=self.shutdown_grace_s,
                    term_s=self.shutdown_term_s,
                    kill_s=self.shutdown_kill_s,
                    write_shutdown=True,
                    on_wait=None,
                    close_pipes=False,
                )
                if not active:
                    abnormal = self._abnormal_exit_error()
                    if abnormal is not None:
                        raise abnormal
                return
            reaped = reap_owned_process(
                self.process,
                grace_s=self.shutdown_grace_s,
                term_s=self.shutdown_term_s,
                kill_s=self.shutdown_kill_s,
                write_shutdown=True,
                on_wait=lambda: self._consume_stderr(classify=False),
                close_pipes=False,
            )
            self._authoritative_drain(reaped)
            self._classify_close(reaped, active)
        finally:
            self._close_owned_pipes()


class CampaignRuntime:
    def __init__(self, args: argparse.Namespace, hooks: CollectHooks | None) -> None:
        self.args = args
        self.hooks = hooks or CollectHooks()
        self.workers: list[WorkerSession] = []
        self.root: Path | None = None
        self.identities: dict[str, str] = {}
        self.fixtures: dict[str, dict[str, Any]] = {}
        self.sleepwake: tuple[str, str, str] | None = None
        self.pmset_log_events: list[dict] | None = None
        self.campaign_start_ns = 0
        self.cell_order = GOLDEN_CELL_ORDER if self.hooks.cell_order is None else list(self.hooks.cell_order)
        self.measured_pairs = MEASURED_PAIRS if self.hooks.measured_pairs is None else self.hooks.measured_pairs
        self.warmup_count = WARMUP_COUNT if self.hooks.warmup_count is None else self.hooks.warmup_count
        self.invocations: list[str] = []

    def start_worker(self, implementation: str, executable: Path, package: Path) -> WorkerSession:
        worker: WorkerSession | None = None
        try:
            if self.hooks.worker_factory is not None:
                worker = self.hooks.worker_factory(implementation, executable, package)
            else:
                worker = WorkerSession(
                    executable,
                    [str(executable), "--worker-stdio", "--package", str(package)],
                    implementation,
                )
            self.workers.append(worker)
            self.invocations.append(implementation)
            return worker
        except BaseException:
            if worker is not None:
                worker.close()
                if worker in self.workers:
                    self.workers.remove(worker)
            raise

    def close_workers(self) -> None:
        while self.workers:
            worker = self.workers.pop()
            worker.close()


def run_command(argv: list[str], hooks: CollectHooks) -> tuple[bytes, bytes, int]:
    if hooks.command_runner is not None:
        return hooks.command_runner(argv)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            check=False,
            timeout=ENVIRONMENT_COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise CollectorError("environment_command_failed", "environment command timeout") from exc
    return completed.stdout, completed.stderr, completed.returncode


def worker_request_timeout_s(_cell_id: str | None = None) -> float:
    return WORKER_REQUEST_TIMEOUT_S


def worker_identity_timeout_s() -> float:
    return WORKER_REQUEST_TIMEOUT_S


def capture_one_line_from_runner(argv: list[str], hooks: CollectHooks) -> str:
    stdout, _stderr, code = run_command(argv, hooks)
    if code != 0:
        raise CollectorError("environment_command_failed", "host command failed")
    if b"\r" in stdout or b"\x00" in stdout:
        raise CollectorError("utf8_invalid", "host command contained CR/NUL")
    text = stdout.decode("utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    if "\n" in text or text == "":
        raise CollectorError("environment_parse_invalid", "host command is not one line")
    return text


def default_environment_raw() -> dict[str, tuple[bytes, bytes]]:
    xml = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        b'<plist version="1.0"><dict><key>AC Power</key><dict>'
        b"<key>LowPowerMode</key><integer>0</integer></dict></dict></plist>\n"
    )
    ioreg = (
        b'"SleepWakeUUID" = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"\n'
        b'"Last Sleep Reason" = "Idle"\n'
        b'"Wake Reason" = "EC.LidOpen"\n'
    )
    return {
        "pmset_batt": (b"Now drawing from 'AC Power'\n", b""),
        "pmset_settings": (b"sleep 0\n", b""),
        "pmset_custom": (b"custom\n", b""),
        "ioreg_power": (ioreg, b""),
        "pmset_log": (b"", b""),
        "system_profiler_power": (b"Power:\n", b""),
        "powermanagement_plist": (xml, b""),
    }


def validate_worker_binary(path: Path, expected_token: str) -> str:
    if not path.is_file():
        raise CollectorError("identity_mismatch", f"worker missing: {path}")
    reject_symlink_ancestors(path)
    code = require_regular_file(path)
    if code is not None:
        raise CollectorError(code, f"worker path invalid: {path}")
    version = capture_one_line([str(path), "--version"])
    if expected_token not in version or EVIDENCE_PATH_CONTRACT not in version:
        raise CollectorError("identity_mismatch", f"worker identity invalid: {version}")
    return version


def validate_fixtures(path: Path, digest_path: Path, hooks: CollectHooks) -> tuple[dict[str, Any], str]:
    reject_symlink_ancestors(path)
    reject_symlink_ancestors(digest_path)
    for candidate in (path, digest_path):
        code = require_regular_file(candidate)
        if code is not None:
            raise CollectorError(code, f"fixture path invalid: {candidate}")
    payload = path.read_bytes()
    digest_text = digest_path.read_text(encoding="ascii").strip()
    actual = sha256_bytes(payload)
    if actual != digest_text or not HEX64.fullmatch(digest_text):
        raise CollectorError("digest_mismatch", "fixture sidecar hash")
    document = load_json_strict(payload)
    if document.get("schema_family") != SCHEMA_FAMILY or document.get("schema_version") != SCHEMA_VERSION:
        raise CollectorError("record_schema_invalid", "fixture schema")
    if document.get("fixture_generation_algorithm") != FIXTURE_GENERATION_ALGORITHM:
        raise CollectorError("record_schema_invalid", "fixture algorithm")
    fixtures = document.get("fixtures")
    if not isinstance(fixtures, list):
        raise CollectorError("record_schema_invalid", "fixtures list")
    if hooks.strict_full_fixtures:
        ids = [item.get("fixture_id") for item in fixtures]
        if ids != ["P_tiny", "P_128", "P_512", "P_1024", "P_2048", "P_near"]:
            raise CollectorError("record_schema_invalid", "strict fixture identity set")
    by_id = {item["fixture_id"]: item for item in fixtures}
    return by_id, actual


def validate_package(package: Path, hooks: CollectHooks) -> None:
    reject_symlink_ancestors(package)
    code = require_directory(package)
    if code is not None:
        raise CollectorError(code, f"model package invalid: {package}")
    if hooks.skip_package_hash:
        return
    gguf = package / GGUF_FILE
    manifest = package / "manifest.txt"
    template = package / CHAT_TEMPLATE_FILE
    for path in (gguf, manifest, template):
        file_code = require_regular_file(path)
        if file_code is not None:
            raise CollectorError("identity_mismatch", f"package file invalid: {path}")
    if gguf.stat().st_size != GGUF_SIZE_BYTES or sha256_file(gguf) != GGUF_SHA256:
        raise CollectorError("identity_mismatch", "GGUF identity")
    if sha256_file(manifest) != PACKAGE_MANIFEST_SHA256:
        raise CollectorError("identity_mismatch", "package manifest")
    if sha256_file(template) != CHAT_TEMPLATE_SHA256:
        raise CollectorError("identity_mismatch", "template hash")
    provenance = Path(PROVENANCE_PATH)
    if provenance.stat().st_size != PROVENANCE_SIZE or sha256_file(provenance) != PROVENANCE_SHA256:
        raise CollectorError("identity_mismatch", "provenance")


def capture_host_fields(hooks: CollectHooks) -> dict[str, str]:
    if hooks.host_fields is not None:
        return dict(hooks.host_fields)
    return {
        "host_system": capture_one_line_from_runner(["/usr/bin/uname", "-s"], hooks),
        "host_kernel_release": capture_one_line_from_runner(["/usr/bin/uname", "-r"], hooks),
        "host_architecture": capture_one_line_from_runner(["/usr/bin/uname", "-m"], hooks),
        "host_model": capture_one_line_from_runner(["/usr/sbin/sysctl", "-n", "hw.model"], hooks),
        "os_product_name": capture_one_line_from_runner(["/usr/bin/sw_vers", "-productName"], hooks),
        "os_product_version": capture_one_line_from_runner(["/usr/bin/sw_vers", "-productVersion"], hooks),
        "os_build_version": capture_one_line_from_runner(["/usr/bin/sw_vers", "-buildVersion"], hooks),
    }


def publish_bytes(runtime: CampaignRuntime, dest: Path, payload: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if runtime.hooks.before_publish is not None:
        runtime.hooks.before_publish(dest)
    publish_exclusive(dest, payload)


def build_environment_sample(
    runtime: CampaignRuntime,
    sample_id: str,
    campaign_id: str,
) -> None:
    assert runtime.root is not None
    plist = runtime.hooks.plist_path or Path("/tmp/unused-plist")
    if runtime.hooks.environment_raw is None and runtime.hooks.command_runner is None:
        plist = discover_powermanagement_plist()
    raw_map = runtime.hooks.environment_raw or default_environment_raw()
    commands = []
    raw_payloads: list[tuple[str, bytes, bytes, list[str]]] = []
    for command_id in COMMAND_IDS:
        argv = command_argv(command_id, plist)
        if runtime.hooks.environment_raw is not None:
            stdout, stderr = raw_map[command_id]
            code = 0
        else:
            stdout, stderr, code = run_command(argv, runtime.hooks)
        if code != 0:
            raise CollectorError("environment_command_failed", command_id)
        raw_payloads.append((command_id, stdout, stderr, argv))
    for command_id, stdout, stderr, argv in raw_payloads:
        stdout_rel = f"environment/raw/{sample_id}/{command_id}.stdout.bin"
        stderr_rel = f"environment/raw/{sample_id}/{command_id}.stderr.bin"
        publish_bytes(runtime, runtime.root / stdout_rel, stdout)
        publish_bytes(runtime, runtime.root / stderr_rel, stderr)
        commands.append(
            {
                "command_id": command_id,
                "argv": argv,
                "exit_status": 0,
                "stdout_relative_path": stdout_rel,
                "stdout_size_bytes": len(stdout),
                "stdout_sha256": sha256_bytes(stdout),
                "stderr_relative_path": stderr_rel,
                "stderr_size_bytes": len(stderr),
                "stderr_sha256": sha256_bytes(stderr),
            }
        )
    by_id = {item[0]: item for item in raw_payloads}
    ac_line, ac_ok = parse_ac(by_id["pmset_batt"][1])
    lpm = parse_lpm_xml(by_id["powermanagement_plist"][1].decode("utf-8"))
    uuid_text, sleep, wake = parse_sleepwake(by_id["ioreg_power"][1])
    if runtime.sleepwake is None:
        runtime.sleepwake = (uuid_text, sleep, wake)
    elif runtime.sleepwake != (uuid_text, sleep, wake):
        raise CollectorError("environment_sleepwake_invalid", "SleepWake changed")
    events = parse_pmset_log_events(by_id["pmset_log"][1], runtime.campaign_start_ns)
    if runtime.pmset_log_events is None:
        runtime.pmset_log_events = events
    else:
        compare_pmset_log_baseline(runtime.pmset_log_events, events)
        runtime.pmset_log_events = events
    clock = runtime.hooks.clock_ns or time.time_ns
    document = {
        "schema_family": SCHEMA_FAMILY,
        "schema_version": SCHEMA_VERSION,
        "record_class": "environment",
        "campaign_id": campaign_id,
        "campaign_manifest_sha256": runtime.identities["campaign_manifest_sha256"],
        "protocol_document_sha256": runtime.identities["protocol_document_sha256"],
        "fixtures_sha256": runtime.identities["fixtures_sha256"],
        "source_commit": runtime.identities["source_commit"],
        "sample_id": sample_id,
        "wall_time_ns": clock(),
        "commands": commands,
        "parsed": {
            "ac_line": ac_line,
            "ac_ok": ac_ok,
            "low_power_mode_value": lpm,
            "low_power_mode_off": lpm == 0,
            "sleepwake_uuid": uuid_text,
            "last_sleep_reason": sleep,
            "wake_reason": wake,
            "pmset_log_disqualifying_events": events,
        },
        "thermal_policy": "uncontrolled",
    }
    publish_bytes(runtime, runtime.root / f"environment/{sample_id}.json", canonical_json_bytes(document))


def identity_request(worker: WorkerSession, implementation: str) -> dict[str, Any]:
    try:
        document = worker.request({"op": "identity"})
        if set(document) != set(IDENTITY_RESPONSE_KEYS):
            raise CollectorError("record_schema_invalid", "identity keys")
        if document.get("ok") is not True or document.get("implementation") != implementation:
            raise CollectorError("identity_mismatch", "worker implementation")
        if document.get("evidence_path_contract") != EVIDENCE_PATH_CONTRACT:
            raise CollectorError("identity_mismatch", "worker contract")
        worker.identity = document
        return document
    except BaseException:
        worker.close()
        raise


def load_request(worker: WorkerSession) -> None:
    document = worker.request({"op": "load"})
    if document.get("ok") is not True:
        raise CollectorError("identity_mismatch", document.get("error_message", "load failed"))


def generation_request(worker: WorkerSession, cell_id: str, user_text: str, max_new: int, threads: int) -> dict[str, Any]:
    if cell_id == "L01":
        document = worker.request({"op": "l01"})
    else:
        document = worker.request(
            {"op": "warm_request", "user_text_utf8": user_text, "max_new_tokens": max_new, "threads": threads}
        )
    validate_generation_response(document)
    return document


def fixture_for_cell(runtime: CampaignRuntime, cell_id: str) -> dict[str, Any]:
    spec = CELL_SPECS[cell_id]
    fixture_id = spec["fixture_id"]
    if fixture_id is None:
        return {
            "fixture_id": None,
            "user_text_utf8": "",
            "official_token_ids": [],
            "expected_token_count": 0,
        }
    fixture = runtime.fixtures[fixture_id]
    return fixture


def compare_generation(left: dict[str, Any], right: dict[str, Any], spec: dict[str, Any]) -> None:
    for key in (
        "prompt_token_ids",
        "generated_token_ids",
        "generated_text_utf8",
        "finish_reason",
        "prompt_token_count_actual",
        "generated_token_count",
    ):
        if left[key] != right[key]:
            raise CollectorError("correctness_mismatch", key)
    if left["finish_reason"] not in spec["finish"]:
        raise CollectorError("finish_reason_invalid", left["finish_reason"])


def make_logical_record(
    runtime: CampaignRuntime,
    record_class: str,
    cell_id: str,
    implementation: str,
    worker: WorkerSession,
    response: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    spec = CELL_SPECS[cell_id]
    fixture = fixture_for_cell(runtime, cell_id)
    prompt_ids = [int(x) for x in response["prompt_token_ids"]]
    generated_ids = [int(x) for x in response["generated_token_ids"]]
    text = str(response["generated_text_utf8"])
    expected_ids = [int(x) for x in fixture.get("official_token_ids", fixture.get("inferedge_token_ids", []))]
    if cell_id != "L01" and prompt_ids != expected_ids:
        raise CollectorError("token_mismatch", "prompt token identity")
    if cell_id != "L01" and len(prompt_ids) != int(fixture.get("expected_token_count", len(expected_ids))):
        raise CollectorError("token_mismatch", "prompt token count")
    if token_ids_sha256(prompt_ids) != token_ids_sha256(expected_ids) and cell_id != "L01":
        raise CollectorError("digest_mismatch", "prompt digest")
    generated_text_digest = sha256_bytes(text.encode("utf-8"))
    finish = response["finish_reason"]
    if finish not in spec["finish"]:
        raise CollectorError("finish_reason_invalid", finish)
    record = {
        "schema_family": SCHEMA_FAMILY,
        "schema_version": SCHEMA_VERSION,
        "record_class": record_class,
        "campaign_id": runtime.identities["campaign_id"],
        "campaign_manifest_sha256": runtime.identities["campaign_manifest_sha256"],
        "protocol_document_sha256": runtime.identities["protocol_document_sha256"],
        "fixtures_sha256": runtime.identities["fixtures_sha256"],
        "source_commit": runtime.identities["source_commit"],
        "cell_id": cell_id,
        "implementation": implementation,
    }
    record.update(extra)
    record.update(
        {
            "measurement_class": spec["measurement_class"],
            "success": True,
            "prompt_token_count_expected": 0 if cell_id == "L01" else len(expected_ids),
            "prompt_token_count_actual": len(prompt_ids),
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
            "generated_token_count": len(generated_ids),
            "generated_token_ids": generated_ids,
            "generated_token_ids_sha256": token_ids_sha256(generated_ids),
            "generated_text_utf8": text,
            "generated_text_sha256": generated_text_digest,
            "finish_reason": finish,
            "utf8_valid": True,
            "threads_requested": 1 if cell_id == "L01" else spec["threads"],
            "threads_effective": int(response["threads_effective"]),
            "n_ctx": int(response["n_ctx"]),
            "n_batch": int(response["n_batch"]),
            "n_ubatch": int(response["n_ubatch"]),
            "n_seq_max": int(response["n_seq_max"]),
            "worker_binary_sha256": runtime.identities[
                "ie_worker_sha256" if implementation == "inferedge" else "ref_worker_sha256"
            ],
            "worker_instance_id": str(response["worker_instance_id"]),
            "pid": int(response["pid"]),
            "t0_ns": int(response["t0_ns"]),
            "t1_ns": int(response["t1_ns"]),
            "duration_ns": int(response["duration_ns"]),
            "error_code": "none",
            "error_message": "",
        }
    )
    if record_class == "measured":
        record["exclusion_status"] = "none"
        record["exclusion_reason"] = ""
        record["publication_eligible"] = True
        return {key: record[key] for key in MEASURED_KEYS}
    return {key: record[key] for key in WARMUP_KEYS}


def maybe_interrupt(runtime: CampaignRuntime, label: str) -> None:
    if runtime.hooks.interrupt_after == label:
        raise CollectorError("campaign_incomplete", "interrupted")


def run_cell(runtime: CampaignRuntime, cell_id: str, ie: Path, ref: Path, package: Path) -> None:
    spec = CELL_SPECS[cell_id]
    fixture = fixture_for_cell(runtime, cell_id)
    user = fixture.get("user_text_utf8", "")
    max_new = 1 if spec["max_new_tokens"] is None else spec["max_new_tokens"]
    threads = 1 if spec["threads"] is None else spec["threads"]
    maybe_interrupt(runtime, f"{cell_id}-pre")

    def one_shot(implementation: str, executable: Path, extra: dict[str, Any], dest: Path) -> dict[str, Any]:
        worker = runtime.start_worker(implementation, executable, package)
        try:
            identity_request(worker, implementation)
            if cell_id != "L01":
                load_request(worker)
            response = generation_request(worker, cell_id, user, max_new, threads)
            record = make_logical_record(runtime, extra["record_class"], cell_id, implementation, worker, response, extra)
            publish_bytes(runtime, dest, canonical_json_bytes(record))
            return record
        finally:
            worker.close()
            if worker in runtime.workers:
                runtime.workers.remove(worker)

    assert runtime.root is not None
    if cell_id == "L01":
        for impl, executable in (("inferedge", ie), ("reference", ref)):
            for index in range(runtime.warmup_count):
                dest = runtime.root / f"warmup/{cell_id}/{impl}/warmup-{index}.json"
                one_shot(impl, executable, {"record_class": "warmup", "warmup_index": index}, dest)
        trials = [
            trial
            for trial in expanded_trials(runtime.cell_order, runtime.measured_pairs)
            if trial["cell_id"] == cell_id
        ]
        pair_records: dict[int, dict[str, dict[str, Any]]] = {}
        for trial in trials:
            dest = (
                runtime.root
                / f"measured/{cell_id}/pair-{trial['pair_index']:02d}/{trial['implementation']}.json"
            )
            extra = {
                "record_class": "measured",
                "pair_index": trial["pair_index"],
                "arm_order": trial["arm_order"],
                "order_position": trial["order_position"],
                "fixture_id": None,
                "max_new_tokens": None,
            }
            executable = ie if trial["implementation"] == "inferedge" else ref
            record = one_shot(trial["implementation"], executable, extra, dest)
            pair_records.setdefault(trial["pair_index"], {})[trial["implementation"]] = record
            if len(pair_records[trial["pair_index"]]) == 2:
                compare_generation(
                    pair_records[trial["pair_index"]]["inferedge"],
                    pair_records[trial["pair_index"]]["reference"],
                    spec,
                )
        return

    workers: dict[str, WorkerSession] = {}
    try:
        workers["inferedge"] = runtime.start_worker("inferedge", ie, package)
        workers["reference"] = runtime.start_worker("reference", ref, package)
        for impl, worker in workers.items():
            identity_request(worker, impl)
            load_request(worker)
        for impl, worker in workers.items():
            for index in range(runtime.warmup_count):
                response = generation_request(worker, cell_id, user, max_new, threads)
                record = make_logical_record(
                    runtime,
                    "warmup",
                    cell_id,
                    impl,
                    worker,
                    response,
                    {"warmup_index": index},
                )
                dest = runtime.root / f"warmup/{cell_id}/{impl}/warmup-{index}.json"
                publish_bytes(runtime, dest, canonical_json_bytes(record))
        pair_records: dict[int, dict[str, dict[str, Any]]] = {}
        for trial in expanded_trials(runtime.cell_order, runtime.measured_pairs):
            if trial["cell_id"] != cell_id:
                continue
            impl = trial["implementation"]
            response = generation_request(workers[impl], cell_id, user, max_new, threads)
            extra = {
                "pair_index": trial["pair_index"],
                "arm_order": trial["arm_order"],
                "order_position": trial["order_position"],
                "fixture_id": spec["fixture_id"],
                "max_new_tokens": spec["max_new_tokens"],
            }
            record = make_logical_record(runtime, "measured", cell_id, impl, workers[impl], response, extra)
            dest = runtime.root / f"measured/{cell_id}/pair-{trial['pair_index']:02d}/{impl}.json"
            publish_bytes(runtime, dest, canonical_json_bytes(record))
            pair_records.setdefault(trial["pair_index"], {})[impl] = record
            if len(pair_records[trial["pair_index"]]) == 2:
                compare_generation(pair_records[trial["pair_index"]]["inferedge"], pair_records[trial["pair_index"]]["reference"], spec)
    finally:
        for worker in workers.values():
            worker.close()
            if worker in runtime.workers:
                runtime.workers.remove(worker)


def build_manifest(runtime: CampaignRuntime, extra: dict[str, Any]) -> dict[str, Any]:
    trials = expanded_trials()
    return {
        "schema_family": SCHEMA_FAMILY,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": extra["campaign_id"],
        "campaign_class": extra["campaign_class"],
        "source_commit": extra["source_commit"],
        "protocol_document_sha256": extra["protocol_document_sha256"],
        "fixtures_sha256": extra["fixtures_sha256"],
        "cell_inventory": list(CONSTRUCTION_CELLS),
        "golden_cell_order": list(GOLDEN_CELL_ORDER),
        "pair_order_rule": PAIR_ORDER_RULE,
        "expanded_trials": trials,
        "warmup_count_per_worker": WARMUP_COUNT,
        "measured_pairs_per_cell": MEASURED_PAIRS,
        "estimator": "mean_paired_difference",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "absolute_threshold_ns": ABSOLUTE_THRESHOLD_NS,
        "relative_threshold_numerator": RELATIVE_THRESHOLD_NUMERATOR,
        "relative_threshold_denominator": RELATIVE_THRESHOLD_DENOMINATOR,
        "relative_threshold_display": 0.05,
        "model_package_directory": PACKAGE_DIRECTORY,
        "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
        "gguf_path": f"{PACKAGE_DIRECTORY}/{GGUF_FILE}",
        "gguf_sha256": GGUF_SHA256,
        "gguf_size_bytes": GGUF_SIZE_BYTES,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "INFEREDGE_LLAMA_CPP_ROOT": LLAMA_PREFIX,
        "provenance_path": PROVENANCE_PATH,
        "provenance_size": PROVENANCE_SIZE,
        "provenance_sha256": PROVENANCE_SHA256,
        "llama_tag": LLAMA_TAG,
        "llama_commit": LLAMA_COMMIT,
        "llama_archive_sha256": LLAMA_ARCHIVE_SHA256,
        "libllama_sha256": LIBLLAMA_SHA256,
        "libggml_sha256": LIBGGML_SHA256,
        "libggml_cpu_sha256": LIBGGML_CPU_SHA256,
        "libggml_base_sha256": LIBGGML_BASE_SHA256,
        "ie_worker_sha256": extra["ie_worker_sha256"],
        "ref_worker_sha256": extra["ref_worker_sha256"],
        "collector_script_sha256": extra["collector_script_sha256"],
        "analyzer_script_sha256": extra["analyzer_script_sha256"],
        "host_system": extra["host_system"],
        "host_kernel_release": extra["host_kernel_release"],
        "host_architecture": extra["host_architecture"],
        "host_model": extra["host_model"],
        "os_product_name": extra["os_product_name"],
        "os_product_version": extra["os_product_version"],
        "os_build_version": extra["os_build_version"],
        "compiler_version_first_line": extra["compiler_version_first_line"],
        "compiler_version_stdout_utf8": extra["compiler_version_stdout_utf8"],
        "compiler_version_stdout_sha256": extra["compiler_version_stdout_sha256"],
        "environment_policy": ENVIRONMENT_POLICY,
        "evidence_path_contract": EVIDENCE_PATH_CONTRACT,
        "raw_schema_version": 1,
        "build_type": extra["build_type"],
    }


def preflight_collect(args: argparse.Namespace, hooks: CollectHooks) -> CampaignRuntime:
    if args.mode != "collect":
        raise CollectorError("campaign_incomplete", "collect mode required")
    if args.campaign_class not in {"campaign", "dry_run"}:
        raise CollectorError("campaign_class_invalid", "unsupported campaign class")
    if args.contract != EVIDENCE_PATH_CONTRACT:
        raise CollectorError("manifest_invalid", "evidence_path_contract")
    if not HEX40.fullmatch(args.source_commit or ""):
        raise CollectorError("identity_mismatch", "source_commit")
    protocol = Path(args.protocol_document)
    fixtures = Path(args.fixtures)
    sidecar = Path(args.fixtures_sha256) if args.fixtures_sha256 else Path(str(fixtures) + ".sha256")
    package = Path(args.model_package)
    ie_worker = Path(args.ie_worker)
    ref_worker = Path(args.ref_worker)
    compiler = Path(args.compiler)
    root = Path(args.evidence_root)
    reject_protected_lexical_path(root)
    reject_symlink_ancestors(root.parent)
    reject_symlink_ancestors(protocol)
    require_nonexistent_or_empty_root(root)
    if (root / "analysis").exists():
        raise CollectorError("campaign_incomplete", "pre-existing analysis directory")
    protocol_bytes = protocol.read_bytes()
    protocol_sha = sha256_bytes(protocol_bytes)
    by_id, fixtures_sha = validate_fixtures(fixtures, sidecar, hooks)
    validate_package(package, hooks)
    ie_version = validate_worker_binary(ie_worker, "inferedge-overhead-ie")
    ref_version = validate_worker_binary(ref_worker, "inferedge-overhead-ref")
    first, compiler_text, compiler_sha = capture_compiler(compiler)
    host = capture_host_fields(hooks)
    runtime = CampaignRuntime(args, hooks)
    runtime.fixtures = by_id
    runtime.identities = {
        "campaign_id": args.campaign_id,
        "campaign_class": args.campaign_class,
        "source_commit": args.source_commit,
        "protocol_document_sha256": protocol_sha,
        "fixtures_sha256": fixtures_sha,
        "ie_worker_sha256": sha256_file(ie_worker),
        "ref_worker_sha256": sha256_file(ref_worker),
        "collector_script_sha256": sha256_file(Path(__file__)),
        "analyzer_script_sha256": sha256_file(Path(__file__).with_name("analyze_generative.py")),
        "compiler_version_first_line": first,
        "compiler_version_stdout_utf8": compiler_text,
        "compiler_version_stdout_sha256": compiler_sha,
        "ie_version": ie_version,
        "ref_version": ref_version,
        "build_type": args.build_type,
        **host,
    }
    runtime.root = root
    return runtime


def run_collect(args: argparse.Namespace, hooks: CollectHooks | None = None) -> dict[str, Any]:
    resolved_hooks = hooks or CollectHooks()
    runtime = preflight_collect(args, resolved_hooks)
    assert runtime.root is not None
    previous = signal.getsignal(signal.SIGINT)
    term_previous = signal.getsignal(signal.SIGTERM)

    def handle_signal(signum: int, _frame: object) -> None:
        runtime.close_workers()
        raise CollectorError("campaign_incomplete", f"signal {signum}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        os.mkdir(runtime.root)
        fsync_directory(runtime.root.parent)
        if resolved_hooks.after_root_created is not None:
            resolved_hooks.after_root_created(runtime.root)
        for rel in (
            "warmup",
            "measured",
            "environment",
            "environment/raw",
        ):
            (runtime.root / rel).mkdir(exist_ok=True)
        manifest = build_manifest(runtime, runtime.identities)
        manifest_bytes = canonical_json_bytes(manifest)
        publish_bytes(runtime, runtime.root / "campaign_manifest.json", manifest_bytes)
        digest = detached_digest_bytes(manifest_bytes)
        publish_bytes(runtime, runtime.root / "campaign_manifest.sha256", digest)
        runtime.identities["campaign_manifest_sha256"] = sha256_bytes(manifest_bytes)
        reread = (runtime.root / "campaign_manifest.json").read_bytes()
        listed = (runtime.root / "campaign_manifest.sha256").read_text(encoding="ascii").strip()
        if sha256_bytes(reread) != listed:
            raise CollectorError("digest_mismatch", "manifest digest")
        runtime.campaign_start_ns = (resolved_hooks.clock_ns or time.time_ns)()
        build_environment_sample(runtime, "campaign-start", args.campaign_id)
        ie = Path(args.ie_worker)
        ref = Path(args.ref_worker)
        package = Path(args.model_package)
        for cell_id in runtime.cell_order:
            build_environment_sample(runtime, f"{cell_id}-pre", args.campaign_id)
            run_cell(runtime, cell_id, ie, ref, package)
            build_environment_sample(runtime, f"{cell_id}-post", args.campaign_id)
        build_environment_sample(runtime, "campaign-end", args.campaign_id)
        expected = inventory_a_relative_paths(
            cells=runtime.cell_order,
            measured_pairs=runtime.measured_pairs,
            warmup_count=runtime.warmup_count,
            cell_order=runtime.cell_order,
        )
        observed = verify_inventory_a(runtime.root, expected)
        fsync_directory(runtime.root)
        return {
            "status": "complete",
            "evidence_root": str(runtime.root),
            "observed_inventory": observed,
            "invocations": runtime.invocations,
        }
    except BaseException:
        runtime.close_workers()
        raise
    finally:
        runtime.close_workers()
        signal.signal(signal.SIGINT, previous)
        signal.signal(signal.SIGTERM, term_previous)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CollectorError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        sys.exit(1)
