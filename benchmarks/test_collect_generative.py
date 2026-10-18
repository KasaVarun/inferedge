#!/usr/bin/env python3
"""Mechanics tests for the R7 generative-overhead collector."""

from __future__ import annotations

import gc
import io
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from benchmarks import collect_generative as collect
from benchmarks.collect_generative import (
    CONSTRUCTION_CELLS,
    EMPTY_SHA256,
    EVIDENCE_PATH_CONTRACT,
    GOLDEN_CELL_ORDER,
    MEASURED_PAIRS,
    WARMUP_COUNT,
    CollectorError,
    canonical_json_bytes,
    SCHEMA_VERSION,
    build_manifest,
    MAX_LPM_XML_BYTES,
    MAX_PMSET_LOG_BYTES,
    MAX_PMSET_LOG_LINE_BYTES,
    MAX_PMSET_LOG_LINES,
    PMSET_TARGET_TOKENS,
    capture_compiler,
    compare_pmset_log_baseline,
    detached_digest_bytes,
    discover_powermanagement_plist,
    encode_token_ids_le,
    environment_sample_ids,
    expanded_trials,
    inventory_a_relative_paths,
    inventory_b_relative_paths,
    parse_ac,
    parse_lpm_xml,
    parse_pmset_log_events,
    parse_sleepwake,
    permute_cells,
    publish_exclusive,
    reject_relative_path,
    render_template,
    sha256_bytes,
    token_ids_sha256,
    validate_contract,
)


class PermutationAndMatrixTests(unittest.TestCase):
    def test_golden_permutation(self) -> None:
        self.assertEqual(permute_cells(CONSTRUCTION_CELLS), GOLDEN_CELL_ORDER)

    def test_expanded_480(self) -> None:
        trials = expanded_trials()
        self.assertEqual(len(trials), 480)
        self.assertEqual(len(CONSTRUCTION_CELLS) * MEASURED_PAIRS * 2, 480)

    def test_warmup_and_pair_counts(self) -> None:
        self.assertEqual(WARMUP_COUNT, 2)
        self.assertEqual(MEASURED_PAIRS, 20)

    def test_ab_ba_order(self) -> None:
        trials = expanded_trials()
        even = [t for t in trials if t["cell_id"] == "C01" and t["pair_index"] == 0]
        odd = [t for t in trials if t["cell_id"] == "C01" and t["pair_index"] == 1]
        self.assertEqual([t["implementation"] for t in even], ["inferedge", "reference"])
        self.assertEqual([t["implementation"] for t in odd], ["reference", "inferedge"])
        self.assertEqual(even[0]["arm_order"], "ie_then_ref")
        self.assertEqual(odd[0]["arm_order"], "ref_then_ie")

    def test_complete_matrix(self) -> None:
        self.assertEqual(set(CONSTRUCTION_CELLS), set(GOLDEN_CELL_ORDER))
        self.assertEqual(len(CONSTRUCTION_CELLS), 12)

    def test_r7_contract(self) -> None:
        self.assertEqual(EVIDENCE_PATH_CONTRACT, "generative-overhead-r7-inventory-v1")
        validate_contract(EVIDENCE_PATH_CONTRACT)
        with self.assertRaises(CollectorError):
            validate_contract("generative-overhead-r6-inventory-v1")


class ProtocolIdentityTests(unittest.TestCase):
    def test_document_revision_distinct_from_json_schema(self) -> None:
        proto = Path(__file__).resolve().parents[1] / "docs/phase-9-b3-generative-overhead-protocol.md"
        text = proto.read_text(encoding="utf-8")
        self.assertIn("| Protocol document revision | integer `5` |", text)
        self.assertNotIn("| Protocol document revision | integer `4` |", text)
        self.assertNotIn("| Protocol document revision | integer `3` |", text)
        self.assertIn("| JSON schema version | integer `1` |", text)
        self.assertIn("`generative-overhead-r7-inventory-v1`", text)
        self.assertNotIn("Changing that file requires protocol version `2`", text)
        self.assertNotIn("A protocol change requires a new protocol version and a new audit.", text)
        self.assertNotRegex(text, r"protocol document revision.*integer `1`")
        self.assertNotIn("| Worker startup | `15` |", text)
        self.assertNotIn("worker startup timeout", text)
        self.assertIn("does not claim a wall-clock timeout for `Popen`", text)
        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual(EVIDENCE_PATH_CONTRACT, "generative-overhead-r7-inventory-v1")
        extra = {
            key: "x"
            for key in (
                "campaign_id",
                "campaign_class",
                "source_commit",
                "protocol_document_sha256",
                "fixtures_sha256",
                "ie_worker_sha256",
                "ref_worker_sha256",
                "collector_script_sha256",
                "analyzer_script_sha256",
                "host_system",
                "host_kernel_release",
                "host_architecture",
                "host_model",
                "os_product_name",
                "os_product_version",
                "os_build_version",
                "compiler_version_first_line",
                "compiler_version_stdout_utf8",
                "compiler_version_stdout_sha256",
                "build_type",
            )
        }
        manifest = build_manifest(None, extra)
        self.assertEqual(len(manifest), 56)
        self.assertEqual(list(manifest.keys())[1], "schema_version")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["evidence_path_contract"], EVIDENCE_PATH_CONTRACT)
        self.assertNotIn("protocol_document_revision", manifest)
        source = Path(collect.__file__).read_text(encoding="utf-8")
        self.assertIn("protocol_bytes = protocol.read_bytes()", source)
        self.assertIn("protocol_sha = sha256_bytes(protocol_bytes)", source)
        self.assertNotIn("class LaunchAttempt", source)
        self.assertNotIn("WORKER_STARTUP_TIMEOUT_S", source)
        self.assertNotIn("import threading", source)
        self.assertNotIn("worker startup timeout", source)
        for prefix in ("54b0" + "e9f0", "f09b" + "0496", "4aed" + "8616"):
            self.assertNotIn(prefix, source)


class CanonicalAndDigestTests(unittest.TestCase):
    def test_canonical_key_order(self) -> None:
        payload = canonical_json_bytes({"b": 1, "a": 2})
        self.assertEqual(payload, b'{"b":1,"a":2}\n')

    def test_token_int32_le_and_empty(self) -> None:
        self.assertEqual(encode_token_ids_le([1, -2]), b"\x01\x00\x00\x00\xfe\xff\xff\xff")
        self.assertEqual(token_ids_sha256([]), EMPTY_SHA256)

    def test_detached_digest(self) -> None:
        payload = b"abc\n"
        self.assertEqual(detached_digest_bytes(payload), (sha256_bytes(payload) + "\n").encode())

    def test_compiler_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "cc"
            fake.write_text("#!/bin/sh\nprintf 'clang version 17\\nTarget: arm64\\n'\n", encoding="utf-8")
            fake.chmod(0o755)
            first, text, digest = capture_compiler(fake)
            self.assertEqual(first, "clang version 17")
            self.assertEqual(text, "clang version 17\nTarget: arm64\n")
            self.assertEqual(digest, sha256_bytes(text.encode("utf-8")))
            self.assertEqual(text.encode("utf-8").decode("utf-8"), text)


class PathAndAtomicTests(unittest.TestCase):
    def test_invalid_paths(self) -> None:
        self.assertEqual(reject_relative_path("/abs"), "path_invalid")
        self.assertEqual(reject_relative_path("."), "path_invalid")
        self.assertEqual(reject_relative_path("a/../b"), "path_invalid")
        self.assertEqual(reject_relative_path("a\\b"), "path_invalid")
        self.assertEqual(reject_relative_path("a\x00b"), "path_invalid")
        self.assertIsNone(reject_relative_path("measured/C01/pair-00/inferedge.json"))

    def test_publish_exclusive_and_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out.json"
            publish_exclusive(dest, b"{}\n")
            self.assertEqual(dest.read_bytes(), b"{}\n")
            with self.assertRaises(CollectorError) as caught:
                publish_exclusive(dest, b"{}\n")
            self.assertEqual(caught.exception.code, "atomic_publish_failure")

    def test_atomic_link_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out.json"

            def boom(*_a: object, **_k: object) -> None:
                raise OSError("link failed")

            with mock.patch.object(collect, "os_link", boom):
                with self.assertRaises(CollectorError) as caught:
                    publish_exclusive(dest, b"x")
                self.assertEqual(caught.exception.code, "atomic_publish_failure")

    def test_file_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out.json"

            def boom(_fd: int) -> None:
                raise OSError("fsync failed")

            with mock.patch.object(collect, "fsync_file", boom):
                with self.assertRaises(OSError):
                    publish_exclusive(dest, b"x")

    def test_directory_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out.json"
            calls = {"n": 0}
            real = collect.fsync_file

            def wrapped(fd: int) -> None:
                calls["n"] += 1
                if calls["n"] > 1:
                    raise OSError("dir fsync")
                return real(fd)

            with mock.patch.object(collect, "fsync_file", wrapped):
                with self.assertRaises(OSError):
                    publish_exclusive(dest, b"x")

    def test_symlink_fifo_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "real.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "link.json"
            os.symlink(target, link)
            self.assertTrue(collect.path_is_symlink(link))
            self.assertEqual(collect.require_regular_file(link), "symlink_rejected")
            fifo = root / "fifo"
            os.mkfifo(fifo)
            self.assertEqual(collect.require_regular_file(fifo), "path_invalid")
            hard = root / "hard.json"
            os.link(target, hard)
            self.assertEqual(collect.require_regular_file(target), "hardlink_rejected")
            self.assertEqual(collect.require_regular_file(hard), "hardlink_rejected")

    def test_stale_legal_temp_ignored_by_grammar(self) -> None:
        self.assertTrue(collect.TEMP_NAME_RE.fullmatch(".out.json.pub." + ("a" * 32) + ".tmp"))
        self.assertIsNone(collect.TEMP_NAME_RE.fullmatch("out.tmp"))

    def test_help_does_not_collect(self) -> None:
        rc = collect.main([])
        self.assertEqual(rc, 0)

    def test_collect_requires_args(self) -> None:
        with self.assertRaises(CollectorError) as caught:
            collect.main(["--mode", "collect"])
        self.assertEqual(caught.exception.code, "campaign_incomplete")


class EnvironmentParseTests(unittest.TestCase):
    def test_ac_ok_and_fail(self) -> None:
        line, ok = parse_ac(b"Now drawing from 'AC Power'\n")
        self.assertTrue(ok)
        self.assertIn("AC Power", line)
        with self.assertRaises(CollectorError) as caught:
            parse_ac(b"Now drawing from 'Battery Power'\n")
        self.assertEqual(caught.exception.code, "environment_ac_invalid")

    def test_lpm_structural_and_value(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>AC Power</key><dict><key>LowPowerMode</key><integer>0</integer></dict></dict></plist>
"""
        self.assertEqual(parse_lpm_xml(xml), 0)
        bad = xml.replace("<integer>0</integer>", "<true/>")
        with self.assertRaises(CollectorError) as caught:
            parse_lpm_xml(bad)
        self.assertEqual(caught.exception.code, "environment_lpm_invalid")

    def test_sleepwake_and_log(self) -> None:
        ioreg = b'"SleepWakeUUID" = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"\n"Last Sleep Reason" = "Idle"\n"Wake Reason" = "EC.LidOpen"\n'
        uuid_text, sleep, wake = parse_sleepwake(ioreg)
        self.assertEqual(uuid_text.startswith("AAAAAAAA"), True)
        self.assertEqual(sleep, "Idle")
        self.assertEqual(wake, "EC.LidOpen")
        with self.assertRaises(CollectorError) as caught:
            parse_pmset_log_events(b"2026-10-17 10:00:00 +0000 Sleep foo\n", 1)
        self.assertEqual(caught.exception.code, "environment_sleepwake_invalid")
        self.assertEqual(caught.exception.message, "sleep/wake during campaign")

    def test_samples_and_raw_count(self) -> None:
        self.assertEqual(len(environment_sample_ids()), 26)
        self.assertEqual(len(inventory_a_relative_paths()), 920)
        self.assertEqual(len(inventory_b_relative_paths()), 922)


def _plist_xml(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        f'<plist version="1.0">{body}</plist>\n'
    )


def _ac_lpm_xml(lpm_inner: str, extra_root: str = "", extra_ac: str = "") -> str:
    return _plist_xml(
        "<dict>"
        "<key>AC Power</key><dict>"
        f"{extra_ac}<key>LowPowerMode</key>{lpm_inner}"
        "</dict>"
        f"{extra_root}"
        "</dict>"
    )


class LpmParserAndDiscoveryTests(unittest.TestCase):
    def test_ac_only_integer_zero_passes(self) -> None:
        self.assertEqual(parse_lpm_xml(_ac_lpm_xml("<integer>0</integer>")), 0)

    def test_battery_sibling_integer_zero_passes(self) -> None:
        sibling = "<key>Battery Power</key><dict><key>LowPowerMode</key><integer>1</integer></dict>"
        self.assertEqual(parse_lpm_xml(_ac_lpm_xml("<integer>0</integer>", extra_root=sibling)), 0)

    def test_sanitized_real_host_shape_passes(self) -> None:
        extra_ac = (
            "<key>Hibernate Mode</key><integer>3</integer>"
            "<key>HighPowerMode</key><integer>0</integer>"
        )
        sibling = (
            "<key>Battery Power</key><dict>"
            "<key>LowPowerMode</key><integer>0</integer>"
            "<key>Hibernate Mode</key><integer>3</integer>"
            "</dict>"
        )
        self.assertEqual(
            parse_lpm_xml(_ac_lpm_xml("<integer>0</integer>", extra_root=sibling, extra_ac=extra_ac)),
            0,
        )

    def test_ac_absent_fails(self) -> None:
        xml = _plist_xml("<dict><key>Battery Power</key><dict></dict></dict>")
        with self.assertRaises(CollectorError) as caught:
            parse_lpm_xml(xml)
        self.assertEqual(caught.exception.code, "environment_lpm_invalid")

    def test_ac_not_dictionary_fails(self) -> None:
        xml = _plist_xml("<dict><key>AC Power</key><string>nope</string></dict>")
        with self.assertRaises(CollectorError) as caught:
            parse_lpm_xml(xml)
        self.assertEqual(caught.exception.code, "environment_lpm_invalid")

    def test_lpm_absent_fails(self) -> None:
        xml = _plist_xml("<dict><key>AC Power</key><dict><key>Hibernate Mode</key><integer>3</integer></dict></dict>")
        with self.assertRaises(CollectorError) as caught:
            parse_lpm_xml(xml)
        self.assertEqual(caught.exception.code, "environment_lpm_invalid")

    def test_lpm_integer_one_and_negative_fail(self) -> None:
        for inner in ("<integer>1</integer>", "<integer>-1</integer>"):
            with self.assertRaises(CollectorError) as caught:
                parse_lpm_xml(_ac_lpm_xml(inner))
            self.assertEqual(caught.exception.code, "environment_lpm_invalid")

    def test_lpm_string_bool_float_fail(self) -> None:
        for inner in ('<string>0</string>', "<false/>", "<real>0.0</real>"):
            with self.assertRaises(CollectorError) as caught:
                parse_lpm_xml(_ac_lpm_xml(inner))
            self.assertEqual(caught.exception.code, "environment_lpm_invalid")

    def test_duplicate_ac_and_lpm_keys_fail(self) -> None:
        dup_root = _plist_xml(
            "<dict><key>AC Power</key><dict><key>LowPowerMode</key><integer>0</integer></dict>"
            "<key>AC Power</key><dict><key>LowPowerMode</key><integer>0</integer></dict></dict>"
        )
        with self.assertRaises(CollectorError) as caught:
            parse_lpm_xml(dup_root)
        self.assertEqual(caught.exception.code, "environment_lpm_invalid")
        dup_lpm = _plist_xml(
            "<dict><key>AC Power</key><dict>"
            "<key>LowPowerMode</key><integer>0</integer>"
            "<key>LowPowerMode</key><integer>0</integer>"
            "</dict></dict>"
        )
        with self.assertRaises(CollectorError) as caught:
            parse_lpm_xml(dup_lpm)
        self.assertEqual(caught.exception.code, "environment_lpm_invalid")

    def test_malformed_empty_nongdict_oversized_fail(self) -> None:
        for payload in ("", "<not-xml", _plist_xml("<array></array>")):
            with self.assertRaises(CollectorError) as caught:
                parse_lpm_xml(payload)
            self.assertEqual(caught.exception.code, "environment_lpm_invalid")
        huge = "x" * (MAX_LPM_XML_BYTES + 1)
        with self.assertRaises(CollectorError) as caught:
            parse_lpm_xml(huge)
        self.assertEqual(caught.exception.code, "environment_lpm_invalid")

    def test_plist_discovery_zero_multiple_symlink_backup(self) -> None:
        uuid_a = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
        uuid_b = "BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF"
        name_a = f"com.apple.PowerManagement.{uuid_a}.plist"
        name_b = f"com.apple.PowerManagement.{uuid_b}.plist"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(CollectorError) as caught:
                discover_powermanagement_plist(root)
            self.assertEqual(caught.exception.code, "environment_lpm_invalid")
            (root / name_a).write_bytes(b"a")
            (root / name_b).write_bytes(b"b")
            with self.assertRaises(CollectorError) as caught:
                discover_powermanagement_plist(root)
            self.assertEqual(caught.exception.code, "environment_lpm_invalid")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "real.plist"
            target.write_bytes(b"x")
            link = root / name_a
            os.symlink(target, link)
            with self.assertRaises(CollectorError) as caught:
                discover_powermanagement_plist(root)
            self.assertEqual(caught.exception.code, "environment_lpm_invalid")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / (name_a + ".backup")).write_bytes(b"stale")
            (root / name_a).write_bytes(b"live")
            self.assertEqual(discover_powermanagement_plist(root).name, name_a)

    def test_ac_remains_independent_of_lpm(self) -> None:
        with self.assertRaises(CollectorError) as caught:
            parse_ac(b"Now drawing from 'Battery Power'\n")
        self.assertEqual(caught.exception.code, "environment_ac_invalid")
        with self.assertRaises(CollectorError) as caught:
            parse_ac(b"Now drawing from 'AC Power'\n -InternalBattery-0\t80%; discharging; 1:00 remaining\n")
        self.assertEqual(caught.exception.code, "environment_ac_invalid")
        line, ok = parse_ac(b"Now drawing from 'AC Power'\n -InternalBattery-0\t100%; charged; 0:00 remaining\n")
        self.assertTrue(ok)
        self.assertEqual(line, "Now drawing from 'AC Power'")
        self.assertEqual(parse_lpm_xml(_ac_lpm_xml("<integer>0</integer>")), 0)
        with self.assertRaises(CollectorError):
            parse_lpm_xml(_ac_lpm_xml("<integer>1</integer>"))

    def test_unknown_harmless_sibling_permitted(self) -> None:
        sibling = "<key>UPS Power</key><dict><key>LowPowerMode</key><integer>1</integer></dict>"
        self.assertEqual(parse_lpm_xml(_ac_lpm_xml("<integer>0</integer>", extra_root=sibling)), 0)

    def test_sleepwake_uniqueness_unchanged(self) -> None:
        good = b'"SleepWakeUUID" = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"\n"Last Sleep Reason" = "Idle"\n"Wake Reason" = "EC.LidOpen"\n'
        self.assertEqual(parse_sleepwake(good)[0], "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")
        with self.assertRaises(CollectorError) as caught:
            parse_sleepwake(b'"Last Sleep Reason" = "Idle"\n')
        self.assertEqual(caught.exception.code, "environment_sleepwake_invalid")
        with self.assertRaises(CollectorError):
            parse_sleepwake(good + b'"SleepWakeUUID" = "BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF"\n')


class PmsetLogParserTests(unittest.TestCase):
    START_NS = 1_800_000_000_000_000_000
    HIST_SLEEP = "2026-01-01 00:00:00 +0000 Sleep hist-sleep"
    HIST_WAKE = "2026-01-01 00:00:01 +0000 Wake hist-wake"
    BASE_LOG = HIST_SLEEP + "\n" + HIST_WAKE + "\n"

    def _parse(self, text: str, start_ns: int | None = None) -> list[dict]:
        return parse_pmset_log_events(text.encode("utf-8"), self.START_NS if start_ns is None else start_ns)

    def _fail(self, text: str, message: str, start_ns: int | None = None) -> None:
        with self.assertRaises(CollectorError) as caught:
            self._parse(text, start_ns)
        self.assertEqual(caught.exception.code, "environment_sleepwake_invalid")
        self.assertEqual(caught.exception.message, message)

    def test_identical_relevant_start_end_pass(self) -> None:
        first = self._parse(self.BASE_LOG)
        second = self._parse(self.BASE_LOG)
        self.assertEqual(first, second)
        compare_pmset_log_baseline(first, second)

    def test_unrelated_assertion_appended_pass(self) -> None:
        extra = self.BASE_LOG + "2026-01-01 00:00:02 +0000 Assertions Created Wake lock\n"
        compare_pmset_log_baseline(self._parse(self.BASE_LOG), self._parse(extra))

    def test_unrelated_kernel_appended_pass(self) -> None:
        extra = self.BASE_LOG + "2026-01-01 00:00:02 +0000 Kernel Wake reason HID\n"
        compare_pmset_log_baseline(self._parse(self.BASE_LOG), self._parse(extra))

    def test_wakedetails_appended_ignored(self) -> None:
        extra = self.BASE_LOG + "2026-01-01 00:00:02 +0000 WakeDetails DarkWake from\n"
        compare_pmset_log_baseline(self._parse(self.BASE_LOG), self._parse(extra))

    def test_new_sleep_fails(self) -> None:
        extra = self.BASE_LOG + "2026-01-02 00:00:00 +0000 Sleep new\n"
        self._fail(extra, "sleep/wake during campaign", start_ns=1_767_225_700_000_000_000)

    def test_new_wake_fails(self) -> None:
        extra = self.BASE_LOG + "2026-01-02 00:00:00 +0000 Wake new\n"
        self._fail(extra, "sleep/wake during campaign", start_ns=1_767_225_700_000_000_000)

    def test_new_darkwake_fails(self) -> None:
        extra = self.BASE_LOG + "2026-01-02 00:00:00 +0000 DarkWake new\n"
        self._fail(extra, "sleep/wake during campaign", start_ns=1_767_225_700_000_000_000)

    def test_wake_versus_wakedetails(self) -> None:
        only_details = "2026-01-01 00:00:00 +0000 WakeDetails extra\n"
        self.assertEqual(self._parse(only_details), [])
        wake = self._parse("2026-01-01 00:00:00 +0000 Wake extra\n")
        self.assertEqual(wake[0]["event_token"], "Wake")

    def test_assertion_text_containing_wake_ignored(self) -> None:
        text = self.BASE_LOG + "2026-01-01 00:00:02 +0000 Assertions Released NoIdleSleepAssertion \"Wake\"\n"
        self.assertEqual([e["event_token"] for e in self._parse(text)], ["Sleep", "Wake"])

    def test_kernel_text_containing_sleep_ignored(self) -> None:
        text = self.BASE_LOG + "2026-01-01 00:00:02 +0000 Kernel Sleep transition\n"
        self.assertEqual([e["event_token"] for e in self._parse(text)], ["Sleep", "Wake"])

    def test_historical_entries_are_valid_baseline(self) -> None:
        events = self._parse(self.BASE_LOG)
        self.assertEqual([e["event_token"] for e in events], ["Sleep", "Wake"])
        self.assertTrue(all(e["aware_unix_second"] < self.START_NS // 1_000_000_000 for e in events))

    def test_historical_unchanged_at_end_pass(self) -> None:
        compare_pmset_log_baseline(self._parse(self.BASE_LOG), self._parse(self.BASE_LOG))

    def test_oldest_prefix_pruned_passes(self) -> None:
        compare_pmset_log_baseline(self._parse(self.BASE_LOG), self._parse(self.HIST_WAKE + "\n"))

    def test_newest_event_removed_fails(self) -> None:
        with self.assertRaises(CollectorError) as caught:
            compare_pmset_log_baseline(self._parse(self.BASE_LOG), self._parse(self.HIST_SLEEP + "\n"))
        self.assertEqual(caught.exception.message, "pmset log newest dropped")

    def test_empty_later_log_fails(self) -> None:
        with self.assertRaises(CollectorError) as caught:
            compare_pmset_log_baseline(self._parse(self.BASE_LOG), self._parse(""))
        self.assertEqual(caught.exception.message, "pmset log empty overlap")

    def test_reordered_relevant_events_fail(self) -> None:
        first = "2026-01-01 00:00:00 +0000 Sleep same-second\n2026-01-01 00:00:00 +0000 Wake same-second\n"
        swapped = "2026-01-01 00:00:00 +0000 Wake same-second\n2026-01-01 00:00:00 +0000 Sleep same-second\n"
        with self.assertRaises(CollectorError) as caught:
            compare_pmset_log_baseline(self._parse(first), self._parse(swapped))
        self.assertEqual(caught.exception.message, "pmset log reordered")

    def test_sleepwake_uuid_changed_still_fails(self) -> None:
        first = b'"SleepWakeUUID" = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"\n"Last Sleep Reason" = "Idle"\n"Wake Reason" = "EC.LidOpen"\n'
        second = b'"SleepWakeUUID" = "BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF"\n"Last Sleep Reason" = "Idle"\n"Wake Reason" = "EC.LidOpen"\n'
        self.assertNotEqual(parse_sleepwake(first), parse_sleepwake(second))

    def test_sleepwake_uuid_missing_fails(self) -> None:
        with self.assertRaises(CollectorError) as caught:
            parse_sleepwake(b'"Last Sleep Reason" = "Idle"\n"Wake Reason" = "EC.LidOpen"\n')
        self.assertEqual(caught.exception.code, "environment_sleepwake_invalid")

    def test_duplicate_uuid_fails(self) -> None:
        blob = (
            b'"SleepWakeUUID" = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"\n'
            b'"Last Sleep Reason" = "Idle"\n"Wake Reason" = "EC.LidOpen"\n'
            b'"SleepWakeUUID" = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"\n'
        )
        with self.assertRaises(CollectorError):
            parse_sleepwake(blob)

    def test_malformed_target_event_fails(self) -> None:
        self._fail("2026-01-01 00:00:00 Sleep missing-tz\n", "pmset log timezone")

    def test_unsupported_target_form_is_not_substring(self) -> None:
        text = "2026-01-01 00:00:00 +0000 SleepWake not-a-target\n"
        self.assertEqual(self._parse(text), [])

    def test_malformed_unrelated_date_shaped_timezone_fails(self) -> None:
        self._fail("2026-01-01 00:00:00 BADTZ Assertions x\n", "pmset log timezone")

    def test_empty_log_is_empty_baseline(self) -> None:
        self.assertEqual(self._parse(""), [])
        compare_pmset_log_baseline([], [])

    def test_invalid_utf8_fails(self) -> None:
        with self.assertRaises(CollectorError) as caught:
            parse_pmset_log_events(b"\xff\xfe", self.START_NS)
        self.assertEqual(caught.exception.message, "pmset log utf-8")

    def test_oversized_log_fails(self) -> None:
        self.assertGreater(MAX_PMSET_LOG_BYTES, 8_475_263)
        with self.assertRaises(CollectorError) as caught:
            parse_pmset_log_events(b"x" * (MAX_PMSET_LOG_BYTES + 1), self.START_NS)
        self.assertEqual(caught.exception.message, "pmset log oversized")

    def test_excessively_long_line_fails(self) -> None:
        line = "2026-01-01 00:00:00 +0000 Assertions " + ("x" * MAX_PMSET_LOG_LINE_BYTES)
        with self.assertRaises(CollectorError) as caught:
            parse_pmset_log_events(line.encode("utf-8"), self.START_NS)
        self.assertEqual(caught.exception.message, "pmset log line length")

    def test_timestamp_regression_fails(self) -> None:
        text = "2026-01-01 00:00:02 +0000 Sleep later\n2026-01-01 00:00:01 +0000 Wake earlier\n"
        self._fail(text, "pmset log timestamp regression")

    def test_event_exactly_at_start_boundary_fails(self) -> None:
        start_ns = 1_767_225_600_000_000_000
        self._fail("2026-01-01 00:00:00 +0000 Sleep boundary\n", "sleep/wake during campaign", start_ns=start_ns)

    def test_event_exactly_at_end_boundary_fails(self) -> None:
        start_ns = 1_767_225_600_000_000_000
        self._fail("2026-01-01 00:00:00 +0000 Wake boundary\n", "sleep/wake during campaign", start_ns=start_ns)

    def test_dst_timezone_transition_is_deterministic(self) -> None:
        text = "2026-11-01 01:30:00 -0400 Sleep pre\n2026-11-01 01:30:00 -0500 Wake post\n"
        events = self._parse(text)
        self.assertEqual(events[0]["aware_unix_second"] + 3600, events[1]["aware_unix_second"])
        again = self._parse(text)
        self.assertEqual(events, again)

    def test_mixed_ignored_classes_with_stable_relevant_state_pass(self) -> None:
        mixed = (
            "PM ASL data store: /var/log/powermanagement\n"
            + self.BASE_LOG
            + "2026-01-01 00:00:02 +0000 Assertions x\n"
            + "2026-01-01 00:00:02 +0000 Kernel x\n"
            + "2026-01-01 00:00:02 +0000 WakeDetails x\n"
            + "2026-01-01 00:00:02 +0000 HibernateStats x\n"
            + "2026-01-01 00:00:02 +0000 PM x\n"
            + "2026-01-01 00:00:02 +0000 WakeTime x\n"
            + "2026-01-01 00:00:02 +0000 Notification x\n"
            + "2026-01-01 00:00:02 +0000 com.apple.sleepservices.sessionStarted x\n"
            + "2026-01-01 00:00:02 +0000 com.apple.sleepservices.sessionTerminated x\n"
            + "2026-01-01 00:00:02 +0000 : Showing all\n"
            + "Sleep/Wakes since boot:12\n"
        )
        compare_pmset_log_baseline(self._parse(self.BASE_LOG), self._parse(mixed))

    def test_impossible_date_fails(self) -> None:
        self._fail("2026-02-30 00:00:00 +0000 Sleep bad\n", "pmset log impossible date")

    def test_duplicate_event_identity_fails(self) -> None:
        line = self.HIST_SLEEP + "\n" + self.HIST_SLEEP + "\n"
        self._fail(line, "pmset log duplicate event")

    def test_nul_and_cr_fail(self) -> None:
        with self.assertRaises(CollectorError) as caught:
            parse_pmset_log_events(b"x\x00y", self.START_NS)
        self.assertEqual(caught.exception.message, "pmset log binary")
        with self.assertRaises(CollectorError) as caught:
            parse_pmset_log_events(b"x\ry", self.START_NS)
        self.assertEqual(caught.exception.message, "pmset log binary")

    def test_target_token_set_and_bounds(self) -> None:
        self.assertEqual(PMSET_TARGET_TOKENS, frozenset({"Sleep", "Wake", "DarkWake"}))
        self.assertEqual(MAX_PMSET_LOG_BYTES, 16_777_216)
        self.assertEqual(MAX_PMSET_LOG_LINE_BYTES, 8192)
        self.assertEqual(MAX_PMSET_LOG_LINES, 200_000)

    def test_hermetic_large_pmset_capture_is_deterministic(self) -> None:
        tokens = ("Sleep", "Wake", "DarkWake")
        origin = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        lines: list[str] = []
        for index in range(2052):
            stamp = (origin + timedelta(seconds=index)).strftime("%Y-%m-%d %H:%M:%S")
            token = tokens[index % 3]
            unique = f"{index:04d}"
            pad = "p" * (4100 - len(unique))
            line = f"{stamp} +0000 {token} {unique}{pad}\n"
            encoded_line = line.encode("utf-8")
            self.assertLess(len(encoded_line), MAX_PMSET_LOG_LINE_BYTES)
            lines.append(line)
        raw = "".join(lines).encode("utf-8")
        snapshot = bytes(raw)
        self.assertGreater(len(raw), 8 * 1024 * 1024)
        self.assertLess(len(raw), MAX_PMSET_LOG_BYTES)
        first = parse_pmset_log_events(raw, self.START_NS)
        second = parse_pmset_log_events(raw, self.START_NS)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2052)
        self.assertEqual({event["event_token"] for event in first}, set(tokens))
        floor = self.START_NS // 1_000_000_000
        self.assertTrue(all(event["aware_unix_second"] < floor for event in first))
        self.assertEqual(raw, snapshot)

    def test_raw_before_json_on_sleepwake_parse_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            raw = collect.default_environment_raw()
            raw["pmset_log"] = (b"2026-01-02 00:00:00 +0000 Sleep during\n", b"")
            hooks = harness.hooks(environment_raw=raw)
            hooks.clock_ns = lambda: 1_767_225_600_000_000_000
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(caught.exception.code, "environment_sleepwake_invalid")
            start_raw = harness.root / "environment/raw/campaign-start"
            self.assertTrue((start_raw / "pmset_log.stdout.bin").is_file())
            self.assertFalse((harness.root / "environment/campaign-start.json").exists())
            self.assertFalse((harness.root / "analysis").exists())
            self.assertFalse((harness.root / "cells").exists())
            self.assertEqual(len(list((harness.root / "warmup").rglob("*"))), 0)


class PmsetHistoricalPruningTests(unittest.TestCase):
    START_NS = 1_800_000_000_000_000_000
    E1 = "2026-01-01 00:00:00 +0000 Sleep e1"
    E2 = "2026-01-01 00:00:01 +0000 Wake e2"
    E3 = "2026-01-01 00:00:02 +0000 DarkWake e3"
    E4 = "2026-01-01 00:00:03 +0000 Sleep e4"
    BASE = E1 + "\n" + E2 + "\n" + E3 + "\n" + E4 + "\n"
    IOREG = (
        b'"SleepWakeUUID" = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"\n'
        b'"Last Sleep Reason" = "Idle"\n"Wake Reason" = "EC.LidOpen"\n'
    )

    def _parse(self, text: str, start_ns: int | None = None) -> list[dict]:
        return parse_pmset_log_events(text.encode("utf-8"), self.START_NS if start_ns is None else start_ns)

    def _cmp(self, baseline: str, observed: str) -> None:
        compare_pmset_log_baseline(self._parse(baseline), self._parse(observed))

    def _cmp_fail(self, baseline: str, observed: str, message: str) -> None:
        with self.assertRaises(CollectorError) as caught:
            self._cmp(baseline, observed)
        self.assertEqual(caught.exception.code, "environment_sleepwake_invalid")
        self.assertEqual(caught.exception.message, message)

    def test_identical_and_oldest_prefix_drops(self) -> None:
        self._cmp(self.BASE, self.BASE)
        self._cmp(self.BASE, self.E2 + "\n" + self.E3 + "\n" + self.E4 + "\n")
        self._cmp(self.BASE, self.E3 + "\n" + self.E4 + "\n")
        self._cmp(self.BASE, self.E4 + "\n")

    def test_drop_all_or_empty_later_fails(self) -> None:
        self._cmp_fail(self.BASE, "", "pmset log empty overlap")
        self._cmp_fail(self.E1 + "\n", "", "pmset log empty overlap")

    def test_newest_middle_and_noncontiguous_drops_fail(self) -> None:
        self._cmp_fail(self.BASE, self.E1 + "\n" + self.E2 + "\n" + self.E3 + "\n", "pmset log newest dropped")
        self._cmp_fail(self.BASE, self.E1 + "\n" + self.E2 + "\n" + self.E4 + "\n", "pmset log interior deletion")
        self._cmp_fail(self.BASE, self.E1 + "\n" + self.E3 + "\n" + self.E4 + "\n", "pmset log interior deletion")
        self._cmp_fail(self.BASE, self.E2 + "\n" + self.E4 + "\n", "pmset log interior deletion")

    def test_reorder_and_mutations_fail(self) -> None:
        same = "2026-01-01 00:00:00 +0000 Sleep same\n2026-01-01 00:00:00 +0000 Wake same\n"
        swapped = "2026-01-01 00:00:00 +0000 Wake same\n2026-01-01 00:00:00 +0000 Sleep same\n"
        self._cmp_fail(same, swapped, "pmset log reordered")
        mutated_time = self.E1 + "\n" + self.E2 + "\n" + self.E3 + "\n" + self.E4.replace("00:00:03", "00:00:09") + "\n"
        self._cmp_fail(self.BASE, mutated_time, "pmset log reordered")
        self._cmp_fail(self.BASE, self.BASE.replace("+0000", "-0000", 1), "pmset log reordered")
        self._cmp_fail(self.BASE, self.BASE.replace("Sleep e1", "Wake e1", 1), "pmset log reordered")
        self._cmp_fail(self.BASE, self.BASE.replace("e1", "e1-mutated", 1), "pmset log reordered")
        dup = self.E2 + "\n" + self.E2 + "\n" + self.E3 + "\n" + self.E4 + "\n"
        with self.assertRaises(CollectorError) as caught:
            self._parse(dup)
        self.assertEqual(caught.exception.message, "pmset log duplicate event")

    def test_insert_and_append_target_events_fail(self) -> None:
        inserted = "2025-12-31 23:59:59 +0000 Sleep older\n" + self.BASE
        self._cmp_fail(self.BASE, inserted, "pmset log reappeared events")
        later_hist = self.BASE + "2026-01-01 00:00:04 +0000 Sleep later-hist\n"
        self._cmp_fail(self.BASE, later_hist, "pmset log new events")
        for token in ("Sleep", "Wake", "DarkWake"):
            extra = self.BASE + f"2027-01-16 00:00:00 +0000 {token} after\n"
            with self.assertRaises(CollectorError) as caught:
                self._parse(extra)
            self.assertEqual(caught.exception.message, "sleep/wake during campaign")
        pruned_plus_new = self.E3 + "\n" + self.E4 + "\n2027-01-16 00:00:00 +0000 Sleep after\n"
        with self.assertRaises(CollectorError) as caught:
            self._parse(pruned_plus_new)
        self.assertEqual(caught.exception.message, "sleep/wake during campaign")
        at_start = self.BASE + "2027-01-15 08:00:00 +0000 Sleep exact-floor\n"
        with self.assertRaises(CollectorError) as caught:
            self._parse(at_start, start_ns=1_800_000_000_000_000_000)
        self.assertEqual(caught.exception.message, "sleep/wake during campaign")

    def test_nontarget_append_with_prefix_prune_passes(self) -> None:
        observed = self.E3 + "\n" + self.E4 + "\n2026-01-01 00:00:04 +0000 Assertions leftover\n"
        self._cmp(self.BASE, observed)

    def test_empty_baseline_rules(self) -> None:
        compare_pmset_log_baseline([], [])
        with self.assertRaises(CollectorError) as caught:
            compare_pmset_log_baseline([], self._parse(self.E1 + "\n"))
        self.assertEqual(caught.exception.message, "pmset log new events")

    def test_timestamp_regression_and_same_length_substitution(self) -> None:
        with self.assertRaises(CollectorError) as caught:
            self._parse(self.E2 + "\n" + self.E1 + "\n")
        self.assertEqual(caught.exception.message, "pmset log timestamp regression")
        other = "2026-01-01 00:00:00 +0000 Sleep other\n2026-01-01 00:00:01 +0000 Wake other\n2026-01-01 00:00:02 +0000 DarkWake other\n2026-01-01 00:00:03 +0000 Sleep other4\n"
        self._cmp_fail(self.BASE, other, "pmset log reordered")

    def test_sequential_prefix_pruning_and_failed_sample_does_not_roll(self) -> None:
        accepted = self._parse(self.BASE)
        first = self._parse(self.E2 + "\n" + self.E3 + "\n" + self.E4 + "\n")
        compare_pmset_log_baseline(accepted, first)
        accepted = first
        second = self._parse(self.E4 + "\n")
        compare_pmset_log_baseline(accepted, second)
        accepted = second
        with self.assertRaises(CollectorError):
            compare_pmset_log_baseline(accepted, self._parse(self.BASE))
        self.assertEqual(accepted, second)
        with self.assertRaises(CollectorError):
            compare_pmset_log_baseline(accepted, self._parse(self.E3 + "\n" + self.E4 + "\n"))
        self.assertEqual(accepted, second)

    def test_reappeared_dropped_prefix_fails(self) -> None:
        rolling = self._parse(self.E3 + "\n" + self.E4 + "\n")
        with self.assertRaises(CollectorError) as caught:
            compare_pmset_log_baseline(rolling, self._parse(self.BASE))
        self.assertEqual(caught.exception.message, "pmset log reappeared events")

    def test_id004_regression_shape_accepts_six_event_prefix_prune(self) -> None:
        history = [
            "2026-01-01 00:00:00 +0000 Sleep e00",
            "2026-01-01 00:00:01 +0000 Wake e01",
            "2026-01-01 00:00:02 +0000 DarkWake e02",
            "2026-01-01 00:00:03 +0000 Sleep e03",
            "2026-01-01 00:00:04 +0000 Wake e04",
            "2026-01-01 00:00:05 +0000 DarkWake e05",
            "2026-01-01 00:00:06 +0000 Sleep e06",
            "2026-01-01 00:00:07 +0000 Wake e07",
        ]
        start_text = "\n".join(history) + "\n"
        pre_text = start_text
        post_text = "\n".join(history[6:]) + "\n"
        start_events = self._parse(start_text)
        pre_events = self._parse(pre_text)
        post_events = self._parse(post_text)
        self.assertEqual(len(start_events), 8)
        self.assertEqual({event["event_token"] for event in start_events}, {"Sleep", "Wake", "DarkWake"})
        self.assertTrue(all(event["aware_unix_second"] < self.START_NS // 1_000_000_000 for event in start_events))
        self.assertEqual(start_events, pre_events)
        self.assertEqual(post_events, start_events[-len(post_events) :])
        self.assertEqual(len(start_events) - len(post_events), 6)
        compare_pmset_log_baseline(start_events, pre_events)
        compare_pmset_log_baseline(pre_events, post_events)
        compare_pmset_log_baseline(start_events, post_events)
        start_triple = parse_sleepwake(self.IOREG)
        pre_triple = parse_sleepwake(self.IOREG)
        post_triple = parse_sleepwake(self.IOREG)
        self.assertEqual(start_triple, pre_triple)
        self.assertEqual(start_triple, post_triple)
        again = parse_pmset_log_events(post_text.encode("utf-8"), self.START_NS)
        self.assertEqual(again, post_events)

    def test_rolling_environment_sample_updates_only_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            raw = collect.default_environment_raw()
            raw["pmset_log"] = (self.BASE.encode(), b"")
            raw["ioreg_power"] = (self.IOREG, b"")
            hooks = harness.hooks(environment_raw=raw)
            runtime = collect.CampaignRuntime(collect.build_parser().parse_args(harness.argv()), hooks)
            runtime.root = Path(tmp) / "root"
            runtime.root.mkdir()
            runtime.identities = {
                "campaign_manifest_sha256": "a" * 64,
                "protocol_document_sha256": "b" * 64,
                "fixtures_sha256": "c" * 64,
                "source_commit": "d" * 40,
            }
            runtime.campaign_start_ns = self.START_NS
            collect.build_environment_sample(runtime, "campaign-start", "camp-synth")
            self.assertEqual(len(runtime.pmset_log_events or []), 4)
            raw["pmset_log"] = ((self.E3 + "\n" + self.E4 + "\n").encode(), b"")
            collect.build_environment_sample(runtime, "C01-pre", "camp-synth")
            self.assertEqual(len(runtime.pmset_log_events or []), 2)
            self.assertTrue((runtime.root / "environment/C01-pre.json").is_file())
            raw["pmset_log"] = ((self.E1 + "\n" + self.E2 + "\n").encode(), b"")
            with self.assertRaises(CollectorError):
                collect.build_environment_sample(runtime, "C01-post", "camp-synth")
            self.assertEqual(len(runtime.pmset_log_events or []), 2)
            self.assertFalse((runtime.root / "environment/C01-post.json").exists())
            self.assertTrue((runtime.root / "environment/raw/C01-post/pmset_log.stdout.bin").is_file())

    def test_new_target_after_prune_does_not_publish_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            raw = collect.default_environment_raw()
            raw["pmset_log"] = ((self.E4 + "\n2026-01-02 00:00:00 +0000 Sleep during\n").encode(), b"")
            hooks = harness.hooks(environment_raw=raw)
            hooks.clock_ns = lambda: 1_767_225_600_000_000_000
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(caught.exception.message, "sleep/wake during campaign")
            self.assertTrue((harness.root / "environment/raw/campaign-start/pmset_log.stdout.bin").is_file())
            self.assertFalse((harness.root / "environment/campaign-start.json").exists())
            self.assertFalse((harness.root / "environment/campaign-end.json").exists())
            self.assertEqual(len(list((harness.root / "warmup").rglob("*.json"))), 0)


class TemplateTests(unittest.TestCase):
    def test_byte_splice(self) -> None:
        template = b"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.assertEqual(len(template), 75)
        rendered = render_template(template, b"OK")
        self.assertEqual(rendered, b"<|im_start|>user\nOK<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


FAKE_WORKER = r"""#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
from pathlib import Path

behavior = os.environ.get("FAKE_WORKER_BEHAVIOR", "ok")
delay_s = float(os.environ.get("FAKE_WORKER_DELAY_S", "0") or "0")
delay_op = os.environ.get("FAKE_WORKER_DELAY_OP", "")
ignore_term = os.environ.get("FAKE_WORKER_IGNORE_TERM") == "1"
spawn_child = os.environ.get("FAKE_WORKER_SPAWN") == "1"
ignore_shutdown = os.environ.get("FAKE_WORKER_IGNORE_SHUTDOWN") == "1"
if ignore_term:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if spawn_child:
    child = os.fork()
    if child == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(30)
        raise SystemExit(0)
name = Path(sys.argv[0]).name
impl = "inferedge" if "ie" in name else "reference"
token = "inferedge-overhead-ie" if impl == "inferedge" else "inferedge-overhead-ref"
prompt_ids = [1, 2]
generated_ids = [3]
text = "x"


def emit(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if "--help" in sys.argv:
    print("Usage: fake-worker --help|--version|--worker-stdio --package PATH")
    raise SystemExit(0)
if "--version" in sys.argv:
    print(f"{token} 0.1.0 generative-overhead-r7-inventory-v1")
    raise SystemExit(0)
if "--worker-stdio" not in sys.argv:
    raise SystemExit(2)
if behavior == "launch_fail":
    raise SystemExit(2)

pid = os.getpid()
worker_id = f"{impl[0:2]}-{pid}"
for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    req = json.loads(line)
    op = req.get("op")
    if delay_s > 0 and (not delay_op or req.get("op") == delay_op):
        time.sleep(delay_s)
    if behavior == "timeout":
        time.sleep(5)
    if behavior == "just_above":
        time.sleep(0.4)
    if behavior == "exact_deadline":
        time.sleep(0.15)
    if behavior == "stderr_flood_ok":
        sys.stderr.write("e" * 4096)
        sys.stderr.flush()
    if behavior == "stderr_only":
        sys.stderr.write("stall\n")
        sys.stderr.flush()
        time.sleep(5)
        continue
    if behavior == "concurrent_fill":
        sys.stderr.write("e" * 8192)
        sys.stderr.flush()
        emit({"ok": True, "implementation": impl, "worker_instance_id": worker_id, "pid": pid, "evidence_path_contract": "generative-overhead-r7-inventory-v1"})
        continue
    if behavior == "stderr_before_stdout":
        sys.stderr.write("e" * 4096)
        sys.stderr.flush()
        time.sleep(0.02)
    if behavior == "stderr_never_close":
        emit({"ok": True, "implementation": impl, "worker_instance_id": worker_id, "pid": pid, "evidence_path_contract": "generative-overhead-r7-inventory-v1"})
        continue
    if behavior == "partial_plus_stderr":
        sys.stdout.write('{"ok":true')
        sys.stdout.flush()
        sys.stderr.write("e" * 2048)
        sys.stderr.flush()
        time.sleep(5)
        continue
    if behavior == "unread_stderr_exit":
        sys.stderr.write("e" * 1024)
        sys.stderr.flush()
        raise SystemExit(0)
    if behavior == "eof":
        raise SystemExit(0)
    if behavior == "crash":
        os._exit(1)
    if behavior == "nonzero":
        raise SystemExit(3)
    if behavior == "signal":
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1)
    if behavior == "partial":
        sys.stdout.write('{"ok":true')
        sys.stdout.flush()
        time.sleep(5)
    if behavior == "extra":
        emit({"ok": True, "implementation": impl, "worker_instance_id": worker_id, "pid": pid, "evidence_path_contract": "generative-overhead-r7-inventory-v1"})
        emit({"ok": True})
        continue
    if behavior == "oversized_stdout":
        sys.stdout.write("x" * (1024 * 1024 + 8) + "\n")
        sys.stdout.flush()
        continue
    if behavior == "oversized_stderr":
        emit({"ok": True, "implementation": impl, "worker_instance_id": worker_id, "pid": pid, "evidence_path_contract": "generative-overhead-r7-inventory-v1"})
        sys.stderr.write("e" * (1024 * 1024 + 8))
        sys.stderr.flush()
        continue
    if behavior == "invalid_utf8":
        sys.stdout.buffer.write(b"\xff\n")
        sys.stdout.buffer.flush()
        continue
    if behavior == "malformed":
        sys.stdout.write("{not-json\n")
        sys.stdout.flush()
        continue
    if behavior == "duplicate_key":
        sys.stdout.write('{"ok":true,"ok":false}\n')
        sys.stdout.flush()
        continue
    if behavior == "broken_pipe":
        sys.stdout.close()
        raise SystemExit(0)
    if op == "identity":
        identity_impl = "wrong" if behavior == "wrong_identity" else impl
        contract = "other-contract" if behavior == "wrong_contract" else "generative-overhead-r7-inventory-v1"
        emit({"ok": True, "implementation": identity_impl, "worker_instance_id": worker_id, "pid": pid, "evidence_path_contract": contract})
        continue
    if op == "shutdown":
        emit({"ok": True})
        if ignore_shutdown:
            continue
        break
    if op == "load":
        emit({"ok": True, "error_code": "none", "error_message": "", "worker_instance_id": worker_id, "pid": pid})
        continue
    if op in {"warm_request", "l01"}:
        local_prompt = [] if op == "l01" else list(prompt_ids)
        local_gen = [] if op == "l01" else list(generated_ids)
        local_text = "" if op == "l01" else text
        finish = "load_success" if op == "l01" else "maximum_tokens"
        if behavior == "unknown_field":
            payload = {
                "ok": True,
                "error_code": "none",
                "error_message": "",
                "finish_reason": finish,
                "prompt_token_ids": local_prompt,
                "generated_token_ids": local_gen,
                "generated_text_utf8": local_text,
                "prompt_token_count": len(local_prompt),
                "generated_token_count": len(local_gen),
                "threads_requested": 1,
                "threads_effective": 1,
                "n_ctx": 4096,
                "n_batch": 4096,
                "n_ubatch": 512,
                "n_seq_max": 1,
                "worker_instance_id": worker_id,
                "pid": pid,
                "t0_ns": 1,
                "t1_ns": 3,
                "duration_ns": 2,
                "unexpected": 1,
            }
            emit(payload)
            continue
        if behavior == "missing_key":
            emit({"ok": True, "error_code": "none"})
            continue
        if behavior == "wrong_token_count":
            local_prompt = []
        if behavior == "wrong_digest":
            local_prompt = [9, 9]
        if behavior == "invalid_stop":
            finish = "nope"
        t0, t1, duration = 1, 3, 2
        if behavior == "negative_timing":
            t0, t1, duration = -1, 3, 4
        if behavior == "noninteger_timing":
            emit({
                "ok": True,
                "error_code": "none",
                "error_message": "",
                "finish_reason": finish,
                "prompt_token_ids": local_prompt,
                "generated_token_ids": local_gen,
                "generated_text_utf8": local_text,
                "prompt_token_count": len(local_prompt),
                "generated_token_count": len(local_gen),
                "threads_requested": 1,
                "threads_effective": 1,
                "n_ctx": 4096,
                "n_batch": 4096,
                "n_ubatch": 512,
                "n_seq_max": 1,
                "worker_instance_id": worker_id,
                "pid": pid,
                "t0_ns": 1.5,
                "t1_ns": 3,
                "duration_ns": 2,
            })
            continue
        if behavior == "nonfinite_timing":
            emit({
                "ok": True,
                "error_code": "none",
                "error_message": "",
                "finish_reason": finish,
                "prompt_token_ids": local_prompt,
                "generated_token_ids": local_gen,
                "generated_text_utf8": local_text,
                "prompt_token_count": len(local_prompt),
                "generated_token_count": len(local_gen),
                "threads_requested": 1,
                "threads_effective": 1,
                "n_ctx": 4096,
                "n_batch": 4096,
                "n_ubatch": 512,
                "n_seq_max": 1,
                "worker_instance_id": worker_id,
                "pid": pid,
                "t0_ns": float("nan"),
                "t1_ns": 3,
                "duration_ns": 2,
            })
            continue
        if behavior == "invalid_timing":
            t0, t1, duration = 5, 3, 2
        payload = {
            "ok": True,
            "error_code": "none",
            "error_message": "",
            "finish_reason": finish,
            "prompt_token_ids": local_prompt,
            "generated_token_ids": local_gen,
            "generated_text_utf8": local_text,
            "prompt_token_count": len(local_prompt),
            "generated_token_count": len(local_gen),
            "threads_requested": 1,
            "threads_effective": 1,
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 512,
            "n_seq_max": 1,
            "worker_instance_id": worker_id,
            "pid": pid,
            "t0_ns": t0,
            "t1_ns": t1,
            "duration_ns": duration,
        }
        emit(payload)
        continue
    emit({"ok": False, "error_code": "internal", "error_message": "unknown op"})
"""


def _write(path: Path, text: str, mode: int = 0o644) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def _fixture_sidecar() -> dict:
    fixtures = []
    for fixture_id, user, ids in (
        ("P_tiny", "OK", [1, 2]),
        ("P_128", "alpha", [1]),
        ("P_512", "beta", [1]),
        ("P_1024", "gamma", [1]),
        ("P_2048", "delta", [1]),
        ("P_near", "epsilon", [1]),
    ):
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "user_text_utf8": user,
                "user_text_sha256": collect.sha256_bytes(user.encode()),
                "rendered_text_utf8": user,
                "rendered_text_sha256": collect.sha256_bytes(user.encode()),
                "official_token_ids": ids,
                "inferedge_token_ids": ids,
                "expected_token_count": len(ids),
            }
        )
    return {
        "schema_family": "generative_overhead",
        "schema_version": 1,
        "fixture_generation_algorithm": collect.FIXTURE_GENERATION_ALGORITHM,
        "fixture_generation_version": 1,
        "model_id": "qwen3-0.6b",
        "fixtures": fixtures,
    }


def _host_fields() -> dict:
    return {
        "host_system": "Darwin",
        "host_kernel_release": "25.0.0",
        "host_architecture": "arm64",
        "host_model": "TestMac",
        "os_product_name": "macOS",
        "os_product_version": "26.0",
        "os_build_version": "25A000",
    }


class CollectHarness:
    def __init__(self, tmp: Path, behavior: str = "ok") -> None:
        self.tmp = tmp
        self.behavior = behavior
        self.protocol = tmp / "protocol.md"
        self.fixtures = tmp / "fixtures.json"
        self.sidecar = tmp / "fixtures.json.sha256"
        self.package = tmp / "package"
        self.ie = tmp / "inferedge-overhead-ie"
        self.ref = tmp / "inferedge-overhead-ref"
        self.compiler = tmp / "clang"
        self.root = tmp / "evidence"
        self.package.mkdir()
        _write(self.protocol, "# protocol\n")
        payload = collect.canonical_json_bytes(_fixture_sidecar())
        _write(self.fixtures, payload.decode("utf-8"))
        _write(self.sidecar, collect.sha256_bytes(payload) + "\n")
        _write(self.ie, FAKE_WORKER, 0o755)
        _write(self.ref, FAKE_WORKER, 0o755)
        _write(self.compiler, "#!/bin/sh\nprintf 'clang version 17\\n'\n", 0o755)

    def argv(self, **overrides: str) -> list[str]:
        values = {
            "--mode": "collect",
            "--campaign-id": "camp-synth",
            "--evidence-root": str(self.root),
            "--campaign-class": "dry_run",
            "--source-commit": "a" * 40,
            "--protocol-document": str(self.protocol),
            "--fixtures": str(self.fixtures),
            "--ie-worker": str(self.ie),
            "--ref-worker": str(self.ref),
            "--compiler": str(self.compiler),
            "--contract": collect.EVIDENCE_PATH_CONTRACT,
            "--model-package": str(self.package),
        }
        for key, value in overrides.items():
            values["--" + key.replace("_", "-")] = value
        argv: list[str] = []
        for key, value in values.items():
            argv.extend([key, value])
        return argv

    def hooks(self, **kwargs: object) -> collect.CollectHooks:
        hooks = collect.CollectHooks()
        hooks.cell_order = ["C01"]
        hooks.measured_pairs = 1
        hooks.warmup_count = 1
        hooks.skip_package_hash = True
        hooks.host_fields = _host_fields()
        hooks.environment_raw = collect.default_environment_raw()
        hooks.clock_ns = lambda: 1_000_000_000
        for key, value in kwargs.items():
            setattr(hooks, key, value)
        os.environ["FAKE_WORKER_BEHAVIOR"] = self.behavior
        return hooks

    def factory(self, timeout_s: float = 2.0):
        created: list[collect.WorkerSession] = []

        def make(implementation: str, executable: Path, package: Path) -> collect.WorkerSession:
            worker = collect.WorkerSession(
                executable,
                [str(executable), "--worker-stdio", "--package", str(package)],
                implementation,
                timeout_s=timeout_s,
            )
            created.append(worker)
            return worker

        return make, created


class CollectLifecycleTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("FAKE_WORKER_BEHAVIOR", None)

    def test_import_causes_no_action(self) -> None:
        self.assertTrue(hasattr(collect, "run_collect"))
        self.assertNotEqual(collect.main.__doc__, "started")

    def test_help_and_missing_args_do_not_create_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evidence"
            collect.main([])
            with self.assertRaises(SystemExit) as help_exit:
                collect.main(["--help"])
            self.assertEqual(help_exit.exception.code, 0)
            with self.assertRaises(CollectorError):
                collect.main(["--mode", "collect"])
            self.assertFalse(root.exists())

    def test_success_path_fake_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            hooks = harness.hooks()
            factory, created = harness.factory()
            hooks.worker_factory = factory
            result = collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(result["status"], "complete")
            expected = collect.inventory_a_relative_paths(
                cells=["C01"], measured_pairs=1, warmup_count=1, cell_order=["C01"]
            )
            present = collect.walk_regular_files(harness.root)
            self.assertEqual(set(present), set(expected))
            self.assertTrue((harness.root / "campaign_manifest.json").is_file())
            digest = (harness.root / "campaign_manifest.sha256").read_text(encoding="ascii").strip()
            self.assertEqual(digest, collect.sha256_bytes((harness.root / "campaign_manifest.json").read_bytes()))
            warmup = json.loads((harness.root / "warmup/C01/inferedge/warmup-0.json").read_text(encoding="utf-8"))
            measured = json.loads((harness.root / "measured/C01/pair-00/inferedge.json").read_text(encoding="utf-8"))
            self.assertEqual(warmup["record_class"], "warmup")
            self.assertEqual(measured["record_class"], "measured")
            self.assertEqual(hooks.cell_order, ["C01"])
            self.assertEqual(created[0].implementation, "inferedge")
            self.assertGreaterEqual(len(result["invocations"]), 2)
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(caught.exception.code, "campaign_incomplete")
            self.assertEqual(set(collect.walk_regular_files(harness.root)), set(expected))

    def test_invalid_campaign_class(self) -> None:
        parser = collect.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--mode", "collect", "--campaign-class", "not-a-class"])

    def test_wrong_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            args = collect.build_parser().parse_args(harness.argv(contract="wrong-contract"))
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(args, harness.hooks())
            self.assertEqual(caught.exception.code, "manifest_invalid")
            self.assertFalse(harness.root.exists())

    def test_wrong_protocol_and_fixture_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            harness.sidecar.write_text("0" * 64 + "\n", encoding="ascii")
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), harness.hooks())
            self.assertEqual(caught.exception.code, "digest_mismatch")

    def test_invalid_model_and_worker_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            hooks = harness.hooks(skip_package_hash=False)
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(caught.exception.code, "identity_mismatch")
            _write(harness.ie, "#!/bin/sh\nprintf 'wrong-worker\\n'\n", 0o755)
            hooks = harness.hooks()
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(caught.exception.code, "identity_mismatch")

    def test_unsafe_and_protected_and_symlink_and_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            protected = Path(tmp) / "baselines" / "out"
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(
                    collect.build_parser().parse_args(harness.argv(evidence_root=str(protected))),
                    harness.hooks(),
                )
            self.assertEqual(caught.exception.code, "path_invalid")
            real = Path(tmp) / "real-parent"
            real.mkdir()
            link = Path(tmp) / "link-parent"
            os.symlink(real, link)
            dest = link / "evidence"
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(
                    collect.build_parser().parse_args(harness.argv(evidence_root=str(dest))),
                    harness.hooks(),
                )
            self.assertEqual(caught.exception.code, "symlink_rejected")
            harness.root.mkdir()
            (harness.root / "partial.json").write_text("x", encoding="utf-8")
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), harness.hooks())
            self.assertEqual(caught.exception.code, "campaign_incomplete")

    def _fail_behavior(self, behavior: str, code: str, timeout_s: float = 0.4) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp), behavior=behavior)
            hooks = harness.hooks()
            factory, created = harness.factory(timeout_s=timeout_s)
            hooks.worker_factory = factory
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(caught.exception.code, code)
            for worker in created:
                if worker.process is not None:
                    self.assertIsNotNone(worker.process.poll())

    def test_worker_failures(self) -> None:
        self._fail_behavior("timeout", "environment_command_failed")
        self._fail_behavior("crash", "environment_command_failed")
        self._fail_behavior("nonzero", "environment_command_failed")
        self._fail_behavior("signal", "environment_command_failed")
        self._fail_behavior("partial", "environment_command_failed")
        self._fail_behavior("extra", "record_schema_invalid")
        self._fail_behavior("oversized_stdout", "record_schema_invalid")
        self._fail_behavior("invalid_utf8", "utf8_invalid")
        self._fail_behavior("malformed", "record_schema_invalid")
        self._fail_behavior("duplicate_key", "record_schema_invalid")
        self._fail_behavior("unknown_field", "record_schema_invalid")
        self._fail_behavior("missing_key", "record_schema_invalid")
        self._fail_behavior("wrong_identity", "identity_mismatch")
        self._fail_behavior("wrong_token_count", "token_mismatch")
        self._fail_behavior("wrong_digest", "token_mismatch")
        self._fail_behavior("invalid_stop", "finish_reason_invalid")
        self._fail_behavior("negative_timing", "timing_invalid")
        self._fail_behavior("noninteger_timing", "timing_invalid")
        self._fail_behavior("nonfinite_timing", "timing_invalid")
        self._fail_behavior("invalid_timing", "timing_invalid")
        self._fail_behavior("broken_pipe", "environment_command_failed")

    def test_oversized_stderr_and_launch_failure(self) -> None:
        self._fail_behavior("oversized_stderr", "record_schema_invalid")
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            hooks = harness.hooks()

            def boom(*_a: object, **_k: object) -> collect.WorkerSession:
                raise collect.CollectorError("environment_command_failed", "worker launch failed")

            hooks.worker_factory = boom
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(caught.exception.code, "environment_command_failed")

    def test_publication_and_fsync_and_manifest_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            hooks = harness.hooks()

            def collide(dest: Path) -> None:
                if dest.name.endswith("inferedge.json"):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(b"{}\n")

            hooks.before_publish = collide
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(caught.exception.code, "atomic_publish_failure")
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            hooks = harness.hooks()
            with mock.patch.object(collect, "fsync_file", side_effect=OSError("file fsync")):
                with self.assertRaises(OSError):
                    collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            hooks = harness.hooks()
            real = collect.fsync_file
            calls = {"n": 0}

            def wrapped(fd: int) -> None:
                calls["n"] += 1
                if calls["n"] > 2:
                    raise OSError("dir fsync")
                return real(fd)

            with mock.patch.object(collect, "fsync_file", wrapped):
                with self.assertRaises(OSError):
                    collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            hooks = harness.hooks()
            real_pub = collect.publish_exclusive

            def fail_manifest(path: Path, payload: bytes) -> None:
                if path.name == "campaign_manifest.json":
                    raise CollectorError("atomic_publish_failure", "manifest")
                return real_pub(path, payload)

            with mock.patch.object(collect, "publish_exclusive", fail_manifest):
                with self.assertRaises(CollectorError) as caught:
                    collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
                self.assertEqual(caught.exception.code, "atomic_publish_failure")

    def test_inventory_mismatch_interrupt_and_no_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            hooks = harness.hooks(interrupt_after="C01-pre")
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(caught.exception.code, "campaign_incomplete")
            self.assertTrue(harness.root.exists())
            self.assertFalse((harness.root / "analysis").exists())
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), harness.hooks())
            self.assertEqual(caught.exception.code, "campaign_incomplete")
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            hooks = harness.hooks()

            def extra(root: Path) -> None:
                (root / "unexpected.bin").write_bytes(b"x")

            real_verify = collect.verify_inventory_a

            def mutated(root: Path, expected: list[str]) -> dict:
                extra(root)
                return real_verify(root, expected)

            with mock.patch.object(collect, "verify_inventory_a", mutated):
                with self.assertRaises(CollectorError) as caught:
                    collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
                self.assertEqual(caught.exception.code, "extra_file")


class WorkerTimeoutPolicyTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("FAKE_WORKER_BEHAVIOR", None)
        os.environ.pop("FAKE_WORKER_DELAY_S", None)
        os.environ.pop("FAKE_WORKER_DELAY_OP", None)
        os.environ.pop("FAKE_WORKER_IGNORE_TERM", None)
        os.environ.pop("FAKE_WORKER_IGNORE_SHUTDOWN", None)
        os.environ.pop("FAKE_WORKER_SPAWN", None)
        collect.WorkerSession.popen_impl = staticmethod(subprocess.Popen)

    def _session(self, tmp: Path, **kwargs: object) -> collect.WorkerSession:
        behavior = os.environ.get("FAKE_WORKER_BEHAVIOR", "ok")
        harness = CollectHarness(tmp, behavior=behavior)
        os.environ["FAKE_WORKER_BEHAVIOR"] = behavior
        return collect.WorkerSession(
            harness.ie,
            [str(harness.ie), "--worker-stdio", "--package", str(harness.package)],
            "inferedge",
            **kwargs,
        )

    def _set_target_request_timeout(self, worker: collect.WorkerSession, timeout_s: float = 0.08) -> None:
        worker.request_timeout_s = timeout_s
        worker.timeout_s = timeout_s

    def _assert_worker_reaped(self, worker: collect.WorkerSession, pid: int) -> None:
        self.assertIsNone(worker.process)
        self.assertTrue(worker.pending_empty())
        self.assertIsNone(worker.cleanup_error)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        self.fail("process group still exists")

    def test_delayed_response_below_timeout_passes_after_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_DELAY_S"] = "0.05"
            os.environ["FAKE_WORKER_DELAY_OP"] = "load"
            worker = self._session(Path(tmp), request_timeout_s=5.0)
            assert worker.process is not None
            pid = worker.process.pid
            try:
                identity = worker.request({"op": "identity"})
                self.assertTrue(identity["ok"])
                sent_before_target = worker.requests_sent
                self.assertEqual(sent_before_target, 1)
                self.assertEqual(os.environ.get("FAKE_WORKER_DELAY_OP"), "load")
                worker.request_timeout_s = 0.25
                worker.timeout_s = 0.25
                document = worker.request({"op": "load"})
                self.assertTrue(document["ok"])
                self.assertEqual(worker.requests_sent, sent_before_target + 1)
                self.assertEqual(worker.requests_sent, 2)
            finally:
                worker.close()
            self._assert_worker_reaped(worker, pid)
            self.assertIsNone(worker.cleanup_error)

    def test_response_exactly_at_deadline_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "exact_deadline"
            worker = self._session(Path(tmp), timeout_s=0.15)
            try:
                with self.assertRaises(CollectorError) as caught:
                    worker.request({"op": "identity"})
                self.assertEqual(caught.exception.code, "environment_command_failed")
                self.assertEqual(caught.exception.message, "worker request timeout")
            finally:
                worker.close()

    def test_response_just_above_timeout_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "just_above"
            worker = self._session(Path(tmp), timeout_s=0.15)
            try:
                with self.assertRaises(CollectorError) as caught:
                    worker.request({"op": "identity"})
                self.assertEqual(caught.exception.message, "worker request timeout")
            finally:
                worker.close()

    def test_delayed_popen_is_owned_without_startup_timeout(self) -> None:
        created: list[subprocess.Popen[bytes]] = []

        def delayed_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            time.sleep(0.05)
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            created.append(process)
            return process

        collect.WorkerSession.popen_impl = staticmethod(delayed_popen)
        with tempfile.TemporaryDirectory() as tmp:
            worker = self._session(Path(tmp), request_timeout_s=0.4, skip_executable_check=True)
            try:
                self.assertIsNotNone(worker.process)
                self.assertIsNone(worker.process.poll())
            finally:
                worker.close()
            self.assertTrue(worker.pending_empty())
        for process in created:
            self.assertIsNotNone(process.poll())

    def test_load_and_generate_request_timeout_messages(self) -> None:
        def run_delayed_op_timeout(
            delay_op: str,
            setup_ops: list[dict[str, object]],
            target: dict[str, object],
            expected_sent: int,
        ) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["FAKE_WORKER_DELAY_S"] = "0.3"
                os.environ["FAKE_WORKER_DELAY_OP"] = delay_op
                worker = self._session(Path(tmp), request_timeout_s=5.0)
                assert worker.process is not None
                pid = worker.process.pid
                try:
                    identity = worker.request({"op": "identity"})
                    self.assertTrue(identity["ok"])
                    for payload in setup_ops:
                        document = worker.request(payload)
                        self.assertTrue(document.get("ok"))
                    sent_before_target = worker.requests_sent
                    self._set_target_request_timeout(worker, 0.08)
                    with self.assertRaises(CollectorError) as caught:
                        worker.request(target)
                    self.assertEqual(caught.exception.message, "worker request timeout")
                    self.assertEqual(target["op"], delay_op)
                    self.assertEqual(worker.requests_sent, sent_before_target + 1)
                    self.assertEqual(worker.requests_sent, expected_sent)
                finally:
                    worker.close()
                self._assert_worker_reaped(worker, pid)

        run_delayed_op_timeout("load", [], {"op": "load"}, 2)
        run_delayed_op_timeout(
            "warm_request",
            [{"op": "load"}],
            {"op": "warm_request", "user_text_utf8": "OK", "max_new_tokens": 1, "threads": 1},
            3,
        )
        run_delayed_op_timeout("l01", [], {"op": "l01"}, 2)

    def test_shutdown_timeout_when_signals_are_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_IGNORE_TERM"] = "1"
            os.environ["FAKE_WORKER_IGNORE_SHUTDOWN"] = "1"
            worker = self._session(
                Path(tmp),
                request_timeout_s=5.0,
                shutdown_grace_s=0.05,
                shutdown_term_s=0.05,
                shutdown_kill_s=0.05,
            )
            worker.request({"op": "identity"})
            with mock.patch.object(collect, "_signal_owned", side_effect=lambda *_args, **_kwargs: None):
                with mock.patch.object(collect, "_process_group_exists", return_value=True):
                    with self.assertRaises(CollectorError) as caught:
                        worker.close()
                    self.assertEqual(caught.exception.message, "worker shutdown timeout")
            if worker.process is not None:
                try:
                    os.killpg(worker.process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                try:
                    worker.process.wait(timeout=1)
                except Exception:
                    pass
                collect._close_process_pipes(worker.process)
                worker.process = None

    def test_partial_stdout_then_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "partial"
            worker = self._session(Path(tmp), timeout_s=0.15)
            try:
                with self.assertRaises(CollectorError) as caught:
                    worker.request({"op": "identity"})
                self.assertEqual(caught.exception.message, "worker request timeout")
            finally:
                worker.close()

    def test_valid_stdout_with_stderr_flood_under_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "stderr_flood_ok"
            worker = self._session(Path(tmp), request_timeout_s=5.0)
            try:
                document = worker.request({"op": "identity"})
                self.assertTrue(document["ok"])
                self.assertGreater(len(worker.stderr_bytes), 0)
            finally:
                worker.close()

    def test_stderr_only_stall_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "stderr_only"
            worker = self._session(Path(tmp), timeout_s=0.15)
            try:
                with self.assertRaises(CollectorError) as caught:
                    worker.request({"op": "identity"})
                self.assertEqual(caught.exception.message, "worker request timeout")
            finally:
                worker.close()

    def test_child_ignores_sigterm_requires_sigkill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_IGNORE_TERM"] = "1"
            os.environ["FAKE_WORKER_IGNORE_SHUTDOWN"] = "1"
            worker = self._session(
                Path(tmp),
                request_timeout_s=5.0,
                shutdown_grace_s=0.05,
                shutdown_term_s=0.05,
                shutdown_kill_s=0.4,
            )
            assert worker.process is not None
            pid = worker.process.pid
            pgid = os.getpgid(pid)
            identity = worker.request({"op": "identity"})
            self.assertTrue(identity["ok"])
            worker.close()
            self._assert_worker_reaped(worker, pid)
            with self.assertRaises(ProcessLookupError):
                os.killpg(pgid, 0)

    def test_descendant_process_group_is_reaped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_SPAWN"] = "1"
            worker = self._session(Path(tmp), request_timeout_s=5.0)
            assert worker.process is not None
            pid = worker.process.pid
            identity = worker.request({"op": "identity"})
            self.assertTrue(identity["ok"])
            pgid = os.getpgid(pid)
            worker.close()
            self._assert_worker_reaped(worker, pid)
            time.sleep(0.05)
            with self.assertRaises(ProcessLookupError):
                os.killpg(pgid, 0)

    def test_nonzero_and_signal_and_eof_before_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "nonzero"
            worker = self._session(Path(tmp), request_timeout_s=5.0)
            try:
                with self.assertRaises(CollectorError) as caught:
                    worker.request({"op": "identity"})
                self.assertIn(caught.exception.message, {"worker nonzero exit 3", "unexpected EOF"})
            finally:
                worker.close()
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "signal"
            worker = self._session(Path(tmp), request_timeout_s=5.0)
            try:
                with self.assertRaises(CollectorError) as caught:
                    worker.request({"op": "identity"})
                self.assertTrue(
                    caught.exception.message.startswith("worker signal exit")
                    or caught.exception.message == "unexpected EOF"
                    or caught.exception.message == "broken pipe"
                )
            finally:
                worker.close()
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "eof"
            worker = self._session(Path(tmp), request_timeout_s=5.0)
            try:
                with self.assertRaises(CollectorError) as caught:
                    worker.request({"op": "identity"})
                self.assertEqual(caught.exception.message, "unexpected EOF")
            finally:
                worker.close()

    def test_no_response_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "timeout"
            worker = self._session(Path(tmp), timeout_s=0.12)
            try:
                with self.assertRaises(CollectorError) as caught:
                    worker.request({"op": "identity"})
                self.assertEqual(caught.exception.message, "worker request timeout")
            finally:
                worker.close()

    def test_oversized_response_before_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "oversized_stdout"
            worker = self._session(Path(tmp), request_timeout_s=5.0)
            try:
                with self.assertRaises(CollectorError) as caught:
                    worker.request({"op": "identity"})
                self.assertEqual(caught.exception.code, "record_schema_invalid")
            finally:
                worker.close()

    def test_identical_request_budget_across_backends_cells_and_l01(self) -> None:
        self.assertEqual(collect.WORKER_REQUEST_TIMEOUT_S, 300.0)
        self.assertEqual(collect.ENVIRONMENT_COMMAND_TIMEOUT_S, 30.0)
        self.assertEqual(collect.WORKER_SHUTDOWN_GRACE_S, 1.0)
        self.assertEqual(collect.WORKER_SHUTDOWN_TERM_S, 1.0)
        self.assertEqual(collect.WORKER_SHUTDOWN_KILL_S, 2.0)
        self.assertFalse(hasattr(collect, "WORKER_STARTUP_TIMEOUT_S"))
        self.assertEqual(collect.worker_request_timeout_s("C06"), 300.0)
        self.assertEqual(collect.worker_request_timeout_s("L01"), 300.0)
        self.assertEqual(collect.worker_request_timeout_s("C01"), 300.0)
        self.assertEqual(collect.worker_request_timeout_s("C08"), 300.0)
        self.assertEqual(collect.worker_identity_timeout_s(), collect.WORKER_REQUEST_TIMEOUT_S)
        source = Path(collect.__file__).read_text(encoding="utf-8")
        self.assertNotIn("worker startup timeout", source)
        self.assertNotIn("class LaunchAttempt", source)
        for cell in collect.GOLDEN_CELL_ORDER:
            self.assertEqual(collect.worker_request_timeout_s(cell), collect.WORKER_REQUEST_TIMEOUT_S)
        source = Path(collect.__file__).read_text(encoding="utf-8")
        self.assertNotIn("os.environ.get(\"WORKER_", source)
        self.assertNotIn("add_argument(\"--timeout", source)
        self.assertNotIn("add_argument('--timeout", source)

    def test_environment_command_keeps_separate_budget(self) -> None:
        self.assertNotEqual(collect.ENVIRONMENT_COMMAND_TIMEOUT_S, collect.WORKER_REQUEST_TIMEOUT_S)

        def sleeper(_argv: list[str]) -> tuple[bytes, bytes, int]:
            time.sleep(0.2)
            return b"x\n", b"", 0

        with mock.patch.object(collect, "ENVIRONMENT_COMMAND_TIMEOUT_S", 0.1):
            with self.assertRaises(CollectorError) as caught:
                collect.run_command(["/bin/sleep", "2"], collect.CollectHooks())
            self.assertEqual(caught.exception.message, "environment command timeout")
        # Injected runners are test-only and are not production timeout bypasses.
        hooks = collect.CollectHooks()
        hooks.command_runner = sleeper
        stdout, _stderr, code = collect.run_command(["unused"], hooks)
        self.assertEqual(code, 0)
        self.assertEqual(stdout, b"x\n")

    def test_timeout_does_not_publish_or_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            os.environ["FAKE_WORKER_DELAY_S"] = "0.4"
            os.environ["FAKE_WORKER_DELAY_OP"] = "warm_request"
            hooks = harness.hooks()
            factory, created = harness.factory(timeout_s=0.12)
            hooks.worker_factory = factory
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(caught.exception.code, "environment_command_failed")
            self.assertEqual(caught.exception.message, "worker request timeout")
            self.assertFalse((harness.root / "warmup/C01/inferedge/warmup-0.json").exists())
            self.assertFalse((harness.root / "measured").exists() and any((harness.root / "measured").rglob("*.json")))
            self.assertFalse((harness.root / "environment/C01-post.json").exists())
            self.assertFalse((harness.root / "environment/campaign-end.json").exists())
            self.assertEqual(created[0].implementation, "inferedge")
            for worker in created:
                self.assertIsNone(worker.process)
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), harness.hooks())
            self.assertEqual(caught.exception.code, "campaign_incomplete")

    def test_production_parser_has_no_timeout_flag(self) -> None:
        parser = collect.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--mode", "collect", "--timeout", "1"])


def _sleeper_popen(*_args: object, **_kwargs: object) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _ignore_term_popen(*_args: object, **_kwargs: object) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


class WorkerIdentityTimeoutTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("FAKE_WORKER_BEHAVIOR", None)
        os.environ.pop("FAKE_WORKER_DELAY_S", None)
        os.environ.pop("FAKE_WORKER_DELAY_OP", None)
        collect.WorkerSession.popen_impl = staticmethod(subprocess.Popen)

    def test_identity_uses_request_budget_not_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_DELAY_S"] = "0.2"
            os.environ["FAKE_WORKER_DELAY_OP"] = "identity"
            harness = CollectHarness(Path(tmp))
            os.environ["FAKE_WORKER_BEHAVIOR"] = "ok"
            worker = collect.WorkerSession(
                harness.ie,
                [str(harness.ie), "--worker-stdio", "--package", str(harness.package)],
                "inferedge",
                request_timeout_s=5.0,
            )
            assert worker.process is not None
            pid = worker.process.pid
            try:
                started = time.monotonic()
                document = worker.request({"op": "identity"})
                elapsed = time.monotonic() - started
                self.assertTrue(document["ok"])
                self.assertEqual(worker.requests_sent, 1)
                self.assertEqual(os.environ.get("FAKE_WORKER_DELAY_OP"), "identity")
                self.assertGreaterEqual(elapsed, 0.15)
            finally:
                worker.close()
            self.assertIsNone(worker.process)
            self.assertTrue(worker.pending_empty())
            self.assertIsNone(worker.cleanup_error)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                pass
            else:
                self.fail("process group still exists")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_DELAY_S"] = "0.25"
            os.environ["FAKE_WORKER_DELAY_OP"] = "identity"
            harness = CollectHarness(Path(tmp))
            worker = collect.WorkerSession(
                harness.ie,
                [str(harness.ie), "--worker-stdio", "--package", str(harness.package)],
                "reference",
                request_timeout_s=0.08,
            )
            assert worker.process is not None
            fail_pid = worker.process.pid
            try:
                with self.assertRaises(CollectorError) as caught:
                    worker.request({"op": "identity"})
                self.assertEqual(caught.exception.message, "worker request timeout")
                self.assertNotEqual(caught.exception.message, "worker startup timeout")
            finally:
                worker.close()
            self.assertTrue(worker.pending_empty())
            self.assertIsNone(worker.cleanup_error)
            with self.assertRaises(ProcessLookupError):
                os.kill(fail_pid, 0)
            try:
                os.killpg(fail_pid, 0)
            except ProcessLookupError:
                pass
            else:
                self.fail("process group still exists")

    def test_identity_partial_and_stderr_do_not_extend_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "partial"
            harness = CollectHarness(Path(tmp), behavior="partial")
            worker = collect.WorkerSession(
                harness.ie,
                [str(harness.ie), "--worker-stdio", "--package", str(harness.package)],
                "inferedge",
                timeout_s=0.12,
            )
            started = time.monotonic()
            try:
                with self.assertRaises(CollectorError) as caught:
                    worker.request({"op": "identity"})
                self.assertEqual(caught.exception.message, "worker request timeout")
            finally:
                worker.close()
            self.assertLess(time.monotonic() - started, 2.0)


def _fd_count() -> int:
    count = 0
    for fd in range(0, 512):
        try:
            os.fstat(fd)
        except OSError:
            continue
        count += 1
    return count


class WorkerDirectPopenTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("FAKE_WORKER_BEHAVIOR", None)
        os.environ.pop("FAKE_WORKER_DELAY_S", None)
        os.environ.pop("FAKE_WORKER_DELAY_OP", None)
        os.environ.pop("FAKE_WORKER_IGNORE_TERM", None)
        os.environ.pop("FAKE_WORKER_IGNORE_SHUTDOWN", None)
        os.environ.pop("FAKE_WORKER_SPAWN", None)
        collect.WorkerSession.popen_impl = staticmethod(subprocess.Popen)

    def _session(self, tmp: Path, **kwargs: object) -> collect.WorkerSession:
        harness = CollectHarness(tmp)
        kwargs.setdefault("request_timeout_s", 5.0)
        return collect.WorkerSession(
            harness.ie,
            [str(harness.ie), "--worker-stdio", "--package", str(harness.package)],
            "inferedge",
            **kwargs,
        )

    def test_successful_direct_popen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            worker = self._session(Path(tmp))
            try:
                self.assertIsNotNone(worker.process)
                self.assertIsNone(worker.process.poll())
                document = worker.request({"op": "identity"})
                self.assertTrue(document["ok"])
            finally:
                worker.close()
            self.assertTrue(worker.pending_empty())

    def test_immediate_popen_exception(self) -> None:
        def boom(*_args: object, **_kwargs: object) -> subprocess.Popen[bytes]:
            raise OSError("noexec")

        collect.WorkerSession.popen_impl = staticmethod(boom)
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            with self.assertRaises(CollectorError) as caught:
                collect.WorkerSession(harness.ie, [str(harness.ie)], "inferedge", timeout_s=0.2)
            self.assertIn("worker launch failed", caught.exception.message)

    def test_permission_missing_nonregular_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing-worker"
            with self.assertRaises(CollectorError) as caught:
                collect.require_launchable_worker(missing)
            self.assertEqual(caught.exception.code, "missing_file")
            directory = root / "dir-worker"
            directory.mkdir()
            with self.assertRaises(CollectorError) as caught:
                collect.require_launchable_worker(directory)
            self.assertEqual(caught.exception.code, "path_invalid")
            harness = CollectHarness(root)
            harness.ie.chmod(0o644)
            with self.assertRaises(CollectorError) as caught:
                collect.require_launchable_worker(harness.ie)
            self.assertEqual(caught.exception.message, "worker is not executable")
            harness.ie.chmod(0o755)
            link = root / "linked-worker"
            link.symlink_to(harness.ie)
            with self.assertRaises(CollectorError):
                collect.require_launchable_worker(link)
            with self.assertRaises(CollectorError):
                collect.require_launchable_worker(Path("relative-worker"))

    def test_process_exits_before_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "eof"
            worker = self._session(Path(tmp))
            try:
                with self.assertRaises(CollectorError) as caught:
                    collect.identity_request(worker, "inferedge")
                self.assertIn(caught.exception.message, {"unexpected EOF", "worker nonzero exit 0"})
            finally:
                worker.close()
            self.assertTrue(worker.pending_empty())

    def test_identity_hang_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_DELAY_S"] = "0.3"
            os.environ["FAKE_WORKER_DELAY_OP"] = "identity"
            worker = self._session(Path(tmp), timeout_s=0.08)
            try:
                with self.assertRaises(CollectorError) as caught:
                    collect.identity_request(worker, "inferedge")
                self.assertEqual(caught.exception.message, "worker request timeout")
            finally:
                worker.close()
            self.assertTrue(worker.pending_empty())

    def test_identity_malformed_wrong_kind_and_contract(self) -> None:
        cases = (
            ("malformed", "record_schema_invalid"),
            ("wrong_identity", "identity_mismatch"),
            ("wrong_contract", "identity_mismatch"),
        )
        for behavior, code in cases:
            with self.subTest(behavior=behavior):
                with tempfile.TemporaryDirectory() as tmp:
                    os.environ["FAKE_WORKER_BEHAVIOR"] = behavior
                    worker = self._session(Path(tmp))
                    try:
                        with self.assertRaises(CollectorError) as caught:
                            collect.identity_request(worker, "inferedge")
                        self.assertEqual(caught.exception.code, code)
                    finally:
                        worker.close()
                    self.assertTrue(worker.pending_empty())

    def test_identity_eof_nonzero_and_signal(self) -> None:
        for behavior in ("eof", "nonzero", "signal"):
            with self.subTest(behavior=behavior):
                with tempfile.TemporaryDirectory() as tmp:
                    os.environ["FAKE_WORKER_BEHAVIOR"] = behavior
                    worker = self._session(Path(tmp))
                    try:
                        with self.assertRaises(CollectorError):
                            collect.identity_request(worker, "inferedge")
                    finally:
                        worker.close()
                    self.assertTrue(worker.pending_empty())

    def test_keyboardinterrupt_and_systemexit_after_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            worker = self._session(Path(tmp))
            with mock.patch.object(worker, "request", side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    collect.identity_request(worker, "inferedge")
            self.assertTrue(worker.pending_empty())
        with tempfile.TemporaryDirectory() as tmp:
            worker = self._session(Path(tmp))
            with mock.patch.object(worker, "request", side_effect=SystemExit(2)):
                with self.assertRaises(SystemExit):
                    collect.identity_request(worker, "inferedge")
            self.assertTrue(worker.pending_empty())
        with tempfile.TemporaryDirectory() as tmp:
            worker = self._session(Path(tmp))
            with mock.patch.object(worker, "request", side_effect=RuntimeError("synthetic")):
                with self.assertRaises(RuntimeError):
                    collect.identity_request(worker, "inferedge")
            self.assertTrue(worker.pending_empty())

    def test_ignore_term_requires_kill_and_reaps_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_IGNORE_TERM"] = "1"
            os.environ["FAKE_WORKER_IGNORE_SHUTDOWN"] = "1"
            os.environ["FAKE_WORKER_SPAWN"] = "1"
            worker = self._session(Path(tmp), shutdown_grace_s=0.05, shutdown_term_s=0.05, shutdown_kill_s=0.4)
            pid = worker.process.pid
            worker.close()
            self.assertTrue(worker.pending_empty())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            with self.assertRaises(ProcessLookupError):
                os.killpg(pid, 0)

    def test_repeated_close_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            worker = self._session(Path(tmp))
            worker.request({"op": "identity"})
            worker.close()
            worker.close()
            self.assertTrue(worker.pending_empty())

    def test_protocol_claims_no_popen_timeout(self) -> None:
        proto = Path(__file__).resolve().parents[1] / "docs/phase-9-b3-generative-overhead-protocol.md"
        text = proto.read_text(encoding="utf-8")
        self.assertNotIn("worker startup timeout", text)
        self.assertIn("The first bounded liveness and protocol check is the `identity` request", text)
        self.assertFalse(hasattr(collect, "LaunchAttempt"))


class WorkerShutdownOwnershipTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("FAKE_WORKER_BEHAVIOR", None)
        os.environ.pop("FAKE_WORKER_DELAY_S", None)
        os.environ.pop("FAKE_WORKER_DELAY_OP", None)
        os.environ.pop("FAKE_WORKER_IGNORE_TERM", None)
        os.environ.pop("FAKE_WORKER_IGNORE_SHUTDOWN", None)
        os.environ.pop("FAKE_WORKER_SPAWN", None)
        collect.WorkerSession.popen_impl = staticmethod(subprocess.Popen)

    def _worker(self, tmp: Path, **kwargs: object) -> collect.WorkerSession:
        harness = CollectHarness(tmp)
        os.environ.setdefault("FAKE_WORKER_BEHAVIOR", "ok")
        kwargs.setdefault("request_timeout_s", 5.0)
        return collect.WorkerSession(
            harness.ie,
            [str(harness.ie), "--worker-stdio", "--package", str(harness.package)],
            "inferedge",
            **kwargs,
        )

    def test_cooperative_and_idempotent_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ResourceWarning)
                worker = self._worker(Path(tmp))
                identity = worker.request({"op": "identity"})
                self.assertTrue(identity["ok"])
                worker.close()
                self.assertIsNone(worker.process)
                worker.close()
                self.assertIsNone(worker.process)
                self.assertTrue(worker.pending_empty())

    def test_shutdown_timeout_retains_handle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_IGNORE_TERM"] = "1"
            os.environ["FAKE_WORKER_IGNORE_SHUTDOWN"] = "1"
            worker = self._worker(Path(tmp), shutdown_grace_s=0.05, shutdown_term_s=0.05, shutdown_kill_s=0.05)
            identity = worker.request({"op": "identity"})
            self.assertTrue(identity["ok"])
            with mock.patch.object(collect, "_signal_owned", side_effect=lambda *_args, **_kwargs: None):
                with mock.patch.object(collect, "_process_group_exists", return_value=True):
                    with self.assertRaises(CollectorError) as caught:
                        worker.close()
                    self.assertEqual(caught.exception.message, "worker shutdown timeout")
            self.assertIsNotNone(worker.process)
            self.assertIsNotNone(worker.cleanup_error)
            pid = worker.process.pid
            os.kill(pid, signal.SIGKILL)
            worker.process.wait(timeout=1)
            collect._close_process_pipes(worker.process)
            worker.process = None

    def test_request_timeout_preserves_primary_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_DELAY_S"] = "0.3"
            os.environ["FAKE_WORKER_DELAY_OP"] = "identity"
            harness = CollectHarness(Path(tmp))
            worker = collect.WorkerSession(
                harness.ie,
                [str(harness.ie), "--worker-stdio", "--package", str(harness.package)],
                "inferedge",
                request_timeout_s=0.08,
            )
            with self.assertRaises(CollectorError) as caught:
                try:
                    worker.request({"op": "identity"})
                finally:
                    worker.close()
            self.assertEqual(caught.exception.message, "worker request timeout")
            self.assertTrue(worker.pending_empty())

    def test_child_exit_before_wait_and_closed_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = "eof"
            worker = self._worker(Path(tmp))
            with self.assertRaises(CollectorError):
                worker.request({"op": "identity"})
            worker.close()
            self.assertTrue(worker.pending_empty())


class WorkerStreamStressTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("FAKE_WORKER_BEHAVIOR", None)
        collect.WorkerSession.popen_impl = staticmethod(subprocess.Popen)

    def _run(self, behavior: str, timeout_s: float = 0.2) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["FAKE_WORKER_BEHAVIOR"] = behavior
            harness = CollectHarness(Path(tmp), behavior=behavior)
            worker = collect.WorkerSession(
                harness.ie,
                [str(harness.ie), "--worker-stdio", "--package", str(harness.package)],
                "inferedge",
                timeout_s=timeout_s,
            )
            try:
                try:
                    worker.request({"op": "identity"})
                except CollectorError:
                    pass
            finally:
                worker.close()
            self.assertTrue(worker.pending_empty())

    def test_stream_cases_do_not_leak(self) -> None:
        self._run("concurrent_fill", 0.4)
        self._run("stderr_before_stdout", 0.4)
        self._run("stderr_never_close", 0.4)
        self._run("partial_plus_stderr", 0.12)
        self._run("unread_stderr_exit", 0.4)


class WorkerPublicationRegressionTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("FAKE_WORKER_BEHAVIOR", None)
        os.environ.pop("FAKE_WORKER_DELAY_S", None)
        os.environ.pop("FAKE_WORKER_DELAY_OP", None)
        collect.WorkerSession.popen_impl = staticmethod(subprocess.Popen)

    def test_identity_timeout_does_not_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = CollectHarness(Path(tmp))
            os.environ["FAKE_WORKER_DELAY_S"] = "0.3"
            os.environ["FAKE_WORKER_DELAY_OP"] = "identity"
            hooks = harness.hooks()
            factory, created = harness.factory(timeout_s=0.1)
            hooks.worker_factory = factory
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
            self.assertEqual(caught.exception.message, "worker request timeout")
            self.assertFalse((harness.root / "warmup/C01/inferedge/warmup-0.json").exists())
            self.assertFalse((harness.root / "environment/C01-post.json").exists())
            self.assertFalse((harness.root / "environment/campaign-end.json").exists())
            for worker in created:
                worker.close()
                self.assertTrue(worker.pending_empty())
            with self.assertRaises(CollectorError) as caught:
                collect.run_collect(collect.build_parser().parse_args(harness.argv()), harness.hooks())
            self.assertEqual(caught.exception.code, "campaign_incomplete")

    def test_identity_failures_publish_no_worker_or_cell_records(self) -> None:
        for behavior, delay_op in (
            ("timeout", "identity"),
            ("malformed", ""),
            ("wrong_identity", ""),
            ("eof", ""),
            ("nonzero", ""),
            ("signal", ""),
        ):
            with self.subTest(behavior=behavior):
                with tempfile.TemporaryDirectory() as tmp:
                    harness = CollectHarness(Path(tmp), behavior=behavior)
                    if delay_op:
                        os.environ["FAKE_WORKER_DELAY_S"] = "0.25"
                        os.environ["FAKE_WORKER_DELAY_OP"] = delay_op
                    hooks = harness.hooks()
                    factory, created = harness.factory(timeout_s=0.08)
                    hooks.worker_factory = factory
                    with self.assertRaises(CollectorError):
                        collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
                    self.assertFalse((harness.root / "warmup").exists() and any((harness.root / "warmup").rglob("*.json")))
                    self.assertFalse((harness.root / "measured").exists() and any((harness.root / "measured").rglob("*.json")))
                    self.assertFalse((harness.root / "environment/C01-post.json").exists())
                    self.assertFalse((harness.root / "environment/campaign-end.json").exists())
                    self.assertFalse((harness.root / "analysis").exists())
                    for worker in created:
                        worker.close()
                        self.assertTrue(worker.pending_empty())
                    with self.assertRaises(CollectorError) as caught:
                        collect.run_collect(collect.build_parser().parse_args(harness.argv()), harness.hooks())
                    self.assertEqual(caught.exception.code, "campaign_incomplete")


class WorkerResourceAndFdTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("FAKE_WORKER_BEHAVIOR", None)
        os.environ.pop("FAKE_WORKER_DELAY_S", None)
        os.environ.pop("FAKE_WORKER_DELAY_OP", None)
        collect.WorkerSession.popen_impl = staticmethod(subprocess.Popen)

    def _collect_identity_failure(self, tmp: Path, behavior: str, timeout_s: float = 0.1) -> None:
        harness = CollectHarness(tmp, behavior=behavior)
        if behavior == "timeout":
            os.environ["FAKE_WORKER_DELAY_S"] = "0.3"
            os.environ["FAKE_WORKER_DELAY_OP"] = "identity"
        hooks = harness.hooks()
        factory, created = harness.factory(timeout_s=timeout_s)
        hooks.worker_factory = factory
        with self.assertRaises(CollectorError):
            collect.run_collect(collect.build_parser().parse_args(harness.argv()), hooks)
        for worker in created:
            worker.close()
            self.assertTrue(worker.pending_empty())
            self.assertIsNone(worker.process)

    def test_run_collect_identity_failures_emit_no_resourcewarning(self) -> None:
        behaviors = ("timeout", "malformed", "wrong_identity", "eof", "nonzero", "signal")
        before = _fd_count()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            for behavior in behaviors:
                for _ in range(2):
                    with tempfile.TemporaryDirectory() as tmp:
                        self._collect_identity_failure(Path(tmp), behavior)
                    gc.collect()
        after = _fd_count()
        self.assertLessEqual(after, before + 2)

    def test_forced_gc_after_identity_timeout_is_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorded: list[warnings.WarningMessage] = []
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                self._collect_identity_failure(Path(tmp), "timeout")
                gc.collect()
                recorded.extend(caught)
            resource = [item for item in recorded if issubclass(item.category, ResourceWarning)]
            self.assertEqual(resource, [])


if __name__ == "__main__":
    unittest.main()
