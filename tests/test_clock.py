from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
import shutil
from struct import pack, pack_into
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid
from zoneinfo import ZoneInfoNotFoundError

from natus_erd.clock import ClockAnchor, ClockEstimate, SNCClock
from natus_erd.errors import DataIntegrityError, ResourceLimitError, UnsupportedFormatError
from natus_erd.limits import ReadLimits


_GUID = bytes.fromhex("d2a98660af60d311986000104b75c151")
_BASE = 116444736000000000  # Synthetic 1970-01-01 UTC, not patient metadata.


class ClockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.temporary.mkdir()
        self.addCleanup(shutil.rmtree, self.temporary, True)
        self.path = self.temporary / "同步测试.SNC"
        self.pairs = ((102, _BASE), (108, _BASE + 30_000_001), (120, _BASE + 90_000_003))
        self.write_snc(self.pairs)

    def write_snc(self, pairs, *, schema: int = 1, base: int = 1, guid: bytes = _GUID) -> None:
        data = bytearray(352)
        data[:16] = guid
        pack_into("<HHI", data, 16, schema, base, 0xFFFFFFFF)
        for stamp, ticks in pairs:
            data.extend(pack("<iQ", stamp, ticks))
        self.path.write_bytes(data)

    def open(self, **kwargs) -> SNCClock:
        options = {"sample_rate": 2.0, "start_stamp": 100, "end_stamp": 120}
        options.update(kwargs)
        return SNCClock.from_file(self.path, **options)

    def test_unicode_path_original_anchors_and_bounds(self) -> None:
        clock = self.open()
        self.assertEqual(clock.sample_rate, 2.0)
        self.assertEqual(clock.anchors, tuple(ClockAnchor(*pair) for pair in self.pairs))
        self.assertEqual(clock.n_samples, 21)
        self.assertEqual(clock.first_anchor_sample, 2)
        self.assertEqual(clock.last_anchor_sample, 20)
        for sample, ticks in ((2, _BASE), (8, _BASE + 30_000_001), (20, _BASE + 90_000_003)):
            with self.subTest(sample=sample):
                estimate = clock.at_sample(sample)
                self.assertEqual(estimate.kind, "anchor")
                self.assertIsInstance(estimate.filetime_ticks, Fraction)
                self.assertEqual(estimate.filetime_ticks, ticks)

    def test_exact_piecewise_interpolation_retains_fractional_ticks(self) -> None:
        clock = self.open()
        before = clock.at_sample(3)
        self.assertEqual(before.kind, "interpolated")
        self.assertEqual(before.filetime_ticks, _BASE + Fraction(30_000_001, 6))
        after = clock.at_sample(9)
        self.assertEqual(after.kind, "interpolated")
        self.assertEqual(after.filetime_ticks, _BASE + 30_000_001 + Fraction(60_000_002, 12))
        # Piecewise interpolation must not be replaced by nominal sample/fs.
        self.assertNotEqual(before.filetime_ticks, _BASE + 5_000_000)

    def test_default_forbids_unanchored_start_and_exclusive_end(self) -> None:
        clock = self.open()
        for sample in (0, 1, clock.n_samples):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                clock.at_sample(sample)
        start = clock.at_sample(0, extrapolation="nominal", max_extrapolation_seconds=1)
        self.assertEqual(start.kind, "extrapolated_nominal")
        self.assertEqual(start.filetime_ticks, _BASE - 10_000_000)
        end = clock.at_sample(clock.n_samples, extrapolation="nominal", max_extrapolation_seconds=0.5)
        self.assertEqual(end.filetime_ticks, _BASE + 95_000_003)
        self.assertEqual(end.kind, "extrapolated_nominal")

    def test_extrapolation_budget_is_distance_to_nearest_anchor(self) -> None:
        clock = self.open()
        with self.assertRaises(ValueError):
            clock.at_sample(0, extrapolation="nominal", max_extrapolation_seconds=0.999)
        with self.assertRaises(ValueError):
            clock.at_sample(21, extrapolation="nominal", max_extrapolation_seconds=0.499)
        clock.at_sample(2, extrapolation="nominal", max_extrapolation_seconds=0)
        with self.assertRaises(ValueError):
            clock.at_sample(1, extrapolation="nominal", max_extrapolation_seconds=0)

    def test_2048_and_noninteger_nominal_rates_use_exact_arithmetic(self) -> None:
        for rate in (512.0, 2048.0, 2.5, 0.1):
            with self.subTest(rate=rate):
                clock = self.open(sample_rate=rate)
                estimate = clock.at_sample(1, extrapolation="nominal", max_extrapolation_seconds=20)
                self.assertEqual(estimate.filetime_ticks, _BASE - Fraction(10_000_000) / Fraction.from_float(rate))
                # Changing the nominal fallback rate does not alter source anchors.
                self.assertEqual(clock.at_sample(2).filetime_ticks, _BASE)
                self.assertEqual(clock.at_sample(3).filetime_ticks, _BASE + Fraction(30_000_001, 6))

    def test_single_anchor_has_explicit_nominal_fallback(self) -> None:
        self.write_snc(((105, _BASE),))
        clock = self.open()
        self.assertEqual(clock.at_sample(5).kind, "anchor")
        with self.assertRaises(ValueError):
            clock.at_sample(6)
        self.assertEqual(clock.at_sample(6, extrapolation="nominal").filetime_ticks, _BASE + 5_000_000)

    def test_creation_field_does_not_determine_clock(self) -> None:
        first = self.open().at_sample(3)
        data = bytearray(self.path.read_bytes())
        pack_into("<I", data, 20, 1)
        self.path.write_bytes(data)
        self.assertEqual(self.open().at_sample(3), first)

    def test_beijing_default_and_explicit_timezone(self) -> None:
        estimate = self.open().at_sample(2)
        self.assertEqual(estimate.to_datetime(), datetime(1970, 1, 1, 8, tzinfo=timezone(timedelta(hours=8))))
        with self.assertRaises(TypeError):
            estimate.to_datetime(None)
        self.assertEqual(estimate.to_datetime("UTC"), datetime(1970, 1, 1, tzinfo=timezone.utc))
        target = timezone(timedelta(hours=9))
        self.assertEqual(estimate.to_datetime(target), datetime(1970, 1, 1, 9, tzinfo=target))
        self.assertIsNotNone(estimate.to_datetime(target).utcoffset())
        with self.assertRaises((ZoneInfoNotFoundError, ValueError)):
            estimate.to_datetime("Natus/NoSuchTimezone")

    def test_nearest_microsecond_half_even_and_exact_original(self) -> None:
        for ticks, microseconds in ((0, 0), (1, 0), (5, 0), (6, 1), (15, 2), (25, 2)):
            with self.subTest(ticks=ticks):
                estimate = ClockEstimate(Fraction(_BASE + ticks), "anchor")
                expected = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=microseconds)
                self.assertEqual(estimate.to_datetime("UTC"), expected)
                self.assertEqual(estimate.filetime_ticks, _BASE + ticks)
        fractional = ClockEstimate(_BASE + Fraction(1, 3), "interpolated")
        self.assertEqual(fractional.filetime_ticks - _BASE, Fraction(1, 3))
        self.assertEqual(fractional.to_datetime("UTC"), datetime(1970, 1, 1, tzinfo=timezone.utc))

    def test_models_are_immutable(self) -> None:
        clock = self.open()
        with self.assertRaises(FrozenInstanceError):
            clock.sample_rate = 1
        with self.assertRaises(FrozenInstanceError):
            clock.anchors[0].stamp = 1
        estimate = clock.at_sample(2)
        with self.assertRaises(FrozenInstanceError):
            estimate.kind = "interpolated"

    def test_invalid_schema_and_guid(self) -> None:
        for schema, base in ((2, 1), (1, 2), (0, 1)):
            with self.subTest(schema=schema, base=base):
                self.write_snc(self.pairs, schema=schema, base=base)
                with self.assertRaises(UnsupportedFormatError):
                    self.open()
        self.write_snc(self.pairs, guid=b"x" * 16)
        with self.assertRaises(DataIntegrityError):
            self.open()

    def test_empty_truncated_header_or_partial_entry(self) -> None:
        valid = self.path.read_bytes()
        for length in (0, 16, 351, 352, 353, 363, len(valid) - 1):
            with self.subTest(length=length):
                self.path.write_bytes(valid[:length])
                with self.assertRaises(DataIntegrityError):
                    self.open()

    def test_invalid_anchor_order_or_time(self) -> None:
        bad_pairs = (
            ((102, _BASE), (102, _BASE + 1)),
            ((108, _BASE), (102, _BASE + 1)),
            ((102, _BASE), (108, _BASE)),
            ((102, _BASE), (108, _BASE - 1)),
            ((102, _BASE), (108, 2**64 - 1)),
        )
        for pairs in bad_pairs:
            with self.subTest(pairs=pairs):
                self.write_snc(pairs)
                with self.assertRaises(DataIntegrityError):
                    self.open()

    def test_anchor_must_belong_to_recording(self) -> None:
        for pairs in (((99, _BASE), (108, _BASE + 1)), ((102, _BASE), (121, _BASE + 1))):
            with self.subTest(pairs=pairs):
                self.write_snc(pairs)
                with self.assertRaises(DataIntegrityError):
                    self.open()

    def test_bad_caller_rates_and_recording_bounds(self) -> None:
        for rate in (0, -1, float("nan"), float("inf"), 5e-324, 10**1000):
            with self.subTest(rate=type(rate).__name__), self.assertRaises(ValueError):
                self.open(sample_rate=rate)
        for rate in (True, "2048", None):
            with self.subTest(rate=rate), self.assertRaises(TypeError):
                self.open(sample_rate=rate)
        for fields in ({"start_stamp": 121}, {"end_stamp": 2**31}, {"start_stamp": -(2**31) - 1}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.open(**fields)
        with self.assertRaises(TypeError):
            self.open(start_stamp=True)
        with self.assertRaises(TypeError):
            self.open(limits=None)

    def test_bad_sample_and_extrapolation_parameters(self) -> None:
        clock = self.open()
        for sample in (-1, 22):
            with self.subTest(sample=sample), self.assertRaises(IndexError):
                clock.at_sample(sample)
        for sample in (True, 1.0, "1"):
            with self.subTest(sample=sample), self.assertRaises(TypeError):
                clock.at_sample(sample)
        for policy in ("guess", None):
            with self.assertRaises(ValueError):
                clock.at_sample(2, extrapolation=policy)
        for budget in (-1, float("inf"), float("nan"), 10**1000):
            with self.assertRaises(ValueError):
                clock.at_sample(2, max_extrapolation_seconds=budget)
        for budget in (True, "10", None):
            with self.assertRaises(TypeError):
                clock.at_sample(2, max_extrapolation_seconds=budget)

    def test_file_and_anchor_budgets_precede_open_or_payload_allocation(self) -> None:
        for limits in (replace(ReadLimits(), max_metadata_bytes=352), replace(ReadLimits(), max_clock_anchors=2)):
            with self.subTest(limits=limits):
                with patch("natus_erd.clock.os.open", side_effect=AssertionError("must not open over-budget file")):
                    with self.assertRaises(ResourceLimitError):
                        self.open(limits=limits)

    def test_reject_nonregular_and_missing_paths(self) -> None:
        for path in (self.temporary, self.temporary / "missing.snc"):
            with self.subTest(path=path.name), self.assertRaises(DataIntegrityError):
                SNCClock.from_file(path, sample_rate=2, start_stamp=100, end_stamp=120)

    def test_reject_symlink_paths_and_cycles(self) -> None:
        link = self.temporary / "alias.snc"
        try:
            link.symlink_to(self.path)
        except (OSError, NotImplementedError):
            self.skipTest("Symbolic-link creation is unavailable on this platform")
        with self.assertRaises(DataIntegrityError):
            SNCClock.from_file(link, sample_rate=2, start_stamp=100, end_stamp=120)
        loop = self.temporary / "loop.snc"
        loop.symlink_to(loop)
        with self.assertRaises(DataIntegrityError):
            SNCClock.from_file(loop, sample_rate=2, start_stamp=100, end_stamp=120)

    def test_public_values_reject_invalid_ticks_and_provenance(self) -> None:
        for ticks in (-1, 2**64 - 1):
            with self.assertRaises(ValueError):
                ClockAnchor(0, ticks)
            with self.assertRaises(ValueError):
                ClockEstimate(Fraction(ticks), "anchor")
        for ticks in (True, 1.0, "1"):
            with self.assertRaises(TypeError):
                ClockAnchor(0, ticks)
            with self.assertRaises(TypeError):
                ClockEstimate(ticks, "anchor")
        with self.assertRaises(ValueError):
            ClockEstimate(Fraction(_BASE), "guessed")
        with self.assertRaises(ValueError):
            SNCClock(2, 100, 120, ())
        with self.assertRaises(TypeError):
            SNCClock(2, 100, 120, ((102, _BASE),))

    def test_fresh_interpreter_metadata_work_does_not_import_numpy(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src"
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from natus_erd.clock import SNCClock; "
            "clock=SNCClock.from_file(sys.argv[2], sample_rate=2, start_stamp=100, end_stamp=120); "
            "clock.at_sample(3).to_datetime('UTC'); "
            "assert 'numpy' not in sys.modules; print('PASS')"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", code, str(source), str(self.path)],
            capture_output=True, text=True, timeout=20, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "PASS")


if __name__ == "__main__":
    unittest.main()
