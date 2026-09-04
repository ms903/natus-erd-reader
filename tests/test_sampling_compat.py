"""Sampling-rate support is metadata-driven, never a resampling operation."""

from __future__ import annotations

import builtins
import math
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from struct import pack_into
from unittest.mock import patch

import numpy as np

from natus_erd import DataIntegrityError, NatusERDReader, ReadLimits, ResourceLimitError
from natus_erd.reader import QUANTUM_UV_SCALE

from ._fixture import SHORTED, build_recording


RATES = (0.5, 128.0, 250.0, 256.0, 256.5, 500.0, 512.0, 1000.0,
         1024.0, 2048.0, 4096.0, 16384.0)


class SamplingCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, True)

    def _fixture(self, rate: float):
        directory = self.root / uuid.uuid4().hex
        directory.mkdir()
        return build_recording(directory, sample_rate=rate)

    def test_rates_preserve_samples_scaling_gaps_and_channel_order(self) -> None:
        for rate in RATES:
            with self.subTest(rate=rate):
                fixture = self._fixture(rate)
                reader = NatusERDReader.open(fixture.directory)
                self.assertEqual(reader.info.sample_rate, rate)
                self.assertEqual(reader.info.n_samples, 10)
                self.assertEqual(reader.info.duration_seconds, 10 / rate)
                selected = [183, 1, 0, 249, 0]
                expected = np.asarray([
                    [np.nan if item is None or channel in SHORTED else item
                     for item in fixture.expected[channel]]
                    for channel in selected
                ], dtype=np.float64)
                digital = reader.read_samples(0, 10, selected, units="digital")
                np.testing.assert_equal(digital, expected)
                np.testing.assert_equal(
                    reader.read_samples(0, 10, [183, "CH001", 0, "CH249", 0]),
                    expected * (QUANTUM_UV_SCALE * 64),
                )
                self.assertEqual(digital[0, 2], 131071)
                self.assertEqual(digital[2, 4], 777777)
                chunks = reader.iter_samples(
                    0, 10, chunk_samples=3, channels=selected, units="digital"
                )
                np.testing.assert_equal(np.concatenate(list(chunks), axis=1), expected)
                self.assertEqual(reader.read_samples(0, 1).shape, (256, 1))
                report = reader.validate()
                self.assertEqual((report.stored_samples, report.missing_samples), (9, 1))

    def test_rates_do_not_change_event_stamps_or_sample_coordinates(self) -> None:
        for rate in RATES:
            with self.subTest(rate=rate):
                reader = NatusERDReader.open(self._fixture(rate).directory)
                event, = reader.read_events()
                self.assertEqual((event.stamp, event.sample), (1001, 1))
                for sample in (0, 1, 5, 9, 10):
                    self.assertEqual(reader.sample_to_stamp(sample), 1000 + sample)
                    self.assertEqual(reader.stamp_to_sample(1000 + sample), sample)

    def test_nonfinite_nonpositive_and_unrepresentable_duration_rejected(self) -> None:
        for rate in (0.0, -0.0, -512.0, math.nan, math.inf, -math.inf,
                     math.nextafter(0.0, 1.0), 1e-308):
            with self.subTest(rate=rate):
                fixture = self._fixture(rate)
                with patch("numpy.full", side_effect=AssertionError("No allocation for invalid metadata")):
                    with self.assertRaises(DataIntegrityError):
                        NatusERDReader.open(fixture.directory)

    def test_extreme_finite_rate_uses_requested_samples_and_existing_budgets(self) -> None:
        for rate in (1e-100, sys.float_info.max):
            with self.subTest(rate=rate):
                fixture = self._fixture(rate)
                reader = NatusERDReader.open(fixture.directory, limits=ReadLimits(max_read_samples=2))
                self.assertTrue(math.isfinite(reader.info.duration_seconds))
                self.assertEqual(reader.read_samples(0, 2, [0]).shape, (1, 2))
                with patch("numpy.full", side_effect=AssertionError("No rate-dependent allocation")):
                    with self.assertRaises(ResourceLimitError):
                        reader.read_samples(0, 3, [0])
                    with self.assertRaises(ResourceLimitError):
                        next(reader.iter_samples(chunk_samples=3, channels=[0]))

    def test_rate_change_in_later_segment_is_rejected_before_output_allocation(self) -> None:
        fixture = self._fixture(512.0)
        second = next(path for path in fixture.directory.glob("*.erd") if path != fixture.first_erd)
        payload = bytearray(second.read_bytes())
        pack_into("<d", payload, 352, 1024.0)
        second.write_bytes(payload)
        reader = NatusERDReader.open(fixture.directory)
        self.assertEqual(reader.read_samples(0, 1, [0], units="digital")[0, 0], 1000)
        for start, stop in ((5, 6), (6, 7), (0, 10)):
            with self.subTest(bounds=(start, stop)), patch(
                "numpy.full", side_effect=AssertionError("No allocation before rejecting mixed rates")
            ):
                with self.assertRaises(DataIntegrityError) as caught:
                    reader.read_samples(start, stop, [0])
                message = str(caught.exception)
                self.assertIn("1", message)
                self.assertIn("512", message)
                self.assertIn("1024", message)
                self.assertNotIn(str(fixture.directory), message)
        with self.assertRaises(DataIntegrityError):
            reader.validate()
        self.assertNotIn(1, reader._etc_cache)

    def test_metadata_validation_and_budget_rejection_do_not_import_numpy(self) -> None:
        original_import = builtins.__import__

        def no_numpy(name, *args, **kwargs):
            if name == "numpy" or name.startswith("numpy."):
                raise AssertionError("Metadata-only operations must not import NumPy")
            return original_import(name, *args, **kwargs)

        for rate in (0.5, 512.0, 256.5, 16384.0, sys.float_info.max):
            with self.subTest(rate=rate):
                fixture = self._fixture(rate)
                with patch("builtins.__import__", side_effect=no_numpy):
                    reader = NatusERDReader.open(fixture.directory, limits=ReadLimits(max_read_bytes=8))
                    self.assertEqual(reader.info.sample_rate, rate)
                    self.assertEqual(len(reader.channels), 276)
                    self.assertEqual(len(reader.read_events()), 1)
                    self.assertEqual(reader.validate().missing_samples, 1)
                    with self.assertRaises(ResourceLimitError):
                        reader.read_samples(0, 10, [0])

    def test_open_preserves_existing_subclass_constructor_signature(self) -> None:
        class ApplicationReader(NatusERDReader):
            def __init__(self, stc_path, *, limits=ReadLimits()):
                super().__init__(stc_path, limits=limits)
                self.application_initialized = True

        fixture = self._fixture(512.0)
        reader = ApplicationReader.open(fixture.directory)
        self.assertIsInstance(reader, ApplicationReader)
        self.assertTrue(reader.application_initialized)
        self.assertEqual(reader.info.sample_rate, 512.0)
        self.assertEqual(reader.read_samples(0, 1, [0], units="digital")[0, 0], 1000)


if __name__ == "__main__":
    unittest.main()
