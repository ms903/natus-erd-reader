"""Resource regressions use tiny files and intercept would-be large operations."""

from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError
import shutil
import unittest
import uuid
from pathlib import Path
from struct import pack_into
from unittest.mock import patch

import numpy as np

from natus_erd import DataIntegrityError, NatusERDReader, ReadLimits, ResourceLimitError
from natus_erd.binary import read_etc, read_stc

from ._fixture import build_recording


class ReadLimitTests(unittest.TestCase):
    def test_resource_limits_are_distinct_from_corruption(self) -> None:
        from natus_erd import NatusERDError
        self.assertEqual(ResourceLimitError.__bases__, (NatusERDError,))

    def setUp(self) -> None:
        self.root = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.fixture = build_recording(self.root)

    def test_limits_are_positive_integers_and_immutable(self) -> None:
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReadLimits(max_read_bytes=value)
        for value in (True, 1.5, "1024", None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                ReadLimits(max_read_bytes=value)
        with self.assertRaises(ValueError):
            ReadLimits(max_parse_depth=129)
        limits = ReadLimits(max_read_bytes=np.int64(128))
        self.assertIs(type(limits.max_read_bytes), int)
        with self.assertRaises(FrozenInstanceError):
            limits.max_read_bytes = 256

    def test_budget_rejection_precedes_numpy_import_allocation_and_packet_io(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory, limits=ReadLimits(max_read_bytes=8))
        original_import = builtins.__import__

        def no_numpy(name, *args, **kwargs):
            if name == "numpy" or name.startswith("numpy."):
                raise AssertionError("NumPy must not load for a rejected read")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_numpy), patch.object(
            reader, "_load_etc", side_effect=AssertionError("No payload indexing before budget check")
        ), patch.object(np, "full", side_effect=AssertionError("No output allocation")):
            with self.assertRaises(ResourceLimitError):
                reader.read_samples(0, 10, [0])

    def test_sample_limit_applies_even_to_one_channel(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory, limits=ReadLimits(max_read_samples=2))
        with self.assertRaises(ResourceLimitError):
            reader.read_samples(0, 3, [0])
        self.assertEqual(reader.read_samples(0, 2, [0]).shape, (1, 2))

    def test_channel_count_is_bounded_before_resolving_names(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory, limits=ReadLimits(max_selected_channels=2))
        with self.assertRaises(ResourceLimitError):
            reader.read_samples(0, 0, ["missing", "missing", "missing"])

    def test_iterator_matches_full_read_across_gaps_packets_segments(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory)
        selected = [183, 0, 249, 183]
        expected = reader.read_samples(0, 10, selected, units="digital")
        chunks = list(reader.iter_samples(0, 10, chunk_samples=3, channels=selected, units="digital"))
        self.assertEqual([value.shape for value in chunks], [(4, 3), (4, 3), (4, 3), (4, 1)])
        np.testing.assert_equal(np.concatenate(chunks, axis=1), expected)
        chunks[0][0, 0] = -999
        self.assertNotEqual(chunks[1][0, 0], -999)

    def test_iterator_limits_and_invalid_arguments(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory, limits=ReadLimits(max_read_samples=2))
        self.assertEqual(len(list(reader.iter_samples(chunk_samples=2, channels=[0]))), 5)
        with self.assertRaises(ResourceLimitError):
            next(reader.iter_samples(chunk_samples=3, channels=[0]))
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                next(reader.iter_samples(chunk_samples=value))
        self.assertEqual(list(reader.iter_samples(0, 0)), [])

    def test_metadata_byte_and_entry_limits_precede_open(self) -> None:
        with patch.object(Path, "open", side_effect=AssertionError("Must not open oversized metadata")):
            with self.assertRaises(ResourceLimitError):
                read_stc(self.fixture.stc, limits=ReadLimits(max_metadata_bytes=32))
            with self.assertRaises(ResourceLimitError):
                read_stc(self.fixture.stc, limits=ReadLimits(max_segments=1))
            with self.assertRaises(ResourceLimitError):
                read_etc(self.fixture.first_erd.with_suffix(".etc"), limits=ReadLimits(max_packets_per_segment=1))

    def test_bad_packet_length_rejected_before_array_allocation(self) -> None:
        # One real tiny packet now declares one sample although it has bytes for
        # three. It remains within file offsets but violates physical bounds.
        etc = self.fixture.first_erd.with_suffix(".etc")
        payload = bytearray(etc.read_bytes())
        pack_into("<h", payload, 352 + 12, 1)
        pack_into("<i", payload, 352 + 16 + 8, 1)
        etc.write_bytes(payload)
        reader = NatusERDReader.open(self.fixture.directory)
        with patch("numpy.full", side_effect=AssertionError("No array for corrupt packet span")):
            with self.assertRaises(DataIntegrityError):
                reader.read_samples(0, 1, [0])

    def test_packet_cap_is_checked_before_array_allocation(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory, limits=ReadLimits(max_packet_bytes=16))
        with patch("numpy.full", side_effect=AssertionError("No array for over-budget packet")):
            with self.assertRaises(ResourceLimitError):
                reader.read_samples(0, 1, [0])

    def test_index_cache_is_bounded(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory, limits=ReadLimits(max_cached_segments=1))
        expected = reader.read_samples(0, 10, [0], units="digital")
        self.assertEqual(len(reader._etc_cache), 1)
        np.testing.assert_equal(reader.read_samples(0, 10, [0], units="digital"), expected)
        self.assertEqual(len(reader._etc_cache), 1)

    def test_cached_recording_changes_are_rejected(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory)
        reader.read_samples(0, 1, [0])
        with self.fixture.first_erd.open("ab") as stream:
            stream.write(b"x")
        with self.assertRaises(DataIntegrityError):
            reader.read_samples(0, 1, [0])

    def test_unknown_erd_is_not_silently_replaced_by_another_recording(self) -> None:
        unknown = self.fixture.directory / "unindexed.erd"
        unknown.write_bytes(self.fixture.first_erd.read_bytes())
        with self.assertRaises(DataIntegrityError):
            NatusERDReader.open(unknown)

    def test_directory_scan_has_explicit_limit(self) -> None:
        with self.assertRaises(ResourceLimitError):
            NatusERDReader.open(self.fixture.directory, limits=ReadLimits(max_directory_entries=1))

    def test_unknown_eeg_does_not_open_the_only_unrelated_stc(self) -> None:
        path = self.fixture.directory / "unrelated.eeg"
        path.write_bytes(b"not sample data")
        with self.assertRaises(FileNotFoundError):
            NatusERDReader.open(path)

    def test_invalid_present_ent_is_not_ignored(self) -> None:
        path = self.fixture.stc.with_suffix(".ent")
        path.unlink()
        path.mkdir()
        with self.assertRaises(DataIntegrityError):
            NatusERDReader.open(self.fixture.directory)


if __name__ == "__main__":
    unittest.main()
