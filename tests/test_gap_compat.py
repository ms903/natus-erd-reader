"""STC stored counts differ from logical stamp spans whenever samples are absent."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from struct import pack, pack_into
from unittest.mock import patch

import numpy as np

from natus_erd import DataIntegrityError, NatusERDReader
from natus_erd.binary import read_stc

from ._fixture import (
    HEADER_SIZE, N_CHANNELS, SHORTED, _encode_packet, _erd_header,
    _generic_header, _stc_entry, build_recording,
)


class GapCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, True)

    def _recording(
        self, intervals: tuple[tuple[int, int], ...], *,
        stored_samples: int | None = None,
        shorted: set[int] | frozenset[int] = frozenset(SHORTED),
    ) -> tuple[Path, np.ndarray]:
        """A 12-sample logical segment; intervals use native stamps and counts."""
        directory = self.root / uuid.uuid4().hex
        directory.mkdir()
        erd = bytearray(_erd_header(sample_rate=512.0, shorted=shorted))
        etc = bytearray(_generic_header(3))
        expected = np.full((N_CHANNELS, 12), np.nan)
        sample_number = 0
        for stamp, count in intervals:
            samples = [
                [50000 + (stamp + offset - 1000) * 10 + channel
                 for channel in range(N_CHANNELS)]
                for offset in range(count)
            ]
            etc.extend(pack("<iiihh", len(erd), stamp, sample_number, count, 0))
            erd.extend(_encode_packet(samples, shorted=shorted))
            for offset, sample in enumerate(samples):
                column = stamp + offset - 1000
                if 0 <= column < 12:
                    for channel in range(N_CHANNELS):
                        if channel not in shorted:
                            expected[channel, column] = sample[channel]
            sample_number += count
        (directory / "recording.erd").write_bytes(erd)
        (directory / "recording.etc").write_bytes(etc)
        count = sample_number if stored_samples is None else stored_samples
        stc = bytearray(_generic_header(1))
        stc.extend(pack("<ii12i", 1, 1, *([0] * 12)))
        stc.extend(_stc_entry("recording", 1000, 1011, 0, stored_samples=count))
        (directory / "recording.stc").write_bytes(stc)
        return directory, expected

    def test_leading_internal_trailing_and_combined_gaps_preserve_time_axis(self) -> None:
        cases = (
            ((1002, 10),),
            ((1000, 3), (1005, 7)),
            ((1000, 10),),
            ((1002, 2), (1007, 2)),
        )
        for intervals in cases:
            with self.subTest(intervals=intervals):
                directory, expected = self._recording(intervals)
                segment, = read_stc(directory / "recording.stc").entries
                self.assertEqual(segment.stored_samples, sum(count for _, count in intervals))
                self.assertEqual(segment.logical_span, 12)
                reader = NatusERDReader.open(directory)
                self.assertEqual((reader.info.start_stamp, reader.info.end_stamp), (1000, 1011))
                self.assertEqual(reader.info.n_samples, 12)
                selected = [0, 1, 249, 183, 0]
                digital = reader.read_samples(0, 12, selected, units="digital")
                np.testing.assert_equal(digital, expected[selected])
                chunks = reader.iter_samples(chunk_samples=3, channels=selected, units="digital")
                np.testing.assert_equal(np.concatenate(list(chunks), axis=1), expected[selected])
                for sample in range(12):
                    np.testing.assert_equal(
                        reader.read_samples(sample, sample + 1, [0], units="digital"),
                        expected[[0], sample:sample + 1],
                    )
                report = reader.validate()
                stored = sum(count for _, count in intervals)
                self.assertEqual((report.stored_samples, report.missing_samples), (stored, 12 - stored))
                self.assertEqual(report.packet_count, len(intervals))

    def test_zero_storage_with_empty_index_and_header_only_erd_is_all_nan(self) -> None:
        directory, expected = self._recording(())
        self.assertEqual((directory / "recording.erd").stat().st_size, HEADER_SIZE)
        reader = NatusERDReader.open(directory)
        np.testing.assert_equal(reader.read_samples(0, 12), expected[:256])
        report = reader.validate()
        self.assertEqual((report.packet_count, report.stored_samples, report.missing_samples), (0, 0, 12))

    def test_zero_storage_cannot_hide_payload_or_packets(self) -> None:
        for kind in ("payload_without_index", "indexed_samples"):
            with self.subTest(kind=kind):
                intervals = ((1000, 1),) if kind == "indexed_samples" else ()
                directory, _ = self._recording(intervals, stored_samples=0)
                if kind == "payload_without_index":
                    erd = directory / "recording.erd"
                    erd.write_bytes(erd.read_bytes() + b"x")
                reader = NatusERDReader.open(directory)
                with patch("numpy.full", side_effect=AssertionError("No corrupt output allocation")):
                    with self.assertRaises(DataIntegrityError):
                        reader.read_samples(0, 1, [0])
                self.assertEqual(len(reader._etc_cache), 0)

    def test_stc_and_etc_stored_count_mismatch_rejected_before_cache_or_allocation(self) -> None:
        for declared in (3, 5, 12):
            with self.subTest(declared=declared):
                directory, _ = self._recording(((1002, 2), (1007, 2)), stored_samples=declared)
                reader = NatusERDReader.open(directory)
                with patch("numpy.full", side_effect=AssertionError("No output for inconsistent stored count")):
                    with self.assertRaises(DataIntegrityError):
                        reader.read_samples(0, 1, [0])
                self.assertEqual(len(reader._etc_cache), 0)
                with self.assertRaises(DataIntegrityError):
                    reader.validate()

    def test_nonzero_storage_cannot_have_an_empty_index(self) -> None:
        directory, _ = self._recording((), stored_samples=1)
        reader = NatusERDReader.open(directory)
        with patch("numpy.full", side_effect=AssertionError("No output for missing stored samples")):
            with self.assertRaises(DataIntegrityError):
                reader.read_samples(0, 1, [0])

    def test_impossible_stc_counts_and_reversed_bounds_are_rejected(self) -> None:
        for start, end, stored in ((1000, 1011, -1), (1000, 1011, 13), (1011, 1000, 0)):
            with self.subTest(start=start, end=end, stored=stored):
                directory, _ = self._recording(())
                path = directory / "recording.stc"
                payload = bytearray(path.read_bytes())
                pack_into("<4i", payload, 408 + 256, start, end, 0, stored)
                path.write_bytes(payload)
                with self.assertRaises(DataIntegrityError):
                    NatusERDReader.open(directory)

    def test_overlapping_packets_and_out_of_segment_stamps_remain_invalid(self) -> None:
        for intervals in (((1000, 3), (1002, 2)), ((999, 2),), ((1011, 2),)):
            with self.subTest(intervals=intervals):
                directory, _ = self._recording(intervals)
                reader = NatusERDReader.open(directory)
                with patch("numpy.full", side_effect=AssertionError("No allocation for invalid stamp bounds")):
                    with self.assertRaises(DataIntegrityError):
                        reader.read_samples(0, 12, [0])

    def test_truncated_erd_payload_is_not_treated_as_a_gap(self) -> None:
        directory, _ = self._recording(((1000, 1),))
        path = directory / "recording.erd"
        path.write_bytes(path.read_bytes()[:-1])
        reader = NatusERDReader.open(directory)
        with patch("numpy.full", side_effect=AssertionError("No output for a truncated packet")):
            with self.assertRaises(DataIntegrityError):
                reader.read_samples(0, 12, [0])

    def test_different_shorted_configurations_skip_only_declared_channels(self) -> None:
        masks = (set(), {0}, {1, 128, 249, 275}, set(range(256)), set(range(N_CHANNELS)))
        for shorted in masks:
            with self.subTest(shorted_count=len(shorted)):
                directory, expected = self._recording(((1000, 3), (1007, 2)), shorted=shorted)
                reader = NatusERDReader.open(directory)
                self.assertEqual({c.index for c in reader.channels if c.shorted}, shorted)
                np.testing.assert_equal(
                    reader.read_samples(0, 12, list(range(N_CHANNELS)), units="digital"), expected
                )
                np.testing.assert_equal(np.isnan(reader.read_samples(0, 12)), np.isnan(expected[:256]))
                self.assertEqual(reader.validate().stored_samples, 5)

    def test_segment_overlap_is_rejected_even_with_reduced_storage(self) -> None:
        fixture = build_recording(self.root)
        payload = bytearray(fixture.stc.read_bytes())
        pack_into("<i", payload, 408 + 272 + 256, 1004)
        fixture.stc.write_bytes(payload)
        with self.assertRaises(DataIntegrityError):
            NatusERDReader.open(fixture.directory)

    def test_empty_first_last_or_all_segments_preserve_logical_recording_length(self) -> None:
        for empty in ((0,), (1,), (0, 1)):
            with self.subTest(empty_segments=empty):
                root = self.root / uuid.uuid4().hex
                root.mkdir()
                fixture = build_recording(root)
                paths = (fixture.first_erd, fixture.first_erd.with_name(f"{fixture.first_erd.stem}_001.erd"))
                stc = bytearray(fixture.stc.read_bytes())
                expected = np.asarray(fixture.expected[0], dtype=np.float64)
                for index in empty:
                    paths[index].write_bytes(_erd_header())
                    paths[index].with_suffix(".etc").write_bytes(_generic_header(3))
                    pack_into("<i", stc, 408 + index * 272 + 268, 0)
                    expected[index * 5:index * 5 + 5] = np.nan
                fixture.stc.write_bytes(stc)
                reader = NatusERDReader.open(fixture.directory)
                self.assertEqual(reader.info.n_samples, 10)
                np.testing.assert_equal(reader.read_samples(0, 10, [0], units="digital")[0], expected)
                chunks = list(reader.iter_samples(chunk_samples=3, channels=[0], units="digital"))
                np.testing.assert_equal(np.concatenate(chunks, axis=1)[0], expected)
                stored = int(np.isfinite(expected).sum())
                report = reader.validate()
                self.assertEqual((report.stored_samples, report.missing_samples), (stored, 10 - stored))

    def test_intersegment_and_leading_packet_gaps_are_both_counted(self) -> None:
        fixture = build_recording(self.root)
        stc = bytearray(fixture.stc.read_bytes())
        # Three stamps between STC segments, then one absent sample inside the
        # second segment. The stored count remains four, not its logical span.
        pack_into("<4i", stc, 408 + 272 + 256, 1008, 1012, 5, 4)
        fixture.stc.write_bytes(stc)
        second = fixture.first_erd.with_name(f"{fixture.first_erd.stem}_001.etc")
        etc = bytearray(second.read_bytes())
        pack_into("<i", etc, 352 + 4, 1009)
        second.write_bytes(etc)
        expected = np.concatenate((
            np.asarray(fixture.expected[0][:5], dtype=np.float64),
            np.full(4, np.nan),
            np.asarray(fixture.expected[0][6:], dtype=np.float64),
        ))
        reader = NatusERDReader.open(fixture.directory)
        self.assertEqual(reader.info.n_samples, 13)
        np.testing.assert_equal(reader.read_samples(0, 13, [0], units="digital")[0], expected)
        self.assertTrue(np.isnan(reader.read_samples(5, 9, [0], units="digital")).all())
        report = reader.validate()
        self.assertEqual((report.logical_samples, report.stored_samples, report.missing_samples), (13, 9, 4))


if __name__ == "__main__":
    unittest.main()
