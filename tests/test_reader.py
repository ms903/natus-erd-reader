from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from struct import pack_into

import numpy as np

from natus_erd import (
    DataIntegrityError,
    NatusERDReader,
    UnsupportedFormatError,
)
from natus_erd.reader import QUANTUM_UV_SCALE

from ._fixture import SHORTED, build_recording


class ReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        # tempfile creates mode-0700 directories that are inaccessible under
        # some Windows AppContainer configurations. A unique mode-0777 test
        # directory exercises the same behavior and remains portable.
        self.temporary = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.temporary.mkdir()
        self.addCleanup(shutil.rmtree, self.temporary, True)
        self.fixture = build_recording(self.temporary)

    def test_open_paths_metadata_montage_and_events(self) -> None:
        for path in (
            self.fixture.directory,
            self.fixture.stc,
            self.fixture.eeg,
            self.fixture.first_erd,
        ):
            reader = NatusERDReader.open(path)
            self.assertEqual(reader.info.n_samples, 10)
            self.assertEqual(reader.info.segment_count, 2)

        reader = NatusERDReader.open(self.fixture.directory)
        self.assertEqual(len(reader.channels), 276)
        self.assertEqual(reader.channels[0].name, "CH000")
        self.assertEqual(reader.channels[275].name, "CH275")
        self.assertEqual(
            [channel.index for channel in reader.channels if channel.shorted],
            sorted(SHORTED),
        )
        self.assertEqual(reader.sample_to_stamp(0), 1000)
        self.assertEqual(reader.stamp_to_sample(1009), 9)
        events = reader.read_events()
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].sample, events[0].text, events[0].user), (1, "marker", "tester"))

    def test_decode_deltas_absolutes_gaps_boundaries_and_selection(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory)
        selected = [1, "CH000", 183, 249, 275, 0]
        data = reader.read_samples(0, 10, selected, units="digital")
        self.assertEqual(data.shape, (6, 10))
        for row, channel in enumerate((1, 0, 183, 249, 275, 0)):
            expected = self.fixture.expected[channel]
            for sample, value in enumerate(expected):
                if value is None or channel in SHORTED:
                    self.assertTrue(np.isnan(data[row, sample]))
                else:
                    self.assertEqual(data[row, sample], value)
        self.assertEqual(data[2, 2], 131071)
        self.assertEqual(data[1, 4], 777777)

    def test_uv_scaling_and_auxiliary_policy(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory)
        digital = reader.read_samples(0, 3, [0, 1], units="digital")
        uv = reader.read_samples(0, 3, [0, 1])
        factor = QUANTUM_UV_SCALE * 64
        np.testing.assert_allclose(uv, digital * factor)
        auxiliary = reader.read_samples(0, 2, [275], units="digital")
        self.assertTrue(np.isfinite(auxiliary).all())
        with self.assertRaises(UnsupportedFormatError):
            reader.read_samples(0, 2, [275], units="uV")

    def test_validate_counts_gap_and_failures(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory)
        report = reader.validate()
        self.assertEqual(report.segment_count, 2)
        self.assertEqual(report.packet_count, 3)
        self.assertEqual(report.logical_samples, 10)
        self.assertEqual(report.stored_samples, 9)
        self.assertEqual(report.missing_samples, 1)
        self.assertEqual(report.event_count, 1)
        self.assertEqual(report.ent_record_count, 2)
        self.assertEqual(report.unparsed_ent_record_count, 0)

        with self.assertRaises(IndexError):
            reader.read_samples(-1, 2)
        with self.assertRaises(IndexError):
            reader.read_samples(0, 11)
        with self.assertRaises(ValueError):
            reader.read_samples(0, 1, units="volts")
        with self.assertRaises(ValueError):
            reader.read_samples(0, 1, ["missing"])

    def test_validation_reports_unparsed_ent_records(self) -> None:
        from struct import pack
        ent = self.fixture.stc.with_suffix('.ent')
        payload = ent.read_bytes()
        text = b'unknown vendor text\0\0'
        # A real additional binary ENT record with unsupported text syntax.
        from struct import unpack_from
        cursor = 352
        previous = 0
        while cursor < len(payload)-16:
            previous = unpack_from('<i', payload, cursor+4)[0]
            cursor += previous
        record = pack('<4i', 3, 16+len(text), previous, 0)+text
        ent.write_bytes(payload[:-16]+record+bytes(16))
        report = NatusERDReader.open(self.fixture.directory).validate()
        self.assertEqual((report.ent_record_count, report.unparsed_ent_record_count, report.event_count), (3, 1, 1))

    def test_truncated_etc_is_reported_as_corruption(self) -> None:
        etc_path = self.fixture.first_erd.with_suffix(".etc")
        etc_path.write_bytes(etc_path.read_bytes()[:-1])
        reader = NatusERDReader.open(self.fixture.directory)
        with self.assertRaises(DataIntegrityError):
            reader.validate()

    def test_rejects_unsupported_schema_and_decimated(self) -> None:
        data = bytearray(self.fixture.first_erd.read_bytes())
        pack_into("<H", data, 16, 8)
        self.fixture.first_erd.write_bytes(data)
        with self.assertRaises(UnsupportedFormatError):
            NatusERDReader.open(self.fixture.directory)

        derivative = Path(self.temporary.name) / "Decimated"
        derivative.mkdir()
        with self.assertRaises(UnsupportedFormatError):
            NatusERDReader.open(derivative)

if __name__ == "__main__":
    unittest.main()
