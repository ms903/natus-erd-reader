"""Small synthetic regressions for file boundaries and layout validation."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from struct import pack_into

from natus_erd import DataIntegrityError, NatusERDReader, UnsupportedFormatError
from natus_erd.binary import read_stc

from ._fixture import build_recording


class FileBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.temporary.mkdir()
        self.addCleanup(shutil.rmtree, self.temporary, True)
        self.fixture = build_recording(self.temporary)

    def test_stc_rejects_windows_unsafe_names_on_every_platform(self) -> None:
        original = self.fixture.stc.read_bytes()
        for name in (
            "../outside", "..\\outside", "D:outside", "recording:stream",
            "CON", "NUL.erd", "COM1", "LPT9", "bad?name", "bad*name",
            "bad\x01name", "trailing.", "trailing ",
        ):
            with self.subTest(name=name):
                data = bytearray(original)
                encoded = name.encode("utf-8")
                data[408:664] = encoded.ljust(256, b"\0")
                self.fixture.stc.write_bytes(data)
                with self.assertRaises(DataIntegrityError):
                    read_stc(self.fixture.stc)

    def test_recording_directory_is_not_searched_recursively(self) -> None:
        with self.assertRaises(FileNotFoundError):
            NatusERDReader.open(self.temporary)

    def test_symlinked_recording_members_cannot_escape_directory(self) -> None:
        for suffix in (".erd", ".etc", ".ent"):
            with self.subTest(suffix=suffix):
                original = self.fixture.first_erd.with_suffix(suffix)
                contents = original.read_bytes()
                external = self.temporary / f"outside{suffix}"
                external.write_bytes(contents)
                original.unlink()
                try:
                    try:
                        original.symlink_to(external)
                    except (OSError, NotImplementedError) as exc:
                        self.skipTest(f"File symlinks unavailable: {exc}")
                    with self.assertRaises(DataIntegrityError):
                        NatusERDReader.open(self.fixture.directory)
                finally:
                    if original.is_symlink():
                        original.unlink()
                    original.write_bytes(contents)

    def test_later_segment_header_is_checked_without_explicit_validate(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory)
        second = self.fixture.first_erd.with_name(
            f"{self.fixture.first_erd.stem}_001.erd"
        )
        payload = bytearray(second.read_bytes())
        pack_into("<H", payload, 16, 8)
        second.write_bytes(payload)
        with self.assertRaises((UnsupportedFormatError, DataIntegrityError)):
            reader.read_samples(6, 10, [0], units="digital")

    def test_later_segment_shorted_layout_is_checked(self) -> None:
        reader = NatusERDReader.open(self.fixture.directory)
        second = self.fixture.first_erd.with_name(
            f"{self.fixture.first_erd.stem}_001.erd"
        )
        payload = bytearray(second.read_bytes())
        pack_into("<h", payload, 4560, 1)
        second.write_bytes(payload)
        with self.assertRaises(DataIntegrityError):
            reader.read_samples(6, 10, [0], units="digital")

    def test_invalid_shorted_flag_is_not_coerced_to_true(self) -> None:
        payload = bytearray(self.fixture.first_erd.read_bytes())
        pack_into("<h", payload, 4560, 7)
        self.fixture.first_erd.write_bytes(payload)
        with self.assertRaises(DataIntegrityError):
            NatusERDReader.open(self.fixture.directory)

    def test_secondary_headbox_is_not_silently_accepted(self) -> None:
        payload = bytearray(self.fixture.first_erd.read_bytes())
        pack_into("<i", payload, 4468, 20)
        self.fixture.first_erd.write_bytes(payload)
        with self.assertRaises((UnsupportedFormatError, DataIntegrityError)):
            NatusERDReader.open(self.fixture.directory)

    def test_negative_or_duplicate_physical_channels_are_rejected(self) -> None:
        original = self.fixture.first_erd.read_bytes()
        for physical_index in (-1, 1):
            with self.subTest(physical_index=physical_index):
                payload = bytearray(original)
                pack_into("<i", payload, 368, physical_index)
                self.fixture.first_erd.write_bytes(payload)
                with self.assertRaises((UnsupportedFormatError, DataIntegrityError)):
                    NatusERDReader.open(self.fixture.directory)

    def test_unvalidated_physical_permutation_is_rejected(self) -> None:
        payload = bytearray(self.fixture.first_erd.read_bytes())
        # A permutation is in range and unique, but could put an auxiliary
        # physical channel into a logical position assigned AC calibration.
        pack_into("<i", payload, 368, 275)
        pack_into("<i", payload, 368 + 275 * 4, 0)
        self.fixture.first_erd.write_bytes(payload)
        with self.assertRaises((UnsupportedFormatError, DataIntegrityError)):
            NatusERDReader.open(self.fixture.directory)

    def test_unsupported_stc_and_etc_schemas_are_rejected(self) -> None:
        for path in (
            self.fixture.stc,
            self.fixture.first_erd.with_suffix(".etc"),
        ):
            with self.subTest(suffix=path.suffix):
                original = path.read_bytes()
                payload = bytearray(original)
                pack_into("<H", payload, 16, 999)
                path.write_bytes(payload)
                try:
                    with self.assertRaises((UnsupportedFormatError, DataIntegrityError)):
                        reader = NatusERDReader.open(self.fixture.directory)
                        reader.read_samples(0, 1, [0], units="digital")
                finally:
                    path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
