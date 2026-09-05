"""Case-insensitive record selection and ENT fallback on synthetic files only."""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from natus_erd import DataIntegrityError, NatusERDReader
from natus_erd.binary import STC_ENTRY_SIZE, STC_PREFIX_SIZE

from ._fixture import build_recording


class RecordingPathCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.temporary.mkdir()
        self.addCleanup(shutil.rmtree, self.temporary, True)
        self.fixture = build_recording(self.temporary)
        self.target = self.temporary / "case fixtures"
        self.target.mkdir()

    def _copy_recording(
        self,
        stem: str = "Record",
        *,
        entry_stem: str | None = None,
        extensions: dict[str, str] | None = None,
    ) -> dict[str, Path]:
        """Rename a tiny generated recording and its embedded STC references."""
        entry_stem = stem if entry_stem is None else entry_stem
        extensions = {} if extensions is None else extensions
        paths: dict[str, Path] = {}
        original_stem = self.fixture.stc.stem
        for source in self.fixture.directory.iterdir():
            tail = source.stem.removeprefix(original_stem)
            suffix = extensions.get(source.suffix, source.suffix)
            destination = self.target / f"{stem}{tail}{suffix}"
            payload = source.read_bytes()
            if source == self.fixture.stc:
                data = bytearray(payload)
                for index, segment_tail in enumerate(("", "_001")):
                    offset = STC_PREFIX_SIZE + index * STC_ENTRY_SIZE
                    name = f"{entry_stem}{segment_tail}".encode("utf-8")
                    data[offset:offset + 256] = name.ljust(256, b"\0")
                payload = bytes(data)
            destination.write_bytes(payload)
            paths[tail + source.suffix] = destination
        return paths

    def _assert_recording(self, source: Path) -> NatusERDReader:
        import numpy as np

        reader = NatusERDReader.open(source)
        self.assertEqual(reader.info.n_samples, 10)
        self.assertEqual(reader.info.segment_count, 2)
        self.assertEqual(reader.channels[0].name, "CH000")
        events = reader.read_events()
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0].sample, events[0].text), (1, "marker"))
        expected = [float("nan") if value is None else value for value in self.fixture.expected[0]]
        np.testing.assert_array_equal(reader.read_samples(0, 10, channels=[0], units="digital"), [expected])
        return reader

    def test_uppercase_extensions_and_case_mismatched_stc_references(self) -> None:
        paths = self._copy_recording(
            "ReCoRd", entry_stem="record",
            extensions={suffix: suffix.upper() for suffix in (".stc", ".eeg", ".erd", ".etc", ".ent")},
        )
        for source in (self.target, paths[".stc"], paths[".eeg"], paths[".erd"], paths["_001.erd"]):
            with self.subTest(entry=source.suffix):
                self._assert_recording(source)

    def test_mixed_extensions_and_eeg_stc_basename_case(self) -> None:
        paths = self._copy_recording(
            "Record", entry_stem="RECORD",
            extensions={".stc": ".sTc", ".eeg": ".eEg", ".erd": ".ErD", ".etc": ".eTc", ".ent": ".EnT"},
        )
        moved_eeg = self.target / "RECORD.eEg"
        paths[".eeg"].rename(moved_eeg)
        self._assert_recording(moved_eeg)
        self._assert_recording(self.target)

    def test_unrelated_ordinary_stc_directory_is_not_a_candidate(self) -> None:
        self._copy_recording()
        (self.target / "backup.stc").mkdir()
        reader = self._assert_recording(self.target)
        self.assertEqual(reader.validate().segment_count, 2)

    def test_two_records_allow_explicit_stc_or_matching_eeg_only(self) -> None:
        first = self._copy_recording("RecordA")
        second = self._copy_recording("RecordB")
        for paths in (first, second):
            self._assert_recording(paths[".stc"])
            self._assert_recording(paths[".eeg"])
        for source in (self.target, first[".erd"], second[".erd"], first["_001.erd"]):
            with self.subTest(entry=source.suffix):
                with self.assertRaisesRegex(DataIntegrityError, "Expected one main STC"):
                    NatusERDReader.open(source)

    def test_explicit_record_selection_does_not_read_other_corrupt_stc(self) -> None:
        paths = self._copy_recording()
        (self.target / "unrelated.STC").write_bytes(b"not a valid STC")
        self._assert_recording(paths[".stc"])
        self._assert_recording(paths[".eeg"])

    def test_mismatched_eeg_is_not_guessed_from_single_stc(self) -> None:
        paths = self._copy_recording()
        other_eeg = self.target / "other.EEG"
        other_eeg.write_bytes(paths[".eeg"].read_bytes())
        with self.assertRaises(FileNotFoundError):
            NatusERDReader.open(other_eeg)

    def test_missing_ent_is_valid_with_default_labels_and_no_events(self) -> None:
        paths = self._copy_recording()
        paths[".ent"].unlink()
        reader = NatusERDReader.open(self.target)
        self.assertEqual(reader.read_events(), ())
        self.assertEqual(reader.channels[0].name, "chan000")
        self.assertEqual(reader.read_samples(0, 1, channels=["chan000"]).shape, (1, 1))

    def test_uppercase_ent_old_supplies_labels_and_events_when_ent_is_absent(self) -> None:
        paths = self._copy_recording()
        paths[".ent"].rename(self.target / "RECORD.ENT.OLD")
        self._assert_recording(self.target)

    def test_existing_ent_takes_priority_over_old(self) -> None:
        self._copy_recording(extensions={".ent": ".ENT"})
        (self.target / "Record.ENT.OLD").write_bytes(b"old data is intentionally corrupt")
        self._assert_recording(self.target)

    def test_corrupt_existing_ent_does_not_fall_back_to_valid_old(self) -> None:
        paths = self._copy_recording(extensions={".ent": ".ENT"})
        (self.target / "Record.ENT.OLD").write_bytes(paths[".ent"].read_bytes())
        paths[".ent"].write_bytes(b"truncated ENT")
        with self.assertRaisesRegex(DataIntegrityError, "truncated generic header"):
            NatusERDReader.open(self.target)

    def test_casefold_collision_is_rejected_for_explicit_and_discovered_stc(self) -> None:
        paths = self._copy_recording()
        collision = self.target / "rECORD.STC"
        if collision.exists():
            self.skipTest("Filesystem is case-insensitive; distinct casefold-colliding files cannot be created")
        collision.write_bytes(paths[".stc"].read_bytes())
        for source in (self.target, paths[".stc"], paths[".eeg"]):
            with self.subTest(entry=source.suffix):
                with self.assertRaisesRegex(DataIntegrityError, "Ambiguous case-insensitive"):
                    NatusERDReader.open(source)

    def test_casefold_collision_in_a_selected_segment_or_ent_is_rejected(self) -> None:
        paths = self._copy_recording()
        for key, spelling in ((".erd", "rECORD.ERD"), (".etc", "rECORD.ETC"), (".ent", "rECORD.ENT")):
            with self.subTest(extension=key):
                collision = self.target / spelling
                if collision.exists():
                    self.skipTest("Filesystem is case-insensitive; distinct casefold-colliding files cannot be created")
                collision.write_bytes(paths[key].read_bytes())
                try:
                    with self.assertRaisesRegex(DataIntegrityError, "Ambiguous case-insensitive"):
                        NatusERDReader.open(paths[".stc"])
                finally:
                    collision.unlink()


if __name__ == "__main__":
    unittest.main()
