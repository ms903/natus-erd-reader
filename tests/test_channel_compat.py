from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from natus_erd import NatusERDReader
from natus_erd.ent import (
    EntNote,
    _names_from_raw_montage,
    _names_from_value,
    _safe_parse_excel,
    channel_names_from_notes,
    complete_channel_names,
    events_from_notes,
)

from ._fixture import N_CHANNELS, build_recording


def _note(value: object, text: str = "", note_type: int = 2) -> EntNote:
    return EntNote(note_type, 0, 0, text, value)


class ChannelNameCompatibilityTests(unittest.TestCase):
    def test_partial_names_preserve_channel_slots_and_unicode(self) -> None:
        values = ["A", None, "", " \t\u3000", 12, False, ["nested"], "B'12", " 左侧 "]
        names = _names_from_value({"ChanNames": values})
        self.assertEqual(names, ("A", None, None, None, None, None, None, "B'12", " 左侧 "))
        self.assertEqual(
            complete_channel_names(names, 11),
            ("A", "chan001", "chan002", "chan003", "chan004", "chan005",
             "chan006", "B'12", " 左侧 ", "chan009", "chan010"),
        )

    def test_missing_or_invalid_name_field_uses_defaults(self) -> None:
        for value in ({}, {"ChanNames": []}, {"ChanNames": "not a sequence"},
                      {"ChanNames": b"not labels"}, {"ChanNames": 123}):
            with self.subTest(value=value):
                self.assertEqual(_names_from_value(value), ())
        self.assertEqual(complete_channel_names((), 3), ("chan000", "chan001", "chan002"))
        self.assertEqual(complete_channel_names(("A",), 0), ())

    def test_defaults_reserve_all_vendor_labels_and_suffixes(self) -> None:
        names = (None, "chan000", "chan000_1", None, "chan003", "chan003_1", "chan003_2")
        self.assertEqual(
            complete_channel_names(names, 7),
            ("chan000_2", "chan000", "chan000_1", "chan003_3", "chan003", "chan003_1", "chan003_2"),
        )
        # Labels outside the output slice must also be reserved first.
        self.assertEqual(complete_channel_names((None, "chan000"), 1), ("chan000_1",))

    def test_real_duplicate_names_are_not_rewritten(self) -> None:
        self.assertEqual(complete_channel_names(("A", "A", None), 3), ("A", "A", "chan002"))

    def test_latest_nonempty_montage_wins_even_when_all_labels_are_missing(self) -> None:
        old = _note({"ChanNames": ["old A", "old B"]})
        missing = _note({"ChanNames": [None, " "]})
        empty = _note({"ChanNames": []})
        self.assertEqual(channel_names_from_notes((old, missing, empty)), (None, None))
        self.assertEqual(channel_names_from_notes((old, empty)), ("old A", "old B"))
        self.assertEqual(channel_names_from_notes((_note({"ChanNames": ["first"]}), old)), ("old A", "old B"))

    def test_raw_montage_fallback_preserves_slots(self) -> None:
        text = '''(.(."Geometry", [unsupported]), (."ChanNames", ("A", "", None, 1, ("nested"), "B'12", "左侧")))'''
        self.assertIsNone(_safe_parse_excel(text))
        expected = ("A", None, None, None, None, "B'12", "左侧")
        self.assertEqual(_names_from_raw_montage(text), expected)
        self.assertEqual(channel_names_from_notes((_note(None, text),)), expected)
        missing = '''(.(."Geometry", [unsupported]), (."ChanNames", (None, " ")))'''
        self.assertEqual(channel_names_from_notes((_note({"ChanNames": ["old"]}), _note(None, missing))), (None, None))

    def test_raw_fallback_does_not_execute_expressions_or_skip_ambiguous_fields(self) -> None:
        unsafe = '''(.(."Geometry", [unsupported]), (."ChanNames", (__import__("os"))), (."ChanNames", ("guess")))'''
        self.assertEqual(_names_from_raw_montage(unsafe), ())
        empty = '''(.(."Geometry", [unsupported]), (."ChanNames", ()), (."ChanNames", ("guess")))'''
        self.assertEqual(_names_from_raw_montage(empty), ())


class ChannelReaderCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.recording = build_recording(self.directory)

    def _reader_with_names(self, names: object) -> NatusERDReader:
        with patch.object(NatusERDReader, "_load_notes", return_value=(_note({"ChanNames": names}),)):
            return NatusERDReader.open(self.recording.directory)

    def test_missing_names_do_not_shift_waveforms(self) -> None:
        import numpy as np

        reader = self._reader_with_names(["A", None, "", 123, "B'12", "左侧"])
        self.assertEqual(len(reader.channels), N_CHANNELS)
        self.assertEqual(tuple(c.name for c in reader.channels[:6]), ("A", "chan001", "chan002", "chan003", "B'12", "左侧"))
        named = reader.read_samples(0, 10, channels=["左侧", "B'12", "chan001"], units="digital")
        indexed = reader.read_samples(0, 10, channels=[5, 4, 1], units="digital")
        np.testing.assert_array_equal(named, indexed)

    def test_duplicates_are_ambiguous_only_for_name_selection(self) -> None:
        import numpy as np

        reader = self._reader_with_names(["A", "A", None, "chan002"])
        self.assertEqual(tuple(c.name for c in reader.channels[:4]), ("A", "A", "chan002_1", "chan002"))
        with self.assertRaises(KeyError):
            reader.read_samples(0, 1, channels=["A"])
        indexed = reader.read_samples(0, 1, channels=[0, 1], units="digital")
        self.assertEqual(indexed.shape, (2, 1))
        np.testing.assert_array_equal(
            reader.read_samples(0, 10, channels=["chan002_1", "chan002"], units="digital"),
            reader.read_samples(0, 10, channels=[2, 3], units="digital"),
        )


class EventTimestampCompatibilityTests(unittest.TestCase):
    def test_integer_stamps_are_preserved_outside_recording_and_sorted_stably(self) -> None:
        notes = tuple(_note({"Stamp": stamp, "Text": text}, note_type=kind) for stamp, text, kind in (
            (1010, "same first", 2), (True, "bool", 1), (1001.0, "float", 1),
            ("1002", "string", 1), (-5, "before origin", 1),
            (1010, "same second", 1), (2**60, "after recording", 3),
        ))
        events = events_from_notes(notes, 1000)
        self.assertEqual(tuple(event.stamp for event in events), (-5, 1010, 1010, 2**60))
        self.assertEqual(tuple(event.sample for event in events), (-1005, 10, 10, 2**60 - 1000))
        self.assertEqual(tuple(event.text for event in events), ("before origin", "same first", "same second", "after recording"))
        self.assertEqual(tuple(event.note_type for event in events), (1, 2, 1, 3))

    def test_non_text_events_are_ignored_and_user_is_optional(self) -> None:
        notes = (
            _note({"Stamp": 42, "Text": None}),
            _note({"Stamp": 42, "Text": "kept", "Data": {"User": 12}}),
            _note({"Stamp": 42, "Text": "Unicode 左侧", "Data": {"User": "tester"}}),
        )
        events = events_from_notes(notes, 42)
        self.assertEqual(tuple(event.user for event in events), (None, "tester"))
        self.assertEqual(tuple(event.sample for event in events), (0, 0))


if __name__ == "__main__":
    unittest.main()
