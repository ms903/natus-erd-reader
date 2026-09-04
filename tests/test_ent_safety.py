from __future__ import annotations

import ast
import io
import shutil
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from struct import pack, pack_into
from unittest.mock import patch

from natus_erd.ent import (
    EntNote,
    _find_named_value,
    _names_from_raw_montage,
    _safe_parse_excel,
    channel_names_from_notes,
    events_from_notes,
    read_ent_notes,
)
from natus_erd.errors import DataIntegrityError, ResourceLimitError, UnsupportedFormatError
from natus_erd.limits import DEFAULT_LIMITS


def _ent_file(*texts: str, schema: int = 3) -> bytes:
    data = bytearray(352)
    pack_into("<HH", data, 16, schema, 1)
    previous = 0
    for text in texts:
        payload = text.encode("utf-8") + b"\0\0"
        length = len(payload) + 16
        data.extend(pack("<4i", 1, length, previous, 0))
        data.extend(payload)
        previous = length
    data.extend(bytes(16))
    return bytes(data)


class _ReadSpy(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.requests: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.requests.append(size)
        if size < 0 or size > 4096:
            raise AssertionError("Diagnostic attempted an unbounded or oversized read")
        return super().read(size)


class EntTextSafetyTests(unittest.TestCase):
    def test_nested_unicode_fields_and_events(self) -> None:
        text = '(.(."Stamp", 1024), (."Text", "左侧事件（测试）"), (."Data", (.(."User", "研究者"))), (."Values", (-1, 2.5, 1e-3, True, None)))'
        value = _safe_parse_excel(text)
        self.assertEqual(value["Values"], [-1, 2.5, 0.001, True, None])
        note = EntNote(1, 0, 0, text, value)
        event, = events_from_notes((note,), 1000)
        self.assertEqual((event.sample, event.text, event.user), (24, "左侧事件（测试）", "研究者"))

    def test_quotes_backslashes_and_unicode_escapes_are_data(self) -> None:
        text = r'''(.(."Text", "quoted \"word\"\n\u5de6\x41"), (."Path", "D:\\data\\file"), (."Unknown", "\q"))'''
        value = _safe_parse_excel(text)
        self.assertEqual(value["Text"], 'quoted "word"\n左A')
        # Recognized escapes are decoded; unknown ones are preserved.
        self.assertEqual(value["Path"], r"D:\data\file")
        self.assertEqual(value["Unknown"], r"\q")

    def test_no_python_parser_or_execution_is_used(self) -> None:
        expressions = (
            '__import__("os").system("must-not-run")',
            '(.(."Text", __import__("os")))',
            '(.(."Text", "x" * 1000000000))',
            '(.(."Text", [x for x in range(1000000000)]))',
            '(.(."Text", object.__class__))',
        )
        with patch("builtins.eval", side_effect=AssertionError("eval called")), patch.object(ast, "literal_eval", side_effect=AssertionError("literal_eval called")):
            for expression in expressions:
                self.assertIsNone(_safe_parse_excel(expression))
            self.assertEqual(_safe_parse_excel('(.(."Stamp", 42))'), {"Stamp": 42})

    def test_vendor_hex_literals_and_short_carriage_escape(self) -> None:
        # Entirely synthetic metadata: wide hex literals are vendor values,
        # and a short hex escape may precede a physical newline in a string.
        text = '(.(."Stamp", 42), (."Text", "first\\xd\nsecond"), (."Token", 0x00112233445566778899aabb), (."Offset", -0x20))'
        value = _safe_parse_excel(text)
        self.assertEqual(value["Token"], int("00112233445566778899aabb", 16))
        self.assertEqual(value["Offset"], -32)
        self.assertEqual(value["Text"], "first\r\nsecond")
        note = EntNote(1, 0, 0, text, value)
        self.assertEqual(len(events_from_notes((note,), 0)), 1)
        for malformed in ("0x", "0x12gg", "0x20.__class__"):
            self.assertIsNone(_safe_parse_excel(malformed))
        with self.assertRaises(ResourceLimitError):
            _safe_parse_excel("0x" + "f" * 64)

    def test_depth_node_and_text_limits(self) -> None:
        limits = replace(DEFAULT_LIMITS, max_parse_depth=4, max_parse_nodes=8, max_ent_record_bytes=128)
        with self.assertRaises(ResourceLimitError):
            _safe_parse_excel("(" * 5 + "0" + ")" * 5, limits=limits)
        with self.assertRaises(ResourceLimitError):
            _safe_parse_excel("(" + ",".join(["0"] * 9) + ")", limits=limits)
        with self.assertRaises(ResourceLimitError):
            _safe_parse_excel('"' + "x" * 128 + '"', limits=limits)
        with self.assertRaises(ResourceLimitError):
            _safe_parse_excel('"' + "左" * 50 + '"', limits=limits)
        with self.assertRaises(ResourceLimitError):
            _safe_parse_excel("9" * 65)

    def test_malformed_literals_do_not_guess_values(self) -> None:
        for text in ("", "(", "(1 2)", "(,1)", "(1,,2)", '(.(."Stamp",1),(."Stamp",2))', '(.(."Text", "unterminated))', '(.(."Value", 1e999))', '(.(."Text", "\\xQQ"))'):
            self.assertIsNone(_safe_parse_excel(text), repr(text))

    def test_nonstandard_montage_extracts_only_quoted_channel_list(self) -> None:
        text = '''(.(."VendorGeometry", [unsupported vendor syntax]), (."ChanNames", ("A1", "B'12", "左侧")), (."Text", "Montage"))'''
        self.assertIsNone(_safe_parse_excel(text))
        self.assertEqual(_names_from_raw_montage(text), ("A1", "B'12", "左侧"))
        note = EntNote(2, 0, 0, text, None)
        self.assertEqual(channel_names_from_notes((note,)), ("A1", "B'12", "左侧"))
        self.assertEqual(_names_from_raw_montage('(.(."Text", "ChanNames"), (."Other", ("not-a-channel")))'), ())
        self.assertEqual(_names_from_raw_montage('(.(."ChanNames", (__import__("os"))))'), ())
        self.assertEqual(_names_from_raw_montage('(.(."ChanNames", ("A", ("nested"))))'), ("A", None))

    def test_montage_fallback_and_field_walk_are_bounded(self) -> None:
        limits = replace(DEFAULT_LIMITS, max_parse_depth=4, max_parse_nodes=8)
        with self.assertRaises(ResourceLimitError):
            _names_from_raw_montage("(" * 5 + '( ."ChanNames", ("A"))', limits=limits)
        with self.assertRaises(ResourceLimitError):
            _names_from_raw_montage('(.(."ChanNames", (' + ",".join(['"A"'] * 10) + ")))", limits=limits)
        cyclic: list[object] = []
        cyclic.append(cyclic)
        self.assertIsNone(_find_named_value(cyclic, "missing"))
        with self.assertRaises(ResourceLimitError):
            _find_named_value([0] * 10, "missing", limits=limits)


class EntFileSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.path = self.directory / "事件.ent"

    def test_valid_file_unicode_and_schema(self) -> None:
        self.path.write_bytes(_ent_file('(.(."Stamp",42),(."Text","安全"))'))
        notes = read_ent_notes(self.path)
        self.assertEqual(notes[0].value, {"Stamp": 42, "Text": "安全"})
        self.path.write_bytes(_ent_file(schema=1))
        with self.assertRaises(UnsupportedFormatError):
            read_ent_notes(self.path)

    def test_declared_giant_payload_is_rejected_before_read(self) -> None:
        data = bytearray(_ent_file())
        pack_into("<4i", data, 352, 1, 2**31 - 1, 0, 0)
        self.path.write_bytes(data)
        spy = _ReadSpy(bytes(data))
        with patch.object(Path, "open", return_value=spy):
            with self.assertRaises(DataIntegrityError):
                read_ent_notes(self.path)
        self.assertEqual(spy.requests, [352, 16])

    def test_valid_but_over_budget_payload_is_rejected_before_read(self) -> None:
        data = _ent_file('"' + "x" * 128 + '"')
        self.path.write_bytes(data)
        spy = _ReadSpy(data)
        with patch.object(Path, "open", return_value=spy):
            with self.assertRaises(ResourceLimitError):
                read_ent_notes(self.path, limits=replace(DEFAULT_LIMITS, max_ent_record_bytes=64))
        self.assertEqual(spy.requests, [352, 16])

    def test_file_size_is_rejected_before_open(self) -> None:
        self.path.write_bytes(_ent_file())
        with patch.object(Path, "open", side_effect=AssertionError("oversized file was opened")):
            with self.assertRaises(ResourceLimitError):
                read_ent_notes(self.path, limits=replace(DEFAULT_LIMITS, max_ent_bytes=352))

    def test_total_parse_nodes_and_record_count_are_bounded(self) -> None:
        self.path.write_bytes(_ent_file("(1,2,3)", "(4,5,6)"))
        with self.assertRaises(ResourceLimitError):
            read_ent_notes(self.path, limits=replace(DEFAULT_LIMITS, max_total_parse_nodes=7))
        with self.assertRaises(ResourceLimitError):
            read_ent_notes(self.path, limits=replace(DEFAULT_LIMITS, max_ent_records=1))
        # Failed parses also consume the aggregate budget.
        self.path.write_bytes(_ent_file("(1,2,3,broken)", "(4,5,6,broken)"))
        with self.assertRaises(ResourceLimitError):
            read_ent_notes(self.path, limits=replace(DEFAULT_LIMITS, max_total_parse_nodes=7))

    def test_truncation_invalid_lengths_and_invalid_utf8(self) -> None:
        for data in (bytes(16), _ent_file()[:360]):
            self.path.write_bytes(data)
            with self.assertRaises(DataIntegrityError):
                read_ent_notes(self.path)
        data = bytearray(_ent_file())
        pack_into("<4i", data, 352, 1, 15, 0, 0)
        self.path.write_bytes(data)
        with self.assertRaises(DataIntegrityError):
            read_ent_notes(self.path)
        data = bytearray(_ent_file('"x"'))
        data[369] = 0xFF
        self.path.write_bytes(data)
        with self.assertRaises(DataIntegrityError):
            read_ent_notes(self.path)


if __name__ == "__main__":
    unittest.main()
