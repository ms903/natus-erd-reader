from __future__ import annotations

import ast
import io
import unittest
from unittest.mock import patch

import numpy as np

from natus_erd.decoder import decode_schema9_packet
from natus_erd.ent import _safe_parse_excel
from natus_erd.errors import DataIntegrityError


class EntSafetyTests(unittest.TestCase):
    def test_excel_parser_never_executes_python(self) -> None:
        event = '(.(."Stamp", 42), (."Text", "safe"))'
        with patch("builtins.eval", side_effect=AssertionError("eval was called")), patch.object(
            ast, "literal_eval", side_effect=AssertionError("AST evaluation was called")
        ):
            parsed = _safe_parse_excel(event)
        self.assertEqual(parsed["Stamp"], 42)
        self.assertEqual(parsed["Text"], "safe")

    def test_expression_is_not_executed(self) -> None:
        expression = '__import__("os").system("this-must-not-run")'
        with patch.object(ast, "literal_eval", side_effect=AssertionError("AST evaluation was called")) as literal:
            self.assertIsNone(_safe_parse_excel(expression))
        literal.assert_not_called()


class DecoderIntegrityTests(unittest.TestCase):
    def test_bad_event_byte_is_rejected(self) -> None:
        # One channel: event, one-byte mask, -1 sentinel, absolute value.
        payload = bytes([2, 1, 0xFF, 0xFF, 1, 0, 0, 0])
        with self.assertRaises(DataIntegrityError):
            decode_schema9_packet(
                io.BytesIO(payload),
                offset=0,
                byte_end=len(payload),
                sample_count=1,
                start=0,
                stop=1,
                n_channels=1,
                shorted=(False,),
                selected=(0,),
            )

    def test_shorted_channel_is_nan_without_payload_bytes(self) -> None:
        payload = bytes([0, 0])
        result = decode_schema9_packet(
            io.BytesIO(payload),
            offset=0,
            byte_end=len(payload),
            sample_count=1,
            start=0,
            stop=1,
            n_channels=1,
            shorted=(True,),
            selected=(0,),
        )
        self.assertTrue(np.isnan(result[0, 0]))


if __name__ == "__main__":
    unittest.main()
