"""Synthetic decoding and pre-allocation budget regressions."""

from __future__ import annotations

import builtins
import io
from struct import pack
import unittest
from unittest.mock import patch

import numpy as np

from natus_erd import DataIntegrityError, ReadLimits, ResourceLimitError
from natus_erd.decoder import decode_schema9_packet, validate_packet_bounds

from ._fixture import N_CHANNELS, SHORTED, _encode_packet


class SpyStream(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.requests: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.requests.append(size)
        if not 0 <= size <= 64 * 1024:
            raise AssertionError(f"Unbounded decoder read: {size}")
        return super().read(size)


def small_packet() -> bytes:
    # Channel 1 is shorted and consumes no delta or absolute bytes, even
    # though its bit is set. A wide -1 is an absolute-value sentinel.
    return (
        bytes((0, 0b111)) + pack("<hhii", -1, -1, 100, -100)
        + bytes((0, 0)) + pack("<bb", -128, 127)
        + bytes((1, 0b101)) + pack("<hhi", -300, -1, 200000)
    )


class DecoderBudgetTests(unittest.TestCase):
    def decode(self, stream: io.BytesIO, **overrides: object) -> np.ndarray:
        options = dict(
            offset=0, byte_end=len(small_packet()), sample_count=3,
            start=0, stop=3, n_channels=3, shorted=(False, True, False),
            selected=(2, 1, 0, 2),
        )
        options.update(overrides)
        return decode_schema9_packet(stream, **options)  # type: ignore[arg-type]

    def test_signed_deltas_sentinel_absolutes_and_shorted_channels(self) -> None:
        output = self.decode(SpyStream(small_packet()))
        np.testing.assert_array_equal(
            output, [[-100, 27, 200000], [np.nan, np.nan, np.nan],
                     [100, -28, -328], [-100, 27, 200000]],
        )

    def test_partial_read_only_loads_bounded_needed_prefix(self) -> None:
        stream = SpyStream(small_packet())
        output = self.decode(stream, stop=1)
        self.assertEqual(stream.requests, [14])
        self.assertLess(sum(stream.requests), len(small_packet()))
        self.assertEqual(output[2, 0], 100)

    def test_mid_packet_window_decodes_prior_deltas(self) -> None:
        output = self.decode(SpyStream(small_packet()), start=2)
        self.assertEqual(output.shape, (4, 1))
        self.assertEqual(output[0, 0], 200000)
        self.assertEqual(output[2, 0], -328)

    def test_out_is_returned_and_non_contiguous_views_work(self) -> None:
        parent = np.zeros((4, 6), dtype=np.float64)
        view = parent[:, ::2]
        output = self.decode(SpyStream(small_packet()), out=view)
        self.assertIs(output, view)
        np.testing.assert_array_equal(parent[:, 1::2], 0)
        self.assertTrue(np.isnan(output[1]).all())
        self.assertEqual(output[0, -1], 200000)

    def test_out_rejects_wrong_shape_dtype_and_readonly_arrays(self) -> None:
        readonly = np.zeros((4, 3), dtype=np.float64)
        readonly.setflags(write=False)
        for out in (np.zeros((4, 2)), np.zeros((4, 3), dtype=np.int32), readonly):
            with self.subTest(shape=out.shape, dtype=str(out.dtype), writable=out.flags.writeable):
                stream = SpyStream(small_packet())
                with self.assertRaises(ValueError):
                    self.decode(stream, out=out)
                self.assertEqual(stream.requests, [])

    def test_read_chunks_never_exceed_64_kib(self) -> None:
        samples = [[1000 + sample + channel for channel in range(N_CHANNELS)]
                   for sample in range(1000)]
        payload = _encode_packet(samples)
        self.assertGreater(len(payload), 64 * 1024)
        stream = SpyStream(payload)
        output = decode_schema9_packet(
            stream, offset=0, byte_end=len(payload), sample_count=len(samples),
            start=0, stop=len(samples), n_channels=N_CHANNELS,
            shorted=tuple(channel in SHORTED for channel in range(N_CHANNELS)),
            selected=(0, 249, 275),
        )
        self.assertGreater(len(stream.requests), 1)
        self.assertLessEqual(max(stream.requests), 64 * 1024)
        self.assertEqual(sum(stream.requests), len(payload))
        self.assertEqual(output[0, -1], 1999)
        self.assertEqual(output[2, -1], 2274)
        self.assertTrue(np.isnan(output[1]).all())

    def test_truncated_bytes_and_unexplained_full_packet_tail_fail(self) -> None:
        with self.assertRaises(DataIntegrityError):
            self.decode(SpyStream(small_packet()[:-1]))
        with self.assertRaises(DataIntegrityError):
            self.decode(SpyStream(small_packet() + b"\0"), byte_end=len(small_packet()) + 1)

    def test_first_sample_must_initialize_all_active_channels(self) -> None:
        payload = bytearray(small_packet())
        payload[1] = 0
        with self.assertRaises(DataIntegrityError):
            self.decode(SpyStream(bytes(payload)))

    def test_invalid_event_byte_is_rejected(self) -> None:
        payload = bytearray(small_packet())
        payload[0] = 7
        with self.assertRaises(DataIntegrityError):
            self.decode(SpyStream(bytes(payload)))

    def test_budget_and_impossible_bounds_fail_before_numpy_import_or_io(self) -> None:
        real_import = builtins.__import__

        def reject_numpy(name: str, *args: object, **kwargs: object) -> object:
            if name == "numpy" or name.startswith("numpy."):
                raise AssertionError("NumPy must not be imported before preflight checks")
            return real_import(name, *args, **kwargs)

        for overrides, expected in (
            ({"limits": ReadLimits(max_packet_bytes=10)}, ResourceLimitError),
            ({"limits": ReadLimits(max_read_bytes=8)}, ResourceLimitError),
            ({"limits": ReadLimits(max_selected_channels=1)}, ResourceLimitError),
            ({"limits": ReadLimits(max_read_samples=2)}, ResourceLimitError),
            ({"byte_end": 2**31}, DataIntegrityError),
            ({"byte_end": 1}, DataIntegrityError),
            ({"selected": (-1,)}, ValueError),
            ({"selected": (True,)}, TypeError),
        ):
            with self.subTest(overrides=overrides):
                stream = SpyStream(small_packet())
                with patch("builtins.__import__", side_effect=reject_numpy):
                    with self.assertRaises(expected):
                        self.decode(stream, **overrides)
                self.assertEqual(stream.requests, [])

    def test_packet_layout_counts_and_offsets_are_validated(self) -> None:
        valid = dict(offset=0, byte_end=len(small_packet()), sample_count=3,
                     n_channels=3, shorted=(False, True, False))
        for override in (
            {"offset": -1}, {"sample_count": 0}, {"sample_count": 32768},
            {"n_channels": 0}, {"n_channels": 1025}, {"shorted": (False,)},
            {"shorted": (False, 2, False)},
        ):
            with self.subTest(override=override):
                options = dict(valid)
                options.update(override)
                with self.assertRaises(DataIntegrityError):
                    validate_packet_bounds(**options)  # type: ignore[arg-type]

    def test_empty_window_does_not_read(self) -> None:
        stream = SpyStream(small_packet())
        self.assertEqual(self.decode(stream, stop=0).shape, (4, 0))
        self.assertEqual(stream.requests, [])


if __name__ == "__main__":
    unittest.main()
