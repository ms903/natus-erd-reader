"""Streaming decoder for schema-9 compressed ERD packets."""

from __future__ import annotations

from collections.abc import Sequence
from struct import unpack_from
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from .errors import DataIntegrityError


def decode_schema9_packet(
    stream: BinaryIO,
    *,
    offset: int,
    byte_end: int,
    sample_count: int,
    start: int,
    stop: int,
    n_channels: int,
    shorted: Sequence[bool],
    selected: Sequence[int],
) -> NDArray[np.float64]:
    """Decode ``[start, stop)`` from one independently compressed packet.

    The compressed packet is read as one bounded byte range. All samples
    before ``start`` are decoded because deltas depend on earlier values in
    the same packet; samples at or after ``stop`` are not decoded.
    """

    if not (0 <= start <= stop <= sample_count):
        raise ValueError("packet sample bounds are invalid")
    if len(shorted) != n_channels:
        raise ValueError("shorted mask length does not match n_channels")
    if byte_end <= offset:
        raise DataIntegrityError("ERD packet has an empty or reversed byte range")

    try:
        stream.seek(offset)
        payload = stream.read(byte_end - offset)
    except OSError as exc:
        raise DataIntegrityError(f"Cannot read ERD packet: {exc}") from exc
    if len(payload) != byte_end - offset:
        raise DataIntegrityError("ERD packet is truncated")

    output = np.full((len(selected), stop - start), np.nan, dtype=np.float64)
    if start == stop:
        return output

    selected_rows: list[tuple[int, int]] = []
    for row, channel in enumerate(selected):
        if not 0 <= channel < n_channels:
            raise ValueError(f"channel index is out of range: {channel}")
        if not shorted[channel]:
            selected_rows.append((row, channel))

    # A deltamask contains every declared channel. Shorted channels have mask
    # bits but no delta or absolute value in the payload.
    mask_size = (n_channels + 7) // 8
    active_layout = tuple(
        (channel, channel >> 3, 1 << (channel & 7))
        for channel in range(n_channels)
        if not shorted[channel]
    )
    state = [0] * n_channels
    position = 0
    payload_length = len(payload)

    def require(size: int, context: str) -> None:
        if position + size > payload_length:
            raise DataIntegrityError(
                f"ERD packet ended while reading {context} at sample {sample_index}"
            )

    for sample_index in range(stop):
        require(1 + mask_size, "event byte and delta mask")
        event = payload[position]
        position += 1
        if event not in (0, 1):
            raise DataIntegrityError(
                f"Invalid ERD event byte 0x{event:02x} at sample {sample_index}"
            )
        mask_position = position
        position += mask_size

        absolute_channels: list[int] = []
        for channel, mask_byte, mask_bit in active_layout:
            wide = bool(payload[mask_position + mask_byte] & mask_bit)
            if wide:
                require(2, "16-bit delta")
                value = payload[position] | (payload[position + 1] << 8)
                position += 2
                if value >= 0x8000:
                    value -= 0x10000
                if value == -1:
                    absolute_channels.append(channel)
                    continue
            else:
                require(1, "8-bit delta")
                value = payload[position]
                position += 1
                if value >= 0x80:
                    value -= 0x100

            if sample_index == 0:
                raise DataIntegrityError(
                    "The first sample of an ERD packet is not absolute"
                )
            state[channel] += value

        for channel in absolute_channels:
            require(4, "absolute channel value")
            state[channel] = unpack_from("<i", payload, position)[0]
            position += 4

        if sample_index == 0 and len(absolute_channels) != len(active_layout):
            raise DataIntegrityError(
                "The first sample of an ERD packet does not initialize every channel"
            )

        if sample_index >= start:
            output_column = sample_index - start
            for row, channel in selected_rows:
                output[row, output_column] = state[channel]

    return output
