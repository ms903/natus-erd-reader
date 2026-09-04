"""Bounded, incremental decoder for schema-9 compressed ERD packets."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from struct import unpack_from
from typing import BinaryIO, TYPE_CHECKING

from .errors import DataIntegrityError
from .limits import DEFAULT_LIMITS, ReadLimits, check_limit, check_output_size

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


_READ_CHUNK_BYTES = 64 * 1024


def _integer(value: int, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{context} must be an integer")
    return int(value)


def validate_packet_bounds(
    *,
    offset: int,
    byte_end: int,
    sample_count: int,
    n_channels: int,
    shorted: Sequence[bool],
    limits: ReadLimits = DEFAULT_LIMITS,
) -> None:
    """Reject impossible or oversized packet spans without reading any bytes."""
    offset = _integer(offset, "packet offset")
    byte_end = _integer(byte_end, "packet end")
    sample_count = _integer(sample_count, "packet sample count")
    n_channels = _integer(n_channels, "packet channel count")
    if offset < 0 or byte_end <= offset:
        raise DataIntegrityError("ERD packet has an invalid byte range")
    if not 1 <= sample_count <= 32767:
        raise DataIntegrityError("ERD packet sample count must be between 1 and 32767")
    if not 1 <= n_channels <= 1024:
        raise DataIntegrityError("ERD packet channel count must be between 1 and 1024")
    if len(shorted) != n_channels or any(value not in (False, True) for value in shorted):
        raise DataIntegrityError("ERD shorted mask must contain one boolean per channel")

    mask_bytes = (n_channels + 7) // 8
    active = sum(not value for value in shorted)
    maximum_sample_bytes = 1 + mask_bytes + 6 * active
    minimum_bytes = maximum_sample_bytes + (sample_count - 1) * (1 + mask_bytes + active)
    maximum_bytes = sample_count * maximum_sample_bytes
    packet_bytes = byte_end - offset
    if not minimum_bytes <= packet_bytes <= maximum_bytes:
        raise DataIntegrityError(
            "ERD packet byte span is impossible for its sample and channel counts"
        )
    check_limit(packet_bytes, limits.max_packet_bytes, "Compressed ERD packet bytes")


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
    limits: ReadLimits = DEFAULT_LIMITS,
    out: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Decode a half-open sample window from an independently compressed packet.

    Byte ranges and allocation budgets are checked before NumPy is imported.
    Reads are at most 64 KiB; the compressed working buffer is bounded by a
    chunk plus one sample. Only the prefix needed to reach stop is decoded.
    A complete-packet read additionally checks exact payload consumption.
    out permits writing directly into a caller-provided float64 array view.
    """
    validate_packet_bounds(
        offset=offset, byte_end=byte_end, sample_count=sample_count,
        n_channels=n_channels, shorted=shorted, limits=limits,
    )
    offset, byte_end = int(offset), int(byte_end)
    sample_count, n_channels = int(sample_count), int(n_channels)
    start, stop = _integer(start, "packet start"), _integer(stop, "packet stop")
    if not 0 <= start <= stop <= sample_count:
        raise ValueError("packet sample bounds are invalid")
    check_output_size(len(selected), stop - start, limits)
    selected_rows: list[tuple[int, int]] = []
    for row, raw_channel in enumerate(selected):
        channel = _integer(raw_channel, "selected channel")
        if not 0 <= channel < n_channels:
            raise ValueError(f"channel index is out of range: {channel}")
        if not shorted[channel]:
            selected_rows.append((row, channel))

    import numpy as np

    shape = (len(selected), stop - start)
    if out is None:
        output = np.full(shape, np.nan, dtype=np.float64)
    else:
        if not isinstance(out, np.ndarray) or out.shape != shape or out.dtype != np.dtype("float64"):
            raise ValueError("out must be a float64 NumPy array with the requested shape")
        if not out.flags.writeable:
            raise ValueError("out must be writeable")
        output = out
        output.fill(np.nan)
    if start == stop:
        return output

    mask_size = (n_channels + 7) // 8
    active_layout = tuple(
        (channel, channel >> 3, 1 << (channel & 7))
        for channel in range(n_channels)
        if not shorted[channel]
    )
    maximum_sample_bytes = 1 + mask_size + 6 * len(active_layout)
    packet_bytes = byte_end - offset
    # Limiting the prefix avoids reading an entire large packet for a tiny window.
    read_limit = min(packet_bytes, stop * maximum_sample_bytes)
    state = [0] * n_channels
    payload = b""
    position = 0
    consumed_before_buffer = 0
    bytes_read = 0
    payload_length = 0
    try:
        stream.seek(offset)
    except OSError as exc:
        raise DataIntegrityError(f"Cannot seek to ERD packet: {exc}") from exc

    def require(size: int, context: str) -> None:
        if position + size > payload_length:
            raise DataIntegrityError(
                f"ERD packet ended while reading {context} at sample {sample_index}"
            )

    for sample_index in range(stop):
        # Refill only between samples, leaving the tight channel loop free of I/O.
        if payload_length - position < maximum_sample_bytes and bytes_read < read_limit:
            tail = payload[position:]
            consumed_before_buffer += position
            position = 0
            requested = min(_READ_CHUNK_BYTES, read_limit - bytes_read)
            try:
                block = stream.read(requested)
            except OSError as exc:
                raise DataIntegrityError(f"Cannot read ERD packet: {exc}") from exc
            if len(block) != requested:
                raise DataIntegrityError("ERD packet is truncated")
            bytes_read += len(block)
            payload = tail + block
            payload_length = len(payload)

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
                raise DataIntegrityError("The first sample of an ERD packet is not absolute")
            state[channel] += value

        for channel in absolute_channels:
            require(4, "absolute channel value")
            state[channel] = unpack_from("<i", payload, position)[0]
            position += 4
        if sample_index == 0 and len(absolute_channels) != len(active_layout):
            raise DataIntegrityError("The first ERD sample does not initialize every channel")
        if sample_index >= start:
            output_column = sample_index - start
            for row, channel in selected_rows:
                output[row, output_column] = state[channel]

    if stop == sample_count and consumed_before_buffer + position != packet_bytes:
        raise DataIntegrityError("ERD packet has unexplained trailing bytes")
    return output
