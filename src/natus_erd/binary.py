"""Binary structure parsing for NeuroWorks schema-9 recordings."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import os
from pathlib import Path
import stat
from struct import Struct, unpack_from

from .errors import DataIntegrityError, UnsupportedFormatError
from .limits import DEFAULT_LIMITS, ReadLimits, check_limit

GENERIC_HEADER_SIZE = 352
ERD_HEADER_SIZE = 8656
STC_PREFIX_SIZE = 408
STC_ENTRY_SIZE = 272
ETC_ENTRY_SIZE = 16


@dataclass(frozen=True, slots=True)
class GenericHeader:
    file_schema: int
    base_schema: int
    creation_time_unix: int


@dataclass(frozen=True, slots=True)
class ErdHeader:
    generic: GenericHeader
    sample_rate: float
    n_channels: int
    delta_bits: int
    physical_channels: tuple[int, ...]
    headbox_types: tuple[int, int, int, int]
    discard_bits: int
    shorted: tuple[bool, ...]
    frequency_factors: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StcEntry:
    index: int
    segment_name: str
    start_stamp: int
    end_stamp: int
    sample_number: int
    sample_span: int


@dataclass(frozen=True, slots=True)
class StcFile:
    generic: GenericHeader
    next_segment: int
    final: int
    entries: tuple[StcEntry, ...]


@dataclass(frozen=True, slots=True)
class EtcEntry:
    index: int
    offset: int
    sample_stamp: int
    sample_number: int
    sample_span: int
    unknown: int

    @property
    def end_stamp_exclusive(self) -> int:
        return self.sample_stamp + self.sample_span


def regular_file_size(path: Path) -> int:
    """Reject special files before opening them (e.g. named pipes/devices)."""
    try:
        info = path.stat()
    except OSError as exc:
        raise DataIntegrityError(f"Cannot stat {path.name}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise DataIntegrityError(f"Expected a regular file: {path.name}")
    return info.st_size


def _read_file(path: Path, *, maximum: int) -> bytes:
    """Check size before allocation and use a finite read even if a file grows."""
    size = regular_file_size(path)
    check_limit(size, maximum, "Metadata file bytes")
    try:
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != size:
                raise DataIntegrityError("Metadata changed while opening")
            data = stream.read(size)
            if len(data) != size or stream.read(1):
                raise DataIntegrityError("Metadata changed or was truncated while reading")
    except OSError as exc:
        raise DataIntegrityError(f"Cannot read {path.name}: {exc}") from exc
    return data


def check_generic_schema(header: GenericHeader, expected: int, context: str) -> None:
    if header.file_schema != expected or header.base_schema != 1:
        raise UnsupportedFormatError(
            f"Unsupported {context} schema {header.file_schema}, base {header.base_schema}"
        )


def _safe_segment_name(name: str) -> bool:
    # Windows device/drive/alternate-stream names are unsafe even on POSIX.
    if not name or name in {".", ".."} or name[-1] in " .":
        return False
    if any(char in '<>:"/\\|?*' or ord(char) < 32 for char in name):
        return False
    device = name.split(".", 1)[0].upper()
    return device not in {"CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$"} and not (
        len(device) == 4 and device[:3] in {"COM", "LPT"}
        and device[3] in "123456789\u00b9\u00b2\u00b3"
    )


def _generic_from_bytes(data: bytes, path: Path) -> GenericHeader:
    if len(data) < GENERIC_HEADER_SIZE:
        raise DataIntegrityError(
            f"{path.name} is shorter than the {GENERIC_HEADER_SIZE}-byte header"
        )
    file_schema, base_schema = unpack_from("<HH", data, 16)
    creation_time = unpack_from("<i", data, 20)[0]
    return GenericHeader(file_schema, base_schema, creation_time)


def read_generic_header(path: Path) -> GenericHeader:
    regular_file_size(path)
    try:
        with path.open("rb") as stream:
            data = stream.read(GENERIC_HEADER_SIZE)
    except OSError as exc:
        raise DataIntegrityError(f"Cannot read {path.name}: {exc}") from exc
    return _generic_from_bytes(data, path)


def read_erd_header(path: Path) -> ErdHeader:
    regular_file_size(path)
    try:
        with path.open("rb") as stream:
            data = stream.read(ERD_HEADER_SIZE)
    except OSError as exc:
        raise DataIntegrityError(f"Cannot read {path.name}: {exc}") from exc

    if len(data) < ERD_HEADER_SIZE:
        raise DataIntegrityError(
            f"{path.name} is shorter than the {ERD_HEADER_SIZE}-byte ERD header"
        )
    generic = _generic_from_bytes(data, path)
    check_generic_schema(generic, 9, "ERD")
    sample_rate = unpack_from("<d", data, 352)[0]
    n_channels, delta_bits = unpack_from("<ii", data, 360)
    if not 1 <= n_channels <= 1024:
        raise DataIntegrityError(
            f"{path.name} declares an invalid channel count: {n_channels}"
        )
    if not isfinite(sample_rate) or sample_rate <= 0:
        raise DataIntegrityError("ERD sample rate must be finite and positive")
    physical = unpack_from(f"<{n_channels}i", data, 368)
    headbox_values = unpack_from("<4i", data, 4464)
    headbox_types: tuple[int, int, int, int] = (
        int(headbox_values[0]),
        int(headbox_values[1]),
        int(headbox_values[2]),
        int(headbox_values[3]),
    )
    discard_bits = unpack_from("<i", data, 4556)[0]
    shorted_raw = unpack_from("<1024h", data, 4560)[:n_channels]
    if any(value not in (0, 1) for value in shorted_raw):
        raise DataIntegrityError("ERD shorted flags must be zero or one")
    frequency_factors = unpack_from("<1024h", data, 6608)[:n_channels]
    return ErdHeader(
        generic=generic,
        sample_rate=sample_rate,
        n_channels=n_channels,
        delta_bits=delta_bits,
        physical_channels=tuple(physical),
        headbox_types=headbox_types,
        discard_bits=discard_bits,
        shorted=tuple(bool(value) for value in shorted_raw),
        frequency_factors=tuple(frequency_factors),
    )


def read_stc(path: Path, *, limits: ReadLimits = DEFAULT_LIMITS) -> StcFile:
    data = _read_file(
        path, maximum=min(limits.max_metadata_bytes, STC_PREFIX_SIZE + limits.max_segments * STC_ENTRY_SIZE)
    )
    generic = _generic_from_bytes(data, path)
    check_generic_schema(generic, 1, "STC")
    if len(data) < STC_PREFIX_SIZE:
        raise DataIntegrityError(f"{path.name} has a truncated STC prefix")
    payload_size = len(data) - STC_PREFIX_SIZE
    if payload_size % STC_ENTRY_SIZE:
        raise DataIntegrityError(
            f"{path.name} has {payload_size % STC_ENTRY_SIZE} trailing STC bytes"
        )

    next_segment, final = unpack_from("<ii", data, GENERIC_HEADER_SIZE)
    entries: list[StcEntry] = []
    names: set[str] = set()
    for index, offset in enumerate(
        range(STC_PREFIX_SIZE, len(data), STC_ENTRY_SIZE)
    ):
        raw_name = data[offset : offset + 256].split(b"\0", 1)[0]
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DataIntegrityError(
                f"STC segment {index} has an invalid UTF-8 name"
            ) from exc
        if not _safe_segment_name(name):
            raise DataIntegrityError(f"STC segment {index} has an unsafe name")
        stem = name[:-4] if name.casefold().endswith(".erd") else name
        if not _safe_segment_name(stem) or stem.casefold() in names:
            raise DataIntegrityError(f"STC segment {index} has an unsafe or duplicate stem")
        names.add(stem.casefold())
        start, end, sample_number, span = unpack_from("<4i", data, offset + 256)
        if end < start or span != end - start + 1:
            raise DataIntegrityError(
                f"STC segment {index} has inconsistent stamp bounds"
            )
        if entries and start <= entries[-1].end_stamp:
            raise DataIntegrityError(f"STC segments overlap at entry {index}")
        entries.append(
            StcEntry(index, name, start, end, sample_number, span)
        )

    if not entries:
        raise DataIntegrityError(f"{path.name} contains no STC entries")
    return StcFile(generic, next_segment, final, tuple(entries))


_ETC_STRUCT = Struct("<iiihh")


def read_etc(
    path: Path, *, erd_size: int | None = None, limits: ReadLimits = DEFAULT_LIMITS
) -> tuple[EtcEntry, ...]:
    data = _read_file(
        path, maximum=min(limits.max_metadata_bytes, GENERIC_HEADER_SIZE + limits.max_packets_per_segment * ETC_ENTRY_SIZE)
    )
    check_generic_schema(_generic_from_bytes(data, path), 3, "ETC")
    payload_size = len(data) - GENERIC_HEADER_SIZE
    if payload_size % ETC_ENTRY_SIZE:
        raise DataIntegrityError(
            f"{path.name} has {payload_size % ETC_ENTRY_SIZE} trailing ETC bytes"
        )

    entries: list[EtcEntry] = []
    for index, offset in enumerate(
        range(GENERIC_HEADER_SIZE, len(data), ETC_ENTRY_SIZE)
    ):
        data_offset, stamp, sample_number, span, unknown = _ETC_STRUCT.unpack_from(
            data, offset
        )
        if span <= 0:
            raise DataIntegrityError(f"ETC packet {index} has invalid span {span}")
        if data_offset < ERD_HEADER_SIZE:
            raise DataIntegrityError(
                f"ETC packet {index} points inside the ERD header"
            )
        if erd_size is not None and data_offset >= erd_size:
            raise DataIntegrityError(
                f"ETC packet {index} offset is outside its ERD file"
            )
        if entries:
            previous = entries[-1]
            if data_offset <= previous.offset:
                raise DataIntegrityError(f"ETC offsets are not increasing at {index}")
            if stamp < previous.end_stamp_exclusive:
                raise DataIntegrityError(f"ETC packets overlap at entry {index}")
            expected_number = previous.sample_number + previous.sample_span
            if sample_number != expected_number:
                raise DataIntegrityError(
                    f"ETC sample numbers are discontinuous at entry {index}"
                )
        entries.append(
            EtcEntry(index, data_offset, stamp, sample_number, span, unknown)
        )
    return tuple(entries)
