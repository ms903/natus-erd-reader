"""Public lazy reader for Natus NeuroWorks schema-9 ERD recordings."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .binary import (
    ERD_HEADER_SIZE,
    ErdHeader,
    EtcEntry,
    StcEntry,
    read_erd_header,
    read_etc,
    read_stc,
)
from .decoder import decode_schema9_packet
from .ent import (
    EntNote,
    channel_names_from_notes,
    events_from_notes,
    read_ent_notes,
)
from .errors import DataIntegrityError, UnsupportedFormatError
from .models import ChannelInfo, Event, RecordingInfo, ValidationReport

ChannelSelector: TypeAlias = int | str | Sequence[int | str] | None

SIGNAL_CHANNEL_COUNT = 256
QUANTUM_HEADBOX_TYPE = 20
QUANTUM_UV_SCALE = -8711.0 / (2**21 - 0.5)


class NatusERDReader:
    """Lazy, read-only access to one native NeuroWorks recording."""

    def __init__(self, stc_path: Path) -> None:
        self._stc_path = stc_path
        self._directory = stc_path.parent
        self._stc = read_stc(stc_path)
        self._segments = self._stc.entries
        self._segment_ends = tuple(segment.end_stamp for segment in self._segments)
        self._etc_cache: dict[int, tuple[EtcEntry, ...]] = {}

        for segment in self._segments:
            erd_path, etc_path = self._segment_paths(segment)
            if not erd_path.is_file() or not etc_path.is_file():
                raise DataIntegrityError(
                    f"Missing ERD/ETC pair for STC segment {segment.index}"
                )

        header_path = self._find_header_erd()
        self._erd_header = read_erd_header(header_path)
        self._check_supported_header(self._erd_header)

        self._origin_stamp = self._segments[0].start_stamp
        end_stamp = self._segments[-1].end_stamp
        n_samples = end_stamp - self._origin_stamp + 1
        self._uv_scale = QUANTUM_UV_SCALE * (2**self._erd_header.discard_bits)

        self._notes = self._load_notes()
        montage_names = channel_names_from_notes(self._notes)
        channel_names = list(montage_names[: self._erd_header.n_channels])
        channel_names.extend(
            f"chan{index:03d}"
            for index in range(len(channel_names), self._erd_header.n_channels)
        )

        self._channels = tuple(
            ChannelInfo(
                index=index,
                name=channel_names[index],
                physical_index=self._erd_header.physical_channels[index],
                shorted=self._erd_header.shorted[index],
                is_signal=index < SIGNAL_CHANNEL_COUNT,
                unit="uV" if index < SIGNAL_CHANNEL_COUNT else None,
                scale_uv_per_count=(
                    self._uv_scale if index < SIGNAL_CHANNEL_COUNT else None
                ),
            )
            for index in range(self._erd_header.n_channels)
        )
        self._name_lookup: dict[str, int | None] = {}
        for channel in self._channels:
            if channel.name in self._name_lookup:
                self._name_lookup[channel.name] = None
            else:
                self._name_lookup[channel.name] = channel.index

        self._info = RecordingInfo(
            sample_rate=self._erd_header.sample_rate,
            n_samples=n_samples,
            n_recorded_channels=self._erd_header.n_channels,
            n_signal_channels=SIGNAL_CHANNEL_COUNT,
            segment_count=len(self._segments),
            start_stamp=self._origin_stamp,
            end_stamp=end_stamp,
            file_schema=self._erd_header.generic.file_schema,
            base_schema=self._erd_header.generic.base_schema,
            headbox_type=self._erd_header.headbox_types[0],
            delta_bits=self._erd_header.delta_bits,
            discard_bits=self._erd_header.discard_bits,
        )
        self._events: tuple[Event, ...] | None = None

    @classmethod
    def open(cls, path: str | Path) -> "NatusERDReader":
        """Open a recording directory or one of its EEG/STC/ERD files."""

        return cls(_resolve_stc(Path(path)))

    @property
    def info(self) -> RecordingInfo:
        return self._info

    @property
    def channels(self) -> tuple[ChannelInfo, ...]:
        return self._channels

    def sample_to_stamp(self, sample: int) -> int:
        """Convert a relative sample index to the native STC stamp."""

        value = _integer("sample", sample)
        if not 0 <= value <= self.info.n_samples:
            raise IndexError("sample is outside the recording")
        return self._origin_stamp + value

    def stamp_to_sample(self, stamp: int) -> int:
        """Convert a native STC stamp to a relative sample index."""

        value = _integer("stamp", stamp)
        sample = value - self._origin_stamp
        if not 0 <= sample <= self.info.n_samples:
            raise IndexError("stamp is outside the recording")
        return sample

    def read_samples(
        self,
        start: int,
        stop: int,
        channels: ChannelSelector = None,
        units: str = "uV",
    ) -> NDArray[np.float64]:
        """Read relative samples in the half-open interval ``[start, stop)``."""

        start_value = _integer("start", start)
        stop_value = _integer("stop", stop)
        if not 0 <= start_value <= stop_value <= self.info.n_samples:
            raise IndexError(
                f"sample range must satisfy 0 <= start <= stop <= {self.info.n_samples}"
            )
        if units not in {"uV", "digital"}:
            raise ValueError("units must be 'uV' or 'digital'")
        selected = self._resolve_channels(channels)
        if units == "uV" and any(index >= SIGNAL_CHANNEL_COUNT for index in selected):
            raise UnsupportedFormatError(
                "Physical-unit conversion is only validated for channels 0-255; "
                "request auxiliary channels with units='digital'"
            )

        output = np.full(
            (len(selected), stop_value - start_value), np.nan, dtype=np.float64
        )
        if start_value == stop_value or not selected:
            return output

        absolute_start = self._origin_stamp + start_value
        absolute_stop = self._origin_stamp + stop_value
        segment_index = bisect_left(self._segment_ends, absolute_start)

        for segment in self._segments[segment_index:]:
            if segment.start_stamp >= absolute_stop:
                break
            segment_start = max(absolute_start, segment.start_stamp)
            segment_stop = min(absolute_stop, segment.end_stamp + 1)
            if segment_start >= segment_stop:
                continue
            entries = self._load_etc(segment)
            if not entries:
                continue

            packet_ends = tuple(entry.end_stamp_exclusive for entry in entries)
            packet_index = bisect_right(packet_ends, segment_start)
            erd_path, _ = self._segment_paths(segment)
            erd_size = erd_path.stat().st_size
            try:
                stream = erd_path.open("rb")
            except OSError as exc:
                raise DataIntegrityError(
                    f"Cannot open ERD segment {segment.index}: {exc}"
                ) from exc
            with stream:
                for index in range(packet_index, len(entries)):
                    entry = entries[index]
                    if entry.sample_stamp >= segment_stop:
                        break
                    take_start = max(segment_start, entry.sample_stamp)
                    take_stop = min(segment_stop, entry.end_stamp_exclusive)
                    if take_start >= take_stop:
                        continue
                    byte_end = (
                        entries[index + 1].offset
                        if index + 1 < len(entries)
                        else erd_size
                    )
                    decoded = decode_schema9_packet(
                        stream,
                        offset=entry.offset,
                        byte_end=byte_end,
                        sample_count=entry.sample_span,
                        start=take_start - entry.sample_stamp,
                        stop=take_stop - entry.sample_stamp,
                        n_channels=self._erd_header.n_channels,
                        shorted=self._erd_header.shorted,
                        selected=selected,
                    )
                    destination_start = take_start - absolute_start
                    destination_stop = take_stop - absolute_start
                    output[:, destination_start:destination_stop] = decoded

        if units == "uV":
            output *= self._uv_scale
        return output

    def read_events(self) -> tuple[Event, ...]:
        """Return safely parsed ENT events sorted by native stamp."""

        if self._events is None:
            self._events = events_from_notes(self._notes, self._origin_stamp)
        return self._events

    def validate(self, *, deep: bool = True) -> ValidationReport:
        """Validate file pairs, headers, ETC offsets, and timestamp coverage."""

        packet_count = 0
        stored_samples = 0
        missing_samples = 0
        coverage_cursor = self._origin_stamp
        expected_header = _header_signature(self._erd_header)

        for segment in self._segments:
            erd_path, etc_path = self._segment_paths(segment)
            if not erd_path.is_file() or not etc_path.is_file():
                raise DataIntegrityError(
                    f"Missing ERD/ETC pair for STC segment {segment.index}"
                )
            if deep:
                actual_header = read_erd_header(erd_path)
                if _header_signature(actual_header) != expected_header:
                    raise DataIntegrityError(
                        f"ERD header differs in segment {segment.index}"
                    )

            entries = self._load_etc(segment)
            packet_count += len(entries)
            for entry in entries:
                if (
                    entry.sample_stamp < segment.start_stamp
                    or entry.end_stamp_exclusive > segment.end_stamp + 1
                ):
                    raise DataIntegrityError(
                        f"ETC packet {entry.index} is outside STC segment "
                        f"{segment.index}"
                    )
                if entry.sample_stamp < coverage_cursor:
                    raise DataIntegrityError(
                        f"Stored sample intervals overlap in segment {segment.index}"
                    )
                if entry.sample_stamp > coverage_cursor:
                    missing_samples += entry.sample_stamp - coverage_cursor
                coverage_cursor = entry.end_stamp_exclusive
                stored_samples += entry.sample_span

        recording_end = self.info.end_stamp + 1
        if coverage_cursor < recording_end:
            missing_samples += recording_end - coverage_cursor
        if stored_samples + missing_samples != self.info.n_samples:
            raise DataIntegrityError(
                "Stored and missing sample counts do not match the STC span"
            )
        return ValidationReport(
            segment_count=len(self._segments),
            packet_count=packet_count,
            logical_samples=self.info.n_samples,
            stored_samples=stored_samples,
            missing_samples=missing_samples,
            event_count=len(self.read_events()),
        )

    def _find_header_erd(self) -> Path:
        for segment in self._segments:
            erd_path, _ = self._segment_paths(segment)
            try:
                if erd_path.stat().st_size >= ERD_HEADER_SIZE:
                    return erd_path
            except OSError:
                continue
        raise DataIntegrityError("No readable ERD header was found")

    def _check_supported_header(self, header: ErdHeader) -> None:
        unsupported: list[str] = []
        if header.generic.file_schema != 9:
            unsupported.append(f"file schema {header.generic.file_schema}")
        if header.generic.base_schema != 1:
            unsupported.append(f"base schema {header.generic.base_schema}")
        if header.headbox_types[0] != QUANTUM_HEADBOX_TYPE:
            unsupported.append(f"headbox type {header.headbox_types[0]}")
        if header.n_channels != 276:
            unsupported.append(f"{header.n_channels} channels")
        if header.sample_rate != 2048.0:
            unsupported.append(f"sample rate {header.sample_rate:g} Hz")
        if header.delta_bits != 8:
            unsupported.append(f"delta width {header.delta_bits}")
        if header.discard_bits != 6:
            unsupported.append(f"discard width {header.discard_bits}")
        if any(factor != 32767 for factor in header.frequency_factors):
            unsupported.append("per-channel frequency factors")
        if unsupported:
            raise UnsupportedFormatError(
                "Unsupported NeuroWorks ERD layout: " + ", ".join(unsupported)
            )

    def _segment_paths(self, segment: StcEntry) -> tuple[Path, Path]:
        stem = segment.segment_name
        if stem.lower().endswith(".erd"):
            stem = stem[:-4]
        return self._directory / f"{stem}.erd", self._directory / f"{stem}.etc"

    def _load_notes(self) -> tuple[EntNote, ...]:
        ent_path = self._stc_path.with_suffix(".ent")
        if not ent_path.is_file():
            ent_path = self._stc_path.with_suffix(".ent.old")
        if not ent_path.is_file():
            return ()
        return read_ent_notes(ent_path)

    def _load_etc(self, segment: StcEntry) -> tuple[EtcEntry, ...]:
        cached = self._etc_cache.get(segment.index)
        if cached is not None:
            return cached
        erd_path, etc_path = self._segment_paths(segment)
        try:
            erd_size = erd_path.stat().st_size
        except OSError as exc:
            raise DataIntegrityError(
                f"Cannot stat ERD segment {segment.index}: {exc}"
            ) from exc
        entries = read_etc(etc_path, erd_size=erd_size)
        for entry in entries:
            if (
                entry.sample_stamp < segment.start_stamp
                or entry.end_stamp_exclusive > segment.end_stamp + 1
            ):
                raise DataIntegrityError(
                    f"ETC packet {entry.index} is outside STC segment {segment.index}"
                )
        self._etc_cache[segment.index] = entries
        return entries

    def _resolve_channels(self, channels: ChannelSelector) -> tuple[int, ...]:
        if channels is None:
            return tuple(range(SIGNAL_CHANNEL_COUNT))
        if isinstance(channels, (str, Integral)) and not isinstance(channels, bool):
            requested: Sequence[int | str] = (channels,)
        elif isinstance(channels, Sequence):
            requested = channels
        else:
            raise TypeError("channels must be names, integer indices, or a sequence")

        resolved: list[int] = []
        for selector in requested:
            if isinstance(selector, str):
                if selector not in self._name_lookup:
                    raise KeyError(f"Unknown channel name: {selector}")
                index = self._name_lookup[selector]
                if index is None:
                    raise KeyError(f"Ambiguous channel name: {selector}")
                resolved.append(index)
            elif isinstance(selector, Integral) and not isinstance(selector, bool):
                index = int(selector)
                if not 0 <= index < self.info.n_recorded_channels:
                    raise IndexError(f"Channel index is out of range: {index}")
                resolved.append(index)
            else:
                raise TypeError(f"Invalid channel selector: {selector!r}")
        return tuple(resolved)


def _integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _header_signature(header: ErdHeader) -> tuple[object, ...]:
    return (
        header.generic.file_schema,
        header.generic.base_schema,
        header.sample_rate,
        header.n_channels,
        header.delta_bits,
        header.physical_channels,
        header.headbox_types,
        header.discard_bits,
        header.shorted,
        header.frequency_factors,
    )


def _resolve_stc(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(path) from exc
    if any(part.casefold() == "decimated" for part in resolved.parts):
        raise UnsupportedFormatError("The Decimated derivative is outside v1 scope")

    if resolved.is_file():
        suffix = resolved.suffix.casefold()
        if suffix == ".stc":
            return resolved
        if suffix not in {".eeg", ".erd"}:
            raise ValueError("Expected a recording directory or EEG/STC/ERD file")
        if suffix == ".eeg":
            matching = resolved.with_suffix(".stc")
            if matching.is_file():
                return matching
        candidates = sorted(resolved.parent.glob("*.stc"))
    elif resolved.is_dir():
        candidates = sorted(resolved.glob("*.stc"))
        if not candidates:
            candidates = sorted(
                candidate
                for candidate in resolved.rglob("*.stc")
                if not any(
                    part.casefold() == "decimated"
                    for part in candidate.relative_to(resolved).parts
                )
            )
    else:
        raise ValueError("Expected a recording directory or EEG/STC/ERD file")

    if not candidates:
        raise FileNotFoundError("No STC file was found")
    if len(candidates) != 1:
        raise DataIntegrityError(
            f"Expected one main STC file, found {len(candidates)}"
        )
    return candidates[0]
