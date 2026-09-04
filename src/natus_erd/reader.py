"""Public lazy reader for Natus NeuroWorks schema-9 ERD recordings."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
import os
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
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
from .decoder import decode_schema9_packet, validate_packet_bounds
from ._paths import RecordingDirectory, resolve_recording
from .ent import (
    EntNote,
    channel_names_from_notes,
    complete_channel_names,
    events_from_notes,
    read_ent_notes,
)
from .errors import DataIntegrityError, ResourceLimitError, UnsupportedFormatError
from .limits import DEFAULT_LIMITS, ReadLimits, check_limit, check_output_size
from .models import ChannelInfo, Event, RecordingInfo, ValidationReport

ChannelSelector: TypeAlias = int | str | Sequence[int | str] | None

SIGNAL_CHANNEL_COUNT = 256
QUANTUM_HEADBOX_TYPE = 20
QUANTUM_UV_SCALE = -8711.0 / (2**21 - 0.5)


@dataclass(frozen=True, slots=True)
class _CachedIndex:
    entries: tuple[EtcEntry, ...]
    packet_ends: tuple[int, ...]
    erd_signature: tuple[int, int, int, int]
    etc_signature: tuple[int, int, int, int]


class NatusERDReader:
    """Lazy, read-only access to one native NeuroWorks recording."""

    def __init__(
        self, stc_path: Path, *, limits: ReadLimits = DEFAULT_LIMITS,
        _files: RecordingDirectory | None = None,
    ) -> None:
        if not isinstance(limits, ReadLimits):
            raise TypeError("limits must be a ReadLimits instance")
        self._limits = limits
        self._cache_lock = RLock()
        self._stc_path = stc_path.resolve(strict=True)
        self._directory = self._stc_path.parent
        self._files = _files if _files is not None else RecordingDirectory(self._directory, limits=limits)
        if self._files.directory != self._directory:
            raise ValueError("Recording directory index does not match STC directory")
        self._files.lookup(self._stc_path.name)
        self._stc = read_stc(self._stc_path, limits=limits)
        self._segments = self._stc.entries
        self._segment_ends = tuple(segment.end_stamp for segment in self._segments)
        self._etc_cache: OrderedDict[int, _CachedIndex] = OrderedDict()

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
        if not isfinite(n_samples / self._erd_header.sample_rate):
            raise DataIntegrityError("Recording duration must be finite for the declared sample rate")
        self._uv_scale = QUANTUM_UV_SCALE * (2**self._erd_header.discard_bits)

        self._notes = self._load_notes()
        montage_names = channel_names_from_notes(self._notes, limits=self._limits)
        channel_names = complete_channel_names(montage_names, self._erd_header.n_channels)

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
    def open(
        cls, path: str | Path, *, limits: ReadLimits = DEFAULT_LIMITS
    ) -> "NatusERDReader":
        """Open an explicit recording directory or an EEG/STC/ERD file.

        Metadata access does not import NumPy. Directory discovery is not
        recursive. Input files must remain unchanged while the reader is used.
        """

        if not isinstance(limits, ReadLimits):
            raise TypeError("limits must be a ReadLimits instance")
        source = Path(path).expanduser().resolve(strict=True)
        stc_path, files = resolve_recording(source, limits=limits)
        if cls is NatusERDReader:
            reader = cls(stc_path, limits=limits, _files=files)
        else:
            # Preserve subclasses implementing the original constructor. They
            # may build a second name index, but need not accept internal args.
            reader = cls(stc_path, limits=limits)
        if source.suffix.casefold() == ".erd" and source.is_file():
            if not any(
                source == reader._segment_paths(segment)[0]
                for segment in reader._segments
            ):
                raise DataIntegrityError("The supplied ERD file is not a member of this STC recording")
        return reader

    @property
    def limits(self) -> ReadLimits:
        """The immutable resource budgets used by this reader."""
        return self._limits

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
        """Read ``[start, stop)`` as float64 ``(channels, samples)``.

        Budgets are checked before importing NumPy or allocating the result.
        Requests exceeding a budget raise ResourceLimitError; use iter_samples
        to process larger intervals without retaining the whole recording.
        """

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

        check_output_size(len(selected), stop_value - start_value, self._limits)
        absolute_start = self._origin_stamp + start_value
        absolute_stop = self._origin_stamp + stop_value
        # Preflight touched headers and indices before allocating even a valid
        # output window. The bounded LRU keeps a long traversal from growing RAM.
        if selected and start_value != stop_value:
            for segment in self._overlapping_segments(absolute_start, absolute_stop):
                self._load_etc(segment)

        import numpy as np

        try:
            output = np.full(
                (len(selected), stop_value - start_value), np.nan, dtype=np.float64
            )
        except MemoryError as exc:
            raise ResourceLimitError("Cannot allocate output; use a smaller window") from exc
        if start_value == stop_value or not selected:
            return output

        for segment in self._overlapping_segments(absolute_start, absolute_stop):
            segment_start = max(absolute_start, segment.start_stamp)
            segment_stop = min(absolute_stop, segment.end_stamp + 1)
            if segment_start >= segment_stop:
                continue
            index_data = self._load_etc(segment)
            entries = index_data.entries
            if not entries:
                continue

            packet_index = bisect_right(index_data.packet_ends, segment_start)
            erd_path, _ = self._segment_paths(segment)
            erd_size = index_data.erd_signature[0]
            try:
                stream = erd_path.open("rb")
            except OSError as exc:
                raise DataIntegrityError(
                    f"Cannot open ERD segment {segment.index}: {exc}"
                ) from exc
            with stream:
                if _stat_signature(os.fstat(stream.fileno())) != index_data.erd_signature:
                    raise DataIntegrityError("ERD changed while opening; reopen a static recording")
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
                    destination_start = take_start - absolute_start
                    destination_stop = take_stop - absolute_start
                    decode_schema9_packet(
                        stream,
                        offset=entry.offset,
                        byte_end=byte_end,
                        sample_count=entry.sample_span,
                        start=take_start - entry.sample_stamp,
                        stop=take_stop - entry.sample_stamp,
                        n_channels=self._erd_header.n_channels,
                        shorted=self._erd_header.shorted,
                        selected=selected,
                        limits=self._limits,
                        out=output[:, destination_start:destination_stop],
                    )
                if _stat_signature(os.fstat(stream.fileno())) != index_data.erd_signature:
                    raise DataIntegrityError("ERD changed while reading; reopen a static recording")

        if units == "uV":
            output *= self._uv_scale
        return output

    def _overlapping_segments(self, start: int, stop: int) -> Iterator[StcEntry]:
        index = bisect_left(self._segment_ends, start)
        while index < len(self._segments):
            segment = self._segments[index]
            if segment.start_stamp >= stop:
                break
            yield segment
            index += 1

    def iter_samples(
        self, start: int = 0, stop: int | None = None, *,
        chunk_samples: int = 20_480, channels: ChannelSelector = None, units: str = "uV",
    ) -> Iterator[NDArray[np.float64]]:
        """Yield successive independent arrays, retaining no earlier chunks.

        Each chunk is subject to the same limits as read_samples. Keeping all
        yielded arrays in a list defeats streaming and uses caller-owned RAM.
        """
        first = _integer("start", start)
        last = self.info.n_samples if stop is None else _integer("stop", stop)
        chunk = _integer("chunk_samples", chunk_samples)
        if not 0 <= first <= last <= self.info.n_samples:
            raise IndexError("iterator sample bounds are outside the recording")
        if chunk <= 0:
            raise ValueError("chunk_samples must be positive")
        selected = self._resolve_channels(channels)
        if units not in ("uV", "digital"):
            raise ValueError("units must be 'uV' or 'digital'")
        if units == "uV" and any(index >= SIGNAL_CHANNEL_COUNT for index in selected):
            raise UnsupportedFormatError("Auxiliary channels require units='digital'")
        check_output_size(len(selected), min(chunk, last - first), self._limits)
        for sample in range(first, last, chunk):
            yield self.read_samples(sample, min(sample + chunk, last), selected, units)

    def read_events(self) -> tuple[Event, ...]:
        """Return safely parsed ENT events sorted by native stamp."""

        with self._cache_lock:
            if self._events is None:
                self._events = events_from_notes(self._notes, self._origin_stamp)
            return self._events

    def validate(self, *, deep: bool = True) -> ValidationReport:
        """Validate file pairs, headers, ETC offsets, and timestamp coverage."""

        packet_count = 0
        stored_samples = 0
        missing_samples = 0
        coverage_cursor = self._origin_stamp

        for segment in self._segments:
            erd_path, etc_path = self._segment_paths(segment)
            if not erd_path.is_file() or not etc_path.is_file():
                raise DataIntegrityError(
                    f"Missing ERD/ETC pair for STC segment {segment.index}"
                )
            if deep:
                with self._cache_lock:
                    self._etc_cache.pop(segment.index, None)

            entries = self._load_etc(segment).entries
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
        if any(header.headbox_types[1:]):
            unsupported.append("multiple headboxes")
        if header.n_channels != 276:
            unsupported.append(f"{header.n_channels} channels")
        if header.delta_bits != 8:
            unsupported.append(f"delta width {header.delta_bits}")
        if header.discard_bits != 6:
            unsupported.append(f"discard width {header.discard_bits}")
        if header.physical_channels != tuple(range(header.n_channels)):
            unsupported.append("nonidentity physical-channel layout")
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
        erd = self._files.lookup(f"{stem}.erd")
        etc = self._files.lookup(f"{stem}.etc")
        assert erd is not None and etc is not None
        return erd, etc

    def _load_notes(self) -> tuple[EntNote, ...]:
        ent_path = self._files.lookup(self._stc_path.stem + ".ent", optional=True)
        if ent_path is None:
            ent_path = self._files.lookup(self._stc_path.stem + ".ent.old", optional=True)
        if ent_path is None:
            return ()
        return read_ent_notes(ent_path, limits=self._limits)

    def _load_etc(self, segment: StcEntry) -> _CachedIndex:
        with self._cache_lock:
            return self._load_etc_locked(segment)

    def _load_etc_locked(self, segment: StcEntry) -> _CachedIndex:
        erd_path, etc_path = self._segment_paths(segment)
        try:
            erd_signature = _stat_signature(erd_path.stat())
            etc_signature = _stat_signature(etc_path.stat())
        except OSError as exc:
            raise DataIntegrityError(f"Cannot stat segment {segment.index}: {exc}") from exc
        cached = self._etc_cache.get(segment.index)
        if cached is not None:
            if (erd_signature, etc_signature) != (cached.erd_signature, cached.etc_signature):
                raise DataIntegrityError("Recording changed after indexing; reopen a static recording")
            self._etc_cache.move_to_end(segment.index)
            return cached
        actual_header = read_erd_header(erd_path)
        self._check_supported_header(actual_header)
        if actual_header.sample_rate != self._erd_header.sample_rate:
            raise DataIntegrityError(
                f"ERD sample rate differs in segment {segment.index}: "
                f"expected {self._erd_header.sample_rate:g} Hz, got {actual_header.sample_rate:g} Hz"
            )
        if _header_signature(actual_header) != _header_signature(self._erd_header):
            raise DataIntegrityError(f"ERD header differs in segment {segment.index}")
        entries = read_etc(etc_path, erd_size=erd_signature[0], limits=self._limits)
        for index, entry in enumerate(entries):
            if (
                entry.sample_stamp < segment.start_stamp
                or entry.end_stamp_exclusive > segment.end_stamp + 1
            ):
                raise DataIntegrityError(
                    f"ETC packet {entry.index} is outside STC segment {segment.index}"
                )
            byte_end = entries[index + 1].offset if index + 1 < len(entries) else erd_signature[0]
            validate_packet_bounds(
                offset=entry.offset, byte_end=byte_end, sample_count=entry.sample_span,
                n_channels=actual_header.n_channels, shorted=actual_header.shorted,
                limits=self._limits,
            )
        if entries and entries[0].offset != ERD_HEADER_SIZE:
            raise DataIntegrityError("Unindexed bytes precede the first ERD packet")
        if not entries and erd_signature[0] != ERD_HEADER_SIZE:
            raise DataIntegrityError("ERD contains payload but its ETC index is empty")
        indexed_samples = sum(entry.sample_span for entry in entries)
        if indexed_samples != segment.stored_samples:
            raise DataIntegrityError(
                f"STC stored sample count differs from ETC in segment {segment.index}: "
                f"declared {segment.stored_samples}, indexed {indexed_samples}"
            )
        if (_stat_signature(erd_path.stat()), _stat_signature(etc_path.stat())) != (erd_signature, etc_signature):
            raise DataIntegrityError("Recording changed while indexing")
        value = _CachedIndex(entries, tuple(entry.end_stamp_exclusive for entry in entries), erd_signature, etc_signature)
        self._etc_cache[segment.index] = value
        while len(self._etc_cache) > self._limits.max_cached_segments:
            self._etc_cache.popitem(last=False)
        return value

    def _resolve_channels(self, channels: ChannelSelector) -> tuple[int, ...]:
        if channels is None:
            return tuple(range(SIGNAL_CHANNEL_COUNT))
        if isinstance(channels, (str, Integral)) and not isinstance(channels, bool):
            requested: Sequence[int | str] = (channels,)
        elif isinstance(channels, Sequence):
            requested = channels
        else:
            raise TypeError("channels must be names, integer indices, or a sequence")

        check_limit(len(requested), self._limits.max_selected_channels, "Selected channel count")

        resolved: list[int] = []
        for selector in requested:
            check_limit(len(resolved) + 1, self._limits.max_selected_channels, "Selected channel count")
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


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino)


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


def _resolve_stc(path: Path, *, limits: ReadLimits = DEFAULT_LIMITS) -> Path:
    return resolve_recording(path, limits=limits)[0]
