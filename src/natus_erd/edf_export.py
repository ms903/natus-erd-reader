"""Preflight and export of continuous, interoperable EDF+C files."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import tzinfo
from fractions import Fraction
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Literal, TypeAlias

from .clock import ClockEstimate
from .errors import ResourceLimitError, UnsupportedFormatError
from .limits import check_limit
from .reader import ChannelSelector, NatusERDReader
from ._parameters import integer as _integer
from ._edf_codec import escape_event, header_labels
from ._export_worker import execution

Progress: TypeAlias = Callable[[dict[str, int | float | str]], None]
EventPolicy: TypeAlias = Literal["full", "types", "none"]
Backend: TypeAlias = Literal["auto", "native", "python"]

def _decimal(value: Fraction | int) -> str:
    value = Fraction(value)
    numerator, denominator = value.numerator, value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1 or max(twos, fives) > 32:
        raise UnsupportedFormatError("Time cannot be expressed as a bounded exact EDF decimal")
    places = max(twos, fives)
    number = abs(numerator)*2**(places-twos)*5**(places-fives)
    digits = str(number).zfill(places+1)
    text = digits if not places else (digits[:-places]+"."+digits[-places:]).rstrip("0").rstrip(".")
    return ("-" if numerator < 0 else "")+text


@lru_cache(maxsize=32768)
def _duration_text(value: Fraction) -> str:
    text = _decimal(value)
    options = [text, text[1:] if text.startswith("0.") else text]
    digits = text.replace(".", "").lstrip("0")
    exponent = -(len(text.split(".")[1]) if "." in text else 0)
    while digits.endswith("0"):
        digits = digits[:-1]
        exponent += 1
    for point in range(1, len(digits)+1):
        coefficient = digits[:point]+("."+digits[point:] if point < len(digits) else "")
        power = exponent+len(digits)-point
        options.append(coefficient+(f"E{power}" if power else ""))
    result = min(options, key=lambda v: (len(v), "E" in v, v))
    if len(result) > 8:
        raise UnsupportedFormatError("Exact EDF data-record duration exceeds its header field")
    return result


def _field(value: object, width: int) -> bytes:
    text = str(value).encode("ascii")
    if len(text) > width:
        raise UnsupportedFormatError("EDF header field exceeds its fixed width")
    return text.ljust(width, b" ")


def _tal(onset: Fraction, text: str | None = None) -> bytes:
    stamp = _decimal(onset)
    if not stamp.startswith("-"):
        stamp = "+"+stamp
    return (stamp+"\x14"+("" if text is None else text)+"\x14\0").encode("utf-8")


@dataclass(frozen=True, slots=True)
class EdfExportPlan:
    """Exact EDF layout and bounded execution configuration, without decoding.

    Channel indices refer to the source recording. ``channel_labels`` contains
    (source index, source name, EDF label) tuples in exported row order.
    """

    start: int
    stop: int
    channels: tuple[int, ...]
    shorted_channels: tuple[int, ...]
    channel_labels: tuple[tuple[int, str, str], ...]
    sample_rate: Fraction
    record_samples: int
    record_duration_text: str
    record_count: int
    annotation_bytes: int
    output_bytes: int
    event_count: int
    backend: str = "python"
    workers: int = 1
    chunk_samples: int = 1
    memory_budget_bytes: int = 256*1024**2
    reserved_buffer_bytes: int = 0
    _header_ticks: int = field(default=0, repr=False)
    _utc_offset_seconds: int = field(default=0, repr=False)
    _origin: Fraction = field(default=Fraction(0), repr=False)
    _events: dict[int, bytes] = field(default_factory=dict, repr=False, compare=False)

    @property
    def logical_samples(self) -> int:
        """Number of samples per exported signal, with no padding."""
        return self.stop-self.start

    def record_starts(self) -> Iterator[int]:
        """Record start samples relative to the exported window."""
        return iter(range(0, self.logical_samples, self.record_samples))


@dataclass(frozen=True, slots=True)
class EdfExportResult:
    """Written file dimensions, calibration error in uV and execution timings."""

    record_count: int
    logical_samples: int
    file_bytes: int
    max_quantization_error_uv: float
    measured_max_error_uv: float
    event_count: int
    channel_count: int
    shorted_channels: tuple[int, ...]
    channel_labels: tuple[tuple[int, str, str], ...]
    uncalibrated_channels: int
    backend: str
    workers: int
    chunk_samples: int
    elapsed_seconds: float
    scan_seconds: float
    write_seconds: float


def _clock_metadata(
    reader: NatusERDReader, first: int, last: int, rate: Fraction,
    timezone: str | tzinfo, extrapolation: Literal["error", "nominal"], maximum: float,
) -> tuple[int, int, Fraction]:
    clock = reader.read_clock()
    for sample in (first, last-1):
        clock.at_sample(sample, extrapolation=extrapolation, max_extrapolation_seconds=maximum)
    anchor = clock.anchors[0]
    ticks = round(Fraction(anchor.filetime_ticks)
                  + Fraction(reader.info.start_stamp+first-anchor.stamp)/rate*10_000_000)
    header_ticks = ticks//10_000_000*10_000_000
    dt = ClockEstimate(Fraction(header_ticks), "anchor").to_datetime(timezone)
    offset = dt.utcoffset()
    if not 1985 <= dt.year <= 2084 or offset is None or offset.total_seconds() % 60:
        raise UnsupportedFormatError("EDF requires a date in 1985–2084 and a whole-minute timezone offset")
    return header_ticks, int(offset.total_seconds()), Fraction(ticks-header_ticks, 10_000_000)


def _layout(
    reader: NatusERDReader, first: int, last: int, selected: tuple[int, ...],
    shorted: tuple[int, ...], labels: tuple[tuple[int, str, str], ...],
    rate: Fraction, points: int, duration: str, source_events: tuple[tuple[int, bytes], ...],
    clock: tuple[int, int, Fraction], slot: int | Literal["auto"],
) -> EdfExportPlan:
    count = (last-first)//points
    if not 1 <= count <= 99_999_999:
        raise UnsupportedFormatError("EDF record count exceeds its 8-character field")
    event_records: dict[int, bytes] = {}
    for sample, tal in source_events:
        index = max(0, min(count-1, sample//points))
        event_records[index] = event_records.get(index, b"")+tal
    origin = clock[2]
    # Bound decimal lengths for every onset without enumerating the records.
    quantum = Fraction(1, (Fraction(points)/rate).denominator*origin.denominator)
    places = len(_decimal(quantum).partition(".")[2])
    last_onset = origin+Fraction(last-first-points)/rate
    time_bound = max(len(str(abs(int(origin)))), len(str(abs(int(last_onset)))))+places+6
    required = time_bound+max(map(len, event_records.values()), default=0)
    required += required % 2
    size = required if slot == "auto" else slot
    limit = min(60000, reader.limits.max_metadata_bytes,
                min(61440, reader.limits.max_packet_bytes)-2*len(selected)*points)
    if size < required or size > limit:
        raise ResourceLimitError("EDF annotation capacity cannot contain complete event TALs")
    total = 256*(len(selected)+2)+count*(2*len(selected)*points+size)
    return EdfExportPlan(first, last, selected, shorted, labels, rate, points, duration,
                         count, size, total, len(source_events), _header_ticks=clock[0],
                         _utc_offset_seconds=clock[1], _origin=origin, _events=event_records)


def plan_edf(
    reader: NatusERDReader, *, timezone: str | tzinfo = "UTC", start: int = 0,
    stop: int | None = None, channels: ChannelSelector = None, events: EventPolicy = "full",
    extrapolation: Literal["error", "nominal"] = "error", max_extrapolation_seconds: float = 10.0,
    annotation_bytes: int | Literal["auto"] = "auto", max_output_bytes: int | None = None,
    backend: Backend = "auto", workers: int | Literal["auto"] = "auto",
    memory_budget_bytes: int = 256*1024**2, chunk_samples: int | None = None,
    progress: Progress | None = None,
) -> EdfExportPlan:
    """Plan a fully stored [start, stop) window using source sample indices.

    Defaults select every recorded channel, dropping shorted channels. Events
    cover the whole source recording; ``types`` replaces text with note types.
    UTC is the default EDF wall clock. Named timezones require timezone data.
    No waveforms are decoded here: export's range scan checks quantization.
    Gaps or an inexact record grid raise UnsupportedFormatError; insufficient
    buffer, annotation or explicit output budgets raise ResourceLimitError.
    """
    first = _integer(start, "start")
    last = reader.info.n_samples if stop is None else _integer(stop, "stop")
    if not 0 <= first < last <= reader.info.n_samples:
        raise ValueError("Invalid export window")
    if max_output_bytes is not None:
        max_output_bytes = _integer(max_output_bytes, "max_output_bytes")
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
    if events not in ("full", "types", "none"):
        raise ValueError("events must be full, types, or none")
    if (isinstance(max_extrapolation_seconds, bool)
            or not isinstance(max_extrapolation_seconds, (int, float))
            or not isfinite(max_extrapolation_seconds) or not 0 <= max_extrapolation_seconds <= 10):
        raise ValueError("Export extrapolation must be between zero and ten seconds")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    if annotation_bytes != "auto":
        annotation_bytes = _integer(annotation_bytes, "annotation_bytes")
        if not 2 <= annotation_bytes <= 60000 or annotation_bytes % 2:
            raise ValueError("annotation_bytes must be auto or an even integer from 2 through 60000")
    selected = (tuple(range(reader.info.n_recorded_channels)) if channels is None
                else reader._resolve_channels(channels))
    check_limit(len(selected), reader.limits.max_selected_channels, "Selected EDF channels")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Export channels must be nonempty and unique")
    shorted = tuple(c for c in selected if reader.channels[c].shorted)
    selected = tuple(c for c in selected if not reader.channels[c].shorted)
    if not selected:
        raise UnsupportedFormatError("Export has no non-shorted channels")
    check_limit(256*(len(selected)+2), reader.limits.max_metadata_bytes, "EDF header bytes")
    check_limit(len(selected)+1, reader.limits.max_parse_nodes, "EDF signal count")
    spans = []
    for span in reader.iter_stored_ranges(first, last):
        spans.append(span)
        check_limit(len(spans), reader.limits.max_segments, "EDF interval count")
    if spans != [(first, last)]:
        choices = ", ".join(f"[{a}, {b})" for a, b in spans) or "none in this window"
        raise UnsupportedFormatError(f"EDF+C requires a continuous, fully stored window; available intervals: {choices}")
    rate = Fraction.from_float(reader.info.sample_rate)
    clock = _clock_metadata(reader, first, last, rate, timezone, extrapolation, max_extrapolation_seconds)
    source_events = []
    event_bytes = 0
    for event in (() if events == "none" else reader.read_events()):
        sample = event.stamp-reader.info.start_stamp-first
        text = event.text if events == "full" else f"type:{event.note_type if event.note_type is not None else 'unknown'}"
        tal = _tal(clock[2]+Fraction(sample)/rate, escape_event(text))
        event_bytes += len(tal)
        check_limit(event_bytes, reader.limits.max_metadata_bytes, "EDF events bytes")
        source_events.append((sample, tal))
    names = tuple(reader.channels[c].name for c in selected)
    labels = tuple(zip(selected, names, header_labels(names)))
    minimum_slot = 8 if annotation_bytes == "auto" else annotation_bytes
    maximum_points = (min(61440, reader.limits.max_packet_bytes)-minimum_slot)//(len(selected)*2)
    if maximum_points < 1:
        raise ResourceLimitError("Minimum EDF record exceeds its byte budget")
    step = rate.numerator
    for prime in (2, 5):
        while step % prime == 0:
            step //= prime
    best = None
    last_error = None
    suggestion = None
    for points in range(step, maximum_points+1, step):
        try:
            duration = _duration_text(Fraction(points)/rate)
        except UnsupportedFormatError:
            continue
        aligned = (last-first)//points*points
        if aligned:
            suggestion = max(suggestion or 0, aligned)
        if (last-first) % points:
            continue
        try:
            candidate = _layout(reader, first, last, selected, shorted, labels, rate,
                                points, duration, tuple(source_events), clock, annotation_bytes)
        except (ResourceLimitError, UnsupportedFormatError) as exc:
            last_error = exc
            continue
        if best is None or (candidate.output_bytes, -points) < (best.output_bytes, -best.record_samples):
            best = candidate
    if best is None:
        if last_error is not None:
            raise last_error
        suffix = f"; candidate aligned window start={first}, stop={first+suggestion}" if suggestion else ""
        raise UnsupportedFormatError("No exact EDF record grid for this window"+suffix)
    if max_output_bytes is not None:
        check_limit(best.output_bytes, max_output_bytes, "Planned EDF file bytes")
    config = execution(reader, selected, best.record_samples, best.annotation_bytes, best.output_bytes,
                       backend, workers, memory_budget_bytes, chunk_samples)
    best = replace(best, backend=config.backend, workers=config.workers, chunk_samples=config.chunk_samples,
                   memory_budget_bytes=config.memory_budget_bytes, reserved_buffer_bytes=config.reserved_bytes)
    if progress:
        progress({"stage": "planned", "file_bytes": best.output_bytes, "channels": len(selected),
                  "record_samples": best.record_samples, "annotation_bytes": best.annotation_bytes,
                  "backend": best.backend, "workers": best.workers, "chunk_samples": best.chunk_samples})
    return best


def export_edf(
    reader: NatusERDReader, path: str | Path, *, timezone: str | tzinfo = "UTC",
    start: int = 0, stop: int | None = None, channels: ChannelSelector = None,
    events: EventPolicy = "full", extrapolation: Literal["error", "nominal"] = "error", max_extrapolation_seconds: float = 10.0,
    annotation_bytes: int | Literal["auto"] = "auto", max_output_bytes: int | None = None,
    max_error_uv: float = 0.5, backend: Backend = "auto", workers: int | Literal["auto"] = "auto",
    memory_budget_bytes: int = 256*1024**2, chunk_samples: int | None = None,
    progress: Progress | None = None,
) -> EdfExportResult:
    """Write EDF+C with a range scan, bounded encoding and checked publication.

    EEG is calibrated in uV with a maximum quantization error of 0.5 uV by
    default. Auxiliary channels preserve integer values with an empty unit.
    Unrepresentable ranges fail before writing. Existing paths are never
    overwritten. Source changes, invalid codes and failed readback comparisons
    raise DataIntegrityError. See plan_edf for window and execution options.
    """
    from ._edf_write import write_export
    return write_export(reader, path, timezone=timezone, start=start, stop=stop,
        channels=channels, events=events, extrapolation=extrapolation,
        max_extrapolation_seconds=max_extrapolation_seconds, annotation_bytes=annotation_bytes,
        max_output_bytes=max_output_bytes, max_error_uv=max_error_uv,
        backend=backend, workers=workers, memory_budget_bytes=memory_budget_bytes,
        chunk_samples=chunk_samples, progress=progress)


