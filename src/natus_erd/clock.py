"""Bounded SNC synchronization clocks, separate from nominal sample time.

SNC schema 1 supplies sparse stamp/FILETIME anchors, not a timestamp measured
for every sample. Interpolation and explicit nominal-rate extrapolation must
not be mistaken for externally certified acquisition-clock accuracy.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone as _timezone, tzinfo
from fractions import Fraction
from math import isfinite
from numbers import Integral, Real
import os
from pathlib import Path
import stat
from struct import iter_unpack, unpack_from
from typing import Literal
from zoneinfo import ZoneInfo

from ._parameters import integer as _integer
from .errors import DataIntegrityError, UnsupportedFormatError
from .limits import DEFAULT_LIMITS, ReadLimits, check_limit


_HEADER_BYTES = 352
_ENTRY_BYTES = 12
_SNC_GUID = bytes.fromhex("d2a98660af60d311986000104b75c151")
_TICKS_PER_SECOND = 10_000_000
_EPOCH = datetime(1601, 1, 1, tzinfo=_timezone.utc)
# Restrict to the datetime-representable interval, including its final whole
# microsecond. Keep sub-microsecond values exact everywhere else.
_MAX_TICKS = (
    (datetime.max.replace(tzinfo=_timezone.utc) - _EPOCH)
    // timedelta(microseconds=1)
) * 10
ClockEstimateKind = Literal["anchor", "interpolated", "extrapolated_nominal"]



def _stamp(value: object, label: str) -> int:
    result = _integer(value, label)
    if not -(2**31) <= result < 2**31:
        raise ValueError(f"{label} must fit a signed 32-bit stamp")
    return result


def _rate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("sample_rate must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("sample_rate must be finite and positive") from exc
    if not isfinite(result) or result <= 0:
        raise ValueError("sample_rate must be finite and positive")
    return result


def _recording_values(sample_rate: object, start_stamp: object, end_stamp: object) -> tuple[float, int, int]:
    rate = _rate(sample_rate)
    start = _stamp(start_stamp, "start_stamp")
    end = _stamp(end_stamp, "end_stamp")
    if end < start:
        raise ValueError("end_stamp must not precede start_stamp")
    if not isfinite((end - start + 1) / rate):
        raise ValueError("Recording duration must be finite")
    return rate, start, end


@dataclass(frozen=True, slots=True)
class ClockAnchor:
    """An original SNC stamp and exact integer FILETIME in 100 ns ticks."""

    stamp: int
    filetime_ticks: int

    def __post_init__(self) -> None:
        stamp = _stamp(self.stamp, "stamp")
        ticks = _integer(self.filetime_ticks, "filetime_ticks")
        if not 0 <= ticks <= _MAX_TICKS:
            raise ValueError("FILETIME is outside the supported datetime range")
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(self, "filetime_ticks", ticks)


@dataclass(frozen=True, slots=True)
class ClockEstimate:
    """Exact rational FILETIME and the way it was obtained.

    ``filetime_ticks`` preserves the original 100 ns resolution, and can have
    a fractional tick after interpolation. ``kind`` distinguishes source
    anchors from estimates; it does not certify the hardware clock.
    """

    filetime_ticks: Fraction
    kind: ClockEstimateKind

    def __post_init__(self) -> None:
        ticks = self.filetime_ticks
        if isinstance(ticks, bool) or not isinstance(ticks, (Fraction, Integral)):
            raise TypeError("filetime_ticks must be an integer or Fraction")
        ticks = Fraction(ticks)
        if not 0 <= ticks <= _MAX_TICKS:
            raise ValueError("FILETIME is outside the supported datetime range")
        if self.kind not in ("anchor", "interpolated", "extrapolated_nominal"):
            raise ValueError("Unknown clock estimate kind")
        object.__setattr__(self, "filetime_ticks", ticks)

    def to_datetime(self, timezone: str | tzinfo) -> datetime:
        """Return an aware datetime in an explicitly supplied timezone.

        FILETIME is interpreted using its UTC epoch. ``"UTC"`` needs no
        timezone database; other strings are IANA ``zoneinfo`` keys and fail
        if that zone is unavailable. No local timezone is guessed.

        Python datetime has microsecond resolution. This conversion rounds
        to the nearest microsecond, with exact halfway cases rounded to the
        even microsecond. The unrounded Fraction remains in filetime_ticks.
        Timezone conversion near datetime's limits can raise OverflowError.
        """
        target: tzinfo
        if isinstance(timezone, str):
            target = _timezone.utc if timezone == "UTC" else ZoneInfo(timezone)
        elif isinstance(timezone, tzinfo):
            target = timezone
        else:
            raise TypeError("timezone must be an explicit zone name or tzinfo")
        microseconds = round(self.filetime_ticks / 10)
        result = (_EPOCH + timedelta(microseconds=microseconds)).astimezone(target)
        if result.utcoffset() is None:
            raise ValueError("timezone must provide an explicit UTC offset")
        return result


def _read_snc(path: Path, limits: ReadLimits) -> bytes:
    """Reject special/link files and enforce budgets before payload allocation."""
    try:
        normalized = Path(os.path.abspath(path))
        if normalized.resolve(strict=True) != normalized:
            raise DataIntegrityError("SNC path must not traverse symbolic links")
        before = normalized.lstat()
        if not stat.S_ISREG(before.st_mode) or normalized.is_symlink():
            raise DataIntegrityError("SNC must be a regular file")
        if getattr(before, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise DataIntegrityError("SNC must not be a reparse-point file")
        if before.st_size < _HEADER_BYTES or (before.st_size - _HEADER_BYTES) % _ENTRY_BYTES:
            raise DataIntegrityError("SNC has a truncated header or entry")
        count = (before.st_size - _HEADER_BYTES) // _ENTRY_BYTES
        if not count:
            raise DataIntegrityError("SNC contains no clock anchors")
        check_limit(before.st_size, limits.max_metadata_bytes, "SNC metadata bytes")
        check_limit(count, limits.max_clock_anchors, "SNC clock anchors")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(normalized, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or (
                opened.st_size, opened.st_mtime_ns, opened.st_dev, opened.st_ino
            ) != (before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino):
                raise DataIntegrityError("SNC changed while opening")
            data = stream.read(before.st_size)
            after = os.fstat(stream.fileno())
            if len(data) != before.st_size or stream.read(1) or (
                after.st_size, after.st_mtime_ns
            ) != (before.st_size, before.st_mtime_ns):
                raise DataIntegrityError("SNC changed or was truncated while reading")
    except (OSError, RuntimeError) as exc:
        raise DataIntegrityError("Cannot safely read SNC metadata") from exc
    return data


@dataclass(frozen=True, slots=True)
class SNCClock:
    """Immutable sparse wall-clock mapping for one recording.

    ``end_stamp`` is the inclusive last sample. Anchors must lie within this
    recording and increase strictly in both stamp and FILETIME. Inside the
    anchor range, estimates use exact piecewise-linear interpolation.
    Nominal sample rate is used only for explicitly authorized extrapolation;
    it is never inferred from file creation/modification times.
    """

    sample_rate: float
    start_stamp: int
    end_stamp: int
    anchors: tuple[ClockAnchor, ...] = field(repr=False)
    _stamps: tuple[int, ...] = field(init=False, repr=False)
    _ticks_per_sample: Fraction = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rate, start, end = _recording_values(self.sample_rate, self.start_stamp, self.end_stamp)
        if not isinstance(self.anchors, tuple) or not self.anchors:
            raise ValueError("anchors must be a nonempty tuple of ClockAnchor values")
        previous = None
        for anchor in self.anchors:
            if not isinstance(anchor, ClockAnchor):
                raise TypeError("anchors must contain only ClockAnchor values")
            if not start <= anchor.stamp <= end:
                raise ValueError("SNC anchor lies outside the recording")
            if previous is not None and (
                anchor.stamp <= previous.stamp or anchor.filetime_ticks <= previous.filetime_ticks
            ):
                raise ValueError("SNC stamps and FILETIMEs must increase strictly")
            previous = anchor
        object.__setattr__(self, "sample_rate", rate)
        object.__setattr__(self, "start_stamp", start)
        object.__setattr__(self, "end_stamp", end)
        object.__setattr__(self, "_stamps", tuple(anchor.stamp for anchor in self.anchors))
        object.__setattr__(self, "_ticks_per_sample", Fraction(_TICKS_PER_SECOND) / Fraction.from_float(rate))

    @classmethod
    def from_file(
        cls, path: str | os.PathLike[str], *, sample_rate: float,
        start_stamp: int, end_stamp: int, limits: ReadLimits = DEFAULT_LIMITS,
    ) -> SNCClock:
        """Read schema-1 SNC metadata without importing NumPy or waveforms."""
        if not isinstance(limits, ReadLimits):
            raise TypeError("limits must be a ReadLimits instance")
        rate, start, end = _recording_values(sample_rate, start_stamp, end_stamp)
        data = _read_snc(Path(path), limits)
        if data[:16] != _SNC_GUID:
            raise DataIntegrityError("SNC file-type GUID does not match")
        if unpack_from("<HH", data, 16) != (1, 1):
            raise UnsupportedFormatError("Only SNC schema 1, base schema 1 is supported")
        try:
            anchors = tuple(ClockAnchor(stamp, ticks) for stamp, ticks in iter_unpack("<iQ", memoryview(data)[_HEADER_BYTES:]))
            return cls(rate, start, end, anchors)
        except (TypeError, ValueError) as exc:
            raise DataIntegrityError("Invalid SNC anchors or recording bounds") from exc

    @property
    def n_samples(self) -> int:
        return self.end_stamp - self.start_stamp + 1

    @property
    def first_anchor_sample(self) -> int:
        return self.anchors[0].stamp - self.start_stamp

    @property
    def last_anchor_sample(self) -> int:
        return self.anchors[-1].stamp - self.start_stamp

    def at_sample(
        self, sample: int, *, extrapolation: Literal["error", "nominal"] = "error",
        max_extrapolation_seconds: float = 10.0,
    ) -> ClockEstimate:
        """Map a relative sample to exact ticks and a provenance label.

        The exclusive end ``sample == n_samples`` is accepted as a boundary
        position, but still requires explicit extrapolation if unanchored.
        ``nominal`` extrapolation is bounded by distance to the closest source
        anchor, not by file creation time. The supplied positive finite sample
        rate is used exactly as its stored floating-point value.
        """
        sample = _integer(sample, "sample")
        if not 0 <= sample <= self.n_samples:
            raise IndexError("sample is outside the recording and its exclusive end")
        if extrapolation not in ("error", "nominal"):
            raise ValueError("extrapolation must be 'error' or 'nominal'")
        if isinstance(max_extrapolation_seconds, bool) or not isinstance(max_extrapolation_seconds, Real):
            raise TypeError("max_extrapolation_seconds must be a finite nonnegative number")
        try:
            maximum = float(max_extrapolation_seconds)
        except (OverflowError, ValueError) as exc:
            raise ValueError("max_extrapolation_seconds must be finite and nonnegative") from exc
        if not isfinite(maximum) or maximum < 0:
            raise ValueError("max_extrapolation_seconds must be finite and nonnegative")
        stamp = self.start_stamp + sample
        index = bisect_left(self._stamps, stamp)
        if index < len(self.anchors) and self.anchors[index].stamp == stamp:
            return ClockEstimate(Fraction(self.anchors[index].filetime_ticks), "anchor")
        if 0 < index < len(self.anchors):
            left, right = self.anchors[index - 1:index + 1]
            fraction = Fraction(stamp - left.stamp, right.stamp - left.stamp)
            ticks = left.filetime_ticks + fraction * (right.filetime_ticks - left.filetime_ticks)
            return ClockEstimate(ticks, "interpolated")
        if extrapolation == "error":
            raise ValueError("Sample is outside SNC anchors; nominal extrapolation must be explicit")
        anchor = self.anchors[0] if index == 0 else self.anchors[-1]
        delta_ticks = (stamp - anchor.stamp) * self._ticks_per_sample
        if abs(delta_ticks) > Fraction.from_float(maximum) * _TICKS_PER_SECOND:
            raise ValueError("Requested nominal extrapolation exceeds its time budget")
        return ClockEstimate(anchor.filetime_ticks + delta_ticks, "extrapolated_nominal")
