"""Immutable public result types for the Natus ERD reader."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecordingInfo:
    """Structural metadata for a recording."""

    sample_rate: float
    n_samples: int
    n_recorded_channels: int
    n_signal_channels: int
    segment_count: int
    start_stamp: int
    end_stamp: int
    file_schema: int
    base_schema: int
    headbox_type: int
    delta_bits: int
    discard_bits: int

    @property
    def duration_seconds(self) -> float:
        """Logical recording duration, including any gaps."""

        return self.n_samples / self.sample_rate


@dataclass(frozen=True, slots=True)
class ChannelInfo:
    """Description of one channel in the ERD header."""

    index: int
    name: str
    physical_index: int
    shorted: bool
    is_signal: bool
    unit: str | None
    scale_uv_per_count: float | None
    name_resolved: bool = True


@dataclass(frozen=True, slots=True)
class Event:
    """An annotation from the ENT file."""

    stamp: int
    sample: int
    text: str
    user: str | None = None
    note_type: int | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Summary returned after structural validation."""

    segment_count: int
    packet_count: int
    logical_samples: int
    stored_samples: int
    missing_samples: int
    event_count: int
    ent_record_count: int
    unparsed_ent_record_count: int
