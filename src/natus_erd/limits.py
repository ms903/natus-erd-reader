"""Explicit per-reader resource budgets; no system-wide settings are changed."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral

from .errors import ResourceLimitError


@dataclass(frozen=True, slots=True)
class ReadLimits:
    """Finite budgets for allocations, index objects and annotation parsing.

    ``max_read_bytes`` bounds each returned float64 array, not total process
    memory. NumPy/BLAS and caller-retained arrays have separate overhead.
    Raising a budget is an explicit trust/resource decision by the caller.
    """

    max_read_bytes: int = 64 * 1024**2
    max_read_samples: int = 131_072
    max_packet_bytes: int = 8 * 1024**2
    max_metadata_bytes: int = 8 * 1024**2
    max_segments: int = 10_000
    max_packets_per_segment: int = 20_000
    max_cached_segments: int = 4
    max_selected_channels: int = 1024
    max_directory_entries: int = 20_000
    max_ent_bytes: int = 8 * 1024**2
    max_ent_record_bytes: int = 256 * 1024
    max_ent_records: int = 4096
    max_parse_depth: int = 32
    max_parse_nodes: int = 20_000
    max_total_parse_nodes: int = 200_000

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer")
            # Normalize integer subclasses to avoid overflow in their operators.
            object.__setattr__(self, field.name, int(value))
        if self.max_parse_depth > 128:
            raise ValueError("max_parse_depth must not exceed 128")


DEFAULT_LIMITS = ReadLimits()


def check_limit(value: int, maximum: int, context: str) -> None:
    if value > maximum:
        raise ResourceLimitError(
            f"{context} requires {value}, exceeding the configured limit {maximum}"
        )


def check_output_size(channels: int, samples: int, limits: ReadLimits) -> None:
    check_limit(channels, limits.max_selected_channels, "Selected channel count")
    check_limit(samples, limits.max_read_samples, "Samples per read; use iter_samples")
    check_limit(channels * samples * 8, limits.max_read_bytes, "Output bytes; use iter_samples or a smaller window")
