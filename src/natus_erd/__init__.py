"""Lazy native reader for Natus NeuroWorks ERD recordings."""

from .errors import (
    DataIntegrityError,
    NatusERDError,
    ResourceLimitError,
    UnsupportedFormatError,
)
from .limits import ReadLimits
from .models import ChannelInfo, Event, RecordingInfo, ValidationReport
from .reader import NatusERDReader
from .clock import ClockAnchor, ClockEstimate, SNCClock
from .edf_export import EdfExportPlan, EdfExportResult, export_edf, plan_edf

__all__ = [
    "ChannelInfo",
    "DataIntegrityError",
    "Event",
    "NatusERDError",
    "NatusERDReader",
    "RecordingInfo",
    "ReadLimits",
    "ResourceLimitError",
    "UnsupportedFormatError",
    "ValidationReport",
    "ClockAnchor",
    "ClockEstimate",
    "SNCClock",
    "EdfExportPlan",
    "EdfExportResult",
    "export_edf",
    "plan_edf",
]

__version__ = "0.3.0rc1"
