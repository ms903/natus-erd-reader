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
]

__version__ = "0.2.1"
