"""Lazy native reader for Natus NeuroWorks ERD recordings."""

from .errors import DataIntegrityError, NatusERDError, UnsupportedFormatError
from .edf import EDFInfo, EDFReader, EDFSignal
from .models import ChannelInfo, Event, RecordingInfo, ValidationReport
from .reader import NatusERDReader

__all__ = [
    "ChannelInfo",
    "DataIntegrityError",
    "EDFInfo",
    "EDFReader",
    "EDFSignal",
    "Event",
    "NatusERDError",
    "NatusERDReader",
    "RecordingInfo",
    "UnsupportedFormatError",
    "ValidationReport",
]

__version__ = "0.1.0"
