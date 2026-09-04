"""Importable example helpers; no command-line interface or import-time I/O."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

from natus_erd import NatusERDReader, ReadLimits

if TYPE_CHECKING:
    import numpy as np


def read_first_second(recording: str | Path) -> np.ndarray:
    """Read two channels with an explicit, conservative output budget."""

    reader = NatusERDReader.open(
        recording,
        limits=ReadLimits(max_read_bytes=16 * 1024 * 1024),
    )
    stop = min(reader.info.n_samples, ceil(reader.info.sample_rate))
    return reader.read_samples(0, stop, channels=[0, 1])


def count_channel_samples(recording: str | Path) -> int:
    """Demonstrate processing chunks without retaining previous arrays."""

    reader = NatusERDReader.open(recording)
    count = 0
    for chunk in reader.iter_samples(chunk_samples=20480, channels=[0, 1]):
        count += chunk.size
    return count
