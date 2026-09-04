"""Read a small window without loading a complete NeuroWorks recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from natus_erd import NatusERDReader


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="seconds")
    parser.add_argument("--duration", type=float, default=1.0, help="seconds")
    args = parser.parse_args()
    reader = NatusERDReader.open(args.recording)
    start = round(args.start * reader.info.sample_rate)
    stop = start + round(args.duration * reader.info.sample_rate)
    values = reader.read_samples(start, stop, channels=[0, 1])
    print({"shape": values.shape, "unit": "uV", "finite_values": int(np.isfinite(values).sum())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
