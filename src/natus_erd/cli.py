"""Command-line interface for :mod:`natus_erd`."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from .errors import NatusERDError
from .reader import NatusERDReader


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="natus-erd", description="Inspect and lazily read Natus ERD recordings"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="show non-identifying structure")
    info.add_argument("path", type=Path)

    validate = subparsers.add_parser("validate", help="validate all segment indices")
    validate.add_argument("path", type=Path)
    validate.add_argument(
        "--quick", action="store_true", help="skip comparing every ERD header"
    )

    sample = subparsers.add_parser("sample", help="read a time window")
    sample.add_argument("path", type=Path)
    sample.add_argument("--start", type=float, required=True, help="start in seconds")
    sample.add_argument(
        "--duration", type=float, required=True, help="duration in seconds"
    )
    sample.add_argument(
        "--channels",
        help="comma-separated channel names or zero-based indices; default: 0-255",
    )
    sample.add_argument("--units", choices=("uV", "digital"), default="uV")
    sample.add_argument("--output", type=Path, help="optional .npy output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        reader = NatusERDReader.open(args.path)
        if args.command == "info":
            _print_info(reader)
        elif args.command == "validate":
            _print_validation(reader, deep=not args.quick)
        elif args.command == "sample":
            _sample(reader, args)
    except (NatusERDError, OSError, ValueError, TypeError, IndexError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _print_info(reader: NatusERDReader) -> None:
    info = reader.info
    print(f"ERD schema: {info.file_schema} (base {info.base_schema})")
    print(f"Headbox type: {info.headbox_type}")
    print(f"Sample rate: {info.sample_rate:g} Hz")
    print(f"Logical samples: {info.n_samples}")
    print(f"Duration: {info.duration_seconds / 3600:.6f} h")
    print(f"Segments: {info.segment_count}")
    print(
        f"Channels: {info.n_recorded_channels} recorded, "
        f"{info.n_signal_channels} calibrated signals"
    )
    print(f"Shorted signal channels: {sum(c.shorted for c in reader.channels[:256])}")
    print(f"ENT events: {len(reader.read_events())}")


def _print_validation(reader: NatusERDReader, *, deep: bool) -> None:
    report = reader.validate(deep=deep)
    print("Validation: OK")
    print(f"Segments: {report.segment_count}")
    print(f"Packets: {report.packet_count}")
    print(f"Logical samples: {report.logical_samples}")
    print(f"Stored samples: {report.stored_samples}")
    print(f"Missing samples: {report.missing_samples}")
    print(f"ENT events: {report.event_count}")


def _sample(reader: NatusERDReader, args: argparse.Namespace) -> None:
    if not math.isfinite(args.start) or args.start < 0:
        raise ValueError("--start must be a finite non-negative number")
    if not math.isfinite(args.duration) or args.duration < 0:
        raise ValueError("--duration must be a finite non-negative number")
    start = round(args.start * reader.info.sample_rate)
    stop = start + round(args.duration * reader.info.sample_rate)
    channels = _parse_channels(args.channels)
    data = reader.read_samples(start, stop, channels=channels, units=args.units)

    finite = data[np.isfinite(data)]
    print(f"Shape: {data.shape}")
    print(f"Units: {args.units}")
    print(f"NaN values: {data.size - finite.size}")
    if finite.size:
        print(f"Finite range: {finite.min():.12g} .. {finite.max():.12g}")
    else:
        print("Finite range: none")
    if args.output is not None:
        np.save(args.output, data, allow_pickle=False)
        output = args.output
        if output.suffix.casefold() != ".npy":
            output = output.with_name(output.name + ".npy")
        print(f"Saved: {output}")


def _parse_channels(value: str | None) -> list[int | str] | None:
    if value is None:
        return None
    selectors: list[int | str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("--channels contains an empty selector")
        try:
            selectors.append(int(item, 10))
        except ValueError:
            selectors.append(item)
    return selectors


if __name__ == "__main__":
    raise SystemExit(main())
