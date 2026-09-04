"""Optional integration check against an EDF exported from the same recording.

This developer tool is intentionally not part of the package dependencies.
Install PyEDFlib separately before running it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from natus_erd import NatusERDReader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("edf", type=Path)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--length", type=int, default=128)
    args = parser.parse_args()

    try:
        import pyedflib
    except ImportError as exc:
        parser.error(f"PyEDFlib is required: {exc}")

    native = NatusERDReader.open(args.recording)
    if args.windows < 1 or args.length < 1 or args.length > native.info.n_samples:
        parser.error("windows and length must be positive and fit the recording")

    edf = pyedflib.EdfReader(
        str(args.edf),
        annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS,
        check_file_size=pyedflib.DO_NOT_CHECK_FILE_SIZE,
    )
    try:
        if edf.signals_in_file < native.info.n_signal_channels:
            raise RuntimeError("EDF has fewer signal channels than the ERD recording")
        if any(
            edf.getSampleFrequency(channel) != native.info.sample_rate
            for channel in range(native.info.n_signal_channels)
        ):
            raise RuntimeError("EDF and ERD sampling frequencies differ")

        maximum_lsb_error = 0.0
        maximum_uv_error = 0.0
        saturation_samples = 0
        saturation_mismatches = 0
        starts = np.linspace(
            0,
            native.info.n_samples - args.length,
            num=args.windows,
            dtype=np.int64,
        )
        for start_value in starts:
            start = int(start_value)
            erd_digital = native.read_samples(
                start, start + args.length, units="digital"
            )
            erd_uv = native.read_samples(start, start + args.length, units="uV")
            edf_digital = np.vstack(
                [
                    edf.readSignal(
                        channel, start=start, n=args.length, digital=True
                    )
                    for channel in range(native.info.n_signal_channels)
                ]
            )
            edf_uv = np.vstack(
                [
                    edf.readSignal(
                        channel, start=start, n=args.length, digital=False
                    )
                    for channel in range(native.info.n_signal_channels)
                ]
            )

            finite = np.isfinite(erd_digital)
            in_edf_range = finite & (erd_digital >= -32768) & (erd_digital <= 32767)
            if np.any(in_edf_range):
                maximum_lsb_error = max(
                    maximum_lsb_error,
                    float(
                        np.max(
                            np.abs(erd_digital[in_edf_range] - edf_digital[in_edf_range])
                        )
                    ),
                )
                maximum_uv_error = max(
                    maximum_uv_error,
                    float(np.max(np.abs(erd_uv[in_edf_range] - edf_uv[in_edf_range]))),
                )
            above = finite & (erd_digital > 32767)
            below = finite & (erd_digital < -32768)
            saturation_samples += int(np.count_nonzero(above | below))
            saturation_mismatches += int(np.count_nonzero(above & (edf_digital != 32767)))
            saturation_mismatches += int(np.count_nonzero(below & (edf_digital != -32768)))

        print(f"Windows compared: {len(starts)} x {args.length} samples")
        print(f"Maximum in-range digital error: {maximum_lsb_error:g} LSB")
        print(f"Maximum in-range physical error: {maximum_uv_error:.9g} uV")
        print(f"ERD samples outside EDF int16 range: {saturation_samples}")
        print(f"Unexpected EDF saturation values: {saturation_mismatches}")
        return (
            0
            if maximum_lsb_error <= 1
            and maximum_uv_error <= 0.5
            and saturation_mismatches == 0
            else 1
        )
    finally:
        edf.close()


if __name__ == "__main__":
    raise SystemExit(main())
