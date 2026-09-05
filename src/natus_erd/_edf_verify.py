"""Private bounded readback of the EDF+C file produced by the exporter."""
from __future__ import annotations

from fractions import Fraction
from datetime import timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import DataIntegrityError
from ._edf_codec import Calibration

if TYPE_CHECKING:
    from .edf_export import EdfExportPlan
    from .reader import NatusERDReader


def verify_export(
    reader: NatusERDReader, path: Path, plan: EdfExportPlan,
    calibrations: tuple[Calibration, ...], max_error_uv: float,
) -> None:
    """Validate all digital codes/TALs and compare bounded windows to ERD."""
    import numpy as np
    from .edf_export import _tal
    from .clock import ClockEstimate

    rows = len(plan.channels)
    header_bytes = 256*(rows+2)
    wave_bytes = rows*plan.record_samples*2
    record_bytes = wave_bytes+plan.annotation_bytes
    if path.stat().st_size != plan.output_bytes:
        raise DataIntegrityError("EDF readback length differs from plan")
    with path.open("rb") as stream:
        header = stream.read(header_bytes)
        try:
            start_time = ClockEstimate(Fraction(plan._header_ticks), "anchor").to_datetime(
                timezone(timedelta(seconds=plan._utc_offset_seconds)))
            if (header[:8].strip() != b"0" or header[192:236].strip() != b"EDF+C"
                    or header[168:176].decode() != start_time.strftime("%d.%m.%y")
                    or header[176:184].decode() != start_time.strftime("%H.%M.%S")
                    or int(header[184:192]) != header_bytes
                    or int(header[236:244]) != plan.record_count
                    or Fraction(header[244:252].decode().strip()) != Fraction(plan.record_samples)/plan.sample_rate
                    or int(header[252:256]) != rows+1):
                raise ValueError("Invalid EDF fixed header")
            fields = []
            cursor = 256
            for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
                fields.append([header[cursor+i*width:cursor+(i+1)*width].decode("ascii").strip()
                               for i in range(rows+1)])
                cursor += width*(rows+1)
            if fields[0] != [label for _, _, label in plan.channel_labels]+["EDF Annotations"]:
                raise ValueError("Invalid EDF labels")
            if [int(n) for n in fields[8]] != [plan.record_samples]*rows+[plan.annotation_bytes//2]:
                raise ValueError("Invalid EDF sample counts")
            for row, cal in enumerate(calibrations):
                if (Fraction(fields[3][row]) != Fraction(cal.pmin)
                        or Fraction(fields[4][row]) != Fraction(cal.pmax)
                        or int(fields[5][row]) != cal.dmin or int(fields[6][row]) != cal.dmax
                        or cal.dmin >= cal.dmax or Fraction(cal.pmin) >= Fraction(cal.pmax)
                        or fields[2][row] != ("" if cal.raw else "uV")):
                    raise ValueError("Invalid EDF calibration")
        except (ValueError, UnicodeError, ZeroDivisionError) as exc:
            raise DataIntegrityError("EDF readback header is invalid") from exc

        # Record size is limited to 61,440 bytes by preflight. All codes are
        # checked, including channels using a narrower digital calibration.
        for index in range(plan.record_count):
            payload = stream.read(record_bytes)
            if len(payload) != record_bytes:
                raise DataIntegrityError("Truncated EDF record")
            codes = np.frombuffer(payload[:wave_bytes], dtype="<i2").reshape(rows, plan.record_samples)
            for row, cal in enumerate(calibrations):
                if np.any(codes[row] < cal.dmin) or np.any(codes[row] > cal.dmax):
                    raise DataIntegrityError("EDF sample code is outside its declared digital range")
            expected = _tal(plan._origin+Fraction(index*plan.record_samples)/plan.sample_rate)+plan._events.get(index, b"")
            if payload[wave_bytes:] != expected.ljust(plan.annotation_bytes, b"\0"):
                raise DataIntegrityError("EDF annotation or record onset differs from plan")

        starts = sorted({0, plan.logical_samples//4, plan.logical_samples//2,
                         plan.logical_samples*3//4, max(0, plan.logical_samples-16)})
        width_limit = min(16, reader.limits.max_read_samples, reader.limits.max_read_bytes//8)
        if width_limit < 1:
            # The independent comparison needs one scalar source sample.
            from .errors import ResourceLimitError
            raise ResourceLimitError("EDF readback requires space for one source sample")
        for start in starts:
            stop = min(start+width_limit, plan.logical_samples)
            for row, (channel, cal) in enumerate(zip(plan.channels, calibrations)):
                source = reader.read_samples(plan.start+start, plan.start+stop, [channel], units="digital")[0]
                physical: list[float] = []
                for sample in range(start, stop):
                    record, column = divmod(sample, plan.record_samples)
                    stream.seek(header_bytes+record*record_bytes+2*(row*plan.record_samples+column))
                    code = int.from_bytes(stream.read(2), "little", signed=True)
                    if cal.raw:
                        physical.append(cal.raw_min+(code-cal.dmin)*cal.raw_step)
                    else:
                        physical.append((code-cal.dmin)*(float(cal.pmax)-float(cal.pmin))/(cal.dmax-cal.dmin)+float(cal.pmin))
                expected_values = source if cal.raw else source*cal.source_scale
                tolerance = 0.0 if cal.raw else max_error_uv+1e-9
                if not np.all(np.abs(np.asarray(physical)-expected_values) <= tolerance):
                    raise DataIntegrityError("EDF readback differs from source waveform")
