"""Private bounded readback of EDF+C/D files produced by the exporter."""
from __future__ import annotations

from fractions import Fraction
from datetime import timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

from .errors import DataIntegrityError
from ._edf_codec import Calibration

if TYPE_CHECKING:
    from .edf_export import EdfExportPlan
    from .reader import NatusERDReader


def verify_export(
    reader: NatusERDReader, path: Path, plan: EdfExportPlan,
    calibrations: tuple[Calibration, ...], max_error_uv: float, progress=None,
) -> None:
    """Validate all digital codes/TALs and compare bounded windows to ERD."""
    import numpy as np
    from numpy.typing import NDArray
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
            if (header[:8].strip() != b"0" or header[192:236].strip() != plan.edf_format.encode("ascii")
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
                        or cal.dmin >= cal.dmax or Fraction(cal.pmin) == Fraction(cal.pmax)
                        or fields[2][row] != cal.unit):
                    raise ValueError("Invalid EDF calibration")
        except (ValueError, UnicodeError, ZeroDivisionError) as exc:
            raise DataIntegrityError("EDF readback header is invalid") from exc

        minimum = np.asarray([c.dmin for c in calibrations])[None, :, None]
        maximum = np.asarray([c.dmax for c in calibrations])[None, :, None]
        shorted_rows = [i for i, c in enumerate(plan.channels) if c in plan.shorted_channels]
        block_records = max(1, plan.chunk_samples//plan.record_samples)

        def check_block(first, payload):
            count = len(payload)//record_bytes
            codes: NDArray[np.int16] = np.ndarray((count, rows, plan.record_samples), dtype="<i2",
                               buffer=payload, strides=(record_bytes, plan.record_samples*2, 2))
            if np.any(codes < minimum) or np.any(codes > maximum):
                raise DataIntegrityError("EDF sample code is outside its declared digital range")
            if shorted_rows and np.any(codes[:, shorted_rows] != 32767):
                raise DataIntegrityError("EDF shorted channel differs from digital 32767")
            annotations: NDArray[np.uint8] = np.ndarray((count, plan.annotation_bytes), dtype="u1", buffer=payload,
                                     offset=wave_bytes, strides=(record_bytes, 1))
            expected = bytearray(count*plan.annotation_bytes)
            for offset in range(count):
                index = first+offset
                tal = _tal(plan._origin+Fraction(plan.record_sample(index))/plan.sample_rate)+plan._events.get(index, b"")
                begin = offset*plan.annotation_bytes
                expected[begin:begin+len(tal)] = tal
            if not np.array_equal(annotations, np.frombuffer(expected, dtype="u1").reshape(count, plan.annotation_bytes)):
                raise DataIntegrityError("EDF annotation or record onset differs from plan")
            return count

        if progress:
            progress({"stage": "verify", "records": 0, "total": plan.record_count})
        # Only the coordinator reads the disk. At most workers blocks are queued.
        with ThreadPoolExecutor(max_workers=plan.workers, thread_name_prefix="natus-verify") as pool:
            pending: deque[Future[int]] = deque()
            completed = 0
            for first in range(0, plan.record_count, block_records):
                if len(pending) == plan.workers:
                    completed += pending.popleft().result()
                    if progress:
                        progress({"stage": "verify", "records": completed, "total": plan.record_count})
                count = min(block_records, plan.record_count-first)
                payload = stream.read(count*record_bytes)
                if len(payload) != count*record_bytes:
                    raise DataIntegrityError("Truncated EDF record")
                pending.append(pool.submit(check_block, first, payload))
                del payload
            while pending:
                completed += pending.popleft().result()
                if progress and completed < plan.record_count:
                    progress({"stage": "verify", "records": completed, "total": plan.record_count})

        width_limit = min(16, reader.limits.max_read_samples, reader.limits.max_read_bytes//8)
        if width_limit < 1:
            from .errors import ResourceLimitError
            raise ResourceLimitError("EDF readback requires space for one source sample")
        # Each window uses one contiguous EDF read and batched source channels.
        # Small user read limits still apply to these independent source reads.
        def windows():
            for span, (a, b) in enumerate(plan.stored_ranges):
                length = b-a
                for local in sorted({0, length//4, length//2, length*3//4, max(0,length-16)}):
                    yield a+local, min(a+local+width_limit,b), plan._record_offsets[span], a

        for start, stop, record_offset, span_start in windows():
            first_record = record_offset+(start-span_start)//plan.record_samples
            last_record = record_offset+(stop-span_start+plan.record_samples-1)//plan.record_samples
            stream.seek(header_bytes+first_record*record_bytes)
            payload = stream.read((last_record-first_record)*record_bytes)
            codes: NDArray[np.int16] = np.ndarray((last_record-first_record, rows, plan.record_samples), dtype="<i2",
                               buffer=payload, strides=(record_bytes, plan.record_samples*2, 2))
            begin = start-span_start-(first_record-record_offset)*plan.record_samples
            values = codes.transpose(1, 0, 2).reshape(rows, -1)[:, begin:begin+stop-start]
            batch_rows = max(1, min(rows, reader.limits.max_selected_channels,
                                   reader.limits.max_read_bytes//(8*(stop-start))))
            for first in range(0, rows, batch_rows):
                selected = plan.channels[first:first+batch_rows]
                source = reader.read_samples(start, stop, selected, units="digital")
                for row, channel in enumerate(selected, first):
                    cal = calibrations[row]
                    if channel in plan.shorted_channels:
                        continue
                    if cal.raw:
                        reconstructed = cal.raw_min+(values[row].astype(np.int64)-cal.dmin)*cal.raw_step
                        expected_values = source[row-first]
                        tolerance = 0.0
                    else:
                        reconstructed = (values[row].astype(float)-cal.dmin)*(float(cal.pmax)-float(cal.pmin))/(cal.dmax-cal.dmin)+float(cal.pmin)
                        expected_values = source[row-first]*cal.source_scale
                        tolerance = max_error_uv+1e-9
                    if not np.all(np.abs(reconstructed-expected_values) <= tolerance):
                        raise DataIntegrityError("EDF readback differs from source waveform")
        if progress:
            progress({"stage": "verify", "records": plan.record_count, "total": plan.record_count})
