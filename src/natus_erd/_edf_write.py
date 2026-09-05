"""Ordered, no-clobber EDF writer; numerical work lives in bounded jobs."""
from __future__ import annotations

from datetime import timedelta, timezone
from fractions import Fraction
from math import isfinite
import os
from pathlib import Path
import shutil
import time
import uuid

from .clock import ClockEstimate
from .edf_export import EdfExportResult, _field, _integer, _tal, plan_edf
from ._edf_codec import calibrate
from ._export_worker import combine_stats, ordered_work, record_blocks
from .errors import DataIntegrityError, ResourceLimitError
from .reader import _stat_signature


def _header(plan, calibrations):
    dt = ClockEstimate(Fraction(plan._header_ticks), "anchor").to_datetime(
        timezone(timedelta(seconds=plan._utc_offset_seconds)))
    months = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    fmt = "EDF+C"
    recording = f"Startdate {dt.day:02d}-{months[dt.month-1]}-{dt.year} X X X"
    n = len(plan.channels)
    header = b"".join((_field(0, 8), _field("X X X X", 80), _field(recording, 80),
        _field(dt.strftime("%d.%m.%y"), 8), _field(dt.strftime("%H.%M.%S"), 8),
        _field(256*(n+2), 8), _field(fmt, 44), _field(plan.record_count, 8),
        _field(plan.record_duration_text, 8), _field(n+1, 4)))
    fields = ((16, [label for _, _, label in plan.channel_labels]+["EDF Annotations"]), (80, [""]*(n+1)),
        (8, ["" if c >= 256 else "uV" for c in plan.channels]+[""]),
        (8, [c.pmin for c in calibrations]+["-1"]), (8, [c.pmax for c in calibrations]+["1"]),
        (8, [c.dmin for c in calibrations]+[-32768]), (8, [c.dmax for c in calibrations]+[32767]),
        (80, [""]*(n+1)), (8, [plan.record_samples]*n+[plan.annotation_bytes//2]), (32, [""]*(n+1)))
    return header+b"".join(_field(v, width) for width, values in fields for v in values), fmt


def write_export(reader, path, *, max_error_uv, progress, **options):
    began = time.perf_counter()
    if isinstance(max_error_uv, bool) or not isinstance(max_error_uv, (int, float)) or not isfinite(max_error_uv) or max_error_uv <= 0:
        raise ValueError("max_error_uv must be finite and positive")
    destination = Path(path).expanduser()
    parent = destination.parent.resolve(strict=True)
    destination = parent/destination.name
    if destination.suffix.casefold() != ".edf":
        raise ValueError("EDF output must have the .edf extension")
    if os.path.lexists(destination):
        raise FileExistsError(f"Export destination already exists: {destination}")
    _check_publication(parent)
    first = _integer(options["start"], "start")
    last = reader.info.n_samples if options["stop"] is None else _integer(options["stop"], "stop")
    if not 0 <= first < last <= reader.info.n_samples:
        raise ValueError("Invalid export window")
    paths = {reader._stc_path, reader._find_header_erd()}
    for suffix in (".snc", ".eeg", ".ent", ".ent.old"):
        member = reader._files.lookup(reader._stc_path.stem+suffix, optional=True)
        if member:
            paths.add(member)
    for segment in reader._overlapping_segments(first+reader.info.start_stamp, last+reader.info.start_stamp):
        paths.update(reader._segment_paths(segment))
    def signatures():
        return {p: _stat_signature(p.stat()) for p in paths}
    original = signatures()
    def check_source():
        if signatures() != original:
            raise DataIntegrityError("Source recording changed during export; refusing publication")
    plan = plan_edf(reader, **options, progress=progress)
    if shutil.disk_usage(parent).free < plan.output_bytes:
        raise ResourceLimitError("Insufficient free space for the bounded EDF output")
    scan_start = time.perf_counter()
    stats, last_notice, completed = None, scan_start, 0
    for _, width, values, _, _ in ordered_work(reader, plan):
        if values is not None:
            stats = combine_stats(stats, values)
        completed += width
        if progress and time.perf_counter()-last_notice >= 1:
            progress({"stage": "range_scan", "samples": completed, "total": plan.record_count*plan.record_samples})
            last_notice = time.perf_counter()
    if stats is None:
        raise DataIntegrityError("No source values were decoded")
    calibrations = tuple(calibrate(reader.channels[c], v, max_error_uv) for c, v in zip(plan.channels, stats))
    scan_seconds = time.perf_counter()-scan_start
    check_source()
    header, fmt = _header(plan, calibrations)
    temporary = parent/(destination.name+".partial-"+uuid.uuid4().hex)
    measured, write_start = 0.0, time.perf_counter()
    origin = plan._origin
    try:
        import numpy as np
        with temporary.open("xb") as output:
            output.write(header)
            record_index = 0
            for sample, width, codes, errors in record_blocks(ordered_work(reader, plan, calibrations), len(plan.channels), plan.record_samples):
                measured = max(measured, max(errors, default=0.0))
                if measured > max_error_uv+1e-9:
                    raise DataIntegrityError("Measured EDF quantization error exceeds its budget")
                count = width//plan.record_samples
                wave_bytes = len(plan.channels)*plan.record_samples*2
                record_bytes = wave_bytes+plan.annotation_bytes
                wave = np.frombuffer(codes, dtype="u1").reshape(len(plan.channels), count, plan.record_samples*2).transpose(1, 0, 2)
                payload = np.zeros((count, record_bytes), dtype="u1")
                payload[:, :wave_bytes] = wave.reshape(count, wave_bytes)
                for offset in range(count):
                    index = record_index+offset
                    onset = origin+Fraction(sample+offset*plan.record_samples)/plan.sample_rate
                    tal = _tal(onset)+plan._events.get(index, b"")
                    if len(tal) > plan.annotation_bytes:
                        raise DataIntegrityError("EDF annotation layout changed after preflight")
                    payload[offset, wave_bytes:wave_bytes+len(tal)] = np.frombuffer(tal, dtype="u1")
                output.write(payload.data)
                record_index += count
                if progress and time.perf_counter()-last_notice >= 1:
                    progress({"stage": "write", "records": record_index, "total": plan.record_count,
                              "bytes": output.tell(), "max_error_uv": measured})
                    last_notice = time.perf_counter()
                del wave, payload, codes
            if record_index != plan.record_count or output.tell() != plan.output_bytes:
                raise DataIntegrityError("EDF file length differs from its finalized plan")
            output.flush()
            os.fsync(output.fileno())
        check_source()
        from ._edf_verify import verify_export
        verify_export(reader, temporary, plan, calibrations, max_error_uv)
        check_source()
        if progress:
            progress({"stage": "publishing", "file_bytes": plan.output_bytes})
            # A callback is application code and may change source files. Its
            # return is still before the no-clobber publication boundary.
            check_source()
        _publish(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return EdfExportResult(
        record_count=plan.record_count, logical_samples=plan.logical_samples,
        file_bytes=plan.output_bytes,
        max_quantization_error_uv=max(c.error_bound for c in calibrations),
        measured_max_error_uv=measured, event_count=plan.event_count,
        channel_count=len(plan.channels), shorted_channels=plan.shorted_channels,
        channel_labels=plan.channel_labels,
        uncalibrated_channels=sum(c >= 256 for c in plan.channels),
        backend=plan.backend, workers=plan.workers, chunk_samples=plan.chunk_samples,
        elapsed_seconds=time.perf_counter()-began, scan_seconds=scan_seconds,
        write_seconds=time.perf_counter()-write_start,
    )


def _publish(temporary: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(temporary, destination)
    else:
        os.link(temporary, destination)


def _check_publication(parent: Path) -> None:
    """Probe POSIX hard-link support before decoding the recording."""
    if os.name == "nt":
        return
    probe = parent/(".edf-publish-"+uuid.uuid4().hex)
    linked = probe.with_suffix(".link")
    try:
        with probe.open("xb"):
            pass
        os.link(probe, linked)
    finally:
        for path in (linked, probe):
            if path.exists():
                path.unlink()
