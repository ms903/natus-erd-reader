"""Private bounded packet jobs. Workers never write the destination file."""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import lcm
import os
from typing import Any

from .errors import DataIntegrityError, ResourceLimitError, UnsupportedFormatError
from ._parameters import integer
from .reader import _stat_signature

MiB = 1024**2


def native_available():
    try:
        from . import _native
    except ImportError:
        return False
    return callable(_native.process)


@dataclass(frozen=True, slots=True)
class Execution:
    backend: str
    workers: int
    chunk_samples: int
    memory_budget_bytes: int
    reserved_bytes: int


def execution(reader, channels, points, annotation_bytes, output_bytes, backend, workers, budget, chunk_samples):
    if backend not in ("auto", "native", "python"):
        raise ValueError("backend must be auto, native, or python")
    budget = integer(budget, "memory_budget_bytes")
    if budget < 1:
        raise ValueError("memory_budget_bytes must be positive")
    available = native_available() if backend != "python" else False
    if backend == "native" and not available:
        raise UnsupportedFormatError("Native export backend is not installed")
    backend = "native" if available else "python"
    automatic = workers == "auto"
    if automatic:
        count = min(4, os.cpu_count() or 1) if available else 1
    else:
        count = integer(workers, "workers")
        if not 1 <= count <= 32:
            raise ValueError("workers must be auto or an integer from 1 through 32")
    if backend == "python" and count != 1:
        raise ValueError("The reference Python export backend requires one worker")
    rows = len(channels)
    if chunk_samples is not None:
        chunk_samples = integer(chunk_samples, "chunk_samples")
        if chunk_samples < points:
            raise ValueError("chunk_samples must be at least one EDF record")
    record_bytes = rows*points*2+annotation_bytes
    target = chunk_samples or max(points, (12*MiB//record_bytes)*points)
    # Known native packet lengths are inspected, never assumed to be 1000.
    spans: list[int] = []
    for segment in reader._segments:
        entries = reader._load_etc(segment).entries
        spans.extend(entry.sample_span for entry in entries[:16])
        if len(spans) >= 16:
            break
    typical = Counter(spans).most_common(1)[0][0] if spans else points
    alignment = lcm(points, typical)
    if alignment <= target:
        aligned = ((target+alignment-1)//alignment)*alignment
        target = aligned if chunk_samples is None and aligned//points*record_bytes <= 16*MiB else target//alignment*alignment
    else:
        target = max(points, target//points*points)
    # Header, source/event metadata and one maximum-sized decoder packet.
    reserve = 256*(rows+2)+reader.limits.max_metadata_bytes
    while True:
        payload = target*rows*2
        scratch = (min(target, 32767)*rows*8+min(target, 32767)*32+65536
                   if backend == "python" else 0)
        # A job reads only one bounded compressed packet at a time. Writer
        # retains at most two extra payload copies while interleaving records.
        encoded_payload = (target//points)*record_bytes
        # Boundary carry may need a joined block, a contiguous view and record
        # interleaving at once; each is explicitly charged, including TALs.
        required = reserve+(count+3)*payload+encoded_payload+2*record_bytes+count*(reader.limits.max_packet_bytes+scratch)
        # Verification retains one input block per worker, one boolean code
        # comparison and an expected TAL block per job. Independent waveform
        # windows hold at most 16 samples per row plus two complete records.
        verify_required = reserve+count*(2*encoded_payload+payload//2)+rows*16*32+4*record_bytes+reader.limits.max_packet_bytes
        required = max(required, verify_required)
        if required <= budget:
            return Execution(backend, count, target, budget, required)
        if automatic and count > 1:
            count = max(1, count//2)
        elif chunk_samples is None and target > points:
            target = max(points, (target//2)//points*points)
        else:
            raise ResourceLimitError("Explicit export configuration exceeds the aggregate buffer budget")


@dataclass(frozen=True, slots=True)
class Packet:
    path: object
    signature: tuple
    offset: int
    byte_end: int
    count: int
    start: int
    stop: int
    column: int


@dataclass(frozen=True, slots=True)
class Job:
    sample: int
    width: int
    packets: tuple[Packet, ...]


def jobs(reader, plan):
    for low, high in ((0, plan.logical_samples),):
        sample = low
        while sample < high:
            width = min(plan.chunk_samples, high-sample)
            first = plan.start+sample+reader.info.start_stamp
            last = min(plan.stop, plan.start+sample+width)+reader.info.start_stamp
            packets = []
            for segment in reader._overlapping_segments(first, last):
                index = reader._load_etc(segment)
                path, _ = reader._segment_paths(segment)
                from bisect import bisect_right
                begin = bisect_right(index.packet_ends, first)
                for at in range(begin, len(index.entries)):
                    entry = index.entries[at]
                    if entry.sample_stamp >= last:
                        break
                    a, b = max(first, entry.sample_stamp), min(last, entry.end_stamp_exclusive)
                    if a >= b:
                        continue
                    end = index.entries[at+1].offset if at+1 < len(index.entries) else index.erd_signature[0]
                    packets.append(Packet(path, index.erd_signature, entry.offset, end,
                                          entry.sample_span, a-entry.sample_stamp, b-entry.sample_stamp, a-first))
            # Prefer a COMPLETE packet boundary. A bounded writer carry joins
            # its partial EDF record to the next job, avoiding repeated packet
            # decompression when the two grids have different origins.
            if sample+width < high:
                boundaries = [p.column+p.stop-p.start for p in packets if p.stop == p.count]
                if boundaries and max(boundaries) >= width//2:
                    width = max(boundaries)
                    packets = [p for p in packets if p.column < width]
            yield Job(sample, width, tuple(packets))
            sample += width


def combine_stats(left, right):
    if left is None:
        return right
    return tuple((min(a, c), max(b, d)) for (a, b), (c, d) in zip(left, right))


def work(job, *, reader, selected, backend, calibrations=None):
    rows = len(selected)
    scanning = calibrations is None
    output = None if scanning else bytearray(rows*job.width*2)
    stats, errors = None, [0.0]*rows
    shorted = bytes(reader._erd_header.shorted)
    stream, current = None, None
    native_params = () if scanning else tuple(c.native for c in calibrations)
    try:
        for packet in job.packets:
            if packet.path != current:
                if stream is not None:
                    stream.close()
                stream = packet.path.open("rb")
                current = packet.path
            assert stream is not None
            if _stat_signature(os.fstat(stream.fileno())) != packet.signature:
                raise DataIntegrityError("Source ERD changed during export")
            if backend == "native":
                from . import _native
                maximum = 1+(len(shorted)+7)//8+6*sum(not s for s in shorted)
                length = min(packet.byte_end-packet.offset, packet.stop*maximum)
                stream.seek(packet.offset)
                payload = stream.read(length)
                if len(payload) != length:
                    raise DataIntegrityError("Source ERD packet was truncated")
                try:
                    result = _native.process(payload, shorted, selected, packet.count,
                        packet.start, packet.stop, 0 if scanning else 2, native_params,
                        output, job.width, packet.column)
                except ValueError as exc:
                    raise DataIntegrityError("Native ERD decoding or quantization rejected a packet") from exc
                if scanning:
                    stats = combine_stats(stats, result)
                else:
                    errors = [max(a, b) for a, b in zip(errors, result)]
                del payload
            else:
                from .decoder import decode_schema9_packet
                from dataclasses import replace
                import numpy as np
                # Separate export budgets, not a mutation of reader.limits.
                limits = replace(reader.limits, max_read_samples=max(1, packet.stop-packet.start),
                                 max_read_bytes=max(8, rows*(packet.stop-packet.start)*8))
                values = decode_schema9_packet(stream, offset=packet.offset, byte_end=packet.byte_end,
                    sample_count=packet.count, start=packet.start, stop=packet.stop,
                    n_channels=len(shorted), shorted=reader._erd_header.shorted, selected=selected, limits=limits)
                if scanning:
                    result = []
                    for row, channel in enumerate(selected):
                        if shorted[channel]:
                            result.append((0, 0))
                            continue
                        integers = values[row].astype(np.int64)
                        low, high = int(integers.min()), int(integers.max())
                        result.append((low, high))
                    stats = combine_stats(stats, tuple(result))
                else:
                    assert output is not None
                    destination = np.frombuffer(output, dtype="<i2").reshape(rows, job.width)
                    for row, channel in enumerate(selected):
                        if shorted[channel]:
                            destination[row, packet.column:packet.column+packet.stop-packet.start] = 32767
                            continue
                        cal = calibrations[row]
                        v = values[row]
                        if cal.raw:
                            integers = v.astype(np.int64)-cal.raw_min
                            if np.any(integers < 0) or np.any(integers % cal.raw_step):
                                raise DataIntegrityError("Auxiliary values changed after range scan")
                            codes = integers//cal.raw_step+cal.dmin
                        else:
                            v *= cal.source_scale
                            pmin, pmax = float(cal.pmin), float(cal.pmax)
                            if not np.isfinite(v).all() or np.any(v < min(pmin, pmax)) or np.any(v > max(pmin, pmax)):
                                raise DataIntegrityError("Signal exceeded its pre-scanned physical range")
                            codes = np.rint((v-pmin)*((cal.dmax-cal.dmin)/(pmax-pmin))+cal.dmin)
                            error = np.max(np.abs((codes-cal.dmin)*((pmax-pmin)/(cal.dmax-cal.dmin))+pmin-v))
                            errors[row] = max(errors[row], float(error))
                        if np.any(codes < cal.dmin) or np.any(codes > cal.dmax):
                            raise DataIntegrityError("EDF quantization would clip data")
                        destination[row, packet.column:packet.column+packet.stop-packet.start] = codes
                del values
            if _stat_signature(os.fstat(stream.fileno())) != packet.signature:
                raise DataIntegrityError("Source ERD changed while exporting a packet")
    finally:
        if stream is not None:
            stream.close()
    return job.sample, job.width, stats, output, tuple(errors)


def ordered_work(reader, plan, calibrations=None):
    from functools import partial
    operation = partial(work, reader=reader, selected=plan.channels,
                        backend=plan.backend, calibrations=calibrations)
    if plan.workers == 1:
        for job in jobs(reader, plan):
            yield operation(job)
        return
    pool = ThreadPoolExecutor(max_workers=plan.workers, thread_name_prefix="natus-export")
    pending: deque[Future[Any]] = deque()
    iterator = iter(jobs(reader, plan))
    try:
        for _ in range(plan.workers):
            job = next(iterator, None)
            if job is not None:
                pending.append(pool.submit(operation, job))
        while pending:
            result = pending.popleft().result()
            yield result
            del result
            job = next(iterator, None)
            if job is not None:
                pending.append(pool.submit(operation, job))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def record_blocks(results, rows, points):
    """Join packet-aligned jobs with less than one EDF record of carry."""
    import numpy as np
    carry, expected = None, None
    pending_errors = [0.0]*rows
    for sample, width, _, codes, errors in results:
        pending_errors = [max(a,b) for a,b in zip(pending_errors,errors)]
        values = np.frombuffer(codes,dtype='<i2').reshape(rows,width)
        if carry is not None:
            if sample != expected:
                raise DataIntegrityError("Packet jobs are discontinuous inside an EDF record")
            sample -= carry.shape[1]
            values = np.concatenate((carry,values),axis=1)
        expected = sample+values.shape[1]
        used = values.shape[1]//points*points
        carry = values[:,used:].copy() if used < values.shape[1] else None
        if used:
            block = np.ascontiguousarray(values[:,:used])
            yield sample,used,block,tuple(pending_errors)
            pending_errors = [0.0]*rows
            del block
        del values, codes
    if carry is not None:
        raise DataIntegrityError("An incomplete EDF record remains after the final packet job")
