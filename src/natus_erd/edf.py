"""Small random-access EDF reader used by the local comparison viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .errors import DataIntegrityError, UnsupportedFormatError


@dataclass(frozen=True, slots=True)
class EDFSignal:
    index: int
    raw_index: int
    label: str
    unit: str
    sample_rate: float
    samples_per_record: int
    physical_min: float
    physical_max: float
    digital_min: int
    digital_max: int


@dataclass(frozen=True, slots=True)
class EDFInfo:
    n_records: int
    record_duration: float
    header_bytes: int
    record_bytes: int
    n_raw_signals: int
    n_data_signals: int


class EDFReader:
    """Read ordinary 16-bit EDF signals without scanning annotations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._parse_header()

    @property
    def info(self) -> EDFInfo:
        return self._info

    @property
    def signals(self) -> tuple[EDFSignal, ...]:
        return self._signals

    @property
    def n_samples(self) -> int:
        if not self._signals:
            return 0
        return self._info.n_records * self._signals[0].samples_per_record

    @property
    def sample_rate(self) -> float:
        if not self._signals:
            raise DataIntegrityError("EDF contains no data signals")
        return self._signals[0].sample_rate

    def read_digital(
        self, start: int, stop: int, channels: tuple[int, ...] | list[int]
    ) -> NDArray[np.int32]:
        """Read EDF digital values as ``(channels, samples)``."""

        if not isinstance(start, int) or not isinstance(stop, int):
            raise TypeError("EDF start and stop must be integers")
        selected = tuple(int(channel) for channel in channels)
        if not 0 <= start <= stop <= self.n_samples:
            raise IndexError(
                f"EDF range must satisfy 0 <= start <= stop <= {self.n_samples}"
            )
        for channel in selected:
            if not 0 <= channel < len(self._signals):
                raise IndexError(f"EDF channel index is out of range: {channel}")
        output = np.empty((len(selected), stop - start), dtype=np.int32)
        if start == stop or not selected:
            return output

        samples_per_record = self._signals[selected[0]].samples_per_record
        for channel in selected:
            signal = self._signals[channel]
            if signal.samples_per_record != samples_per_record:
                raise UnsupportedFormatError(
                    "Selected EDF channels use different sampling frequencies"
                )

        first_record = start // samples_per_record
        last_record = (stop - 1) // samples_per_record
        record_count = last_record - first_record + 1
        byte_count = record_count * self._info.record_bytes
        byte_offset = self._info.header_bytes + first_record * self._info.record_bytes
        try:
            with self.path.open("rb") as stream:
                stream.seek(byte_offset)
                payload = stream.read(byte_count)
        except OSError as exc:
            raise DataIntegrityError(f"Cannot read EDF data: {exc}") from exc
        if len(payload) != byte_count:
            raise DataIntegrityError("EDF data window is truncated")

        for row, channel in enumerate(selected):
            signal = self._signals[channel]
            raw_index = signal.raw_index
            channel_byte_offset = self._raw_channel_offsets[raw_index] * 2
            destination = 0
            for record in range(first_record, last_record + 1):
                record_start_sample = record * samples_per_record
                take_start = max(start, record_start_sample)
                take_stop = min(stop, record_start_sample + samples_per_record)
                source_start = take_start - record_start_sample
                source_stop = take_stop - record_start_sample
                local_record = record - first_record
                source_offset = (
                    local_record * self._info.record_bytes + channel_byte_offset
                )
                values = np.frombuffer(
                    payload,
                    dtype="<i2",
                    count=samples_per_record,
                    offset=source_offset,
                )
                count = source_stop - source_start
                output[row, destination : destination + count] = values[
                    source_start:source_stop
                ]
                destination += count
        return output

    def digital_to_physical(
        self, digital: NDArray[np.integer], channels: tuple[int, ...] | list[int]
    ) -> NDArray[np.float64]:
        selected = tuple(int(channel) for channel in channels)
        if digital.ndim != 2 or digital.shape[0] != len(selected):
            raise ValueError("digital array shape does not match selected channels")
        physical = digital.astype(np.float64, copy=True)
        for row, channel in enumerate(selected):
            signal = self._signals[channel]
            denominator = signal.digital_max - signal.digital_min
            if denominator == 0:
                raise DataIntegrityError(
                    f"EDF channel {channel} has a zero digital range"
                )
            slope = (signal.physical_max - signal.physical_min) / denominator
            physical[row] = (
                (physical[row] - signal.digital_min) * slope + signal.physical_min
            )
        return physical

    def read_physical(
        self, start: int, stop: int, channels: tuple[int, ...] | list[int]
    ) -> NDArray[np.float64]:
        digital = self.read_digital(start, stop, channels)
        return self.digital_to_physical(digital, channels)

    def _parse_header(self) -> None:
        try:
            with self.path.open("rb") as stream:
                fixed = stream.read(256)
                if len(fixed) != 256:
                    raise DataIntegrityError("EDF fixed header is truncated")
                if fixed[:1] == b"\xff":
                    raise UnsupportedFormatError("BDF is not supported by this viewer")
                if fixed[192:236].strip().startswith(b"EDF+D"):
                    raise UnsupportedFormatError(
                        "Discontinuous EDF+D requires time-annotation alignment"
                    )
                header_bytes = _ascii_int(fixed[184:192], "header byte count")
                declared_records = _ascii_int(fixed[236:244], "record count")
                record_duration = _ascii_float(fixed[244:252], "record duration")
                n_raw_signals = _ascii_int(fixed[252:256], "signal count")
                if n_raw_signals <= 0 or record_duration <= 0:
                    raise DataIntegrityError("EDF declares invalid signal metadata")
                expected_header = 256 + n_raw_signals * 256
                if header_bytes != expected_header:
                    raise DataIntegrityError(
                        f"EDF header size is {header_bytes}, expected {expected_header}"
                    )
                signal_header = stream.read(header_bytes - 256)
        except OSError as exc:
            raise DataIntegrityError(f"Cannot read EDF header: {exc}") from exc
        if len(signal_header) != header_bytes - 256:
            raise DataIntegrityError("EDF signal header is truncated")

        offset = 0

        def text_fields(width: int) -> tuple[str, ...]:
            nonlocal offset
            fields = tuple(
                signal_header[offset + index * width : offset + (index + 1) * width]
                .decode("latin-1")
                .strip()
                for index in range(n_raw_signals)
            )
            offset += width * n_raw_signals
            return fields

        labels = text_fields(16)
        text_fields(80)  # transducer
        units = text_fields(8)
        physical_min = tuple(
            _field_float(value, "physical minimum") for value in text_fields(8)
        )
        physical_max = tuple(
            _field_float(value, "physical maximum") for value in text_fields(8)
        )
        digital_min = tuple(
            _field_int(value, "digital minimum") for value in text_fields(8)
        )
        digital_max = tuple(
            _field_int(value, "digital maximum") for value in text_fields(8)
        )
        text_fields(80)  # prefiltering
        samples_per_record = tuple(
            _field_int(value, "samples per record") for value in text_fields(8)
        )
        text_fields(32)  # reserved

        if any(value <= 0 for value in samples_per_record):
            raise DataIntegrityError("EDF contains a non-positive samples-per-record value")
        raw_offsets: list[int] = []
        running_offset = 0
        for count in samples_per_record:
            raw_offsets.append(running_offset)
            running_offset += count
        record_bytes = running_offset * 2
        file_size = self.path.stat().st_size
        available_data_bytes = file_size - header_bytes
        if available_data_bytes < 0 or available_data_bytes % record_bytes:
            raise DataIntegrityError("EDF file size is not an integral number of records")
        available_records = available_data_bytes // record_bytes
        if declared_records < 0:
            n_records = available_records
        else:
            n_records = declared_records
            if n_records > available_records:
                raise DataIntegrityError("EDF declares more records than the file contains")

        data_raw_indices = tuple(
            index
            for index, label in enumerate(labels)
            if label.casefold() != "edf annotations"
        )
        signals: list[EDFSignal] = []
        for index, raw_index in enumerate(data_raw_indices):
            signals.append(
                EDFSignal(
                    index=index,
                    raw_index=raw_index,
                    label=labels[raw_index] or f"signal{index:03d}",
                    unit=units[raw_index],
                    sample_rate=samples_per_record[raw_index] / record_duration,
                    samples_per_record=samples_per_record[raw_index],
                    physical_min=physical_min[raw_index],
                    physical_max=physical_max[raw_index],
                    digital_min=digital_min[raw_index],
                    digital_max=digital_max[raw_index],
                )
            )
        self._signals = tuple(signals)
        self._raw_channel_offsets = tuple(raw_offsets)
        self._info = EDFInfo(
            n_records=n_records,
            record_duration=record_duration,
            header_bytes=header_bytes,
            record_bytes=record_bytes,
            n_raw_signals=n_raw_signals,
            n_data_signals=len(signals),
        )


def _ascii_int(value: bytes, field: str) -> int:
    try:
        return int(value.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError) as exc:
        raise DataIntegrityError(f"EDF {field} is invalid") from exc


def _ascii_float(value: bytes, field: str) -> float:
    try:
        return float(value.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError) as exc:
        raise DataIntegrityError(f"EDF {field} is invalid") from exc


def _field_int(value: str, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise DataIntegrityError(f"EDF {field} is invalid") from exc


def _field_float(value: str, field: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise DataIntegrityError(f"EDF {field} is invalid") from exc
