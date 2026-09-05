from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from struct import pack, pack_into

N_CHANNELS = 276
SHORTED = {249, 251, 253, 255}
HEADER_SIZE = 8656


@dataclass(frozen=True)
class SyntheticRecording:
    directory: Path
    stc: Path
    eeg: Path
    first_erd: Path
    expected: tuple[tuple[int | None, ...], ...]


def build_recording(
    root: Path, *, sample_rate: float = 2048.0,
    shorted: set[int] | frozenset[int] = frozenset(SHORTED),
) -> SyntheticRecording:
    directory = root / "中文记录"
    directory.mkdir()
    stem = "记录"
    segment_names = (stem, f"{stem}_001")

    stored = _stored_values()
    packet_groups = ((stored[:3], stored[3:5]), (stored[5:9],))
    packet_stamps = ((1000, 1003), (1006,))

    for segment_index, segment_name in enumerate(segment_names):
        packets = [_encode_packet(samples, shorted=shorted) for samples in packet_groups[segment_index]]
        erd = bytearray(_erd_header(sample_rate=sample_rate, shorted=shorted))
        offsets: list[int] = []
        for packet in packets:
            offsets.append(len(erd))
            erd.extend(packet)
        (directory / f"{segment_name}.erd").write_bytes(erd)

        etc = bytearray(_generic_header(3))
        sample_number = 0
        for offset, stamp, samples in zip(
            offsets, packet_stamps[segment_index], packet_groups[segment_index]
        ):
            etc.extend(pack("<iiihh", offset, stamp, sample_number, len(samples), 0))
            sample_number += len(samples)
        (directory / f"{segment_name}.etc").write_bytes(etc)

    stc = directory / f"{stem}.stc"
    stc_data = bytearray(_generic_header(1))
    stc_data.extend(pack("<ii12i", 1, 1, *([0] * 12)))
    stc_data.extend(_stc_entry(segment_names[0], 1000, 1004, 0))
    stc_data.extend(_stc_entry(segment_names[1], 1005, 1009, 5, stored_samples=4))
    stc.write_bytes(stc_data)

    eeg = directory / f"{stem}.eeg"
    eeg.write_bytes(_generic_header(1))
    (directory / f"{stem}.ent").write_bytes(_ent_file())

    expected_by_channel: list[tuple[int | None, ...]] = []
    for channel in range(N_CHANNELS):
        expected_by_channel.append(
            tuple(
                [sample[channel] for sample in stored[:5]]
                + [None]
                + [sample[channel] for sample in stored[5:9]]
            )
        )
    return SyntheticRecording(
        directory=directory,
        stc=stc,
        eeg=eeg,
        first_erd=directory / f"{stem}.erd",
        expected=tuple(expected_by_channel),
    )


def _generic_header(schema: int) -> bytes:
    header = bytearray(352)
    pack_into("<HH", header, 16, schema, 1)
    pack_into("<i", header, 20, 1_700_000_000)
    return bytes(header)


def _erd_header(
    *, sample_rate: float = 2048.0, shorted: set[int] | frozenset[int] = frozenset(SHORTED)
) -> bytes:
    header = bytearray(HEADER_SIZE)
    header[:352] = _generic_header(9)
    pack_into("<dii", header, 352, sample_rate, N_CHANNELS, 8)
    pack_into(f"<{N_CHANNELS}i", header, 368, *range(N_CHANNELS))
    pack_into("<4i", header, 4464, 20, 0, 0, 0)
    pack_into("<i", header, 4556, 6)
    for channel in shorted:
        pack_into("<h", header, 4560 + channel * 2, 1)
    for channel in range(N_CHANNELS):
        pack_into("<h", header, 6608 + channel * 2, 32767)
    return bytes(header)


def _stc_entry(
    name: str, start: int, end: int, sample_number: int,
    *, stored_samples: int | None = None,
) -> bytes:
    encoded = name.encode("utf-8")
    if len(encoded) >= 256:
        raise AssertionError("synthetic segment name is too long")
    name_field = encoded + bytes(256 - len(encoded))
    count = end - start + 1 if stored_samples is None else stored_samples
    return name_field + pack("<4i", start, end, sample_number, count)


def _stored_values() -> list[list[int]]:
    sample0 = [1000 + channel * 3 for channel in range(N_CHANNELS)]
    sample1 = [value + 1 for value in sample0]
    sample2 = [value + 2 for value in sample1]
    sample2[0] = sample1[0] + 300
    sample2[1] = sample1[1] - 300
    sample2[183] = 131071

    sample3 = [-2000 + channel for channel in range(N_CHANNELS)]
    sample4 = [value - 1 for value in sample3]
    sample4[0] = 777777

    sample5 = [50000 - channel * 2 for channel in range(N_CHANNELS)]
    sample6 = [value + 5 for value in sample5]
    sample7 = [value - 7 for value in sample6]
    sample8 = [value + 1 for value in sample7]
    return [
        sample0,
        sample1,
        sample2,
        sample3,
        sample4,
        sample5,
        sample6,
        sample7,
        sample8,
    ]


def _encode_packet(
    samples: tuple[list[int], ...] | list[list[int]],
    *, shorted: set[int] | frozenset[int] = frozenset(SHORTED),
) -> bytes:
    active = [channel for channel in range(N_CHANNELS) if channel not in shorted]
    mask_size = (N_CHANNELS + 7) // 8
    encoded = bytearray()
    previous: list[int] | None = None

    for sample_index, values in enumerate(samples):
        mask = bytearray(mask_size)
        delta_bytes = bytearray()
        absolutes: list[int] = []
        for channel in active:
            if previous is None:
                wide = True
                absolute = True
                delta = -1
            else:
                delta = values[channel] - previous[channel]
                wide = not -128 <= delta <= 127
                absolute = not -32768 <= delta <= 32767
                if absolute:
                    wide = True
                    delta = -1

            if wide:
                mask[channel >> 3] |= 1 << (channel & 7)
                delta_bytes.extend(pack("<h", delta))
            else:
                delta_bytes.extend(pack("<b", delta))
            if absolute:
                absolutes.append(values[channel])

        encoded.append(1 if sample_index == 2 else 0)
        encoded.extend(mask)
        encoded.extend(delta_bytes)
        for value in absolutes:
            encoded.extend(pack("<i", value))
        previous = values
    return bytes(encoded)


def _ent_file() -> bytes:
    names = ", ".join(f'"CH{channel:03d}"' for channel in range(N_CHANNELS))
    montage = f'(.(."ChanNames", ({names})), (."Text", "Montage"))'
    event = (
        '(.(."Stamp", 1001), (."Text", "marker"), '
        '(."Data", (.(."User", "tester"))))'
    )
    output = bytearray(_generic_header(3))
    previous_length = 0
    for note_type, text in ((2, montage), (1, event)):
        payload = text.encode("utf-8") + b"\0\0"
        length = 16 + len(payload)
        output.extend(pack("<4i", note_type, length, previous_length, 0))
        output.extend(payload)
        previous_length = length
    output.extend(bytes(16))
    return bytes(output)


from datetime import datetime, timedelta, timezone
from fractions import Fraction

SNC_GUID = bytes.fromhex("d2a98660af60d311986000104b75c151")
BASE_DATETIME = datetime(2000, 1, 1, 12, 0, 0, 125000, tzinfo=timezone.utc)
BASE_TICKS = ((BASE_DATETIME - datetime(1601, 1, 1, tzinfo=timezone.utc))
              // timedelta(microseconds=1)) * 10


def _snc(path, rate, end=1009):
    header = bytearray(352)
    header[:16] = SNC_GUID
    pack_into("<HH", header, 16, 1, 1)
    last_ticks = BASE_TICKS + round(Fraction(end - 1000) / Fraction(str(rate)) * 10_000_000)
    path.write_bytes(header + pack("<iQ", 1000, BASE_TICKS) + pack("<iQ", end, last_ticks))


def _annotation_layout(payload):
    signals = int(payload[252:256])
    header_bytes = int(payload[184:192])
    samples_start = 256 + signals * (16 + 80 + 8 * 5 + 80)
    counts = [int(payload[samples_start + 8 * i:samples_start + 8 * (i + 1)])
              for i in range(signals)]
    record_bytes = sum(counts) * 2
    return header_bytes + sum(counts[:-1]) * 2, counts[-1] * 2, record_bytes


def build_continuous_recording(root: Path, *, sample_rate: float = 512.0,
                               samples: int = 1024, packet_samples: int = 63,
                               shorted: set[int] | frozenset[int] = frozenset(SHORTED)) -> SyntheticRecording:
    """Continuous, multi-packet signal with calibrated and exact auxiliary rows."""
    fixture = build_recording(root, sample_rate=sample_rate, shorted=shorted)
    values = [[1500+channel+((sample*7)%257)*2 for channel in range(N_CHANNELS)]
              for sample in range(samples)]
    for row in values:
        row[273] = row[274] = 131070
        row[275] = 0
    erd = bytearray(_erd_header(sample_rate=sample_rate, shorted=shorted))
    etc = bytearray(_generic_header(3))
    for start in range(0, samples, packet_samples):
        block = values[start:start+packet_samples]
        etc.extend(pack("<iiihh", len(erd), 1000+start, start, len(block), 0))
        erd.extend(_encode_packet(block, shorted=shorted))
    fixture.first_erd.write_bytes(erd)
    fixture.first_erd.with_suffix('.etc').write_bytes(etc)
    fixture.stc.write_bytes(_generic_header(1)+pack("<ii12i", 1, 1, *([0]*12))
                            +_stc_entry(fixture.stc.stem, 1000, 999+samples, 0))
    _snc(fixture.stc.with_suffix('.snc'), sample_rate, 999+samples)
    return SyntheticRecording(fixture.directory, fixture.stc, fixture.eeg, fixture.first_erd,
                               tuple(tuple(row) for row in zip(*values)))


def build_discontinuous_recording(root: Path, *, spans=((0, 32), (544, 576)),
                                  sample_rate=512.0, packet_samples=7):
    """Synthetic stored spans; waveform values depend on original sample positions."""
    fixture = build_continuous_recording(root, sample_rate=sample_rate, samples=spans[-1][1])
    erd = bytearray(_erd_header(sample_rate=sample_rate))
    etc = bytearray(_generic_header(3))
    stored = 0
    for a, b in spans:
        for start in range(a, b, packet_samples):
            stop = min(b, start+packet_samples)
            block = [list(row) for row in zip(*(c[start:stop] for c in fixture.expected))]
            etc.extend(pack('<iiihh',len(erd),1000+start,stored,len(block),0))
            erd.extend(_encode_packet(block))
            stored += len(block)
    fixture.first_erd.write_bytes(erd)
    fixture.first_erd.with_suffix('.etc').write_bytes(etc)
    fixture.stc.write_bytes(_generic_header(1)+pack('<ii12i',1,1,*([0]*12))
        +_stc_entry(fixture.stc.stem,1000,999+spans[-1][1],0,stored_samples=stored))
    return fixture
