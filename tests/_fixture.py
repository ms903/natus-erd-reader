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
    edf: Path
    expected: tuple[tuple[int | None, ...], ...]


def build_recording(root: Path) -> SyntheticRecording:
    directory = root / "中文记录"
    directory.mkdir()
    stem = "记录"
    segment_names = (stem, f"{stem}_001")

    stored = _stored_values()
    packet_groups = ((stored[:3], stored[3:5]), (stored[5:9],))
    packet_stamps = ((1000, 1003), (1006,))

    for segment_index, segment_name in enumerate(segment_names):
        packets = [_encode_packet(samples) for samples in packet_groups[segment_index]]
        erd = bytearray(_erd_header())
        offsets: list[int] = []
        for packet in packets:
            offsets.append(len(erd))
            erd.extend(packet)
        (directory / f"{segment_name}.erd").write_bytes(erd)

        etc = bytearray(_generic_header(1))
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
    stc_data.extend(_stc_entry(segment_names[1], 1005, 1009, 5))
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
    edf = root / "export.edf"
    edf.write_bytes(_edf_file(tuple(expected_by_channel)))
    return SyntheticRecording(
        directory=directory,
        stc=stc,
        eeg=eeg,
        first_erd=directory / f"{stem}.erd",
        edf=edf,
        expected=tuple(expected_by_channel),
    )


def _generic_header(schema: int) -> bytes:
    header = bytearray(352)
    pack_into("<HH", header, 16, schema, 1)
    pack_into("<i", header, 20, 1_700_000_000)
    return bytes(header)


def _erd_header() -> bytes:
    header = bytearray(HEADER_SIZE)
    header[:352] = _generic_header(9)
    pack_into("<dii", header, 352, 2048.0, N_CHANNELS, 8)
    pack_into(f"<{N_CHANNELS}i", header, 368, *range(N_CHANNELS))
    pack_into("<4i", header, 4464, 20, 0, 0, 0)
    pack_into("<i", header, 4556, 6)
    for channel in SHORTED:
        pack_into("<h", header, 4560 + channel * 2, 1)
    for channel in range(N_CHANNELS):
        pack_into("<h", header, 6608 + channel * 2, 32767)
    return bytes(header)


def _stc_entry(name: str, start: int, end: int, sample_number: int) -> bytes:
    encoded = name.encode("utf-8")
    if len(encoded) >= 256:
        raise AssertionError("synthetic segment name is too long")
    name_field = encoded + bytes(256 - len(encoded))
    return name_field + pack("<4i", start, end, sample_number, end - start + 1)


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


def _encode_packet(samples: tuple[list[int], ...] | list[list[int]]) -> bytes:
    active = [channel for channel in range(N_CHANNELS) if channel not in SHORTED]
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
    output = bytearray(_generic_header(1))
    previous_length = 0
    for note_type, text in ((2, montage), (1, event)):
        payload = text.encode("utf-8") + b"\0\0"
        length = 16 + len(payload)
        output.extend(pack("<4i", note_type, length, previous_length, 0))
        output.extend(payload)
        previous_length = length
    output.extend(bytes(16))
    return bytes(output)


def _edf_file(expected: tuple[tuple[int | None, ...], ...]) -> bytes:
    n_raw_signals = N_CHANNELS + 1
    header_bytes = 256 + n_raw_signals * 256

    def field(value: object, width: int) -> bytes:
        encoded = str(value).encode("ascii")
        if len(encoded) > width:
            raise AssertionError(f"EDF field is wider than {width}: {value}")
        return encoded.ljust(width, b" ")

    fixed = bytearray()
    fixed.extend(field("0", 8))
    fixed.extend(field("synthetic", 80))
    fixed.extend(field("synthetic", 80))
    fixed.extend(field("01.01.24", 8))
    fixed.extend(field("00.00.00", 8))
    fixed.extend(field(header_bytes, 8))
    fixed.extend(field("EDF+C", 44))
    fixed.extend(field(1, 8))
    fixed.extend(field("0.03125", 8))
    fixed.extend(field(n_raw_signals, 4))
    if len(fixed) != 256:
        raise AssertionError("invalid synthetic EDF fixed header")

    labels = [f"CH{channel:03d}" for channel in range(N_CHANNELS)] + [
        "EDF Annotations"
    ]
    signal_header = bytearray()
    for values, width in (
        (labels, 16),
        ([""] * n_raw_signals, 80),
        (["uV"] * N_CHANNELS + [""], 8),
        (["8711"] * N_CHANNELS + ["-1"], 8),
        (["-8711"] * N_CHANNELS + ["1"], 8),
        (["-32768"] * n_raw_signals, 8),
        (["32767"] * n_raw_signals, 8),
        ([""] * n_raw_signals, 80),
        (["64"] * n_raw_signals, 8),
        ([""] * n_raw_signals, 32),
    ):
        for value in values:
            signal_header.extend(field(value, width))
    if len(signal_header) != n_raw_signals * 256:
        raise AssertionError("invalid synthetic EDF signal header")

    record = bytearray()
    for channel in range(N_CHANNELS):
        values = []
        for value in expected[channel]:
            if value is None:
                values.append(0)
            else:
                values.append(max(-32768, min(32767, value)))
        values.extend([0] * (64 - len(values)))
        record.extend(pack("<64h", *values))
    record.extend(bytes(64 * 2))
    return bytes(fixed + signal_header + record)
