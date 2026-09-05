"""EDF signal labels, annotations and exact calibration helpers."""
from __future__ import annotations
import math
from dataclasses import dataclass
from .errors import UnsupportedFormatError

def escape_event(text):
    result = "".join("\\\\" if c == "\\" else f"\\u{ord(c):04x}" if ord(c) < 32 or ord(c) == 127 else c for c in text)
    return result


def header_labels(names):
    def legal(name):
        return (bool(name) and name == name.strip() and len(name) <= 16
                and all(32 <= ord(c) < 127 for c in name) and name != "EDF Annotations")
    counts = {name: names.count(name) for name in set(names)}
    reserved = {name for name in names if legal(name)}
    result = []
    for index, name in enumerate(names):
        if legal(name) and counts[name] == 1:
            result.append(name)
            continue
        alias, suffix = f"NATUS{index:04d}", 0
        while alias in reserved:
            suffix += 1
            alias = f"NATUS{index:04d}_{suffix}"
        if len(alias) > 16:
            raise UnsupportedFormatError("Cannot represent unique EDF labels")
        reserved.add(alias)
        result.append(alias)
    return tuple(result)


def outward_field(value, lower):
    number = math.floor(value) if lower else math.ceil(value)
    if len(str(number)) <= 8:
        return str(number)
    for exponent in range(1, 24):
        power = 10 ** exponent
        quotient = number // power if lower else -((-number) // power)
        text = f"{quotient}E{exponent}"
        if len(text) <= 8:
            return text
    raise UnsupportedFormatError("Physical range cannot fit EDF calibration fields")


@dataclass(frozen=True, slots=True)
class Calibration:
    pmin: str
    pmax: str
    dmin: int
    dmax: int
    source_scale: float
    raw_min: int = 0
    raw_step: int = 1
    raw: bool = False
    unit: str = "uV"

    @property
    def native(self):
        return (int(self.raw), self.source_scale, float(self.pmin), float(self.pmax),
                self.dmin, self.dmax, self.raw_min, self.raw_step)

    @property
    def error_bound(self):
        return 0.0 if self.raw else abs(float(self.pmax) - float(self.pmin)) / (self.dmax-self.dmin) / 2


def channel_unit(index: int) -> str:
    return "%" if index == 273 else "bpm" if index == 274 else "uV"


def official_calibration(index: int) -> Calibration:
    """Quantum calibration from six complete official reference exports."""
    if index < 256:
        return Calibration("8711", "-8711", -32768, 32767, 0.0, raw=True)
    if index <= 272:
        return Calibration("5151600", "-5151600", -32768, 32767,
                           1.0, -32768, 1, True)
    if index in (273, 274):
        return Calibration("0", "102.3" if index == 273 else "1023", 0, 32767,
                           1.0, 131070, 1, True, channel_unit(index))
    return Calibration("4.29e+09", "32767", -32768, 32767,
                       1.0, -32768, 1, True)


def calibrate(channel, stats, max_error_uv):
    low, high = stats
    if channel.shorted:
        return official_calibration(channel.index)
    if not channel.is_signal:
        cal = official_calibration(channel.index)
        if channel.index in (273, 274) and (low, high) != (131070, 131070):
            name = "OSAT" if channel.index == 273 else "PR"
            raise UnsupportedFormatError(f"{name} channel index {channel.index}: only missing code 131070 is verified; select other channels")
        if channel.index == 275 and (low, high) != (0, 0):
            raise UnsupportedFormatError("Pleth channel index 275: only source code 0 is verified; select other channels")
        if low < cal.raw_min or high > cal.raw_min+(cal.dmax-cal.dmin)*cal.raw_step:
            raise UnsupportedFormatError(f"Auxiliary channel index {channel.index} exceeds the official digital range")
        return cal
    a, b = sorted((low*channel.scale_uv_per_count, high*channel.scale_uv_per_count))
    if a == b:
        a, b = a-1, b+1
    result = Calibration(outward_field(a, True), outward_field(b, False),
                         -32768, 32767, channel.scale_uv_per_count)
    if result.error_bound > max_error_uv:
        raise UnsupportedFormatError("No-clipping 16-bit quantization exceeds max_error_uv")
    return result
