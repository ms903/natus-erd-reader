"""EDF signal labels, annotations and exact calibration helpers."""
from __future__ import annotations
import math
from dataclasses import dataclass
from fractions import Fraction
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


def exact_integer_field(value):
    candidates = [str(value)]
    for exponent in range(1, 20):
        if value % (10 ** exponent) == 0:
            candidates.append(f"{value // 10 ** exponent}E{exponent}")
    result = min(candidates, key=lambda v: (len(v), "E" in v, v))
    if len(result) > 8 or Fraction(result) != value:
        raise UnsupportedFormatError("Auxiliary integer endpoint exceeds the EDF 8-character field")
    return result


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

    @property
    def native(self):
        return (int(self.raw), self.source_scale, float(self.pmin), float(self.pmax),
                self.dmin, self.dmax, self.raw_min, self.raw_step)

    @property
    def error_bound(self):
        return 0.0 if self.raw else (float(self.pmax) - float(self.pmin)) / (self.dmax-self.dmin) / 2


def calibrate(channel, stats, max_error_uv):
    low, high, step = stats
    if channel.shorted:
        return Calibration("-1", "1", -32768, 32767, 0.0)
    if not channel.is_signal:
        if low == high:
            # Valid nondegenerate endpoints; digital zero reconstructs low.
            return Calibration(exact_integer_field(low-1), exact_integer_field(low+1),
                               -1, 1, 1.0, low-1, 1, True)
        if step < 1 or (high-low) % step or (high-low)//step > 65535:
            raise UnsupportedFormatError(f"Auxiliary channel index {channel.index} cannot be represented losslessly in EDF")
        return Calibration(exact_integer_field(low), exact_integer_field(high),
                           -32768, -32768+(high-low)//step, 1.0, low, step, True)
    a, b = sorted((low*channel.scale_uv_per_count, high*channel.scale_uv_per_count))
    if a == b:
        a, b = a-1, b+1
    result = Calibration(outward_field(a, True), outward_field(b, False),
                         -32768, 32767, channel.scale_uv_per_count)
    if result.error_bound > max_error_uv:
        raise UnsupportedFormatError("No-clipping 16-bit quantization exceeds max_error_uv")
    return result
