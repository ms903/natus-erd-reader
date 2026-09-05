"""Shared validation for integer-valued public arguments."""
from numbers import Integral


def integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)
