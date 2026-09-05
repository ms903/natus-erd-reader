from typing import Any

def process(payload: bytes, shorted: bytes, selected: tuple[int, ...], count: int,
            start: int, stop: int, operation: int, calibrations: tuple[Any, ...],
            output: Any, width: int, column: int) -> Any: ...
