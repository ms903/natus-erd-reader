"""Bounded, non-executing parsing of NeuroWorks ENT notes.

The vendor text uses parenthesized lists and dotted key/value fields, for
example ``(.(."Stamp", 42), (."Text", "marker"))``. It is parsed directly,
without interpreting any text as Python source code.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from struct import Struct, unpack_from
from typing import Any

from .binary import GENERIC_HEADER_SIZE, regular_file_size
from .errors import DataIntegrityError, ResourceLimitError, UnsupportedFormatError
from .limits import DEFAULT_LIMITS, ReadLimits
from .models import Event

_NOTE_HEADER = Struct("<4i")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_HEX_NUMBER = re.compile(r"[+-]?0[xX][0-9a-fA-F]+\Z")
_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "v": "\v", "a": "\a"}


@dataclass(frozen=True, slots=True)
class EntNote:
    note_type: int
    length: int
    previous_length: int
    text: str
    value: Any | None


@dataclass(frozen=True, slots=True)
class _Field:
    name: str
    value: Any


@dataclass(slots=True)
class _Frame:
    values: list[Any] = field(default_factory=list)
    dotted: bool = False
    at_start: bool = True
    needs_value: bool = True


@dataclass(slots=True)
class _ParseBudget:
    total_nodes: int = 0


class _InvalidText(ValueError):
    """A token or container is outside the supported literal grammar."""


def _check_text_size(text: str, limits: ReadLimits) -> None:
    # Check characters first, so checking UTF-8 size cannot allocate an
    # unbounded temporary buffer even when called directly with hostile text.
    if len(text) > limits.max_ent_record_bytes:
        raise ResourceLimitError("ENT text exceeds the configured record size limit")
    try:
        byte_count = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise _InvalidText("Invalid Unicode in ENT text") from exc
    if byte_count > limits.max_ent_record_bytes:
        raise ResourceLimitError("ENT text exceeds the configured record size limit")


def _quoted(text: str, start: int) -> tuple[str, int]:
    """Read a string without evaluating Python or another language."""
    quote = text[start]
    position = start + 1
    fragment_start = position
    fragments: list[str] = []
    while position < len(text):
        char = text[position]
        if char == quote:
            fragments.append(text[fragment_start:position])
            return "".join(fragments), position + 1
        if char == "\0":
            raise _InvalidText("An ENT string contains an unescaped NUL")
        if char != "\\":
            position += 1
            continue
        fragments.append(text[fragment_start:position])
        position += 1
        if position == len(text):
            raise _InvalidText("Unterminated ENT string escape")
        escaped = text[position]
        position += 1
        if escaped in ("\\", '"', "'"):
            fragments.append(escaped)
        elif escaped in _ESCAPES:
            fragments.append(_ESCAPES[escaped])
        elif escaped in ("x", "u", "U"):
            width = {"x": 2, "u": 4, "U": 8}[escaped]
            # NeuroWorks also writes a single hex digit before whitespace,
            # notably a carriage-return escape immediately before a newline.
            # This is a bounded literal escape, not a global text rewrite.
            if (
                escaped == "x"
                and position + 1 < len(text)
                and text[position] in "0123456789abcdefABCDEF"
                and text[position + 1].isspace()
            ):
                width = 1
            digits = text[position:position + width]
            if len(digits) != width or any(c not in "0123456789abcdefABCDEF" for c in digits):
                raise _InvalidText("Invalid hexadecimal ENT string escape")
            value = int(digits, 16)
            if value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
                raise _InvalidText("Invalid Unicode ENT string escape")
            fragments.append(chr(value))
            position += width
        else:
            # Unknown escapes are preserved as vendor text (e.g. paths).
            fragments.append("\\" + escaped)
        fragment_start = position
    raise _InvalidText("Unterminated ENT string")


def _public_value(value: Any) -> Any:
    return {value.name: value.value} if isinstance(value, _Field) else value


def _close_frame(frame: _Frame) -> Any:
    if not frame.dotted:
        return [_public_value(value) for value in frame.values]
    if len(frame.values) == 2 and isinstance(frame.values[0], str):
        return _Field(frame.values[0], _public_value(frame.values[1]))
    result: dict[str, Any] = {}
    for value in frame.values:
        if not isinstance(value, _Field) or value.name in result:
            raise _InvalidText("Invalid or duplicate dotted ENT field")
        result[value.name] = value.value
    return result


def _parse_text(
    text: str,
    limits: ReadLimits,
    budget: _ParseBudget,
    *,
    require_end: bool = True,
    start: int = 0,
    initial_nodes: int = 0,
    base_depth: int = 0,
) -> Any:
    """Parse using a bounded explicit stack; input depth never uses recursion."""
    frames: list[_Frame] = []
    position = start
    nodes = initial_nodes
    root: Any = None
    has_root = False

    def count_node() -> None:
        nonlocal nodes
        nodes += 1
        budget.total_nodes += 1
        if nodes > limits.max_parse_nodes:
            raise ResourceLimitError("ENT literal exceeds the configured node count")
        if budget.total_nodes > limits.max_total_parse_nodes:
            raise ResourceLimitError("ENT file exceeds the configured total parser node count")

    while position < len(text):
        char = text[position]
        if char.isspace():
            position += 1
            continue
        if has_root and not frames:
            if require_end:
                raise _InvalidText("Unexpected text after ENT literal")
            break
        if char == "(":
            if frames and not frames[-1].needs_value:
                raise _InvalidText("Missing ENT list separator")
            count_node()
            if base_depth + len(frames) >= limits.max_parse_depth:
                raise ResourceLimitError("ENT literal exceeds the configured nesting depth")
            frames.append(_Frame())
            position += 1
            continue
        if char == "." and (position + 1 == len(text) or not text[position + 1].isdigit()):
            if not frames or not frames[-1].at_start:
                raise _InvalidText("Unexpected dotted ENT marker")
            frames[-1].dotted = True
            frames[-1].at_start = False
            position += 1
            continue
        if char == ",":
            if not frames or frames[-1].needs_value:
                raise _InvalidText("Unexpected ENT list separator")
            frames[-1].needs_value = True
            frames[-1].at_start = False
            position += 1
            continue
        if char == ")":
            if not frames:
                raise _InvalidText("Unexpected closing ENT delimiter")
            value = _close_frame(frames.pop())
            position += 1
        else:
            if frames and not frames[-1].needs_value:
                raise _InvalidText("Missing ENT list separator")
            count_node()
            if char in ('"', "'"):
                value, position = _quoted(text, position)
            else:
                end = position
                while end < len(text) and not text[end].isspace() and text[end] not in "(),":
                    end += 1
                atom = text[position:end]
                if len(atom) > 64:
                    raise ResourceLimitError("ENT numeric or identifier token is too long")
                if atom in ("True", "False", "None"):
                    value = {"True": True, "False": False, "None": None}[atom]
                elif _HEX_NUMBER.fullmatch(atom):
                    value = int(atom, 16)
                elif _NUMBER.fullmatch(atom):
                    if any(c in atom for c in ".eE"):
                        value = float(atom)
                        if not math.isfinite(value):
                            raise _InvalidText("Non-finite ENT number")
                    else:
                        value = int(atom)
                else:
                    raise _InvalidText("Unsupported ENT token")
                position = end
        if frames:
            if not frames[-1].needs_value:
                raise _InvalidText("Missing ENT list separator")
            frames[-1].values.append(value)
            frames[-1].needs_value = False
            frames[-1].at_start = False
        else:
            root = _public_value(value)
            has_root = True
    if frames or not has_root:
        raise _InvalidText("Incomplete ENT literal")
    return root


def _parse_note(text: str, limits: ReadLimits, budget: _ParseBudget) -> Any | None:
    try:
        _check_text_size(text, limits)
        return _parse_text(text, limits, budget)
    except _InvalidText:
        return None


def _safe_parse_excel(text: str, *, limits: ReadLimits = DEFAULT_LIMITS) -> Any | None:
    """Parse ENT literals; unknown syntax returns None, resource violations raise.

    Function calls, attributes, operators and comprehensions are not supported.
    """
    return _parse_note(text, limits, _ParseBudget())


def read_ent_notes(path: Path, *, limits: ReadLimits = DEFAULT_LIMITS) -> tuple[EntNote, ...]:
    """Check file size, schema and remaining record bytes before payload reads."""
    file_size = regular_file_size(path)
    if file_size < GENERIC_HEADER_SIZE:
        raise DataIntegrityError("ENT file has a truncated generic header")
    if file_size > limits.max_ent_bytes:
        raise ResourceLimitError("ENT file exceeds the configured size limit")
    notes: list[EntNote] = []
    budget = _ParseBudget()
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            if stream.tell() != file_size:
                raise DataIntegrityError("ENT file changed while opening")
            stream.seek(0)
            generic = stream.read(GENERIC_HEADER_SIZE)
            if len(generic) != GENERIC_HEADER_SIZE:
                raise DataIntegrityError("ENT file has a truncated generic header")
            if unpack_from("<HH", generic, 16) != (3, 1):
                raise UnsupportedFormatError("Only ENT schema 3 / base schema 1 is supported")
            while stream.tell() < file_size:
                remaining = file_size - stream.tell()
                if remaining < _NOTE_HEADER.size:
                    raise DataIntegrityError("ENT file has a truncated note header")
                header = stream.read(_NOTE_HEADER.size)
                if len(header) != _NOTE_HEADER.size:
                    raise DataIntegrityError("ENT file changed or has a truncated note header")
                note_type, length, previous_length, _unused = _NOTE_HEADER.unpack(header)
                if note_type == 0:
                    break
                if length < _NOTE_HEADER.size or length > remaining:
                    raise DataIntegrityError("ENT note length lies outside the remaining file")
                if length > limits.max_ent_record_bytes:
                    raise ResourceLimitError("ENT note exceeds the configured record size limit")
                if len(notes) >= limits.max_ent_records:
                    raise ResourceLimitError("ENT file exceeds the configured record count")
                payload_size = length - _NOTE_HEADER.size
                payload = stream.read(payload_size)
                if len(payload) != payload_size:
                    raise DataIntegrityError("ENT file changed or has a truncated note")
                if payload.endswith(b"\0\0"):
                    payload = payload[:-2]
                elif payload.endswith(b"\0"):
                    payload = payload[:-1]
                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise DataIntegrityError("ENT note contains invalid UTF-8") from exc
                value = _parse_note(text, limits, budget)
                notes.append(EntNote(note_type, length, previous_length, text, value))
            stream.seek(0, 2)
            if stream.tell() != file_size:
                raise DataIntegrityError("ENT file changed while reading")
    except OSError as exc:
        raise DataIntegrityError("Cannot read ENT file") from exc
    return tuple(notes)


def _find_named_value(value: Any, key: str, *, limits: ReadLimits = DEFAULT_LIMITS) -> Any | None:
    stack = [value]
    visited: set[int] = set()
    count = 0
    while stack:
        current = stack.pop()
        count += 1
        if count > limits.max_parse_nodes:
            raise ResourceLimitError("ENT field search exceeds the configured node count")
        if isinstance(current, Mapping):
            if id(current) in visited:
                continue
            visited.add(id(current))
            if key in current:
                return current[key]
            if count + len(stack) + len(current) > limits.max_parse_nodes:
                raise ResourceLimitError("ENT field search exceeds the configured node count")
            stack.extend(reversed(tuple(current.values())))
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            if id(current) in visited:
                continue
            visited.add(id(current))
            if count + len(stack) + len(current) > limits.max_parse_nodes:
                raise ResourceLimitError("ENT field search exceeds the configured node count")
            stack.extend(reversed(current))
    return None


def _name_slot(value: Any) -> str | None:
    """Keep valid vendor text verbatim, without shifting empty label slots."""
    return value if isinstance(value, str) and value.strip() else None


def _names_from_value(value: Any, *, limits: ReadLimits = DEFAULT_LIMITS) -> tuple[str | None, ...]:
    found = _find_named_value(value, "ChanNames", limits=limits)
    if isinstance(found, Sequence) and not isinstance(found, (str, bytes)):
        if len(found) > limits.max_parse_nodes:
            raise ResourceLimitError("ENT channel names exceed the configured node count")
        return tuple(_name_slot(item) for item in found)
    return ()


def _names_from_raw_montage(text: str, *, limits: ReadLimits = DEFAULT_LIMITS) -> tuple[str | None, ...]:
    """Recover only a quoted ChanNames field from unsupported montage syntax."""
    try:
        _check_text_size(text, limits)
        position = 0
        depth = 0
        nodes = 0
        while position < len(text):
            char = text[position]
            if char in ('"', "'"):
                start = position
                value, position = _quoted(text, position)
                nodes += 1
                prefix_end = start
                while prefix_end > 0 and text[prefix_end - 1].isspace():
                    prefix_end -= 1
                if value == "ChanNames" and text[max(0, prefix_end - 2):prefix_end] == "(.":
                    tail = position
                    while tail < len(text) and text[tail].isspace():
                        tail += 1
                    if tail < len(text) and text[tail] == ",":
                        tail += 1
                        while tail < len(text) and text[tail].isspace():
                            tail += 1
                        if tail < len(text) and text[tail] == "(":
                            names = _parse_text(
                                text, limits, _ParseBudget(nodes),
                                require_end=False, start=tail,
                                initial_nodes=nodes, base_depth=depth,
                            )
                            if isinstance(names, list) and names:
                                return tuple(_name_slot(name) for name in names)
                            # Do not search repeated or nested candidates after
                            # a malformed channel list; the field is ambiguous.
                            return ()
            else:
                position += 1
                if char == "(":
                    depth += 1
                    nodes += 1
                elif char == ")":
                    depth = max(0, depth - 1)
            if depth > limits.max_parse_depth or nodes > limits.max_parse_nodes:
                raise ResourceLimitError("ENT montage exceeds the configured parser limits")
    except _InvalidText:
        return ()
    return ()


def channel_names_from_notes(notes: Sequence[EntNote], *, limits: ReadLimits = DEFAULT_LIMITS) -> tuple[str | None, ...]:
    """Return positional names from the last nonempty montage list.

    Even a list consisting entirely of missing labels is usable: falling back
    to an older montage would silently assign stale labels to these channels.
    """
    for note in reversed(notes):
        names = _names_from_value(note.value, limits=limits)
        if not names:
            names = _names_from_raw_montage(note.text, limits=limits)
        if names:
            return names
    return ()


def complete_channel_names(names: Sequence[str | None], n_channels: int) -> tuple[str, ...]:
    """Fill missing slots with collision-free, zero-based ``chanNNN`` names.

    Reserve all real vendor labels before generating defaults, including labels
    beyond the recorded channel count. Real duplicate names remain duplicates;
    the reader can reject ambiguous name selection without rewriting metadata.
    """
    if isinstance(n_channels, bool) or not isinstance(n_channels, int):
        raise TypeError("n_channels must be an integer")
    if n_channels < 0:
        raise ValueError("n_channels must be nonnegative")
    reserved = {name for name in names if _name_slot(name) is not None}
    result: list[str] = []
    for index in range(n_channels):
        name = _name_slot(names[index]) if index < len(names) else None
        if name is None:
            base = f"chan{index:03d}"
            name = base
            suffix = 0
            while name in reserved:
                suffix += 1
                name = f"{base}_{suffix}"
            reserved.add(name)
        result.append(name)
    return tuple(result)


def events_from_notes(notes: Sequence[EntNote], origin_stamp: int) -> tuple[Event, ...]:
    """Convert parsed ENT mappings with Stamp/Text fields into events."""
    events: list[Event] = []
    for note in notes:
        value = note.value
        if not isinstance(value, Mapping):
            continue
        stamp = value.get("Stamp")
        text = value.get("Text")
        if isinstance(stamp, bool) or not isinstance(stamp, int):
            continue
        if not isinstance(text, str):
            continue
        user: str | None = None
        data = value.get("Data")
        if isinstance(data, Mapping) and isinstance(data.get("User"), str):
            user = data["User"]
        events.append(Event(stamp=stamp, sample=stamp - origin_stamp, text=text, user=user, note_type=note.note_type))
    events.sort(key=lambda event: event.stamp)
    return tuple(events)
