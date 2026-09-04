"""Safe parsing of NeuroWorks ENT notes and montage channel names."""

# The legacy-text normalization in _safe_parse_excel is adapted from
# Wonambi (Copyright 2014-2021 Gio Piantoni, Jordan O'Byrne), BSD-3-Clause.
# See THIRD_PARTY_NOTICES.md for the complete upstream notice.

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from struct import Struct
from typing import Any

from .binary import GENERIC_HEADER_SIZE
from .errors import DataIntegrityError
from .models import Event

_NOTE_HEADER = Struct("<4i")
_QUOTED_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


@dataclass(frozen=True, slots=True)
class EntNote:
    note_type: int
    length: int
    previous_length: int
    text: str
    value: Any | None


def _safe_parse_excel(text: str) -> Any | None:
    """Translate the legacy Excel-list notation and parse literals only."""

    converted = text.replace("\n", " ").replace("\\xd ", "")
    converted = converted.replace("(.", "{")
    converted = re.sub(r'\(([A-Za-z0-9," ]*)\)', r"[\1]", converted)
    converted = converted.replace(")", "}")
    converted = re.sub(r'(\{[\w"]*),', r"\1 :", converted)
    converted = converted.replace('{"', '"')
    converted = converted.replace("},", ",")
    converted = converted.replace("}}", "}")
    converted = re.sub(r'\(([0-9 ,\-.]*)\}', r"[\1]", converted)
    try:
        return ast.literal_eval(converted)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def read_ent_notes(path: Path) -> tuple[EntNote, ...]:
    """Read every non-terminal note from an ENT file."""

    try:
        stream = path.open("rb")
    except OSError as exc:
        raise DataIntegrityError(f"Cannot read {path.name}: {exc}") from exc

    notes: list[EntNote] = []
    with stream:
        try:
            stream.seek(GENERIC_HEADER_SIZE)
        except OSError as exc:
            raise DataIntegrityError(f"Cannot seek in {path.name}: {exc}") from exc
        while True:
            header = stream.read(_NOTE_HEADER.size)
            if not header:
                break
            if len(header) != _NOTE_HEADER.size:
                raise DataIntegrityError(f"{path.name} has a truncated ENT note header")
            note_type, length, previous_length, _unused = _NOTE_HEADER.unpack(header)
            if note_type == 0:
                break
            if length < _NOTE_HEADER.size:
                raise DataIntegrityError(
                    f"{path.name} has an invalid ENT note length: {length}"
                )
            payload = stream.read(length - _NOTE_HEADER.size)
            if len(payload) != length - _NOTE_HEADER.size:
                raise DataIntegrityError(f"{path.name} has a truncated ENT note")
            if payload.endswith(b"\0\0"):
                payload = payload[:-2]
            elif payload.endswith(b"\0"):
                payload = payload[:-1]
            text = payload.decode("utf-8", errors="replace")
            notes.append(
                EntNote(
                    note_type=note_type,
                    length=length,
                    previous_length=previous_length,
                    text=text,
                    value=_safe_parse_excel(text),
                )
            )
    return tuple(notes)


def _find_named_value(value: Any, key: str) -> Any | None:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
        for child in value.values():
            found = _find_named_value(child, key)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found = _find_named_value(child, key)
            if found is not None:
                return found
    return None


def _names_from_value(value: Any) -> tuple[str, ...]:
    found = _find_named_value(value, "ChanNames")
    if isinstance(found, Sequence) and not isinstance(found, (str, bytes)):
        names = tuple(item for item in found if isinstance(item, str))
        if names:
            return names
    return ()


def _names_from_raw_montage(text: str) -> tuple[str, ...]:
    """Extract only the quoted ChanNames list from an unparsed montage note."""

    marker = text.find("ChanNames")
    if marker < 0:
        return ()
    opening = text.find("(", marker + len("ChanNames"))
    if opening < 0:
        return ()

    depth = 0
    in_string = False
    escaped = False
    closing = -1
    for position in range(opening, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                closing = position
                break
    if closing < 0:
        return ()

    names: list[str] = []
    for token in _QUOTED_STRING.findall(text[opening + 1 : closing]):
        try:
            value = ast.literal_eval(token)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, str):
            names.append(value)
    return tuple(names)


def channel_names_from_notes(notes: Sequence[EntNote]) -> tuple[str, ...]:
    """Return names from the last usable montage note."""

    for note in reversed(notes):
        names = _names_from_value(note.value)
        if not names:
            names = _names_from_raw_montage(note.text)
        if names:
            return names
    return ()


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
        events.append(
            Event(
                stamp=stamp,
                sample=stamp - origin_stamp,
                text=text,
                user=user,
                note_type=note.note_type,
            )
        )
    events.sort(key=lambda event: event.stamp)
    return tuple(events)
