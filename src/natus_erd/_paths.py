"""Bounded, non-recursive lookup of the real names in a recording directory."""

from __future__ import annotations

import os
from pathlib import Path

from .binary import regular_file_size
from .errors import DataIntegrityError, UnsupportedFormatError
from .limits import DEFAULT_LIMITS, ReadLimits, check_limit


class RecordingDirectory:
    """Snapshot directory names, not payloads; resolve only selected members.

    Windows recordings often use mixed-case extensions. Use unique casefold
    matches on every platform without guessing between colliding filenames.
    The top-level path is still resolved by the operating system as supplied.
    """

    def __init__(self, directory: Path, *, limits: ReadLimits = DEFAULT_LIMITS) -> None:
        self.directory = directory.resolve(strict=True)
        if any(part.casefold() == "decimated" for part in self.directory.parts):
            raise UnsupportedFormatError("The Decimated derivative is outside the supported scope")
        self._names: dict[str, list[Path]] = {}
        self._stc_names: list[str] = []
        with os.scandir(self.directory) as entries:
            for count, entry in enumerate(entries, 1):
                check_limit(count, limits.max_directory_entries, "Recording directory entries")
                self._names.setdefault(entry.name.casefold(), []).append(Path(entry.path))
                # An unrelated ordinary backup.stc/ folder is not a recording
                # candidate. Links remain candidates so boundary checks apply.
                if entry.name.casefold().endswith(".stc") and not entry.is_dir(follow_symlinks=False):
                    self._stc_names.append(entry.name)

    def lookup(self, name: str, *, optional: bool = False) -> Path | None:
        matches = self._names.get(name.casefold(), ())
        if len(matches) > 1:
            raise DataIntegrityError("Ambiguous case-insensitive recording filename")
        if not matches:
            if optional:
                return None
            raise DataIntegrityError("Missing required recording file")
        try:
            resolved = matches[0].resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise DataIntegrityError("Missing or inaccessible recording file") from exc
        if resolved.parent != self.directory:
            raise DataIntegrityError("Recording file resolves outside its directory")
        regular_file_size(resolved)
        return resolved

    def stc_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        for name in self._stc_names:
            path = self.lookup(name)
            assert path is not None  # Required lookup raises instead of returning None.
            paths.append(path)
        return tuple(paths)

    def resolve_stc(self, source: Path) -> Path:
        """Select one recording without scanning other records' metadata."""
        if source.is_file():
            suffix = source.suffix.casefold()
            if suffix not in {".stc", ".eeg", ".erd"}:
                raise ValueError("Expected a recording directory or EEG/STC/ERD file")
            self.lookup(source.name)  # Also reject case collisions on an explicit entry.
            if suffix == ".stc":
                return source
            if suffix == ".eeg":
                stc = self.lookup(source.stem + ".stc", optional=True)
                if stc is None:
                    raise FileNotFoundError("No STC matches the supplied EEG file")
                return stc
        elif not source.is_dir():
            raise ValueError("Expected a recording directory or EEG/STC/ERD file")

        candidates = self.stc_paths()
        if not candidates:
            raise FileNotFoundError("No STC file found; pass the recording directory itself (discovery is not recursive)")
        if len(candidates) != 1:
            raise DataIntegrityError(
                f"Expected one main STC file, found {len(candidates)}; pass the target STC or matching EEG file"
            )
        return candidates[0]


def resolve_recording(path: Path, *, limits: ReadLimits = DEFAULT_LIMITS) -> tuple[Path, RecordingDirectory]:
    source = path.expanduser().resolve(strict=True)
    directory = source if source.is_dir() else source.parent
    files = RecordingDirectory(directory, limits=limits)
    return files.resolve_stc(source), files
