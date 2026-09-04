"""Audit wheel and source archives before publication; never extract them."""

from __future__ import annotations

import argparse
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


FORBIDDEN_SUFFIXES = {
    ".erd", ".etc", ".stc", ".eeg", ".ent", ".edf", ".bdf",
    ".npy", ".npz", ".avi", ".mp4", ".mov", ".pem", ".key",
}
FORBIDDEN_PARTS = {"data", "figures", "reports", ".git", "__pycache__"}
REMOVED_MEMBERS = {
    "cli.py", "__main__.py", "edf.py", "viewer.py", "test_edf_viewer.py",
    "compare_edf.py", "plot_error_summary.py", "entry_points.txt",
}
SOURCE_ROOT_FILES = {
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md",
    "CONTRIBUTING.md", "SECURITY.md", "pyproject.toml", "MANIFEST.in",
    "PKG-INFO", "setup.cfg",
}
REQUIRED_PACKAGE_FILES = {
    "natus_erd/__init__.py", "natus_erd/py.typed",
    "natus_erd/reader.py", "natus_erd/decoder.py", "natus_erd/binary.py",
    "natus_erd/ent.py", "natus_erd/errors.py", "natus_erd/limits.py",
    "natus_erd/models.py",
}


def check_member(name: str, size: int, *, wheel: bool, directory: bool = False) -> None:
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
        or any(ord(character) < 32 for character in name)
        or any(part.endswith((" ", ".")) for part in path.parts)
    ):
        raise ValueError(f"Unsafe archive path: {name}")
    parts = path.parts if wheel else path.parts[1:]
    if not parts:
        if directory:
            return
        raise ValueError(f"Source file is outside its package root: {name}")
    lowered = {part.casefold() for part in parts}
    if "web" in lowered or path.name.casefold() in REMOVED_MEMBERS:
        raise ValueError(f"Removed non-ERD application feature in archive: {name}")
    if lowered & FORBIDDEN_PARTS:
        raise ValueError(f"Private/generated directory in archive: {name}")
    if path.suffix.casefold() in FORBIDDEN_SUFFIXES or ".ent.old" in name.casefold():
        raise ValueError(f"Recording/credential file in archive: {name}")
    if any(part.casefold().startswith((".env", ".venv")) for part in parts):
        raise ValueError(f"Local environment file in archive: {name}")
    if size > 5_000_000:
        raise ValueError(f"Unexpected file larger than 5 MB: {name}")
    if wheel:
        if parts[0] != "natus_erd" and not parts[0].endswith(".dist-info"):
            raise ValueError(f"Unexpected wheel root: {name}")
    elif parts[0] not in {"src", "tests", "tools", "examples"}:
        if len(parts) != 1 or parts[0] not in SOURCE_ROOT_FILES:
            raise ValueError(f"Unexpected source archive member: {name}")


def audit(path: Path) -> int:
    is_wheel = path.suffix == ".whl"
    if is_wheel:
        with zipfile.ZipFile(path) as archive:
            zip_entries = archive.infolist()
            for info in zip_entries:
                mode = info.external_attr >> 16
                kind = stat.S_IFMT(mode)
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError("Wheel contains a link or special file")
            entries = [(info.filename, info.file_size, info.is_dir()) for info in zip_entries]
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            tar_entries = archive.getmembers()
            if any(not (item.isfile() or item.isdir()) for item in tar_entries):
                raise ValueError("Source archive contains links or special files")
            entries = [(item.name, item.size, item.isdir()) for item in tar_entries]
    else:
        raise ValueError(f"Unsupported distribution archive: {path.name}")

    seen: set[str] = set()
    roots: set[str] = set()
    for name, size, directory in entries:
        check_member(name, size, wheel=is_wheel, directory=directory)
        # Check all entries, including directories. Normalize POSIX spelling
        # and case so paths cannot alias on case-insensitive installations.
        canonical = PurePosixPath(name).as_posix().casefold()
        if canonical in seen:
            raise ValueError(f"Duplicate canonical archive member: {name}")
        seen.add(canonical)
        roots.add(PurePosixPath(name).parts[0])
    if not is_wheel and len(roots) != 1:
        raise ValueError("Source archive must contain exactly one package root")
    members = [(name, size) for name, size, directory in entries if not directory]
    if not members:
        raise ValueError("Empty distribution archive")
    names = {name for name, _ in members}
    if is_wheel:
        missing = REQUIRED_PACKAGE_FILES - names
        if missing:
            raise ValueError(f"Wheel is missing package resources: {sorted(missing)}")
    for notice in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        if not any(PurePosixPath(name).name == notice for name in names):
            raise ValueError(f"Archive is missing license notice: {notice}")
    return len(members)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("dist"))
    args = parser.parse_args()
    archives = sorted(args.directory.glob("*.whl")) + sorted(args.directory.glob("*.tar.gz"))
    if not archives:
        parser.error("No wheel or source archives found")
    for archive in archives:
        count = audit(archive)
        print(f"PASS {archive.name}: {count} audited files; no recordings or reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
