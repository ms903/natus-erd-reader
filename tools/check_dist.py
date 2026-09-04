"""Audit wheel and source archives before publication; never extract them."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


FORBIDDEN_SUFFIXES = {
    ".erd", ".etc", ".stc", ".eeg", ".ent", ".edf", ".bdf",
    ".npy", ".npz", ".avi", ".mp4", ".mov", ".pem", ".key",
}
FORBIDDEN_PARTS = {"data", "figures", "reports", ".git", "__pycache__"}
SOURCE_ROOT_FILES = {
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md",
    "CONTRIBUTING.md", "SECURITY.md", "pyproject.toml", "MANIFEST.in",
    "PKG-INFO", "setup.cfg",
}
REQUIRED_PACKAGE_FILES = {
    "natus_erd/__init__.py", "natus_erd/py.typed",
    "natus_erd/web/index.html", "natus_erd/web/app.css", "natus_erd/web/app.js",
}


def check_member(name: str, size: int, *, wheel: bool) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    parts = path.parts if wheel else path.parts[1:]
    if not parts:
        return
    lowered = {part.casefold() for part in parts}
    if lowered & FORBIDDEN_PARTS:
        raise ValueError(f"Private/generated directory in archive: {name}")
    if path.suffix.casefold() in FORBIDDEN_SUFFIXES or ".ent.old" in name.casefold():
        raise ValueError(f"Recording/credential file in archive: {name}")
    if any(part.startswith((".env", ".venv")) for part in parts):
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
            members = [(info.filename, info.file_size) for info in archive.infolist() if not info.is_dir()]
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            entries = archive.getmembers()
            if any(item.issym() or item.islnk() for item in entries):
                raise ValueError("Source archive contains links")
            members = [(item.name, item.size) for item in entries if item.isfile()]
    else:
        raise ValueError(f"Unsupported distribution archive: {path.name}")

    if not members:
        raise ValueError("Empty distribution archive")
    for name, size in members:
        check_member(name, size, wheel=is_wheel)
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
