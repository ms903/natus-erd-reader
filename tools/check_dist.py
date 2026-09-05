"""Audit wheel and source archives before publication; never extract them."""

from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import re
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


FORBIDDEN_SUFFIXES = {
    ".erd", ".etc", ".stc", ".eeg", ".ent", ".snc", ".edf", ".bdf",
    ".npy", ".npz", ".avi", ".mp4", ".mov", ".pem", ".key",
}
FORBIDDEN_PARTS = {"data", "figures", "reports", "archive", ".git", "__pycache__"}
SOURCE_ROOT_FILES = {
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md",
    "CONTRIBUTING.md", "SECURITY.md", "pyproject.toml", "MANIFEST.in",
    "PKG-INFO", "setup.cfg", "setup.py",
}
REQUIRED_PACKAGE_FILES = {"natus_erd/__init__.py", "natus_erd/py.typed"}


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
    elif parts[0] == "src" and len(parts) > 1:
        if parts[1] not in {"natus_erd", "natus_erd_reader.egg-info"}:
            raise ValueError(f"Unexpected source package: {name}")
    elif parts[0] not in {"src", "tests", "tools", "examples"}:
        if len(parts) != 1 or parts[0] not in SOURCE_ROOT_FILES:
            raise ValueError(f"Unexpected source archive member: {name}")


def audit(path: Path, *, version: str | None = None) -> int:
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
    metadata_names = [name for name in names if
                      (name.endswith(".dist-info/METADATA") if is_wheel else name == next(iter(roots))+"/PKG-INFO")]
    if len(metadata_names) != 1:
        raise ValueError("Archive must contain one package metadata file")
    metadata_name = metadata_names[0]
    if is_wheel:
        with zipfile.ZipFile(path) as archive:
            raw = archive.read(metadata_name)
            package_init = archive.read("natus_erd/__init__.py")
    else:
        with tarfile.open(path, "r:gz") as archive:
            stream = archive.extractfile(metadata_name)
            assert stream is not None
            raw = stream.read()
            stream = archive.extractfile(next(iter(roots))+"/src/natus_erd/__init__.py")
            if stream is None:
                raise ValueError("Source package initializer is missing")
            package_init = stream.read()
    metadata = BytesParser().parsebytes(raw)
    identity = metadata.get("Version", "")
    if (metadata.get_all("Name") != ["natus-erd-reader"]
            or metadata.get_all("Version") != [identity]
            or not re.fullmatch(r"\d+\.\d+\.\d+(?:rc\d+)?", identity)
            or (version is not None and identity != version)):
        raise ValueError("Distribution metadata identity does not match the release")
    stem = f"natus_erd_reader-{identity}"
    assignments = [node for node in ast.parse(package_init).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]
    if len(assignments) != 1 or ast.literal_eval(assignments[0].value) != identity:
        raise ValueError("Package source version does not match its metadata")
    if (not path.name.startswith(stem+"-") if is_wheel else path.name != stem+".tar.gz"):
        raise ValueError("Archive filename does not match its metadata")
    if is_wheel:
        if roots != {"natus_erd", stem+".dist-info"}:
            raise ValueError("Unexpected wheel metadata directory")
        if metadata_name != stem+".dist-info/METADATA":
            raise ValueError("Wheel metadata directory does not match its version")
        with zipfile.ZipFile(path) as archive:
            wheel = BytesParser().parsebytes(archive.read(stem+".dist-info/WHEEL"))
        tags = wheel.get_all("Tag", [])
        filename_tags = path.name.removesuffix(".whl").rsplit("-", 3)[1:]
        advertised = {f"{python}-{abi}-{platform}" for python in filename_tags[0].split(".")
                      for abi in filename_tags[1].split(".") for platform in filename_tags[2].split(".")}
        if set(tags) != advertised:
            raise ValueError("Wheel tags do not match its filename")
        native = any(name.endswith((".pyd", ".so")) for name in names)
        pure = wheel.get("Root-Is-Purelib", "").lower() == "true"
        if pure == native or (pure and advertised != {"py3-none-any"}):
            raise ValueError("Wheel binary contents do not match its platform tags")
    return len(members)


def audit_set(directory: Path, version: str, *, complete: bool = False) -> list[Path]:
    """Check one version's assets; complete requires the supported wheel matrix."""
    archives = sorted(directory.iterdir())
    if not archives or any(not p.is_file() or not p.name.endswith((".whl", ".tar.gz")) for p in archives):
        raise ValueError("Distribution directory must contain only wheel and source archives")
    for path in archives:
        audit(path, version=version)
    if complete:
        stem = f"natus_erd_reader-{version}"
        required = {stem+"-py3-none-any.whl", stem+".tar.gz"}
        names = {path.name for path in archives}
        if not required <= names or len(names) != 12:
            raise ValueError("Release requires a pure wheel, sdist and ten native wheels")
        for python in ("cp310", "cp311", "cp312", "cp313", "cp314"):
            for platform in ("win_amd64", "manylinux"):
                matches = [n for n in names if n.startswith(f"{stem}-{python}-{python}-")
                           and (n.endswith("-win_amd64.whl") if platform == "win_amd64"
                                else "manylinux" in n and n.endswith("_x86_64.whl"))]
                if len(matches) != 1:
                    raise ValueError(f"Release requires one {python} {platform} wheel")
    return archives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("dist"))
    parser.add_argument("--version")
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    archives = sorted(args.directory.glob("*.whl")) + sorted(args.directory.glob("*.tar.gz"))
    if not archives:
        parser.error("No wheel or source archives found")
    if args.complete:
        if not args.version:
            parser.error("--complete requires --version")
        audit_set(args.directory, args.version, complete=True)
    for archive in archives:
        count = audit(archive, version=args.version)
        print(f"PASS {archive.name}: {count} audited files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
