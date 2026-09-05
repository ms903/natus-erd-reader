"""Build release assets from a clean, explicitly selected source staging tree."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT_FILES = ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md",
              "CONTRIBUTING.md", "SECURITY.md", "pyproject.toml", "MANIFEST.in", "setup.py")


def stage_source(source: Path, staging: Path) -> None:
    staging.mkdir(parents=True)
    for name in ROOT_FILES:
        shutil.copy2(source/name, staging/name)
    for name in ("src", "tests", "tools", "examples"):
        shutil.copytree(source/name, staging/name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyd", "*.so", "*.egg-info"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("pure", "native"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    build_root = source/"build"
    build_root.mkdir(exist_ok=True)
    staging = build_root/("release-stage-"+uuid.uuid4().hex)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["NATUS_ERD_NO_NATIVE"] = "1" if args.kind == "pure" else "0"
    environment["NATUS_ERD_REQUIRE_NATIVE"] = "0" if args.kind == "pure" else "1"
    try:
        stage_source(source, staging)
        command = [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output)]
        if args.kind == "native":
            command.append("--wheel")
        subprocess.run(command, cwd=staging, env=environment, check=True)
    finally:
        if staging.exists() and staging.resolve().parent == build_root.resolve():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
