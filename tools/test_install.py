"""Install one wheel/sdist into an isolated target and run the source tests."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--backend", choices=("native", "pure"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root/"build"/("install-test-"+uuid.uuid4().hex)
    target.mkdir(parents=True)
    environment = os.environ.copy()
    environment["NATUS_ERD_NO_NATIVE"] = "1" if args.backend == "pure" else "0"
    environment["NATUS_ERD_REQUIRE_NATIVE"] = "0" if args.backend == "pure" else "1"
    environment["PYTHONPATH"] = str(target)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                        "--target", str(target), str(args.archive.resolve())], env=environment, check=True)
        probe = ("from pathlib import Path; import natus_erd; from importlib.metadata import version; "
                 "assert version('natus-erd-reader') == natus_erd.__version__; "
                 "from natus_erd._export_worker import native_available; "
                 f"assert Path(natus_erd.__file__).is_relative_to({str(target)!r}); "
                 f"assert native_available() is {args.backend == 'native'}")
        subprocess.run([sys.executable, "-c", probe], cwd=root, env=environment, check=True)
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                       cwd=root, env=environment, check=True)
    finally:
        if target.exists() and target.resolve().parent == (root/"build").resolve():
            shutil.rmtree(target)


if __name__ == "__main__":
    main()
