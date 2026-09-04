from __future__ import annotations

import io
import shutil
import stat
import tarfile
import unittest
import uuid
import warnings
import zipfile
from pathlib import Path

from tools.check_dist import REQUIRED_PACKAGE_FILES, audit, check_member


class DistributionAuditTests(unittest.TestCase):
    """Small synthetic archives only; no extraction or real recording data."""

    def setUp(self) -> None:
        # Match the other fixtures: tempfile's mode-0700 directories are not
        # accessible in some Windows AppContainer configurations.
        self.root = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)

    def _wheel(self, extra: list[str | zipfile.ZipInfo] | None = None) -> Path:
        path = self.root / "synthetic.whl"
        names: list[str | zipfile.ZipInfo] = sorted(REQUIRED_PACKAGE_FILES)
        names.extend([
            "natus_erd_reader-0.2.0.dist-info/licenses/LICENSE",
            "natus_erd_reader-0.2.0.dist-info/licenses/THIRD_PARTY_NOTICES.md",
        ])
        names.extend(extra or [])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w") as archive:
                for name in names:
                    archive.writestr(name, b"synthetic")
        return path

    def _sdist(self, extra: list[str | tarfile.TarInfo] | None = None) -> Path:
        path = self.root / "synthetic.tar.gz"
        entries: list[str | tarfile.TarInfo] = [
            "synthetic/LICENSE", "synthetic/THIRD_PARTY_NOTICES.md",
            "synthetic/src/natus_erd/__init__.py",
        ]
        entries.extend(extra or [])
        with tarfile.open(path, "w:gz") as archive:
            for item in entries:
                if isinstance(item, str):
                    info = tarfile.TarInfo(item)
                    info.size = 9
                    archive.addfile(info, io.BytesIO(b"synthetic"))
                else:
                    archive.addfile(item)
        return path

    def test_minimal_python_only_archives_pass(self) -> None:
        self.assertEqual(audit(self._wheel()), len(REQUIRED_PACKAGE_FILES) + 2)
        self.assertEqual(audit(self._sdist()), 3)

    def test_windows_drive_and_alternate_stream_paths_are_rejected(self) -> None:
        for name, wheel in (
            ("C:/src/hidden.py", False),
            ("C:src/hidden.py", False),
            ("synthetic/src/natus_erd/notes:secret", False),
            ("natus_erd/notes:secret", True),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe archive path"):
                check_member(name, 1, wheel=wheel)

    def test_environment_names_are_rejected_case_insensitively(self) -> None:
        for part in (".ENV", ".EnV.production", ".VenV-cache"):
            with self.subTest(part=part), self.assertRaisesRegex(ValueError, "Local environment"):
                check_member(f"synthetic/src/{part}", 1, wheel=False)

    def test_removed_features_and_all_entry_points_are_rejected(self) -> None:
        for name in (
            "natus_erd/cli.py", "natus_erd/__main__.py", "natus_erd/edf.py",
            "natus_erd/viewer.py", "natus_erd/web/index.html",
            "natus_erd_reader-0.2.0.dist-info/entry_points.txt",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Removed non-ERD"):
                audit(self._wheel([name]))
        with self.assertRaisesRegex(ValueError, "Removed non-ERD"):
            audit(self._sdist(["synthetic/src/natus_erd_reader.egg-info/entry_points.txt"]))

    def test_duplicate_and_canonical_duplicate_zip_members_are_rejected(self) -> None:
        for name in (
            "natus_erd/reader.py", "natus_erd/./reader.py",
            "natus_erd//reader.py", "natus_erd/READER.PY",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Duplicate canonical"):
                audit(self._wheel([name]))

    def test_duplicate_source_members_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate canonical"):
            audit(self._sdist(["synthetic/./LICENSE"]))

    def test_multiple_source_roots_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one package root"):
            audit(self._sdist(["other/LICENSE"]))
        extra_directory = tarfile.TarInfo("other")
        extra_directory.type = tarfile.DIRTYPE
        with self.assertRaisesRegex(ValueError, "exactly one package root"):
            audit(self._sdist([extra_directory]))

    def test_zip_symlinks_are_rejected(self) -> None:
        info = zipfile.ZipInfo("natus_erd/symlink.py")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaisesRegex(ValueError, "link or special file"):
            audit(self._wheel([info]))

    def test_source_links_and_special_files_are_rejected(self) -> None:
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE):
            info = tarfile.TarInfo("synthetic/src/unsafe")
            info.type = kind
            info.linkname = "elsewhere"
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "links or special files"):
                audit(self._sdist([info]))

    def test_unsafe_directory_entries_are_checked(self) -> None:
        info = tarfile.TarInfo("synthetic/src/.ENV")
        info.type = tarfile.DIRTYPE
        with self.assertRaisesRegex(ValueError, "Local environment"):
            audit(self._sdist([info]))


if __name__ == "__main__":
    unittest.main()
