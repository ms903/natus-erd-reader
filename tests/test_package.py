from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib.resources import files

import natus_erd


class PackageContractTests(unittest.TestCase):
    def test_public_exports_and_version(self) -> None:
        self.assertRegex(natus_erd.__version__, r"^\d+\.\d+\.\d+$")
        for name in natus_erd.__all__:
            self.assertIsNotNone(getattr(natus_erd, name))

    def test_typed_marker_and_offline_viewer_assets_are_packaged(self) -> None:
        root = files("natus_erd")
        self.assertTrue(root.joinpath("py.typed").is_file())
        for name in ("index.html", "app.css", "app.js"):
            self.assertTrue(root.joinpath("web", name).is_file())
        html = root.joinpath("web", "index.html").read_text(encoding="utf-8")
        self.assertIn('/assets/app.js', html)
        self.assertNotIn('<script src="http', html)

    def test_viewer_help_works_with_legacy_console_encoding(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "cp1252:strict"
        completed = subprocess.run(
            [sys.executable, "-m", "natus_erd.viewer", "--help"],
            env=environment,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("ascii", errors="replace"))
        self.assertIn(b"synchronized Natus ERD and EDF", completed.stdout)


if __name__ == "__main__":
    unittest.main()
