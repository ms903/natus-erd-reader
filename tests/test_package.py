from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib.resources import files
from pathlib import Path

import natus_erd


class PackageContractTests(unittest.TestCase):
    def test_public_exports_and_version(self) -> None:
        self.assertRegex(natus_erd.__version__, r"^\d+\.\d+\.\d+(?:rc\d+)?$")
        for name in natus_erd.__all__:
            self.assertIsNotNone(getattr(natus_erd, name))

    def test_typed_marker(self) -> None:
        root = files("natus_erd")
        self.assertTrue(root.joinpath("py.typed").is_file())

    def test_import_does_not_load_numpy_or_change_environment(self) -> None:
        environment = os.environ.copy()
        # Select the same source or installed package as this test process.
        environment["PYTHONPATH"] = str(Path(natus_erd.__file__).resolve().parents[1])
        probe = """
import os
import sys
before = dict(os.environ)
class NoNumpy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('numpy.'):
            raise AssertionError('metadata import must not initialize NumPy')
sys.meta_path.insert(0, NoNumpy())
import natus_erd
assert 'numpy' not in sys.modules
assert dict(os.environ) == before
assert natus_erd.ReadLimits().max_read_bytes == 64 * 1024 * 1024
"""
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            env=environment,
            capture_output=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("ascii", errors="replace"))


if __name__ == "__main__":
    unittest.main()
