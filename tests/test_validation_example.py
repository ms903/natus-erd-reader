"""The diagnostic example only sees tiny synthetic, non-identifying files."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from struct import pack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import numpy as np

from examples import validate_recording as example
from examples.read_window import read_first_second
from natus_erd import DataIntegrityError, NatusERDReader, UnsupportedFormatError, __version__

from ._fixture import N_CHANNELS, _generic_header, build_recording


class ValidationExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        # Match existing fixtures; mode-0700 tempfile directories can be
        # inaccessible under some Windows AppContainer configurations.
        self.root = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.fixture = build_recording(self.root)

    def _run(self, **kwargs):
        output: list[str] = []
        # Synthetic tests may run from source before a matching distribution
        # is installed. Real diagnostic runs still check installed metadata.
        with patch.object(example.metadata, "version", return_value=kwargs.pop("installed_version", __version__)), patch.object(
            example.metadata, "distribution", return_value=SimpleNamespace(entry_points=kwargs.pop("entry_points", ()))
        ):
            results = example.validate_recording(
                kwargs.pop("recording", self.fixture.directory),
                preferred_channel=kwargs.pop("preferred_channel", None),
                # Never recursively discover this test from the example itself.
                run_synthetic_tests=False,
                emit=output.append,
                **kwargs,
            )
        return results, "\n".join(output)

    def _write_names(self, names: list[str]) -> None:
        text = '(.(."ChanNames", (' + ", ".join(f'"{name}"' for name in names) + ')))'
        payload = text.encode("utf-8") + b"\0\0"
        self.fixture.stc.with_suffix(".ent").write_bytes(
            _generic_header(3) + pack("<4i", 2, 16 + len(payload), 0, 0) + payload + bytes(16)
        )

    def _assert_redacted(self, output: str) -> None:
        for private in (
            str(self.fixture.directory),
            self.fixture.directory.name,
            self.fixture.stc.name,
            "CH000",
            "marker",
            "tester",
        ):
            self.assertNotIn(private, output)

    def test_public_api_checks_pass_on_synthetic_recording(self) -> None:
        before = {path.name: path.read_bytes() for path in self.fixture.directory.iterdir()}
        results, output = self._run()
        self.assertIsInstance(results, tuple)
        self.assertGreaterEqual(len(results), 20)
        self.assertFalse(any(item.status == "FAIL" for item in results), output)
        self.assertTrue(all(item.seconds >= 0 for item in results))
        self.assertIn('"missing_samples": 1', output)
        self.assertIn('"event_count": 1', output)
        for entry in ("directory", "stc", "eeg", "erd"):
            self.assertIn(f'"entry_type": "{entry}"', output)
        self.assertIn("Synthetic regression suite disabled", output)
        self.assertIn("FAIL=0", output)
        self._assert_redacted(output)
        after = {path.name: path.read_bytes() for path in self.fixture.directory.iterdir()}
        self.assertEqual(before, after, "Diagnostic checks must not modify source recordings")

    def test_missing_preferred_name_fails_redacted_and_stops(self) -> None:
        sensitive_name = "PRIVATE_MISSING_CHANNEL_IDENTIFIER"
        results, output = self._run(preferred_channel=sensitive_name)
        failures = [item for item in results if item.status == "FAIL"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0], results[-1])
        self.assertEqual(failures[0].detail, "AssertionError")
        self.assertIn("Resolve signal channel by index or explicit name", failures[0].name)
        self.assertNotIn("Scalar/list/tuple", output)
        self.assertNotIn(sensitive_name, output)
        self.assertIn("later checks were NOT run", output)
        self._assert_redacted(output)

    def test_full_index_opt_out_skips_without_calling_validate(self) -> None:
        with patch.object(NatusERDReader, "validate", side_effect=AssertionError("Must not traverse all indexes")) as validate:
            results, output = self._run(full_index_validation=False)
        validate.assert_not_called()
        index_result = next(item for item in results if item.name.startswith("All segment indexes:"))
        self.assertEqual(index_result.status, "SKIP")
        self.assertEqual(index_result.detail, "Full index traversal disabled")
        self.assertFalse(any(item.status == "FAIL" for item in results), output)

    def test_already_loaded_numpy_is_an_explicit_lazy_check_skip(self) -> None:
        self.assertIsNotNone(np.ndarray)
        results, output = self._run(full_index_validation=False)
        lazy = next(item for item in results if item.name == "Metadata and events leave NumPy unloaded")
        self.assertEqual(lazy.status, "SKIP")
        self.assertIn("NumPy was already imported", lazy.detail)
        self.assertFalse(any(item.status == "FAIL" for item in results), output)

    def test_none_preferred_name_selects_a_unique_signal(self) -> None:
        results, output = self._run(preferred_channel=None, full_index_validation=False)
        self.assertFalse(any(item.status == "FAIL" for item in results), output)
        self._assert_redacted(output)

    def test_default_selector_accepts_duplicated_signal_names(self) -> None:
        self._write_names(["PRIVATE_DUPLICATE"] * N_CHANNELS)
        results, output = self._run(full_index_validation=False)
        self.assertFalse(any(item.status == "FAIL" for item in results), output)
        name_result = next(item for item in results if item.name.startswith("Unique-name/index"))
        self.assertEqual(name_result.status, "SKIP")
        self.assertNotIn("PRIVATE_DUPLICATE", output)
        self.assertIn("Scalar/list/tuple index selectors", output)

    def test_all_shorted_signal_channels_are_still_valid(self) -> None:
        another = self.root / "all-shorted"
        another.mkdir()
        self.fixture = build_recording(another, shorted=frozenset(range(256)))
        results, output = self._run(full_index_validation=False)
        self.assertFalse(any(item.status == "FAIL" for item in results), output)
        self.assertIn('"shorted_channels": 256', output)

    def test_single_shorted_preferred_name_is_valid(self) -> None:
        results, output = self._run(preferred_channel="CH249", full_index_validation=False)
        self.assertFalse(any(item.status == "FAIL" for item in results), output)
        self.assertNotIn("CH249", output)

    def test_explicit_stc_and_eeg_among_multiple_recordings(self) -> None:
        (self.fixture.directory / "another.STC").write_bytes(self.fixture.stc.read_bytes())
        (self.fixture.directory / "another.EEG").write_bytes(self.fixture.eeg.read_bytes())
        for entry in (self.fixture.stc, self.fixture.eeg):
            with self.subTest(suffix=entry.suffix):
                results, output = self._run(recording=entry, full_index_validation=False)
                self.assertFalse(any(item.status == "FAIL" for item in results), output)
                self.assertEqual(next(item for item in results if item.name == "Optional directory entry point").status, "SKIP")
                self.assertEqual(next(item for item in results if item.name == "Optional .erd entry point").status, "SKIP")
                self.assertIn('"entry_type": "eeg"', output)
                self.assertIn('"entry_type": "stc"', output)

    def test_uppercase_extensions_and_irrelevant_stc_directory(self) -> None:
        for path in list(self.fixture.directory.iterdir()):
            path.rename(path.with_name(path.stem + path.suffix.upper()))
        (self.fixture.directory / "unrelated.stc").mkdir()
        results, output = self._run(full_index_validation=False)
        self.assertFalse(any(item.status == "FAIL" for item in results), output)
        for entry in ("directory", "stc", "eeg", "erd"):
            self.assertIn(f'"entry_type": "{entry}"', output)

    def test_window_duration_uses_actual_rate(self) -> None:
        another = self.root / "rate-512"
        another.mkdir()
        self.fixture = build_recording(another, sample_rate=512.0)
        results, output = self._run(full_index_validation=False)
        self.assertFalse(any(item.status == "FAIL" for item in results), output)
        self.assertIn('"sample_rate_hz": 512.0', output)
        self.assertIn('"window_duration_seconds": 0.01953125', output)

    def test_rate_dependent_window_stays_bounded(self) -> None:
        for rate, expected in ((0.5, 1), (256.5, 257), (512.0, 512), (2048.0, 2048), (4096.0, 2048), (1e300, 2048)):
            with self.subTest(rate=rate):
                self.assertEqual(example._window_width(rate, 100_000, 276), expected)
        self.assertEqual(example._window_width(2048.0, 10, 276), 10)
        self.assertLessEqual(example._window_width(2048.0, 100_000, 1024) * 1024 * 8, example.MAX_OUTPUT_BYTES)

    def test_first_second_example_uses_ceiling_and_recording_end(self) -> None:
        for rate, n_samples, stop in ((256.5, 1000, 257), (0.5, 100, 1), (512.0, 10, 10)):
            with self.subTest(rate=rate):
                reader = SimpleNamespace(info=SimpleNamespace(sample_rate=rate, n_samples=n_samples), read_samples=Mock())
                with patch("examples.read_window.NatusERDReader.open", return_value=reader):
                    read_first_second(Path("synthetic-recording"))
                reader.read_samples.assert_called_once_with(0, stop, channels=[0, 1])

    def test_older_installation_is_rejected_before_opening(self) -> None:
        with patch.object(NatusERDReader, "open") as opened:
            results, output = self._run(installed_version="0.2.0")
        opened.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "FAIL")
        self._assert_redacted(output)

    def test_legacy_installed_entrypoint_is_rejected(self) -> None:
        results, output = self._run(entry_points=(SimpleNamespace(group="console_scripts"),), full_index_validation=False)
        self.assertEqual(results[-1].status, "FAIL")
        self.assertEqual(results[-1].name, "Python-only installation: no CLI/EDF/viewer entry points")
        self._assert_redacted(output)

    def test_trusted_numeric_diagnostic_shows_reason(self) -> None:
        messages = (
            (UnsupportedFormatError, "Unsupported NeuroWorks ERD layout: headbox type 17, 256 channels"),
            (UnsupportedFormatError, "Unsupported ERD schema 8, base 1"),
            (DataIntegrityError, "ERD sample rate must be finite and positive"),
            (DataIntegrityError, "Recording duration must be finite for the declared sample rate"),
            (DataIntegrityError, "ERD sample rate differs in segment 2: expected 512 Hz, got 2048 Hz"),
            (DataIntegrityError, "STC stored sample count differs from ETC in segment 2: declared 4, indexed 3"),
        )
        for category, message in messages:
            with self.subTest(message=message):
                with patch.object(NatusERDReader, "open", side_effect=category(message)):
                    results, output = self._run()
                self.assertEqual(results[-1].detail, f"{category.__name__}: {message}")
                self.assertIn(message, output)
                self._assert_redacted(output)

    def test_diagnostic_allowlist_cannot_be_extended_by_private_suffix_or_type(self) -> None:
        trusted = "Unsupported ERD schema 8, base 1"
        for category, message in (
            (UnsupportedFormatError, trusted + " PRIVATE_IDENTIFIER"),
            (UnsupportedFormatError, trusted + "\nPRIVATE_IDENTIFIER"),
            (UnsupportedFormatError, "Unsupported PRIVATE_IDENTIFIER schema 8, base 1"),
            (DataIntegrityError, "ERD sample rate differs in segment PRIVATE_IDENTIFIER: expected 512 Hz, got 2048 Hz"),
            (RuntimeError, trusted),
        ):
            with self.subTest(category=category, message=message):
                self.assertEqual(example._safe_error(category(message)), category.__name__)

    def test_unexpected_open_failure_does_not_expose_exception_text(self) -> None:
        private = "PRIVATE_PATIENT_PATH_AND_EVENT_TEXT"
        with patch.object(NatusERDReader, "open", side_effect=RuntimeError(private)):
            results, output = self._run()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[-1].status, "FAIL")
        self.assertEqual(results[-1].detail, "RuntimeError")
        self.assertNotIn(private, output)
        self.assertNotIn("Traceback", output)
        self.assertNotIn("[START] Metadata", output)
        self._assert_redacted(output)

    def test_main_refuses_an_existing_numpy_session_without_changing_environment(self) -> None:
        before = dict(os.environ)
        output: list[str] = []
        with patch.object(example, "_print", side_effect=output.append), patch.object(
            example, "validate_recording", side_effect=AssertionError("No workload in reused NumPy process")
        ) as validate:
            code = example.main(self.fixture.directory, preferred_channel="CH000")
        self.assertEqual(code, 2)
        validate.assert_not_called()
        self.assertEqual(dict(os.environ), before)
        self.assertIn("fresh Python process", "\n".join(output))


if __name__ == "__main__":
    unittest.main()
