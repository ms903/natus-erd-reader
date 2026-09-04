from __future__ import annotations

import json
import shutil
import threading
import unittest
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from natus_erd import EDFReader, UnsupportedFormatError
from natus_erd.viewer import ViewerApplication, create_server

from ._fixture import build_recording


class EDFAndViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path.cwd() / f".natus-erd-test-{uuid.uuid4().hex}"
        self.temporary.mkdir()
        self.addCleanup(shutil.rmtree, self.temporary, True)
        self.fixture = build_recording(self.temporary)

    def test_edf_random_access_and_physical_conversion(self) -> None:
        reader = EDFReader(self.fixture.edf)
        self.assertEqual(reader.info.n_raw_signals, 277)
        self.assertEqual(reader.info.n_data_signals, 276)
        self.assertEqual(reader.n_samples, 64)
        self.assertEqual(reader.sample_rate, 2048)
        self.assertEqual(reader.signals[0].label, "CH000")

        digital = reader.read_digital(2, 8, [0, 1])
        self.assertEqual(digital.shape, (2, 6))
        self.assertEqual(digital[0, 2], 32767)
        physical = reader.digital_to_physical(digital, [0, 1])
        self.assertEqual(physical.shape, digital.shape)
        self.assertTrue(np.isfinite(physical).all())

    def test_viewer_payload_and_http_endpoints(self) -> None:
        application = ViewerApplication(self.fixture.directory, None)
        info = application.info_payload()
        self.assertEqual(info["sampleRate"], 2048)
        self.assertEqual(info["nSamples"], 10)
        self.assertEqual(info["labelMatches"], 256)
        window = application.window_payload(
            {
                "start": ["0"],
                "duration": ["0.002"],
                "channels": ["0,1"],
                "points": ["200"],
            }
        )
        self.assertEqual(len(window["channels"]), 2)
        self.assertEqual(window["sourceSamples"], 4)
        self.assertEqual(window["channels"][0]["metrics"]["maxLsb"], 0.0)
        self.assertEqual(len(window["channels"][0]["diff"]), 4)
        self.assertTrue(
            all(value is None or isinstance(value, float) for value in window["channels"][0]["diff"])
        )

        server = create_server(self.fixture.directory, self.fixture.edf, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(f"{base}/api/info", timeout=5) as response:
            payload = json.load(response)
        self.assertEqual(payload["nSamples"], 10)
        with urllib.request.urlopen(f"{base}/", timeout=5) as response:
            html = response.read().decode("utf-8")
        self.assertIn("NeuroScope", html)

    def test_discontinuous_edf_is_rejected(self) -> None:
        with self.fixture.edf.open("r+b") as stream:
            stream.seek(192)
            stream.write(b"EDF+D".ljust(44, b" "))
        with self.assertRaisesRegex(UnsupportedFormatError, "EDF\\+D"):
            EDFReader(self.fixture.edf)


if __name__ == "__main__":
    unittest.main()
