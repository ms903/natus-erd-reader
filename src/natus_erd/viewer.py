"""Local, dependency-free web viewer for synchronized ERD and EDF windows."""

from __future__ import annotations

import argparse
import json
import math
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .edf import EDFReader
from .errors import NatusERDError
from .reader import NatusERDReader, SIGNAL_CHANNEL_COUNT

_WEB_ROOT = Path(__file__).with_name("web")
_MAX_CHANNELS = 8
_MAX_WINDOW_SECONDS = 10.0


class ViewerApplication:
    def __init__(self, recording: str | Path, edf_path: str | Path | None) -> None:
        self.erd = NatusERDReader.open(recording)
        self.edf = EDFReader(edf_path or _discover_edf(Path(recording)))
        if len(self.edf.signals) < SIGNAL_CHANNEL_COUNT:
            raise ValueError("EDF contains fewer than 256 signal channels")
        if not math.isclose(
            self.edf.sample_rate, self.erd.info.sample_rate, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("ERD and EDF sampling frequencies do not match")
        for channel in range(SIGNAL_CHANNEL_COUNT):
            if not math.isclose(
                self.edf.signals[channel].sample_rate,
                self.erd.info.sample_rate,
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"EDF channel {channel} has a different sample rate")
            signal = self.edf.signals[channel]
            if signal.label != self.erd.channels[channel].name:
                raise ValueError(f"ERD/EDF channel labels differ at index {channel}")
            if signal.unit.replace("µ", "u").replace("μ", "u").lower() != "uv":
                raise ValueError(f"EDF channel {channel} is not calibrated in microvolts")
        self.n_samples = min(self.erd.info.n_samples, self.edf.n_samples)
        self._events = self.erd.read_events()
        self._read_lock = threading.Lock()

    def info_payload(self) -> dict[str, Any]:
        sample_rate = self.erd.info.sample_rate
        label_matches = sum(
            self.erd.channels[index].name == self.edf.signals[index].label
            for index in range(SIGNAL_CHANNEL_COUNT)
        )
        channels = [
            {
                "index": index,
                "name": self.erd.channels[index].name,
                "edfName": self.edf.signals[index].label,
                "shorted": self.erd.channels[index].shorted,
            }
            for index in range(SIGNAL_CHANNEL_COUNT)
        ]
        defaults = [
            channel["index"]
            for channel in channels
            if not channel["shorted"]
        ][:4]
        return {
            "sampleRate": sample_rate,
            "nSamples": self.n_samples,
            "durationSeconds": self.n_samples / sample_rate,
            "segments": self.erd.info.segment_count,
            "erdSamples": self.erd.info.n_samples,
            "edfSamples": self.edf.n_samples,
            "labelMatches": label_matches,
            "channels": channels,
            "defaultChannels": defaults,
            "maxChannels": _MAX_CHANNELS,
            "maxWindowSeconds": _MAX_WINDOW_SECONDS,
            "eventCount": len(self._events),
        }

    def window_payload(self, query: dict[str, list[str]]) -> dict[str, Any]:
        sample_rate = self.erd.info.sample_rate
        start_seconds = _query_float(query, "start", 0.0)
        duration_seconds = _query_float(query, "duration", 2.0)
        max_points = _query_int(query, "points", 2400)
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise ValueError("start must be a finite non-negative number")
        if (
            not math.isfinite(duration_seconds)
            or duration_seconds <= 0
            or duration_seconds > _MAX_WINDOW_SECONDS
        ):
            raise ValueError(
                f"duration must be greater than 0 and at most {_MAX_WINDOW_SECONDS:g}"
            )
        if not 200 <= max_points <= 4000:
            raise ValueError("points must be between 200 and 4000")

        channels_text = _query_value(query, "channels", "0,1,2,3")
        try:
            channels = tuple(int(value) for value in channels_text.split(","))
        except ValueError as exc:
            raise ValueError("channels must be comma-separated integer indices") from exc
        if not channels or len(channels) > _MAX_CHANNELS:
            raise ValueError(f"select between 1 and {_MAX_CHANNELS} channels")
        if len(set(channels)) != len(channels):
            raise ValueError("channels must not contain duplicates")
        if any(not 0 <= channel < SIGNAL_CHANNEL_COUNT for channel in channels):
            raise ValueError("channel index must be between 0 and 255")

        start_sample = round(start_seconds * sample_rate)
        window_samples = max(1, round(duration_seconds * sample_rate))
        if start_sample >= self.n_samples:
            raise ValueError("start is beyond the common ERD/EDF duration")
        stop_sample = min(self.n_samples, start_sample + window_samples)
        actual_count = stop_sample - start_sample

        with self._read_lock:
            erd_digital = self.erd.read_samples(
                start_sample, stop_sample, channels, units="digital"
            )
            edf_digital = self.edf.read_digital(start_sample, stop_sample, channels)
        scales = np.asarray(
            [self.erd.channels[channel].scale_uv_per_count for channel in channels],
            dtype=np.float64,
        )
        erd_uv = erd_digital * scales[:, np.newaxis]
        edf_uv = self.edf.digital_to_physical(edf_digital, channels)
        difference_uv = erd_uv - edf_uv

        point_count = min(actual_count, max_points)
        if point_count == actual_count:
            indices = np.arange(actual_count, dtype=np.int64)
        else:
            indices = np.unique(
                np.linspace(0, actual_count - 1, point_count, dtype=np.int64)
            )
        times = indices.astype(np.float64) / sample_rate

        channel_payloads: list[dict[str, Any]] = []
        for row, channel in enumerate(channels):
            native = erd_digital[row]
            exported = edf_digital[row].astype(np.float64)
            finite = np.isfinite(native)
            in_range = finite & (native >= -32768) & (native <= 32767)
            clipped = finite & ~in_range
            max_lsb = _maximum_or_none(np.abs(native[in_range] - exported[in_range]))
            max_uv = _maximum_or_none(
                np.abs(erd_uv[row, in_range] - edf_uv[row, in_range])
            )
            rms_uv = _rms_or_none(erd_uv[row, in_range] - edf_uv[row, in_range])
            channel_payloads.append(
                {
                    "index": channel,
                    "name": self.erd.channels[channel].name,
                    "shorted": self.erd.channels[channel].shorted,
                    "erd": _finite_list(erd_uv[row, indices]),
                    "edf": _finite_list(edf_uv[row, indices]),
                    "diff": _finite_list(difference_uv[row, indices]),
                    "metrics": {
                        "maxLsb": max_lsb,
                        "maxUv": max_uv,
                        "rmsUv": rms_uv,
                        "clipped": int(np.count_nonzero(clipped)),
                    },
                }
            )

        events = [
            {
                "sample": event.sample,
                "offsetSeconds": (event.sample - start_sample) / sample_rate,
                "text": event.text,
            }
            for event in self._events
            if start_sample <= event.sample < stop_sample
        ][:100]
        return {
            "startSample": start_sample,
            "stopSample": stop_sample,
            "startSeconds": start_sample / sample_rate,
            "durationSeconds": actual_count / sample_rate,
            "sampleRate": sample_rate,
            "sourceSamples": actual_count,
            "displayPoints": len(indices),
            "times": [round(float(value), 9) for value in times],
            "channels": channel_payloads,
            "events": events,
        }


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True


def create_server(
    recording: str | Path,
    edf_path: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ViewerServer:
    application = ViewerApplication(recording, edf_path)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/info":
                    self._json(application.info_payload())
                elif parsed.path == "/api/window":
                    self._json(application.window_payload(parse_qs(parsed.query)))
                elif parsed.path == "/api/health":
                    self._json({"status": "ok"})
                elif parsed.path in {"/", "/index.html"}:
                    self._asset("index.html", "text/html; charset=utf-8")
                elif parsed.path == "/assets/app.css":
                    self._asset("app.css", "text/css; charset=utf-8")
                elif parsed.path == "/assets/app.js":
                    self._asset("app.js", "text/javascript; charset=utf-8")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (ValueError, IndexError, KeyError, NatusERDError) as exc:
                self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception:
                self._json(
                    {"error": "读取窗口时发生内部错误"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _asset(self, name: str, content_type: str) -> None:
            try:
                payload = (_WEB_ROOT / name).read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self._headers(content_type, len(payload), cache="public, max-age=300")
            self.end_headers()
            self.wfile.write(payload)

        def _json(
            self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            encoded = json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(status)
            self._headers(
                "application/json; charset=utf-8", len(encoded), cache="no-store"
            )
            self.end_headers()
            self.wfile.write(encoded)

        def _headers(self, content_type: str, length: int, *, cache: str) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
            )

        def log_message(self, format: str, *args: object) -> None:
            return

    return ViewerServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="natus-erd-viewer",
        description="Browse synchronized Natus ERD and EDF waveforms locally",
    )
    parser.add_argument("recording", nargs="?", default="data")
    parser.add_argument("--edf", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        server = create_server(
            args.recording, args.edf, host=args.host, port=args.port
        )
    except (OSError, ValueError, NatusERDError) as exc:
        parser.error(str(exc))
    host = args.host if args.host not in {"0.0.0.0", "::"} else "127.0.0.1"
    url = f"http://{host}:{server.server_port}/"
    print(f"ERD / EDF viewer: {url}")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _discover_edf(recording: Path) -> Path:
    resolved = recording.expanduser().resolve(strict=True)
    bases = [resolved if resolved.is_dir() else resolved.parent]
    if bases[0].parent not in bases:
        bases.append(bases[0].parent)
    candidates: dict[Path, None] = {}
    for base in bases:
        for candidate in base.rglob("*.edf"):
            if not any(part.casefold() == "decimated" for part in candidate.parts):
                candidates[candidate.resolve()] = None
        if candidates:
            break
    found = sorted(candidates)
    if not found:
        raise FileNotFoundError("No EDF file found; select one with --edf")
    if len(found) != 1:
        raise ValueError(f"Found {len(found)} EDF files; select one with --edf")
    return found[0]


def _query_value(query: dict[str, list[str]], name: str, default: str) -> str:
    values = query.get(name)
    return values[0] if values else default


def _query_float(query: dict[str, list[str]], name: str, default: float) -> float:
    try:
        return float(_query_value(query, name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _query_int(query: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(_query_value(query, name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _finite_list(values: np.ndarray[Any, Any]) -> list[float | None]:
    return [
        round(float(value), 6) if math.isfinite(float(value)) else None
        for value in values
    ]


def _maximum_or_none(values: np.ndarray[Any, Any]) -> float | None:
    return float(np.max(values)) if values.size else None


def _rms_or_none(values: np.ndarray[Any, Any]) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else None


if __name__ == "__main__":
    raise SystemExit(main())
