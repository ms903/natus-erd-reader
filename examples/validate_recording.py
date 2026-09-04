"""Bounded public-API integration checks plus the checkout's synthetic tests.

Edit RECORDING_PATH, then run this file in a fresh Python process. No recording
is modified, no array/event text/path is printed, and nothing is uploaded.
This is an example script, NOT an installed package command-line application.
Passing these checks is not full-recording decoding or clinical validation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, dataclass, replace
import importlib
from importlib import metadata, util
import json
from math import ceil
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Callable
import unittest


# Use an ordinary raw Windows path: no Markdown escapes or HTML entities.
RECORDING_PATH = Path(r"D:\path\to\recording")
PREFERRED_CHANNEL = None  # Optional exact name; default checks use channel indices.
RUN_FULL_INDEX_VALIDATION = True  # Indexes only; not all waveform payloads.
RUN_SYNTHETIC_TESTS = True  # Requires this repository's tests/ directory.
WINDOW_SAMPLES = 2048
MAX_OUTPUT_BYTES = 8 * 1024**2


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    seconds: float
    detail: str = ""


class _Skip(Exception):
    pass


class _Stop(Exception):
    pass


def _print(text: str) -> None:
    print(text, flush=True)


def _require(condition: bool, message: str) -> None:
    # Explicit checks continue to work when Python is started with -O.
    if not condition:
        raise AssertionError(message)


def _raises(expected: type[Exception], action: Callable[[], object]) -> None:
    try:
        action()
    except expected:
        return
    raise AssertionError("An invalid request was unexpectedly accepted")


_NUMBER = r"-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?"
_LAYOUT_ITEM = (
    rf"(?:file schema [0-9]+|base schema [0-9]+|headbox type -?[0-9]+|"
    rf"[0-9]+ channels|sample rate {_NUMBER} Hz|delta width -?[0-9]+|"
    r"discard width -?[0-9]+|multiple headboxes|nonidentity physical-channel layout|"
    r"per-channel frequency factors)"
)
_SAFE_ERROR_PATTERNS = (
    rf"Unsupported NeuroWorks ERD layout: {_LAYOUT_ITEM}(?:, {_LAYOUT_ITEM})*",
    r"Unsupported (?:ERD|STC|ETC|ENT) schema [0-9]+, base [0-9]+",
    r"ERD sample rate must be finite and positive",
    r"Recording duration must be finite for the declared sample rate",
    rf"ERD sample rate differs in segment [0-9]+: expected {_NUMBER} Hz, got {_NUMBER} Hz",
    r"STC stored sample count differs from ETC in segment [0-9]+: declared [0-9]+, indexed [0-9]+",
    r"ERD header differs in segment [0-9]+",
    r"STC segment [0-9]+ has inconsistent stamp bounds",
    r"STC segments overlap at entry [0-9]+",
    r"Missing ERD/ETC pair for STC segment [0-9]+",
    r"ETC packet [0-9]+ is outside STC segment [0-9]+",
    r"ETC (?:offsets are not increasing at|packets overlap at entry|sample numbers are discontinuous at entry) [0-9]+",
    r"Expected one main STC file, found [0-9]+",
    r"A recording file resolves outside its directory",
    r"Recording file resolves outside its directory",
    r"Ambiguous case-insensitive recording filename",
    r"Missing required recording file",
    r"The supplied ERD file is not a member of this STC recording",
    r"The Decimated derivative is outside the supported scope",
    r"ERD contains payload but its ETC index is empty",
    r"Unindexed bytes precede the first ERD packet",
)


def _safe_error(exc: Exception) -> str:
    """Only explicitly enumerated reader diagnostics may expose their text.

    Whole-message matching, numeric-only substitutions and a length bound
    prevent a path/event-bearing exception from matching an innocent prefix.
    Unknown exception classes and messages remain type-only.
    """
    category = type(exc).__name__
    try:
        from natus_erd import DataIntegrityError, ResourceLimitError, UnsupportedFormatError
    except ImportError:
        return category
    if type(exc) not in {DataIntegrityError, ResourceLimitError, UnsupportedFormatError}:
        return category
    message = str(exc)
    if len(message) <= 512 and any(re.fullmatch(pattern, message) for pattern in _SAFE_ERROR_PATTERNS):
        return f"{category}: {message}"
    return category


def _window_width(sample_rate: float, n_samples: int, n_channels: int) -> int:
    return min(ceil(sample_rate), WINDOW_SAMPLES, n_samples, MAX_OUTPUT_BYTES // (n_channels * 8))


class _Checks:
    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit = emit
        self.results: list[CheckResult] = []

    def note(self, **values: object) -> None:
        self.emit("  " + json.dumps(values, ensure_ascii=True))

    def step(self, name: str, action: Callable[[], object]):
        self.emit(f"[START] {name}")
        started = perf_counter()
        try:
            value = action()
        except _Skip as exc:
            status, detail, value = "SKIP", str(exc), None
        except Exception as exc:
            # Exception messages and tracebacks may disclose recording paths,
            # channel names or ENT content. Unknown messages stay type-only.
            status, detail, value = "FAIL", _safe_error(exc), None
        else:
            status, detail = "PASS", ""
        elapsed = perf_counter() - started
        self.results.append(CheckResult(name, status, elapsed, detail))
        self.emit(f"[{status}] {name} ({elapsed:.3f}s)" + (f" - {detail}" if detail else ""))
        if status == "FAIL":
            raise _Stop()
        return value


def validate_recording(
    recording: str | Path,
    *,
    preferred_channel: str | None = None,
    full_index_validation: bool = True,
    run_synthetic_tests: bool = False,
    emit: Callable[[str], None] = _print,
) -> tuple[CheckResult, ...]:
    """Target one-second windows, with <= 2048 points and <= 8 MiB per output.

    Stops after the first unexpected failure. Optional or data-dependent
    coverage is reported as SKIP, never silently counted as passing. No global
    thread settings are changed here; main() sets its own startup policy.
    """
    checks = _Checks(emit)
    try:
        _validate(checks, recording, preferred_channel, full_index_validation, run_synthetic_tests)
    except _Stop:
        emit("Stopped after failure; later checks were NOT run. No automatic retry.")
    counts = Counter(item.status for item in checks.results)
    emit(f"SUMMARY: PASS={counts['PASS']} SKIP={counts['SKIP']} FAIL={counts['FAIL']}")
    emit("Coverage: public API consistency + optional synthetic regressions; not clinical validation or a full payload scan.")
    return tuple(checks.results)


def _validate(checks, recording, preferred_channel, full_index_validation, run_synthetic_tests):
    numpy_was_loaded = "numpy" in sys.modules

    def package_check():
        package = importlib.import_module("natus_erd")
        for name in (
            "NatusERDReader", "ReadLimits", "RecordingInfo", "ChannelInfo", "Event",
            "ValidationReport", "NatusERDError", "DataIntegrityError",
            "ResourceLimitError", "UnsupportedFormatError",
        ):
            _require(hasattr(package, name), "Required public API is missing; install 0.2.1 or newer")
        installed_version = metadata.version("natus-erd-reader")
        _require(installed_version == package.__version__, "Distribution/module version mismatch")
        release = re.fullmatch(r"([0-9]+)\.([0-9]+)\.([0-9]+)", installed_version)
        _require(release is not None and tuple(map(int, release.groups())) >= (0, 2, 1), "Install version 0.2.1 or newer")
        for name in ("DataIntegrityError", "UnsupportedFormatError"):
            _require(issubclass(getattr(package, name), package.NatusERDError), "Exception hierarchy mismatch")
        _require(issubclass(package.ResourceLimitError, package.DataIntegrityError), "Resource error hierarchy mismatch")
        checks.note(python=sys.version.split()[0], package_version=installed_version)
        return package

    package = checks.step("Package version and public exports (requires 0.2.1+)", package_check)
    Reader, Limits = package.NatusERDReader, package.ReadLimits
    limits = Limits(max_read_bytes=MAX_OUTPUT_BYTES, max_read_samples=WINDOW_SAMPLES)
    reader = checks.step("Open recording with bounded reads", lambda: Reader.open(recording, limits=limits))
    info, channels = reader.info, reader.channels

    def metadata_check():
        _require(isinstance(info, package.RecordingInfo), "Wrong info type")
        _require(isinstance(channels, tuple), "Channels must be a tuple")
        _require(info.n_samples > 0 and info.sample_rate > 0, "Empty or invalid recording")
        _require(info.n_samples == info.end_stamp - info.start_stamp + 1, "Sample span mismatch")
        _require(info.duration_seconds == info.n_samples / info.sample_rate, "Duration mismatch")
        _require(len(channels) == info.n_recorded_channels, "Channel count mismatch")
        _require(sum(c.is_signal for c in channels) == info.n_signal_channels, "Signal count mismatch")
        _require(reader.limits == limits, "Reader did not retain its limits")
        for index, channel in enumerate(channels):
            _require(isinstance(channel, package.ChannelInfo) and channel.index == index, "Channel index mismatch")
            _require(channel.physical_index == index, "Unsupported physical map")
            _require(channel.is_signal == (index < info.n_signal_channels), "Signal layout mismatch")
            _require(isinstance(channel.name, str) and isinstance(channel.shorted, bool), "Channel metadata type mismatch")
            if not channel.is_signal:
                _require(channel.unit is None and channel.scale_uv_per_count is None, "Auxiliary calibration must be absent")
        _raises(FrozenInstanceError, lambda: setattr(limits, "max_read_bytes", 1))
        _raises(FrozenInstanceError, lambda: setattr(info, "n_samples", 0))
        checks.note(segments=info.segment_count, recorded_channels=len(channels), signal_channels=info.n_signal_channels,
                    shorted_channels=sum(c.shorted for c in channels), samples=info.n_samples,
                    sample_rate_hz=info.sample_rate, duration_hours=round(info.duration_seconds / 3600, 6))

    checks.step("Metadata, channel layout and immutable resource limits", metadata_check)

    def events_check():
        events = reader.read_events()
        _require(isinstance(events, tuple), "Events must be a tuple")
        _require(events == reader.read_events(), "Repeated event reads differ")
        previous = None
        outside = 0
        for event in events:
            _require(isinstance(event, package.Event), "Wrong event type")
            _require(isinstance(event.stamp, int) and isinstance(event.sample, int), "Invalid event position type")
            _require(isinstance(event.text, str), "Invalid event text type")
            _require(event.user is None or isinstance(event.user, str), "Invalid event user type")
            _require(event.note_type is None or isinstance(event.note_type, int), "Invalid event note type")
            _require(event.sample == event.stamp - info.start_stamp, "Event position mismatch")
            _require(previous is None or event.stamp >= previous, "Events are not sorted")
            previous = event.stamp
            if 0 <= event.sample <= info.n_samples:
                _require(reader.sample_to_stamp(event.sample) == event.stamp, "Event conversion mismatch")
            else:
                outside += 1  # Legitimate vendor notes can be outside the sample span.
        checks.note(event_count=len(events), events_outside_sample_span=outside)
        return len(events)

    event_count = checks.step("ENT event types, ordering and relative positions (no text output)", events_check)

    def time_check():
        for sample in {0, min(1, info.n_samples), info.n_samples // 2, info.n_samples - 1, info.n_samples}:
            stamp = reader.sample_to_stamp(sample)
            _require(stamp == info.start_stamp + sample, "Sample-to-stamp mismatch")
            _require(reader.stamp_to_sample(stamp) == sample, "Stamp round trip mismatch")
        _require(reader.sample_to_stamp(info.n_samples) == info.end_stamp + 1, "Exclusive-end mismatch")

    checks.step("Sample/stamp conversions including the exclusive end", time_check)

    def index_check():
        if not full_index_validation:
            raise _Skip("Full index traversal disabled")
        report = reader.validate(deep=True)
        _require(isinstance(report, package.ValidationReport), "Wrong report type")
        _require(report.segment_count == info.segment_count and report.logical_samples == info.n_samples, "Validation counts differ")
        _require(report.stored_samples + report.missing_samples == info.n_samples, "Coverage accounting differs")
        _require(report.event_count == event_count, "Validation event count differs")
        _require(report == reader.validate(deep=False), "Cached and refreshed validation differ")
        checks.note(packets=report.packet_count, stored_samples=report.stored_samples, missing_samples=report.missing_samples)

    checks.step("All segment indexes: validate(deep=True/False), NOT full waveform decoding", index_check)

    def lazy_check():
        if numpy_was_loaded:
            raise _Skip("NumPy was already imported; rerun in a fresh process to test lazy import")
        _require("numpy" not in sys.modules, "Metadata unexpectedly imported NumPy")

    checks.step("Metadata and events leave NumPy unloaded", lazy_check)
    np = checks.step("Load NumPy for bounded array checks", lambda: importlib.import_module("numpy"))
    checks.note(numpy_version=np.__version__, max_output_mib=MAX_OUTPUT_BYTES // 1024**2, max_window_samples=WINDOW_SAMPLES)
    width = _window_width(info.sample_rate, info.n_samples, len(channels))
    checks.note(window_samples=width, window_duration_seconds=width / info.sample_rate, target_duration_seconds=1)
    names = Counter(c.name for c in channels)
    signals = [c for c in channels if c.is_signal]

    def select_channel():
        _require(bool(signals), "No signal channels available")
        if preferred_channel is not None:
            matches = [c for c in signals if c.name == preferred_channel and names[c.name] == 1]
            _require(len(matches) == 1, "Requested channel is missing or ambiguous; edit PREFERRED_CHANNEL")
            return matches[0]
        return next((c for c in signals if not c.shorted), signals[0])

    chosen = checks.step("Resolve signal channel by index or explicit name (name kept private)", select_channel)
    signal_indices = [c.index for c in signals]
    selected = list(dict.fromkeys([chosen.index, signal_indices[0], signal_indices[-1]]))

    def array_check(data, rows, samples):
        _require(isinstance(data, np.ndarray), "Not an ndarray")
        _require(data.shape == (rows, samples) and data.dtype == np.dtype("float64"), "Array shape/dtype mismatch")
        _require(not np.isinf(data).any(), "Unexpected infinite value")

    def same(left, right):
        _require(np.array_equal(left, right, equal_nan=True), "Array consistency mismatch")

    def selector_check():
        reference = reader.read_samples(0, width, channels=[chosen.index], units="digital")
        array_check(reference, 1, width)
        for selector in (chosen.index, [chosen.index], (chosen.index,)):
            same(reader.read_samples(0, width, channels=selector, units="digital"), reference)
        other = next(c.index for c in channels if c.is_signal and c.index != chosen.index)
        ordered = reader.read_samples(0, width, channels=[other, chosen.index, other, chosen.index], units="digital")
        array_check(ordered, 4, width)
        same(ordered[1:2], reference)
        same(ordered[3:4], reference)
        same(ordered[0:1], reader.read_samples(0, width, channels=[other], units="digital"))
        same(ordered[0:1], ordered[2:3])

    checks.step("Scalar/list/tuple index selectors, ordering and duplicates", selector_check)

    def name_selector_check():
        named = chosen if names[chosen.name] == 1 else next((c for c in signals if names[c.name] == 1), None)
        if named is None:
            raise _Skip("No unique signal name; index-based checks remain valid")
        reference = reader.read_samples(0, width, [named.index], units="digital")
        for selector in (named.name, [named.name]):
            same(reader.read_samples(0, width, selector, units="digital"), reference)
        mixed = reader.read_samples(0, width, [named.name, named.index, named.name], units="digital")
        array_check(mixed, 3, width)
        for row in range(3):
            same(mixed[row:row + 1], reference)

    checks.step("Unique-name/index equivalence and mixed selectors", name_selector_check)

    def calibration_check():
        digital = reader.read_samples(0, width, selected, units="digital")
        uv = reader.read_samples(0, width, selected)
        array_check(digital, len(selected), width)
        array_check(uv, len(selected), width)
        factor = (-8711.0 / (2**21 - 0.5)) * 2**info.discard_bits
        for index in selected:
            _require(channels[index].unit == "uV" and channels[index].scale_uv_per_count == factor, "Calibration metadata mismatch")
        expected = digital * factor
        _require(np.allclose(uv, expected, rtol=1e-12, atol=1e-12, equal_nan=True), "Calibration differs")
        finite = np.isfinite(uv)
        maximum = float(np.max(np.abs(uv[finite] - expected[finite]))) if finite.any() else None
        checks.note(calibration_consistency_max_abs_uv=maximum, independent_edf_comparison=False)

    checks.step("uV/digital conversion and matching NaN masks", calibration_check)

    def default_check():
        data = reader.read_samples(0, width)
        array_check(data, info.n_signal_channels, width)
        same(data[selected], reader.read_samples(0, width, selected))
        shorted = [c.index for c in channels if c.is_signal and c.shorted]
        if shorted:
            _require(np.isnan(data[shorted]).all(), "Shorted signal channels must be NaN")
        checks.note(default_output_mib=round(data.nbytes / 1024**2, 4))

    checks.step("Default signal channels and shorted-channel NaNs in uV", default_check)

    def recorded_check():
        data = reader.read_samples(0, width, list(range(len(channels))), units="digital")
        array_check(data, len(channels), width)
        values = data[np.isfinite(data)]
        _require(np.equal(values, np.trunc(values)).all(), "Digital counts must be integral")
        shorted = [c.index for c in channels if c.shorted]
        if shorted:
            _require(np.isnan(data[shorted]).all(), "Shorted digital channels must be NaN")
        checks.note(finite_digital_values=int(values.size), observed_values_outside_int16=int(np.count_nonzero((values < -32768) | (values > 32767))))

    checks.step("All recorded channels including auxiliary digital data", recorded_check)

    def auxiliary_check():
        auxiliary = [c.index for c in channels if not c.is_signal]
        if not auxiliary:
            raise _Skip("No auxiliary channels in this recording")
        _raises(package.UnsupportedFormatError, lambda: reader.read_samples(0, 1, auxiliary, units="uV"))
        _raises(package.UnsupportedFormatError, lambda: next(reader.iter_samples(0, 1, channels=auxiliary, units="uV")))

    checks.step("Auxiliary uV conversion is explicitly rejected", auxiliary_check)

    def empty_check():
        array_check(reader.read_samples(0, 0, selected), len(selected), 0)
        array_check(reader.read_samples(info.n_samples, info.n_samples, chosen.index), 1, 0)
        array_check(reader.read_samples(0, width, []), 0, width)
        array_check(reader.read_samples(0, 0, []), 0, 0)
        _require(next(reader.iter_samples(0, 0), None) is None, "Empty iterator yielded data")

    checks.step("Empty windows, empty selectors and empty iterator", empty_check)

    def windows_check():
        starts = sorted({((info.n_samples - width) * index) // 4 for index in range(5)})
        for index, start in enumerate(starts, 1):
            stop = start + width
            checks.note(window=index, relative_start_sample=start, samples=width, duration_seconds=width / info.sample_rate)
            whole = reader.read_samples(start, stop, selected, units="digital")
            array_check(whole, len(selected), width)
            split = start + width // 2
            same(whole[:, :split - start], reader.read_samples(start, split, selected, units="digital"))
            same(whole[:, split - start:], reader.read_samples(split, stop, selected, units="digital"))
            same(whole, reader.read_samples(start, stop, selected, units="digital"))
            del whole

    checks.step("Beginning/quarter/middle/three-quarter/end windows and repeatability", windows_check)

    def chunks_check():
        for units in ("uV", "digital"):
            reference = reader.read_samples(0, width, selected, units=units)
            offset = 0
            chunk_size = max(1, min(257, width // 3))
            previous = None
            for chunk in reader.iter_samples(0, width, chunk_samples=chunk_size, channels=selected, units=units):
                expected_width = min(chunk_size, width - offset)
                array_check(chunk, len(selected), expected_width)
                same(chunk, reference[:, offset:offset + expected_width])
                if previous is not None:
                    _require(not np.shares_memory(previous, chunk), "Iterator chunks share storage")
                previous = chunk  # At most two tiny chunks; never collect a recording.
                offset += expected_width
            _require(offset == width, "Iterator coverage differs")
        # Exercise stop=None/default chunk size only over the short final window.
        iterator = reader.iter_samples(info.n_samples - width, channels=selected)
        last = next(iterator)
        same(last, reader.read_samples(info.n_samples - width, info.n_samples, selected))
        _require(next(iterator, None) is None, "Tail iterator exceeded the recording")

    checks.step("Streaming uV/digital chunks, remainder, independence and default stop", chunks_check)

    def entry_check():
        # This example shares the package's bounded, case-insensitive discovery
        # rules. The helper is internal and not an additional public API.
        from natus_erd._paths import RecordingDirectory

        source = Path(recording).expanduser().resolve(strict=True)
        directory = source if source.is_dir() else source.parent
        files = RecordingDirectory(directory, limits=limits)
        candidates = files.stc_paths()
        stc = files.resolve_stc(source)
        paths = [("stc", stc)]
        if len(candidates) == 1:
            paths.insert(0, ("directory", directory))
        else:
            checks.step("Optional directory entry point", lambda: (_ for _ in ()).throw(_Skip("Multiple recordings; use an explicit STC or matching EEG")))
        for suffix in (".eeg", ".erd"):
            if suffix == ".erd" and len(candidates) != 1:
                checks.step("Optional .erd entry point", lambda: (_ for _ in ()).throw(_Skip("Multiple recordings; ERD alone cannot select a unique STC")))
                continue
            sibling = files.lookup(stc.stem + suffix, optional=True)
            if source.is_file() and source.suffix.casefold() == suffix:
                sibling = source
            if sibling is not None:
                paths.append((suffix[1:], sibling))
            else:
                checks.step(f"Optional {suffix} entry point", lambda: (_ for _ in ()).throw(_Skip("Matching entry file absent; no alternate recording guessed")))
        reference = reader.read_samples(0, min(16, width), chosen.index, units="digital")
        for label, path in paths:
            alternate = Reader.open(path, limits=limits)
            _require(alternate.info == info and alternate.channels == channels, "Entry point metadata differs")
            _require(alternate.read_events() == reader.read_events(), "Entry point events differ")
            same(alternate.read_samples(0, min(16, width), chosen.index, units="digital"), reference)
            checks.note(entry_type=label, consistent=True)
            del alternate

    checks.step("Directory/STC/EEG/ERD entry-point equivalence", entry_check)

    def invalid_check():
        unknown = "__NONEXISTENT_VALIDATION_CHANNEL__"
        while unknown in names:
            unknown += "_"
        cases = [
            (IndexError, lambda: reader.read_samples(-1, 0)),
            (IndexError, lambda: reader.read_samples(1, 0)),
            (IndexError, lambda: reader.read_samples(0, info.n_samples + 1)),
            (TypeError, lambda: reader.read_samples(True, 1)),
            (TypeError, lambda: reader.read_samples(0.5, 1)),
            (ValueError, lambda: reader.read_samples(0, 1, units="V")),
            (IndexError, lambda: reader.read_samples(0, 1, [-1])),
            (IndexError, lambda: reader.read_samples(0, 1, [len(channels)])),
            (KeyError, lambda: reader.read_samples(0, 1, [unknown])),
            (TypeError, lambda: reader.read_samples(0, 1, [True])),
            (TypeError, lambda: reader.read_samples(0, 1, [0.5])),
            (IndexError, lambda: reader.sample_to_stamp(-1)),
            (IndexError, lambda: reader.sample_to_stamp(info.n_samples + 1)),
            (IndexError, lambda: reader.stamp_to_sample(info.start_stamp - 1)),
            (IndexError, lambda: reader.stamp_to_sample(info.end_stamp + 2)),
            (TypeError, lambda: reader.sample_to_stamp(True)),
            (TypeError, lambda: reader.stamp_to_sample(0.5)),
            (ValueError, lambda: next(reader.iter_samples(0, 1, chunk_samples=0))),
            (ValueError, lambda: next(reader.iter_samples(0, 1, chunk_samples=-1))),
            (TypeError, lambda: next(reader.iter_samples(0, 1, chunk_samples=True))),
            (ValueError, lambda: next(reader.iter_samples(0, 1, units="V"))),
            (IndexError, lambda: next(reader.iter_samples(-1, 1))),
        ]
        for expected, action in cases:
            _raises(expected, action)
        for value in (0, -1, True, 1.5):
            _raises(ValueError, lambda value=value: Limits(max_read_bytes=value))
        _raises(ValueError, lambda: Limits(max_parse_depth=129))
        _raises(TypeError, lambda: Reader.open(recording, limits={}))
        checks.note(invalid_api_cases=len(cases), invalid_limit_cases=6)

    checks.step("Invalid ranges, selectors, units, iterator arguments and limits", invalid_check)

    def budget_check():
        # Never attempt a huge allocation. Eight bytes already exceeds a 1-byte
        # configured output budget, including for the iterator's first chunk.
        tiny = Reader.open(recording, limits=replace(limits, max_read_bytes=1))
        _raises(package.ResourceLimitError, lambda: tiny.read_samples(0, 1, [0]))
        _raises(package.ResourceLimitError, lambda: next(tiny.iter_samples(0, 1, channels=[0])))
        del tiny
        tiny = Reader.open(recording, limits=replace(limits, max_selected_channels=1))
        _raises(package.ResourceLimitError, lambda: tiny.read_samples(0, 0, [0, 1]))
        del tiny
        if info.n_samples >= 2:
            tiny = Reader.open(recording, limits=replace(limits, max_read_samples=1))
            _raises(package.ResourceLimitError, lambda: tiny.read_samples(0, 2, [0]))
            _raises(package.ResourceLimitError, lambda: next(tiny.iter_samples(0, 2, chunk_samples=2, channels=[0])))
        else:
            checks.step("Sample-count budget rejection", lambda: (_ for _ in ()).throw(_Skip("Recording has fewer than two samples")))

    checks.step("Output/channel/sample budgets reject tiny over-budget requests", budget_check)

    def surface_check():
        for name in ("cli", "__main__", "edf", "viewer"):
            _require(util.find_spec(f"natus_erd.{name}") is None, "Removed application module is still installed")
        dist = metadata.distribution("natus-erd-reader")
        _require(not any(ep.group in {"console_scripts", "gui_scripts"} for ep in dist.entry_points), "Installed application entry point remains")

    checks.step("Python-only installation: no CLI/EDF/viewer entry points", surface_check)

    def synthetic_check():
        if not run_synthetic_tests:
            raise _Skip("Synthetic regression suite disabled")
        root = Path(__file__).resolve().parents[1]
        if not (root / "tests" / "_fixture.py").is_file() or not (root / "pyproject.toml").is_file():
            raise _Skip("Repository tests unavailable; a wheel alone does not include tests")
        # Use only this checkout's explicitly located tests. Discovery performs
        # no clinical-file access; corruption/security inputs are synthetic.
        previous_path = sys.path[:]
        try:
            loader = unittest.TestLoader()
            suite = loader.discover(str(root / "tests"), pattern="test_*.py", top_level_dir=str(root))
            _require(not loader.errors, "Synthetic test discovery failed")
            result = unittest.TestResult()
            suite.run(result)
        finally:
            sys.path[:] = previous_path
        checks.note(synthetic_tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
        _require(result.testsRun > 0 and result.wasSuccessful(), "Synthetic regression suite failed")
        if result.skipped:
            checks.step("Optional synthetic test cases", lambda: (_ for _ in ()).throw(_Skip(f"{len(result.skipped)} cases skipped by unittest; not fully exercised")))

    checks.step("Synthetic decoder/boundary/gap/corruption/ENT/path-security regressions", synthetic_check)


def main(recording: str | Path = RECORDING_PATH, *, preferred_channel: str | None = PREFERRED_CHANNEL) -> int:
    _print("Bounded Natus ERD validation. No raw arrays, patient fields or paths will be printed.")
    if "numpy" in sys.modules:
        _print("STOP: NumPy is already loaded. Run this script in a fresh Python process, not an existing notebook console.")
        return 2
    # Explicit policy for THIS diagnostic application, before any NumPy import.
    # This neither changes Conda configuration nor guarantees against OS faults.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    _print("Diagnostic process policy: one numerical-backend thread; <= 8 MiB per array; <= 2048 samples per window.")
    _print("These are reader/application budgets, NOT a process-wide memory cap or a remedy for kernel crashes.")
    results = validate_recording(
        recording, preferred_channel=preferred_channel,
        full_index_validation=RUN_FULL_INDEX_VALIDATION,
        run_synthetic_tests=RUN_SYNTHETIC_TESTS,
    )
    return 1 if any(item.status == "FAIL" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
