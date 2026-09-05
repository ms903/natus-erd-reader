# natus-erd-reader

[中文文档](https://natus-reader.github.io/) · [English documentation](https://natus-reader.github.io/en/)

Read Natus NeuroWorks ERD signals, ENT events and SNC clocks in Python, and
export continuous windows as standard EDF+C for other analysis tools.
The source version is **0.3.0rc1**, a development candidate.

## Install

Python 3.10+ and NumPy 1.24+ are required. Install this checkout with:

```sh
python -m pip install .
```

The optional C extension accelerates EDF export. A single-worker Python
implementation is available on other platforms. For named timezones on systems
without an IANA database, install `.[timezones]`.

## Read a window

```python
from natus_erd import NatusERDReader

reader = NatusERDReader.open("path/to/recording")
data = reader.read_samples(0, min(2048, reader.info.n_samples), channels=[0, 1, 2])
print(reader.info.sample_rate, data.shape)
events = reader.read_events()
```

Arrays have shape `(channels, samples)` and float64 values. Signals default to
microvolts; `units="digital"` returns decoded counts. Gaps and shorted channels
are NaN. Use `iter_samples()` for longer intervals.

## Export EDF+C

```python
from natus_erd import plan_edf, export_edf

print(tuple(reader.iter_stored_ranges()))
start, stop = 0, 2048  # Choose an exact, fully stored interval from this recording.
plan = plan_edf(reader, start=start, stop=stop)
result = export_edf(reader, "window.edf", start=start, stop=stop)
```

Export defaults to all recorded channels, drops shorted channels, preserves
auxiliary integer values and limits EEG quantization error to 0.5 uV. It retains
parsed events from the entire recording. See the [EDF export guide](https://natus-reader.github.io/en/edf-export/)
for exact record alignment, clocks, annotations and third-party reading.

## Supported recordings

The reader supports schema-9/base-1 ERD, 8-bit delta encoding, Quantum headbox 20
and the 276-channel layout (256 signal channels plus 20 auxiliary channels).
The sampling rate comes from the header. Input recordings must remain unchanged
while a reader or export is using them. See [compatibility](https://natus-reader.github.io/en/compatibility/).

This is independent research software, not a certified diagnostic device or an
official Natus product. Development and release instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).
