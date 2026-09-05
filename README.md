# natus-erd-reader

[中文文档](https://natus-reader.github.io/) · [English documentation](https://natus-reader.github.io/en/)

Read Natus NeuroWorks ERD signals, ENT events and SNC clocks in Python, and
export stored signals as standard EDF+C or EDF+D for other analysis tools.
The source version is **0.3.0rc3**, a development candidate.

## Install

Use Python 3.10+. NumPy and tqdm are installed automatically:

```sh
python -m pip install "natus-erd-reader==0.3.0rc3"
```

The optional C extension accelerates EDF export. A single-worker Python
implementation is available on other platforms. For named timezones on systems
without an IANA database, install `natus-erd-reader[timezones]==0.3.0rc3`.
The default Beijing offset does not require this extra.

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

## Export EDF+C or EDF+D

```python
from natus_erd import NatusERDReader, export_edf

reader = NatusERDReader.open(r"D:\data\recording")
result = export_edf(reader, r"D:\output.edf")
```

Defaults export all 276 channels with fixed Beijing UTC+08:00 and separate
scan, write and verification progress bars. Set `progress=False` to hide them.
Shorted channels retain their names and digital 32767. EEG quantization error
is bounded by 0.5 uV; auxiliaries use the documented official calibration and
units. Continuous stored data produces EDF+C; gaps produce EDF+D with original
record onsets. Every stored interval must fit one exact common record grid;
no samples are padded or discarded. Select `start`, `stop` or `channels` when needed.

`result.edf_format` identifies the output. `stored_samples` counts genuine samples
per channel, while `logical_samples` is the requested span including gaps.
`stored_seconds`, `time_span_seconds` and `gap_seconds` distinguish valid data
duration from elapsed recording time; `elapsed_seconds` measures conversion work.
Use an EDF+D-aware reader for discontinuous files. The MNE example requires EDF+C.

Parsed events from the entire recording are retained. Results include output
labels, units and separate `scan_seconds`, `write_seconds`, `verify_seconds`
and total `elapsed_seconds`. See the [EDF export guide](https://natus-reader.github.io/en/edf-export/)
for auxiliary value support, window selection, clocks and third-party reading.

## Supported recordings

The reader supports schema-9/base-1 ERD, 8-bit delta encoding, Quantum headbox 20
and the 276-channel layout (256 signal channels plus 20 auxiliary channels).
The sampling rate comes from the header. Input recordings must remain unchanged
while a reader or export is using them. See [compatibility](https://natus-reader.github.io/en/compatibility/).

This is independent research software, not a certified diagnostic device or an
official Natus product. Development and release instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).
