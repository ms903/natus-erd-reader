# natus-erd-reader

[![Tests and package build](https://github.com/ms903/natus-erd-reader/actions/workflows/ci.yml/badge.svg)](https://github.com/ms903/natus-erd-reader/actions/workflows/ci.yml)

A small Python library for native Natus NeuroWorks ERD recordings. It reads
sample windows through the STC/ETC indexes, decodes schema-9 packets and
extracts ENT annotations. It has **no command-line application, web server,
EDF reader, plotting tool or vendor SDK dependency**.

Version 0.2.1 adds header-driven sample rates and compatibility fixes while
preserving bounded file reads and allocations. Python 3.10+ is
required; NumPy is the only runtime dependency and is imported only when a
valid sample-read request needs an array. Opening a recording and inspecting
metadata or events do not initialize NumPy.

This is **experimental research software**, not a certified diagnostic device.
It is not affiliated with or endorsed by Natus. The supported hardware/layout
is deliberately narrow, and no guarantee is made against operating-system,
driver, hardware or numerical-library failures.

## Installation

With Python 3.10 or newer, install **0.2.1** directly from GitHub Releases
(no Git installation or manual download required):

```shell
python -m pip install --upgrade "https://github.com/ms903/natus-erd-reader/releases/download/v0.2.1/natus_erd_reader-0.2.1-py3-none-any.whl"
```

Run this command in the Python environment where you want to use the reader.
NumPy is installed automatically if needed. To preserve an existing,
compatible NumPy installation, add `--no-deps`.

Once the version is available on [PyPI](https://pypi.org/project/natus-erd-reader/),
it can also be installed by name:

```shell
python -m pip install --upgrade "natus-erd-reader==0.2.1"
```

The GitHub Release command above does not depend on PyPI availability.

Check the installed version without loading NumPy or opening a recording:

```shell
python -c "from importlib.metadata import version; print(version('natus-erd-reader'))"
```

For development source from the current `main` branch, or from a local checkout:

```shell
python -m pip install --upgrade "https://github.com/ms903/natus-erd-reader/archive/refs/heads/main.zip"
python -m pip install .
```

Tagged releases are listed in
[GitHub Releases](https://github.com/ms903/natus-erd-reader/releases).
The distribution name is
`natus-erd-reader`; the import name is `natus_erd`.

## Read a small window

```python
from math import ceil

from natus_erd import NatusERDReader

# Pass the recording directory itself, or its .stc, .eeg or .erd file.
reader = NatusERDReader.open(r"D:\path\to\recording")
print(reader.info)  # Structural counts, not patient header fields.

# Start is inclusive; stop is exclusive. Both are sample numbers.
# Target one second, using the actual ERD header rate and recording length.
stop = min(reader.info.n_samples, ceil(reader.info.sample_rate))
data = reader.read_samples(0, stop, channels=[0, 1])
print(data.shape, data.dtype)  # (2, stop), float64

digital = reader.read_samples(0, stop, channels=[0, 1], units="digital")
events = reader.read_events()
```

Arrays have shape `(channels, samples)`, always with `float64` dtype. The
default unit is microvolts (`units="uV"`); `"digital"` returns decoded native
counts without clipping to int16. Channel selection accepts zero-based
indices or names and preserves the requested order. Omitting `channels`
selects the first 256 signal channels. Shorted channels and known gaps are
returned as `NaN`, not zero.
STC stored-sample counts can be smaller than the stamp-defined time span;
the ETC packet stamps determine where samples are present. Leading, internal
and trailing gaps remain on the original time axis: they are not removed,
interpolated or compressed. Reads crossing a gap retain it as `NaN`.

Sample rates are read from the ERD header, not assumed to be 2048 Hz. For
example, 2048 samples cover four seconds at 512 Hz and one second at 2048 Hz.
The reader does not resample. A one-second request at an unusually high rate
can exceed the unchanged resource limits; use a shorter window or chunks.

Sample zero corresponds to the first STC stamp. `sample_to_stamp()` and
`stamp_to_sample()` convert between relative sample positions and native
stamps. `reader.info`, `reader.channels`, `reader.read_events()` and
`reader.validate()` expose structural metadata, channel descriptions, events
and a structural validation report, respectively. Validation is not a
complete decoding or clinical validation of every sample.

The reader does not recursively search arbitrary parent directories. Pass
the actual recording directory and keep its index and segment files together.
Filenames are matched case-insensitively, including their extensions, but
ambiguous case-folded duplicates are rejected. In a directory containing
multiple recordings, pass an explicit `.stc` or its matching `.eeg`; a
directory or `.erd` alone cannot select a unique recording. An ENT file is
optional; `.ent.old` is used only if `.ent` is absent, never to conceal a
corrupt `.ent`. Missing channel labels receive positional fallback names;
ambiguous vendor labels must be selected by index.
Only completed, static recordings are supported: do not modify the files or
continue acquisition into them while a reader is open. Size, timestamp and
file-identity checks detect some concurrent changes, but detection is
best-effort and does not provide a consistent snapshot of changing files.

## Keep larger reads bounded

The default output budget is **64 MiB per `read_samples` call**. A request
that would exceed it raises `ResourceLimitError` before allocating the output
or importing NumPy. An output needs `selected_channels × samples × 8` bytes.
Thus a one-second, two-channel read at 2048 Hz needs 32 KiB for its output;
reading an entire long recording at once is intentionally rejected.
There is also a default limit of **131,072 samples per call** (64 seconds at
2048 Hz), even when only one channel is selected, to bound decoding work.

Process long recordings incrementally:

```python
for chunk in reader.iter_samples(
    start=0,
    stop=reader.info.n_samples,
    chunk_samples=20480,
    channels=[0, 1],
    units="uV",
):
    # Analyze or write this chunk, then let it go before requesting the next.
    print(chunk.shape)
```

Each chunk is checked against the same read budget. **Do not collect all
chunks into a list** unless you deliberately want to retain that memory.
The caller owns arrays returned by the reader; repeated reads retained by
the caller can still consume arbitrary memory.

For a stricter output policy:

```python
from natus_erd import NatusERDReader, ReadLimits

reader = NatusERDReader.open(
    r"D:\path\to\recording",
    limits=ReadLimits(max_read_bytes=16 * 1024 * 1024),
)
```

Metadata lengths, parser complexity and packet sizes are also checked before
large reads. Packet decoding uses a bounded input buffer rather than reading
the entire compressed packet into memory. Invalid offsets, truncated data,
overlap and unsupported layouts raise explicit errors. These checks limit
the reader's own resource requests; **they are not a process-wide memory
limit or a sandbox**.

Other defaults include 8 MiB per metadata file or compressed packet, 256 KiB
per ENT record, 4,096 ENT records, parser depth 32 and a cache of four segment
indexes. `ReadLimits` exposes these policies explicitly; increasing a limit
should follow an assessment of the input and available resources.

### NumPy backend resource use

Some NumPy builds initialize a multi-threaded numerical backend with a large
memory footprint, even though this reader does not need matrix multiplication.
The package does not modify process-wide thread settings or environment
variables. On a memory-constrained system, an application may explicitly set
its own backend policy **before importing NumPy or any library that imports
it**, for example at the very start of the application:

```python
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # For an OpenBLAS-backed NumPy build.

from natus_erd import NatusERDReader
```

This setting is backend-specific and is not a remedy or guarantee for kernel
crashes. If a machine has frozen or restarted, do not retry an unbounded
workload; investigate the operating-system failure separately.

## Supported format and errors

The supported layout is ERD schema 9, base schema 1, Quantum headbox type 20:
276 recorded channels, 8-bit delta base and 6 discarded bits. Any finite
positive header sample rate is accepted if its recording duration is finite;
there is no sample-rate whitelist. All segments must have the same rate.
This does not extend support to other hardware or per-channel sampling rates.
STC schema 1, ETC schema 3 and ENT schema 3 are supported, each with base
schema 1. The ERD layout must contain a single headbox, an identity physical
channel mapping and a frequency-factor value of 32767 for every channel.
The first 256 channels use the Quantum AC calibration:

```text
uV = digital × (-8711 / (2**21 - 0.5)) × 2**discard_bits
```

The last 20 auxiliary channels remain in `reader.channels` and can be read
as digital counts, but have no verified physical-unit conversion. Native
UTF-8 segment and montage names are supported. ENT is parsed as data with a
restricted, bounded grammar; it is never evaluated as Python code.
Notes with unsupported or malformed text syntax may be skipped; a restricted
fallback extracts channel names from recognized nonstandard montage fields.
Consequently, `read_events()` is not guaranteed to include every annotation
stored by the vendor. Invalid binary lengths, schemas and exceeded parser
budgets still raise errors rather than being silently skipped.

`.eeg` is an accepted entry point, not the sample-data source. Samples come
from `.stc`/`.etc` indexes and `.erd` payloads. Other schemas/headboxes,
`Decimated`, video, EDF and additional auxiliary calibration are out of scope.

- `UnsupportedFormatError`: unsupported format or hardware layout.
- `DataIntegrityError`: malformed, truncated or inconsistent input.
- `ResourceLimitError`: a configured resource limit would be exceeded.
- `NatusERDError`: common base class for these reader exceptions.

Ordinary invalid API arguments may raise `TypeError`, `ValueError` or
`IndexError`. Filesystem failures may propagate the corresponding `OSError`.

## Privacy, testing and maintenance

The library does not upload recordings or contact network services. Channel
labels and ENT event text can contain sensitive information; avoid printing
or sharing them without reviewing them. Only generic source and synthetic
tests belong in this repository. Real recordings, event text and derived
patient outputs must not be distributed.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the test/build workflow and
[SECURITY.md](SECURITY.md) for resource-limit boundaries. Public API changes
from 0.1.0 are listed in [CHANGELOG.md](CHANGELOG.md).

## 中文速览

这是只提供 Python API 的 Natus ERD 读取包，不再包含命令行、网页、EDF 或绘图功能。
支持的范围限定为 schema 9 / Quantum headbox 20。默认单次输出上限为 64 MiB；
采样率从 ERD 文件头读取，不再固定为 2048 Hz；采样空档保留原时间轴并返回 NaN。
长记录请使用 `iter_samples()` 分块处理，不要把所有块保留在内存中。
读取器会检查文件长度、索引和解析资源上限，但不能保证避免操作系统或硬件故障。
仅查看记录信息和事件不会初始化 NumPy；真正读取数组时才加载它。

## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for applicable terms and retained attribution. Both are included in source
and wheel distributions.
