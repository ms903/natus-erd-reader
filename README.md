# natus-erd-reader

[![Tests and package build](https://github.com/ms903/natus-erd-reader/actions/workflows/ci.yml/badge.svg)](https://github.com/ms903/natus-erd-reader/actions/workflows/ci.yml)

`natus-erd-reader` is a small, lazy Python reader for a specific Natus
NeuroWorks ERD layout: file schema 9, base schema 1, and Quantum headbox type
20. It reads native `.stc`, `.etc`, `.erd`, and `.ent` files without loading a
complete segment or recording into memory. Python 3.10+ and NumPy are the only
core requirements; no Wonambi, MNE or vendor SDK is required at runtime.

This is an **experimental research tool**, not a certified diagnostic device.
It is not affiliated with or endorsed by Natus. Other schema/headbox layouts
are explicitly rejected rather than interpreted with unverified parameters.

The first 256 channels are exposed as calibrated sEEG signals. The remaining
20 recorded channels remain visible in metadata and can be requested as
digital counts, but are deliberately not assigned an unverified physical-unit
conversion.

## Installation

Install the tagged GitHub release:

```shell
python -m pip install "git+https://github.com/ms903/natus-erd-reader.git@v0.1.0"
```

Alternatively, download a wheel from [GitHub Releases](https://github.com/ms903/natus-erd-reader/releases)
and install it with `python -m pip install path/to/package.whl`.
This project is not yet published on PyPI; the distribution name and Python
import name are `natus-erd-reader` and `natus_erd`, respectively.

For development or the optional comparison/plotting scripts:

```shell
git clone https://github.com/ms903/natus-erd-reader.git
cd natus-erd-reader
python -m pip install -e ".[dev,plot,validation]"
```

## Python API

```python
from natus_erd import NatusERDReader

reader = NatusERDReader.open(r"D:\path\to\recording")
print(reader.info)

# Shape: (3, 2048), unit: microvolts
data = reader.read_samples(0, 2048, channels=["A1", "A2", 10])

# Native, unclipped ERD digital values
digital = reader.read_samples(0, 2048, channels=[0, 1], units="digital")

events = reader.read_events()
report = reader.validate()
```

`start` is inclusive, `stop` is exclusive, and both are relative to the first
STC stamp. Known gaps are returned as `NaN`; shorted channels are always
`NaN`.

## Command line

```text
natus-erd info PATH
natus-erd validate PATH
natus-erd sample PATH --start 10 --duration 2 --channels 0,1,2
```

For `sample`, start and duration are seconds. Add `--output window.npy` to save
the resulting NumPy array. The default CLI output contains structural counts,
not patient header fields or full event text.

## Local ERD / EDF browser

Start the synchronized local waveform page with:

```text
natus-erd-viewer data
```

The viewer finds the single EDF below or next to the recording automatically.
When more than one EDF exists, select it explicitly:

```text
natus-erd-viewer RECORDING --edf EXPORTED.edf
```

The page binds to `127.0.0.1:8765` by default. It reads only the selected time
window, supports up to eight of the first 256 calibrated channels, overlays or
splits ERD and EDF traces, and adds a dedicated zero-centred `ERD - EDF`
difference curve in µV. It also shows ENT events, in-range LSB/µV differences,
and ERD values that were clipped by the EDF export. No recording data is
uploaded or copied.

## Supported scope

- ERD file schema 9 and base schema 1
- Quantum headbox type 20
- 2048 Hz, 276 recorded channels, 8-bit delta base, 6 discarded bits
- Native UTF-8 segment and montage names
- ENT event and montage extraction using `ast.literal_eval`, never `eval`

Other schemas and headboxes raise `UnsupportedFormatError` instead of silently
using a possibly incorrect conversion.

The `.eeg` path is accepted as a recording entry point; samples are obtained
through its `.stc`/`.etc` indexes and `.erd` payloads. `Decimated`, video,
BDF, EDF+D time discontinuities, and vendor-specific auxiliary-channel
calibration are outside the verified scope. The lightweight EDF helper is
intended for continuous, equally sampled 16-bit EDF exports from the same
recording. Comparison aligns by sample index and channel index, not by an
automatic time-offset search.

## Static error report

From a source checkout, the optional plotting tool compares uniformly spaced
windows across the common ERD/EDF duration:

```shell
python tools/plot_error_summary.py RECORDING --edf EXPORTED.edf --output-dir reports
```

It produces a PNG, a vector PDF, JSON statistics and a per-window CSV. The
default is **48 windows of 2 seconds**, not an exhaustive whole-recording
scan. Moments and extrema use all valid samples in those windows; quantiles
use a deterministic stride-16 subsample. Normal int16-range errors and EDF
clipping distortion are reported separately. Outputs remain local and are
ignored by Git. Install the `plot` extra to use this tool.

Only synthetic tests and generic code are distributed. No clinical recording,
patient identifier, real event text, exported waveform or derived patient
plot is included in this repository or its package archives.

## Verification

The synthetic unit suite uses only `unittest` and NumPy:

```text
python -m unittest discover -v -s tests -t .
```

For a one-off comparison with an EDF exported from the same recording, install
PyEDFlib separately and run:

```text
python tools/compare_edf.py RECORDING EDF_FILE
```

The comparison checks five distributed windows by default, verifies the
in-range digital and physical errors, and confirms that native ERD values
outside the EDF int16 range are clipped at the corresponding EDF limits.

The local viewer's EDF helper and the optional PyEDFlib validation tool are
separate implementations. Use independent validation on representative data
before relying on a new recording or hardware configuration. See
[CONTRIBUTING.md](CONTRIBUTING.md) for building and auditing distributions.

## 中文速览

这是面向 Natus NeuroWorks schema 9 / Quantum headbox 20 的轻量懒加载读取器。
核心运行时只依赖 NumPy，支持中文路径、按采样窗口读取、ENT 事件、原始数字量及
前 256 个信号通道的 µV 换算。缺失区间与 shorted 通道返回 `NaN`，不会伪造为零。
仓库不包含真实 EEG 数据。当前版本用于研究验证，不应视为经过认证的临床软件。

## License and acknowledgements

Original project code is [MIT licensed](LICENSE). ENT legacy-text
normalization is adapted from Wonambi under BSD-3-Clause; its full notice is
preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and both package
archives. Consequently the distribution's SPDX expression is
`MIT AND BSD-3-Clause`. There is no Wonambi runtime dependency.
