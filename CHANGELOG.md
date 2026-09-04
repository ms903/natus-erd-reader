# Changelog

## 0.2.1 — 2026-09-04

Compatibility fixes within the existing schema-9 Quantum layout; no new
application entry points, hardware calibration or resampling behavior.

- Read the finite positive sample rate from ERD headers instead of requiring
  2048 Hz. Reject non-finite recording duration and inconsistent segment rates.
- Interpret STC stored counts separately from stamp spans, verify ETC counts,
  and preserve leading, internal, trailing and fully empty gaps as NaN.
- Resolve complete recording filenames case-insensitively with ambiguity and
  directory-boundary checks. Explicit STC or matching EEG can select among
  multiple recordings; an unrelated ordinary `.stc` directory is ignored.
- Keep positional channel-label placeholders, avoid fallback-name collisions
  and require index selection for genuinely duplicated vendor labels.
- Add bounded public-API validation examples without a fixed channel label;
  adapt sample windows to the actual rate and redact unknown error details.
- Extend synthetic compatibility coverage, add Python 3.13 CI, and test the
  minimum NumPy 1.24.x dependency on Python 3.10 for Windows and Linux.
- Preserve all existing read/parser budgets and Python-only runtime surface.

## 0.2.0 — source-only update

Python-library-only release with explicit resource limits. This is a breaking
change from 0.1.0, and remains experimental research software.

- Remove both installed command-line programs and `python -m natus_erd`.
- Remove EDF APIs, comparison tools, plotting extras and the local web viewer.
- Keep native ERD window reads, structural metadata, ENT events and validation.
- Add `ReadLimits` and `ResourceLimitError`; default output budget is 64 MiB
  per read, with at most 131,072 samples per call. Reject oversized requests
  before allocation and NumPy import.
- Add `iter_samples()` for chunked processing without retaining a full record.
- Bound metadata reads and ENT parsing; validate compressed-packet lengths and
  decode through a bounded buffer instead of loading a whole packet.
- Parse ENT using an independent restricted grammar; never execute input.
- Load NumPy only when an accepted sample request needs an array. Do not
  modify global numerical-backend thread settings.
- Require an explicit recording directory or entry-point file rather than
  recursively searching broad parent directories.
- Extend synthetic resource-boundary tests and distribution checks. No
  operating-system crash-prevention or clinical-safety claim is made.

### Migration from 0.1.0

Use `NatusERDReader` directly from Python; there is no replacement executable.
`EDFInfo`, `EDFReader` and `EDFSignal` are no longer exported or included.
Remove optional `plot`/`validation` extras from dependency declarations.
Split large `read_samples` calls with `iter_samples`, or choose an explicit
`ReadLimits` policy appropriate to the application. Raising the limits is an
application decision and requires its own memory planning.

## 0.1.0

Initial research release of the schema-9 Quantum reader, before the bounded
Python-only API introduced in 0.2.0. The legacy release did not sufficiently
constrain output allocation, metadata reads or compressed-packet reads.
