# Changelog

## 0.1.0

Initial research release.

- Lazy STC/ETC-indexed schema-9 ERD decoding for Quantum headbox type 20.
- UTF-8 segment names, packet state resets, gaps and shorted channels as NaN.
- Unclipped digital values and validated conversion for the first 256 channels.
- Safe ENT event and montage parsing without `eval`.
- Random-access EDF reader and local-only synchronized comparison viewer.
- CLI, type information, synthetic tests, optional comparison and plotting tools.
- Source/wheel audits exclude recordings, derived patient data and credentials.

This release supports the explicitly documented layout only. It is not a
general-purpose reader for every Natus or NeuroWorks version.
