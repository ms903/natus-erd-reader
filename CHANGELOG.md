# Changelog

## 0.3.0rc2

- Use fixed Beijing UTC+08:00 by default for clocks and EDF export.
- Display scan, write and verification progress with tqdm; support disabling or callbacks.
- Preserve all selected channels, including shorted channels with digital 32767.
- Apply official Quantum auxiliary units, negative-gain calibration and verified missing codes.
- Batch sequential readback with bounded worker queues and batched source windows.
- Report channel units and separate verification timing.
- Provide direct scripts and an anonymous EDF+D development design.

## 0.3.0rc1

- Add SNC schema-1 clocks and stored-range discovery.
- Export continuous, exactly aligned EDF+C with bounded C/Python workers.
- Preserve exact auxiliary values, whole-record events and channel label mappings.
- Check quantization, digital ranges, source waveform readback and atomic publication.
- Report ENT parsing completeness and distinguish resource limits from corrupt input.

## 0.2.1 — 2026-09-04

- Read sampling rates from headers and improve channel, path and gap handling.

## 0.2.0

- Add a bounded ENT parser, resource limits and package type information.

## 0.1.0

- Initial schema-9 Quantum ERD reader with windowed signal and event access.
