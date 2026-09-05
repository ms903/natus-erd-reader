# EDF+D observations and implementation design

`export_edf` automatically writes EDF+C for one stored interval and EDF+D
for multiple stored intervals. This document records its mapping and evidence. The accompanying JSON contains only
anonymous structural measurements; no source paths, identities or event text.

## Reference evidence

Seven official Quantum exports have 276 ordinary signals, 64 samples per
record, and a record duration of 1/32 second (2048 Hz). Cases 1–5 and 7 are
complete exports; case 6 is the small template. Header and first/last records
were checked in all seven. All 9,501 timekeeping TALs were checked in case 6;
the complete files were sampled, not exhaustively scanned for each jump.
`official-edf-observations.json` retains exact rational times and calibration.

The template has exactly one jump at zero-based record 9,469: an additional
1,388,031/32 seconds separates adjacent records. Its stored record duration is
9,501/32 seconds, while its elapsed record span is 349,383/8 seconds. A reader
must distinguish these quantities. The first TAL contributes a fractional
offset from the header's whole second.

Six complete exports use DC1–DC16 and TRIG in uV, physical endpoints
5151600 and -5151600, digital endpoints -32768 and 32767. The template alone
uses an empty TRIG unit and identity physical calibration. This does not select
a second export mode. EEG reference endpoints are 8711 and -8711. OSAT uses
0–102.3 %, PR 0–1023 bpm, both digital 0–32767. Pleth uses physical endpoints
4.29e+09 and 32767, digital -32768–32767; its unusual scale is reproduced as
observed and is not interpreted as a clinically calibrated pulsatile voltage.

Paired source snippets show DC/TRIG digital values on the same scale; their
EDF physical values use the full affine calibration, including its half-count
offset and negative gain. Source OSAT/PR missing code 131070 corresponds to
official digital zero. Valid measurements are not established by these pairs.
Source Pleth zero corresponds to official digital zero; nonzero source values
are likewise not established. Export refuses these unverified values explicitly.
The source examples are 512 Hz and their starts differ from the reference
exports. No resampling or clock shift is inferred from near-matching snippets.

For cases 1, 2, 3, 4, 6 and 7, the first record has four constant-32767 rows at
249, 251, 253 and 255. Case 5 does not. Export uses source shorted flags, never
this list. In the last record, the apparent trailing fill lengths for cases
1–7 are 56, 38, 41, 52, 2, 48 and 56 samples. EEG/DC and shorted rows use zero
in those positions, while Pleth uses 32767. OSAT/PR are zero throughout.
These observations distinguish padding from valid shorted samples (32767)
and valid Pleth zeros. They do not prove an official padding algorithm for
every signal state or internal segment boundary.

## Format constraints

The [EDF+ specification](https://www.edfplus.info/specs/edfplus.html), sections
2.1.1–2.1.3 and 2.2.4, defines EDF+D record order, equal sampling intervals
within each record, a timekeeping TAL per record, and negative amplifier gain.
Physical endpoints may decrease; digital endpoints must increase. The record
onset marks its first sample, so a gap belongs between records, not inside a
record. Header local time and the initial fractional TAL jointly locate the
first sample. EDF contains no timezone identifier.

## Implemented mapping

Use the source stored intervals `[a, b)` after intersecting the requested
window. Select one exact record size that divides **every** interval length,
has a finite decimal duration at the source rate, fits the header fields, and
leaves room for complete annotations within the record byte limit. Generate
record starts inside each interval independently. A record may never straddle
a source gap. Keep source channel order, sampling rate and SNC-derived origin.

For record start `s`, write `origin + (s - window_start) / sample_rate` as its
timekeeping onset. The next segment therefore retains the real sample-stamp
gap. Store only available samples. Report stored sample count, logical sample
span and record-duration sum separately. Reuse bounded packet jobs and the
existing signal calibration, ordered writer and all-record verifier.

Use sorted segment starts and cumulative record counts to assign events,
without building an index entry for every record. An event inside a record
goes there; an event in a gap goes in the next record with its original onset.
Before/after-window events go in the first/last record, preserving the whole
source event policy. Annotation capacity is checked before writing; event
times are never moved to match their storage record.

The implementation rejects a window if no common exact grid exists.
EDF+D requires at least two samples per record here: the specification uses
zero-duration records for the single-sample case, which cannot retain the
source rate in the positive-duration layout used by this exporter. Do not silently pad, trim, join gaps or invent valid duration. A future
explicit padding feature needs a documented validity representation and
third-party tests before adopting the observed vendor tail behavior. Zero
padding alone does not communicate the number of genuine source samples.

## Acceptance cases

`edf-d-cases.json` gives an anonymous two-segment case with exact TALs,
boundary/gap events, and a fractional-rate case that must fail without output.
Implementation tests decode stored samples, preserve the
gap in an EDF+D-aware reader, check events on either side of a boundary, cover
shorted channels and incomplete tails, and reject non-monotonic onsets,
records crossing gaps, changed TALs and truncated records. Reusing the EDF+C
publication checks must preserve cancellation cleanup and no-clobber output.

## Measured continuous-export performance

`edf-performance.json` records 60-second and 10-minute windows on the same
machine and disks, with automatic native workers and the 256 MiB buffer budget.
It separates scan, write and verify wall times. The rc1 write time is corrected
by subtracting its separately measured verifier time because that result field
included verification. The comparison uses each release's default selection:
272 output channels in rc1 and 276 in rc2. Results depend on cache and storage;
these measurements do not imply a fixed conversion speed for other recordings.
