# Security and data privacy

This is experimental research software, not certified diagnostic software.
Treat recordings as untrusted input. Unsupported layouts, detected corruption
and exceeded limits raise errors; this is not a guarantee that every malformed
file will be detected, nor a guarantee against a system crash.

## Resource boundaries

Version 0.2.0 bounds individual output allocations, metadata lengths, parser
complexity and packet input reads. The default decoded-output limit is 64 MiB
per call. A rejected oversized sample request is checked before loading NumPy.
The default work budget also limits a call to 131,072 samples, independent
of channel count.
Metadata inspection does not load NumPy. The library never changes application
thread or environment settings.

These safeguards are **not a total process-memory limit**. Python, NumPy and
its numerical backend have their own overhead; the application can also retain
arbitrarily many returned arrays or run concurrent reads. In particular,
`list(reader.iter_samples())` deliberately defeats incremental processing.
Choose conservative limits and release processed chunks promptly. A service
handling untrusted uploads should additionally enforce operating-system process
memory and time limits and keep its dependencies current.

There is no command-line program, web server, network client, automatic upload
or telemetry in the library. Channel labels and ENT annotations can contain
patient information. Do not publish them, real EEG files, derived waveforms,
local paths or credentials in issues, test fixtures or package archives.

## Reporting

Use GitHub private vulnerability reporting when enabled, or contact the
repository owner through a private channel for sensitive reports. Public issues
should contain a sanitized description and, where possible, a small synthetic
reproducer. Do not repeatedly run a workload that has frozen or restarted your
machine; investigate the system failure before trying it again.

The 0.2.x line is the maintenance target. The older 0.1.x design lacks the
resource boundaries described here and should not be used for untrusted input.
There has been no independent security audit or clinical certification.
