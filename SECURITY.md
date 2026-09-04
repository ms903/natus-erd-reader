# Security and data privacy

This is a research data reader, not certified diagnostic software. Treat
recordings as untrusted input. Unsupported layouts and detected corruption
raise errors; this does not guarantee detection of every possible malformed
file.

The optional viewer binds to `127.0.0.1` by default. It has no authentication
layer and can display channel labels and event text from your local recording.
Do not expose it to a network or publish its responses. There are no external
analytics, uploads, CDNs or hosted data-processing services in the viewer.

Do not post patient data or credentials in public issues. Use GitHub private
vulnerability reporting when enabled, or contact the repository owner through
a private channel for sensitive reports. Public issues should contain only a
sanitized description and, where possible, a synthetic reproducer.

Only version 0.1.x is currently maintained. There is no security audit or
clinical validation claim for this initial release.
