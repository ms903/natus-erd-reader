# Security and resource boundaries

Report a reproducible vulnerability through the repository's
[security advisories](https://github.com/ms903/natus-erd-reader/security/advisories/new).
A small synthetic example, affected version and traceback help investigation.

The reader bounds output arrays, compressed packets, metadata, directory/index
counts and ENT parsing depth/node counts through `ReadLimits`. It validates
recording member paths and regular file types. ENT text is parsed with a limited
grammar. EDF export has a separate aggregate work-buffer budget and publishes
only a verified file to a previously unused destination.

These limits cover library allocations and parsing work, rather than the total
memory of an application, numerical runtime or operating system. Select limits
appropriate to the input and workload. Keep recording files unchanged throughout
an operation; reopen the reader after intentional source changes.
