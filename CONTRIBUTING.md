# Contributing

Use Python 3.10 or newer. A clean development environment is recommended:

```shell
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -e ".[dev]"
python -m unittest discover -v -s tests -t .
python -m build
python -m twine check dist/*
python tools/check_dist.py dist
```

All automated tests generate synthetic recordings at runtime. Do not add
clinical recordings, patient names, event text, exported waveforms, local
credentials or derived plots to issues, pull requests, commits or artifacts.
When reporting a format problem, supply sanitized structural metadata and a
synthetic reproducer where possible.

Keep unsupported schemas and headboxes explicit. Do not silently reuse the
Quantum calibration for unverified hardware. Changes to packet decoding need
tests for absolute values, delta sentinels, shorted channels and packet edges.
Resource-boundary tests must intercept oversized requests or use synthetic
bounded input; do not actually allocate gigabytes to test rejection paths.
Keep metadata-only imports free of NumPy and never alter the user's global
numerical-backend settings from package code.

CI sets `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for the test runner
to keep backend initialization predictable. These are application-side test
settings, not package behavior or a substitute for an operating-system memory
limit. Backend-specific failure diagnosis should record the actual environment
and avoid unsafe reproduction on a user's machine.

The installed distribution exposes Python APIs only. Do not add executables,
web assets, EDF readers, optional plotting dependencies or real data to it.
`tools/check_dist.py` is a development audit script, not an installed command.

## Release checklist

1. Update the versions in `pyproject.toml` and `src/natus_erd/__init__.py`.
2. Update `CHANGELOG.md`; run the unit suite and distribution audit from a
   clean staging tree so removed files cannot survive in build caches.
3. Confirm `git ls-files` contains no recordings, reports or credentials.
4. Merge to `main` and wait for CI to succeed.
5. Push an annotated `vX.Y.Z` tag. The release workflow builds and attaches a
   wheel and source archive to the GitHub release.

6. For an explicitly authorized PyPI publication, configure a PyPI Trusted
   Publisher for owner `ms903`, repository `natus-erd-reader`, workflow
   `publish-pypi.yml`, and environment `pypi`. A new PyPI project requires a
   pending publisher under the account's Publishing settings. See the
   [official PyPI setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
7. Manually run `publish-pypi.yml` with the released tag. It audits the existing
   GitHub Release archives and uploads those same files using short-lived OIDC
   credentials. It does not rebuild them or require a stored PyPI API token.

PyPI publication is a separate, explicitly authorized step, not triggered by
ordinary pushes or by the GitHub Release workflow. Never commit credentials.
