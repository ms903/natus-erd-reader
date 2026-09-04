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

## Release checklist

1. Update the versions in `pyproject.toml` and `src/natus_erd/__init__.py`.
2. Update `CHANGELOG.md`; run the unit suite and distribution audit.
3. Confirm `git ls-files` contains no recordings, reports or credentials.
4. Merge to `main` and wait for CI to succeed.
5. Push an annotated `vX.Y.Z` tag. The release workflow builds and attaches a
   wheel and source archive to the GitHub release.

The GitHub workflows do not publish to PyPI. PyPI publication is a separate,
explicitly authorized step; no PyPI credentials are required for this repository.
