# Contributing

Use Python 3.10 or newer. Install development and interoperability tools in a
virtual environment:

```sh
python -m venv .venv
# Activate .venv for your shell.
python -m pip install ".[dev,interop,timezones]"
python -m unittest discover -s tests -t .
python -m mypy src/natus_erd
```

Synthetic fixtures in `tests/_fixture.py` exercise native packet decoding,
channel order, gaps, safe ENT parsing, resource limits and EDF interoperability.
C tests run when the extension is built; native CI explicitly requires it.
On Windows, run native builds from a Visual Studio x64 developer shell.
`NATUS_ERD_REQUIRE_NATIVE=1` makes compilation failure fatal.
`NATUS_ERD_NO_NATIVE=1` selects a pure Python build.

## Build and validate distributions

The staging helper copies package source, tests, examples and license files into
a clean directory. Each build has its own staging tree.

```sh
python tools/build_artifacts.py --kind pure --output build/candidate
python tools/build_artifacts.py --kind native --output build/candidate
python tools/check_dist.py build/candidate --version 0.3.0rc3
python -m twine check --strict build/candidate/*
python tools/test_install.py build/candidate/natus_erd_reader-0.3.0rc3-py3-none-any.whl --backend pure
python tools/test_install.py build/candidate/natus_erd_reader-0.3.0rc3.tar.gz --backend native
```

Use `test_install.py --backend native` on the wheel for the current platform as
well. It verifies the imported package location and executes the regression suite
against the installation. Optional MNE and pyEDFlib tests run when installed.

`build-assets.yml` uses cibuildwheel for CPython 3.10–3.14 Windows x64 and
manylinux x86_64 wheels, plus a pure wheel and sdist. Its final audit requires
all twelve assets with matching versions, metadata, tags and licenses.
Linux CI also runs ASan/UBSan against the native implementation.

## Release

Update the version in `pyproject.toml`, `src/natus_erd/__init__.py` and the brief
changelog. Run the tests and distribution checks. The `vX.Y.Z` or `vX.Y.ZrcN`
tag must match package metadata. The GitHub release workflow builds and attaches
the complete asset set; candidate versions are marked as prereleases.

PyPI publication is a separate manual `publish-pypi.yml` dispatch with the public
release tag. It audits all release archives, verifies hashes and uses the `pypi`
environment's OIDC Trusted Publisher. Publishing requires the corresponding
repository/environment permissions.

The bilingual documentation is maintained in the independent
[natus-reader.github.io repository](https://github.com/natus-reader/natus-reader.github.io).
Its contribution guide describes selecting package source and checking generated
API data. Include both language updates when changing public behavior.
