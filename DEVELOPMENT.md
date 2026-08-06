# Development guide

## Python checks

Create the Python 3.12 environment and install development dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run the local quality checks:

```powershell
.\.venv\Scripts\python.exe -m compileall -q -f .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

The Ruff selection is explicit in `pyproject.toml`, so upgrading Ruff cannot silently change the repository policy.

## Native build

Use CPython 3.12 and the pinned vcpkg baseline. On Windows:

```powershell
$env:VCPKG_ROOT = "C:\path\to\vcpkg"
.\BUILD_NATIVE.ps1
```

After changing the C++ ABI or pybind11 bindings, run the native tests and verify that the built module exports both `IngestionEngine` and `L2Synchronizer`.

## Change discipline

Before editing:

1. Identify the correctness or maintainability contract being changed.
2. List the files that need modification.
3. Define measurable acceptance criteria.

After editing:

1. Run the narrowest relevant tests.
2. Run the complete Python test suite.
3. Rebuild the native extension when C++ bindings change.
4. Run CTest and the workstation verifier.
5. Update design documentation when behavior or failure semantics change.

## Version policy

`project_version.py` defines the runtime version and HTTP User-Agent. The same public version must appear in `pyproject.toml`, `CMakeLists.txt`, and `vcpkg.json`; `tests/test_quality_policy.py` enforces that contract. Schema versions for binary formats, models, and reports remain independent and change only when their compatibility contract changes.

## Line endings

`.gitattributes` keeps source and documentation files in LF form across platforms while preserving CRLF for Windows batch files. Do not override the repository attributes with manual whole-tree conversion.
