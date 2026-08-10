"""Repository/runtime path resolution used by packaged and source-tree UI modes."""


from __future__ import annotations

from pathlib import Path

from market_engine.runtime import (
    application_root as _application_root,
    prepare_native_runtime as _prepare_native_runtime,
)


def application_root() -> Path:
    return _application_root()


ROOT = application_root()
RECORDINGS_DIR = ROOT / "recordings"
ARTIFACTS_DIR = ROOT / "artifacts"
LOGS_DIR = ROOT / "logs"

for directory in (RECORDINGS_DIR, ARTIFACTS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def prepare_native_runtime() -> None:
    _prepare_native_runtime(ROOT)

