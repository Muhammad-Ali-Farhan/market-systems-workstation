
from __future__ import annotations

import sys
from pathlib import Path

from native_runtime import prepare_native_runtime as _prepare_native_runtime


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


ROOT = application_root()
RECORDINGS_DIR = ROOT / "recordings"
ARTIFACTS_DIR = ROOT / "artifacts"
LOGS_DIR = ROOT / "logs"

for directory in (RECORDINGS_DIR, ARTIFACTS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def prepare_native_runtime() -> None:
    _prepare_native_runtime(ROOT)

