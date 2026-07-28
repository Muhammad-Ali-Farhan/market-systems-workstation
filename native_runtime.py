
from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_DIRECTORY_HANDLES: list[object] = []


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def prepare_native_runtime(root: str | Path | None = None) -> Path:
    """Make the native extension and its dependent DLLs discoverable.

    Windows removes a directory from the DLL search path when the object
    returned by os.add_dll_directory is closed or garbage-collected. Retaining
    the handle for process lifetime is therefore intentional.
    """
    resolved = Path(root).resolve() if root is not None else application_root()
    root_text = str(resolved)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        try:
            handle = os.add_dll_directory(root_text)
        except OSError:
            pass
        else:
            _DLL_DIRECTORY_HANDLES.append(handle)
    return resolved

