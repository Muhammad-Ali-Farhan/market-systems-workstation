

from __future__ import annotations

import ctypes
import os
import traceback
from pathlib import Path

import certifi

from native_runtime import prepare_native_runtime

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
prepare_native_runtime(ROOT)
os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def enable_high_dpi_mode() -> None:
    """Enable crisp per-monitor DPI rendering before Tk is imported."""
    if os.name != "nt":
        return

    try:
        # Windows 10+: best available per-monitor DPI behavior.
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass

    try:
        # Windows 8.1 fallback: PROCESS_PER_MONITOR_DPI_AWARE.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass

    try:
        # Older Windows fallback.
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


enable_high_dpi_mode()


def main() -> int:
    try:
        from market_ui.app import main as run
        return run()
    except Exception:
        crash = traceback.format_exc()
        (LOGS / "desktop_crash.log").write_text(crash, encoding="utf-8")
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror(
                "Market Systems Workstation",
                "The workstation could not start. See logs\\desktop_crash.log",
            )
        except Exception:
            print(crash)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

