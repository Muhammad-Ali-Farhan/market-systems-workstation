from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

import tkinter as tk
from tkinter import ttk

BG = "#0b1220"
PANEL = "#111a2b"
PANEL_ALT = "#172338"
TEXT = "#e8eef8"
MUTED = "#95a4ba"
ACCENT = "#4f8cff"
GOOD = "#39d98a"
WARN = "#ffcc66"
BAD = "#ff6b6b"
GRID = "#26364f"


def window_dpi(window: tk.Misc) -> int:
    """Return the effective DPI for the monitor containing the window."""
    if os.name == "nt":
        try:
            dpi = int(ctypes.windll.user32.GetDpiForWindow(window.winfo_id()))
            if dpi > 0:
                return dpi
        except Exception:
            pass

        try:
            dpi = int(ctypes.windll.user32.GetDpiForSystem())
            if dpi > 0:
                return dpi
        except Exception:
            pass

    try:
        return max(96, int(round(float(window.winfo_fpixels("1i")))))
    except Exception:
        return 96


def widget_scale(widget: tk.Misc) -> float:
    try:
        return float(getattr(widget.winfo_toplevel(), "ui_scale", 1.0))
    except Exception:
        return 1.0


def scaled_px(widget: tk.Misc, value: int | float) -> int:
    return max(1, int(round(float(value) * widget_scale(widget))))


def safe_float(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:,.{digits}f}"


def safe_percent(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{100.0 * number:,.{digits}f}%"


def safe_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def newest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def configure_styles(window: tk.Misc, ui_scale: float) -> None:
    style = ttk.Style(window)
    style.theme_use("clam")
    def px(value: int | float) -> int:
        return max(1, int(round(value * ui_scale)))
    style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("Card.TFrame", background=PANEL_ALT)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
    style.configure(
        "CardTitle.TLabel",
        background=PANEL_ALT,
        foreground=MUTED,
        font=("Segoe UI", 9),
    )
    style.configure(
        "Metric.TLabel",
        background=PANEL_ALT,
        foreground=TEXT,
        font=("Segoe UI Semibold", 18),
    )
    style.configure(
        "Header.TLabel",
        background=BG,
        foreground=TEXT,
        font=("Segoe UI Semibold", 22),
    )
    style.configure(
        "Subheader.TLabel",
        background=BG,
        foreground=MUTED,
        font=("Segoe UI", 10),
    )
    style.configure(
        "TButton",
        background=PANEL_ALT,
        foreground=TEXT,
        padding=(px(12), px(8)),
        borderwidth=0,
    )
    style.map("TButton", background=[("active", ACCENT), ("pressed", "#376dcc")])
    style.configure(
        "Accent.TButton",
        background=ACCENT,
        foreground="white",
        padding=(px(14), px(9)),
    )
    style.map(
        "Accent.TButton",
        background=[("active", "#6aa0ff"), ("pressed", "#376dcc")],
    )
    style.configure("Danger.TButton", background="#6d2530", foreground="white")
    style.map("Danger.TButton", background=[("active", "#9a3342")])
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=PANEL,
        foreground=MUTED,
        padding=(px(16), px(10)),
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", PANEL_ALT)],
        foreground=[("selected", TEXT)],
    )
    style.configure(
        "Treeview",
        background=PANEL,
        fieldbackground=PANEL,
        foreground=TEXT,
        rowheight=px(28),
        borderwidth=0,
    )
    style.configure(
        "Treeview.Heading", background=PANEL_ALT, foreground=TEXT, relief="flat"
    )
    style.map("Treeview", background=[("selected", ACCENT)])
    style.configure(
        "TEntry", fieldbackground=PANEL_ALT, foreground=TEXT, insertcolor=TEXT
    )
    style.configure("TCombobox", fieldbackground=PANEL_ALT, foreground=TEXT)
    style.configure(
        "Horizontal.TProgressbar", background=ACCENT, troughcolor=PANEL_ALT
    )
