from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import ttk

from .theme import GRID, MUTED, PANEL, TEXT, scaled_px, widget_scale


class LineChart(tk.Canvas):
    def __init__(self, master, title: str = "", height: int = 240, **kwargs):
        super().__init__(
            master,
            bg=PANEL,
            highlightthickness=0,
            height=scaled_px(master, height),
            **kwargs,
        )
        self.title = title
        self.series: list[tuple[str, list[float], str]] = []
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_series(
        self, series: list[tuple[str, list[float] | np.ndarray, str]]
    ) -> None:
        self.series = [
            (name, [float(value) for value in values], color)
            for name, values, color in series
        ]
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 100)
        height = max(self.winfo_height(), 80)
        scale = widget_scale(self)
        left = int(round(54 * scale))
        right = int(round(18 * scale))
        top = int(round(34 * scale))
        bottom = int(round(30 * scale))

        self.create_text(
            left,
            16,
            text=self.title,
            fill=TEXT,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
        )
        if not self.series or not any(values for _, values, _ in self.series):
            self.create_text(
                width / 2,
                height / 2,
                text="No data",
                fill=MUTED,
                font=("Segoe UI", 11),
            )
            return

        all_values = [
            value
            for _, values, _ in self.series
            for value in values
            if math.isfinite(value)
        ]
        if not all_values:
            self.create_text(
                width / 2, height / 2, text="No finite data", fill=MUTED
            )
            return

        minimum = min(all_values)
        maximum = max(all_values)
        if maximum == minimum:
            maximum += 1.0
            minimum -= 1.0
        padding = (maximum - minimum) * 0.08
        minimum -= padding
        maximum += padding

        plot_w = max(width - left - right, 1)
        plot_h = max(height - top - bottom, 1)
        for index in range(5):
            y = top + plot_h * index / 4
            self.create_line(left, y, width - right, y, fill=GRID)
            label = maximum - (maximum - minimum) * index / 4
            self.create_text(
                left - 8,
                y,
                text=f"{label:.4g}",
                fill=MUTED,
                anchor="e",
                font=("Consolas", 8),
            )

        for series_index, (name, values, color) in enumerate(self.series):
            finite = [
                (index, value)
                for index, value in enumerate(values)
                if math.isfinite(value)
            ]
            if len(finite) >= 2:
                count = max(len(values) - 1, 1)
                points: list[float] = []
                for index, value in finite:
                    x = left + plot_w * index / count
                    y = top + plot_h * (maximum - value) / (maximum - minimum)
                    points.extend((x, y))
                self.create_line(
                    *points,
                    fill=color,
                    width=max(2, int(round(2 * scale))),
                    smooth=False,
                )
            legend_x = left + series_index * int(round(150 * scale))
            self.create_line(
                legend_x,
                height - int(round(13 * scale)),
                legend_x + int(round(20 * scale)),
                height - int(round(13 * scale)),
                fill=color,
                width=max(3, int(round(3 * scale))),
            )
            self.create_text(
                legend_x + int(round(26 * scale)),
                height - int(round(13 * scale)),
                text=name,
                fill=MUTED,
                anchor="w",
                font=("Segoe UI", 8),
            )


class MetricCard(ttk.Frame):
    def __init__(self, master, title: str, value: str = "—"):
        super().__init__(
            master,
            style="Card.TFrame",
            padding=(scaled_px(master, 14), scaled_px(master, 10)),
        )
        ttk.Label(self, text=title, style="CardTitle.TLabel").pack(anchor="w")
        self.value_var = tk.StringVar(value=value)
        ttk.Label(self, textvariable=self.value_var, style="Metric.TLabel").pack(
            anchor="w", pady=(4, 0)
        )

    def set(self, value: object) -> None:
        self.value_var.set(str(value))


class LogBox(tk.Text):
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            bg="#080d17",
            fg="#c9d4e5",
            insertbackground=TEXT,
            relief="flat",
            wrap="word",
            font=("Consolas", 9),
            **kwargs,
        )
        self.configure(state="disabled")

    def write(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.configure(state="normal")
        self.insert("end", f"[{timestamp}] {message.rstrip()}\n")
        self.see("end")
        self.configure(state="disabled")
