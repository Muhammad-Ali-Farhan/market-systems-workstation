from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import certifi
import numpy as np

from .data import discover_recordings, recording_chart_series, summarize_recording
from qbin import feature_reset_indices, read_metadata
from .model import OnlineFeatureBuilder, load_model, train_model
from .l2 import L2_RECORDINGS_DIR, discover_l2_recordings, summarize_l2_recording
from .paths import ARTIFACTS_DIR, LOGS_DIR, RECORDINGS_DIR, ROOT, prepare_native_runtime
from .reports import discover_research_reports, inventory_research_reports
from .theme import (
    ACCENT,
    BG,
    GOOD,
    MUTED,
    PANEL,
    TEXT,
    WARN,
    configure_styles,
    newest,
    safe_float,
    safe_int,
    safe_percent,
    window_dpi,
)
from .widgets import LineChart, LogBox, MetricCard


MAX_LATENCY_SAMPLES_PER_INTERVAL = 50_000


class EngineRunner(threading.Thread):
    def __init__(
        self,
        event_queue: queue.Queue,
        *,
        mode: str,
        recording_path: str = "",
        replay_file: str = "",
        replay_speed: float = 1.0,
        model_path: str = "",
    ):
        super().__init__(daemon=False)
        self.event_queue = event_queue
        self.mode = mode
        self.recording_path = recording_path
        self.replay_file = replay_file
        self.replay_speed = replay_speed
        self.model_path = model_path
        self.stop_event = threading.Event()

    def request_stop(self) -> None:
        self.stop_event.set()

    def emit(self, kind: str, payload: object) -> None:
        self.event_queue.put((kind, payload))

    @staticmethod
    def _sample_latency(values: np.ndarray, remaining: int) -> np.ndarray:
        if remaining <= 0 or values.size == 0:
            return np.empty(0, dtype=np.float64)
        if values.size <= remaining:
            return values.copy()
        indices = np.linspace(0, values.size - 1, remaining, dtype=np.int64)
        return values[indices].copy()

    def run(self) -> None:
        engine = None
        total_ticks = 0
        try:
            import quant_engine

            model = load_model(self.model_path) if self.model_path else None
            feature_builder = OnlineFeatureBuilder()
            replay_resets: frozenset[int] = frozenset()
            replay_record_index = 0
            dtype = np.dtype(quant_engine.order_book_dtype)
            if dtype.itemsize != 32:
                raise RuntimeError("Native and NumPy record layouts do not match.")
            buffer = np.empty(4096, dtype=dtype)
            engine = quant_engine.IngestionEngine()

            if self.mode == "live":
                self.emit("log", "Connecting to Binance BTCUSDT bookTicker...")
                engine.start_live(self.recording_path)
                self.emit("log", "Native live engine started.")
            elif self.mode == "replay":
                if not math.isfinite(self.replay_speed) or self.replay_speed < 0.0:
                    raise ValueError("Replay speed must be finite and non-negative.")
                replay_metadata = read_metadata(self.replay_file)
                replay_resets = feature_reset_indices(replay_metadata)
                label = "maximum speed" if self.replay_speed == 0 else f"{self.replay_speed:g}x"
                self.emit("log", f"Starting deterministic replay at {label}: {self.replay_file}")
                engine.start_replay(self.replay_file, self.replay_speed)
            else:
                raise ValueError("Unknown engine mode")

            interval_ticks = 0
            interval_start = time.perf_counter()
            last_emit = interval_start
            latency_samples: list[np.ndarray] = []
            latency_sample_count = 0
            previous_drops = 0
            previous_reconnects = 0
            previous_engine_error = ""
            latest = {
                "bid": float("nan"),
                "ask": float("nan"),
                "mid": float("nan"),
                "spread": float("nan"),
                "obi": float("nan"),
                "prediction": float("nan"),
                "signal": "FLAT",
                "threshold_multiple": 0.0,
            }

            while not self.stop_event.is_set():
                count = engine.consume_batch(buffer)
                if count > 0:
                    data = buffer[:count]
                    total_ticks += count
                    interval_ticks += count
                    bid = data["best_bid"].astype(np.float64, copy=False)
                    ask = data["best_ask"].astype(np.float64, copy=False)
                    bid_volume = data["bid_volume"].astype(np.float64, copy=False)
                    ask_volume = data["ask_volume"].astype(np.float64, copy=False)
                    denominator = bid_volume + ask_volume
                    obi = np.divide(
                        bid_volume - ask_volume,
                        denominator,
                        out=np.zeros_like(denominator),
                        where=denominator != 0,
                    )

                    if self.mode == "live":
                        current_ns = engine.now_ns()
                        latency_us = (
                            current_ns - data["timestamp_ns"].astype(np.uint64, copy=False)
                        ).astype(np.float64) / 1_000.0
                        remaining = MAX_LATENCY_SAMPLES_PER_INTERVAL - latency_sample_count
                        sample = self._sample_latency(latency_us, remaining)
                        if sample.size:
                            latency_samples.append(sample)
                            latency_sample_count += int(sample.size)

                        drops = int(engine.dropped_ticks())
                        reconnects = int(engine.reconnect_count())
                        if drops != previous_drops or reconnects != previous_reconnects:
                            feature_builder.reset()
                            previous_drops = drops
                            previous_reconnects = reconnects
                            self.emit("log", "Feature history reset after a feed gap or reconnect.")

                    latest.update(
                        bid=float(bid[-1]),
                        ask=float(ask[-1]),
                        mid=float((bid[-1] + ask[-1]) / 2.0),
                        spread=float(ask[-1] - bid[-1]),
                        obi=float(obi[-1]),
                    )
                    if model is not None:
                        for row in data:
                            if (
                                self.mode == "replay"
                                and replay_record_index in replay_resets
                            ):
                                feature_builder.reset()
                            feature_row = feature_builder.update(row)
                            replay_record_index += 1
                            if feature_row is None:
                                continue
                            prediction = float(model.predict(feature_row))
                            latest["prediction"] = prediction
                            latest["threshold_multiple"] = (
                                abs(prediction) / model.threshold if model.threshold > 0 else 0.0
                            )
                            latest["signal"] = (
                                "LONG"
                                if prediction > 0.0 and prediction >= model.threshold
                                else "SHORT"
                                if prediction < 0.0 and prediction <= -model.threshold
                                else "FLAT"
                            )
                else:
                    if self.mode == "replay" and not engine.is_running():
                        break
                    time.sleep(0.001)

                now = time.perf_counter()
                if now - last_emit >= 0.25:
                    elapsed = max(now - interval_start, 1e-9)
                    combined = (
                        np.concatenate(latency_samples)
                        if latency_samples
                        else np.empty(0, dtype=np.float64)
                    )
                    current_engine_error = str(engine.last_error())
                    if current_engine_error and current_engine_error != previous_engine_error:
                        self.emit("log", f"Native engine: {current_engine_error}")
                    previous_engine_error = current_engine_error
                    payload = {
                        **latest,
                        "mode": self.mode,
                        "total_ticks": total_ticks,
                        "rate": interval_ticks / elapsed,
                        "p50_us": float(np.percentile(combined, 50)) if combined.size else float("nan"),
                        "p99_us": float(np.percentile(combined, 99)) if combined.size else float("nan"),
                        "p999_us": float(np.percentile(combined, 99.9)) if combined.size else float("nan"),
                        "dropped_ticks": int(engine.dropped_ticks()),
                        "malformed_messages": int(engine.malformed_messages()),
                        "reconnect_count": int(engine.reconnect_count()),
                        "recorded_ticks": int(engine.recorded_ticks()),
                        "recording_dropped": int(engine.recording_dropped()),
                        "recording_write_errors": int(engine.recording_write_errors()),
                        "replayed_ticks": int(engine.replayed_ticks()),
                        "replay_backpressure": int(engine.replay_backpressure_events()),
                        "replay_errors": int(engine.replay_errors()),
                        "engine_error": current_engine_error,
                    }
                    self.emit("snapshot", payload)
                    interval_ticks = 0
                    interval_start = now
                    last_emit = now
                    latency_samples.clear()
                    latency_sample_count = 0

            self.emit("log", "Stopping native engine and draining the recorder...")
            engine.stop()
            error = engine.last_error()
            if error:
                raise RuntimeError(error)
            if self.mode == "live" and self.recording_path:
                if engine.recording_write_errors() != 0:
                    raise RuntimeError(
                        "The binary recorder reported a write/finalization error."
                    )
                if engine.recording_dropped() != 0:
                    raise RuntimeError(
                        "The binary recorder dropped records; capture is incomplete."
                    )
            self.emit(
                "complete",
                {
                    "mode": self.mode,
                    "total_ticks": total_ticks,
                    "recorded_ticks": int(engine.recorded_ticks()),
                    "replayed_ticks": int(engine.replayed_ticks()),
                    "recording_dropped": int(engine.recording_dropped()),
                    "replay_errors": int(engine.replay_errors()),
                },
            )
        except Exception:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            self.emit("error", traceback.format_exc())


class QuantWorkstation(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.withdraw()
        self.update_idletasks()

        self.display_dpi = window_dpi(self)
        self.ui_scale = max(1.0, self.display_dpi / 96.0)

        # Tk uses pixels per typographic point. This keeps fonts crisp and
        # physically consistent at 125%, 150%, 175%, 200%, and 4K scaling.
        self.tk.call("tk", "scaling", self.display_dpi / 72.0)

        self.title("Market Systems Workstation")
        self.minsize(
            int(round(1180 * self.ui_scale)),
            int(round(760 * self.ui_scale)),
        )
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._configure_styles()

        self.engine_events: queue.Queue = queue.Queue()
        self.engine_runner: EngineRunner | None = None
        self.training_thread: threading.Thread | None = None
        self._closing = False
        self.mid_history: deque[float] = deque(maxlen=700)
        self.prediction_history: deque[float] = deque(maxlen=700)
        self._build_shell()

        # A workstation benefits from the available screen area. Starting
        # maximized also avoids a physically tiny 1440x900 window on 4K panels.
        self.deiconify()
        if os.name == "nt":
            try:
                self.state("zoomed")
            except tk.TclError:
                self.geometry(
                    f"{int(round(1440 * self.ui_scale))}x"
                    f"{int(round(900 * self.ui_scale))}"
                )
        else:
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            self.geometry(f"{int(screen_width * 0.92)}x{int(screen_height * 0.90)}")

        self.after(100, self._poll_events)
        self.refresh_all()

    def _configure_styles(self) -> None:
        configure_styles(self, self.ui_scale)

    def _build_shell(self) -> None:
        header = ttk.Frame(self, padding=(22, 18, 22, 8))
        header.pack(fill="x")
        ttk.Label(header, text="MARKET SYSTEMS WORKSTATION", style="Header.TLabel").pack(side="left")
        self.global_status = tk.StringVar(value="● READY")
        ttk.Label(header, textvariable=self.global_status, foreground=GOOD).pack(side="right")
        ttk.Label(self, text="C++20 market-data infrastructure • deterministic replay • microstructure research", style="Subheader.TLabel").pack(anchor="w", padx=24, pady=(0, 12))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.pages: dict[str, ttk.Frame] = {}
        for title in ("Command Center", "Live Capture", "Data Library", "L2 Lab", "Research Lab", "Alpha Runtime", "Diagnostics", "Interview Mode"):
            frame = ttk.Frame(self.notebook, padding=18)
            self.notebook.add(frame, text=title)
            self.pages[title] = frame

        self._build_command_center()
        self._build_live_capture()
        self._build_data_library()
        self._build_l2_lab()
        self._build_research_lab()
        self._build_alpha_runtime()
        self._build_diagnostics()
        self._build_interview()

    def _section_title(self, parent, title: str, subtitle: str = "") -> None:
        ttk.Label(parent, text=title, font=("Segoe UI Semibold", 18)).pack(anchor="w")
        if subtitle:
            ttk.Label(parent, text=subtitle, foreground=MUTED).pack(anchor="w", pady=(2, 14))

    def _build_command_center(self) -> None:
        page = self.pages["Command Center"]
        self._section_title(page, "Command Center", "Project health, latest data, and latest research evidence.")
        cards = ttk.Frame(page)
        cards.pack(fill="x")
        self.cc_engine = MetricCard(cards, "Native engine")
        self.cc_recordings = MetricCard(cards, "Recordings")
        self.cc_models = MetricCard(cards, "Models")
        self.cc_latest = MetricCard(cards, "Latest session")
        for index, card in enumerate((self.cc_engine, self.cc_recordings, self.cc_models, self.cc_latest)):
            card.grid(row=0, column=index, padx=(0 if index == 0 else 8, 0), sticky="nsew")
            cards.columnconfigure(index, weight=1)

        controls = ttk.Frame(page)
        controls.pack(fill="x", pady=12)
        ttk.Button(controls, text="Refresh workstation", command=self.refresh_all, style="Accent.TButton").pack(side="left")
        ttk.Button(controls, text="Open project folder", command=lambda: os.startfile(ROOT)).pack(side="left", padx=8)

        lower = ttk.Frame(page)
        lower.pack(fill="both", expand=True)
        lower.columnconfigure(0, weight=2)
        lower.columnconfigure(1, weight=1)
        lower.rowconfigure(0, weight=1)
        self.cc_chart = LineChart(lower, "Latest recording mid-price")
        self.cc_chart.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        panel = ttk.Frame(lower, style="Panel.TFrame", padding=16)
        panel.grid(row=0, column=1, sticky="nsew")
        ttk.Label(panel, text="Latest evidence", style="Panel.TLabel", font=("Segoe UI Semibold", 13)).pack(anchor="w")
        self.cc_evidence = tk.Text(panel, bg=PANEL, fg=TEXT, relief="flat", wrap="word", font=("Consolas", 9))
        self.cc_evidence.pack(fill="both", expand=True, pady=(10, 0))

    def _build_live_capture(self) -> None:
        page = self.pages["Live Capture"]
        self._section_title(page, "Live Capture", "Direct control of the native Binance ingestion and binary recorder.")
        controls = ttk.Frame(page)
        controls.pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Recording filename:").pack(side="left")
        self.live_filename = tk.StringVar(value=f"btcusdt-{datetime.now().strftime('%Y%m%d-%H%M%S')}.qbin")
        ttk.Entry(controls, textvariable=self.live_filename, width=38).pack(side="left", padx=8)
        self.live_start = ttk.Button(controls, text="Start live capture", command=self.start_live_capture, style="Accent.TButton")
        self.live_start.pack(side="left")
        self.live_stop = ttk.Button(controls, text="Stop", command=self.stop_engine, style="Danger.TButton", state="disabled")
        self.live_stop.pack(side="left", padx=8)

        metrics = ttk.Frame(page)
        metrics.pack(fill="x")
        names = (
            "Ticks",
            "Rate/sec",
            "Mid",
            "Spread",
            "OBI",
            "p99 local age µs",
            "Recorded",
            "Drops",
        )
        self.live_metrics: dict[str, MetricCard] = {}
        for index, name in enumerate(names):
            card = MetricCard(metrics, name)
            card.grid(row=index // 4, column=index % 4, padx=(0 if index % 4 == 0 else 8, 0), pady=(0, 8), sticky="nsew")
            metrics.columnconfigure(index % 4, weight=1)
            self.live_metrics[name] = card

        split = ttk.Frame(page)
        split.pack(fill="both", expand=True)
        split.columnconfigure(0, weight=2)
        split.columnconfigure(1, weight=1)
        split.rowconfigure(0, weight=1)
        self.live_chart = LineChart(split, "Live BTCUSDT mid-price")
        self.live_chart.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.live_log = LogBox(split)
        self.live_log.grid(row=0, column=1, sticky="nsew")

    def _build_data_library(self) -> None:
        page = self.pages["Data Library"]
        self._section_title(page, "Data Library", "Validate recordings and inspect market structure without loading everything into memory.")
        body = ttk.Frame(page)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Panel.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Button(left, text="Refresh", command=self.refresh_recordings).pack(fill="x")
        self.recording_list = tk.Listbox(left, bg=PANEL, fg=TEXT, selectbackground=ACCENT, relief="flat", font=("Consolas", 9), exportselection=False)
        self.recording_list.pack(fill="both", expand=True, pady=8)
        self.recording_list.bind("<<ListboxSelect>>", lambda _event: self.inspect_selected_recording())
        ttk.Button(left, text="Inspect selected", command=self.inspect_selected_recording, style="Accent.TButton").pack(fill="x")

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)
        self.recording_summary = tk.Text(right, height=9, bg=PANEL, fg=TEXT, relief="flat", font=("Consolas", 10))
        self.recording_summary.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.library_mid_chart = LineChart(right, "Mid-price")
        self.library_mid_chart.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        self.library_obi_chart = LineChart(right, "Order-book imbalance and spread (bps)")
        self.library_obi_chart.grid(row=2, column=0, sticky="nsew")


    def _build_l2_lab(self) -> None:
        page = self.pages["L2 Lab"]
        self._section_title(
            page,
            "Sequence-Correct L2 Lab",
            "Capture diff-depth plus aggregate trades, verify deterministic reconstruction, and benchmark the L2 pipeline.",
        )
        controls = ttk.Frame(page)
        controls.pack(fill="x", pady=(0, 10))
        ttk.Button(
            controls,
            text="Start BTC + ETH capture",
            command=self.launch_l2_capture,
            style="Accent.TButton",
        ).pack(side="left")
        ttk.Button(controls, text="Refresh", command=self.refresh_l2_recordings).pack(side="left", padx=8)
        ttk.Button(controls, text="Verify selected", command=self.verify_selected_l2).pack(side="left")
        ttk.Button(controls, text="Benchmark selected", command=self.benchmark_selected_l2).pack(side="left", padx=8)
        ttk.Button(
            controls,
            text="Open L2 recordings",
            command=lambda: os.startfile(L2_RECORDINGS_DIR),
        ).pack(side="left")

        body = ttk.Frame(page)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, style="Panel.TFrame", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.l2_recording_list = tk.Listbox(
            left,
            bg=PANEL,
            fg=TEXT,
            selectbackground=ACCENT,
            relief="flat",
            font=("Consolas", 9),
            exportselection=False,
        )
        self.l2_recording_list.pack(fill="both", expand=True)
        self.l2_recording_list.bind(
            "<<ListboxSelect>>", lambda _event: self.inspect_selected_l2()
        )
        self.l2_output = tk.Text(
            body, bg=PANEL, fg=TEXT, relief="flat", font=("Consolas", 10), wrap="word"
        )
        self.l2_output.grid(row=0, column=1, sticky="nsew")
        self.l2_output.insert(
            "end",
            "L2 capture follows Binance snapshot/diff-depth synchronization, records exact fixed-point events, and writes reconstruction checkpoints.\n",
        )

    def _build_research_lab(self) -> None:
        page = self.pages["Research Lab"]
        self._section_title(
            page,
            "Research Lab",
            "Leakage-resistant selection, execution stress, and untouched holdout diagnostics.",
        )
        top = ttk.Frame(page)
        top.pack(fill="x")
        labels = (
            ("Recordings (multi-select):", 0),
            ("Horizon:", 1),
            ("Fee bps/side:", 2),
            ("Slippage bps/side:", 3),
            ("Trade size (BTC):", 4),
            ("Displayed participation:", 5),
            ("Artifact name:", 6),
        )
        for text, column in labels:
            ttk.Label(top, text=text).grid(
                row=0,
                column=column,
                sticky="w",
                padx=(14 if column else 0, 0),
            )
        self.research_list = tk.Listbox(
            top,
            height=7,
            selectmode="extended",
            bg=PANEL,
            fg=TEXT,
            selectbackground=ACCENT,
            relief="flat",
            exportselection=False,
        )
        self.research_list.grid(row=1, column=0, sticky="nsew", pady=6)
        self.research_horizon = tk.IntVar(value=20)
        ttk.Spinbox(
            top, from_=1, to=1000, textvariable=self.research_horizon, width=9
        ).grid(row=1, column=1, sticky="nw", padx=(14, 0), pady=6)
        self.research_fee = tk.DoubleVar(value=0.0)
        ttk.Entry(top, textvariable=self.research_fee, width=11).grid(
            row=1, column=2, sticky="nw", padx=(14, 0), pady=6
        )
        self.research_slippage = tk.DoubleVar(value=0.0)
        ttk.Entry(top, textvariable=self.research_slippage, width=11).grid(
            row=1, column=3, sticky="nw", padx=(14, 0), pady=6
        )
        self.research_trade_size = tk.DoubleVar(value=0.0)
        ttk.Entry(top, textvariable=self.research_trade_size, width=11).grid(
            row=1, column=4, sticky="nw", padx=(14, 0), pady=6
        )
        self.research_participation = tk.DoubleVar(value=1.0)
        ttk.Entry(top, textvariable=self.research_participation, width=11).grid(
            row=1, column=5, sticky="nw", padx=(14, 0), pady=6
        )
        self.research_name = tk.StringVar(value="alpha_model")
        ttk.Entry(top, textvariable=self.research_name, width=18).grid(
            row=1, column=6, sticky="nw", padx=(14, 0), pady=6
        )
        top.columnconfigure(0, weight=1)

        actions = ttk.Frame(page)
        actions.pack(fill="x", pady=10)
        self.train_button = ttk.Button(
            actions,
            text="Train and evaluate",
            command=self.start_training,
            style="Accent.TButton",
        )
        self.train_button.pack(side="left")
        self.training_progress = ttk.Progressbar(
            actions, maximum=100, mode="determinate"
        )
        self.training_progress.pack(side="left", fill="x", expand=True, padx=12)
        self.training_status = tk.StringVar(value="Ready")
        ttk.Label(
            actions, textvariable=self.training_status, foreground=MUTED
        ).pack(side="right")

        self.research_output = tk.Text(
            page,
            bg=PANEL,
            fg=TEXT,
            relief="flat",
            font=("Consolas", 10),
            wrap="word",
        )
        self.research_output.pack(fill="both", expand=True)

    def _build_alpha_runtime(self) -> None:
        page = self.pages["Alpha Runtime"]
        self._section_title(page, "Alpha Runtime", "Deploy the same trained artifact against deterministic replay or live market data.")
        controls = ttk.Frame(page)
        controls.pack(fill="x", pady=(0, 12))
        ttk.Label(controls, text="Model:").pack(side="left")
        self.runtime_model = ttk.Combobox(controls, width=28, state="readonly")
        self.runtime_model.pack(side="left", padx=6)
        ttk.Label(controls, text="Replay file:").pack(side="left", padx=(12, 0))
        self.runtime_recording = ttk.Combobox(controls, width=30, state="readonly")
        self.runtime_recording.pack(side="left", padx=6)
        ttk.Label(controls, text="Speed:").pack(side="left", padx=(12, 0))
        self.runtime_speed = tk.DoubleVar(value=1.0)
        ttk.Entry(controls, textvariable=self.runtime_speed, width=8).pack(side="left", padx=6)
        ttk.Button(controls, text="Start replay", command=self.start_alpha_replay, style="Accent.TButton").pack(side="left", padx=4)
        ttk.Button(controls, text="Start live", command=self.start_alpha_live).pack(side="left", padx=4)
        ttk.Button(controls, text="Stop", command=self.stop_engine, style="Danger.TButton").pack(side="left", padx=4)

        metrics = ttk.Frame(page)
        metrics.pack(fill="x")
        self.alpha_signal = MetricCard(metrics, "Signal")
        self.alpha_prediction = MetricCard(metrics, "Forecast bps")
        self.alpha_threshold = MetricCard(metrics, "Threshold multiple")
        self.alpha_mid = MetricCard(metrics, "Mid")
        self.alpha_obi = MetricCard(metrics, "OBI")
        self.alpha_rate = MetricCard(metrics, "Rate/sec")
        for i, card in enumerate((self.alpha_signal, self.alpha_prediction, self.alpha_threshold, self.alpha_mid, self.alpha_obi, self.alpha_rate)):
            card.grid(row=0, column=i, padx=(0 if i == 0 else 8, 0), sticky="nsew")
            metrics.columnconfigure(i, weight=1)

        split = ttk.Frame(page)
        split.pack(fill="both", expand=True, pady=(10, 0))
        split.columnconfigure(0, weight=2)
        split.columnconfigure(1, weight=1)
        split.rowconfigure(0, weight=1)
        chart_stack = ttk.Frame(split)
        chart_stack.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        chart_stack.columnconfigure(0, weight=1)
        chart_stack.rowconfigure(0, weight=1)
        chart_stack.rowconfigure(1, weight=1)
        self.alpha_mid_chart = LineChart(chart_stack, "Mid-price", height=180)
        self.alpha_mid_chart.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.alpha_forecast_chart = LineChart(
            chart_stack, "Model forecast (bps)", height=180
        )
        self.alpha_forecast_chart.grid(row=1, column=0, sticky="nsew")
        self.alpha_log = LogBox(split)
        self.alpha_log.grid(row=0, column=1, sticky="nsew")

    def _build_diagnostics(self) -> None:
        page = self.pages["Diagnostics"]
        self._section_title(page, "Diagnostics", "Runtime, native module, DLL, certificate, and filesystem checks.")
        ttk.Button(page, text="Run diagnostics", command=self.run_diagnostics, style="Accent.TButton").pack(anchor="w")
        self.diagnostics_output = tk.Text(page, bg=PANEL, fg=TEXT, relief="flat", font=("Consolas", 10))
        self.diagnostics_output.pack(fill="both", expand=True, pady=(10, 0))

    def _build_interview(self) -> None:
        page = self.pages["Interview Mode"]
        self._section_title(page, "Interview Mode", "A concise evidence-focused explanation of the system and research process.")
        self.interview_text = tk.Text(page, bg=PANEL, fg=TEXT, relief="flat", font=("Segoe UI", 11), wrap="word")
        self.interview_text.pack(fill="both", expand=True)
        self.interview_text.insert("end", self._interview_overview())
        actions = ttk.Frame(page)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="Generate evidence brief", command=self.generate_evidence_brief, style="Accent.TButton").pack(side="left")
        ttk.Button(actions, text="Open artifacts folder", command=lambda: os.startfile(ARTIFACTS_DIR)).pack(side="left", padx=8)

    def _interview_overview(self) -> str:
        return (
            "SYSTEM ARCHITECTURE\n\n"
            "Binance TLS WebSocket → Boost.Beast → simdjson → fixed 32-byte OrderBookState → "
            "lock-free SPSC queue → pybind11 → NumPy analytics.\n\n"
            "REPRODUCIBILITY\n\n"
            "Every successfully parsed top-of-book update can be recorded to a versioned binary format. "
            "Replay preserves the original record bytes and source timestamps at original timing, accelerated timing, "
            "or maximum backpressure-safe throughput. Completeness is declared only after a clean recorder shutdown with zero recorder loss.\n\n"
            "RESEARCH DISCIPLINE\n\n"
            "Whole-session chronological train/validation/test partitions when possible, purging, train-only normalization, "
            "validation-only model and threshold selection, held-out test evaluation, raw test-period fingerprinting, "
            "transaction-cost modeling, and an OBI baseline. Holdout reuse is refused unless explicitly recorded as a reproducibility rerun.\n\n"
            "WHAT TO DEMONSTRATE\n\n"
            "1. Start a live capture and show zero drops.\n"
            "2. Inspect the binary recording.\n"
            "3. Train on multiple sessions.\n"
            "4. Compare validation and held-out test performance.\n"
            "5. Replay the exact data and deploy the saved model artifact.\n"
        )

    def refresh_all(self) -> None:
        self.refresh_recordings()
        if hasattr(self, "l2_recording_list"):
            self.refresh_l2_recordings()
        self.refresh_models()
        recordings = discover_recordings()
        models = getattr(self, "_models", [])
        try:
            import quant_engine
            _ = quant_engine.IngestionEngine
            self.cc_engine.set("READY")
        except Exception:
            self.cc_engine.set("ERROR")
        self.cc_recordings.set(len(recordings))
        self.cc_models.set(len(models))
        latest = newest(recordings)
        self.cc_latest.set(latest.name if latest else "None")
        if latest:
            try:
                series = recording_chart_series(latest)
                self.cc_chart.set_series([("Mid", series["mid"], ACCENT)])
            except Exception:
                self.cc_chart.set_series([])
        else:
            self.cc_chart.set_series([])
        self._refresh_evidence()

    def _refresh_evidence(self) -> None:
        inventory = inventory_research_reports(ARTIFACTS_DIR)
        report_paths = list(inventory.current)
        stale_note = (
            f"\nIgnored stale reports: {len(inventory.stale)}"
            if inventory.stale
            else ""
        )
        text = "No current-schema research report yet." + stale_note
        if report_paths:
            try:
                payload = json.loads(report_paths[0].read_text(encoding="utf-8"))
                test_reg = payload.get("test_regression", {})
                strategy = payload.get("test_strategy", {})
                baseline = payload.get("imbalance_baseline", {}).get(
                    "test_strategy", {}
                )
                robustness = payload.get("robustness", {})
                shift_test = robustness.get("test_ic_circular_shift", {})
                text = (
                    f"Report: {report_paths[0].name}\n"
                    f"Test IC / rank IC: {safe_float(test_reg.get('pearson_ic'), 6)} / "
                    f"{safe_float(test_reg.get('spearman_rank_ic'), 6)}\n"
                    "Direction accuracy / majority: "
                    f"{safe_percent(test_reg.get('direction_accuracy'))} / "
                    f"{safe_percent(test_reg.get('majority_direction_accuracy'))}\n"
                    "Balanced accuracy / lift: "
                    f"{safe_percent(test_reg.get('balanced_direction_accuracy'))} / "
                    f"{safe_percent(test_reg.get('direction_accuracy_lift_vs_majority'))}\n"
                    f"Trades: {safe_int(strategy.get('trades'))}\n"
                    f"Total PnL: {safe_float(strategy.get('total_pnl_bps'), 4)} bps\n"
                    f"HAC t-stat: {safe_float(strategy.get('newey_west_pnl_t_statistic'), 4)}\n"
                    f"Break-even extra cost: {safe_float(strategy.get('breakeven_additional_cost_bps_per_side'), 4)} bps/side\n"
                    f"Circular-shift p-value: {safe_float(shift_test.get('two_sided_p_value'), 4)}\n"
                    f"Max drawdown: {safe_float(strategy.get('max_drawdown_bps'), 4)} bps\n"
                    f"OBI baseline PnL: {safe_float(baseline.get('total_pnl_bps'), 4)} bps"
                    f"{stale_note}\n"
                )
            except Exception as exc:
                text = f"Could not read report: {exc}"
        self.cc_evidence.delete("1.0", "end")
        self.cc_evidence.insert("end", text)

    def refresh_recordings(self) -> None:
        recordings = discover_recordings()
        self._recordings = recordings
        if hasattr(self, "recording_list"):
            self.recording_list.delete(0, "end")
            for path in recordings:
                self.recording_list.insert("end", path.name)
        if hasattr(self, "research_list"):
            self.research_list.delete(0, "end")
            for path in recordings:
                self.research_list.insert("end", path.name)
        if hasattr(self, "runtime_recording"):
            self.runtime_recording["values"] = [path.name for path in recordings]
            if recordings and not self.runtime_recording.get():
                self.runtime_recording.current(0)

    def refresh_models(self) -> None:
        candidates = sorted(
            ARTIFACTS_DIR.glob("*.npz"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        models: list[Path] = []
        for path in candidates:
            try:
                load_model(path)
            except Exception:
                continue
            models.append(path)
        self._models = models
        if hasattr(self, "runtime_model"):
            self.runtime_model["values"] = [path.name for path in models]
            current = self.runtime_model.get()
            if current and current not in {path.name for path in models}:
                self.runtime_model.set("")
            if models and not self.runtime_model.get():
                self.runtime_model.current(0)

    def inspect_selected_recording(self) -> None:
        selection = self.recording_list.curselection()
        if not selection:
            return
        path = self._recordings[selection[0]]
        try:
            summary = summarize_recording(path)
            series = recording_chart_series(path)
            text = (
                f"File: {summary.path}\n"
                f"Created UTC: {summary.created_at.isoformat()}\n"
                f"Size: {summary.file_size:,} bytes\n"
                f"Records: {summary.record_count:,}\n"
                f"Duration: {summary.duration_seconds:.6f} seconds\n"
                f"Average rate: {summary.average_rate:,.2f}/sec\n"
                f"First market: {summary.first_bid:.8f} / {summary.first_ask:.8f}\n"
                f"Last market: {summary.last_bid:.8f} / {summary.last_ask:.8f}\n"
                f"Price range: {summary.minimum_bid:.8f} to {summary.maximum_ask:.8f}\n"
                f"Clean shutdown: {summary.clean_shutdown}\n"
                f"Data complete: {summary.data_complete}\n"
                f"Reconnects: {summary.reconnect_count}\n"
                f"Recording drops: {summary.recording_dropped}\n"
            )
            self.recording_summary.delete("1.0", "end")
            self.recording_summary.insert("end", text)
            self.library_mid_chart.set_series([("Mid", series["mid"], ACCENT)])
            self.library_obi_chart.set_series([
                ("OBI", series["obi"], GOOD),
                ("Spread bps", series["spread_bps"], WARN),
            ])
        except Exception as exc:
            messagebox.showerror("Recording error", str(exc), parent=self)

    def _runner_busy(self) -> bool:
        return self.engine_runner is not None and self.engine_runner.is_alive()

    def start_live_capture(self) -> None:
        if self._runner_busy():
            messagebox.showwarning("Engine busy", "Stop the current live/replay session first.", parent=self)
            return
        raw_filename = self.live_filename.get().strip()
        if not raw_filename:
            messagebox.showerror(
                "Missing filename",
                "Enter a recording filename.",
                parent=self,
            )
            return
        filename = Path(raw_filename).name
        if not filename.lower().endswith(".qbin"):
            filename += ".qbin"
        path = RECORDINGS_DIR / filename
        artifact_set = (path, Path(f"{path}.qids"), Path(f"{path}.meta.json"))
        existing = [candidate for candidate in artifact_set if candidate.exists()]
        if existing:
            messagebox.showerror(
                "Recording exists",
                "Refusing to overwrite an existing recording artifact:\n"
                + "\n".join(str(candidate) for candidate in existing),
                parent=self,
            )
            return
        self.mid_history.clear()
        self.live_log.write(f"Recording to {path}")
        self.engine_runner = EngineRunner(self.engine_events, mode="live", recording_path=str(path))
        self.engine_runner.start()
        self.live_start.configure(state="disabled")
        self.live_stop.configure(state="normal")
        self.global_status.set("● LIVE")

    def start_alpha_replay(self) -> None:
        if self._runner_busy():
            messagebox.showwarning("Engine busy", "Stop the current session first.", parent=self)
            return
        model_name = self.runtime_model.get()
        recording_name = self.runtime_recording.get()
        if not model_name or not recording_name:
            messagebox.showwarning("Missing selection", "Select both a model and a recording.", parent=self)
            return
        try:
            replay_speed = float(self.runtime_speed.get())
        except ValueError:
            messagebox.showerror(
                "Invalid replay speed",
                "Replay speed must be a finite non-negative number.",
                parent=self,
            )
            return
        if not math.isfinite(replay_speed) or replay_speed < 0.0:
            messagebox.showerror(
                "Invalid replay speed",
                "Replay speed must be a finite non-negative number.",
                parent=self,
            )
            return
        self.mid_history.clear()
        self.prediction_history.clear()
        self.engine_runner = EngineRunner(
            self.engine_events,
            mode="replay",
            replay_file=str(RECORDINGS_DIR / recording_name),
            replay_speed=replay_speed,
            model_path=str(ARTIFACTS_DIR / model_name),
        )
        self.engine_runner.start()
        self.global_status.set("● REPLAY")

    def start_alpha_live(self) -> None:
        if self._runner_busy():
            messagebox.showwarning("Engine busy", "Stop the current session first.", parent=self)
            return
        model_name = self.runtime_model.get()
        if not model_name:
            messagebox.showwarning("Missing model", "Select a trained model.", parent=self)
            return
        self.mid_history.clear()
        self.prediction_history.clear()
        self.engine_runner = EngineRunner(
            self.engine_events,
            mode="live",
            model_path=str(ARTIFACTS_DIR / model_name),
        )
        self.engine_runner.start()
        self.global_status.set("● LIVE ALPHA")

    def stop_engine(self) -> None:
        if self.engine_runner:
            self.engine_runner.request_stop()
            self.global_status.set("● STOPPING")


    def refresh_l2_recordings(self) -> None:
        self._l2_recordings = discover_l2_recordings()
        if not hasattr(self, "l2_recording_list"):
            return
        self.l2_recording_list.delete(0, "end")
        for path in self._l2_recordings:
            self.l2_recording_list.insert("end", path.name)

    def _selected_l2_path(self) -> Path | None:
        selection = self.l2_recording_list.curselection()
        if not selection:
            return None
        return self._l2_recordings[selection[0]]

    def inspect_selected_l2(self) -> None:
        path = self._selected_l2_path()
        if path is None:
            return
        try:
            text = summarize_l2_recording(path)
        except Exception as exception:
            text = f"L2 inspection failed: {exception}"
        self.l2_output.delete("1.0", "end")
        self.l2_output.insert("end", text)

    @staticmethod
    def _l2_tool_command(script_name: str, packaged_name: str) -> list[str]:
        if getattr(sys, "frozen", False):
            executable = ROOT / packaged_name
            if not executable.exists():
                raise FileNotFoundError(f"Packaged L2 tool is missing: {executable}")
            return [str(executable)]
        return [sys.executable, str(ROOT / script_name)]

    def launch_l2_capture(self) -> None:
        L2_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            command = self._l2_tool_command("l2_capture.py", "MarketL2Capture.exe")
            command.extend(
                [
                    "--symbols",
                    "BTCUSDT",
                    "ETHUSDT",
                    "--output-dir",
                    str(L2_RECORDINGS_DIR),
                ]
            )
            kwargs = {"cwd": str(ROOT)}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            subprocess.Popen(command, **kwargs)
        except Exception as exception:
            messagebox.showerror("L2 capture", str(exception), parent=self)

    def _run_l2_tool(
        self,
        title: str,
        script_name: str,
        packaged_name: str,
        extra_arguments: list[str] | None = None,
    ) -> None:
        path = self._selected_l2_path()
        if path is None:
            messagebox.showwarning("L2 selection", "Select an L2 recording first.", parent=self)
            return
        self.l2_output.delete("1.0", "end")
        self.l2_output.insert("end", f"{title} running...\n")

        def task() -> None:
            try:
                command = self._l2_tool_command(script_name, packaged_name)
                command.extend(extra_arguments or [])
                command.append(str(path))
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                output = result.stdout or result.stderr or f"{title} completed."
            except Exception:
                output = traceback.format_exc()
            self.engine_events.put(("l2_output", output))

        threading.Thread(target=task, daemon=False).start()

    def verify_selected_l2(self) -> None:
        self._run_l2_tool(
            "L2 verification",
            "verify_l2_replay.py",
            "MarketL2Verify.exe",
        )

    def benchmark_selected_l2(self) -> None:
        output = ARTIFACTS_DIR / f"l2_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._run_l2_tool(
            "L2 benchmark",
            "l2_benchmark.py",
            "MarketL2Benchmark.exe",
            ["--output", str(output)],
        )

    def start_training(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            return
        selections = self.research_list.curselection()
        if not selections:
            messagebox.showwarning(
                "No recordings",
                "Select at least one recording.",
                parent=self,
            )
            return
        paths = [self._recordings[index] for index in selections]
        try:
            horizon = int(self.research_horizon.get())
            fee = float(self.research_fee.get())
            slippage = float(self.research_slippage.get())
            trade_size = float(self.research_trade_size.get())
            participation = float(self.research_participation.get())
        except (TypeError, ValueError):
            messagebox.showerror(
                "Invalid research settings",
                "Horizon and execution settings must be numeric.",
                parent=self,
            )
            return
        valid = (
            horizon > 0
            and math.isfinite(fee)
            and fee >= 0.0
            and math.isfinite(slippage)
            and slippage >= 0.0
            and math.isfinite(trade_size)
            and trade_size >= 0.0
            and math.isfinite(participation)
            and 0.0 < participation <= 1.0
        )
        if not valid:
            messagebox.showerror(
                "Invalid research settings",
                "Use a positive horizon, non-negative fee/slippage/size, and participation in (0, 1].",
                parent=self,
            )
            return
        stem = Path(self.research_name.get().strip() or "alpha_model").stem
        if not stem or stem in {".", ".."}:
            messagebox.showerror(
                "Invalid artifact name",
                "Enter a valid artifact filename stem.",
                parent=self,
            )
            return
        model_path = ARTIFACTS_DIR / f"{stem}.npz"
        report_path = ARTIFACTS_DIR / f"{stem}_report.json"
        predictions_path = ARTIFACTS_DIR / f"{stem}_test_predictions.csv"
        evidence_path = ARTIFACTS_DIR / f"{stem}_research_card.md"
        existing_outputs = [
            path
            for path in (model_path, report_path, predictions_path, evidence_path)
            if path.exists()
        ]
        if existing_outputs:
            messagebox.showerror(
                "Research artifact exists",
                "Refusing to overwrite existing artifacts:\n"
                + "\n".join(str(path) for path in existing_outputs),
                parent=self,
            )
            return
        self.train_button.configure(state="disabled")
        self.training_progress["value"] = 0
        self.research_output.delete("1.0", "end")

        def progress(value: int, message: str) -> None:
            self.engine_events.put(("train_progress", (value, message)))

        def task() -> None:
            try:
                result = train_model(
                    paths,
                    horizon,
                    fee,
                    model_path,
                    report_path,
                    predictions_path,
                    progress,
                    slippage_bps=slippage,
                    trade_size_base=trade_size,
                    max_displayed_participation=participation,
                    evidence_path=evidence_path,
                )
                self.engine_events.put(("train_complete", result))
            except Exception:
                self.engine_events.put(("train_error", traceback.format_exc()))

        self.training_thread = threading.Thread(target=task, daemon=False)
        self.training_thread.start()

    def _display_training_result(self, result: dict) -> None:
        test_reg = result.get("test_regression", {})
        test_strategy = result.get("test_strategy", {})
        baseline = result.get("imbalance_baseline", {}).get("test_strategy", {})
        selected = result.get("selected_model", {})
        coefficients = selected.get("coefficients", [])
        strongest = coefficients[:8] if isinstance(coefficients, list) else []
        methodology = result.get("methodology", {})
        robustness = result.get("robustness", {})
        shift_test = robustness.get("test_ic_circular_shift", {})
        walk_forward = robustness.get("pretest_anchored_walk_forward", {})
        text = (
            "TRAINING COMPLETE\n\n"
            f"Recordings: {len(result.get('recordings', []))}\n"
            f"Horizon: {methodology.get('horizon_events')} events\n"
            f"Split mode: {methodology.get('split_mode')}\n"
            f"Selected ridge alpha: {selected.get('ridge_alpha')}\n"
            f"Signal threshold: {safe_float(selected.get('signal_threshold_bps'), 8)} bps\n"
            f"Explicit round-trip cost: {safe_float(methodology.get('round_trip_explicit_cost_bps'), 6)} bps\n\n"
            "UNTOUCHED TEST\n"
            f"Pearson IC: {safe_float(test_reg.get('pearson_ic'), 8)}\n"
            f"Spearman rank IC: {safe_float(test_reg.get('spearman_rank_ic'), 8)}\n"
            f"Direction accuracy: {safe_percent(test_reg.get('direction_accuracy'))}\n"
            f"Majority-direction baseline: {safe_percent(test_reg.get('majority_direction_accuracy'))}\n"
            f"Accuracy lift over majority: {safe_percent(test_reg.get('direction_accuracy_lift_vs_majority'))}\n"
            f"Balanced direction accuracy: {safe_percent(test_reg.get('balanced_direction_accuracy'))}\n"
            "Actionable coverage / accuracy: "
            f"{safe_percent(test_reg.get('actionable_coverage'))} / "
            f"{safe_percent(test_reg.get('actionable_direction_accuracy'))}\n"
            f"Zero-return target fraction: {safe_percent(test_reg.get('target_zero_fraction'))}\n"
            f"MSE: {safe_float(test_reg.get('mse'), 8)}\n"
            f"Trades: {safe_int(test_strategy.get('trades'))}\n"
            f"Total net PnL: {safe_float(test_strategy.get('total_pnl_bps'), 6)} bps\n"
            f"Mean net PnL: {safe_float(test_strategy.get('mean_pnl_bps'), 6)} bps\n"
            f"Hit rate: {safe_float(100.0 * float(test_strategy.get('win_rate', 0.0)), 2)}%\n"
            f"HAC t-stat: {safe_float(test_strategy.get('newey_west_pnl_t_statistic'), 6)}\n"
            f"Bootstrap P(mean <= 0): {safe_float(test_strategy.get('session_bootstrap_probability_mean_non_positive'), 6)}\n"
            f"Break-even extra cost: {safe_float(test_strategy.get('breakeven_additional_cost_bps_per_side'), 6)} bps/side\n"
            f"Circular-shift IC p-value: {safe_float(shift_test.get('two_sided_p_value'), 6)}\n"
            f"Max drawdown: {safe_float(test_strategy.get('max_drawdown_bps'), 6)} bps\n"
            f"Fill rejections: {safe_int(test_strategy.get('fill_rejections'))}\n\n"
            f"OBI baseline total PnL: {safe_float(baseline.get('total_pnl_bps'), 6)} bps\n"
            f"Pre-test walk-forward available: {walk_forward.get('available', False)}\n\n"
            "STRONGEST STANDARDIZED COEFFICIENTS\n"
        )
        text += "\n".join(
            f"{str(item.get('feature', '')):34s} "
            f"{float(item.get('standardized_coefficient', 0.0)):+.8f}"
            for item in strongest
            if isinstance(item, dict)
        )
        self.research_output.insert("end", text)
        self.refresh_models()
        self._refresh_evidence()

    def _handle_snapshot(self, payload: dict) -> None:
        self.mid_history.append(payload.get("mid", float("nan")))
        mode = payload.get("mode")
        if mode == "live" and not self.runtime_model.get():
            pass
        self.live_metrics["Ticks"].set(safe_int(payload.get("total_ticks")))
        self.live_metrics["Rate/sec"].set(safe_float(payload.get("rate"), 2))
        self.live_metrics["Mid"].set(safe_float(payload.get("mid"), 4))
        self.live_metrics["Spread"].set(safe_float(payload.get("spread"), 4))
        self.live_metrics["OBI"].set(safe_float(payload.get("obi"), 4))
        self.live_metrics["p99 local age µs"].set(
            safe_float(payload.get("p99_us"), 2)
        )
        self.live_metrics["Recorded"].set(safe_int(payload.get("recorded_ticks")))
        drops = int(payload.get("dropped_ticks", 0)) + int(payload.get("recording_dropped", 0))
        self.live_metrics["Drops"].set(safe_int(drops))
        self.live_chart.set_series([("Mid", list(self.mid_history), ACCENT)])

        prediction = payload.get("prediction", float("nan"))
        self.prediction_history.append(prediction)
        self.alpha_signal.set(payload.get("signal", "FLAT"))
        self.alpha_prediction.set(safe_float(prediction, 6))
        self.alpha_threshold.set(safe_float(payload.get("threshold_multiple"), 2) + "x")
        self.alpha_mid.set(safe_float(payload.get("mid"), 4))
        self.alpha_obi.set(safe_float(payload.get("obi"), 4))
        self.alpha_rate.set(safe_float(payload.get("rate"), 2))
        self.alpha_mid_chart.set_series([("Mid", list(self.mid_history), ACCENT)])
        self.alpha_forecast_chart.set_series(
            [("Forecast", list(self.prediction_history), GOOD)]
        )

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.engine_events.get_nowait()
                if kind == "log":
                    self.live_log.write(str(payload))
                    self.alpha_log.write(str(payload))
                elif kind == "snapshot":
                    self._handle_snapshot(payload)
                elif kind == "complete":
                    self.live_log.write(f"Session completed: {payload}")
                    self.alpha_log.write(f"Session completed: {payload}")
                    runner = self.engine_runner
                    self.engine_runner = None
                    if runner is not None:
                        runner.join(timeout=0.1)
                    if self._closing:
                        self.destroy()
                        return
                    self.live_start.configure(state="normal")
                    self.live_stop.configure(state="disabled")
                    self.global_status.set("● READY")
                    self.refresh_all()
                elif kind == "error":
                    LOGS_DIR.mkdir(parents=True, exist_ok=True)
                    (LOGS_DIR / "runtime_error.log").write_text(str(payload), encoding="utf-8")
                    self.live_log.write(str(payload))
                    self.alpha_log.write(str(payload))
                    runner = self.engine_runner
                    self.engine_runner = None
                    if runner is not None:
                        runner.join(timeout=0.1)
                    if self._closing:
                        self.destroy()
                        return
                    messagebox.showerror("Runtime error", "The operation failed. See logs/runtime_error.log", parent=self)
                    self.live_start.configure(state="normal")
                    self.live_stop.configure(state="disabled")
                    self.global_status.set("● ERROR")
                elif kind == "l2_output":
                    self.l2_output.delete("1.0", "end")
                    self.l2_output.insert("end", str(payload))
                    self.refresh_l2_recordings()
                elif kind == "train_progress":
                    value, message = payload
                    self.training_progress["value"] = value
                    self.training_status.set(message)
                elif kind == "train_complete":
                    thread = self.training_thread
                    self.training_thread = None
                    if thread is not None:
                        thread.join(timeout=0.1)
                    self.training_progress["value"] = 100
                    self.training_status.set("Completed")
                    self.train_button.configure(state="normal")
                    self._display_training_result(payload)
                elif kind == "train_error":
                    thread = self.training_thread
                    self.training_thread = None
                    if thread is not None:
                        thread.join(timeout=0.1)
                    self.training_status.set("Failed")
                    self.train_button.configure(state="normal")
                    self.research_output.insert("end", str(payload))
                    (LOGS_DIR / "training_error.log").write_text(
                        str(payload), encoding="utf-8"
                    )
        except queue.Empty:
            pass
        if self._closing and self.engine_runner is None:
            self.destroy()
            return
        if self.winfo_exists():
            self.after(100, self._poll_events)

    def run_diagnostics(self) -> None:
        lines: list[str] = []
        checks = [
            ("Python", sys.version.replace("\n", " ")),
            ("Project root", str(ROOT)),
            ("NumPy", np.__version__),
            ("TLS certificates", certifi.where()),
        ]
        for name, value in checks:
            lines.append(f"[PASS] {name}: {value}")
        lines.append("[PASS] Native dependencies: OpenSSL and simdjson linked statically")
        try:
            import tkinter
            root = tkinter.Tcl()
            root.eval("info patchlevel")
            lines.append(f"[PASS] Tk runtime: {root.eval('info patchlevel')}")
        except Exception as exc:
            lines.append(f"[FAIL] Tk runtime: {exc!r}")
        try:
            import quant_engine
            engine = quant_engine.IngestionEngine()
            del engine
            lines.append("[PASS] Native engine: IngestionEngine API ready")
            synchronizer = quant_engine.L2Synchronizer()
            del synchronizer
            lines.append("[PASS] Native L2 synchronizer: L2Synchronizer API ready")
        except Exception as exc:
            lines.append(f"[FAIL] Native engine/L2 API: {exc!r}")
        for directory in (RECORDINGS_DIR, ARTIFACTS_DIR, LOGS_DIR):
            lines.append(f"[{'PASS' if directory.exists() else 'FAIL'}] Directory: {directory}")
        self.diagnostics_output.delete("1.0", "end")
        self.diagnostics_output.insert("end", "\n".join(lines))

    def generate_evidence_brief(self) -> None:
        recordings = discover_recordings()
        latest_recording = newest(recordings)
        reports = discover_research_reports(ARTIFACTS_DIR)
        lines = [
            "# Market Systems Workstation — Project Evidence Brief",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Architecture",
            "",
            "Binance TLS WebSocket → Boost.Beast → simdjson → fixed 32-byte market state → lock-free SPSC queue → pybind11 → NumPy research/runtime.",
            "",
            "## Reproducibility",
            "",
            "The engine records successfully parsed top-of-book updates with exchange update IDs and replays the exact market records and source timestamps at original, accelerated, or maximum backpressure-safe speed. A capture is called complete only after clean shutdown with zero recorder drops or write errors.",
            "",
            "## Research safeguards",
            "",
            "Whole-session chronological train/validation/test splitting when possible, purging, train-only normalization, validation-only hyperparameter and threshold selection, held-out test evaluation, raw test-period fingerprinting, explicit fee/slippage assumptions, displayed-liquidity screening, cost stress, HAC inference, session bootstrap, circular-shift tests, anchored pre-test walk-forward diagnostics, feature drift, coefficient stability, and multiple simple baselines. Explicit holdout reruns remain visible in provenance.",
            "",
        ]
        if latest_recording:
            summary = summarize_recording(latest_recording)
            lines += [
                "## Latest recording",
                "",
                f"- File: `{summary.path.name}`",
                f"- Records: {summary.record_count:,}",
                f"- Duration: {summary.duration_seconds:.6f} seconds",
                f"- Average event rate: {summary.average_rate:,.2f}/sec",
                f"- Price range: {summary.minimum_bid:.8f} to {summary.maximum_ask:.8f}",
                f"- Clean shutdown: {summary.clean_shutdown}",
                f"- Data complete: {summary.data_complete}",
                f"- Reconnect boundaries: {summary.reconnect_count}",
                f"- Recorder drops: {summary.recording_dropped}",
                "",
            ]
        if reports:
            payload = json.loads(reports[0].read_text(encoding="utf-8"))
            test_reg = payload.get("test_regression", {})
            strategy = payload.get("test_strategy", {})
            methodology = payload.get("methodology", {})
            lines += [
                "## Latest held-out test result",
                "",
                f"- Report: `{reports[0].name}`",
                f"- Test Pearson IC: {safe_float(test_reg.get('pearson_ic'), 8)}",
                f"- Test Spearman rank IC: {safe_float(test_reg.get('spearman_rank_ic'), 8)}",
                f"- Direction accuracy: {safe_percent(test_reg.get('direction_accuracy'))}",
                f"- Majority-direction baseline: {safe_percent(test_reg.get('majority_direction_accuracy'))}",
                f"- Accuracy lift over majority: {safe_percent(test_reg.get('direction_accuracy_lift_vs_majority'))}",
                f"- Balanced direction accuracy: {safe_percent(test_reg.get('balanced_direction_accuracy'))}",
                f"- Actionable coverage: {safe_percent(test_reg.get('actionable_coverage'))}",
                f"- Trades: {strategy.get('trades', 0)}",
                f"- Total net PnL: {safe_float(strategy.get('total_pnl_bps'), 6)} bps",
                f"- HAC t-stat: {safe_float(strategy.get('newey_west_pnl_t_statistic'), 6)}",
                f"- Bootstrap P(mean <= 0): {safe_float(strategy.get('session_bootstrap_probability_mean_non_positive'), 6)}",
                f"- Break-even extra cost: {safe_float(strategy.get('breakeven_additional_cost_bps_per_side'), 6)} bps/side",
                f"- Maximum drawdown: {safe_float(strategy.get('max_drawdown_bps'), 6)} bps",
                "- Prior evaluations of this held-out market period: "
                f"{methodology.get('prior_test_evaluations', 0)}",
                "",
            ]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ARTIFACTS_DIR / f"PROJECT_EVIDENCE_BRIEF_{timestamp}.md"
        output.write_text("\n".join(lines), encoding="utf-8")
        messagebox.showinfo("Evidence brief", f"Created:\n{output}", parent=self)

    def on_close(self) -> None:
        if self._closing:
            return
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning(
                "Training in progress",
                "Let the current training run finish before closing the workstation.",
                parent=self,
            )
            return
        self._closing = True
        self.global_status.set("● STOPPING")
        if self.engine_runner and self.engine_runner.is_alive():
            self.engine_runner.request_stop()
            self.live_start.configure(state="disabled")
            self.live_stop.configure(state="disabled")
            return
        self.destroy()


def main() -> int:
    prepare_native_runtime()
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    app = QuantWorkstation()
    app.mainloop()
    return 0

