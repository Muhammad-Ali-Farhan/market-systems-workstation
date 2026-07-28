
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import certifi
import numpy as np

from native_runtime import prepare_native_runtime

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
prepare_native_runtime()

import quant_engine  # noqa: E402
from qbin import (  # noqa: E402
    open_records,
    open_update_ids,
    read_metadata,
    validate_records,
    validate_update_ids,
)

BATCH_SIZE = 4096
REPORT_INTERVAL_SECONDS = 5.0
VOLUME_SCALE = 1_000_000.0
MAX_LATENCY_SAMPLES_PER_REPORT = 200_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect live ingestion or deterministic binary replay."
    )
    parser.add_argument("mode", nargs="?", choices=("live", "replay"), default="live")
    parser.add_argument("--record", default="")
    parser.add_argument("--file", default="")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow explicit inspection of a recording marked incomplete.",
    )
    arguments = parser.parse_args()
    if arguments.mode == "replay" and not arguments.file:
        parser.error("--file is required in replay mode.")
    if arguments.mode == "replay" and arguments.record:
        parser.error("--record is only valid in live mode.")
    if arguments.mode == "live" and arguments.file:
        parser.error("--file is only valid in replay mode.")
    if arguments.mode == "live" and arguments.allow_incomplete:
        parser.error("--allow-incomplete is only valid in replay mode.")
    if not math.isfinite(arguments.speed) or arguments.speed < 0.0:
        parser.error("--speed must be finite and non-negative.")
    return arguments


def add_latency_sample(
    batches: list[np.ndarray],
    current_count: int,
    values: np.ndarray,
) -> int:
    remaining = MAX_LATENCY_SAMPLES_PER_REPORT - current_count
    if remaining <= 0:
        return current_count
    if values.size <= remaining:
        sample = values.copy()
    else:
        indices = np.linspace(0, values.size - 1, remaining, dtype=np.int64)
        sample = values[indices].copy()
    batches.append(sample)
    return current_count + int(sample.size)


def main() -> None:
    arguments = parse_arguments()
    engine = quant_engine.IngestionEngine()
    dtype = np.dtype(quant_engine.order_book_dtype)
    if dtype.itemsize != 32:
        raise RuntimeError(f"Expected 32-byte records, received {dtype.itemsize}.")
    buffer = np.empty(BATCH_SIZE, dtype=dtype)

    if arguments.mode == "live":
        print("[Initialization] Connecting to Binance BTCUSDT bookTicker...")
        if arguments.record:
            recording_path = Path(arguments.record).expanduser().resolve()
            if recording_path.suffix.lower() != ".qbin":
                raise ValueError("Recording destination must use the .qbin suffix.")
            existing = [
                candidate
                for candidate in (
                    recording_path,
                    Path(f"{recording_path}.qids"),
                    Path(f"{recording_path}.meta.json"),
                )
                if candidate.exists()
            ]
            if existing:
                raise FileExistsError(
                    "Refusing to overwrite recording artifacts:\n"
                    + "\n".join(f"  {candidate}" for candidate in existing)
                )
            arguments.record = str(recording_path)
            print(f"[Recorder] Writing to: {recording_path}")
        engine.start_live(arguments.record)
    else:
        replay_metadata = read_metadata(arguments.file)
        if replay_metadata.data_complete is False and not arguments.allow_incomplete:
            raise RuntimeError(
                "Recording is marked incomplete. Use --allow-incomplete only "
                "for an explicit diagnostic replay."
            )
        replay_records = open_records(arguments.file, replay_metadata)
        validate_records(replay_records, context=str(replay_metadata.path))
        validate_update_ids(
            open_update_ids(replay_metadata),
            replay_metadata,
            context=str(replay_metadata.path),
        )
        print(f"[Initialization] Replaying: {Path(arguments.file).resolve()}")
        print(
            "[Replay] Mode: "
            + ("maximum lossless speed" if arguments.speed == 0.0 else f"{arguments.speed:g}x")
        )
        engine.start_replay(
            arguments.file, arguments.speed, arguments.allow_incomplete
        )

    total_ticks = previous_total = 0
    start = previous_report = time.perf_counter()
    latency_batches: list[np.ndarray] = []
    latency_count = 0
    latest_bid = latest_ask = latest_spread = latest_mid = latest_imbalance = float("nan")

    try:
        while True:
            count = engine.consume_batch(buffer)
            if count:
                total_ticks += count
                data = buffer[:count]
                bid = data["best_bid"]
                ask = data["best_ask"]
                bid_volume = data["bid_volume"].astype(np.float64) / VOLUME_SCALE
                ask_volume = data["ask_volume"].astype(np.float64) / VOLUME_SCALE
                total_volume = bid_volume + ask_volume
                imbalance = np.divide(
                    bid_volume - ask_volume,
                    total_volume,
                    out=np.zeros_like(total_volume),
                    where=total_volume != 0.0,
                )
                latest_bid = float(bid[-1])
                latest_ask = float(ask[-1])
                latest_spread = latest_ask - latest_bid
                latest_mid = (latest_bid + latest_ask) / 2.0
                latest_imbalance = float(imbalance[-1])

                # Recorded timestamps share the current steady-clock epoch only
                # in live mode. Replay deliberately preserves source timestamps.
                if arguments.mode == "live":
                    timestamps = data["timestamp_ns"].astype(np.uint64, copy=False)
                    age_us = (engine.now_ns() - timestamps).astype(np.float64) / 1_000.0
                    latency_count = add_latency_sample(
                        latency_batches, latency_count, age_us
                    )
            else:
                if arguments.mode == "replay" and not engine.is_running():
                    break
                time.sleep(0.001)

            now = time.perf_counter()
            elapsed = now - previous_report
            if elapsed < REPORT_INTERVAL_SECONDS:
                continue
            interval_rate = (total_ticks - previous_total) / elapsed
            average_rate = total_ticks / max(now - start, 1e-9)

            if latency_batches:
                combined = np.concatenate(latency_batches)
                latency_text = (
                    f"p50 receipt-to-consumer: {np.percentile(combined, 50):,.2f} us | "
                    f"p99: {np.percentile(combined, 99):,.2f} us | "
                    f"p99.9: {np.percentile(combined, 99.9):,.2f} us"
                )
            else:
                latency_text = "receipt-to-consumer latency: N/A in replay"

            common = (
                f"Ticks: {total_ticks:,} | Rate: {interval_rate:,.2f}/sec | "
                f"Average: {average_rate:,.2f}/sec | Bid: {latest_bid:.2f} | "
                f"Ask: {latest_ask:.2f} | Spread: {latest_spread:.4f} | "
                f"Mid: {latest_mid:.4f} | OBI: {latest_imbalance:.4f} | {latency_text}"
            )
            if arguments.mode == "live":
                print(
                    common
                    + f" | Feed drops: {engine.dropped_ticks():,}"
                    + f" | Malformed: {engine.malformed_messages():,}"
                    + f" | Reconnects: {engine.reconnect_count():,}"
                    + f" | Recorded: {engine.recorded_ticks():,}"
                    + f" | Record drops: {engine.recording_dropped():,}"
                    + f" | Write errors: {engine.recording_write_errors():,}"
                )
            else:
                print(
                    common
                    + f" | Replayed: {engine.replayed_ticks():,}"
                    + f" | Backpressure episodes: {engine.replay_backpressure_events():,}"
                    + f" | Replay errors: {engine.replay_errors():,}"
                )

            latency_batches.clear()
            latency_count = 0
            previous_total = total_ticks
            previous_report = now
    except KeyboardInterrupt:
        print("\n[Termination] Stop requested.")
    finally:
        engine.stop()

    print(f"[Final] State: {engine.state()} | Consumed: {total_ticks:,}")
    engine_error = engine.last_error()
    if engine_error:
        print(f"[Final] Error: {engine_error}")
        raise RuntimeError(engine_error)
    if arguments.mode == "live" and engine.recording_write_errors() != 0:
        raise RuntimeError("The binary recorder reported a write/finalization error.")
    if arguments.mode == "live" and engine.recording_dropped() != 0:
        raise RuntimeError("The binary recorder dropped records; capture is incomplete.")
    if arguments.mode == "replay" and engine.replay_errors() != 0:
        raise RuntimeError("The binary replay reported an error.")


if __name__ == "__main__":
    main()

