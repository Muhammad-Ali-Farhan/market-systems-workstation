
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
from qbin import read_metadata  # noqa: E402


BATCH_SIZE = 4096
REPORT_INTERVAL_SECONDS = 5.0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a live Binance session for a fixed duration."
    )
    parser.add_argument("--record", required=True, help="Destination .qbin file.")
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=0.0,
        help="Auto-stop duration in minutes. Use 0 to run until interrupted.",
    )
    arguments = parser.parse_args()
    if (
        not math.isfinite(arguments.duration_minutes)
        or arguments.duration_minutes < 0.0
    ):
        parser.error("--duration-minutes must be finite and non-negative.")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    path = Path(arguments.record).expanduser().resolve()
    if path.suffix.lower() != ".qbin":
        raise ValueError("Recording destination must use the .qbin suffix.")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing recording: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    engine = quant_engine.IngestionEngine()
    dtype = np.dtype(quant_engine.order_book_dtype)
    if dtype.itemsize != 32:
        raise RuntimeError("Expected a 32-byte OrderBookState.")

    buffer = np.empty(BATCH_SIZE, dtype=dtype)
    deadline = (
        time.monotonic() + arguments.duration_minutes * 60.0
        if arguments.duration_minutes > 0.0
        else None
    )
    total_ticks = 0
    previous_ticks = 0
    start = time.monotonic()
    previous_report = start
    latest_bid = float("nan")
    latest_ask = float("nan")

    print(f"[Capture] Destination: {path}")
    print(
        "[Capture] Duration: manual stop"
        if deadline is None
        else f"[Capture] Duration: {arguments.duration_minutes:g} minutes"
    )
    engine.start_live(str(path))

    try:
        while engine.is_running():
            count = engine.consume_batch(buffer)
            if count:
                total_ticks += count
                latest_bid = float(buffer[count - 1]["best_bid"])
                latest_ask = float(buffer[count - 1]["best_ask"])
            else:
                time.sleep(0.001)

            now = time.monotonic()
            if deadline is not None and now >= deadline:
                print("[Capture] Requested duration completed.")
                break
            elapsed = now - previous_report
            if elapsed < REPORT_INTERVAL_SECONDS:
                continue

            interval_ticks = total_ticks - previous_ticks
            runtime = now - start
            message = (
                f"Ticks: {total_ticks:,} | "
                f"Rate: {interval_ticks / elapsed:,.2f}/sec | "
                f"Bid: {latest_bid:.2f} | Ask: {latest_ask:.2f} | "
                f"Accepted: {engine.recording_accepted():,} | "
                f"Recorded: {engine.recorded_ticks():,} | "
                f"Consumer Drops: {engine.dropped_ticks():,} | "
                f"Record Drops: {engine.recording_dropped():,} | "
                f"Malformed: {engine.malformed_messages():,} | "
                f"Reconnects: {engine.reconnect_count():,} | Runtime: {runtime:.1f}s"
            )
            if deadline is not None:
                message += f" | Remaining: {max(0.0, deadline - now):.1f}s"
            print(message)
            previous_ticks = total_ticks
            previous_report = now
    except KeyboardInterrupt:
        print("\n[Capture] Graceful stop requested.")
    finally:
        engine.stop()

    error = engine.last_error()
    print(
        f"[Capture Summary] Consumed: {total_ticks:,} | "
        f"Accepted: {engine.recording_accepted():,} | "
        f"Recorded: {engine.recorded_ticks():,} | "
        f"Consumer Drops: {engine.dropped_ticks():,} | "
        f"Record Drops: {engine.recording_dropped():,} | "
        f"Write Errors: {engine.recording_write_errors():,} | "
        f"Reconnects: {engine.reconnect_count():,}"
    )
    if error:
        raise RuntimeError(error)
    if engine.recording_write_errors() != 0:
        raise RuntimeError("The binary recorder reported a write/finalization error.")
    if engine.recording_dropped() != 0:
        raise RuntimeError("The binary recorder dropped records; capture is incomplete.")
    metadata = read_metadata(path)
    if metadata.data_complete is not True:
        raise RuntimeError("Capture did not finalize as a complete recording.")


if __name__ == "__main__":
    main()

