
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
from microstructure import AlphaModel, OnlineFeatureBuilder  # noqa: E402
from qbin import (  # noqa: E402
    feature_reset_indices,
    open_records,
    open_update_ids,
    read_metadata,
    validate_records,
    validate_update_ids,
)

BATCH_SIZE = 4096
REPORT_INTERVAL_SECONDS = 1.0
VOLUME_SCALE = 1_000_000.0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one canonical alpha artifact against live data or deterministic replay."
    )
    parser.add_argument("mode", choices=("live", "replay"))
    parser.add_argument("--model", default="artifacts/alpha_model.npz")
    parser.add_argument("--record", default="")
    parser.add_argument("--file", default="")
    parser.add_argument("--speed", type=float, default=1.0)
    arguments = parser.parse_args()
    if arguments.mode == "replay" and not arguments.file:
        parser.error("--file is required in replay mode.")
    if arguments.mode == "replay" and arguments.record:
        parser.error("--record is only valid in live mode.")
    if arguments.mode == "live" and arguments.file:
        parser.error("--file is only valid in replay mode.")
    if not math.isfinite(arguments.speed) or arguments.speed < 0.0:
        parser.error("--speed must be finite and non-negative.")
    return arguments


def classify_signal(prediction_bps: float, threshold_bps: float) -> str:
    if prediction_bps > 0.0 and prediction_bps >= threshold_bps:
        return "LONG"
    if prediction_bps < 0.0 and prediction_bps <= -threshold_bps:
        return "SHORT"
    return "FLAT"


def main() -> None:
    arguments = parse_arguments()
    model = AlphaModel.load(arguments.model)
    engine = quant_engine.IngestionEngine()
    dtype = np.dtype(quant_engine.order_book_dtype)
    if dtype.itemsize != 32:
        raise RuntimeError("Expected a 32-byte OrderBookState.")
    buffer = np.empty(BATCH_SIZE, dtype=dtype)
    feature_builder = OnlineFeatureBuilder(VOLUME_SCALE)
    replay_resets: frozenset[int] = frozenset()
    replay_record_index = 0

    if arguments.mode == "live":
        print("[Alpha Runtime] Starting live Binance BTCUSDT bookTicker feed")
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
            print(f"[Alpha Runtime] Recording to: {recording_path}")
        engine.start_live(arguments.record)
    else:
        replay_metadata = read_metadata(arguments.file)
        if replay_metadata.data_complete is False:
            raise RuntimeError(
                "Refusing to deploy a model against an incomplete recording."
            )
        replay_records = open_records(arguments.file, replay_metadata)
        validate_records(replay_records, context=str(replay_metadata.path))
        validate_update_ids(
            open_update_ids(replay_metadata),
            replay_metadata,
            context=str(replay_metadata.path),
        )
        replay_resets = feature_reset_indices(replay_metadata)
        print(f"[Alpha Runtime] Replaying: {Path(arguments.file).resolve()}")
        print(
            "[Alpha Runtime] Replay speed: "
            + ("maximum lossless" if arguments.speed == 0.0 else f"{arguments.speed:g}x")
        )
        engine.start_replay(arguments.file, arguments.speed)

    print(f"[Alpha Runtime] Model horizon: {model.horizon} events")
    print(f"[Alpha Runtime] Signal threshold: {model.signal_threshold_bps:.6f} bps")

    total_ticks = 0
    previous_ticks = 0
    previous_report = time.perf_counter()
    previous_drops = 0
    previous_reconnects = 0
    latest_prediction = float("nan")
    latest_bid = latest_ask = latest_mid = latest_imbalance = float("nan")
    warmed = False

    try:
        while True:
            count = engine.consume_batch(buffer)
            if count == 0:
                if arguments.mode == "replay" and not engine.is_running():
                    break
                time.sleep(0.001)
            else:
                total_ticks += count
                if arguments.mode == "live":
                    drops = int(engine.dropped_ticks())
                    reconnects = int(engine.reconnect_count())
                    if drops != previous_drops or reconnects != previous_reconnects:
                        feature_builder.reset()
                        warmed = False
                        previous_drops = drops
                        previous_reconnects = reconnects

                for record in buffer[:count]:
                    if (
                        arguments.mode == "replay"
                        and replay_record_index in replay_resets
                    ):
                        feature_builder.reset()
                        warmed = False
                    latest_bid = float(record["best_bid"])
                    latest_ask = float(record["best_ask"])
                    latest_mid = (latest_bid + latest_ask) / 2.0
                    features = feature_builder.update(
                        timestamp_ns=int(record["timestamp_ns"]),
                        best_bid=latest_bid,
                        best_ask=latest_ask,
                        bid_volume=int(record["bid_volume"]),
                        ask_volume=int(record["ask_volume"]),
                    )
                    replay_record_index += 1
                    if features is None:
                        continue
                    warmed = True
                    latest_prediction = model.predict_one(features)
                    latest_imbalance = float(features[1])

            now = time.perf_counter()
            elapsed = now - previous_report
            if elapsed < REPORT_INTERVAL_SECONDS:
                continue
            rate = (total_ticks - previous_ticks) / elapsed
            if not warmed:
                print(f"Ticks: {total_ticks:,} | Rate: {rate:,.2f}/sec | Warming feature history...")
            else:
                signal = classify_signal(latest_prediction, model.signal_threshold_bps)
                multiple = (
                    abs(latest_prediction) / model.signal_threshold_bps
                    if model.signal_threshold_bps > 0.0
                    else float("inf")
                )
                print(
                    f"Ticks: {total_ticks:,} | Rate: {rate:,.2f}/sec | "
                    f"Bid: {latest_bid:.2f} | Ask: {latest_ask:.2f} | Mid: {latest_mid:.4f} | "
                    f"OBI: {latest_imbalance:+.4f} | Forecast({model.horizon}): "
                    f"{latest_prediction:+.6f} bps | Signal: {signal} | Threshold: {multiple:.2f}x"
                )
            previous_ticks = total_ticks
            previous_report = now
    except KeyboardInterrupt:
        print("\n[Alpha Runtime] Stop requested.")
    finally:
        engine.stop()

    print(f"[Alpha Runtime] Final ticks consumed: {total_ticks:,}")
    engine_error = engine.last_error()
    if engine_error:
        print(f"[Alpha Runtime] Engine error: {engine_error}")
    if arguments.mode == "live":
        print(
            f"[Alpha Runtime] Feed drops: {engine.dropped_ticks():,} | "
            f"Record drops: {engine.recording_dropped():,} | "
            f"Write errors: {engine.recording_write_errors():,}"
        )
    else:
        print(
            f"[Alpha Runtime] Replayed: {engine.replayed_ticks():,} | "
            f"Backpressure episodes: {engine.replay_backpressure_events():,} | "
            f"Replay errors: {engine.replay_errors():,}"
        )

    if engine_error:
        raise RuntimeError(engine_error)
    if arguments.mode == "live" and engine.recording_write_errors() != 0:
        raise RuntimeError("The binary recorder reported a write/finalization error.")
    if arguments.mode == "live" and engine.recording_dropped() != 0:
        raise RuntimeError("The binary recorder dropped records; capture is incomplete.")
    if arguments.mode == "replay" and engine.replay_errors() != 0:
        raise RuntimeError("The binary replay reported an error.")


if __name__ == "__main__":
    main()

