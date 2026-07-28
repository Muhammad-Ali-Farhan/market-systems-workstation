from __future__ import annotations

import csv
from pathlib import Path

from execution_simulator import QueueModel
from l2_execution_sensitivity import run_sensitivity
from l2bin import L2Writer
from l2book import PRICE_SCALE, QUANTITY_SCALE, DepthUpdate, L2OrderBook, Level, Snapshot, Trade


def test_execution_sensitivity_runs_selected_signal(tmp_path: Path) -> None:
    recording = tmp_path / "session.l2bin"
    def price(value: float) -> int:
        return int(value * PRICE_SCALE)

    def quantity(value: float) -> int:
        return int(value * QUANTITY_SCALE)
    snapshot = Snapshot(1_000, 100, (Level(price(100), quantity(1)),), (Level(price(101), quantity(1)),))
    update = DepthUpdate(2_000, 2, 101, 101, (), ())
    trade = Trade(3_000, 3, 1, price(100), quantity(2), True)
    book = L2OrderBook()
    writer = L2Writer(recording, "BTCUSDT")
    writer.write(snapshot)
    book.install_snapshot(snapshot)
    writer.write(update)
    book.apply(update)
    writer.write(trade)
    writer.write_checkpoint(book.last_update_id, book.state_hash())
    writer.finalize(final_update_id=book.last_update_id, final_state_hash=book.state_hash())

    predictions = tmp_path / "predictions.csv"
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        output = csv.writer(stream)
        output.writerow([
            "global_row", "timestamp_ns", "session_id", "best_bid", "best_ask",
            "selected_trade", "side",
        ])
        output.writerow([1, 1_000, 0, 100.0, 101.0, 1, 1])

    result = run_sensitivity(
        predictions,
        {0: recording},
        quantity=quantity(0.1),
        style="passive",
        latencies_us=(0.0,),
        queue_models=(QueueModel.TRADE_ONLY,),
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        queue_ahead_fraction=0.0,
        time_to_live_ns=10_000,
    )
    assert result["cases"][0]["orders"] == 1
    assert result["cases"][0]["fill_rate"] == 1.0
    assert result["prediction_file"]["file"] == predictions.name
    assert "sha256" in result["prediction_file"]
    assert result["recordings"]["0"]["file"] == recording.name
    assert "checkpoint_sha256" in result["recordings"]["0"]
    assert str(tmp_path) not in str(result)
