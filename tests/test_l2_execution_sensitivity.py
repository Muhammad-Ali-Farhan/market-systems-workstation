from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from execution_simulator import QueueModel
from l2_execution_sensitivity import run_sensitivity
from l2bin import L2Writer, read_metadata, sha256_file
from l2book import (
    PRICE_SCALE,
    QUANTITY_SCALE,
    DepthUpdate,
    L2OrderBook,
    Level,
    Snapshot,
    Trade,
)


def price(value: float) -> int:
    return int(value * PRICE_SCALE)


def quantity(value: float) -> int:
    return int(value * QUANTITY_SCALE)


def write_recording(path: Path, *, mid: float = 100.0) -> Path:
    snapshot = Snapshot(
        1_000,
        100,
        (Level(price(mid), quantity(1)),),
        (Level(price(mid + 1), quantity(1)),),
    )
    update = DepthUpdate(2_000, 2, 101, 101, (), ())
    trade = Trade(3_000, 3, 1, price(mid), quantity(2), True)
    book = L2OrderBook()
    writer = L2Writer(path, "BTCUSDT")
    writer.write(snapshot)
    book.install_snapshot(snapshot)
    writer.write(update)
    book.apply(update)
    writer.write(trade)
    writer.write_checkpoint(book.last_update_id, book.state_hash())
    writer.finalize(
        final_update_id=book.last_update_id,
        final_state_hash=book.state_hash(),
    )
    return path


def write_predictions(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        output = csv.writer(stream)
        output.writerow(
            [
                "global_row",
                "timestamp_ns",
                "session_id",
                "best_bid",
                "best_ask",
                "selected_trade",
                "side",
            ]
        )
        output.writerow([1, 1_000, 0, 100.0, 101.0, 1, 1])
    return path


def write_research_report(
    path: Path,
    predictions: Path,
    recordings: list[Path],
    *,
    test_sessions: list[int] | None = None,
) -> Path:
    entries = []
    for recording in recordings:
        metadata = read_metadata(recording, verify_hashes=True)
        entries.append(
            {
                "file": recording.name,
                "symbol": metadata.symbol,
                "sha256": metadata.sha256,
                "checkpoint_sha256": metadata.checkpoint_sha256,
            }
        )
    payload = {
        "schema_version": 2,
        "symbol": "BTCUSDT",
        "test_fingerprint_sha256": "a" * 64,
        "split": {"test_sessions": test_sessions or [0]},
        "recordings": entries,
        "artifacts": {
            "predictions": {
                "file": predictions.name,
                "sha256": sha256_file(predictions),
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_single_case(
    predictions: Path,
    recording: Path,
    research_report: Path,
) -> dict[str, object]:
    return run_sensitivity(
        predictions,
        {0: recording},
        research_report=research_report,
        quantity=quantity(0.1),
        style="passive",
        latencies_us=(0.0,),
        queue_models=(QueueModel.TRADE_ONLY,),
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        queue_ahead_fraction=0.0,
        time_to_live_ns=10_000,
    )


def test_execution_sensitivity_runs_provenance_verified_signal(tmp_path: Path) -> None:
    recording = write_recording(tmp_path / "session.l2bin")
    predictions = write_predictions(tmp_path / "predictions.csv")
    report = write_research_report(
        tmp_path / "l2_h20_report.json",
        predictions,
        [recording],
    )

    result = run_single_case(predictions, recording, report)

    assert result["schema_version"] == 2
    assert result["research_provenance"]["verified"] is True
    assert result["cases"][0]["orders"] == 1
    assert result["cases"][0]["fill_rate"] == 1.0
    assert result["prediction_file"]["file"] == predictions.name
    assert result["prediction_file"]["sha256"] == sha256_file(predictions)
    assert result["recordings"]["0"]["file"] == recording.name
    assert "checkpoint_sha256" in result["recordings"]["0"]
    assert str(tmp_path) not in str(result)


def test_execution_sensitivity_rejects_prediction_not_produced_by_report(
    tmp_path: Path,
) -> None:
    recording = write_recording(tmp_path / "session.l2bin")
    predictions = write_predictions(tmp_path / "predictions.csv")
    report = write_research_report(tmp_path / "report.json", predictions, [recording])
    predictions.write_text(
        predictions.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Prediction CSV does not match"):
        run_single_case(predictions, recording, report)


def test_execution_sensitivity_rejects_wrong_recording_for_session(
    tmp_path: Path,
) -> None:
    expected_recording = write_recording(tmp_path / "expected.l2bin", mid=100.0)
    wrong_recording = write_recording(tmp_path / "wrong.l2bin", mid=200.0)
    predictions = write_predictions(tmp_path / "predictions.csv")
    report = write_research_report(
        tmp_path / "report.json",
        predictions,
        [expected_recording],
    )

    with pytest.raises(RuntimeError, match="does not match research provenance"):
        run_single_case(predictions, wrong_recording, report)


def test_execution_sensitivity_requires_exact_held_out_session_mapping(
    tmp_path: Path,
) -> None:
    first = write_recording(tmp_path / "first.l2bin", mid=100.0)
    second = write_recording(tmp_path / "second.l2bin", mid=200.0)
    predictions = write_predictions(tmp_path / "predictions.csv")
    report = write_research_report(
        tmp_path / "report.json",
        predictions,
        [first, second],
        test_sessions=[0],
    )

    with pytest.raises(ValueError, match="held-out sessions exactly"):
        run_sensitivity(
            predictions,
            {0: first, 1: second},
            research_report=report,
            quantity=quantity(0.1),
            style="passive",
            latencies_us=(0.0,),
            queue_models=(QueueModel.TRADE_ONLY,),
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
            queue_ahead_fraction=0.0,
            time_to_live_ns=10_000,
        )
