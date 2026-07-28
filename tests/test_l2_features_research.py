from __future__ import annotations

import json

import numpy as np
import pytest
from pathlib import Path

from l2_features import FEATURE_NAMES, FEATURE_SCHEMA_HASH, FEATURE_SCHEMA_VERSION, build_feature_set
from l2_research import run_experiment
from l2bin import Boundary, BoundaryReason, L2Writer, sha256_file
from l2book import PRICE_SCALE, QUANTITY_SCALE, DepthUpdate, L2OrderBook, Level, Snapshot, Trade


def write_session(
    path: Path, seed: int, count: int = 260, symbol: str = "BTCUSDT"
) -> Path:
    tick = int(0.01 * PRICE_SCALE)
    mid_ticks = 10_000 + seed * 50
    bid = mid_ticks * tick
    ask = bid + tick
    snapshot = Snapshot(
        1_000_000_000 + seed * 10_000_000_000,
        100,
        (Level(bid, 5 * QUANTITY_SCALE),),
        (Level(ask, 5 * QUANTITY_SCALE),),
    )
    book = L2OrderBook()
    writer = L2Writer(path, symbol, checkpoint_interval=50)
    writer.write(Boundary(snapshot.receipt_timestamp_ns - 1, BoundaryReason.CONNECTION_START))
    writer.write(snapshot)
    book.install_snapshot(snapshot)
    previous_bid = bid
    previous_ask = ask
    timestamp = snapshot.receipt_timestamp_ns
    for index in range(1, count + 1):
        direction = 1 if (index * 17 + seed) % 7 in (0, 1, 2, 3) else -1
        mid_ticks += direction
        new_bid = mid_ticks * tick
        new_ask = new_bid + tick
        bid_quantity = (3 + ((index + seed) % 7)) * QUANTITY_SCALE
        ask_quantity = (3 + ((index * 3 + seed) % 7)) * QUANTITY_SCALE
        timestamp += 500_000 + ((index * 13 + seed) % 100_000)
        update = DepthUpdate(
            timestamp,
            timestamp // 1_000_000,
            100 + index,
            100 + index,
            (Level(previous_bid, 0), Level(new_bid, bid_quantity)),
            (Level(previous_ask, 0), Level(new_ask, ask_quantity)),
        )
        writer.write(update)
        book.apply(update)
        if index % 3 == 0:
            writer.write(
                Trade(
                    timestamp + 1,
                    timestamp // 1_000_000,
                    index,
                    new_ask if direction > 0 else new_bid,
                    QUANTITY_SCALE // 10,
                    direction < 0,
                )
            )
        if index % 50 == 0:
            writer.write_checkpoint(book.last_update_id, book.state_hash())
        previous_bid, previous_ask = new_bid, new_ask
    writer.write_checkpoint(book.last_update_id, book.state_hash())
    writer.finalize(final_update_id=book.last_update_id, final_state_hash=book.state_hash())
    return path


def test_l2_feature_schema_and_research_pipeline(tmp_path: Path) -> None:
    recordings = [
        write_session(tmp_path / f"session-{index}.l2bin", index)
        for index in range(3)
    ]
    features = build_feature_set(recordings, horizon=5)
    assert features.size > 500
    assert features.X.shape[1] == len(FEATURE_NAMES)
    assert features.current_bid_quantity.shape == features.y.shape
    output = tmp_path / "artifacts"
    result = run_experiment(
        recordings,
        horizon=5,
        output_directory=output,
        fee_bps_per_side=0.0,
        slippage_bps_per_side=0.0,
        trade_size_base=0.0,
        max_displayed_participation=0.1,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    report_path = Path(result["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["symbol"] == "BTCUSDT"
    assert report["predeclared_primary_model"] == "full_l2"
    assert report["feature_schema"]["version"] == FEATURE_SCHEMA_VERSION
    assert report["feature_schema"]["sha256"] == FEATURE_SCHEMA_HASH
    assert all(Path(item["file"]).name == item["file"] for item in report["recordings"])
    assert set(report["feature_group_comparison"]) == {
        "top_of_book",
        "multi_level_depth",
        "full_l2",
    }
    assert len(report["latency_stress"]) == 6
    model_path = Path(result["model"])
    predictions_path = Path(result["predictions"])
    card_path = Path(result["research_card"])
    assert model_path.exists()
    assert card_path.exists()
    assert report["artifacts"]["model"]["sha256"] == sha256_file(model_path)
    assert report["artifacts"]["predictions"]["sha256"] == sha256_file(predictions_path)
    assert report["artifacts"]["research_card"]["sha256"] == sha256_file(card_path)
    with np.load(model_path, allow_pickle=False) as archive:
        assert int(archive["schema_version"]) == 2
        assert int(archive["feature_schema_version"]) == FEATURE_SCHEMA_VERSION
        assert str(archive["feature_schema_hash"].item()) == FEATURE_SCHEMA_HASH
        assert str(archive["symbol"].item()) == "BTCUSDT"
    assert not tuple(output.glob(".*.tmp*"))

    with pytest.raises(RuntimeError, match="exact L2 holdout"):
        run_experiment(
            recordings,
            horizon=5,
            output_directory=output,
            fee_bps_per_side=0.0,
            slippage_bps_per_side=0.0,
            trade_size_base=0.0,
            max_displayed_participation=0.1,
            bootstrap_samples=100,
            bootstrap_seed=7,
        )


def test_l2_research_rejects_duplicate_and_mixed_symbol_inputs(tmp_path: Path) -> None:
    btc = write_session(tmp_path / "btc.l2bin", 1)
    eth = write_session(tmp_path / "eth.l2bin", 2, symbol="ETHUSDT")
    with pytest.raises(RuntimeError, match="Duplicate"):
        run_experiment(
            [btc, btc],
            horizon=5,
            output_directory=tmp_path / "duplicate",
            fee_bps_per_side=0.0,
            slippage_bps_per_side=0.0,
            trade_size_base=0.0,
            max_displayed_participation=0.1,
            bootstrap_samples=100,
            bootstrap_seed=7,
        )
    with pytest.raises(ValueError, match="exactly one symbol"):
        run_experiment(
            [btc, eth],
            horizon=5,
            output_directory=tmp_path / "mixed",
            fee_bps_per_side=0.0,
            slippage_bps_per_side=0.0,
            trade_size_base=0.0,
            max_displayed_participation=0.1,
            bootstrap_samples=100,
            bootstrap_seed=7,
        )


def test_feature_builder_rejects_incomplete_recording_by_default(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.l2bin"
    writer = L2Writer(path, "BTCUSDT")
    writer.write(Boundary(1, BoundaryReason.CONNECTION_START))
    writer.abort()
    with pytest.raises(RuntimeError, match="incomplete"):
        build_feature_set([path], horizon=5)
