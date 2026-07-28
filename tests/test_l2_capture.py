from __future__ import annotations

import json

import pytest

from l2_capture import parse_stream_message
from l2book import DepthUpdate, Trade


def test_parse_combined_depth_message() -> None:
    raw = json.dumps(
        {
            "stream": "btcusdt@depth@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 123,
                "s": "BTCUSDT",
                "U": 10,
                "u": 12,
                "b": [["100.00000000", "1.25000000"]],
                "a": [["101.00000000", "0.00000000"]],
            },
        }
    )
    symbol, event = parse_stream_message(raw, 999)
    assert symbol == "BTCUSDT"
    assert isinstance(event, DepthUpdate)
    assert event.first_update_id == 10
    assert event.final_update_id == 12
    assert event.bids[0].price == 10_000_000_000
    assert event.bids[0].quantity == 125_000_000


def test_parse_combined_aggregate_trade() -> None:
    raw = json.dumps(
        {
            "stream": "ethusdt@aggTrade",
            "data": {
                "e": "aggTrade",
                "E": 123,
                "s": "ETHUSDT",
                "a": 44,
                "p": "2500.50000000",
                "q": "0.25000000",
                "m": True,
            },
        }
    )
    symbol, event = parse_stream_message(raw, 999)
    assert symbol == "ETHUSDT"
    assert isinstance(event, Trade)
    assert event.aggregate_trade_id == 44
    assert event.buyer_is_maker is True


def test_unknown_event_is_rejected() -> None:
    raw = json.dumps({"data": {"e": "mystery", "s": "BTCUSDT"}})
    with pytest.raises(ValueError, match="Unsupported"):
        parse_stream_message(raw, 999)


def test_symbol_capture_bootstraps_and_recovers_gap(tmp_path) -> None:
    from l2_capture import SymbolCapture
    from l2bin import L2Writer, iter_events
    from l2book import Level, Snapshot, SyncState

    snapshots = iter(
        [
            Snapshot(
                10,
                100,
                (Level(10_000, 500),),
                (Level(10_100, 600),),
            ),
            Snapshot(
                20,
                102,
                (Level(10_000, 450),),
                (Level(10_100, 550),),
            ),
        ]
    )
    writer = L2Writer(tmp_path / "sync.l2bin", "BTCUSDT")
    capture = SymbolCapture(
        "BTCUSDT",
        writer,
        snapshot_fetcher=lambda _symbol: next(snapshots),
    )
    capture.on_depth(
        DepthUpdate(11, 11, 101, 101, (Level(10_000, 450),), ())
    )
    assert capture.synchronizer.state is SyncState.LIVE
    assert capture.synchronizer.book.last_update_id == 101

    capture.on_depth(
        DepthUpdate(21, 21, 103, 103, (), (Level(10_100, 525),))
    )
    assert capture.sequence_gaps == 1
    assert capture.synchronizer.state is SyncState.LIVE
    assert capture.synchronizer.book.last_update_id == 103
    writer.abort()

    events = tuple(iter_events(tmp_path / "sync.l2bin"))
    assert sum(isinstance(item, Snapshot) for item in events) == 2
    assert sum(isinstance(item, DepthUpdate) for item in events) == 2
