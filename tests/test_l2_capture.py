from __future__ import annotations

import json

import pytest

from l2_capture import ReconnectBackoff, parse_stream_message
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


def test_stream_integer_fields_do_not_coerce_strings_or_booleans() -> None:
    string_timestamp = json.dumps(
        {
            "data": {
                "e": "depthUpdate",
                "E": "123",
                "s": "BTCUSDT",
                "U": 10,
                "u": 12,
                "b": [],
                "a": [],
            }
        }
    )
    with pytest.raises(ValueError, match="'E'.*integer"):
        parse_stream_message(string_timestamp, 999)

    boolean_trade_id = json.dumps(
        {
            "data": {
                "e": "aggTrade",
                "E": 123,
                "s": "BTCUSDT",
                "a": True,
                "p": "100.00000000",
                "q": "1.00000000",
                "m": False,
            }
        }
    )
    with pytest.raises(ValueError, match="'a'.*integer"):
        parse_stream_message(boolean_trade_id, 999)


def test_aggregate_trade_maker_flag_must_be_boolean() -> None:
    raw = json.dumps(
        {
            "data": {
                "e": "aggTrade",
                "E": 123,
                "s": "BTCUSDT",
                "a": 44,
                "p": "100.00000000",
                "q": "1.00000000",
                "m": "false",
            }
        }
    )
    with pytest.raises(ValueError, match="'m'.*boolean"):
        parse_stream_message(raw, 999)


def test_reconnect_backoff_resets_after_successful_open() -> None:
    backoff = ReconnectBackoff(initial_delay_seconds=0.5, maximum_delay_seconds=2.0)
    assert backoff.consume() == 0.5
    assert backoff.consume() == 1.0
    assert backoff.consume() == 2.0
    backoff.reset()
    assert backoff.consume() == 0.5


def test_stream_client_resets_backoff_when_connection_opens(monkeypatch) -> None:
    import queue
    import sys
    from types import SimpleNamespace

    import l2_capture
    from l2_capture import CombinedStreamClient

    observed_delays: list[float] = []

    class SpyBackoff:
        def __init__(self) -> None:
            self.inner = ReconnectBackoff(
                initial_delay_seconds=0.5,
                maximum_delay_seconds=2.0,
            )

        def consume(self) -> float:
            delay = self.inner.consume()
            observed_delays.append(delay)
            return delay

        def reset(self) -> None:
            self.inner.reset()

    monkeypatch.setattr(l2_capture, "ReconnectBackoff", SpyBackoff)

    client = CombinedStreamClient(
        ("BTCUSDT",),
        queue.Queue(),
        queue_drop_callback=lambda: None,
    )
    runs = 0

    class FakeWebSocketApp:
        def __init__(self, _target, *, on_open, on_message, on_error, on_close) -> None:
            self.on_open = on_open
            self.on_message = on_message
            self.on_error = on_error
            self.on_close = on_close

        def run_forever(self, **_kwargs) -> None:
            nonlocal runs
            runs += 1
            if runs == 3:
                self.on_open(self)
            if runs == 4:
                client.stop_event.set()

        def close(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "websocket",
        SimpleNamespace(WebSocketApp=FakeWebSocketApp),
    )
    clock = iter(float(value) for value in range(0, 1000, 100))
    monkeypatch.setattr(l2_capture.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(l2_capture.time, "sleep", lambda _seconds: None)

    client.run()

    assert runs == 4
    assert observed_delays == [0.5, 1.0, 0.5]
