from __future__ import annotations

from pathlib import Path

import pytest

from execution_simulator import (
    ExecutionConfig,
    ExecutionSimulator,
    OrderRequest,
    OrderStatus,
    OrderType,
    QueueModel,
    Side,
)
from l2bin import Boundary, BoundaryReason, L2Writer
from l2book import PRICE_SCALE, QUANTITY_SCALE, DepthUpdate, L2OrderBook, Level, Snapshot, Trade


def p(value: float) -> int:
    return int(value * PRICE_SCALE)


def q(value: float) -> int:
    return int(value * QUANTITY_SCALE)


def write_market(path: Path, *, include_trade: bool = True, depth_decrease: bool = False) -> Path:
    snapshot = Snapshot(
        1_000,
        100,
        (Level(p(100.0), q(5.0)), Level(p(99.0), q(10.0))),
        (Level(p(101.0), q(5.0)), Level(p(102.0), q(10.0))),
    )
    update = DepthUpdate(
        2_000,
        2,
        101,
        101,
        (Level(p(100.0), q(3.0)),) if depth_decrease else (),
        (),
    )
    final_update = DepthUpdate(
        5_000_000,
        5,
        102,
        102,
        (Level(p(100.5), q(5.0)), Level(p(100.0), 0)),
        (Level(p(101.5), q(5.0)), Level(p(101.0), 0)),
    )
    book = L2OrderBook()
    writer = L2Writer(path, "BTCUSDT")
    writer.write(Boundary(999, BoundaryReason.CONNECTION_START))
    writer.write(snapshot)
    book.install_snapshot(snapshot)
    writer.write(update)
    book.apply(update)
    if include_trade:
        writer.write(Trade(3_000, 3, 1, p(100.0), q(6.0), True))
    writer.write(final_update)
    book.apply(final_update)
    writer.write_checkpoint(book.last_update_id, book.state_hash())
    writer.finalize(final_update_id=book.last_update_id, final_state_hash=book.state_hash())
    return path


def test_market_order_sweeps_visible_depth(tmp_path: Path) -> None:
    path = write_market(tmp_path / "market.l2bin")
    simulator = ExecutionSimulator(ExecutionConfig(transmission_latency_ns=0))
    simulator.submit(OrderRequest("m1", 1_000, Side.BUY, OrderType.MARKET, q(6.0)))
    result = simulator.run(path)
    assert result.orders[0].status is OrderStatus.FILLED
    assert len(result.fills) == 2
    assert result.fills[0].price == p(101.0)
    assert result.fills[1].price == p(102.0)
    assert result.summary()["filled_quantity"] == pytest.approx(6.0)


def test_passive_fill_uses_observed_trade_and_queue_ahead(tmp_path: Path) -> None:
    path = write_market(tmp_path / "passive.l2bin")
    simulator = ExecutionSimulator(
        ExecutionConfig(transmission_latency_ns=0, queue_ahead_fraction=1.0)
    )
    simulator.submit(
        OrderRequest("l1", 1_000, Side.BUY, OrderType.LIMIT, q(1.0), p(100.0))
    )
    result = simulator.run(path)
    assert result.orders[0].status is OrderStatus.FILLED
    assert len(result.fills) == 1
    assert result.fills[0].maker is True
    assert result.fills[0].quantity == q(1.0)


def test_trade_only_does_not_treat_cancellation_as_fill(tmp_path: Path) -> None:
    path = write_market(
        tmp_path / "cancellation.l2bin", include_trade=False, depth_decrease=True
    )
    simulator = ExecutionSimulator(
        ExecutionConfig(
            transmission_latency_ns=0,
            queue_model=QueueModel.TRADE_ONLY,
            queue_ahead_fraction=0.0,
        )
    )
    simulator.submit(
        OrderRequest("l1", 1_000, Side.BUY, OrderType.LIMIT, q(1.0), p(100.0))
    )
    result = simulator.run(path)
    assert result.orders[0].status is OrderStatus.EXPIRED
    assert not result.fills


def test_optimistic_depth_is_explicit_sensitivity_case(tmp_path: Path) -> None:
    path = write_market(
        tmp_path / "optimistic.l2bin", include_trade=False, depth_decrease=True
    )
    simulator = ExecutionSimulator(
        ExecutionConfig(
            transmission_latency_ns=0,
            queue_model=QueueModel.OPTIMISTIC_DEPTH,
            queue_ahead_fraction=0.0,
        )
    )
    simulator.submit(
        OrderRequest("l1", 1_000, Side.BUY, OrderType.LIMIT, q(1.0), p(100.0))
    )
    result = simulator.run(path)
    assert result.orders[0].status is OrderStatus.FILLED
    assert result.fills[0].maker is True


def test_market_orders_cannot_reuse_same_displayed_liquidity(tmp_path: Path) -> None:
    path = write_market(tmp_path / "no_reuse.l2bin")
    simulator = ExecutionSimulator(ExecutionConfig(transmission_latency_ns=0))
    simulator.submit(OrderRequest("m1", 1_000, Side.BUY, OrderType.MARKET, q(5.0)))
    simulator.submit(OrderRequest("m2", 1_000, Side.BUY, OrderType.MARKET, q(1.0)))
    result = simulator.run(path)
    first = next(order for order in result.orders if order.request.order_id == "m1")
    second = next(order for order in result.orders if order.request.order_id == "m2")
    assert first.status is OrderStatus.FILLED
    assert second.status is OrderStatus.FILLED
    second_fill = next(fill for fill in result.fills if fill.order_id == "m2")
    assert second_fill.price == p(102.0)


def test_later_own_limit_order_queues_behind_earlier_order(tmp_path: Path) -> None:
    path = write_market(tmp_path / "own_fifo.l2bin")
    simulator = ExecutionSimulator(
        ExecutionConfig(
            transmission_latency_ns=0,
            queue_ahead_fraction=0.0,
            queue_model=QueueModel.TRADE_ONLY,
        )
    )
    simulator.submit(OrderRequest("l1", 1_000, Side.BUY, OrderType.LIMIT, q(1.0), p(100.0)))
    simulator.submit(OrderRequest("l2", 1_000, Side.BUY, OrderType.LIMIT, q(1.0), p(100.0)))
    result = simulator.run(path)
    fills = {fill.order_id: fill for fill in result.fills}
    assert set(fills) == {"l1", "l2"}
    assert fills["l1"].quantity == q(1.0)
    assert fills["l2"].quantity == q(1.0)


def write_event_order_market(path: Path) -> Path:
    snapshot = Snapshot(
        1_000,
        100,
        (Level(p(100.0), q(5.0)),),
        (Level(p(101.0), q(5.0)),),
    )
    update = DepthUpdate(
        2_000,
        2,
        101,
        101,
        (),
        (Level(p(101.0), 0), Level(p(102.0), q(5.0))),
    )
    book = L2OrderBook()
    writer = L2Writer(path, "BTCUSDT")
    writer.write(snapshot)
    book.install_snapshot(snapshot)
    writer.write(update)
    book.apply(update)
    writer.write_checkpoint(book.last_update_id, book.state_hash())
    writer.finalize(final_update_id=book.last_update_id, final_state_hash=book.state_hash())
    return path


def test_order_arriving_before_market_event_uses_pre_event_book(tmp_path: Path) -> None:
    path = write_event_order_market(tmp_path / "pre_event.l2bin")
    simulator = ExecutionSimulator(ExecutionConfig(transmission_latency_ns=0))
    simulator.submit(OrderRequest("m1", 1_500, Side.BUY, OrderType.MARKET, q(1.0)))
    result = simulator.run(path)
    assert result.fills[0].price == p(101.0)
    assert result.fills[0].timestamp_ns == 2_000


def test_exact_timestamp_tie_follows_recorded_market_event_first(tmp_path: Path) -> None:
    path = write_event_order_market(tmp_path / "tie.l2bin")
    simulator = ExecutionSimulator(ExecutionConfig(transmission_latency_ns=0))
    simulator.submit(OrderRequest("m1", 2_000, Side.BUY, OrderType.MARKET, q(1.0)))
    result = simulator.run(path)
    assert result.fills[0].price == p(102.0)


def test_market_data_boundary_invalidates_pending_markouts(tmp_path: Path) -> None:
    path = tmp_path / "markout_boundary.l2bin"
    first = Snapshot(1_000, 100, (Level(p(100.0), q(5.0)),), (Level(p(101.0), q(5.0)),))
    second = Snapshot(2_000_000, 200, (Level(p(110.0), q(5.0)),), (Level(p(111.0), q(5.0)),))
    update = DepthUpdate(200_000_000, 200, 201, 201, (), ())
    book = L2OrderBook()
    writer = L2Writer(path, "BTCUSDT")
    writer.write(first)
    book.install_snapshot(first)
    writer.write(Boundary(1_500, BoundaryReason.SEQUENCE_GAP))
    writer.write(second)
    book.install_snapshot(second)
    writer.write(update)
    book.apply(update)
    writer.write_checkpoint(book.last_update_id, book.state_hash())
    writer.finalize(final_update_id=book.last_update_id, final_state_hash=book.state_hash())

    simulator = ExecutionSimulator(ExecutionConfig(transmission_latency_ns=0))
    simulator.submit(OrderRequest("m1", 1_000, Side.BUY, OrderType.MARKET, q(1.0)))
    result = simulator.run(path)
    assert result.fills[0].markouts_bps == {}


def test_simulator_instance_cannot_be_reused(tmp_path: Path) -> None:
    path = write_market(tmp_path / "once.l2bin")
    simulator = ExecutionSimulator(ExecutionConfig(transmission_latency_ns=0))
    simulator.run(path)
    with pytest.raises(RuntimeError, match="only once"):
        simulator.run(path)


def test_execution_configuration_rejects_ambiguous_values() -> None:
    with pytest.raises(TypeError):
        ExecutionConfig(transmission_latency_ns=True)
    with pytest.raises(ValueError):
        ExecutionConfig(kill_switch_loss_quote=float("nan"))
    with pytest.raises(ValueError):
        ExecutionConfig(markout_horizons_ns=(1, 1))
