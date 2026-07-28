from __future__ import annotations

import argparse
import csv
import enum
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from l2bin import Boundary, iter_events
from l2book import (
    PRICE_SCALE,
    QUANTITY_SCALE,
    UINT64_MAX,
    DepthUpdate,
    L2OrderBook,
    Snapshot,
    Trade,
    parse_price,
    parse_quantity,
)


class Side(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class QueueModel(str, enum.Enum):
    TRADE_ONLY = "trade_only"
    PRO_RATA_DEPTH = "pro_rata_depth"
    OPTIMISTIC_DEPTH = "optimistic_depth"


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    decision_latency_ns: int = 0
    transmission_latency_ns: int = 250_000
    cancel_latency_ns: int = 250_000
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    queue_model: QueueModel = QueueModel.TRADE_ONLY
    queue_ahead_fraction: float = 1.0
    depth_depletion_fill_fraction: float = 0.5
    max_absolute_position: int = 10 * QUANTITY_SCALE
    kill_switch_loss_quote: float = math.inf
    markout_horizons_ns: tuple[int, ...] = (1_000_000, 10_000_000, 100_000_000)

    def __post_init__(self) -> None:
        integer_values = {
            "decision_latency_ns": self.decision_latency_ns,
            "transmission_latency_ns": self.transmission_latency_ns,
            "cancel_latency_ns": self.cancel_latency_ns,
            "max_absolute_position": self.max_absolute_position,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 0 or value > UINT64_MAX:
                raise ValueError(f"{name} must fit in an unsigned 64-bit integer.")
        if not isinstance(self.queue_model, QueueModel):
            raise TypeError("queue_model must be a QueueModel value.")
        if not math.isfinite(self.queue_ahead_fraction) or not 0.0 <= self.queue_ahead_fraction <= 5.0:
            raise ValueError("queue_ahead_fraction must be finite and between 0 and 5.")
        if (
            not math.isfinite(self.depth_depletion_fill_fraction)
            or not 0.0 <= self.depth_depletion_fill_fraction <= 1.0
        ):
            raise ValueError(
                "depth_depletion_fill_fraction must be finite and between 0 and 1."
            )
        if not math.isfinite(self.maker_fee_bps) or self.maker_fee_bps < -100.0:
            raise ValueError("Maker fee must be finite and cannot be below -100 bps.")
        if not math.isfinite(self.taker_fee_bps) or self.taker_fee_bps < 0.0:
            raise ValueError("Taker fee must be finite and non-negative.")
        if math.isnan(self.kill_switch_loss_quote) or self.kill_switch_loss_quote <= 0.0:
            raise ValueError("kill_switch_loss_quote must be positive or +infinity.")
        if not isinstance(self.markout_horizons_ns, tuple):
            raise TypeError("markout_horizons_ns must be a tuple of integers.")
        if len(set(self.markout_horizons_ns)) != len(self.markout_horizons_ns):
            raise ValueError("Markout horizons must be unique.")
        for horizon in self.markout_horizons_ns:
            if isinstance(horizon, bool) or not isinstance(horizon, int):
                raise TypeError("Markout horizons must be integers.")
            if horizon <= 0 or horizon > UINT64_MAX:
                raise ValueError("Markout horizons must be positive 64-bit values.")


@dataclass(frozen=True, slots=True)
class OrderRequest:
    order_id: str
    decision_timestamp_ns: int
    side: Side
    order_type: OrderType
    quantity: int
    limit_price: int | None = None
    time_to_live_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError("order_id cannot be empty.")
        if not isinstance(self.side, Side) or not isinstance(self.order_type, OrderType):
            raise TypeError("side and order_type must use their enum types.")
        for name, value in (
            ("decision_timestamp_ns", self.decision_timestamp_ns),
            ("quantity", self.quantity),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0 or value > UINT64_MAX:
                raise ValueError(f"{name} must be positive and fit in 64 bits.")
        if self.order_type is OrderType.LIMIT:
            if (
                self.limit_price is None
                or isinstance(self.limit_price, bool)
                or not isinstance(self.limit_price, int)
                or self.limit_price <= 0
            ):
                raise ValueError("Limit orders require a positive integer limit price.")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("Market orders cannot specify a limit price.")
        if self.time_to_live_ns is not None:
            if isinstance(self.time_to_live_ns, bool) or not isinstance(self.time_to_live_ns, int):
                raise TypeError("time_to_live_ns must be an integer when provided.")
            if self.time_to_live_ns <= 0 or self.time_to_live_ns > UINT64_MAX:
                raise ValueError("time_to_live_ns must be a positive 64-bit value.")


@dataclass(frozen=True, slots=True)
class CancelRequest:
    order_id: str
    decision_timestamp_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError("order_id cannot be empty.")
        if (
            isinstance(self.decision_timestamp_ns, bool)
            or not isinstance(self.decision_timestamp_ns, int)
        ):
            raise TypeError("decision_timestamp_ns must be an integer.")
        if self.decision_timestamp_ns <= 0 or self.decision_timestamp_ns > UINT64_MAX:
            raise ValueError("decision_timestamp_ns must be positive and fit in 64 bits.")


@dataclass(slots=True)
class SimulatedOrder:
    request: OrderRequest
    arrival_timestamp_ns: int
    expiry_timestamp_ns: int | None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    queue_ahead: float = 0.0
    rejection_reason: str = ""

    @property
    def remaining_quantity(self) -> int:
        return self.request.quantity - self.filled_quantity


@dataclass(slots=True)
class Fill:
    order_id: str
    timestamp_ns: int
    side: Side
    quantity: int
    price: int
    maker: bool
    fee_quote: float
    markouts_bps: dict[int, float] = field(default_factory=dict)

    @property
    def quantity_float(self) -> float:
        return self.quantity / QUANTITY_SCALE

    @property
    def price_float(self) -> float:
        return self.price / PRICE_SCALE


@dataclass(frozen=True, slots=True)
class SimulationResult:
    orders: tuple[SimulatedOrder, ...]
    fills: tuple[Fill, ...]
    ending_position: int
    ending_cash_quote: float
    marked_equity_quote: float
    realized_fees_quote: float
    killed: bool

    def summary(self) -> dict[str, object]:
        requested = sum(order.request.quantity for order in self.orders)
        filled = sum(fill.quantity for fill in self.fills)
        maker_quantity = sum(fill.quantity for fill in self.fills if fill.maker)
        taker_quantity = filled - maker_quantity
        markouts: dict[str, float | None] = {}
        horizons = sorted({horizon for fill in self.fills for horizon in fill.markouts_bps})
        for horizon in horizons:
            values = [fill.markouts_bps[horizon] for fill in self.fills if horizon in fill.markouts_bps]
            markouts[f"{horizon}_ns"] = sum(values) / len(values) if values else None
        return {
            "orders": len(self.orders),
            "fills": len(self.fills),
            "requested_quantity": requested / QUANTITY_SCALE,
            "filled_quantity": filled / QUANTITY_SCALE,
            "fill_rate": filled / requested if requested else 0.0,
            "maker_share": maker_quantity / filled if filled else 0.0,
            "maker_quantity": maker_quantity / QUANTITY_SCALE,
            "taker_quantity": taker_quantity / QUANTITY_SCALE,
            "ending_position": self.ending_position / QUANTITY_SCALE,
            "ending_cash_quote": self.ending_cash_quote,
            "marked_equity_quote": self.marked_equity_quote,
            "realized_fees_quote": self.realized_fees_quote,
            "killed": self.killed,
            "order_status_counts": {
                status.value: sum(order.status is status for order in self.orders)
                for status in OrderStatus
            },
            "mean_signed_markout_bps": markouts,
        }


class ExecutionSimulator:
    """Event-driven L2 execution simulator.

    Passive fills use aggregate trades. Depth-depletion queue models are explicit
    sensitivity scenarios because aggregate depth updates do not reveal exact
    order-level queue position or whether a decrease was a trade or cancellation.
    """

    def __init__(self, config: ExecutionConfig = ExecutionConfig()) -> None:
        self.config = config
        self.book = L2OrderBook()
        self.position = 0
        self.cash_quote = 0.0
        self.realized_fees_quote = 0.0
        self.orders: dict[str, SimulatedOrder] = {}
        self.fills: list[Fill] = []
        self._pending_requests: list[OrderRequest] = []
        self._pending_cancels: list[tuple[int, CancelRequest]] = []
        self._active_order_ids: list[str] = []
        self._pending_markouts: list[int] = []
        self.killed = False
        self.latest_mid: float | None = None
        self._consumed_liquidity: dict[tuple[str, int], int] = {}
        self._has_run = False

    def submit(self, request: OrderRequest) -> None:
        if request.order_id in self.orders or any(
            item.order_id == request.order_id for item in self._pending_requests
        ):
            raise ValueError(f"Duplicate order ID: {request.order_id}")
        self._pending_requests.append(request)
        self._pending_requests.sort(key=lambda item: (item.decision_timestamp_ns, item.order_id))

    def cancel(self, request: CancelRequest) -> None:
        arrival = request.decision_timestamp_ns + self.config.cancel_latency_ns
        if arrival > UINT64_MAX:
            raise OverflowError("Cancel arrival timestamp exceeds 64-bit storage.")
        self._pending_cancels.append((arrival, request))
        self._pending_cancels.sort(key=lambda item: (item[0], item[1].order_id))

    def run(self, recording: str | Path) -> SimulationResult:
        if self._has_run:
            raise RuntimeError("One ExecutionSimulator instance can run only once.")
        self._has_run = True
        for event in iter_events(recording):
            timestamp = event.receipt_timestamp_ns

            # Requests and cancels that arrived strictly before this market-data
            # event must observe the pre-event book. Exact timestamp ties follow
            # file order: the recorded market event is applied first, then local
            # controls at the same timestamp. This avoids silently granting a
            # simulated order priority over an exchange event with an identical
            # local receipt timestamp.
            self._activate_requests(timestamp, inclusive=False)
            self._process_cancels(timestamp, inclusive=False)
            self._expire_orders(timestamp)

            if isinstance(event, Boundary):
                self._cancel_all("market_data_boundary")
                self.book.clear()
                self.latest_mid = None
                self._pending_markouts.clear()
                self._consumed_liquidity.clear()
            elif isinstance(event, Snapshot):
                self.book.install_snapshot(event)
                self._consumed_liquidity.clear()
                self._update_mid_and_markouts(timestamp)
            elif isinstance(event, DepthUpdate):
                self._process_depth_depletion(event)
                if event.final_update_id > self.book.last_update_id:
                    if event.first_update_id > self.book.last_update_id + 1:
                        raise RuntimeError("Execution simulation encountered an L2 sequence gap.")
                    self.book.apply(event)
                    self._consumed_liquidity.clear()
                    self._update_mid_and_markouts(timestamp)
            elif isinstance(event, Trade):
                self._process_trade(event)

            self._activate_requests(timestamp, inclusive=True)
            self._process_cancels(timestamp, inclusive=True)
            self._expire_orders(timestamp)
            self._check_kill_switch()
        self._expire_orders(UINT64_MAX, final=True)
        self._finalize_pending_requests()
        marked_equity = self.cash_quote
        if self.latest_mid is not None:
            marked_equity += self.position / QUANTITY_SCALE * self.latest_mid
        return SimulationResult(
            tuple(self.orders.values()),
            tuple(self.fills),
            self.position,
            self.cash_quote,
            marked_equity,
            self.realized_fees_quote,
            self.killed,
        )

    def _activate_requests(self, timestamp_ns: int, *, inclusive: bool) -> None:
        while self._pending_requests:
            request = self._pending_requests[0]
            arrival = (
                request.decision_timestamp_ns
                + self.config.decision_latency_ns
                + self.config.transmission_latency_ns
            )
            if arrival > UINT64_MAX:
                raise OverflowError("Order arrival timestamp exceeds 64-bit storage.")
            if arrival > timestamp_ns or (not inclusive and arrival == timestamp_ns):
                break
            self._pending_requests.pop(0)
            expiry = (
                arrival + request.time_to_live_ns
                if request.time_to_live_ns is not None
                else None
            )
            order = SimulatedOrder(request, arrival, expiry)
            self.orders[request.order_id] = order
            if self.killed:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "kill_switch_active"
                continue
            if not self._position_allows(request.side, request.quantity):
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "position_limit"
                continue
            if self.book.last_update_id == 0:
                order.status = OrderStatus.REJECTED
                order.rejection_reason = "book_unavailable"
                continue
            if request.order_type is OrderType.MARKET:
                self._execute_taker(order, timestamp_ns)
                continue
            assert request.limit_price is not None
            marketable = (
                request.side is Side.BUY and request.limit_price >= self.book.best_ask.price
            ) or (
                request.side is Side.SELL and request.limit_price <= self.book.best_bid.price
            )
            if marketable:
                self._execute_taker(order, timestamp_ns, limit_price=request.limit_price)
            else:
                side = "bid" if request.side is Side.BUY else "ask"
                displayed = self.book.quantity_at(side, request.limit_price)
                own_queue = sum(
                    self.orders[active_id].remaining_quantity
                    for active_id in self._active_order_ids
                    if self.orders[active_id].request.side is request.side
                    and self.orders[active_id].request.limit_price == request.limit_price
                    and self.orders[active_id].status
                    in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
                )
                order.queue_ahead = (
                    displayed * self.config.queue_ahead_fraction + own_queue
                )
                order.status = OrderStatus.OPEN
                self._active_order_ids.append(request.order_id)

    def _execute_taker(
        self,
        order: SimulatedOrder,
        timestamp_ns: int,
        *,
        limit_price: int | None = None,
    ) -> None:
        levels = self.book.asks() if order.request.side is Side.BUY else self.book.bids()
        remaining = order.remaining_quantity
        book_side = "ask" if order.request.side is Side.BUY else "bid"
        for level in levels:
            if remaining <= 0:
                break
            if limit_price is not None:
                if order.request.side is Side.BUY and level.price > limit_price:
                    break
                if order.request.side is Side.SELL and level.price < limit_price:
                    break
            key = (book_side, level.price)
            available = max(0, level.quantity - self._consumed_liquidity.get(key, 0))
            quantity = min(remaining, available)
            if quantity <= 0:
                continue
            self._record_fill(order, timestamp_ns, quantity, level.price, maker=False)
            self._consumed_liquidity[key] = self._consumed_liquidity.get(key, 0) + quantity
            remaining -= quantity
        if order.remaining_quantity == 0:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
            order.rejection_reason = "insufficient_visible_depth"
        else:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "insufficient_visible_depth"

    def _process_trade(self, trade: Trade) -> None:
        if not self._active_order_ids:
            return
        aggressor_side = Side.SELL if trade.buyer_is_maker else Side.BUY
        trade_quantity = float(trade.quantity)
        eligible: list[SimulatedOrder] = []
        for order_id in self._active_order_ids:
            order = self.orders[order_id]
            if order.status not in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                continue
            if order.request.side is aggressor_side:
                continue
            if order.request.limit_price != trade.price:
                continue
            eligible.append(order)
        eligible.sort(key=lambda item: (item.arrival_timestamp_ns, item.request.order_id))
        for order in eligible:
            queue_before = order.queue_ahead
            order.queue_ahead = max(0.0, queue_before - trade_quantity)
            executable = max(0.0, trade_quantity - queue_before)
            fill_quantity = min(order.remaining_quantity, int(executable))
            if fill_quantity > 0:
                self._record_fill(
                    order,
                    trade.receipt_timestamp_ns,
                    fill_quantity,
                    trade.price,
                    maker=True,
                )
                order.status = (
                    OrderStatus.FILLED
                    if order.remaining_quantity == 0
                    else OrderStatus.PARTIALLY_FILLED
                )
        self._active_order_ids = [
            order_id
            for order_id in self._active_order_ids
            if self.orders[order_id].status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
        ]

    def _process_depth_depletion(self, update: DepthUpdate) -> None:
        if self.config.queue_model is QueueModel.TRADE_ONLY or self.book.last_update_id == 0:
            return
        decreases: dict[tuple[str, int], int] = {}
        for level in update.bids:
            old = self.book.quantity_at("bid", level.price)
            if level.quantity < old:
                decreases[("bid", level.price)] = old - level.quantity
        for level in update.asks:
            old = self.book.quantity_at("ask", level.price)
            if level.quantity < old:
                decreases[("ask", level.price)] = old - level.quantity
        fraction = (
            1.0
            if self.config.queue_model is QueueModel.OPTIMISTIC_DEPTH
            else self.config.depth_depletion_fill_fraction
        )
        for order_id in list(self._active_order_ids):
            order = self.orders[order_id]
            assert order.request.limit_price is not None
            key = ("bid" if order.request.side is Side.BUY else "ask", order.request.limit_price)
            depletion = decreases.get(key, 0) * fraction
            if depletion <= 0:
                continue
            queue_reduction = min(order.queue_ahead, depletion)
            order.queue_ahead -= queue_reduction
            depletion -= queue_reduction
            fill_quantity = min(order.remaining_quantity, int(depletion))
            if fill_quantity > 0:
                self._record_fill(
                    order,
                    update.receipt_timestamp_ns,
                    fill_quantity,
                    order.request.limit_price,
                    maker=True,
                )
                order.status = (
                    OrderStatus.FILLED
                    if order.remaining_quantity == 0
                    else OrderStatus.PARTIALLY_FILLED
                )
        self._active_order_ids = [
            order_id
            for order_id in self._active_order_ids
            if self.orders[order_id].status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
        ]

    def _record_fill(
        self,
        order: SimulatedOrder,
        timestamp_ns: int,
        quantity: int,
        price: int,
        *,
        maker: bool,
    ) -> None:
        if quantity <= 0:
            return
        if not self._position_allows(order.request.side, quantity):
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "position_limit_during_partial_fill"
            return
        price_float = price / PRICE_SCALE
        quantity_float = quantity / QUANTITY_SCALE
        notional = price_float * quantity_float
        fee_bps = self.config.maker_fee_bps if maker else self.config.taker_fee_bps
        fee = notional * fee_bps / 10_000.0
        sign = 1 if order.request.side is Side.BUY else -1
        self.position += sign * quantity
        self.cash_quote -= sign * notional
        self.cash_quote -= fee
        self.realized_fees_quote += fee
        order.filled_quantity += quantity
        fill = Fill(order.request.order_id, timestamp_ns, order.request.side, quantity, price, maker, fee)
        self.fills.append(fill)
        self._pending_markouts.append(len(self.fills) - 1)

    def _update_mid_and_markouts(self, timestamp_ns: int) -> None:
        if self.book.last_update_id == 0:
            return
        self.latest_mid = (self.book.best_bid.price + self.book.best_ask.price) / (2 * PRICE_SCALE)
        remaining: list[int] = []
        for index in self._pending_markouts:
            fill = self.fills[index]
            for horizon in self.config.markout_horizons_ns:
                if horizon in fill.markouts_bps or timestamp_ns < fill.timestamp_ns + horizon:
                    continue
                raw = (self.latest_mid - fill.price_float) / fill.price_float * 10_000.0
                fill.markouts_bps[horizon] = raw if fill.side is Side.BUY else -raw
            if len(fill.markouts_bps) < len(self.config.markout_horizons_ns):
                remaining.append(index)
        self._pending_markouts = remaining

    def _process_cancels(self, timestamp_ns: int, *, inclusive: bool) -> None:
        while self._pending_cancels:
            arrival, request = self._pending_cancels[0]
            if arrival > timestamp_ns or (not inclusive and arrival == timestamp_ns):
                break
            self._pending_cancels.pop(0)
            order = self.orders.get(request.order_id)
            if order and order.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                order.status = OrderStatus.CANCELLED
                self._active_order_ids = [
                    item for item in self._active_order_ids if item != request.order_id
                ]
                continue

            # A separately configured cancel path may beat the original order
            # to the venue. Preserve that outcome instead of silently dropping
            # the cancel request and activating the order later.
            pending_index = next(
                (
                    index
                    for index, pending in enumerate(self._pending_requests)
                    if pending.order_id == request.order_id
                ),
                None,
            )
            if pending_index is not None:
                pending = self._pending_requests.pop(pending_index)
                order_arrival = (
                    pending.decision_timestamp_ns
                    + self.config.decision_latency_ns
                    + self.config.transmission_latency_ns
                )
                expiry = (
                    order_arrival + pending.time_to_live_ns
                    if pending.time_to_live_ns is not None
                    else None
                )
                cancelled = SimulatedOrder(
                    pending, order_arrival, expiry, status=OrderStatus.CANCELLED
                )
                cancelled.rejection_reason = "cancel_arrived_before_order"
                self.orders[pending.order_id] = cancelled

    def _expire_orders(self, timestamp_ns: int, *, final: bool = False) -> None:
        for order_id in list(self._active_order_ids):
            order = self.orders[order_id]
            if final or (
                order.expiry_timestamp_ns is not None
                and order.expiry_timestamp_ns <= timestamp_ns
            ):
                order.status = OrderStatus.EXPIRED
                self._active_order_ids.remove(order_id)

    def _finalize_pending_requests(self) -> None:
        for request in self._pending_requests:
            arrival = (
                request.decision_timestamp_ns
                + self.config.decision_latency_ns
                + self.config.transmission_latency_ns
            )
            expiry = (
                arrival + request.time_to_live_ns
                if request.time_to_live_ns is not None
                else None
            )
            order = SimulatedOrder(request, arrival, expiry, status=OrderStatus.EXPIRED)
            order.rejection_reason = "recording_ended_before_arrival"
            self.orders[request.order_id] = order
        self._pending_requests.clear()

    def _cancel_all(self, reason: str) -> None:
        for order_id in self._active_order_ids:
            order = self.orders[order_id]
            if order.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                order.status = OrderStatus.CANCELLED
                order.rejection_reason = reason
        self._active_order_ids.clear()

    def _position_allows(self, side: Side, quantity: int) -> bool:
        projected = self.position + (quantity if side is Side.BUY else -quantity)
        return abs(projected) <= self.config.max_absolute_position

    def _check_kill_switch(self) -> None:
        if self.killed or not math.isfinite(self.config.kill_switch_loss_quote):
            return
        equity = self.cash_quote
        if self.latest_mid is not None:
            equity += self.position / QUANTITY_SCALE * self.latest_mid
        if equity <= -self.config.kill_switch_loss_quote:
            self.killed = True
            self._cancel_all("kill_switch")


def load_orders(path: str | Path) -> list[OrderRequest]:
    orders: list[OrderRequest] = []
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"order_id", "decision_timestamp_ns", "side", "order_type", "quantity"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Order CSV must contain columns: {sorted(required)}")
        for row in reader:
            order_type = OrderType(row["order_type"].strip().lower())
            raw_limit = (row.get("limit_price") or "").strip()
            raw_ttl = (row.get("time_to_live_ns") or "").strip()
            orders.append(
                OrderRequest(
                    order_id=row["order_id"],
                    decision_timestamp_ns=int(row["decision_timestamp_ns"]),
                    side=Side(row["side"].strip().lower()),
                    order_type=order_type,
                    quantity=parse_quantity(row["quantity"]),
                    limit_price=parse_price(raw_limit) if raw_limit else None,
                    time_to_live_ns=int(raw_ttl) if raw_ttl else None,
                )
            )
    return orders


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the event-driven L2 execution simulator.")
    parser.add_argument("recording")
    parser.add_argument("orders", help="CSV of order requests.")
    parser.add_argument("--queue-model", choices=[item.value for item in QueueModel], default="trade_only")
    parser.add_argument("--decision-latency-us", type=float, default=0.0)
    parser.add_argument("--transmission-latency-us", type=float, default=250.0)
    parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    parser.add_argument("--taker-fee-bps", type=float, default=0.0)
    parser.add_argument("--queue-ahead-fraction", type=float, default=1.0)
    parser.add_argument("--depth-depletion-fill-fraction", type=float, default=0.5)
    parser.add_argument("--max-position", type=float, default=10.0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = ExecutionConfig(
        decision_latency_ns=int(arguments.decision_latency_us * 1_000),
        transmission_latency_ns=int(arguments.transmission_latency_us * 1_000),
        maker_fee_bps=arguments.maker_fee_bps,
        taker_fee_bps=arguments.taker_fee_bps,
        queue_model=QueueModel(arguments.queue_model),
        queue_ahead_fraction=arguments.queue_ahead_fraction,
        depth_depletion_fill_fraction=arguments.depth_depletion_fill_fraction,
        max_absolute_position=parse_quantity(arguments.max_position),
    )
    simulator = ExecutionSimulator(config)
    for order in load_orders(arguments.orders):
        simulator.submit(order)
    result = simulator.run(arguments.recording)
    payload = result.summary()
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if arguments.output:
        Path(arguments.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
