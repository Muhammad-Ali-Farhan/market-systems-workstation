from __future__ import annotations

import enum
import re
import struct
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence

PRICE_SCALE = 100_000_000
QUANTITY_SCALE = 100_000_000
SCALE_DECIMALS = 8
UINT64_MAX = (1 << 64) - 1
INT64_MAX = (1 << 63) - 1
_FIXED_DECIMAL_TEXT = re.compile(r"^\+?(?:\d+(?:\.\d*)?|\.\d+)$")


def _validate_u64(name: str, value: int, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    minimum = 1 if positive else 0
    if value < minimum or value > UINT64_MAX:
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{name} must be a {qualifier}unsigned 64-bit integer.")


@dataclass(frozen=True, slots=True)
class Level:
    price: int
    quantity: int

    def __post_init__(self) -> None:
        if self.price <= 0 or self.price > INT64_MAX:
            raise ValueError("L2 price must be a positive signed 64-bit integer.")
        if self.quantity < 0 or self.quantity > UINT64_MAX:
            raise ValueError("L2 quantity must fit in an unsigned 64-bit integer.")


@dataclass(frozen=True, slots=True)
class Snapshot:
    receipt_timestamp_ns: int
    last_update_id: int
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]

    def __post_init__(self) -> None:
        _validate_u64("Snapshot receipt timestamp", self.receipt_timestamp_ns, positive=True)
        _validate_u64("Snapshot update ID", self.last_update_id, positive=True)


@dataclass(frozen=True, slots=True)
class DepthUpdate:
    receipt_timestamp_ns: int
    event_time_ms: int
    first_update_id: int
    final_update_id: int
    bids: tuple[Level, ...] = field(default_factory=tuple)
    asks: tuple[Level, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_u64("Depth receipt timestamp", self.receipt_timestamp_ns, positive=True)
        _validate_u64("Depth exchange timestamp", self.event_time_ms)
        _validate_u64("Depth first update ID", self.first_update_id, positive=True)
        _validate_u64("Depth final update ID", self.final_update_id, positive=True)
        if self.first_update_id > self.final_update_id:
            raise ValueError("Invalid depth update-ID range.")


@dataclass(frozen=True, slots=True)
class Trade:
    receipt_timestamp_ns: int
    event_time_ms: int
    aggregate_trade_id: int
    price: int
    quantity: int
    buyer_is_maker: bool

    def __post_init__(self) -> None:
        _validate_u64("Trade receipt timestamp", self.receipt_timestamp_ns, positive=True)
        _validate_u64("Trade exchange timestamp", self.event_time_ms)
        _validate_u64("Aggregate trade ID", self.aggregate_trade_id, positive=True)
        if not isinstance(self.buyer_is_maker, bool):
            raise TypeError("buyer_is_maker must be a boolean.")
        Level(self.price, self.quantity)
        if self.quantity == 0:
            raise ValueError("Trade quantity must be positive.")


def parse_fixed_decimal(value: str | int | float | Decimal, scale: int) -> int:
    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise ValueError("Scale must be a positive integer.")
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid fixed-point decimals.")
    if isinstance(value, str) and _FIXED_DECIMAL_TEXT.fullmatch(value) is None:
        raise ValueError(f"Invalid fixed-point decimal syntax: {value!r}")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exception:
        raise ValueError(f"Invalid fixed-point decimal: {value!r}") from exception
    if not decimal_value.is_finite() or decimal_value < 0:
        raise ValueError(f"Fixed-point decimal must be finite and non-negative: {value!r}")
    scaled = decimal_value * scale
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise ValueError(
            f"Value {value!r} has more precision than scale {scale:,} supports."
        )
    result = int(integral)
    if result > UINT64_MAX:
        raise OverflowError("Fixed-point value exceeds 64-bit storage.")
    return result


def parse_price(value: str | int | float | Decimal) -> int:
    result = parse_fixed_decimal(value, PRICE_SCALE)
    if result <= 0 or result > INT64_MAX:
        raise ValueError("Price must be positive and fit in signed 64-bit storage.")
    return result


def parse_quantity(value: str | int | float | Decimal) -> int:
    return parse_fixed_decimal(value, QUANTITY_SCALE)


def price_to_float(value: int) -> float:
    return int(value) / PRICE_SCALE


def quantity_to_float(value: int) -> float:
    return int(value) / QUANTITY_SCALE


def parse_levels(values: Iterable[Sequence[object]]) -> tuple[Level, ...]:
    output: list[Level] = []
    for item in values:
        if len(item) != 2:
            raise ValueError("Each depth level must contain price and quantity.")
        output.append(Level(parse_price(item[0]), parse_quantity(item[1])))
    return tuple(output)


class L2OrderBook:
    """Exact fixed-point reference L2 book.

    Dictionaries make mutations unambiguous; sorting is performed only when an
    ordered view or hash is requested. The native implementation uses a flat,
    sorted representation and is cross-checked against this reference model.
    """

    def __init__(self) -> None:
        self._bids: dict[int, int] = {}
        self._asks: dict[int, int] = {}
        self.last_update_id = 0

    def clear(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self.last_update_id = 0

    def install_snapshot(self, snapshot: Snapshot) -> None:
        bids: dict[int, int] = {}
        asks: dict[int, int] = {}
        for level in snapshot.bids:
            if level.quantity:
                bids[level.price] = level.quantity
            else:
                bids.pop(level.price, None)
        for level in snapshot.asks:
            if level.quantity:
                asks[level.price] = level.quantity
            else:
                asks.pop(level.price, None)
        self._bids = bids
        self._asks = asks
        self.last_update_id = snapshot.last_update_id
        self.validate()

    def apply(self, update: DepthUpdate) -> None:
        for level in update.bids:
            self._set_level(self._bids, level)
        for level in update.asks:
            self._set_level(self._asks, level)
        self.last_update_id = update.final_update_id
        self.validate()

    @staticmethod
    def _set_level(side: dict[int, int], level: Level) -> None:
        if level.quantity == 0:
            side.pop(level.price, None)
        else:
            side[level.price] = level.quantity

    @property
    def best_bid(self) -> Level:
        if not self._bids:
            raise RuntimeError("Bid book is empty.")
        price = max(self._bids)
        return Level(price, self._bids[price])

    @property
    def best_ask(self) -> Level:
        if not self._asks:
            raise RuntimeError("Ask book is empty.")
        price = min(self._asks)
        return Level(price, self._asks[price])

    def bids(self, limit: int = 0) -> tuple[Level, ...]:
        prices = sorted(self._bids, reverse=True)
        if limit > 0:
            prices = prices[:limit]
        return tuple(Level(price, self._bids[price]) for price in prices)

    def asks(self, limit: int = 0) -> tuple[Level, ...]:
        prices = sorted(self._asks)
        if limit > 0:
            prices = prices[:limit]
        return tuple(Level(price, self._asks[price]) for price in prices)

    def quantity_at(self, side: str, price: int) -> int:
        if side == "bid":
            return self._bids.get(int(price), 0)
        if side == "ask":
            return self._asks.get(int(price), 0)
        raise ValueError("Side must be 'bid' or 'ask'.")

    def validate(self) -> None:
        if not self._bids or not self._asks:
            raise RuntimeError("L2 order book must remain two-sided.")
        if any(price <= 0 or quantity <= 0 for price, quantity in self._bids.items()):
            raise RuntimeError("Bid book contains an invalid active level.")
        if any(price <= 0 or quantity <= 0 for price, quantity in self._asks.items()):
            raise RuntimeError("Ask book contains an invalid active level.")
        if max(self._bids) >= min(self._asks):
            raise RuntimeError("L2 order book is crossed or locked.")

    def state_hash(self) -> int:
        value = 14_695_981_039_346_656_037

        def mix(raw: bytes) -> None:
            nonlocal value
            for byte in raw:
                value ^= byte
                value = (value * 1_099_511_628_211) & UINT64_MAX

        bid_levels = self.bids()
        ask_levels = self.asks()
        mix(struct.pack("<Q", self.last_update_id))
        mix(struct.pack("<Q", len(bid_levels)))
        for level in bid_levels:
            mix(struct.pack("<qQ", level.price, level.quantity))
        mix(struct.pack("<Q", len(ask_levels)))
        for level in ask_levels:
            mix(struct.pack("<qQ", level.price, level.quantity))
        return value


class SyncState(str, enum.Enum):
    AWAITING_SNAPSHOT = "awaiting_snapshot"
    LIVE = "live"
    GAP = "gap"


class ApplyResult(str, enum.Enum):
    APPLIED = "applied"
    IGNORED_STALE = "ignored_stale"
    BUFFERED = "buffered"
    GAP_DETECTED = "gap_detected"


class SnapshotResult(str, enum.Enum):
    SYNCHRONIZED = "synchronized"
    AWAITING_BRIDGE = "awaiting_bridge"
    SNAPSHOT_TOO_OLD = "snapshot_too_old"
    GAP_DETECTED = "gap_detected"


@dataclass(frozen=True, slots=True)
class SnapshotInstallResult:
    result: SnapshotResult
    stale_events: int
    applied_events: tuple[DepthUpdate, ...]
    first_applied_buffer_index: int


class DepthSynchronizer:
    """Implements Binance's documented snapshot + diff-depth procedure."""

    def __init__(self, maximum_buffered_events: int = 200_000) -> None:
        if maximum_buffered_events <= 0:
            raise ValueError("maximum_buffered_events must be positive.")
        self.maximum_buffered_events = int(maximum_buffered_events)
        self.book = L2OrderBook()
        self.state = SyncState.AWAITING_SNAPSHOT
        self._buffer: list[DepthUpdate] = []
        self.reset_count = 0

    @property
    def buffered_events(self) -> tuple[DepthUpdate, ...]:
        return tuple(self._buffer)

    def reset(self, *, preserve_buffer: bool = False) -> None:
        existing = self._buffer if preserve_buffer else []
        self.book.clear()
        self._buffer = existing
        self.state = SyncState.AWAITING_SNAPSHOT
        self.reset_count += 1

    def ingest(self, update: DepthUpdate) -> ApplyResult:
        if self.state is not SyncState.LIVE:
            self._buffer_event(update)
            return ApplyResult.BUFFERED
        return self._apply_live(update)

    def install_snapshot(self, snapshot: Snapshot) -> SnapshotInstallResult:
        stale = 0
        while self._buffer and self._buffer[0].final_update_id <= snapshot.last_update_id:
            self._buffer.pop(0)
            stale += 1

        if not self._buffer:
            self.book.install_snapshot(snapshot)
            self.state = SyncState.AWAITING_SNAPSHOT
            return SnapshotInstallResult(
                SnapshotResult.AWAITING_BRIDGE,
                stale,
                (),
                stale,
            )

        expected = snapshot.last_update_id + 1
        first = self._buffer[0]
        if first.first_update_id > expected:
            return SnapshotInstallResult(
                SnapshotResult.SNAPSHOT_TOO_OLD,
                stale,
                (),
                stale,
            )
        if first.final_update_id < expected:
            raise RuntimeError("Stale events remained after snapshot filtering.")

        self.book.install_snapshot(snapshot)
        self.state = SyncState.LIVE
        pending = self._buffer
        self._buffer = []
        applied: list[DepthUpdate] = []
        for index, update in enumerate(pending):
            result = self._apply_live(update)
            if result is ApplyResult.APPLIED:
                applied.append(update)
            elif result is ApplyResult.GAP_DETECTED:
                self._buffer.extend(pending[index + 1 :])
                return SnapshotInstallResult(
                    SnapshotResult.GAP_DETECTED,
                    stale,
                    tuple(applied),
                    stale,
                )
        return SnapshotInstallResult(
            SnapshotResult.SYNCHRONIZED,
            stale,
            tuple(applied),
            stale,
        )

    def _buffer_event(self, update: DepthUpdate) -> None:
        if len(self._buffer) >= self.maximum_buffered_events:
            self.state = SyncState.GAP
            raise RuntimeError("L2 synchronization buffer capacity exceeded.")
        if self._buffer and update.final_update_id < self._buffer[-1].final_update_id:
            self.state = SyncState.GAP
            raise RuntimeError("L2 depth events arrived out of order.")
        self._buffer.append(update)

    def _apply_live(self, update: DepthUpdate) -> ApplyResult:
        local = self.book.last_update_id
        if update.final_update_id <= local:
            return ApplyResult.IGNORED_STALE
        if update.first_update_id > local + 1:
            self.state = SyncState.GAP
            self._buffer_event(update)
            return ApplyResult.GAP_DETECTED
        self.book.apply(update)
        self.state = SyncState.LIVE
        return ApplyResult.APPLIED
