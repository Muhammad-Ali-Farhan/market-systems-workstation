from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from l2bin import Boundary, iter_events, read_metadata
from l2book import (
    PRICE_SCALE,
    QUANTITY_SCALE,
    DepthUpdate,
    L2OrderBook,
    Snapshot,
    Trade,
)

FEATURE_NAMES = (
    "spread_bps",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "imbalance_20",
    "microprice_edge_1_bps",
    "depth_weighted_mid_edge_5_bps",
    "log_bid_depth_5",
    "log_ask_depth_5",
    "bid_depth_slope_10",
    "ask_depth_slope_10",
    "bid_concentration_1_10",
    "ask_concentration_1_10",
    "bid_convexity_5_20",
    "ask_convexity_5_20",
    "top_of_book_ofi",
    "depth_addition_imbalance",
    "cancellation_imbalance",
    "trade_imbalance_20",
    "mid_return_1",
    "mid_return_5",
    "realized_volatility_20",
    "log_interarrival_us",
    "event_rate_10",
    "spread_change_1",
    "imbalance_change_1",
)
FEATURE_SCHEMA_VERSION = 1
FEATURE_SCHEMA_DEFINITION = "\n".join(
    (
        f"feature_schema_version={FEATURE_SCHEMA_VERSION}",
        "source=sequence_correct_aggregated_l2",
        f"price_scale={PRICE_SCALE}",
        f"quantity_scale={QUANTITY_SCALE}",
        "depth_levels=1,5,10,20",
        "history_depth_events=25",
        "trade_window=20_aggregate_trades",
        "returns=log_bps",
        "volatility=population_std_latest_20_depth_returns",
        *FEATURE_NAMES,
    )
)
FEATURE_SCHEMA_HASH = hashlib.sha256(
    FEATURE_SCHEMA_DEFINITION.encode("utf-8")
).hexdigest()
MAX_HISTORY = 25
EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    features: np.ndarray
    timestamp_ns: int
    update_id: int
    mid: float
    best_bid: float
    best_ask: float
    best_bid_quantity: float
    best_ask_quantity: float
    imbalance_1: float
    microprice_edge_bps: float


@dataclass(frozen=True, slots=True)
class L2FeatureSet:
    X: np.ndarray
    y: np.ndarray
    timestamps_ns: np.ndarray
    update_ids: np.ndarray
    session_id: np.ndarray
    event_index: np.ndarray
    current_mid: np.ndarray
    current_bid: np.ndarray
    current_ask: np.ndarray
    current_bid_quantity: np.ndarray
    current_ask_quantity: np.ndarray
    future_bid_quantity: np.ndarray
    future_ask_quantity: np.ndarray
    future_bid: np.ndarray
    future_ask: np.ndarray
    imbalance: np.ndarray
    microprice_edge_bps: np.ndarray
    horizon: int

    @property
    def size(self) -> int:
        return int(self.y.size)


def _imbalance(bid_quantity: float, ask_quantity: float) -> float:
    total = bid_quantity + ask_quantity
    return (bid_quantity - ask_quantity) / total if total > 0.0 else 0.0


def _depth_slope(levels: tuple, best_price: int, *, bid_side: bool, limit: int) -> float:
    selected = levels[:limit]
    if len(selected) < 2:
        return 0.0
    distances: list[float] = []
    cumulative: list[float] = []
    running = 0.0
    for level in selected:
        running += level.quantity / QUANTITY_SCALE
        distance_bps = (
            (best_price - level.price) / best_price * 10_000.0
            if bid_side
            else (level.price - best_price) / best_price * 10_000.0
        )
        distances.append(max(distance_bps, 0.0))
        cumulative.append(math.log1p(running))
    x = np.asarray(distances, dtype=np.float64)
    y = np.asarray(cumulative, dtype=np.float64)
    centered = x - x.mean()
    denominator = float(centered @ centered)
    return float(centered @ (y - y.mean()) / denominator) if denominator > 0.0 else 0.0


def _convexity(levels: tuple, near: int = 5, far: int = 20) -> float:
    near_quantity = sum(level.quantity for level in levels[:near]) / QUANTITY_SCALE
    far_quantity = sum(level.quantity for level in levels[:far]) / QUANTITY_SCALE
    return math.log((far_quantity + EPSILON) / (near_quantity + EPSILON))


class L2FeatureBuilder:
    def __init__(self) -> None:
        self._mid_history: deque[float] = deque(maxlen=MAX_HISTORY + 1)
        self._return_history: deque[float] = deque(maxlen=20)
        self._timestamp_history: deque[int] = deque(maxlen=11)
        self._trade_signed_quantity: deque[float] = deque(maxlen=20)
        self._previous_spread_bps: float | None = None
        self._previous_imbalance: float | None = None
        self._previous_best_bid: tuple[int, int] | None = None
        self._previous_best_ask: tuple[int, int] | None = None

    def reset(self) -> None:
        self._mid_history.clear()
        self._return_history.clear()
        self._timestamp_history.clear()
        self._trade_signed_quantity.clear()
        self._previous_spread_bps = None
        self._previous_imbalance = None
        self._previous_best_bid = None
        self._previous_best_ask = None

    def on_trade(self, trade: Trade) -> None:
        quantity = trade.quantity / QUANTITY_SCALE
        # buyer_is_maker=True means the seller was the aggressor.
        signed = -quantity if trade.buyer_is_maker else quantity
        self._trade_signed_quantity.append(signed)

    def apply_and_build(
        self,
        book: L2OrderBook,
        update: DepthUpdate,
    ) -> FeatureObservation | None:
        old_bid_quantities = {
            level.price: book.quantity_at("bid", level.price) for level in update.bids
        }
        old_ask_quantities = {
            level.price: book.quantity_at("ask", level.price) for level in update.asks
        }
        previous_best_bid = (
            (book.best_bid.price, book.best_bid.quantity) if book.last_update_id else None
        )
        previous_best_ask = (
            (book.best_ask.price, book.best_ask.quantity) if book.last_update_id else None
        )
        book.apply(update)

        bid_levels = book.bids(20)
        ask_levels = book.asks(20)
        best_bid = bid_levels[0]
        best_ask = ask_levels[0]
        best_bid_float = best_bid.price / PRICE_SCALE
        best_ask_float = best_ask.price / PRICE_SCALE
        mid = (best_bid_float + best_ask_float) / 2.0
        spread_bps = (best_ask_float - best_bid_float) / mid * 10_000.0

        def depth(side_levels: tuple, count: int) -> float:
            return sum(level.quantity for level in side_levels[:count]) / QUANTITY_SCALE

        imbalances: dict[int, float] = {}
        for count in (1, 5, 10, 20):
            imbalances[count] = _imbalance(
                depth(bid_levels, count), depth(ask_levels, count)
            )

        bid_quantity = best_bid.quantity / QUANTITY_SCALE
        ask_quantity = best_ask.quantity / QUANTITY_SCALE
        total_top = bid_quantity + ask_quantity
        microprice = (
            (best_ask_float * bid_quantity + best_bid_float * ask_quantity) / total_top
            if total_top > 0.0
            else mid
        )

        weighted_numerator = 0.0
        weighted_denominator = 0.0
        for level in bid_levels[:5]:
            quantity = level.quantity / QUANTITY_SCALE
            weighted_numerator += (level.price / PRICE_SCALE) * quantity
            weighted_denominator += quantity
        for level in ask_levels[:5]:
            quantity = level.quantity / QUANTITY_SCALE
            weighted_numerator += (level.price / PRICE_SCALE) * quantity
            weighted_denominator += quantity
        depth_weighted_mid = (
            weighted_numerator / weighted_denominator if weighted_denominator else mid
        )

        additions_bid = cancellations_bid = 0
        additions_ask = cancellations_ask = 0
        for level in update.bids:
            change = level.quantity - old_bid_quantities[level.price]
            if change >= 0:
                additions_bid += change
            else:
                cancellations_bid -= change
        for level in update.asks:
            change = level.quantity - old_ask_quantities[level.price]
            if change >= 0:
                additions_ask += change
            else:
                cancellations_ask -= change
        addition_imbalance = _imbalance(additions_bid, additions_ask)
        cancellation_imbalance = _imbalance(cancellations_ask, cancellations_bid)

        top_of_book_ofi = 0.0
        if previous_best_bid is not None and previous_best_ask is not None:
            old_bid_price, old_bid_quantity = previous_best_bid
            old_ask_price, old_ask_quantity = previous_best_ask
            if best_bid.price > old_bid_price:
                top_of_book_ofi += best_bid.quantity
            elif best_bid.price == old_bid_price:
                top_of_book_ofi += best_bid.quantity - old_bid_quantity
            else:
                top_of_book_ofi -= old_bid_quantity
            if best_ask.price < old_ask_price:
                top_of_book_ofi -= best_ask.quantity
            elif best_ask.price == old_ask_price:
                top_of_book_ofi -= best_ask.quantity - old_ask_quantity
            else:
                top_of_book_ofi += old_ask_quantity
            top_of_book_ofi /= QUANTITY_SCALE

        previous_mid = self._mid_history[-1] if self._mid_history else None
        mid_return_1 = (
            math.log(mid / previous_mid) * 10_000.0 if previous_mid else 0.0
        )
        self._mid_history.append(mid)
        self._return_history.append(mid_return_1)
        self._timestamp_history.append(update.receipt_timestamp_ns)
        mid_return_5 = (
            math.log(mid / self._mid_history[-6]) * 10_000.0
            if len(self._mid_history) >= 6
            else 0.0
        )
        realized_volatility = (
            float(np.std(np.asarray(self._return_history, dtype=np.float64), ddof=0))
            if len(self._return_history) >= 20
            else 0.0
        )
        interarrival_ns = (
            self._timestamp_history[-1] - self._timestamp_history[-2]
            if len(self._timestamp_history) >= 2
            else 0
        )
        event_rate_10 = (
            10.0
            / ((self._timestamp_history[-1] - self._timestamp_history[0]) / 1e9)
            if len(self._timestamp_history) >= 11
            and self._timestamp_history[-1] > self._timestamp_history[0]
            else 0.0
        )
        trade_total = sum(abs(value) for value in self._trade_signed_quantity)
        trade_imbalance = (
            sum(self._trade_signed_quantity) / trade_total if trade_total else 0.0
        )
        spread_change = (
            spread_bps - self._previous_spread_bps
            if self._previous_spread_bps is not None
            else 0.0
        )
        imbalance_change = (
            imbalances[1] - self._previous_imbalance
            if self._previous_imbalance is not None
            else 0.0
        )
        self._previous_spread_bps = spread_bps
        self._previous_imbalance = imbalances[1]
        self._previous_best_bid = (best_bid.price, best_bid.quantity)
        self._previous_best_ask = (best_ask.price, best_ask.quantity)

        features = np.asarray(
            [
                spread_bps,
                imbalances[1],
                imbalances[5],
                imbalances[10],
                imbalances[20],
                (microprice - mid) / mid * 10_000.0,
                (depth_weighted_mid - mid) / mid * 10_000.0,
                math.log1p(depth(bid_levels, 5)),
                math.log1p(depth(ask_levels, 5)),
                _depth_slope(bid_levels, best_bid.price, bid_side=True, limit=10),
                _depth_slope(ask_levels, best_ask.price, bid_side=False, limit=10),
                bid_quantity / max(depth(bid_levels, 10), EPSILON),
                ask_quantity / max(depth(ask_levels, 10), EPSILON),
                _convexity(bid_levels),
                _convexity(ask_levels),
                top_of_book_ofi,
                addition_imbalance,
                cancellation_imbalance,
                trade_imbalance,
                mid_return_1,
                mid_return_5,
                realized_volatility,
                math.log1p(max(interarrival_ns, 0) / 1_000.0),
                event_rate_10,
                spread_change,
                imbalance_change,
            ],
            dtype=np.float64,
        )
        if features.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(features)):
            return None
        if len(self._mid_history) < MAX_HISTORY + 1:
            return None
        return FeatureObservation(
            features=features,
            timestamp_ns=update.receipt_timestamp_ns,
            update_id=update.final_update_id,
            mid=mid,
            best_bid=best_bid_float,
            best_ask=best_ask_float,
            best_bid_quantity=bid_quantity,
            best_ask_quantity=ask_quantity,
            imbalance_1=imbalances[1],
            microprice_edge_bps=(microprice - mid) / mid * 10_000.0,
        )


def build_feature_set(
    recordings: Iterable[str | Path],
    *,
    horizon: int,
    require_complete: bool = True,
) -> L2FeatureSet:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    recording_paths = tuple(Path(value).expanduser().resolve() for value in recordings)
    if not recording_paths:
        raise ValueError("At least one L2 recording is required.")
    if require_complete:
        for recording in recording_paths:
            metadata = read_metadata(recording, verify_hashes=True)
            if metadata.data_complete is not True:
                raise RuntimeError(
                    f"Refusing to build research features from an incomplete L2 recording: {recording}"
                )
    all_rows: list[tuple[int, int, FeatureObservation, FeatureObservation]] = []
    for session_id, recording in enumerate(recording_paths):
        book = L2OrderBook()
        builder = L2FeatureBuilder()
        segment: list[FeatureObservation] = []
        event_index = 0

        def finish_segment() -> None:
            nonlocal event_index
            if len(segment) > horizon:
                for index in range(len(segment) - horizon):
                    all_rows.append(
                        (session_id, event_index, segment[index], segment[index + horizon])
                    )
                    event_index += 1
            segment.clear()

        for event in iter_events(recording):
            if isinstance(event, Boundary):
                finish_segment()
                book.clear()
                builder.reset()
            elif isinstance(event, Snapshot):
                finish_segment()
                book.install_snapshot(event)
                builder.reset()
            elif isinstance(event, Trade):
                builder.on_trade(event)
            else:
                if book.last_update_id == 0:
                    raise RuntimeError("Depth event appeared before an L2 snapshot.")
                if event.first_update_id > book.last_update_id + 1:
                    raise RuntimeError("L2 feature build encountered a sequence gap.")
                if event.final_update_id <= book.last_update_id:
                    continue
                observation = builder.apply_and_build(book, event)
                if observation is not None:
                    segment.append(observation)
        finish_segment()

    if not all_rows:
        raise ValueError("L2 recordings produced no usable feature rows.")
    current = [row[2] for row in all_rows]
    future = [row[3] for row in all_rows]
    current_mid = np.asarray([row.mid for row in current], dtype=np.float64)
    future_mid = np.asarray([row.mid for row in future], dtype=np.float64)
    return L2FeatureSet(
        X=np.ascontiguousarray(
            np.vstack([row.features for row in current]), dtype=np.float64
        ),
        y=np.ascontiguousarray(np.log(future_mid / current_mid) * 10_000.0),
        timestamps_ns=np.asarray([row.timestamp_ns for row in current], dtype=np.uint64),
        update_ids=np.asarray([row.update_id for row in current], dtype=np.uint64),
        session_id=np.asarray([row[0] for row in all_rows], dtype=np.int32),
        event_index=np.asarray([row[1] for row in all_rows], dtype=np.int64),
        current_mid=current_mid,
        current_bid=np.asarray([row.best_bid for row in current], dtype=np.float64),
        current_ask=np.asarray([row.best_ask for row in current], dtype=np.float64),
        current_bid_quantity=np.asarray(
            [row.best_bid_quantity for row in current], dtype=np.float64
        ),
        current_ask_quantity=np.asarray(
            [row.best_ask_quantity for row in current], dtype=np.float64
        ),
        future_bid_quantity=np.asarray(
            [row.best_bid_quantity for row in future], dtype=np.float64
        ),
        future_ask_quantity=np.asarray(
            [row.best_ask_quantity for row in future], dtype=np.float64
        ),
        future_bid=np.asarray([row.best_bid for row in future], dtype=np.float64),
        future_ask=np.asarray([row.best_ask for row in future], dtype=np.float64),
        imbalance=np.asarray([row.imbalance_1 for row in current], dtype=np.float64),
        microprice_edge_bps=np.asarray(
            [row.microprice_edge_bps for row in current], dtype=np.float64
        ),
        horizon=horizon,
    )
