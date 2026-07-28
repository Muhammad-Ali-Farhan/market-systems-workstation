
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

FEATURE_NAMES = (
    "spread_bps",
    "imbalance",
    "microprice_edge_bps",
    "log_quantity_ratio",
    "mid_return_1",
    "mid_return_2",
    "mid_return_5",
    "mid_return_10",
    "mid_return_25",
    "imbalance_change_1",
    "imbalance_change_5",
    "spread_change_1",
    "log_interarrival_us",
    "event_rate_10",
    "realized_volatility_20",
    "log_total_quantity_change_1",
)
MODEL_SCHEMA_VERSION = 3
FEATURE_SCHEMA_VERSION = 2
LEGACY_FEATURE_NAME_HASH = hashlib.sha256(
    "\n".join(FEATURE_NAMES).encode("utf-8")
).hexdigest()
FEATURE_SCHEMA_DEFINITION = "\n".join(
    (
        f"feature_schema_version={FEATURE_SCHEMA_VERSION}",
        "source=top_of_book",
        "price_returns=log_bps",
        "quantity_scale=artifact_volume_scale",
        "rolling_volatility=population_std_latest_20_one_event_returns",
        "event_rate=10_events_over_source_timestamp_delta",
        "history_events=25",
        *FEATURE_NAMES,
    )
)
FEATURE_SCHEMA_HASH = hashlib.sha256(
    FEATURE_SCHEMA_DEFINITION.encode("utf-8")
).hexdigest()
MAX_HISTORY = 25
VOLATILITY_WINDOW = 20
EPSILON = 1e-12
DEFAULT_MAX_GAP_NS = 5_000_000_000
UINT32_MAX = int(np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class FeatureSet:
    X: np.ndarray
    y: np.ndarray
    timestamps_ns: np.ndarray
    session_id: np.ndarray
    event_index: np.ndarray
    current_mid: np.ndarray
    current_bid: np.ndarray
    current_ask: np.ndarray
    future_bid: np.ndarray
    future_ask: np.ndarray
    current_bid_quantity: np.ndarray
    current_ask_quantity: np.ndarray
    future_bid_quantity: np.ndarray
    future_ask_quantity: np.ndarray
    imbalance: np.ndarray
    microprice_edge_bps: np.ndarray
    horizon: int

    @property
    def size(self) -> int:
        return int(self.y.size)


@dataclass(frozen=True)
class AlphaModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    horizon: int
    signal_threshold_bps: float
    ridge_alpha: float
    fee_bps_per_side: float
    provenance: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = MODEL_SCHEMA_VERSION

    @property
    def threshold(self) -> float:
        return self.signal_threshold_bps

    @property
    def fee_bps(self) -> float:
        return self.fee_bps_per_side

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("Feature matrix must be two-dimensional.")
        if matrix.shape[1] != len(self.feature_names):
            raise ValueError("Feature count does not match the trained model.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Feature matrix contains non-finite values.")
        standardized = np.clip((matrix - self.mean) / self.scale, -20.0, 20.0)
        return self.intercept + standardized @ self.coefficients

    def predict_one(self, features: np.ndarray) -> float:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (len(self.feature_names),):
            raise ValueError("Single feature vector has the wrong shape.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Feature vector contains non-finite values.")
        standardized = np.clip((vector - self.mean) / self.scale, -20.0, 20.0)
        return float(self.intercept + standardized @ self.coefficients)

    def predict(self, features: np.ndarray) -> np.ndarray | float:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            return self.predict_one(values)
        return self.predict_matrix(values)

    def save(
        self,
        file_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        self._validate()
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite model artifact: {path}")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp.npz",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(
                temporary,
                schema_version=np.asarray(MODEL_SCHEMA_VERSION, dtype=np.int64),
                feature_schema_version=np.asarray(
                    FEATURE_SCHEMA_VERSION, dtype=np.int64
                ),
                feature_schema_hash=np.asarray(FEATURE_SCHEMA_HASH, dtype=np.str_),
                feature_names=np.asarray(self.feature_names, dtype=np.str_),
                mean=np.asarray(self.mean, dtype=np.float64),
                scale=np.asarray(self.scale, dtype=np.float64),
                coefficients=np.asarray(self.coefficients, dtype=np.float64),
                intercept=np.asarray(self.intercept, dtype=np.float64),
                horizon=np.asarray(self.horizon, dtype=np.int64),
                signal_threshold_bps=np.asarray(
                    self.signal_threshold_bps, dtype=np.float64
                ),
                ridge_alpha=np.asarray(self.ridge_alpha, dtype=np.float64),
                fee_bps_per_side=np.asarray(
                    self.fee_bps_per_side, dtype=np.float64
                ),
                provenance_json=np.asarray(
                    json.dumps(dict(self.provenance), sort_keys=True),
                    dtype=np.str_,
                ),
            )
            if path.exists() and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite model artifact: {path}"
                )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def load(file_path: str | Path) -> "AlphaModel":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact does not exist: {path}")

        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            names = tuple(str(name) for name in archive["feature_names"].tolist())

            # Current canonical schema.
            if "signal_threshold_bps" in keys:
                schema_version = (
                    int(np.asarray(archive["schema_version"]).reshape(-1)[0])
                    if "schema_version" in keys
                    else 1
                )
                if schema_version > MODEL_SCHEMA_VERSION or schema_version <= 0:
                    raise RuntimeError(
                        f"Unsupported model artifact schema version: {schema_version}"
                    )
                if "feature_schema_hash" in keys:
                    artifact_hash = str(np.asarray(archive["feature_schema_hash"]).item())
                    compatible_hashes = {FEATURE_SCHEMA_HASH}
                    if schema_version <= 2:
                        compatible_hashes.add(LEGACY_FEATURE_NAME_HASH)
                    if artifact_hash not in compatible_hashes:
                        raise RuntimeError(
                            "Model feature-schema hash does not match this code."
                        )
                if "feature_schema_version" in keys:
                    artifact_feature_version = int(
                        np.asarray(archive["feature_schema_version"]).reshape(-1)[0]
                    )
                    if artifact_feature_version > FEATURE_SCHEMA_VERSION:
                        raise RuntimeError(
                            "Model requires a newer feature implementation."
                        )
                provenance: Mapping[str, object] = {}
                if "provenance_json" in keys:
                    raw = str(np.asarray(archive["provenance_json"]).item())
                    try:
                        loaded = json.loads(raw)
                    except json.JSONDecodeError as exception:
                        raise RuntimeError(
                            "Model provenance metadata is not valid JSON."
                        ) from exception
                    provenance = loaded if isinstance(loaded, dict) else {}
                model = AlphaModel(
                    feature_names=names,
                    mean=np.asarray(archive["mean"], dtype=np.float64),
                    scale=np.asarray(archive["scale"], dtype=np.float64),
                    coefficients=np.asarray(archive["coefficients"], dtype=np.float64),
                    intercept=float(np.asarray(archive["intercept"]).reshape(-1)[0]),
                    horizon=int(np.asarray(archive["horizon"]).reshape(-1)[0]),
                    signal_threshold_bps=float(
                        np.asarray(archive["signal_threshold_bps"]).reshape(-1)[0]
                    ),
                    ridge_alpha=float(np.asarray(archive["ridge_alpha"]).reshape(-1)[0]),
                    fee_bps_per_side=float(
                        np.asarray(archive["fee_bps_per_side"]).reshape(-1)[0]
                    ),
                    provenance=provenance,
                    schema_version=schema_version,
                )
            # Legacy Tk artifact migration.
            elif {"threshold", "selected_alpha", "fee_bps"}.issubset(keys):
                model = AlphaModel(
                    feature_names=names,
                    mean=np.asarray(archive["mean"], dtype=np.float64),
                    scale=np.asarray(archive["scale"], dtype=np.float64),
                    coefficients=np.asarray(archive["coefficients"], dtype=np.float64),
                    intercept=float(np.asarray(archive["intercept"]).reshape(-1)[0]),
                    horizon=int(np.asarray(archive["horizon"]).reshape(-1)[0]),
                    signal_threshold_bps=float(
                        np.asarray(archive["threshold"]).reshape(-1)[0]
                    ),
                    ridge_alpha=float(
                        np.asarray(archive["selected_alpha"]).reshape(-1)[0]
                    ),
                    fee_bps_per_side=float(
                        np.asarray(archive["fee_bps"]).reshape(-1)[0]
                    ),
                    provenance={"migrated_from": "legacy_tk_artifact"},
                    schema_version=1,
                )
            else:
                raise RuntimeError(
                    "Unrecognized model artifact schema. Retrain with the current code."
                )

        model._validate()
        return model

    def _validate(self) -> None:
        if self.feature_names != FEATURE_NAMES:
            raise RuntimeError("Model feature definition does not match this code.")
        expected = (len(FEATURE_NAMES),)
        if self.mean.shape != expected:
            raise RuntimeError("Model mean vector has the wrong shape.")
        if self.scale.shape != expected:
            raise RuntimeError("Model scale vector has the wrong shape.")
        if self.coefficients.shape != expected:
            raise RuntimeError("Model coefficient vector has the wrong shape.")
        if not np.all(np.isfinite(self.mean)):
            raise RuntimeError("Model mean vector contains non-finite values.")
        if not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0.0):
            raise RuntimeError("Model contains an invalid feature scale.")
        if not np.all(np.isfinite(self.coefficients)):
            raise RuntimeError("Model coefficients contain non-finite values.")
        scalar_values = (
            self.intercept,
            self.signal_threshold_bps,
            self.ridge_alpha,
            self.fee_bps_per_side,
        )
        if not all(np.isfinite(value) for value in scalar_values):
            raise RuntimeError("Model metadata contains non-finite values.")
        if (
            self.horizon <= 0
            or self.signal_threshold_bps < 0.0
            or self.ridge_alpha < 0.0
            or self.fee_bps_per_side < 0.0
            or self.schema_version <= 0
            or self.schema_version > MODEL_SCHEMA_VERSION
        ):
            raise RuntimeError("Model horizon, threshold, regularization, fee, or schema is invalid.")


def _lagged_log_return(mid_price: np.ndarray, lag: int) -> np.ndarray:
    output = np.full(mid_price.size, np.nan, dtype=np.float64)
    output[lag:] = np.log(mid_price[lag:] / mid_price[:-lag]) * 10_000.0
    return output


def _lagged_difference(values: np.ndarray, lag: int) -> np.ndarray:
    output = np.full(values.size, np.nan, dtype=np.float64)
    output[lag:] = values[lag:] - values[:-lag]
    return output


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    output = np.full(values.size, np.nan, dtype=np.float64)
    if values.size <= window:
        return output
    usable = values[1:]
    cumulative = np.concatenate(([0.0], np.cumsum(usable, dtype=np.float64)))
    cumulative_square = np.concatenate(
        ([0.0], np.cumsum(usable * usable, dtype=np.float64))
    )
    rolling_sum = cumulative[window:] - cumulative[:-window]
    rolling_square_sum = cumulative_square[window:] - cumulative_square[:-window]
    mean = rolling_sum / window
    variance = rolling_square_sum / window - mean * mean
    output[window:] = np.sqrt(np.maximum(variance, 0.0))
    return output


def build_feature_set(
    records: np.ndarray,
    *,
    volume_scale: float,
    horizon: int,
    session_id: int,
) -> FeatureSet:
    if horizon <= 0:
        raise ValueError("Prediction horizon must be positive.")
    if not math.isfinite(volume_scale) or volume_scale <= 0.0:
        raise ValueError("Volume scale must be finite and positive.")

    record_count = int(records.size)
    minimum_required = MAX_HISTORY + horizon + 2
    if record_count < minimum_required:
        raise ValueError(
            f"Recording segment needs at least {minimum_required:,} records; "
            f"received {record_count:,}."
        )

    timestamps_ns = np.asarray(records["timestamp_ns"], dtype=np.uint64)
    bid = np.asarray(records["best_bid"], dtype=np.float64)
    ask = np.asarray(records["best_ask"], dtype=np.float64)
    bid_quantity = np.asarray(records["bid_volume"], dtype=np.float64) / volume_scale
    ask_quantity = np.asarray(records["ask_volume"], dtype=np.float64) / volume_scale

    if np.any(timestamps_ns == 0) or np.any(timestamps_ns[1:] < timestamps_ns[:-1]):
        raise ValueError("Recording segment timestamps must be positive and monotonic.")
    if (
        not np.all(np.isfinite(bid))
        or not np.all(np.isfinite(ask))
        or np.any(bid <= 0.0)
        or np.any(ask <= 0.0)
        or np.any(bid > ask)
    ):
        raise ValueError("Recording segment contains invalid top-of-book states.")

    mid = (bid + ask) / 2.0
    spread_bps = (ask - bid) / mid * 10_000.0
    total_quantity = bid_quantity + ask_quantity
    imbalance = np.divide(
        bid_quantity - ask_quantity,
        total_quantity,
        out=np.zeros_like(total_quantity),
        where=total_quantity > 0.0,
    )
    microprice = np.divide(
        ask * bid_quantity + bid * ask_quantity,
        total_quantity,
        out=mid.copy(),
        where=total_quantity > 0.0,
    )
    microprice_edge_bps = (microprice - mid) / mid * 10_000.0
    log_quantity_ratio = np.log(
        (bid_quantity + EPSILON) / (ask_quantity + EPSILON)
    )

    mid_return_1 = _lagged_log_return(mid, 1)
    feature_columns = (
        spread_bps,
        imbalance,
        microprice_edge_bps,
        log_quantity_ratio,
        mid_return_1,
        _lagged_log_return(mid, 2),
        _lagged_log_return(mid, 5),
        _lagged_log_return(mid, 10),
        _lagged_log_return(mid, 25),
        _lagged_difference(imbalance, 1),
        _lagged_difference(imbalance, 5),
        _lagged_difference(spread_bps, 1),
    )

    log_interarrival_us = np.full(record_count, np.nan, dtype=np.float64)
    timestamp_delta_ns = np.diff(timestamps_ns.astype(np.float64))
    log_interarrival_us[1:] = np.log1p(np.maximum(timestamp_delta_ns, 0.0) / 1_000.0)

    event_rate_10 = np.full(record_count, np.nan, dtype=np.float64)
    ten_event_delta_seconds = (
        timestamps_ns[10:].astype(np.float64)
        - timestamps_ns[:-10].astype(np.float64)
    ) / 1_000_000_000.0
    event_rate_10[10:] = np.divide(
        10.0,
        ten_event_delta_seconds,
        out=np.zeros_like(ten_event_delta_seconds),
        where=ten_event_delta_seconds > 0.0,
    )

    log_total_quantity_change_1 = np.full(record_count, np.nan, dtype=np.float64)
    log_total_quantity_change_1[1:] = np.log(
        (total_quantity[1:] + EPSILON) / (total_quantity[:-1] + EPSILON)
    )

    all_columns = feature_columns + (
        log_interarrival_us,
        event_rate_10,
        _rolling_std(mid_return_1, VOLATILITY_WINDOW),
        log_total_quantity_change_1,
    )

    start = MAX_HISTORY
    stop = record_count - horizon
    X = np.column_stack(tuple(column[start:stop] for column in all_columns))
    current_mid = mid[start:stop]
    current_bid = bid[start:stop]
    current_ask = ask[start:stop]
    future_bid = bid[start + horizon : stop + horizon]
    future_ask = ask[start + horizon : stop + horizon]
    current_bid_quantity = bid_quantity[start:stop]
    current_ask_quantity = ask_quantity[start:stop]
    future_bid_quantity = bid_quantity[start + horizon : stop + horizon]
    future_ask_quantity = ask_quantity[start + horizon : stop + horizon]
    future_mid = mid[start + horizon : stop + horizon]
    y = np.log(future_mid / current_mid) * 10_000.0
    event_index = np.arange(start, stop, dtype=np.int64)

    valid = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(y)
        & (current_bid > 0.0)
        & (current_ask > 0.0)
        & (future_bid > 0.0)
        & (future_ask > 0.0)
    )
    valid_size = int(np.count_nonzero(valid))
    if valid_size == 0:
        raise ValueError("Recording segment produced no valid feature rows.")

    return FeatureSet(
        X=np.ascontiguousarray(X[valid], dtype=np.float64),
        y=np.ascontiguousarray(y[valid], dtype=np.float64),
        timestamps_ns=np.ascontiguousarray(timestamps_ns[start:stop][valid], dtype=np.uint64),
        session_id=np.full(valid_size, session_id, dtype=np.int32),
        event_index=np.ascontiguousarray(event_index[valid], dtype=np.int64),
        current_mid=np.ascontiguousarray(current_mid[valid], dtype=np.float64),
        current_bid=np.ascontiguousarray(current_bid[valid], dtype=np.float64),
        current_ask=np.ascontiguousarray(current_ask[valid], dtype=np.float64),
        future_bid=np.ascontiguousarray(future_bid[valid], dtype=np.float64),
        future_ask=np.ascontiguousarray(future_ask[valid], dtype=np.float64),
        current_bid_quantity=np.ascontiguousarray(
            current_bid_quantity[valid], dtype=np.float64
        ),
        current_ask_quantity=np.ascontiguousarray(
            current_ask_quantity[valid], dtype=np.float64
        ),
        future_bid_quantity=np.ascontiguousarray(
            future_bid_quantity[valid], dtype=np.float64
        ),
        future_ask_quantity=np.ascontiguousarray(
            future_ask_quantity[valid], dtype=np.float64
        ),
        imbalance=np.ascontiguousarray(imbalance[start:stop][valid], dtype=np.float64),
        microprice_edge_bps=np.ascontiguousarray(
            microprice_edge_bps[start:stop][valid], dtype=np.float64
        ),
        horizon=horizon,
    )


def concatenate_feature_sets(feature_sets: Iterable[FeatureSet]) -> FeatureSet:
    sets = tuple(feature_sets)
    if not sets:
        raise ValueError("No feature sets were supplied.")
    horizon = sets[0].horizon
    if any(item.horizon != horizon for item in sets):
        raise ValueError("All feature sets must use the same horizon.")

    def join(attribute: str) -> np.ndarray:
        return np.concatenate(tuple(getattr(item, attribute) for item in sets))

    return FeatureSet(
        X=join("X"),
        y=join("y"),
        timestamps_ns=join("timestamps_ns"),
        session_id=join("session_id"),
        event_index=join("event_index"),
        current_mid=join("current_mid"),
        current_bid=join("current_bid"),
        current_ask=join("current_ask"),
        future_bid=join("future_bid"),
        future_ask=join("future_ask"),
        current_bid_quantity=join("current_bid_quantity"),
        current_ask_quantity=join("current_ask_quantity"),
        future_bid_quantity=join("future_bid_quantity"),
        future_ask_quantity=join("future_ask_quantity"),
        imbalance=join("imbalance"),
        microprice_edge_bps=join("microprice_edge_bps"),
        horizon=horizon,
    )


@dataclass(frozen=True)
class BookTick:
    timestamp_ns: int
    best_bid: float
    best_ask: float
    bid_quantity: float
    ask_quantity: float


class OnlineFeatureBuilder:
    def __init__(
        self,
        volume_scale: float = 1_000_000.0,
        *,
        max_gap_ns: int = DEFAULT_MAX_GAP_NS,
    ) -> None:
        if not math.isfinite(volume_scale) or volume_scale <= 0.0:
            raise ValueError("Volume scale must be finite and positive.")
        if max_gap_ns <= 0:
            raise ValueError("max_gap_ns must be positive.")
        self._volume_scale = float(volume_scale)
        self._max_gap_ns = int(max_gap_ns)
        self._history: deque[BookTick] = deque(maxlen=MAX_HISTORY + 1)
        self._reset_count = 0

    def reset(self) -> None:
        if self._history:
            self._reset_count += 1
        self._history.clear()

    @staticmethod
    def _mid(tick: BookTick) -> float:
        return (tick.best_bid + tick.best_ask) / 2.0

    @staticmethod
    def _spread_bps(tick: BookTick, mid: float) -> float:
        return (tick.best_ask - tick.best_bid) / mid * 10_000.0

    @staticmethod
    def _total_quantity(tick: BookTick) -> float:
        return tick.bid_quantity + tick.ask_quantity

    @classmethod
    def _imbalance(cls, tick: BookTick) -> float:
        total = cls._total_quantity(tick)
        return (tick.bid_quantity - tick.ask_quantity) / total if total > 0.0 else 0.0

    def update(
        self,
        timestamp_ns: int,
        best_bid: float,
        best_ask: float,
        bid_volume: int,
        ask_volume: int,
    ) -> np.ndarray | None:
        timestamp = int(timestamp_ns)
        bid = float(best_bid)
        ask = float(best_ask)
        bid_volume_value = int(bid_volume)
        ask_volume_value = int(ask_volume)
        if (
            timestamp <= 0
            or not math.isfinite(bid)
            or not math.isfinite(ask)
            or bid <= 0.0
            or ask <= 0.0
            or bid > ask
            or bid_volume_value < 0
            or ask_volume_value < 0
            or bid_volume_value > UINT32_MAX
            or ask_volume_value > UINT32_MAX
        ):
            return None

        tick = BookTick(
            timestamp_ns=timestamp,
            best_bid=bid,
            best_ask=ask,
            bid_quantity=bid_volume_value / self._volume_scale,
            ask_quantity=ask_volume_value / self._volume_scale,
        )

        if self._history:
            previous_timestamp = self._history[-1].timestamp_ns
            if (
                tick.timestamp_ns < previous_timestamp
                or tick.timestamp_ns - previous_timestamp > self._max_gap_ns
            ):
                self.reset()

        self._history.append(tick)
        if len(self._history) < MAX_HISTORY + 1:
            return None

        history = tuple(self._history)
        current = history[-1]
        current_mid = self._mid(current)
        current_spread = self._spread_bps(current, current_mid)
        current_total = self._total_quantity(current)
        current_imbalance = self._imbalance(current)
        previous = history[-2]
        previous_mid = self._mid(previous)
        previous_spread = self._spread_bps(previous, previous_mid)
        previous_total = self._total_quantity(previous)
        previous_imbalance = self._imbalance(previous)
        imbalance_lag_5 = self._imbalance(history[-6])

        microprice = (
            (
                current.best_ask * current.bid_quantity
                + current.best_bid * current.ask_quantity
            )
            / current_total
            if current_total > 0.0
            else current_mid
        )

        def lag_return(lag: int) -> float:
            return math.log(current_mid / self._mid(history[-1 - lag])) * 10_000.0

        timestamp_delta_ns = current.timestamp_ns - previous.timestamp_ns
        ten_event_delta_ns = current.timestamp_ns - history[-11].timestamp_ns

        # Offline realized_volatility_20 at the current event uses the latest
        # twenty one-event log returns, which require the latest twenty-one mids.
        volatility_returns: list[float] = []
        for index in range(len(history) - VOLATILITY_WINDOW, len(history)):
            current_window_mid = self._mid(history[index])
            previous_window_mid = self._mid(history[index - 1])
            volatility_returns.append(
                math.log(current_window_mid / previous_window_mid) * 10_000.0
            )
        volatility_mean = sum(volatility_returns) / VOLATILITY_WINDOW
        volatility_variance = max(
            0.0,
            sum(value * value for value in volatility_returns) / VOLATILITY_WINDOW
            - volatility_mean * volatility_mean,
        )

        features = np.asarray(
            [
                current_spread,
                current_imbalance,
                (microprice - current_mid) / current_mid * 10_000.0,
                math.log(
                    (current.bid_quantity + EPSILON)
                    / (current.ask_quantity + EPSILON)
                ),
                lag_return(1),
                lag_return(2),
                lag_return(5),
                lag_return(10),
                lag_return(25),
                current_imbalance - previous_imbalance,
                current_imbalance - imbalance_lag_5,
                current_spread - previous_spread,
                math.log1p(max(timestamp_delta_ns, 0) / 1_000.0),
                (
                    10.0 / (ten_event_delta_ns / 1_000_000_000.0)
                    if ten_event_delta_ns > 0
                    else 0.0
                ),
                math.sqrt(volatility_variance),
                math.log((current_total + EPSILON) / (previous_total + EPSILON)),
            ],
            dtype=np.float64,
        )
        return features if all(math.isfinite(float(value)) for value in features) else None

    @property
    def latest_tick(self) -> BookTick | None:
        return self._history[-1] if self._history else None

    @property
    def reset_count(self) -> int:
        return self._reset_count


