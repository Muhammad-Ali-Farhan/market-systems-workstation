
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import tempfile
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from microstructure import (
    AlphaModel,
    DEFAULT_MAX_GAP_NS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    FeatureSet,
    build_feature_set,
    concatenate_feature_sets,
)
from research_diagnostics import (
    coefficient_bootstrap_stability,
    cost_stress_curve,
    directional_classification_metrics,
    circular_shift_permutation_test,
    feature_drift_report,
    newey_west_mean_t_statistic,
    prediction_quantile_table,
    render_research_card,
    session_bootstrap_summary,
    session_regression_table,
    spearman_correlation,
)
from qbin import (
    RecordingMetadata,
    contiguous_slices,
    open_records,
    open_update_ids,
    read_metadata,
    sha256_file,
    validate_records,
    validate_update_ids,
)

RIDGE_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
THRESHOLD_QUANTILES = (0.00, 0.25, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95)
REPORT_SCHEMA_VERSION = 4
MINIMUM_PARTITION_ROWS = 100


@dataclass(frozen=True)
class RegressionMetrics:
    samples: int
    mse: float
    mae: float
    pearson_ic: float
    spearman_rank_ic: float
    direction_accuracy: float
    majority_direction_accuracy: float
    direction_accuracy_lift_vs_majority: float
    balanced_direction_accuracy: float
    actionable_coverage: float
    actionable_direction_accuracy: float
    actionable_matthews_correlation: float
    target_zero_fraction: float
    positive_target_fraction: float
    positive_prediction_fraction: float
    nonzero_target_samples: int
    r_squared: float


@dataclass(frozen=True)
class StrategyMetrics:
    trades: int
    win_rate: float
    mean_pnl_bps: float
    median_pnl_bps: float
    total_pnl_bps: float
    gross_mean_pnl_bps: float
    gross_total_pnl_bps: float
    round_trip_cost_bps: float
    max_drawdown_bps: float
    naive_pnl_t_statistic: float
    newey_west_pnl_t_statistic: float
    trade_sharpe: float
    profit_factor: float
    payoff_ratio: float
    expected_shortfall_5pct_bps: float
    breakeven_additional_cost_bps_per_side: float
    session_bootstrap_mean_ci_low_bps: float
    session_bootstrap_mean_ci_high_bps: float
    session_bootstrap_probability_mean_non_positive: float
    sessions_traded: int
    fill_rejections: int


@dataclass(frozen=True)
class ExecutionAssumptions:
    fee_bps_per_side: float
    slippage_bps_per_side: float = 0.0
    trade_size_base: float = 0.0
    max_displayed_participation: float = 1.0

    def validate(self) -> None:
        values = (
            self.fee_bps_per_side,
            self.slippage_bps_per_side,
            self.trade_size_base,
            self.max_displayed_participation,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Execution assumptions must be finite.")
        if self.fee_bps_per_side < 0.0 or self.slippage_bps_per_side < 0.0:
            raise ValueError("Fees and slippage must be non-negative.")
        if self.trade_size_base < 0.0:
            raise ValueError("Trade size must be non-negative.")
        if not 0.0 < self.max_displayed_participation <= 1.0:
            raise ValueError("Displayed participation must be in (0, 1].")

    @property
    def round_trip_cost_bps(self) -> float:
        return 2.0 * (self.fee_bps_per_side + self.slippage_bps_per_side)


@dataclass(frozen=True)
class TradeResults:
    net_pnl_bps: np.ndarray
    gross_pnl_bps: np.ndarray
    selected_rows: np.ndarray
    sessions: np.ndarray
    sides: np.ndarray
    displayed_capacity_base: np.ndarray
    fill_rejections: int


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    mode: str
    train_sessions: tuple[int, ...]
    validation_sessions: tuple[int, ...]
    test_sessions: tuple[int, ...]
    purge_rows: int


@dataclass(frozen=True)
class PreparedRecording:
    metadata: RecordingMetadata
    sha256: str
    segment_session_ids: tuple[int, ...]
    segment_sha256: tuple[str, ...]
    feature_rows: int


ProgressCallback = Callable[[int, str], None]


def _progress(callback: ProgressCallback | None, value: int, message: str) -> None:
    if callback is not None:
        callback(int(value), message)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = result.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def _test_set_fingerprint(
    data: FeatureSet,
    test_indices: np.ndarray,
    *,
    horizon: int,
) -> str:
    """Fingerprint the exact examples supplied to the fitted model.

    This fingerprint is intentionally feature-implementation and horizon
    specific. It complements ``_test_period_fingerprint``, which identifies
    the underlying raw held-out market segments independent of feature code.
    """
    digest = hashlib.sha256()
    digest.update(FEATURE_SCHEMA_HASH.encode("ascii"))
    digest.update(str(horizon).encode("ascii"))
    digest.update(str(int(test_indices.size)).encode("ascii"))
    for values in (
        data.timestamps_ns[test_indices],
        data.event_index[test_indices],
        data.current_bid[test_indices],
        data.current_ask[test_indices],
        data.future_bid[test_indices],
        data.future_ask[test_indices],
        data.y[test_indices],
    ):
        array = np.ascontiguousarray(values)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _test_period_fingerprint(
    prepared: Sequence[PreparedRecording],
    test_sessions: Sequence[int],
) -> str:
    """Fingerprint raw contiguous market segments assigned to test.

    The result does not depend on arbitrary session numbering, the model
    horizon, or the current feature implementation. Adding earlier training
    sessions therefore cannot disguise reuse of the same held-out market
    period.
    """
    selected = set(int(value) for value in test_sessions)
    segment_hashes: list[str] = []
    for item in prepared:
        for session_id, segment_hash in zip(
            item.segment_session_ids,
            item.segment_sha256,
            strict=True,
        ):
            if session_id in selected:
                segment_hashes.append(segment_hash)
    if not segment_hashes:
        raise RuntimeError("The chronological split contains no raw test segments.")
    payload = json.dumps(segment_hashes, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _prior_holdout_reports(
    report_directory: Path,
    *,
    test_period_fingerprint: str,
    test_set_fingerprint: str,
) -> list[Path]:
    matches: list[Path] = []
    if not report_directory.exists():
        return matches
    for path in report_directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            continue
        prior_period = provenance.get("test_period_fingerprint")
        prior_exact = provenance.get("test_set_fingerprint")
        if (
            prior_period == test_period_fingerprint
            or (prior_period is None and prior_exact == test_set_fingerprint)
        ):
            matches.append(path)
    return sorted(matches, key=lambda path: str(path.resolve()))


def prepare_feature_data(
    recording_paths: Sequence[str | Path],
    *,
    horizon: int,
    max_gap_ns: int = DEFAULT_MAX_GAP_NS,
    allow_incomplete: bool = False,
    progress: ProgressCallback | None = None,
) -> tuple[FeatureSet, tuple[PreparedRecording, ...]]:
    if horizon <= 0:
        raise ValueError("Prediction horizon must be positive.")
    if max_gap_ns <= 0:
        raise ValueError("Maximum gap must be positive.")
    if not recording_paths:
        raise ValueError("At least one recording is required.")

    metadata_items = [read_metadata(path) for path in recording_paths]
    metadata_items.sort(key=lambda item: (item.created_unix_ns, str(item.path.resolve())))

    hashes: dict[str, Path] = {}
    feature_sets: list[FeatureSet] = []
    prepared: list[PreparedRecording] = []
    next_session_id = 0

    for recording_index, metadata in enumerate(metadata_items):
        if metadata.data_complete is False and not allow_incomplete:
            raise RuntimeError(
                f"Recording is marked incomplete and cannot be used for research: {metadata.path}"
            )

        digest = sha256_file(metadata.path)
        previous_path = hashes.get(digest)
        if previous_path is not None:
            raise RuntimeError(
                "Duplicate recording content detected:\n"
                f"  {previous_path}\n"
                f"  {metadata.path}"
            )
        hashes[digest] = metadata.path

        records = open_records(metadata.path, metadata)
        validate_records(records, context=str(metadata.path))
        validate_update_ids(
            open_update_ids(metadata),
            metadata,
            context=str(metadata.path),
        )
        segment_ids: list[int] = []
        segment_hashes: list[str] = []
        rows_before = sum(item.size for item in feature_sets)

        for segment in contiguous_slices(records, metadata, max_gap_ns=max_gap_ns):
            segment_records = records[segment]
            try:
                features = build_feature_set(
                    segment_records,
                    volume_scale=float(metadata.volume_scale),
                    horizon=horizon,
                    session_id=next_session_id,
                )
            except ValueError:
                continue
            feature_sets.append(features)
            segment_ids.append(next_session_id)
            segment_hashes.append(
                hashlib.sha256(
                    np.ascontiguousarray(segment_records).tobytes(order="C")
                ).hexdigest()
            )
            next_session_id += 1

        feature_rows = sum(item.size for item in feature_sets) - rows_before
        if feature_rows == 0:
            raise RuntimeError(
                f"Recording produced no usable contiguous feature segment: {metadata.path}"
            )

        prepared.append(
            PreparedRecording(
                metadata=metadata,
                sha256=digest,
                segment_session_ids=tuple(segment_ids),
                segment_sha256=tuple(segment_hashes),
                feature_rows=feature_rows,
            )
        )
        _progress(
            progress,
            5 + int(25 * (recording_index + 1) / len(metadata_items)),
            f"Validated {metadata.path.name}",
        )

    data = concatenate_feature_sets(feature_sets)
    if data.size < 500:
        raise ValueError(
            f"At least 500 feature rows are required; received {data.size:,}."
        )
    return data, tuple(prepared)


def _session_row_ranges(data: FeatureSet) -> list[tuple[int, int, int]]:
    sessions = np.asarray(data.session_id, dtype=np.int32)
    if sessions.size == 0:
        return []
    changes = np.flatnonzero(sessions[1:] != sessions[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [sessions.size]))
    ranges: list[tuple[int, int, int]] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        ranges.append((int(sessions[start]), int(start), int(stop)))
    return ranges


def make_chronological_split(data: FeatureSet, purge_size: int) -> SplitIndices:
    if purge_size < 0:
        raise ValueError("Purge size cannot be negative.")
    ranges = _session_row_ranges(data)

    # Prefer whole-session partitions. This prevents later sessions from being
    # used to train a model evaluated on an earlier session.
    if len(ranges) >= 3:
        cumulative = np.asarray([stop for _session, _start, stop in ranges], dtype=np.int64)
        target_train = data.size * 0.60
        target_validation = data.size * 0.80
        candidates: list[tuple[float, int, int]] = []
        for train_boundary in range(1, len(ranges) - 1):
            for validation_boundary in range(train_boundary + 1, len(ranges)):
                train_stop = int(cumulative[train_boundary - 1])
                validation_stop = int(cumulative[validation_boundary - 1])
                sizes = (train_stop, validation_stop - train_stop, data.size - validation_stop)
                if min(sizes) < MINIMUM_PARTITION_ROWS:
                    continue
                score = abs(train_stop - target_train) + abs(validation_stop - target_validation)
                candidates.append((score, train_boundary, validation_boundary))

        if candidates:
            _score, train_boundary, validation_boundary = min(candidates)
            train_stop = ranges[train_boundary - 1][2]
            validation_stop = ranges[validation_boundary - 1][2]
            train = np.arange(0, train_stop, dtype=np.int64)
            validation = np.arange(train_stop, validation_stop, dtype=np.int64)
            test = np.arange(validation_stop, data.size, dtype=np.int64)
            return SplitIndices(
                train=train,
                validation=validation,
                test=test,
                mode="whole_session",
                train_sessions=tuple(item[0] for item in ranges[:train_boundary]),
                validation_sessions=tuple(
                    item[0] for item in ranges[train_boundary:validation_boundary]
                ),
                test_sessions=tuple(item[0] for item in ranges[validation_boundary:]),
                purge_rows=0,
            )

    # Fallback for one or two recordings: a global chronological event split
    # with purged boundaries. Feature targets never cross a recording segment.
    train_end = int(data.size * 0.60)
    validation_end = int(data.size * 0.80)
    validation_start = train_end + purge_size
    test_start = validation_end + purge_size
    train = np.arange(0, train_end, dtype=np.int64)
    validation = np.arange(validation_start, validation_end, dtype=np.int64)
    test = np.arange(test_start, data.size, dtype=np.int64)
    if min(train.size, validation.size, test.size) < MINIMUM_PARTITION_ROWS:
        raise ValueError(
            "Data is too short after chronological splitting and purging. "
            "Collect more independent sessions."
        )

    def session_tuple(indices: np.ndarray) -> tuple[int, ...]:
        return tuple(int(value) for value in np.unique(data.session_id[indices]).tolist())

    return SplitIndices(
        train=train,
        validation=validation,
        test=test,
        mode="purged_event_fallback",
        train_sessions=session_tuple(train),
        validation_sessions=session_tuple(validation),
        test_sessions=session_tuple(test),
        purge_rows=purge_size,
    )


def standardize_training_data(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0, dtype=np.float64)
    scale = np.std(X_train, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return np.clip((X_train - mean) / scale, -20.0, 20.0), mean, scale


def standardize_with_parameters(
    X: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.clip((X - mean) / scale, -20.0, 20.0)


def fit_ridge(
    X_standardized: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(X_standardized, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != target.size:
        raise ValueError("Ridge inputs have incompatible shapes.")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("Ridge alpha must be finite and non-negative.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("Ridge inputs contain non-finite values.")

    y_mean = float(np.mean(target))
    centered_y = target - y_mean
    if alpha > 0.0:
        feature_count = matrix.shape[1]
        augmented_X = np.vstack(
            (matrix, math.sqrt(alpha) * np.eye(feature_count, dtype=np.float64))
        )
        augmented_y = np.concatenate(
            (centered_y, np.zeros(feature_count, dtype=np.float64))
        )
        coefficients = np.linalg.lstsq(augmented_X, augmented_y, rcond=None)[0]
    else:
        coefficients = np.linalg.lstsq(matrix, centered_y, rcond=None)[0]
    return np.asarray(coefficients, dtype=np.float64), y_mean


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.size < 2 or right_values.size < 2:
        return float("nan")
    if np.std(left_values) <= 1e-15 or np.std(right_values) <= 1e-15:
        return 0.0
    return float(np.corrcoef(left_values, right_values)[0, 1])


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> RegressionMetrics:
    prediction_values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction_values.size != target_values.size or prediction_values.size == 0:
        raise ValueError("Regression metric inputs must be non-empty and equal length.")
    if not np.all(np.isfinite(prediction_values)) or not np.all(np.isfinite(target_values)):
        raise ValueError("Regression metric inputs contain non-finite values.")
    residual = prediction_values - target_values
    target_variance = float(np.sum((target_values - np.mean(target_values)) ** 2))
    residual_variance = float(np.sum(residual * residual))
    direction = directional_classification_metrics(
        prediction_values, target_values
    )
    return RegressionMetrics(
        samples=int(target_values.size),
        mse=float(np.mean(residual * residual)),
        mae=float(np.mean(np.abs(residual))),
        pearson_ic=pearson_correlation(prediction_values, target_values),
        spearman_rank_ic=spearman_correlation(prediction_values, target_values),
        direction_accuracy=float(direction["direction_accuracy"]),
        majority_direction_accuracy=float(direction["majority_direction_accuracy"]),
        direction_accuracy_lift_vs_majority=float(
            direction["direction_accuracy_lift_vs_majority"]
        ),
        balanced_direction_accuracy=float(direction["balanced_direction_accuracy"]),
        actionable_coverage=float(direction["actionable_coverage"]),
        actionable_direction_accuracy=float(
            direction["actionable_direction_accuracy"]
        ),
        actionable_matthews_correlation=float(
            direction["actionable_matthews_correlation"]
        ),
        target_zero_fraction=float(direction["target_zero_fraction"]),
        positive_target_fraction=float(direction["positive_target_fraction"]),
        positive_prediction_fraction=float(
            direction["positive_prediction_fraction"]
        ),
        nonzero_target_samples=int(direction["nonzero_target_samples"]),
        r_squared=(
            float(1.0 - residual_variance / target_variance)
            if target_variance > 1e-15
            else 0.0
        ),
    )


def strategy_trades(
    prediction: np.ndarray,
    data: FeatureSet,
    indices: np.ndarray,
    *,
    threshold: float,
    execution: ExecutionAssumptions,
) -> TradeResults:
    execution.validate()
    forecast_values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    row_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if forecast_values.size != row_indices.size:
        raise ValueError("Prediction count must match evaluation row count.")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Signal threshold must be finite and non-negative.")
    if np.any(row_indices < 0) or np.any(row_indices >= data.size):
        raise ValueError("Evaluation row index is outside the feature set.")

    net_values: list[float] = []
    gross_values: list[float] = []
    selected_rows: list[int] = []
    selected_sessions: list[int] = []
    selected_sides: list[int] = []
    displayed_capacities: list[float] = []
    fill_rejections = 0
    current_session: int | None = None
    next_eligible_event = -1

    for local_index, row_index in enumerate(row_indices.tolist()):
        session = int(data.session_id[row_index])
        event = int(data.event_index[row_index])
        if session != current_session:
            current_session = session
            next_eligible_event = -1
        if event < next_eligible_event:
            continue

        forecast = float(forecast_values[local_index])
        if not math.isfinite(forecast):
            continue
        if forecast > 0.0 and forecast >= threshold:
            side = 1
            entry_price = float(data.current_ask[row_index])
            exit_price = float(data.future_bid[row_index])
            displayed_capacity = min(
                float(data.current_ask_quantity[row_index]),
                float(data.future_bid_quantity[row_index]),
            ) * execution.max_displayed_participation
            gross = math.log(exit_price / entry_price) * 10_000.0
        elif forecast < 0.0 and forecast <= -threshold:
            side = -1
            entry_price = float(data.current_bid[row_index])
            exit_price = float(data.future_ask[row_index])
            displayed_capacity = min(
                float(data.current_bid_quantity[row_index]),
                float(data.future_ask_quantity[row_index]),
            ) * execution.max_displayed_participation
            gross = math.log(entry_price / exit_price) * 10_000.0
        else:
            continue

        if execution.trade_size_base > 0.0 and displayed_capacity < execution.trade_size_base:
            fill_rejections += 1
            continue

        net = gross - execution.round_trip_cost_bps
        gross_values.append(float(gross))
        net_values.append(float(net))
        selected_rows.append(row_index)
        selected_sessions.append(session)
        selected_sides.append(side)
        displayed_capacities.append(displayed_capacity)
        next_eligible_event = event + data.horizon

    return TradeResults(
        net_pnl_bps=np.asarray(net_values, dtype=np.float64),
        gross_pnl_bps=np.asarray(gross_values, dtype=np.float64),
        selected_rows=np.asarray(selected_rows, dtype=np.int64),
        sessions=np.asarray(selected_sessions, dtype=np.int32),
        sides=np.asarray(selected_sides, dtype=np.int8),
        displayed_capacity_base=np.asarray(displayed_capacities, dtype=np.float64),
        fill_rejections=fill_rejections,
    )


def strategy_pnls(
    prediction: np.ndarray,
    data: FeatureSet,
    indices: np.ndarray,
    *,
    threshold: float,
    fee_bps_per_side: float,
    slippage_bps_per_side: float = 0.0,
    trade_size_base: float = 0.0,
    max_displayed_participation: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trades = strategy_trades(
        prediction,
        data,
        indices,
        threshold=threshold,
        execution=ExecutionAssumptions(
            fee_bps_per_side=fee_bps_per_side,
            slippage_bps_per_side=slippage_bps_per_side,
            trade_size_base=trade_size_base,
            max_displayed_participation=max_displayed_participation,
        ),
    )
    return trades.net_pnl_bps, trades.selected_rows, trades.sessions


def strategy_metrics(
    pnl: np.ndarray,
    sessions: np.ndarray,
    *,
    gross_pnl: np.ndarray | None = None,
    round_trip_cost_bps: float = 0.0,
    fill_rejections: int = 0,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 0,
) -> StrategyMetrics:
    net = np.asarray(pnl, dtype=np.float64).reshape(-1)
    session_values = np.asarray(sessions, dtype=np.int32).reshape(-1)
    gross = net.copy() if gross_pnl is None else np.asarray(gross_pnl, dtype=np.float64).reshape(-1)
    if not (net.size == gross.size == session_values.size):
        raise ValueError("Strategy metric arrays must have the same length.")
    if not math.isfinite(round_trip_cost_bps) or round_trip_cost_bps < 0.0:
        raise ValueError("Round-trip cost must be finite and non-negative.")
    if fill_rejections < 0:
        raise ValueError("Fill rejections cannot be negative.")
    if net.size == 0:
        return StrategyMetrics(
            trades=0,
            win_rate=0.0,
            mean_pnl_bps=0.0,
            median_pnl_bps=0.0,
            total_pnl_bps=0.0,
            gross_mean_pnl_bps=0.0,
            gross_total_pnl_bps=0.0,
            round_trip_cost_bps=round_trip_cost_bps,
            max_drawdown_bps=0.0,
            naive_pnl_t_statistic=0.0,
            newey_west_pnl_t_statistic=0.0,
            trade_sharpe=0.0,
            profit_factor=0.0,
            payoff_ratio=0.0,
            expected_shortfall_5pct_bps=0.0,
            breakeven_additional_cost_bps_per_side=0.0,
            session_bootstrap_mean_ci_low_bps=float("nan"),
            session_bootstrap_mean_ci_high_bps=float("nan"),
            session_bootstrap_probability_mean_non_positive=float("nan"),
            sessions_traded=0,
            fill_rejections=fill_rejections,
        )

    cumulative = np.cumsum(net)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[1:]
    drawdown = cumulative - running_peak
    standard_deviation = float(np.std(net, ddof=1)) if net.size > 1 else 0.0
    mean_pnl = float(np.mean(net))
    naive_t = (
        mean_pnl / standard_deviation * math.sqrt(net.size)
        if standard_deviation > 1e-15
        else 0.0
    )
    bootstrap = session_bootstrap_summary(
        net,
        session_values,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    positive = net[net > 0.0]
    negative = net[net < 0.0]
    profit_factor = (
        float(np.sum(positive) / abs(np.sum(negative)))
        if negative.size and abs(float(np.sum(negative))) > 1e-15
        else (float("inf") if positive.size else 0.0)
    )
    payoff_ratio = (
        float(np.mean(positive) / abs(np.mean(negative)))
        if positive.size and negative.size and abs(float(np.mean(negative))) > 1e-15
        else 0.0
    )
    tail_count = max(1, int(math.ceil(net.size * 0.05)))
    expected_shortfall = float(np.mean(np.sort(net)[:tail_count]))
    gross_mean = float(np.mean(gross))
    return StrategyMetrics(
        trades=int(net.size),
        win_rate=float(np.mean(net > 0.0)),
        mean_pnl_bps=mean_pnl,
        median_pnl_bps=float(np.median(net)),
        total_pnl_bps=float(np.sum(net)),
        gross_mean_pnl_bps=gross_mean,
        gross_total_pnl_bps=float(np.sum(gross)),
        round_trip_cost_bps=round_trip_cost_bps,
        max_drawdown_bps=float(-np.min(drawdown)),
        naive_pnl_t_statistic=float(naive_t),
        newey_west_pnl_t_statistic=newey_west_mean_t_statistic(net),
        trade_sharpe=(mean_pnl / standard_deviation if standard_deviation > 1e-15 else 0.0),
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        expected_shortfall_5pct_bps=expected_shortfall,
        breakeven_additional_cost_bps_per_side=max(0.0, mean_pnl / 2.0),
        session_bootstrap_mean_ci_low_bps=bootstrap.mean_ci_low,
        session_bootstrap_mean_ci_high_bps=bootstrap.mean_ci_high,
        session_bootstrap_probability_mean_non_positive=(
            bootstrap.probability_mean_non_positive
        ),
        sessions_traded=int(np.unique(session_values).size),
        fill_rejections=fill_rejections,
    )


def select_signal_threshold(
    prediction: np.ndarray,
    data: FeatureSet,
    validation_indices: np.ndarray,
    fee_bps_per_side: float,
    *,
    slippage_bps_per_side: float = 0.0,
    trade_size_base: float = 0.0,
    max_displayed_participation: float = 1.0,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 0,
) -> tuple[float, StrategyMetrics]:
    execution = ExecutionAssumptions(
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        trade_size_base=trade_size_base,
        max_displayed_participation=max_displayed_participation,
    )
    execution.validate()
    prediction_values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if prediction_values.size != np.asarray(validation_indices).size:
        raise ValueError("Prediction count must match validation rows.")
    candidates = sorted(
        {
            float(np.quantile(np.abs(prediction_values), quantile))
            for quantile in THRESHOLD_QUANTILES
        }
    )
    minimum_trades = max(10, int(np.asarray(validation_indices).size * 0.005))
    best_threshold = candidates[0]
    best_metrics: StrategyMetrics | None = None

    for threshold in candidates:
        trades = strategy_trades(
            prediction_values,
            data,
            validation_indices,
            threshold=threshold,
            execution=execution,
        )
        metrics = strategy_metrics(
            trades.net_pnl_bps,
            trades.sessions,
            gross_pnl=trades.gross_pnl_bps,
            round_trip_cost_bps=execution.round_trip_cost_bps,
            fill_rejections=trades.fill_rejections,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        if metrics.trades < minimum_trades:
            continue
        score = (
            metrics.mean_pnl_bps,
            metrics.total_pnl_bps,
            -metrics.max_drawdown_bps,
        )
        if best_metrics is None:
            best_threshold, best_metrics = threshold, metrics
        else:
            best_score = (
                best_metrics.mean_pnl_bps,
                best_metrics.total_pnl_bps,
                -best_metrics.max_drawdown_bps,
            )
            if score > best_score:
                best_threshold, best_metrics = threshold, metrics

    if best_metrics is None:
        trades = strategy_trades(
            prediction_values,
            data,
            validation_indices,
            threshold=best_threshold,
            execution=execution,
        )
        best_metrics = strategy_metrics(
            trades.net_pnl_bps,
            trades.sessions,
            gross_pnl=trades.gross_pnl_bps,
            round_trip_cost_bps=execution.round_trip_cost_bps,
            fill_rejections=trades.fill_rejections,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    return best_threshold, best_metrics


def choose_ridge_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[float, np.ndarray, float, list[dict[str, float]]]:
    best: tuple[float, float, np.ndarray, float] | None = None
    search: list[dict[str, float]] = []
    for alpha in RIDGE_GRID:
        coefficients, intercept = fit_ridge(X_train, y_train, alpha)
        prediction = intercept + X_validation @ coefficients
        metrics = regression_metrics(prediction, y_validation)
        search.append(
            {
                "ridge_alpha": float(alpha),
                "validation_mse": metrics.mse,
                "validation_pearson_ic": metrics.pearson_ic,
            }
        )
        candidate = (metrics.mse, alpha, coefficients, intercept)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _mse, alpha, coefficients, intercept = best
    return float(alpha), coefficients, float(intercept), search



def _indices_for_sessions(data: FeatureSet, sessions: Sequence[int]) -> np.ndarray:
    selected = np.asarray(tuple(int(value) for value in sessions), dtype=np.int32)
    if selected.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.isin(data.session_id, selected)).astype(np.int64)


def _strategy_session_table(
    trades: TradeResults,
    evaluation_sessions: np.ndarray,
) -> dict[str, object]:
    rows: list[dict[str, float | int]] = []
    for session in np.unique(np.asarray(evaluation_sessions, dtype=np.int32)):
        mask = trades.sessions == session
        session_pnl = trades.net_pnl_bps[mask]
        rows.append(
            {
                "session_id": int(session),
                "trades": int(session_pnl.size),
                "total_pnl_bps": float(np.sum(session_pnl)) if session_pnl.size else 0.0,
                "mean_pnl_bps": float(np.mean(session_pnl)) if session_pnl.size else 0.0,
                "win_rate": float(np.mean(session_pnl > 0.0)) if session_pnl.size else 0.0,
            }
        )
    totals = np.asarray([row["total_pnl_bps"] for row in rows], dtype=np.float64)
    return {
        "sessions": rows,
        "positive_session_pnl_fraction": (
            float(np.mean(totals > 0.0)) if totals.size else float("nan")
        ),
        "median_session_total_pnl_bps": (
            float(np.median(totals)) if totals.size else float("nan")
        ),
    }


def _evaluate_baseline(
    signal: np.ndarray,
    data: FeatureSet,
    split: SplitIndices,
    execution: ExecutionAssumptions,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    validation_signal = np.asarray(signal[split.validation], dtype=np.float64)
    test_signal = np.asarray(signal[split.test], dtype=np.float64)
    threshold, validation_metrics = select_signal_threshold(
        validation_signal,
        data,
        split.validation,
        execution.fee_bps_per_side,
        slippage_bps_per_side=execution.slippage_bps_per_side,
        trade_size_base=execution.trade_size_base,
        max_displayed_participation=execution.max_displayed_participation,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    test_trades = strategy_trades(
        test_signal,
        data,
        split.test,
        threshold=threshold,
        execution=execution,
    )
    test_metrics = strategy_metrics(
        test_trades.net_pnl_bps,
        test_trades.sessions,
        gross_pnl=test_trades.gross_pnl_bps,
        round_trip_cost_bps=execution.round_trip_cost_bps,
        fill_rejections=test_trades.fill_rejections,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "validation_threshold": threshold,
        "validation_strategy": asdict(validation_metrics),
        "test_strategy": asdict(test_metrics),
    }


def pretest_walk_forward_diagnostics(
    data: FeatureSet,
    split: SplitIndices,
    execution: ExecutionAssumptions,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    ordered_ranges = _session_row_ranges(data)
    pretest_set = set((*split.train_sessions, *split.validation_sessions))
    pretest_sessions = [session for session, _start, _stop in ordered_ranges if session in pretest_set]
    if len(pretest_sessions) < 3:
        return {
            "available": False,
            "reason": "At least three pre-test sessions are required.",
            "folds": [],
        }

    folds: list[dict[str, object]] = []
    aggregate_prediction: list[np.ndarray] = []
    aggregate_target: list[np.ndarray] = []
    aggregate_net: list[np.ndarray] = []
    aggregate_gross: list[np.ndarray] = []
    aggregate_trade_sessions: list[np.ndarray] = []
    total_fill_rejections = 0

    for target_position in range(2, len(pretest_sessions)):
        train_sessions = pretest_sessions[: target_position - 1]
        validation_session = pretest_sessions[target_position - 1]
        target_session = pretest_sessions[target_position]
        train_indices = _indices_for_sessions(data, train_sessions)
        validation_indices = _indices_for_sessions(data, [validation_session])
        target_indices = _indices_for_sessions(data, [target_session])
        if min(train_indices.size, validation_indices.size, target_indices.size) < MINIMUM_PARTITION_ROWS:
            continue

        X_train, mean, scale = standardize_training_data(data.X[train_indices])
        X_validation = standardize_with_parameters(data.X[validation_indices], mean, scale)
        X_target = standardize_with_parameters(data.X[target_indices], mean, scale)
        alpha, coefficients, intercept, _search = choose_ridge_alpha(
            X_train,
            data.y[train_indices],
            X_validation,
            data.y[validation_indices],
        )
        validation_prediction = intercept + X_validation @ coefficients
        target_prediction = intercept + X_target @ coefficients
        threshold, _validation_strategy = select_signal_threshold(
            validation_prediction,
            data,
            validation_indices,
            execution.fee_bps_per_side,
            slippage_bps_per_side=execution.slippage_bps_per_side,
            trade_size_base=execution.trade_size_base,
            max_displayed_participation=execution.max_displayed_participation,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + target_position,
        )
        trades = strategy_trades(
            target_prediction,
            data,
            target_indices,
            threshold=threshold,
            execution=execution,
        )
        metrics = strategy_metrics(
            trades.net_pnl_bps,
            trades.sessions,
            gross_pnl=trades.gross_pnl_bps,
            round_trip_cost_bps=execution.round_trip_cost_bps,
            fill_rejections=trades.fill_rejections,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + target_position,
        )
        folds.append(
            {
                "train_sessions": list(train_sessions),
                "validation_session": validation_session,
                "target_session": target_session,
                "ridge_alpha": alpha,
                "signal_threshold_bps": threshold,
                "target_regression": asdict(
                    regression_metrics(target_prediction, data.y[target_indices])
                ),
                "target_strategy": asdict(metrics),
            }
        )
        aggregate_prediction.append(target_prediction)
        aggregate_target.append(data.y[target_indices])
        aggregate_net.append(trades.net_pnl_bps)
        aggregate_gross.append(trades.gross_pnl_bps)
        aggregate_trade_sessions.append(trades.sessions)
        total_fill_rejections += trades.fill_rejections

    if not folds:
        return {
            "available": False,
            "reason": "Pre-test sessions were too short for anchored folds.",
            "folds": [],
        }

    prediction = np.concatenate(aggregate_prediction)
    target = np.concatenate(aggregate_target)
    net = np.concatenate(aggregate_net) if aggregate_net else np.empty(0, dtype=np.float64)
    gross = np.concatenate(aggregate_gross) if aggregate_gross else np.empty(0, dtype=np.float64)
    trade_sessions = (
        np.concatenate(aggregate_trade_sessions)
        if aggregate_trade_sessions
        else np.empty(0, dtype=np.int32)
    )
    aggregate_strategy = strategy_metrics(
        net,
        trade_sessions,
        gross_pnl=gross,
        round_trip_cost_bps=execution.round_trip_cost_bps,
        fill_rejections=total_fill_rejections,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "available": True,
        "folds": folds,
        "aggregate_regression": asdict(regression_metrics(prediction, target)),
        "aggregate_strategy": asdict(aggregate_strategy),
    }

def _serialize_recordings(prepared: Iterable[PreparedRecording]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for item in prepared:
        metadata = item.metadata
        output.append(
            {
                "file": metadata.path.name,
                "sha256": item.sha256,
                "update_id_sha256": (
                    sha256_file(metadata.update_id_path)
                    if metadata.update_id_path is not None
                    else None
                ),
                "created_unix_ns": metadata.created_unix_ns,
                "record_count": metadata.record_count,
                "feature_rows": item.feature_rows,
                "segment_session_ids": list(item.segment_session_ids),
                "segment_sha256": list(item.segment_sha256),
                "clean_shutdown": metadata.clean_shutdown,
                "data_complete": metadata.data_complete,
                "recording_dropped": metadata.recording_dropped,
                "recording_write_errors": metadata.recording_write_errors,
                "reconnect_count": metadata.reconnect_count,
            }
        )
    return output


def save_predictions(
    file_path: str | Path,
    data: FeatureSet,
    indices: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
    trades: TradeResults | None = None,
) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trade_lookup: dict[int, tuple[float, float, int, float]] = {}
    if trades is not None:
        for local_index, row_index in enumerate(trades.selected_rows.tolist()):
            trade_lookup[int(row_index)] = (
                float(trades.gross_pnl_bps[local_index]),
                float(trades.net_pnl_bps[local_index]),
                int(trades.sides[local_index]),
                float(trades.displayed_capacity_base[local_index]),
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "timestamp_ns",
                "session_id",
                "event_index",
                "current_bid",
                "current_ask",
                "future_bid",
                "future_ask",
                "actual_return_bps",
                "predicted_return_bps",
                "executed",
                "executed_side",
                "gross_trade_pnl_bps",
                "net_trade_pnl_bps",
                "displayed_capacity_base",
                "signal",
            ]
        )
        for local_index, row_index in enumerate(np.asarray(indices).tolist()):
            forecast = float(prediction[local_index])
            signal = (
                "LONG"
                if forecast > 0.0 and forecast >= threshold
                else "SHORT"
                if forecast < 0.0 and forecast <= -threshold
                else "FLAT"
            )
            executed = trade_lookup.get(int(row_index))
            writer.writerow(
                [
                    int(data.timestamps_ns[row_index]),
                    int(data.session_id[row_index]),
                    int(data.event_index[row_index]),
                    float(data.current_bid[row_index]),
                    float(data.current_ask[row_index]),
                    float(data.future_bid[row_index]),
                    float(data.future_ask[row_index]),
                    float(data.y[row_index]),
                    forecast,
                    executed is not None,
                    (
                        "LONG"
                        if executed is not None and executed[2] > 0
                        else "SHORT"
                        if executed is not None and executed[2] < 0
                        else ""
                    ),
                    executed[0] if executed is not None else "",
                    executed[1] if executed is not None else "",
                    executed[3] if executed is not None else "",
                    signal,
                ]
            )


def train_and_evaluate(
    recording_paths: Sequence[str | Path],
    *,
    horizon: int,
    fee_bps_per_side: float,
    model_path: str | Path,
    report_path: str | Path,
    predictions_path: str | Path,
    evidence_path: str | Path | None = None,
    slippage_bps_per_side: float = 0.0,
    trade_size_base: float = 0.0,
    max_displayed_participation: float = 1.0,
    diagnostic_resamples: int = 500,
    diagnostic_seed: int = 0,
    cost_stress_grid_bps_per_side: Sequence[float] = (
        0.0,
        0.01,
        0.05,
        0.10,
        0.25,
        0.50,
        1.00,
    ),
    max_gap_ns: int = DEFAULT_MAX_GAP_NS,
    allow_incomplete: bool = False,
    overwrite: bool = False,
    allow_test_reuse: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    execution = ExecutionAssumptions(
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        trade_size_base=trade_size_base,
        max_displayed_participation=max_displayed_participation,
    )
    execution.validate()
    if diagnostic_resamples < 100:
        raise ValueError("At least 100 diagnostic resamples are required.")
    if not isinstance(diagnostic_seed, int):
        raise ValueError("Diagnostic seed must be an integer.")

    output_paths = [Path(model_path), Path(report_path), Path(predictions_path)]
    if evidence_path is not None:
        output_paths.append(Path(evidence_path))
    output_tuple = tuple(output_paths)
    if len({path.resolve() for path in output_tuple}) != len(output_tuple):
        raise ValueError("Research artifact outputs must be distinct files.")
    existing = [path for path in output_tuple if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing research artifacts:\n" + joined
        )

    _progress(progress, 2, "Loading and validating recordings")
    data, prepared = prepare_feature_data(
        recording_paths,
        horizon=horizon,
        max_gap_ns=max_gap_ns,
        allow_incomplete=allow_incomplete,
        progress=progress,
    )
    purge_size = max(horizon, 25)
    split = make_chronological_split(data, purge_size)
    test_set_fingerprint = _test_set_fingerprint(
        data,
        split.test,
        horizon=horizon,
    )
    test_period_fingerprint = _test_period_fingerprint(
        prepared,
        split.test_sessions,
    )
    prior_holdout_reports = _prior_holdout_reports(
        Path(report_path).parent,
        test_period_fingerprint=test_period_fingerprint,
        test_set_fingerprint=test_set_fingerprint,
    )
    if prior_holdout_reports and not allow_test_reuse:
        joined = "\n".join(f"  {path}" for path in prior_holdout_reports)
        raise RuntimeError(
            "This held-out market period has already been evaluated. Reusing it can "
            "turn the holdout into a tuning set. Use new later recordings, or "
            "pass allow_test_reuse=True / --allow-test-reuse only for an "
            "explicit reproducibility rerun. Prior reports:\n" + joined
        )

    _progress(progress, 35, "Fitting training-only normalization")
    X_train, mean, scale = standardize_training_data(data.X[split.train])
    X_validation = standardize_with_parameters(data.X[split.validation], mean, scale)
    X_test = standardize_with_parameters(data.X[split.test], mean, scale)
    y_train = data.y[split.train]
    y_validation = data.y[split.validation]
    y_test = data.y[split.test]

    _progress(progress, 48, "Selecting ridge regularization on validation data")
    best_alpha, coefficients, intercept, alpha_search = choose_ridge_alpha(
        X_train, y_train, X_validation, y_validation
    )
    validation_prediction = intercept + X_validation @ coefficients
    test_prediction = intercept + X_test @ coefficients
    threshold, validation_strategy = select_signal_threshold(
        validation_prediction,
        data,
        split.validation,
        execution.fee_bps_per_side,
        slippage_bps_per_side=execution.slippage_bps_per_side,
        trade_size_base=execution.trade_size_base,
        max_displayed_participation=execution.max_displayed_participation,
        bootstrap_samples=min(diagnostic_resamples, 200),
        bootstrap_seed=diagnostic_seed,
    )

    _progress(progress, 62, "Evaluating the untouched chronological test partition")
    test_trades = strategy_trades(
        test_prediction,
        data,
        split.test,
        threshold=threshold,
        execution=execution,
    )
    test_strategy = strategy_metrics(
        test_trades.net_pnl_bps,
        test_trades.sessions,
        gross_pnl=test_trades.gross_pnl_bps,
        round_trip_cost_bps=execution.round_trip_cost_bps,
        fill_rejections=test_trades.fill_rejections,
        bootstrap_samples=diagnostic_resamples,
        bootstrap_seed=diagnostic_seed,
    )
    validation_regression = regression_metrics(validation_prediction, y_validation)
    test_regression = regression_metrics(test_prediction, y_test)

    _progress(progress, 72, "Running baselines and execution-cost stress tests")
    baseline_signals = {
        "order_book_imbalance": data.imbalance,
        "microprice_edge": data.microprice_edge_bps,
        "one_event_momentum": data.X[:, FEATURE_NAMES.index("mid_return_1")],
    }
    baselines = {
        name: _evaluate_baseline(
            signal,
            data,
            split,
            execution,
            bootstrap_samples=min(diagnostic_resamples, 500),
            bootstrap_seed=diagnostic_seed + index + 1,
        )
        for index, (name, signal) in enumerate(baseline_signals.items())
    }
    cost_stress = cost_stress_curve(
        test_trades.gross_pnl_bps,
        base_fee_bps_per_side=execution.fee_bps_per_side,
        base_slippage_bps_per_side=execution.slippage_bps_per_side,
        extra_cost_bps_per_side=cost_stress_grid_bps_per_side,
    )

    _progress(progress, 80, "Measuring stability, drift, and inferential robustness")
    robustness = {
        "test_ic_circular_shift": asdict(
            circular_shift_permutation_test(
                test_prediction,
                y_test,
                data.session_id[split.test],
                samples=diagnostic_resamples,
                seed=diagnostic_seed,
            )
        ),
        "test_prediction_quantiles": prediction_quantile_table(
            test_prediction,
            y_test,
            bins=10,
        ),
        "test_session_regression": session_regression_table(
            test_prediction,
            y_test,
            data.session_id[split.test],
        ),
        "test_session_strategy": _strategy_session_table(
            test_trades,
            data.session_id[split.test],
        ),
        "pretest_anchored_walk_forward": pretest_walk_forward_diagnostics(
            data,
            split,
            execution,
            bootstrap_samples=min(diagnostic_resamples, 500),
            bootstrap_seed=diagnostic_seed,
        ),
        "training_coefficient_bootstrap": coefficient_bootstrap_stability(
            X_train,
            y_train,
            data.session_id[split.train],
            feature_names=FEATURE_NAMES,
            alpha=best_alpha,
            fit_function=fit_ridge,
            samples=diagnostic_resamples,
            seed=diagnostic_seed,
        ),
    }
    drift = feature_drift_report(
        data.X[split.train],
        data.X[split.validation],
        data.X[split.test],
        FEATURE_NAMES,
    )

    root = Path(__file__).resolve().parent
    recording_report = _serialize_recordings(prepared)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "recording_sha256": [item.sha256 for item in prepared],
        "update_id_sha256": [
            sha256_file(item.metadata.update_id_path)
            if item.metadata.update_id_path is not None
            else None
            for item in prepared
        ],
        "split_mode": split.mode,
        "test_period_fingerprint": test_period_fingerprint,
        "test_set_fingerprint": test_set_fingerprint,
        "prior_test_evaluations": [path.name for path in prior_holdout_reports],
        "command_line": list(sys.argv),
        "python_implementation": platform.python_implementation(),
        "configuration": {
            "horizon_events": horizon,
            "fee_bps_per_side": execution.fee_bps_per_side,
            "slippage_bps_per_side": execution.slippage_bps_per_side,
            "trade_size_base": execution.trade_size_base,
            "max_displayed_participation": execution.max_displayed_participation,
            "diagnostic_resamples": diagnostic_resamples,
            "diagnostic_seed": diagnostic_seed,
            "cost_stress_grid_bps_per_side": [
                float(value) for value in cost_stress_grid_bps_per_side
            ],
            "max_gap_ns": max_gap_ns,
            "allow_incomplete": allow_incomplete,
            "allow_test_reuse": allow_test_reuse,
        },
    }
    model = AlphaModel(
        feature_names=FEATURE_NAMES,
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        intercept=intercept,
        horizon=horizon,
        signal_threshold_bps=threshold,
        ridge_alpha=best_alpha,
        fee_bps_per_side=fee_bps_per_side,
        provenance=provenance,
    )

    coefficient_order = np.argsort(np.abs(coefficients))[::-1]
    coefficient_report = [
        {
            "feature": FEATURE_NAMES[index],
            "standardized_coefficient": float(coefficients[index]),
        }
        for index in coefficient_order.tolist()
    ]
    displayed_capacity = test_trades.displayed_capacity_base
    capacity_summary = {
        "filter_enabled": execution.trade_size_base > 0.0,
        "trade_size_base": execution.trade_size_base,
        "max_displayed_participation": execution.max_displayed_participation,
        "fill_rejections": test_trades.fill_rejections,
        "minimum_displayed_capacity_base": (
            float(np.min(displayed_capacity)) if displayed_capacity.size else None
        ),
        "median_displayed_capacity_base": (
            float(np.median(displayed_capacity)) if displayed_capacity.size else None
        ),
        "maximum_displayed_capacity_base": (
            float(np.max(displayed_capacity)) if displayed_capacity.size else None
        ),
        "interpretation": (
            "A top-of-book displayed-liquidity screen, not a queue-position or fill model."
        ),
    }

    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": provenance["created_utc"],
        "provenance": provenance,
        "methodology": {
            "model": "ridge_regression",
            "solver": "augmented_least_squares",
            "target": "future_mid_log_return_bps",
            "horizon_events": horizon,
            "max_gap_ns": max_gap_ns,
            "split_mode": split.mode,
            "purge_rows": split.purge_rows,
            "train_only_normalization": True,
            "validation_only_model_selection": True,
            "spread_crossing": True,
            "fee_bps_per_side": execution.fee_bps_per_side,
            "slippage_bps_per_side": execution.slippage_bps_per_side,
            "round_trip_explicit_cost_bps": execution.round_trip_cost_bps,
            "trade_size_base": execution.trade_size_base,
            "max_displayed_participation": execution.max_displayed_participation,
            "overlapping_trades": False,
            "test_used_for_selection": False,
            "test_period_fingerprint": test_period_fingerprint,
            "test_set_fingerprint": test_set_fingerprint,
            "test_reuse_allowed": allow_test_reuse,
            "prior_test_evaluations": len(prior_holdout_reports),
            "ridge_candidates": len(RIDGE_GRID),
            "threshold_candidates": len(THRESHOLD_QUANTILES),
        },
        "recordings": recording_report,
        "split": {
            "train_sessions": list(split.train_sessions),
            "validation_sessions": list(split.validation_sessions),
            "test_sessions": list(split.test_sessions),
        },
        "samples": {
            "total": data.size,
            "train": int(split.train.size),
            "validation": int(split.validation.size),
            "test": int(split.test.size),
        },
        "selected_model": {
            "ridge_alpha": best_alpha,
            "signal_threshold_bps": threshold,
            "intercept": intercept,
            "coefficients": coefficient_report,
        },
        "alpha_search": alpha_search,
        "validation_regression": asdict(validation_regression),
        "test_regression": asdict(test_regression),
        "validation_strategy": asdict(validation_strategy),
        "test_strategy": asdict(test_strategy),
        "baselines": baselines,
        "imbalance_baseline": baselines["order_book_imbalance"],
        "execution_cost_stress": cost_stress,
        "displayed_liquidity_screen": capacity_summary,
        "feature_drift": drift,
        "robustness": robustness,
        "selected_test_trade_rows": int(test_trades.selected_rows.size),
        "limitations": [
            "Top-of-book bookTicker data is not a full L2 order-book reconstruction.",
            "Displayed top-of-book quantity does not establish queue position or guaranteed fillability.",
            "Fixed slippage is a stress assumption, not an empirically calibrated impact model.",
            "The ordinary trade t-statistic assumes independent trades; HAC and session-bootstrap diagnostics are also reported.",
            "Holdout reuse is refused by default; explicit reproducibility reruns remain visible in provenance.",
        ],
    }
    safe_report = _json_safe(report)
    if not isinstance(safe_report, dict):
        raise RuntimeError("Internal report serialization failed.")
    report = safe_report
    evidence_markdown = render_research_card(report)

    _progress(progress, 92, "Writing model, report, predictions, and evidence card")
    temporary_paths: list[Path] = []
    try:
        for output_path in output_tuple:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        def temporary_for(path: Path, suffix: str) -> Path:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=suffix,
                dir=path.parent,
            )
            os.close(descriptor)
            temporary = Path(name)
            temporary_paths.append(temporary)
            return temporary

        model_temporary = temporary_for(Path(model_path), ".tmp.npz")
        report_temporary = temporary_for(Path(report_path), ".tmp.json")
        predictions_temporary = temporary_for(Path(predictions_path), ".tmp.csv")
        evidence_temporary = (
            temporary_for(Path(evidence_path), ".tmp.md")
            if evidence_path is not None
            else None
        )
        model.save(model_temporary, overwrite=True)
        save_predictions(
            predictions_temporary,
            data,
            split.test,
            test_prediction,
            threshold,
            test_trades,
        )
        if evidence_temporary is not None:
            evidence_temporary.write_text(evidence_markdown, encoding="utf-8")

        report["artifacts"] = {
            "model": {
                "file": Path(model_path).name,
                "sha256": sha256_file(model_temporary),
            },
            "predictions": {
                "file": Path(predictions_path).name,
                "sha256": sha256_file(predictions_temporary),
            },
            "evidence_card": (
                {
                    "file": Path(evidence_path).name,
                    "sha256": sha256_file(evidence_temporary),
                }
                if evidence_temporary is not None and evidence_path is not None
                else None
            ),
        }
        report_temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

        if not overwrite:
            raced = [path for path in output_tuple if path.exists()]
            if raced:
                joined = "\n".join(f"  {path}" for path in raced)
                raise FileExistsError(
                    "Research artifacts appeared during training; refusing to "
                    "overwrite them:\n" + joined
                )

        # All files are completely staged before publication. The JSON report
        # is committed last, so report discovery never exposes a partial set.
        committed: list[Path] = []
        commit_pairs: list[tuple[Path, Path]] = [
            (model_temporary, Path(model_path)),
            (predictions_temporary, Path(predictions_path)),
        ]
        if evidence_temporary is not None and evidence_path is not None:
            commit_pairs.append((evidence_temporary, Path(evidence_path)))
        commit_pairs.append((report_temporary, Path(report_path)))
        try:
            for source, destination in commit_pairs:
                os.replace(source, destination)
                committed.append(destination)
        except Exception:
            for destination in reversed(committed):
                destination.unlink(missing_ok=True)
            raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)

    _progress(progress, 100, "Training completed")
    return report

