from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from l2bin import L2Metadata, read_metadata, sha256_file

from l2_features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    L2FeatureSet,
    build_feature_set,
)
from research import (
    ExecutionAssumptions,
    choose_ridge_alpha,
    make_chronological_split,
    pretest_walk_forward_diagnostics,
    regression_metrics,
    select_signal_threshold,
    standardize_training_data,
    standardize_with_parameters,
    strategy_metrics,
    strategy_trades,
)

MODEL_SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 2

@dataclass(frozen=True, slots=True)
class PreparedL2Recording:
    path: Path
    metadata: L2Metadata
    sha256: str
    checkpoint_sha256: str


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
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def prepare_recordings(recordings: Iterable[str | Path]) -> tuple[PreparedL2Recording, ...]:
    prepared: list[PreparedL2Recording] = []
    seen_hashes: dict[str, Path] = {}
    symbols: set[str] = set()
    for raw_path in recordings:
        path = Path(raw_path).expanduser().resolve()
        metadata = read_metadata(path, verify_hashes=True)
        if metadata.data_complete is not True:
            raise RuntimeError(
                f"Refusing to research an incomplete L2 recording: {path}"
            )
        if metadata.sha256 is None or metadata.checkpoint_sha256 is None:
            raise RuntimeError(f"Complete L2 recording lacks hashes: {path}")
        previous = seen_hashes.get(metadata.sha256)
        if previous is not None:
            raise RuntimeError(
                "Duplicate L2 recording content was supplied:\n"
                f"  {previous}\n  {path}"
            )
        seen_hashes[metadata.sha256] = path
        symbols.add(metadata.symbol)
        prepared.append(
            PreparedL2Recording(
                path=path,
                metadata=metadata,
                sha256=metadata.sha256,
                checkpoint_sha256=metadata.checkpoint_sha256,
            )
        )
    if not prepared:
        raise ValueError("At least one complete L2 recording is required.")
    if len(symbols) != 1:
        raise ValueError(
            "One research experiment must use exactly one symbol; received: "
            + ", ".join(sorted(symbols))
        )
    prepared.sort(
        key=lambda item: (
            item.metadata.created_unix_ns,
            item.path.name.lower(),
            str(item.path).lower(),
        )
    )
    return tuple(prepared)


def _prior_test_reports(
    output_directory: Path,
    test_fingerprint: str,
) -> tuple[Path, ...]:
    matches: list[Path] = []
    for path in output_directory.glob("*_report.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("test_fingerprint_sha256") == test_fingerprint:
            matches.append(path.resolve())
    return tuple(sorted(matches))


def _recording_report(
    prepared: tuple[PreparedL2Recording, ...],
) -> list[dict[str, object]]:
    return [
        {
            "file": item.path.name,
            "symbol": item.metadata.symbol,
            "created_unix_ns": item.metadata.created_unix_ns,
            "event_count": item.metadata.event_count,
            "snapshot_count": item.metadata.snapshot_count,
            "depth_count": item.metadata.depth_count,
            "trade_count": item.metadata.trade_count,
            "boundary_count": item.metadata.boundary_count,
            "sequence_gaps": item.metadata.sequence_gaps,
            "snapshot_retries": item.metadata.snapshot_retries,
            "sha256": item.sha256,
            "checkpoint_sha256": item.checkpoint_sha256,
        }
        for item in prepared
    ]


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "top_of_book": (
        "spread_bps",
        "imbalance_1",
        "microprice_edge_1_bps",
        "top_of_book_ofi",
        "trade_imbalance_20",
        "mid_return_1",
        "mid_return_5",
        "realized_volatility_20",
        "log_interarrival_us",
        "event_rate_10",
        "spread_change_1",
        "imbalance_change_1",
    ),
    "multi_level_depth": (
        "imbalance_5",
        "imbalance_10",
        "imbalance_20",
        "depth_weighted_mid_edge_5_bps",
        "log_bid_depth_5",
        "log_ask_depth_5",
        "bid_depth_slope_10",
        "ask_depth_slope_10",
        "bid_concentration_1_10",
        "ask_concentration_1_10",
        "bid_convexity_5_20",
        "ask_convexity_5_20",
        "depth_addition_imbalance",
        "cancellation_imbalance",
    ),
    "full_l2": FEATURE_NAMES,
}


def _indices(names: Iterable[str]) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(FEATURE_NAMES)}
    return np.asarray([mapping[name] for name in names], dtype=np.int64)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite research artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _staging_path(destination: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _format_metric(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}f}" if math.isfinite(number) else "N/A"


def _fingerprint(data: L2FeatureSet, indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (
        data.timestamps_ns[indices],
        data.update_ids[indices],
        data.X[indices],
        data.y[indices],
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _delay_predictions(
    predictions: np.ndarray,
    session_ids: np.ndarray,
    delay_events: int,
) -> np.ndarray:
    if delay_events <= 0:
        return predictions.copy()
    output = np.zeros_like(predictions)
    for session in np.unique(session_ids):
        local = np.flatnonzero(session_ids == session)
        if local.size > delay_events:
            output[local[delay_events:]] = predictions[local[:-delay_events]]
    return output


def _evaluate_predictions(
    prediction: np.ndarray,
    data: L2FeatureSet,
    indices: np.ndarray,
    *,
    threshold: float,
    execution: ExecutionAssumptions,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    trades = strategy_trades(
        prediction,
        data,  # type: ignore[arg-type]
        indices,
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
    return {
        "regression": asdict(regression_metrics(prediction, data.y[indices])),
        "strategy": asdict(metrics),
        "selected_rows": trades.selected_rows,
        "sides": trades.sides,
        "net_pnl_bps": trades.net_pnl_bps,
        "gross_pnl_bps": trades.gross_pnl_bps,
    }


def _regime_diagnostics(
    prediction: np.ndarray,
    data: L2FeatureSet,
    test_indices: np.ndarray,
    threshold: float,
    execution: ExecutionAssumptions,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    spread_column = FEATURE_NAMES.index("spread_bps")
    volatility_column = FEATURE_NAMES.index("realized_volatility_20")
    result: dict[str, object] = {}
    for label, values in (
        ("spread", data.X[test_indices, spread_column]),
        ("volatility", data.X[test_indices, volatility_column]),
    ):
        low, high = np.quantile(values, [1 / 3, 2 / 3])
        buckets = {
            "low": values <= low,
            "medium": (values > low) & (values <= high),
            "high": values > high,
        }
        bucket_results: dict[str, object] = {}
        for offset, (bucket, mask) in enumerate(buckets.items()):
            local_indices = np.flatnonzero(mask)
            if local_indices.size < 10:
                bucket_results[bucket] = {"samples": int(local_indices.size)}
                continue
            selected_indices = test_indices[local_indices]
            evaluation = _evaluate_predictions(
                prediction[local_indices],
                data,
                selected_indices,
                threshold=threshold,
                execution=execution,
                bootstrap_samples=max(200, bootstrap_samples // 4),
                bootstrap_seed=bootstrap_seed + offset,
            )
            bucket_results[bucket] = {
                "samples": int(local_indices.size),
                "regression": evaluation["regression"],
                "strategy": evaluation["strategy"],
            }
        result[label] = {
            "lower_cut": float(low),
            "upper_cut": float(high),
            "buckets": bucket_results,
        }
    return result


def _fit_group(
    group_name: str,
    data: L2FeatureSet,
    split,
    execution: ExecutionAssumptions,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    names = FEATURE_GROUPS[group_name]
    columns = _indices(names)
    X_train_raw = data.X[split.train][:, columns]
    X_validation_raw = data.X[split.validation][:, columns]
    X_test_raw = data.X[split.test][:, columns]
    X_train, mean, scale = standardize_training_data(X_train_raw)
    X_validation = standardize_with_parameters(X_validation_raw, mean, scale)
    X_test = standardize_with_parameters(X_test_raw, mean, scale)
    alpha, coefficients, intercept, search = choose_ridge_alpha(
        X_train,
        data.y[split.train],
        X_validation,
        data.y[split.validation],
    )
    validation_prediction = intercept + X_validation @ coefficients
    test_prediction = intercept + X_test @ coefficients
    threshold, validation_strategy = select_signal_threshold(
        validation_prediction,
        data,  # type: ignore[arg-type]
        split.validation,
        execution.fee_bps_per_side,
        slippage_bps_per_side=execution.slippage_bps_per_side,
        trade_size_base=execution.trade_size_base,
        max_displayed_participation=execution.max_displayed_participation,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    test_evaluation = _evaluate_predictions(
        test_prediction,
        data,
        split.test,
        threshold=threshold,
        execution=execution,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    coefficient_table = sorted(
        (
            {
                "feature": name,
                "standardized_coefficient": float(value),
                "absolute_standardized_coefficient": abs(float(value)),
            }
            for name, value in zip(names, coefficients)
        ),
        key=lambda row: row["absolute_standardized_coefficient"],
        reverse=True,
    )
    return {
        "group": group_name,
        "feature_names": list(names),
        "columns": columns,
        "mean": mean,
        "scale": scale,
        "coefficients": coefficients,
        "intercept": float(intercept),
        "ridge_alpha": float(alpha),
        "ridge_search": search,
        "signal_threshold_bps": float(threshold),
        "validation_regression": asdict(
            regression_metrics(validation_prediction, data.y[split.validation])
        ),
        "validation_strategy": asdict(validation_strategy),
        "test_prediction": test_prediction,
        "test_regression": test_evaluation["regression"],
        "test_strategy": test_evaluation["strategy"],
        "test_selected_rows": test_evaluation["selected_rows"],
        "test_sides": test_evaluation["sides"],
        "test_net_pnl_bps": test_evaluation["net_pnl_bps"],
        "test_gross_pnl_bps": test_evaluation["gross_pnl_bps"],
        "coefficient_table": coefficient_table,
    }


def run_experiment(
    recordings: list[Path],
    *,
    horizon: int,
    output_directory: Path,
    fee_bps_per_side: float,
    slippage_bps_per_side: float,
    trade_size_base: float,
    max_displayed_participation: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    allow_test_reuse: bool = False,
) -> dict[str, object]:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100.")
    prepared = prepare_recordings(recordings)
    canonical_recordings = [item.path for item in prepared]
    data = build_feature_set(
        canonical_recordings, horizon=horizon, require_complete=False
    )
    split = make_chronological_split(
        data, purge_size=max(horizon, 25)
    )  # type: ignore[arg-type]
    test_fingerprint = _fingerprint(data, split.test)
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    prior_reports = _prior_test_reports(output_directory, test_fingerprint)
    if prior_reports and not allow_test_reuse:
        joined = "\n".join(f"  {path}" for path in prior_reports)
        raise RuntimeError(
            "This exact L2 holdout has already been evaluated. Use later data, "
            "or set allow_test_reuse=True / --allow-test-reuse only for an "
            "explicit reproducibility rerun. Prior reports:\n" + joined
        )

    stem = f"l2_h{horizon}"
    report_path = output_directory / f"{stem}_report.json"
    model_path = output_directory / f"{stem}_model.npz"
    predictions_path = output_directory / f"{stem}_test_predictions.csv"
    card_path = output_directory / f"{stem}_research_card.md"
    destinations = (model_path, predictions_path, card_path, report_path)
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing L2 research artifacts:\n"
            + "\n".join(f"  {path}" for path in existing)
        )

    execution = ExecutionAssumptions(
        fee_bps_per_side=fee_bps_per_side,
        slippage_bps_per_side=slippage_bps_per_side,
        trade_size_base=trade_size_base,
        max_displayed_participation=max_displayed_participation,
    )
    execution.validate()
    group_results = {
        group: _fit_group(
            group,
            data,
            split,
            execution,
            bootstrap_samples,
            bootstrap_seed + offset * 100,
        )
        for offset, group in enumerate(FEATURE_GROUPS)
    }
    primary = group_results["full_l2"]
    test_prediction = np.asarray(primary["test_prediction"], dtype=np.float64)
    threshold = float(primary["signal_threshold_bps"])

    latency_stress: list[dict[str, object]] = []
    for delay in (0, 1, 2, 5, 10, 20):
        delayed = _delay_predictions(
            test_prediction,
            data.session_id[split.test],
            delay,
        )
        evaluation = _evaluate_predictions(
            delayed,
            data,
            split.test,
            threshold=threshold,
            execution=execution,
            bootstrap_samples=max(200, bootstrap_samples // 4),
            bootstrap_seed=bootstrap_seed + delay,
        )
        latency_stress.append(
            {
                "delay_events": delay,
                "regression": evaluation["regression"],
                "strategy": evaluation["strategy"],
            }
        )

    cost_stress: list[dict[str, object]] = []
    for additional_per_side in (0.0, 0.05, 0.10, 0.25, 0.50, 1.00):
        stressed = ExecutionAssumptions(
            fee_bps_per_side=execution.fee_bps_per_side,
            slippage_bps_per_side=execution.slippage_bps_per_side + additional_per_side,
            trade_size_base=execution.trade_size_base,
            max_displayed_participation=execution.max_displayed_participation,
        )
        evaluation = _evaluate_predictions(
            test_prediction,
            data,
            split.test,
            threshold=threshold,
            execution=stressed,
            bootstrap_samples=max(200, bootstrap_samples // 4),
            bootstrap_seed=bootstrap_seed + int(additional_per_side * 100),
        )
        cost_stress.append(
            {
                "additional_cost_bps_per_side": additional_per_side,
                "strategy": evaluation["strategy"],
            }
        )

    baselines: dict[str, object] = {}
    baseline_signals = {
        "top_of_book_imbalance": data.imbalance,
        "microprice_edge": data.microprice_edge_bps,
        "top_of_book_ofi": data.X[:, FEATURE_NAMES.index("top_of_book_ofi")],
    }
    for offset, (name, signal) in enumerate(baseline_signals.items()):
        validation_signal = signal[split.validation]
        test_signal = signal[split.test]
        baseline_threshold, validation_metrics = select_signal_threshold(
            validation_signal,
            data,  # type: ignore[arg-type]
            split.validation,
            execution.fee_bps_per_side,
            slippage_bps_per_side=execution.slippage_bps_per_side,
            trade_size_base=execution.trade_size_base,
            max_displayed_participation=execution.max_displayed_participation,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 1_000 + offset,
        )
        evaluation = _evaluate_predictions(
            test_signal,
            data,
            split.test,
            threshold=baseline_threshold,
            execution=execution,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 2_000 + offset,
        )
        baselines[name] = {
            "validation_threshold": baseline_threshold,
            "validation_strategy": asdict(validation_metrics),
            "test_regression": evaluation["regression"],
            "test_strategy": evaluation["strategy"],
        }

    walk_forward = pretest_walk_forward_diagnostics(
        data,  # type: ignore[arg-type]
        split,
        execution,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    regime = _regime_diagnostics(
        test_prediction,
        data,
        split.test,
        threshold,
        execution,
        bootstrap_samples,
        bootstrap_seed,
    )

    root = Path(__file__).resolve().parent
    provenance: dict[str, object] = {
        "git_commit": _git_commit(root),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "command_line": list(sys.argv),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "allow_test_reuse": allow_test_reuse,
        "prior_holdout_reports": [path.name for path in prior_reports],
    }
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": (
            "Do multi-level L2 features improve short-horizon prediction and "
            "execution-adjusted diagnostics beyond top-of-book baselines?"
        ),
        "symbol": prepared[0].metadata.symbol,
        "recordings": _recording_report(prepared),
        "horizon_events": horizon,
        "feature_schema": {
            "version": FEATURE_SCHEMA_VERSION,
            "sha256": FEATURE_SCHEMA_HASH,
            "names": list(FEATURE_NAMES),
        },
        "samples": data.size,
        "split": {
            "mode": split.mode,
            "train_rows": int(split.train.size),
            "validation_rows": int(split.validation.size),
            "test_rows": int(split.test.size),
            "train_sessions": list(split.train_sessions),
            "validation_sessions": list(split.validation_sessions),
            "test_sessions": list(split.test_sessions),
            "purge_rows": split.purge_rows,
        },
        "test_fingerprint_sha256": test_fingerprint,
        "test_reuse": {
            "explicitly_allowed": allow_test_reuse,
            "prior_matching_reports": [path.name for path in prior_reports],
        },
        "execution_assumptions": asdict(execution),
        "predeclared_primary_model": "full_l2",
        "feature_group_comparison": {
            name: {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "columns",
                    "mean",
                    "scale",
                    "coefficients",
                    "test_prediction",
                    "test_selected_rows",
                    "test_sides",
                    "test_net_pnl_bps",
                    "test_gross_pnl_bps",
                }
            }
            for name, result in group_results.items()
        },
        "baselines": baselines,
        "latency_stress": latency_stress,
        "cost_stress": cost_stress,
        "regime_diagnostics": regime,
        "pretest_walk_forward": walk_forward,
        "provenance": provenance,
        "limitations": [
            "Aggregated depth does not reveal exact order-level queue position.",
            "Passive-fill results must be reported across queue-model sensitivity cases.",
            "Direction accuracy must be compared with its majority-direction baseline.",
            "The experiment is research evidence, not a claim of live profitability.",
        ],
    }

    selected_rows = set(
        np.asarray(primary["test_selected_rows"], dtype=np.int64).tolist()
    )
    side_by_row = {
        int(row): int(side)
        for row, side in zip(
            np.asarray(primary["test_selected_rows"], dtype=np.int64),
            np.asarray(primary["test_sides"], dtype=np.int8),
        )
    }

    staged_model = _staging_path(model_path, ".tmp.npz")
    staged_predictions = _staging_path(predictions_path, ".tmp.csv")
    staged_card = _staging_path(card_path, ".tmp.md")
    staged_report = _staging_path(report_path, ".tmp.json")
    staged = (staged_model, staged_predictions, staged_card, staged_report)
    committed: list[Path] = []
    try:
        model_provenance = {
            **provenance,
            "symbol": prepared[0].metadata.symbol,
            "recording_sha256": [item.sha256 for item in prepared],
            "checkpoint_sha256": [item.checkpoint_sha256 for item in prepared],
            "test_fingerprint_sha256": test_fingerprint,
            "execution_assumptions": asdict(execution),
        }
        np.savez_compressed(
            staged_model,
            schema_version=np.asarray(MODEL_SCHEMA_VERSION, dtype=np.int64),
            feature_schema_version=np.asarray(
                FEATURE_SCHEMA_VERSION, dtype=np.int64
            ),
            feature_schema_hash=np.asarray(FEATURE_SCHEMA_HASH, dtype=np.str_),
            feature_names=np.asarray(primary["feature_names"], dtype=np.str_),
            mean=np.asarray(primary["mean"], dtype=np.float64),
            scale=np.asarray(primary["scale"], dtype=np.float64),
            coefficients=np.asarray(primary["coefficients"], dtype=np.float64),
            intercept=np.asarray(primary["intercept"], dtype=np.float64),
            ridge_alpha=np.asarray(primary["ridge_alpha"], dtype=np.float64),
            signal_threshold_bps=np.asarray(
                primary["signal_threshold_bps"], dtype=np.float64
            ),
            horizon=np.asarray(horizon, dtype=np.int64),
            symbol=np.asarray(prepared[0].metadata.symbol, dtype=np.str_),
            test_fingerprint_sha256=np.asarray(test_fingerprint, dtype=np.str_),
            provenance_json=np.asarray(
                json.dumps(_json_safe(model_provenance), sort_keys=True, allow_nan=False),
                dtype=np.str_,
            ),
        )

        with staged_predictions.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "global_row",
                    "timestamp_ns",
                    "update_id",
                    "session_id",
                    "prediction_bps",
                    "target_bps",
                    "best_bid",
                    "best_ask",
                    "selected_trade",
                    "side",
                ]
            )
            for local, global_row in enumerate(split.test.tolist()):
                writer.writerow(
                    [
                        global_row,
                        int(data.timestamps_ns[global_row]),
                        int(data.update_ids[global_row]),
                        int(data.session_id[global_row]),
                        float(test_prediction[local]),
                        float(data.y[global_row]),
                        float(data.current_bid[global_row]),
                        float(data.current_ask[global_row]),
                        int(global_row in selected_rows),
                        side_by_row.get(global_row, 0),
                    ]
                )

        primary_strategy = primary["test_strategy"]
        primary_regression = primary["test_regression"]
        top_strategy = group_results["top_of_book"]["test_strategy"]
        top_regression = group_results["top_of_book"]["test_regression"]
        card = (
            f"# L2 Research Evidence — Horizon {horizon}\n\n"
            f"**Question:** {report['experiment']}\n\n"
            f"- Symbol: {prepared[0].metadata.symbol}\n"
            f"- Complete recordings: {len(prepared)}\n"
            f"- Samples: {data.size:,}\n"
            f"- Split: {split.mode}\n"
            f"- Held-out test fingerprint: `{test_fingerprint}`\n"
            f"- Feature schema: `{FEATURE_SCHEMA_HASH}`\n"
            f"- Full-L2 Pearson IC: {_format_metric(primary_regression.get('pearson_ic'))}\n"
            "- Full-L2 balanced direction accuracy: "
            f"{_format_metric(primary_regression.get('balanced_direction_accuracy'))}\n"
            "- Full-L2 majority-direction baseline: "
            f"{_format_metric(primary_regression.get('majority_direction_accuracy'))}\n"
            "- Full-L2 accuracy lift over majority: "
            f"{_format_metric(primary_regression.get('direction_accuracy_lift_vs_majority'))}\n"
            f"- Top-of-book Pearson IC: {_format_metric(top_regression.get('pearson_ic'))}\n"
            f"- Full-L2 net PnL: {_format_metric(primary_strategy.get('total_pnl_bps'))} "
            f"bps across {int(primary_strategy.get('trades', 0))} trades\n"
            f"- Top-of-book net PnL: {_format_metric(top_strategy.get('total_pnl_bps'))} "
            f"bps across {int(top_strategy.get('trades', 0))} trades\n"
            "- Full-L2 Newey-West t-stat: "
            f"{_format_metric(primary_strategy.get('newey_west_pnl_t_statistic'))}\n\n"
            "Direction metrics are shown relative to their class baseline. PnL values "
            "are execution-adjusted research diagnostics, not live-profitability claims.\n"
        )
        staged_card.write_text(card, encoding="utf-8")

        report["artifacts"] = {
            "model": {"file": model_path.name, "sha256": sha256_file(staged_model)},
            "predictions": {
                "file": predictions_path.name,
                "sha256": sha256_file(staged_predictions),
            },
            "research_card": {
                "file": card_path.name,
                "sha256": sha256_file(staged_card),
            },
        }
        staged_report.write_text(
            json.dumps(_json_safe(report), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

        # All files are fully staged before publication. The report is committed
        # last, so report discovery never exposes a partially written artifact set.
        if any(path.exists() for path in destinations):
            raise FileExistsError(
                "An L2 research artifact appeared while the experiment was running."
            )
        for source, destination in (
            (staged_model, model_path),
            (staged_predictions, predictions_path),
            (staged_card, card_path),
            (staged_report, report_path),
        ):
            os.replace(source, destination)
            committed.append(destination)
    except Exception:
        for destination in reversed(committed):
            destination.unlink(missing_ok=True)
        raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)

    return {
        "horizon": horizon,
        "report": str(report_path),
        "model": str(model_path),
        "predictions": str(predictions_path),
        "research_card": str(card_path),
    }



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical multi-level L2 research experiment.")
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--output-dir", default="artifacts/l2")
    parser.add_argument("--fee-bps-per-side", type=float, default=0.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=0.0)
    parser.add_argument("--trade-size-base", type=float, default=0.0)
    parser.add_argument("--max-displayed-participation", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument(
        "--allow-test-reuse",
        action="store_true",
        help="Permit an explicitly recorded reproducibility rerun of an already evaluated holdout.",
    )
    arguments = parser.parse_args()
    if any(horizon <= 0 for horizon in arguments.horizons):
        parser.error("All horizons must be positive.")
    if arguments.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100.")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    recordings = [Path(value).expanduser().resolve() for value in arguments.recordings]
    output_directory = Path(arguments.output_dir).expanduser().resolve()
    results = [
        run_experiment(
            recordings,
            horizon=horizon,
            output_directory=output_directory,
            fee_bps_per_side=arguments.fee_bps_per_side,
            slippage_bps_per_side=arguments.slippage_bps_per_side,
            trade_size_base=arguments.trade_size_base,
            max_displayed_participation=arguments.max_displayed_participation,
            bootstrap_samples=arguments.bootstrap_samples,
            bootstrap_seed=arguments.bootstrap_seed + horizon,
            allow_test_reuse=arguments.allow_test_reuse,
        )
        for horizon in arguments.horizons
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
