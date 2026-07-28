from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

FitFunction = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, float]]


@dataclass(frozen=True)
class BootstrapSummary:
    samples: int
    mean_ci_low: float
    mean_ci_high: float
    probability_mean_non_positive: float


@dataclass(frozen=True)
class PermutationSummary:
    samples: int
    observed: float
    null_mean: float
    null_standard_deviation: float
    two_sided_p_value: float


def _finite_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_values.size != right_values.size:
        raise ValueError("Input vectors must have the same length.")
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    return left_values[mask], right_values[mask]


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic one-based average ranks with tie handling."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("Rank input contains non-finite values.")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and array[order[stop]] == array[order[start]]:
            stop += 1
        average = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_values, right_values = _finite_pair(left, right)
    if left_values.size < 2:
        return float("nan")
    if np.std(left_values) <= 1e-15 or np.std(right_values) <= 1e-15:
        return 0.0
    return float(np.corrcoef(left_values, right_values)[0, 1])


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_values, right_values = _finite_pair(left, right)
    if left_values.size < 2:
        return float("nan")
    return pearson_correlation(average_ranks(left_values), average_ranks(right_values))


def newey_west_mean_t_statistic(
    values: np.ndarray,
    *,
    max_lag: int | None = None,
) -> float:
    """HAC t-statistic for a sample mean using Bartlett kernel weights."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count < 2:
        return 0.0
    centered = array - float(np.mean(array))
    if max_lag is None:
        max_lag = int(math.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))
    lag_limit = min(max(int(max_lag), 0), count - 1)
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, lag_limit + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        weight = 1.0 - lag / (lag_limit + 1.0)
        long_run_variance += 2.0 * weight * covariance
    variance_of_mean = max(long_run_variance, 0.0) / count
    if variance_of_mean <= 1e-30:
        return 0.0
    return float(np.mean(array) / math.sqrt(variance_of_mean))


def session_bootstrap_summary(
    values: np.ndarray,
    sessions: np.ndarray,
    *,
    samples: int = 2_000,
    seed: int = 0,
) -> BootstrapSummary:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    session_values = np.asarray(sessions, dtype=np.int64).reshape(-1)
    if array.size != session_values.size:
        raise ValueError("Values and sessions must have the same length.")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")
    finite = np.isfinite(array)
    array = array[finite]
    session_values = session_values[finite]
    unique_sessions = np.unique(session_values)
    if array.size == 0 or unique_sessions.size < 2:
        return BootstrapSummary(samples, float("nan"), float("nan"), float("nan"))

    groups = [array[session_values == session] for session in unique_sessions]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = rng.integers(0, len(groups), size=len(groups))
        total = 0.0
        count = 0
        for group_index in selected.tolist():
            group = groups[group_index]
            total += float(np.sum(group, dtype=np.float64))
            count += int(group.size)
        means[sample_index] = total / max(count, 1)
    low, high = np.quantile(means, [0.025, 0.975])
    probability = (1.0 + float(np.count_nonzero(means <= 0.0))) / (samples + 1.0)
    return BootstrapSummary(samples, float(low), float(high), probability)


def circular_shift_permutation_test(
    prediction: np.ndarray,
    target: np.ndarray,
    sessions: np.ndarray,
    *,
    samples: int = 1_000,
    seed: int = 0,
) -> PermutationSummary:
    prediction_values, target_values = _finite_pair(prediction, target)
    raw_sessions = np.asarray(sessions, dtype=np.int64).reshape(-1)
    if raw_sessions.size != np.asarray(prediction).size:
        raise ValueError("Sessions must match the unfiltered prediction vector length.")
    finite = np.isfinite(np.asarray(prediction, dtype=np.float64).reshape(-1)) & np.isfinite(
        np.asarray(target, dtype=np.float64).reshape(-1)
    )
    session_values = raw_sessions[finite]
    observed = pearson_correlation(prediction_values, target_values)
    if samples <= 0:
        raise ValueError("Permutation sample count must be positive.")
    if prediction_values.size < 3 or not math.isfinite(observed):
        return PermutationSummary(samples, observed, float("nan"), float("nan"), float("nan"))

    groups = [np.flatnonzero(session_values == session) for session in np.unique(session_values)]
    rng = np.random.default_rng(seed)
    null = np.empty(samples, dtype=np.float64)
    shifted = np.empty_like(prediction_values)
    for sample_index in range(samples):
        for group in groups:
            if group.size <= 1:
                shifted[group] = prediction_values[group]
                continue
            offset = int(rng.integers(1, group.size))
            shifted[group] = np.roll(prediction_values[group], offset)
        null[sample_index] = pearson_correlation(shifted, target_values)
    p_value = (1.0 + float(np.count_nonzero(np.abs(null) >= abs(observed)))) / (
        samples + 1.0
    )
    return PermutationSummary(
        samples=samples,
        observed=float(observed),
        null_mean=float(np.mean(null)),
        null_standard_deviation=float(np.std(null, ddof=1)) if samples > 1 else 0.0,
        two_sided_p_value=float(p_value),
    )


def prediction_quantile_table(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, object]:
    prediction_values, target_values = _finite_pair(prediction, target)
    if bins < 2:
        raise ValueError("At least two quantile bins are required.")
    if prediction_values.size == 0:
        return {"bins": [], "mean_target_monotonicity": float("nan")}

    boundaries = np.unique(
        np.quantile(prediction_values, np.linspace(0.0, 1.0, bins + 1))
    )
    if boundaries.size <= 2:
        assignments = np.zeros(prediction_values.size, dtype=np.int64)
        bin_count = 1
    else:
        assignments = np.searchsorted(boundaries[1:-1], prediction_values, side="right")
        bin_count = boundaries.size - 1

    rows: list[dict[str, float | int]] = []
    bin_numbers: list[float] = []
    mean_targets: list[float] = []
    for bin_index in range(bin_count):
        mask = assignments == bin_index
        if not np.any(mask):
            continue
        predicted = prediction_values[mask]
        actual = target_values[mask]
        rows.append(
            {
                "quantile": bin_index + 1,
                "samples": int(actual.size),
                "mean_prediction_bps": float(np.mean(predicted)),
                "mean_actual_bps": float(np.mean(actual)),
                "median_actual_bps": float(np.median(actual)),
                "positive_actual_rate": float(np.mean(actual > 0.0)),
            }
        )
        bin_numbers.append(float(bin_index + 1))
        mean_targets.append(float(np.mean(actual)))

    monotonicity = (
        spearman_correlation(np.asarray(bin_numbers), np.asarray(mean_targets))
        if len(rows) >= 2
        else float("nan")
    )
    return {"bins": rows, "mean_target_monotonicity": monotonicity}


def _population_stability_index(reference: np.ndarray, comparison: np.ndarray) -> float:
    reference_values = np.asarray(reference, dtype=np.float64)
    comparison_values = np.asarray(comparison, dtype=np.float64)
    reference_values = reference_values[np.isfinite(reference_values)]
    comparison_values = comparison_values[np.isfinite(comparison_values)]
    if reference_values.size == 0 or comparison_values.size == 0:
        return float("nan")
    interior = np.unique(np.quantile(reference_values, np.linspace(0.1, 0.9, 9)))
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    reference_count, _ = np.histogram(reference_values, bins=edges)
    comparison_count, _ = np.histogram(comparison_values, bins=edges)
    reference_share = np.maximum(reference_count / reference_values.size, 1e-6)
    comparison_share = np.maximum(comparison_count / comparison_values.size, 1e-6)
    return float(
        np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share))
    )


def feature_drift_report(
    X_train: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    feature_names: Sequence[str],
) -> dict[str, object]:
    train = np.asarray(X_train, dtype=np.float64)
    validation = np.asarray(X_validation, dtype=np.float64)
    test = np.asarray(X_test, dtype=np.float64)
    if train.ndim != 2 or validation.ndim != 2 or test.ndim != 2:
        raise ValueError("Feature drift inputs must be matrices.")
    if train.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match feature matrices.")
    train_mean = np.mean(train, axis=0)
    train_scale = np.std(train, axis=0)
    train_scale = np.where(train_scale > 1e-12, train_scale, 1.0)

    rows: list[dict[str, float | str]] = []
    for index, name in enumerate(feature_names):
        validation_shift = float((np.mean(validation[:, index]) - train_mean[index]) / train_scale[index])
        test_shift = float((np.mean(test[:, index]) - train_mean[index]) / train_scale[index])
        rows.append(
            {
                "feature": str(name),
                "validation_standardized_mean_shift": validation_shift,
                "test_standardized_mean_shift": test_shift,
                "validation_psi": _population_stability_index(
                    train[:, index], validation[:, index]
                ),
                "test_psi": _population_stability_index(train[:, index], test[:, index]),
            }
        )
    rows.sort(
        key=lambda row: max(
            abs(float(row["test_standardized_mean_shift"])),
            float(row["test_psi"]),
        ),
        reverse=True,
    )
    max_test_psi = max((float(row["test_psi"]) for row in rows), default=float("nan"))
    max_test_shift = max(
        (abs(float(row["test_standardized_mean_shift"])) for row in rows),
        default=float("nan"),
    )
    return {
        "features": rows,
        "maximum_test_psi": max_test_psi,
        "maximum_absolute_test_standardized_mean_shift": max_test_shift,
    }



def directional_classification_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | int]:
    """Return honest sign-classification diagnostics for a regression forecast.

    Zero target returns are excluded from directional scoring and reported
    separately. Zero predictions count as abstentions: they are incorrect in
    the all-sample direction/balanced-accuracy metrics, while the actionable
    metrics report both their coverage and accuracy explicitly.
    """
    prediction_values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction_values.size != target_values.size or prediction_values.size == 0:
        raise ValueError("Directional metric inputs must be non-empty and equal length.")
    if not np.all(np.isfinite(prediction_values)) or not np.all(np.isfinite(target_values)):
        raise ValueError("Directional metric inputs contain non-finite values.")

    nonzero_target = target_values != 0.0
    nonzero_count = int(np.count_nonzero(nonzero_target))
    total_count = int(target_values.size)
    if nonzero_count == 0:
        return {
            "nonzero_target_samples": 0,
            "target_zero_fraction": 1.0,
            "positive_target_fraction": 0.0,
            "direction_accuracy": 0.0,
            "majority_direction_accuracy": 0.0,
            "direction_accuracy_lift_vs_majority": 0.0,
            "balanced_direction_accuracy": 0.0,
            "actionable_coverage": 0.0,
            "actionable_direction_accuracy": 0.0,
            "actionable_matthews_correlation": 0.0,
            "positive_prediction_fraction": 0.0,
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "abstentions": 0,
        }

    true_sign = np.sign(target_values[nonzero_target]).astype(np.int8)
    predicted_sign = np.sign(prediction_values[nonzero_target]).astype(np.int8)
    positive_targets = true_sign > 0
    negative_targets = true_sign < 0
    positive_count = int(np.count_nonzero(positive_targets))
    negative_count = int(np.count_nonzero(negative_targets))
    abstentions = int(np.count_nonzero(predicted_sign == 0))

    true_positive = int(np.count_nonzero(positive_targets & (predicted_sign > 0)))
    false_negative = int(np.count_nonzero(positive_targets & (predicted_sign < 0)))
    true_negative = int(np.count_nonzero(negative_targets & (predicted_sign < 0)))
    false_positive = int(np.count_nonzero(negative_targets & (predicted_sign > 0)))

    direction_accuracy = float(np.mean(predicted_sign == true_sign))
    majority_accuracy = max(positive_count, negative_count) / nonzero_count
    positive_recall = true_positive / positive_count if positive_count else 0.0
    negative_recall = true_negative / negative_count if negative_count else 0.0
    balanced_accuracy = 0.5 * (positive_recall + negative_recall)

    actionable = predicted_sign != 0
    actionable_count = int(np.count_nonzero(actionable))
    actionable_accuracy = (
        float(np.mean(predicted_sign[actionable] == true_sign[actionable]))
        if actionable_count
        else 0.0
    )
    denominator = math.sqrt(
        max(
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative),
            0,
        )
    )
    matthews = (
        (true_positive * true_negative - false_positive * false_negative) / denominator
        if denominator > 0.0
        else 0.0
    )

    return {
        "nonzero_target_samples": nonzero_count,
        "target_zero_fraction": 1.0 - nonzero_count / total_count,
        "positive_target_fraction": positive_count / nonzero_count,
        "direction_accuracy": direction_accuracy,
        "majority_direction_accuracy": float(majority_accuracy),
        "direction_accuracy_lift_vs_majority": float(direction_accuracy - majority_accuracy),
        "balanced_direction_accuracy": float(balanced_accuracy),
        "actionable_coverage": actionable_count / nonzero_count,
        "actionable_direction_accuracy": actionable_accuracy,
        "actionable_matthews_correlation": float(matthews),
        "positive_prediction_fraction": (
            float(np.mean(predicted_sign[actionable] > 0)) if actionable_count else 0.0
        ),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "abstentions": abstentions,
    }

def session_regression_table(
    prediction: np.ndarray,
    target: np.ndarray,
    sessions: np.ndarray,
) -> dict[str, object]:
    prediction_values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    session_values = np.asarray(sessions, dtype=np.int64).reshape(-1)
    if not (prediction_values.size == target_values.size == session_values.size):
        raise ValueError("Session regression inputs must have the same length.")

    rows: list[dict[str, float | int]] = []
    for session in np.unique(session_values):
        mask = session_values == session
        session_prediction = prediction_values[mask]
        session_target = target_values[mask]
        direction = directional_classification_metrics(
            session_prediction, session_target
        )
        rows.append(
            {
                "session_id": int(session),
                "samples": int(np.count_nonzero(mask)),
                "pearson_ic": pearson_correlation(session_prediction, session_target),
                "spearman_rank_ic": spearman_correlation(session_prediction, session_target),
                "mae": float(np.mean(np.abs(session_prediction - session_target))),
                "direction_accuracy": float(direction["direction_accuracy"]),
                "majority_direction_accuracy": float(
                    direction["majority_direction_accuracy"]
                ),
                "balanced_direction_accuracy": float(
                    direction["balanced_direction_accuracy"]
                ),
                "direction_accuracy_lift_vs_majority": float(
                    direction["direction_accuracy_lift_vs_majority"]
                ),
                "target_zero_fraction": float(direction["target_zero_fraction"]),
            }
        )
    pearson_values = np.asarray([row["pearson_ic"] for row in rows], dtype=np.float64)
    finite = pearson_values[np.isfinite(pearson_values)]
    return {
        "sessions": rows,
        "session_count": len(rows),
        "median_session_pearson_ic": float(np.median(finite)) if finite.size else float("nan"),
        "positive_session_pearson_ic_fraction": (
            float(np.mean(finite > 0.0)) if finite.size else float("nan")
        ),
    }


def coefficient_bootstrap_stability(
    X: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    *,
    feature_names: Sequence[str],
    alpha: float,
    fit_function: FitFunction,
    samples: int = 500,
    seed: int = 0,
) -> dict[str, object]:
    matrix = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    session_values = np.asarray(sessions, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != target.size or target.size != session_values.size:
        raise ValueError("Coefficient bootstrap inputs have incompatible shapes.")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match coefficient bootstrap matrix.")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive.")

    unique_sessions = np.unique(session_values)
    groups = [np.flatnonzero(session_values == session) for session in unique_sessions]
    rng = np.random.default_rng(seed)
    coefficients = np.empty((samples, matrix.shape[1]), dtype=np.float64)
    for sample_index in range(samples):
        if len(groups) >= 2:
            selected_groups = rng.integers(0, len(groups), size=len(groups))
            selected_rows = np.concatenate([groups[index] for index in selected_groups.tolist()])
        else:
            block = min(max(25, int(math.sqrt(target.size))), target.size)
            starts = rng.integers(0, max(target.size - block + 1, 1), size=max(target.size // block, 1))
            selected_rows = np.concatenate(
                [np.arange(start, min(start + block, target.size)) for start in starts.tolist()]
            )
        sample_coefficients, _intercept = fit_function(
            matrix[selected_rows], target[selected_rows], alpha
        )
        coefficients[sample_index] = sample_coefficients

    median = np.median(coefficients, axis=0)
    low = np.quantile(coefficients, 0.05, axis=0)
    high = np.quantile(coefficients, 0.95, axis=0)
    dominant_sign = np.sign(median)
    sign_consistency = np.mean(np.sign(coefficients) == dominant_sign, axis=0)
    rows = [
        {
            "feature": str(name),
            "median_standardized_coefficient": float(median[index]),
            "p05_standardized_coefficient": float(low[index]),
            "p95_standardized_coefficient": float(high[index]),
            "sign_consistency": float(sign_consistency[index]),
        }
        for index, name in enumerate(feature_names)
    ]
    rows.sort(
        key=lambda row: abs(float(row["median_standardized_coefficient"])),
        reverse=True,
    )
    return {
        "samples": samples,
        "resampling_unit": "session" if len(groups) >= 2 else "moving_event_block",
        "features": rows,
    }


def cost_stress_curve(
    gross_pnl_bps: np.ndarray,
    *,
    base_fee_bps_per_side: float,
    base_slippage_bps_per_side: float,
    extra_cost_bps_per_side: Sequence[float],
) -> list[dict[str, float | int]]:
    gross = np.asarray(gross_pnl_bps, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(gross)):
        raise ValueError("Gross PnL contains non-finite values.")
    rows: list[dict[str, float | int]] = []
    for extra in extra_cost_bps_per_side:
        extra_value = float(extra)
        if not math.isfinite(extra_value) or extra_value < 0.0:
            raise ValueError("Cost-stress values must be finite and non-negative.")
        per_side = base_fee_bps_per_side + base_slippage_bps_per_side + extra_value
        net = gross - 2.0 * per_side
        rows.append(
            {
                "extra_cost_bps_per_side": extra_value,
                "total_cost_bps_per_side": per_side,
                "trades": int(net.size),
                "mean_net_pnl_bps": float(np.mean(net)) if net.size else 0.0,
                "total_net_pnl_bps": float(np.sum(net)) if net.size else 0.0,
                "win_rate": float(np.mean(net > 0.0)) if net.size else 0.0,
            }
        )
    return rows



def _format_optional_number(value: object, format_spec: str) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return format(number, format_spec)

def render_research_card(report: dict[str, object]) -> str:
    methodology = report.get("methodology", {})
    samples = report.get("samples", {})
    model = report.get("selected_model", {})
    regression = report.get("test_regression", {})
    strategy = report.get("test_strategy", {})
    robustness = report.get("robustness", {})
    baseline = report.get("baselines", {})
    drift = report.get("feature_drift", {})
    provenance = report.get("provenance", {})

    lines = [
        "# Market Systems Evidence Card",
        "",
        "> Generated from the canonical training pipeline. Metrics are diagnostic evidence, "
        "not a claim of deployable profitability.",
        "",
        "## Research design",
        "",
        f"- Model: `{methodology.get('model', 'unknown')}`",
        f"- Target: `{methodology.get('target', 'unknown')}`",
        f"- Horizon: {methodology.get('horizon_events', '—')} events",
        f"- Split: `{methodology.get('split_mode', 'unknown')}`",
        f"- Purge: {methodology.get('purge_rows', '—')} rows",
        f"- Train/validation/test rows: {samples.get('train', '—')} / "
        f"{samples.get('validation', '—')} / {samples.get('test', '—')}",
        "- Normalization fitted only on training data; model and signal threshold selected "
        "only on validation data.",
        "",
        "## Selected model",
        "",
        f"- Ridge alpha: {model.get('ridge_alpha', '—')}",
        f"- Signal threshold: {_format_optional_number(model.get('signal_threshold_bps'), '.6f')} bps",
        "",
        "## Untouched holdout",
        "",
        f"- Pearson IC: {_format_optional_number(regression.get('pearson_ic'), '.6f')}",
        f"- Spearman rank IC: {_format_optional_number(regression.get('spearman_rank_ic'), '.6f')}",
        f"- Direction accuracy: {_format_optional_number(regression.get('direction_accuracy'), '.2%')}",
        "- Majority-direction baseline: "
        f"{_format_optional_number(regression.get('majority_direction_accuracy'), '.2%')}",
        "- Direction lift vs majority: "
        f"{_format_optional_number(regression.get('direction_accuracy_lift_vs_majority'), '+.2%')}",
        "- Balanced direction accuracy: "
        f"{_format_optional_number(regression.get('balanced_direction_accuracy'), '.2%')}",
        "- Target zero-return fraction: "
        f"{_format_optional_number(regression.get('target_zero_fraction'), '.2%')}",
        f"- Trades: {strategy.get('trades', 0)}",
        f"- Mean net PnL: {_format_optional_number(strategy.get('mean_pnl_bps'), '.6f')} bps/trade",
        f"- Total net PnL: {_format_optional_number(strategy.get('total_pnl_bps'), '.6f')} bps",
        f"- HAC t-statistic: {_format_optional_number(strategy.get('newey_west_pnl_t_statistic'), '.4f')}",
        "- Session-bootstrap P(mean ≤ 0): "
        f"{_format_optional_number(strategy.get('session_bootstrap_probability_mean_non_positive'), '.4f')}",
        "- Break-even additional cost: "
        f"{_format_optional_number(strategy.get('breakeven_additional_cost_bps_per_side'), '.6f')} bps/side",
        "",
        "## Robustness",
        "",
        "- Circular-shift IC p-value: "
        f"{_format_optional_number(robustness.get('test_ic_circular_shift', {}).get('two_sided_p_value'), '.4f')}",
        "- Test-session positive-IC fraction: "
        f"{_format_optional_number(robustness.get('test_session_regression', {}).get('positive_session_pearson_ic_fraction'), '.2%')}",
        f"- Maximum test PSI: {_format_optional_number(drift.get('maximum_test_psi'), '.4f')}",
        "",
        "## Baselines",
        "",
    ]
    if isinstance(baseline, dict):
        for name, values in baseline.items():
            if not isinstance(values, dict):
                continue
            test_values = values.get("test_strategy", {})
            lines.append(
                f"- `{name}`: {_format_optional_number(test_values.get('total_pnl_bps'), '.6f')} bps total, "
                f"{test_values.get('trades', 0)} trades"
            )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Feature schema: `{provenance.get('feature_schema_hash', 'unknown')}`",
            f"- Test-period fingerprint: `{provenance.get('test_period_fingerprint', 'unknown')}`",
            f"- Exact test-set fingerprint: `{provenance.get('test_set_fingerprint', 'unknown')}`",
            f"- Git commit: `{provenance.get('git_commit') or 'unavailable'}`",
            "",
            "## Scope limits",
            "",
        ]
    )
    limitations = report.get("limitations", [])
    if isinstance(limitations, list):
        lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines) + "\n"
