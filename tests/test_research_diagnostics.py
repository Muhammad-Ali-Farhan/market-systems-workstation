from __future__ import annotations

import numpy as np
import pytest

from conftest import synthetic_records
from microstructure import build_feature_set
from research import ExecutionAssumptions, strategy_metrics, strategy_trades
from research_diagnostics import (
    circular_shift_permutation_test,
    cost_stress_curve,
    directional_classification_metrics,
    newey_west_mean_t_statistic,
    prediction_quantile_table,
    session_bootstrap_summary,
    spearman_correlation,
)


def test_execution_costs_and_displayed_liquidity_filter_are_explicit() -> None:
    data = build_feature_set(
        synthetic_records(180, seed=901),
        volume_scale=1_000_000.0,
        horizon=10,
        session_id=0,
    )
    indices = np.arange(data.size, dtype=np.int64)
    prediction = np.full(data.size, 100.0)
    unconstrained = strategy_trades(
        prediction,
        data,
        indices,
        threshold=1.0,
        execution=ExecutionAssumptions(0.1, slippage_bps_per_side=0.2),
    )
    impossible_size = float(np.max(data.current_ask_quantity)) * 10.0
    constrained = strategy_trades(
        prediction,
        data,
        indices,
        threshold=1.0,
        execution=ExecutionAssumptions(
            0.1,
            slippage_bps_per_side=0.2,
            trade_size_base=impossible_size,
            max_displayed_participation=0.5,
        ),
    )
    assert unconstrained.net_pnl_bps.size > 0
    np.testing.assert_allclose(
        unconstrained.gross_pnl_bps - unconstrained.net_pnl_bps,
        0.6,
    )
    assert constrained.net_pnl_bps.size == 0
    assert constrained.fill_rejections > 0


def test_robust_diagnostics_are_deterministic_and_cost_stress_is_monotone() -> None:
    values = np.asarray([0.4, -0.1, 0.2, 0.5, -0.2, 0.3], dtype=np.float64)
    sessions = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)
    first = session_bootstrap_summary(values, sessions, samples=200, seed=7)
    second = session_bootstrap_summary(values, sessions, samples=200, seed=7)
    assert first == second
    assert np.isfinite(newey_west_mean_t_statistic(values))

    curve = cost_stress_curve(
        values,
        base_fee_bps_per_side=0.01,
        base_slippage_bps_per_side=0.02,
        extra_cost_bps_per_side=(0.0, 0.1, 0.5),
    )
    totals = [float(row["total_net_pnl_bps"]) for row in curve]
    assert totals[0] > totals[1] > totals[2]


def test_rank_and_circular_shift_inference_behave_sensibly() -> None:
    prediction = np.arange(1.0, 21.0)
    target = prediction * 2.0
    sessions = np.repeat(np.arange(4), 5)
    assert spearman_correlation(prediction, target) == 1.0
    result = circular_shift_permutation_test(
        prediction,
        target,
        sessions,
        samples=200,
        seed=3,
    )
    assert result.observed > 0.99
    assert 0.0 < result.two_sided_p_value <= 1.0
    quantiles = prediction_quantile_table(prediction, target, bins=5)
    assert len(quantiles["bins"]) == 5
    assert float(quantiles["mean_target_monotonicity"]) > 0.99


def test_strategy_metrics_report_hac_tail_and_break_even() -> None:
    pnl = np.asarray([0.2, -0.1, 0.4, -0.2, 0.3, 0.1], dtype=np.float64)
    gross = pnl + 0.1
    sessions = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)
    metrics = strategy_metrics(
        pnl,
        sessions,
        gross_pnl=gross,
        round_trip_cost_bps=0.1,
        bootstrap_samples=200,
    )
    assert metrics.trades == pnl.size
    assert np.isfinite(metrics.newey_west_pnl_t_statistic)
    assert metrics.expected_shortfall_5pct_bps <= 0.0
    assert metrics.breakeven_additional_cost_bps_per_side > 0.0


def test_directional_metrics_expose_majority_baseline_and_zero_targets() -> None:
    prediction = np.asarray([1.0, 1.0, -1.0, 0.0, 1.0, -1.0])
    target = np.asarray([1.0, -1.0, -1.0, 1.0, 0.0, -1.0])
    metrics = directional_classification_metrics(prediction, target)
    assert metrics["nonzero_target_samples"] == 5
    assert metrics["target_zero_fraction"] == pytest.approx(1.0 / 6.0)
    assert metrics["majority_direction_accuracy"] == pytest.approx(3.0 / 5.0)
    assert metrics["direction_accuracy"] == pytest.approx(3.0 / 5.0)
    assert metrics["direction_accuracy_lift_vs_majority"] == 0.0
    assert metrics["balanced_direction_accuracy"] == pytest.approx(7.0 / 12.0)
    assert metrics["actionable_coverage"] == pytest.approx(4.0 / 5.0)
    assert metrics["actionable_direction_accuracy"] == pytest.approx(3.0 / 4.0)
    assert metrics["abstentions"] == 1
