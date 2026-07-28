
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from microstructure import build_feature_set, concatenate_feature_sets
from research import make_chronological_split, strategy_metrics, strategy_pnls, train_and_evaluate
from conftest import synthetic_records, write_qbin
from qbin import sha256_file


def test_whole_session_split_is_globally_chronological() -> None:
    feature_sets = [
        build_feature_set(synthetic_records(450, seed=seed), volume_scale=1_000_000.0, horizon=10, session_id=seed)
        for seed in (1, 2, 3, 4, 5)
    ]
    data = concatenate_feature_sets(feature_sets)
    split = make_chronological_split(data, purge_size=25)
    assert split.mode == "whole_session"
    assert max(split.train_sessions) < min(split.validation_sessions)
    assert max(split.validation_sessions) < min(split.test_sessions)
    assert int(np.max(split.train)) < int(np.min(split.validation))
    assert int(np.max(split.validation)) < int(np.min(split.test))


def test_strategy_cooldown_resets_per_session() -> None:
    first = build_feature_set(synthetic_records(120, seed=10), volume_scale=1_000_000.0, horizon=10, session_id=0)
    second = build_feature_set(synthetic_records(120, seed=11), volume_scale=1_000_000.0, horizon=10, session_id=1)
    data = concatenate_feature_sets([first, second])
    indices = np.arange(data.size, dtype=np.int64)
    predictions = np.full(data.size, 100.0)
    pnl, selected, sessions = strategy_pnls(
        predictions,
        data,
        indices,
        threshold=1.0,
        fee_bps_per_side=0.0,
    )
    assert selected.size == pnl.size == sessions.size
    assert set(sessions.tolist()) == {0, 1}
    metrics = strategy_metrics(pnl, sessions)
    assert metrics.max_drawdown_bps >= 0.0


def test_end_to_end_training_artifacts_are_compatible(tmp_path: Path) -> None:
    recordings = []
    for index in range(4):
        path = write_qbin(
            tmp_path / f"session-{index}.qbin",
            synthetic_records(
                650,
                seed=100 + index,
                start_timestamp_ns=1_000_000_000 + index * 100_000_000_000,
            ),
            created_unix_ns=1_700_000_000_000_000_000 + index,
        )
        recordings.append(path)

    model_path = tmp_path / "alpha_model.npz"
    report_path = tmp_path / "report.json"
    predictions_path = tmp_path / "predictions.csv"
    evidence_path = tmp_path / "research_card.md"
    report = train_and_evaluate(
        recordings,
        horizon=10,
        fee_bps_per_side=0.01,
        model_path=model_path,
        report_path=report_path,
        predictions_path=predictions_path,
        evidence_path=evidence_path,
        diagnostic_resamples=100,
    )
    assert report["schema_version"] == 4
    assert report["methodology"]["split_mode"] == "whole_session"
    assert model_path.exists() and report_path.exists() and predictions_path.exists()
    assert evidence_path.exists()
    assert report["test_strategy"]["max_drawdown_bps"] >= 0.0
    assert "newey_west_pnl_t_statistic" in report["test_strategy"]
    assert "feature_drift" in report and "robustness" in report
    assert "order_book_imbalance" in report["baselines"]
    report_text = report_path.read_text(encoding="utf-8")
    assert "NaN" not in report_text
    saved = json.loads(report_text)
    assert all("file" in item and "path" not in item for item in saved["recordings"])
    assert "working_directory" not in saved["provenance"]
    assert "python_executable" not in saved["provenance"]
    artifacts = saved["artifacts"]
    assert artifacts["model"]["sha256"] == sha256_file(model_path)
    assert artifacts["predictions"]["sha256"] == sha256_file(predictions_path)
    assert artifacts["evidence_card"]["sha256"] == sha256_file(evidence_path)
    assert not list(tmp_path.glob(".*.tmp*"))


def test_zero_forecast_remains_flat_at_zero_threshold() -> None:
    data = build_feature_set(
        synthetic_records(120, seed=20),
        volume_scale=1_000_000.0,
        horizon=10,
        session_id=0,
    )
    indices = np.arange(data.size, dtype=np.int64)
    pnl, selected, sessions = strategy_pnls(
        np.zeros(data.size, dtype=np.float64),
        data,
        indices,
        threshold=0.0,
        fee_bps_per_side=0.0,
    )
    assert pnl.size == selected.size == sessions.size == 0


def test_prediction_export_keeps_zero_forecast_flat(tmp_path: Path) -> None:
    from research import save_predictions

    data = build_feature_set(
        synthetic_records(120, seed=21),
        volume_scale=1_000_000.0,
        horizon=10,
        session_id=0,
    )
    output = tmp_path / "predictions.csv"
    indices = np.arange(data.size, dtype=np.int64)
    save_predictions(
        output,
        data,
        indices,
        np.zeros(data.size, dtype=np.float64),
        threshold=0.0,
    )
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines
    assert all(line.endswith(",FLAT") for line in lines[1:])


def test_repeated_holdout_requires_explicit_override(tmp_path: Path) -> None:
    recordings = []
    for index in range(4):
        recordings.append(
            write_qbin(
                tmp_path / f"holdout-session-{index}.qbin",
                synthetic_records(
                    650,
                    seed=300 + index,
                    start_timestamp_ns=1_000_000_000 + index * 100_000_000_000,
                ),
                created_unix_ns=1_800_000_000_000_000_000 + index,
            )
        )

    train_and_evaluate(
        recordings,
        horizon=10,
        fee_bps_per_side=0.01,
        model_path=tmp_path / "first.npz",
        report_path=tmp_path / "first.json",
        predictions_path=tmp_path / "first.csv",
    )

    import pytest

    with pytest.raises(RuntimeError, match="held-out market period has already been evaluated"):
        train_and_evaluate(
            recordings,
            horizon=10,
            fee_bps_per_side=0.01,
            model_path=tmp_path / "second.npz",
            report_path=tmp_path / "second.json",
            predictions_path=tmp_path / "second.csv",
        )

    report = train_and_evaluate(
        recordings,
        horizon=10,
        fee_bps_per_side=0.01,
        model_path=tmp_path / "rerun.npz",
        report_path=tmp_path / "rerun.json",
        predictions_path=tmp_path / "rerun.csv",
        allow_test_reuse=True,
    )
    assert report["methodology"]["prior_test_evaluations"] == 1


def test_holdout_reuse_detected_when_earlier_training_session_is_added(
    tmp_path: Path,
) -> None:
    recordings = []
    for index in range(4):
        recordings.append(
            write_qbin(
                tmp_path / f"stable-holdout-{index}.qbin",
                synthetic_records(
                    650,
                    seed=500 + index,
                    start_timestamp_ns=10_000_000_000 + index * 100_000_000_000,
                ),
                created_unix_ns=1_900_000_000_000_000_000 + index,
            )
        )

    first = train_and_evaluate(
        recordings,
        horizon=10,
        fee_bps_per_side=0.01,
        model_path=tmp_path / "baseline.npz",
        report_path=tmp_path / "baseline.json",
        predictions_path=tmp_path / "baseline.csv",
    )

    earlier = write_qbin(
        tmp_path / "new-earlier-training.qbin",
        synthetic_records(
            650,
            seed=499,
            start_timestamp_ns=1_000_000_000,
        ),
        created_unix_ns=1_899_999_999_999_999_999,
    )

    import pytest

    with pytest.raises(
        RuntimeError,
        match="held-out market period has already been evaluated",
    ):
        train_and_evaluate(
            [earlier, *recordings],
            horizon=10,
            fee_bps_per_side=0.01,
            model_path=tmp_path / "shifted.npz",
            report_path=tmp_path / "shifted.json",
            predictions_path=tmp_path / "shifted.csv",
        )

    assert first["provenance"]["test_period_fingerprint"]

