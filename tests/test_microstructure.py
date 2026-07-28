
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microstructure import (
    AlphaModel,
    FEATURE_NAMES,
    OnlineFeatureBuilder,
    build_feature_set,
)
from conftest import synthetic_records


def test_offline_online_feature_parity() -> None:
    records = synthetic_records(500)
    offline = build_feature_set(records, volume_scale=1_000_000.0, horizon=10, session_id=7)
    builder = OnlineFeatureBuilder()
    online_by_index: dict[int, np.ndarray] = {}
    for index, row in enumerate(records):
        features = builder.update(
            int(row["timestamp_ns"]),
            float(row["best_bid"]),
            float(row["best_ask"]),
            int(row["bid_volume"]),
            int(row["ask_volume"]),
        )
        if features is not None:
            online_by_index[index] = features
    online = np.vstack([online_by_index[int(index)] for index in offline.event_index])
    np.testing.assert_allclose(online, offline.X, rtol=1e-12, atol=1e-10)


def test_online_builder_resets_after_gap() -> None:
    records = synthetic_records(80)
    builder = OnlineFeatureBuilder(max_gap_ns=1_000_000_000)
    for row in records[:40]:
        builder.update(
            int(row["timestamp_ns"]),
            float(row["best_bid"]),
            float(row["best_ask"]),
            int(row["bid_volume"]),
            int(row["ask_volume"]),
        )
    row = records[40].copy()
    row["timestamp_ns"] = int(records[39]["timestamp_ns"]) + 2_000_000_000
    assert builder.update(
        int(row["timestamp_ns"]),
        float(row["best_bid"]),
        float(row["best_ask"]),
        int(row["bid_volume"]),
        int(row["ask_volume"]),
    ) is None
    assert builder.reset_count == 1


def test_model_round_trip_and_schema_validation(tmp_path: Path) -> None:
    model = AlphaModel(
        feature_names=FEATURE_NAMES,
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.arange(len(FEATURE_NAMES), dtype=np.float64) / 100.0,
        intercept=0.5,
        horizon=25,
        signal_threshold_bps=0.2,
        ridge_alpha=0.01,
        fee_bps_per_side=0.05,
        provenance={"test": True},
    )
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = AlphaModel.load(path)
    np.testing.assert_array_equal(loaded.coefficients, model.coefficients)
    assert loaded.provenance == {"test": True}

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["feature_schema_hash"] = np.asarray("wrong")
    np.savez_compressed(path, **payload)
    with pytest.raises(RuntimeError, match="feature-schema hash"):
        AlphaModel.load(path)


def test_model_save_refuses_accidental_overwrite(tmp_path: Path) -> None:
    model = AlphaModel(
        feature_names=FEATURE_NAMES,
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.zeros(len(FEATURE_NAMES)),
        intercept=0.0,
        horizon=10,
        signal_threshold_bps=0.0,
        ridge_alpha=0.0,
        fee_bps_per_side=0.0,
    )
    path = tmp_path / "model.npz"
    model.save(path)
    with pytest.raises(FileExistsError):
        model.save(path)
    model.save(path, overwrite=True)


def test_model_rejects_invalid_metadata_and_nonfinite_features() -> None:
    base = dict(
        feature_names=FEATURE_NAMES,
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.zeros(len(FEATURE_NAMES)),
        intercept=0.0,
        horizon=10,
        signal_threshold_bps=0.0,
        ridge_alpha=0.0,
        fee_bps_per_side=0.0,
    )
    for field in ("ridge_alpha", "fee_bps_per_side", "signal_threshold_bps"):
        invalid = dict(base)
        invalid[field] = -0.01
        with pytest.raises(RuntimeError, match="invalid"):
            AlphaModel(**invalid)._validate()

    model = AlphaModel(**base)
    vector = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    vector[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        model.predict_one(vector)


def test_schema_two_names_only_hash_remains_loadable(tmp_path: Path) -> None:
    import hashlib

    model = AlphaModel(
        feature_names=FEATURE_NAMES,
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.ones(len(FEATURE_NAMES)),
        coefficients=np.zeros(len(FEATURE_NAMES)),
        intercept=0.0,
        horizon=10,
        signal_threshold_bps=0.0,
        ridge_alpha=0.0,
        fee_bps_per_side=0.0,
    )
    path = tmp_path / "legacy-schema-two.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["schema_version"] = np.asarray(2, dtype=np.int64)
    payload["feature_schema_version"] = np.asarray(1, dtype=np.int64)
    payload["feature_schema_hash"] = np.asarray(
        hashlib.sha256("\n".join(FEATURE_NAMES).encode("utf-8")).hexdigest()
    )
    np.savez_compressed(path, **payload)
    loaded = AlphaModel.load(path)
    assert loaded.schema_version == 2

