
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from microstructure import AlphaModel, OnlineFeatureBuilder as CanonicalOnlineFeatureBuilder
from research import train_and_evaluate


@dataclass(frozen=True)
class TrainedModel:
    model: AlphaModel

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.model.feature_names

    @property
    def mean(self) -> np.ndarray:
        return self.model.mean

    @property
    def scale(self) -> np.ndarray:
        return self.model.scale

    @property
    def coefficients(self) -> np.ndarray:
        return self.model.coefficients

    @property
    def intercept(self) -> float:
        return self.model.intercept

    @property
    def horizon(self) -> int:
        return self.model.horizon

    @property
    def threshold(self) -> float:
        return self.model.signal_threshold_bps

    @property
    def fee_bps(self) -> float:
        return self.model.fee_bps_per_side

    def predict(self, features: np.ndarray) -> np.ndarray | float:
        return self.model.predict(features)


def load_model(path: str | Path) -> TrainedModel:
    return TrainedModel(AlphaModel.load(path))


class OnlineFeatureBuilder:
    def __init__(self, volume_scale: float = 1_000_000.0) -> None:
        self._builder = CanonicalOnlineFeatureBuilder(volume_scale)

    def reset(self) -> None:
        self._builder.reset()

    def update(self, row: np.void) -> np.ndarray | None:
        return self._builder.update(
            timestamp_ns=int(row["timestamp_ns"]),
            best_bid=float(row["best_bid"]),
            best_ask=float(row["best_ask"]),
            bid_volume=int(row["bid_volume"]),
            ask_volume=int(row["ask_volume"]),
        )


def train_model(
    recording_paths: list[Path],
    horizon: int,
    fee_bps: float,
    model_path: Path,
    report_path: Path,
    predictions_path: Path,
    progress: Callable[[int, str], None] | None = None,
    *,
    slippage_bps: float = 0.0,
    trade_size_base: float = 0.0,
    max_displayed_participation: float = 1.0,
    evidence_path: Path | None = None,
) -> dict[str, object]:
    return train_and_evaluate(
        recording_paths,
        horizon=horizon,
        fee_bps_per_side=fee_bps,
        slippage_bps_per_side=slippage_bps,
        trade_size_base=trade_size_base,
        max_displayed_participation=max_displayed_participation,
        model_path=model_path,
        report_path=report_path,
        predictions_path=predictions_path,
        evidence_path=evidence_path,
        progress=progress,
    )

