
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from qbin import (
    RecordingMetadata,
    open_records,
    open_update_ids,
    read_metadata,
    validate_records,
    validate_update_ids,
)
from .paths import RECORDINGS_DIR


@dataclass(frozen=True)
class RecordingSummary:
    path: Path
    file_size: int
    record_count: int
    duration_seconds: float
    average_rate: float
    created_at: datetime
    volume_scale: int
    first_bid: float
    first_ask: float
    last_bid: float
    last_ask: float
    minimum_bid: float
    maximum_ask: float
    clean_shutdown: bool | None
    data_complete: bool | None
    reconnect_count: int | None
    recording_dropped: int | None


class RecordingFormatError(RuntimeError):
    pass


def discover_recordings() -> list[Path]:
    return sorted(
        RECORDINGS_DIR.glob("*.qbin"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def map_records(path: str | Path) -> tuple[RecordingMetadata, np.ndarray]:
    try:
        metadata = read_metadata(path)
        records = open_records(path, metadata)
        validate_records(records, context=str(path))
        validate_update_ids(open_update_ids(metadata), metadata, context=str(path))
        return metadata, records
    except (OSError, RuntimeError, ValueError) as exception:
        raise RecordingFormatError(str(exception)) from exception


def summarize_recording(path: str | Path) -> RecordingSummary:
    recording_path = Path(path)
    metadata, records = map_records(recording_path)
    count = len(records)
    created_at = datetime.fromtimestamp(
        metadata.created_unix_ns / 1_000_000_000,
        tz=timezone.utc,
    )

    if count == 0:
        nan = float("nan")
        return RecordingSummary(
            path=recording_path,
            file_size=metadata.file_size,
            record_count=0,
            duration_seconds=0.0,
            average_rate=0.0,
            created_at=created_at,
            volume_scale=metadata.volume_scale,
            first_bid=nan,
            first_ask=nan,
            last_bid=nan,
            last_ask=nan,
            minimum_bid=nan,
            maximum_ask=nan,
            clean_shutdown=metadata.clean_shutdown,
            data_complete=metadata.data_complete,
            reconnect_count=metadata.reconnect_count,
            recording_dropped=metadata.recording_dropped,
        )

    timestamps = np.asarray(records["timestamp_ns"], dtype=np.uint64)
    duration = max(0.0, (int(timestamps[-1]) - int(timestamps[0])) / 1_000_000_000.0)
    average_rate = (count - 1) / duration if duration > 0.0 and count > 1 else 0.0
    bid = np.asarray(records["best_bid"], dtype=np.float64)
    ask = np.asarray(records["best_ask"], dtype=np.float64)

    return RecordingSummary(
        path=recording_path,
        file_size=metadata.file_size,
        record_count=count,
        duration_seconds=duration,
        average_rate=average_rate,
        created_at=created_at,
        volume_scale=metadata.volume_scale,
        first_bid=float(bid[0]),
        first_ask=float(ask[0]),
        last_bid=float(bid[-1]),
        last_ask=float(ask[-1]),
        minimum_bid=float(np.min(bid)),
        maximum_ask=float(np.max(ask)),
        clean_shutdown=metadata.clean_shutdown,
        data_complete=metadata.data_complete,
        reconnect_count=metadata.reconnect_count,
        recording_dropped=metadata.recording_dropped,
    )


def downsample(
    values: Iterable[float] | np.ndarray,
    maximum_points: int = 2_000,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= maximum_points:
        return array
    indices = np.linspace(0, len(array) - 1, maximum_points, dtype=np.int64)
    return array[indices]


def recording_chart_series(
    path: str | Path,
    maximum_points: int = 2_000,
) -> dict[str, np.ndarray]:
    _metadata, records = map_records(path)
    if len(records) == 0:
        empty = np.array([], dtype=np.float64)
        return {"mid": empty, "obi": empty, "spread_bps": empty}

    bid = np.asarray(records["best_bid"], dtype=np.float64)
    ask = np.asarray(records["best_ask"], dtype=np.float64)
    bid_volume = np.asarray(records["bid_volume"], dtype=np.float64)
    ask_volume = np.asarray(records["ask_volume"], dtype=np.float64)
    mid = (bid + ask) / 2.0
    denominator = bid_volume + ask_volume
    obi = np.divide(
        bid_volume - ask_volume,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0.0,
    )
    spread_bps = np.divide(
        ask - bid,
        mid,
        out=np.zeros_like(mid),
        where=mid != 0.0,
    ) * 10_000.0
    return {
        "mid": downsample(mid, maximum_points),
        "obi": downsample(obi, maximum_points),
        "spread_bps": downsample(spread_bps, maximum_points),
    }

