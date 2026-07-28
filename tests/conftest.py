
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from qbin import (
    EXPECTED_MAGIC,
    EXPECTED_VERSION,
    EXPECTED_VOLUME_SCALE,
    HEADER_SIZE,
    RECORD_DTYPE,
    UPDATE_ID_HEADER_STRUCT,
    UPDATE_ID_MAGIC,
    UPDATE_ID_VERSION,
)

HEADER = struct.Struct("<8sIIIIQQQQQ")


def write_qbin(
    path: Path,
    records: np.ndarray,
    *,
    created_unix_ns: int = 1_700_000_000_000_000_000,
    data_complete: bool = True,
    boundaries: list[dict[str, object]] | None = None,
    with_update_ids: bool = True,
) -> Path:
    records = np.asarray(records, dtype=RECORD_DTYPE)
    header = HEADER.pack(
        EXPECTED_MAGIC,
        EXPECTED_VERSION,
        HEADER_SIZE,
        RECORD_DTYPE.itemsize,
        0,
        EXPECTED_VOLUME_SCALE,
        created_unix_ns,
        0,
        0,
        0,
    )
    path.write_bytes(header + records.tobytes())
    sidecar = {
        "schema_version": 1,
        "binary_version": EXPECTED_VERSION,
        "record_size": RECORD_DTYPE.itemsize,
        "volume_scale": EXPECTED_VOLUME_SCALE,
        "source": "binance_spot",
        "symbol": "BTCUSDT",
        "stream": "bookTicker",
        "recording_file": path.name,
        "created_unix_ns": created_unix_ns,
        "clean_shutdown": True,
        "data_complete": data_complete,
        "accepted_records": len(records),
        "recorded_records": len(records),
        "recording_dropped": 0,
        "recording_write_errors": 0,
        "consumer_queue_dropped": 0,
        "malformed_messages": 0,
        "reconnect_count": 0,
        "boundaries": boundaries or [{"record_index": 0, "kind": "connection_start"}],
    }
    if with_update_ids:
        update_ids = np.arange(1, len(records) + 1, dtype="<u8")
        update_path = Path(f"{path}.qids")
        update_header = UPDATE_ID_HEADER_STRUCT.pack(
            UPDATE_ID_MAGIC,
            UPDATE_ID_VERSION,
            UPDATE_ID_HEADER_STRUCT.size,
            8,
            0,
            created_unix_ns,
        )
        update_path.write_bytes(update_header + update_ids.tobytes())
        sidecar.update(
            {
                "update_id_file": update_path.name,
                "update_id_version": UPDATE_ID_VERSION,
                "recorded_update_ids": len(records),
                "first_update_id": int(update_ids[0]) if len(records) else 0,
                "last_update_id": int(update_ids[-1]) if len(records) else 0,
            }
        )
    Path(f"{path}.meta.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path


def synthetic_records(
    count: int,
    *,
    start_timestamp_ns: int = 1_000_000_000,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    records = np.empty(count, dtype=RECORD_DTYPE)
    intervals = rng.integers(100_000, 3_000_000, size=count, dtype=np.uint64)
    records["timestamp_ns"] = start_timestamp_ns + np.cumsum(intervals, dtype=np.uint64)
    innovations = rng.normal(0.0, 0.01, size=count)
    mid = 100.0 + np.cumsum(innovations)
    spread = rng.choice([0.01, 0.02, 0.03], size=count)
    records["best_bid"] = mid - spread / 2.0
    records["best_ask"] = mid + spread / 2.0
    records["bid_volume"] = rng.integers(100_000, 3_000_000, size=count, dtype=np.uint32)
    records["ask_volume"] = rng.integers(100_000, 3_000_000, size=count, dtype=np.uint32)
    return records

