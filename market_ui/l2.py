from __future__ import annotations

import json
from pathlib import Path

from l2bin import read_metadata
from .paths import RECORDINGS_DIR

L2_RECORDINGS_DIR = RECORDINGS_DIR / "l2"


def discover_l2_recordings() -> list[Path]:
    L2_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        L2_RECORDINGS_DIR.glob("*.l2bin"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def summarize_l2_recording(path: str | Path) -> str:
    metadata = read_metadata(path, verify_hashes=False)
    payload = {
        "file": metadata.path.name,
        "symbol": metadata.symbol,
        "events": metadata.event_count,
        "snapshots": metadata.snapshot_count,
        "depth_events": metadata.depth_count,
        "trades": metadata.trade_count,
        "boundaries": metadata.boundary_count,
        "clean_shutdown": metadata.clean_shutdown,
        "data_complete": metadata.data_complete,
        "final_update_id": metadata.final_update_id,
        "final_state_hash": metadata.final_state_hash,
        "sequence_gaps": metadata.sequence_gaps,
        "snapshot_retries": metadata.snapshot_retries,
        "queue_drops": metadata.queue_drops,
        "malformed_messages": metadata.malformed_messages,
    }
    return json.dumps(payload, indent=2, sort_keys=True)
