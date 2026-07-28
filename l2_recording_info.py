from __future__ import annotations

import argparse
import json

from l2bin import read_checkpoints, read_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate an L2 recording.")
    parser.add_argument("file")
    parser.add_argument("--verify-hashes", action="store_true")
    arguments = parser.parse_args()
    metadata = read_metadata(arguments.file, verify_hashes=arguments.verify_hashes)
    checkpoints = (
        read_checkpoints(metadata.checkpoint_path, metadata.created_unix_ns)
        if metadata.checkpoint_path is not None
        else ()
    )
    payload = {
        "path": str(metadata.path),
        "symbol": metadata.symbol,
        "created_unix_ns": metadata.created_unix_ns,
        "events": metadata.event_count,
        "snapshots": metadata.snapshot_count,
        "depth_events": metadata.depth_count,
        "trades": metadata.trade_count,
        "boundaries": metadata.boundary_count,
        "checkpoints": len(checkpoints),
        "clean_shutdown": metadata.clean_shutdown,
        "data_complete": metadata.data_complete,
        "final_update_id": metadata.final_update_id,
        "final_state_hash": metadata.final_state_hash,
        "sequence_gaps": metadata.sequence_gaps,
        "snapshot_retries": metadata.snapshot_retries,
        "queue_drops": metadata.queue_drops,
        "malformed_messages": metadata.malformed_messages,
        "sha256": metadata.sha256,
        "checkpoint_sha256": metadata.checkpoint_sha256,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
