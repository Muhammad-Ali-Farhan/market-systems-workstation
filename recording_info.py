
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from microstructure import DEFAULT_MAX_GAP_NS
from qbin import (
    contiguous_slices,
    open_records,
    open_update_ids,
    read_metadata,
    sha256_file,
    validate_records,
    validate_update_ids,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and validate a market-data binary market-data recording."
    )
    parser.add_argument("file", help="Path to the .qbin recording.")
    parser.add_argument(
        "--sha256",
        action="store_true",
        help="Calculate and print the recording SHA-256 digest.",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=DEFAULT_MAX_GAP_NS / 1_000_000_000.0,
        help="Gap threshold used to count contiguous research segments.",
    )
    arguments = parser.parse_args()
    if not np.isfinite(arguments.max_gap_seconds) or arguments.max_gap_seconds <= 0.0:
        parser.error("--max-gap-seconds must be finite and positive.")
    return arguments


def optional(value: object) -> str:
    return "unknown (legacy recording without sidecar)" if value is None else str(value)


def main() -> None:
    arguments = parse_arguments()
    path = Path(arguments.file)
    metadata = read_metadata(path)
    records = open_records(path, metadata)
    validate_records(records, context=str(path))
    update_ids = open_update_ids(metadata)
    validate_update_ids(update_ids, metadata, context=str(path))
    segments = contiguous_slices(
        records,
        metadata,
        max_gap_ns=int(arguments.max_gap_seconds * 1_000_000_000.0),
    )

    created_at = datetime.fromtimestamp(
        metadata.created_unix_ns / 1_000_000_000,
        tz=timezone.utc,
    )
    print(f"File: {path.resolve()}")
    print(f"Size: {metadata.file_size:,} bytes")
    print(f"Format version: {metadata.version}")
    print(f"Created: {created_at.isoformat()}")
    print(f"Volume scale: {metadata.volume_scale:,}")
    print(f"Records: {metadata.record_count:,}")
    print(f"Sidecar: {metadata.sidecar_path or 'not present'}")
    print(f"Update-ID file: {metadata.update_id_path or 'not present (legacy recording)'}")
    print(f"Update IDs: {optional(metadata.update_id_count)}")
    print(f"First update ID: {optional(metadata.first_update_id)}")
    print(f"Last update ID: {optional(metadata.last_update_id)}")
    print(f"Clean shutdown: {optional(metadata.clean_shutdown)}")
    print(f"Data complete: {optional(metadata.data_complete)}")
    print(f"Accepted records: {optional(metadata.accepted_records)}")
    print(f"Recorder drops: {optional(metadata.recording_dropped)}")
    print(f"Write errors: {optional(metadata.recording_write_errors)}")
    print(f"Consumer-queue drops: {optional(metadata.consumer_queue_dropped)}")
    print(f"Malformed messages: {optional(metadata.malformed_messages)}")
    print(f"Reconnects: {optional(metadata.reconnect_count)}")
    print(f"Research-contiguous segments: {len(segments):,}")
    if metadata.boundaries:
        print("Boundaries:")
        for boundary in metadata.boundaries:
            print(f"  - record {boundary.record_index:,}: {boundary.kind}")

    if arguments.sha256:
        print(f"SHA-256: {sha256_file(path)}")

    if metadata.record_count == 0:
        print("The recording contains no market updates.")
        return

    timestamps = np.asarray(records["timestamp_ns"], dtype=np.uint64)
    bid = np.asarray(records["best_bid"], dtype=np.float64)
    ask = np.asarray(records["best_ask"], dtype=np.float64)
    duration_seconds = max(
        0.0,
        (int(timestamps[-1]) - int(timestamps[0])) / 1_000_000_000.0,
    )
    average_rate = (
        (metadata.record_count - 1) / duration_seconds
        if duration_seconds > 0.0 and metadata.record_count > 1
        else 0.0
    )
    print(f"Duration: {duration_seconds:.6f} seconds")
    print(f"Average source rate: {average_rate:,.2f} records/sec")
    print(f"First market: bid={bid[0]:.8f}, ask={ask[0]:.8f}")
    print(f"Last market: bid={bid[-1]:.8f}, ask={ask[-1]:.8f}")
    print(f"Observed price range: {np.min(bid):.8f} to {np.max(ask):.8f}")


if __name__ == "__main__":
    main()

