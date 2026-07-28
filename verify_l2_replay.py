from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from l2bin import Boundary, iter_events, read_checkpoints, read_metadata
from l2book import DepthUpdate, L2OrderBook, Snapshot, Trade


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify deterministic L2 reconstruction and checkpoint hashes."
    )
    parser.add_argument("file")
    parser.add_argument("--speeds", nargs="+", type=float, default=[0.0, 10.0, 1.0])
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--timed",
        action="store_true",
        help="Honor source timing. Without this flag, speeds are logical determinism trials.",
    )
    arguments = parser.parse_args()
    if any(not math.isfinite(speed) or speed < 0.0 for speed in arguments.speeds):
        parser.error("Replay speeds must be finite and non-negative.")
    return arguments


def replay(path: Path, speed: float, *, timed: bool) -> dict[str, object]:
    metadata = read_metadata(path, verify_hashes=True)
    checkpoints = (
        read_checkpoints(metadata.checkpoint_path, metadata.created_unix_ns)
        if metadata.checkpoint_path is not None
        else ()
    )
    checkpoint_by_event = {item.event_index: item for item in checkpoints}
    book = L2OrderBook()
    state_digest = hashlib.sha256()
    first_receipt: int | None = None
    replay_start = time.perf_counter_ns()
    event_count = depth_count = trade_count = checkpoint_matches = 0

    for event in iter_events(path):
        if timed and speed > 0.0:
            if first_receipt is None:
                first_receipt = event.receipt_timestamp_ns
            target = replay_start + int((event.receipt_timestamp_ns - first_receipt) / speed)
            while time.perf_counter_ns() < target:
                remaining = target - time.perf_counter_ns()
                if remaining > 2_000_000:
                    time.sleep(min(remaining / 2e9, 0.005))
                else:
                    time.sleep(0)
        event_count += 1
        if isinstance(event, Boundary):
            book.clear()
        elif isinstance(event, Snapshot):
            book.install_snapshot(event)
        elif isinstance(event, DepthUpdate):
            if event.final_update_id <= book.last_update_id:
                continue
            if event.first_update_id > book.last_update_id + 1:
                raise RuntimeError("Replay encountered an unmarked depth sequence gap.")
            book.apply(event)
            depth_count += 1
            state_digest.update(book.state_hash().to_bytes(8, "little"))
        elif isinstance(event, Trade):
            trade_count += 1
        checkpoint = checkpoint_by_event.get(event_count)
        if checkpoint is not None:
            if (
                checkpoint.update_id != book.last_update_id
                or checkpoint.state_hash != book.state_hash()
            ):
                raise RuntimeError(
                    f"Checkpoint mismatch after event {event_count:,}: "
                    f"expected update={checkpoint.update_id}, hash={checkpoint.state_hash}; "
                    f"received update={book.last_update_id}, hash={book.state_hash()}."
                )
            checkpoint_matches += 1

    if metadata.final_update_id is not None and book.last_update_id != metadata.final_update_id:
        raise RuntimeError("Final reconstructed update ID does not match metadata.")
    if metadata.final_state_hash is not None and book.state_hash() != metadata.final_state_hash:
        raise RuntimeError("Final reconstructed L2 hash does not match metadata.")
    return {
        "speed": speed,
        "events": event_count,
        "depth_events": depth_count,
        "trades": trade_count,
        "checkpoints_verified": checkpoint_matches,
        "final_update_id": book.last_update_id,
        "final_state_hash": book.state_hash(),
        "state_sequence_sha256": state_digest.hexdigest(),
    }


def main() -> None:
    arguments = parse_arguments()
    path = Path(arguments.file).expanduser().resolve()
    metadata = read_metadata(path, verify_hashes=True)
    if metadata.data_complete is False and not arguments.allow_incomplete:
        raise RuntimeError("Refusing to verify a recording marked incomplete.")
    trials = [replay(path, speed, timed=arguments.timed) for speed in arguments.speeds]
    canonical = trials[0]
    for trial in trials[1:]:
        for key in ("events", "depth_events", "trades", "final_update_id", "final_state_hash", "state_sequence_sha256"):
            if trial[key] != canonical[key]:
                raise RuntimeError(f"Replay determinism failed for field {key!r}.")
    output = {
        "recording": str(path),
        "symbol": metadata.symbol,
        "data_complete": metadata.data_complete,
        "deterministic": True,
        "trials": trials,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
