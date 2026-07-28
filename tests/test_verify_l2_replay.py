from __future__ import annotations

from pathlib import Path

import pytest

from l2bin import Boundary, BoundaryReason, L2Writer
from l2book import DepthUpdate, L2OrderBook, Level, Snapshot
from verify_l2_replay import replay


def make_recording(path: Path) -> Path:
    snapshot = Snapshot(1, 10, (Level(100, 2),), (Level(101, 3),))
    update = DepthUpdate(2, 2, 11, 11, (Level(100, 4),), ())
    book = L2OrderBook()
    writer = L2Writer(path, "BTCUSDT")
    writer.write(Boundary(1, BoundaryReason.CONNECTION_START))
    writer.write(snapshot)
    book.install_snapshot(snapshot)
    writer.write(update)
    book.apply(update)
    writer.write_checkpoint(book.last_update_id, book.state_hash())
    writer.finalize(final_update_id=book.last_update_id, final_state_hash=book.state_hash())
    return path


def test_replay_is_identical_across_logical_speeds(tmp_path: Path) -> None:
    path = make_recording(tmp_path / "replay.l2bin")
    first = replay(path, 0.0, timed=False)
    second = replay(path, 100.0, timed=False)
    assert first["final_state_hash"] == second["final_state_hash"]
    assert first["state_sequence_sha256"] == second["state_sequence_sha256"]
    assert first["checkpoints_verified"] == 1


def test_checkpoint_corruption_is_rejected(tmp_path: Path) -> None:
    path = make_recording(tmp_path / "bad_checkpoint.l2bin")
    checkpoint = Path(f"{path}.l2chk")
    raw = bytearray(checkpoint.read_bytes())
    raw[-1] ^= 0x01
    checkpoint.write_bytes(raw)
    with pytest.raises(RuntimeError, match="checkpoint|SHA-256"):
        replay(path, 0.0, timed=False)
