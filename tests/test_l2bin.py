from __future__ import annotations

from pathlib import Path

import pytest

from l2bin import (
    Boundary,
    BoundaryReason,
    L2Writer,
    iter_events,
    read_checkpoints,
    read_metadata,
)
from l2book import DepthUpdate, L2OrderBook, Level, Snapshot, Trade


def create_recording(path: Path) -> Path:
    book = L2OrderBook()
    snapshot = Snapshot(
        1,
        100,
        (Level(10_000, 500), Level(9_900, 700)),
        (Level(10_100, 600), Level(10_200, 800)),
    )
    update = DepthUpdate(
        2,
        2,
        101,
        101,
        (Level(10_000, 450),),
        (Level(10_100, 550),),
    )
    writer = L2Writer(path, "BTCUSDT", checkpoint_interval=1)
    writer.write(Boundary(1, BoundaryReason.CONNECTION_START))
    writer.write(snapshot)
    book.install_snapshot(snapshot)
    writer.write(update)
    book.apply(update)
    writer.write(Trade(3, 3, 1, 10_100, 25, False))
    writer.write_checkpoint(book.last_update_id, book.state_hash())
    writer.finalize(
        final_update_id=book.last_update_id,
        final_state_hash=book.state_hash(),
    )
    return path


def test_round_trip_and_hash_verification(tmp_path: Path) -> None:
    path = create_recording(tmp_path / "sample.l2bin")
    events = tuple(iter_events(path))
    assert len(events) == 4
    assert isinstance(events[1], Snapshot)
    assert isinstance(events[2], DepthUpdate)
    assert isinstance(events[3], Trade)
    metadata = read_metadata(path, verify_hashes=True)
    assert metadata.data_complete is True
    assert metadata.depth_count == 1
    checkpoints = read_checkpoints(metadata.checkpoint_path, metadata.created_unix_ns)
    assert len(checkpoints) == 1
    assert checkpoints[0].update_id == 101


def test_payload_corruption_is_rejected(tmp_path: Path) -> None:
    path = create_recording(tmp_path / "corrupt.l2bin")
    data = bytearray(path.read_bytes())
    # Header + boundary + snapshot header/payload + depth header.
    payload_offset = 128 + 80 + 80 + 4 * 16 + 80
    data[payload_offset] ^= 0xFF
    path.write_bytes(data)
    with pytest.raises(RuntimeError, match="CRC"):
        tuple(iter_events(path))


def test_truncation_is_rejected(tmp_path: Path) -> None:
    path = create_recording(tmp_path / "truncated.l2bin")
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(RuntimeError, match="truncated"):
        tuple(iter_events(path))


def test_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = create_recording(tmp_path / "exists.l2bin")
    with pytest.raises(FileExistsError):
        L2Writer(path, "BTCUSDT")


def test_interrupted_current_recording_is_explicitly_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.l2bin"
    writer = L2Writer(path, "BTCUSDT")
    writer.write(Boundary(1, BoundaryReason.CONNECTION_START))
    writer.abort()
    metadata = read_metadata(path)
    assert metadata.sidecar_path is None
    assert metadata.checkpoint_path == Path(f"{path}.l2chk")
    assert metadata.clean_shutdown is False
    assert metadata.data_complete is False


def test_malformed_counter_prevents_complete_recording(tmp_path: Path) -> None:
    path = tmp_path / "malformed.l2bin"
    snapshot = Snapshot(
        1,
        100,
        (Level(10_000, 500),),
        (Level(10_100, 600),),
    )
    update = DepthUpdate(2, 2, 101, 101, (Level(10_000, 450),), ())
    book = L2OrderBook()
    writer = L2Writer(path, "BTCUSDT")
    writer.write(snapshot)
    book.install_snapshot(snapshot)
    writer.write(update)
    book.apply(update)
    writer.write_checkpoint(book.last_update_id, book.state_hash())
    metadata = writer.finalize(
        final_update_id=book.last_update_id,
        final_state_hash=book.state_hash(),
        malformed_messages=1,
    )
    assert metadata.data_complete is False


def test_sidecar_boolean_type_is_strict(tmp_path: Path) -> None:
    path = create_recording(tmp_path / "typed.l2bin")
    sidecar = Path(f"{path}.meta.json")
    import json

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["clean_shutdown"] = "true"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="boolean"):
        read_metadata(path)


def test_symbol_is_restricted_to_exchange_style_identifier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="letters or digits"):
        L2Writer(tmp_path / "bad.l2bin", "BTC/USDT")


def test_sidecar_free_l2_file_is_incomplete_even_without_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "orphan.l2bin"
    writer = L2Writer(path, "BTCUSDT")
    writer.write(Boundary(1, BoundaryReason.CONNECTION_START))
    writer.abort()
    Path(f"{path}.l2chk").unlink()
    metadata = read_metadata(path)
    assert metadata.clean_shutdown is False
    assert metadata.data_complete is False


def test_snapshot_with_unknown_flags_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "flags.l2bin"
    writer = L2Writer(path, "BTCUSDT")
    snapshot = Snapshot(1, 100, (Level(10_000, 1),), (Level(10_100, 1),))
    writer.write(snapshot)
    writer.abort()
    data = bytearray(path.read_bytes())
    # The first event begins at byte 128; flags are the second uint32.
    data[132:136] = (1).to_bytes(4, "little")
    path.write_bytes(data)
    with pytest.raises(RuntimeError, match="snapshot header"):
        tuple(iter_events(path))


def test_duplicate_checkpoint_event_index_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.l2bin"
    writer = L2Writer(path, "BTCUSDT")
    writer.write(Boundary(1, BoundaryReason.CONNECTION_START))
    writer.write_checkpoint(1, 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        writer.write_checkpoint(1, 1)
    writer.abort()


def test_final_checkpoint_must_match_final_state(tmp_path: Path) -> None:
    path = tmp_path / "final_mismatch.l2bin"
    writer = L2Writer(path, "BTCUSDT")
    snapshot = Snapshot(1, 100, (Level(10_000, 1),), (Level(10_100, 1),))
    update = DepthUpdate(2, 2, 101, 101, (), ())
    book = L2OrderBook()
    writer.write(snapshot)
    book.install_snapshot(snapshot)
    writer.write(update)
    book.apply(update)
    writer.write_checkpoint(book.last_update_id, book.state_hash())
    with pytest.raises(RuntimeError, match="Final checkpoint"):
        writer.finalize(final_update_id=book.last_update_id, final_state_hash=book.state_hash() + 1)
    writer.abort()


def test_zero_l2_creation_timestamp_is_rejected(tmp_path: Path) -> None:
    path = create_recording(tmp_path / "zero-created.l2bin")
    payload = bytearray(path.read_bytes())
    # created_unix_ns occupies bytes 40..47 in the 128-byte L2 header.
    payload[40:48] = (0).to_bytes(8, "little")
    path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="timestamp cannot be zero"):
        read_metadata(path)


@pytest.mark.parametrize("field", ["sha256", "checkpoint_sha256"])
def test_current_l2_sidecar_requires_well_formed_hashes(
    tmp_path: Path,
    field: str,
) -> None:
    import json

    path = create_recording(tmp_path / f"missing-{field}.l2bin")
    sidecar = Path(f"{path}.meta.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload.pop(field)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match=field):
        read_metadata(path)

    path = create_recording(tmp_path / f"malformed-{field}.l2bin")
    sidecar = Path(f"{path}.meta.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload[field] = "not-a-sha256"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 is malformed"):
        read_metadata(path)
