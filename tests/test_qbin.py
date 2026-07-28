
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qbin import (
    UPDATE_ID_HEADER_SIZE,
    contiguous_slices,
    open_records,
    open_update_ids,
    read_metadata,
    validate_records,
    validate_update_ids,
)
from conftest import synthetic_records, write_qbin


def test_metadata_and_segments(tmp_path: Path) -> None:
    records = synthetic_records(100)
    records["timestamp_ns"][60:] += np.uint64(10_000_000_000)
    path = write_qbin(
        tmp_path / "sample.qbin",
        records,
        boundaries=[
            {"record_index": 0, "kind": "connection_start"},
            {"record_index": 40, "kind": "recording_queue_drop"},
        ],
    )
    metadata = read_metadata(path)
    mapped = open_records(path, metadata)
    validate_records(mapped)
    assert metadata.record_count == 100
    assert metadata.data_complete is True
    assert contiguous_slices(mapped, metadata, max_gap_ns=5_000_000_000) == (
        slice(0, 40),
        slice(40, 60),
        slice(60, 100),
    )


def test_partial_record_is_rejected(tmp_path: Path) -> None:
    path = write_qbin(tmp_path / "partial.qbin", synthetic_records(10))
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="partial record"):
        read_metadata(path)


def test_invalid_market_record_is_rejected(tmp_path: Path) -> None:
    records = synthetic_records(10)
    records["best_bid"][4] = records["best_ask"][4] + 1.0
    path = write_qbin(tmp_path / "invalid.qbin", records)
    with pytest.raises(RuntimeError, match="index 4"):
        validate_records(open_records(path))


def test_malformed_sidecar_boolean_is_rejected(tmp_path: Path) -> None:
    path = write_qbin(tmp_path / "bad-sidecar.qbin", synthetic_records(10))
    sidecar_path = Path(f"{path}.meta.json")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["clean_shutdown"] = "false"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="boolean"):
        read_metadata(path)


def test_current_format_update_ids_are_validated(tmp_path: Path) -> None:
    path = write_qbin(
        tmp_path / "current.qbin",
        synthetic_records(10),
        with_update_ids=True,
    )
    metadata = read_metadata(path)
    update_ids = open_update_ids(metadata)
    validate_update_ids(update_ids, metadata)
    assert update_ids is not None
    assert update_ids.tolist() == list(range(1, 11))


def test_zero_update_id_is_rejected(tmp_path: Path) -> None:
    path = write_qbin(
        tmp_path / "zero-id.qbin",
        synthetic_records(10),
        with_update_ids=True,
    )
    update_path = Path(f"{path}.qids")
    payload = bytearray(update_path.read_bytes())
    payload[UPDATE_ID_HEADER_SIZE:UPDATE_ID_HEADER_SIZE + 8] = (0).to_bytes(8, "little")
    update_path.write_bytes(payload)
    metadata = read_metadata(path)
    with pytest.raises(RuntimeError, match="zero exchange update ID"):
        validate_update_ids(open_update_ids(metadata), metadata)


def test_current_format_without_completion_sidecar_is_incomplete(tmp_path: Path) -> None:
    path = write_qbin(
        tmp_path / "interrupted.qbin",
        synthetic_records(10),
        with_update_ids=True,
    )
    Path(f"{path}.meta.json").unlink()
    metadata = read_metadata(path)
    assert metadata.clean_shutdown is False
    assert metadata.data_complete is False


def test_sidecar_creation_timestamp_must_match_binary(tmp_path: Path) -> None:
    path = write_qbin(tmp_path / "wrong-created.qbin", synthetic_records(10))
    sidecar_path = Path(f"{path}.meta.json")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["created_unix_ns"] += 1
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="creation timestamp"):
        read_metadata(path)


def test_complete_flag_requires_consistent_counts(tmp_path: Path) -> None:
    path = write_qbin(tmp_path / "inconsistent-complete.qbin", synthetic_records(10))
    sidecar_path = Path(f"{path}.meta.json")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["recording_dropped"] = 1
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="recorder loss"):
        read_metadata(path)



def test_unsupported_binary_flags_are_rejected(tmp_path: Path) -> None:
    path = write_qbin(tmp_path / "binary-flags.qbin", synthetic_records(10))
    payload = bytearray(path.read_bytes())
    payload[20:24] = (1).to_bytes(4, "little")
    path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="recording flags"):
        read_metadata(path)


def test_unsupported_update_id_flags_are_rejected(tmp_path: Path) -> None:
    path = write_qbin(tmp_path / "update-flags.qbin", synthetic_records(10))
    update_path = Path(f"{path}.qids")
    payload = bytearray(update_path.read_bytes())
    payload[20:24] = (1).to_bytes(4, "little")
    update_path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="update-ID flags"):
        read_metadata(path)


def test_sidecar_market_identity_must_match(tmp_path: Path) -> None:
    path = write_qbin(tmp_path / "wrong-symbol.qbin", synthetic_records(10))
    sidecar_path = Path(f"{path}.meta.json")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["symbol"] = "ETHUSDT"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="symbol"):
        read_metadata(path)
