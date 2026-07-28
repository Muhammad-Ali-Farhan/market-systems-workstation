
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

HEADER_SIZE = 64
RECORD_SIZE = 32
EXPECTED_MAGIC = b"QENGINE1"
EXPECTED_VERSION = 1
EXPECTED_VOLUME_SCALE = 1_000_000
HEADER_STRUCT = struct.Struct("<8sIIIIQQQQQ")
UPDATE_ID_MAGIC = b"QUPDID1\x00"
UPDATE_ID_VERSION = 1
UPDATE_ID_HEADER_STRUCT = struct.Struct("<8sIIIIQ")
UPDATE_ID_HEADER_SIZE = UPDATE_ID_HEADER_STRUCT.size
RECORD_DTYPE = np.dtype(
    [
        ("timestamp_ns", "<u8"),
        ("best_bid", "<f8"),
        ("best_ask", "<f8"),
        ("bid_volume", "<u4"),
        ("ask_volume", "<u4"),
    ],
    align=False,
)

if HEADER_STRUCT.size != HEADER_SIZE:
    raise RuntimeError("Binary header definition is not 64 bytes.")
if RECORD_DTYPE.itemsize != RECORD_SIZE:
    raise RuntimeError("Binary record definition is not 32 bytes.")


@dataclass(frozen=True)
class RecordingBoundary:
    record_index: int
    kind: str


@dataclass(frozen=True)
class RecordingMetadata:
    path: Path
    file_size: int
    version: int
    header_size: int
    record_size: int
    flags: int
    volume_scale: int
    created_unix_ns: int
    record_count: int
    sidecar_path: Path | None
    update_id_path: Path | None
    update_id_count: int | None
    first_update_id: int | None
    last_update_id: int | None
    clean_shutdown: bool | None
    data_complete: bool | None
    accepted_records: int | None
    recorded_records: int | None
    recording_dropped: int | None
    recording_write_errors: int | None
    consumer_queue_dropped: int | None
    malformed_messages: int | None
    reconnect_count: int | None
    boundaries: tuple[RecordingBoundary, ...]


def _optional_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeError(f"Sidecar field {key!r} must be an integer.")
    return int(value)


def _optional_bool(payload: dict[str, object], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise RuntimeError(f"Sidecar field {key!r} must be a boolean.")
    return value


def _read_sidecar(path: Path, record_count: int) -> tuple[Path | None, dict[str, object]]:
    sidecar = Path(f"{path}.meta.json")
    if not sidecar.exists():
        return None, {}

    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise RuntimeError(f"Could not read recording sidecar {sidecar}: {exception}") from exception

    if not isinstance(payload, dict):
        raise RuntimeError(f"Recording sidecar must contain a JSON object: {sidecar}")
    if int(payload.get("schema_version", 0)) != 1:
        raise RuntimeError(f"Unsupported recording sidecar schema: {sidecar}")
    if "update_id_file" not in payload:
        raise RuntimeError(
            f"Current recording sidecar does not name its update-ID file: {sidecar}"
        )

    expected_numeric_fields = {
        "binary_version": EXPECTED_VERSION,
        "record_size": RECORD_SIZE,
        "volume_scale": EXPECTED_VOLUME_SCALE,
    }
    for key, expected in expected_numeric_fields.items():
        if key in payload and int(payload[key]) != expected:
            raise RuntimeError(
                f"Sidecar field {key!r} does not match the binary file: {sidecar}"
            )

    expected_identity_fields = {
        "source": "binance_spot",
        "symbol": "BTCUSDT",
        "stream": "bookTicker",
    }
    for key, expected in expected_identity_fields.items():
        if str(payload.get(key, "")) != expected:
            raise RuntimeError(
                f"Sidecar field {key!r} is invalid for this reader: {sidecar}"
            )
    if "recording_file" in payload and str(payload["recording_file"]) != path.name:
        raise RuntimeError(f"Sidecar belongs to a different recording: {sidecar}")

    raw_boundaries = payload.get("boundaries", [])
    if not isinstance(raw_boundaries, list):
        raise RuntimeError(f"Sidecar boundaries must be a list: {sidecar}")

    boundaries: list[dict[str, object]] = []
    for item in raw_boundaries:
        if not isinstance(item, dict):
            raise RuntimeError(f"Invalid sidecar boundary entry: {sidecar}")
        index = int(item.get("record_index", -1))
        kind = str(item.get("kind", "")).strip()
        if index < 0 or index > record_count or not kind:
            raise RuntimeError(f"Invalid sidecar boundary entry: {item!r}")
        boundaries.append({"record_index": index, "kind": kind})

    boundaries.sort(key=lambda item: (int(item["record_index"]), str(item["kind"])))
    deduplicated: list[dict[str, object]] = []
    for item in boundaries:
        if not deduplicated or item != deduplicated[-1]:
            deduplicated.append(item)

    payload = dict(payload)
    payload["boundaries"] = deduplicated
    return sidecar, payload


def _read_update_id_metadata(
    recording_path: Path,
    payload: dict[str, object],
    *,
    created_unix_ns: int,
    record_count: int,
) -> tuple[Path | None, int | None]:
    raw_name = payload.get("update_id_file")
    if raw_name is None:
        return None, None
    if int(payload.get("update_id_version", UPDATE_ID_VERSION)) != UPDATE_ID_VERSION:
        raise RuntimeError("Unsupported sidecar update-ID version.")
    name = str(raw_name)
    if not name or Path(name).name != name:
        raise RuntimeError("Sidecar update_id_file must be a filename in the recording folder.")
    path = recording_path.parent / name
    if not path.exists():
        raise RuntimeError(f"Recording update-ID file is missing: {path}")
    size = path.stat().st_size
    if size < UPDATE_ID_HEADER_SIZE:
        raise RuntimeError(f"Update-ID file is smaller than its header: {path}")
    with path.open("rb") as stream:
        header = stream.read(UPDATE_ID_HEADER_SIZE)
    magic, version, header_size, record_size, flags, created = (
        UPDATE_ID_HEADER_STRUCT.unpack(header)
    )
    if magic != UPDATE_ID_MAGIC or version != UPDATE_ID_VERSION:
        raise RuntimeError(f"Unsupported update-ID file format: {path}")
    if header_size != UPDATE_ID_HEADER_SIZE or record_size != 8:
        raise RuntimeError(f"Invalid update-ID layout: {path}")
    if flags != 0:
        raise RuntimeError(f"Unsupported update-ID flags in {path}: {flags}")
    if created != created_unix_ns:
        raise RuntimeError(f"Update-ID file belongs to another recording: {path}")
    payload_size = size - header_size
    if payload_size % record_size != 0:
        raise RuntimeError(f"Update-ID file ends with a partial record: {path}")
    count = payload_size // record_size
    if count != record_count:
        raise RuntimeError(
            f"Update-ID count ({count}) does not match market-record count "
            f"({record_count}): {path}"
        )
    sidecar_count = _optional_int(payload, "recorded_update_ids")
    if sidecar_count is not None and sidecar_count != count:
        raise RuntimeError(f"Update-ID count does not match the sidecar: {path}")
    return path, count


def read_metadata(file_path: str | Path) -> RecordingMetadata:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Recording does not exist: {path}")
    if not path.is_file():
        raise RuntimeError(f"Recording path is not a file: {path}")

    file_size = path.stat().st_size
    if file_size < HEADER_SIZE:
        raise RuntimeError(
            f"Recording is smaller than its {HEADER_SIZE}-byte header: {path}"
        )

    with path.open("rb") as stream:
        header = stream.read(HEADER_SIZE)

    (
        magic,
        version,
        header_size,
        record_size,
        flags,
        volume_scale,
        created_unix_ns,
        _reserved_1,
        _reserved_2,
        _reserved_3,
    ) = HEADER_STRUCT.unpack(header)

    if magic != EXPECTED_MAGIC:
        raise RuntimeError(f"Invalid recording magic in {path}: {magic!r}")
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"Unsupported recording version in {path}: {version}")
    if header_size != HEADER_SIZE:
        raise RuntimeError(f"Unexpected header size in {path}: {header_size}")
    if record_size != RECORD_SIZE:
        raise RuntimeError(f"Unexpected record size in {path}: {record_size}")
    if flags != 0:
        raise RuntimeError(f"Unsupported recording flags in {path}: {flags}")
    if volume_scale != EXPECTED_VOLUME_SCALE:
        raise RuntimeError(f"Unexpected volume scale in {path}: {volume_scale}")

    payload_size = file_size - header_size
    if payload_size % record_size != 0:
        raise RuntimeError(f"Recording ends with a partial record: {path}")
    record_count = payload_size // record_size

    sidecar_path, sidecar = _read_sidecar(path, record_count)
    if sidecar_path is not None:
        sidecar_created = _optional_int(sidecar, "created_unix_ns")
        if sidecar_created is None or sidecar_created != created_unix_ns:
            raise RuntimeError(
                f"Sidecar creation timestamp does not match the binary file: {path}"
            )
    inferred_incomplete = False
    if sidecar_path is None:
        conventional_update_ids = Path(f"{path}.qids")
        if conventional_update_ids.exists():
            # Current-format captures always create the parallel update-ID file.
            # Its presence without the atomic completion sidecar indicates an
            # interrupted or failed finalization, so treat the recording as
            # incomplete rather than silently accepting it as a legacy file.
            sidecar = {
                "update_id_file": conventional_update_ids.name,
                "update_id_version": UPDATE_ID_VERSION,
            }
            inferred_incomplete = True
    update_id_path, update_id_count = _read_update_id_metadata(
        path,
        sidecar,
        created_unix_ns=created_unix_ns,
        record_count=record_count,
    )
    boundaries = tuple(
        RecordingBoundary(int(item["record_index"]), str(item["kind"]))
        for item in sidecar.get("boundaries", [])
    )

    sidecar_recorded = _optional_int(sidecar, "recorded_records")
    if sidecar_recorded is not None and sidecar_recorded != record_count:
        raise RuntimeError(
            f"Recording payload count ({record_count}) does not match sidecar "
            f"recorded_records ({sidecar_recorded}): {path}"
        )

    accepted_records = _optional_int(sidecar, "accepted_records")
    recording_dropped = _optional_int(sidecar, "recording_dropped")
    write_errors = _optional_int(sidecar, "recording_write_errors")
    for key, value in (
        ("accepted_records", accepted_records),
        ("recording_dropped", recording_dropped),
        ("recording_write_errors", write_errors),
    ):
        if value is not None and value < 0:
            raise RuntimeError(f"Sidecar field {key!r} cannot be negative: {path}")
    if accepted_records is not None and sidecar_recorded is not None:
        if sidecar_recorded > accepted_records:
            raise RuntimeError(f"Recorded count exceeds accepted count: {path}")

    clean_shutdown = _optional_bool(sidecar, "clean_shutdown")
    data_complete = _optional_bool(sidecar, "data_complete")
    if sidecar_path is not None and (clean_shutdown is None or data_complete is None):
        raise RuntimeError(
            f"Current recording sidecar must declare clean_shutdown and data_complete: {path}"
        )
    if data_complete is True:
        if clean_shutdown is not True:
            raise RuntimeError(f"Complete recording is not marked cleanly shut down: {path}")
        if recording_dropped != 0 or write_errors != 0:
            raise RuntimeError(f"Complete recording reports recorder loss or write errors: {path}")
        if accepted_records != record_count:
            raise RuntimeError(f"Complete recording accepted/recorded counts disagree: {path}")
        if update_id_count != record_count:
            raise RuntimeError(f"Complete recording update-ID count disagrees: {path}")

    first_update_id = _optional_int(sidecar, "first_update_id")
    last_update_id = _optional_int(sidecar, "last_update_id")
    for key, value in (("first_update_id", first_update_id), ("last_update_id", last_update_id)):
        if value is not None and value < 0:
            raise RuntimeError(f"Sidecar field {key!r} cannot be negative: {path}")
    if (
        first_update_id not in (None, 0)
        and last_update_id not in (None, 0)
        and first_update_id > last_update_id
    ):
        raise RuntimeError(f"Sidecar update-ID range is invalid: {path}")

    return RecordingMetadata(
        path=path,
        file_size=file_size,
        version=version,
        header_size=header_size,
        record_size=record_size,
        flags=flags,
        volume_scale=volume_scale,
        created_unix_ns=created_unix_ns,
        record_count=record_count,
        sidecar_path=sidecar_path,
        update_id_path=update_id_path,
        update_id_count=update_id_count,
        first_update_id=first_update_id,
        last_update_id=last_update_id,
        clean_shutdown=(False if inferred_incomplete else clean_shutdown),
        data_complete=(False if inferred_incomplete else data_complete),
        accepted_records=accepted_records,
        recorded_records=sidecar_recorded,
        recording_dropped=recording_dropped,
        recording_write_errors=write_errors,
        consumer_queue_dropped=_optional_int(sidecar, "consumer_queue_dropped"),
        malformed_messages=_optional_int(sidecar, "malformed_messages"),
        reconnect_count=_optional_int(sidecar, "reconnect_count"),
        boundaries=boundaries,
    )


def open_records(
    file_path: str | Path,
    metadata: RecordingMetadata | None = None,
) -> np.ndarray:
    info = metadata if metadata is not None else read_metadata(file_path)
    if info.record_count == 0:
        return np.empty(0, dtype=RECORD_DTYPE)
    return np.memmap(
        info.path,
        mode="r",
        dtype=RECORD_DTYPE,
        offset=info.header_size,
        shape=(info.record_count,),
    )


def open_update_ids(
    metadata: RecordingMetadata,
) -> np.ndarray | None:
    if metadata.update_id_path is None or metadata.update_id_count is None:
        return None
    if metadata.update_id_count == 0:
        return np.empty(0, dtype="<u8")
    return np.memmap(
        metadata.update_id_path,
        mode="r",
        dtype="<u8",
        offset=UPDATE_ID_HEADER_SIZE,
        shape=(metadata.update_id_count,),
    )


def validate_update_ids(
    update_ids: np.ndarray | None,
    metadata: RecordingMetadata,
    *,
    context: str = "recording",
) -> None:
    if update_ids is None or update_ids.size == 0:
        return
    values = np.asarray(update_ids, dtype=np.uint64)
    if np.any(values == 0):
        first = int(np.flatnonzero(values == 0)[0])
        raise RuntimeError(
            f"{context} contains a zero exchange update ID at index {first:,}."
        )
    if np.any(values[1:] <= values[:-1]):
        first = int(np.flatnonzero(values[1:] <= values[:-1])[0] + 1)
        raise RuntimeError(
            f"{context} exchange update IDs are not strictly increasing "
            f"near index {first:,}."
        )
    if metadata.first_update_id not in (None, 0) and (
        int(values[0]) != metadata.first_update_id
    ):
        raise RuntimeError(f"{context} first update ID does not match its sidecar.")
    if metadata.last_update_id not in (None, 0) and (
        int(values[-1]) != metadata.last_update_id
    ):
        raise RuntimeError(f"{context} last update ID does not match its sidecar.")


def validate_records(records: np.ndarray, *, context: str = "recording") -> None:
    if records.dtype != RECORD_DTYPE:
        try:
            records = np.asarray(records, dtype=RECORD_DTYPE)
        except (TypeError, ValueError) as exception:
            raise RuntimeError(f"{context} does not use the expected 32-byte dtype.") from exception

    if records.size == 0:
        return

    timestamps = np.asarray(records["timestamp_ns"], dtype=np.uint64)
    bid = np.asarray(records["best_bid"], dtype=np.float64)
    ask = np.asarray(records["best_ask"], dtype=np.float64)

    invalid = (
        (timestamps == 0)
        | ~np.isfinite(bid)
        | ~np.isfinite(ask)
        | (bid <= 0.0)
        | (ask <= 0.0)
        | (bid > ask)
    )
    if np.any(invalid):
        first = int(np.flatnonzero(invalid)[0])
        raise RuntimeError(f"{context} contains an invalid market record at index {first:,}.")

    decreasing = timestamps[1:] < timestamps[:-1]
    if np.any(decreasing):
        first = int(np.flatnonzero(decreasing)[0] + 1)
        raise RuntimeError(f"{context} timestamps decrease at index {first:,}.")


def contiguous_slices(
    records: np.ndarray,
    metadata: RecordingMetadata,
    *,
    max_gap_ns: int,
) -> tuple[slice, ...]:
    if max_gap_ns <= 0:
        raise ValueError("max_gap_ns must be positive.")
    count = int(records.size)
    if count == 0:
        return ()

    cut_points: set[int] = {0, count}
    for boundary in metadata.boundaries:
        if boundary.kind in {"connection_start", "recording_queue_drop"}:
            cut_points.add(boundary.record_index)

    timestamps = np.asarray(records["timestamp_ns"], dtype=np.uint64)
    if timestamps.size > 1:
        delta = timestamps[1:] - timestamps[:-1]
        for index in np.flatnonzero(delta > np.uint64(max_gap_ns)).tolist():
            cut_points.add(int(index) + 1)

    ordered = sorted(point for point in cut_points if 0 <= point <= count)
    slices: list[slice] = []
    for start, stop in zip(ordered, ordered[1:]):
        if stop > start:
            slices.append(slice(start, stop))
    return tuple(slices)


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def unique_boundary_indices(boundaries: Iterable[RecordingBoundary]) -> tuple[int, ...]:
    return tuple(sorted({item.record_index for item in boundaries}))


def feature_reset_indices(metadata: RecordingMetadata) -> frozenset[int]:
    """Record indices that begin a new online feature-history segment."""
    return frozenset(
        boundary.record_index
        for boundary in metadata.boundaries
        if boundary.kind in {"connection_start", "recording_queue_drop"}
        and boundary.record_index > 0
    )

