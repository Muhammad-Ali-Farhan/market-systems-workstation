from __future__ import annotations

import enum
import hashlib
import json
import os
import re
import struct
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from l2book import (
    PRICE_SCALE,
    QUANTITY_SCALE,
    DepthUpdate,
    Level,
    Snapshot,
    Trade,
    UINT64_MAX,
)

MAGIC = b"QL2EVT1\0"
VERSION = 1
HEADER_STRUCT = struct.Struct("<8sIIIIQQQ16s8Q")
HEADER_SIZE = HEADER_STRUCT.size
EVENT_HEADER_STRUCT = struct.Struct("<IIQQQQIIqQIIQ")
EVENT_HEADER_SIZE = EVENT_HEADER_STRUCT.size
LEVEL_STRUCT = struct.Struct("<qQ")
LEVEL_SIZE = LEVEL_STRUCT.size
CHECKPOINT_MAGIC = b"QL2CHK1\0"
CHECKPOINT_HEADER_STRUCT = struct.Struct("<8sIIIIQ")
CHECKPOINT_HEADER_SIZE = CHECKPOINT_HEADER_STRUCT.size
CHECKPOINT_RECORD_STRUCT = struct.Struct("<QQQ")
MAX_LEVELS_PER_EVENT = 20_000
MAX_RECORD_SIZE = EVENT_HEADER_SIZE + MAX_LEVELS_PER_EVENT * LEVEL_SIZE * 2

if HEADER_SIZE != 128:
    raise RuntimeError(f"L2 header must be 128 bytes, received {HEADER_SIZE}.")
if EVENT_HEADER_SIZE != 80:
    raise RuntimeError(f"L2 event header must be 80 bytes, received {EVENT_HEADER_SIZE}.")
if LEVEL_SIZE != 16:
    raise RuntimeError(f"L2 level must be 16 bytes, received {LEVEL_SIZE}.")


class EventType(enum.IntEnum):
    SNAPSHOT = 1
    DEPTH = 2
    TRADE = 3
    BOUNDARY = 4


class BoundaryReason(enum.IntEnum):
    CONNECTION_START = 1
    SEQUENCE_GAP = 2
    SNAPSHOT_RETRY = 3
    USER_STOP = 4
    QUEUE_OVERFLOW = 5
    CONNECTION_END = 6


TRADE_BUYER_IS_MAKER = 1 << 0
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{1,15}$")


@dataclass(frozen=True, slots=True)
class Boundary:
    receipt_timestamp_ns: int
    reason: BoundaryReason

    def __post_init__(self) -> None:
        if (
            isinstance(self.receipt_timestamp_ns, bool)
            or not isinstance(self.receipt_timestamp_ns, int)
        ):
            raise TypeError("Boundary receipt timestamp must be an integer.")
        if self.receipt_timestamp_ns <= 0 or self.receipt_timestamp_ns > UINT64_MAX:
            raise ValueError("Boundary receipt timestamp must be a positive 64-bit value.")
        if not isinstance(self.reason, BoundaryReason):
            raise TypeError("Boundary reason must be a BoundaryReason value.")


L2Event = Snapshot | DepthUpdate | Trade | Boundary


@dataclass(frozen=True, slots=True)
class Checkpoint:
    event_index: int
    update_id: int
    state_hash: int

    def __post_init__(self) -> None:
        for name, value in (
            ("event_index", self.event_index),
            ("update_id", self.update_id),
            ("state_hash", self.state_hash),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Checkpoint {name} must be an integer.")
            if value <= 0 or value > UINT64_MAX:
                raise ValueError(f"Checkpoint {name} must be a positive 64-bit value.")


@dataclass(frozen=True, slots=True)
class L2Metadata:
    path: Path
    symbol: str
    created_unix_ns: int
    event_count: int
    snapshot_count: int
    depth_count: int
    trade_count: int
    boundary_count: int
    checkpoint_path: Path | None
    clean_shutdown: bool | None
    data_complete: bool | None
    final_update_id: int | None
    final_state_hash: int | None
    sidecar_path: Path | None
    sha256: str | None
    checkpoint_sha256: str | None
    sequence_gaps: int | None
    snapshot_retries: int | None
    queue_drops: int | None
    malformed_messages: int | None


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _encode_symbol(symbol: str) -> bytes:
    if not isinstance(symbol, str):
        raise TypeError("Symbol must be a string.")
    normalized = symbol.strip().upper()
    if _SYMBOL_PATTERN.fullmatch(normalized) is None:
        raise ValueError("Symbol must contain 1-15 uppercase ASCII letters or digits.")
    return normalized.encode("ascii").ljust(16, b"\0")


def _decode_symbol(value: bytes) -> str:
    if len(value) != 16:
        raise RuntimeError("L2 binary symbol field has the wrong size.")
    raw, separator, padding = value.partition(b"\0")
    if separator and any(padding):
        raise RuntimeError("L2 binary symbol padding is not zero-filled.")
    try:
        symbol = raw.decode("ascii")
    except UnicodeDecodeError as exception:
        raise RuntimeError("L2 binary symbol is not ASCII.") from exception
    if _SYMBOL_PATTERN.fullmatch(symbol) is None:
        raise RuntimeError("L2 binary symbol contains unsupported characters.")
    return symbol




def _required_int(payload: dict[str, object], key: str, *, minimum: int = 0) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"L2 sidecar field {key!r} must be an integer.")
    if value < minimum:
        raise RuntimeError(
            f"L2 sidecar field {key!r} must be at least {minimum}."
        )
    return value


def _required_bool(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"L2 sidecar field {key!r} must be a boolean.")
    return value


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"L2 sidecar field {key!r} must be a non-empty string.")
    return value

def _level_payload(bids: tuple[Level, ...], asks: tuple[Level, ...]) -> bytes:
    if len(bids) > MAX_LEVELS_PER_EVENT or len(asks) > MAX_LEVELS_PER_EVENT:
        raise ValueError("L2 event contains too many levels.")
    output = bytearray((len(bids) + len(asks)) * LEVEL_SIZE)
    offset = 0
    for level in bids + asks:
        LEVEL_STRUCT.pack_into(output, offset, level.price, level.quantity)
        offset += LEVEL_SIZE
    return bytes(output)


def _pack_event(event: L2Event) -> bytes:
    if isinstance(event, Snapshot):
        event_type = EventType.SNAPSHOT
        flags = 0
        receipt = event.receipt_timestamp_ns
        exchange_time = 0
        first_id = final_id = event.last_update_id
        bids, asks = event.bids, event.asks
        trade_price = trade_quantity = 0
    elif isinstance(event, DepthUpdate):
        event_type = EventType.DEPTH
        flags = 0
        receipt = event.receipt_timestamp_ns
        exchange_time = event.event_time_ms
        first_id = event.first_update_id
        final_id = event.final_update_id
        bids, asks = event.bids, event.asks
        trade_price = trade_quantity = 0
    elif isinstance(event, Trade):
        event_type = EventType.TRADE
        flags = TRADE_BUYER_IS_MAKER if event.buyer_is_maker else 0
        receipt = event.receipt_timestamp_ns
        exchange_time = event.event_time_ms
        first_id = event.aggregate_trade_id
        final_id = 0
        bids = asks = ()
        trade_price = event.price
        trade_quantity = event.quantity
    elif isinstance(event, Boundary):
        event_type = EventType.BOUNDARY
        flags = int(event.reason)
        receipt = event.receipt_timestamp_ns
        exchange_time = first_id = final_id = 0
        bids = asks = ()
        trade_price = trade_quantity = 0
    else:
        raise TypeError(f"Unsupported L2 event type: {type(event)!r}")

    if receipt <= 0:
        raise ValueError("Event receipt timestamp must be positive.")
    payload = _level_payload(bids, asks)
    payload_crc = zlib.crc32(payload) & 0xFFFFFFFF
    record_size = EVENT_HEADER_SIZE + len(payload)
    header = EVENT_HEADER_STRUCT.pack(
        int(event_type),
        flags,
        receipt,
        exchange_time,
        first_id,
        final_id,
        len(bids),
        len(asks),
        trade_price,
        trade_quantity,
        payload_crc,
        record_size,
        0,
    )
    return header + payload


def _write_exact(stream: BinaryIO, payload: bytes, *, context: str) -> None:
    written = stream.write(payload)
    if written != len(payload):
        raise OSError(f"Could not write complete {context}.")


class L2Writer:
    def __init__(
        self,
        path: str | Path,
        symbol: str,
        *,
        checkpoint_interval: int = 1_000,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if self.path.suffix.lower() != ".l2bin":
            raise ValueError("L2 recording must use the .l2bin suffix.")
        self.sidecar_path = Path(f"{self.path}.meta.json")
        self.checkpoint_path = Path(f"{self.path}.l2chk")
        self.symbol = symbol.strip().upper()
        self.checkpoint_interval = int(checkpoint_interval)
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive.")
        for candidate in (self.path, self.sidecar_path, self.checkpoint_path):
            if candidate.exists():
                raise FileExistsError(f"Refusing to overwrite L2 artifact: {candidate}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.created_unix_ns = time.time_ns()
        self._stream: BinaryIO | None = None
        self._checkpoint_stream: BinaryIO | None = None
        try:
            self._stream = self.path.open("xb")
            self._checkpoint_stream = self.checkpoint_path.open("xb")
            _write_exact(
                self._stream,
                HEADER_STRUCT.pack(
                    MAGIC,
                    VERSION,
                    HEADER_SIZE,
                    EVENT_HEADER_SIZE,
                    LEVEL_SIZE,
                    PRICE_SCALE,
                    QUANTITY_SCALE,
                    self.created_unix_ns,
                    _encode_symbol(self.symbol),
                    *([0] * 8),
                ),
                context="L2 file header",
            )
            _write_exact(
                self._checkpoint_stream,
                CHECKPOINT_HEADER_STRUCT.pack(
                    CHECKPOINT_MAGIC,
                    VERSION,
                    CHECKPOINT_HEADER_SIZE,
                    CHECKPOINT_RECORD_STRUCT.size,
                    0,
                    self.created_unix_ns,
                ),
                context="L2 checkpoint header",
            )
        except Exception:
            if self._stream is not None:
                self._stream.close()
            if self._checkpoint_stream is not None:
                self._checkpoint_stream.close()
            self.path.unlink(missing_ok=True)
            self.checkpoint_path.unlink(missing_ok=True)
            raise
        self.event_count = 0
        self.snapshot_count = 0
        self.depth_count = 0
        self.trade_count = 0
        self.boundary_count = 0
        self.checkpoint_count = 0
        self._last_checkpoint: Checkpoint | None = None
        self._closed = False

    def write(self, event: L2Event) -> int:
        if self._closed:
            raise RuntimeError("L2 writer is closed.")
        encoded = _pack_event(event)
        assert self._stream is not None
        _write_exact(self._stream, encoded, context="L2 event")
        self.event_count += 1
        if isinstance(event, Snapshot):
            self.snapshot_count += 1
        elif isinstance(event, DepthUpdate):
            self.depth_count += 1
        elif isinstance(event, Trade):
            self.trade_count += 1
        else:
            self.boundary_count += 1
        return self.event_count - 1

    def write_checkpoint(self, update_id: int, state_hash: int) -> None:
        if self._closed:
            raise RuntimeError("L2 writer is closed.")
        checkpoint = Checkpoint(self.event_count, update_id, state_hash)
        if (
            self._last_checkpoint is not None
            and checkpoint.event_index <= self._last_checkpoint.event_index
        ):
            raise ValueError("Checkpoint event indices must be strictly increasing.")
        assert self._checkpoint_stream is not None
        _write_exact(
            self._checkpoint_stream,
            CHECKPOINT_RECORD_STRUCT.pack(
                checkpoint.event_index, checkpoint.update_id, checkpoint.state_hash
            ),
            context="L2 checkpoint",
        )
        self._last_checkpoint = checkpoint
        self.checkpoint_count += 1

    def finalize(
        self,
        *,
        final_update_id: int,
        final_state_hash: int,
        sequence_gaps: int = 0,
        snapshot_retries: int = 0,
        queue_drops: int = 0,
        malformed_messages: int = 0,
        clean_shutdown: bool = True,
        extra: dict[str, object] | None = None,
    ) -> L2Metadata:
        if self._closed:
            return read_metadata(self.path, verify_hashes=True)
        if not isinstance(clean_shutdown, bool):
            raise TypeError("clean_shutdown must be a boolean.")
        for name, value in (
            ("final_update_id", final_update_id),
            ("final_state_hash", final_state_hash),
            ("sequence_gaps", sequence_gaps),
            ("snapshot_retries", snapshot_retries),
            ("queue_drops", queue_drops),
            ("malformed_messages", malformed_messages),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > UINT64_MAX
            ):
                raise ValueError(f"{name} must be a non-negative 64-bit integer.")
        if extra is not None:
            # Validate serializability before closing the streams; otherwise a
            # malformed supplementary payload could strand a finalized binary
            # without an explainable metadata sidecar.
            json.dumps(extra, allow_nan=False)
        data_complete = bool(
            clean_shutdown
            and queue_drops == 0
            and malformed_messages == 0
            and self.event_count > 0
            and self.snapshot_count > 0
            and self.depth_count > 0
            and self.checkpoint_count > 0
            and final_update_id > 0
            and final_state_hash > 0
        )
        if data_complete:
            if self._last_checkpoint is None:
                raise RuntimeError("Complete L2 recording requires a final checkpoint.")
            if (
                self._last_checkpoint.event_index != self.event_count
                or self._last_checkpoint.update_id != final_update_id
                or self._last_checkpoint.state_hash != final_state_hash
            ):
                raise RuntimeError(
                    "Final checkpoint must match the final event index, update ID, and state hash."
                )
        assert self._stream is not None and self._checkpoint_stream is not None
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._checkpoint_stream.flush()
        os.fsync(self._checkpoint_stream.fileno())
        self._checkpoint_stream.close()
        self._closed = True
        payload: dict[str, object] = {
            "schema_version": 1,
            "binary_version": VERSION,
            "source": "binance_spot",
            "symbol": self.symbol,
            "stream": "diff_depth_100ms+aggTrade",
            "recording_file": self.path.name,
            "checkpoint_file": self.checkpoint_path.name,
            "created_unix_ns": self.created_unix_ns,
            "price_scale": PRICE_SCALE,
            "quantity_scale": QUANTITY_SCALE,
            "clean_shutdown": clean_shutdown,
            "data_complete": data_complete,
            "event_count": self.event_count,
            "snapshot_count": self.snapshot_count,
            "depth_count": self.depth_count,
            "trade_count": self.trade_count,
            "boundary_count": self.boundary_count,
            "checkpoint_count": self.checkpoint_count,
            "final_update_id": final_update_id,
            "final_state_hash": final_state_hash,
            "sequence_gaps": sequence_gaps,
            "snapshot_retries": snapshot_retries,
            "queue_drops": queue_drops,
            "malformed_messages": malformed_messages,
            "sha256": sha256_file(self.path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
        }
        if extra:
            payload["extra"] = extra
        _atomic_json(self.sidecar_path, payload)
        return read_metadata(self.path, verify_hashes=True)

    def abort(self) -> None:
        if self._closed:
            return
        if self._stream is not None:
            self._stream.close()
        if self._checkpoint_stream is not None:
            self._checkpoint_stream.close()
        self._closed = True

    def __enter__(self) -> "L2Writer":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        # Finalization needs the reconstructed final book state and must be
        # explicit. Always close an unfinalized writer so a context-manager
        # exit cannot leak file descriptors; the absent sidecar deliberately
        # marks the artifact as interrupted/incomplete.
        if not self._closed:
            self.abort()


def _read_header(stream: BinaryIO) -> tuple[str, int]:
    raw = stream.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise RuntimeError("L2 file does not contain a complete header.")
    (
        magic,
        version,
        header_size,
        event_header_size,
        level_size,
        price_scale,
        quantity_scale,
        created_unix_ns,
        symbol_raw,
        *reserved,
    ) = HEADER_STRUCT.unpack(raw)
    if magic != MAGIC or version != VERSION:
        raise RuntimeError("Unsupported L2 binary format.")
    if header_size != HEADER_SIZE or event_header_size != EVENT_HEADER_SIZE:
        raise RuntimeError("L2 binary header layout does not match this reader.")
    if level_size != LEVEL_SIZE:
        raise RuntimeError("L2 level layout does not match this reader.")
    if price_scale != PRICE_SCALE or quantity_scale != QUANTITY_SCALE:
        raise RuntimeError("L2 fixed-point scales do not match this reader.")
    if created_unix_ns == 0:
        raise RuntimeError("L2 binary creation timestamp cannot be zero.")
    if any(reserved):
        raise RuntimeError("L2 binary header contains unsupported flags.")
    return _decode_symbol(symbol_raw), created_unix_ns


def _read_levels(
    payload: bytes,
    bid_count: int,
    ask_count: int,
) -> tuple[tuple[Level, ...], tuple[Level, ...]]:
    expected = (bid_count + ask_count) * LEVEL_SIZE
    if len(payload) != expected:
        raise RuntimeError("L2 event payload size does not match its level counts.")
    levels: list[Level] = []
    for offset in range(0, len(payload), LEVEL_SIZE):
        price, quantity = LEVEL_STRUCT.unpack_from(payload, offset)
        levels.append(Level(price, quantity))
    return tuple(levels[:bid_count]), tuple(levels[bid_count:])


def iter_events(path: str | Path) -> Iterator[L2Event]:
    file_path = Path(path)
    with file_path.open("rb") as stream:
        _read_header(stream)
        while True:
            raw_header = stream.read(EVENT_HEADER_SIZE)
            if not raw_header:
                break
            if len(raw_header) != EVENT_HEADER_SIZE:
                raise RuntimeError("L2 file ends with a truncated event header.")
            (
                raw_type,
                flags,
                receipt,
                exchange_time,
                first_id,
                final_id,
                bid_count,
                ask_count,
                trade_price,
                trade_quantity,
                payload_crc,
                record_size,
                reserved,
            ) = EVENT_HEADER_STRUCT.unpack(raw_header)
            if reserved != 0:
                raise RuntimeError("L2 event contains unsupported reserved data.")
            if record_size < EVENT_HEADER_SIZE or record_size > MAX_RECORD_SIZE:
                raise RuntimeError("L2 event record size is invalid.")
            if bid_count > MAX_LEVELS_PER_EVENT or ask_count > MAX_LEVELS_PER_EVENT:
                raise RuntimeError("L2 event declares too many levels.")
            expected_size = EVENT_HEADER_SIZE + (bid_count + ask_count) * LEVEL_SIZE
            if record_size != expected_size:
                raise RuntimeError("L2 event record size disagrees with its level counts.")
            payload = stream.read(record_size - EVENT_HEADER_SIZE)
            if len(payload) != record_size - EVENT_HEADER_SIZE:
                raise RuntimeError("L2 file ends with a truncated event payload.")
            if zlib.crc32(payload) & 0xFFFFFFFF != payload_crc:
                raise RuntimeError("L2 event payload CRC mismatch.")
            bids, asks = _read_levels(payload, bid_count, ask_count)
            try:
                event_type = EventType(raw_type)
            except ValueError as exception:
                raise RuntimeError(f"Unsupported L2 event type: {raw_type}") from exception
            if event_type is EventType.SNAPSHOT:
                if (
                    flags != 0
                    or first_id != final_id
                    or first_id == 0
                    or exchange_time != 0
                    or trade_price != 0
                    or trade_quantity != 0
                ):
                    raise RuntimeError("Invalid L2 snapshot header.")
                yield Snapshot(receipt, first_id, bids, asks)
            elif event_type is EventType.DEPTH:
                if flags != 0 or trade_price != 0 or trade_quantity != 0:
                    raise RuntimeError("Invalid L2 depth header.")
                yield DepthUpdate(receipt, exchange_time, first_id, final_id, bids, asks)
            elif event_type is EventType.TRADE:
                if (
                    flags & ~TRADE_BUYER_IS_MAKER
                    or bids
                    or asks
                    or final_id != 0
                    or first_id == 0
                    or trade_price <= 0
                    or trade_quantity <= 0
                ):
                    raise RuntimeError("Invalid L2 trade header.")
                yield Trade(
                    receipt,
                    exchange_time,
                    first_id,
                    trade_price,
                    trade_quantity,
                    bool(flags & TRADE_BUYER_IS_MAKER),
                )
            else:
                if (
                    exchange_time
                    or bids
                    or asks
                    or first_id
                    or final_id
                    or trade_price
                    or trade_quantity
                ):
                    raise RuntimeError("Invalid L2 boundary header.")
                try:
                    reason = BoundaryReason(flags)
                except ValueError as exception:
                    raise RuntimeError(f"Unsupported L2 boundary reason: {flags}") from exception
                yield Boundary(receipt, reason)


def read_checkpoints(
    path: str | Path,
    expected_created_unix_ns: int | None = None,
) -> tuple[Checkpoint, ...]:
    checkpoint_path = Path(path)
    with checkpoint_path.open("rb") as stream:
        header = stream.read(CHECKPOINT_HEADER_SIZE)
        if len(header) != CHECKPOINT_HEADER_SIZE:
            raise RuntimeError("L2 checkpoint file has a truncated header.")
        (
            magic,
            version,
            header_size,
            record_size,
            flags,
            created,
        ) = CHECKPOINT_HEADER_STRUCT.unpack(header)
        if (
            magic != CHECKPOINT_MAGIC
            or version != VERSION
            or header_size != CHECKPOINT_HEADER_SIZE
            or record_size != CHECKPOINT_RECORD_STRUCT.size
            or flags != 0
        ):
            raise RuntimeError("Invalid L2 checkpoint file header.")
        if expected_created_unix_ns is not None and created != expected_created_unix_ns:
            raise RuntimeError("L2 checkpoint file belongs to another recording.")
        payload = stream.read()
    if len(payload) % CHECKPOINT_RECORD_STRUCT.size:
        raise RuntimeError("L2 checkpoint file ends with a partial record.")
    checkpoints = tuple(
        Checkpoint(*CHECKPOINT_RECORD_STRUCT.unpack_from(payload, offset))
        for offset in range(0, len(payload), CHECKPOINT_RECORD_STRUCT.size)
    )
    if any(
        current.event_index <= previous.event_index
        for previous, current in zip(checkpoints, checkpoints[1:])
    ):
        raise RuntimeError("L2 checkpoint indices are not strictly increasing.")
    return checkpoints


def read_metadata(path: str | Path, *, verify_hashes: bool = False) -> L2Metadata:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"L2 recording does not exist: {file_path}")
    with file_path.open("rb") as stream:
        symbol, created_unix_ns = _read_header(stream)

    sidecar_path = Path(f"{file_path}.meta.json")
    conventional_checkpoint = Path(f"{file_path}.l2chk")
    sidecar: dict[str, object] = {}
    expected_recording_hash: str | None = None
    expected_checkpoint_hash: str | None = None
    inferred_incomplete = False
    if sidecar_path.exists():
        try:
            loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            raise RuntimeError(f"Could not read L2 sidecar: {sidecar_path}") from exception
        if not isinstance(loaded, dict):
            raise RuntimeError("L2 metadata sidecar must contain a JSON object.")
        sidecar = loaded
        if _required_int(sidecar, "schema_version", minimum=1) != 1:
            raise RuntimeError("Unsupported L2 metadata schema.")
        if _required_int(sidecar, "binary_version", minimum=1) != VERSION:
            raise RuntimeError("L2 sidecar binary version does not match this reader.")
        if _required_string(sidecar, "source") != "binance_spot":
            raise RuntimeError("L2 sidecar source is unsupported.")
        if _required_string(sidecar, "stream") != "diff_depth_100ms+aggTrade":
            raise RuntimeError("L2 sidecar stream identity is unsupported.")
        if _required_string(sidecar, "symbol") != symbol:
            raise RuntimeError("L2 sidecar symbol does not match the binary header.")
        if _required_int(sidecar, "created_unix_ns", minimum=1) != created_unix_ns:
            raise RuntimeError("L2 sidecar timestamp does not match the binary header.")
        if _required_string(sidecar, "recording_file") != file_path.name:
            raise RuntimeError("L2 sidecar belongs to another recording.")
        if _required_int(sidecar, "price_scale", minimum=1) != PRICE_SCALE:
            raise RuntimeError("L2 sidecar price scale does not match the binary file.")
        if _required_int(sidecar, "quantity_scale", minimum=1) != QUANTITY_SCALE:
            raise RuntimeError("L2 sidecar quantity scale does not match the binary file.")
        expected_recording_hash = _required_string(sidecar, "sha256")
        expected_checkpoint_hash = _required_string(sidecar, "checkpoint_sha256")
        if re.fullmatch(r"[0-9a-f]{64}", expected_recording_hash) is None:
            raise RuntimeError("L2 sidecar recording SHA-256 is malformed.")
        if re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint_hash) is None:
            raise RuntimeError("L2 sidecar checkpoint SHA-256 is malformed.")
    else:
        # Version 1 has never had a legacy sidecar-free completed form. Any
        # .l2bin without its atomically published sidecar is therefore an
        # interrupted or manually altered current-format artifact, regardless
        # of whether the checkpoint companion also survived.
        inferred_incomplete = True

    event_counts = {item: 0 for item in EventType}
    for event in iter_events(file_path):
        if isinstance(event, Snapshot):
            event_counts[EventType.SNAPSHOT] += 1
        elif isinstance(event, DepthUpdate):
            event_counts[EventType.DEPTH] += 1
        elif isinstance(event, Trade):
            event_counts[EventType.TRADE] += 1
        else:
            event_counts[EventType.BOUNDARY] += 1
    event_count = sum(event_counts.values())

    checkpoint_path: Path | None = None
    checkpoints: tuple[Checkpoint, ...] = ()
    if sidecar:
        expected_counts = {
            "event_count": event_count,
            "snapshot_count": event_counts[EventType.SNAPSHOT],
            "depth_count": event_counts[EventType.DEPTH],
            "trade_count": event_counts[EventType.TRADE],
            "boundary_count": event_counts[EventType.BOUNDARY],
        }
        for key, expected in expected_counts.items():
            if _required_int(sidecar, key) != expected:
                raise RuntimeError(
                    f"L2 sidecar field {key!r} does not match the binary file."
                )
        raw_name = _required_string(sidecar, "checkpoint_file")
        if Path(raw_name).name != raw_name:
            raise RuntimeError("L2 checkpoint file must be in the recording directory.")
        checkpoint_path = file_path.parent / raw_name
        if not checkpoint_path.is_file():
            raise RuntimeError(f"L2 checkpoint file is missing: {checkpoint_path}")
        checkpoints = read_checkpoints(checkpoint_path, created_unix_ns)
        if _required_int(sidecar, "checkpoint_count") != len(checkpoints):
            raise RuntimeError("L2 sidecar checkpoint count does not match its file.")
    elif conventional_checkpoint.exists():
        checkpoint_path = conventional_checkpoint
        checkpoints = read_checkpoints(checkpoint_path, created_unix_ns)

    clean_shutdown: bool | None
    data_complete: bool | None
    final_update_id: int | None
    final_state_hash: int | None
    sequence_gaps: int | None
    snapshot_retries: int | None
    queue_drops: int | None
    malformed_messages: int | None
    if sidecar:
        clean_shutdown = _required_bool(sidecar, "clean_shutdown")
        data_complete = _required_bool(sidecar, "data_complete")
        final_update_id = _required_int(sidecar, "final_update_id")
        final_state_hash = _required_int(sidecar, "final_state_hash")
        sequence_gaps = _required_int(sidecar, "sequence_gaps")
        snapshot_retries = _required_int(sidecar, "snapshot_retries")
        queue_drops = _required_int(sidecar, "queue_drops")
        malformed_messages = _required_int(sidecar, "malformed_messages")
        expected_complete = bool(
            clean_shutdown
            and queue_drops == 0
            and malformed_messages == 0
            and event_count > 0
            and event_counts[EventType.SNAPSHOT] > 0
            and event_counts[EventType.DEPTH] > 0
            and checkpoints
            and final_update_id > 0
            and final_state_hash > 0
        )
        if data_complete != expected_complete:
            raise RuntimeError(
                "L2 sidecar data_complete is inconsistent with capture counters."
            )
        if data_complete:
            final_checkpoint = checkpoints[-1]
            if final_checkpoint.event_index != event_count:
                raise RuntimeError(
                    "Complete L2 recording does not end with a final checkpoint."
                )
            if (
                final_checkpoint.update_id != final_update_id
                or final_checkpoint.state_hash != final_state_hash
            ):
                raise RuntimeError(
                    "Complete L2 recording final checkpoint disagrees with its sidecar."
                )
    else:
        clean_shutdown = False if inferred_incomplete else None
        data_complete = False if inferred_incomplete else None
        final_update_id = None
        final_state_hash = None
        sequence_gaps = None
        snapshot_retries = None
        queue_drops = None
        malformed_messages = None

    if verify_hashes and sidecar:
        assert expected_recording_hash is not None
        assert expected_checkpoint_hash is not None
        if sha256_file(file_path) != expected_recording_hash:
            raise RuntimeError("L2 recording SHA-256 does not match its sidecar.")
        if checkpoint_path is None or sha256_file(checkpoint_path) != expected_checkpoint_hash:
            raise RuntimeError("L2 checkpoint SHA-256 does not match its sidecar.")

    return L2Metadata(
        path=file_path,
        symbol=symbol,
        created_unix_ns=created_unix_ns,
        event_count=event_count,
        snapshot_count=event_counts[EventType.SNAPSHOT],
        depth_count=event_counts[EventType.DEPTH],
        trade_count=event_counts[EventType.TRADE],
        boundary_count=event_counts[EventType.BOUNDARY],
        checkpoint_path=checkpoint_path,
        clean_shutdown=clean_shutdown,
        data_complete=data_complete,
        final_update_id=final_update_id,
        final_state_hash=final_state_hash,
        sidecar_path=(sidecar_path if sidecar else None),
        sha256=expected_recording_hash,
        checkpoint_sha256=expected_checkpoint_hash,
        sequence_gaps=sequence_gaps,
        snapshot_retries=snapshot_retries,
        queue_drops=queue_drops,
        malformed_messages=malformed_messages,
    )

