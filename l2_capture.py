from __future__ import annotations

import argparse
import json
import math
import queue
import ssl
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import certifi

from l2bin import Boundary, BoundaryReason, L2Writer
from project_version import PROJECT_USER_AGENT
from l2book import (
    ApplyResult,
    DepthSynchronizer,
    DepthUpdate,
    Snapshot,
    SnapshotResult,
    SyncState,
    Trade,
    parse_levels,
    parse_price,
    parse_quantity,
)


@dataclass(frozen=True, slots=True)
class ConnectionBoundary:
    receipt_timestamp_ns: int
    generation: int
    connected: bool


StreamEvent = DepthUpdate | Trade | ConnectionBoundary


def monotonic_ns() -> int:
    return time.monotonic_ns()



def _required_json_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise ValueError(f"Stream field {key!r} must be an integer.")
    return value


def _required_json_bool(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Stream field {key!r} must be a boolean.")
    return value


def parse_stream_message(raw_message: str, receipt_timestamp_ns: int) -> tuple[str, DepthUpdate | Trade]:
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError as exception:
        raise ValueError("WebSocket message is not valid JSON.") from exception
    if not isinstance(payload, dict):
        raise ValueError("WebSocket message must contain a JSON object.")
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise ValueError("Combined stream payload does not contain an object.")
    event_type = data.get("e")
    symbol_value = data.get("s")
    if not isinstance(event_type, str):
        raise ValueError("Stream event type must be a string.")
    if not isinstance(symbol_value, str) or not symbol_value:
        raise ValueError("Stream event does not name a valid symbol.")
    symbol = symbol_value.upper()
    if event_type == "depthUpdate":
        return symbol, DepthUpdate(
            receipt_timestamp_ns=receipt_timestamp_ns,
            event_time_ms=_required_json_int(data, "E"),
            first_update_id=_required_json_int(data, "U"),
            final_update_id=_required_json_int(data, "u"),
            bids=parse_levels(data["b"]),
            asks=parse_levels(data["a"]),
        )
    if event_type == "aggTrade":
        return symbol, Trade(
            receipt_timestamp_ns=receipt_timestamp_ns,
            event_time_ms=_required_json_int(data, "E"),
            aggregate_trade_id=_required_json_int(data, "a"),
            price=parse_price(data["p"]),
            quantity=parse_quantity(data["q"]),
            buyer_is_maker=_required_json_bool(data, "m"),
        )
    raise ValueError(f"Unsupported stream event type: {event_type!r}")


def fetch_snapshot(symbol: str, *, limit: int = 5_000, timeout_seconds: float = 10.0) -> Snapshot:
    if limit not in (5, 10, 20, 50, 100, 500, 1_000, 5_000):
        raise ValueError("Unsupported Binance depth snapshot limit.")
    query = urllib.parse.urlencode({"symbol": symbol.upper(), "limit": limit})
    url = f"https://api.binance.com/api/v3/depth?{query}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": PROJECT_USER_AGENT},
        method="GET",
    )
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=timeout_seconds, context=context) as response:
        if response.status != 200:
            raise RuntimeError(f"Snapshot request failed with HTTP {response.status}.")
        raw = response.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exception:
        raise RuntimeError("Snapshot endpoint returned invalid JSON.") from exception
    if not isinstance(payload, dict) or "lastUpdateId" not in payload:
        raise RuntimeError(f"Snapshot endpoint returned an error: {payload!r}")
    return Snapshot(
        receipt_timestamp_ns=monotonic_ns(),
        last_update_id=_required_json_int(payload, "lastUpdateId"),
        bids=parse_levels(payload.get("bids", ())),
        asks=parse_levels(payload.get("asks", ())),
    )


@dataclass(slots=True)
class ReconnectBackoff:
    initial_delay_seconds: float = 0.5
    maximum_delay_seconds: float = 10.0
    current_delay_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.initial_delay_seconds)
            or self.initial_delay_seconds <= 0.0
        ):
            raise ValueError("Initial reconnect delay must be finite and positive.")
        if (
            not math.isfinite(self.maximum_delay_seconds)
            or self.maximum_delay_seconds < self.initial_delay_seconds
        ):
            raise ValueError(
                "Maximum reconnect delay must be finite and no smaller than the initial delay."
            )
        self.current_delay_seconds = self.initial_delay_seconds

    def consume(self) -> float:
        delay = self.current_delay_seconds
        self.current_delay_seconds = min(
            self.current_delay_seconds * 2.0,
            self.maximum_delay_seconds,
        )
        return delay

    def reset(self) -> None:
        self.current_delay_seconds = self.initial_delay_seconds


class CombinedStreamClient(threading.Thread):
    def __init__(
        self,
        symbols: tuple[str, ...],
        output_queue: queue.Queue[tuple[str | None, StreamEvent]],
        *,
        queue_drop_callback: Callable[[], None],
    ) -> None:
        super().__init__(name="binance-l2-stream", daemon=False)
        self.symbols = symbols
        self.output_queue = output_queue
        self.queue_drop_callback = queue_drop_callback
        self.stop_event = threading.Event()
        self._websocket = None
        self.generation = 0
        self.malformed_messages = 0
        self.last_error = ""

    def request_stop(self) -> None:
        self.stop_event.set()
        websocket = self._websocket
        if websocket is not None:
            try:
                websocket.close()
            except Exception:
                pass

    def _emit(self, symbol: str | None, event: StreamEvent) -> None:
        try:
            self.output_queue.put_nowait((symbol, event))
        except queue.Full:
            self.queue_drop_callback()

    def run(self) -> None:
        try:
            import websocket
        except ImportError:
            self.last_error = (
                "websocket-client is required for L2 capture. "
                "Install requirements-desktop.txt."
            )
            return
        stream_names = [
            name
            for symbol in self.symbols
            for name in (f"{symbol.lower()}@depth@100ms", f"{symbol.lower()}@aggTrade")
        ]
        target = "wss://stream.binance.com:9443/stream?streams=" + "/".join(stream_names)
        reconnect_backoff = ReconnectBackoff()
        while not self.stop_event.is_set():
            self.generation += 1
            generation = self.generation

            def on_open(_ws) -> None:
                reconnect_backoff.reset()
                self._emit(
                    None,
                    ConnectionBoundary(monotonic_ns(), generation, True),
                )

            def on_message(_ws, message: str) -> None:
                receipt = monotonic_ns()
                try:
                    symbol, event = parse_stream_message(message, receipt)
                except Exception:
                    self.malformed_messages += 1
                    return
                if symbol in self.symbols:
                    self._emit(symbol, event)

            def on_error(_ws, error: object) -> None:
                self.last_error = str(error)

            def on_close(_ws, _status, _message) -> None:
                self._emit(
                    None,
                    ConnectionBoundary(monotonic_ns(), generation, False),
                )

            self._websocket = websocket.WebSocketApp(
                target,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            try:
                self._websocket.run_forever(
                    sslopt={
                        "cert_reqs": ssl.CERT_REQUIRED,
                        "ca_certs": certifi.where(),
                        "check_hostname": True,
                    },
                    ping_interval=15,
                    ping_timeout=10,
                )
            except Exception as exception:
                self.last_error = str(exception)
            self._websocket = None
            if self.stop_event.is_set():
                break
            reconnect_delay = reconnect_backoff.consume()
            deadline = time.monotonic() + reconnect_delay
            while time.monotonic() < deadline and not self.stop_event.is_set():
                time.sleep(0.05)


@dataclass(slots=True)
class SymbolCapture:
    symbol: str
    writer: L2Writer
    synchronizer: DepthSynchronizer = field(default_factory=DepthSynchronizer)
    sequence_gaps: int = 0
    snapshot_retries: int = 0
    discarded_pre_snapshot_trades: int = 0
    depth_since_checkpoint: int = 0
    last_snapshot: Snapshot | None = None
    snapshot_fetcher: Callable[[str], Snapshot] = fetch_snapshot
    snapshot_attempt_limit: int = 20

    def connection_boundary(self, receipt_timestamp_ns: int, connected: bool) -> None:
        reason = (
            BoundaryReason.CONNECTION_START if connected else BoundaryReason.CONNECTION_END
        )
        self.writer.write(Boundary(receipt_timestamp_ns, reason))
        self.synchronizer.reset()
        self.last_snapshot = None

    def _record_applied(self, update: DepthUpdate) -> None:
        self.writer.write(update)
        self.depth_since_checkpoint += 1
        if self.depth_since_checkpoint >= self.writer.checkpoint_interval:
            self.writer.write_checkpoint(
                self.synchronizer.book.last_update_id,
                self.synchronizer.book.state_hash(),
            )
            self.depth_since_checkpoint = 0

    def synchronize(self) -> bool:
        if not self.synchronizer.buffered_events:
            return False
        for _attempt in range(self.snapshot_attempt_limit):
            snapshot = self.snapshot_fetcher(self.symbol)
            result = self.synchronizer.install_snapshot(snapshot)
            if result.result is SnapshotResult.SNAPSHOT_TOO_OLD:
                self.snapshot_retries += 1
                self.writer.write(
                    Boundary(monotonic_ns(), BoundaryReason.SNAPSHOT_RETRY)
                )
                continue
            if result.result is SnapshotResult.GAP_DETECTED:
                self.sequence_gaps += 1
                self.writer.write(Boundary(monotonic_ns(), BoundaryReason.SEQUENCE_GAP))
                self.synchronizer.reset(preserve_buffer=True)
                continue
            if result.result is SnapshotResult.AWAITING_BRIDGE:
                self.last_snapshot = snapshot
                return False
            self.writer.write(snapshot)
            self.last_snapshot = snapshot
            for update in result.applied_events:
                self._record_applied(update)
            return True
        raise RuntimeError(
            f"Could not synchronize {self.symbol} after "
            f"{self.snapshot_attempt_limit} snapshot attempts."
        )

    def on_depth(self, update: DepthUpdate) -> None:
        result = self.synchronizer.ingest(update)
        if result is ApplyResult.APPLIED:
            self._record_applied(update)
        elif result is ApplyResult.GAP_DETECTED:
            self.sequence_gaps += 1
            self.writer.write(Boundary(update.receipt_timestamp_ns, BoundaryReason.SEQUENCE_GAP))
            self.synchronizer.reset(preserve_buffer=True)
            self.synchronize()
        elif result is ApplyResult.BUFFERED:
            self.synchronize()

    def on_trade(self, trade: Trade) -> None:
        if self.synchronizer.state is SyncState.LIVE:
            self.writer.write(trade)
        else:
            self.discarded_pre_snapshot_trades += 1


def capture(
    symbols: tuple[str, ...],
    output_directory: Path,
    *,
    duration_seconds: float,
    checkpoint_interval: int,
    queue_capacity: int,
) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    captures = {
        symbol: SymbolCapture(
            symbol,
            L2Writer(
                output_directory / f"{symbol.lower()}-{stamp}.l2bin",
                symbol,
                checkpoint_interval=checkpoint_interval,
            ),
        )
        for symbol in symbols
    }
    event_queue: queue.Queue[tuple[str | None, StreamEvent]] = queue.Queue(
        maxsize=queue_capacity
    )
    queue_drops = 0
    queue_drop_lock = threading.Lock()

    def count_queue_drop() -> None:
        nonlocal queue_drops
        with queue_drop_lock:
            queue_drops += 1

    client = CombinedStreamClient(
        symbols,
        event_queue,
        queue_drop_callback=count_queue_drop,
    )
    failure: BaseException | None = None
    graceful_interrupt = False
    started = time.monotonic()
    last_report = started
    try:
        client.start()
        while True:
            if duration_seconds > 0 and time.monotonic() - started >= duration_seconds:
                break
            if not client.is_alive():
                detail = client.last_error or "WebSocket capture thread stopped unexpectedly."
                raise RuntimeError(detail)
            try:
                symbol, event = event_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if isinstance(event, ConnectionBoundary):
                for state in captures.values():
                    state.connection_boundary(event.receipt_timestamp_ns, event.connected)
                continue
            if symbol is None or symbol not in captures:
                continue
            state = captures[symbol]
            if isinstance(event, DepthUpdate):
                state.on_depth(event)
            else:
                state.on_trade(event)
            now = time.monotonic()
            if now - last_report >= 5.0:
                summary = " | ".join(
                    f"{item.symbol}: update={item.synchronizer.book.last_update_id:,} "
                    f"events={item.writer.event_count:,} gaps={item.sequence_gaps:,}"
                    for item in captures.values()
                )
                print(f"[L2 Capture] {summary} | queue_drops={queue_drops:,}")
                last_report = now
    except KeyboardInterrupt:
        graceful_interrupt = True
        print("\n[L2 Capture] Graceful stop requested.")
    except BaseException as exception:
        failure = exception
    finally:
        client.request_stop()
        if client.ident is not None:
            client.join(timeout=15.0)
        if client.is_alive() and failure is None:
            failure = RuntimeError(
                "WebSocket capture thread did not stop within the shutdown deadline."
            )

    clean_shutdown = failure is None and not client.is_alive()
    output_paths: list[Path] = []
    finalization_errors: list[BaseException] = []
    for state in captures.values():
        try:
            if queue_drops > 0:
                state.writer.write(
                    Boundary(monotonic_ns(), BoundaryReason.QUEUE_OVERFLOW)
                )
            state.writer.write(Boundary(monotonic_ns(), BoundaryReason.USER_STOP))
            final_update_id = state.synchronizer.book.last_update_id
            final_state_hash = (
                state.synchronizer.book.state_hash() if final_update_id > 0 else 0
            )
            if final_update_id > 0:
                state.writer.write_checkpoint(final_update_id, final_state_hash)
            metadata = state.writer.finalize(
                final_update_id=final_update_id,
                final_state_hash=final_state_hash,
                sequence_gaps=state.sequence_gaps,
                snapshot_retries=state.snapshot_retries,
                queue_drops=queue_drops,
                malformed_messages=client.malformed_messages,
                clean_shutdown=clean_shutdown,
                extra={
                    "discarded_pre_snapshot_trades": (
                        state.discarded_pre_snapshot_trades
                    ),
                    "websocket_last_error": client.last_error,
                    "graceful_keyboard_interrupt": graceful_interrupt,
                    "capture_failure": (
                        f"{type(failure).__name__}: {failure}"
                        if failure is not None
                        else None
                    ),
                },
            )
            output_paths.append(metadata.path)
            print(
                f"[L2 Capture] {state.symbol}: events={metadata.event_count:,} "
                f"depth={metadata.depth_count:,} trades={metadata.trade_count:,} "
                f"complete={metadata.data_complete} path={metadata.path}"
            )
        except BaseException as exception:
            state.writer.abort()
            finalization_errors.append(exception)

    if finalization_errors:
        messages = "; ".join(
            f"{type(error).__name__}: {error}" for error in finalization_errors
        )
        if failure is not None:
            raise RuntimeError(
                f"Capture failed ({failure}) and artifact finalization also failed: "
                f"{messages}"
            ) from failure
        raise RuntimeError(f"L2 artifact finalization failed: {messages}")
    if failure is not None:
        raise failure
    return output_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture sequence-correct Binance Spot L2 depth and aggregate trades."
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--output-dir", default="recordings/l2")
    parser.add_argument("--duration-minutes", type=float, default=0.0)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--queue-capacity", type=int, default=200_000)
    arguments = parser.parse_args()
    if not math.isfinite(arguments.duration_minutes) or arguments.duration_minutes < 0.0:
        parser.error("--duration-minutes must be finite and non-negative.")
    if arguments.checkpoint_interval <= 0 or arguments.queue_capacity <= 0:
        parser.error("Checkpoint interval and queue capacity must be positive.")
    arguments.symbols = tuple(dict.fromkeys(value.strip().upper() for value in arguments.symbols))
    if not arguments.symbols or any(not symbol.isalnum() for symbol in arguments.symbols):
        parser.error("Symbols must be non-empty alphanumeric Binance symbols.")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    capture(
        arguments.symbols,
        Path(arguments.output_dir).expanduser().resolve(),
        duration_seconds=arguments.duration_minutes * 60.0,
        checkpoint_interval=arguments.checkpoint_interval,
        queue_capacity=arguments.queue_capacity,
    )


if __name__ == "__main__":
    main()
