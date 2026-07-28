
# Architecture

## System boundary

The current engine consumes Binance Spot `BTCUSDT` `bookTicker` messages. Each accepted message becomes one fixed-layout top-of-book state plus one exchange update ID.

`bookTicker` is not a multi-level depth feed. The project therefore describes itself as a top-of-book ingestion and research workstation, not an L2 reconstruction engine.

## Live path

```text
network thread
    BinanceFeed::run
      -> bounded DNS resolution
      -> verified TLS and hostname handshake
      -> WebSocket read
      -> simdjson validation
      -> OrderBookState + exchange update ID
            |                         |
            v                         v
      primary SPSC queue       recorder SPSC queue
            |                         |
            v                         v
      Python consumer          recorder writer thread
                                      |
                                      v
                         qbin + qids + metadata sidecar
```

### Thread ownership

- `IngestionEngine` owns the primary queue, recorder, feed/replay object, and native worker thread.
- The live network thread is the only producer for the primary queue.
- Python is the only permitted consumer. `IngestionEngine::consume_batch` rejects overlapping consumers.
- The live network thread is the only producer for the recorder queue.
- `BinaryRecorder::writer_loop` is the only recorder-queue consumer and the only owner of binary stream writes.
- Tk widgets are touched only from the Tk main thread. `EngineRunner` communicates through `queue.Queue` events.

### SPSC memory ordering

Producer publication uses a release store to the tail index. The consumer acquires the tail before reading published values. Consumer progress uses a release store to the head, and the producer acquires the head before reusing capacity.

The queue is correct only for one producer and one consumer. That contract is enforced at the engine binding instead of being left as a convention.

## Recording format

The hot record remains binary format version 1 and is intentionally unchanged:

```text
BinaryFileHeader: 64 bytes
OrderBookState:    32 bytes per record
```

A current-format capture also writes:

- `<recording>.qids`: a versioned 64-bit exchange update ID for every market record.
- `<recording>.meta.json`: atomic completion metadata, counters, hashes inputs, and reconnect/drop boundaries.

The recorder refuses to overwrite any member of the artifact set.

The metadata sidecar is created only after:

1. The producer has stopped.
2. The recorder queue has drained.
3. Both binary streams have flushed and closed.

A `.qids` file without the metadata sidecar is treated as an interrupted current-format recording. Legacy version-1 `.qbin` files without either companion remain readable, but their completeness is reported as unknown.

`data_complete` describes the recorder stream: clean shutdown, zero recorder drops, zero write errors, and accepted count equal to recorded count. Primary consumer-queue drops are reported separately because they do not imply missing records on disk.

## Replay path

```text
qbin + optional qids
        |
        v
BinaryReplay validation
        |
        v
wall-clock scheduler using source timestamp deltas
        |
        v
unchanged OrderBookState -> primary SPSC queue -> Python
```

Replay scheduling and record data are separate. `timestamp_ns` is never replaced with replay wall-clock time. Therefore speed changes affect only delivery timing, not the record stream or time-based research features.

`verify_replay.py` hashes the consumed record bytes and compares online features across replay speeds.

## Lifecycle

`IngestionEngine` serializes lifecycle transitions and exposes these states:

```text
stopped -> starting -> live/replaying -> stopping -> stopped
                                  \-> completed
                                  \-> failed
```

Before a new start, a naturally completed worker is joined. Assigning over a joinable `std::thread` is never permitted.

`stop()`:

1. Marks the engine stopping.
2. Requests live DNS/socket cancellation when applicable.
3. Joins the producer worker.
4. Supplies final feed counters to the recorder.
5. Drains and joins the recorder writer.
6. Clears mode objects and returns to stopped.

`stop()` is idempotent. The binding releases the Python GIL while waiting.

## Research data flow

`qbin.py` is the canonical binary reader. It validates header layout, full-record sizing, market semantics, update-ID semantics, metadata consistency, and contiguous-session boundaries.

`microstructure.py` owns:

- The 16-feature schema and schema hash.
- Offline feature generation.
- Online feature generation.
- The canonical model artifact.

`research.py` owns:

- Recording sorting and content deduplication.
- Session segmentation.
- Chronological split construction.
- Train-only normalization.
- Validation-only ridge and signal-threshold selection.
- Untouched test evaluation.
- Spread-crossing strategy diagnostics.
- Session-aware non-overlapping trade selection.
- OBI baseline and session bootstrap interval.
- Test-set fingerprinting and reuse protection.
- Provenance and atomic artifact staging.

Both `train_alpha.py` and the Tk Research Lab call the same functions. Both `alpha_monitor.py` and the Tk Alpha Runtime load the same `AlphaModel` schema.

## Latency semantics

Live `timestamp_ns` is captured immediately after a complete WebSocket message is read. The workstation therefore reports **local receipt-to-consumer age**. It does not claim exchange-to-host network latency.

Replay does not report this latency because source steady-clock timestamps may come from another boot and intentionally remain unchanged.

## Failure semantics

- Invalid network messages increment `malformed_messages` and are not published.
- A full primary queue increments `dropped_ticks`; live feature history resets when this counter changes.
- A full recorder queue increments `recording_dropped`, writes a boundary marker, and makes `data_complete` false.
- A reconnect writes a `connection_start` boundary. Offline and replay feature history resets at those boundaries.
- Binary corruption, unknown header flags, mismatched sidecar market identity, invalid prices, crossed markets, truncated records, zero current-format update IDs, and nonmonotonic IDs/timestamps are rejected.
- Failure to finalize the atomic metadata sidecar is surfaced through `recording_write_errors`; capture tools fail rather than reporting a successful complete recording.
- Live connection errors are retained and exposed instead of disappearing into stderr only.


## Execution and robustness layer

The canonical `FeatureSet` now retains current and future displayed top-of-book quantities in addition to model features and prices. These arrays are research metadata and do not alter the native 32-byte `OrderBookState` ABI or the saved model feature schema.

`ExecutionAssumptions` defines:

- Explicit fee in basis points per side.
- Fixed slippage stress in basis points per side.
- Optional base-asset trade size.
- Maximum fraction of displayed top-of-book quantity available to the screening rule.

`strategy_trades` returns gross PnL, net PnL, selected rows, session IDs, side, displayed capacity, and fill-screen rejections. Non-overlapping trade selection remains session-aware.

The robustness layer is implemented in `research_diagnostics.py` and receives already-separated train/validation/test arrays. It does not fit production artifacts or change final-test predictions. It provides:

- Rank IC and prediction-quantile diagnostics.
- HAC inference and session-cluster bootstrap inference.
- Within-session circular-shift null tests.
- Feature drift and training coefficient stability.
- Extra-cost stress curves.
- Strict Markdown/JSON evidence rendering.

Anchored walk-forward diagnostics are restricted to the union of the original train and validation sessions. The final holdout remains outside those folds.

## Binding-level replay guard

The Python `start_replay` binding calls the canonical `qbin.read_metadata` validator before starting the native worker. A recording explicitly marked incomplete is rejected by default even when a caller bypasses the higher-level CLI and GUI checks. The binding exposes an explicit `allow_incomplete` diagnostic override used by `analytics.py`. Legacy recordings with unknown completeness remain supported, while native replay continues to validate every binary record and optional exchange update ID.
