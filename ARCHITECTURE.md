# Architecture

## System boundary

Market Systems Workstation contains two related but deliberately separate market-data paths:

1. A native C++ top-of-book path for Binance Spot `BTCUSDT` `bookTicker` messages.
2. A sequence-correct Level-2 path for Binance diff-depth streams, REST snapshots, and aggregate trades across configured symbols.

The top-of-book path preserves a compact 32-byte ABI for ingestion, recording, replay, and online feature parity. The L2 path reconstructs an aggregated order book from a snapshot plus ordered depth updates, records exact fixed-point events, and drives deterministic replay and execution-sensitivity studies.

The system does not claim market-by-order data, exact exchange queue priority, or historical fill ground truth.

## Top-of-book live path

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

## Top-of-book recording format

The hot record remains binary format version 1 and is intentionally unchanged:

```text
BinaryFileHeader: 64 bytes
OrderBookState:    32 bytes per record
```

A current-format capture also writes:

- `<recording>.qids`: a versioned 64-bit exchange update ID for every market record.
- `<recording>.meta.json`: atomic completion metadata, counters, identity, and reconnect/drop boundaries.

The reader rejects unknown header flags, nonzero reserved header fields, partial records, malformed metadata types, identity mismatches, inconsistent counters, and invalid update-ID companions.

The metadata sidecar is created only after:

1. The producer has stopped.
2. The recorder queue has drained.
3. Both binary streams have flushed and closed.

A `.qids` file without the metadata sidecar is treated as an interrupted current-format recording. Legacy version-1 `.qbin` files without either companion remain readable, but their completeness is reported as unknown.

`data_complete` describes the recorder stream: clean shutdown, zero recorder drops, zero write errors, and accepted count equal to recorded count. Primary consumer-queue drops are reported separately because they do not imply missing records on disk.

## Top-of-book replay path

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

## L2 capture and synchronization path

```text
combined WebSocket stream
  diff depth + aggregate trades
             |
             v
     per-symbol event queue
             |
             v
     DepthSynchronizer
       1. buffer updates
       2. fetch REST snapshot
       3. discard stale updates
       4. locate bridge update
       5. apply contiguous updates
       6. resnapshot after a gap
             |
             v
      exact L2OrderBook
             |
             v
 l2bin + checkpoint file + metadata sidecar
```

Prices and quantities are stored as scaled integers. Update IDs, timestamps, trade-side booleans, and fixed-point values are type-checked before entering the book. The Python reference implementation and native implementation share the same synchronization contract and are cross-checked by tests.

A completed `.l2bin` artifact includes:

- The binary event stream.
- A checkpoint companion containing deterministic state hashes.
- An atomically published metadata sidecar containing event counts, final state identity, capture counters, and SHA-256 hashes.

A sidecar-free L2 recording is always treated as incomplete.

## L2 deterministic replay and execution path

`verify_l2_replay.py` reconstructs the book, validates sequence continuity and checkpoints, and verifies the final state identity at multiple replay speeds.

`ExecutionSimulator` consumes only complete, hash-verified L2 recordings. It applies market events and local control events in a documented timestamp order, models visible-liquidity consumption, and exposes queue assumptions as explicit sensitivity scenarios. Aggregate data cannot reveal exact queue position, cancellation attribution, or hidden liquidity, so passive-fill results are never presented as historical truth.

## Native lifecycle

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

`qbin.py` and `l2bin.py` are the canonical binary readers. They validate layout, full-record sizing, market semantics, metadata types and identity, companion files, completeness, and deterministic hashes.

`microstructure.py` owns:

- The top-of-book feature schema and schema hash.
- Offline feature generation.
- Online feature generation.
- The canonical top-of-book model artifact.

`l2_features.py` and `l2_research.py` own L2 feature extraction and chronological research over complete recordings.

`research.py` owns:

- Recording sorting and content deduplication.
- Session segmentation.
- Chronological split construction.
- Train-only normalization.
- Validation-only ridge and signal-threshold selection.
- Untouched test evaluation.
- Spread-crossing strategy diagnostics.
- Session-aware non-overlapping trade selection.
- Direction diagnostics against majority baselines.
- Test-set fingerprinting and reuse protection.
- Provenance and atomic artifact staging.

Both command-line and Tk workflows call the same underlying research functions.

## Execution and robustness diagnostics

Top-of-book research retains current and future displayed quantities as evaluation metadata without changing the native 32-byte record ABI or model feature schema. Explicit execution assumptions cover fees, slippage stress, trade size, and displayed-liquidity participation.

`research_diagnostics.py` receives already-separated train, validation, and test arrays. It does not fit production artifacts or alter final-test predictions. It provides rank correlation, prediction-quantile diagnostics, HAC inference, session-cluster bootstrap inference, within-session null tests, feature drift, coefficient stability, and extra-cost stress curves. Anchored walk-forward diagnostics remain restricted to the original train and validation sessions; the final holdout stays outside those folds.

## Latency semantics

Live `timestamp_ns` is captured immediately after a complete WebSocket message is read. The workstation therefore reports **local receipt-to-consumer age**. It does not claim exchange-to-host network latency.

Replay does not report this latency because source steady-clock timestamps may come from another boot and intentionally remain unchanged.

## Failure semantics

- Invalid network messages increment `malformed_messages` and are not published.
- A full primary queue increments `dropped_ticks`; live feature history resets when this counter changes.
- A full recorder queue increments the relevant drop counter, writes a boundary marker where supported, and makes `data_complete` false.
- A reconnect writes a connection boundary. Offline and replay feature history resets at continuity boundaries.
- An L2 sequence gap invalidates the active book and triggers resynchronization from a new snapshot.
- Binary corruption, unknown flags, nonzero reserved fields, malformed metadata types, mismatched identity, invalid fixed-point values, crossed books, partial records, zero current-format update IDs, and nonmonotonic IDs or timestamps are rejected.
- Execution simulation rejects incomplete or hash-mismatched L2 recordings before processing events.
- Failure to publish an atomic completion sidecar is surfaced as an incomplete or failed capture rather than a successful recording.
- Live connection errors are retained and exposed instead of disappearing into stderr only.

## Binding-level replay guard

The Python `start_replay` binding calls the canonical `qbin.read_metadata` validator before starting the native worker. A recording explicitly marked incomplete is rejected by default even when a caller bypasses higher-level CLI and GUI checks. The binding exposes an explicit diagnostic override used by `analytics.py`. Legacy recordings with unknown completeness remain supported, while native replay continues to validate every binary record and optional exchange update ID.
