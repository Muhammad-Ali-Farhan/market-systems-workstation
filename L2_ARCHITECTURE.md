# Sequence-Correct L2 Architecture

## Purpose

The L2 subsystem reconstructs Binance Spot aggregated depth from a REST snapshot and diff-depth update ranges, records the exact logical event stream, and proves deterministic reconstruction through state checkpoints.

## Components

### `L2Types.hpp` / `l2book.py`

- Signed 64-bit price and unsigned 64-bit quantity fixed point.
- Scale: `100,000,000` units per decimal unit.
- Exact decimal parsing; excess precision and overflow are rejected.
- Snapshot, depth-update, trade, and level types.

### `L2Book.hpp`

- `MapOrderBook`: independent correctness reference.
- `FlatOrderBook`: sorted contiguous production representation.
- Zero quantity deletes a level.
- Bids are strictly descending; asks strictly ascending.
- Two-sided books must satisfy `best_bid < best_ask`.
- Cross-language FNV-1a state hash covers update ID and every ordered level.

### `L2Synchronizer.hpp` / `DepthSynchronizer`

State machine:

```text
awaiting_snapshot -> live -> gap -> awaiting_snapshot
```

Contract:

1. Buffer updates while no synchronized snapshot exists.
2. Discard buffered updates with `u <= lastUpdateId`.
3. Require the first retained range to contain `lastUpdateId + 1`.
4. Apply future events only when `U <= local_id + 1 <= u`.
5. Ignore stale events with `u <= local_id`.
6. Enter gap state and resynchronize when `U > local_id + 1`.

A connection boundary clears book and feature state. A sequence gap is recorded before resynchronization.

### `l2_capture.py`

- Combined diff-depth and aggregate-trade WebSocket streams.
- One sequence state and artifact writer per symbol.
- Bounded cross-thread event queue.
- Verified TLS using the certifi CA bundle.
- REST snapshots with bounded timeout.
- A fetched snapshot is retained while awaiting its bridge event; queued depth events are tested against that same snapshot before another REST request is made.
- Atomic sidecar finalization.
- Capture failures still close and finalize artifacts as incomplete.

### `l2bin.py` / `L2BinaryFormat.hpp`

- Fixed 128-byte file header.
- Variable-length 80-byte event header plus fixed 16-byte levels.
- CRC32 over every level payload.
- Separate checkpoint file containing event index, update ID, and state hash.
- Atomic JSON sidecar with hashes, counters, completeness, and provenance.

### Replay

`verify_l2_replay.py` applies the exact event stream, verifies checkpoints after their recorded event indices, verifies the final metadata state, and compares the state-hash sequence across replay speeds.

### Feature and research path

Only synchronized depth events produce model observations. Boundaries reset history, and labels never cross session boundaries. `l2_research.py` performs chronological model selection and holds the final test partition untouched until evaluation.

### Execution path

`execution_simulator.py` consumes the same event stream. Market orders interact with current visible depth. Passive orders use aggregate trades plus explicit queue assumptions. Book boundaries cancel active orders because continuity is lost.

## Thread ownership

- `CombinedStreamClient` owns the WebSocket callback thread.
- The capture/main thread is the sole consumer of the bounded Python queue.
- Each `SymbolCapture` is mutated only by the capture thread.
- Each `L2Writer` has exactly one caller.
- The native `L2Synchronizer` binding is not advertised as internally thread-safe; callers must serialize access.

## Failure semantics

- Queue overflow: counter + boundary + incomplete artifact.
- Malformed stream message: counter + incomplete artifact.
- Depth sequence gap: boundary + snapshot resynchronization.
- Snapshot too old: retry boundary and bounded retry loop.
- Buffer overflow/out-of-order update: hard failure and incomplete artifact.
- CRC/truncation/header mismatch: replay rejection.
- Missing sidecar with checkpoint companion: explicitly interrupted/incomplete.
- Final checkpoint mismatch: recording rejection.
