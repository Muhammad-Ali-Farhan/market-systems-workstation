# System Design Walkthrough

This document explains the design contracts that matter most when reviewing or maintaining the platform. It is written as an engineering reference, not as a script to memorize.

## 1. Sequence-correct L2 synchronization

The depth stream and REST snapshot are separate data sources, so synchronization must establish one continuous update sequence before the book is trusted.

1. Connect to the diff-depth stream and buffer events immediately.
2. Fetch the REST snapshot while buffering continues.
3. Discard buffered events whose final update ID is not newer than the snapshot.
4. Require the first retained event to bridge `lastUpdateId + 1`.
5. Apply later events only while update ranges remain continuous.
6. On a gap or reconnect, mark the continuity boundary, cancel continuity-dependent simulated orders, reset streaming features, and acquire a new snapshot.

The bridge condition prevents an off-by-one error in which the first post-snapshot event is accepted even though one or more updates are missing.

## 2. Exact price-level identity

Price and quantity values are parsed into unsigned 64-bit fixed-point integers with eight decimal places. Floating point is avoided for book keys because decimal exchange prices that look equal can have different binary floating-point representations.

The parser rejects:

- scientific notation;
- negative or non-finite values;
- more than eight decimal places;
- values that overflow 64-bit storage.

Conversion to floating point occurs only at research and presentation boundaries.

## 3. Reference and production books

`MapOrderBook` is the correctness-oriented reference implementation. `FlatOrderBook` is the cache-oriented production representation.

The map provides straightforward ordered semantics and logarithmic updates. The flat representation improves locality and top-N traversal but can require shifting elements. Randomized tests apply identical mutation streams to both implementations and compare complete state hashes. This gives evidence of behavioral equivalence without assuming either implementation is self-validating.

## 4. Deterministic recording and replay

The recorder persists snapshots, depth deltas, aggregate trades, continuity boundaries, and state checkpoints. Each event has a CRC for local corruption detection. Completed artifacts also carry SHA-256 hashes for whole-file identity.

Replay preserves source event contents and receipt timestamps. Logical replay speed controls delivery timing only. Verification checks intermediate checkpoints and the final book hash, which localizes divergence instead of merely reporting that the final state differs.

CRC and SHA-256 serve different purposes:

- CRC detects accidental record corruption cheaply while streaming.
- SHA-256 identifies the complete finalized artifact and supports provenance.

## 5. SPSC publication and lifecycle ownership

The top-of-book hot path uses a single-producer/single-consumer ring buffer.

- The producer writes the element, then publishes the tail with a release store.
- The consumer acquires the tail before reading the published element.
- The consumer publishes head progress with a release store.
- The producer acquires the head before reusing capacity.

This contract is correct only for one producer and one consumer. The Python binding rejects overlapping consumers. Start, stop, and restart use ordinary locks because lifecycle transitions are rare and correctness matters more than making control flow lock-free.

## 6. Execution-model boundaries

Aggregated depth cannot reveal exact FIFO queue position, hidden liquidity, or whether every depth reduction was a cancellation or execution. The simulator therefore exposes assumptions rather than presenting them as observations.

- `trade_only` permits passive fills only after observed aggregate trades deplete queue-ahead estimates.
- `pro_rata_depth` treats a configurable portion of qualifying depth reduction as fill sensitivity.
- `optimistic_depth` is an explicitly optimistic upper-bound case.

Market orders sweep displayed levels and cannot reuse liquidity already consumed before the next book update. Limit orders support decision, transmission, cancellation, and expiration latency; own simulated orders at the same price are ordered FIFO.

## 7. Research-discipline contract

The research pipeline separates engineering evidence from profitability claims.

- Complete sessions are ordered chronologically.
- Normalization is fitted only on training data.
- Ridge regularization and signal threshold are selected only on validation data.
- The final test set is fingerprinted and untouched until evaluation.
- Exact holdout reuse is refused by default and must be explicitly recorded.
- Direction accuracy is reported beside its majority baseline, lift, balanced accuracy, actionable coverage, and zero-return prevalence.
- Economic diagnostics include explicit costs, latency sensitivity, trade count, HAC inference, session bootstrap, circular-shift tests, and baseline comparisons.

A high predictive metric is not treated as executable alpha unless it survives spread, fees, latency, fill uncertainty, and independent sessions.

## 8. Failure semantics

The system rejects or marks incomplete artifacts when it encounters:

- sequence gaps;
- queue overflow;
- malformed network messages;
- truncated or corrupt binary records;
- checkpoint mismatch;
- recorder write failure;
- interrupted finalization;
- mixed-symbol or duplicate research inputs.

The design prefers an explicit failure over silently accepting ambiguous market state.
