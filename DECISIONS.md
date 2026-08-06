
# Engineering Decisions

## One supported desktop stack

Tk/ttk is the supported UI. The obsolete Qt and process-managed dashboard layers are removed so installation, documentation, and artifact discovery describe one application.

## Preserve the 32-byte market ABI

`OrderBookState` and binary format version 1 remain unchanged. Exchange update IDs and recording/session metadata are stored in coordinated companion files rather than silently changing existing payload records.

## Source timestamps are immutable

Replay scheduling uses recorded timestamp deltas but never writes replay wall-clock time into a market record. This is required for deterministic time-dependent features.

## Recording completion is explicit

Current captures use `.qbin`, `.qids`, and an atomic `.meta.json` sidecar. The sidecar appears only after queue drain and stream close. Its absence is observable evidence of interruption.

## Lifecycle correctness over lock-free lifecycle control

The data path remains SPSC and lock-free. Start/stop/restart transitions use mutexes because lifecycle operations are rare and correctness is more valuable than avoiding a non-contentious lock.

## One research and artifact implementation

CLI and GUI call the same feature, split, model, evaluation, and serialization code. A shared schema version and feature hash prevent incompatible `.npz` files from sharing a filename convention.

## Whole sessions before event fractions

When enough sessions exist, complete early sessions train the model, following sessions validate it, and the latest sessions form the test set. A purged event-level fallback exists only for limited data.

## Holdout reuse is not silent

Exact test examples are hashed. Reusing them is refused by default and requires an explicit reproducibility override recorded in provenance.

## Honest metrics

The live latency metric is named local receipt-to-consumer age. Strategy output is called an execution-adjusted diagnostic. The ordinary t-statistic is labeled naive and accompanied by a session-bootstrap interval.

## Portable release builds by default

`/arch:AVX2` and `-march=native` are opt-in through `MARKET_ENABLE_NATIVE_ARCH`. Default artifacts do not assume a development machine's instruction set.


## Robust inference without a heavyweight dependency stack

The project keeps NumPy as the only numerical runtime dependency. HAC inference, session bootstrap, circular-shift testing, rank IC, feature drift, and coefficient stability are implemented directly and tested deterministically. This keeps installation predictable while making the methodology auditable.

## Explicit execution assumptions, not hidden optimism

Gross spread-crossing PnL is stored separately from explicit fees and fixed slippage. A displayed-liquidity screen is optional and is never described as a fill or impact model. Cost stress is applied to a fixed selected trade set so the effect of assumptions is visible without silently reselecting the model.

## Stable least squares over normal equations

Ridge fitting uses an augmented least-squares system. With only sixteen features the runtime difference is negligible, while numerical behavior is more defensible than solving a regularized Gram matrix directly.

## Strict JSON artifacts

Non-finite diagnostics are represented as JSON `null`. Reports never rely on Python's non-standard `NaN` encoding.

## Aggregated L2, not market-by-order

The final-stage pipeline reconstructs sequence-correct aggregated price levels. It does not claim exact order priority. Aggregate trades are recorded separately, and passive execution is reported across explicit queue-model sensitivity cases.

## Exact fixed point for L2 identity

L2 prices and quantities use eight-decimal fixed-point integers. Floating point remains appropriate for statistical features, but not for price-level keys, deterministic hashes, or binary identity.

## Reference and production books

A `std::map` book is retained as an independent correctness oracle. A flat sorted book is the production candidate. Randomized equivalence and identical final hashes are required before performance comparisons matter.

## Event format separate from top-of-book ABI

The 32-byte top-of-book ABI and qbin v1 remain unchanged. Variable-length L2 snapshots, deltas, trades, boundaries, and checkpoints use a distinct versioned format so compatibility is explicit.

## Fills are sensitivity, not fact

The default passive model requires observed aggregate trades. Depth-depletion models are named and reported as sensitivity assumptions. No output may describe estimated queue position as exchange truth.

## Execution sensitivity is provenance-bound

A sensitivity run must include the L2 research report that produced its prediction CSV. The report hash, prediction hash, held-out session IDs, recording hashes, checkpoint hashes, and symbol are verified before simulation. A manually remapped session cannot silently become new evidence.

## Overwrite publication is transactional

Top-of-book research artifacts are fully staged first. When explicit overwrite is enabled, every previous destination is moved to a same-directory backup before any new artifact is published. A partial publication failure removes new files and restores the complete prior set; the report remains the final commit marker.

## Public version and schema versions are separate

The application version is shared by the Python package, CMake project, vcpkg manifest, and runtime User-Agent and is enforced by tests. Binary-format, feature, model, and report schema versions remain independent compatibility contracts.
