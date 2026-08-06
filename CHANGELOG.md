# Changelog

## Unreleased

- Bound execution-sensitivity runs to the exact research report, prediction hash, held-out session IDs, recording hashes, and checkpoint hashes.
- Made top-of-book research overwrite publication restore every prior artifact after a partial commit failure.
- Reset WebSocket reconnect backoff after a confirmed successful connection.
- Unified the public runtime, Python, CMake, and vcpkg version at `1.0.0` and added an executable consistency policy.
- Corrected fixed-point type documentation and replaced obsolete first-push instructions with repository maintenance guidance.
- Fixed Python lint annotations and selected the MSVC toolchain explicitly in Windows CI.
- Made the local Windows native build activate x64 MSVC and use Ninja deterministically.
- Updated the architecture document to describe both top-of-book and sequence-correct L2 paths.
- Tightened top-of-book sidecar and binary-header type validation.
- Rejected boolean and coercible scalar values at L2 parsing boundaries.
- Required complete, hash-verified L2 recordings for execution simulation.
- Added regression tests for each validation and execution guard.

## 1.0.0 — Initial public release

- Published the source under the neutral **Market Systems Workstation** name.
- Preserved the tested C++/Python architecture and native ABI.
- Added sequence-correct L2 capture, deterministic replay, execution simulation, and chronological research workflows.
- Added cross-platform CI, native correctness tests, sanitizers, and fuzz-smoke coverage.
- Removed private workspace artifacts, generated reports, recordings, compiled binaries, and portfolio-specific notes from the public source tree.
- Reworked public documentation around engineering contracts, reproducibility, and limitations rather than job-targeting language.
