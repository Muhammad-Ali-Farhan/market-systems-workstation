# Changelog

## Unreleased

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
