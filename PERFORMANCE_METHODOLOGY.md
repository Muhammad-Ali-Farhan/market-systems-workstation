# Performance and Benchmark Methodology

## Principles

A benchmark result is evidence only when its environment and method are reproducible. Never mix Debug and Release results or quote another machine’s output as local evidence.

## Required provenance

Record:

- Git commit.
- Compiler and version.
- CMake generator and build type.
- CPU, logical cores, RAM, operating system, and power mode.
- Python and NumPy versions for Python benchmarks.
- Recording SHA-256 and event counts.
- Trial count and warmup count.
- Whether antivirus, debugger, profiler, and recording were enabled.

## Native book benchmark

`l2_book_benchmark` compares the independent `std::map` reference and flat sorted book using identical seeded mutations. It reports:

- Snapshot rebuild time.
- Sustained update throughput.
- Per-update batch p99.
- Top-N extraction cost.
- Final state-hash equality.

The reference implementation is a correctness baseline, not expected to win.

## Pipeline benchmark

`l2_benchmark.py` measures:

- Binary parse/validation.
- Deterministic reconstruction.
- Streaming L2 feature generation.
- Peak Python memory where available.

Use at least one warmup and seven measured trials. Report median and range. For tail latency, collect independent operation batches instead of deriving p99 from a single wall-clock total.

## Profiling sequence

1. Establish deterministic correctness.
2. Benchmark a Release build.
3. Profile the measured bottleneck.
4. Make one controlled change.
5. Re-run correctness and benchmark tests.
6. Keep the change only when evidence supports it.

Potential controlled comparisons include flat versus map books, snapshot size, active depth, event batch size, queue capacity, parser mode, feature calculation location, and recording on/off.

## Claims

Use precise language:

- “local receipt-to-consumer age,” not network latency.
- “updates per second on machine X,” not universally low latency.
- “p99 within this benchmark distribution,” not a production SLA.
- “zero allocations in measured mutation path” only after profiler/allocation evidence.
