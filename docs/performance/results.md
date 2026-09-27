# Benchmark Results

These results are local component-level microbenchmarks. They are not measurements of end-to-end exchange latency, complete market-data pipeline throughput, or production trading-system performance.

## Environment

- **Git commit:** `933ef8316abda15e07a8587a0ca55dd2d9456a46`
- **CPU:** Intel Core Ultra 7 155H
- **CPU topology reported by Windows:** 16 cores / 22 logical processors
- **RAM:** 31.5 GB
- **Operating system:** Microsoft Windows 11 Home, version 10.0.26200, build 26200
- **Power source:** AC
- **Active Windows power scheme:** Balanced
- **Compiler:** MSVC 19.50.35728.0 (x64)
- **MSVC toolset path:** 14.50.35717
- **CMake:** 4.0.3 from the repository build environment
- **Generator:** Ninja
- **Build type:** Release
- **C++ standard:** C++20

The native build completed successfully and all four native tests passed before the final measurements were collected.

## Protocol

Each native benchmark used:

1. One complete warm-up run, discarded.
2. Seven measured runs.
3. The same executable and workload size for every measured run.
4. No result removal from the final seven-run set.

Earlier exploratory measurements collected on battery power under a different Windows power mode were not used in the reported medians below.

## Flat L2 order book

Command:

```powershell
.\build-native\l2_order_book_benchmark.exe 1000000
```

Each measured run applied 1,000,000 deterministic updates to both the independent `std::map` reference book and the flat sorted book. The benchmark reports throughput only after confirming that both implementations finish with the same logical state hash.

### Throughput

| Run | `std::map` updates/s | Flat updates/s |
| ---: | ---: | ---: |
| 1 | 21,933.83 | 141,250.17 |
| 2 | 18,277.12 | 139,356.86 |
| 3 | 20,757.24 | 140,058.23 |
| 4 | 20,943.20 | 138,942.72 |
| 5 | 20,642.57 | 138,933.76 |
| 6 | 15,808.68 | 138,821.81 |
| 7 | 20,934.52 | 139,294.79 |

- **Flat median:** 139,294.79 updates/s
- **Flat range:** 138,821.81–141,250.17 updates/s
- **`std::map` median:** 20,757.24 updates/s
- **Ratio of medians:** 6.71×
- **Flat batch p99 median:** 8.47 µs/update
- **Flat snapshot-install median:** 133.3 µs
- **Flat Top-20 extraction median:** 98.96 ns

The `std::map` reference showed substantially more run-to-run variance than the flat implementation. The relative figure above is therefore reported as the ratio of the two seven-run medians rather than a selected per-run speedup.

### Correctness guards

Every measured run produced:

- **Final state hash:** `6233951501269747521`
- **Anti-optimization hash:** `13980092870487377440`

The identical final state hash confirms that the reference and flat books reached the same logical state on the deterministic workload.

## SPSC ring buffer

Command:

```powershell
.\build-native\spsc_queue_benchmark.exe 1000000000
```

Each measured run transferred 1,000,000,000 records between one producer and one consumer. Each record is 32 bytes, and the benchmark observes and validates the full payload rather than measuring index movement alone.

| Run | Records/s | Payload GiB/s |
| ---: | ---: | ---: |
| 1 | 519,196,512.25 | 15.47 |
| 2 | 528,943,189.49 | 15.76 |
| 3 | 528,526,532.51 | 15.75 |
| 4 | 524,011,902.20 | 15.62 |
| 5 | 528,467,933.41 | 15.75 |
| 6 | 517,222,447.19 | 15.41 |
| 7 | 528,449,473.76 | 15.75 |

- **Median throughput:** 528,449,473.76 records/s
- **Median payload throughput:** 15.75 GiB/s
- **Throughput range:** 517,222,447.19–528,943,189.49 records/s
- **Payload checksum in every run:** `10268419940996299668`

## Interpretation

These measurements characterize isolated native components on the machine and software environment listed above. They should not be interpreted as exchange-to-strategy latency, network latency, complete ingestion throughput, or a production service-level claim.

For the measurement rules and interpretation constraints used by this repository, see [methodology.md](methodology.md).
