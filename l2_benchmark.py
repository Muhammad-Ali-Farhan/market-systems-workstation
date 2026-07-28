from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from l2bin import Boundary, iter_events, read_metadata
from l2_features import build_feature_set
from l2book import DepthUpdate, L2OrderBook, Snapshot, Trade

T = TypeVar("T")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark L2 binary parsing, reconstruction, and feature generation."
    )
    parser.add_argument("recording")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", default="artifacts/l2_benchmark.json")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.horizon <= 0:
        parser.error("--horizon must be positive.")
    if arguments.trials < 3:
        parser.error("--trials must be at least 3.")
    return arguments


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _timed(function: Callable[[], T], trials: int) -> tuple[list[float], T]:
    latest = function()
    values: list[float] = []
    for _ in range(trials):
        started = time.perf_counter_ns()
        latest = function()
        values.append((time.perf_counter_ns() - started) / 1e9)
    return values, latest


def _summary(values: list[float], units: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "trials": int(array.size),
        "units": int(units),
        "median_seconds": median,
        "minimum_seconds": float(np.min(array)),
        "maximum_seconds": float(np.max(array)),
        "median_units_per_second": units / max(median, 1e-12),
    }


def _parse(path: Path) -> tuple[int, int, int]:
    events = depth = trades = 0
    for event in iter_events(path):
        events += 1
        depth += isinstance(event, DepthUpdate)
        trades += isinstance(event, Trade)
    return events, depth, trades


def _reconstruct(path: Path) -> tuple[int, int]:
    book = L2OrderBook()
    depth = 0
    for event in iter_events(path):
        if isinstance(event, Boundary):
            book.clear()
        elif isinstance(event, Snapshot):
            book.install_snapshot(event)
        elif isinstance(event, DepthUpdate):
            if event.final_update_id <= book.last_update_id:
                continue
            if event.first_update_id > book.last_update_id + 1:
                raise RuntimeError("Benchmark recording contains a sequence gap.")
            book.apply(event)
            depth += 1
    return depth, book.state_hash()


def _measure_peak_memory(function: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        function()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def _write_json_atomic(path: Path, payload: dict[str, object], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite benchmark report: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    arguments = parse_arguments()
    path = Path(arguments.recording).expanduser().resolve()
    metadata = read_metadata(path, verify_hashes=True)
    if metadata.data_complete is False:
        raise RuntimeError("Refusing to benchmark an incomplete L2 recording.")

    parse_times, parse_counts = _timed(lambda: _parse(path), arguments.trials)
    reconstruction_times, reconstruction = _timed(
        lambda: _reconstruct(path), arguments.trials
    )
    feature_times, feature_set = _timed(
        lambda: build_feature_set([path], horizon=arguments.horizon),
        arguments.trials,
    )
    events, depth_events, trades = parse_counts
    reconstructed_depth, final_hash = reconstruction
    report: dict[str, object] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": _git_commit(Path(__file__).resolve().parent),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "command_line": list(os.sys.argv),
        },
        "recording": {
            "path": str(path),
            "symbol": metadata.symbol,
            "events": events,
            "depth_events": depth_events,
            "trades": trades,
            "sha256": metadata.sha256,
            "data_complete": metadata.data_complete,
        },
        "configuration": {
            "horizon_events": arguments.horizon,
            "trials": arguments.trials,
            "warmup_trials": 1,
        },
        "binary_parse": _summary(parse_times, events),
        "book_reconstruction": _summary(reconstruction_times, reconstructed_depth),
        "feature_generation": _summary(feature_times, feature_set.size),
        "peak_memory_bytes": {
            "binary_parse": _measure_peak_memory(lambda: _parse(path)),
            "book_reconstruction": _measure_peak_memory(lambda: _reconstruct(path)),
            "feature_generation": _measure_peak_memory(
                lambda: build_feature_set([path], horizon=arguments.horizon)
            ),
        },
        "determinism": {
            "final_state_hash": final_hash,
            "metadata_final_state_hash": metadata.final_state_hash,
            "match": final_hash == metadata.final_state_hash,
        },
    }
    output = Path(arguments.output).expanduser().resolve()
    _write_json_atomic(output, report, overwrite=arguments.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"L2 benchmark report: {output}")


if __name__ == "__main__":
    main()
