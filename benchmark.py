
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from microstructure import AlphaModel, build_feature_set
from qbin import (
    open_records,
    open_update_ids,
    read_metadata,
    sha256_file,
    validate_records,
    validate_update_ids,
)

T = TypeVar("T")
BENCHMARK_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible qbin, feature, and model-inference benchmark."
    )
    parser.add_argument("file", help="Validated .qbin benchmark recording.")
    parser.add_argument("--model", default="", help="Optional canonical .npz model.")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", default="artifacts/benchmark.json")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing benchmark report.",
    )
    arguments = parser.parse_args()
    if arguments.horizon <= 0:
        parser.error("--horizon must be positive.")
    if arguments.trials < 3:
        parser.error("--trials must be at least 3.")
    return arguments


def git_commit(root: Path) -> str | None:
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
    value = result.stdout.strip()
    return value or None


def timed_trials(function: Callable[[], T], trials: int) -> tuple[list[float], T]:
    latest = function()  # warm-up
    results: list[float] = []
    for _ in range(trials):
        started = time.perf_counter_ns()
        latest = function()
        elapsed_ns = time.perf_counter_ns() - started
        results.append(elapsed_ns / 1_000_000_000.0)
    return results, latest


def summary(seconds: list[float], units: int) -> dict[str, float | int]:
    values = np.asarray(seconds, dtype=np.float64)
    median = float(np.median(values))
    return {
        "trials": int(values.size),
        "units": int(units),
        "median_seconds": median,
        "minimum_seconds": float(np.min(values)),
        "maximum_seconds": float(np.max(values)),
        "median_units_per_second": units / max(median, 1e-12),
    }


def write_json_atomic(path: Path, payload: dict[str, object], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing benchmark report: {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Benchmark report appeared during execution: {path}"
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    arguments = parse_arguments()
    path = Path(arguments.file).resolve()
    metadata = read_metadata(path)
    records = open_records(path, metadata)
    validate_records(records, context=str(path))
    validate_update_ids(open_update_ids(metadata), metadata, context=str(path))
    if metadata.data_complete is False:
        raise RuntimeError("Refusing to benchmark an incomplete recording.")
    if metadata.record_count < arguments.horizon + 30:
        raise ValueError("Benchmark recording is too short for the selected horizon.")

    qbin_times, _scan_value = timed_trials(
        lambda: float(np.sum(records["best_bid"], dtype=np.float64)),
        arguments.trials,
    )

    def build_features():
        return build_feature_set(
            records,
            volume_scale=float(metadata.volume_scale),
            horizon=arguments.horizon,
            session_id=0,
        )

    feature_times, latest_feature_set = timed_trials(
        build_features,
        arguments.trials,
    )

    root = Path(__file__).resolve().parent
    report: dict[str, object] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": git_commit(root),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "command_line": list(os.sys.argv),
        },
        "recording": {
            "path": str(path),
            "sha256": sha256_file(path),
            "records": metadata.record_count,
            "data_complete": metadata.data_complete,
            "update_id_sha256": (
                sha256_file(metadata.update_id_path)
                if metadata.update_id_path is not None
                else None
            ),
        },
        "configuration": {
            "horizon_events": arguments.horizon,
            "trials": arguments.trials,
            "warmup_trials": 1,
        },
        "qbin_memory_scan": summary(qbin_times, metadata.record_count),
        "feature_build": summary(feature_times, latest_feature_set.size),
    }

    if arguments.model:
        model_path = Path(arguments.model).resolve()
        model = AlphaModel.load(model_path)
        if model.horizon != arguments.horizon:
            raise ValueError(
                f"Model horizon {model.horizon} does not match "
                f"--horizon {arguments.horizon}."
            )
        inference_times, _prediction = timed_trials(
            lambda: model.predict_matrix(latest_feature_set.X),
            arguments.trials,
        )
        report["model_inference"] = summary(
            inference_times,
            latest_feature_set.size,
        )
        report["model"] = {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "schema_version": model.schema_version,
        }

    output = Path(arguments.output)
    write_json_atomic(output, report, overwrite=arguments.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Benchmark report: {output.resolve()}")


if __name__ == "__main__":
    main()

