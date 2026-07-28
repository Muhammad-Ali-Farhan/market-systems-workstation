
from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from pathlib import Path

import certifi
import numpy as np

from native_runtime import prepare_native_runtime

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
prepare_native_runtime()

import quant_engine  # noqa: E402
from microstructure import OnlineFeatureBuilder  # noqa: E402
from qbin import (  # noqa: E402
    feature_reset_indices,
    open_records,
    open_update_ids,
    read_metadata,
    validate_records,
    validate_update_ids,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove that native replay preserves every 32-byte record and produces "
            "the same online features at multiple replay speeds."
        )
    )
    parser.add_argument("file", help="Recording to verify.")
    parser.add_argument(
        "--speeds",
        nargs="+",
        type=float,
        default=[0.0, 10.0],
        help="Finite non-negative replay speeds to compare.",
    )
    arguments = parser.parse_args()
    if any(not math.isfinite(speed) or speed < 0.0 for speed in arguments.speeds):
        parser.error("Every replay speed must be finite and non-negative.")
    return arguments


def source_digest(records: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(records).tobytes(order="C")).hexdigest()


def replay_once(
    path: Path,
    speed: float,
    dtype: np.dtype,
    reset_indices: frozenset[int],
) -> tuple[str, np.ndarray, int, float]:
    engine = quant_engine.IngestionEngine()
    buffer = np.empty(4096, dtype=dtype)
    digest = hashlib.sha256()
    builder = OnlineFeatureBuilder()
    feature_rows: list[np.ndarray] = []
    record_index = 0
    started = time.perf_counter()
    engine.start_replay(str(path), speed)
    try:
        while True:
            count = engine.consume_batch(buffer)
            if count == 0:
                if not engine.is_running():
                    break
                time.sleep(0.0005)
                continue
            batch = buffer[:count]
            digest.update(batch.tobytes(order="C"))
            for row in batch:
                if record_index in reset_indices:
                    builder.reset()
                features = builder.update(
                    int(row["timestamp_ns"]),
                    float(row["best_bid"]),
                    float(row["best_ask"]),
                    int(row["bid_volume"]),
                    int(row["ask_volume"]),
                )
                record_index += 1
                if features is not None:
                    feature_rows.append(features)
    finally:
        engine.stop()
    elapsed = time.perf_counter() - started
    if engine.last_error():
        raise RuntimeError(engine.last_error())
    if engine.replay_errors() != 0:
        raise RuntimeError(f"Replay reported {engine.replay_errors()} error(s).")
    matrix = (
        np.vstack(feature_rows)
        if feature_rows
        else np.empty((0, 16), dtype=np.float64)
    )
    return digest.hexdigest(), matrix, record_index, elapsed


def main() -> None:
    arguments = parse_arguments()
    path = Path(arguments.file).resolve()
    metadata = read_metadata(path)
    if metadata.data_complete is False:
        raise RuntimeError("Refusing to certify an incomplete recording.")
    records = open_records(path, metadata)
    validate_records(records, context=str(path))
    validate_update_ids(open_update_ids(metadata), metadata, context=str(path))
    expected_digest = source_digest(records)
    reset_indices = feature_reset_indices(metadata)
    dtype = np.dtype(quant_engine.order_book_dtype)
    if dtype.itemsize != 32:
        raise RuntimeError("Expected the native 32-byte record layout.")

    reference_features: np.ndarray | None = None
    for speed in arguments.speeds:
        digest, features, count, elapsed = replay_once(
            path,
            speed,
            dtype,
            reset_indices,
        )
        if count != metadata.record_count:
            raise RuntimeError(
                f"Replay at {speed:g}x consumed {count:,} records; "
                f"expected {metadata.record_count:,}."
            )
        if digest != expected_digest:
            raise RuntimeError(f"Replay at {speed:g}x changed the record stream.")
        if reference_features is None:
            reference_features = features
        else:
            np.testing.assert_allclose(
                features,
                reference_features,
                rtol=0.0,
                atol=0.0,
            )
        label = "maximum" if speed == 0.0 else f"{speed:g}x"
        rate = count / max(elapsed, 1e-9)
        print(
            f"[PASS] {label}: {count:,} records | {features.shape[0]:,} "
            f"feature rows | {rate:,.2f} records/sec | SHA-256 {digest}"
        )

    print("[PASS] Replay records and online features are deterministic across speeds.")


if __name__ == "__main__":
    main()

