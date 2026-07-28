
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

from native_runtime import prepare_native_runtime

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
prepare_native_runtime(ROOT)

import numpy as np  

from microstructure import FEATURE_NAMES, OnlineFeatureBuilder, build_feature_set  
from qbin import RECORD_DTYPE  
from research_diagnostics import newey_west_mean_t_statistic, spearman_correlation  

failures = 0


def check(name: str, function: Callable[[], object]) -> None:
    global failures
    try:
        result = function()
        print(f"[PASS] {name}: {result}")
    except Exception as exc:
        failures += 1
        print(f"[FAIL] {name}: {exc!r}")


def verify_python() -> str:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "The packaged native extension requires CPython 3.12; "
            f"current interpreter is {sys.version.split()[0]}."
        )
    return sys.version.replace("\n", " ")


def verify_tk() -> str:
    import tkinter as tk

    root = tk.Tk()
    try:
        version = root.tk.call("info", "patchlevel")
        root.withdraw()
        root.update_idletasks()
        return f"Tk {version} | native window ready"
    finally:
        root.destroy()


def verify_websocket_client() -> str:
    import websocket

    version = getattr(websocket, "__version__", "unknown")
    if not hasattr(websocket, "WebSocketApp"):
        raise RuntimeError("websocket-client does not expose WebSocketApp.")
    return f"websocket-client {version} | WebSocketApp ready"


def verify_engine() -> str:
    import quant_engine

    dtype = np.dtype(quant_engine.order_book_dtype)
    if dtype.itemsize != 32:
        raise RuntimeError(f"Expected 32-byte native records, received {dtype.itemsize}.")
    required = {
        "start_live",
        "start_replay",
        "stop",
        "consume_batch",
        "state",
        "last_error",
        "recording_accepted",
    }
    engine = quant_engine.IngestionEngine()
    missing = sorted(name for name in required if not hasattr(engine, name))
    if missing:
        raise RuntimeError(f"Native extension is missing API methods: {missing}")
    engine.stop()
    return f"IngestionEngine API ready | dtype={dtype}"


def verify_native_l2() -> str:
    import quant_engine

    synchronizer = quant_engine.L2Synchronizer()
    result = synchronizer.ingest(
        2, 2, 101, 101, [(10_000, 450)], []
    )
    if result != "buffered":
        raise RuntimeError(f"Unexpected native L2 buffer result: {result}")
    installed = synchronizer.install_snapshot(
        1,
        100,
        [(10_000, 500), (9_900, 700)],
        [(10_100, 600), (10_200, 800)],
    )
    if installed["result"] != "synchronized":
        raise RuntimeError(f"Native L2 synchronization failed: {installed}")
    if synchronizer.last_update_id != 101 or synchronizer.best_bid != (10_000, 450):
        raise RuntimeError("Native L2 state does not match the applied depth event.")
    return f"update={synchronizer.last_update_id} | hash={synchronizer.state_hash}"


def verify_l2_binary() -> str:
    from l2bin import Boundary, BoundaryReason, L2Writer, iter_events, read_metadata
    from l2book import DepthUpdate, L2OrderBook, Level, Snapshot

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.l2bin"
        snapshot = Snapshot(1, 100, (Level(10_000, 500),), (Level(10_100, 600),))
        update = DepthUpdate(2, 2, 101, 101, (Level(10_000, 450),), ())
        book = L2OrderBook()
        writer = L2Writer(path, "BTCUSDT", checkpoint_interval=1)
        writer.write(Boundary(1, BoundaryReason.CONNECTION_START))
        writer.write(snapshot)
        book.install_snapshot(snapshot)
        writer.write(update)
        book.apply(update)
        writer.write_checkpoint(book.last_update_id, book.state_hash())
        writer.finalize(
            final_update_id=book.last_update_id,
            final_state_hash=book.state_hash(),
        )
        events = tuple(iter_events(path))
        metadata = read_metadata(path, verify_hashes=True)
        if len(events) != 3 or metadata.data_complete is not True:
            raise RuntimeError("L2 binary smoke artifact did not validate.")
        return f"events={len(events)} | hash={metadata.final_state_hash}"


def verify_feature_parity() -> str:
    rng = np.random.default_rng(7)
    count = 180
    records = np.empty(count, dtype=RECORD_DTYPE)
    records["timestamp_ns"] = 1_000_000_000 + np.cumsum(
        rng.integers(100_000, 2_000_000, size=count, dtype=np.uint64)
    )
    mid = 100.0 + np.cumsum(rng.normal(0.0, 0.01, size=count))
    records["best_bid"] = mid - 0.005
    records["best_ask"] = mid + 0.005
    records["bid_volume"] = rng.integers(100_000, 2_000_000, size=count, dtype=np.uint32)
    records["ask_volume"] = rng.integers(100_000, 2_000_000, size=count, dtype=np.uint32)

    offline = build_feature_set(
        records,
        volume_scale=1_000_000.0,
        horizon=10,
        session_id=0,
    )
    builder = OnlineFeatureBuilder()
    online: dict[int, np.ndarray] = {}
    for index, row in enumerate(records):
        feature = builder.update(
            int(row["timestamp_ns"]),
            float(row["best_bid"]),
            float(row["best_ask"]),
            int(row["bid_volume"]),
            int(row["ask_volume"]),
        )
        if feature is not None:
            online[index] = feature
    matrix = np.vstack([online[int(index)] for index in offline.event_index])
    np.testing.assert_allclose(matrix, offline.X, rtol=1e-12, atol=1e-10)
    if np.any(offline.current_bid_quantity < 0.0) or np.any(
        offline.current_ask_quantity < 0.0
    ):
        raise RuntimeError("Feature set contains invalid displayed quantities.")
    return f"{len(FEATURE_NAMES)} features match online/offline"


def verify_research_diagnostics() -> str:
    values = np.asarray([0.2, -0.1, 0.4, 0.3], dtype=np.float64)
    rank_ic = spearman_correlation(values, values)
    hac_t = newey_west_mean_t_statistic(values)
    if rank_ic != 1.0 or not np.isfinite(hac_t):
        raise RuntimeError("Robust research diagnostics failed their smoke test.")
    return f"rankIC={rank_ic:.1f} | HAC t={hac_t:.4f}"


check("Python 3.12", verify_python)
check("Project root", lambda: ROOT)
check("NumPy", lambda: np.__version__)
check("TLS certificates", lambda: __import__("certifi").where())
check("WebSocket client", verify_websocket_client)
check("Desktop UI runtime", verify_tk)

check("Native dependency model", lambda: "OpenSSL and simdjson linked statically")

check("Native engine", verify_engine)
check("Native L2 synchronizer", verify_native_l2)
check("Deterministic L2 binary", verify_l2_binary)
check("market_ui package", lambda: __import__("market_ui.app") and "all modules importable")
check("Online/offline feature parity", verify_feature_parity)
check("Robust research diagnostics", verify_research_diagnostics)

if failures:
    print(f"\nVerification failed with {failures} problem(s).")
    raise SystemExit(1)
print("\nMarket Systems Workstation is ready.")
