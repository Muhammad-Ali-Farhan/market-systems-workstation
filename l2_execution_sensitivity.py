from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from execution_simulator import (
    ExecutionConfig,
    ExecutionSimulator,
    OrderRequest,
    OrderType,
    QueueModel,
    Side,
)
from l2bin import read_metadata, sha256_file
from l2book import parse_price, parse_quantity


@dataclass(frozen=True, slots=True)
class PredictionOrder:
    session_id: int
    global_row: int
    timestamp_ns: int
    side: Side
    bid: int
    ask: int


def load_prediction_orders(path: str | Path) -> tuple[PredictionOrder, ...]:
    output: list[PredictionOrder] = []
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "global_row",
            "timestamp_ns",
            "session_id",
            "best_bid",
            "best_ask",
            "selected_trade",
            "side",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                "Prediction CSV is missing required columns: "
                + ", ".join(sorted(required))
            )
        for row in reader:
            if int(row["selected_trade"]) != 1:
                continue
            raw_side = int(row["side"])
            if raw_side == 0:
                continue
            output.append(
                PredictionOrder(
                    session_id=int(row["session_id"]),
                    global_row=int(row["global_row"]),
                    timestamp_ns=int(row["timestamp_ns"]),
                    side=Side.BUY if raw_side > 0 else Side.SELL,
                    bid=parse_price(row["best_bid"]),
                    ask=parse_price(row["best_ask"]),
                )
            )
    return tuple(output)


def parse_recording_mapping(values: list[str]) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for value in values:
        raw_session, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError("Each --recording value must use SESSION_ID=PATH.")
        session_id = int(raw_session)
        path = Path(raw_path).expanduser().resolve()
        if session_id in mapping:
            raise ValueError(f"Duplicate recording mapping for session {session_id}.")
        if not path.is_file():
            raise FileNotFoundError(f"L2 recording does not exist: {path}")
        metadata = read_metadata(path, verify_hashes=True)
        if metadata.data_complete is not True:
            raise RuntimeError(f"Execution sensitivity requires a complete recording: {path}")
        mapping[session_id] = path
    return mapping


def aggregate_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    orders = sum(int(item["orders"]) for item in summaries)
    fills = sum(int(item["fills"]) for item in summaries)
    requested = sum(float(item["requested_quantity"]) for item in summaries)
    filled = sum(float(item["filled_quantity"]) for item in summaries)
    maker = sum(float(item["maker_quantity"]) for item in summaries)
    taker = sum(float(item["taker_quantity"]) for item in summaries)
    status_counts: dict[str, int] = {}
    for item in summaries:
        for name, count in dict(item["order_status_counts"]).items():
            status_counts[name] = status_counts.get(name, 0) + int(count)
    return {
        "orders": orders,
        "fills": fills,
        "requested_quantity": requested,
        "filled_quantity": filled,
        "fill_rate": filled / requested if requested else 0.0,
        "maker_share": maker / filled if filled else 0.0,
        "maker_quantity": maker,
        "taker_quantity": taker,
        "ending_cash_quote": sum(float(item["ending_cash_quote"]) for item in summaries),
        "marked_equity_quote": sum(float(item["marked_equity_quote"]) for item in summaries),
        "realized_fees_quote": sum(float(item["realized_fees_quote"]) for item in summaries),
        "killed_sessions": sum(bool(item["killed"]) for item in summaries),
        "order_status_counts": status_counts,
    }


def run_sensitivity(
    predictions: str | Path,
    recordings: dict[int, Path],
    *,
    quantity: int,
    style: str,
    latencies_us: tuple[float, ...],
    queue_models: tuple[QueueModel, ...],
    maker_fee_bps: float,
    taker_fee_bps: float,
    queue_ahead_fraction: float,
    time_to_live_ns: int,
) -> dict[str, object]:
    orders = load_prediction_orders(predictions)
    missing = sorted({item.session_id for item in orders} - set(recordings))
    if missing:
        raise ValueError(f"No recording mapping was supplied for sessions: {missing}")
    cases: list[dict[str, object]] = []
    models = queue_models if style == "passive" else (QueueModel.TRADE_ONLY,)
    for queue_model in models:
        for latency_us in latencies_us:
            session_summaries: list[dict[str, object]] = []
            for session_id, recording in sorted(recordings.items()):
                config = ExecutionConfig(
                    transmission_latency_ns=int(round(latency_us * 1_000.0)),
                    maker_fee_bps=maker_fee_bps,
                    taker_fee_bps=taker_fee_bps,
                    queue_model=queue_model,
                    queue_ahead_fraction=queue_ahead_fraction,
                )
                simulator = ExecutionSimulator(config)
                session_orders = [item for item in orders if item.session_id == session_id]
                for item in session_orders:
                    order_type = OrderType.LIMIT if style == "passive" else OrderType.MARKET
                    limit = (
                        item.bid if item.side is Side.BUY else item.ask
                    ) if order_type is OrderType.LIMIT else None
                    simulator.submit(
                        OrderRequest(
                            order_id=f"s{session_id}-r{item.global_row}",
                            decision_timestamp_ns=item.timestamp_ns,
                            side=item.side,
                            order_type=order_type,
                            quantity=quantity,
                            limit_price=limit,
                            time_to_live_ns=(
                                time_to_live_ns if order_type is OrderType.LIMIT else None
                            ),
                        )
                    )
                session_summaries.append(simulator.run(recording).summary())
            cases.append(
                {
                    "style": style,
                    "queue_model": queue_model.value,
                    "transmission_latency_us": latency_us,
                    **aggregate_summaries(session_summaries),
                }
            )
    return {
        "schema_version": 1,
        "prediction_file": {
            "file": Path(predictions).name,
            "sha256": sha256_file(Path(predictions)),
        },
        "recordings": {
            str(key): {
                "file": value.name,
                "sha256": sha256_file(value),
                "checkpoint_sha256": read_metadata(value).checkpoint_sha256,
            }
            for key, value in recordings.items()
        },
        "configuration": {
            "style": style,
            "quantity": quantity / 100_000_000,
            "latencies_us": list(latencies_us),
            "queue_models": [item.value for item in models],
            "maker_fee_bps": maker_fee_bps,
            "taker_fee_bps": taker_fee_bps,
            "queue_ahead_fraction": queue_ahead_fraction,
            "time_to_live_ns": time_to_live_ns,
        },
        "cases": cases,
        "interpretation": {
            "trade_only": "Passive fills require observed aggregate trades after queue-ahead depletion.",
            "pro_rata_depth": "Depth decreases contribute only a configured proportional sensitivity fill.",
            "optimistic_depth": "All qualifying depth depletion is treated as executable; this is an upper-bound sensitivity case.",
        },
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay held-out L2 research signals through execution sensitivity cases."
    )
    parser.add_argument("predictions", help="L2 research test-prediction CSV.")
    parser.add_argument(
        "--recording",
        action="append",
        required=True,
        help="Map a research session to its recording as SESSION_ID=PATH. Repeat as needed.",
    )
    parser.add_argument("--style", choices=("passive", "market"), default="passive")
    parser.add_argument("--quantity", default="0.00100000")
    parser.add_argument("--latencies-us", nargs="+", type=float, default=[0.0, 100.0, 250.0, 500.0, 1000.0])
    parser.add_argument(
        "--queue-models",
        nargs="+",
        choices=[item.value for item in QueueModel],
        default=[item.value for item in QueueModel],
    )
    parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    parser.add_argument("--taker-fee-bps", type=float, default=0.0)
    parser.add_argument("--queue-ahead-fraction", type=float, default=1.0)
    parser.add_argument("--ttl-ms", type=float, default=100.0)
    parser.add_argument("--output", default="artifacts/l2_execution_sensitivity.json")
    arguments = parser.parse_args()
    if any(not math.isfinite(value) or value < 0.0 for value in arguments.latencies_us):
        parser.error("Latencies must be finite and non-negative.")
    if not math.isfinite(arguments.ttl_ms) or arguments.ttl_ms <= 0.0:
        parser.error("--ttl-ms must be finite and positive.")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    payload = run_sensitivity(
        arguments.predictions,
        parse_recording_mapping(arguments.recording),
        quantity=parse_quantity(arguments.quantity),
        style=arguments.style,
        latencies_us=tuple(arguments.latencies_us),
        queue_models=tuple(QueueModel(value) for value in arguments.queue_models),
        maker_fee_bps=arguments.maker_fee_bps,
        taker_fee_bps=arguments.taker_fee_bps,
        queue_ahead_fraction=arguments.queue_ahead_fraction,
        time_to_live_ns=int(round(arguments.ttl_ms * 1_000_000.0)),
    )
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    output = Path(arguments.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite execution report: {output}")
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"Execution sensitivity report: {output}")


if __name__ == "__main__":
    main()
