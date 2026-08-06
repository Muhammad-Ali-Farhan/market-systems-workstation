from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping
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
from l2bin import L2Metadata, read_metadata, sha256_file
from l2book import parse_price, parse_quantity

RESEARCH_REPORT_SCHEMA_VERSION = 2
EXECUTION_SENSITIVITY_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class PredictionOrder:
    session_id: int
    global_row: int
    timestamp_ns: int
    side: Side
    bid: int
    ask: int


@dataclass(frozen=True, slots=True)
class ExpectedRecording:
    session_id: int
    file: str
    symbol: str
    sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedResearchProvenance:
    report_path: Path
    report_sha256: str
    prediction_file: str
    prediction_sha256: str
    test_fingerprint_sha256: str | None
    recordings: dict[int, ExpectedRecording]
    metadata: dict[int, L2Metadata]


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Research report field {key!r} must be an object.")
    return value


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Research report field {key!r} must be a non-empty string.")
    return value


def _required_integer(payload: Mapping[str, object], key: str, *, minimum: int = 0) -> int:
    value = payload.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"Research report field {key!r} must be an integer >= {minimum}."
        )
    return value


def _required_sha256(payload: Mapping[str, object], key: str) -> str:
    value = _required_string(payload, key).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Research report field {key!r} must be a SHA-256 digest.")
    return value


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
        if session_id < 0:
            raise ValueError("Recording session IDs must be non-negative.")
        path = Path(raw_path).expanduser().resolve()
        if session_id in mapping:
            raise ValueError(f"Duplicate recording mapping for session {session_id}.")
        if not path.is_file():
            raise FileNotFoundError(f"L2 recording does not exist: {path}")
        mapping[session_id] = path
    return mapping


def verify_research_provenance(
    report: str | Path,
    predictions: str | Path,
    recordings: Mapping[int, Path],
    orders: tuple[PredictionOrder, ...],
) -> VerifiedResearchProvenance:
    report_path = Path(report).expanduser().resolve()
    predictions_path = Path(predictions).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"L2 research report does not exist: {report_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Prediction CSV does not exist: {predictions_path}")

    try:
        raw_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exception:
        raise ValueError(f"L2 research report is not valid JSON: {report_path}") from exception
    if not isinstance(raw_payload, Mapping):
        raise ValueError("L2 research report root must be an object.")

    schema_version = _required_integer(raw_payload, "schema_version", minimum=1)
    if schema_version != RESEARCH_REPORT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported L2 research report schema: "
            f"{schema_version}; expected {RESEARCH_REPORT_SCHEMA_VERSION}."
        )

    artifacts = _required_mapping(raw_payload, "artifacts")
    prediction_artifact = _required_mapping(artifacts, "predictions")
    expected_prediction_file = _required_string(prediction_artifact, "file")
    expected_prediction_hash = _required_sha256(prediction_artifact, "sha256")
    actual_prediction_hash = sha256_file(predictions_path)
    if actual_prediction_hash != expected_prediction_hash:
        raise RuntimeError(
            "Prediction CSV does not match the supplied L2 research report: "
            f"expected {expected_prediction_hash}, received {actual_prediction_hash}."
        )

    raw_recordings = raw_payload.get("recordings")
    if not isinstance(raw_recordings, list) or not raw_recordings:
        raise ValueError("Research report field 'recordings' must be a non-empty array.")
    report_symbol = _required_string(raw_payload, "symbol")
    expected_all: dict[int, ExpectedRecording] = {}
    seen_recording_hashes: set[str] = set()
    for session_id, item in enumerate(raw_recordings):
        if not isinstance(item, Mapping):
            raise ValueError("Each research-report recording entry must be an object.")
        expected = ExpectedRecording(
            session_id=session_id,
            file=_required_string(item, "file"),
            symbol=_required_string(item, "symbol"),
            sha256=_required_sha256(item, "sha256"),
            checkpoint_sha256=_required_sha256(item, "checkpoint_sha256"),
        )
        if expected.symbol != report_symbol:
            raise ValueError(
                f"Research report recording {session_id} does not match symbol {report_symbol}."
            )
        if expected.sha256 in seen_recording_hashes:
            raise ValueError("Research report contains duplicate recording content hashes.")
        seen_recording_hashes.add(expected.sha256)
        expected_all[session_id] = expected

    split = _required_mapping(raw_payload, "split")
    raw_test_sessions = split.get("test_sessions")
    if not isinstance(raw_test_sessions, list) or not raw_test_sessions:
        raise ValueError("Research report split.test_sessions must be a non-empty array.")
    test_sessions: list[int] = []
    for value in raw_test_sessions:
        if type(value) is not int or value < 0:
            raise ValueError("Research report test session IDs must be non-negative integers.")
        if value not in expected_all:
            raise ValueError(f"Research report references unknown test session {value}.")
        test_sessions.append(value)
    if len(set(test_sessions)) != len(test_sessions):
        raise ValueError("Research report test session IDs must be unique.")

    expected_sessions = set(test_sessions)
    actual_sessions = set(recordings)
    if actual_sessions != expected_sessions:
        missing = sorted(expected_sessions - actual_sessions)
        extra = sorted(actual_sessions - expected_sessions)
        raise ValueError(
            "Recording mappings must match the research report's held-out sessions exactly; "
            f"missing={missing}, extra={extra}."
        )

    order_sessions = {item.session_id for item in orders}
    unexpected_order_sessions = sorted(order_sessions - expected_sessions)
    if unexpected_order_sessions:
        raise RuntimeError(
            "Prediction CSV contains selected trades outside the report's held-out sessions: "
            f"{unexpected_order_sessions}."
        )

    verified_metadata: dict[int, L2Metadata] = {}
    expected_test_recordings: dict[int, ExpectedRecording] = {}
    for session_id in sorted(expected_sessions):
        path = Path(recordings[session_id]).expanduser().resolve()
        metadata = read_metadata(path, verify_hashes=True)
        if metadata.data_complete is not True:
            raise RuntimeError(f"Execution sensitivity requires a complete recording: {path}")
        expected = expected_all[session_id]
        if metadata.symbol != expected.symbol:
            raise RuntimeError(
                f"Recording symbol mismatch for session {session_id}: "
                f"expected {expected.symbol}, received {metadata.symbol}."
            )
        if metadata.sha256 != expected.sha256:
            raise RuntimeError(
                f"Recording content does not match research provenance for session {session_id}."
            )
        if metadata.checkpoint_sha256 != expected.checkpoint_sha256:
            raise RuntimeError(
                "Recording checkpoint sidecar does not match research provenance for "
                f"session {session_id}."
            )
        verified_metadata[session_id] = metadata
        expected_test_recordings[session_id] = expected

    fingerprint = raw_payload.get("test_fingerprint_sha256")
    if fingerprint is not None:
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in fingerprint)
        ):
            raise ValueError("Research report test_fingerprint_sha256 is invalid.")
        fingerprint = fingerprint.lower()

    return VerifiedResearchProvenance(
        report_path=report_path,
        report_sha256=sha256_file(report_path),
        prediction_file=expected_prediction_file,
        prediction_sha256=actual_prediction_hash,
        test_fingerprint_sha256=fingerprint,
        recordings=expected_test_recordings,
        metadata=verified_metadata,
    )


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
    research_report: str | Path,
    quantity: int,
    style: str,
    latencies_us: tuple[float, ...],
    queue_models: tuple[QueueModel, ...],
    maker_fee_bps: float,
    taker_fee_bps: float,
    queue_ahead_fraction: float,
    time_to_live_ns: int,
) -> dict[str, object]:
    predictions_path = Path(predictions).expanduser().resolve()
    canonical_recordings = {
        session_id: Path(path).expanduser().resolve()
        for session_id, path in recordings.items()
    }
    orders = load_prediction_orders(predictions_path)
    provenance = verify_research_provenance(
        research_report,
        predictions_path,
        canonical_recordings,
        orders,
    )
    orders_by_session: dict[int, list[PredictionOrder]] = {
        session_id: [] for session_id in canonical_recordings
    }
    for order in orders:
        orders_by_session[order.session_id].append(order)

    cases: list[dict[str, object]] = []
    models = queue_models if style == "passive" else (QueueModel.TRADE_ONLY,)
    for queue_model in models:
        for latency_us in latencies_us:
            session_summaries: list[dict[str, object]] = []
            for session_id, recording in sorted(canonical_recordings.items()):
                config = ExecutionConfig(
                    transmission_latency_ns=int(round(latency_us * 1_000.0)),
                    maker_fee_bps=maker_fee_bps,
                    taker_fee_bps=taker_fee_bps,
                    queue_model=queue_model,
                    queue_ahead_fraction=queue_ahead_fraction,
                )
                simulator = ExecutionSimulator(config)
                for item in orders_by_session[session_id]:
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
        "schema_version": EXECUTION_SENSITIVITY_SCHEMA_VERSION,
        "research_provenance": {
            "verified": True,
            "research_report": {
                "file": provenance.report_path.name,
                "sha256": provenance.report_sha256,
                "schema_version": RESEARCH_REPORT_SCHEMA_VERSION,
            },
            "test_fingerprint_sha256": provenance.test_fingerprint_sha256,
        },
        "prediction_file": {
            "file": predictions_path.name,
            "research_file": provenance.prediction_file,
            "sha256": provenance.prediction_sha256,
        },
        "recordings": {
            str(session_id): {
                "file": canonical_recordings[session_id].name,
                "research_file": provenance.recordings[session_id].file,
                "symbol": provenance.metadata[session_id].symbol,
                "sha256": provenance.metadata[session_id].sha256,
                "checkpoint_sha256": provenance.metadata[session_id].checkpoint_sha256,
            }
            for session_id in sorted(canonical_recordings)
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
            "trade_only": (
                "Passive fills require observed aggregate trades after queue-ahead depletion."
            ),
            "pro_rata_depth": (
                "Depth decreases contribute only a configured proportional sensitivity fill."
            ),
            "optimistic_depth": (
                "All qualifying depth depletion is treated as executable; this is an "
                "upper-bound sensitivity case."
            ),
        },
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay held-out L2 research signals through execution sensitivity cases."
    )
    parser.add_argument("predictions", help="L2 research test-prediction CSV.")
    parser.add_argument(
        "--research-report",
        required=True,
        help="L2 research report that produced the prediction CSV.",
    )
    parser.add_argument(
        "--recording",
        action="append",
        required=True,
        help=(
            "Map each held-out research session to its recording as SESSION_ID=PATH. "
            "Repeat as needed."
        ),
    )
    parser.add_argument("--style", choices=("passive", "market"), default="passive")
    parser.add_argument("--quantity", default="0.00100000")
    parser.add_argument(
        "--latencies-us",
        nargs="+",
        type=float,
        default=[0.0, 100.0, 250.0, 500.0, 1000.0],
    )
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
        research_report=arguments.research_report,
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
