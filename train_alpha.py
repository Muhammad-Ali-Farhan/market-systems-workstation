
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Sequence

from microstructure import DEFAULT_MAX_GAP_NS
from qbin import read_metadata
from research import train_and_evaluate


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a leakage-resistant top-of-book alpha model."
    )
    parser.add_argument(
        "recordings",
        nargs="+",
        help="One or more .qbin files or quoted glob patterns.",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Fixed slippage stress per side in basis points.",
    )
    parser.add_argument(
        "--trade-size-base",
        type=float,
        default=0.0,
        help=(
            "Optional base-asset trade size. Nonzero values require the selected "
            "size to fit within the configured displayed top-of-book participation."
        ),
    )
    parser.add_argument(
        "--max-displayed-participation",
        type=float,
        default=1.0,
        help="Maximum fraction of displayed top-of-book quantity used by the screen.",
    )
    parser.add_argument(
        "--diagnostic-resamples",
        type=int,
        default=500,
        help="Deterministic bootstrap/permutation sample count (minimum 100).",
    )
    parser.add_argument("--diagnostic-seed", type=int, default=0)
    parser.add_argument("--max-gap-ms", type=float, default=DEFAULT_MAX_GAP_NS / 1_000_000.0)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--allow-test-reuse",
        action="store_true",
        help=(
            "Allow an explicit reproducibility rerun against a test set that "
            "already appears in another report in the output folder."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing model/report/prediction artifacts.",
    )
    parser.add_argument("--output", default="artifacts/alpha_model.npz")
    parser.add_argument("--report", default="artifacts/research_report.json")
    parser.add_argument("--predictions", default="artifacts/test_predictions.csv")
    parser.add_argument("--evidence", default="artifacts/research_card.md")
    arguments = parser.parse_args()

    if arguments.horizon <= 0:
        parser.error("--horizon must be positive.")
    if not math.isfinite(arguments.fee_bps) or arguments.fee_bps < 0.0:
        parser.error("--fee-bps must be finite and non-negative.")
    if not math.isfinite(arguments.slippage_bps) or arguments.slippage_bps < 0.0:
        parser.error("--slippage-bps must be finite and non-negative.")
    if not math.isfinite(arguments.trade_size_base) or arguments.trade_size_base < 0.0:
        parser.error("--trade-size-base must be finite and non-negative.")
    if (
        not math.isfinite(arguments.max_displayed_participation)
        or not 0.0 < arguments.max_displayed_participation <= 1.0
    ):
        parser.error("--max-displayed-participation must be in (0, 1].")
    if arguments.diagnostic_resamples < 100:
        parser.error("--diagnostic-resamples must be at least 100.")
    if not math.isfinite(arguments.max_gap_ms) or arguments.max_gap_ms <= 0.0:
        parser.error("--max-gap-ms must be finite and positive.")
    return arguments


def expand_recording_paths(patterns: Sequence[str]) -> list[Path]:
    discovered: dict[Path, None] = {}
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        for match in matches:
            resolved = match.resolve()
            if resolved.suffix.lower() == ".qbin":
                discovered[resolved] = None
    if not discovered:
        raise FileNotFoundError("No .qbin recordings matched the supplied paths.")
    metadata = [read_metadata(path) for path in discovered]
    metadata.sort(key=lambda item: (item.created_unix_ns, str(item.path)))
    return [item.path for item in metadata]


def main() -> None:
    arguments = parse_arguments()
    paths = expand_recording_paths(arguments.recordings)

    def progress(value: int, message: str) -> None:
        print(f"[Research {value:3d}%] {message}")

    report = train_and_evaluate(
        paths,
        horizon=arguments.horizon,
        fee_bps_per_side=arguments.fee_bps,
        model_path=arguments.output,
        report_path=arguments.report,
        predictions_path=arguments.predictions,
        evidence_path=arguments.evidence,
        slippage_bps_per_side=arguments.slippage_bps,
        trade_size_base=arguments.trade_size_base,
        max_displayed_participation=arguments.max_displayed_participation,
        diagnostic_resamples=arguments.diagnostic_resamples,
        diagnostic_seed=arguments.diagnostic_seed,
        max_gap_ns=int(arguments.max_gap_ms * 1_000_000.0),
        allow_incomplete=arguments.allow_incomplete,
        overwrite=arguments.overwrite,
        allow_test_reuse=arguments.allow_test_reuse,
        progress=progress,
    )

    selected = report["selected_model"]
    test_regression = report["test_regression"]
    test_strategy = report["test_strategy"]
    baseline = report["imbalance_baseline"]["test_strategy"]
    print()
    print(f"Selected ridge alpha: {selected['ridge_alpha']:g}")
    print(f"Selected signal threshold: {selected['signal_threshold_bps']:.6f} bps")
    print(
        "Untouched test: "
        f"IC={test_regression['pearson_ic']:.6f} | "
        f"rankIC={test_regression['spearman_rank_ic']:.6f} | "
        f"direction={test_regression['direction_accuracy']:.2%} | "
        f"majority={test_regression['majority_direction_accuracy']:.2%} | "
        f"lift={test_regression['direction_accuracy_lift_vs_majority']:+.2%} | "
        f"balanced={test_regression['balanced_direction_accuracy']:.2%} | "
        f"R²={test_regression['r_squared']:.6f}"
    )
    print(
        "Execution-adjusted signal diagnostic: "
        f"trades={test_strategy['trades']} | "
        f"mean={test_strategy['mean_pnl_bps']:.6f} bps | "
        f"total={test_strategy['total_pnl_bps']:.6f} bps | "
        f"maxDD={test_strategy['max_drawdown_bps']:.6f} bps | "
        f"HAC t={test_strategy['newey_west_pnl_t_statistic']:.4f}"
    )
    print(f"OBI baseline total PnL: {baseline['total_pnl_bps']:.6f} bps")
    print(f"Model: {Path(arguments.output).resolve()}")
    print(f"Report: {Path(arguments.report).resolve()}")
    print(f"Predictions: {Path(arguments.predictions).resolve()}")
    print(f"Evidence card: {Path(arguments.evidence).resolve()}")


if __name__ == "__main__":
    main()

