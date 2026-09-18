"""Cross-judge saved quality responses with a different judge model.

Reads a comparison quality JSON (with per-case rows persisted by the
comparison harness), re-scores every saved agent response under a new
judge, and writes an agreement report. No agent generation happens -
only judge calls.

Example:
    uv run python scripts/cross_judge_quality.py \
        --results data/evaluation/results/model_comparison_quality_20260918_135513.json \
        --judge ollama:deepseek-v4-pro:0813
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from langchain_core.language_models.chat_models import BaseChatModel
from finance_ai.evaluation.models import QualityCase

from finance_ai.core.logging import get_logger
from finance_ai.evaluation.datasets import load_quality_dataset
from finance_ai.evaluation.model_comparison import _default_model_factory, parse_model_spec
from finance_ai.evaluation.models import ModelComparisonEntry
from finance_ai.evaluation.quality_evaluator import evaluate_single_quality_case

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the cross-judge script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Comparison quality JSON path")
    parser.add_argument(
        "--judge",
        default="ollama:deepseek-v4-pro:0813",
        help="Cross judge spec (default: ollama:deepseek-v4-pro:0813)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/evaluation/results",
        help="Directory for the agreement report",
    )
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def _rescore_entry(
    entry: dict[str, Any],
    case_map: "dict[str, QualityCase]",
    judge_model: "BaseChatModel",
    workers: int,
) -> list[dict[str, Any]]:
    """Re-score all saved responses of one entry under the cross judge."""
    rows = entry["per_case_results"]

    def score_row(row: dict[str, Any]) -> dict[str, Any]:
        case = case_map[row["case_id"]]
        fresh = evaluate_single_quality_case(
            agent_model=MagicMock(),
            judge_model=judge_model,
            case=case,
            agent_response=row["agent_response"],
        )
        return {
            "model_name": entry["model_name"],
            "case_id": row["case_id"],
            "original_overall": str(row["scores"]["overall"]),
            "cross_overall": str(fresh.scores.overall),
            "cross_reasoning": fresh.scores.judge_reasoning[:200],
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(score_row, row) for row in rows]
        return [f.result() for f in as_completed(futures)]


def _mean_overall(rows: list[dict[str, Any]]) -> Decimal:
    """Mean of the overall scores across rows."""
    total = sum(Decimal(r["cross_overall"]) for r in rows)
    return total / Decimal(len(rows)) if rows else Decimal("0")


def _judge_mean_entry(entry: dict[str, Any]) -> Decimal:
    """Mean overall from the original judge's per-case rows."""
    scores = [Decimal(r["scores"]["overall"]) for r in entry["per_case_results"]]
    return sum(scores) / Decimal(len(scores)) if scores else Decimal("0")


def _agreement_stats(all_rows: list[dict[str, Any]]) -> tuple[Decimal, int]:
    """Exact-agreement rate and count of exact matches."""
    exact = sum(1 for r in all_rows if r["original_overall"] == r["cross_overall"])
    rate = Decimal(exact) / Decimal(len(all_rows)) if all_rows else Decimal("0")
    return rate, exact


def _build_markdown(
    judge_spec: str,
    per_model: dict[str, dict[str, Any]],
    exact_rate: Decimal,
    exact_count: int,
) -> str:
    """Build the cross-judge agreement markdown report.

    Args:
        judge_spec: Cross judge model spec string.
        per_model: Per-model original vs cross means.
        exact_rate: Fraction of rows with identical scores.
        exact_count: Number of rows with identical scores.

    Returns:
        Markdown report text.
    """
    lines = [
        "# Quality Cross-Judge Agreement",
        "",
        "Re-scored saved agent responses with a second judge; no",
        "regeneration. Exact-agreement rate: " f"{exact_rate:.2f} ({exact_count} rows).",
        "",
        f"**Cross judge:** `{judge_spec}`",
        "",
        "| Model | minimax-m3 judge | cross judge | delta |",
        "|---|---|---|---|",
    ]
    for model_name, stats in sorted(per_model.items(), key=lambda kv: kv[1]["cross"], reverse=True):
        delta = stats["cross"] - stats["original"]
        lines.append(
            f"| {model_name} | {stats['original']:.4f} " f"| {stats['cross']:.4f} | {delta:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run cross-judging over a saved quality comparison report."""
    args = parse_args()
    document = json.loads(Path(args.results).read_text(encoding="utf-8"))
    dataset = load_quality_dataset("data/evaluation/quality_dataset.yaml")
    case_map = {case.case_id: case for case in dataset.cases}
    spec = parse_model_spec(args.judge)
    judge_model = _default_model_factory(spec)

    per_model: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_rescore_entry, entry, case_map, judge_model, args.workers): entry
            for entry in document["entries"]
            if entry["status"] == "ok"
        }
        for future in as_completed(futures):
            entry = futures[future]
            rows = future.result()
            all_rows.extend(rows)
            per_model[entry["model_name"]] = {
                "original": _judge_mean_entry(entry),
                "cross": _mean_overall(rows),
                "rows": rows,
            }

    rate, exact = _agreement_stats(all_rows)
    report = _build_markdown(args.judge, per_model, rate, exact)
    out = Path(args.output_dir) / "quality_cross_judge_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
