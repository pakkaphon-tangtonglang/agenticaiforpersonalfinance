"""Evaluate the guru personas on data/evaluation/guru_dataset.yaml.

Deterministic checks always run. Pass --judge to also score persona
adherence with the configured model as judge (costs extra LLM calls;
ideally point it at a different provider than the one evaluated).

Example:
    uv run python scripts/evaluate_gurus.py
    uv run python scripts/evaluate_gurus.py --judge
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from finance_ai.agents.llm_factory import create_chat_model
from finance_ai.database.session import create_database_engine, create_session_factory
from finance_ai.evaluation.datasets import load_guru_eval_dataset
from finance_ai.evaluation.guru_evaluator import make_graph_answer_function, run_guru_evaluation

DATASET_PATH = "data/evaluation/guru_dataset.yaml"
RESULTS_DIR = Path("data/evaluation/results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="store_true", help="Also score persona adherence")
    args = parser.parse_args()

    chat_model = create_chat_model()
    session_factory = create_session_factory(create_database_engine())
    summary = run_guru_evaluation(
        load_guru_eval_dataset(DATASET_PATH).cases,
        make_graph_answer_function(chat_model, session_factory),
        judge_model=create_chat_model(temperature=0.0) if args.judge else None,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"guru_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"pass_rate={summary.pass_rate} failures={summary.failure_counts}")
    print(f"mean_persona_score={summary.mean_persona_score} -> {output_path}")
