"""CLI entry point for running evaluations.

Usage:
    python -m finance_ai.evaluation.cli --eval all --provider google --model gemini-2.0-flash
    python -m finance_ai.evaluation.cli --compare  # multi-model routing comparison
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from finance_ai.core.logging import get_logger
from finance_ai.evaluation.model_comparison import VALID_COMPARISON_DIMENSIONS

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.language_models.chat_models import BaseChatModel

    from finance_ai.core.config import Settings
    from finance_ai.evaluation.models import (
        EvaluationReport,
        ModelComparisonResult,
    )
    from finance_ai.evaluation.runner import EvaluationRunner
    from finance_ai.rag.vector_store import FinanceVectorStore

logger = get_logger(__name__)

VALID_EVAL_TYPES = (
    "routing",
    "rag",
    "accuracy",
    "accuracy-forced",
    "hallucination",
    "quality",
    "safety",
    "performance",
    "all",
)
VALID_PROVIDERS = ("google", "ollama", "openrouter")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the evaluation CLI.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Parsed argument namespace.

    Example:
        >>> args = parse_args(["--eval", "routing"])
    """
    parser = argparse.ArgumentParser(
        description="Run evaluation framework for Personal Finance AI",
    )
    parser.add_argument(
        "--eval",
        choices=VALID_EVAL_TYPES,
        default="all",
        help="Which evaluation to run.",
    )
    parser.add_argument(
        "--provider",
        choices=VALID_PROVIDERS,
        default="google",
        help="LLM provider for the evaluated model.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Model name to evaluate.",
    )
    parser.add_argument(
        "--judge-provider",
        choices=VALID_PROVIDERS,
        default="openrouter",
        help="LLM provider for the judge model.",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="Judge model name (e.g., gpt-4o).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/evaluation",
        help="Path to evaluation dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/evaluation/results",
        help="Path to save evaluation results.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison across multiple models.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["routing", "rag", "accuracy", "hallucination", "quality", "safety", "performance"],
        help="Dimensions to skip (e.g., --skip rag quality).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help=(
            "With --compare: candidate models as provider:model strings "
            "(e.g. ollama:minimax-m3 google:gemini-3.5). Defaults to the "
            "built-in comparison shortlist."
        ),
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["routing"],
        choices=VALID_COMPARISON_DIMENSIONS,
        help=(
            "With --compare: dimensions to run per model. "
            f"Default: routing. Options: {', '.join(VALID_COMPARISON_DIMENSIONS)}."
        ),
    )
    parser.add_argument(
        "--comparison-judge",
        default="ollama:minimax-m3",
        help=(
            "With --compare: provider:model of the fixed LLM-as-judge "
            "used for the quality dimension (same judge for all models)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="With --compare: number of candidate models evaluated concurrently.",
    )
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help="Reuse existing ChromaDB index instead of re-indexing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the evaluation CLI.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Example:
        >>> main(["--eval", "routing", "--provider", "google"])
    """
    args = parse_args(argv)
    print(f"[Eval] provider={args.provider} model={args.model} eval={args.eval}")

    if args.compare:
        _run_model_comparison(args)
        return

    from finance_ai.evaluation.reporter import (  # noqa: PLC0415
        save_json_report,
        save_markdown_report,
    )
    from finance_ai.evaluation.runner import EvaluationRunner  # noqa: PLC0415

    print("[Eval] Creating model...")
    chat_model = _create_model(args.provider, args.model)
    needs_rag = args.eval in ("rag", "all")
    vector_store = _create_vector_store(reuse_index=args.reuse_index) if needs_rag else None
    needs_judge = args.eval in ("quality", "all")
    judge_model = (
        _create_judge_model(args.judge_provider, args.judge_model) if needs_judge else None
    )

    runner = EvaluationRunner(
        chat_model=chat_model,
        vector_store=vector_store,
        llm_provider=args.provider,
        llm_model=args.model,
        judge_model=judge_model,
        data_dir=args.data_dir,
    )

    skip_set = set(args.skip)
    if skip_set:
        print(f"[Eval] Skipping: {', '.join(skip_set)}")
    print(f"[Eval] Running {args.eval}...")
    report = _run_selected_evaluation(runner, args.eval, skip_set)
    json_path = save_json_report(report, args.output_dir)
    md_path = save_markdown_report(report, args.output_dir)
    print(f"[Eval] Done! Results saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


def _run_model_comparison(args: argparse.Namespace) -> None:
    """Run the routing comparison across candidate models and save reports.

    Args:
        args: Parsed CLI arguments (uses .models, .data_dir, .output_dir).

    Example:
        >>> _run_model_comparison(parse_args(["--compare"]))  # doctest: +SKIP
    """
    from datetime import datetime
    from pathlib import Path

    from finance_ai.evaluation.model_comparison import (  # noqa: PLC0415
        DEFAULT_COMPARISON_MODELS,
        parse_model_spec,
        run_multi_dimension_comparison,
    )

    specs = [parse_model_spec(text) for text in args.models] or DEFAULT_COMPARISON_MODELS
    dimensions = args.dimensions
    print(
        f"[Eval] Comparing {len(specs)} models on {len(dimensions)} "
        f"dimension(s): {', '.join(dimensions)}..."
    )
    results = run_multi_dimension_comparison(
        specs,
        dimensions,
        data_dir=args.data_dir,
        max_workers=args.workers,
        judge_spec=parse_model_spec(args.comparison_judge),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for result in results:
        _save_dimension_report(result, output_dir, stamp)


def _save_dimension_report(
    result: "ModelComparisonResult",
    output_dir: "Path",
    stamp: str,
) -> None:
    """Save and print one dimension's comparison report.

    Args:
        result: Comparison result for one dimension.
        output_dir: Directory to write JSON and markdown reports to.
        stamp: Timestamp string shared by this run's filenames.
    """
    from finance_ai.evaluation.model_comparison import (  # noqa: PLC0415
        format_comparison_markdown,
    )

    json_path = output_dir / f"model_comparison_{result.dimension}_{stamp}.json"
    md_path = output_dir / f"model_comparison_{result.dimension}_{stamp}.md"
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(format_comparison_markdown(result), encoding="utf-8")
    print(f"\n[Eval] Dimension: {result.dimension}")
    print(format_comparison_markdown(result))
    print("[Eval] Comparison saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


def _run_selected_evaluation(
    runner: EvaluationRunner,
    eval_type: str,
    skip: set[str] | None = None,
) -> EvaluationReport:
    """Run the selected evaluation type.

    Args:
        runner: Configured EvaluationRunner.
        eval_type: Type of evaluation to run.
        skip: Set of dimension names to skip.

    Returns:
        EvaluationReport with results.
    """
    import uuid  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    from finance_ai.evaluation.models import EvaluationReport  # noqa: PLC0415

    if eval_type == "all":
        return runner.run_all(skip=skip or set())

    report = EvaluationReport(
        report_id=str(uuid.uuid4()),
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        llm_provider="",
        llm_model="",
    )
    eval_map = {
        "routing": ("routing", runner.run_routing),
        "rag": ("rag_retrieval", runner.run_rag),
        "accuracy": ("tax_accuracy", runner.run_tax_accuracy),
        "accuracy-forced": ("tax_accuracy", runner.run_tax_accuracy_forced),
        "hallucination": ("hallucination", runner.run_hallucination),
        "quality": ("quality", runner.run_quality),
        "safety": ("recommendation_safety", runner.run_recommendation_safety),
        "performance": ("performance", runner.run_performance),
    }
    if eval_type in eval_map:
        field, method = eval_map[eval_type]
        setattr(report, field, method())
    return report


def _create_model(
    provider: str,
    model_name: str,
) -> BaseChatModel:
    """Create a chat model for evaluation.

    Args:
        provider: LLM provider name (google, ollama, openrouter).
        model_name: Model name to evaluate.

    Returns:
        BaseChatModel instance.
    """
    settings = _build_settings(provider, model_name)

    from finance_ai.agents.llm_factory import (  # noqa: PLC0415
        create_chat_model,
    )

    return create_chat_model(settings=settings)


def _create_vector_store(
    reuse_index: bool = False,
) -> FinanceVectorStore:
    """Create and populate the vector store for RAG evaluation.

    Args:
        reuse_index: If True, reuse existing ChromaDB index.

    Returns:
        FinanceVectorStore instance populated with knowledge base.
    """
    from pathlib import Path  # noqa: PLC0415

    from finance_ai.core.config import get_settings  # noqa: PLC0415
    from finance_ai.rag.embedding_factory import create_embeddings  # noqa: PLC0415
    from finance_ai.rag.knowledge_base import KnowledgeBaseManager  # noqa: PLC0415
    from finance_ai.rag.vector_store import (  # noqa: PLC0415
        FinanceVectorStore as _FinanceVectorStore,
    )

    settings = get_settings()
    embeddings = create_embeddings(settings)
    store = _FinanceVectorStore(embeddings=embeddings)

    if reuse_index and store.get_document_count() > 0:
        count = store.get_document_count()
        print(f"[Eval] Reusing existing index ({count} chunks)")
        return store

    manager = KnowledgeBaseManager(
        vector_store=store,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    kb_dir = Path(settings.rag_knowledge_base_directory)
    print(f"[Eval] Indexing knowledge base from {kb_dir}...")
    total = manager.rebuild_index(kb_dir)
    print(f"[Eval] Indexed {total} chunks")
    return store


def _create_judge_model(
    provider: str,
    model_name: str,
) -> BaseChatModel | None:
    """Create the judge model for quality evaluation.

    Args:
        provider: Judge LLM provider.
        model_name: Judge model name.

    Returns:
        BaseChatModel instance, or None if no model specified.
    """
    if not model_name:
        return None
    settings = _build_settings(provider, model_name)

    from finance_ai.agents.llm_factory import (  # noqa: PLC0415
        create_chat_model,
    )

    return create_chat_model(settings=settings)


def _build_settings(provider: str, model_name: str) -> "Settings":  # noqa: F821
    """Build Settings with CLI-provided provider and model.

    Args:
        provider: LLM provider name.
        model_name: Model name.

    Returns:
        Settings instance with overridden values.
    """
    from finance_ai.core.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    settings.llm_provider = provider  # type: ignore[assignment]
    _MODEL_FIELD = {
        "google": "google_model",
        "ollama": "ollama_model",
        "openrouter": "openrouter_model",
    }
    field = _MODEL_FIELD.get(provider)
    if field:
        setattr(settings, field, model_name)
    return settings


if __name__ == "__main__":
    main()
