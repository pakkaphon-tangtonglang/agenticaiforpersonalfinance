"""Run DeepSeek bypass evaluation and generate PDF in EvalPDF style.

Usage:
    python scripts/run_deepseek_bypass_report.py

Runs two tests:
  1. Hallucination: DeepSeek answers tax fact questions with no RAG
  2. Tax accuracy: DeepSeek computes tax with no tools

Output:
    data/evaluation/results/deepseek_bypass_report.pdf
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "deepseek/deepseek-chat"
OUTPUT_PDF = "data/evaluation/results/deepseek_bypass_report.pdf"


def _create_model():
    from finance_ai.core.config import get_settings
    from finance_ai.agents.llm_factory import create_chat_model

    settings = get_settings()
    settings.llm_provider = "openrouter"
    settings.openrouter_model = MODEL_NAME
    return create_chat_model(settings=settings)


def _run_hallucination(model) -> object:
    """Run hallucination evaluation: DeepSeek answers directly, no RAG."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from finance_ai.evaluation.datasets import load_hallucination_dataset
    from finance_ai.evaluation.hallucination_evaluator import evaluate_hallucination_dataset

    print("  Loading hallucination dataset...")
    dataset = load_hallucination_dataset("data/evaluation/hallucination_dataset.yaml")

    SYSTEM = (
        "คุณเป็นผู้เชี่ยวชาญด้านภาษีไทย ตอบคำถามเกี่ยวกับกฎหมายและข้อบังคับภาษีเงินได้บุคคลธรรมดา"
    )

    print(f"  Asking DeepSeek {len(dataset.cases)} questions directly (no RAG)...")
    responses: dict[str, str] = {}
    for case in dataset.cases:
        try:
            resp = model.invoke(
                [SystemMessage(content=SYSTEM), HumanMessage(content=case.query)]
            )
            responses[case.case_id] = str(resp.content)
            print(f"    [{case.case_id}] done")
        except Exception as exc:  # noqa: BLE001
            responses[case.case_id] = ""
            print(f"    [{case.case_id}] ERROR: {exc}")

    return evaluate_hallucination_dataset(dataset, agent_responses=responses)


def _run_tax_bypass(model) -> object:
    """Run tax self-calculation benchmark (no tools)."""
    from finance_ai.evaluation.tool_bypass_benchmark import run_tool_bypass_benchmark

    print("  Running tax self-calculation benchmark (8 cases)...")
    return run_tool_bypass_benchmark(model, model_name=MODEL_NAME)


def main() -> None:
    import importlib.util

    print(f"Model: {MODEL_NAME}")
    print("Creating model...")
    model = _create_model()
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    print("\n[1/2] Hallucination test (no RAG)...")
    hal_result = _run_hallucination(model)
    print(
        f"  Compliance: {float(hal_result.compliance_rate) * 100:.1f}%"
        f" ({hal_result.compliant_count}/{hal_result.total_cases})"
    )

    print("\n[2/2] Tax self-calculation test (no tool)...")
    tax_result = _run_tax_bypass(model)
    print(f"  Accuracy: {tax_result.notool_accuracy_pct:.1f}%")

    print("\nGenerating PDF...")
    script_path = os.path.join(os.path.dirname(__file__), "deepseek_bypass_pdf.py")
    spec = importlib.util.spec_from_file_location("deepseek_bypass_pdf", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pdf_path = mod.generate_bypass_pdf(
        hal_result=hal_result,
        tax_result=tax_result,
        model_name=MODEL_NAME,
        timestamp=timestamp,
        output_path=OUTPUT_PDF,
    )
    print(f"\nPDF saved: {pdf_path}")


if __name__ == "__main__":
    main()
