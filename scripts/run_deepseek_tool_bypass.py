"""Run DeepSeek tool-bypass benchmark and generate PDF report.

Usage:
    python scripts/run_deepseek_tool_bypass.py

Requires:
    OPENROUTER_API_KEY in .env

Output:
    data/evaluation/results/deepseek_tool_bypass.md
    data/evaluation/results/deepseek_tool_bypass.pdf
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "deepseek/deepseek-chat"
MD_OUTPUT = "data/evaluation/results/deepseek_tool_bypass.md"
PDF_OUTPUT = "data/evaluation/results/deepseek_tool_bypass.pdf"


def _create_deepseek_model():
    """Create DeepSeek chat model via OpenRouter.

    Returns:
        BaseChatModel configured for deepseek-chat.
    """
    from finance_ai.core.config import get_settings
    from finance_ai.agents.llm_factory import create_chat_model

    settings = get_settings()
    settings.llm_provider = "openrouter"  # type: ignore[assignment]
    settings.openrouter_model = MODEL_NAME  # type: ignore[assignment]
    return create_chat_model(settings=settings)


def _render_pdf(md_path: str, pdf_path: str) -> None:
    """Convert markdown file to PDF using md_to_pdf script.

    Args:
        md_path: Path to input markdown file.
        pdf_path: Path for output PDF.
    """
    import importlib.util

    script_path = os.path.join(os.path.dirname(__file__), "md_to_pdf.py")
    spec = importlib.util.spec_from_file_location("md_to_pdf", script_path)
    md_to_pdf = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(md_to_pdf)  # type: ignore[union-attr]
    md_to_pdf.render_markdown_to_pdf(md_path, pdf_path)


def main() -> None:
    """Run benchmark, save markdown and PDF."""
    from finance_ai.evaluation.tool_bypass_benchmark import run_tool_bypass_benchmark
    from finance_ai.evaluation.tool_bypass_reporter import save_tool_bypass_report

    print(f"Model: {MODEL_NAME}")
    print("Creating DeepSeek model...")
    model = _create_deepseek_model()

    print("\nRunning benchmark (8 cases x 2 modes = 16 LLM calls)...")
    result = run_tool_bypass_benchmark(model, model_name=MODEL_NAME)

    print("\nResults:")
    print(f"  No-tool accuracy : {result.notool_accuracy_pct:.0f}%")
    print(f"  Tool accuracy    : {result.tool_accuracy_pct:.0f}%")

    md_path = save_tool_bypass_report(result, MD_OUTPUT)
    print(f"\nMarkdown: {md_path}")

    _render_pdf(MD_OUTPUT, PDF_OUTPUT)
    print(f"PDF: {os.path.abspath(PDF_OUTPUT)}")


if __name__ == "__main__":
    main()
