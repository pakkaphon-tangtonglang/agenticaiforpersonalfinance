# mypy: ignore-errors
"""Generate PDF report for DeepSeek bypass tests in EvalPDF style.

Produces a PDF matching the visual style of evaluation_report_3models.pdf:
Cordia font, blue section headers, dark table headers, alternating rows,
pink note_box for failed cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fpdf import FPDF

if TYPE_CHECKING:
    from finance_ai.evaluation.hallucination_evaluator import HallucinationAggregateResult
    from finance_ai.evaluation.tool_bypass_benchmark import ToolBypassBenchmarkResult

FONT_PATH = "C:/Windows/Fonts/cordia.ttc"
OUTPUT_PATH = "data/evaluation/results/deepseek_bypass_report.pdf"


# ---------------------------------------------------------------------------
# EvalPDF (same class as generate_eval_pdf.py)
# ---------------------------------------------------------------------------

class EvalPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("Cordia", "", FONT_PATH)
        self.add_font("Cordia", "B", FONT_PATH)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Cordia", "B", 12)
        self.cell(
            0, 8,
            "Personal Finance AI - DeepSeek Bypass Evaluation",
            align="C", new_x="LMARGIN", new_y="NEXT",
        )
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Cordia", "", 10)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_font("Cordia", "B", 16)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def sub_title(self, title: str):
        self.set_font("Cordia", "B", 14)
        self.set_text_color(41, 128, 185)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Cordia", "", 13)
        self.multi_cell(0, 7, text)
        self.ln(1)

    def add_table(self, headers: list[str], rows: list[list[str]], col_widths: list[int]):
        self.set_font("Cordia", "B", 12)
        self.set_fill_color(52, 73, 94)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font("Cordia", "", 12)
        for ri, row in enumerate(rows):
            self.set_fill_color(236, 240, 241) if ri % 2 == 0 else self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 7, str(val), border=1, fill=True, align=align)
            self.ln()
        self.ln(3)

    def note_box(self, title: str, body: str):
        self.set_font("Cordia", "B", 11)
        self.set_fill_color(253, 237, 236)
        self.set_text_color(192, 57, 43)
        self.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Cordia", "", 11)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 6, body, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _trunc(text: str, n: int = 60) -> str:
    return text[:n] + "..." if len(text) > n else text


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def _build_title_page(pdf: EvalPDF, model_name: str, timestamp: str) -> None:
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Cordia", "B", 28)
    pdf.cell(0, 15, "Personal Finance AI", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Cordia", "B", 22)
    pdf.cell(0, 12, "DeepSeek Bypass Evaluation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Cordia", "", 16)
    pdf.cell(0, 10, f"Model: {model_name}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.cell(
        0, 10,
        "2 Tests: Hallucination (No RAG) + Tax Accuracy (No Tool)",
        align="C", new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(20)
    pdf.set_font("Cordia", "", 14)
    pdf.cell(0, 8, f"Date: {timestamp[:10]}", align="C", new_x="LMARGIN", new_y="NEXT")


def _build_hallucination_page(
    pdf: EvalPDF,
    result: "HallucinationAggregateResult",
) -> None:
    pdf.add_page()
    pdf.section_title("1. Hallucination Test (No RAG, No Tools)")
    pdf.body_text(
        f"Compliance Rate: {_fmt_pct(float(result.compliance_rate) * 100)}"
        f"  ({result.compliant_count}/{result.total_cases} compliant)"
    )
    pdf.body_text(
        "Method: DeepSeek called directly (no RAG, no agent graph)."
        " Checks if LLM returns correct Thai tax facts from training knowledge."
    )

    pdf.sub_title("Per-Case Results")
    headers = ["Case", "Query", "Pass?", "Violations"]
    col_w = [20, 80, 18, 72]
    rows = []
    for r in result.results:
        ok = "Y" if r.is_compliant else "N"
        viols = "; ".join(r.violations_found[:2]) if r.violations_found else "-"
        rows.append([r.case_id, _trunc(r.query, 55), ok, _trunc(viols, 50)])
    pdf.add_table(headers, rows, col_w)

    # Full response for ALL cases (not just failed)
    pdf.sub_title(f"Full LLM Responses ({len(result.results)} cases)")
    for r in result.results:
        status = "PASS" if r.is_compliant else "FAIL"
        viols_text = ""
        if r.violations_found:
            viols_text = "\nViolations: " + "; ".join(r.violations_found)
        body = (
            f"Query: {r.query}\n"
            f"Full Response:\n{r.agent_response}"
            f"{viols_text}"
        )
        if r.is_compliant:
            pdf.sub_title(f"  {r.case_id} [{status}]")
            pdf.body_text(body)
        else:
            pdf.note_box(f"{r.case_id} [{status}] — {r.category}", body)


def _build_tax_bypass_page(
    pdf: EvalPDF,
    result: "ToolBypassBenchmarkResult",
) -> None:
    pdf.add_page()
    pdf.section_title("2. Tax Accuracy: No Tool (LLM calculates itself)")

    total = len(result.cases)
    ok_count = sum(1 for c in result.cases if c.notool_within_tolerance)
    pdf.body_text(
        f"Accuracy: {_fmt_pct(result.notool_accuracy_pct)}"
        f"  ({ok_count}/{total} within tolerance)"
    )
    pdf.body_text(
        "Method: DeepSeek asked to compute Thai income tax with NO tools available."
        " Ground truth is Python calculate_tax() function."
    )

    pdf.sub_title("Per-Case Results")
    headers = ["Case", "Expected (THB)", "LLM Answer", "Error (THB)", "Pass?"]
    col_w = [25, 38, 38, 38, 22]
    rows = []
    for c in result.cases:
        expected = f"{int(c.expected_tax):,}"
        llm_ans = f"{int(c.notool_extracted_tax):,}" if c.notool_extracted_tax is not None else "N/A"
        if c.notool_extracted_tax is not None:
            err = f"{int(abs(c.notool_extracted_tax - c.expected_tax)):,}"
        else:
            err = "N/A"
        ok = "Y" if c.notool_within_tolerance else "N"
        rows.append([c.case_id, expected, llm_ans, err, ok])
    pdf.add_table(headers, rows, col_w)

    pdf.sub_title("Tool Ground Truth vs LLM")
    headers2 = ["Case", "Expected (THB)", "Tool Answer", "Pass?"]
    col_w2 = [25, 50, 50, 36]
    rows2 = []
    for c in result.cases:
        expected = f"{int(c.expected_tax):,}"
        tool_ans = f"{int(c.tool_tax):,}"
        ok = "Y" if c.tool_within_tolerance else "N"
        rows2.append([c.case_id, expected, tool_ans, ok])
    pdf.add_table(headers2, rows2, col_w2)

    # Full LLM response for ALL cases
    pdf.sub_title(f"Full LLM Responses ({len(result.cases)} cases)")
    for c in result.cases:
        status = "PASS" if c.notool_within_tolerance else "FAIL"
        if c.notool_extracted_tax is not None:
            extracted_info = f"Extracted number: {int(c.notool_extracted_tax):,} THB"
        else:
            extracted_info = "Extracted number: N/A (could not parse)"
        body = (
            f"Query: {c.query}\n"
            f"Expected: {int(c.expected_tax):,} THB  |  Tool: {int(c.tool_tax):,} THB\n"
            f"{extracted_info}\n"
            f"Full Response:\n{c.notool_response}"
        )
        if c.notool_within_tolerance:
            pdf.sub_title(f"  {c.case_id} [{status}]")
            pdf.body_text(body)
        else:
            pdf.note_box(f"{c.case_id} [{status}]: {c.description}", body)


def _build_summary_page(
    pdf: EvalPDF,
    hal: "HallucinationAggregateResult",
    tax: "ToolBypassBenchmarkResult",
) -> None:
    pdf.add_page()
    pdf.section_title("3. Summary")

    hal_ok = sum(1 for r in hal.results if r.is_compliant)
    tax_ok = sum(1 for c in tax.cases if c.notool_within_tolerance)
    tax_tool_ok = sum(1 for c in tax.cases if c.tool_within_tolerance)

    headers = ["Dimension", "DeepSeek (No RAG/Tool)", "Python Tool"]
    col_w = [70, 65, 50]
    rows = [
        ["Hallucination Compliance",
         f"{_fmt_pct(float(hal.compliance_rate) * 100)} ({hal_ok}/{hal.total_cases})", "N/A"],
        ["Tax Accuracy",
         f"{_fmt_pct(tax.notool_accuracy_pct)} ({tax_ok}/{len(tax.cases)})",
         f"{_fmt_pct(tax.tool_accuracy_pct)} ({tax_tool_ok}/{len(tax.cases)})"],
        ["Tax Accuracy Gap",
         f"{tax.tool_accuracy_pct - tax.notool_accuracy_pct:+.1f}% vs tool", "Baseline"],
    ]
    pdf.add_table(headers, rows, col_w)

    pdf.sub_title("Key Findings")
    pdf.body_text(
        f"1. Hallucination: DeepSeek answered {hal_ok}/{hal.total_cases} tax fact"
        f" questions correctly without RAG ({_fmt_pct(float(hal.compliance_rate) * 100)})."
    )
    pdf.body_text(
        f"2. Tax Calculation: Without tools, DeepSeek computed tax correctly"
        f" in only {tax_ok}/{len(tax.cases)} cases ({_fmt_pct(tax.notool_accuracy_pct)})."
    )
    pdf.body_text(
        f"3. Python calculate_tax() achieved {_fmt_pct(tax.tool_accuracy_pct)}"
        " accuracy — proving tool-based calculation is essential."
    )
    pdf.body_text(
        "4. Common LLM errors: wrong expense deduction (50% cap),"
        " incorrect tax bracket boundaries, wrong child/RMF deduction limits."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_bypass_pdf(
    hal_result: "HallucinationAggregateResult",
    tax_result: "ToolBypassBenchmarkResult",
    model_name: str,
    timestamp: str,
    output_path: str = OUTPUT_PATH,
) -> str:
    """Generate DeepSeek bypass evaluation PDF.

    Args:
        hal_result: Hallucination evaluation result.
        tax_result: Tool-bypass benchmark result.
        model_name: Model display name.
        timestamp: ISO timestamp string.
        output_path: Path for output PDF.

    Returns:
        Absolute path of the written PDF.
    """
    pdf = EvalPDF()
    pdf.alias_nb_pages()

    _build_title_page(pdf, model_name, timestamp)
    _build_hallucination_page(pdf, hal_result)
    _build_tax_bypass_page(pdf, tax_result)
    _build_summary_page(pdf, hal_result, tax_result)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)
    return str(Path(output_path).resolve())
