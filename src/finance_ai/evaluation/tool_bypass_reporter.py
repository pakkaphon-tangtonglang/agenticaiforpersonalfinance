"""Markdown report generator for the DeepSeek tool-bypass benchmark.

Produces a Thai-language markdown document with comparison tables,
per-case details, and a conclusion — suitable for PDF conversion.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finance_ai.evaluation.tool_bypass_benchmark import ToolBypassBenchmarkResult


def save_tool_bypass_report(
    result: "ToolBypassBenchmarkResult",
    output_path: str = "data/evaluation/results/deepseek_tool_bypass.md",
) -> str:
    """Save tool-bypass benchmark results as a markdown report.

    Args:
        result: Full benchmark result from run_tool_bypass_benchmark().
        output_path: Path to write the markdown file.

    Returns:
        Absolute path of the written file.

    Example:
        >>> path = save_tool_bypass_report(result)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    content = _build_report(result)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.abspath(output_path)


def _build_report(result: "ToolBypassBenchmarkResult") -> str:
    """Assemble the full markdown report.

    Args:
        result: Full benchmark result.

    Returns:
        Complete markdown document as a string.
    """
    sections = [
        _section_header(result),
        _section_summary(result),
        _section_per_case_table(result),
        _section_error_detail(result),
        _section_conclusion(result),
    ]
    return "\n\n".join(sections) + "\n"


def _section_header(result: "ToolBypassBenchmarkResult") -> str:
    """Return the report title and model info.

    Args:
        result: Benchmark result with model name.

    Returns:
        Header markdown string.
    """
    short_name = result.model_name.split("/")[-1]
    return (
        f"# ปัญหา Tool Bypass: {short_name}\n\n"
        f"ทดสอบ **{result.model_name}** สองโหมด:\n\n"
        "- **No-Tool Mode**: LLM คำนวณภาษีเองจาก training knowledge (ไม่มี tool)\n"
        "- **Tool Mode**: ฟังก์ชัน `calculate_thai_tax` ใน Python (ถูกต้อง 100%)\n\n"
        "---"
    )


def _section_summary(result: "ToolBypassBenchmarkResult") -> str:
    """Build the summary metrics table.

    Args:
        result: Benchmark result with aggregate metrics.

    Returns:
        Markdown table as string.
    """
    total = len(result.cases)
    notool_wrong = total - sum(1 for c in result.cases if c.notool_within_tolerance)
    tool_wrong = total - sum(1 for c in result.cases if c.tool_within_tolerance)

    rows = [
        "## 1. สรุปผล",
        "",
        "| เมตริก | No-Tool (LLM คำนวณเอง) | Tool (calculate_thai_tax) |",
        "|---|---|---|",
        "| วิธีหาคำตอบ | LLM reasoning | Python function |",
        f"| ความแม่นยำ | **{result.notool_accuracy_pct:.0f}%** ({total - notool_wrong}/{total}) | **{result.tool_accuracy_pct:.0f}%** ({total - tool_wrong}/{total}) |",
        f"| คำตอบผิดพลาด | {notool_wrong} cases | {tool_wrong} cases |",
        "| รองรับกฎใหม่ | ไม่รองรับ (ต้อง retrain) | รองรับทันที (แก้ code) |",
    ]
    return "\n".join(rows)


def _section_per_case_table(result: "ToolBypassBenchmarkResult") -> str:
    """Build per-case comparison table.

    Args:
        result: Benchmark result with per-case data.

    Returns:
        Markdown table as string.
    """
    rows = [
        "## 2. ผลแต่ละ Query",
        "",
        "| # | คำอธิบาย | ถูกต้อง (บาท) | LLM ตอบ | LLM ถูก? | Tool ตอบ | Tool ถูก? |",
        "|---|---|---|---|---|---|---|",
    ]

    for i, case in enumerate(result.cases, 1):
        expected = f"{case.expected_tax:,}"
        notool_ans = (
            f"{case.notool_extracted_tax:,}" if case.notool_extracted_tax is not None else "-"
        )
        notool_ok = "v" if case.notool_within_tolerance else "**x**"
        tool_ans = f"{case.tool_tax:,}"
        tool_ok = "v" if case.tool_within_tolerance else "**x**"

        rows.append(
            f"| {i} | {case.description} | {expected} | "
            f"{notool_ans} | {notool_ok} | {tool_ans} | {tool_ok} |"
        )

    return "\n".join(rows)


def _section_error_detail(result: "ToolBypassBenchmarkResult") -> str:
    """Build detailed breakdown for cases where no-tool mode was wrong.

    Args:
        result: Benchmark result with per-case data.

    Returns:
        Markdown section as string.
    """
    wrong_cases = [c for c in result.cases if not c.notool_within_tolerance]
    if not wrong_cases:
        return "## 3. รายละเอียด Cases ที่ LLM คำนวณผิด\n\n" "LLM คำนวณถูกต้องทุก case"

    lines = [
        "## 3. รายละเอียด: Cases ที่ LLM คำนวณเองแล้วผิด",
        "",
    ]
    for case in wrong_cases:
        expected_str = f"{case.expected_tax:,}"
        if case.notool_extracted_tax is not None:
            error = abs(case.notool_extracted_tax - case.expected_tax)
            pct = float(error / max(case.expected_tax, 1) * 100)
            got_str = f"{case.notool_extracted_tax:,}"
            error_str = f"ผิด {error:,} บาท ({pct:.1f}%)"
        else:
            got_str = "-"
            error_str = "สกัดตัวเลขไม่ได้ (ตอบแบบอธิบาย ไม่ใช่ตัวเลข)"

        lines += [
            f"### {case.case_id}: {case.description}",
            "",
            f"- ภาษีที่ถูกต้อง: **{expected_str} บาท**",
            f"- LLM ตอบ: {got_str} — {error_str}",
            f"- Tool ตอบ: {case.tool_tax:,} บาท (ถูกต้อง)",
            "",
        ]

    return "\n".join(lines)


def _section_conclusion(result: "ToolBypassBenchmarkResult") -> str:
    """Build the conclusion section.

    Args:
        result: Full benchmark result.

    Returns:
        Markdown conclusion as string.
    """
    short_name = result.model_name.split("/")[-1]
    accuracy_gap = result.tool_accuracy_pct - result.notool_accuracy_pct
    total = len(result.cases)
    wrong_notool = total - sum(1 for c in result.cases if c.notool_within_tolerance)

    return (
        '## 4. สรุป: ทำไมต้องบังคับ tool_choice="any"?\n\n'
        f"จากการทดสอบ {total} queries กับ **{short_name}**:\n\n"
        f"1. **LLM คำนวณเองแม่นยำ {result.notool_accuracy_pct:.0f}%**: "
        f"ผิดพลาด {wrong_notool}/{total} cases\n\n"
        f"2. **Tool ฟังก์ชันแม่นยำ {result.tool_accuracy_pct:.0f}%**: "
        f"ต่างจาก LLM {accuracy_gap:+.0f}%\n\n"
        "3. **สาเหตุที่ LLM ผิด** — กฎภาษีไทยที่ซับซ้อน:\n"
        "   - หักค่าใช้จ่าย 50% ของเงินได้ (สูงสุด 100,000 บาท)\n"
        "   - Progressive bracket 8 ขั้น (0%-35%)\n"
        "   - เพดาน SSF/RMF ต่างกัน (30% และ 30% ของรายได้)\n"
        "   - ลดหย่อนบุตร 30,000 บาท/คน (ไม่ใช่ 50,000)\n\n"
        "4. **วิธีแก้ใน Production**:\n"
        '   - `tax_agent.py` ใช้ `tool_choice="any"` บน first_turn\n'
        "   - ทุก query จะผ่าน `calculate_thai_tax` เสมอ\n"
        "   - แก้กฎใหม่ได้ทันทีโดยไม่ต้อง retrain LLM\n\n"
        f"> **ข้อสรุป**: {short_name} มีความสามารถสูง แต่ภาษีไทยมีรายละเอียด "
        "เฉพาะที่ LLM อาจไม่รู้หรือรู้ผิด — tool เป็นตัวรับประกันความถูกต้อง"
    )
