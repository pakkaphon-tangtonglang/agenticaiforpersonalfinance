"""PDF and CSV export for financial reports.

Uses fpdf2 for PDF generation (supports Thai via TrueType fonts)
and csv module for CSV export with BOM for Excel compatibility.
"""

import csv
import io
from decimal import Decimal
from pathlib import Path
from typing import Any

from fpdf import FPDF

from finance_ai.tools.report_models import (
    ExpenseBreakdownSection,
    FinancialReport,
    GoalProgressSection,
    InvestmentPortfolioSection,
    MonthlyOverviewSection,
    TaxStatusSection,
)

FONT_DIR = Path(__file__).parent / "fonts"
FONT_FILE = FONT_DIR / "NotoSansThai-Regular.ttf"

THAI_MONTHS = [
    "",
    "มกราคม",
    "กุมภาพันธ์",
    "มีนาคม",
    "เมษายน",
    "พฤษภาคม",
    "มิถุนายน",
    "กรกฎาคม",
    "สิงหาคม",
    "กันยายน",
    "ตุลาคม",
    "พฤศจิกายน",
    "ธันวาคม",
]

CATEGORY_LABELS: dict[str, str] = {
    "food": "อาหาร",
    "transport": "เดินทาง",
    "health": "สุขภาพ",
    "education": "การศึกษา",
    "shopping": "ช้อปปิ้ง",
    "utilities": "สาธารณูปโภค",
    "entertainment": "บันเทิง",
    "other": "อื่นๆ",
}


def format_thai_currency(amount: Decimal | float | str) -> str:
    """Format amount as Thai currency string.

    Args:
        amount: Numeric amount.

    Returns:
        Formatted string like '50,000.00 บาท'.

    Example:
        >>> format_thai_currency(Decimal("50000"))
        '50,000.00 บาท'
    """
    value = float(Decimal(str(amount)))
    return f"{value:,.2f} บาท"


def generate_report_pdf(report: FinancialReport) -> bytes:
    """Generate a PDF report from a FinancialReport model.

    Args:
        report: Complete financial report.

    Returns:
        PDF file contents as bytes.

    Example:
        >>> pdf_bytes = generate_report_pdf(report)
    """
    pdf = _create_pdf()
    pdf.add_page()
    _add_pdf_header(pdf, report)
    _add_monthly_overview(pdf, report.monthly_overview)
    _add_expense_breakdown(pdf, report.expense_breakdown)
    _add_investment_section(pdf, report.investment_portfolio)
    _add_goal_section(pdf, report.goal_progress)
    _add_tax_section(pdf, report.tax_status)
    _add_health_score(pdf, report.health_score)
    return bytes(pdf.output())


def _create_pdf() -> FPDF:
    """Create an FPDF instance with Thai font registered.

    Returns:
        Configured FPDF instance.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    if FONT_FILE.exists():
        pdf.add_font("Thai", "", str(FONT_FILE))
        pdf.set_font("Thai", size=12)
    else:
        pdf.set_font("Helvetica", size=12)
    return pdf


def _add_pdf_header(pdf: FPDF, report: FinancialReport) -> None:
    """Add report title and period header.

    Args:
        pdf: FPDF instance.
        report: Financial report.
    """
    pdf.set_font_size(18)
    pdf.cell(0, 12, "รายงานการเงินส่วนบุคคล", new_x="LMARGIN", new_y="NEXT", align="C")
    month_name = THAI_MONTHS[report.month] if report.month else ""
    pdf.set_font_size(12)
    pdf.cell(0, 8, f"{month_name} {report.year}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)


def _section_title(pdf: FPDF, title: str) -> None:
    """Add a section title with underline.

    Args:
        pdf: FPDF instance.
        title: Section title text.
    """
    pdf.set_font_size(14)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font_size(12)


def _add_row(pdf: FPDF, label: str, value: str) -> None:
    """Add a label-value row.

    Args:
        pdf: FPDF instance.
        label: Row label.
        value: Row value.
    """
    pdf.cell(90, 7, label)
    pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")


def _add_monthly_overview(
    pdf: FPDF,
    overview: MonthlyOverviewSection,
) -> None:
    """Add monthly overview section to PDF.

    Args:
        pdf: FPDF instance.
        overview: Monthly overview data.
    """
    _section_title(pdf, "ภาพรวมรายเดือน")
    _add_row(pdf, "รายได้", format_thai_currency(overview.total_income))
    _add_row(pdf, "รายจ่าย", format_thai_currency(overview.total_expenses))
    _add_row(pdf, "เงินออม", format_thai_currency(overview.net_savings))
    rate = int(overview.savings_rate * 100)
    _add_row(pdf, "อัตราการออม", f"{rate}%")
    pdf.ln(5)


def _add_expense_breakdown(
    pdf: FPDF,
    breakdown: ExpenseBreakdownSection,
) -> None:
    """Add expense breakdown section to PDF.

    Args:
        pdf: FPDF instance.
        breakdown: Expense breakdown data.
    """
    _section_title(pdf, "ค่าใช้จ่ายแยกตามหมวด")
    _add_row(pdf, "รวมค่าใช้จ่าย", format_thai_currency(breakdown.total_amount))
    _add_row(pdf, "จำนวนรายการ", str(breakdown.transaction_count))

    for cat in breakdown.categories:
        name = CATEGORY_LABELS.get(cat.get("category", ""), cat.get("category", ""))
        amount = format_thai_currency(cat.get("amount", "0"))
        pct = cat.get("percentage", "0")
        _add_row(pdf, f"  {name}", f"{amount} ({pct}%)")
    pdf.ln(5)


def _add_investment_section(
    pdf: FPDF,
    portfolio: InvestmentPortfolioSection,
) -> None:
    """Add investment portfolio section to PDF.

    Args:
        pdf: FPDF instance.
        portfolio: Investment portfolio data.
    """
    _section_title(pdf, "พอร์ตการลงทุน")
    _add_row(pdf, "มูลค่ารวม", format_thai_currency(portfolio.total_value))
    _add_row(pdf, "ต้นทุนรวม", format_thai_currency(portfolio.total_cost))
    _add_row(pdf, "กำไร/ขาดทุน", format_thai_currency(portfolio.total_gain_loss))
    _add_row(pdf, "ผลตอบแทน", f"{portfolio.gain_loss_percentage}%")
    pdf.ln(5)


def _add_goal_section(
    pdf: FPDF,
    goals: GoalProgressSection,
) -> None:
    """Add financial goals section to PDF.

    Args:
        pdf: FPDF instance.
        goals: Goal progress data.
    """
    _section_title(pdf, "เป้าหมายการเงิน")
    _add_row(pdf, "เป้าหมายทั้งหมด", str(goals.total_goals))
    _add_row(pdf, "กำลังดำเนินการ", str(goals.active_goals))
    _add_row(pdf, "สำเร็จแล้ว", str(goals.completed_goals))
    _add_row(pdf, "ความคืบหน้ารวม", f"{goals.overall_percentage}%")
    pdf.ln(5)


def _add_tax_section(
    pdf: FPDF,
    tax: TaxStatusSection,
) -> None:
    """Add tax status section to PDF.

    Args:
        pdf: FPDF instance.
        tax: Tax status data.
    """
    _section_title(pdf, f"สถานะภาษีปี {tax.tax_year}")
    _add_row(pdf, "รายได้รวม", format_thai_currency(tax.gross_income))
    _add_row(pdf, "ลดหย่อนรวม", format_thai_currency(tax.total_deductions))
    _add_row(pdf, "ภาษีที่ต้องจ่าย", format_thai_currency(tax.total_tax))
    _add_row(pdf, "อัตราภาษีที่แท้จริง", f"{tax.effective_rate}%")
    pdf.ln(5)


def _add_health_score(pdf: FPDF, score: int) -> None:
    """Add health score section to PDF.

    Args:
        pdf: FPDF instance.
        score: Health score 0-100.
    """
    _section_title(pdf, "คะแนนสุขภาพการเงิน")
    pdf.set_font_size(20)
    pdf.cell(0, 12, f"{score}/100", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font_size(12)


def generate_transactions_csv(
    transactions: list[dict[str, Any]],
) -> str:
    """Generate CSV string from transaction data.

    Includes UTF-8 BOM for Excel Thai compatibility.

    Args:
        transactions: List of transaction dicts with date, type,
            category, description, amount keys.

    Returns:
        CSV string with BOM prefix.

    Example:
        >>> csv_str = generate_transactions_csv([
        ...     {"date": "2026-03-01", "type": "expense",
        ...      "category": "food", "description": "lunch",
        ...      "amount": "350.00"},
        ... ])
    """
    output = io.StringIO()
    output.write("\ufeff")  # BOM for Excel Thai
    writer = csv.writer(output)
    writer.writerow(["วันที่", "ประเภท", "หมวดหมู่", "รายละเอียด", "จำนวนเงิน"])

    for txn in transactions:
        writer.writerow(
            [
                txn.get("date", ""),
                txn.get("type", ""),
                CATEGORY_LABELS.get(txn.get("category", ""), txn.get("category", "")),
                txn.get("description", ""),
                txn.get("amount", ""),
            ]
        )
    return output.getvalue()
