"""LangGraph tool wrappers for autonomous financial report generation.

Tools gather all user financial data and build a structured report
for the LLM to format and present to the user.
"""

from datetime import date
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.session_helper import get_tool_session


@tool
def generate_financial_report_tool(
    year: str = "",
    month: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """สร้างรายงานการเงินฉบับสมบูรณ์อัตโนมัติ.

    รวบรวมข้อมูลจากทุกด้าน (รายได้, ค่าใช้จ่าย, การลงทุน,
    เป้าหมาย, ภาษี) แล้วสร้างรายงานวิเคราะห์ 7 ส่วน

    Args:
        year: ปีที่ต้องการรายงาน (ค่าเริ่มต้น: ปีปัจจุบัน).
        month: เดือนที่ต้องการรายงาน (ค่าเริ่มต้น: เดือนปัจจุบัน).
        user_id: UUID ของผู้ใช้ (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with complete financial report data.
    """
    today = date.today()
    parsed_year = int(year) if year else today.year
    parsed_month = int(month) if month else today.month
    return _fetch_report(db_session_factory, user_id, parsed_year, parsed_month)


def _fetch_report(
    db_session_factory: Any,
    user_id: str,
    year: int,
    month: int,
) -> dict[str, Any]:
    """Fetch report via the service layer.

    Args:
        db_session_factory: Session factory or None.
        user_id: UUID of the user.
        year: Year to report.
        month: Month to report.

    Returns:
        Serialized FinancialReport dict.
    """
    from finance_ai.tools.report_service import (  # noqa: PLC0415
        generate_financial_report,
    )

    with get_tool_session(db_session_factory) as session:
        report = generate_financial_report(session, user_id, year, month)
    return _serialize_report(report)


def _serialize_report(report: Any) -> dict[str, Any]:
    """Convert FinancialReport to LLM-friendly dict.

    Args:
        report: FinancialReport instance.

    Returns:
        Dict with all Decimal values converted to strings.
    """
    return {
        "action": "financial_report",
        "user_id": report.user_id,
        "generated_at": report.generated_at,
        "report_type": report.report_type,
        "year": report.year,
        "month": report.month,
        "health_score": report.health_score,
        "monthly_overview": _serialize_section(report.monthly_overview),
        "expense_breakdown": _serialize_section(report.expense_breakdown),
        "investment_portfolio": _serialize_section(report.investment_portfolio),
        "goal_progress": _serialize_section(report.goal_progress),
        "tax_status": _serialize_section(report.tax_status),
        "highlights": [
            {
                "type": h.highlight_type,
                "title": h.title,
                "description": h.description,
            }
            for h in report.highlights
        ],
    }


def _serialize_section(section: Any) -> dict[str, Any]:
    """Serialize a Pydantic section model to dict with string values.

    Args:
        section: A Pydantic BaseModel section instance.

    Returns:
        Dict with Decimal values converted to strings.
    """
    data = section.model_dump()
    return {k: str(v) if hasattr(v, "quantize") else v for k, v in data.items()}


@tool
def get_financial_summary(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """ดึงภาพรวมการเงินแบบย่อ (สรุปสั้น).

    แสดงเฉพาะรายได้, ค่าใช้จ่าย, เงินออม, และคะแนนสุขภาพการเงิน

    Args:
        user_id: UUID ของผู้ใช้ (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with summary overview and health score.
    """
    today = date.today()
    report_data = _fetch_report(db_session_factory, user_id, today.year, today.month)
    overview = report_data["monthly_overview"]
    return {
        "action": "financial_summary",
        "health_score": report_data["health_score"],
        "total_income": overview.get("total_income", "0"),
        "total_expenses": overview.get("total_expenses", "0"),
        "net_savings": overview.get("net_savings", "0"),
        "savings_rate": overview.get("savings_rate", "0"),
        "highlight_count": len(report_data.get("highlights", [])),
    }


REPORT_TOOLS = [
    generate_financial_report_tool,
    get_financial_summary,
]
