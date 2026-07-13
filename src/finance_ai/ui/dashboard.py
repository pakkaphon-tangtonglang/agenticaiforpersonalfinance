"""Streamlit dashboard view with Plotly charts.

Renders a financial overview dashboard using data from the
existing report_service.generate_financial_report().
"""

from datetime import datetime
from typing import Any, Callable

import streamlit as st
from sqlalchemy.orm import Session

from finance_ai.tools.report_models import (
    ExpenseBreakdownSection,
    FinancialReport,
    GoalProgressSection,
    InvestmentPortfolioSection,
    MonthlyOverviewSection,
)
from finance_ai.ui.charts import (
    create_expense_pie_chart,
    create_goal_progress_bar,
    create_health_gauge,
    create_income_vs_expense_bar,
    create_portfolio_pie_chart,
)


def render_dashboard(
    user_id: str,
    db_session_factory: Callable[[], Session],
) -> None:
    """Render the full financial dashboard.

    Args:
        user_id: UUID of the current user.
        db_session_factory: Callable that creates DB sessions.
    """
    year, month = _render_period_selector()
    report = _load_report(user_id, db_session_factory, year, month)
    if report is None:
        st.info("ยังไม่มีข้อมูลเพียงพอสำหรับแสดง Dashboard")
        return

    if _is_empty_report(report):
        thai_months = [
            "",
            "ม.ค.",
            "ก.พ.",
            "มี.ค.",
            "เม.ย.",
            "พ.ค.",
            "มิ.ย.",
            "ก.ค.",
            "ส.ค.",
            "ก.ย.",
            "ต.ค.",
            "พ.ย.",
            "ธ.ค.",
        ]
        month_name = thai_months[month]
        st.info(
            f"ไม่พบข้อมูลสำหรับ {month_name} {year}  \n"
            "ลองเลือกเดือนอื่น หรือนำเข้า Bank Statement ที่แท็บ **📂 นำเข้าข้อมูล**"
        )
        return

    _render_header(report)
    _render_top_metrics(report)
    _render_charts(report)
    _render_highlights(report)


def _render_period_selector() -> tuple[int, int]:
    """Render year/month selector for the dashboard.

    Returns:
        Tuple of (year, month).
    """
    now = datetime.now()
    col_year, col_month = st.columns(2)
    with col_year:
        year = st.selectbox(
            "ปี",
            options=list(range(now.year, now.year - 3, -1)),
            index=0,
            key="dashboard_year",
        )
    with col_month:
        thai_months = [
            "ม.ค.",
            "ก.พ.",
            "มี.ค.",
            "เม.ย.",
            "พ.ค.",
            "มิ.ย.",
            "ก.ค.",
            "ส.ค.",
            "ก.ย.",
            "ต.ค.",
            "พ.ย.",
            "ธ.ค.",
        ]
        month = st.selectbox(
            "เดือน",
            options=list(range(1, 13)),
            format_func=lambda m: thai_months[m - 1],
            index=now.month - 1,
            key="dashboard_month",
        )
    return year, month


def _is_empty_report(report: FinancialReport) -> bool:
    """Return True when the report has no meaningful financial data.

    Args:
        report: The generated financial report.

    Returns:
        True if expenses, income, investments, and goals are all absent.
    """
    return (
        report.monthly_overview.total_expenses == 0
        and report.monthly_overview.total_income == 0
        and report.investment_portfolio.holding_count == 0
        and report.goal_progress.total_goals == 0
    )


def _load_report(
    user_id: str,
    db_session_factory: Callable[[], Session],
    year: int,
    month: int,
) -> FinancialReport | None:
    """Load financial report data from the DB.

    Args:
        user_id: UUID of the user.
        db_session_factory: Session factory callable.
        year: Year to report on.
        month: Month to report on.

    Returns:
        FinancialReport or None if loading fails.
    """
    from finance_ai.tools.report_service import (  # noqa: PLC0415
        generate_financial_report,
    )

    session = db_session_factory()
    try:
        return generate_financial_report(session, user_id, year, month)
    except Exception as exc:  # noqa: BLE001
        st.error(f"โหลดข้อมูล Dashboard ไม่สำเร็จ: {exc}")
        return None
    finally:
        session.close()


def _render_header(report: FinancialReport) -> None:
    """Render dashboard header with report period.

    Args:
        report: The financial report.
    """
    thai_months = [
        "",
        "ม.ค.",
        "ก.พ.",
        "มี.ค.",
        "เม.ย.",
        "พ.ค.",
        "มิ.ย.",
        "ก.ค.",
        "ส.ค.",
        "ก.ย.",
        "ต.ค.",
        "พ.ย.",
        "ธ.ค.",
    ]
    month_name = thai_months[report.month] if report.month else ""
    st.subheader(f"ภาพรวมการเงิน — {month_name} {report.year}")


def _render_top_metrics(report: FinancialReport) -> None:
    """Render top-level metric cards.

    Args:
        report: The financial report.
    """
    overview = report.monthly_overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("รายได้/เดือน", _fmt(overview.total_income))
    with col2:
        st.metric("รายจ่าย/เดือน", _fmt(overview.total_expenses))
    with col3:
        st.metric("เงินออม", _fmt(overview.net_savings))
    with col4:
        rate = int(overview.savings_rate * 100)
        st.metric("อัตราการออม", f"{rate}%")


def _fmt(amount: Any) -> str:
    """Format amount as Thai currency string.

    Args:
        amount: Numeric amount.

    Returns:
        Formatted string like '฿50,000'.
    """
    return f"฿{float(amount):,.0f}"


def _render_charts(report: FinancialReport) -> None:
    """Render all chart sections.

    Args:
        report: The financial report.
    """
    _render_income_expense_chart(report.monthly_overview)

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        _render_expense_chart(report.expense_breakdown)
    with col_right:
        _render_portfolio_chart(report.investment_portfolio)

    st.divider()
    _render_goals_chart(report.goal_progress)

    st.divider()
    _render_health_chart(report.health_score)


def _render_income_expense_chart(
    overview: MonthlyOverviewSection,
) -> None:
    """Render income vs expense bar chart.

    Args:
        overview: Monthly overview section.
    """
    fig = create_income_vs_expense_bar(overview.total_income, overview.total_expenses)
    st.plotly_chart(fig, use_container_width=True)


def _render_expense_chart(
    breakdown: ExpenseBreakdownSection,
) -> None:
    """Render expense breakdown pie chart.

    Args:
        breakdown: Expense breakdown section.
    """
    if not breakdown.categories:
        st.caption("ยังไม่มีข้อมูลค่าใช้จ่าย")
        return
    fig = create_expense_pie_chart(breakdown.categories)
    st.plotly_chart(fig, use_container_width=True)


def _render_portfolio_chart(
    portfolio: InvestmentPortfolioSection,
) -> None:
    """Render portfolio allocation pie chart.

    Args:
        portfolio: Investment portfolio section.
    """
    if not portfolio.holdings:
        st.caption("ยังไม่มีข้อมูลการลงทุน")
        return
    fig = create_portfolio_pie_chart(portfolio.holdings)
    st.plotly_chart(fig, use_container_width=True)


def _render_goals_chart(
    goals: GoalProgressSection,
) -> None:
    """Render goal progress bar chart.

    Args:
        goals: Goal progress section.
    """
    if not goals.goals:
        st.caption("ยังไม่มีเป้าหมายการเงิน")
        return
    fig = create_goal_progress_bar(goals.goals)
    st.plotly_chart(fig, use_container_width=True)


def _render_health_chart(score: int) -> None:
    """Render financial health score gauge.

    Args:
        score: Health score 0-100.
    """
    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        fig = create_health_gauge(score)
        st.plotly_chart(fig, use_container_width=True)


def _render_highlights(report: FinancialReport) -> None:
    """Render report highlights as info/warning boxes.

    Args:
        report: The financial report.
    """
    if not report.highlights:
        return

    st.subheader("ไฮไลท์สำคัญ")
    for highlight in report.highlights:
        _show_highlight(highlight.highlight_type, highlight.title, highlight.description)


def _show_highlight(
    highlight_type: str,
    title: str,
    description: str,
) -> None:
    """Show a single highlight in appropriate Streamlit format.

    Args:
        highlight_type: 'achievement', 'warning', or 'info'.
        title: Highlight title.
        description: Highlight description.
    """
    text = f"**{title}**  \n{description}"
    if highlight_type == "warning":
        st.warning(text)
    elif highlight_type == "achievement":
        st.success(text)
    else:
        st.info(text)
