"""Pure functions that create Plotly figures for financial data.

Each function takes simple data (dicts/lists) and returns a
plotly.graph_objects.Figure. No Streamlit dependency here — this
module is purely data-to-figure so it's easy to unit test.
"""

from decimal import Decimal
from typing import Any

import plotly.graph_objects as go

THAI_CATEGORY_LABELS: dict[str, str] = {
    "food": "อาหาร",
    "transport": "เดินทาง",
    "health": "สุขภาพ",
    "education": "การศึกษา",
    "shopping": "ช้อปปิ้ง",
    "utilities": "สาธารณูปโภค",
    "entertainment": "บันเทิง",
    "other": "อื่นๆ",
}


def _to_float(value: Any) -> float:
    """Convert Decimal/str/int to float for Plotly.

    Args:
        value: A numeric value (Decimal, str, int, or float).

    Returns:
        Python float.
    """
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _detect_template() -> str:
    """Return Plotly template name matching Streamlit theme.

    Returns:
        'plotly_dark' or 'plotly_white'.
    """
    try:
        import streamlit as st  # noqa: PLC0415

        base = st.get_option("theme.base")
        if base == "dark":
            return "plotly_dark"
    except Exception:  # noqa: BLE001
        pass
    return "plotly_white"


def _thai_label(category: str) -> str:
    """Translate English category key to Thai label.

    Args:
        category: English category key.

    Returns:
        Thai label string.
    """
    return THAI_CATEGORY_LABELS.get(category, category)


def create_expense_pie_chart(
    categories: list[dict[str, Any]],
) -> go.Figure:
    """Create a pie chart for expense breakdown by category.

    Args:
        categories: List of dicts with 'category' and 'amount' keys.

    Returns:
        Plotly Figure with a pie chart.

    Example:
        >>> fig = create_expense_pie_chart([
        ...     {"category": "food", "amount": "15000"},
        ... ])
    """
    labels = [_thai_label(c.get("category", "other")) for c in categories]
    values = [_to_float(c.get("amount", 0)) for c in categories]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo="label+percent",
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="ค่าใช้จ่ายแยกตามหมวดหมู่",
        template=_detect_template(),
        showlegend=True,
        margin={"t": 60, "b": 20, "l": 20, "r": 20},
        height=400,
    )
    return fig


def create_portfolio_pie_chart(
    holdings: list[dict[str, Any]],
) -> go.Figure:
    """Create a pie chart for portfolio allocation by holding.

    Args:
        holdings: List of dicts with 'symbol' and 'current_value' keys.

    Returns:
        Plotly Figure with a pie chart.

    Example:
        >>> fig = create_portfolio_pie_chart([
        ...     {"symbol": "PTT.BK", "current_value": "100000"},
        ... ])
    """
    labels = [h.get("symbol", "N/A") for h in holdings]
    values = [_to_float(h.get("current_value", 0)) for h in holdings]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo="label+percent",
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="สัดส่วนพอร์ตการลงทุน",
        template=_detect_template(),
        showlegend=True,
        margin={"t": 60, "b": 20, "l": 20, "r": 20},
        height=400,
    )
    return fig


def create_goal_progress_bar(
    goals: list[dict[str, Any]],
) -> go.Figure:
    """Create a horizontal bar chart for goal progress.

    Args:
        goals: List of dicts with 'name' and 'percentage' keys.

    Returns:
        Plotly Figure with horizontal bars.

    Example:
        >>> fig = create_goal_progress_bar([
        ...     {"name": "เกษียณ", "percentage": "45"},
        ... ])
    """
    names = [g.get("name", "เป้าหมาย") for g in goals]
    percentages = [min(_to_float(g.get("percentage", 0)), 100) for g in goals]

    colors = ["#51CF66" if p >= 80 else "#FFA94D" if p >= 50 else "#FF6B6B" for p in percentages]

    fig = go.Figure(
        data=[
            go.Bar(
                x=percentages,
                y=names,
                orientation="h",
                marker_color=colors,
                text=[f"{p:.0f}%" for p in percentages],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="ความคืบหน้าเป้าหมายการเงิน",
        xaxis={"range": [0, 100], "title": "เปอร์เซ็นต์"},
        template=_detect_template(),
        margin={"t": 60, "b": 40, "l": 120, "r": 20},
        height=max(200, len(goals) * 60 + 100),
    )
    return fig


def create_health_gauge(score: int) -> go.Figure:
    """Create a gauge chart for financial health score.

    Args:
        score: Health score 0-100.

    Returns:
        Plotly Figure with a gauge indicator.

    Example:
        >>> fig = create_health_gauge(75)
    """
    color = _score_color(score)

    fig = go.Figure(
        data=[
            go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "คะแนนสุขภาพการเงิน"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(255,107,107,0.2)"},
                        {"range": [40, 70], "color": "rgba(255,169,77,0.2)"},
                        {"range": [70, 100], "color": "rgba(81,207,102,0.2)"},
                    ],
                },
                number={"suffix": "/100"},
            )
        ]
    )
    fig.update_layout(
        template=_detect_template(),
        margin={"t": 80, "b": 20, "l": 40, "r": 40},
        height=300,
    )
    return fig


def _score_color(score: int) -> str:
    """Return color for a health score value.

    Args:
        score: Health score 0-100.

    Returns:
        Hex color string.
    """
    if score >= 70:
        return "#51CF66"
    if score >= 40:
        return "#FFA94D"
    return "#FF6B6B"


def create_income_vs_expense_bar(
    income: Decimal,
    expenses: Decimal,
) -> go.Figure:
    """Create a bar chart comparing income and expenses.

    Args:
        income: Monthly income amount.
        expenses: Monthly expense amount.

    Returns:
        Plotly Figure with side-by-side bars.

    Example:
        >>> fig = create_income_vs_expense_bar(
        ...     Decimal("50000"), Decimal("35000")
        ... )
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=["รายได้", "รายจ่าย", "เงินออม"],
                y=[
                    _to_float(income),
                    _to_float(expenses),
                    _to_float(income - expenses),
                ],
                marker_color=["#339AF0", "#FF6B6B", "#51CF66"],
                text=[
                    f"฿{_to_float(income):,.0f}",
                    f"฿{_to_float(expenses):,.0f}",
                    f"฿{_to_float(income - expenses):,.0f}",
                ],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="รายได้ vs รายจ่าย (รายเดือน)",
        yaxis={"title": "บาท"},
        template=_detect_template(),
        margin={"t": 60, "b": 40, "l": 60, "r": 20},
        height=400,
    )
    return fig
