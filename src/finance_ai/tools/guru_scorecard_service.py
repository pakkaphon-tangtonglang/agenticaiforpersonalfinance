"""Deterministic guru scorecards computed from the user's own data.

Inspired by persona-agent designs where code computes the checks and the
LLM only explains them in the guru's voice. Every number here is Decimal
math over data gathered by recommendation_service.gather_all_financial_data;
missing data is reported as "missing", never guessed.
"""

from collections.abc import Callable
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel

CheckStatus = Literal["good", "warn", "bad", "missing"]
FinancialData = dict[str, Any]

GOOD_SAVINGS_RATE = Decimal("0.20")
MIN_SAVINGS_RATE = Decimal("0.10")
GOOD_EMERGENCY_MONTHS = Decimal("6")
MIN_EMERGENCY_MONTHS = Decimal("3")
GOOD_INVESTING_RATE = Decimal("0.10")
MAX_SINGLE_HOLDING_SHARE = Decimal("0.30")
WARN_SINGLE_HOLDING_SHARE = Decimal("0.50")
GOOD_FUND_SHARE = Decimal("0.50")
HIGH_EQUITY_SHARE = Decimal("0.70")
HIGH_RISK_LEVEL = 4


class ScorecardCheck(BaseModel):
    """One deterministic check the guru explains to the user."""

    key: str
    label: str
    value: str | None
    status: CheckStatus
    note: str


class GuruScorecard(BaseModel):
    """All checks for one guru's lens."""

    guru_id: str
    checks: list[ScorecardCheck]


def _to_decimal(raw_value: Any) -> Decimal:
    """Convert a service string amount to Decimal (missing -> 0)."""
    return Decimal(str(raw_value or "0"))


def _ratio(numerator: Decimal, denominator: Decimal) -> Decimal | None:
    """Return numerator / denominator, or None when the denominator is not positive."""
    if denominator <= 0:
        return None
    return numerator / denominator


def savings_rate(data: FinancialData) -> Decimal | None:
    """Share of monthly income not spent.

    Example:
        >>> savings_rate(data)  # income 50,000, expense 35,000
        Decimal('0.3')
    """
    income = _to_decimal(data["income_monthly"].get("total_income"))
    expense = _to_decimal(data["expense"].get("total_amount"))
    return _ratio(income - expense, income)


def emergency_fund_months(data: FinancialData) -> Decimal | None:
    """Months of expenses covered by money saved in emergency-fund goals.

    Example:
        >>> emergency_fund_months(data)  # 120,000 saved, 35,000/month
        Decimal('3.428...')
    """
    saved = sum(
        (
            _to_decimal(goal.get("current_amount"))
            for goal in data["goals"].get("goals", [])
            if goal.get("goal_type") == "emergency_fund"
        ),
        Decimal("0"),
    )
    return _ratio(saved, _to_decimal(data["expense"].get("total_amount")))


def investing_rate(data: FinancialData) -> Decimal | None:
    """Share of monthly income recorded in the 'investment' expense category.

    Example:
        >>> investing_rate(data)  # 5,000 invested of 50,000
        Decimal('0.1')
    """
    invested = sum(
        (
            _to_decimal(row.get("amount"))
            for row in data["expense"].get("category_breakdown", [])
            if row.get("category") == "investment"
        ),
        Decimal("0"),
    )
    return _ratio(invested, _to_decimal(data["income_monthly"].get("total_income")))


def _holding_values(data: FinancialData, asset_type: str | None = None) -> list[Decimal]:
    """Current values of holdings, optionally of one asset type."""
    return [
        _to_decimal(holding.get("current_value"))
        for holding in data["portfolio"].get("holdings", [])
        if asset_type is None or holding.get("asset_type") == asset_type
    ]


def _portfolio_value(data: FinancialData) -> Decimal:
    """Total current portfolio value."""
    return sum(_holding_values(data), Decimal("0"))


def largest_holding_share(data: FinancialData) -> Decimal | None:
    """Biggest single holding as a share of the portfolio.

    Example:
        >>> largest_holding_share(data)  # 60,000 of 100,000
        Decimal('0.6')
    """
    values = _holding_values(data)
    return _ratio(max(values), _portfolio_value(data)) if values else None


def equity_share(data: FinancialData) -> Decimal | None:
    """Share of the portfolio held in individual stocks.

    Example:
        >>> equity_share(data)  # 40,000 stock of 100,000
        Decimal('0.4')
    """
    return _ratio(sum(_holding_values(data, "stock"), Decimal("0")), _portfolio_value(data))


def fund_share(data: FinancialData) -> Decimal | None:
    """Share of the portfolio held in mutual funds (incl. index funds)."""
    return _ratio(sum(_holding_values(data, "mutual_fund"), Decimal("0")), _portfolio_value(data))


def _percent(value: Decimal) -> str:
    """Format a ratio as a 2-decimal percentage string."""
    return f"{(value * 100).quantize(Decimal('0.01'))}%"


def _threshold_check(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    key: str,
    label: str,
    value: Decimal | None,
    good_at: Decimal,
    warn_at: Decimal,
    note: str,
) -> ScorecardCheck:
    """Build a 'higher is better' ratio check (percent display)."""
    if value is None:
        return ScorecardCheck(key=key, label=label, value=None, status="missing", note=note)
    status: CheckStatus = "good" if value >= good_at else "warn" if value >= warn_at else "bad"
    return ScorecardCheck(key=key, label=label, value=_percent(value), status=status, note=note)


def _savings_rate_check(data: FinancialData) -> ScorecardCheck:
    """Savings rate check: good >= 20%, warn >= 10%."""
    return _threshold_check(
        "savings_rate",
        "อัตราการออมต่อเดือน",
        savings_rate(data),
        GOOD_SAVINGS_RATE,
        MIN_SAVINGS_RATE,
        "เป้าหมายทั่วไปคือออมอย่างน้อย 20% ของรายได้",
    )


def _investing_rate_check(data: FinancialData) -> ScorecardCheck:
    """Monthly investing check: good >= 10% of income."""
    return _threshold_check(
        "investing_rate",
        "สัดส่วนเงินลงทุนต่อรายได้",
        investing_rate(data),
        GOOD_INVESTING_RATE,
        Decimal("0.01"),
        "การลงทุนสม่ำเสมอทุกเดือนคือหัวใจของผลตอบแทนทบต้น",
    )


def _emergency_fund_check(data: FinancialData) -> ScorecardCheck:
    """Months of cover: good >= 6, warn >= 3, bad below."""
    months = emergency_fund_months(data)
    note = "เงินสำรองฉุกเฉินที่แนะนำคือ 3-6 เดือนของค่าใช้จ่าย"
    label = "เงินสำรองฉุกเฉิน (เดือน)"
    if months is None:
        return ScorecardCheck(
            key="emergency_fund_months", label=label, value=None, status="missing", note=note
        )
    status: CheckStatus = (
        "good"
        if months >= GOOD_EMERGENCY_MONTHS
        else "warn" if months >= MIN_EMERGENCY_MONTHS else "bad"
    )
    value = str(months.quantize(Decimal("0.1")))
    return ScorecardCheck(
        key="emergency_fund_months", label=label, value=value, status=status, note=note
    )


def _has_goal_check(data: FinancialData, goal_type: str, label: str, note: str) -> ScorecardCheck:
    """Good when the user has a goal of the given type."""
    goal_types = {goal.get("goal_type") for goal in data["goals"].get("goals", [])}
    has_goal = goal_type in goal_types
    return ScorecardCheck(
        key=f"has_{goal_type}_goal",
        label=label,
        value="มี" if has_goal else "ไม่มี",
        status="good" if has_goal else "bad",
        note=note,
    )


def _largest_holding_check(data: FinancialData) -> ScorecardCheck:
    """Lower is better: good <= 30%, warn <= 50%, bad above."""
    share = largest_holding_share(data)
    label = "สัดส่วนสินทรัพย์ตัวที่ใหญ่ที่สุด"
    note = "ไม่ควรมีสินทรัพย์ตัวเดียวเกิน 30% ของพอร์ต"
    if share is None:
        return ScorecardCheck(
            key="largest_holding_share", label=label, value=None, status="missing", note=note
        )
    status: CheckStatus = (
        "good"
        if share <= MAX_SINGLE_HOLDING_SHARE
        else "warn" if share <= WARN_SINGLE_HOLDING_SHARE else "bad"
    )
    return ScorecardCheck(
        key="largest_holding_share", label=label, value=_percent(share), status=status, note=note
    )


def _asset_type_count_check(data: FinancialData) -> ScorecardCheck:
    """Number of asset types held (the app tracks stocks and mutual funds)."""
    asset_types = {holding.get("asset_type") for holding in data["portfolio"].get("holdings", [])}
    note = "แอปติดตามได้แค่หุ้นและกองทุน ทองคำ/พันธบัตรให้ถามผู้ใช้เพิ่ม"
    label = "จำนวนประเภทสินทรัพย์"
    if not asset_types:
        return ScorecardCheck(
            key="asset_type_count", label=label, value=None, status="missing", note=note
        )
    status: CheckStatus = "good" if len(asset_types) >= 2 else "warn"
    return ScorecardCheck(
        key="asset_type_count", label=label, value=str(len(asset_types)), status=status, note=note
    )


def _fund_share_check(data: FinancialData) -> ScorecardCheck:
    """Mutual fund share check: good >= 50% of the portfolio."""
    return _threshold_check(
        "fund_share",
        "สัดส่วนกองทุนรวมในพอร์ต",
        fund_share(data),
        GOOD_FUND_SHARE,
        Decimal("0.01"),
        "คนที่ไม่มีเวลาวิเคราะห์หุ้น กองทุนดัชนีค่าธรรมเนียมต่ำเหมาะกว่า",
    )


def _equity_share_check(data: FinancialData, risk_level: int | None) -> ScorecardCheck:
    """High stock share is good only for users with a high risk level."""
    share = equity_share(data)
    label = "สัดส่วนหุ้นในพอร์ต"
    note = "ลงทุนหุ้นสัดส่วนสูงเหมาะกับคนที่รับความเสี่ยงได้และถือยาว"
    if share is None or risk_level is None:
        value = None if share is None else _percent(share)
        return ScorecardCheck(
            key="equity_share", label=label, value=value, status="missing", note=note
        )
    suits_user = share < HIGH_EQUITY_SHARE or risk_level >= HIGH_RISK_LEVEL
    status: CheckStatus = "good" if suits_user else "warn"
    return ScorecardCheck(
        key="equity_share", label=label, value=_percent(share), status=status, note=note
    )


def _coach_noom_checks(data: FinancialData, _risk_level: int | None) -> list[ScorecardCheck]:
    """โค้ชหนุ่ม: saving first, emergency fund, retirement, steady investing."""
    return [
        _savings_rate_check(data),
        _emergency_fund_check(data),
        _has_goal_check(data, "retirement", "มีเป้าหมายเกษียณ", "ยิ่งเริ่มวางแผนเกษียณเร็วยิ่งดี"),
        _investing_rate_check(data),
    ]


def _buffett_checks(data: FinancialData, _risk_level: int | None) -> list[ScorecardCheck]:
    """Buffett: steady investing, low-cost funds, cash buffer, concentration."""
    return [
        _investing_rate_check(data),
        _fund_share_check(data),
        _emergency_fund_check(data),
        _largest_holding_check(data),
    ]


def _dalio_checks(data: FinancialData, _risk_level: int | None) -> list[ScorecardCheck]:
    """Dalio: diversification across assets and a cash buffer."""
    return [
        _asset_type_count_check(data),
        _largest_holding_check(data),
        _emergency_fund_check(data),
    ]


def _niwes_checks(data: FinancialData, risk_level: int | None) -> list[ScorecardCheck]:
    """ดร.นิเวศน์: stock share vs risk level, steady investing, saving."""
    return [
        _equity_share_check(data, risk_level),
        _investing_rate_check(data),
        _savings_rate_check(data),
    ]


_SCORECARD_BUILDERS: dict[str, Callable[[FinancialData, int | None], list[ScorecardCheck]]] = {
    "coach_noom": _coach_noom_checks,
    "buffett": _buffett_checks,
    "dalio": _dalio_checks,
    "niwes": _niwes_checks,
}


def build_guru_scorecard(
    guru_id: str,
    data: FinancialData,
    risk_level: int | None = None,
) -> GuruScorecard:
    """Run one guru's deterministic checks on the user's financial data.

    Args:
        guru_id: Persona id.
        data: Output of recommendation_service.gather_all_financial_data.
        risk_level: Questionnaire risk level 1-5, or None.

    Returns:
        GuruScorecard with one entry per check.

    Raises:
        ValueError: If guru_id has no scorecard.

    Example:
        >>> build_guru_scorecard("coach_noom", data).checks[0].key
        'savings_rate'
    """
    builder = _SCORECARD_BUILDERS.get(guru_id)
    if builder is None:
        available = ", ".join(sorted(_SCORECARD_BUILDERS))
        raise ValueError(f"No scorecard for guru '{guru_id}'. Available: {available}")
    return GuruScorecard(guru_id=guru_id, checks=builder(data, risk_level))
