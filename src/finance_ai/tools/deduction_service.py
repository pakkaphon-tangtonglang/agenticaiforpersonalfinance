"""Record the tax deductions a user claims (insurance, RMF/SSF, donations...).

Personal and family allowances are not recorded here: they come from the
user profile in tax_service.build_auto_allowances.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel
from sqlalchemy.orm import Session

from finance_ai.database.crud.deduction_crud import DeductionCRUD
from finance_ai.database.models.deduction import Deduction
from finance_ai.tools.tax_constants import (
    DEDUCTION_LIMITS,
    PERCENTAGE_CAPPED_DEDUCTIONS,
    SOCIAL_SECURITY_RATE,
    SOCIAL_SECURITY_WAGE_CEILING,
    SSF_FINAL_TAX_YEAR,
)

AUTOMATIC_ALLOWANCE_TYPES: frozenset[str] = frozenset(
    {"personal_allowance", "spouse_allowance", "child_allowance", "parent_allowance"}
)
CLAIMABLE_DEDUCTION_TYPES: frozenset[str] = (
    frozenset(DEDUCTION_LIMITS) - AUTOMATIC_ALLOWANCE_TYPES
) | {"donations"}

TAX_SAVING_FUND_MARKERS: dict[str, tuple[str, ...]] = {
    "thai_esg": ("THAIESG", "ไทยเพื่อความยั่งยืน"),
    "rmf": ("RMF", "เพื่อการเลี้ยงชีพ", "RETIREMENT MUTUAL FUND"),
    "ssf": ("SSF", "เพื่อการออม", "SUPER SAVINGS"),
}


class DeductionRecordResult(BaseModel):
    """Outcome of recording one deduction, with the year's running total.

    Example:
        >>> DeductionRecordResult(status="recorded", deduction_type="ssf",
        ...     amount=Decimal("1"), tax_year=2026, year_total=Decimal("1")).status
        'recorded'
    """

    status: str  # "recorded" or "duplicate"
    deduction_type: str
    amount: Decimal
    tax_year: int
    year_total: Decimal
    maximum_allowed: Optional[Decimal] = None
    remaining_room: Optional[Decimal] = None
    exceeds_limit: bool = False
    percentage_of_income_cap: Optional[Decimal] = None


def validate_claimable_deduction_type(deduction_type: str) -> str:
    """Normalize a deduction type and check the user can claim it by record.

    Args:
        deduction_type: Type key such as "life_insurance".

    Returns:
        Lower-case type key.

    Raises:
        ValueError: If the type is unknown or is an automatic allowance.

    Example:
        >>> validate_claimable_deduction_type("RMF")
        'rmf'
    """
    normalized = deduction_type.strip().lower()
    if normalized in CLAIMABLE_DEDUCTION_TYPES:
        return normalized
    accepted = ", ".join(sorted(CLAIMABLE_DEDUCTION_TYPES))
    raise ValueError(
        f"Deduction type '{normalized}' cannot be recorded. Accepted: {accepted}"
        " (personal/spouse/child/parent allowances come from the user profile)."
    )


def validate_deduction_still_offered(deduction_type: str, tax_year: int) -> None:
    """Reject deductions the law no longer gives for that tax year (SSF after 2024).

    Raises:
        ValueError: If the deduction has ended for the year.

    Example:
        >>> validate_deduction_still_offered("ssf", 2024)
    """
    if deduction_type == "ssf" and tax_year > SSF_FINAL_TAX_YEAR:
        raise ValueError(
            f"SSF purchases are deductible only up to tax year {SSF_FINAL_TAX_YEAR}"
            f" (พ.ศ. {SSF_FINAL_TAX_YEAR + 543}). Received tax year: {tax_year}"
        )


def record_deduction(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    session: Session,
    user_id: str,
    deduction_type: str,
    amount: Decimal,
    tax_year: int,
    description: str = "",
) -> DeductionRecordResult:
    """Save a deduction unless the same type and amount was already saved today.

    Args:
        session: Database session (committed here).
        user_id: UUID of the user.
        deduction_type: Claimable type key.
        amount: Amount paid in THB.
        tax_year: Tax year the payment counts for.
        description: Optional note (e.g. "ประกันชีวิต AIA").

    Returns:
        Result with status and the year's total for that type.

    Raises:
        ValueError: If the type is not claimable or the amount is not positive.

    Example:
        >>> record_deduction(session, uid, "life_insurance", Decimal("20000"), 2026).status
        'recorded'
    """
    normalized = validate_claimable_deduction_type(deduction_type)
    if amount <= 0:
        raise ValueError(f"Deduction amount must be greater than 0. Received: {amount} THB")
    validate_deduction_still_offered(normalized, tax_year)
    existing = DeductionCRUD().get_by_type(session, user_id, tax_year, normalized)
    if any(_is_same_payment(row, amount) for row in existing):
        return _build_result("duplicate", normalized, amount, tax_year, existing)
    created = DeductionCRUD().create(
        session,
        user_id=user_id,
        deduction_type=normalized,
        description=description or None,
        amount=amount,
        maximum_allowed=DEDUCTION_LIMITS.get(normalized, amount),
        tax_year=tax_year,
    )
    session.commit()
    return _build_result("recorded", normalized, amount, tax_year, [*existing, created])


def _is_same_payment(row: Deduction, amount: Decimal) -> bool:
    """Same amount saved today: a repeated message, not next month's premium."""
    created = row.created_at
    if created.tzinfo is None:  # SQLite drops the zone; values are written in UTC
        created = created.replace(tzinfo=timezone.utc)
    return row.amount == amount and created.astimezone(timezone.utc).date() == _today_utc()


def _today_utc() -> date:
    """Today's date in UTC (created_at is stored in UTC)."""
    return datetime.now(timezone.utc).date()


def _build_result(
    status: str, deduction_type: str, amount: Decimal, tax_year: int, rows: list[Deduction]
) -> DeductionRecordResult:
    """Totals and remaining room under the fixed THB limit (None for donations)."""
    year_total = sum((row.amount for row in rows), Decimal("0"))
    limit = DEDUCTION_LIMITS.get(deduction_type)
    return DeductionRecordResult(
        status=status,
        deduction_type=deduction_type,
        amount=amount,
        tax_year=tax_year,
        year_total=year_total,
        maximum_allowed=limit,
        remaining_room=None if limit is None else max(limit - year_total, Decimal("0")),
        exceeds_limit=limit is not None and year_total > limit,
        percentage_of_income_cap=PERCENTAGE_CAPPED_DEDUCTIONS.get(deduction_type),
    )


def remove_latest_deduction(
    session: Session, user_id: str, deduction_type: str, tax_year: int
) -> Optional[Decimal]:
    """Delete the most recently created deduction of a type (undo).

    Args:
        session: Database session (committed here).
        user_id: UUID of the user.
        deduction_type: Claimable type key.
        tax_year: Tax year.

    Returns:
        Amount removed, or None when nothing of that type was recorded.

    Example:
        >>> remove_latest_deduction(session, uid, "ssf", 2026)
        Decimal('30000.00')
    """
    normalized = validate_claimable_deduction_type(deduction_type)
    rows = DeductionCRUD().get_by_type(session, user_id, tax_year, normalized)
    if not rows:
        return None
    latest = max(rows, key=lambda row: row.created_at)
    DeductionCRUD().delete(session, latest.id)
    session.commit()
    return latest.amount


def has_deduction_for_year(
    session: Session, user_id: str, deduction_type: str, tax_year: int
) -> bool:
    """Whether any deduction of the type is recorded for the year.

    Example:
        >>> has_deduction_for_year(session, uid, "social_security", 2026)
        False
    """
    return bool(DeductionCRUD().get_by_type(session, user_id, tax_year, deduction_type))


def calculate_yearly_social_security(monthly_salary: Decimal) -> Decimal:
    """Employee social security for a year: 5% of wage up to 17,500, x12.

    Args:
        monthly_salary: Monthly salary in THB.

    Returns:
        Yearly contribution (2 decimals).

    Raises:
        ValueError: If the salary is not positive.

    Example:
        >>> calculate_yearly_social_security(Decimal("50000"))
        Decimal('10500.00')
    """
    if monthly_salary <= 0:
        raise ValueError(f"Monthly salary must be greater than 0. Received: {monthly_salary} THB")
    insured_wage = min(monthly_salary, SOCIAL_SECURITY_WAGE_CEILING)
    return (insured_wage * SOCIAL_SECURITY_RATE * 12).quantize(Decimal("0.01"))


def detect_tax_saving_fund_type(symbol: str, name: str = "") -> Optional[str]:
    """Recognize an RMF or SSF from its fund code or name.

    Args:
        symbol: Fund code (e.g. "K-USARMF").
        name: Fund name, Thai or English (e.g. "... เพื่อการเลี้ยงชีพ").

    Returns:
        "rmf", "ssf", or None for any other asset.

    Example:
        >>> detect_tax_saving_fund_type("K-CHANGESSF")
        'ssf'
    """
    text = f"{symbol} {name}".upper()
    for fund_type, markers in TAX_SAVING_FUND_MARKERS.items():
        if any(marker.upper() in text for marker in markers):
            return fund_type
    return None
