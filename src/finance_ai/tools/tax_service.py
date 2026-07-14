"""Database-integrated tax calculation service.

Orchestrates tax calculations by querying user financial data from the
database and delegating computation to the pure tax calculator.
"""

from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.crud.deduction_crud import DeductionCRUD
from finance_ai.database.crud.income_crud import IncomeCRUD
from finance_ai.database.crud.tax_filing_crud import TaxFilingCRUD
from finance_ai.database.crud.user_crud import UserCRUD
from finance_ai.database.models.deduction import Deduction
from finance_ai.database.models.user import User
from finance_ai.tools.tax_calculator import TaxCalculationResult, calculate_tax


def build_auto_allowances(user: User) -> dict[str, Decimal]:
    """
    Build automatic allowances based on user profile.

    Args:
        user: User model with family information.

    Returns:
        Dict of auto-applied deduction types and amounts.

    Example:
        >>> allowances = build_auto_allowances(user)
        >>> allowances["personal_allowance"]
        Decimal('60000')
    """
    allowances: dict[str, Decimal] = {
        "personal_allowance": Decimal("60000"),
    }
    if user.marital_status == "married":
        allowances["spouse_allowance"] = Decimal("60000")
    if user.number_of_children > 0:
        allowances["child_allowance"] = Decimal("30000") * user.number_of_children
    if user.number_of_parents > 0:
        allowances["parent_allowance"] = Decimal("30000") * user.number_of_parents
    return allowances


def aggregate_deductions_by_type(
    deductions: list[Deduction],
) -> dict[str, Decimal]:
    """
    Aggregate deduction records into a dict by type.

    Args:
        deductions: List of Deduction model instances.

    Returns:
        Dict mapping deduction type to total claimed amount.

    Example:
        >>> aggregate_deductions_by_type([])
        {}
    """
    result: dict[str, Decimal] = {}
    for deduction in deductions:
        current = result.get(deduction.deduction_type, Decimal("0"))
        result[deduction.deduction_type] = current + deduction.amount
    return result


def calculate_total_withholding_tax(session: Session, user_id: str, tax_year: int) -> Decimal:
    """
    Sum all withholding tax from income records for the year.

    Args:
        session: Database session.
        user_id: UUID of the user.
        tax_year: Tax year to query.

    Returns:
        Total withholding tax paid as Decimal.

    Example:
        >>> calculate_total_withholding_tax(session, user_id, 2024)
        Decimal('60000.00')
    """
    income_crud = IncomeCRUD()
    incomes = income_crud.get_by_user_and_year(session, user_id, tax_year)
    return sum((income.withholding_tax for income in incomes), Decimal("0"))


def merge_deductions(
    auto_allowances: dict[str, Decimal],
    user_deductions: dict[str, Decimal],
) -> dict[str, Decimal]:
    """
    Merge auto-applied allowances with user-entered deductions.

    Auto allowances take precedence (user cannot override personal/family allowances).

    Args:
        auto_allowances: System-generated allowances.
        user_deductions: User-entered deductions from database.

    Returns:
        Merged deductions dict.

    Example:
        >>> merge_deductions({"personal_allowance": Decimal("60000")}, {"rmf": Decimal("200000")})
    """
    merged = dict(user_deductions)
    merged.update(auto_allowances)
    return merged


def store_tax_filing_result(
    session: Session,
    user_id: str,
    tax_year: int,
    result: TaxCalculationResult,
) -> None:
    """
    Save or update a tax filing record in the database.

    Args:
        session: Database session.
        user_id: UUID of the user.
        tax_year: Tax year of the filing.
        result: Calculated tax result to store.

    Example:
        >>> store_tax_filing_result(session, user_id, 2024, result)
    """
    import json

    tax_filing_crud = TaxFilingCRUD()
    existing = tax_filing_crud.get_by_user_and_year(session, user_id, tax_year)
    breakdown_json = json.dumps(
        [bracket.model_dump(mode="json") for bracket in result.tax_breakdown]
    )
    filing_data = {
        "gross_income": result.gross_income,
        "total_deductions": result.total_deductions,
        "net_income": result.net_income,
        "total_tax": result.total_tax,
        "effective_tax_rate": result.effective_tax_rate,
        "withholding_tax_paid": result.withholding_tax_paid,
        "tax_due_or_refund": result.tax_due_or_refund,
        "tax_breakdown_json": breakdown_json,
    }
    if existing:
        tax_filing_crud.update(session, existing.id, **filing_data)
    else:
        tax_filing_crud.create(session, user_id=user_id, tax_year=tax_year, **filing_data)
    session.commit()


def calculate_tax_for_user(session: Session, user_id: str, tax_year: int) -> TaxCalculationResult:
    """
    Calculate tax for a user by querying their financial data.

    Args:
        session: Database session.
        user_id: UUID of the user.
        tax_year: Tax year to calculate.

    Returns:
        TaxCalculationResult with complete tax breakdown.

    Raises:
        ValueError: If user is not found.

    Example:
        >>> result = calculate_tax_for_user(session, user_id, 2024)
        >>> result.total_tax
        Decimal('75000.00')
    """
    user = UserCRUD().get_by_id(session, user_id)
    if user is None:
        raise ValueError(f"User not found. User ID: {user_id}")
    gross_income = IncomeCRUD().get_total_income_for_year(session, user_id, tax_year)
    user_deductions = aggregate_deductions_by_type(
        DeductionCRUD().get_by_user_and_year(session, user_id, tax_year)
    )
    auto_allowances = build_auto_allowances(user)
    all_deductions = merge_deductions(auto_allowances, user_deductions)
    withholding = calculate_total_withholding_tax(session, user_id, tax_year)
    result = calculate_tax(gross_income, all_deductions, withholding)
    store_tax_filing_result(session, user_id, tax_year, result)
    return result
