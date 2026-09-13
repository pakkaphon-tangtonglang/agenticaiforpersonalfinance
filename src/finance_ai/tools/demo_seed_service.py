"""Demo seed service: wipe-and-reseed a fixed demo user for the defense demo.

Seeds a Thai-language demo profile with ~3 months of expenses, monthly
salary income, RMF/SSF tax deductions (to showcase tax optimization),
spending patterns that trigger recommendations, financial goals, a stock
watchlist, and investment holdings. Safe to run repeatedly: the demo user
and all related records are deleted before re-seeding.

Example:
    uv run python scripts/seed_demo.py
"""

from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from finance_ai.database.base import Base
from finance_ai.database.models.deduction import Deduction
from finance_ai.database.models.financial_goal import FinancialGoal
from finance_ai.database.models.income import Income
from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.database.models.transaction import Transaction
from finance_ai.database.models.user import User
from finance_ai.database.models.watched_asset import WatchedAsset
from finance_ai.database.session import create_database_engine, create_session_factory
from finance_ai.tools.tax_constants import DEDUCTION_LIMITS

DEMO_USER_ID = "00000000-de20-4000-8000-000000000001"
DEMO_USER_EMAIL = "demo@finance-ai.local"
_SALARY_AMOUNT = Decimal("50000.00")
_SALARY_WITHHOLDING = Decimal("2500.00")

# (deduction_type, amount, description) — all within DEDUCTION_LIMITS
DEDUCTION_PLAN: list[tuple[str, Decimal, str]] = [
    ("social_security", Decimal("9000.00"), "ประกันสังคม"),
    ("life_insurance", Decimal("50000.00"), "เบี้ยประกันชีวิต"),
    ("rmf", Decimal("60000.00"), "ลงทุนกองทุนรวม RMF"),
    ("ssf", Decimal("40000.00"), "ลงทุนกองทุนรวม SSF"),
]

# Recurring monthly expenses: (category, day_of_month, amount, description)
_MONTHLY_EXPENSE_TEMPLATE: list[tuple[str, int, Decimal, str]] = [
    ("housing", 1, Decimal("9000.00"), "ค่าห้องเช่า"),
    ("transport", 1, Decimal("60.00"), "ค่ารถไฟฟ้า BTS"),
    ("food", 3, Decimal("70.00"), "ข้าวมันไก่"),
    ("utilities", 5, Decimal("1250.00"), "ค่าไฟฟ้าและค่าน้ำ"),
    ("transport", 8, Decimal("150.00"), "ค่าแท็กซี่"),
    ("food", 10, Decimal("60.00"), "กาแฟและขนม"),
    ("shopping", 12, Decimal("850.00"), "ช้อปปิ้งที่เซ็นทรัล"),
    ("health", 14, Decimal("300.00"), "ค่าตรวจสุขภาพคลินิก"),
    ("transport", 15, Decimal("500.00"), "ค่าน้ำมันรถ"),
    ("food", 17, Decimal("120.00"), "บุฟเฟ่ต์กลางวัน"),
    ("entertainment", 20, Decimal("240.00"), "ตั๋วหนัง"),
    ("transport", 22, Decimal("80.00"), "ค่าจอดรถ"),
    ("food", 24, Decimal("250.00"), "ซุปเปอร์มาร์เก็ตสัปดาห์ท้าย"),
]

_WATCHLIST_PLAN: list[tuple[str, str]] = [
    ("PTT.BK", "ปตท."),
    ("KBANK.BK", "ธ.กสิกรไทย"),
    ("AOT.BK", "ท่าอากาศยานไทย"),
]


def seed_demo_database() -> str:
    """
    Seed the application database with demo data and return a summary.

    Returns:
        str: Human-readable confirmation with the demo user email.
    """
    engine = create_database_engine()
    Base.metadata.create_all(bind=engine)
    session = create_session_factory(engine)()
    try:
        user = seed_demo_data(session)
        return f"Seeded demo user {user.full_name} <{DEMO_USER_EMAIL}> (id={user.id})"
    finally:
        session.close()


def seed_demo_data(session: Session, as_of: date | None = None) -> User:
    """
    Wipe any existing demo user and seed fresh demo data.

    Args:
        session: Database session used for all writes.
        as_of: Reference date for the 3-month expense window
            (defaults to today, so the demo always looks current).

    Returns:
        User: The freshly seeded demo user.

    Example:
        >>> user = seed_demo_data(session)
    """
    effective_date = as_of if as_of is not None else date.today()
    wipe_demo_data(session)
    user = _create_demo_user(session)
    _create_incomes(session, effective_date)
    _create_deductions(session, effective_date)
    _create_expense_transactions(session, effective_date)
    _create_goals(session, effective_date)
    _create_watchlist(session)
    _create_holdings(session, effective_date)
    session.commit()
    return user


def wipe_demo_data(session: Session) -> None:
    """
    Delete the demo user and all related records (cascaded).

    Args:
        session: Database session used for the deletion.

    Example:
        >>> wipe_demo_data(session)
    """
    demo_users = session.scalars(
        select(User).where(or_(User.id == DEMO_USER_ID, User.email == DEMO_USER_EMAIL))
    ).all()
    for user in demo_users:
        session.delete(user)
    session.commit()


def _create_demo_user(session: Session) -> User:
    """Create and persist the fixed demo user profile."""
    user = User(
        id=DEMO_USER_ID,
        email=DEMO_USER_EMAIL,
        hashed_password="not-a-real-login",
        full_name="สมชาย ใจดี",
        tax_id="1100700123456",
        date_of_birth=date(1992, 5, 20),
        marital_status="single",
        number_of_children=0,
        number_of_parents=2,
    )
    session.add(user)
    session.flush()
    return user


def _create_incomes(session: Session, as_of: date) -> None:
    """Create monthly salary records for the last 3 months."""
    for month_start in _recent_month_starts(as_of, 3):
        session.add(
            Income(
                user_id=DEMO_USER_ID,
                income_type="salary",
                description=f"เงินเดือน {month_start:%Y-%m}",
                amount=_SALARY_AMOUNT,
                tax_year=month_start.year,
                pay_period="monthly",
                employer_name="บริษัท ไทยซอฟต์แวร์ จำกัด",
                withholding_tax=_SALARY_WITHHOLDING,
            )
        )


def _create_deductions(session: Session, as_of: date) -> None:
    """Create RMF/SSF and other deductions for the current tax year."""
    for deduction_type, amount, description in DEDUCTION_PLAN:
        session.add(
            Deduction(
                user_id=DEMO_USER_ID,
                deduction_type=deduction_type,
                description=description,
                amount=amount,
                maximum_allowed=DEDUCTION_LIMITS[deduction_type],
                tax_year=as_of.year,
            )
        )


def _create_expense_transactions(session: Session, as_of: date) -> None:
    """Create ~3 months of deterministic expense transactions."""
    for transaction in _build_expense_transactions(as_of):
        session.add(transaction)


def _build_expense_transactions(as_of: date) -> list[Transaction]:
    """Build expense transactions for each of the last 3 months."""
    transactions: list[Transaction] = []
    for month_start in _recent_month_starts(as_of, 3):
        transactions.extend(_build_month_expenses(month_start, as_of))
    return transactions


def _build_month_expenses(month_start: date, as_of: date) -> list[Transaction]:
    """Build one month of expenses, skipping days after the reference date."""
    transactions: list[Transaction] = []
    for category, day_of_month, amount, description in _MONTHLY_EXPENSE_TEMPLATE:
        transaction_date = date(month_start.year, month_start.month, day_of_month)
        if transaction_date > as_of:
            continue
        transactions.append(
            Transaction(
                user_id=DEMO_USER_ID,
                transaction_type="expense",
                category=category,
                description=description,
                amount=amount,
                transaction_date=transaction_date,
            )
        )
    return transactions


def _create_goals(session: Session, as_of: date) -> None:
    """Create two financial goals shown in the planning demo."""
    session.add_all(
        [
            FinancialGoal(
                user_id=DEMO_USER_ID,
                goal_type="emergency_fund",
                name="กองทุนสำรองฉุกเฉิน",
                target_amount=Decimal("180000.00"),
                current_amount=Decimal("96000.00"),
                target_date=_shift_months(as_of, 12),
                priority=1,
            ),
            FinancialGoal(
                user_id=DEMO_USER_ID,
                goal_type="travel",
                name="ท่องเที่ยวญี่ปุ่น",
                target_amount=Decimal("120000.00"),
                current_amount=Decimal("35000.00"),
                target_date=_shift_months(as_of, 18),
                priority=2,
            ),
        ]
    )


def _create_watchlist(session: Session) -> None:
    """Create the stock watchlist entries (SYMBOL.BK format)."""
    for symbol, name in _WATCHLIST_PLAN:
        session.add(WatchedAsset(user_id=DEMO_USER_ID, symbol=symbol, name=name))


def _create_holdings(session: Session, as_of: date) -> None:
    """Create one mutual-fund and one stock holding with current prices."""
    session.add_all(
        [
            InvestmentHolding(
                user_id=DEMO_USER_ID,
                asset_type="mutual_fund",
                symbol="KT-RMFX",
                name="กองทุน RMF หุ้นไทย",
                quantity=Decimal("12000.0000"),
                average_cost_per_unit=Decimal("12.5000"),
                total_cost=Decimal("150000.00"),
                current_price_per_unit=Decimal("14.2000"),
                current_value=Decimal("170400.00"),
                unrealized_gain_loss=Decimal("20400.00"),
                purchase_date=_shift_months(as_of, -24),
                last_price_update=datetime.now(timezone.utc),
            ),
            InvestmentHolding(
                user_id=DEMO_USER_ID,
                asset_type="stock",
                symbol="PTT.BK",
                name="ปตท.",
                quantity=Decimal("1000.0000"),
                average_cost_per_unit=Decimal("55.0000"),
                total_cost=Decimal("55000.00"),
                current_price_per_unit=Decimal("58.0000"),
                current_value=Decimal("58000.00"),
                unrealized_gain_loss=Decimal("3000.00"),
                purchase_date=_shift_months(as_of, -12),
                last_price_update=datetime.now(timezone.utc),
            ),
        ]
    )


def _recent_month_starts(as_of: date, months: int) -> list[date]:
    """Return the first day of each of the last `months` months."""
    return [_shift_months(as_of, -offset).replace(day=1) for offset in range(months - 1, -1, -1)]


def _shift_months(anchor: date, offset: int) -> date:
    """Shift a date by whole months, clamped to the 1st of the result month."""
    total_months = anchor.year * 12 + (anchor.month - 1) + offset
    return date(total_months // 12, total_months % 12 + 1, 1)
