"""Tests for demo seed service."""

from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from finance_ai.database.models.deduction import Deduction
from finance_ai.database.models.financial_goal import FinancialGoal
from finance_ai.database.models.income import Income
from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.database.models.transaction import Transaction
from finance_ai.database.models.user import User
from finance_ai.database.models.watched_asset import WatchedAsset
from finance_ai.tools.demo_seed_service import (
    DEDUCTION_PLAN,
    DEMO_USER_EMAIL,
    DEMO_USER_ID,
    seed_demo_data,
    wipe_demo_data,
)
from finance_ai.tools.expense_constants import VALID_EXPENSE_CATEGORIES
from finance_ai.tools.tax_constants import DEDUCTION_LIMITS

AS_OF = date(2026, 2, 28)


class TestSeedDemoData:
    """Tests for seed_demo_data."""

    def test_creates_demo_user_with_fixed_id(self, test_session: Session) -> None:
        """Demo user is created with the fixed id and Thai profile."""
        user = seed_demo_data(test_session, as_of=AS_OF)
        assert user.id == DEMO_USER_ID
        assert user.email == DEMO_USER_EMAIL
        assert user.full_name == "สมชาย ใจดี"

    def test_returns_persisted_user(self, test_session: Session) -> None:
        """Returned user is persisted and refreshable from the database."""
        user = seed_demo_data(test_session, as_of=AS_OF)
        test_session.expire(user)
        reloaded = test_session.get(User, DEMO_USER_ID)
        assert reloaded is not None
        assert reloaded.email == DEMO_USER_EMAIL

    def test_creates_three_months_of_expenses(self, test_session: Session) -> None:
        """At least 3 months of expense transactions are seeded."""
        seed_demo_data(test_session, as_of=AS_OF)
        expense_count = len(
            test_session.scalars(
                select(Transaction.id).where(
                    Transaction.user_id == DEMO_USER_ID,
                    Transaction.transaction_type == "expense",
                )
            ).all()
        )
        assert expense_count >= 36
        expense_dates = test_session.scalars(
            select(Transaction.transaction_date).where(
                Transaction.user_id == DEMO_USER_ID,
                Transaction.transaction_type == "expense",
            )
        ).all()
        assert min(expense_dates) >= AS_OF - timedelta(days=100)
        assert max(expense_dates) <= AS_OF

    def test_expense_categories_are_valid(self, test_session: Session) -> None:
        """Every seeded expense uses a valid category key."""
        seed_demo_data(test_session, as_of=AS_OF)
        categories = test_session.scalars(
            select(Transaction.category).where(
                Transaction.user_id == DEMO_USER_ID,
                Transaction.transaction_type == "expense",
            )
        ).all()
        assert categories
        assert set(categories).issubset(set(VALID_EXPENSE_CATEGORIES))

    def test_creates_monthly_income_records(self, test_session: Session) -> None:
        """Three monthly salary records with withholding tax are seeded."""
        seed_demo_data(test_session, as_of=AS_OF)
        incomes = test_session.scalars(select(Income).where(Income.user_id == DEMO_USER_ID)).all()
        assert len(incomes) == 3
        assert all(income.income_type == "salary" for income in incomes)
        assert all(
            income.amount == Decimal("50000.00") and income.withholding_tax == Decimal("2500.00")
            for income in incomes
        )

    def test_creates_rmf_and_ssf_deductions(self, test_session: Session) -> None:
        """Deductions include RMF and SSF for tax optimization demo."""
        seed_demo_data(test_session, as_of=AS_OF)
        deductions = test_session.scalars(
            select(Deduction).where(Deduction.user_id == DEMO_USER_ID)
        ).all()
        types = {deduction.deduction_type for deduction in deductions}
        assert {"rmf", "ssf", "social_security", "life_insurance"}.issubset(types)
        assert all(d.tax_year == AS_OF.year for d in deductions)

    def test_deduction_amounts_respect_statutory_limits(self, test_session: Session) -> None:
        """Seeded amounts never exceed the tax law limits."""
        seed_demo_data(test_session, as_of=AS_OF)
        deductions = test_session.scalars(
            select(Deduction).where(Deduction.user_id == DEMO_USER_ID)
        ).all()
        planned = {d_type: amount for d_type, amount, _ in DEDUCTION_PLAN}
        for deduction in deductions:
            assert deduction.amount == planned[deduction.deduction_type]
            assert deduction.amount <= DEDUCTION_LIMITS[deduction.deduction_type]
            assert deduction.maximum_allowed == DEDUCTION_LIMITS[deduction.deduction_type]

    def test_creates_goals_watchlist_and_holdings(self, test_session: Session) -> None:
        """Financial goals, watchlist entries, and holdings are seeded."""
        seed_demo_data(test_session, as_of=AS_OF)
        goals = test_session.scalars(
            select(FinancialGoal).where(FinancialGoal.user_id == DEMO_USER_ID)
        ).all()
        watched = test_session.scalars(
            select(WatchedAsset).where(WatchedAsset.user_id == DEMO_USER_ID)
        ).all()
        holdings = test_session.scalars(
            select(InvestmentHolding).where(InvestmentHolding.user_id == DEMO_USER_ID)
        ).all()
        assert len(goals) == 2
        assert len(watched) == 3
        assert all(watched_asset.symbol.endswith(".BK") for watched_asset in watched)
        assert len(holdings) == 2

    def test_seeding_twice_is_idempotent(self, test_session: Session) -> None:
        """Re-seeding wipes and recreates without duplicating rows."""
        seed_demo_data(test_session, as_of=AS_OF)
        seed_demo_data(test_session, as_of=AS_OF)
        user_ids = test_session.scalars(
            select(User.id).where(or_(User.id == DEMO_USER_ID, User.email == DEMO_USER_EMAIL))
        ).all()
        transaction_ids = test_session.scalars(
            select(Transaction.id).where(Transaction.user_id == DEMO_USER_ID)
        ).all()
        assert len(user_ids) == 1
        assert len(transaction_ids) >= 36

    def test_reseed_preserves_other_users(self, test_session: Session) -> None:
        """Other users are untouched by the demo wipe-and-reseed."""
        other = User(
            email="other@example.com",
            hashed_password="hashed",
            full_name="Other User",
        )
        test_session.add(other)
        test_session.commit()
        seed_demo_data(test_session, as_of=AS_OF)
        remaining = test_session.scalars(
            select(User.id).where(User.email == "other@example.com")
        ).all()
        assert len(remaining) == 1


class TestWipeDemoData:
    """Tests for wipe_demo_data."""

    def test_removes_demo_user_and_children(self, test_session: Session) -> None:
        """Wiping removes the demo user and cascades to all records."""
        seed_demo_data(test_session, as_of=AS_OF)
        wipe_demo_data(test_session)
        user_ids = test_session.scalars(
            select(User.id).where(or_(User.id == DEMO_USER_ID, User.email == DEMO_USER_EMAIL))
        ).all()
        transaction_ids = test_session.scalars(
            select(Transaction.id).where(Transaction.user_id == DEMO_USER_ID)
        ).all()
        assert len(user_ids) == 0
        assert len(transaction_ids) == 0

    def test_wiping_empty_database_is_safe(self, test_session: Session) -> None:
        """Wiping a database without the demo user raises no error."""
        wipe_demo_data(test_session)
        assert len(test_session.scalars(select(User.id)).all()) == 0
