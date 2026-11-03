"""Tests for per-run databases, database checks and the market snapshot."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import select

from finance_ai.database.models.financial_goal import FinancialGoal
from finance_ai.database.models.transaction import Transaction
from finance_ai.database.models.watched_asset import WatchedAsset
from finance_ai.evaluation.architectures.environment import (
    build_template_database,
    check_database,
    fresh_database,
    open_template,
)
from finance_ai.evaluation.architectures.market_snapshot import (
    PRICES,
    SnapshotTicker,
    market_snapshot,
    snapshot_fund_search,
    snapshot_news_rss,
    snapshot_symbol_search,
)
from finance_ai.evaluation.architectures.task_models import DbCheck
from finance_ai.tools.demo_seed_service import DEMO_USER_ID


@pytest.fixture(name="template")
def fixture_template(tmp_path: Path) -> Path:
    """Build a seeded template database in a temp folder."""
    return build_template_database(tmp_path, date(2026, 11, 2))


def _add_expense(session_factory: object, amount: str) -> None:
    """Write one expense row for the demo user."""
    with session_factory() as session:  # type: ignore[operator]
        session.add(
            Transaction(
                user_id=DEMO_USER_ID,
                transaction_type="expense",
                category="food",
                amount=Decimal(amount),
                description="test",
                transaction_date=date(2026, 11, 2),
            )
        )
        session.commit()


class TestFreshDatabase:
    """Each run works on its own copy."""

    def test_copy_starts_with_demo_data(self, template: Path) -> None:
        """The copy contains the seeded watchlist."""
        with fresh_database(template, "a") as factory, factory() as session:
            symbols = set(session.scalars(select(WatchedAsset.symbol)))
        assert {"PTT.BK", "KBANK.BK", "AOT.BK"} <= symbols

    def test_writes_do_not_reach_template(self, template: Path) -> None:
        """Rows written in a copy are absent from the next copy."""
        with fresh_database(template, "a") as factory:
            _add_expense(factory, "123.45")
        with fresh_database(template, "b") as factory, factory() as session:
            amounts = set(session.scalars(select(Transaction.amount)))
        assert Decimal("123.45") not in amounts

    def test_copy_file_removed(self, template: Path) -> None:
        """The copy is deleted when the block ends."""
        with fresh_database(template, "gone"):
            pass
        assert not (template.parent / "run_gone.db").exists()


class TestCheckDatabase:
    """Database checks compare the copy with the template."""

    def test_new_expense_detected(self, template: Path) -> None:
        """An expense amount that already exists still needs a new row."""
        checks = [DbCheck(kind="expense", amount=Decimal("80"))]
        with open_template(template) as template_factory, fresh_database(template, "x") as run:
            assert check_database(run, template_factory, checks) == [False]
            _add_expense(run, "80")
            assert check_database(run, template_factory, checks) == [True]

    def test_goal_current_amount(self, template: Path) -> None:
        """goal_current compares the emergency fund balance."""
        checks = [DbCheck(kind="goal_current", amount=Decimal("106000"))]
        with open_template(template) as template_factory, fresh_database(template, "g") as run:
            assert check_database(run, template_factory, checks) == [False]
            with run() as session:
                goal = session.scalars(
                    select(FinancialGoal).where(FinancialGoal.goal_type == "emergency_fund")
                ).one()
                goal.current_amount = Decimal("106000")
                session.commit()
            assert check_database(run, template_factory, checks) == [True]

    @pytest.mark.parametrize(
        "check",
        [
            DbCheck(kind="income", amount=Decimal("30000")),
            DbCheck(kind="goal_target", amount=Decimal("500000")),
            DbCheck(kind="watchlist", symbol="CPALL.BK"),
            DbCheck(kind="holding", symbol="ADVANC.BK"),
        ],
    )
    def test_untouched_copy_fails_row_checks(self, template: Path, check: DbCheck) -> None:
        """Without writes, every row check fails."""
        with open_template(template) as template_factory, fresh_database(template, "r") as run:
            assert check_database(run, template_factory, [check]) == [False]


class TestMarketSnapshot:
    """Canned market data."""

    def test_ticker_price_and_history(self) -> None:
        """Known symbols return the snapshot price everywhere."""
        ticker = SnapshotTicker("ptt.bk")
        assert ticker.fast_info.last_price == float(PRICES["PTT.BK"])
        assert ticker.history(period="5d")["Close"].iloc[-1] == float(PRICES["PTT.BK"])
        assert ticker.info["currency"] == "THB"

    def test_unknown_ticker_is_empty(self) -> None:
        """Unknown symbols return nothing."""
        ticker = SnapshotTicker("NOPE")
        assert ticker.fast_info.last_price is None and not ticker.info
        assert ticker.history().empty

    def test_news_feed(self) -> None:
        """Known symbols get one allowlisted item; others get none."""
        assert "ผู้โดยสาร" in snapshot_news_rss("AOT.BK")
        assert "<item>" not in snapshot_news_rss("NOPE")

    def test_symbol_search_thai_and_english(self) -> None:
        """Thai and English names resolve to the same symbol."""
        assert snapshot_symbol_search("ปตท")["quotes"][0]["symbol"] == "PTT.BK"
        assert snapshot_symbol_search("ptt")["quotes"][0]["symbol"] == "PTT.BK"
        assert snapshot_symbol_search("unknown") == {"quotes": []}

    def test_fund_search(self) -> None:
        """SET50 queries return the two index funds."""
        assert len(snapshot_fund_search("กองทุน SET50").results) == 2
        assert not snapshot_fund_search("ทองคำ").results

    def test_patch_active_only_inside_block(self) -> None:
        """yfinance.Ticker is replaced only inside the block."""
        import yfinance  # pylint: disable=import-outside-toplevel

        with market_snapshot():
            assert yfinance.Ticker is SnapshotTicker
        assert yfinance.Ticker is not SnapshotTicker
