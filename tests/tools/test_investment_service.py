"""Tests for investment service layer."""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.database.models.user import User
from finance_ai.tools.investment_calculator import HoldingRecord, PortfolioSummaryResult
from finance_ai.tools.investment_service import (
    add_investment_holding,
    convert_holding_to_record,
    get_portfolio_summary,
    get_user_holding_by_symbol,
    get_user_holdings,
    get_user_holdings_by_asset_type,
    import_holdings_from_csv,
    refresh_portfolio_prices,
)


@pytest.fixture
def sample_holding(
    test_session: Session,
    sample_user: User,
) -> InvestmentHolding:
    """Create a sample stock holding for testing."""
    holding = InvestmentHolding(
        user_id=sample_user.id,
        asset_type="stock",
        symbol="PTT.BK",
        name="PTT",
        quantity=Decimal("100"),
        average_cost_per_unit=Decimal("35.50"),
        total_cost=Decimal("3550.00"),
        purchase_date=date(2025, 1, 15),
    )
    test_session.add(holding)
    test_session.commit()
    test_session.refresh(holding)
    return holding


@pytest.fixture
def sample_fund_holding(
    test_session: Session,
    sample_user: User,
) -> InvestmentHolding:
    """Create a sample mutual fund holding for testing."""
    holding = InvestmentHolding(
        user_id=sample_user.id,
        asset_type="mutual_fund",
        symbol="K-EQUITY",
        name="K Equity Fund",
        quantity=Decimal("500"),
        average_cost_per_unit=Decimal("14.50"),
        total_cost=Decimal("7250.00"),
        purchase_date=date(2025, 3, 1),
    )
    test_session.add(holding)
    test_session.commit()
    test_session.refresh(holding)
    return holding


# ── Convert Holding to Record ────────────────────────────────────────


class TestConvertHoldingToRecord:
    """Tests for convert_holding_to_record."""

    def test_converts_all_fields(self, sample_holding: InvestmentHolding) -> None:
        """Converts all fields from ORM to Pydantic model."""
        record = convert_holding_to_record(sample_holding)
        assert isinstance(record, HoldingRecord)
        assert record.symbol == "PTT.BK"
        assert record.name == "PTT"
        assert record.asset_type == "stock"
        assert record.quantity == Decimal("100")
        assert record.average_cost_per_unit == Decimal("35.50")
        assert record.total_cost == Decimal("3550.00")

    def test_handles_null_optional_fields(self, sample_holding: InvestmentHolding) -> None:
        """Handles None for optional price fields."""
        record = convert_holding_to_record(sample_holding)
        assert record.current_price_per_unit is None
        assert record.current_value is None
        assert record.unrealized_gain_loss is None

    def test_handles_null_name(self, test_session: Session, sample_user: User) -> None:
        """Converts None name to empty string."""
        holding = InvestmentHolding(
            user_id=sample_user.id,
            asset_type="stock",
            symbol="AOT.BK",
            name=None,
            quantity=Decimal("50"),
            average_cost_per_unit=Decimal("60.00"),
            total_cost=Decimal("3000.00"),
            purchase_date=date(2025, 1, 1),
        )
        test_session.add(holding)
        test_session.commit()
        record = convert_holding_to_record(holding)
        assert record.name == ""


# ── Get User Holdings ────────────────────────────────────────────────


class TestGetUserHoldings:
    """Tests for get_user_holdings."""

    def test_returns_holdings(
        self,
        test_session: Session,
        sample_user: User,
        sample_holding: InvestmentHolding,
    ) -> None:
        """Returns list of HoldingRecord for user."""
        result = get_user_holdings(test_session, sample_user.id)
        assert len(result) == 1
        assert result[0].symbol == "PTT.BK"

    def test_empty_for_no_holdings(self, test_session: Session, sample_user: User) -> None:
        """Returns empty list when user has no holdings."""
        result = get_user_holdings(test_session, sample_user.id)
        assert result == []


# ── Get User Holdings By Asset Type ──────────────────────────────────


class TestGetUserHoldingsByAssetType:
    """Tests for get_user_holdings_by_asset_type."""

    def test_filters_by_stock(
        self,
        test_session: Session,
        sample_user: User,
        sample_holding: InvestmentHolding,
        sample_fund_holding: InvestmentHolding,
    ) -> None:
        """Filters to stock holdings only."""
        result = get_user_holdings_by_asset_type(test_session, sample_user.id, "stock")
        assert len(result) == 1
        assert result[0].symbol == "PTT.BK"

    def test_invalid_asset_type_raises(self, test_session: Session, sample_user: User) -> None:
        """Raises ValueError for unknown asset type."""
        with pytest.raises(ValueError, match="Unknown asset type"):
            get_user_holdings_by_asset_type(test_session, sample_user.id, "crypto")


# ── Get User Holding By Symbol ───────────────────────────────────────


class TestGetUserHoldingBySymbol:
    """Tests for get_user_holding_by_symbol."""

    def test_found(
        self,
        test_session: Session,
        sample_user: User,
        sample_holding: InvestmentHolding,
    ) -> None:
        """Returns HoldingRecord when found."""
        result = get_user_holding_by_symbol(test_session, sample_user.id, "PTT.BK")
        assert result is not None
        assert result.symbol == "PTT.BK"

    def test_not_found(self, test_session: Session, sample_user: User) -> None:
        """Returns None when not found."""
        result = get_user_holding_by_symbol(test_session, sample_user.id, "KBANK.BK")
        assert result is None

    def test_normalizes_symbol(
        self,
        test_session: Session,
        sample_user: User,
        sample_holding: InvestmentHolding,
    ) -> None:
        """Normalizes symbol to uppercase."""
        result = get_user_holding_by_symbol(test_session, sample_user.id, "ptt.bk")
        assert result is not None


# ── Add Investment Holding ───────────────────────────────────────────


class TestAddInvestmentHolding:
    """Tests for add_investment_holding."""

    def test_creates_stock_holding(self, test_session: Session, sample_user: User) -> None:
        """Creates a stock holding with correct values."""
        holding = add_investment_holding(
            test_session,
            sample_user.id,
            asset_type="stock",
            symbol="AOT.BK",
            name="AOT",
            quantity=Decimal("50"),
            price_per_unit=Decimal("60.00"),
            purchase_date=date(2025, 2, 1),
        )
        assert holding.symbol == "AOT.BK"
        assert holding.total_cost == Decimal("3000.00")

    def test_creates_mutual_fund_holding(self, test_session: Session, sample_user: User) -> None:
        """Creates a mutual fund holding."""
        holding = add_investment_holding(
            test_session,
            sample_user.id,
            asset_type="mutual_fund",
            symbol="K-EQUITY",
            name="K Equity",
            quantity=Decimal("1000"),
            price_per_unit=Decimal("14.50"),
            purchase_date=date(2025, 3, 1),
        )
        assert holding.symbol == "K-EQUITY"
        assert holding.asset_type == "mutual_fund"

    def test_invalid_asset_type_raises(self, test_session: Session, sample_user: User) -> None:
        """Raises ValueError for invalid asset type."""
        with pytest.raises(ValueError, match="Unknown asset type"):
            add_investment_holding(
                test_session,
                sample_user.id,
                asset_type="crypto",
                symbol="BTC",
                name="Bitcoin",
                quantity=Decimal("1"),
                price_per_unit=Decimal("100000"),
                purchase_date=date(2025, 1, 1),
            )

    def test_invalid_symbol_raises(self, test_session: Session, sample_user: User) -> None:
        """Raises ValueError for invalid stock symbol."""
        with pytest.raises(ValueError, match="must end with"):
            add_investment_holding(
                test_session,
                sample_user.id,
                asset_type="stock",
                symbol="PTT",
                name="PTT",
                quantity=Decimal("100"),
                price_per_unit=Decimal("35.50"),
                purchase_date=date(2025, 1, 1),
            )

    def test_invalid_quantity_raises(self, test_session: Session, sample_user: User) -> None:
        """Raises ValueError for zero quantity."""
        with pytest.raises(ValueError, match="must be at least"):
            add_investment_holding(
                test_session,
                sample_user.id,
                asset_type="stock",
                symbol="PTT.BK",
                name="PTT",
                quantity=Decimal("0"),
                price_per_unit=Decimal("35.50"),
                purchase_date=date(2025, 1, 1),
            )


# ── Import Holdings from CSV ────────────────────────────────────────


class TestImportHoldingsFromCsv:
    """Tests for import_holdings_from_csv."""

    def test_imports_multiple_rows(self, test_session: Session, sample_user: User) -> None:
        """Creates holdings from CSV data."""
        csv_content = (
            "symbol,asset_type,name,quantity,price_per_unit,purchase_date\n"
            "AOT.BK,stock,AOT,50,60.00,2025-02-01\n"
            "K-EQUITY,mutual_fund,K Equity,1000,14.50,2025-03-01\n"
        )
        result = import_holdings_from_csv(test_session, sample_user.id, csv_content)
        assert len(result) == 2
        assert result[0].symbol == "AOT.BK"
        assert result[1].symbol == "K-EQUITY"

    def test_invalid_csv_raises(self, test_session: Session, sample_user: User) -> None:
        """Raises ValueError for invalid CSV."""
        with pytest.raises(ValueError):
            import_holdings_from_csv(test_session, sample_user.id, "")


# ── Refresh Portfolio Prices ─────────────────────────────────────────


class TestRefreshPortfolioPrices:
    """Tests for refresh_portfolio_prices."""

    @patch("finance_ai.tools.price_client.fetch_multiple_prices")
    def test_updates_prices(
        self,
        mock_fetch: MagicMock,
        test_session: Session,
        sample_user: User,
        sample_holding: InvestmentHolding,
    ) -> None:
        """Updates holding prices from yfinance."""
        mock_fetch.return_value = {"PTT.BK": Decimal("42.50")}
        count = refresh_portfolio_prices(test_session, sample_user.id)
        assert count == 1
        test_session.refresh(sample_holding)
        assert sample_holding.current_price_per_unit == Decimal("42.50")

    @patch("finance_ai.tools.price_client.fetch_multiple_prices")
    def test_skips_unavailable_prices(
        self,
        mock_fetch: MagicMock,
        test_session: Session,
        sample_user: User,
        sample_holding: InvestmentHolding,
    ) -> None:
        """Skips holdings with no price available."""
        mock_fetch.return_value = {"PTT.BK": None}
        count = refresh_portfolio_prices(test_session, sample_user.id)
        assert count == 0

    @patch("finance_ai.tools.price_client.fetch_multiple_prices")
    def test_no_holdings_returns_zero(
        self,
        mock_fetch: MagicMock,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Returns 0 when user has no holdings."""
        count = refresh_portfolio_prices(test_session, sample_user.id)
        assert count == 0
        mock_fetch.assert_not_called()


# ── Get Portfolio Summary ────────────────────────────────────────────


class TestGetPortfolioSummary:
    """Tests for get_portfolio_summary."""

    def test_with_holdings(
        self,
        test_session: Session,
        sample_user: User,
        sample_holding: InvestmentHolding,
        sample_fund_holding: InvestmentHolding,
    ) -> None:
        """Returns summary with multiple holdings."""
        result = get_portfolio_summary(test_session, sample_user.id)
        assert isinstance(result, PortfolioSummaryResult)
        assert result.holding_count == 2
        assert result.total_cost == Decimal("10800.00")

    def test_empty_portfolio(self, test_session: Session, sample_user: User) -> None:
        """Returns empty summary for no holdings."""
        result = get_portfolio_summary(test_session, sample_user.id)
        assert result.holding_count == 0
        assert result.total_cost == Decimal("0")
