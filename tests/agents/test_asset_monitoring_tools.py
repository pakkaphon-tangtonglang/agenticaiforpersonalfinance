"""Tests for asset monitoring agent tool wrappers."""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from finance_ai.agents.asset_monitoring_tools import (
    add_holding,
    create_asset_schedule,
    delete_asset_schedule,
    get_asset_notifications,
    get_investment_advice,
    import_csv,
    lookup_holding,
    refresh_prices,
    view_asset_schedules,
    view_portfolio,
)
from finance_ai.database.models.user import User


class TestViewPortfolio:
    """Tests for view_portfolio tool."""

    def test_returns_portfolio_data(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns dict with portfolio summary from database."""
        result = view_portfolio.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "view_portfolio"
        assert "total_value" in result
        assert "holding_count" in result

    def test_empty_portfolio(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns zero values for empty portfolio."""
        result = view_portfolio.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["holding_count"] == 0
        assert result["holdings"] == []


class TestAddHolding:
    """Tests for add_holding tool."""

    def test_valid_stock(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Parses, validates, and persists a stock holding."""
        result = add_holding.invoke(
            {
                "symbol": "PTT.BK",
                "asset_type": "stock",
                "name": "PTT",
                "quantity": "100",
                "price_per_unit": "35.50",
                "purchase_date": "2025-01-15",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "add_holding"
        assert result["symbol"] == "PTT.BK"
        assert result["asset_type"] == "stock"
        assert result["asset_type_label"] == "หุ้น"
        assert result["quantity"] == "100"
        assert result["price_per_unit"] == "35.50"
        assert result["purchase_date"] == "2025-01-15"

    def test_valid_mutual_fund(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Parses and validates a mutual fund holding."""
        result = add_holding.invoke(
            {
                "symbol": "K-EQUITY",
                "asset_type": "mutual_fund",
                "name": "K Equity Fund",
                "quantity": "500",
                "price_per_unit": "14.50",
                "purchase_date": "2025-03-01",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["symbol"] == "K-EQUITY"
        assert result["asset_type_label"] == "กองทุนรวม"

    def test_default_date_is_today(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Uses today's date when purchase_date is empty."""
        result = add_holding.invoke(
            {
                "symbol": "PTT.BK",
                "asset_type": "stock",
                "name": "PTT",
                "quantity": "100",
                "price_per_unit": "35.50",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["purchase_date"] == date.today().isoformat()

    def test_invalid_asset_type_raises(self) -> None:
        """Raises ValueError for unknown asset type."""
        with pytest.raises(Exception, match="Unknown asset type"):
            add_holding.invoke(
                {
                    "symbol": "BTC",
                    "asset_type": "crypto",
                    "name": "Bitcoin",
                    "quantity": "1",
                    "price_per_unit": "100000",
                }
            )

    def test_invalid_stock_symbol_raises(self) -> None:
        """Raises ValueError for stock without .BK suffix."""
        with pytest.raises(Exception, match="must end with"):
            add_holding.invoke(
                {
                    "symbol": "PTT",
                    "asset_type": "stock",
                    "name": "PTT",
                    "quantity": "100",
                    "price_per_unit": "35.50",
                }
            )

    def test_invalid_quantity_raises(self) -> None:
        """Raises ValueError for zero quantity."""
        with pytest.raises(Exception, match="must be at least"):
            add_holding.invoke(
                {
                    "symbol": "PTT.BK",
                    "asset_type": "stock",
                    "name": "PTT",
                    "quantity": "0",
                    "price_per_unit": "35.50",
                }
            )

    def test_invalid_price_raises(self) -> None:
        """Raises ValueError for zero price."""
        with pytest.raises(Exception, match="must be at least"):
            add_holding.invoke(
                {
                    "symbol": "PTT.BK",
                    "asset_type": "stock",
                    "name": "PTT",
                    "quantity": "100",
                    "price_per_unit": "0",
                }
            )

    def test_normalizes_symbol_case(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Normalizes symbol to uppercase."""
        result = add_holding.invoke(
            {
                "symbol": "ptt.bk",
                "asset_type": "stock",
                "name": "PTT",
                "quantity": "100",
                "price_per_unit": "35.50",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["symbol"] == "PTT.BK"

    def test_holding_persisted_to_database(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Verifies holding is visible in portfolio after adding."""
        add_holding.invoke(
            {
                "symbol": "PTT.BK",
                "asset_type": "stock",
                "name": "PTT",
                "quantity": "100",
                "price_per_unit": "35.50",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        portfolio = view_portfolio.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert portfolio["holding_count"] == 1
        assert portfolio["holdings"][0]["symbol"] == "PTT.BK"


class TestImportCsv:
    """Tests for import_csv tool."""

    def test_valid_csv(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Parses valid CSV content and persists holdings."""
        csv_content = (
            "symbol,asset_type,name,quantity,price_per_unit,purchase_date\n"
            "PTT.BK,stock,PTT,100,35.50,2025-01-15\n"
        )
        result = import_csv.invoke(
            {
                "csv_content": csv_content,
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "import_csv"
        assert result["row_count"] == 1

    def test_invalid_csv_raises(self) -> None:
        """Raises ValueError for invalid CSV."""
        with pytest.raises(Exception, match="empty"):
            import_csv.invoke({"csv_content": ""})


class TestRefreshPrices:
    """Tests for refresh_prices tool."""

    def test_returns_updated_count(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns dict with updated count."""
        with patch(
            "finance_ai.tools.price_client.fetch_multiple_prices",
            return_value={},
        ):
            result = refresh_prices.invoke(
                {
                    "user_id": sample_user.id,
                    "db_session_factory": db_session_factory,
                }
            )
        assert result["action"] == "refresh_prices"
        assert result["updated_count"] == 0


class TestLookupHolding:
    """Tests for lookup_holding tool."""

    def test_not_found(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns not_found status for non-existent holding."""
        result = lookup_holding.invoke(
            {
                "symbol": "PTT.BK",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "lookup_holding"
        assert result["symbol"] == "PTT.BK"
        assert result["status"] == "not_found"

    def test_strips_whitespace(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Strips whitespace from symbol."""
        result = lookup_holding.invoke(
            {
                "symbol": "  AOT.BK  ",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["symbol"] == "AOT.BK"

    def test_found_after_adding(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns found status after adding a holding."""
        add_holding.invoke(
            {
                "symbol": "PTT.BK",
                "asset_type": "stock",
                "name": "PTT",
                "quantity": "100",
                "price_per_unit": "35.50",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        result = lookup_holding.invoke(
            {
                "symbol": "PTT.BK",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["status"] == "found"
        assert result["symbol"] == "PTT.BK"


class TestGetInvestmentAdvice:
    """Tests for get_investment_advice tool."""

    def test_returns_advice_data(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns dict with portfolio data for advice."""
        result = get_investment_advice.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "get_investment_advice"
        assert "total_value" in result
        assert "holdings" in result


# ---------------------------------------------------------------------------
# Schedule + Notification Tool Tests
# ---------------------------------------------------------------------------


class TestCreateAssetSchedule:
    """Tests for create_asset_schedule tool."""

    def test_creates_schedule(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Create a schedule and verify returned data."""
        result = create_asset_schedule.invoke(
            {
                "symbol": "GC=F",
                "cron_expression": "0 21 * * *",
                "description": "ราคาทอง",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "create_asset_schedule"
        assert result["symbol"] == "GC=F"
        assert result["cron_expression"] == "0 21 * * *"
        assert result["description"] == "ราคาทอง"
        assert result["schedule_id"] is not None

    def test_defaults_description_to_symbol(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Uses symbol as description when none provided."""
        result = create_asset_schedule.invoke(
            {
                "symbol": "PTT.BK",
                "cron_expression": "0 9 * * 1-5",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["description"] == "PTT.BK"


class TestViewAssetSchedules:
    """Tests for view_asset_schedules tool."""

    def test_returns_empty(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns empty list when no schedules exist."""
        result = view_asset_schedules.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "view_asset_schedules"
        assert result["count"] == 0
        assert result["schedules"] == []

    def test_returns_active_schedules(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns only active schedules."""
        create_asset_schedule.invoke(
            {
                "symbol": "GC=F",
                "cron_expression": "0 21 * * *",
                "description": "ทอง",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        result = view_asset_schedules.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["count"] == 1


class TestDeleteAssetSchedule:
    """Tests for delete_asset_schedule tool."""

    def test_delete_existing(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Delete an existing schedule."""
        created = create_asset_schedule.invoke(
            {
                "symbol": "GC=F",
                "cron_expression": "0 21 * * *",
                "description": "ทอง",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        result = delete_asset_schedule.invoke(
            {
                "schedule_id": created["schedule_id"],
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "delete_asset_schedule"
        assert result["deleted"] is True

    def test_delete_nonexistent(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns deleted=False for non-existent schedule."""
        result = delete_asset_schedule.invoke(
            {
                "schedule_id": "fake-id",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["deleted"] is False


class TestGetAssetNotifications:
    """Tests for get_asset_notifications tool."""

    def test_returns_empty(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Returns empty list when no notifications exist."""
        result = get_asset_notifications.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "get_asset_notifications"
        assert result["count"] == 0
        assert result["notifications"] == []
