"""Tests for investment calculator pure functions."""

from decimal import Decimal

import pytest

from finance_ai.tools.investment_calculator import (
    AssetTypeSummary,
    HoldingRecord,
    HoldingSummary,
    PortfolioSummaryResult,
    build_asset_type_summary,
    build_holding_summary,
    calculate_allocation_percentage,
    calculate_current_value,
    calculate_gain_loss_percentage,
    calculate_portfolio_current_value,
    calculate_portfolio_total_cost,
    calculate_total_cost,
    calculate_unrealized_gain_loss,
    get_active_asset_types,
    summarize_portfolio,
    validate_asset_type,
    validate_mutual_fund_symbol,
    validate_price_per_unit,
    validate_quantity,
    validate_stock_symbol,
    validate_symbol,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def stock_holding() -> HoldingRecord:
    """PTT.BK stock holding with known values."""
    return HoldingRecord(
        symbol="PTT.BK",
        name="PTT",
        asset_type="stock",
        quantity=Decimal("100"),
        average_cost_per_unit=Decimal("35.50"),
        total_cost=Decimal("3550.00"),
        current_price_per_unit=Decimal("42.50"),
        current_value=Decimal("4250.00"),
        unrealized_gain_loss=Decimal("700.00"),
    )


@pytest.fixture
def fund_holding() -> HoldingRecord:
    """K-EQUITY mutual fund holding."""
    return HoldingRecord(
        symbol="K-EQUITY",
        name="K Equity Fund",
        asset_type="mutual_fund",
        quantity=Decimal("500"),
        average_cost_per_unit=Decimal("14.50"),
        total_cost=Decimal("7250.00"),
        current_price_per_unit=Decimal("15.00"),
        current_value=Decimal("7500.00"),
        unrealized_gain_loss=Decimal("250.00"),
    )


@pytest.fixture
def holding_without_price() -> HoldingRecord:
    """Holding with no current price data."""
    return HoldingRecord(
        symbol="AOT.BK",
        name="AOT",
        asset_type="stock",
        quantity=Decimal("50"),
        average_cost_per_unit=Decimal("60.00"),
        total_cost=Decimal("3000.00"),
    )


@pytest.fixture
def mixed_holdings(
    stock_holding: HoldingRecord,
    fund_holding: HoldingRecord,
) -> list[HoldingRecord]:
    """Portfolio with both stocks and mutual funds."""
    return [stock_holding, fund_holding]


# ── Validate Asset Type ──────────────────────────────────────────────


class TestValidateAssetType:
    """Tests for validate_asset_type."""

    def test_valid_stock(self) -> None:
        """Accepts 'stock' as valid."""
        assert validate_asset_type("stock") == "stock"

    def test_valid_mutual_fund(self) -> None:
        """Accepts 'mutual_fund' as valid."""
        assert validate_asset_type("mutual_fund") == "mutual_fund"

    def test_normalizes_case(self) -> None:
        """Normalizes to lowercase."""
        assert validate_asset_type("Stock") == "stock"

    def test_strips_whitespace(self) -> None:
        """Strips leading/trailing whitespace."""
        assert validate_asset_type("  stock  ") == "stock"

    def test_invalid_type_raises(self) -> None:
        """Raises ValueError for unknown type."""
        with pytest.raises(ValueError, match="Unknown asset type"):
            validate_asset_type("crypto")


# ── Validate Stock Symbol ────────────────────────────────────────────


class TestValidateStockSymbol:
    """Tests for validate_stock_symbol."""

    def test_valid_symbol(self) -> None:
        """Accepts valid .BK stock symbol."""
        assert validate_stock_symbol("PTT.BK") == "PTT.BK"

    def test_normalizes_case(self) -> None:
        """Normalizes to uppercase."""
        assert validate_stock_symbol("ptt.bk") == "PTT.BK"

    def test_strips_whitespace(self) -> None:
        """Strips whitespace."""
        assert validate_stock_symbol("  AOT.BK  ") == "AOT.BK"

    def test_missing_suffix_raises(self) -> None:
        """Raises ValueError without .BK suffix."""
        with pytest.raises(ValueError, match="must end with"):
            validate_stock_symbol("PTT")

    def test_empty_raises(self) -> None:
        """Raises ValueError for empty string."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_stock_symbol("")

    def test_whitespace_only_raises(self) -> None:
        """Raises ValueError for whitespace-only string."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_stock_symbol("   ")


# ── Validate Mutual Fund Symbol ──────────────────────────────────────


class TestValidateMutualFundSymbol:
    """Tests for validate_mutual_fund_symbol."""

    def test_valid_k_prefix(self) -> None:
        """Accepts K- prefix."""
        assert validate_mutual_fund_symbol("K-EQUITY") == "K-EQUITY"

    def test_valid_kt_prefix(self) -> None:
        """Accepts KT- prefix."""
        assert validate_mutual_fund_symbol("KT-BOND") == "KT-BOND"

    def test_valid_scb_prefix(self) -> None:
        """Accepts SCB- prefix."""
        assert validate_mutual_fund_symbol("SCB-SSF") == "SCB-SSF"

    def test_normalizes_case(self) -> None:
        """Normalizes to uppercase."""
        assert validate_mutual_fund_symbol("k-equity") == "K-EQUITY"

    def test_invalid_prefix_raises(self) -> None:
        """Raises ValueError for invalid prefix."""
        with pytest.raises(ValueError, match="must start with"):
            validate_mutual_fund_symbol("TISCO-FUND")

    def test_empty_raises(self) -> None:
        """Raises ValueError for empty string."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_mutual_fund_symbol("")


# ── Validate Symbol (dispatcher) ─────────────────────────────────────


class TestValidateSymbol:
    """Tests for validate_symbol."""

    def test_dispatches_stock(self) -> None:
        """Dispatches to stock validation for stock type."""
        assert validate_symbol("PTT.BK", "stock") == "PTT.BK"

    def test_dispatches_mutual_fund(self) -> None:
        """Dispatches to fund validation for mutual_fund type."""
        assert validate_symbol("K-EQUITY", "mutual_fund") == "K-EQUITY"

    def test_unknown_type_normalizes(self) -> None:
        """Unknown types just normalize case."""
        assert validate_symbol("  abc  ", "other") == "ABC"


# ── Validate Quantity ────────────────────────────────────────────────


class TestValidateQuantity:
    """Tests for validate_quantity."""

    def test_valid_quantity(self) -> None:
        """Accepts valid quantity."""
        validate_quantity(Decimal("100"))

    def test_minimum_quantity(self) -> None:
        """Accepts minimum quantity."""
        validate_quantity(Decimal("0.0001"))

    def test_below_minimum_raises(self) -> None:
        """Raises ValueError below minimum."""
        with pytest.raises(ValueError, match="must be at least"):
            validate_quantity(Decimal("0.00001"))

    def test_zero_raises(self) -> None:
        """Raises ValueError for zero."""
        with pytest.raises(ValueError, match="must be at least"):
            validate_quantity(Decimal("0"))

    def test_negative_raises(self) -> None:
        """Raises ValueError for negative."""
        with pytest.raises(ValueError, match="must be at least"):
            validate_quantity(Decimal("-1"))


# ── Validate Price Per Unit ──────────────────────────────────────────


class TestValidatePricePerUnit:
    """Tests for validate_price_per_unit."""

    def test_valid_price(self) -> None:
        """Accepts valid price."""
        validate_price_per_unit(Decimal("35.50"))

    def test_minimum_price(self) -> None:
        """Accepts minimum price."""
        validate_price_per_unit(Decimal("0.01"))

    def test_below_minimum_raises(self) -> None:
        """Raises ValueError below minimum."""
        with pytest.raises(ValueError, match="must be at least"):
            validate_price_per_unit(Decimal("0.001"))

    def test_above_maximum_raises(self) -> None:
        """Raises ValueError above maximum."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_price_per_unit(Decimal("200000000.00"))

    def test_zero_raises(self) -> None:
        """Raises ValueError for zero."""
        with pytest.raises(ValueError, match="must be at least"):
            validate_price_per_unit(Decimal("0"))


# ── Calculate Total Cost ─────────────────────────────────────────────


class TestCalculateTotalCost:
    """Tests for calculate_total_cost."""

    def test_normal_calculation(self) -> None:
        """Computes quantity * price."""
        result = calculate_total_cost(Decimal("100"), Decimal("35.50"))
        assert result == Decimal("3550.00")

    def test_quantizes_to_two_decimals(self) -> None:
        """Result has 2 decimal places."""
        result = calculate_total_cost(Decimal("3"), Decimal("10.333"))
        assert result == Decimal("31.00")

    def test_zero_quantity(self) -> None:
        """Zero quantity yields zero cost."""
        result = calculate_total_cost(Decimal("0"), Decimal("35.50"))
        assert result == Decimal("0.00")


# ── Calculate Current Value ──────────────────────────────────────────


class TestCalculateCurrentValue:
    """Tests for calculate_current_value."""

    def test_normal_calculation(self) -> None:
        """Computes quantity * current_price."""
        result = calculate_current_value(Decimal("100"), Decimal("42.50"))
        assert result == Decimal("4250.00")

    def test_zero_quantity(self) -> None:
        """Zero quantity yields zero value."""
        result = calculate_current_value(Decimal("0"), Decimal("42.50"))
        assert result == Decimal("0.00")


# ── Calculate Unrealized Gain/Loss ───────────────────────────────────


class TestCalculateUnrealizedGainLoss:
    """Tests for calculate_unrealized_gain_loss."""

    def test_profit(self) -> None:
        """Positive when current value > cost."""
        result = calculate_unrealized_gain_loss(Decimal("4250"), Decimal("3550"))
        assert result == Decimal("700")

    def test_loss(self) -> None:
        """Negative when current value < cost."""
        result = calculate_unrealized_gain_loss(Decimal("3000"), Decimal("3550"))
        assert result == Decimal("-550")

    def test_breakeven(self) -> None:
        """Zero when value equals cost."""
        result = calculate_unrealized_gain_loss(Decimal("3550"), Decimal("3550"))
        assert result == Decimal("0")


# ── Calculate Gain/Loss Percentage ───────────────────────────────────


class TestCalculateGainLossPercentage:
    """Tests for calculate_gain_loss_percentage."""

    def test_profit_percentage(self) -> None:
        """Computes correct profit percentage."""
        result = calculate_gain_loss_percentage(Decimal("700"), Decimal("3550"))
        assert result == Decimal("19.7183")

    def test_loss_percentage(self) -> None:
        """Computes correct loss percentage (negative)."""
        result = calculate_gain_loss_percentage(Decimal("-550"), Decimal("3550"))
        assert result == Decimal("-15.4930")

    def test_zero_cost_returns_zero(self) -> None:
        """Returns 0 when total cost is zero."""
        result = calculate_gain_loss_percentage(Decimal("100"), Decimal("0"))
        assert result == Decimal("0.0000")


# ── Calculate Allocation Percentage ──────────────────────────────────


class TestCalculateAllocationPercentage:
    """Tests for calculate_allocation_percentage."""

    def test_normal_allocation(self) -> None:
        """Computes correct allocation percentage."""
        result = calculate_allocation_percentage(Decimal("3550"), Decimal("10000"))
        assert result == Decimal("35.5000")

    def test_full_allocation(self) -> None:
        """100% when holding equals total."""
        result = calculate_allocation_percentage(Decimal("10000"), Decimal("10000"))
        assert result == Decimal("100.0000")

    def test_zero_total_returns_zero(self) -> None:
        """Returns 0 when total is zero."""
        result = calculate_allocation_percentage(Decimal("100"), Decimal("0"))
        assert result == Decimal("0.0000")


# ── Calculate Portfolio Total Cost ───────────────────────────────────


class TestCalculatePortfolioTotalCost:
    """Tests for calculate_portfolio_total_cost."""

    def test_multiple_holdings(self, mixed_holdings: list[HoldingRecord]) -> None:
        """Sums cost across holdings."""
        result = calculate_portfolio_total_cost(mixed_holdings)
        assert result == Decimal("10800.00")

    def test_empty_list(self) -> None:
        """Returns 0 for empty list."""
        result = calculate_portfolio_total_cost([])
        assert result == Decimal("0")


# ── Calculate Portfolio Current Value ────────────────────────────────


class TestCalculatePortfolioCurrentValue:
    """Tests for calculate_portfolio_current_value."""

    def test_all_priced(self, mixed_holdings: list[HoldingRecord]) -> None:
        """Sums values when all holdings have prices."""
        result = calculate_portfolio_current_value(mixed_holdings)
        assert result == Decimal("11750.00")

    def test_some_missing_returns_none(
        self,
        stock_holding: HoldingRecord,
        holding_without_price: HoldingRecord,
    ) -> None:
        """Returns None when any holding lacks a price."""
        result = calculate_portfolio_current_value([stock_holding, holding_without_price])
        assert result is None

    def test_empty_list_returns_none(self) -> None:
        """Returns None for empty list."""
        result = calculate_portfolio_current_value([])
        assert result is None


# ── Get Active Asset Types ───────────────────────────────────────────


class TestGetActiveAssetTypes:
    """Tests for get_active_asset_types."""

    def test_mixed_holdings(self, mixed_holdings: list[HoldingRecord]) -> None:
        """Returns sorted unique types."""
        result = get_active_asset_types(mixed_holdings)
        assert result == ["mutual_fund", "stock"]

    def test_empty_list(self) -> None:
        """Returns empty list for no holdings."""
        assert get_active_asset_types([]) == []


# ── Build Holding Summary ────────────────────────────────────────────


class TestBuildHoldingSummary:
    """Tests for build_holding_summary."""

    def test_with_price(self, stock_holding: HoldingRecord) -> None:
        """Builds summary with gain/loss percentage."""
        summary = build_holding_summary(stock_holding, Decimal("10800"))
        assert isinstance(summary, HoldingSummary)
        assert summary.symbol == "PTT.BK"
        assert summary.asset_type_label == "หุ้น"
        assert summary.gain_loss_percentage == Decimal("19.7183")
        assert summary.allocation_percentage > Decimal("0")

    def test_without_price(self, holding_without_price: HoldingRecord) -> None:
        """Builds summary with None for gain/loss percentage."""
        summary = build_holding_summary(holding_without_price, Decimal("10000"))
        assert summary.gain_loss_percentage is None
        assert summary.current_value is None


# ── Build Asset Type Summary ─────────────────────────────────────────


class TestBuildAssetTypeSummary:
    """Tests for build_asset_type_summary."""

    def test_stock_summary(self, mixed_holdings: list[HoldingRecord]) -> None:
        """Aggregates stock holdings correctly."""
        summary = build_asset_type_summary(mixed_holdings, "stock", Decimal("10800"))
        assert isinstance(summary, AssetTypeSummary)
        assert summary.asset_type == "stock"
        assert summary.asset_type_label == "หุ้น"
        assert summary.holding_count == 1
        assert summary.total_cost == Decimal("3550.00")
        assert summary.current_value == Decimal("4250.00")

    def test_no_matching_type(self, mixed_holdings: list[HoldingRecord]) -> None:
        """Empty result for type with no holdings."""
        summary = build_asset_type_summary(mixed_holdings, "bond", Decimal("10800"))
        assert summary.holding_count == 0
        assert summary.total_cost == Decimal("0")


# ── Summarize Portfolio ──────────────────────────────────────────────


class TestSummarizePortfolio:
    """Tests for summarize_portfolio."""

    def test_mixed_portfolio(self, mixed_holdings: list[HoldingRecord]) -> None:
        """Full integration with mixed holdings."""
        result = summarize_portfolio(mixed_holdings)
        assert isinstance(result, PortfolioSummaryResult)
        assert result.holding_count == 2
        assert result.total_cost == Decimal("10800.00")
        assert result.total_current_value == Decimal("11750.00")
        assert result.total_unrealized_gain_loss == Decimal("950.00")
        assert result.total_gain_loss_percentage is not None
        assert len(result.holdings) == 2
        assert len(result.asset_type_breakdown) == 2

    def test_empty_portfolio(self) -> None:
        """Handles empty portfolio gracefully."""
        result = summarize_portfolio([])
        assert result.holding_count == 0
        assert result.total_cost == Decimal("0")
        assert result.total_current_value is None
        assert result.total_unrealized_gain_loss is None
        assert result.holdings == []
        assert result.asset_type_breakdown == []

    def test_portfolio_with_missing_prices(
        self,
        stock_holding: HoldingRecord,
        holding_without_price: HoldingRecord,
    ) -> None:
        """Returns None for value when any holding lacks price."""
        result = summarize_portfolio([stock_holding, holding_without_price])
        assert result.holding_count == 2
        assert result.total_cost == Decimal("6550.00")
        assert result.total_current_value is None
        assert result.total_unrealized_gain_loss is None
        assert result.total_gain_loss_percentage is None

    def test_single_holding(self, stock_holding: HoldingRecord) -> None:
        """Works with a single holding."""
        result = summarize_portfolio([stock_holding])
        assert result.holding_count == 1
        assert result.total_cost == Decimal("3550.00")
        assert result.total_current_value == Decimal("4250.00")
        assert len(result.asset_type_breakdown) == 1
        assert result.asset_type_breakdown[0].allocation_percentage == Decimal("100.0000")
