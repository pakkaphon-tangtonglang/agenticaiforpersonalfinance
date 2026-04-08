"""Tests for expense tracking constants."""

from decimal import Decimal

from finance_ai.tools.expense_constants import (
    EXPENSE_CATEGORIES,
    EXPENSE_TRANSACTION_TYPE,
    MAX_SINGLE_EXPENSE_AMOUNT,
    MIN_EXPENSE_AMOUNT,
    VALID_EXPENSE_CATEGORIES,
)


class TestExpenseCategories:
    """Tests for expense category constants."""

    def test_has_ten_categories(self) -> None:
        """Test that there are exactly 10 expense categories."""
        assert len(EXPENSE_CATEGORIES) == 10

    def test_all_expected_categories_present(self) -> None:
        """Test that all expected category keys are defined."""
        expected = {
            "food",
            "transport",
            "housing",
            "entertainment",
            "utilities",
            "health",
            "education",
            "shopping",
            "investment",
            "other",
        }
        assert set(EXPENSE_CATEGORIES.keys()) == expected

    def test_every_category_has_thai_label(self) -> None:
        """Test that every category has a non-empty Thai label."""
        for key, label in EXPENSE_CATEGORIES.items():
            assert isinstance(label, str), f"Label for {key} is not a string"
            assert len(label) > 0, f"Label for {key} is empty"

    def test_food_category_label(self) -> None:
        """Test food category Thai label."""
        assert EXPENSE_CATEGORIES["food"] == "อาหาร"

    def test_transport_category_label(self) -> None:
        """Test transport category Thai label."""
        assert EXPENSE_CATEGORIES["transport"] == "การเดินทาง"


class TestValidExpenseCategories:
    """Tests for valid expense category list."""

    def test_matches_category_keys(self) -> None:
        """Test that VALID_EXPENSE_CATEGORIES matches EXPENSE_CATEGORIES keys."""
        assert set(VALID_EXPENSE_CATEGORIES) == set(EXPENSE_CATEGORIES.keys())

    def test_is_list_type(self) -> None:
        """Test that VALID_EXPENSE_CATEGORIES is a list."""
        assert isinstance(VALID_EXPENSE_CATEGORIES, list)


class TestExpenseTransactionType:
    """Tests for expense transaction type constant."""

    def test_transaction_type_value(self) -> None:
        """Test expense transaction type is 'expense'."""
        assert EXPENSE_TRANSACTION_TYPE == "expense"


class TestExpenseAmountLimits:
    """Tests for expense amount validation limits."""

    def test_max_amount_is_decimal(self) -> None:
        """Test MAX_SINGLE_EXPENSE_AMOUNT is a Decimal."""
        assert isinstance(MAX_SINGLE_EXPENSE_AMOUNT, Decimal)

    def test_min_amount_is_decimal(self) -> None:
        """Test MIN_EXPENSE_AMOUNT is a Decimal."""
        assert isinstance(MIN_EXPENSE_AMOUNT, Decimal)

    def test_min_amount_is_positive(self) -> None:
        """Test MIN_EXPENSE_AMOUNT is greater than zero."""
        assert MIN_EXPENSE_AMOUNT > Decimal("0")

    def test_max_greater_than_min(self) -> None:
        """Test MAX is greater than MIN."""
        assert MAX_SINGLE_EXPENSE_AMOUNT > MIN_EXPENSE_AMOUNT
