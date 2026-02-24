"""Tests for pure expense calculation functions."""

from datetime import date
from decimal import Decimal

import pytest

from finance_ai.tools.expense_calculator import (
    CategorySummary,
    ExpenseRecord,
    ExpenseSummaryResult,
    build_category_summary,
    calculate_category_percentage,
    calculate_category_total,
    calculate_grand_total,
    count_expenses_by_category,
    get_active_categories,
    summarize_expenses,
    validate_expense_amount,
    validate_expense_category,
)


@pytest.fixture
def food_expense() -> ExpenseRecord:
    """A single food expense record."""
    return ExpenseRecord(
        amount=Decimal("80.00"),
        category="food",
        description="กาแฟ",
        transaction_date=date(2026, 2, 1),
    )


@pytest.fixture
def transport_expense() -> ExpenseRecord:
    """A single transport expense record."""
    return ExpenseRecord(
        amount=Decimal("120.00"),
        category="transport",
        description="BTS",
        transaction_date=date(2026, 2, 1),
    )


@pytest.fixture
def mixed_expenses(
    food_expense: ExpenseRecord,
    transport_expense: ExpenseRecord,
) -> list[ExpenseRecord]:
    """Multiple expenses across categories."""
    return [
        food_expense,
        ExpenseRecord(
            amount=Decimal("200.00"),
            category="food",
            description="ข้าวกลางวัน",
            transaction_date=date(2026, 2, 2),
        ),
        transport_expense,
    ]


class TestValidateExpenseAmount:
    """Tests for expense amount validation."""

    def test_valid_amount(self) -> None:
        """Accepts a normal expense amount."""
        validate_expense_amount(Decimal("100.00"))

    def test_minimum_amount(self) -> None:
        """Accepts the minimum allowed amount."""
        validate_expense_amount(Decimal("0.01"))

    def test_below_minimum_raises(self) -> None:
        """Rejects amount below minimum."""
        with pytest.raises(ValueError, match="must be at least"):
            validate_expense_amount(Decimal("0"))

    def test_negative_amount_raises(self) -> None:
        """Rejects negative amount."""
        with pytest.raises(ValueError, match="must be at least"):
            validate_expense_amount(Decimal("-50"))

    def test_above_maximum_raises(self) -> None:
        """Rejects amount above maximum."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_expense_amount(Decimal("20000000"))

    def test_maximum_amount(self) -> None:
        """Accepts the maximum allowed amount."""
        validate_expense_amount(Decimal("10000000.00"))


class TestValidateExpenseCategory:
    """Tests for expense category validation."""

    def test_valid_category(self) -> None:
        """Accepts a valid category key."""
        assert validate_expense_category("food") == "food"

    def test_normalizes_uppercase(self) -> None:
        """Normalizes uppercase input to lowercase."""
        assert validate_expense_category("Food") == "food"

    def test_normalizes_whitespace(self) -> None:
        """Strips whitespace from input."""
        assert validate_expense_category("  transport  ") == "transport"

    def test_all_valid_categories(self) -> None:
        """Accepts all 8 valid categories."""
        valid = [
            "food",
            "transport",
            "entertainment",
            "utilities",
            "health",
            "education",
            "shopping",
            "other",
        ]
        for category in valid:
            assert validate_expense_category(category) == category

    def test_invalid_category_raises(self) -> None:
        """Rejects unknown category."""
        with pytest.raises(ValueError, match="Unknown expense category"):
            validate_expense_category("gambling")

    def test_empty_string_raises(self) -> None:
        """Rejects empty string."""
        with pytest.raises(ValueError, match="Unknown expense category"):
            validate_expense_category("")


class TestCalculateCategoryTotal:
    """Tests for category total calculation."""

    def test_single_expense(self, food_expense: ExpenseRecord) -> None:
        """Sums a single expense in the category."""
        total = calculate_category_total([food_expense], "food")
        assert total == Decimal("80.00")

    def test_multiple_same_category(
        self,
        mixed_expenses: list[ExpenseRecord],
    ) -> None:
        """Sums multiple expenses in the same category."""
        total = calculate_category_total(mixed_expenses, "food")
        assert total == Decimal("280.00")

    def test_filters_other_categories(
        self,
        mixed_expenses: list[ExpenseRecord],
    ) -> None:
        """Excludes expenses from other categories."""
        total = calculate_category_total(mixed_expenses, "transport")
        assert total == Decimal("120.00")

    def test_empty_list(self) -> None:
        """Returns zero for empty expense list."""
        assert calculate_category_total([], "food") == Decimal("0")

    def test_no_matching_category(
        self,
        food_expense: ExpenseRecord,
    ) -> None:
        """Returns zero when no expenses match the category."""
        assert calculate_category_total([food_expense], "transport") == Decimal("0")


class TestCountExpensesByCategory:
    """Tests for counting expenses by category."""

    def test_counts_matching(
        self,
        mixed_expenses: list[ExpenseRecord],
    ) -> None:
        """Counts expenses matching the category."""
        assert count_expenses_by_category(mixed_expenses, "food") == 2

    def test_empty_list(self) -> None:
        """Returns zero for empty list."""
        assert count_expenses_by_category([], "food") == 0

    def test_no_matching(
        self,
        food_expense: ExpenseRecord,
    ) -> None:
        """Returns zero when no expenses match."""
        assert count_expenses_by_category([food_expense], "health") == 0


class TestCalculateCategoryPercentage:
    """Tests for category percentage calculation."""

    def test_normal_percentage(self) -> None:
        """Calculates correct percentage."""
        result = calculate_category_percentage(Decimal("250"), Decimal("1000"))
        assert result == Decimal("0.2500")

    def test_full_percentage(self) -> None:
        """Returns 1.0000 when category equals total."""
        result = calculate_category_percentage(Decimal("500"), Decimal("500"))
        assert result == Decimal("1.0000")

    def test_zero_grand_total(self) -> None:
        """Returns 0.0000 when grand total is zero."""
        result = calculate_category_percentage(Decimal("0"), Decimal("0"))
        assert result == Decimal("0.0000")

    def test_small_percentage(self) -> None:
        """Handles small percentages with precision."""
        result = calculate_category_percentage(Decimal("1"), Decimal("3"))
        assert result == Decimal("0.3333")


class TestBuildCategorySummary:
    """Tests for building category summary."""

    def test_with_expenses(
        self,
        mixed_expenses: list[ExpenseRecord],
    ) -> None:
        """Builds correct summary for a category with expenses."""
        summary = build_category_summary(
            mixed_expenses,
            "food",
            Decimal("400.00"),
        )
        assert summary.category == "food"
        assert summary.category_label == "อาหาร"
        assert summary.total_amount == Decimal("280.00")
        assert summary.transaction_count == 2
        assert summary.percentage_of_total == Decimal("0.7000")

    def test_empty_category(self) -> None:
        """Builds summary with zeros for empty category."""
        summary = build_category_summary([], "food", Decimal("1000"))
        assert summary.total_amount == Decimal("0")
        assert summary.transaction_count == 0
        assert summary.percentage_of_total == Decimal("0.0000")


class TestCalculateGrandTotal:
    """Tests for grand total calculation."""

    def test_with_expenses(
        self,
        mixed_expenses: list[ExpenseRecord],
    ) -> None:
        """Sums all expenses correctly."""
        assert calculate_grand_total(mixed_expenses) == Decimal("400.00")

    def test_empty_list(self) -> None:
        """Returns zero for empty list."""
        assert calculate_grand_total([]) == Decimal("0")

    def test_single_expense(self, food_expense: ExpenseRecord) -> None:
        """Returns the amount of a single expense."""
        assert calculate_grand_total([food_expense]) == Decimal("80.00")


class TestGetActiveCategories:
    """Tests for getting active categories."""

    def test_multiple_categories(
        self,
        mixed_expenses: list[ExpenseRecord],
    ) -> None:
        """Returns sorted unique categories."""
        categories = get_active_categories(mixed_expenses)
        assert categories == ["food", "transport"]

    def test_empty_list(self) -> None:
        """Returns empty list for no expenses."""
        assert get_active_categories([]) == []

    def test_single_category(self, food_expense: ExpenseRecord) -> None:
        """Returns single category."""
        assert get_active_categories([food_expense]) == ["food"]


class TestSummarizeExpenses:
    """Tests for the main summarize_expenses function."""

    def test_multiple_categories(
        self,
        mixed_expenses: list[ExpenseRecord],
    ) -> None:
        """Summarizes expenses across multiple categories."""
        result = summarize_expenses(
            mixed_expenses,
            date(2026, 2, 1),
            date(2026, 2, 28),
        )
        assert isinstance(result, ExpenseSummaryResult)
        assert result.total_amount == Decimal("400.00")
        assert result.transaction_count == 3
        assert len(result.category_breakdown) == 2
        assert result.start_date == date(2026, 2, 1)
        assert result.end_date == date(2026, 2, 28)

    def test_empty_expenses(self) -> None:
        """Returns zero summary for no expenses."""
        result = summarize_expenses([], date(2026, 2, 1), date(2026, 2, 28))
        assert result.total_amount == Decimal("0")
        assert result.transaction_count == 0
        assert result.category_breakdown == []

    def test_single_category(self, food_expense: ExpenseRecord) -> None:
        """Summarizes single-category expenses."""
        result = summarize_expenses(
            [food_expense],
            date(2026, 2, 1),
            date(2026, 2, 28),
        )
        assert len(result.category_breakdown) == 1
        assert result.category_breakdown[0].category == "food"
        assert result.category_breakdown[0].percentage_of_total == Decimal("1.0000")

    def test_category_breakdown_sorted(
        self,
        mixed_expenses: list[ExpenseRecord],
    ) -> None:
        """Category breakdown is sorted alphabetically."""
        result = summarize_expenses(
            mixed_expenses,
            date(2026, 2, 1),
            date(2026, 2, 28),
        )
        categories = [cat.category for cat in result.category_breakdown]
        assert categories == sorted(categories)


class TestExpenseRecordModel:
    """Tests for ExpenseRecord Pydantic model."""

    def test_creates_with_required_fields(self) -> None:
        """Creates record with required fields only."""
        record = ExpenseRecord(
            amount=Decimal("100"),
            category="food",
            transaction_date=date(2026, 2, 1),
        )
        assert record.amount == Decimal("100")
        assert record.description == ""

    def test_creates_with_all_fields(self) -> None:
        """Creates record with all fields."""
        record = ExpenseRecord(
            amount=Decimal("80"),
            category="food",
            description="กาแฟ",
            transaction_date=date(2026, 2, 1),
        )
        assert record.description == "กาแฟ"


class TestCategorySummaryModel:
    """Tests for CategorySummary Pydantic model."""

    def test_default_percentage(self) -> None:
        """Percentage defaults to 0.0000."""
        summary = CategorySummary(
            category="food",
            category_label="อาหาร",
            total_amount=Decimal("100"),
            transaction_count=1,
        )
        assert summary.percentage_of_total == Decimal("0.0000")
