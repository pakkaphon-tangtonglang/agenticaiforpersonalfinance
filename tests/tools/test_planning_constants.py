"""Tests for planning constants and validation functions."""

from decimal import Decimal

import pytest

from finance_ai.tools.planning_constants import (
    DEFAULT_PRIORITY,
    GOAL_TYPES,
    MAX_GOAL_AMOUNT,
    MIN_GOAL_AMOUNT,
    PRIORITY_LEVELS,
    VALID_GOAL_TYPES,
    validate_goal_amount,
    validate_goal_type,
    validate_priority,
)


class TestGoalTypes:
    """Tests for goal type constants."""

    def test_all_types_have_thai_labels(self) -> None:
        """Every goal type has a non-empty Thai label."""
        for key, label in GOAL_TYPES.items():
            assert label, f"Goal type '{key}' has empty label"

    def test_valid_goal_types_matches_keys(self) -> None:
        """VALID_GOAL_TYPES matches GOAL_TYPES keys."""
        assert set(VALID_GOAL_TYPES) == set(GOAL_TYPES.keys())

    def test_expected_types_present(self) -> None:
        """Core goal types are defined."""
        expected = ["savings", "emergency_fund", "retirement", "home_purchase"]
        for goal_type in expected:
            assert goal_type in VALID_GOAL_TYPES


class TestPriorityLevels:
    """Tests for priority level constants."""

    def test_five_levels_defined(self) -> None:
        """Exactly 5 priority levels exist."""
        assert len(PRIORITY_LEVELS) == 5

    def test_levels_range_one_to_five(self) -> None:
        """Priority levels cover 1 through 5."""
        assert set(PRIORITY_LEVELS.keys()) == {1, 2, 3, 4, 5}

    def test_default_priority_valid(self) -> None:
        """Default priority is within valid range."""
        assert DEFAULT_PRIORITY in PRIORITY_LEVELS


class TestValidateGoalType:
    """Tests for validate_goal_type function."""

    def test_valid_type_returns_normalized(self) -> None:
        """Returns normalized lowercase key."""
        assert validate_goal_type("Savings") == "savings"

    def test_lowercase_passthrough(self) -> None:
        """Lowercase input passes through."""
        assert validate_goal_type("retirement") == "retirement"

    def test_strips_whitespace(self) -> None:
        """Strips leading/trailing whitespace."""
        assert validate_goal_type("  education  ") == "education"

    def test_invalid_type_raises(self) -> None:
        """Raises ValueError for unknown type."""
        with pytest.raises(ValueError, match="Unknown goal type"):
            validate_goal_type("crypto_moon")

    def test_empty_string_raises(self) -> None:
        """Raises ValueError for empty string."""
        with pytest.raises(ValueError, match="Unknown goal type"):
            validate_goal_type("")


class TestValidatePriority:
    """Tests for validate_priority function."""

    def test_valid_priorities(self) -> None:
        """All valid priorities return the value."""
        for level in range(1, 6):
            assert validate_priority(level) == level

    def test_zero_raises(self) -> None:
        """Priority 0 raises ValueError."""
        with pytest.raises(ValueError, match="Priority must be between 1 and 5"):
            validate_priority(0)

    def test_six_raises(self) -> None:
        """Priority 6 raises ValueError."""
        with pytest.raises(ValueError, match="Priority must be between 1 and 5"):
            validate_priority(6)

    def test_negative_raises(self) -> None:
        """Negative priority raises ValueError."""
        with pytest.raises(ValueError, match="Priority must be between 1 and 5"):
            validate_priority(-1)


class TestValidateGoalAmount:
    """Tests for validate_goal_amount function."""

    def test_valid_amount(self) -> None:
        """Normal amount passes validation."""
        validate_goal_amount(Decimal("100000"))

    def test_minimum_amount(self) -> None:
        """Exact minimum amount passes."""
        validate_goal_amount(MIN_GOAL_AMOUNT)

    def test_below_minimum_raises(self) -> None:
        """Amount below minimum raises ValueError."""
        with pytest.raises(ValueError, match="Goal amount must be at least"):
            validate_goal_amount(Decimal("0.50"))

    def test_zero_raises(self) -> None:
        """Zero amount raises ValueError."""
        with pytest.raises(ValueError, match="Goal amount must be at least"):
            validate_goal_amount(Decimal("0"))

    def test_negative_raises(self) -> None:
        """Negative amount raises ValueError."""
        with pytest.raises(ValueError, match="Goal amount must be at least"):
            validate_goal_amount(Decimal("-100"))

    def test_above_maximum_raises(self) -> None:
        """Amount above maximum raises ValueError."""
        with pytest.raises(ValueError, match="Goal amount exceeds maximum"):
            validate_goal_amount(MAX_GOAL_AMOUNT + Decimal("1"))
