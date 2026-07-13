"""Planning constants for financial goal validation and display.

Contains goal types with Thai labels, priority levels,
and validation functions for planning operations.
"""

from decimal import Decimal

# Goal types: key -> Thai display label
GOAL_TYPES: dict[str, str] = {
    "savings": "ออมเงิน",
    "emergency_fund": "เงินสำรองฉุกเฉิน",
    "retirement": "เกษียณอายุ",
    "home_purchase": "ซื้อบ้าน",
    "education": "การศึกษา",
    "investment": "การลงทุน",
    "debt_payoff": "ปลดหนี้",
    "travel": "ท่องเที่ยว",
}

# Valid goal type keys for validation
VALID_GOAL_TYPES: list[str] = list(GOAL_TYPES.keys())

# Priority levels: value -> Thai display label
PRIORITY_LEVELS: dict[int, str] = {
    1: "ต่ำมาก",
    2: "ต่ำ",
    3: "ปานกลาง",
    4: "สูง",
    5: "สูงมาก",
}

# Default priority for new goals
DEFAULT_PRIORITY: int = 3

# Minimum goal amount (must be positive)
MIN_GOAL_AMOUNT: Decimal = Decimal("1.00")

# Maximum goal amount (sanity check)
MAX_GOAL_AMOUNT: Decimal = Decimal("999999999999.99")


def validate_goal_type(goal_type: str) -> str:
    """Validate and normalize a goal type key.

    Args:
        goal_type: Goal type key to validate.

    Returns:
        Normalized (lowercase) goal type key.

    Raises:
        ValueError: If goal type is not recognized.

    Example:
        >>> validate_goal_type("Savings")
        'savings'
    """
    normalized = goal_type.strip().lower()
    if normalized not in VALID_GOAL_TYPES:
        raise ValueError(f"Unknown goal type: '{goal_type}'. " f"Valid types: {VALID_GOAL_TYPES}")
    return normalized


def validate_priority(priority: int) -> int:
    """Validate that priority is within the allowed range (1-5).

    Args:
        priority: Priority level to validate.

    Returns:
        Validated priority value.

    Raises:
        ValueError: If priority is not between 1 and 5.

    Example:
        >>> validate_priority(3)
        3
    """
    if priority < 1 or priority > 5:
        raise ValueError(f"Priority must be between 1 and 5. Received: {priority}")
    return priority


def validate_goal_amount(amount: Decimal) -> None:
    """Validate that a goal amount is within the allowed range.

    Args:
        amount: Goal amount to validate.

    Raises:
        ValueError: If amount is below minimum or above maximum.

    Example:
        >>> validate_goal_amount(Decimal("100000"))
    """
    if amount < MIN_GOAL_AMOUNT:
        raise ValueError(
            f"Goal amount must be at least {MIN_GOAL_AMOUNT} THB. " f"Received: {amount} THB"
        )
    if amount > MAX_GOAL_AMOUNT:
        raise ValueError(
            f"Goal amount exceeds maximum of {MAX_GOAL_AMOUNT} THB. " f"Received: {amount} THB"
        )
