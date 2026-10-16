"""Pydantic model for the pillar-based financial health score."""

from decimal import Decimal

from pydantic import BaseModel, Field


class HealthScoreResult(BaseModel):
    """Financial health score with per-pillar breakdown.

    Attributes:
        score: Overall score 0-100, or None when data is insufficient.
        status: "scored" or "insufficient_data".
        pillars: Pillar key -> score 0-100, or None when that pillar has no data.
        coverage: Fraction of pillars with data (4 decimal places).
        missing_data: Thai labels of data needed before a score can be given.

    Example:
        >>> HealthScoreResult(score=None, status="insufficient_data",
        ...     pillars={}, coverage=Decimal("0"), missing_data=["รายรับเดือนนี้"])
    """

    score: int | None = Field(default=None, ge=0, le=100)
    status: str = "insufficient_data"
    pillars: dict[str, int | None] = Field(default_factory=dict)
    coverage: Decimal = Decimal("0")
    missing_data: list[str] = Field(default_factory=list)
