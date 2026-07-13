"""CRUD operations for the InvestmentHolding model."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.investment_holding import InvestmentHolding


class InvestmentHoldingCRUD(BaseCRUD[InvestmentHolding]):
    """
    CRUD operations specific to InvestmentHolding model.

    Example:
        >>> investment_crud = InvestmentHoldingCRUD()
        >>> holdings = investment_crud.get_by_user(session, user_id)
    """

    def __init__(self) -> None:
        """Initialize InvestmentHoldingCRUD with InvestmentHolding model."""
        super().__init__(InvestmentHolding)

    def get_by_user(self, session: Session, user_id: str) -> list[InvestmentHolding]:
        """
        Get all holdings for a user.

        Args:
            session: Database session.
            user_id: UUID of the user.

        Returns:
            List of InvestmentHolding instances.
        """
        statement = select(InvestmentHolding).where(InvestmentHolding.user_id == user_id)
        return list(session.execute(statement).scalars().all())

    def get_by_symbol(
        self, session: Session, user_id: str, symbol: str
    ) -> Optional[InvestmentHolding]:
        """
        Get a specific holding by symbol for a user.

        Args:
            session: Database session.
            user_id: UUID of the user.
            symbol: Asset symbol (e.g., "PTT.BK").

        Returns:
            InvestmentHolding instance or None.
        """
        statement = select(InvestmentHolding).where(
            InvestmentHolding.user_id == user_id,
            InvestmentHolding.symbol == symbol,
        )
        return session.execute(statement).scalar_one_or_none()

    def get_by_asset_type(
        self, session: Session, user_id: str, asset_type: str
    ) -> list[InvestmentHolding]:
        """
        Get holdings filtered by asset type for a user.

        Args:
            session: Database session.
            user_id: UUID of the user.
            asset_type: Type of asset (stock/mutual_fund/bond/deposit).

        Returns:
            List of matching InvestmentHolding instances.
        """
        statement = select(InvestmentHolding).where(
            InvestmentHolding.user_id == user_id,
            InvestmentHolding.asset_type == asset_type,
        )
        return list(session.execute(statement).scalars().all())

    def update_market_price(
        self,
        session: Session,
        holding_id: str,
        current_price: Decimal,
    ) -> Optional[InvestmentHolding]:
        """
        Update the market price and recalculate values for a holding.

        Args:
            session: Database session.
            holding_id: UUID of the holding.
            current_price: New price per unit.

        Returns:
            Updated InvestmentHolding or None if not found.
        """
        holding = self.get_by_id(session, holding_id)
        if holding is None:
            return None
        holding.current_price_per_unit = current_price
        holding.current_value = holding.quantity * current_price
        holding.unrealized_gain_loss = holding.current_value - holding.total_cost
        holding.last_price_update = datetime.now(timezone.utc)
        session.commit()
        session.refresh(holding)
        return holding
