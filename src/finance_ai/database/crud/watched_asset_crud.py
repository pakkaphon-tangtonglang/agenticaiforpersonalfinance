"""CRUD operations for WatchedAsset model."""

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.watched_asset import WatchedAsset


class WatchedAssetCRUD(BaseCRUD[WatchedAsset]):
    """CRUD operations for watched assets."""

    def __init__(self) -> None:
        """Initialize with the WatchedAsset model."""
        super().__init__(WatchedAsset)

    def add(
        self,
        session: Session,
        user_id: str,
        symbol: str,
        name: str = "",
    ) -> WatchedAsset | None:
        """Add a new watched asset.

        Args:
            session: Database session.
            user_id: UUID of the user.
            symbol: Ticker symbol.
            name: Optional asset name.

        Returns:
            Created WatchedAsset or None if duplicate.
        """
        asset = WatchedAsset(
            user_id=user_id,
            symbol=symbol.strip().upper(),
            name=name,
        )
        session.add(asset)
        try:
            session.flush()
        except IntegrityError:
            session.rollback()
            return None
        return asset

    def get_by_user(self, session: Session, user_id: str) -> list[WatchedAsset]:
        """Get all watched assets for a user.

        Args:
            session: Database session.
            user_id: UUID of the user.

        Returns:
            List of watched assets.
        """
        stmt = select(WatchedAsset).where(WatchedAsset.user_id == user_id)
        return list(session.execute(stmt).scalars().all())

    def get_by_symbol(self, session: Session, user_id: str, symbol: str) -> WatchedAsset | None:
        """Get a watched asset by symbol for a user.

        Args:
            session: Database session.
            user_id: UUID of the user.
            symbol: Ticker symbol (e.g., "PTT.BK"); case-insensitive.

        Returns:
            WatchedAsset instance or None.
        """
        stmt = select(WatchedAsset).where(
            WatchedAsset.user_id == user_id,
            WatchedAsset.symbol == symbol.strip().upper(),
        )
        return session.execute(stmt).scalar_one_or_none()

    def list_for_user(self, session: Session, user_id: str) -> list[WatchedAsset]:
        """Get all watched assets for a user, oldest first.

        Args:
            session: Database session.
            user_id: UUID of the user.

        Returns:
            List of watched assets ordered by created_at ascending.
        """
        stmt = (
            select(WatchedAsset)
            .where(WatchedAsset.user_id == user_id)
            .order_by(WatchedAsset.created_at.asc())
        )
        return list(session.execute(stmt).scalars().all())

    def remove(self, session: Session, user_id: str, symbol: str) -> bool:
        """Remove a watched asset.

        Args:
            session: Database session.
            user_id: UUID of the user.
            symbol: Ticker symbol to remove.

        Returns:
            True if removed, False if not found.
        """
        stmt = delete(WatchedAsset).where(
            WatchedAsset.user_id == user_id,
            WatchedAsset.symbol == symbol.strip().upper(),
        )
        result = session.execute(stmt)
        rowcount: int = result.rowcount  # type: ignore[attr-defined]
        return rowcount > 0

    def delete(  # type: ignore[override]  # pylint: disable=arguments-differ
        self,
        session: Session,
        asset_id: str,
        user_id: str,
    ) -> bool:
        """Delete a watched asset by ID (must belong to the user).

        Args:
            session: Database session.
            asset_id: UUID of the watched asset.
            user_id: UUID of the owner.

        Returns:
            True if deleted, False when missing or owned by another user.
        """
        asset = session.get(WatchedAsset, asset_id)
        if asset is None or asset.user_id != user_id:
            return False
        session.delete(asset)
        session.flush()
        return True
