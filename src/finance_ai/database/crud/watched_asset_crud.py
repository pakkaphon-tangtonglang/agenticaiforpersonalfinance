"""CRUD operations for the WatchedAsset model."""

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from finance_ai.database.models.watched_asset import WatchedAsset


class WatchedAssetCRUD:
    """Create, read, and delete operations for the user asset watchlist.

    Example:
        >>> crud = WatchedAssetCRUD()
        >>> crud.add(session, user_id, "PTT.BK", "PTT")
    """

    def add(
        self,
        session: Session,
        user_id: str,
        symbol: str,
        name: str = "",
    ) -> WatchedAsset | None:
        """Add a symbol to the watchlist, ignoring duplicates.

        Args:
            session: Database session.
            user_id: UUID of the user.
            symbol: Ticker symbol (e.g., "PTT.BK").
            name: Display name (e.g., "PTT"). Defaults to symbol.

        Returns:
            New WatchedAsset record, or None if already exists.

        Example:
            >>> asset = crud.add(session, uid, "PTT.BK", "PTT")
        """
        record = WatchedAsset(
            user_id=user_id,
            symbol=symbol.upper(),
            name=name or symbol.upper(),
        )
        try:
            session.add(record)
            session.flush()
            return record
        except IntegrityError:
            session.rollback()
            return None

    def remove(
        self,
        session: Session,
        user_id: str,
        symbol: str,
    ) -> bool:
        """Remove a symbol from the watchlist.

        Args:
            session: Database session.
            user_id: UUID of the user.
            symbol: Ticker symbol to remove.

        Returns:
            True if a record was deleted, False if not found.

        Example:
            >>> deleted = crud.remove(session, uid, "PTT.BK")
        """
        cursor = session.execute(
            delete(WatchedAsset).where(
                WatchedAsset.user_id == user_id,
                WatchedAsset.symbol == symbol.upper(),
            )
        )
        return bool(getattr(cursor, "rowcount", 0) > 0)

    def list_all(
        self,
        session: Session,
        user_id: str,
    ) -> list[WatchedAsset]:
        """List all watched assets for a user.

        Args:
            session: Database session.
            user_id: UUID of the user.

        Returns:
            List of WatchedAsset records ordered by symbol.

        Example:
            >>> assets = crud.list_all(session, uid)
        """
        return list(
            session.execute(
                select(WatchedAsset)
                .where(WatchedAsset.user_id == user_id)
                .order_by(WatchedAsset.symbol)
            ).scalars()
        )
