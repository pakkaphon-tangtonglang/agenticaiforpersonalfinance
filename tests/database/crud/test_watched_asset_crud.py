"""Tests for WatchedAssetCRUD."""

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.crud.watched_asset_crud import WatchedAssetCRUD
from finance_ai.database.models.user import User


@pytest.fixture
def crud() -> WatchedAssetCRUD:
    """Create a WatchedAssetCRUD instance."""
    return WatchedAssetCRUD()


class TestWatchedAssetCRUD:
    """Tests for WatchedAssetCRUD operations."""

    def test_add_new_asset(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Add a new watched asset."""
        asset = crud.add(test_session, sample_user.id, "PTT.BK", "PTT Public")

        assert asset is not None
        assert asset.id is not None
        assert asset.symbol == "PTT.BK"
        assert asset.name == "PTT Public"

    def test_add_duplicate_returns_none(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Adding duplicate symbol returns None."""
        crud.add(test_session, sample_user.id, "PTT.BK", "PTT")
        test_session.commit()

        result = crud.add(test_session, sample_user.id, "PTT.BK", "PTT")

        assert result is None

    def test_add_normalizes_symbol(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Symbol should be stripped and uppercased."""
        asset = crud.add(test_session, sample_user.id, "  ptt.bk  ", "PTT")

        assert asset is not None
        assert asset.symbol == "PTT.BK"

    def test_get_by_user(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Get all watched assets for a user."""
        crud.add(test_session, sample_user.id, "PTT.BK", "PTT")
        crud.add(test_session, sample_user.id, "AOT.BK", "AOT")
        test_session.commit()

        assets = crud.get_by_user(test_session, sample_user.id)

        assert len(assets) == 2

    def test_get_by_user_empty(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Return empty list when no watched assets exist."""
        assert crud.get_by_user(test_session, sample_user.id) == []

    def test_remove_asset(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Remove a watched asset."""
        crud.add(test_session, sample_user.id, "PTT.BK", "PTT")
        test_session.commit()

        assert crud.remove(test_session, sample_user.id, "PTT.BK") is True
        assert crud.get_by_user(test_session, sample_user.id) == []

    def test_remove_not_found(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Removing a non-existent asset returns False."""
        assert crud.remove(test_session, sample_user.id, "FAKE.BK") is False
