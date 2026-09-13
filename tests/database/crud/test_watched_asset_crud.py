"""Tests for WatchedAssetCRUD."""

# pylint: disable=redefined-outer-name

from datetime import datetime

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.crud.watched_asset_crud import WatchedAssetCRUD
from finance_ai.database.models.user import User
from finance_ai.database.models.watched_asset import WatchedAsset


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


class TestListForUser:
    """Tests for the ordered list_for_user query."""

    def test_orders_by_created_at_ascending(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Assets are listed oldest-first by created_at."""
        later = WatchedAsset(
            user_id=sample_user.id,
            symbol="AOT.BK",
            name="AOT",
            created_at=datetime(2026, 9, 2, 10, 0),
        )
        earlier = WatchedAsset(
            user_id=sample_user.id,
            symbol="PTT.BK",
            name="PTT",
            created_at=datetime(2026, 9, 1, 10, 0),
        )
        test_session.add_all([later, earlier])
        test_session.commit()

        assets = crud.list_for_user(test_session, sample_user.id)

        assert [asset.symbol for asset in assets] == ["PTT.BK", "AOT.BK"]

    def test_scoped_to_user(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Assets belonging to other users are not listed."""
        other_user = User(
            email="other@example.com",
            hashed_password="x",
            full_name="Other",
        )
        test_session.add(other_user)
        test_session.commit()
        test_session.refresh(other_user)
        crud.add(test_session, other_user.id, "KBANK.BK", "KBANK")
        crud.add(test_session, sample_user.id, "PTT.BK", "PTT")
        test_session.commit()

        assets = crud.list_for_user(test_session, sample_user.id)

        assert [asset.symbol for asset in assets] == ["PTT.BK"]

    def test_empty_for_new_user(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """A user with no assets gets an empty list."""
        assert crud.list_for_user(test_session, sample_user.id) == []


class TestDeleteByUser:
    """Tests for the user-scoped delete by asset id."""

    def test_deletes_own_asset(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """The owner can delete their asset by id."""
        asset = crud.add(test_session, sample_user.id, "PTT.BK", "PTT")
        assert asset is not None
        test_session.commit()

        assert crud.delete(test_session, asset.id, sample_user.id) is True
        assert crud.get_by_user(test_session, sample_user.id) == []

    def test_rejects_other_users_asset(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """A user cannot delete another user's asset."""
        other_user = User(
            email="deleter@example.com",
            hashed_password="x",
            full_name="Deleter",
        )
        test_session.add(other_user)
        test_session.commit()
        test_session.refresh(other_user)
        asset = crud.add(test_session, sample_user.id, "PTT.BK", "PTT")
        assert asset is not None
        test_session.commit()

        assert crud.delete(test_session, asset.id, other_user.id) is False
        assert crud.get_by_user(test_session, sample_user.id) != []

    def test_missing_id_returns_false(
        self,
        crud: WatchedAssetCRUD,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Deleting a non-existent asset id returns False."""
        assert crud.delete(test_session, "missing-id", sample_user.id) is False
