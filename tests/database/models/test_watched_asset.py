"""Tests for WatchedAsset model."""

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from finance_ai.database.models.user import User
from finance_ai.database.models.watched_asset import WatchedAsset


class TestWatchedAsset:
    """Tests for WatchedAsset model."""

    def test_create_watched_asset(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Create a watched asset with required fields."""
        asset = WatchedAsset(user_id=sample_user.id, symbol="PTT.BK", name="PTT")
        test_session.add(asset)
        test_session.commit()
        test_session.refresh(asset)
        assert asset.id is not None
        assert asset.symbol == "PTT.BK"
        assert asset.created_at is not None

    def test_unique_constraint(
        self,
        test_session: Session,
        sample_user: User,
    ) -> None:
        """Duplicate user+symbol raises IntegrityError."""
        asset_one = WatchedAsset(user_id=sample_user.id, symbol="PTT.BK")
        asset_two = WatchedAsset(user_id=sample_user.id, symbol="PTT.BK")
        test_session.add(asset_one)
        test_session.commit()
        test_session.add(asset_two)
        with pytest.raises(IntegrityError):
            test_session.commit()
