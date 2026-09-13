"""Tests for the user provisioning service (auto-create missing users)."""

from typing import Any

from sqlalchemy.orm import Session

from finance_ai.database.crud.user_crud import UserCRUD
from finance_ai.tools.user_provisioning_service import ensure_user_exists

NEW_USER_ID = "11111111-2222-3333-4444-555555555555"


class TestEnsureUserExists:
    """Integration tests for ensure_user_exists."""

    def test_creates_missing_user_with_derived_email(self, test_session: Session) -> None:
        """A user_id not in the DB gets a user row with a derived email."""
        ensure_user_exists(test_session, NEW_USER_ID)
        user = UserCRUD().get_by_id(test_session, NEW_USER_ID)
        assert user is not None
        assert user.email == f"{NEW_USER_ID}@users.finance-ai.local"

    def test_existing_user_unchanged(self, test_session: Session, sample_user: Any) -> None:
        """An existing user row is left untouched."""
        original_email = sample_user.email
        ensure_user_exists(test_session, sample_user.id)
        user = UserCRUD().get_by_id(test_session, sample_user.id)
        assert user is not None
        assert user.email == original_email

    def test_idempotent_when_called_twice(self, test_session: Session) -> None:
        """Calling twice does not raise or duplicate."""
        ensure_user_exists(test_session, NEW_USER_ID)
        ensure_user_exists(test_session, NEW_USER_ID)
        user = UserCRUD().get_by_id(test_session, NEW_USER_ID)
        assert user is not None

    def test_line_mapped_user_not_shadowed(self, test_session: Session) -> None:
        """A LINE-mapped user id (created elsewhere) is not recreated."""
        from finance_ai.database.models.user import User

        line_user = User(
            id="line-existing-id",
            email="line-abc@line.users.finance-ai.local",
            hashed_password="line-login-not-supported",
            full_name="ผู้ใช้ LINE",
        )
        test_session.add(line_user)
        test_session.commit()
        ensure_user_exists(test_session, "line-existing-id")
        user = UserCRUD().get_by_id(test_session, "line-existing-id")
        assert user is not None
        assert user.email == "line-abc@line.users.finance-ai.local"
