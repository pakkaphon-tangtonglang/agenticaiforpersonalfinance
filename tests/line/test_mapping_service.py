"""Tests for the LINE user mapping service."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.models.conversation import Conversation
from finance_ai.database.models.line_user_mapping import LineUserMapping
from finance_ai.database.models.user import User
from finance_ai.line.mapping_service import get_or_create_line_mapping

LINE_USER_ID = "Uline-user-abc123"


class TestGetOrCreateLineMapping:
    """Tests for get_or_create_line_mapping."""

    def test_creates_mapping_with_user_and_conversation(self, test_session: Session) -> None:
        """First message auto-creates a LINE user, conversation, and mapping."""
        mapping = get_or_create_line_mapping(test_session, LINE_USER_ID)
        assert mapping.line_user_id == LINE_USER_ID
        user = test_session.get(User, mapping.user_id)
        conversation = test_session.get(Conversation, mapping.conversation_id)
        assert user is not None
        assert user.email.startswith("line-")
        assert conversation is not None

    def test_second_lookup_returns_same_mapping(self, test_session: Session) -> None:
        """Lookups are idempotent — one mapping per LINE user."""
        first = get_or_create_line_mapping(test_session, LINE_USER_ID)
        second = get_or_create_line_mapping(test_session, LINE_USER_ID)
        assert first.id == second.id
        users = test_session.scalars(select(User)).all()
        assert len(users) == 1
