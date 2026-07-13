"""Tests for ConversationMessageCRUD operations."""

import time

from sqlalchemy.orm import Session

from finance_ai.database.crud.conversation_crud import ConversationCRUD
from finance_ai.database.crud.conversation_message_crud import ConversationMessageCRUD
from finance_ai.database.models.user import User


class TestConversationMessageCRUD:
    """Tests for ConversationMessage-specific CRUD operations."""

    def _create_conversation(self, session: Session, user: User) -> str:
        """Helper to create a conversation and return its id."""
        conv = ConversationCRUD().create(session, user_id=user.id, title="test")
        return conv.id

    def test_create_message(self, test_session: Session, sample_user: User) -> None:
        """Test creating a message."""
        conv_id = self._create_conversation(test_session, sample_user)
        crud = ConversationMessageCRUD()
        msg = crud.create(
            test_session,
            conversation_id=conv_id,
            role="user",
            content="สวัสดี",
        )
        assert msg.id is not None
        assert msg.role == "user"
        assert msg.content == "สวัสดี"

    def test_get_by_conversation(self, test_session: Session, sample_user: User) -> None:
        """Test getting messages for a conversation in order."""
        conv_id = self._create_conversation(test_session, sample_user)
        crud = ConversationMessageCRUD()
        crud.create(test_session, conversation_id=conv_id, role="user", content="msg 1")
        crud.create(test_session, conversation_id=conv_id, role="assistant", content="msg 2")
        crud.create(test_session, conversation_id=conv_id, role="user", content="msg 3")
        messages = crud.get_by_conversation(test_session, conv_id)
        assert len(messages) == 3
        assert messages[0].content == "msg 1"
        assert messages[2].content == "msg 3"

    def test_get_by_conversation_respects_limit(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test that limit parameter is respected."""
        conv_id = self._create_conversation(test_session, sample_user)
        crud = ConversationMessageCRUD()
        for i in range(10):
            crud.create(test_session, conversation_id=conv_id, role="user", content=f"msg {i}")
        messages = crud.get_by_conversation(test_session, conv_id, limit=5)
        assert len(messages) == 5

    def test_get_by_conversation_empty(self, test_session: Session, sample_user: User) -> None:
        """Test getting messages for empty conversation."""
        conv_id = self._create_conversation(test_session, sample_user)
        crud = ConversationMessageCRUD()
        messages = crud.get_by_conversation(test_session, conv_id)
        assert messages == []

    def test_get_recent_by_conversation(self, test_session: Session, sample_user: User) -> None:
        """Test getting recent messages with correct ordering."""
        conv_id = self._create_conversation(test_session, sample_user)
        crud = ConversationMessageCRUD()
        for i in range(6):
            crud.create(test_session, conversation_id=conv_id, role="user", content=f"msg {i}")
            time.sleep(0.01)  # ensure distinct timestamps
        recent = crud.get_recent_by_conversation(test_session, conv_id, limit=3)
        assert len(recent) == 3
        # Should be in chronological order (oldest first)
        assert recent[0].content == "msg 3"
        assert recent[2].content == "msg 5"

    def test_get_recent_empty(self, test_session: Session, sample_user: User) -> None:
        """Test getting recent messages from empty conversation."""
        conv_id = self._create_conversation(test_session, sample_user)
        crud = ConversationMessageCRUD()
        recent = crud.get_recent_by_conversation(test_session, conv_id)
        assert recent == []

    def test_get_recent_fewer_than_limit(self, test_session: Session, sample_user: User) -> None:
        """Test when fewer messages exist than the limit."""
        conv_id = self._create_conversation(test_session, sample_user)
        crud = ConversationMessageCRUD()
        crud.create(test_session, conversation_id=conv_id, role="user", content="only one")
        recent = crud.get_recent_by_conversation(test_session, conv_id, limit=10)
        assert len(recent) == 1
        assert recent[0].content == "only one"
