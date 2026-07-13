"""Tests for the Conversation database model."""

from sqlalchemy.orm import Session

from finance_ai.database.models.conversation import Conversation
from finance_ai.database.models.user import User


class TestConversationModel:
    """Tests for Conversation model creation, defaults, and relationships."""

    def test_create_conversation(self, test_session: Session, sample_user: User) -> None:
        """Test creating a conversation with all fields."""
        conv = Conversation(
            user_id=sample_user.id,
            title="ทดสอบแชท",
        )
        test_session.add(conv)
        test_session.commit()
        test_session.refresh(conv)
        assert conv.id is not None
        assert conv.title == "ทดสอบแชท"
        assert conv.user_id == sample_user.id

    def test_conversation_defaults(self, test_session: Session, sample_user: User) -> None:
        """Test that defaults are correctly applied."""
        conv = Conversation(user_id=sample_user.id)
        test_session.add(conv)
        test_session.commit()
        test_session.refresh(conv)
        assert conv.title == "แชทใหม่"
        assert conv.is_active is True

    def test_conversation_relationship_to_user(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test that conversation links back to its user."""
        conv = Conversation(user_id=sample_user.id, title="Test")
        test_session.add(conv)
        test_session.commit()
        test_session.refresh(conv)
        assert conv.user.id == sample_user.id
        assert conv in sample_user.conversations

    def test_conversation_timestamps(self, test_session: Session, sample_user: User) -> None:
        """Test that created_at and updated_at are set."""
        conv = Conversation(user_id=sample_user.id)
        test_session.add(conv)
        test_session.commit()
        test_session.refresh(conv)
        assert conv.created_at is not None
        assert conv.updated_at is not None

    def test_multiple_conversations_for_user(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test creating multiple conversations for one user."""
        for i in range(3):
            conv = Conversation(user_id=sample_user.id, title=f"แชท {i}")
            test_session.add(conv)
        test_session.commit()
        assert len(sample_user.conversations) == 3

    def test_soft_delete_with_is_active(self, test_session: Session, sample_user: User) -> None:
        """Test soft-deleting a conversation by setting is_active=False."""
        conv = Conversation(user_id=sample_user.id)
        test_session.add(conv)
        test_session.commit()
        conv.is_active = False
        test_session.commit()
        test_session.refresh(conv)
        assert conv.is_active is False
