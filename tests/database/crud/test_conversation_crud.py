"""Tests for ConversationCRUD operations."""

from sqlalchemy.orm import Session

from finance_ai.database.crud.conversation_crud import ConversationCRUD
from finance_ai.database.models.user import User


class TestConversationCRUD:
    """Tests for Conversation-specific CRUD operations."""

    def test_create_conversation(self, test_session: Session, sample_user: User) -> None:
        """Test creating a conversation."""
        crud = ConversationCRUD()
        conv = crud.create(test_session, user_id=sample_user.id, title="ทดสอบ")
        assert conv.id is not None
        assert conv.title == "ทดสอบ"
        assert conv.user_id == sample_user.id

    def test_get_by_user(self, test_session: Session, sample_user: User) -> None:
        """Test getting all conversations for a user."""
        crud = ConversationCRUD()
        crud.create(test_session, user_id=sample_user.id, title="แชท 1")
        crud.create(test_session, user_id=sample_user.id, title="แชท 2")
        conversations = crud.get_by_user(test_session, sample_user.id)
        assert len(conversations) == 2

    def test_get_by_user_respects_limit(self, test_session: Session, sample_user: User) -> None:
        """Test that limit parameter is respected."""
        crud = ConversationCRUD()
        for i in range(5):
            crud.create(test_session, user_id=sample_user.id, title=f"แชท {i}")
        conversations = crud.get_by_user(test_session, sample_user.id, limit=3)
        assert len(conversations) == 3

    def test_get_by_user_empty(self, test_session: Session, sample_user: User) -> None:
        """Test getting conversations when none exist."""
        crud = ConversationCRUD()
        conversations = crud.get_by_user(test_session, sample_user.id)
        assert conversations == []

    def test_get_active_by_user(self, test_session: Session, sample_user: User) -> None:
        """Test filtering out inactive conversations."""
        crud = ConversationCRUD()
        crud.create(test_session, user_id=sample_user.id, title="Active")
        inactive = crud.create(test_session, user_id=sample_user.id, title="Inactive")
        inactive.is_active = False
        test_session.commit()
        active = crud.get_active_by_user(test_session, sample_user.id)
        assert len(active) == 1
        assert active[0].title == "Active"

    def test_get_latest_by_user(self, test_session: Session, sample_user: User) -> None:
        """Test getting the most recently updated conversation."""
        crud = ConversationCRUD()
        crud.create(test_session, user_id=sample_user.id, title="แชท 1")
        crud.create(test_session, user_id=sample_user.id, title="แชท 2")
        latest = crud.get_latest_by_user(test_session, sample_user.id)
        assert latest is not None
        assert latest.title == "แชท 2"

    def test_get_latest_by_user_empty(self, test_session: Session, sample_user: User) -> None:
        """Test getting latest when no conversations exist."""
        crud = ConversationCRUD()
        latest = crud.get_latest_by_user(test_session, sample_user.id)
        assert latest is None

    def test_get_latest_skips_inactive(self, test_session: Session, sample_user: User) -> None:
        """Test that latest only returns active conversations."""
        crud = ConversationCRUD()
        conv = crud.create(test_session, user_id=sample_user.id, title="Only one")
        conv.is_active = False
        test_session.commit()
        latest = crud.get_latest_by_user(test_session, sample_user.id)
        assert latest is None
