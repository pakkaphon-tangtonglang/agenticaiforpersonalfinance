"""Tests for the ConversationMessage database model."""

from sqlalchemy.orm import Session

from finance_ai.database.models.conversation import Conversation
from finance_ai.database.models.conversation_message import ConversationMessage
from finance_ai.database.models.user import User


class TestConversationMessageModel:
    """Tests for ConversationMessage model creation and relationships."""

    def test_create_user_message(self, test_session: Session, sample_user: User) -> None:
        """Test creating a user message."""
        conv = Conversation(user_id=sample_user.id)
        test_session.add(conv)
        test_session.commit()
        msg = ConversationMessage(
            conversation_id=conv.id,
            role="user",
            content="คำนวณภาษี",
        )
        test_session.add(msg)
        test_session.commit()
        test_session.refresh(msg)
        assert msg.id is not None
        assert msg.role == "user"
        assert msg.content == "คำนวณภาษี"
        assert msg.intent is None

    def test_create_assistant_message_with_intent(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test creating an assistant message with intent."""
        conv = Conversation(user_id=sample_user.id)
        test_session.add(conv)
        test_session.commit()
        msg = ConversationMessage(
            conversation_id=conv.id,
            role="assistant",
            content="ผลคำนวณภาษี...",
            intent="tax",
        )
        test_session.add(msg)
        test_session.commit()
        test_session.refresh(msg)
        assert msg.role == "assistant"
        assert msg.intent == "tax"

    def test_message_relationship_to_conversation(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test that message links back to its conversation."""
        conv = Conversation(user_id=sample_user.id)
        test_session.add(conv)
        test_session.commit()
        msg = ConversationMessage(conversation_id=conv.id, role="user", content="test")
        test_session.add(msg)
        test_session.commit()
        test_session.refresh(msg)
        assert msg.conversation.id == conv.id
        assert msg in conv.messages

    def test_message_timestamps(self, test_session: Session, sample_user: User) -> None:
        """Test that timestamps are set on message creation."""
        conv = Conversation(user_id=sample_user.id)
        test_session.add(conv)
        test_session.commit()
        msg = ConversationMessage(conversation_id=conv.id, role="user", content="test")
        test_session.add(msg)
        test_session.commit()
        test_session.refresh(msg)
        assert msg.created_at is not None
        assert msg.updated_at is not None

    def test_cascade_delete_messages(self, test_session: Session, sample_user: User) -> None:
        """Test that deleting conversation cascades to messages."""
        conv = Conversation(user_id=sample_user.id)
        test_session.add(conv)
        test_session.commit()
        for i in range(3):
            msg = ConversationMessage(conversation_id=conv.id, role="user", content=f"msg {i}")
            test_session.add(msg)
        test_session.commit()
        assert len(conv.messages) == 3
        test_session.delete(conv)
        test_session.commit()
        remaining = test_session.query(ConversationMessage).all()
        assert len(remaining) == 0

    def test_multiple_messages_in_conversation(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test adding multiple messages to a conversation."""
        conv = Conversation(user_id=sample_user.id)
        test_session.add(conv)
        test_session.commit()
        messages_data = [
            ("user", "คำนวณภาษี"),
            ("assistant", "ผลคำนวณ..."),
            ("user", "ขอบคุณ"),
        ]
        for role, content in messages_data:
            msg = ConversationMessage(conversation_id=conv.id, role=role, content=content)
            test_session.add(msg)
        test_session.commit()
        assert len(conv.messages) == 3
