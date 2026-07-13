"""Tests for conversation service layer."""

from typing import Any

from sqlalchemy.orm import Session

from finance_ai.database.crud.conversation_crud import ConversationCRUD
from finance_ai.database.crud.conversation_message_crud import ConversationMessageCRUD
from finance_ai.database.models.user import User
from finance_ai.tools.conversation_constants import (
    CONVERSATION_TITLE_MAX_LENGTH,
    DEFAULT_CONVERSATION_TITLE,
)
from finance_ai.tools.conversation_service import (
    create_conversation,
    get_or_create_active_conversation,
    get_recent_history_as_tuples,
    list_user_conversations,
    load_conversation_messages,
    save_assistant_message,
    save_user_message,
    update_conversation_title,
)


class TestCreateConversation:
    """Tests for create_conversation function."""

    def test_creates_with_default_title(self, test_session: Session, sample_user: User) -> None:
        """Test creating a conversation with default title."""
        conv = create_conversation(test_session, sample_user.id)
        assert conv.title == DEFAULT_CONVERSATION_TITLE
        assert conv.user_id == sample_user.id

    def test_creates_with_custom_title(self, test_session: Session, sample_user: User) -> None:
        """Test creating a conversation with custom title."""
        conv = create_conversation(test_session, sample_user.id, title="ภาษี 2026")
        assert conv.title == "ภาษี 2026"


class TestSaveMessages:
    """Tests for save_user_message and save_assistant_message."""

    def _make_conversation(self, session: Session, user: User) -> str:
        """Helper to create a conversation."""
        conv = create_conversation(session, user.id)
        return conv.id

    def test_save_user_message(self, test_session: Session, sample_user: User) -> None:
        """Test saving a user message."""
        conv_id = self._make_conversation(test_session, sample_user)
        msg = save_user_message(test_session, conv_id, "คำนวณภาษี")
        assert msg.role == "user"
        assert msg.content == "คำนวณภาษี"
        assert msg.intent is None

    def test_save_assistant_message(self, test_session: Session, sample_user: User) -> None:
        """Test saving an assistant message with intent."""
        conv_id = self._make_conversation(test_session, sample_user)
        msg = save_assistant_message(test_session, conv_id, "ผลลัพธ์...", "tax")
        assert msg.role == "assistant"
        assert msg.content == "ผลลัพธ์..."
        assert msg.intent == "tax"

    def test_save_assistant_message_no_intent(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test saving an assistant message without intent."""
        conv_id = self._make_conversation(test_session, sample_user)
        msg = save_assistant_message(test_session, conv_id, "ผลลัพธ์...")
        assert msg.intent is None


class TestGetRecentHistoryAsTuples:
    """Tests for get_recent_history_as_tuples function."""

    def _setup_conversation_with_messages(self, session: Session, user: User, count: int) -> str:
        """Helper to create conversation with N messages."""
        conv_id = create_conversation(session, user.id).id
        for i in range(count):
            role = "user" if i % 2 == 0 else "assistant"
            (
                save_user_message(session, conv_id, f"msg {i}")
                if role == "user"
                else save_assistant_message(session, conv_id, f"msg {i}")
            )
        return conv_id

    def test_returns_tuples(self, test_session: Session, sample_user: User) -> None:
        """Test that history is returned as (role, content) tuples."""
        conv_id = create_conversation(test_session, sample_user.id).id
        save_user_message(test_session, conv_id, "สวัสดี")
        save_assistant_message(test_session, conv_id, "สวัสดีครับ")
        history = get_recent_history_as_tuples(test_session, conv_id)
        assert len(history) == 2
        assert history[0] == ("user", "สวัสดี")
        assert history[1] == ("assistant", "สวัสดีครับ")

    def test_respects_limit(self, test_session: Session, sample_user: User) -> None:
        """Test that limit restricts the number of messages."""
        conv_id = self._setup_conversation_with_messages(test_session, sample_user, 8)
        history = get_recent_history_as_tuples(test_session, conv_id, limit=4)
        assert len(history) == 4

    def test_empty_conversation(self, test_session: Session, sample_user: User) -> None:
        """Test with no messages."""
        conv_id = create_conversation(test_session, sample_user.id).id
        history = get_recent_history_as_tuples(test_session, conv_id)
        assert history == []


class TestLoadConversationMessages:
    """Tests for load_conversation_messages function."""

    def test_returns_streamlit_format(self, test_session: Session, sample_user: User) -> None:
        """Test that messages are in Streamlit-compatible format."""
        conv_id = create_conversation(test_session, sample_user.id).id
        save_user_message(test_session, conv_id, "ถาม")
        save_assistant_message(test_session, conv_id, "ตอบ", "tax")
        messages: list[dict[str, Any]] = load_conversation_messages(test_session, conv_id)
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "ถาม"}
        assert messages[1] == {"role": "assistant", "content": "ตอบ", "intent": "tax"}

    def test_excludes_none_intent(self, test_session: Session, sample_user: User) -> None:
        """Test that None intent is not included in dict."""
        conv_id = create_conversation(test_session, sample_user.id).id
        save_assistant_message(test_session, conv_id, "ตอบ")
        messages = load_conversation_messages(test_session, conv_id)
        assert "intent" not in messages[0]

    def test_empty_conversation(self, test_session: Session, sample_user: User) -> None:
        """Test loading from empty conversation."""
        conv_id = create_conversation(test_session, sample_user.id).id
        messages = load_conversation_messages(test_session, conv_id)
        assert messages == []


class TestGetOrCreateActiveConversation:
    """Tests for get_or_create_active_conversation."""

    def test_creates_new_when_none_exists(self, test_session: Session, sample_user: User) -> None:
        """Test that a new conversation is created if none exist."""
        conv = get_or_create_active_conversation(test_session, sample_user.id)
        assert conv is not None
        assert conv.user_id == sample_user.id
        assert conv.title == DEFAULT_CONVERSATION_TITLE

    def test_returns_existing_conversation(self, test_session: Session, sample_user: User) -> None:
        """Test that existing active conversation is returned."""
        existing = create_conversation(test_session, sample_user.id, "existing")
        result = get_or_create_active_conversation(test_session, sample_user.id)
        assert result.id == existing.id


class TestUpdateConversationTitle:
    """Tests for update_conversation_title."""

    def test_updates_title(self, test_session: Session, sample_user: User) -> None:
        """Test updating conversation title."""
        conv = create_conversation(test_session, sample_user.id)
        result = update_conversation_title(test_session, conv.id, "ภาษี 2026")
        assert result is not None
        assert result.title == "ภาษี 2026"

    def test_truncates_long_title(self, test_session: Session, sample_user: User) -> None:
        """Test that long titles are truncated."""
        conv = create_conversation(test_session, sample_user.id)
        long_title = "ก" * 100
        result = update_conversation_title(test_session, conv.id, long_title)
        assert result is not None
        assert len(result.title) == CONVERSATION_TITLE_MAX_LENGTH

    def test_returns_none_for_nonexistent(self, test_session: Session) -> None:
        """Test updating nonexistent conversation returns None."""
        result = update_conversation_title(test_session, "fake-id", "test")
        assert result is None


class TestListUserConversations:
    """Tests for list_user_conversations."""

    def test_lists_active_conversations(self, test_session: Session, sample_user: User) -> None:
        """Test listing active conversations."""
        create_conversation(test_session, sample_user.id, "Conv 1")
        create_conversation(test_session, sample_user.id, "Conv 2")
        convs = list_user_conversations(test_session, sample_user.id)
        assert len(convs) == 2

    def test_excludes_inactive(self, test_session: Session, sample_user: User) -> None:
        """Test that inactive conversations are excluded."""
        create_conversation(test_session, sample_user.id, "Active")
        inactive = create_conversation(test_session, sample_user.id, "Inactive")
        inactive.is_active = False
        test_session.commit()
        convs = list_user_conversations(test_session, sample_user.id)
        assert len(convs) == 1

    def test_respects_limit(self, test_session: Session, sample_user: User) -> None:
        """Test that limit parameter works."""
        for i in range(10):
            create_conversation(test_session, sample_user.id, f"Conv {i}")
        convs = list_user_conversations(test_session, sample_user.id, limit=5)
        assert len(convs) == 5

    def test_empty_user(self, test_session: Session, sample_user: User) -> None:
        """Test listing for user with no conversations."""
        convs = list_user_conversations(test_session, sample_user.id)
        assert convs == []
