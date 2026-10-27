"""Tests for conversation title service (auto-title, rename, delete)."""

from typing import Any

from langchain_core.messages import AIMessage
from sqlalchemy.orm import Session

from finance_ai.database.models.conversation import Conversation
from finance_ai.database.models.user import User
from finance_ai.tools.conversation_constants import (
    CONVERSATION_TITLE_MAX_LENGTH,
    DEFAULT_CONVERSATION_TITLE,
)
from finance_ai.tools.conversation_service import (
    create_conversation,
    save_assistant_message,
    save_user_message,
)
from finance_ai.tools.conversation_title_service import (
    auto_title_conversation,
    clean_generated_title,
    delete_user_conversation,
    generate_conversation_title,
    rename_conversation,
)


class FakeChatModel:
    """Chat model double that records calls and returns a fixed reply."""

    def __init__(
        self,
        reply: str | list[str | dict[str, Any]] = "วางแผนภาษีปีนี้",
        error: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.error = error
        self.calls = 0

    def invoke(self, _messages: Any) -> AIMessage:
        """Return the canned reply, or raise the configured error."""
        self.calls += 1
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.reply)


def _conversation_with_exchange(session: Session, user: User) -> Conversation:
    """Create a default-titled conversation with one user/assistant exchange."""
    conversation = create_conversation(session, user.id)
    save_user_message(session, conversation.id, "ช่วยคำนวณภาษีเงินเดือน 50,000")
    save_assistant_message(session, conversation.id, "ภาษีของคุณคือ...", "tax")
    return conversation


class TestCleanGeneratedTitle:
    """Tests for clean_generated_title."""

    def test_strips_quotes_and_whitespace(self) -> None:
        """Surrounding quotes and spaces are removed."""
        assert clean_generated_title('  "วางแผนภาษี"  ') == "วางแผนภาษี"

    def test_strips_markdown_markers(self) -> None:
        """Markdown heading and bold markers are removed."""
        assert clean_generated_title("## **วางแผนภาษี**") == "วางแผนภาษี"

    def test_keeps_first_non_empty_line(self) -> None:
        """Only the first non-empty line is kept."""
        assert clean_generated_title("\nวางแผนภาษี\nคำอธิบายเพิ่ม") == "วางแผนภาษี"

    def test_truncates_long_title(self) -> None:
        """Titles longer than the max length are truncated."""
        cleaned = clean_generated_title("ก" * 200)
        assert len(cleaned) == CONVERSATION_TITLE_MAX_LENGTH

    def test_empty_input_returns_empty(self) -> None:
        """Whitespace-only input yields an empty string."""
        assert clean_generated_title("   \n  ") == ""


class TestGenerateConversationTitle:
    """Tests for generate_conversation_title."""

    def test_returns_cleaned_llm_title(self) -> None:
        """The LLM reply is cleaned and returned."""
        model = FakeChatModel(reply='"สรุปรายจ่ายเดือนนี้"')
        title = generate_conversation_title(model, "สรุปรายจ่าย", "รายจ่ายคือ...")
        assert title == "สรุปรายจ่ายเดือนนี้"

    def test_gemini_content_blocks_are_joined_as_text(self) -> None:
        """Gemini 3 returns a list of parts; the title is their text, not the list repr."""
        model = FakeChatModel(
            reply=[{"type": "text", "text": "วางแผนการออมเงิน", "extras": {"signature": "x"}}]
        )
        title = generate_conversation_title(model, "ช่วยวางแผนการออม", "ได้เลย...")
        assert title == "วางแผนการออมเงิน"

    def test_llm_error_falls_back_to_user_message(self) -> None:
        """An LLM failure falls back to the first user message."""
        model = FakeChatModel(error=RuntimeError("rate limited"))
        title = generate_conversation_title(model, "สวัสดีครับ", "สวัสดี")
        assert title == "สวัสดีครับ"

    def test_empty_llm_reply_falls_back_to_user_message(self) -> None:
        """An empty LLM reply falls back to the first user message."""
        model = FakeChatModel(reply="   ")
        title = generate_conversation_title(model, "ดูพอร์ตลงทุน", "พอร์ตของคุณ...")
        assert title == "ดูพอร์ตลงทุน"


class TestAutoTitleConversation:
    """Tests for auto_title_conversation."""

    def test_sets_title_on_default_conversation(
        self, test_session: Session, sample_user: User
    ) -> None:
        """A default-titled conversation gets the generated title."""
        conversation = _conversation_with_exchange(test_session, sample_user)
        model = FakeChatModel(reply="ภาษีเงินเดือน")
        title = auto_title_conversation(test_session, conversation.id, sample_user.id, model)
        assert title == "ภาษีเงินเดือน"
        assert conversation.title == "ภาษีเงินเดือน"

    def test_keeps_existing_title_without_calling_llm(
        self, test_session: Session, sample_user: User
    ) -> None:
        """A named conversation is left alone and the LLM is not called."""
        conversation = _conversation_with_exchange(test_session, sample_user)
        rename_conversation(test_session, conversation.id, sample_user.id, "ชื่อของฉัน")
        model = FakeChatModel()
        title = auto_title_conversation(test_session, conversation.id, sample_user.id, model)
        assert title == "ชื่อของฉัน"
        assert model.calls == 0

    def test_no_messages_keeps_default(self, test_session: Session, sample_user: User) -> None:
        """Without a user message there is nothing to title from."""
        conversation = create_conversation(test_session, sample_user.id)
        model = FakeChatModel()
        title = auto_title_conversation(test_session, conversation.id, sample_user.id, model)
        assert title == DEFAULT_CONVERSATION_TITLE
        assert model.calls == 0

    def test_wrong_user_returns_none(self, test_session: Session, sample_user: User) -> None:
        """Another user's conversation cannot be titled."""
        conversation = _conversation_with_exchange(test_session, sample_user)
        title = auto_title_conversation(test_session, conversation.id, "other", FakeChatModel())
        assert title is None

    def test_missing_conversation_returns_none(
        self, test_session: Session, sample_user: User
    ) -> None:
        """An unknown id returns None."""
        title = auto_title_conversation(test_session, "missing", sample_user.id, FakeChatModel())
        assert title is None


class TestRenameConversation:
    """Tests for rename_conversation."""

    def test_renames_owned_conversation(self, test_session: Session, sample_user: User) -> None:
        """The owner can rename; the title is stripped."""
        conversation = create_conversation(test_session, sample_user.id)
        renamed = rename_conversation(test_session, conversation.id, sample_user.id, "  ภาษี  ")
        assert renamed is not None
        assert renamed.title == "ภาษี"

    def test_truncates_long_title(self, test_session: Session, sample_user: User) -> None:
        """Long titles are truncated to the max length."""
        conversation = create_conversation(test_session, sample_user.id)
        renamed = rename_conversation(test_session, conversation.id, sample_user.id, "ข" * 99)
        assert renamed is not None
        assert len(renamed.title) == CONVERSATION_TITLE_MAX_LENGTH

    def test_wrong_user_returns_none(self, test_session: Session, sample_user: User) -> None:
        """Another user cannot rename the conversation."""
        conversation = create_conversation(test_session, sample_user.id)
        assert rename_conversation(test_session, conversation.id, "other", "x") is None
        assert conversation.title == DEFAULT_CONVERSATION_TITLE

    def test_missing_conversation_returns_none(
        self, test_session: Session, sample_user: User
    ) -> None:
        """An unknown id returns None."""
        assert rename_conversation(test_session, "missing", sample_user.id, "x") is None


class TestDeleteUserConversation:
    """Tests for delete_user_conversation."""

    def test_deletes_conversation_and_messages(
        self, test_session: Session, sample_user: User
    ) -> None:
        """The owner can delete; messages are removed with it."""
        conversation = _conversation_with_exchange(test_session, sample_user)
        conversation_id = conversation.id
        assert delete_user_conversation(test_session, conversation_id, sample_user.id) is True
        assert test_session.get(Conversation, conversation_id) is None

    def test_wrong_user_is_rejected(self, test_session: Session, sample_user: User) -> None:
        """Another user's conversation is not deleted."""
        conversation = create_conversation(test_session, sample_user.id)
        assert delete_user_conversation(test_session, conversation.id, "other") is False
        assert test_session.get(Conversation, conversation.id) is not None

    def test_missing_conversation_returns_false(
        self, test_session: Session, sample_user: User
    ) -> None:
        """An unknown id returns False."""
        assert delete_user_conversation(test_session, "missing", sample_user.id) is False
