"""Tests for conversation constants."""

from finance_ai.tools.conversation_constants import (
    CONVERSATION_TITLE_MAX_LENGTH,
    DEFAULT_CONVERSATION_TITLE,
    MAX_CONVERSATIONS_PER_USER,
    MAX_HISTORY_MESSAGES,
    VALID_MESSAGE_ROLES,
)


class TestConversationConstants:
    """Tests for conversation constant values."""

    def test_max_history_messages_is_positive(self) -> None:
        """MAX_HISTORY_MESSAGES should be a positive integer."""
        assert isinstance(MAX_HISTORY_MESSAGES, int)
        assert MAX_HISTORY_MESSAGES > 0

    def test_max_history_messages_value(self) -> None:
        """MAX_HISTORY_MESSAGES should be 10."""
        assert MAX_HISTORY_MESSAGES == 10

    def test_default_conversation_title(self) -> None:
        """DEFAULT_CONVERSATION_TITLE should be Thai text."""
        assert isinstance(DEFAULT_CONVERSATION_TITLE, str)
        assert len(DEFAULT_CONVERSATION_TITLE) > 0

    def test_title_max_length_is_positive(self) -> None:
        """CONVERSATION_TITLE_MAX_LENGTH should be positive."""
        assert isinstance(CONVERSATION_TITLE_MAX_LENGTH, int)
        assert CONVERSATION_TITLE_MAX_LENGTH > 0

    def test_valid_message_roles_contains_user_and_assistant(self) -> None:
        """VALID_MESSAGE_ROLES must include 'user' and 'assistant'."""
        assert "user" in VALID_MESSAGE_ROLES
        assert "assistant" in VALID_MESSAGE_ROLES

    def test_valid_message_roles_is_tuple(self) -> None:
        """VALID_MESSAGE_ROLES should be a tuple (immutable)."""
        assert isinstance(VALID_MESSAGE_ROLES, tuple)

    def test_max_conversations_per_user_is_positive(self) -> None:
        """MAX_CONVERSATIONS_PER_USER should be positive."""
        assert isinstance(MAX_CONVERSATIONS_PER_USER, int)
        assert MAX_CONVERSATIONS_PER_USER > 0

    def test_max_conversations_per_user_value(self) -> None:
        """MAX_CONVERSATIONS_PER_USER should be 50."""
        assert MAX_CONVERSATIONS_PER_USER == 50
