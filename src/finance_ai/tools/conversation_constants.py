"""Constants for conversation history and memory management.

Contains limits, defaults, and validation values for
conversation persistence and chat history retrieval.
"""

# Maximum number of recent messages to send as LLM context
MAX_HISTORY_MESSAGES: int = 10

# Default title for new conversations
DEFAULT_CONVERSATION_TITLE: str = "แชทใหม่"

# Maximum length for conversation title (auto-generated from first message)
CONVERSATION_TITLE_MAX_LENGTH: int = 50

# Valid message roles
VALID_MESSAGE_ROLES: tuple[str, ...] = ("user", "assistant")

# Maximum conversations stored per user
MAX_CONVERSATIONS_PER_USER: int = 50
