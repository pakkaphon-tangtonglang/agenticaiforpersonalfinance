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

# System prompt for the LLM that names a conversation after its first exchange
CONVERSATION_TITLE_PROMPT: str = (
    "ตั้งชื่อบทสนทนานี้ให้สั้นและสื่อความหมาย สำหรับแสดงในรายการประวัติแชท\n"
    "\n"
    "กฎ:\n"
    "- ภาษาไทย ไม่เกิน 6 คำ หรือ 30 ตัวอักษร (คำศัพท์เฉพาะใช้ภาษาอังกฤษได้)\n"
    "- สรุปหัวข้อหลักที่ผู้ใช้ถาม เช่น 'คำนวณภาษีเงินเดือน' หรือ 'วางแผนออมซื้อรถ'\n"
    "- ตอบเฉพาะชื่อเท่านั้น ห้ามมีเครื่องหมายคำพูด คำอธิบาย หรือขึ้นบรรทัดใหม่\n"
)
