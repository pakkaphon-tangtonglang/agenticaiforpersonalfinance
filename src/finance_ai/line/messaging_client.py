"""LINE Push Message API client with the Thai clarify Quick Reply menu."""

from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessageAction,
    MessagingApi,
    PushMessageRequest,
    QuickReply,
    QuickReplyItem,
    TextMessage,
)

# Thai clarify menu mapped to tappable LINE Quick Reply buttons
CLARIFY_QUICK_REPLY_LABELS: tuple[str, ...] = (
    "บันทึกรายจ่าย",
    "วางแผน",
    "หุ้น",
    "ภาษี",
)


def send_line_push(access_token: str, line_user_id: str, text: str) -> None:
    """Send a text answer to a LINE user via the Push Message API.

    Push is used instead of Reply because agent calls take 10-30s on
    ollama while reply tokens expire after ~30s.

    Args:
        access_token: LINE channel access token.
        line_user_id: LINE platform userId of the recipient.
        text: Answer text (in Thai) to deliver.

    Example:
        >>> send_line_push(token, "U4af4980629...", "คำตอบ")
    """
    messaging_api = MessagingApi(ApiClient(Configuration(access_token=access_token)))
    messaging_api.push_message(
        push_message_request=PushMessageRequest(
            to=line_user_id,
            messages=[TextMessage(text=text, quick_reply=_clarify_quick_reply())],
        )
    )


def _clarify_quick_reply() -> QuickReply:
    """Build the Quick Reply menu from the Thai clarify options."""
    return QuickReply(
        items=[
            QuickReplyItem(action=MessageAction(label=label, text=label))
            for label in CLARIFY_QUICK_REPLY_LABELS
        ]
    )
