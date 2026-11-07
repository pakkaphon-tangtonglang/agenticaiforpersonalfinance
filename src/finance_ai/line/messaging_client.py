"""LINE Messaging API client: push answers and show the loading indicator."""

from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessageAction,
    MessagingApi,
    MessagingApiBlob,
    PushMessageRequest,
    QuickReply,
    QuickReplyItem,
    ReplyMessageRequest,
    ShowLoadingAnimationRequest,
    TextMessage,
)

from finance_ai.core.logging import get_logger
from finance_ai.line.text_split import split_line_text

logger = get_logger(__name__)

# LINE accepts 5-60 seconds in steps of 5; the animation stops early
# as soon as the answer message arrives.
LOADING_ANIMATION_SECONDS: int = 60

# Thai clarify menu mapped to tappable LINE Quick Reply buttons
CLARIFY_QUICK_REPLY_LABELS: tuple[str, ...] = (
    "บันทึกรายจ่าย",
    "วางแผน",
    "หุ้น",
    "ภาษี",
)


def send_line_push(
    access_token: str,
    line_user_id: str,
    text: str,
    quick_reply_labels: tuple[str, ...] = CLARIFY_QUICK_REPLY_LABELS,
) -> None:
    """Send a text answer to a LINE user via the Push Message API.

    Push is used instead of Reply because agent calls take 10-30s on
    ollama while reply tokens expire after ~30s.

    Args:
        access_token: LINE channel access token.
        line_user_id: LINE platform userId of the recipient.
        text: Answer text (in Thai) to deliver; a long text is split into
            several messages (see split_line_text).
        quick_reply_labels: Tappable buttons (defaults to the clarify menu),
            attached to the last message.

    Example:
        >>> send_line_push(token, "U4af4980629...", "คำตอบ")
    """
    chunks = split_line_text(text)
    messages = [TextMessage(text=chunk) for chunk in chunks[:-1]]
    messages.append(TextMessage(text=chunks[-1], quick_reply=_quick_reply(quick_reply_labels)))
    messaging_api = MessagingApi(ApiClient(Configuration(access_token=access_token)))
    messaging_api.push_message(
        push_message_request=PushMessageRequest(to=line_user_id, messages=messages)
    )


def send_line_reply(
    access_token: str,
    reply_token: str,
    text: str,
    quick_reply_labels: tuple[str, ...] = (),
) -> None:
    """Send a text with the Reply API (free, does not count toward quota).

    The reply token is single-use and expires about a minute after the
    event, so this suits instant messages such as a processing notice.

    Args:
        access_token: LINE channel access token.
        reply_token: Reply token from the webhook event.
        text: Text (in Thai) to send.
        quick_reply_labels: Optional tappable buttons (none by default).

    Example:
        >>> send_line_reply(token, event.reply_token, "ได้รับคำถามแล้ว")
    """
    quick_reply = _quick_reply(quick_reply_labels)
    messaging_api = MessagingApi(ApiClient(Configuration(access_token=access_token)))
    messaging_api.reply_message(
        reply_message_request=ReplyMessageRequest(
            reply_token=reply_token,
            messages=[TextMessage(text=text, quick_reply=quick_reply)],
        )
    )


def download_line_content(access_token: str, message_id: str) -> bytes:
    """Download the image or file a user sent (receipt, slip, statement PDF).

    Args:
        access_token: LINE channel access token.
        message_id: Id of the image/file message.

    Returns:
        The raw file bytes.

    Example:
        >>> content = download_line_content(token, event.message.id)
    """
    blob_api = MessagingApiBlob(ApiClient(Configuration(access_token=access_token)))
    return bytes(blob_api.get_message_content(message_id))


def _quick_reply(labels: tuple[str, ...]) -> QuickReply | None:
    """Build a Quick Reply menu whose buttons send their own label (None if no labels)."""
    if not labels:
        return None
    return QuickReply(
        items=[QuickReplyItem(action=MessageAction(label=label, text=label)) for label in labels]
    )


def show_line_loading(access_token: str, line_user_id: str) -> None:
    """Show LINE's "typing" loading animation in the user's chat.

    Works in 1:1 chats only. It is cosmetic, so any API failure is
    logged and swallowed rather than blocking the answer.

    Args:
        access_token: LINE channel access token.
        line_user_id: LINE platform userId of the chat to animate.

    Example:
        >>> show_line_loading(token, "U4af4980629...")
    """
    messaging_api = MessagingApi(ApiClient(Configuration(access_token=access_token)))
    try:
        messaging_api.show_loading_animation(
            show_loading_animation_request=ShowLoadingAnimationRequest(
                chat_id=line_user_id, loading_seconds=LOADING_ANIMATION_SECONDS
            )
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to show LINE loading animation: %s", error)
