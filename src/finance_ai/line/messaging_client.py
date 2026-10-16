"""LINE Messaging API client: push answers and show the loading indicator."""

from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessageAction,
    MessagingApi,
    PushMessageRequest,
    QuickReply,
    QuickReplyItem,
    ReplyMessageRequest,
    ShowLoadingAnimationRequest,
    TextMessage,
)

from finance_ai.core.logging import get_logger

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


def send_line_reply(access_token: str, reply_token: str, text: str) -> None:
    """Send a text with the Reply API (free, does not count toward quota).

    The reply token is single-use and expires about a minute after the
    event, so this suits instant messages such as a processing notice.

    Args:
        access_token: LINE channel access token.
        reply_token: Reply token from the webhook event.
        text: Text (in Thai) to send.

    Example:
        >>> send_line_reply(token, event.reply_token, "ได้รับคำถามแล้ว")
    """
    messaging_api = MessagingApi(ApiClient(Configuration(access_token=access_token)))
    messaging_api.reply_message(
        reply_message_request=ReplyMessageRequest(
            reply_token=reply_token, messages=[TextMessage(text=text)]
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
