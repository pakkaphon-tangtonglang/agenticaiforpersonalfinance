"""Pydantic models for LINE Messaging API webhook payloads."""

from pydantic import BaseModel, Field, ValidationError


class LineEventSource(BaseModel):
    """The `source` object of a LINE webhook event."""

    type: str = ""
    user_id: str = Field(default="", alias="userId")

    model_config = {"populate_by_name": True}


class LineEventMessage(BaseModel):
    """The `message` object of a LINE message event."""

    type: str = ""
    text: str = ""


class LineEvent(BaseModel):
    """A single event from the LINE webhook `events` array."""

    type: str = ""
    reply_token: str = Field(default="", alias="replyToken")
    source: LineEventSource = Field(default_factory=LineEventSource)
    message: LineEventMessage = Field(default_factory=LineEventMessage)

    model_config = {"populate_by_name": True}

    def is_text_message(self) -> bool:
        """Return True when the event is a user text message.

        Example:
            >>> event.is_text_message()
            True
        """
        return self.type == "message" and self.message.type == "text"


class LineWebhookBody(BaseModel):
    """The full LINE webhook request body."""

    destination: str = ""
    events: list[LineEvent] = Field(default_factory=list)

    def text_message_events(self) -> list[LineEvent]:
        """Return only events the agent should process (user text messages).

        Example:
            >>> body.text_message_events()
            [LineEvent(...)]
        """
        return [event for event in self.events if event.is_text_message()]


def parse_line_webhook_body(raw_body: str) -> LineWebhookBody:
    """Parse a raw webhook body into a LineWebhookBody.

    Args:
        raw_body: Raw JSON request body string.

    Returns:
        LineWebhookBody parsed from the JSON.

    Raises:
        ValidationError: When the body is not valid JSON.

    Example:
        >>> parse_line_webhook_body('{"events": []}')
        LineWebhookBody(destination='', events=[])
    """
    try:
        return LineWebhookBody.model_validate_json(raw_body)
    except ValidationError as exc:
        raise ValueError("Invalid LINE webhook payload") from exc
