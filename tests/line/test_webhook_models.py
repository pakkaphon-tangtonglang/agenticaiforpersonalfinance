"""Tests for LINE webhook payload parsing."""

import pytest

from finance_ai.line.webhook_models import parse_line_webhook_body

TEXT_MESSAGE_BODY = """
{
  "destination": "Udestination",
  "events": [
    {
      "type": "message",
      "replyToken": "token-1",
      "source": {"type": "user", "userId": "Uline-user-1"},
      "message": {"type": "text", "text": "ภาษีของฉันเท่าไหร่"}
    },
    {
      "type": "follow",
      "replyToken": "token-2",
      "source": {"type": "user", "userId": "Uline-user-1"}
    },
    {
      "type": "message",
      "replyToken": "token-3",
      "source": {"type": "user", "userId": "Uline-user-2"},
      "message": {"type": "sticker", "stickerId": "1", "packageId": "1"}
    }
  ]
}
"""


class TestParseLineWebhookBody:
    """Tests for parse_line_webhook_body."""

    def test_parses_destination_and_events(self) -> None:
        """Destination and event count are parsed from the raw body."""
        body = parse_line_webhook_body(TEXT_MESSAGE_BODY)
        assert body.destination == "Udestination"
        assert len(body.events) == 3

    def test_text_message_events_filters_non_text(self) -> None:
        """Only message/text events are returned for agent processing."""
        body = parse_line_webhook_body(TEXT_MESSAGE_BODY)
        text_events = body.text_message_events()
        assert len(text_events) == 1
        assert text_events[0].reply_token == "token-1"
        assert text_events[0].source.user_id == "Uline-user-1"
        assert text_events[0].message.text == "ภาษีของฉันเท่าไหร่"

    def test_empty_body_parses_to_no_events(self) -> None:
        """An empty events list produces no text message events."""
        body = parse_line_webhook_body('{"destination": "Ux", "events": []}')
        assert body.text_message_events() == []

    def test_invalid_json_raises_validation_error(self) -> None:
        """Malformed JSON is rejected with a validation error."""
        with pytest.raises(ValueError):
            parse_line_webhook_body("not-json")
