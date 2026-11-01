"""Tests for alert_delivery (LINE push of fired watchlist alerts)."""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.models.conversation import Conversation
from finance_ai.database.models.line_user_mapping import LineUserMapping
from finance_ai.database.models.user import User
from finance_ai.tools.alert_delivery import find_line_user_ids, push_alert_to_line

_MODULE = "finance_ai.tools.alert_delivery"


@pytest.fixture
def session(test_session: Session) -> Session:
    """Shared in-memory session (tests/conftest.py)."""
    return test_session


@pytest.fixture
def linked_user_id(session: Session, sample_user: User) -> str:
    """A user linked to one LINE account."""
    user = sample_user
    conversation = Conversation(user_id=user.id)
    session.add(conversation)
    session.flush()
    session.add(
        LineUserMapping(line_user_id="U123", user_id=user.id, conversation_id=conversation.id)
    )
    session.commit()
    return str(user.id)


def _settings(token: str | None) -> MagicMock:
    settings = MagicMock()
    settings.line_channel_access_token = token
    return settings


class TestFindLineUserIds:
    """Tests for find_line_user_ids."""

    def test_linked_user(self, session: Session, linked_user_id: str) -> None:
        """Returns the linked LINE id."""
        assert find_line_user_ids(session, linked_user_id) == ["U123"]

    def test_unlinked_user(self, session: Session) -> None:
        """No mapping, no ids."""
        assert not find_line_user_ids(session, "nobody")


class TestPushAlertToLine:
    """Tests for push_alert_to_line."""

    def test_pushes_to_linked_account(self, session: Session, linked_user_id: str) -> None:
        """One push with the title and plain-text content."""
        with (
            patch(f"{_MODULE}.get_settings", return_value=_settings("tok")),
            patch(f"{_MODULE}.send_line_push") as push,
        ):
            sent = push_alert_to_line(session, linked_user_id, "PTT.BK", "**ราคา** 35 บาท")
        assert sent == 1
        token, line_id, text = push.call_args.args
        assert (token, line_id) == ("tok", "U123")
        assert text.startswith("🔔 ข่าวประจำวัน PTT.BK")
        assert "**" not in text

    def test_no_token_sends_nothing(self, session: Session, linked_user_id: str) -> None:
        """Without a LINE token nothing is sent."""
        with (
            patch(f"{_MODULE}.get_settings", return_value=_settings(None)),
            patch(f"{_MODULE}.send_line_push") as push,
        ):
            assert push_alert_to_line(session, linked_user_id, "PTT.BK", "x") == 0
        push.assert_not_called()

    def test_push_failure_is_swallowed(self, session: Session, linked_user_id: str) -> None:
        """A LINE API error is logged, not raised."""
        with (
            patch(f"{_MODULE}.get_settings", return_value=_settings("tok")),
            patch(f"{_MODULE}.send_line_push", side_effect=RuntimeError("quota")),
        ):
            assert push_alert_to_line(session, linked_user_id, "PTT.BK", "x") == 0


def test_long_alert_is_cut_to_line_limit(session: Session, linked_user_id: str) -> None:
    """LINE rejects texts over 5,000 characters, so long alerts are cut."""
    from finance_ai.tools.alert_delivery import LINE_TEXT_LIMIT

    with (
        patch(f"{_MODULE}.get_settings", return_value=_settings("tok")),
        patch(f"{_MODULE}.send_line_push") as push,
    ):
        push_alert_to_line(session, linked_user_id, "PTT.BK", "ข่าว " * 3000)
    text = push.call_args.args[2]
    assert len(text) == LINE_TEXT_LIMIT
    assert text.endswith("…")
