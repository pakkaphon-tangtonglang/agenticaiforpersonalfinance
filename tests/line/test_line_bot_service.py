"""Tests for the LINE bot service (event -> agent -> reply flow)."""

from typing import Any
from unittest.mock import MagicMock, patch

from sqlalchemy.orm import Session, sessionmaker

from finance_ai.database.models.user import User
from finance_ai.line.line_bot_service import handle_line_event, process_line_message
from finance_ai.line.mapping_service import get_or_create_line_mapping

LINE_USER_ID = "Uline-user-xyz"


class TestProcessLineMessage:
    """Tests for process_line_message."""

    def test_returns_agent_response_and_saves_messages(
        self,
        test_session: Session,
        monkeypatch: Any,
    ) -> None:
        """Agent output is returned and stored in the mapped conversation."""
        mapping = get_or_create_line_mapping(test_session, LINE_USER_ID)
        captured: dict[str, Any] = {}

        def fake_orchestrate(**kwargs: Any) -> dict[str, str]:
            captured["query"] = kwargs["query"]
            captured["user_id"] = kwargs["user_id"]
            return {"intent": "tax", "response": "คำตอบภาษี"}

        monkeypatch.setattr("finance_ai.line.line_bot_service.orchestrate_query", fake_orchestrate)
        factory = sessionmaker(bind=test_session.get_bind())
        reply = process_line_message(factory, None, LINE_USER_ID, "ภาษีของฉัน")
        assert reply == "คำตอบภาษี"
        assert captured["query"] == "ภาษีของฉัน"
        assert captured["user_id"] == mapping.user_id

    def test_chat_model_provider_is_called_lazily(
        self,
        test_session: Session,
        monkeypatch: Any,
    ) -> None:
        """The chat model provider callable resolves the model at call time."""
        get_or_create_line_mapping(test_session, LINE_USER_ID)
        provider = MagicMock(return_value="fake-model")

        def fake_orchestrate(**kwargs: Any) -> dict[str, str]:
            assert kwargs["chat_model"] == "fake-model"
            return {"intent": "other", "response": "ok"}

        monkeypatch.setattr("finance_ai.line.line_bot_service.orchestrate_query", fake_orchestrate)
        factory = sessionmaker(bind=test_session.get_bind())
        process_line_message(factory, provider, LINE_USER_ID, "สวัสดี")
        provider.assert_called_once()

    def test_link_command_short_circuits_agent(
        self,
        test_session: Session,
        monkeypatch: Any,
    ) -> None:
        """A link command replies directly without invoking any agent."""
        get_or_create_line_mapping(test_session, LINE_USER_ID)
        target_user_id = "00000000-de20-4000-8000-000000000001"
        test_session.add(
            User(
                id=target_user_id,
                email="web@finance-ai.local",
                hashed_password="not-a-login",
                full_name="ผู้ใช้เว็บ",
            )
        )
        test_session.commit()

        def fail_orchestrate(**kwargs: Any) -> dict[str, str]:
            raise AssertionError("agent must not run for link commands")

        monkeypatch.setattr("finance_ai.line.line_bot_service.orchestrate_query", fail_orchestrate)
        factory = sessionmaker(bind=test_session.get_bind())
        reply = process_line_message(factory, None, LINE_USER_ID, f"เชื่อมต่อ {target_user_id}")
        assert "เชื่อมต่อบัญชีเรียบร้อย" in reply

    def test_unlink_command_short_circuits_agent(
        self,
        test_session: Session,
        monkeypatch: Any,
    ) -> None:
        """An unlink command replies directly without invoking any agent."""
        get_or_create_line_mapping(test_session, LINE_USER_ID)

        def fail_orchestrate(**kwargs: Any) -> dict[str, str]:
            raise AssertionError("agent must not run for unlink commands")

        monkeypatch.setattr("finance_ai.line.line_bot_service.orchestrate_query", fail_orchestrate)
        factory = sessionmaker(bind=test_session.get_bind())
        reply = process_line_message(factory, None, LINE_USER_ID, "ยกเลิกเชื่อมต่อ")
        assert "ยกเลิกการเชื่อมต่อเรียบร้อย" in reply


class TestHandleLineEvent:
    """Tests for the webhook background task (agent -> reply)."""

    def test_pushes_converted_reply_without_ack(self) -> None:
        """Only the converted reply is pushed — no processing ack.

        The ack was removed because the duplicate waiting message
        annoyed users more than the short silence did.
        """
        markdown_reply = (
            "## ราคาหุ้น PTT\n\n" + "| **ราคาปัจจุบัน** | 42.00 บาท |\n\n" + "- **ชื่อ**: PTT\n"
        )
        with (
            patch(
                "finance_ai.line.line_bot_service.process_line_message",
                return_value=markdown_reply,
            ),
            patch("finance_ai.line.line_bot_service.send_line_push") as mock_push,
        ):
            handle_line_event(
                line_user_id=LINE_USER_ID,
                text="ราคา PTT",
                session_factory=MagicMock(),
                chat_model_provider=None,
                access_token="token-123",
            )

        assert mock_push.call_count == 1
        pushed_text = mock_push.call_args[0][2]
        assert pushed_text == ("ราคาหุ้น PTT\n\nราคาปัจจุบัน | 42.00 บาท\n\n• ชื่อ: PTT")
        assert "##" not in pushed_text
        assert "**" not in pushed_text

    def test_link_command_pushes_reply(self) -> None:
        """Link commands push the instant reply like any other message."""
        link_reply = "เชื่อมต่อบัญชีเรียบร้อย"
        with (
            patch(
                "finance_ai.line.line_bot_service.process_line_message",
                return_value=link_reply,
            ),
            patch("finance_ai.line.line_bot_service.send_line_push") as mock_push,
        ):
            handle_line_event(
                line_user_id=LINE_USER_ID,
                text="เชื่อมต่อ 00000000-1111-2222-3333-444444444444",
                session_factory=MagicMock(),
                chat_model_provider=None,
                access_token="token-123",
            )

        assert mock_push.call_count == 1
        assert mock_push.call_args[0][2] == link_reply

    def test_pushes_error_reply_on_agent_failure(self) -> None:
        """An agent exception still pushes a Thai error reply."""
        with (
            patch(
                "finance_ai.line.line_bot_service.process_line_message",
                side_effect=RuntimeError("boom"),
            ),
            patch("finance_ai.line.line_bot_service.send_line_push") as mock_push,
        ):
            handle_line_event(
                line_user_id=LINE_USER_ID,
                text="ราคา PTT",
                session_factory=MagicMock(),
                chat_model_provider=None,
                access_token="token-123",
            )

        assert mock_push.call_count == 1
        pushed_text = mock_push.call_args[0][2]
        assert "ขออภัย" in pushed_text
        assert "boom" in pushed_text
