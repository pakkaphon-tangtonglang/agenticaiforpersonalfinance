"""Tests for the LINE bot service (event -> agent -> reply flow)."""

from typing import Any
from unittest.mock import MagicMock

from sqlalchemy.orm import Session, sessionmaker

from finance_ai.database.models.user import User
from finance_ai.line.line_bot_service import process_line_message
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
