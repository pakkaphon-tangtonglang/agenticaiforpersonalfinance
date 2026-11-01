"""Tests for the LINE bot service (event -> agent -> reply flow)."""

from typing import Any
from unittest.mock import MagicMock, patch

from sqlalchemy.orm import Session, sessionmaker

from finance_ai.database.models.user import User
from finance_ai.line.import_conversation import LineReply
from finance_ai.line.line_bot_service import (
    DOCUMENT_NOTICE,
    LINE_ERROR_REPLY,
    PROCESSING_NOTICE,
    handle_line_document_event,
    handle_line_event,
    process_line_message,
)
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
            patch("finance_ai.line.line_bot_service.show_line_loading"),
            patch("finance_ai.line.line_bot_service.send_line_reply"),
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
            patch("finance_ai.line.line_bot_service.show_line_loading"),
            patch("finance_ai.line.line_bot_service.send_line_reply"),
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
            patch("finance_ai.line.line_bot_service.show_line_loading"),
            patch("finance_ai.line.line_bot_service.send_line_reply"),
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
        assert "boom" not in pushed_text  # internal details (SQL, ids) stay in logs

    def _run_with_manager(self, text: str, reply_token: str) -> MagicMock:
        """Run handle_line_event with all LINE calls recorded on one mock."""
        manager = MagicMock()
        manager.process.return_value = "คำตอบ"
        with (
            patch("finance_ai.line.line_bot_service.send_line_reply", manager.notice),
            patch("finance_ai.line.line_bot_service.show_line_loading", manager.loading),
            patch("finance_ai.line.line_bot_service.process_line_message", manager.process),
            patch("finance_ai.line.line_bot_service.send_line_push", manager.push),
        ):
            handle_line_event(
                line_user_id=LINE_USER_ID,
                text=text,
                session_factory=MagicMock(),
                chat_model_provider=None,
                access_token="token-123",
                reply_token=reply_token,
            )
        return manager

    def test_notice_then_loading_then_answer(self) -> None:
        """The notice goes first, since any bot message hides the loading animation."""
        manager = self._run_with_manager("ราคา PTT", "reply-token-1")
        manager.notice.assert_called_once_with("token-123", "reply-token-1", PROCESSING_NOTICE)
        manager.loading.assert_called_once_with("token-123", LINE_USER_ID)
        called_in_order = [name for name, _, _ in manager.mock_calls]
        assert called_in_order == ["notice", "loading", "process", "push"]

    def test_notice_mentions_wait_time(self) -> None:
        """The notice tells the user roughly how long to wait."""
        assert "10-15 วินาที" in PROCESSING_NOTICE

    def test_no_notice_for_link_command(self) -> None:
        """Link commands answer instantly, so no notice is sent."""
        manager = self._run_with_manager(
            "เชื่อมต่อ 00000000-1111-2222-3333-444444444444", "reply-token-1"
        )
        manager.notice.assert_not_called()

    def test_no_notice_without_reply_token(self) -> None:
        """Without a reply token there is nothing to reply to, so no notice."""
        manager = self._run_with_manager("ราคา PTT", "")
        manager.notice.assert_not_called()
        manager.loading.assert_called_once()

    def test_notice_failure_does_not_block_answer(self) -> None:
        """A failed notice is logged and the answer is still pushed."""
        with (
            patch(
                "finance_ai.line.line_bot_service.send_line_reply",
                side_effect=ConnectionError("LINE API unreachable"),
            ),
            patch("finance_ai.line.line_bot_service.show_line_loading"),
            patch("finance_ai.line.line_bot_service.process_line_message", return_value="คำตอบ"),
            patch("finance_ai.line.line_bot_service.send_line_push") as mock_push,
        ):
            handle_line_event(
                line_user_id=LINE_USER_ID,
                text="ราคา PTT",
                session_factory=MagicMock(),
                chat_model_provider=None,
                access_token="token-123",
                reply_token="reply-token-1",
            )
        assert mock_push.call_args[0][2] == "คำตอบ"


SERVICE = "finance_ai.line.line_bot_service"


class TestPendingImportText:
    """Text that answers a waiting import never reaches the agent or chat history."""

    def test_pending_reply_pushed_instead_of_agent(self) -> None:
        """A password/confirm reply is pushed with its buttons; no notice, no agent."""
        reply = LineReply("📄 อ่านได้ 1 รายการ", ("บันทึก", "ยกเลิก"))
        with (
            patch(f"{SERVICE}.continue_pending_import", return_value=reply),
            patch(f"{SERVICE}.process_line_message") as mock_process,
            patch(f"{SERVICE}.send_line_push") as mock_push,
            patch(f"{SERVICE}.send_line_reply") as mock_notice,
            patch(f"{SERVICE}.show_line_loading"),
        ):
            handle_line_event(
                line_user_id=LINE_USER_ID,
                text="19900101",
                session_factory=MagicMock(),
                chat_model_provider=None,
                access_token="token-123",
                reply_token="rt",
            )
        mock_process.assert_not_called()
        mock_notice.assert_not_called()
        mock_push.assert_called_once_with(
            "token-123",
            LINE_USER_ID,
            "📄 อ่านได้ 1 รายการ",
            quick_reply_labels=("บันทึก", "ยกเลิก"),
        )

    def test_pending_failure_gives_error_reply(self) -> None:
        """A crash while handling the import still answers the user."""
        with (
            patch(f"{SERVICE}.continue_pending_import", side_effect=RuntimeError("boom")),
            patch(f"{SERVICE}.process_line_message") as mock_process,
            patch(f"{SERVICE}.send_line_push") as mock_push,
            patch(f"{SERVICE}.show_line_loading"),
        ):
            handle_line_event(
                line_user_id=LINE_USER_ID,
                text="บันทึก",
                session_factory=MagicMock(),
                chat_model_provider=None,
                access_token="token-123",
            )
        mock_process.assert_not_called()
        assert mock_push.call_args[0][2] == LINE_ERROR_REPLY


class TestHandleLineDocumentEvent:
    """Images and files sent in LINE."""

    def _run(self, file_size: int = 1000, **patches: Any) -> dict[str, MagicMock]:
        mocks = {name: MagicMock(**kwargs) for name, kwargs in patches.items()}
        targets = ("send_line_reply", "show_line_loading", "download_line_content",
                   "start_document_import", "send_line_push")  # fmt: skip
        for target in targets:
            mocks.setdefault(target, MagicMock())
        with patch.multiple(SERVICE, **{target: mocks[target] for target in targets}):
            handle_line_document_event(
                line_user_id=LINE_USER_ID,
                message_id="m-1",
                file_size=file_size,
                ocr_model_provider=lambda: "ocr",
                chat_model_provider=lambda: "chat",
                access_token="token-123",
                reply_token="rt",
            )
        return mocks

    def test_reads_document_and_pushes_summary(self) -> None:
        """Notice, loading, download, read, then the summary with its buttons."""
        reply = LineReply("📄 อ่านได้ 2 รายการ", ("บันทึก", "ยกเลิก"))
        mocks = self._run(
            download_line_content={"return_value": b"%PDF"},
            start_document_import={"return_value": reply},
        )
        mocks["send_line_reply"].assert_called_once_with("token-123", "rt", DOCUMENT_NOTICE)
        mocks["download_line_content"].assert_called_once_with("token-123", "m-1")
        assert mocks["start_document_import"].call_args.args[:2] == (LINE_USER_ID, b"%PDF")
        mocks["send_line_push"].assert_called_once_with(
            "token-123",
            LINE_USER_ID,
            "📄 อ่านได้ 2 รายการ",
            quick_reply_labels=("บันทึก", "ยกเลิก"),
        )

    def test_too_large_is_refused_without_download(self) -> None:
        """Files over the limit are refused before downloading."""
        mocks = self._run(file_size=11 * 1024 * 1024)
        mocks["download_line_content"].assert_not_called()
        assert "10 MB" in mocks["send_line_push"].call_args[0][2]

    def test_failure_gives_error_reply(self) -> None:
        """Download or read failures still answer the user."""
        mocks = self._run(download_line_content={"side_effect": ConnectionError("down")})
        assert mocks["send_line_push"].call_args[0][2] == LINE_ERROR_REPLY
