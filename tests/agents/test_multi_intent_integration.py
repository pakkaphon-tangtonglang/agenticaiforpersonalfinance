"""Multi-intent routing across storage, API events, LINE and the architecture study."""

# pylint: disable=unsubscriptable-object
# ^ false positive: sessionmaker IS subscriptable (SQLAlchemy 2.x generic).

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy.orm import Session, sessionmaker

from finance_ai.agents.stream_utils import StreamEvent
from finance_ai.agents.tool_trace import traces_from_messages
from finance_ai.database.models.user import User
from finance_ai.evaluation.architectures import hub_spoke
from finance_ai.evaluation.architectures.design_context import DesignContext
from finance_ai.evaluation.architectures.hub_spoke import run_hub_multi, run_hub_spoke
from finance_ai.evaluation.architectures.run_result import ArchitectureRun
from finance_ai.evaluation.architectures.runner import ORIGINAL_DESIGNS, StudyConfig
from finance_ai.evaluation.architectures.scoring import score_run
from finance_ai.evaluation.architectures.study_cli import build_parser
from finance_ai.evaluation.architectures.task_models import ArchitectureTask
from finance_ai.line import line_bot_service
from finance_ai.line.line_bot_service import process_line_message
from finance_ai.line.text_split import LINE_MAX_MESSAGES, split_line_text
from finance_ai.main import _format_sse_event
from finance_ai.tools.conversation_service import (
    create_conversation,
    get_last_reply_intents,
    load_conversation_messages,
    save_assistant_message,
    save_user_message,
)


class TestTracesFromMessages:
    """Tool calls of the latest turn are read back from a finished run."""

    def test_only_latest_turn(self) -> None:
        """Calls before the last human message are ignored."""
        messages = [
            HumanMessage(content="old"),
            AIMessage(content="", tool_calls=[{"id": "a", "name": "add_income", "args": {}}]),
            ToolMessage(content="ok", tool_call_id="a", name="add_income"),
            HumanMessage(content="new"),
            AIMessage(
                content="", tool_calls=[{"id": "b", "name": "add_expense", "args": {"amount": 80}}]
            ),
            ToolMessage(content="ok", tool_call_id="b", name="add_expense"),
            AIMessage(content="done"),
        ]
        traces = traces_from_messages(messages)
        assert [(trace.name, trace.args) for trace in traces] == [("add_expense", {"amount": "80"})]

    def test_no_messages(self) -> None:
        """An empty run has no traces."""
        assert traces_from_messages([]) == []


class TestConversationIntents:
    """Multi-topic replies remember every topic."""

    def test_several_topics_are_stored_and_read(
        self, test_session: Session, sample_user: User
    ) -> None:
        """The latest reply's topics are returned and shown in history."""
        conv_id = create_conversation(test_session, sample_user.id).id
        save_user_message(test_session, conv_id, "ภาษีเท่าไร แล้วควรออมเท่าไร")
        save_assistant_message(test_session, conv_id, "...", "tax", intents=["tax", "planning"])
        assert get_last_reply_intents(test_session, conv_id) == ["tax", "planning"]
        assert load_conversation_messages(test_session, conv_id)[-1]["intents"] == [
            "tax",
            "planning",
        ]

    def test_single_topic_stores_nothing(self, test_session: Session, sample_user: User) -> None:
        """A one-topic reply keeps intents empty (single-topic routing unchanged)."""
        conv_id = create_conversation(test_session, sample_user.id).id
        message = save_assistant_message(test_session, conv_id, "...", "tax", intents=["tax"])
        assert message.intents is None
        assert get_last_reply_intents(test_session, conv_id) == []
        assert "intents" not in load_conversation_messages(test_session, conv_id)[-1]

    def test_new_conversation_has_no_topics(self, test_session: Session, sample_user: User) -> None:
        """No reply yet means no previous topics."""
        conv_id = create_conversation(test_session, sample_user.id).id
        assert get_last_reply_intents(test_session, conv_id) == []


class TestSseIntents:
    """The web chat learns every topic before the stream completes."""

    def test_intents_event_precedes_complete(self) -> None:
        """A multi-topic complete event is preceded by an intents event."""
        sse = _format_sse_event(
            StreamEvent(event_type="complete", intent="tax", intents=["tax", "planning"])
        )
        assert sse is not None
        assert sse.startswith('event: intents\ndata: ["tax", "planning"]\n\n')
        assert sse.endswith('event: complete\ndata: "tax"\n\n')

    def test_single_topic_complete_is_unchanged(self) -> None:
        """A one-topic reply sends only the complete event, as before."""
        sse = _format_sse_event(StreamEvent(event_type="complete", intent="tax"))
        assert sse == 'event: complete\ndata: "tax"\n\n'


class TestSplitLineText:
    """Long replies fit LINE's limits."""

    def test_short_text_is_one_message(self) -> None:
        """A normal reply is sent as it is."""
        assert split_line_text("สวัสดี") == ["สวัสดี"]

    def test_splits_at_paragraphs(self) -> None:
        """Paragraphs are kept whole when they fit."""
        assert split_line_text("aaa\n\nbbb\n\nccc", limit=8) == ["aaa\n\nbbb", "ccc"]

    def test_long_paragraph_is_cut(self) -> None:
        """A paragraph longer than the limit is cut into pieces."""
        assert split_line_text("abcdefghij", limit=4) == ["abcd", "efgh", "ij"]

    def test_message_count_is_capped(self) -> None:
        """At most five messages; the last ends with an ellipsis."""
        chunks = split_line_text("\n\n".join(["x" * 4] * 9), limit=4)
        assert len(chunks) == LINE_MAX_MESSAGES
        assert chunks[-1].endswith("…") and len(chunks[-1]) <= 4

    def test_empty_text(self) -> None:
        """An empty reply still gives one message."""
        assert split_line_text("") == [""]


class TestLinePushSplit:
    """send_line_push sends several messages for a long reply."""

    def test_long_reply_becomes_several_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the last message carries the quick-reply menu."""
        sent: list[Any] = []

        class FakeApi:
            def __init__(self, *args: Any) -> None:
                """Accept the ApiClient."""

            def push_message(self, **kwargs: Any) -> None:
                """Record the request."""
                sent.append(kwargs["push_message_request"])

        monkeypatch.setattr("finance_ai.line.messaging_client.MessagingApi", FakeApi)
        from finance_ai.line.messaging_client import send_line_push  # noqa: PLC0415

        send_line_push("token", "U1", "ก" * 6000)
        messages = sent[0].messages
        assert [len(message.text) for message in messages] == [5000, 1000]
        assert messages[0].quick_reply is None and messages[-1].quick_reply is not None


class TestLineIntents:
    """LINE passes the previous topics and stores every topic."""

    def test_previous_topics_and_saved_intents(
        self, test_session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second message sees the first reply's topics."""
        seen: list[Any] = []

        def fake_orchestrate(**kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs["previous_intents"])
            return {"intent": "tax", "intents": ["tax", "planning"], "response": "สองเรื่อง"}

        monkeypatch.setattr(line_bot_service, "orchestrate_query", fake_orchestrate)
        factory = sessionmaker(bind=test_session.get_bind())
        process_line_message(factory, None, "Uline-multi", "ภาษี แล้วออม")
        process_line_message(factory, None, "Uline-multi", "แล้วเรื่องภาษีล่ะ")
        assert seen == [[], ["tax", "planning"]]


class TestHubMultiDesign:
    """The study can run multi-intent routing next to the original hub."""

    def test_route_lists_every_agent(
        self, monkeypatch: pytest.MonkeyPatch, make_context: Any
    ) -> None:
        """hub_multi's route is every agent that answered."""
        captured: dict[str, Any] = {}

        def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"intent": "tax", "intents": ["tax", "general"], "response": "ok"}

        monkeypatch.setattr(hub_spoke, "orchestrate_query", fake)
        assert run_hub_multi("q", make_context()).route == ["tax", "general"]
        assert captured["allow_multi_intent"] is True

    def test_original_hub_turns_multi_intent_off(
        self, monkeypatch: pytest.MonkeyPatch, make_context: Any
    ) -> None:
        """hub_spoke keeps one agent per message, so old results stay comparable."""
        captured: dict[str, Any] = {}

        def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"intent": "tax", "response": "ok"}

        monkeypatch.setattr(hub_spoke, "orchestrate_query", fake)
        assert run_hub_spoke("q", make_context()).route == ["tax"]
        assert captured["allow_multi_intent"] is False

    def test_route_counts_if_any_agent_fits(self) -> None:
        """A cross-topic run is routed correctly when an acceptable agent answered."""
        task = ArchitectureTask(
            task_id="crs_x", query="q", category="cross", acceptable_agents=["planning"]
        )
        run = ArchitectureRun(
            task_id="crs_x",
            design="hub_multi",
            round_index=0,
            answer="ok",
            route=["tax", "planning"],
        )
        assert score_run(task, run, []).route_ok is True

    def test_default_designs_are_the_original_four(self) -> None:
        """A plain study run reproduces the original four designs."""
        assert build_parser().parse_args([]).designs == list(ORIGINAL_DESIGNS)
        assert "hub_multi" not in StudyConfig(model_label="m", output_dir=Path(".")).designs
        assert build_parser().parse_args(["--designs", "hub_multi"]).designs == ["hub_multi"]


@pytest.fixture(name="make_context")
def fixture_make_context() -> Any:
    """Build a design context without a model (orchestrate_query is faked)."""

    def make() -> DesignContext:
        return DesignContext(chat_model=None, user_id="u", db_session_factory=None)  # type: ignore[arg-type]

    return make
