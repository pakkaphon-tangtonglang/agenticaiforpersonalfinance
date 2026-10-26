"""Tests for streaming a guru answer."""

from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from finance_ai.agents.guru_personas import UnknownGuruError
from finance_ai.agents.guru_stream import stream_guru_query
from finance_ai.agents.stream_utils import StreamEvent


def _fake_stream(*_args: Any, **_kwargs: Any) -> Iterator[StreamEvent]:
    """Stand-in for stream_agent_response."""
    yield StreamEvent(event_type="token", content="มุมมองแบบ Dalio")
    yield StreamEvent(event_type="complete", intent="guru:dalio")


class TestStreamGuruQuery:
    """Tests for stream_guru_query."""

    @patch("finance_ai.agents.guru_stream.stream_agent_response", side_effect=_fake_stream)
    @patch("finance_ai.agents.guru_stream.get_compiled_graph")
    def test_builds_guru_state_with_history(
        self, mock_get_graph: MagicMock, mock_stream: MagicMock
    ) -> None:
        """History plus the new question is sent with guru_id in state."""
        events = list(
            stream_guru_query(
                query="ควรกระจายพอร์ตยังไง",
                guru_id="dalio",
                user_id="u1",
                db_session_factory=None,
                chat_model=MagicMock(),
                chat_history=[("user", "สวัสดี"), ("assistant", "สวัสดีครับ")],
            )
        )

        graph, state, intent = mock_stream.call_args.args
        assert graph is mock_get_graph.return_value
        assert intent == "guru:dalio"
        assert state["guru_id"] == "dalio"
        assert state["messages"][-1] == ("user", "ควรกระจายพอร์ตยังไง")
        assert len(state["messages"]) == 3
        assert events[-1].intent == "guru:dalio"

    @patch("finance_ai.agents.guru_stream.get_compiled_graph", return_value=None)
    def test_missing_graph_registration_raises(self, _mock_get_graph: MagicMock) -> None:
        """A missing cache registration is reported instead of streaming None."""
        with pytest.raises(RuntimeError, match="not registered"):
            list(stream_guru_query("q", "dalio", "u1", None, MagicMock()))

    def test_unknown_guru_raises_before_calling_model(self) -> None:
        """An invalid guru_id fails fast with the list of valid ids."""
        with pytest.raises(UnknownGuruError, match="Available"):
            list(
                stream_guru_query(
                    query="q",
                    guru_id="elon",
                    user_id="u1",
                    db_session_factory=None,
                    chat_model=MagicMock(),
                )
            )
