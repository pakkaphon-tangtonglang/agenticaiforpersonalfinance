"""Tests for the guru panel (ask every guru one question)."""

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from finance_ai.agents.guru_panel import (
    GuruStance,
    PanelSynthesis,
    answer_as_guru,
    run_guru_panel,
    synthesis_to_markdown,
    synthesize_panel,
)


def _graph_answering(text_by_guru: dict[str, str]) -> MagicMock:
    """A fake compiled graph whose answer depends on state['guru_id']."""
    graph = MagicMock()

    def _invoke(state: dict[str, Any]) -> dict[str, Any]:
        answer = text_by_guru[state["guru_id"]]
        if answer == "RAISE":
            raise RuntimeError("model quota exceeded")
        return {"messages": [HumanMessage(content="q"), AIMessage(content=answer)]}

    graph.invoke.side_effect = _invoke
    return graph


SYNTHESIS = PanelSynthesis(
    stances=[
        GuruStance(guru_id="buffett", stance="ซื้อเพิ่ม", key_reason="ราคาถูก", risk="ผันผวน")
    ],
    consensus="ทุกคนเน้นถือยาว",
    differences="Dalio เน้นกระจาย",
)


class TestSynthesisToMarkdown:
    """Tests for synthesis_to_markdown."""

    def test_renders_table_and_summary(self) -> None:
        """One row per guru plus consensus and differences lines."""
        markdown = synthesis_to_markdown(SYNTHESIS)

        assert "| buffett | ซื้อเพิ่ม | ราคาถูก | ผันผวน |" in markdown
        assert "**เห็นตรงกัน:** ทุกคนเน้นถือยาว" in markdown
        assert "**เห็นต่าง:** Dalio เน้นกระจาย" in markdown


class TestAnswerAsGuru:
    """Tests for answer_as_guru."""

    @patch("finance_ai.agents.guru_panel.get_compiled_graph")
    def test_returns_final_ai_text_with_panel_instruction(self, mock_get_graph: MagicMock) -> None:
        """The guru gets a short-answer instruction and its final text is returned."""
        graph = _graph_answering({"dalio": "จุดยืน: กระจายความเสี่ยง"})
        mock_get_graph.return_value = graph

        answer = answer_as_guru("ตลาดตก", "dalio", "u1", None, MagicMock())

        sent_state = graph.invoke.call_args.args[0]
        assert answer == "จุดยืน: กระจายความเสี่ยง"
        assert sent_state["guru_id"] == "dalio"
        assert "จุดยืน:" in sent_state["messages"][0][1]


class TestSynthesizePanel:
    """Tests for synthesize_panel."""

    def test_uses_structured_output(self) -> None:
        """One structured call turns the answers into a comparison."""
        chat_model = MagicMock()
        chat_model.with_structured_output.return_value.invoke.return_value = SYNTHESIS

        result = synthesize_panel("ตลาดตก", {"buffett": "ซื้อเพิ่ม"}, chat_model)

        assert result == SYNTHESIS
        chat_model.with_structured_output.assert_called_once_with(PanelSynthesis)
        prompt = chat_model.with_structured_output.return_value.invoke.call_args.args[0]
        assert "buffett" in prompt and "ซื้อเพิ่ม" in prompt

    def test_returns_none_when_model_fails(self) -> None:
        """A broken synthesis never breaks the panel."""
        chat_model = MagicMock()
        chat_model.with_structured_output.side_effect = ValueError("no structured output")

        assert synthesize_panel("q", {"buffett": "a"}, chat_model) is None


class TestRunGuruPanel:
    """Tests for run_guru_panel."""

    @patch("finance_ai.agents.guru_panel.synthesize_panel", return_value=SYNTHESIS)
    @patch("finance_ai.agents.guru_panel.get_compiled_graph")
    def test_one_failing_guru_does_not_break_the_panel(
        self, mock_get_graph: MagicMock, mock_synthesize: MagicMock
    ) -> None:
        """Answers and errors are both reported, then the synthesis."""
        mock_get_graph.return_value = _graph_answering(
            {"buffett": "ซื้อเพิ่ม", "dalio": "RAISE", "niwes": "ถือยาว"}
        )

        events = list(
            run_guru_panel("ตลาดตก", ["buffett", "dalio", "niwes"], "u1", None, MagicMock())
        )

        answers = {e.guru_id: e.content for e in events if e.event_type == "guru_answer"}
        errors = [e.guru_id for e in events if e.event_type == "guru_error"]
        assert answers == {"buffett": "ซื้อเพิ่ม", "niwes": "ถือยาว"}
        assert errors == ["dalio"]
        error_text = next(e.content for e in events if e.event_type == "guru_error")
        assert "quota" not in error_text
        assert events[-1].event_type == "synthesis"
        assert mock_synthesize.call_args.args[1] == answers

    @patch("finance_ai.agents.guru_panel.get_compiled_graph")
    def test_no_synthesis_when_every_guru_fails(self, mock_get_graph: MagicMock) -> None:
        """Nothing to compare means no synthesis call."""
        mock_get_graph.return_value = _graph_answering({"buffett": "RAISE"})

        events = list(run_guru_panel("q", ["buffett"], "u1", None, MagicMock()))

        assert [e.event_type for e in events] == ["guru_error"]

    @patch("finance_ai.agents.guru_panel.synthesize_panel", return_value=None)
    @patch("finance_ai.agents.guru_panel.get_compiled_graph")
    def test_failed_synthesis_is_reported(
        self, mock_get_graph: MagicMock, _mock_synthesize: MagicMock
    ) -> None:
        """A None synthesis becomes a synthesis_error event."""
        mock_get_graph.return_value = _graph_answering({"buffett": "ซื้อเพิ่ม"})

        events = list(run_guru_panel("q", ["buffett"], "u1", None, MagicMock()))

        assert events[-1].event_type == "synthesis_error"
