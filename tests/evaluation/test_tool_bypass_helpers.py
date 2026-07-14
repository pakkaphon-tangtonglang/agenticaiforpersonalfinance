"""Tests for tool-bypass benchmark graph helpers."""

from decimal import Decimal
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from finance_ai.evaluation.tool_bypass_benchmark import (
    BYPASS_TEST_CASES,
    ToolBypassBenchmarkResult,
    _build_notool_graph,
    _evaluate_case,
    _run_notool,
    run_tool_bypass_benchmark,
)


class TestToolBypassHelpers:
    """Tests for tool-bypass benchmark helpers."""

    def test_run_notool_returns_response_and_latency(self) -> None:
        """_run_notool returns message content and latency."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="tax 25000")],
        }
        response, latency = _run_notool(mock_graph, "query")
        assert response == "tax 25000"
        assert latency >= 0.0
        mock_graph.invoke.assert_called_once()

    def test_run_notool_handles_exception(self) -> None:
        """_run_notool returns error string when graph raises."""
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("model error")
        response, latency = _run_notool(mock_graph, "query")
        assert "ERROR" in response
        assert latency >= 0.0

    def test_build_notool_graph_returns_compiled_graph(self) -> None:
        """_build_notool_graph returns a compiled StateGraph."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="test")
        graph = _build_notool_graph(mock_model)
        assert graph is not None

    def test_evaluate_case_with_mock_graph(self) -> None:
        """_evaluate_case computes tool tax and evaluates no-tool response."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [AIMessage(content="tax 21500")],
        }
        case = BYPASS_TEST_CASES[0]
        result = _evaluate_case(case, mock_graph)
        assert result.case_id == case["case_id"]
        assert result.tool_tax == case["expected_tax"]
        assert result.tool_within_tolerance is True

    def test_run_tool_bypass_benchmark(self) -> None:
        """run_tool_bypass_benchmark returns aggregate result."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = AIMessage(content="tax 21500")
        result = run_tool_bypass_benchmark(
            mock_model,
            "test-model",
            cases=[BYPASS_TEST_CASES[0]],
        )
        assert isinstance(result, ToolBypassBenchmarkResult)
        assert result.model_name == "test-model"
        assert len(result.cases) == 1
        assert result.tool_accuracy_pct == 100.0
