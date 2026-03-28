"""Tests for LLM-as-Judge quality evaluator."""

import json
from decimal import Decimal
from unittest.mock import MagicMock

from finance_ai.evaluation.models import QualityCase, QualityDataset
from finance_ai.evaluation.quality_evaluator import (
    build_judge_prompt,
    evaluate_quality_dataset,
    evaluate_single_quality_case,
    parse_judge_response,
)


def _make_judge_response(
    relevance: int = 4,
    completeness: int = 4,
    accuracy: int = 5,
    thai_quality: int = 4,
    overall: int = 4,
) -> str:
    """Create a valid judge JSON response."""
    return json.dumps(
        {
            "relevance": relevance,
            "completeness": completeness,
            "accuracy": accuracy,
            "thai_language_quality": thai_quality,
            "overall": overall,
            "reasoning": "คำตอบดี ครบถ้วน",
        }
    )


def _make_mock_judge(response_json: str) -> MagicMock:
    """Create a mock judge model returning given JSON."""
    judge = MagicMock()
    judge.invoke.return_value = MagicMock(content=response_json)
    return judge


class TestBuildJudgePrompt:
    """Tests for build_judge_prompt."""

    def test_contains_query_and_response(self) -> None:
        """Test that prompt contains the query and response."""
        prompt = build_judge_prompt("คำถาม", "คำตอบ")
        assert "คำถาม" in prompt
        assert "คำตอบ" in prompt

    def test_contains_criteria(self) -> None:
        """Test that prompt contains scoring criteria."""
        prompt = build_judge_prompt("q", "r")
        assert "relevance" in prompt
        assert "completeness" in prompt
        assert "accuracy" in prompt


class TestParseJudgeResponse:
    """Tests for parse_judge_response."""

    def test_valid_json(self) -> None:
        """Test parsing valid judge JSON."""
        content = _make_judge_response()
        score = parse_judge_response(content)
        assert score.relevance == Decimal("4")
        assert score.overall == Decimal("4")

    def test_invalid_json_returns_default(self) -> None:
        """Test that invalid JSON returns default scores."""
        score = parse_judge_response("not json")
        assert score.overall == Decimal("1")
        assert "Failed" in score.judge_reasoning

    def test_code_fence_wrapped(self) -> None:
        """Test parsing JSON wrapped in code fence."""
        content = f"```json\n{_make_judge_response()}\n```"
        score = parse_judge_response(content)
        assert score.relevance == Decimal("4")

    def test_missing_field_returns_default(self) -> None:
        """Test that missing field returns default scores."""
        content = json.dumps({"relevance": 4})
        score = parse_judge_response(content)
        assert score.overall == Decimal("1")


class TestEvaluateSingleQualityCase:
    """Tests for evaluate_single_quality_case."""

    def test_with_pre_computed_response(self) -> None:
        """Test evaluation with pre-computed agent response."""
        agent = MagicMock()
        judge = _make_mock_judge(_make_judge_response())
        case = QualityCase(
            case_id="q_001",
            query="คำนวณภาษี",
            expected_agent="tax",
        )
        result = evaluate_single_quality_case(agent, judge, case, agent_response="ภาษี 29,000 บาท")
        assert result.scores.overall == Decimal("4")
        assert result.agent_response == "ภาษี 29,000 บาท"
        agent.invoke.assert_not_called()

    def test_generates_response_if_not_provided(self) -> None:
        """Test that agent is called when no response provided."""
        agent = MagicMock()
        agent.invoke.return_value = MagicMock(content="agent response")
        judge = _make_mock_judge(_make_judge_response())
        case = QualityCase(
            case_id="q_002",
            query="test",
            expected_agent="tax",
        )
        result = evaluate_single_quality_case(agent, judge, case)
        assert result.agent_response == "agent response"
        agent.invoke.assert_called_once()

    def test_latency_recorded(self) -> None:
        """Test that latency is recorded."""
        agent = MagicMock()
        judge = _make_mock_judge(_make_judge_response())
        case = QualityCase(
            case_id="q_003",
            query="test",
            expected_agent="tax",
        )
        result = evaluate_single_quality_case(agent, judge, case, agent_response="response")
        assert result.latency_seconds >= 0.0


class TestEvaluateQualityDataset:
    """Tests for evaluate_quality_dataset."""

    def test_aggregate_scores(self) -> None:
        """Test that dataset evaluation aggregates scores."""
        agent = MagicMock()
        judge = _make_mock_judge(
            _make_judge_response(relevance=4, completeness=3, accuracy=5, thai_quality=4, overall=4)
        )
        dataset = QualityDataset(
            version="1.0",
            cases=[
                QualityCase(case_id="q_001", query="q1", expected_agent="tax"),
                QualityCase(case_id="q_002", query="q2", expected_agent="expense"),
            ],
        )
        responses = {"q_001": "response1", "q_002": "response2"}
        agg = evaluate_quality_dataset(agent, judge, dataset, responses)
        assert agg.total_cases == 2
        assert agg.mean_overall == Decimal("4.0000")
        assert "tax" in agg.per_agent_scores
        assert "expense" in agg.per_agent_scores

    def test_empty_dataset(self) -> None:
        """Test with empty dataset."""
        agent = MagicMock()
        judge = MagicMock()
        dataset = QualityDataset(version="1.0", cases=[])
        agg = evaluate_quality_dataset(agent, judge, dataset)
        assert agg.total_cases == 0
        assert agg.mean_overall == Decimal("0.0000")
