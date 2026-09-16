"""Tests for hallucination compliance evaluator."""

import json
from unittest.mock import MagicMock

from finance_ai.evaluation.hallucination_evaluator import (
    _parse_judge_violations,
    check_forbidden_patterns,
    check_known_facts,
    evaluate_hallucination_dataset,
    evaluate_single_hallucination_case,
    evaluate_with_llm_judge,
)
from finance_ai.evaluation.models import HallucinationCase, HallucinationDataset


class TestCheckForbiddenPatterns:
    """Tests for check_forbidden_patterns."""

    def test_no_violations(self) -> None:
        """Test when no forbidden patterns are found."""
        result = check_forbidden_patterns("60,000 บาท", ["70,000", "80,000"])
        assert result == []

    def test_single_violation(self) -> None:
        """Test when one forbidden pattern is found."""
        result = check_forbidden_patterns("70,000 บาท", ["70,000"])
        assert len(result) == 1

    def test_multiple_violations(self) -> None:
        """Test when multiple forbidden patterns are found."""
        result = check_forbidden_patterns("70,000 และ 80,000 บาท", ["70,000", "80,000"])
        assert len(result) == 2

    def test_empty_patterns(self) -> None:
        """Test with no patterns to check."""
        result = check_forbidden_patterns("any text", [])
        assert result == []

    def test_regex_pattern(self) -> None:
        """Test with regex pattern."""
        result = check_forbidden_patterns("rate is 15%", [r"\d+%"])
        assert len(result) == 1


class TestCheckKnownFacts:
    """Tests for check_known_facts."""

    def test_facts_present(self) -> None:
        """Test when known facts are correctly present."""
        result = check_known_facts(
            "ค่าลดหย่อนส่วนตัว 60,000 บาท",
            ["ค่าลดหย่อนส่วนตัว 60,000 บาท"],
        )
        assert result == []

    def test_fact_number_missing(self) -> None:
        """Test when a fact's number is missing from response."""
        result = check_known_facts(
            "ค่าลดหย่อนส่วนตัว สูงสุด",
            ["ค่าลดหย่อนส่วนตัว 60,000 บาท"],
        )
        assert len(result) > 0

    def test_unrelated_response(self) -> None:
        """Test with response that doesn't mention the topic."""
        result = check_known_facts(
            "the weather is nice",
            ["ค่าลดหย่อนส่วนตัว 60,000 บาท"],
        )
        assert result == []


class TestParseJudgeViolations:
    """Tests for _parse_judge_violations."""

    def test_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        content = json.dumps({"violations": ["error1"], "is_compliant": False})
        result = _parse_judge_violations(content)
        assert result == ["error1"]

    def test_empty_violations(self) -> None:
        """Test parsing compliant response."""
        content = json.dumps({"violations": [], "is_compliant": True})
        result = _parse_judge_violations(content)
        assert result == []

    def test_invalid_json(self) -> None:
        """Test parsing invalid JSON returns empty list."""
        result = _parse_judge_violations("not json")
        assert result == []

    def test_code_fence_wrapped(self) -> None:
        """Test parsing JSON wrapped in code fence."""
        content = '```json\n{"violations": ["err"], "is_compliant": false}\n```'
        result = _parse_judge_violations(content)
        assert result == ["err"]


class TestEvaluateWithLLMJudge:
    """Tests for evaluate_with_llm_judge."""

    def test_judge_returns_violations(self) -> None:
        """Test when judge finds violations."""
        judge = MagicMock()
        judge.invoke.return_value = MagicMock(
            content=json.dumps({"violations": ["wrong number"], "is_compliant": False})
        )
        result = evaluate_with_llm_judge("response", ["fact1"], judge)
        assert result == ["wrong number"]

    def test_judge_returns_compliant(self) -> None:
        """Test when judge finds no violations."""
        judge = MagicMock()
        judge.invoke.return_value = MagicMock(
            content=json.dumps({"violations": [], "is_compliant": True})
        )
        result = evaluate_with_llm_judge("response", ["fact1"], judge)
        assert result == []


class TestEvaluateSingleHallucinationCase:
    """Tests for evaluate_single_hallucination_case."""

    def test_compliant_response(self) -> None:
        """Test a compliant response (no violations)."""
        case = HallucinationCase(
            case_id="hal_001",
            query="ค่าลดหย่อน",
            category="fabricated_number",
            known_facts=["ค่าลดหย่อนส่วนตัว 60,000 บาท"],
            forbidden_patterns=["70,000"],
        )
        result = evaluate_single_hallucination_case(case, "ค่าลดหย่อนส่วนตัว 60,000 บาท")
        assert result.is_compliant is True
        assert result.violations_found == []

    def test_non_compliant_forbidden_pattern(self) -> None:
        """Test non-compliant due to forbidden pattern."""
        case = HallucinationCase(
            case_id="hal_002",
            query="ค่าลดหย่อน",
            category="fabricated_number",
            known_facts=["60,000"],
            forbidden_patterns=["70,000"],
        )
        result = evaluate_single_hallucination_case(case, "ค่าลดหย่อน 70,000 บาท")
        assert result.is_compliant is False

    def test_with_llm_judge(self) -> None:
        """Test using Stage 2 LLM judge."""
        case = HallucinationCase(
            case_id="hal_003",
            query="test",
            category="fabricated_law",
            known_facts=["fact1"],
        )
        judge = MagicMock()
        judge.invoke.return_value = MagicMock(
            content=json.dumps({"violations": ["llm_violation"], "is_compliant": False})
        )
        result = evaluate_single_hallucination_case(
            case, "response", use_llm_judge=True, judge_model=judge
        )
        assert "llm_violation" in result.violations_found


class TestEvaluateHallucinationDataset:
    """Tests for evaluate_hallucination_dataset."""

    def test_aggregate_results(self) -> None:
        """Test dataset aggregation."""
        dataset = HallucinationDataset(
            version="1.0",
            cases=[
                HallucinationCase(
                    case_id="hal_001",
                    query="q1",
                    category="fabricated_number",
                    known_facts=["60,000"],
                    forbidden_patterns=["70,000"],
                ),
                HallucinationCase(
                    case_id="hal_002",
                    query="q2",
                    category="fabricated_law",
                    known_facts=["fact"],
                ),
            ],
        )
        responses: dict[str, str | None] = {
            "hal_001": "ค่าลดหย่อน 60,000 บาท",
            "hal_002": "response text",
        }
        agg = evaluate_hallucination_dataset(dataset, responses)
        assert agg.total_cases == 2
        assert len(agg.results) == 2

    def test_empty_dataset(self) -> None:
        """Test with empty dataset."""
        dataset = HallucinationDataset(version="1.0", cases=[])
        agg = evaluate_hallucination_dataset(dataset, {})
        assert agg.total_cases == 0


class TestFailedGenerationHandling:
    """A missing (None) agent response must be non-compliant."""

    def test_none_response_marked_non_compliant(self) -> None:
        """A None response fails with a generation_failed violation."""
        case = HallucinationCase(
            case_id="hal_010",
            query="ค่าลดหย่อน",
            category="fabricated_number",
            known_facts=["ค่าลดหย่อนส่วนตัว 60,000 บาท"],
            forbidden_patterns=[],
        )
        result = evaluate_single_hallucination_case(case, None)
        assert result.is_compliant is False
        assert "generation_failed" in result.violations_found

    def test_none_response_skips_llm_judge(self) -> None:
        """The judge is not called for a failed generation."""
        case = HallucinationCase(
            case_id="hal_011",
            query="ค่าลดหย่อน",
            category="fabricated_number",
            known_facts=["60,000"],
            forbidden_patterns=[],
        )
        judge = MagicMock()
        result = evaluate_single_hallucination_case(
            case, None, use_llm_judge=True, judge_model=judge
        )
        assert result.is_compliant is False
        judge.invoke.assert_not_called()

    def test_dataset_treats_missing_response_as_non_compliant(self) -> None:
        """Dataset evaluation counts a None response as non-compliant."""
        dataset = HallucinationDataset(
            version="1.0",
            cases=[
                HallucinationCase(
                    case_id="hal_012",
                    query="ค่าลดหย่อน",
                    category="fabricated_number",
                    known_facts=["60,000"],
                    forbidden_patterns=[],
                )
            ],
        )
        aggregate = evaluate_hallucination_dataset(dataset, {"hal_012": None})
        assert aggregate.compliance_rate == 0
        assert aggregate.compliant_count == 0
