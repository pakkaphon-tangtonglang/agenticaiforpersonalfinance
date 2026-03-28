"""LLM-as-Judge response quality evaluation module.

Uses a separate LLM (judge) to score agent responses on 4 dimensions:
relevance, completeness, accuracy, and Thai language quality.
"""

import json
import time
from decimal import Decimal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from finance_ai.evaluation.metrics import compute_mean_decimal
from finance_ai.evaluation.models import (
    QualityAggregateResult,
    QualityCase,
    QualityDataset,
    QualityResult,
    QualityScore,
)

JUDGE_PROMPT_TEMPLATE = """คุณเป็นผู้ตัดสินคุณภาพคำตอบระบบ AI การเงินส่วนบุคคล

ให้คะแนน 1-5 สำหรับแต่ละเกณฑ์:
1. relevance (ตรงประเด็น): คำตอบตรงกับคำถามแค่ไหน
2. completeness (ครบถ้วน): ครอบคลุมข้อมูลสำคัญหรือไม่
3. accuracy (ถูกต้อง): ข้อมูลถูกต้องตามหลักการเงินไทย
4. thai_language_quality (คุณภาพภาษา): ภาษาไทยเป็นธรรมชาติ อ่านง่าย

คำถาม: {query}
คำตอบ: {response}

ตอบเป็น JSON เท่านั้น:
{{"relevance": X, "completeness": X, "accuracy": X,
"thai_language_quality": X, "overall": X, "reasoning": "..."}}"""

DEFAULT_SCORE = QualityScore(
    relevance=Decimal("1"),
    completeness=Decimal("1"),
    accuracy=Decimal("1"),
    thai_language_quality=Decimal("1"),
    overall=Decimal("1"),
    judge_reasoning="Failed to parse judge response",
)


def build_judge_prompt(query: str, response: str) -> str:
    """Build the judge prompt from query and response.

    Args:
        query: Original user query.
        response: Agent response to evaluate.

    Returns:
        Formatted judge prompt string.

    Example:
        >>> prompt = build_judge_prompt("คำนวณภาษี", "ภาษี 29,000 บาท")
    """
    return JUDGE_PROMPT_TEMPLATE.format(query=query, response=response)


def parse_judge_response(content: str) -> QualityScore:
    """Parse judge LLM response into QualityScore.

    Args:
        content: Raw response from the judge LLM.

    Returns:
        QualityScore parsed from JSON, or default score on failure.

    Example:
        >>> score = parse_judge_response('{"relevance": 4, ...}')
    """
    try:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return QualityScore(
            relevance=Decimal(str(data["relevance"])),
            completeness=Decimal(str(data["completeness"])),
            accuracy=Decimal(str(data["accuracy"])),
            thai_language_quality=Decimal(str(data["thai_language_quality"])),
            overall=Decimal(str(data["overall"])),
            judge_reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return DEFAULT_SCORE


def evaluate_single_quality_case(
    agent_model: BaseChatModel,
    judge_model: BaseChatModel,
    case: QualityCase,
    agent_response: str | None = None,
) -> QualityResult:
    """Evaluate a single quality case using LLM-as-Judge.

    If agent_response is not provided, the agent_model is called
    to generate a response first.

    Args:
        agent_model: Model being evaluated (for generating response).
        judge_model: Model used as judge.
        case: Quality test case.
        agent_response: Pre-computed agent response (optional).

    Returns:
        QualityResult with judge scores.

    Example:
        >>> result = evaluate_single_quality_case(agent, judge, case)
    """
    start = time.perf_counter()

    if agent_response is None:
        agent_result = agent_model.invoke([HumanMessage(content=case.query)])
        agent_response = str(agent_result.content)

    scores = _get_judge_scores(judge_model, case.query, agent_response)
    latency = time.perf_counter() - start

    return QualityResult(
        case_id=case.case_id,
        query=case.query,
        agent_response=agent_response,
        scores=scores,
        latency_seconds=latency,
    )


def _get_judge_scores(
    judge_model: BaseChatModel,
    query: str,
    response: str,
) -> QualityScore:
    """Get quality scores from the judge model.

    Args:
        judge_model: LLM to use as judge.
        query: Original user query.
        response: Agent response to judge.

    Returns:
        QualityScore from the judge.
    """
    prompt = build_judge_prompt(query, response)
    result = judge_model.invoke(
        [
            SystemMessage(content="You are a quality evaluation judge."),
            HumanMessage(content=prompt),
        ]
    )
    return parse_judge_response(str(result.content))


def evaluate_quality_dataset(
    agent_model: BaseChatModel,
    judge_model: BaseChatModel,
    dataset: QualityDataset,
    agent_responses: dict[str, str] | None = None,
) -> QualityAggregateResult:
    """Evaluate all cases in a quality dataset.

    Args:
        agent_model: Model being evaluated.
        judge_model: Model used as judge.
        dataset: Quality evaluation dataset.
        agent_responses: Pre-computed responses keyed by case_id.

    Returns:
        QualityAggregateResult with aggregated scores.

    Example:
        >>> agg = evaluate_quality_dataset(agent, judge, dataset)
    """
    responses = agent_responses or {}
    results = [
        evaluate_single_quality_case(
            agent_model,
            judge_model,
            case,
            agent_response=responses.get(case.case_id),
        )
        for case in dataset.cases
    ]
    return _aggregate_quality_results(results, dataset.cases)


def _aggregate_quality_results(
    results: list[QualityResult],
    cases: list[QualityCase],
) -> QualityAggregateResult:
    """Aggregate individual quality results.

    Args:
        results: List of individual quality results.
        cases: Original test cases (for agent grouping).

    Returns:
        QualityAggregateResult with mean scores.
    """
    per_agent = _compute_per_agent_scores(results, cases)
    latencies = [r.latency_seconds for r in results]

    return QualityAggregateResult(
        total_cases=len(results),
        mean_relevance=compute_mean_decimal([r.scores.relevance for r in results]),
        mean_completeness=compute_mean_decimal([r.scores.completeness for r in results]),
        mean_accuracy=compute_mean_decimal([r.scores.accuracy for r in results]),
        mean_thai_language_quality=compute_mean_decimal(
            [r.scores.thai_language_quality for r in results]
        ),
        mean_overall=compute_mean_decimal([r.scores.overall for r in results]),
        per_agent_scores=per_agent,
        mean_latency_seconds=sum(latencies) / max(len(latencies), 1),
        results=results,
    )


def _compute_per_agent_scores(
    results: list[QualityResult],
    cases: list[QualityCase],
) -> dict[str, dict[str, Decimal]]:
    """Compute per-agent mean quality scores.

    Args:
        results: List of quality results.
        cases: Original test cases.

    Returns:
        Dict mapping agent name to mean scores per dimension.
    """
    agent_scores: dict[str, list[QualityScore]] = {}
    for result, case in zip(results, cases):
        agent = case.expected_agent
        if agent not in agent_scores:
            agent_scores[agent] = []
        agent_scores[agent].append(result.scores)

    per_agent: dict[str, dict[str, Decimal]] = {}
    for agent, scores in agent_scores.items():
        per_agent[agent] = {
            "relevance": compute_mean_decimal([s.relevance for s in scores]),
            "overall": compute_mean_decimal([s.overall for s in scores]),
        }
    return per_agent
