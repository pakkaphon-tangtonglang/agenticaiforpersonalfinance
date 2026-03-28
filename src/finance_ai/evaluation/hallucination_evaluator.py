"""Anti-hallucination compliance evaluation module.

Stage 1: Rule-based checks (regex patterns, fact verification).
Stage 2: Optional LLM-as-Judge for nuanced claims.
"""

import json
import re
import time
from decimal import Decimal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from finance_ai.evaluation.models import (
    HallucinationAggregateResult,
    HallucinationCase,
    HallucinationDataset,
    HallucinationResult,
)

JUDGE_SYSTEM_PROMPT = """คุณเป็นผู้ตรวจสอบความถูกต้องของคำตอบระบบ AI การเงิน

ให้ตรวจสอบว่าคำตอบมีข้อมูลที่ขัดแย้งกับข้อเท็จจริงหรือไม่

ข้อเท็จจริง:
{facts}

คำตอบที่ต้องตรวจ:
{response}

ตอบเป็น JSON:
{{"violations": ["รายละเอียดข้อที่ผิด..."], "is_compliant": true/false}}"""


def check_forbidden_patterns(
    response: str,
    patterns: list[str],
) -> list[str]:
    """Check response for forbidden regex patterns.

    Args:
        response: Agent response text.
        patterns: List of regex patterns that indicate hallucination.

    Returns:
        List of matched forbidden patterns found.

    Example:
        >>> check_forbidden_patterns("70,000 บาท", ["70,000"])
        ['70,000']
    """
    violations = []
    for pattern in patterns:
        if re.search(pattern, response):
            violations.append(f"Forbidden pattern matched: {pattern}")
    return violations


def check_known_facts(
    response: str,
    known_facts: list[str],
) -> list[str]:
    """Verify known facts are not contradicted in the response.

    Simple check: each fact's key numbers should appear in the response
    if the response mentions the related topic.

    Args:
        response: Agent response text.
        known_facts: List of ground truth fact strings.

    Returns:
        List of potential fact violations.

    Example:
        >>> check_known_facts("ลดหย่อน 70,000", ["ลดหย่อนส่วนตัว 60,000 บาท"])
        []
    """
    violations = []
    for fact in known_facts:
        numbers = re.findall(r"[\d,]+\.?\d*", fact)
        for number in numbers:
            if number not in response and _topic_mentioned(fact, response):
                violations.append(f"Expected number {number} not found for: {fact}")
    return violations


def _topic_mentioned(fact: str, response: str) -> bool:
    """Check if the topic of a fact is mentioned in the response.

    Args:
        fact: A fact string containing a topic keyword.
        response: Agent response text.

    Returns:
        True if any Thai keyword from the fact appears in the response.
    """
    keywords = re.findall(r"[\u0E00-\u0E7F]+", fact)
    return any(kw in response for kw in keywords if len(kw) > 2)


def evaluate_with_llm_judge(
    response: str,
    known_facts: list[str],
    judge_model: BaseChatModel,
) -> list[str]:
    """Use LLM-as-Judge to check for hallucinations.

    Args:
        response: Agent response text.
        known_facts: Ground truth facts.
        judge_model: LLM to use as judge.

    Returns:
        List of violations identified by the judge.

    Example:
        >>> violations = evaluate_with_llm_judge(response, facts, model)
    """
    facts_text = "\n".join(f"- {f}" for f in known_facts)
    prompt = JUDGE_SYSTEM_PROMPT.format(facts=facts_text, response=response)
    result = judge_model.invoke(
        [
            SystemMessage(content="You are a fact-checking assistant."),
            HumanMessage(content=prompt),
        ]
    )
    return _parse_judge_violations(str(result.content))


def _parse_judge_violations(content: str) -> list[str]:
    """Parse violation list from judge LLM response.

    Args:
        content: Raw judge response (expected JSON).

    Returns:
        List of violation strings, empty if parsing fails.
    """
    try:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return list(data.get("violations", []))
    except (json.JSONDecodeError, AttributeError):
        return []


def evaluate_single_hallucination_case(
    case: HallucinationCase,
    agent_response: str,
    use_llm_judge: bool = False,
    judge_model: BaseChatModel | None = None,
) -> HallucinationResult:
    """Evaluate a single hallucination compliance case.

    Args:
        case: Hallucination test case.
        agent_response: The agent's response to check.
        use_llm_judge: Whether to use Stage 2 LLM judge.
        judge_model: LLM model for Stage 2 (required if use_llm_judge).

    Returns:
        HallucinationResult with compliance status.

    Example:
        >>> result = evaluate_single_hallucination_case(case, response)
    """
    start = time.perf_counter()
    violations = check_forbidden_patterns(agent_response, case.forbidden_patterns)
    violations.extend(check_known_facts(agent_response, case.known_facts))

    if use_llm_judge and judge_model is not None:
        llm_violations = evaluate_with_llm_judge(agent_response, case.known_facts, judge_model)
        violations.extend(llm_violations)

    latency = time.perf_counter() - start
    return HallucinationResult(
        case_id=case.case_id,
        query=case.query,
        category=case.category,
        violations_found=violations,
        is_compliant=len(violations) == 0,
        agent_response=agent_response,
        latency_seconds=latency,
    )


def evaluate_hallucination_dataset(
    dataset: HallucinationDataset,
    agent_responses: dict[str, str],
    use_llm_judge: bool = False,
    judge_model: BaseChatModel | None = None,
) -> HallucinationAggregateResult:
    """Evaluate all cases in a hallucination dataset.

    Args:
        dataset: Hallucination evaluation dataset.
        agent_responses: Dict mapping case_id to agent response.
        use_llm_judge: Whether to use Stage 2 LLM judge.
        judge_model: LLM model for Stage 2.

    Returns:
        HallucinationAggregateResult with compliance metrics.

    Example:
        >>> agg = evaluate_hallucination_dataset(dataset, responses)
    """
    results = []
    for case in dataset.cases:
        response = agent_responses.get(case.case_id, "")
        result = evaluate_single_hallucination_case(case, response, use_llm_judge, judge_model)
        results.append(result)
    return _aggregate_hallucination_results(results)


def _aggregate_hallucination_results(
    results: list[HallucinationResult],
) -> HallucinationAggregateResult:
    """Aggregate individual hallucination results.

    Args:
        results: List of individual hallucination results.

    Returns:
        HallucinationAggregateResult with summary metrics.
    """
    total = len(results)
    compliant = sum(1 for r in results if r.is_compliant)
    violations_by_cat: dict[str, int] = {}
    for result in results:
        if not result.is_compliant:
            violations_by_cat[result.category] = violations_by_cat.get(result.category, 0) + 1
    latencies = [r.latency_seconds for r in results]

    return HallucinationAggregateResult(
        total_cases=total,
        compliant_count=compliant,
        compliance_rate=(Decimal(compliant) / Decimal(max(total, 1))).quantize(Decimal("0.0001")),
        violations_by_category=violations_by_cat,
        mean_latency_seconds=sum(latencies) / max(len(latencies), 1),
        results=results,
    )
