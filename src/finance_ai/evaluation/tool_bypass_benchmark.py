"""Benchmark comparing DeepSeek tax accuracy vs Python calculate_thai_tax tool.

Demonstrates the tool-bypass problem: when a LLM computes Thai income tax
directly from its training knowledge (no tool), the results are often wrong.

Compares two modes:
- No-tool mode: LLM answers with no tools available (pure LLM math)
- Tool ground truth: calculate_tax() Python function (always accurate)
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from finance_ai.agents.schemas import TaxAgentState
from finance_ai.evaluation.accuracy_evaluator import extract_tax_from_response
from finance_ai.tools.tax_calculator import calculate_tax

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

BYPASS_TEST_CASES: list[dict[str, Any]] = [
    {
        "case_id": "bypass_001",
        "query": "คำนวณภาษี เงินเดือน 50,000 บาทต่อเดือน ค่าลดหย่อนส่วนตัว 60,000",
        "expected_tax": Decimal("21500"),
        "tolerance": Decimal("500"),
        "description": "รายได้ปานกลาง ลดหย่อนส่วนตัวเท่านั้น",
        "gross_income": Decimal("600000"),
        "deductions": {"personal_allowance": Decimal("60000")},
    },
    {
        "case_id": "bypass_002",
        "query": "รายได้ 1,200,000 บาท ลดหย่อนส่วนตัว 60,000 RMF 200,000 คำนวณภาษีให้หน่อย",
        "expected_tax": Decimal("83000"),
        "tolerance": Decimal("1000"),
        "description": "รายได้สูง มี RMF",
        "gross_income": Decimal("1200000"),
        "deductions": {"personal_allowance": Decimal("60000"), "rmf": Decimal("200000")},
    },
    {
        "case_id": "bypass_003",
        "query": "รายได้ 2,000,000 บาท ลดหย่อนส่วนตัว 60,000 ประกันชีวิต 100,000 RMF 500,000 ภาษีเท่าไหร่",
        "expected_tax": Decimal("175000"),
        "tolerance": Decimal("2000"),
        "description": "รายได้สูงมาก หลายลดหย่อน",
        "gross_income": Decimal("2000000"),
        "deductions": {
            "personal_allowance": Decimal("60000"),
            "life_insurance": Decimal("100000"),
            "rmf": Decimal("500000"),
        },
    },
    {
        "case_id": "bypass_004",
        "query": "รายได้ 5,000,000 บาท ค่าลดหย่อนส่วนตัว 60,000 ต้องเสียภาษีเท่าไหร่",
        "expected_tax": Decimal("1217000"),
        "tolerance": Decimal("5000"),
        "description": "รายได้สูงมาก bracket 35%",
        "gross_income": Decimal("5000000"),
        "deductions": {"personal_allowance": Decimal("60000")},
    },
    {
        "case_id": "bypass_005",
        "query": "เงินเดือน 30,000 ค่าลดหย่อนส่วนตัว 60,000 ประกันสังคม 9,000 คำนวณภาษี",
        "expected_tax": Decimal("2050"),
        "tolerance": Decimal("500"),
        "description": "รายได้น้อย มีประกันสังคม",
        "gross_income": Decimal("360000"),
        "deductions": {
            "personal_allowance": Decimal("60000"),
            "social_security": Decimal("9000"),
        },
    },
    {
        "case_id": "bypass_006",
        "query": "รายได้ 800,000 ลดหย่อนส่วนตัว 60,000 SSF 200,000 ประกันสุขภาพ 15,000 ภาษีเท่าไหร่",
        "expected_tax": Decimal("20000"),
        "tolerance": Decimal("1000"),
        "description": "รายได้ปานกลาง มี SSF และประกันสุขภาพ",
        "gross_income": Decimal("800000"),
        "deductions": {
            "personal_allowance": Decimal("60000"),
            "ssf": Decimal("200000"),
            "health_insurance": Decimal("15000"),
        },
    },
    {
        "case_id": "bypass_007",
        "query": "รายได้ 3,000,000 บาท ลดหย่อนส่วนตัว 60,000 RMF 500,000 SSF 200,000 คำนวณภาษี",
        "expected_tax": Decimal("407000"),
        "tolerance": Decimal("5000"),
        "description": "รายได้สูงมาก RMF + SSF เต็มขั้น",
        "gross_income": Decimal("3000000"),
        "deductions": {
            "personal_allowance": Decimal("60000"),
            "rmf": Decimal("500000"),
            "ssf": Decimal("200000"),
        },
    },
    {
        "case_id": "bypass_008",
        "query": "เงินเดือน 70,000 ลดหย่อนส่วนตัว 60,000 ลูก 2 คน ประกันสังคม 9,000 ประกันชีวิต 50,000",
        "expected_tax": Decimal("41150"),
        "tolerance": Decimal("1000"),
        "description": "ครอบครัว หลายลดหย่อน",
        "gross_income": Decimal("840000"),
        "deductions": {
            "personal_allowance": Decimal("60000"),
            "child": Decimal("60000"),
            "social_security": Decimal("9000"),
            "life_insurance": Decimal("50000"),
        },
    },
]


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ToolBypassCaseResult(BaseModel):
    """Result for a single tax case in no-tool vs tool comparison.

    Attributes:
        case_id: Unique case identifier.
        description: Human-readable case description.
        query: The tax query sent to the model.
        expected_tax: Ground truth from calculate_tax() in THB.
        notool_response: Raw LLM response with no tools available.
        notool_extracted_tax: Tax extracted from no-tool response.
        notool_within_tolerance: Whether no-tool answer is accurate.
        notool_latency_seconds: Time for no-tool run.
        tool_tax: Tax computed by calculate_tax() Python function.
        tool_within_tolerance: Whether tool answer is accurate.
    """

    case_id: str
    description: str
    query: str
    expected_tax: Decimal
    notool_response: str = Field(default="")
    notool_extracted_tax: Decimal | None = None
    notool_within_tolerance: bool = False
    notool_latency_seconds: float = 0.0
    tool_tax: Decimal = Decimal("0")
    tool_within_tolerance: bool = False


class ToolBypassBenchmarkResult(BaseModel):
    """Aggregate results for the tool-bypass benchmark.

    Attributes:
        model_name: Name of the LLM tested in no-tool mode.
        cases: Per-case results.
        notool_accuracy_pct: Accuracy when LLM calculates without tool.
        tool_accuracy_pct: Accuracy of calculate_tax() Python function.
    """

    model_name: str
    cases: list[ToolBypassCaseResult]
    notool_accuracy_pct: float = 0.0
    tool_accuracy_pct: float = 0.0


# ---------------------------------------------------------------------------
# No-tool graph
# ---------------------------------------------------------------------------

_NO_TOOL_PROMPT = (
    "คุณเป็นผู้เชี่ยวชาญด้านภาษีไทย ตอบคำถามเกี่ยวกับภาษีเงินได้บุคคลธรรมดา "
    "คำนวณและตอบโดยละเอียด แสดงตัวเลขภาษีที่ต้องจ่ายในหน่วย บาท"
)


def _build_notool_graph(chat_model: BaseChatModel) -> Any:
    """Build a minimal graph with NO tools — LLM must calculate by itself.

    Args:
        chat_model: LangChain ChatModel to use.

    Returns:
        Compiled LangGraph StateGraph.
    """

    def notool_node(state: TaxAgentState) -> dict[str, Any]:
        """LLM node with no tools available.

        Args:
            state: Current agent state with messages.

        Returns:
            Dict with updated messages list.
        """
        msgs = [SystemMessage(content=_NO_TOOL_PROMPT)] + state["messages"]
        return {"messages": [chat_model.invoke(msgs)]}

    graph = StateGraph(TaxAgentState)
    graph.add_node("llm", notool_node)
    graph.set_entry_point("llm")
    graph.add_edge("llm", END)
    return graph.compile()


def _run_notool(graph: Any, query: str) -> tuple[str, float]:
    """Invoke no-tool graph and return response text and latency.

    Args:
        graph: Compiled LangGraph StateGraph.
        query: Tax query to evaluate.

    Returns:
        Tuple of (response_text, latency_seconds).
    """
    start = time.perf_counter()
    try:
        result = graph.invoke(
            {
                "messages": [("user", query)],
                "user_id": "",
                "db_session_factory": None,
            }
        )
        latency = time.perf_counter() - start
        messages = result.get("messages", [])
        response = str(messages[-1].content) if messages else ""
        return response, latency
    except Exception as exc:  # noqa: BLE001
        latency = time.perf_counter() - start
        return f"ERROR: {exc}", latency


# ---------------------------------------------------------------------------
# Ground truth via Python tool
# ---------------------------------------------------------------------------


def _compute_tool_tax(case: dict[str, Any]) -> Decimal:
    """Compute ground truth tax using the Python calculate_tax function.

    Args:
        case: Case dict with gross_income and deductions.

    Returns:
        Tax amount in THB from the Python calculator.
    """
    result = calculate_tax(
        gross_income=case["gross_income"],
        deductions_by_type=case["deductions"],
    )
    return result.total_tax


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def _evaluate_case(
    case: dict[str, Any],
    notool_graph: Any,
) -> ToolBypassCaseResult:
    """Run one test case through no-tool graph and Python tool.

    Args:
        case: Case dict with query, expected_tax, gross_income, deductions.
        notool_graph: Graph with no tools — LLM calculates by itself.

    Returns:
        ToolBypassCaseResult with both mode results.
    """
    tolerance = case["tolerance"]

    notool_resp, notool_latency = _run_notool(notool_graph, case["query"])
    notool_extracted = extract_tax_from_response(notool_resp)
    notool_ok = (
        notool_extracted is not None and abs(notool_extracted - case["expected_tax"]) <= tolerance
    )

    tool_tax = _compute_tool_tax(case)
    tool_ok = abs(tool_tax - case["expected_tax"]) <= tolerance

    return ToolBypassCaseResult(
        case_id=case["case_id"],
        description=case["description"],
        query=case["query"],
        expected_tax=case["expected_tax"],
        notool_response=notool_resp,
        notool_extracted_tax=notool_extracted,
        notool_within_tolerance=notool_ok,
        notool_latency_seconds=notool_latency,
        tool_tax=tool_tax,
        tool_within_tolerance=tool_ok,
    )


def run_tool_bypass_benchmark(
    chat_model: BaseChatModel,
    model_name: str,
    cases: list[dict[str, Any]] | None = None,
) -> ToolBypassBenchmarkResult:
    """Run tool-bypass benchmark: LLM self-calculation vs Python calculate_tax.

    Args:
        chat_model: LangChain ChatModel to test in no-tool mode.
        model_name: Display name for the model (used in report).
        cases: Optional custom test cases (defaults to BYPASS_TEST_CASES).

    Returns:
        ToolBypassBenchmarkResult with per-case and aggregate metrics.

    Example:
        >>> result = run_tool_bypass_benchmark(model, "deepseek-chat")
    """
    if cases is None:
        cases = BYPASS_TEST_CASES

    notool_graph = _build_notool_graph(chat_model)

    results: list[ToolBypassCaseResult] = []
    for case in cases:
        print(f"  [{case['case_id']}] running...")
        result = _evaluate_case(case, notool_graph)
        print(
            f"    No-tool : extracted={result.notool_extracted_tax}, ok={result.notool_within_tolerance}"
        )
        print(f"    Tool    : result={result.tool_tax}, ok={result.tool_within_tolerance}")
        results.append(result)

    total = len(results)
    notool_ok = sum(1 for r in results if r.notool_within_tolerance)
    tool_ok = sum(1 for r in results if r.tool_within_tolerance)

    return ToolBypassBenchmarkResult(
        model_name=model_name,
        cases=results,
        notool_accuracy_pct=notool_ok / max(total, 1) * 100,
        tool_accuracy_pct=tool_ok / max(total, 1) * 100,
    )
