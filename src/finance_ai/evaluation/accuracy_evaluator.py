"""Agent response accuracy evaluation module.

Evaluates tax agent accuracy by comparing LLM responses against
ground truth from pure tax calculation functions.

Supports two modes:
- End-to-end: uses production graph via execute_tax_agent()
- Forced tool: uses eval-only graph that forces calculate_thai_tax
"""

import re
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from sqlalchemy.orm import Session

from finance_ai.agents.prompts import TAX_AGENT_SYSTEM_PROMPT
from finance_ai.agents.router_agent import execute_tax_agent
from finance_ai.agents.schemas import TaxAgentState
from finance_ai.agents.graph_utils import should_continue
from finance_ai.agents.tax_tools import calculate_thai_tax
from finance_ai.evaluation.metrics import extract_thai_number
from finance_ai.evaluation.models import (
    AccuracyAggregateResult,
    TaxAccuracyCase,
    TaxAccuracyDataset,
    TaxAccuracyResult,
)
from finance_ai.tools.tax_calculator import TaxCalculationResult, calculate_tax

EVAL_TAX_TOOLS = [calculate_thai_tax]

ZERO_TAX_PATTERNS = [
    r"ไม่ต้องเสียภาษี",
    r"ไม่ต้องจ่ายภาษี",
    r"ไม่ต้องชำระภาษี",
    r"ภาษี[:\s*]*0(?:\.0+)?\s*บาท",
]

TAX_PATTERNS = [
    r"ภาษีที่ต้องจ่าย[^\d\n]{0,30}([\d,]+\.?\d*)\s*บาท",
    r"ภาษีที่ต้องชำระ[^\d\n]{0,30}([\d,]+\.?\d*)\s*บาท",
    r"ภาษีเงินได้[^\d\n]{0,30}([\d,]+\.?\d*)\s*บาท",
    r"รวมภาษี[^\d\n]{0,30}([\d,]+\.?\d*)\s*บาท",
    r"ภาษีสุทธิ[:\s*]{0,10}([\d,]+\.?\d*)\s*บาท",
    r"ภาษีรวม[:\s*]{0,10}([\d,]+\.?\d*)\s*บาท",
    r"ต้องจ่ายภาษี[:\s*]{0,10}([\d,]+\.?\d*)\s*บาท",
    r"จำนวนภาษี[:\s*]{0,10}([\d,]+\.?\d*)\s*บาท",
    r"ภาษี[:\s*]{0,10}([\d,]+\.?\d*)\s*บาท",
]


def compute_tax_ground_truth(case: TaxAccuracyCase) -> TaxCalculationResult:
    """Compute ground truth tax using pure calculator functions.

    Args:
        case: Tax accuracy test case with income and deductions.

    Returns:
        TaxCalculationResult from the pure calculator.

    Example:
        >>> ground_truth = compute_tax_ground_truth(case)
    """
    return calculate_tax(
        gross_income=case.gross_income,
        deductions_by_type=case.deductions_by_type,
        withholding_tax_paid=case.withholding_tax_paid,
    )


def extract_tax_from_response(response: str) -> Decimal | None:
    """Extract tax amount from a Thai agent response.

    Checks explicit tax amount patterns first, then falls back to
    zero-tax phrasing. Ordering matters: agents often explain that the
    first 150,000 THB band is tax-free before stating a non-zero tax,
    so zero phrasing alone must not override an explicit amount.

    Args:
        response: Agent response text in Thai.

    Returns:
        Extracted tax amount as Decimal, or None if not found.

    Example:
        >>> extract_tax_from_response("ภาษีที่ต้องจ่าย 29,000 บาท")
        Decimal('29000')
    """
    for pattern in TAX_PATTERNS:
        result = extract_thai_number(response, pattern)
        if result is not None:
            return result
    for pattern in ZERO_TAX_PATTERNS:
        if re.search(pattern, response):
            return Decimal("0")
    return None


def evaluate_single_tax_case(
    model: BaseChatModel,
    case: TaxAccuracyCase,
    db_session_factory: Callable[[], Session] | None = None,
) -> TaxAccuracyResult:
    """Evaluate a single tax accuracy case.

    Sends the query to the tax agent, extracts the numeric answer,
    and compares against ground truth from the pure calculator.

    Args:
        model: Chat model for the tax agent.
        case: Tax accuracy test case.
        db_session_factory: Optional DB session factory.

    Returns:
        TaxAccuracyResult with accuracy comparison.

    Example:
        >>> result = evaluate_single_tax_case(model, case)
    """
    start = time.perf_counter()
    agent_result = execute_tax_agent(
        query=case.query,
        chat_model=model,
        db_session_factory=db_session_factory,
    )
    latency = time.perf_counter() - start

    response = agent_result.get("response", "")
    extracted = extract_tax_from_response(response)

    return _build_tax_result(case, extracted, response, latency)


def _build_tax_result(
    case: TaxAccuracyCase,
    extracted: Decimal | None,
    response: str,
    latency: float,
) -> TaxAccuracyResult:
    """Build a TaxAccuracyResult from extraction results.

    Args:
        case: Original test case.
        extracted: Extracted tax amount (may be None).
        response: Raw agent response.
        latency: Time taken in seconds.

    Returns:
        TaxAccuracyResult with error calculations.
    """
    if extracted is not None:
        error = abs(extracted - case.expected_total_tax)
        within_tolerance = error <= case.tolerance_thb
    else:
        error = None
        within_tolerance = False

    return TaxAccuracyResult(
        case_id=case.case_id,
        query=case.query,
        expected_total_tax=case.expected_total_tax,
        extracted_total_tax=extracted,
        absolute_error_thb=error,
        is_within_tolerance=within_tolerance,
        agent_response=response,
        latency_seconds=latency,
    )


def evaluate_tax_accuracy_dataset(
    model: BaseChatModel,
    dataset: TaxAccuracyDataset,
    db_session_factory: Callable[[], Session] | None = None,
) -> AccuracyAggregateResult:
    """Evaluate all cases in a tax accuracy dataset.

    Args:
        model: Chat model for the tax agent.
        dataset: Tax accuracy evaluation dataset.
        db_session_factory: Optional DB session factory.

    Returns:
        AccuracyAggregateResult with aggregated metrics.

    Example:
        >>> agg = evaluate_tax_accuracy_dataset(model, dataset)
    """
    results = [evaluate_single_tax_case(model, case, db_session_factory) for case in dataset.cases]
    return _aggregate_tax_results(results)


def _aggregate_tax_results(
    results: list[TaxAccuracyResult],
) -> AccuracyAggregateResult:
    """Aggregate individual tax accuracy results.

    Args:
        results: List of individual tax results.

    Returns:
        AccuracyAggregateResult with summary metrics.
    """
    total = len(results)
    within = sum(1 for r in results if r.is_within_tolerance)
    errors = [r.absolute_error_thb for r in results if r.absolute_error_thb is not None]
    mean_error = sum(errors, Decimal("0")) / Decimal(max(len(errors), 1)) if errors else None
    latencies = [r.latency_seconds for r in results]

    return AccuracyAggregateResult(
        agent_type="tax",
        total_cases=total,
        within_tolerance_count=within,
        accuracy_rate=(Decimal(within) / Decimal(max(total, 1))).quantize(Decimal("0.0001")),
        mean_absolute_error_thb=mean_error,
        mean_latency_seconds=sum(latencies) / max(len(latencies), 1),
        results=results,
    )


# --- Eval-only forced tool mode ---


def _build_eval_tax_graph(chat_model: BaseChatModel) -> Any:
    """Build eval-only graph that forces calculate_thai_tax.

    Binds only [calculate_thai_tax] on first turn so
    tool_choice='any' forces exactly that tool.
    Does not affect the production tax agent graph.

    Args:
        chat_model: LangChain ChatModel to use.

    Returns:
        Compiled LangGraph StateGraph.
    """
    model_force = chat_model.bind_tools(EVAL_TAX_TOOLS, tool_choice="any")
    model_respond = chat_model.bind_tools(EVAL_TAX_TOOLS)

    def first_turn(state: TaxAgentState) -> dict[str, Any]:
        """Force call calculate_thai_tax on first turn."""
        msgs = [SystemMessage(content=TAX_AGENT_SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [model_force.invoke(msgs)]}

    def respond(state: TaxAgentState) -> dict[str, Any]:
        """Format tool results without forcing tool choice."""
        msgs = [SystemMessage(content=TAX_AGENT_SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [model_respond.invoke(msgs)]}

    graph = StateGraph(TaxAgentState)
    graph.add_node("first_turn", first_turn)
    graph.add_node("tools", ToolNode(EVAL_TAX_TOOLS))
    graph.add_node("respond", respond)
    graph.set_entry_point("first_turn")
    graph.add_edge("first_turn", "tools")
    graph.add_edge("tools", "respond")
    graph.add_conditional_edges(
        "respond",
        should_continue,
        {"tools": "tools", "end": END},
    )
    return graph.compile()


def _execute_eval_tax_agent(
    query: str,
    chat_model: BaseChatModel,
    db_session_factory: Callable[[], Session] | None = None,
) -> str:
    """Execute eval-only tax graph and return response text.

    Args:
        query: User query about tax calculation.
        chat_model: LangChain ChatModel to use.
        db_session_factory: Optional DB session factory.

    Returns:
        Agent response content as string.
    """
    graph = _build_eval_tax_graph(chat_model)
    result = graph.invoke(
        {
            "messages": [("user", query)],
            "user_id": "",
            "db_session_factory": db_session_factory,
        }
    )
    return str(result["messages"][-1].content)


def evaluate_single_tax_case_forced(
    model: BaseChatModel,
    case: TaxAccuracyCase,
    db_session_factory: Callable[[], Session] | None = None,
) -> TaxAccuracyResult:
    """Evaluate a tax case using forced tool calling.

    Uses eval-only graph that binds only calculate_thai_tax,
    ensuring the tool is always called on first turn.

    Args:
        model: Chat model for the tax agent.
        case: Tax accuracy test case.
        db_session_factory: Optional DB session factory.

    Returns:
        TaxAccuracyResult with accuracy comparison.
    """
    start = time.perf_counter()
    response = _execute_eval_tax_agent(
        query=case.query,
        chat_model=model,
        db_session_factory=db_session_factory,
    )
    latency = time.perf_counter() - start

    extracted = extract_tax_from_response(response)
    return _build_tax_result(case, extracted, response, latency)


def evaluate_tax_accuracy_dataset_forced(
    model: BaseChatModel,
    dataset: TaxAccuracyDataset,
    db_session_factory: Callable[[], Session] | None = None,
) -> AccuracyAggregateResult:
    """Evaluate all cases using forced tool calling.

    Args:
        model: Chat model for the tax agent.
        dataset: Tax accuracy evaluation dataset.
        db_session_factory: Optional DB session factory.

    Returns:
        AccuracyAggregateResult with aggregated metrics.
    """
    results = [
        evaluate_single_tax_case_forced(model, case, db_session_factory) for case in dataset.cases
    ]
    return _aggregate_tax_results(results)
