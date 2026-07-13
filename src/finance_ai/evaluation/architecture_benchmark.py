"""Multi-agent architecture benchmark: Hub-and-Spoke vs P2P vs Hierarchical.

Runs 5 test queries through each architecture, measuring wall-clock latency
and routing LLM call count. Used to demonstrate Hub-and-Spoke advantages.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

# ──────────────────────── Constants ────────────────────────

N_AGENTS: int = 6
LEVELS_HIERARCHICAL: int = 2
P2P_STARTING_AGENT: str = "expense"

DEFAULT_BENCHMARK_QUERIES: list[str] = [
    "คำนวณภาษีเงินเดือน 60,000 บาทต่อเดือน มีลูก 1 คน",
    "บันทึกค่ากาแฟ 80 บาท",
    "ดูราคาหุ้น PTT.BK",
    "อยากออมเงิน 200,000 บาทเพื่อดาวน์บ้าน",
    "สรุปรายงานการเงินเดือนนี้",
]

MACRO_CLASSIFIER_PROMPT: str = (
    "จำแนกคำถามเป็น 3 กลุ่มใหญ่:\n"
    '- "financial_operations": ภาษี, ค่าใช้จ่าย, รายรับ\n'
    '- "financial_analysis": คำแนะนำการเงิน, รายงาน, วิเคราะห์\n'
    '- "financial_planning": วางแผนการเงิน, เป้าหมาย, การลงทุน, ติดตามสินทรัพย์\n'
    'ตอบเป็น JSON เท่านั้น: {"macro": "<category>", "confidence": <0.0-1.0>}'
)

# ──────────────────────── Models ────────────────────────


class QueryBenchmarkResult(BaseModel):
    """Benchmark result for a single query through one architecture.

    Attributes:
        architecture: Architecture name.
        query: The query string.
        query_index: Index in the benchmark set.
        latency_ms: Total wall-clock time in milliseconds.
        routing_llm_calls: Number of LLM API calls used for routing only.
        agent_intent: Final intent dispatched to.
        hops: Number of agent hops (P2P only, otherwise 1).

    Example:
        >>> r = QueryBenchmarkResult(
        ...     architecture="hub_spoke", query="test", query_index=0,
        ...     latency_ms=1234.5, routing_llm_calls=1, agent_intent="tax", hops=1,
        ... )
    """

    architecture: Literal["hub_spoke", "p2p", "hierarchical"]
    query: str
    query_index: int
    latency_ms: float
    routing_llm_calls: int
    agent_intent: str
    hops: int = Field(default=1)


class ArchitectureSummary(BaseModel):
    """Aggregate benchmark results for one architecture.

    Attributes:
        architecture: Architecture name.
        mean_latency_ms: Mean wall-clock latency across all queries.
        routing_llm_calls_per_query: Average routing LLM calls per query.
        coupling_score: Number of agent-to-agent connections for N_AGENTS.
        per_query_results: Individual query results.

    Example:
        >>> s = ArchitectureSummary(
        ...     architecture="hub_spoke", mean_latency_ms=1000.0,
        ...     routing_llm_calls_per_query=1.0, coupling_score=6, per_query_results=[],
        ... )
    """

    architecture: str
    mean_latency_ms: float
    routing_llm_calls_per_query: float
    coupling_score: int
    per_query_results: list[QueryBenchmarkResult] = Field(default_factory=list)


class ArchitectureBenchmarkResult(BaseModel):
    """Top-level benchmark result comparing all three architectures.

    Attributes:
        hub_spoke: Hub-and-Spoke summary.
        p2p: Peer-to-Peer summary.
        hierarchical: Hierarchical summary.
        n_agents: Number of agents in the system.
        scalability_data: Theoretical coupling connections for N=2..10.
        benchmark_queries: The queries used in the benchmark.

    Example:
        >>> r = ArchitectureBenchmarkResult(
        ...     hub_spoke=summary, p2p=summary, hierarchical=summary,
        ...     n_agents=6, scalability_data={}, benchmark_queries=[],
        ... )
    """

    hub_spoke: ArchitectureSummary
    p2p: ArchitectureSummary
    hierarchical: ArchitectureSummary
    n_agents: int = N_AGENTS
    scalability_data: dict[str, list[int]] = Field(default_factory=dict)
    benchmark_queries: list[str] = Field(default_factory=list)


# ──────────────────────── Agent Dispatch Map ────────────────────────


def _get_agent_dispatch_map(
    model: BaseChatModel,
    user_id: str,
    db_factory: Optional[Callable[[], Session]],
) -> dict[str, Callable[[str], dict[str, Any]]]:
    """Build a map from intent string to agent executor.

    Args:
        model: LLM chat model.
        user_id: UUID of the user.
        db_factory: DB session factory.

    Returns:
        Dict mapping intent key to a callable that executes the agent.
    """
    from finance_ai.agents.router_agent import (  # noqa: PLC0415
        execute_asset_monitoring_agent,
        execute_expense_agent,
        execute_general_chat,
        execute_planning_agent,
        execute_recommendation_agent,
        execute_report_agent,
        execute_tax_agent,
    )

    kwargs: dict[str, Any] = {
        "chat_model": model,
        "user_id": user_id,
        "db_session_factory": db_factory,
    }

    return {
        "tax": lambda q: execute_tax_agent(q, **kwargs),
        "expense": lambda q: execute_expense_agent(q, **kwargs),
        "asset_monitoring": lambda q: execute_asset_monitoring_agent(q, **kwargs),
        "planning": lambda q: execute_planning_agent(q, **kwargs),
        "recommendation": lambda q: execute_recommendation_agent(q, **kwargs),
        "report": lambda q: execute_report_agent(q, **kwargs),
        "general": lambda q: execute_general_chat(q, **kwargs),
    }


# ──────────────────────── Hub-and-Spoke Runner ────────────────────────


def run_hub_spoke_benchmark(
    queries: list[str],
    model: BaseChatModel,
    db_factory: Optional[Callable[[], Session]],
    user_id: str,
) -> ArchitectureSummary:
    """Run the Hub-and-Spoke benchmark.

    Each query goes through `orchestrate_query` which does exactly
    1 routing LLM call (classify_query) then dispatches to one agent.

    Args:
        queries: List of test queries.
        model: LLM chat model.
        db_factory: DB session factory.
        user_id: UUID of the user.

    Returns:
        ArchitectureSummary for hub-and-spoke.

    Example:
        >>> summary = run_hub_spoke_benchmark(queries, model, None, "")
    """
    from finance_ai.agents.router_agent import orchestrate_query  # noqa: PLC0415

    results: list[QueryBenchmarkResult] = []
    for i, query in enumerate(queries):
        logger.info("Hub-and-Spoke query %d/%d: %s", i + 1, len(queries), query[:40])
        t0 = time.perf_counter()
        result = orchestrate_query(
            query,
            chat_model=model,
            user_id=user_id,
            db_session_factory=db_factory,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        results.append(
            QueryBenchmarkResult(
                architecture="hub_spoke",
                query=query,
                query_index=i,
                latency_ms=latency_ms,
                routing_llm_calls=1,
                agent_intent=result.get("intent", "unknown"),
                hops=1,
            )
        )

    return _build_summary("hub_spoke", results, N_AGENTS)


# ──────────────────────── P2P Runner ────────────────────────


def run_p2p_benchmark(
    queries: list[str],
    model: BaseChatModel,
    db_factory: Optional[Callable[[], Session]],
    user_id: str,
) -> ArchitectureSummary:
    """Run the Peer-to-Peer benchmark.

    Simulates P2P by starting all queries at the "expense" agent.
    If the query doesn't belong to that agent (classified as a different
    intent), the starting agent "forwards" it — incurring 1 extra routing
    LLM call (total routing_calls = 2, hops = 2).

    Agent execution is done via direct execute_*_agent() calls (not via
    orchestrate_query) to avoid triple-counting the routing classification.

    Args:
        queries: List of test queries.
        model: LLM chat model.
        db_factory: DB session factory.
        user_id: UUID of the user.

    Returns:
        ArchitectureSummary for P2P.

    Example:
        >>> summary = run_p2p_benchmark(queries, model, None, "")
    """
    from finance_ai.agents.router_agent import classify_query  # noqa: PLC0415

    dispatch = _get_agent_dispatch_map(model, user_id, db_factory)
    results: list[QueryBenchmarkResult] = []

    for i, query in enumerate(queries):
        logger.info("P2P query %d/%d: %s", i + 1, len(queries), query[:40])
        t0 = time.perf_counter()

        decision1 = classify_query(query, chat_model=model)
        routing_calls = 1
        hops = 1
        final_intent = decision1.intent

        if decision1.intent != P2P_STARTING_AGENT:
            decision2 = classify_query(query, chat_model=model)
            routing_calls = 2
            hops = 2
            final_intent = decision2.intent

        agent_fn = dispatch.get(final_intent, dispatch["general"])
        agent_fn(query)

        latency_ms = (time.perf_counter() - t0) * 1000
        results.append(
            QueryBenchmarkResult(
                architecture="p2p",
                query=query,
                query_index=i,
                latency_ms=latency_ms,
                routing_llm_calls=routing_calls,
                agent_intent=final_intent,
                hops=hops,
            )
        )

    coupling = N_AGENTS * (N_AGENTS - 1) // 2
    return _build_summary("p2p", results, coupling)


# ──────────────────────── Hierarchical Runner ────────────────────────


def _classify_macro(query: str, model: BaseChatModel) -> str:
    """Run Level-1 macro classification for Hierarchical architecture.

    Args:
        query: User query.
        model: LLM chat model.

    Returns:
        Macro category string.
    """
    messages = [
        SystemMessage(content=MACRO_CLASSIFIER_PROMPT),
        HumanMessage(content=query),
    ]
    response = model.invoke(messages)
    try:
        text = str(response.content).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        return str(data.get("macro", "financial_operations"))
    except (json.JSONDecodeError, KeyError, IndexError):
        return "financial_operations"


def run_hierarchical_benchmark(
    queries: list[str],
    model: BaseChatModel,
    db_factory: Optional[Callable[[], Session]],
    user_id: str,
) -> ArchitectureSummary:
    """Run the Hierarchical benchmark.

    Simulates 2-level routing:
    - Level 1: macro classifier (custom prompt, 1 LLM call)
    - Level 2: specific intent classifier (existing classify_query, 1 LLM call)
    Total routing_llm_calls = 2 for every query.

    Agent execution uses direct execute_*_agent() calls.

    Args:
        queries: List of test queries.
        model: LLM chat model.
        db_factory: DB session factory.
        user_id: UUID of the user.

    Returns:
        ArchitectureSummary for hierarchical.

    Example:
        >>> summary = run_hierarchical_benchmark(queries, model, None, "")
    """
    from finance_ai.agents.router_agent import classify_query  # noqa: PLC0415

    dispatch = _get_agent_dispatch_map(model, user_id, db_factory)
    results: list[QueryBenchmarkResult] = []

    for i, query in enumerate(queries):
        logger.info("Hierarchical query %d/%d: %s", i + 1, len(queries), query[:40])
        t0 = time.perf_counter()

        _classify_macro(query, model)
        decision = classify_query(query, chat_model=model)

        agent_fn = dispatch.get(decision.intent, dispatch["general"])
        agent_fn(query)

        latency_ms = (time.perf_counter() - t0) * 1000
        results.append(
            QueryBenchmarkResult(
                architecture="hierarchical",
                query=query,
                query_index=i,
                latency_ms=latency_ms,
                routing_llm_calls=2,
                agent_intent=decision.intent,
                hops=1,
            )
        )

    coupling = N_AGENTS + LEVELS_HIERARCHICAL
    return _build_summary("hierarchical", results, coupling)


# ──────────────────────── Scalability Data ────────────────────────


def _compute_scalability_data(
    n_range: range = range(2, 11),
) -> dict[str, list[int]]:
    """Compute theoretical coupling connections as number of agents grows.

    Args:
        n_range: Range of agent counts to compute.

    Returns:
        Dict with keys 'n_agents', 'p2p', 'hierarchical', 'hub_spoke'.

    Example:
        >>> data = _compute_scalability_data()
        >>> data["p2p"][4]  # N=6: 6×5/2 = 15
        15
    """
    data: dict[str, list[int]] = {
        "n_agents": [],
        "p2p": [],
        "hierarchical": [],
        "hub_spoke": [],
    }
    for n in n_range:
        data["n_agents"].append(n)
        data["p2p"].append(n * (n - 1) // 2)
        data["hierarchical"].append(n + LEVELS_HIERARCHICAL)
        data["hub_spoke"].append(n)
    return data


# ──────────────────────── Helpers ────────────────────────


def _build_summary(
    architecture: str,
    results: list[QueryBenchmarkResult],
    coupling_score: int,
) -> ArchitectureSummary:
    """Build ArchitectureSummary from individual query results.

    Args:
        architecture: Architecture name.
        results: Per-query results.
        coupling_score: Number of agent connections.

    Returns:
        ArchitectureSummary with computed means.
    """
    n = len(results)
    mean_latency = sum(r.latency_ms for r in results) / n if n else 0.0
    mean_calls = sum(r.routing_llm_calls for r in results) / n if n else 0.0
    return ArchitectureSummary(
        architecture=architecture,
        mean_latency_ms=mean_latency,
        routing_llm_calls_per_query=mean_calls,
        coupling_score=coupling_score,
        per_query_results=results,
    )


# ──────────────────────── Top-Level Entry ────────────────────────


def run_architecture_benchmark(
    queries: Optional[list[str]] = None,
    model: Optional[BaseChatModel] = None,
    db_factory: Optional[Callable[[], Session]] = None,
    user_id: str = "",
) -> ArchitectureBenchmarkResult:
    """Run the full architecture comparison benchmark.

    Runs all three architectures on the same set of queries and returns
    aggregate results for comparison.

    Args:
        queries: Test queries (defaults to DEFAULT_BENCHMARK_QUERIES).
        model: LLM chat model (defaults to create_chat_model()).
        db_factory: DB session factory (optional, no DB calls if None).
        user_id: UUID of the user (optional).

    Returns:
        ArchitectureBenchmarkResult with all three summaries.

    Example:
        >>> result = run_architecture_benchmark()
        >>> result.hub_spoke.mean_latency_ms
        1234.5
    """
    if queries is None:
        queries = DEFAULT_BENCHMARK_QUERIES
    if model is None:
        from finance_ai.agents.llm_factory import create_chat_model  # noqa: PLC0415

        model = create_chat_model()

    logger.info("Starting architecture benchmark with %d queries", len(queries))

    hub = run_hub_spoke_benchmark(queries, model, db_factory, user_id)
    p2p = run_p2p_benchmark(queries, model, db_factory, user_id)
    hier = run_hierarchical_benchmark(queries, model, db_factory, user_id)

    return ArchitectureBenchmarkResult(
        hub_spoke=hub,
        p2p=p2p,
        hierarchical=hier,
        n_agents=N_AGENTS,
        scalability_data=_compute_scalability_data(),
        benchmark_queries=queries,
    )
