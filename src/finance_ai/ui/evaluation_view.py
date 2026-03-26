"""Streamlit UI for running evaluations interactively."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from finance_ai.rag.vector_store import FinanceVectorStore

EVAL_DIMENSIONS: list[dict[str, str]] = [
    {
        "key": "routing",
        "label": "🎯 Intent Routing",
        "desc": "ทดสอบการจำแนก intent (35 cases)",
    },
    {
        "key": "rag",
        "label": "📚 RAG Retrieval",
        "desc": "ทดสอบคุณภาพการดึงเอกสาร (15 cases)",
    },
    {
        "key": "accuracy",
        "label": "🧮 Tax Accuracy",
        "desc": "ทดสอบความแม่นยำคำนวณภาษี (10 cases)",
    },
    {
        "key": "hallucination",
        "label": "🛡️ Hallucination",
        "desc": "ตรวจสอบการสร้างข้อมูลเท็จ (10 cases)",
    },
    {
        "key": "quality",
        "label": "⭐ Quality (LLM Judge)",
        "desc": "ให้คะแนนคุณภาพคำตอบ 1-5 (15 cases)",
    },
    {
        "key": "performance",
        "label": "⚡ Performance",
        "desc": "วัด latency และ cost (15 cases)",
    },
]


def render_evaluation_view(
    chat_model: BaseChatModel,
    session_factory: object,
) -> None:
    """Render the evaluation dashboard UI.

    Args:
        chat_model: The chat model to evaluate.
        session_factory: Database session factory.
    """
    st.subheader("🔬 Evaluation Framework")
    st.caption("ประเมินประสิทธิภาพระบบ AI แบบอัตโนมัติ 6 มิติ")

    _render_dimension_selector()
    _render_run_button(chat_model, session_factory)
    _render_results()


def _render_dimension_selector() -> None:
    """Render checkboxes for selecting evaluation dimensions."""
    st.markdown("#### เลือกมิติที่ต้องการประเมิน")

    col_left, col_right = st.columns(2)
    for i, dim in enumerate(EVAL_DIMENSIONS):
        target = col_left if i % 2 == 0 else col_right
        with target:
            st.checkbox(
                dim["label"],
                value=True,
                help=dim["desc"],
                key=f"eval_{dim['key']}",
            )


def _get_selected_dimensions() -> list[str]:
    """Return list of selected dimension keys.

    Returns:
        List of selected evaluation dimension keys.
    """
    return [
        dim["key"] for dim in EVAL_DIMENSIONS if st.session_state.get(f"eval_{dim['key']}", False)
    ]


def _render_run_button(
    chat_model: BaseChatModel,
    session_factory: object,
) -> None:
    """Render the run evaluation button.

    Args:
        chat_model: The chat model to evaluate.
        session_factory: Database session factory.
    """
    st.divider()
    selected = _get_selected_dimensions()

    if not selected:
        st.warning("กรุณาเลือกอย่างน้อย 1 มิติ")
        return

    label = f"▶️ เริ่มประเมิน ({len(selected)} มิติ)"
    if st.button(label, type="primary", use_container_width=True):
        _run_evaluation(chat_model, session_factory, selected)


def _run_evaluation(
    chat_model: BaseChatModel,
    session_factory: object,
    selected: list[str],
) -> None:
    """Execute selected evaluations with progress bar.

    Args:
        chat_model: The chat model to evaluate.
        session_factory: Database session factory.
        selected: List of dimension keys to evaluate.
    """
    from finance_ai.evaluation.runner import (  # noqa: PLC0415
        EvaluationRunner,
    )

    store = _get_vector_store()
    runner = EvaluationRunner(
        chat_model=chat_model,
        vector_store=store,
        llm_provider="",
        llm_model="",
        db_session_factory=session_factory,  # type: ignore[arg-type]
    )

    results: dict[str, object] = {}
    progress = st.progress(0, text="กำลังเริ่มประเมิน...")

    eval_methods = {
        "routing": ("🎯 Routing", runner.run_routing),
        "rag": ("📚 RAG", runner.run_rag),
        "accuracy": ("🧮 Accuracy", runner.run_tax_accuracy),
        "hallucination": ("🛡️ Hallucination", runner.run_hallucination),
        "performance": ("⚡ Performance", runner.run_performance),
    }

    for i, key in enumerate(selected):
        frac = (i + 1) / len(selected)
        if key == "quality":
            progress.progress(frac, text="⭐ Quality — ต้องใช้ Judge model")
            continue
        if key not in eval_methods:
            continue
        label, method = eval_methods[key]
        progress.progress(frac, text=f"กำลังประเมิน {label}...")
        try:
            results[key] = method()
        except Exception as exc:  # noqa: BLE001
            results[key] = str(exc)

    progress.progress(1.0, text="เสร็จสิ้น!")
    st.session_state["eval_results"] = results
    st.rerun()


def _get_vector_store() -> FinanceVectorStore:
    """Create or get cached vector store.

    Returns:
        FinanceVectorStore instance.
    """
    from finance_ai.rag.embedding_factory import (  # noqa: PLC0415
        create_embeddings,
    )
    from finance_ai.rag.vector_store import (  # noqa: PLC0415
        FinanceVectorStore as _Store,
    )

    return _Store(embeddings=create_embeddings())


def _render_results() -> None:
    """Render evaluation results if available."""
    results = st.session_state.get("eval_results")
    if not results:
        return

    st.divider()
    st.markdown("#### 📋 ผลการประเมิน")

    for key, result in results.items():
        if isinstance(result, str):
            st.error(f"**{key}**: {result}")
            continue
        _render_single_result(key, result)


def _render_single_result(key: str, result: object) -> None:
    """Render a single evaluation result as metrics.

    Args:
        key: Dimension key.
        result: Aggregate result object.
    """
    label_map = {
        "routing": "🎯 Intent Routing",
        "rag": "📚 RAG Retrieval",
        "accuracy": "🧮 Tax Accuracy",
        "hallucination": "🛡️ Hallucination",
        "quality": "⭐ Quality",
        "performance": "⚡ Performance",
    }
    with st.expander(label_map.get(key, key), expanded=True):
        _render_metrics_for(key, result)


def _render_metrics_for(key: str, result: object) -> None:
    """Render metric cards for a given result.

    Args:
        key: Dimension key.
        result: Aggregate result object.
    """
    if key == "routing":
        _render_routing(result)
    elif key == "rag":
        _render_rag(result)
    elif key == "accuracy":
        _render_accuracy(result)
    elif key == "hallucination":
        _render_hallucination(result)
    elif key == "performance":
        _render_performance(result)


def _render_routing(result: object) -> None:
    """Render routing evaluation metrics.

    Args:
        result: RoutingAggregateResult.
    """
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{float(result.accuracy) * 100:.1f}%")  # type: ignore[attr-defined]
    c2.metric("Correct", f"{result.correct_count}/{result.total_cases}")  # type: ignore[attr-defined]
    c3.metric("Latency", f"{result.mean_latency_seconds:.2f}s")  # type: ignore[attr-defined]


def _render_rag(result: object) -> None:
    """Render RAG evaluation metrics.

    Args:
        result: RAGAggregateResult.
    """
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision@k", f"{result.mean_precision_at_k:.2f}")  # type: ignore[attr-defined]
    c2.metric("Recall@k", f"{result.mean_recall_at_k:.2f}")  # type: ignore[attr-defined]
    c3.metric("MRR", f"{result.mean_reciprocal_rank:.2f}")  # type: ignore[attr-defined]
    c4.metric("Keyword Hit", f"{result.mean_keyword_hit_rate:.2f}")  # type: ignore[attr-defined]


def _render_accuracy(result: object) -> None:
    """Render tax accuracy metrics.

    Args:
        result: AccuracyAggregateResult.
    """
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{float(result.accuracy_rate) * 100:.1f}%")  # type: ignore[attr-defined]
    c2.metric("Within Tolerance", f"{result.within_tolerance_count}/{result.total_cases}")  # type: ignore[attr-defined]
    c3.metric("Latency", f"{result.mean_latency_seconds:.2f}s")  # type: ignore[attr-defined]


def _render_hallucination(result: object) -> None:
    """Render hallucination compliance metrics.

    Args:
        result: HallucinationAggregateResult.
    """
    c1, c2 = st.columns(2)
    c1.metric("Compliance", f"{float(result.compliance_rate) * 100:.1f}%")  # type: ignore[attr-defined]
    c2.metric("Compliant", f"{result.compliant_count}/{result.total_cases}")  # type: ignore[attr-defined]


def _render_performance(result: object) -> None:
    """Render performance metrics.

    Args:
        result: PerformanceAggregateResult.
    """
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Latency", f"{result.mean_total_latency:.2f}s")  # type: ignore[attr-defined]
    c2.metric("P50", f"{result.p50_latency:.2f}s")  # type: ignore[attr-defined]
    c3.metric("P95", f"{result.p95_latency:.2f}s")  # type: ignore[attr-defined]
    c4.metric("Total Cost", f"${result.total_estimated_cost_usd}")  # type: ignore[attr-defined]
