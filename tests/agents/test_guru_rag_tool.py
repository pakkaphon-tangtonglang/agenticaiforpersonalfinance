"""Tests for the guru knowledge search tool."""

# pylint: disable=redefined-outer-name

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from finance_ai.agents.guru_rag_tool import (
    get_guru_retriever,
    reset_guru_retriever,
    search_guru_knowledge,
)
from finance_ai.rag.document_models import RetrievalResult


@pytest.fixture(autouse=True)
def _reset_singleton() -> Generator[None, None, None]:
    """Make sure no retriever leaks between tests."""
    reset_guru_retriever()
    yield
    reset_guru_retriever()


@pytest.fixture
def buffett_result() -> RetrievalResult:
    """A single Buffett retrieval result."""
    return RetrievalResult(
        content="Be fearful when others are greedy",
        source_file="letter_2008.md",
        domain="guru",
        relevance_score=0.3,
        guru_id="buffett",
        source_url="https://example.com/2008.pdf",
    )


def _invoke_tool(query: str, guru_id: str) -> dict[str, Any]:
    """Call the tool the way ToolNode does, with guru_id injected from state."""
    result: dict[str, Any] = search_guru_knowledge.func(  # type: ignore[attr-defined]
        query=query, guru_id=guru_id
    )
    return result


class TestGapLogging:
    """search_guru_knowledge logs questions its sources cannot answer."""

    @patch("finance_ai.agents.guru_rag_tool.record_gap_if_needed")
    @patch("finance_ai.agents.guru_rag_tool.get_guru_retriever")
    def test_passes_results_and_threshold_to_gap_logger(
        self, mock_get_retriever: MagicMock, mock_record: MagicMock
    ) -> None:
        """The session factory from state and the configured threshold are used."""
        mock_get_retriever.return_value.retrieve_by_metadata.return_value = []
        factory = MagicMock()

        search_guru_knowledge.func(  # type: ignore[attr-defined]
            query="ทองคำ", guru_id="buffett", db_session_factory=factory
        )

        args = mock_record.call_args.args
        assert args[:4] == (factory, "buffett", "ทองคำ", [])
        assert args[4] == pytest.approx(1.0)


class TestSearchGuruKnowledge:
    """Tests for search_guru_knowledge."""

    @patch("finance_ai.agents.guru_rag_tool.get_guru_retriever")
    def test_filters_by_guru_id_from_state(
        self, mock_get_retriever: MagicMock, buffett_result: RetrievalResult
    ) -> None:
        """Only the active guru's sources are searched."""
        mock_get_retriever.return_value.retrieve_by_metadata.return_value = [buffett_result]

        result = _invoke_tool("ตลาดตก", "buffett")

        mock_get_retriever.return_value.retrieve_by_metadata.assert_called_once_with(
            "ตลาดตก", {"guru_id": "buffett"}
        )
        assert result["results"][0]["guru_id"] == "buffett"
        assert "https://example.com/2008.pdf" in str(result["context"])

    @patch("finance_ai.agents.guru_rag_tool.get_guru_retriever")
    def test_no_results_returns_not_found_message(self, mock_get_retriever: MagicMock) -> None:
        """Empty retrieval tells the model to say the sources don't cover it."""
        mock_get_retriever.return_value.retrieve_by_metadata.return_value = []

        result = _invoke_tool("คริปโต", "buffett")

        assert result["results"] == []
        assert "ไม่พบในแหล่งข้อมูลของกูรู" in str(result["context"])

    def test_missing_guru_id_returns_error(self) -> None:
        """Without a guru in state the tool refuses rather than searching everyone."""
        result = _invoke_tool("อะไรก็ได้", "")

        assert "error" in result

    def test_guru_id_is_hidden_from_the_model(self) -> None:
        """The LLM-facing schema exposes only the query argument."""
        llm_schema = convert_to_openai_tool(search_guru_knowledge)["function"]["parameters"]

        assert set(llm_schema["properties"]) == {"query"}


class TestGetGuruRetriever:
    """Tests for the guru retriever singleton."""

    @patch("finance_ai.agents.guru_rag_tool._create_guru_retriever_from_settings")
    def test_creates_once_and_reuses(self, mock_create: MagicMock) -> None:
        """The retriever is built once per process."""
        first = get_guru_retriever()
        second = get_guru_retriever()

        assert first is second
        mock_create.assert_called_once()

    @patch("finance_ai.rag.embedding_factory.create_embeddings")
    @patch("finance_ai.rag.vector_store.FinanceVectorStore")
    def test_uses_guru_collection_from_settings(
        self, mock_store_class: MagicMock, _mock_embeddings: MagicMock
    ) -> None:
        """The guru collection is separate from finance_knowledge."""
        get_guru_retriever()

        assert mock_store_class.call_args.kwargs["collection_name"] == "guru_knowledge"
