"""Tests for the Postgres/SQLite-backed embedding cache."""

# pylint: disable=redefined-outer-name,unused-argument,protected-access

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from finance_ai.database.crud.embedding_cache_crud import EmbeddingCacheCRUD
from finance_ai.rag.cached_embeddings import (
    CachedEmbeddings,
    cache_key,
    decode_vector,
    encode_vector,
)


@pytest.fixture
def provider() -> MagicMock:
    """Underlying embeddings that return [len(text), 1.0] per text."""
    mock = MagicMock()
    mock.embed_documents.side_effect = lambda texts: [[float(len(t)), 1.0] for t in texts]
    mock.embed_query.return_value = [9.0, 9.0]
    return mock


class TestVectorEncoding:
    """Tests for the compact float32 storage format."""

    def test_round_trip(self) -> None:
        """Vectors survive encode -> decode (float32 precision)."""
        assert decode_vector(encode_vector([0.5, -1.25, 3.0])) == [0.5, -1.25, 3.0]

    def test_cache_key_depends_on_namespace_and_text(self) -> None:
        """Different models or texts never share a cache entry."""
        assert cache_key("google:a", "x") != cache_key("google:b", "x")
        assert cache_key("google:a", "x") != cache_key("google:a", "y")
        assert len(cache_key("google:a", "x")) == 64


class TestCachedEmbeddings:
    """Tests for CachedEmbeddings."""

    def test_second_call_uses_cache(
        self, provider: MagicMock, db_session_factory: Callable[[], Session]
    ) -> None:
        """Texts embedded once are never sent to the provider again."""
        embeddings = CachedEmbeddings(provider, db_session_factory, namespace="google:m")

        first = embeddings.embed_documents(["aa", "bbb"])
        second = embeddings.embed_documents(["aa", "bbb"])

        assert first == second == [[2.0, 1.0], [3.0, 1.0]]
        provider.embed_documents.assert_called_once_with(["aa", "bbb"])

    def test_only_missing_texts_are_embedded_and_order_is_kept(
        self, provider: MagicMock, db_session_factory: Callable[[], Session]
    ) -> None:
        """A mixed batch embeds only the new text and returns vectors in input order."""
        embeddings = CachedEmbeddings(provider, db_session_factory, namespace="google:m")
        embeddings.embed_documents(["aa"])

        result = embeddings.embed_documents(["cccc", "aa"])

        assert result == [[4.0, 1.0], [2.0, 1.0]]
        assert provider.embed_documents.call_args.args[0] == ["cccc"]

    def test_queries_are_not_cached(
        self, provider: MagicMock, db_session_factory: Callable[[], Session]
    ) -> None:
        """Search queries go straight to the provider."""
        embeddings = CachedEmbeddings(provider, db_session_factory, namespace="google:m")

        assert embeddings.embed_query("q") == [9.0, 9.0]
        provider.embed_query.assert_called_once_with("q")

    def test_database_failure_falls_back_to_provider(self, provider: MagicMock) -> None:
        """A broken cache never blocks indexing."""
        broken_factory = MagicMock(side_effect=RuntimeError("db down"))
        embeddings = CachedEmbeddings(provider, broken_factory, namespace="google:m")

        assert embeddings.embed_documents(["aa"]) == [[2.0, 1.0]]


class TestEmbeddingCacheCRUD:
    """Tests for EmbeddingCacheCRUD."""

    def test_add_skips_existing_keys(self, test_session: Session) -> None:
        """Re-adding a key keeps one row and does not raise."""
        crud = EmbeddingCacheCRUD()
        crud.add_vectors(test_session, {"k1": b"a"})
        test_session.commit()
        crud.add_vectors(test_session, {"k1": b"b", "k2": b"c"})
        test_session.commit()

        assert crud.get_vectors(test_session, ["k1", "k2", "missing"]) == {"k1": b"a", "k2": b"c"}

    def test_get_with_no_keys(self, test_session: Session) -> None:
        """An empty lookup returns an empty dict."""
        assert EmbeddingCacheCRUD().get_vectors(test_session, []) == {}
