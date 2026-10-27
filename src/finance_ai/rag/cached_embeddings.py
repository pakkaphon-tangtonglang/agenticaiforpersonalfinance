"""Embeddings wrapper that stores document vectors in the database.

Each chunk is embedded once, ever: restarts and redeploys (Render's disk is
wiped on every start) reuse the stored vectors with zero provider calls, so
free-tier quotas (e.g. Gemini's 1,000 embedding requests/day) are not spent
re-indexing unchanged documents. Search queries are not cached.
"""

import hashlib
from array import array
from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from sqlalchemy.orm import Session

from finance_ai.core.logging import get_logger
from finance_ai.database.crud.embedding_cache_crud import EmbeddingCacheCRUD

logger = get_logger(__name__)

_cache_crud = EmbeddingCacheCRUD()


def cache_key(namespace: str, text: str) -> str:
    """SHA-256 key for a text under one provider/model namespace.

    Example:
        >>> len(cache_key("google:models/gemini-embedding-001", "ภาษี"))
        64
    """
    return hashlib.sha256(f"{namespace}\n{text}".encode("utf-8")).hexdigest()


def encode_vector(vector: list[float]) -> bytes:
    """Pack a vector as float32 bytes (4 bytes per dimension).

    Example:
        >>> len(encode_vector([0.1, 0.2]))
        8
    """
    return array("f", vector).tobytes()


def decode_vector(data: bytes) -> list[float]:
    """Unpack float32 bytes into a vector.

    Example:
        >>> decode_vector(encode_vector([0.5]))
        [0.5]
    """
    values = array("f")
    values.frombytes(data)
    return values.tolist()


class CachedEmbeddings(Embeddings):
    """Embed documents through a database cache; queries pass straight through.

    Example:
        >>> embeddings = CachedEmbeddings(google_embeddings, session_factory, "google:m")
        >>> embeddings.embed_documents(["ค่าลดหย่อน"])  # second call: no API request
    """

    def __init__(
        self,
        underlying: Embeddings,
        session_factory: Callable[[], Session],
        namespace: str,
    ) -> None:
        """Wrap a provider's embeddings.

        Args:
            underlying: The real embedding provider.
            session_factory: Creates sessions for the cache table.
            namespace: Provider and model, so switching models never mixes vectors.
        """
        self._underlying = underlying
        self._session_factory = session_factory
        self._namespace = namespace

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return vectors for texts, embedding only those not cached yet.

        Args:
            texts: Document chunks.

        Returns:
            One vector per text, in input order.
        """
        keys = [cache_key(self._namespace, text) for text in texts]
        vectors = {key: decode_vector(data) for key, data in self._load(keys).items()}
        missing = [(key, text) for key, text in zip(keys, texts) if key not in vectors]
        if missing:
            new_vectors = self._underlying.embed_documents([text for _, text in missing])
            fresh = {key: vector for (key, _), vector in zip(missing, new_vectors)}
            self._save(fresh)
            vectors.update(fresh)
        logger.info("Embedding cache: %d hit, %d new", len(texts) - len(missing), len(missing))
        return [vectors[key] for key in keys]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (not cached).

        Args:
            text: Query text.

        Returns:
            The query vector.
        """
        return self._underlying.embed_query(text)

    def _load(self, keys: list[str]) -> dict[str, bytes]:
        """Read cached vectors; any database error means 'nothing cached'."""
        try:
            session = self._session_factory()
            try:
                return _cache_crud.get_vectors(session, keys)
            finally:
                session.close()
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Broad by design: a broken cache must never block indexing.
            logger.warning("Embedding cache read failed: %s", error)
            return {}

    def _save(self, vectors: dict[str, list[float]]) -> None:
        """Store new vectors; failures are logged and ignored."""
        try:
            session = self._session_factory()
            try:
                _cache_crud.add_vectors(
                    session, {key: encode_vector(vector) for key, vector in vectors.items()}
                )
                session.commit()
            finally:
                session.close()
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.warning("Embedding cache write failed: %s", error)
