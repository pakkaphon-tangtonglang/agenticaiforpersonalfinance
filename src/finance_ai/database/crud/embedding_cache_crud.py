"""CRUD operations for the EmbeddingCacheEntry model."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.models.embedding_cache_entry import EmbeddingCacheEntry


class EmbeddingCacheCRUD(BaseCRUD[EmbeddingCacheEntry]):
    """Read and write cached embedding vectors."""

    def __init__(self) -> None:
        """Initialize with the EmbeddingCacheEntry model."""
        super().__init__(EmbeddingCacheEntry)

    def get_vectors(self, session: Session, cache_keys: list[str]) -> dict[str, bytes]:
        """Return stored vectors for the given keys (missing keys are omitted).

        Args:
            session: Database session.
            cache_keys: Keys to look up.

        Returns:
            Mapping of cache_key to encoded vector.

        Example:
            >>> EmbeddingCacheCRUD().get_vectors(session, ["3f2a..."])
            {'3f2a...': b'...'}
        """
        if not cache_keys:
            return {}
        statement = select(EmbeddingCacheEntry.cache_key, EmbeddingCacheEntry.vector).where(
            EmbeddingCacheEntry.cache_key.in_(cache_keys)
        )
        return {row.cache_key: row.vector for row in session.execute(statement)}

    def add_vectors(self, session: Session, vectors: dict[str, bytes]) -> None:
        """Add vectors whose keys are not stored yet. Caller must commit.

        Args:
            session: Database session.
            vectors: Mapping of cache_key to encoded vector.

        Example:
            >>> EmbeddingCacheCRUD().add_vectors(session, {"3f2a...": b"..."})
            >>> session.commit()
        """
        existing = self.get_vectors(session, list(vectors))
        for key, vector in vectors.items():
            if key not in existing:
                session.add(EmbeddingCacheEntry(cache_key=key, vector=vector))
        session.flush()
