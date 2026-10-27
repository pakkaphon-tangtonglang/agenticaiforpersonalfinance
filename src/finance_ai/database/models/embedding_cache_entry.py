"""EmbeddingCacheEntry model: stored embedding vectors keyed by text hash."""

from sqlalchemy import LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column

from finance_ai.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class EmbeddingCacheEntry(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One embedded text chunk, so it is never sent to the provider twice.

    Render's free disk is wiped on every start, which would otherwise
    re-embed the whole knowledge base each time. Vectors live in Postgres
    instead and are reused across restarts and deploys.

    Example:
        >>> EmbeddingCacheEntry(cache_key="3f2a...", vector=b"...")
    """

    __tablename__ = "embedding_cache"

    cache_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
