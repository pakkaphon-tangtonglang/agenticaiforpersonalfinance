"""Retrievers compared in the study: four embedding models and BM25.

Every dense model gets an exact cosine index over the same chunks (the
production Chroma index is approximate HNSW; exact search removes that noise
from the comparison). Document vectors are cached on disk per model.
"""

import hashlib
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np
from pydantic import BaseModel

from finance_ai.evaluation.embedding_study.corpus import Chunk

E5_QUERY_PREFIX = (
    "Instruct: Given a Thai personal finance question, retrieve the passage that answers it\n"
    "Query: "
)


class Retriever(Protocol):
    """Anything that indexes chunks and ranks them for a query."""

    name: str

    def index(self, chunks: list[Chunk]) -> None:
        """Build the index."""

    def search(self, query: str, top_k: int) -> list[str]:
        """Return chunk ids, best first."""


class ModelSpec(BaseModel):
    """An embedding model and how to call it."""

    name: str
    kind: str  # "google" | "sentence_transformers" | "bm25"
    model_id: str = ""
    query_prefix: str = ""
    document_prefix: str = ""


MODEL_SPECS: dict[str, ModelSpec] = {
    "gemini": ModelSpec(name="gemini", kind="google", model_id="models/gemini-embedding-001"),
    "e5_large_instruct": ModelSpec(
        name="e5_large_instruct",
        kind="sentence_transformers",
        model_id="intfloat/multilingual-e5-large-instruct",
        query_prefix=E5_QUERY_PREFIX,
    ),
    "bge_m3": ModelSpec(name="bge_m3", kind="sentence_transformers", model_id="BAAI/bge-m3"),
    "minilm": ModelSpec(
        name="minilm",
        kind="sentence_transformers",
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ),
    "bm25": ModelSpec(name="bm25", kind="bm25"),
}


class Embedder(Protocol):
    """LangChain-style embedding interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""


class SentenceTransformerEmbedder:
    """Local sentence-transformers model with optional query / passage prefixes."""

    def __init__(self, spec: ModelSpec) -> None:
        """Load the model (downloads it on first use).

        Args:
            spec: Model id and prefixes.
        """
        # pylint: disable=import-outside-toplevel
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(spec.model_id)
        self._spec = spec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages (normalised)."""
        inputs = [self._spec.document_prefix + text for text in texts]
        vectors = self._model.encode(inputs, batch_size=16, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query (normalised)."""
        vector = self._model.encode([self._spec.query_prefix + text], normalize_embeddings=True)
        return list(map(float, vector[0]))


def create_embedder(spec: ModelSpec) -> Embedder:
    """Create the embedder for a dense model.

    Args:
        spec: A model with kind "google" or "sentence_transformers".

    Returns:
        The embedder.
    """
    if spec.kind == "google":
        # pylint: disable=import-outside-toplevel
        from finance_ai.core.config import get_settings
        from finance_ai.rag.embedding_factory import create_google_embeddings

        settings = get_settings().model_copy(update={"rag_embedding_model": spec.model_id})
        return create_google_embeddings(settings)
    return SentenceTransformerEmbedder(spec)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.asarray(matrix / np.where(norms == 0, 1, norms))


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    """Short hash of the chunk ids, so a cache never mixes corpora.

    Example:
        >>> len(corpus_fingerprint([]))
        12
    """
    return hashlib.sha256("|".join(c.chunk_id for c in chunks).encode()).hexdigest()[:12]


class DenseRetriever:
    """Exact cosine search over one embedding model's vectors."""

    def __init__(
        self, spec: ModelSpec, embedder_factory: Callable[[ModelSpec], Embedder] = create_embedder
    ) -> None:
        """Store the model spec; the model loads on index().

        Args:
            spec: Dense model.
            embedder_factory: Builds the embedder (replaced in tests).
        """
        self.name = spec.name
        self._spec = spec
        self._factory = embedder_factory
        self._embedder: Optional[Embedder] = None
        self._ids: list[str] = []
        self._matrix = np.zeros((0, 0))
        self.cache_dir: Optional[Path] = None
        self.index_seconds = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        """Embed every chunk (or load cached vectors) and time it.

        Args:
            chunks: Chunks to index.
        """
        self._embedder = self._factory(self._spec)
        self._ids = [chunk.chunk_id for chunk in chunks]
        start = time.perf_counter()
        self._matrix = _normalise(self._load_or_embed(chunks))
        self.index_seconds = time.perf_counter() - start

    def _load_or_embed(self, chunks: list[Chunk]) -> np.ndarray:
        """Return cached document vectors, embedding them on a cache miss."""
        cache = self._cache_path(chunks)
        if cache is not None and cache.exists():
            return np.asarray(np.load(cache))
        assert self._embedder is not None
        matrix = np.asarray(self._embedder.embed_documents([c.text for c in chunks]))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, matrix)
        return matrix

    def _cache_path(self, chunks: list[Chunk]) -> Optional[Path]:
        """Where this model's vectors for this corpus are cached, if caching is on."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.name}_{corpus_fingerprint(chunks)}.npy"

    def search(self, query: str, top_k: int) -> list[str]:
        """Rank chunks by cosine similarity to the query.

        Args:
            query: Question text.
            top_k: Number of ids to return.

        Returns:
            Chunk ids, best first.
        """
        assert self._embedder is not None, "index() first"
        vector = np.asarray(self._embedder.embed_query(query))
        scores = self._matrix @ (vector / (np.linalg.norm(vector) or 1.0))
        order = np.argsort(-scores)[:top_k]
        return [self._ids[i] for i in order]


def thai_tokens(text: str) -> list[str]:
    """Lower-cased word tokens (PyThaiNLP newmm handles Thai without spaces).

    Example:
        >>> "ภาษี" in thai_tokens("ภาษีเงินได้")
        True
    """
    # pylint: disable=import-outside-toplevel
    from pythainlp.tokenize import word_tokenize

    return [token.lower() for token in word_tokenize(text, engine="newmm") if token.strip()]


class BM25Retriever:
    """Okapi BM25 keyword search (k1 = 1.5, b = 0.75), the no-embedding baseline."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Set the BM25 parameters."""
        self.name = "bm25"
        self._k1, self._b = k1, b
        self._ids: list[str] = []
        self._docs: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._average_length = 0.0
        self.index_seconds = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        """Tokenise every chunk and compute document frequencies."""
        start = time.perf_counter()
        self._ids = [chunk.chunk_id for chunk in chunks]
        self._docs = [Counter(thai_tokens(chunk.text)) for chunk in chunks]
        total = len(self._docs)
        frequency: Counter[str] = Counter(t for doc in self._docs for t in doc)
        self._idf = {t: math.log(1 + (total - n + 0.5) / (n + 0.5)) for t, n in frequency.items()}
        self._average_length = sum(sum(d.values()) for d in self._docs) / max(total, 1)
        self.index_seconds = time.perf_counter() - start

    def _score(self, doc: Counter[str], terms: list[str]) -> float:
        """BM25 score of one document."""
        length = sum(doc.values())
        norm = self._k1 * (1 - self._b + self._b * length / (self._average_length or 1))
        return sum(
            self._idf.get(t, 0.0) * doc[t] * (self._k1 + 1) / (doc[t] + norm)
            for t in terms
            if t in doc
        )

    def search(self, query: str, top_k: int) -> list[str]:
        """Rank chunks by BM25 score."""
        terms = thai_tokens(query)
        scores = [self._score(doc, terms) for doc in self._docs]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self._ids[i] for i in order]


def build_retriever(spec: ModelSpec, cache_dir: Optional[Path] = None) -> Retriever:
    """Create the retriever for a model spec.

    Args:
        spec: Model to wrap.
        cache_dir: Where dense vectors are cached (None = no cache).

    Returns:
        An un-indexed retriever.
    """
    if spec.kind == "bm25":
        return BM25Retriever()
    retriever = DenseRetriever(spec)
    retriever.cache_dir = cache_dir
    return retriever
