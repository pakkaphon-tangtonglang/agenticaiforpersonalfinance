"""Tests for the embedding study: corpus, metrics, retrievers, report."""

from pathlib import Path

import pytest

from finance_ai.evaluation.embedding_study import study
from finance_ai.evaluation.embedding_study.corpus import (
    Chunk,
    RagQuestion,
    keyword_present,
    load_questions,
    relevant_chunk_ids,
)
from finance_ai.evaluation.embedding_study.metrics import (
    lexical_overlap,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from finance_ai.evaluation.embedding_study.retrievers import (
    MODEL_SPECS,
    BM25Retriever,
    DenseRetriever,
    build_retriever,
    corpus_fingerprint,
)

CHUNKS = [
    Chunk(chunk_id="c1", source_file="tax.md", text="ค่าลดหย่อนส่วนตัว 60,000 บาท"),
    Chunk(chunk_id="c2", source_file="tax.md", text="ดอกเบี้ยบ้านลดหย่อนได้ 100,000 บาท"),
    Chunk(chunk_id="c3", source_file="debt.md", text="หนี้บัตรเครดิต ดอกเบี้ย 16%"),
]
QUESTION = RagQuestion(
    question_id="q1", query="ผ่อนบ้าน ดอกเบี้ย หักภาษี", gold_files=["tax.md"], keywords=["100,000"]
)


class FakeEmbedder:
    """Maps text to a 3-d vector by which topic words it contains."""

    calls = 0

    def _vector(self, text: str) -> list[float]:
        return [float("ส่วนตัว" in text), float("บ้าน" in text), float("บัตร" in text)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        FakeEmbedder.calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class TestCorpus:
    """Relevance labels."""

    def test_relevant_needs_gold_file_and_keywords(self) -> None:
        """Only c2 is in a gold file and contains 100,000."""
        assert relevant_chunk_ids(QUESTION, CHUNKS) == {"c2"}

    def test_keyword_alternatives(self) -> None:
        """'a|b' matches either option."""
        assert keyword_present("× 25|25 เท่า", "25 เท่า") and not keyword_present("x|y", "z")

    def test_real_dataset_every_question_answerable(self) -> None:
        """All 120 questions load with unique ids."""
        ids = [q.question_id for q in load_questions()]
        assert len(ids) == len(set(ids)) == 120


class TestMetrics:
    """Ranking metrics."""

    def test_recall_and_rr(self) -> None:
        """Hit within k; reciprocal of the first relevant rank."""
        assert recall_at_k(["a", "b"], {"b"}, 2) == 1.0
        assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
        assert reciprocal_rank(["a"], {"z"}) == 0.0

    def test_ndcg_perfect_and_none(self) -> None:
        """Best order scores 1, no relevant scores 0."""
        assert ndcg_at_k(["a", "b"], {"a", "b"}) == pytest.approx(1.0)
        assert ndcg_at_k(["a"], set()) == 0.0

    def test_overlap(self) -> None:
        """Copied wording scores high, unrelated text zero; empty query is zero."""
        assert lexical_overlap("ค่าลดหย่อนส่วนตัว", CHUNKS[0].text) == 1.0
        assert lexical_overlap("abcdefgh", "zzzzzzzz") == 0.0
        assert lexical_overlap("ab", "ab") == 0.0


class TestRetrievers:
    """Dense and BM25 retrieval."""

    def test_dense_ranks_and_caches(self, tmp_path: Path) -> None:
        """The matching chunk ranks first; a second index loads vectors from cache."""
        FakeEmbedder.calls = 0
        retriever = DenseRetriever(MODEL_SPECS["minilm"], lambda _spec: FakeEmbedder())
        retriever.cache_dir = tmp_path
        retriever.index(CHUNKS)
        assert retriever.search("บ้าน", 1) == ["c2"]
        retriever.index(CHUNKS)
        assert FakeEmbedder.calls == 1
        assert (tmp_path / f"minilm_{corpus_fingerprint(CHUNKS)}.npy").exists()

    def test_bm25_keyword_match(self) -> None:
        """BM25 ranks the chunk sharing the query's words first."""
        retriever = BM25Retriever()
        retriever.index(CHUNKS)
        assert retriever.search("หนี้บัตรเครดิต", 1) == ["c3"]

    def test_build_retriever_kinds(self) -> None:
        """bm25 → BM25Retriever; others → DenseRetriever."""
        assert isinstance(build_retriever(MODEL_SPECS["bm25"]), BM25Retriever)
        assert isinstance(build_retriever(MODEL_SPECS["bge_m3"]), DenseRetriever)


class TestStudy:
    """Scoring and report."""

    def test_question_groups(self) -> None:
        """English, Thai PDF and Thai guide questions are told apart."""
        pdf = QUESTION.model_copy(update={"gold_files": ["a.pdf"]})
        english = pdf.model_copy(update={"query": "What is EMH?"})
        assert [study.question_group(q) for q in (QUESTION, pdf, english)] == [
            "thai_guide",
            "thai_pdf",
            "english_pdf",
        ]

    def test_end_to_end_with_fakes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two retrievers → report with a paired comparison and leakage line."""
        monkeypatch.setattr(study, "RESULTS_DIR", tmp_path)
        relevant = {"q1": {"c2"}}
        dense = DenseRetriever(MODEL_SPECS["minilm"], lambda _spec: FakeEmbedder())
        dense.name = "gemini"
        dense.index(CHUNKS)
        bm25 = BM25Retriever()
        bm25.index(CHUNKS)
        results = {
            "gemini": [study.score_question(dense, QUESTION, relevant["q1"])],
            "bm25": [study.score_question(bm25, QUESTION, relevant["q1"])],
        }
        leakage = study.leakage_scores([QUESTION], CHUNKS, relevant)
        report = study.write_outputs(results, {"gemini": 1.0, "bm25": 0.1}, leakage)
        text = report.read_text(encoding="utf-8")
        assert results["gemini"][0].recall_1 == 1.0
        assert "| bm25 |" in text and "Leakage check" in text
        assert (tmp_path / "retrieval.jsonl").exists()

    def test_main_runs_selected_models(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() runs only the chosen models."""
        monkeypatch.setattr(study, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(study, "load_chunks", lambda: CHUNKS)
        monkeypatch.setattr(study, "load_questions", lambda: [QUESTION])
        report = study.main(["--models", "bm25"])
        assert "| bm25 |" in report.read_text(encoding="utf-8")


class TestLoadersAndFactories:
    """Chunk loading and embedder creation (no downloads)."""

    def test_load_chunks_from_folder(self, tmp_path: Path) -> None:
        """A Markdown file is split into chunks named after the file."""
        # pylint: disable=import-outside-toplevel
        from finance_ai.evaluation.embedding_study.corpus import load_chunks

        (tmp_path / "Tax_Test.md").write_text("# ภาษี\n\n" + "ลดหย่อน " * 200, encoding="utf-8")
        chunks = load_chunks(tmp_path)
        assert chunks and {c.source_file for c in chunks} == {"Tax_Test.md"}

    def test_sentence_transformer_prefixes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Query prefix is added to queries only; vectors are returned as floats."""
        # pylint: disable=import-outside-toplevel
        import sys
        import types

        import numpy as np

        from finance_ai.evaluation.embedding_study import retrievers

        seen: list[list[str]] = []

        class FakeModel:  # pylint: disable=too-few-public-methods
            def __init__(self, _model_id: str) -> None:
                pass

            def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
                seen.append(texts)
                return np.ones((len(texts), 2))

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.SimpleNamespace(SentenceTransformer=FakeModel),
        )
        embedder = retrievers.create_embedder(MODEL_SPECS["e5_large_instruct"])
        assert embedder.embed_query("ภาษี") == [1.0, 1.0]
        embedder.embed_documents(["doc"])
        assert seen[0][0].startswith("Instruct:") and seen[1] == ["doc"]

    def test_google_embedder_uses_production_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Gemini goes through create_google_embeddings with the study's model id."""
        # pylint: disable=import-outside-toplevel
        from finance_ai.evaluation.embedding_study import retrievers
        from finance_ai.rag import embedding_factory

        captured: dict[str, str] = {}
        monkeypatch.setattr(
            embedding_factory,
            "create_google_embeddings",
            lambda settings: captured.setdefault("model", settings.rag_embedding_model),
        )
        retrievers.create_embedder(MODEL_SPECS["gemini"])
        assert captured["model"] == "models/gemini-embedding-001"
