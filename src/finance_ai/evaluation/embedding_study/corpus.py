"""Knowledge-base chunks and RAG questions for the embedding study."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from finance_ai.rag.document_loader import discover_documents, load_document
from finance_ai.rag.text_splitter import split_text_into_chunks

KNOWLEDGE_BASE = Path("docs/knowledge_base")
QUESTIONS_PATH = Path("data/evaluation/rag_v2_dataset.yaml")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class Chunk(BaseModel):
    """One production chunk."""

    chunk_id: str
    source_file: str
    text: str


class RagQuestion(BaseModel):
    """One question with the files and keywords that mark a relevant chunk.

    Example:
        >>> RagQuestion(question_id="q1", query="?", gold_files=["a.md"], keywords=["x"]).keywords
        ['x']
    """

    question_id: str
    query: str
    gold_files: list[str]
    keywords: list[str] = Field(min_length=1)


def load_chunks(directory: Path = KNOWLEDGE_BASE) -> list[Chunk]:
    """Split every knowledge-base document exactly as production does.

    Args:
        directory: Knowledge-base folder.

    Returns:
        Chunks in document order.
    """
    chunks: list[Chunk] = []
    for path in discover_documents(directory):
        text, metadata = load_document(path)
        for piece in split_text_into_chunks(text, metadata, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(Chunk(chunk_id=piece.chunk_id, source_file=path.name, text=piece.content))
    return chunks


def load_questions(path: Path = QUESTIONS_PATH) -> list[RagQuestion]:
    """Load the RAG questions.

    Args:
        path: YAML file with a ``questions`` list.

    Returns:
        Questions in file order.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [RagQuestion(**item) for item in data["questions"]]


def keyword_present(keyword: str, text: str) -> bool:
    """Tell whether a keyword (``"a|b"`` = either) appears in text.

    Example:
        >>> keyword_present("× 25|25 เท่า", "เก็บ 25 เท่าของค่าใช้จ่าย")
        True
    """
    return any(option in text for option in keyword.split("|"))


def relevant_chunk_ids(question: RagQuestion, chunks: list[Chunk]) -> set[str]:
    """Return the ids of chunks that answer a question.

    Args:
        question: The question.
        chunks: Every chunk.

    Returns:
        Ids of chunks from a gold file that contain every keyword.
    """
    return {
        chunk.chunk_id
        for chunk in chunks
        if chunk.source_file in question.gold_files
        and all(keyword_present(keyword, chunk.text) for keyword in question.keywords)
    }
