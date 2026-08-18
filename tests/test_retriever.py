"""Tests for the local retrieval stack. No LLM, no network — everything
here runs against a real embedding model and real SQLite/FAISS, per the
project's own testing convention (retrieval is entirely local)."""

import asyncio
from pathlib import Path

import pytest

from agentic_rag.retrieval.embedding import EmbeddingModel
from agentic_rag.retrieval.index import FaissSearchIndex
from agentic_rag.retrieval.retriever import KnowledgeBaseRetriever
from agentic_rag.retrieval.store import ChunkStore

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_FLOOR = 0.2


@pytest.fixture(scope="module")
def embedder() -> EmbeddingModel:
    return EmbeddingModel(EMBEDDING_MODEL_NAME)


def _build_retriever(
    tmp_path: Path, embedder: EmbeddingModel, content: str, db_name: str = "index.sqlite3"
) -> KnowledgeBaseRetriever:
    kb_path = tmp_path / "kb.txt"
    kb_path.write_text(content, encoding="utf-8")
    return KnowledgeBaseRetriever(
        kb_path=kb_path,
        store=ChunkStore(tmp_path / db_name),
        embedder=embedder,
        index=FaissSearchIndex(embedder.dimension),
        similarity_floor=SIMILARITY_FLOOR,
    )


def test_chunks_split_on_blank_lines(tmp_path, embedder):
    content = (
        "Paragraph one about cats.\n\n"
        "Paragraph two about dogs.\n\n"
        "Paragraph three about travel policy."
    )
    retriever = _build_retriever(tmp_path, embedder, content)
    assert len(retriever) == 3


def test_search_returns_relevant_chunk(tmp_path, embedder):
    content = (
        "International Travel Policy: employees must submit a travel "
        "request 10 days before departure and get manager approval.\n\n"
        "General Facts About Cats: cats sleep 12 to 16 hours a day and are "
        "obligate carnivores."
    )
    retriever = _build_retriever(tmp_path, embedder, content)
    results = asyncio.run(retriever.search("What is the policy on international travel?", k=1))
    assert len(results) == 1
    assert "Travel Policy" in results[0]


def test_search_returns_empty_list_when_no_chunks_exist(tmp_path, embedder):
    retriever = _build_retriever(tmp_path, embedder, "")
    assert asyncio.run(retriever.search("anything", k=4)) == []


def test_missing_knowledge_base_raises_clear_error(tmp_path, embedder):
    store = ChunkStore(tmp_path / "index.sqlite3")
    index = FaissSearchIndex(embedder.dimension)

    with pytest.raises(FileNotFoundError, match="Knowledge base file not found"):
        KnowledgeBaseRetriever(
            kb_path=tmp_path / "missing.txt",
            store=store,
            embedder=embedder,
            index=index,
            similarity_floor=SIMILARITY_FLOOR,
        )


def test_cache_is_reused_across_instances(tmp_path, embedder):
    content = "Paragraph one.\n\nParagraph two."
    kb_path = tmp_path / "kb.txt"
    kb_path.write_text(content, encoding="utf-8")
    db_path = tmp_path / "index.sqlite3"

    first = KnowledgeBaseRetriever(
        kb_path=kb_path,
        store=ChunkStore(db_path),
        embedder=embedder,
        index=FaissSearchIndex(embedder.dimension),
        similarity_floor=SIMILARITY_FLOOR,
    )
    assert db_path.exists()
    assert len(first) == 2

    second = KnowledgeBaseRetriever(
        kb_path=kb_path,
        store=ChunkStore(db_path),
        embedder=embedder,
        index=FaissSearchIndex(embedder.dimension),
        similarity_floor=SIMILARITY_FLOOR,
    )
    assert len(second) == 2


def test_cache_invalidated_on_content_change(tmp_path, embedder):
    kb_path = tmp_path / "kb.txt"
    db_path = tmp_path / "index.sqlite3"
    kb_path.write_text("Paragraph one.\n\nParagraph two.", encoding="utf-8")

    KnowledgeBaseRetriever(
        kb_path=kb_path,
        store=ChunkStore(db_path),
        embedder=embedder,
        index=FaissSearchIndex(embedder.dimension),
        similarity_floor=SIMILARITY_FLOOR,
    )

    kb_path.write_text(
        "Paragraph one.\n\nParagraph two.\n\nParagraph three.", encoding="utf-8"
    )
    retriever = KnowledgeBaseRetriever(
        kb_path=kb_path,
        store=ChunkStore(db_path),
        embedder=embedder,
        index=FaissSearchIndex(embedder.dimension),
        similarity_floor=SIMILARITY_FLOOR,
    )
    assert len(retriever) == 3


def test_chunk_store_invalid_for_missing_file(tmp_path):
    store = ChunkStore(tmp_path / "nope.sqlite3")
    assert store.is_valid("some-hash") is False


def test_chunk_store_invalid_for_corrupted_file(tmp_path):
    db_path = tmp_path / "bad.sqlite3"
    db_path.write_bytes(b"not a real sqlite file")
    store = ChunkStore(db_path)
    assert store.is_valid("some-hash") is False


def test_search_returns_empty_when_nothing_relevant(tmp_path, embedder):
    content = (
        "International Travel Policy: employees must submit a travel "
        "request 10 days before departure and get manager approval.\n\n"
        "General Facts About Cats: cats sleep 12 to 16 hours a day."
    )
    retriever = _build_retriever(tmp_path, embedder, content)
    results = asyncio.run(retriever.search("xkqzvortex flibbergarnish quantum toaster", k=3))
    assert results == []


def test_concurrent_searches_return_independent_correct_results(tmp_path, embedder):
    """Multiple in-flight search() calls (e.g. the Data Retriever agent
    issuing several concurrent tool calls for a multi-part question) must
    not interfere with each other."""
    content = (
        "International Travel Policy: employees must submit a travel "
        "request 10 days before departure and get manager approval.\n\n"
        "General Facts About Cats: cats sleep 12 to 16 hours a day and are "
        "obligate carnivores."
    )
    retriever = _build_retriever(tmp_path, embedder, content)

    async def _run_concurrently():
        return await asyncio.gather(
            retriever.search("What is the policy on international travel?", k=1),
            retriever.search("Tell me about cats", k=1),
        )

    travel_results, cat_results = asyncio.run(_run_concurrently())
    assert "Travel Policy" in travel_results[0]
    assert "Cats" in cat_results[0]
