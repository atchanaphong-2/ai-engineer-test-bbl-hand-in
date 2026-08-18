"""Tests for AppContainer's wiring of Settings into the real object graph.

These exercise the exact composition bugs a typo in bootstrap.py would
cause (e.g. a swapped Settings field) — the retrieval stack runs locally
(real embedder/FAISS, no LLM), and ChatAnthropic construction validates
config without making a network call, so no live LLM/API key is needed.
"""

import asyncio
from pathlib import Path

from pydantic import SecretStr

from agentic_rag.bootstrap import AppContainer
from agentic_rag.config import Settings


def _write_kb(tmp_path: Path, content: str) -> Path:
    kb_path = tmp_path / "kb.txt"
    kb_path.write_text(content, encoding="utf-8")
    return kb_path


def test_retriever_property_wires_similarity_floor_from_settings(tmp_path):
    kb_path = _write_kb(
        tmp_path,
        "General Facts About Cats: cats sleep 12 to 16 hours a day and are "
        "obligate carnivores.",
    )

    strict_settings = Settings(
        kb_path=kb_path, kb_index_path=tmp_path / "strict.sqlite3", similarity_floor=0.99
    )
    strict = AppContainer(strict_settings)
    assert asyncio.run(strict.retriever.search("Tell me about cats", k=1)) == []

    lenient_settings = Settings(
        kb_path=kb_path, kb_index_path=tmp_path / "lenient.sqlite3", similarity_floor=0.0
    )
    lenient = AppContainer(lenient_settings)
    assert asyncio.run(lenient.retriever.search("Tell me about cats", k=1)) != []


def test_orchestrator_property_wires_retrieval_k_into_search_tool(tmp_path):
    kb_path = _write_kb(
        tmp_path,
        "Cats sleep 12 to 16 hours a day.\n\n"
        "Cats are obligate carnivores that require meat in their diet.\n\n"
        "Domestic cats can run up to 30 miles per hour in short bursts.",
    )

    def _search_tool_results(retrieval_k: int) -> list[str]:
        settings = Settings(
            anthropic_api_key=SecretStr("sk-ant-test-fake"),
            kb_path=kb_path,
            kb_index_path=tmp_path / f"k{retrieval_k}.sqlite3",
            retrieval_k=retrieval_k,
        )
        container = AppContainer(settings)
        search_tool = container.orchestrator._retriever_agent._search_tool
        return asyncio.run(search_tool.ainvoke({"query": "Tell me about cats"}))

    assert len(_search_tool_results(retrieval_k=1)) == 1
    assert len(_search_tool_results(retrieval_k=3)) == 3
