"""Tests for RagOrchestrator's sequencing and multi-turn memory, with fake
agents standing in for DataRetrieverAgent/ReportGeneratorAgent — no live
LLM required."""

import asyncio

from langchain_core.messages import BaseMessage

from agentic_rag.models import ReportOutput, RetrievalResult
from agentic_rag.orchestrator import RagOrchestrator


class FakeRetrieverAgent:
    """Duck-typed stand-in for DataRetrieverAgent."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, list[BaseMessage]]] = []

    async def retrieve(
        self, query: str, history: list[BaseMessage] | None = None
    ) -> RetrievalResult:
        self.calls.append((query, history or []))
        return RetrievalResult(chunks=self._chunks)


class FakeGeneratorAgent:
    """Duck-typed stand-in for ReportGeneratorAgent."""

    def __init__(self, answer: str, grounded: bool = True) -> None:
        self._answer = answer
        self._grounded = grounded
        self.calls: list[tuple[str, list[str], list[BaseMessage]]] = []

    async def generate(
        self,
        query: str,
        chunks: list[str],
        history: list[BaseMessage] | None = None,
    ) -> ReportOutput:
        self.calls.append((query, chunks, history or []))
        return ReportOutput(answer=self._answer, grounded=self._grounded)


async def _collect(agen):
    return [item async for item in agen]


def test_run_passes_retrieved_chunks_to_generator():
    retriever = FakeRetrieverAgent(chunks=["snippet one", "snippet two"])
    generator = FakeGeneratorAgent(answer="final answer", grounded=True)
    orchestrator = RagOrchestrator(retriever, generator)

    state = asyncio.run(orchestrator.run("what is the policy?", "thread-1"))

    assert retriever.calls[0][0] == "what is the policy?"
    assert generator.calls[0][:2] == ("what is the policy?", ["snippet one", "snippet two"])
    assert state["retrieved_chunks"] == ["snippet one", "snippet two"]
    assert state["final_report"] == ReportOutput(answer="final answer", grounded=True)


def test_run_generates_from_empty_chunks_when_nothing_found():
    retriever = FakeRetrieverAgent(chunks=[])
    generator = FakeGeneratorAgent(answer="no information found", grounded=False)
    orchestrator = RagOrchestrator(retriever, generator)

    state = asyncio.run(orchestrator.run("unrelated query", "thread-1"))

    assert generator.calls[0][:2] == ("unrelated query", [])
    assert state["final_report"].grounded is False


def test_stream_yields_retrieve_before_generate():
    retriever = FakeRetrieverAgent(chunks=["a chunk"])
    generator = FakeGeneratorAgent(answer="answer", grounded=True)
    orchestrator = RagOrchestrator(retriever, generator)

    events = asyncio.run(_collect(orchestrator.stream("query", "thread-1")))
    node_names = [name for name, _update in events]

    assert node_names == ["retrieve", "generate"]
    assert events[0][1]["retrieved_chunks"] == ["a chunk"]
    assert events[1][1]["final_report"] == ReportOutput(answer="answer", grounded=True)


def test_second_turn_on_same_thread_sees_first_turn_as_history():
    retriever = FakeRetrieverAgent(chunks=["chunk"])
    generator = FakeGeneratorAgent(answer="16 weeks for primary caregivers", grounded=True)
    orchestrator = RagOrchestrator(retriever, generator)

    asyncio.run(orchestrator.run("I just had a baby, what policy applies?", "thread-a"))
    asyncio.run(orchestrator.run("what about secondary caregivers?", "thread-a"))

    # Second turn's agents saw the first turn's (query, answer) as history,
    # as alternating Human/AI messages.
    second_turn_history = retriever.calls[1][1]
    assert len(second_turn_history) == 2
    assert second_turn_history[0].content == "I just had a baby, what policy applies?"
    assert second_turn_history[1].content == "16 weeks for primary caregivers"


def test_different_threads_do_not_share_history():
    retriever = FakeRetrieverAgent(chunks=["chunk"])
    generator = FakeGeneratorAgent(answer="answer", grounded=True)
    orchestrator = RagOrchestrator(retriever, generator)

    asyncio.run(orchestrator.run("first thread's question", "thread-a"))
    asyncio.run(orchestrator.run("second thread's question", "thread-b"))

    # thread-b's only call has no history, since it's a separate conversation.
    thread_b_history = retriever.calls[1][1]
    assert thread_b_history == []
