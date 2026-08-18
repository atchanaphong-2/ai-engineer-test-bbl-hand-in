"""Tests for the /api/chat SSE endpoint, with fake agents standing in for
the real ones via a RagOrchestrator dependency override — no live LLM needed."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import BaseMessage

from agentic_rag.backend.dependencies import get_orchestrator
from agentic_rag.backend.routers.chat import router as chat_router
from agentic_rag.models import ReportOutput, RetrievalResult
from agentic_rag.orchestrator import RagOrchestrator


class _FakeRetrieverAgent:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def retrieve(
        self, query: str, history: list[BaseMessage] | None = None
    ) -> RetrievalResult:
        return RetrievalResult(chunks=self._chunks)


class _FakeGeneratorAgent:
    def __init__(self, answer: str, grounded: bool) -> None:
        self._answer = answer
        self._grounded = grounded

    async def generate(
        self,
        query: str,
        chunks: list[str],
        history: list[BaseMessage] | None = None,
    ) -> ReportOutput:
        return ReportOutput(answer=self._answer, grounded=self._grounded)


def _make_client(orchestrator: RagOrchestrator) -> TestClient:
    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    return TestClient(app)


def test_chat_streams_chunks_then_report_then_done():
    orchestrator = RagOrchestrator(
        _FakeRetrieverAgent(chunks=["snippet one"]),
        _FakeGeneratorAgent(answer="final answer", grounded=True),
    )
    client = _make_client(orchestrator)

    response = client.post(
        "/api/chat", json={"query": "what is the policy?", "thread_id": "test-thread"}
    )

    assert response.status_code == 200
    body = response.text
    assert "event: chunks_retrieved" in body
    assert '"snippet one"' in body
    assert "event: report_generated" in body
    assert '"final answer"' in body
    assert "event: done" in body
    assert (
        body.index("event: chunks_retrieved")
        < body.index("event: report_generated")
        < body.index("event: done")
    )


def test_chat_rejects_empty_query():
    orchestrator = RagOrchestrator(
        _FakeRetrieverAgent(chunks=[]),
        _FakeGeneratorAgent(answer="", grounded=False),
    )
    client = _make_client(orchestrator)

    response = client.post("/api/chat", json={"query": "", "thread_id": "test-thread"})

    assert response.status_code == 422


def test_chat_rejects_missing_thread_id():
    orchestrator = RagOrchestrator(
        _FakeRetrieverAgent(chunks=[]),
        _FakeGeneratorAgent(answer="", grounded=False),
    )
    client = _make_client(orchestrator)

    response = client.post("/api/chat", json={"query": "a question"})

    assert response.status_code == 422
