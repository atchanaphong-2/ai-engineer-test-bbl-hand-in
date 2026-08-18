"""Tests for main.py's own logic — arg parsing, error branches, output
formatting — via an injected orchestrator_factory. No live LLM needed."""

from agentic_rag.errors import AgentInvocationError
from agentic_rag.models import ReportOutput
from main import main


class _FakeOrchestrator:
    def __init__(self, chunks: list[str], answer: str, grounded: bool) -> None:
        self._chunks = chunks
        self._answer = answer
        self._grounded = grounded

    async def run(self, query: str, thread_id: str) -> dict:
        return {
            "retrieved_chunks": self._chunks,
            "final_report": ReportOutput(answer=self._answer, grounded=self._grounded),
        }


def test_main_prints_query_chunks_and_answer_on_success(capsys):
    def factory():
        return _FakeOrchestrator(chunks=["snippet one"], answer="final answer", grounded=True)

    exit_code = main(["what is the policy?"], orchestrator_factory=factory)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "what is the policy?" in captured.out
    assert "snippet one" in captured.out
    assert "final answer" in captured.out
    assert "[Note:" not in captured.out


def test_main_prints_note_when_not_grounded(capsys):
    def factory():
        return _FakeOrchestrator(chunks=[], answer="no info found", grounded=False)

    exit_code = main(["unrelated query"], orchestrator_factory=factory)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[Note: answer may be incomplete" in captured.out


def test_main_returns_1_on_missing_knowledge_base(capsys):
    def factory():
        raise FileNotFoundError("Knowledge base file not found: knowledge_base.txt")

    exit_code = main(["query"], orchestrator_factory=factory)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error:" in captured.err


def test_main_returns_1_on_config_error(capsys):
    def factory():
        raise ValueError("ANTHROPIC_API_KEY must be set")

    exit_code = main(["query"], orchestrator_factory=factory)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Configuration error:" in captured.err


def test_main_returns_1_on_agent_invocation_error(capsys):
    def factory():
        raise AgentInvocationError("DataRetrieverAgent failed to produce a response")

    exit_code = main(["query"], orchestrator_factory=factory)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error:" in captured.err
