"""Unit tests for DataRetrieverAgent/ReportGeneratorAgent.

Construction uses a real `GenericFakeChatModel` to prove the actual
create_agent(..., response_format=ToolStrategy(...)) wiring is valid — no
live LLM needed, and no coupling to LangChain's internal tool-call wire
format. Invocation behavior (structured_response extraction, error
wrapping, prompt construction) is tested by swapping in a fake compiled
graph for the one real seam BaseAgent exposes: `self._agent`.
"""

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agentic_rag.agents.data_retriever import DataRetrieverAgent
from agentic_rag.agents.report_generator import ReportGeneratorAgent
from agentic_rag.errors import AgentInvocationError
from agentic_rag.models import ReportOutput, RetrievalResult
from agentic_rag.tools import make_search_tool


class _FakeRetriever:
    async def search(self, query: str, k: int) -> list[str]:
        return ["fake chunk"]


class _CapturingGraph:
    """Fake compiled graph: records its ainvoke() input and returns a canned
    structured_response, standing in for the real LangGraph-compiled agent."""

    def __init__(self, structured_response: object) -> None:
        self.structured_response = structured_response
        self.last_inputs: dict | None = None

    async def ainvoke(self, inputs: dict) -> dict:
        self.last_inputs = inputs
        return {"structured_response": self.structured_response}


class _RaisingGraph:
    async def ainvoke(self, inputs: dict) -> dict:
        raise RuntimeError("boom")


def _fake_model() -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter(["unused"]))


def test_data_retriever_agent_builds_with_real_create_agent_wiring():
    tool = make_search_tool(_FakeRetriever(), k=4)
    agent = DataRetrieverAgent(_fake_model(), tool)
    assert agent._agent is not None


def test_data_retriever_agent_extracts_structured_response():
    agent = DataRetrieverAgent(_fake_model(), make_search_tool(_FakeRetriever(), k=4))
    agent._agent = _CapturingGraph(RetrievalResult(chunks=["a", "b"]))

    result = asyncio.run(agent.retrieve("what is the policy?"))

    assert result == RetrievalResult(chunks=["a", "b"])


def test_data_retriever_agent_wraps_errors():
    agent = DataRetrieverAgent(_fake_model(), make_search_tool(_FakeRetriever(), k=4))
    agent._agent = _RaisingGraph()

    with pytest.raises(AgentInvocationError, match="DataRetrieverAgent"):
        asyncio.run(agent.retrieve("query"))


def test_report_generator_agent_builds_with_real_create_agent_wiring():
    agent = ReportGeneratorAgent(_fake_model())
    assert agent._agent is not None


def test_report_generator_agent_extracts_structured_response():
    agent = ReportGeneratorAgent(_fake_model())
    agent._agent = _CapturingGraph(ReportOutput(answer="the answer", grounded=True))

    result = asyncio.run(agent.generate("query", ["chunk one"]))

    assert result == ReportOutput(answer="the answer", grounded=True)


def test_report_generator_agent_wraps_errors():
    agent = ReportGeneratorAgent(_fake_model())
    agent._agent = _RaisingGraph()

    with pytest.raises(AgentInvocationError, match="ReportGeneratorAgent"):
        asyncio.run(agent.generate("query", []))


def test_report_generator_agent_prompt_includes_query_and_chunks():
    agent = ReportGeneratorAgent(_fake_model())
    graph = _CapturingGraph(ReportOutput(answer="x", grounded=True))
    agent._agent = graph

    asyncio.run(agent.generate("what is the policy?", ["snippet one", "snippet two"]))

    prompt_text = graph.last_inputs["messages"][0].content
    assert "what is the policy?" in prompt_text
    assert "snippet one" in prompt_text
    assert "snippet two" in prompt_text


def test_report_generator_agent_prompt_notes_empty_chunks():
    agent = ReportGeneratorAgent(_fake_model())
    graph = _CapturingGraph(ReportOutput(answer="x", grounded=False))
    agent._agent = graph

    asyncio.run(agent.generate("unrelated query", []))

    prompt_text = graph.last_inputs["messages"][0].content
    assert "no relevant snippets found" in prompt_text


def test_data_retriever_agent_includes_history_before_current_query():
    agent = DataRetrieverAgent(_fake_model(), make_search_tool(_FakeRetriever(), k=4))
    graph = _CapturingGraph(RetrievalResult(chunks=[]))
    agent._agent = graph
    history = [HumanMessage(content="earlier question"), AIMessage(content="earlier answer")]

    asyncio.run(agent.retrieve("follow-up question", history))

    messages = graph.last_inputs["messages"]
    assert [m.content for m in messages] == [
        "earlier question",
        "earlier answer",
        "follow-up question",
    ]


def test_report_generator_agent_includes_history_before_current_prompt():
    agent = ReportGeneratorAgent(_fake_model())
    graph = _CapturingGraph(ReportOutput(answer="x", grounded=True))
    agent._agent = graph
    history = [HumanMessage(content="earlier question"), AIMessage(content="earlier answer")]

    asyncio.run(agent.generate("follow-up question", ["chunk"], history))

    messages = graph.last_inputs["messages"]
    assert messages[0].content == "earlier question"
    assert messages[1].content == "earlier answer"
    assert "follow-up question" in messages[2].content
