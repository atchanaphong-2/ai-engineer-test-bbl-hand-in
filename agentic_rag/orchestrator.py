"""Deterministic sequential Retriever -> Generator conversation orchestrator."""

import operator
from collections.abc import AsyncIterator
from typing import Annotated, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.agents.data_retriever import DataRetrieverAgent
from agentic_rag.agents.report_generator import ReportGeneratorAgent
from agentic_rag.models import ReportOutput


class OrchestratorState(TypedDict):
    """State threaded through the orchestrator's graph, checkpointed per
    `thread_id`. `history` accumulates across turns (see `_generate_node`);
    everything else is scratch state for the current turn only."""

    query: str
    history: Annotated[list[BaseMessage], operator.add]
    retrieved_chunks: list[str]
    final_report: ReportOutput | None


class RagOrchestrator:
    """Owns the compiled StateGraph. Both main.py and the API call this.

    The graph's edges are fixed (retrieve -> generate) — no agent decides
    whether or how to invoke the other; that's what keeps the workflow
    deterministic. Multi-turn memory is LangGraph's own checkpointer, keyed
    by the caller-supplied `thread_id` — state for one conversation is
    invisible to another, and nothing is persisted beyond this process
    (`MemorySaver` is in-memory: lost on restart, not shared across workers).
    """

    def __init__(
        self,
        retriever_agent: DataRetrieverAgent,
        generator_agent: ReportGeneratorAgent,
    ) -> None:
        self._retriever_agent = retriever_agent
        self._generator_agent = generator_agent
        self._graph: CompiledStateGraph = self._build_graph()

    async def run(self, query: str, thread_id: str) -> OrchestratorState:
        """Run one turn of conversation `thread_id` to completion."""
        state = await self._graph.ainvoke(
            self._turn_input(query), config=self._config(thread_id)
        )
        return cast(OrchestratorState, state)

    async def stream(self, query: str, thread_id: str) -> AsyncIterator[tuple[str, dict]]:
        """Yield (node_name, partial_state_update) as each node finishes.

        Unlike a full-state stream, this unambiguously identifies which node
        just ran even when its update is an empty list/falsy value — the
        caller doesn't have to guess from the state's shape.
        """
        async for update in self._graph.astream(
            self._turn_input(query), config=self._config(thread_id), stream_mode="updates"
        ):
            for item in update.items():
                yield item

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(OrchestratorState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("generate", self._generate_node)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile(checkpointer=MemorySaver())

    async def _retrieve_node(self, state: OrchestratorState) -> dict:
        result = await self._retriever_agent.retrieve(state["query"], state["history"])
        return {"retrieved_chunks": result.chunks}

    async def _generate_node(self, state: OrchestratorState) -> dict:
        report = await self._generator_agent.generate(
            state["query"], state["retrieved_chunks"], state["history"]
        )
        turn = [HumanMessage(content=state["query"]), AIMessage(content=report.answer)]
        return {"final_report": report, "history": turn}

    @staticmethod
    def _turn_input(query: str) -> dict:
        return {"query": query, "retrieved_chunks": [], "final_report": None}

    @staticmethod
    def _config(thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}
