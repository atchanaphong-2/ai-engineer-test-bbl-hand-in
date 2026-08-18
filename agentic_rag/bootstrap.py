"""Composition root: the one place that wires the full object graph."""

from functools import cached_property

from agentic_rag.agents.data_retriever import DataRetrieverAgent
from agentic_rag.agents.report_generator import ReportGeneratorAgent
from agentic_rag.config import ChatModelFactory, Settings
from agentic_rag.orchestrator import RagOrchestrator
from agentic_rag.retrieval.embedding import EmbeddingModel
from agentic_rag.retrieval.index import FaissSearchIndex
from agentic_rag.retrieval.retriever import KnowledgeBaseRetriever
from agentic_rag.retrieval.store import ChunkStore
from agentic_rag.tools import make_search_tool


class AppContainer:
    """Builds the full object graph once: Settings -> ChatModelFactory ->
    KnowledgeBaseRetriever -> agents -> RagOrchestrator. Both main.py (CLI) and
    backend/main.py's create_app (API) read `.orchestrator` off an AppContainer
    to get the same wiring, so there's exactly one place that constructs the app."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()

    @cached_property
    def settings(self) -> Settings:
        return self._settings

    @cached_property
    def retriever(self) -> KnowledgeBaseRetriever:
        embedder = EmbeddingModel(self._settings.embedding_model_name)
        return KnowledgeBaseRetriever(
            kb_path=self._settings.kb_path,
            store=ChunkStore(self._settings.kb_index_path),
            embedder=embedder,
            index=FaissSearchIndex(embedder.dimension),
            similarity_floor=self._settings.similarity_floor,
        )

    @cached_property
    def orchestrator(self) -> RagOrchestrator:
        model = ChatModelFactory(self._settings).create()
        search_tool = make_search_tool(self.retriever, k=self._settings.retrieval_k)
        retriever_agent = DataRetrieverAgent(model, search_tool)
        generator_agent = ReportGeneratorAgent(model)
        return RagOrchestrator(retriever_agent, generator_agent)
