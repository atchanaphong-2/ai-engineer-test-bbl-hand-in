"""The custom retrieval tool the Data Retriever agent calls."""

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from agentic_rag.retrieval.retriever import KnowledgeBaseRetriever


class SearchKnowledgeBaseInput(BaseModel):
    """Input schema for the knowledge base search tool."""

    query: str = Field(description="The user's question or search query.")


def make_search_tool(retriever: KnowledgeBaseRetriever, k: int) -> BaseTool:
    """Build the LangChain tool the Data Retriever agent calls.

    Bound to a specific `retriever` instance (no module-level global
    retriever), so each `create_agent` call gets its own wiring.
    """

    async def search_knowledge_base(query: str) -> list[str]:
        return await retriever.search(query, k=k)

    return StructuredTool.from_function(
        coroutine=search_knowledge_base,
        name="search_knowledge_base",
        description=(
            "Search the knowledge base for raw text snippets relevant to a "
            "query. Returns a list of verbatim text chunks — never a "
            "synthesized answer."
        ),
        args_schema=SearchKnowledgeBaseInput,
    )
