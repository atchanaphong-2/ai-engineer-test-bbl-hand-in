"""FastAPI dependency providers."""

from fastapi import Request

from agentic_rag.orchestrator import RagOrchestrator


def get_orchestrator(request: Request) -> RagOrchestrator:
    """Return the singleton RagOrchestrator built once by AppContainer in create_app."""
    return request.app.state.orchestrator
