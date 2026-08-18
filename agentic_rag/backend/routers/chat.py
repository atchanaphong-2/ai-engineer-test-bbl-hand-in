"""POST /api/chat — streams orchestrator progress as Server-Sent Events."""

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from agentic_rag.backend.dependencies import get_orchestrator
from agentic_rag.backend.schemas import ChatRequest
from agentic_rag.errors import AgentInvocationError
from agentic_rag.orchestrator import RagOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_events(
    orchestrator: RagOrchestrator, query: str, thread_id: str
) -> AsyncIterator[str]:
    try:
        async for node_name, update in orchestrator.stream(query, thread_id):
            if node_name == "retrieve":
                yield _sse("chunks_retrieved", {"chunks": update["retrieved_chunks"]})
            elif node_name == "generate":
                report = update["final_report"]
                yield _sse("report_generated", report.model_dump())
    except AgentInvocationError as exc:
        logger.warning("Orchestrator failed for query: %s", exc)
        yield _sse("error", {"message": str(exc)})
    yield _sse("done", {})


@router.post("/chat")
async def chat(
    request: ChatRequest,
    orchestrator: Annotated[RagOrchestrator, Depends(get_orchestrator)],
) -> StreamingResponse:
    """Stream chunks_retrieved -> report_generated -> done as SSE."""
    return StreamingResponse(
        _stream_events(orchestrator, request.query, request.thread_id),
        media_type="text/event-stream",
    )
