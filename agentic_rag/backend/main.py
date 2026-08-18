"""FastAPI app factory: wires the orchestrator and mounts the static frontend."""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from agentic_rag.backend.routers.chat import router as chat_router
from agentic_rag.bootstrap import AppContainer

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


def create_app(container: AppContainer) -> FastAPI:
    """Build the FastAPI app: chat API + static frontend, one process/port."""
    app = FastAPI(title="Agentic RAG Test")
    app.state.orchestrator = container.orchestrator
    app.include_router(chat_router)
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app
