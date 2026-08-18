"""API request schemas."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body of POST /api/chat.

    `thread_id` identifies the conversation for the orchestrator's
    checkpointer — the client generates it once per session and resends it
    on every turn so follow-up questions can see prior turns.
    """

    query: str = Field(min_length=1, description="The user's question.")
    thread_id: str = Field(min_length=1, description="Client-generated conversation id.")
