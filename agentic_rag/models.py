"""Shared Pydantic data shapes used across agents, orchestrator, and API layers."""

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Structured output for the Data Retriever agent."""

    chunks: list[str] = Field(
        default_factory=list,
        description=(
            "Raw, relevant text snippets copied verbatim from the knowledge "
            "base tool's results. Empty if nothing relevant was found. "
            "Never include commentary, summaries, or answers here."
        ),
    )


class ReportOutput(BaseModel):
    """Structured output for the Report Generator agent."""

    answer: str = Field(description="The final, polished, non-redundant answer for the user.")
    grounded: bool = Field(
        description=(
            "True if the answer is fully supported by the provided snippets; "
            "False if the agent had to note missing/incomplete information."
        )
    )
