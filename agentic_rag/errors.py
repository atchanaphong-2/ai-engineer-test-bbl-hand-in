"""Shared exception types surfaced at the CLI/API boundary."""


class AgentInvocationError(RuntimeError):
    """Raised when an agent fails to produce a usable response."""
