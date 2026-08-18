"""Shared construction/error-handling base class for both agents."""

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from agentic_rag.errors import AgentInvocationError


class BaseAgent(ABC):
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model
        self._agent = self._build()

    @abstractmethod
    def _build(self) -> Runnable:
        """Construct the underlying create_agent-compiled graph."""

    async def _invoke(self, messages: list[BaseMessage]) -> dict:
        try:
            return await self._agent.ainvoke({"messages": messages})
        except Exception as exc:
            raise AgentInvocationError(
                f"{type(self).__name__} failed to produce a response: {exc}"
            ) from exc
