"""Application settings and the Anthropic chat model factory."""

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic-settings, loaded from env/.env. One source of truth for config."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    embedding_model_name: str = "all-MiniLM-L6-v2"
    kb_path: Path = Path("knowledge_base.txt")
    kb_index_path: Path = Path("kb_index.sqlite3")
    retrieval_k: int = 4
    similarity_floor: float = 0.2

    host: str = "0.0.0.0"
    port: int = 8000

    temperature: float = 0.0


class ChatModelFactory:
    """Builds the Anthropic chat model. Agents/orchestrator code never import
    `langchain_anthropic` directly — only this factory does."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create(self) -> BaseChatModel:
        """Build the configured ChatAnthropic model.

        Raises ValueError if no API key is configured — callers at the
        CLI/API boundary catch this alongside FileNotFoundError.
        """
        if not self._settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set. Get a key at "
                "https://console.anthropic.com/settings/keys."
            )

        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model_name=self._settings.anthropic_model,
            api_key=self._settings.anthropic_api_key,
            temperature=self._settings.temperature,
            timeout=None,
            stop=None,
        )
