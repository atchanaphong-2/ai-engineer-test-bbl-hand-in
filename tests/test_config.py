"""Tests for ChatModelFactory. Constructing ChatAnthropic validates and
stores its config but makes no network call, so these need no live LLM."""

import pytest
from langchain_anthropic import ChatAnthropic

from agentic_rag.config import ChatModelFactory, Settings


def test_anthropic_requires_api_key():
    settings = Settings(anthropic_api_key=None)
    factory = ChatModelFactory(settings)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        factory.create()


def test_create_builds_chat_anthropic_from_settings():
    settings = Settings(
        anthropic_api_key="sk-ant-test-fake",
        anthropic_model="claude-sonnet-4-5-20250929",
        temperature=0.7,
    )
    factory = ChatModelFactory(settings)

    model = factory.create()

    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-4-5-20250929"
    assert model.temperature == 0.7
