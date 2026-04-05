"""integration tests for c1_providers.providers -- real API calls only.

every test requires its provider's API key to be explicitly set.
missing keys cause hard failures, not silent skips.
"""

import os
import pytest

from c1_providers.providers import ChatResult, OpenAIProvider, AnthropicProvider, create_provider


# ---------------------------------------------------------------------------
# required credentials -- crash if missing
# ---------------------------------------------------------------------------

def _require_env(var_name: str) -> str:
    """return env var value or raise with a clear message."""
    value = os.environ.get(var_name)
    if not value:
        raise ValueError(
            f"{var_name} is required but not set. "
            f"Add it to llm-providers/.env or export it."
        )
    return value


# ---------------------------------------------------------------------------
# OpenAIProvider (via OpenRouter endpoint -- real API call)
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    """test OpenAIProvider by pointing it at OpenRouter (OpenAI-compatible)."""

    def _make_provider(self):
        key = _require_env("OPENROUTER_API_KEY")
        return OpenAIProvider(
            model="openai/gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )

    def test_chat_returns_chat_result(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Reply with exactly: hello"}])

        assert isinstance(result, ChatResult)
        assert result.message["role"] == "assistant"
        assert isinstance(result.message["content"], str)
        assert len(result.message["content"]) > 0

    def test_usage_populated(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Say hi"}])

        assert result.usage["prompt_tokens"] > 0
        assert result.usage["completion_tokens"] > 0
        assert result.usage["total_tokens"] == (
            result.usage["prompt_tokens"] + result.usage["completion_tokens"]
        )

    def test_model_name(self):
        provider = self._make_provider()
        assert provider.model_name == "openai/gpt-4o-mini"

    def test_chat_with_system_message(self):
        provider = self._make_provider()
        result = provider.chat([
            {"role": "system", "content": "You only respond with the word 'pong'."},
            {"role": "user", "content": "ping"},
        ])
        assert isinstance(result.message["content"], str)

    def test_raw_response_present(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Say ok"}])
        assert result.raw is not None


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class TestAnthropicProvider:

    def _make_provider(self):
        _require_env("ANTHROPIC_API_KEY")
        return AnthropicProvider(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    def test_chat_returns_chat_result(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Reply with exactly: hello"}])

        assert isinstance(result, ChatResult)
        assert result.usage["prompt_tokens"] > 0
        assert result.usage["total_tokens"] > 0

    def test_system_message_works(self):
        provider = self._make_provider()
        result = provider.chat([
            {"role": "system", "content": "You only respond with the word 'pong'."},
            {"role": "user", "content": "ping"},
        ])
        assert isinstance(result.message["content"], (str, list))

    def test_model_name(self):
        provider = self._make_provider()
        assert provider.model_name == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# create_provider factory (real API calls)
# ---------------------------------------------------------------------------

class TestCreateProviderFactory:

    def test_openrouter_via_factory(self):
        key = _require_env("OPENROUTER_API_KEY")
        p = create_provider(
            "openai",
            model="openai/gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        result = p.chat([{"role": "user", "content": "Say ok"}])
        assert isinstance(result, ChatResult)
        assert len(result.message["content"]) > 0
