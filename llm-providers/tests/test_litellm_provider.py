"""integration tests for LiteLLMProvider -- real API calls only.

skips if no OpenRouter-compatible API key is available.
"""

import os
import pytest

from providers.litellm_provider import LiteLLMProvider
from providers.providers import ChatResult


_has_openrouter_key = bool(
    os.environ.get("OPENROUTER_API_KEY")
    or (os.environ.get("OPENAI_API_KEY", "").startswith("sk-or-"))
)
_openrouter_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


@pytest.mark.skipif(not _has_openrouter_key, reason="no OpenRouter-compatible API key")
class TestLiteLLMProvider:
    """real API calls via LiteLLM routing through OpenRouter."""

    def _make_provider(self, model="openai/gpt-4o-mini"):
        return LiteLLMProvider(
            litellm_model=f"openrouter/{model}",
            api_key=_openrouter_key,
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
        assert result.usage["total_tokens"] > 0

    def test_model_name(self):
        provider = self._make_provider()
        assert provider.model_name == "openrouter/openai/gpt-4o-mini"
