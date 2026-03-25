"""integration tests for c1_providers.pipeline_providers -- real API calls only.

every test requires its provider's API key to be explicitly set.
missing keys cause hard failures, not silent skips.
"""

import os
import json
import pytest

from c1_providers.pipeline_providers import (
    PipelineChatResult,
    AnthropicOAuthProvider,
    AnthropicAPIProvider,
    OpenAICompatProvider,
    ZAIProvider,
    OpenRouterProvider,
    ClaudeCodeProvider,
    create_pipeline_provider,
)


# ---------------------------------------------------------------------------
# required credentials -- crash if missing
# ---------------------------------------------------------------------------

def _require_env(var_name: str) -> str:
    """return env var value or raise with a clear message."""
    value = os.environ.get(var_name)
    if not value:
        raise ValueError(
            f"{var_name} is required but not set. "
            f"Add it to llm-skills.llm-providers/.env or export it."
        )
    return value


# ---------------------------------------------------------------------------
# OpenAICompatProvider (via OpenRouter -- real API call)
# ---------------------------------------------------------------------------

class TestOpenAICompatProvider:

    def _make_provider(self, model="openai/gpt-4o-mini"):
        key = _require_env("OPENROUTER_API_KEY")
        return OpenAICompatProvider(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )

    def test_chat_returns_pipeline_chat_result(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Reply with exactly: hello"}])

        assert isinstance(result, PipelineChatResult)
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

    def test_raw_response_populated(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Say ok"}])

        assert isinstance(result.raw_response, dict)
        assert len(result.raw_response) > 0

    def test_model_name(self):
        provider = self._make_provider(model="openai/gpt-4o-mini")
        assert provider.model_name == "openai/gpt-4o-mini"

    def test_no_tool_calls_for_plain_text(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Say hello"}])
        assert result.message["tool_calls"] is None

    def test_chat_with_system_message(self):
        provider = self._make_provider()
        result = provider.chat([
            {"role": "system", "content": "You only respond with the word 'pong'."},
            {"role": "user", "content": "ping"},
        ])
        assert isinstance(result.message["content"], str)

    def test_tool_calling(self):
        """provider sends tools and model returns a tool call."""
        provider = self._make_provider()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        result = provider.chat(
            [{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=tools,
        )

        assert result.message["tool_calls"] is not None
        tc = result.message["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        args = json.loads(tc["function"]["arguments"])
        assert "city" in args


# ---------------------------------------------------------------------------
# OpenRouterProvider (real API call)
# ---------------------------------------------------------------------------

class TestOpenRouterProvider:

    def _make_provider(self, model="openai/gpt-4o-mini"):
        key = _require_env("OPENROUTER_API_KEY")
        return OpenRouterProvider(model=model, api_key=key)

    def test_chat_returns_result(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Reply with exactly: hello"}])

        assert isinstance(result, PipelineChatResult)
        assert result.message["role"] == "assistant"
        assert len(result.message["content"]) > 0

    def test_usage_populated(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Say hi"}])
        assert result.usage["prompt_tokens"] > 0
        assert result.usage["total_tokens"] > 0

    def test_model_name(self):
        provider = self._make_provider()
        assert provider.model_name == "openai/gpt-4o-mini"

    def test_tool_calling(self):
        provider = self._make_provider()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]
        result = provider.chat(
            [{"role": "user", "content": "Add 3 and 5"}],
            tools=tools,
        )
        assert result.message["tool_calls"] is not None
        tc = result.message["tool_calls"][0]
        assert tc["function"]["name"] == "add"


# ---------------------------------------------------------------------------
# ZAIProvider (real API call)
# ---------------------------------------------------------------------------

class TestZAIProvider:

    def _make_provider(self, model="glm-5"):
        _require_env("ZHIPU_API_KEY")
        return ZAIProvider(model=model)

    def test_chat_returns_result(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Reply with exactly: hello"}])

        assert isinstance(result, PipelineChatResult)
        assert result.message["role"] == "assistant"
        assert isinstance(result.message["content"], str)
        assert len(result.message["content"]) > 0

    def test_usage_populated(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Say hi"}])
        assert result.usage["prompt_tokens"] > 0
        assert result.usage["total_tokens"] > 0

    def test_model_name(self):
        provider = self._make_provider()
        assert provider.model_name == "glm-5"

    def test_tools_suppressed(self):
        """even when tools are passed, ZAI provider strips them before calling API."""
        provider = self._make_provider()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a command",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            }
        ]
        result = provider.chat(
            [{"role": "user", "content": "Say hello"}],
            tools=tools,
        )
        assert isinstance(result, PipelineChatResult)

    def test_raw_response_populated(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Say ok"}])
        assert isinstance(result.raw_response, dict)
        assert len(result.raw_response) > 0


# ---------------------------------------------------------------------------
# AnthropicOAuthProvider (real API call)
# ---------------------------------------------------------------------------

class TestAnthropicOAuthProvider:

    def _make_provider(self):
        # requires either anthropic_oauth package or Claude Code OAuth credentials
        try:
            from anthropic_oauth import create_oauth_client
        except ImportError:
            from c1_providers.credentials import get_claude_oauth_token
            if not get_claude_oauth_token():
                raise ValueError(
                    "AnthropicOAuthProvider requires either the anthropic_oauth package "
                    "or Claude Code OAuth credentials (~/.claude/.credentials.json). "
                    "Run 'claude login' or 'pip install anthropic-oauth'."
                )
        return AnthropicOAuthProvider(model="claude-haiku-4-5-20251001")

    def test_chat_returns_result(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Reply with exactly: hello"}])

        assert isinstance(result, PipelineChatResult)
        assert result.message["role"] == "assistant"
        assert isinstance(result.message["content"], str)

    def test_usage_populated(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Say hi"}])
        assert result.usage["prompt_tokens"] > 0
        assert result.usage["total_tokens"] > 0

    def test_history_message_built(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Say ok"}])
        assert result.history_message is not None
        assert result.history_message["role"] == "assistant"
        assert isinstance(result.history_message["content"], list)

    def test_tool_calling(self):
        provider = self._make_provider()
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            }
        ]
        result = provider.chat(
            [{"role": "user", "content": "What is the weather in Tokyo? Use the get_weather tool."}],
            tools=tools,
        )
        assert result.message["tool_calls"] is not None
        tc = result.message["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# AnthropicAPIProvider (real API call)
# ---------------------------------------------------------------------------

class TestAnthropicAPIProvider:

    def _make_provider(self):
        _require_env("ANTHROPIC_API_KEY")
        return AnthropicAPIProvider(
            model="claude-haiku-4-5-20251001",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )

    def test_chat_returns_result(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Reply with exactly: hello"}])

        assert isinstance(result, PipelineChatResult)
        assert result.message["role"] == "assistant"
        assert isinstance(result.message["content"], str)

    def test_usage_populated(self):
        provider = self._make_provider()
        result = provider.chat([{"role": "user", "content": "Say hi"}])
        assert result.usage["prompt_tokens"] > 0
        assert result.usage["total_tokens"] > 0

    def test_system_message_works(self):
        provider = self._make_provider()
        result = provider.chat([
            {"role": "system", "content": "You only respond with the word 'pong'."},
            {"role": "user", "content": "ping"},
        ])
        assert isinstance(result.message["content"], str)


# ---------------------------------------------------------------------------
# create_pipeline_provider factory (real API calls)
# ---------------------------------------------------------------------------

class TestCreatePipelineProvider:

    def test_openai_via_factory(self):
        key = _require_env("OPENROUTER_API_KEY")
        p = create_pipeline_provider(
            "openai",
            model="openai/gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        result = p.chat([{"role": "user", "content": "Say ok"}])
        assert isinstance(result, PipelineChatResult)
        assert len(result.message["content"]) > 0

    def test_openrouter_via_factory(self):
        key = _require_env("OPENROUTER_API_KEY")
        p = create_pipeline_provider(
            "openrouter",
            model="openai/gpt-4o-mini",
            api_key=key,
        )
        result = p.chat([{"role": "user", "content": "Say ok"}])
        assert isinstance(result, PipelineChatResult)
        assert len(result.message["content"]) > 0

    def test_zai_via_factory(self):
        _require_env("ZHIPU_API_KEY")
        p = create_pipeline_provider("zai")
        result = p.chat([{"role": "user", "content": "Say ok"}])
        assert isinstance(result, PipelineChatResult)
