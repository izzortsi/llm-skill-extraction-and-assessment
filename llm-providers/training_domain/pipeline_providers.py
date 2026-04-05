"""
pipeline_providers.py

Provider abstraction for LLM backends (training pipeline).

Normalizes responses to a common format:
  message: {"role": "assistant", "content": str, "tool_calls": [...] | None}
  usage:   {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}

Swap providers by changing one constructor call. The harness never
knows which backend is active.

This module is the canonical source for the rich provider hierarchy
used by training-pipeline and related packages.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineChatResult:
    """normalized llm response."""
    message: dict[str, Any]         # normalized: role, content, tool_calls (for training data)
    usage: dict[str, int]           # prompt_tokens, completion_tokens, total_tokens
    raw_response: dict[str, Any]    # full provider-specific response for training data
    reasoning: str | None = None    # chain-of-thought if available
    history_message: dict[str, Any] | None = None  # provider-native format for conversation history


class PipelineProvider(ABC):
    """base class for llm providers."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    def uses_native_tools(self) -> bool:
        """whether the provider supports native tool calling (tool_calls in response).
        if False, tool descriptions go in system prompt and tool calls are parsed from text."""
        return True

    @property
    def message_format(self) -> str:
        """message format expected by the provider ("openai" or "anthropic")."""
        return "openai"

    @property
    def is_external(self) -> bool:
        """whether this provider uses an external agent process (e.g. claude code CLI).
        if True, use ClaudeCodeRunner instead of LinearHarness."""
        return False

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> PipelineChatResult: ...


CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


class AnthropicOAuthProvider(PipelineProvider):
    """anthropic via oauth -- $0 inference through claude pro/max subscription.

    auth priority:
    1. anthropic_oauth package (browser-based OAuth flow)
    2. Claude Code local credentials (~/.claude/.credentials.json)

    OAuth requires the 'anthropic-beta: oauth-2025-04-20' header.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 32768, **kwargs):
        self._model = model
        self._max_tokens = max_tokens

        # try anthropic_oauth library first
        try:
            from anthropic_oauth import create_oauth_client
            self._client = create_oauth_client(**kwargs)
            return
        except ImportError:
            pass

        # fall back to Claude Code local credentials
        import anthropic
        from providers.credentials import get_claude_oauth_token
        token = get_claude_oauth_token()
        if not token:
            raise RuntimeError(
                "anthropic_oauth not installed and no Claude Code credentials found. "
                "install anthropic-oauth, or log in with Claude Code (claude login)."
            )
        self._client = anthropic.Anthropic(
            auth_token=token,
            base_url="https://api.anthropic.com",
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def message_format(self) -> str:
        return "anthropic"

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> PipelineChatResult:
        # anthropic messages api: system goes as top-level param, not in messages
        system_parts = [CLAUDE_CODE_IDENTITY]
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            else:
                api_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
            "system": "\n\n".join(system_parts),
        }
        if tools:
            kwargs["tools"] = tools

        # use streaming to avoid SDK's non-streaming timeout limit
        # (ValueError at max_tokens >= ~16K: "Streaming is required")
        with self._client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
        return self._normalize(response)

    def _normalize(self, response) -> PipelineChatResult:
        """convert anthropic response to common format."""
        content_text = ""
        tool_calls = []
        reasoning = None

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "thinking" and len(block.thinking) < 5000:
                reasoning = block.thinking
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        message = {
            "role": "assistant",
            "content": content_text or None,
            "tool_calls": tool_calls or None,
        }
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        # build provider-native message for conversation history
        # anthropic expects content as list of blocks, not tool_calls field
        history_content = []
        for block in response.content:
            if block.type == "text":
                history_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                history_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        history_message = {"role": "assistant", "content": history_content}

        return PipelineChatResult(
            message=message,
            usage=usage,
            raw_response=response.model_dump(),
            reasoning=reasoning,
            history_message=history_message,
        )


class AnthropicAPIProvider(PipelineProvider):
    """standard anthropic api -- per-token billing.

    if no api_key is provided, falls back to ANTHROPIC_API_KEY env var,
    then Claude Code local credentials (~/.claude/.credentials.json).
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 32768, api_key: str | None = None):
        import anthropic
        from providers.credentials import get_claude_oauth_token, get_anthropic_api_key
        self._model = model
        self._max_tokens = max_tokens

        if api_key:
            self._client = anthropic.Anthropic(api_key=api_key)
            return

        # try Claude Code OAuth credentials
        oauth_token = get_claude_oauth_token()
        if oauth_token:
            self._client = anthropic.Anthropic(
                auth_token=oauth_token,
                base_url="https://api.anthropic.com",
                default_headers={"anthropic-beta": "oauth-2025-04-20"},
            )
            return

        # fall back to API key from env
        env_key = get_anthropic_api_key()
        if env_key:
            self._client = anthropic.Anthropic(api_key=env_key)
            return

        self._client = anthropic.Anthropic()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def message_format(self) -> str:
        return "anthropic"

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> PipelineChatResult:
        # anthropic messages api: system goes as top-level param, not in messages
        system_parts = []
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg.get("content", ""))
            else:
                api_messages.append(msg)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if tools:
            kwargs["tools"] = tools

        # use streaming to avoid SDK's non-streaming timeout limit
        with self._client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
        # reuse oauth provider's normalizer -- same response format
        return AnthropicOAuthProvider._normalize(self, response)


class OpenAICompatProvider(PipelineProvider):
    """openai-compatible provider -- works with openai api, zai paas, vllm, etc."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 32768,
    ):
        from openai import OpenAI
        self._model = model
        self._max_tokens = max_tokens
        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self._client = OpenAI(**kwargs)

    @property
    def model_name(self) -> str:
        return self._model

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> PipelineChatResult:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        return self._normalize(response)

    def _normalize(self, response) -> PipelineChatResult:
        choice = response.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        message = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": tool_calls,
        }

        reasoning = None
        if hasattr(msg, "reasoning_content") and msg.reasoning_content and len(msg.reasoning_content) < 5000:
            reasoning = msg.reasoning_content

        usage_data = {}
        if response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return PipelineChatResult(
            message=message,
            usage=usage_data,
            raw_response=response.model_dump(),
            reasoning=reasoning,
        )


class ZAIProvider(OpenAICompatProvider):
    """z.ai paas -- subscription billing via glm-5/glm-4.x.

    uses openai sdk pointed at z.ai paas endpoint.
    does NOT send tools parameter -- glm-5 ignores native tool calling
    via paas. tool descriptions go in system prompt instead.
    """

    def __init__(
        self,
        model: str = "glm-5",
        api_key: str | None = None,
        max_tokens: int = 32768,
    ):
        import os
        effective_key = api_key or os.environ.get("ZHIPU_API_KEY")
        if not effective_key:
            raise ValueError(
                "ZHIPU_API_KEY is required for ZAIProvider. "
                "Set it as an environment variable or in a .env file."
            )
        super().__init__(
            model=model,
            base_url="https://api.z.ai/api/coding/paas/v4",
            api_key=effective_key,
            max_tokens=max_tokens,
        )

    @property
    def uses_native_tools(self) -> bool:
        return False

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> PipelineChatResult:
        # intentionally suppress tools -- glm-5 ignores them via paas
        # tool descriptions should be in the system prompt
        return super().chat(messages, tools=None)


class OpenRouterProvider(OpenAICompatProvider):
    """openrouter provider -- openai-compatible api via openrouter.ai.

    uses openai sdk pointed at openrouter endpoint.
    default auth uses OPENROUTER_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        max_tokens: int = 32768,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        import os
        effective_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not effective_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for OpenRouterProvider. "
                "Set it as an environment variable or in a .env file."
            )
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=effective_key,
            max_tokens=max_tokens,
        )


class ClaudeCodeProvider(PipelineProvider):
    """claude code CLI as an external agent.

    this provider does NOT call the api directly. the claude code CLI
    handles its own auth and api calls. a capture proxy intercepts
    the traffic for training data.

    chat() raises NotImplementedError because the agent loop is
    managed by ClaudeCodeRunner, not LinearHarness.
    """

    def __init__(self, model: str = "claude-code", **kwargs):
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_external(self) -> bool:
        return True

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> PipelineChatResult:
        raise NotImplementedError(
            "ClaudeCodeProvider does not support chat(). "
            "use ClaudeCodeRunner instead of LinearHarness."
        )


def create_pipeline_provider(provider_type: str, **kwargs) -> PipelineProvider:
    """factory for creating providers by name."""
    providers = {
        "claude-code": ClaudeCodeProvider,
        "anthropic-oauth": AnthropicOAuthProvider,
        "anthropic": AnthropicAPIProvider,
        "openai": OpenAICompatProvider,
        "openrouter": OpenRouterProvider,
        "zai": ZAIProvider,
    }
    if provider_type not in providers:
        raise ValueError(f"unknown provider: {provider_type}. available: {list(providers.keys())}")
    return providers[provider_type](**kwargs)
