"""
providers.py

LLM provider abstraction. Supports OpenAI-compatible APIs (Ollama, vLLM,
OpenRouter, Z.AI) and Anthropic (API key or OAuth).

Provider usage examples:

    # Ollama (local)
    create_provider("openai", "qwen2.5:3b", base_url="http://localhost:11434/v1")

    # OpenRouter (requires OPENROUTER_API_KEY env var)
    create_provider("openrouter", "anthropic/claude-sonnet-4-5-20250929")

    # Z.AI / ZhipuAI (OpenAI-compatible, requires OPENAI_API_KEY env var set to ZhipuAI key)
    create_provider("openai", "glm-5-turbo", base_url="https://api.z.ai/api/coding/paas/v4")

    # Anthropic (tries OAuth first, falls back to ANTHROPIC_API_KEY)
    create_provider("anthropic", "claude-opus-4-6")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChatResult:
    """Result from a provider.chat() call."""
    message: Dict[str, Any]
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None


class OpenAIProvider:
    """OpenAI-compatible API provider (works with Ollama, vLLM, OpenAI)."""

    def __init__(
        self,
        model: str,
        base_url: str = "",
        api_key: str = "",
        tools: list = None,
        headers: dict = None,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.model = model
        self._tools = tools or []
        effective_base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        effective_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = openai.OpenAI(
            base_url=effective_base_url or None,
            api_key=effective_api_key or "none",
            default_headers=headers or {},
        )

    @property
    def model_name(self) -> str:
        return self.model

    def chat(self, messages: List[Dict[str, Any]], tools: list = None) -> ChatResult:
        kwargs = {"model": self.model, "messages": messages}
        effective_tools = tools or self._tools
        if effective_tools:
            kwargs["tools"] = effective_tools

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        msg = {"role": "assistant", "content": None}
        if choice.message.content:
            msg["content"] = choice.message.content

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
        if tool_calls:
            msg["tool_calls"] = tool_calls

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        return ChatResult(message=msg, usage=usage, raw=response)


class AnthropicProvider:
    """Anthropic provider supporting API key and OAuth token auth."""

    def __init__(self, model: str, api_key: str = ""):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        self.model = model

        try:
            from anthropic_oauth import create_oauth_client
            self._client = create_oauth_client()
            return
        except ImportError:
            pass
        except Exception:
            pass

        # try Claude Code local credentials (OAuth token)
        from c1_providers.credentials import get_claude_oauth_token, get_anthropic_api_key
        oauth_token = get_claude_oauth_token()
        if not api_key and oauth_token:
            self._client = anthropic.Anthropic(
                auth_token=oauth_token,
                base_url="https://api.anthropic.com",
                default_headers={"anthropic-beta": "oauth-2025-04-20"},
            )
            return

        # fall back to API key
        token = api_key or get_anthropic_api_key()
        if not token:
            raise ValueError(
                "Anthropic credentials required: set ANTHROPIC_API_KEY, "
                "log in with Claude Code (claude login), or pass --api-key"
            )
        self._client = anthropic.Anthropic(api_key=token)

    @property
    def model_name(self) -> str:
        return self.model

    def chat(self, messages: List[Dict[str, Any]], tools: list = None) -> ChatResult:
        system_text = ""
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                conversation.append(msg)

        kwargs = {"model": self.model, "messages": conversation, "max_tokens": 4096}
        if system_text:
            kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)

        content_blocks = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    },
                })

        msg = {"role": "assistant", "content": content_blocks}
        if tool_calls:
            msg["tool_calls"] = tool_calls

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return ChatResult(message=msg, usage=usage, raw=response)


def create_provider(
    provider_name: str,
    model: str,
    base_url: str = "",
    api_key: str = "",
    tools: list = None,
) -> Any:
    """Create a provider by name ('openai', 'anthropic', 'mock')."""
    if provider_name == "mock":
        from c1_providers.mock_provider import MockProvider
        return MockProvider(model=model, seed=42)

    if provider_name == "openai":
        return OpenAIProvider(model=model, base_url=base_url, api_key=api_key, tools=tools)

    if provider_name == "openrouter":
        effective_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not effective_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required for openrouter provider. "
                "Set it as an environment variable or in a .env file."
            )
        return OpenAIProvider(
            model=model,
            base_url=base_url or "https://openrouter.ai/api/v1",
            api_key=effective_key,
            tools=tools,
            headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://localhost"),
                "X-Title": os.environ.get("OPENROUTER_TITLE", "skillmix"),
            },
        )

    if provider_name in ("anthropic", "anthropic-oauth"):
        return AnthropicProvider(model=model, api_key=api_key)

    if provider_name == "litellm":
        from c1_providers.litellm_provider import LiteLLMProvider
        return LiteLLMProvider(
            litellm_model=model,
            api_base=base_url,
            api_key=api_key,
        )

    if provider_name == "claude-code":
        import sys
        from pathlib import Path
        # Walk up from providers.py to find repo root (contains llm-skills.cli/)
        candidate = Path(__file__).resolve().parent
        cli_root = None
        for _i in range(6):
            candidate = candidate.parent
            if (candidate / "llm-skills.cli" / "c1_tools").is_dir():
                cli_root = candidate / "llm-skills.cli"
                break
        if cli_root is None:
            raise RuntimeError("cannot locate llm-skills.cli relative to providers.py")
        if str(cli_root) not in sys.path:
            sys.path.insert(0, str(cli_root))
        from c1_tools.claude_code_provider import ClaudeCodeProvider
        return ClaudeCodeProvider(model=model)

    raise ValueError(f"Unknown provider: {provider_name}")
