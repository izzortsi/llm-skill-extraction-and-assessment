"""LiteLLM-based provider for unified model routing."""

from typing import Any, Dict, List, Optional

import litellm
from providers.providers import ChatResult


class LiteLLMProvider:
    """Provider that routes to any backend via litellm SDK.

    Supports vLLM, Ollama, OpenAI, Anthropic, and 100+ other providers
    through litellm model-string routing (e.g. "openai/qwen2.5:3b",
    "anthropic/claude-opus-4-6").
    """

    def __init__(
        self,
        litellm_model: str,
        api_base: str = "",
        api_key: str = "",
    ):
        self._litellm_model = litellm_model
        self._api_base = api_base
        self._api_key = api_key

    @property
    def model_name(self) -> str:
        return self._litellm_model

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[list] = None,
    ) -> ChatResult:
        """Send chat completion via litellm routing."""
        kwargs: Dict[str, Any] = {
            "model": self._litellm_model,
            "messages": messages,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if tools:
            kwargs["tools"] = tools

        response = litellm.completion(**kwargs)

        choice = response.choices[0]
        message = {
            "role": choice.message.role,
            "content": choice.message.content or "",
        }
        usage = {
            "total_tokens": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

        return ChatResult(message=message, usage=usage, raw=response)
