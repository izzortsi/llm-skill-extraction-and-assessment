"""
pipeline_providers.py -- backwards-compatibility shim.

Canonical location: training_domain.pipeline_providers
"""

from training_domain.pipeline_providers import (  # noqa: F401
    PipelineChatResult,
    PipelineProvider,
    AnthropicOAuthProvider,
    AnthropicAPIProvider,
    OpenAICompatProvider,
    ZAIProvider,
    OpenRouterProvider,
    ClaudeCodeProvider,
    create_pipeline_provider,
    CLAUDE_CODE_IDENTITY,
)

__all__ = [
    "PipelineChatResult",
    "PipelineProvider",
    "AnthropicOAuthProvider",
    "AnthropicAPIProvider",
    "OpenAICompatProvider",
    "ZAIProvider",
    "OpenRouterProvider",
    "ClaudeCodeProvider",
    "create_pipeline_provider",
    "CLAUDE_CODE_IDENTITY",
]
