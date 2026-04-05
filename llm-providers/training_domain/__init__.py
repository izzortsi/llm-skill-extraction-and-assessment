"""training_domain -- training pipeline provider abstraction layer.

Re-exports the pipeline provider hierarchy for LLM backends.
Depends on providers for credentials.
"""

from training_domain.pipeline_providers import (
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
