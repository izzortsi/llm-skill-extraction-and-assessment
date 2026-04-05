"""llm-providers -- shared LLM provider abstraction layer.

public API re-exports. imports are deferred to avoid hard failures
when optional dependencies (pyyaml, etc.) are not installed.
"""

from c1_providers.providers import ChatResult, create_provider
from c1_providers.credentials import get_claude_oauth_token, get_anthropic_api_key
from c1_providers.mock_provider import MockProvider, MockChatResult

# re-export from c0_utils (canonical location for utilities)
from c0_utils.uid import generate_uid
from c0_utils.stat_utils import bootstrap_ci, pass_rate, pass_rate_delta_pp, permutation_test

# re-export from c2_training_domain (canonical location for pipeline providers)
# deferred to avoid bootstrap ordering issues
try:
    from c2_training_domain.pipeline_providers import (
        PipelineChatResult,
        PipelineProvider,
        AnthropicOAuthProvider,
        AnthropicAPIProvider,
        OpenAICompatProvider,
        ZAIProvider,
        OpenRouterProvider,
        ClaudeCodeProvider,
        create_pipeline_provider,
    )
except ImportError:
    pass

# model_config and schema_validator require pyyaml
try:
    from c1_providers.model_config import ModelConfig, ModelEntry, load_model_config
except ImportError:
    pass

try:
    from c1_providers.schema_validator import validate_skills_json, validate_tasks_json
except ImportError:
    pass

__all__ = [
    "ChatResult",
    "create_provider",
    "PipelineChatResult",
    "PipelineProvider",
    "AnthropicOAuthProvider",
    "AnthropicAPIProvider",
    "OpenAICompatProvider",
    "ZAIProvider",
    "OpenRouterProvider",
    "ClaudeCodeProvider",
    "create_pipeline_provider",
    "get_claude_oauth_token",
    "get_anthropic_api_key",
    "generate_uid",
    "MockProvider",
    "MockChatResult",
    "bootstrap_ci",
    "pass_rate",
    "pass_rate_delta_pp",
    "permutation_test",
    "ModelConfig",
    "ModelEntry",
    "load_model_config",
    "validate_skills_json",
    "validate_tasks_json",
]
