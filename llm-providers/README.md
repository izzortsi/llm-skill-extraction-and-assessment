# llm-providers

shared LLM provider abstraction layer used by all llm-skills pipelines.

## what this project does

provides a unified interface for calling LLMs across providers (Anthropic, OpenAI-compatible,
LiteLLM). includes model config loading from YAML, credential management, mock providers for
testing, schema validation, and statistical utilities (bootstrap confidence interval (CI), pass rate, permutation test).

also contains training-domain pipeline providers for agentic training data generation.

## usage

```python
# registered via _bootstrap.setup_providers()
from providers.providers import create_provider
from providers.model_config import load_model_config

provider = create_provider("anthropic", "claude-opus-4-6")
result = provider.chat([{"role": "user", "content": "hello"}])
```

## structure

```
utils/              uid, stat_utils
providers/          providers, credentials, litellm_provider, mock_provider,
                       model_config, schema_validator, stat_utils
training_domain/    pipeline_providers (agentic training)
configs/               models.yaml, models.example.yaml
docs/                  environment-setup.md
scripts/               oauth_query.py, zai_oauth_query.py
tests/                 test_providers, test_pipeline_providers, test_litellm_provider
```

## configuration

model routing is configured via `configs/models.yaml`. see `docs/environment-setup.md`
for API key and OAuth setup.
