# Environment Variable Setup

This document describes the environment variables required by each LLM provider.

## Required Variables by Provider

### ZAI (Z.AI PaaS -- GLM-5)

```bash
export ZHIPU_API_KEY="your-zhipu-api-key"
```

- Endpoint: `https://api.z.ai/api/coding/paas/v4`
- Billing: subscription-based
- Native tool calling: not supported (tool descriptions injected into system prompt)

### OpenRouter

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key"
```

- Endpoint: `https://openrouter.ai/api/v1`
- Billing: per-token
- Supports 100+ models via routing (e.g., `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-5`)

### Anthropic API

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key"
```

- Endpoint: managed by anthropic SDK
- Billing: per-token
- Credential priority: `api_key` param > `ANTHROPIC_API_KEY` env var > Claude Code local credentials > `CLAUDE_CODE_OAUTH_TOKEN` env var

### Anthropic OAuth

No environment variable needed. Requires the `anthropic_oauth` Python package:

```bash
pip install anthropic-oauth
```

- Billing: $0 inference through Claude Pro/Max subscription
- Auth: browser-based OAuth flow managed by the library
- First run will open a browser to authenticate
- Token is managed and refreshed by the library automatically

Note: Claude Code's local OAuth token (`~/.claude/.credentials.json`) cannot be reused
by external tools — the Anthropic API rejects OAuth tokens outside of Claude Code's
own auth flow.

### OpenAI

```bash
export OPENAI_API_KEY="sk-your-key"
```

- Endpoint: `https://api.openai.com/v1` (default), or set `OPENAI_BASE_URL` for compatible APIs
- Also works with Ollama, vLLM, or any OpenAI-compatible endpoint

### LiteLLM

LiteLLM routes to any backend. Set the environment variable for whichever backend you target:

```bash
# for OpenRouter routing
export OPENROUTER_API_KEY="sk-or-v1-your-key"

# for direct OpenAI
export OPENAI_API_KEY="sk-your-key"

# for Anthropic
export ANTHROPIC_API_KEY="sk-ant-your-key"
```

## Optional Variables

```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"   # point OpenAI provider at local Ollama/vLLM
export OPENROUTER_REFERER="https://your-app.com"     # OpenRouter tracking header
export OPENROUTER_TITLE="your-app-name"              # OpenRouter tracking header
export CLAUDE_CODE_OAUTH_TOKEN="your-token"           # fallback for Anthropic auth
```

## Configuration File

The `configs/models.yaml` file maps model names to provider settings. API keys are referenced by environment variable name, not stored directly:

```yaml
models:
  glm-5-turbo:
    litellm_model: "openai/glm-5-turbo"
    api_base: "https://api.z.ai/api/coding/paas/v4"
    api_key_env: "ZHIPU_API_KEY"

  claude-opus-4-6:
    provider: "anthropic"
    litellm_model: "claude-opus-4-6"
```

The `api_key_env` field is resolved at runtime via `os.environ.get()`.

## Setting Variables

### Option 1: `.env` file (recommended for local development)

Create a `.env` file in the `llm-providers/` directory:

```bash
ZHIPU_API_KEY="your-zhipu-key"
OPENROUTER_API_KEY="sk-or-v1-your-key"
ANTHROPIC_API_KEY="sk-ant-your-key"
```

The `.env` file is loaded automatically by conftest.py when running tests.
It is already in `.gitignore` and will not be committed.

### Option 2: Shell environment variables

#### Linux / macOS

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export ZHIPU_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

Then reload: `source ~/.bashrc`

#### Windows

```powershell
[System.Environment]::SetEnvironmentVariable("ZHIPU_API_KEY", "your-key", "User")
[System.Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", "your-key", "User")
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "your-key", "User")
```

Or via Settings > System > Advanced > Environment Variables.

## Running Tests

```bash
cd llm-providers

# run all pipeline provider tests
pytest tests/test_pipeline_providers.py -v

# run a specific provider's tests
pytest tests/test_pipeline_providers.py::TestZAIProvider -v
```

Providers will raise `ValueError` at construction time if the required API key
is missing. Tests use `pytest.mark.skipif` to skip providers whose keys are
not set, so you will see SKIPs for unconfigured providers and hard crashes
for misconfigured ones.

## Security

- Never commit API keys to the repository
- The `.env` file is in `.gitignore` and must not be checked in
- All keys are read from environment variables at runtime
- The `configs/models.yaml` file only stores environment variable names, not actual keys
