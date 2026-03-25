from unittest.mock import patch, MagicMock
from c1_providers.litellm_provider import LiteLLMProvider

def test_litellm_provider_chat_returns_chat_result():
    """LiteLLMProvider.chat() calls litellm.completion and returns ChatResult."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello from vLLM"
    mock_response.choices[0].message.role = "assistant"
    mock_response.usage.total_tokens = 42
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 32

    with patch("c1_providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.completion.return_value = mock_response
        provider = LiteLLMProvider(
            litellm_model="openai/qwen2.5:3b",
            api_base="http://localhost:8000/v1",
        )
        result = provider.chat([{"role": "user", "content": "Hi"}])

    assert result.message["content"] == "Hello from vLLM"
    assert result.message["role"] == "assistant"
    assert result.usage["total_tokens"] == 42
    mock_litellm.completion.assert_called_once()
    call_kwargs = mock_litellm.completion.call_args
    assert call_kwargs.kwargs["model"] == "openai/qwen2.5:3b"
    assert call_kwargs.kwargs["api_base"] == "http://localhost:8000/v1"


def test_litellm_provider_passes_api_key():
    """LiteLLMProvider passes api_key to litellm.completion."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"
    mock_response.choices[0].message.role = "assistant"
    mock_response.usage.total_tokens = 5
    mock_response.usage.prompt_tokens = 3
    mock_response.usage.completion_tokens = 2

    with patch("c1_providers.litellm_provider.litellm") as mock_litellm:
        mock_litellm.completion.return_value = mock_response
        provider = LiteLLMProvider(
            litellm_model="anthropic/claude-opus-4-6",
            api_key="sk-test",
        )
        provider.chat([{"role": "user", "content": "Hi"}])

    call_kwargs = mock_litellm.completion.call_args
    assert call_kwargs.kwargs["api_key"] == "sk-test"


def test_litellm_provider_model_name_property():
    """model_name property returns the litellm_model string."""
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    provider._litellm_model = "openai/qwen2.5:3b"
    assert provider.model_name == "openai/qwen2.5:3b"


def test_create_provider_litellm_returns_litellm_provider():
    """create_provider('litellm', ...) returns a LiteLLMProvider."""
    from c1_providers.providers import create_provider
    provider = create_provider(
        "litellm",
        model="openai/qwen2.5:3b",
        base_url="http://localhost:8000/v1",
    )
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model_name == "openai/qwen2.5:3b"
