import os
import tempfile
import yaml
import pytest


def test_load_model_config_returns_model_entries():
    """load_model_config parses YAML and returns ModelConfig with model entries."""
    from providers.model_config import load_model_config
    config_data = {
        "models": {
            "qwen2.5-3b": {
                "litellm_model": "openai/qwen2.5:3b",
                "api_base": "http://localhost:8000/v1",
            }
        },
        "judge": {"model": "qwen2.5-3b"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        path = f.name

    try:
        config = load_model_config(path)
        assert "qwen2.5-3b" in config.models
        assert config.models["qwen2.5-3b"].litellm_model == "openai/qwen2.5:3b"
        assert config.models["qwen2.5-3b"].api_base == "http://localhost:8000/v1"
        assert config.judge_model_name == "qwen2.5-3b"
    finally:
        os.unlink(path)


def test_load_model_config_resolves_api_key_from_env():
    """api_key_env field resolves to actual key from environment."""
    from providers.model_config import load_model_config
    config_data = {
        "models": {
            "glm": {
                "litellm_model": "openai/glm-5-turbo",
                "api_key_env": "TEST_GLM_KEY",
            }
        },
        "judge": {"model": "glm"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        path = f.name

    try:
        os.environ["TEST_GLM_KEY"] = "sk-test-123"
        config = load_model_config(path)
        assert config.models["glm"].api_key == "sk-test-123"
    finally:
        os.environ.pop("TEST_GLM_KEY", None)
        os.unlink(path)


def test_load_model_config_raises_on_missing_models_key():
    """Config without 'models' key raises ValueError."""
    from providers.model_config import load_model_config
    config_data = {"judge": {"model": "x"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        path = f.name

    try:
        with pytest.raises(ValueError, match="models"):
            load_model_config(path)
    finally:
        os.unlink(path)


def test_model_entry_list_returns_all_model_names():
    """ModelConfig.model_names returns list of all configured model aliases."""
    from providers.model_config import load_model_config
    config_data = {
        "models": {
            "model-a": {"litellm_model": "openai/a"},
            "model-b": {"litellm_model": "anthropic/b"},
        },
        "judge": {"model": "model-a"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        path = f.name

    try:
        config = load_model_config(path)
        assert sorted(config.model_names) == ["model-a", "model-b"]
    finally:
        os.unlink(path)
