import os
import tempfile
import yaml
import pytest
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path


# ---------------------------------------------------------------------------
# Inline model config (replaces c1_providers.model_config)
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    litellm_model: str
    provider: str = "lmproxy"
    api_base: str = ""
    api_key: str = ""
    api_key_env: str = ""


@dataclass
class ModelConfig:
    models: Dict[str, ModelEntry]
    judge_model_name: str

    @property
    def model_names(self) -> List[str]:
        return list(self.models.keys())

    def get_judge_entry(self) -> ModelEntry:
        return self.models[self.judge_model_name]


def load_model_config(path: str) -> ModelConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    if not raw or "models" not in raw:
        raise ValueError("Config must contain a 'models' key")
    models: Dict[str, ModelEntry] = {}
    for name, entry_data in raw["models"].items():
        litellm_model = entry_data.get("litellm_model", "")
        if not litellm_model:
            raise ValueError(f"Model '{name}' missing 'litellm_model' field")
        api_key_env = entry_data.get("api_key_env", "")
        api_key = entry_data.get("api_key", "")
        if api_key_env and not api_key:
            api_key = os.environ.get(api_key_env, "")
        models[name] = ModelEntry(
            litellm_model=litellm_model,
            api_base=entry_data.get("api_base", ""),
            api_key=api_key,
            api_key_env=api_key_env,
        )
    judge_section = raw.get("judge", {})
    judge_model_name = judge_section.get("model", "")
    if judge_model_name and judge_model_name not in models:
        raise ValueError(f"Judge model '{judge_model_name}' not found in models config")
    return ModelConfig(models=models, judge_model_name=judge_model_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_model_config_returns_model_entries():
    """load_model_config parses YAML and returns ModelConfig with model entries."""
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
