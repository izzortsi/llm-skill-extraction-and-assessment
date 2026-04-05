"""Load and validate YAML model configuration for LiteLLM routing."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class ModelEntry:
    """Single model routing entry."""
    litellm_model: str          # e.g. "openai/qwen2.5:3b" or "claude-opus-4-6"
    provider: str = "litellm"   # "litellm" (default), "anthropic", "openai", etc.
    api_base: str = ""          # e.g. "http://vllm-host:8000/v1"
    api_key: str = ""           # resolved from api_key_env or direct
    api_key_env: str = ""       # env var name to read API key from


@dataclass
class ModelConfig:
    """Parsed model configuration."""
    models: Dict[str, ModelEntry]
    judge_model_name: str

    @property
    def model_names(self) -> List[str]:
        return list(self.models.keys())

    def get_judge_entry(self) -> ModelEntry:
        return self.models[self.judge_model_name]


def load_model_config(path: str) -> ModelConfig:
    """Load model config from YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        ModelConfig with resolved API keys.

    Raises:
        ValueError: If required keys are missing.
        FileNotFoundError: If config file does not exist.
    """
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
            provider=entry_data.get("provider", "litellm"),
            api_base=entry_data.get("api_base", ""),
            api_key=api_key,
            api_key_env=api_key_env,
        )

    judge_section = raw.get("judge", {})
    judge_model_name = judge_section.get("model", "")
    if judge_model_name and judge_model_name not in models:
        raise ValueError(
            f"Judge model '{judge_model_name}' not found in models config"
        )

    return ModelConfig(models=models, judge_model_name=judge_model_name)
