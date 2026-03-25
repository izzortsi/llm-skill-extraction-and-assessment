"""
config_generator.py

Generates models.yaml from a PipelineProfile's eval_models list.
Handles lmproxy, anthropic, ollama, and openai providers.
"""

import re
import yaml
from pathlib import Path

from c0_config.pipeline_profile import PipelineProfile


def _resolve_litellm_model(entry):
    """Resolve the litellm_model string for a model entry.

    Resolution order:
    (a) use entry["litellm_model"] if present
    (b) anthropic: bare model name
    (c) openai: openai/{model}
    (d) lmproxy/ollama: apply hyphen-to-colon heuristic then prefix with openai/
    """
    if "litellm_model" in entry:
        return entry["litellm_model"]

    provider = entry.get("provider", "")
    model = entry["model"]

    if provider == "claude-code":
        return model  # pass through the actual model name (e.g. claude-sonnet-4-6)

    if provider == "anthropic":
        return model

    if provider == "openai":
        return "openai/" + model

    # lmproxy, ollama, or unknown: apply heuristic
    # converts trailing -<version> to :<version>, e.g. qwen2.5-3b -> qwen2.5:3b
    converted = re.sub(r"-(\d[\d.]*[a-zA-Z]?)$", r":\1", model)
    return "openai/" + converted


def _build_model_entry(entry, profile):
    """Build one YAML model entry dict for the given model entry and profile."""
    provider = entry.get("provider", "")
    model_entry = {}

    model_entry["litellm_model"] = _resolve_litellm_model(entry)

    if provider == "anthropic":
        model_entry["api_key_env"] = "ANTHROPIC_API_KEY"
    elif provider == "ollama":
        model_entry["api_base"] = profile.ollama_url
    elif provider == "claude-code":
        model_entry["provider"] = "claude-code"
    else:
        # lmproxy, openai, or unknown default to lmproxy base url
        model_entry["api_base"] = profile.lmproxy_base_url

    return model_entry


def generate_models_yaml(profile, output_path):
    """Generate models.yaml from the profile's eval_models list and write to output_path.

    Adds the judge model to the models section if it is not already present from eval_models.
    Writes the judge section with the judge model name.
    """
    models = {}

    for entry in profile.eval_models:
        model_name = entry["model"]
        models[model_name] = _build_model_entry(entry, profile)

    # add judge model if not already covered by eval_models
    judge_model_name = profile.judge_model
    if judge_model_name not in models:
        judge_entry = {
            "provider": profile.judge_provider,
            "model": judge_model_name,
        }
        models[judge_model_name] = _build_model_entry(judge_entry, profile)

    config = {
        "models": models,
        "judge": {
            "model": judge_model_name,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
