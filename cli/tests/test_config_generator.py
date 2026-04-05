# tests/test_config_generator.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from config.pipeline_profile import PipelineProfile
from orchestration.config_generator import generate_models_yaml


def test_generate_lmproxy_models(tmp_path):
    profile = PipelineProfile()
    profile.eval_models = [
        {"provider": "lmproxy", "model": "qwen2.5-3b", "litellm_model": "openai/qwen2.5:3b"},
    ]
    profile.judge_provider = "lmproxy"
    profile.judge_model = "claude-opus-4-6"
    output = tmp_path / "models.yaml"
    generate_models_yaml(profile, output)
    with open(output) as f:
        data = yaml.safe_load(f)
    assert "qwen2.5-3b" in data["models"]
    assert data["models"]["qwen2.5-3b"]["litellm_model"] == "openai/qwen2.5:3b"
    assert data["models"]["qwen2.5-3b"]["api_base"] == profile.lmproxy_base_url


def test_generate_anthropic_model(tmp_path):
    profile = PipelineProfile()
    profile.eval_models = [
        {"provider": "anthropic", "model": "claude-opus-4-6"},
    ]
    profile.judge_provider = "anthropic"
    profile.judge_model = "claude-opus-4-6"
    output = tmp_path / "models.yaml"
    generate_models_yaml(profile, output)
    with open(output) as f:
        data = yaml.safe_load(f)
    entry = data["models"]["claude-opus-4-6"]
    assert entry["litellm_model"] == "claude-opus-4-6"
    assert entry["api_key_env"] == "ANTHROPIC_API_KEY"


def test_judge_included_even_if_not_in_eval(tmp_path):
    profile = PipelineProfile()
    profile.eval_models = [
        {"provider": "lmproxy", "model": "qwen2.5-3b", "litellm_model": "openai/qwen2.5:3b"},
    ]
    profile.judge_provider = "anthropic"
    profile.judge_model = "claude-opus-4-6"
    output = tmp_path / "models.yaml"
    generate_models_yaml(profile, output)
    with open(output) as f:
        data = yaml.safe_load(f)
    assert "claude-opus-4-6" in data["models"]
    assert data["judge"]["model"] == "claude-opus-4-6"


def test_ollama_model_uses_ollama_url(tmp_path):
    profile = PipelineProfile()
    profile.eval_models = [
        {"provider": "ollama", "model": "llama3.2-1b", "litellm_model": "openai/llama3.2:1b"},
    ]
    profile.judge_provider = "lmproxy"
    profile.judge_model = "qwen2.5-3b"
    output = tmp_path / "models.yaml"
    generate_models_yaml(profile, output)
    with open(output) as f:
        data = yaml.safe_load(f)
    assert data["models"]["llama3.2-1b"]["api_base"] == profile.ollama_url


def test_custom_model_without_litellm_model_uses_heuristic(tmp_path):
    profile = PipelineProfile()
    profile.eval_models = [
        {"provider": "lmproxy", "model": "qwen2.5-3b"},
    ]
    profile.judge_provider = "lmproxy"
    profile.judge_model = "qwen2.5-3b"
    output = tmp_path / "models.yaml"
    generate_models_yaml(profile, output)
    with open(output) as f:
        data = yaml.safe_load(f)
    assert data["models"]["qwen2.5-3b"]["litellm_model"] == "openai/qwen2.5:3b"


def test_anthropic_model_no_heuristic(tmp_path):
    profile = PipelineProfile()
    profile.eval_models = [
        {"provider": "anthropic", "model": "claude-opus-4-6"},
    ]
    profile.judge_provider = "anthropic"
    profile.judge_model = "claude-opus-4-6"
    output = tmp_path / "models.yaml"
    generate_models_yaml(profile, output)
    with open(output) as f:
        data = yaml.safe_load(f)
    assert data["models"]["claude-opus-4-6"]["litellm_model"] == "claude-opus-4-6"
