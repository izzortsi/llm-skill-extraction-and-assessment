import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.pipeline_profile import PipelineProfile
from orchestration.stage_output_wirer import build_stage_args


def test_stage5_eval_models_join(tmp_path):
    """Stage 5 args contain comma-joined model names from dict eval_models."""
    profile = PipelineProfile()
    profile.eval_models = [
        {"provider": "lmproxy", "model": "qwen2.5-3b"},
        {"provider": "ollama", "model": "llama3.2:1b"},
    ]
    # Stage 5 checks config_file existence — create a dummy
    config_dir = tmp_path / "llm-providers" / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "models.yaml").write_text("models: {}\njudge:\n  model: x\n")
    profile.config_file = str(config_dir / "models.yaml")

    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    repo_root = tmp_path
    stage_outputs = {
        "1b": {"tasks": "/tmp/tasks.json"},
        "4": {"skills": "/tmp/skills.json"},
    }
    args = build_stage_args("5", profile, run_dir, repo_root, stage_outputs, mode="singlecall")
    # Find --models arg
    models_idx = args.index("--models")
    models_value = args[models_idx + 1]
    assert "qwen2.5-3b" in models_value
    assert "llama3.2:1b" in models_value


def test_stage1b_lmproxy_provider_becomes_openai(tmp_path):
    """When extraction_provider is lmproxy, stage 1b gets --provider openai --base-url."""
    profile = PipelineProfile()
    profile.extraction_provider = "lmproxy"
    profile.extraction_model = "claude-opus-4-6"
    profile.lmproxy_base_url = "http://proxy:8080/v1"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    repo_root = tmp_path
    stage_outputs = {"1a": {"passages": "/tmp/passages.json"}}
    args = build_stage_args("1b", profile, run_dir, repo_root, stage_outputs)
    assert "--provider" in args
    provider_idx = args.index("--provider")
    assert args[provider_idx + 1] == "openai"
    # base_url is set via env var, not --base-url CLI arg
    assert "--base-url" not in args


def test_stage1b_anthropic_direct(tmp_path):
    """When extraction_provider is anthropic, stage 1b gets --provider anthropic, no --base-url."""
    profile = PipelineProfile()
    profile.extraction_provider = "anthropic"
    profile.extraction_model = "claude-opus-4-6"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    repo_root = tmp_path
    stage_outputs = {"1a": {"passages": "/tmp/passages.json"}}
    args = build_stage_args("1b", profile, run_dir, repo_root, stage_outputs)
    provider_idx = args.index("--provider")
    assert args[provider_idx + 1] == "anthropic"
    assert "--base-url" not in args


def test_stage1b_ollama_uses_openai_provider(tmp_path):
    """When extraction_provider is ollama, stage 1b gets --provider openai (base_url via env)."""
    profile = PipelineProfile()
    profile.extraction_provider = "ollama"
    profile.extraction_model = "qwen2.5:3b"
    profile.ollama_url = "http://ollama:11434/v1"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    repo_root = tmp_path
    stage_outputs = {"1a": {"passages": "/tmp/passages.json"}}
    args = build_stage_args("1b", profile, run_dir, repo_root, stage_outputs)
    provider_idx = args.index("--provider")
    assert args[provider_idx + 1] == "openai"
    assert "--base-url" not in args


def test_provider_env_iosys():
    """iosys provider_env returns OPENAI_BASE_URL pointing at iosys."""
    from orchestration.stage_output_wirer import provider_env
    profile = PipelineProfile()
    profile.iosys_base_url = "http://llm.iosys.net/v1"
    env = provider_env("iosys", profile)
    assert env["OPENAI_BASE_URL"] == "http://llm.iosys.net/v1"
