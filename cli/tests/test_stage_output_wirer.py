import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c0_config.pipeline_profile import PipelineProfile
from c2_orchestration.stage_output_wirer import build_stage_args


def test_stage5_eval_models_join(tmp_path):
    """Stage 5 args contain comma-joined model names from dict eval_models."""
    profile = PipelineProfile()
    profile.eval_models = [
        {"provider": "lmproxy", "model": "qwen2.5-3b"},
        {"provider": "ollama", "model": "llama3.2:1b"},
    ]
    # Stage 5 checks config_file existence — create a dummy
    config_dir = tmp_path / "llm-skills.llm-providers" / "configs"
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
