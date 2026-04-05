import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.pipeline_profile import PipelineProfile
from tools.profile_loader import load_profile, save_profile, PROFILES_DIR


def test_load_old_string_eval_models(tmp_path):
    """Old profiles with eval_models as plain strings normalize to dicts."""
    yaml_content = """
profile_name: test-old
eval_models:
  - qwen2.5-3b
  - qwen2.5-7b
"""
    profile_file = tmp_path / "test-old.yaml"
    profile_file.write_text(yaml_content)

    import tools.profile_loader as loader
    original_dir = loader.PROFILES_DIR
    loader.PROFILES_DIR = tmp_path
    try:
        profile = load_profile("test-old")
        assert len(profile.eval_models) == 2
        assert profile.eval_models[0] == {"provider": "ollama", "model": "qwen2.5-3b"}
        assert profile.eval_models[1] == {"provider": "ollama", "model": "qwen2.5-7b"}
    finally:
        loader.PROFILES_DIR = original_dir


def test_load_new_dict_eval_models(tmp_path):
    """New profiles with dict eval_models pass through unchanged."""
    yaml_content = """
profile_name: test-new
eval_models:
  - provider: lmproxy
    model: qwen2.5-3b
"""
    profile_file = tmp_path / "test-new.yaml"
    profile_file.write_text(yaml_content)

    import tools.profile_loader as loader
    original_dir = loader.PROFILES_DIR
    loader.PROFILES_DIR = tmp_path
    try:
        profile = load_profile("test-new")
        assert profile.eval_models[0]["provider"] == "lmproxy"
    finally:
        loader.PROFILES_DIR = original_dir


def test_save_and_load_roundtrip(tmp_path):
    """Profile with dict eval_models survives save/load."""
    import tools.profile_loader as loader
    original_dir = loader.PROFILES_DIR
    loader.PROFILES_DIR = tmp_path
    try:
        profile = PipelineProfile()
        profile.profile_name = "roundtrip"
        save_profile(profile)
        loaded = load_profile("roundtrip")
        assert loaded.eval_models == profile.eval_models
        assert loaded.lmproxy_base_url == profile.lmproxy_base_url
    finally:
        loader.PROFILES_DIR = original_dir
