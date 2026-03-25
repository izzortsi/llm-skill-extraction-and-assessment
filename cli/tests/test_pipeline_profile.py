import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c0_config.pipeline_profile import PipelineProfile, MINIMAL_OVERRIDES, apply_minimal


def test_eval_models_default_is_list_of_dicts():
    profile = PipelineProfile()
    assert len(profile.eval_models) == 2
    assert profile.eval_models[0]["provider"] == "lmproxy"
    assert profile.eval_models[0]["model"] == "qwen2.5-3b"
    assert profile.eval_models[1]["model"] == "qwen2.5-7b"


def test_lmproxy_base_url_default():
    profile = PipelineProfile()
    assert profile.lmproxy_base_url == "http://localhost:8080/v1"


def test_minimal_overrides_eval_models_are_dicts():
    assert "eval_models" in MINIMAL_OVERRIDES
    for entry in MINIMAL_OVERRIDES["eval_models"]:
        assert "provider" in entry
        assert "model" in entry


def test_apply_minimal_sets_eval_models():
    profile = PipelineProfile()
    apply_minimal(profile)
    assert len(profile.eval_models) == 2
    assert profile.eval_models[0]["provider"] == "lmproxy"
