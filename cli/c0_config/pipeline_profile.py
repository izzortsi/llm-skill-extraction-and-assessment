"""
pipeline_profile.py

Experiment profile: all configuration needed to run a pipeline end-to-end.
Serializable to/from YAML. Each field corresponds to a CLI argument or
environment variable consumed by one or more pipeline stages.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineProfile:
    # identity
    profile_name: str = "default"
    run_dir: str = "llm-skills.shared-data/skillmix-pipeline-run"

    # source data (stage 1a)
    dataset: str = "wikimedia/wikipedia"
    subset: str = "20231101.en"
    domain: str = "language-skills"
    max_chunks: int = 5
    chunk_size: int = 4000
    tasks_per_chunk: int = 2

    # extraction model (stages 1b, 3)
    extraction_provider: str = "anthropic"
    extraction_model: str = "claude-opus-4-6"

    # trace capture (stage 2)
    trace_provider: str = "anthropic"
    trace_model: str = "claude-opus-4-6"

    # evaluation (stage 5)
    config_file: str = ""
    ollama_url: str = "http://localhost:11434/v1"
    eval_models: List[str] = field(default_factory=lambda: ["qwen2.5:3b", "qwen2.5:7b"])
    modes: List[str] = field(default_factory=lambda: ["singlecall", "stepwise", "guided"])

    # optional API models (stage 5 legacy path)
    zai_model: str = "glm-5-turbo"
    zai_base_url: str = "https://api.z.ai/api/coding/paas/v4"
    anthropic_eval_model: str = "claude-opus-4-6"

    # judge (stage 5)
    judge_provider: str = "anthropic"
    judge_model: str = "claude-opus-4-6"

    # skill extraction (stage 3)
    max_skills: int = 8


MINIMAL_OVERRIDES = {
    "max_chunks": 1,
    "tasks_per_chunk": 1,
    "max_skills": 3,
    "modes": ["singlecall"],
    "eval_models": ["qwen2.5:3b"],
    "zai_model": "",
    "anthropic_eval_model": "",
}


def apply_minimal(profile: PipelineProfile) -> PipelineProfile:
    """Apply minimal overrides to a profile for fewest API calls."""
    for key, value in MINIMAL_OVERRIDES.items():
        setattr(profile, key, value)
    return profile
