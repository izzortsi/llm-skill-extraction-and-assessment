"""
experiment_config.py

Experiment configuration, trial specifications, and condition types.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ConditionType(Enum):
    """Experimental conditions for skill benchmarking."""

    BASELINE = "baseline"             # No skill injected
    CURATED_SKILL = "curated"         # Skill from extraction suite injected as system context
    SELF_GENERATED = "self_generated" # Agent generates its own skill before solving


@dataclass
class ModelSpec:
    """A model to benchmark."""

    provider: str   # Provider name (anthropic-oauth, openai, openrouter, zai)
    model: str      # Model identifier (claude-sonnet-4-6, gpt-4o, etc.)
    base_url: str = ""  # Optional base URL for OpenAI-compatible endpoints (Ollama, vLLM)

    def key(self) -> str:
        """Unique key for this model spec."""
        if self.base_url:
            return f"{self.provider}/{self.model}@{self.base_url}"
        return f"{self.provider}/{self.model}"

    def display_name(self) -> str:
        """Short display name for heatmap labels."""
        return self.model


@dataclass
class SkillSourceConfig:
    """Configuration for skill sources."""

    atomic_dir: str = ""
    composed_dir: str = ""
    k_values: List[int] = field(default_factory=list)
    composition_types: List[str] = field(default_factory=list)


@dataclass
class TaskSourceConfig:
    """Configuration for task sources."""

    verification_tasks: str = ""
    problem_dirs: List[str] = field(default_factory=list)


@dataclass
class SelfGeneratedConfig:
    """Configuration for self-generated skill condition."""

    enabled: bool = False
    generator_provider: str = ""
    generator_model: str = ""


@dataclass
class JudgeConfig:
    """Configuration for LLM-as-judge evaluator."""

    provider: str = ""      # Provider for judge model (e.g., "anthropic")
    model: str = ""         # Judge model identifier (e.g., "claude-opus-4-6")
    base_url: str = ""      # Optional base URL for judge provider

    @property
    def enabled(self) -> bool:
        return bool(self.provider and self.model)

    def to_model_spec(self) -> "ModelSpec":
        return ModelSpec(provider=self.provider, model=self.model, base_url=self.base_url)


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    experiment_id: str
    skill_sources: SkillSourceConfig
    task_sources: TaskSourceConfig
    self_generated: SelfGeneratedConfig
    models: List[ModelSpec]
    conditions: List[ConditionType]
    repetitions: int = 5
    max_steps: int = 15
    seed: int = 42
    output_dir: str = "../shared-data/skilleval-results"
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    @classmethod
    def from_json(cls, filepath: Path) -> "ExperimentConfig":
        """Load experiment config from JSON file.

        Args:
            filepath: Path to experiment config JSON

        Returns:
            ExperimentConfig object
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        skill_src = data.get("skill_sources", {})
        task_src = data.get("task_sources", {})
        self_gen = data.get("self_generated", {})
        judge_data = data.get("judge", {})

        # Support flat judge fields for backward compatibility
        if not judge_data:
            judge_data = {
                "provider": data.get("judge_provider", ""),
                "model": data.get("judge_model", ""),
                "base_url": data.get("judge_base_url", ""),
            }

        return cls(
            experiment_id=data["experiment_id"],
            skill_sources=SkillSourceConfig(
                atomic_dir=skill_src.get("atomic_dir", ""),
                composed_dir=skill_src.get("composed_dir", ""),
                k_values=skill_src.get("k_values", []),
                composition_types=skill_src.get("composition_types", []),
            ),
            task_sources=TaskSourceConfig(
                verification_tasks=task_src.get("verification_tasks", ""),
                problem_dirs=task_src.get("problem_dirs", []),
            ),
            self_generated=SelfGeneratedConfig(
                enabled=self_gen.get("enabled", False),
                generator_provider=self_gen.get("generator_provider", ""),
                generator_model=self_gen.get("generator_model", ""),
            ),
            models=[ModelSpec(**m) for m in data.get("models", [])],
            conditions=[ConditionType(c) for c in data.get("conditions", [])],
            repetitions=data.get("repetitions", 5),
            max_steps=data.get("max_steps", 15),
            seed=data.get("seed", 42),
            output_dir=data.get("output_dir", "../shared-data/skilleval-results"),
            judge=JudgeConfig(
                provider=judge_data.get("provider", ""),
                model=judge_data.get("model", ""),
                base_url=judge_data.get("base_url", ""),
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict."""
        return {
            "experiment_id": self.experiment_id,
            "skill_sources": {
                "atomic_dir": self.skill_sources.atomic_dir,
                "composed_dir": self.skill_sources.composed_dir,
                "k_values": self.skill_sources.k_values,
                "composition_types": self.skill_sources.composition_types,
            },
            "task_sources": {
                "verification_tasks": self.task_sources.verification_tasks,
                "problem_dirs": self.task_sources.problem_dirs,
            },
            "self_generated": {
                "enabled": self.self_generated.enabled,
                "generator_provider": self.self_generated.generator_provider,
                "generator_model": self.self_generated.generator_model,
            },
            "models": [
                {k: v for k, v in [("provider", m.provider), ("model", m.model), ("base_url", m.base_url)] if v}
                for m in self.models
            ],
            "judge": {
                "provider": self.judge.provider,
                "model": self.judge.model,
                "base_url": self.judge.base_url,
            },
            "conditions": [c.value for c in self.conditions],
            "repetitions": self.repetitions,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "output_dir": self.output_dir,
        }


@dataclass
class TrialSpec:
    """Specification for a single trial (one run of task x skill x model x condition)."""

    task_uid: str
    domain: str
    model_spec: ModelSpec
    condition: ConditionType
    skill_name: str = ""            # Empty for baseline
    skill_content: str = ""         # Full skill text for injection
    k_value: int = 0                # Composition depth (0 for atomic/baseline)
    composition_type: str = ""      # seq/par/cond/sem or empty
    repetition_index: int = 0
    max_steps: int = 15
    seed: int = 42

    def trial_key(self) -> str:
        """Unique key for this trial spec (excluding repetition)."""
        return f"{self.task_uid}|{self.model_spec.key()}|{self.condition.value}|{self.skill_name}"
