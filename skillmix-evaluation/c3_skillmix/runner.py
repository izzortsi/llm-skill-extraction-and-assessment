"""
runner.py

SkillMix experiment runner: evaluate multiple models on composed skills.

Usage:
    python -m c3_skillmix.runner --config experiment.json
    python -m c3_skillmix.runner --tasks tasks.json --skills-dir skills/ --models qwen3:0.6b,llama3.2:1b
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import requests
import openai

from c2_analytics.summary import compute_summary
from c2_evaluation.llm_judge import LLMJudgeEvaluator
from c3_skillmix.harness import run_skillmix_episode, SkillMixEpisode


def _lmproxy_base_url() -> str:
    return os.environ.get("LMPROXY_BASE_URL", "http://localhost:8080/v1")


def _ensure_lmproxy_session(worker_id: str = "skillmix-pipeline") -> str:
    """Register worker and start session with lmproxy. Returns worker_id."""
    base = _lmproxy_base_url()
    try:
        requests.post(f"{base}/register", json={"identifier": worker_id}, timeout=5)
        requests.post(f"{base}/session/start", json={"worker_id": f"worker-{worker_id}", "service": "openai"}, timeout=5)
    except Exception:
        pass  # proxy may not require sessions (direct mode)
    return f"worker-{worker_id}"


def _create_lmproxy_client(worker_id: str) -> openai.OpenAI:
    """Create an OpenAI client pointed at lmproxy with worker ID header."""
    return openai.OpenAI(
        base_url=_lmproxy_base_url(),
        api_key="lmproxy",
        default_headers={"X-Worker-ID": worker_id},
    )


class _OpenAIProvider:
    """Thin adapter that wraps an openai.OpenAI client to match the provider
    interface expected by c3_skillmix.harness (`.chat(messages)` returning an
    object with `.message` dict and `.usage` dict, plus `.model_name`)."""

    def __init__(self, client: openai.OpenAI, model: str):
        self._client = client
        self.model_name = model

    def chat(self, messages):
        response = self._client.chat.completions.create(
            model=self.model_name, messages=messages
        )
        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        } if response.usage else {}

        class _Result:
            pass

        result = _Result()
        result.message = {"content": choice.message.content}
        result.usage = usage
        return result



@dataclass
class SkillMixConfig:
    """Configuration for SkillMix experiment."""

    models: List[Dict[str, str]]       # [{"provider": "openai", "model": "...", "base_url": "..."}]
    judge_provider: str = "anthropic"
    judge_model: str = "claude-opus-4-6"
    tasks_path: str = ""
    atomic_dir: str = ""
    composed_dir: str = ""
    k_values: List[int] = None
    composition_types: List[str] = None
    output_dir: str = "skillmix-results"


@dataclass
class SkillMixResult:
    """Aggregated result for a SkillMix experiment."""
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def run_skillmix_experiment(
    tasks: List[Dict],
    skills: Dict[str, str],
    model_configs: List[Dict[str, str]],
    judge: Optional[LLMJudgeEvaluator] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> SkillMixResult:
    """Run full SkillMix experiment: models x skills x tasks.

    Args:
        tasks: List of task dicts with passage, challenge, acceptance_criteria
        skills: Dict of skill_name -> skill_content (markdown)
        model_configs: List of model config dicts
        judge: LLM judge evaluator
        output_dir: Directory for output files
        verbose: Print progress

    Returns:
        SkillMixResult with episodes and summary
    """
    result = SkillMixResult()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Register with lmproxy (used as fallback when model config has no base_url)
    worker_id = _ensure_lmproxy_session()
    lmproxy_client = _create_lmproxy_client(worker_id)

    total_episodes = len(model_configs) * len(tasks) * (1 + len(skills))
    episode_count = 0

    for model_cfg in model_configs:
        cfg_provider = model_cfg.get("provider", "")
        model_base_url = model_cfg.get("base_url", "")

        if cfg_provider == "claude-code":
            # Use ClaudeCodeProvider directly via create_provider
            from c1_providers.providers import create_provider
            provider = create_provider("claude-code", model_cfg.get("model", ""))
        elif model_base_url:
            # config-driven: create client pointing at the model's own endpoint
            client = openai.OpenAI(
                base_url=model_base_url,
                api_key=model_cfg.get("api_key", "na"),
            )
            provider = _OpenAIProvider(client, model_cfg.get("model", ""))
        else:
            # fallback: use lmproxy
            client = lmproxy_client
            provider = _OpenAIProvider(client, model_cfg.get("model", ""))
        model_name = model_cfg.get("alias", model_cfg.get("model", "unknown"))

        if verbose:
            print(f"\nModel: {model_name}")

        for task in tasks:
            # Baseline (no skill)
            episode_count += 1
            if verbose:
                print(f"  [{episode_count}/{total_episodes}] {task.get('task_id', '')} baseline")

            ep = run_skillmix_episode(task, None, "", provider, judge, verbose=verbose)
            result.episodes.append(_episode_to_dict(ep))

            # With each skill
            for skill_name, skill_content in skills.items():
                episode_count += 1
                if verbose:
                    print(f"  [{episode_count}/{total_episodes}] {task.get('task_id', '')} + {skill_name}")

                ep = run_skillmix_episode(task, skill_content, skill_name, provider, judge, verbose=verbose)
                result.episodes.append(_episode_to_dict(ep))

    # Compute summary
    result.summary = _compute_summary(result.episodes)

    # Save
    if output_dir:
        with open(output_dir / "episodes.json", "w") as f:
            json.dump(result.episodes, f, indent=2)
        with open(output_dir / "summary.json", "w") as f:
            json.dump(result.summary, f, indent=2)

    return result


def _episode_to_dict(ep: SkillMixEpisode) -> Dict[str, Any]:
    d = {
        "task_uid": ep.task_uid,
        "skill_name": ep.skill_name,
        "model": ep.model,
        "condition": ep.condition,
        "tokens": ep.tokens,
        "elapsed_s": ep.elapsed_s,
    }
    if ep.judge_result:
        d["score"] = ep.judge_result.score
        d["passed"] = ep.judge_result.passed
        d["rationale"] = ep.judge_result.rationale
    return d


def _compute_summary(episodes: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics from episodes.

    Delegates to c2_analytics.summary.compute_summary.
    Kept for backwards compatibility.
    """
    return compute_summary(episodes)


def main() -> None:
    """CLI entry point for backwards compatibility.

    Delegates to c4_cli.run_skillmix for argument parsing and execution.
    Prefer using ``python -m c4_cli.main run-skillmix`` instead.
    """
    from c4_cli.run_skillmix import run

    parser = argparse.ArgumentParser(description="Run SkillMix benchmark experiment")
    parser.add_argument("--tasks", type=Path, required=True, help="Path to tasks JSON")
    parser.add_argument("--skills-dir", type=Path, help="Directory with composed skill .md files")
    parser.add_argument("--atomic-dir", type=Path, help="Directory with atomic skill .md files")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names (for Ollama)")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1", help="Ollama base URL")
    parser.add_argument("--provider", type=str, default="openai", help="Model provider")
    parser.add_argument("--judge-provider", type=str, default="anthropic")
    parser.add_argument("--judge-model", type=str, default="claude-opus-4-6")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("stage5-corpus-evaluation/skillmix-results"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
