"""
run_skillsbench.py

CLI entry point for SkillsBench corpus evaluation.
Extracted from c3_skillsbench.corpus_harness to keep c3 as a pure functional API.

Usage:
    python -m c4_cli.run_skillsbench \
      --tasks tasks.json --skills skills.json \
      --models qwen3:0.6b,qwen2.5:1.5b,llama3.2:1b \
      --base-url http://host.docker.internal:11434/v1 \
      --mode guided
"""

from __future__ import annotations

import argparse
import os
import yaml
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

from openai import OpenAI

from c2_evaluation.llm_judge import LLMJudgeEvaluator
from c1_types.extracted_task import load_extracted_tasks
from c1_types.extracted_skill import load_extracted_skills
from c3_skillsbench.corpus_harness import (
    run_corpus_evaluation,
    run_multi_model_evaluation,
    _save_episodes,
)


# ---------------------------------------------------------------------------
# Inline model config (replaces c1_providers.model_config)
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    litellm_model: str
    provider: str = "lmproxy"
    api_base: str = ""
    api_key: str = ""
    api_key_env: str = ""


@dataclass
class ModelConfig:
    models: Dict[str, ModelEntry]
    judge_model_name: str

    @property
    def model_names(self) -> List[str]:
        return list(self.models.keys())

    def get_judge_entry(self) -> ModelEntry:
        return self.models[self.judge_model_name]


def load_model_config(path: str) -> ModelConfig:
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
            api_base=entry_data.get("api_base", ""),
            api_key=api_key,
            api_key_env=api_key_env,
        )
    judge_section = raw.get("judge", {})
    judge_model_name = judge_section.get("model", "")
    if judge_model_name and judge_model_name not in models:
        raise ValueError(f"Judge model '{judge_model_name}' not found in models config")
    return ModelConfig(models=models, judge_model_name=judge_model_name)


# ---------------------------------------------------------------------------
# OpenAI-SDK-based provider (replaces c1_providers.providers.create_provider)
# ---------------------------------------------------------------------------

_LMPROXY_BASE_URL = os.environ.get("LMPROXY_BASE_URL", "http://localhost:8080")


def _create_openai_provider(model: str, base_url: str = "", api_key: str = "lmproxy"):
    """Return a thin wrapper around the OpenAI SDK pointed at lmproxy."""
    effective_base = base_url or _LMPROXY_BASE_URL
    client = OpenAI(base_url=effective_base, api_key=api_key)

    class _Provider:
        def __init__(self):
            self.model_name = model
            self._client = client

        def chat(self, messages, tools=None):
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            }

            class _R:
                pass
            r = _R()
            r.message = {"role": "assistant", "content": content}
            r.usage = usage
            return r

    return _Provider()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SkillsBench corpus evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  singlecall   One LLM call per episode (default)
  stepwise     Multi-turn: model reasons one action primitive per step
  guided       Multi-turn: each turn follows one step of the skill's procedure

Examples:
  # Singlecall, multiple models
  python -m c4_cli.run_skillsbench \\
    --tasks ../llm-skills.shared-data/tasks.json --skills ../llm-skills.shared-data/verified_skills.json \\
    --models qwen3:0.6b,qwen2.5:1.5b,llama3.2:1b,qwen2.5:3b \\
    --base-url http://host.docker.internal:11434/v1

  # Stepwise mode
  python -m c4_cli.run_skillsbench \\
    --tasks ../llm-skills.shared-data/tasks.json --skills ../llm-skills.shared-data/verified_skills.json \\
    --models qwen3:0.6b,llama3.2:1b \\
    --base-url http://host.docker.internal:11434/v1 \\
    --mode stepwise --max-steps 6

  # Guided mode (procedure-driven multi-turn)
  python -m c4_cli.run_skillsbench \\
    --tasks ../llm-skills.shared-data/tasks.json --skills ../llm-skills.shared-data/verified_skills.json \\
    --models qwen3:0.6b,llama3.2:1b \\
    --base-url http://host.docker.internal:11434/v1 \\
    --mode guided
""",
    )
    parser.add_argument("--tasks", type=Path, required=True, help="Extracted tasks JSON")
    parser.add_argument("--skills", type=Path, required=True, help="Extracted skills JSON")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated model names/aliases. Required for legacy mode; "
                             "optional with --config (filters to subset of config models)")
    parser.add_argument("--config", type=str, default="",
                        help="Path to YAML model config for config-driven routing via LiteLLM")
    parser.add_argument("--mode", choices=["singlecall", "stepwise", "guided"],
                        default="singlecall", help="Scaffolding mode (default: singlecall)")
    parser.add_argument("--max-steps", type=int, default=8,
                        help="Max steps per episode for stepwise (default: 8)")
    parser.add_argument("--provider", type=str, default="openai", help="Model provider (default: openai)")
    parser.add_argument("--base-url", type=str, default="", help="Base URL for provider")
    parser.add_argument("--judge-provider", type=str, default="anthropic")
    parser.add_argument("--judge-model", type=str, default="claude-opus-4-6")
    parser.add_argument("--output", "-o", type=Path, default=Path("stage5-corpus-evaluation/results.json"))
    parser.add_argument("--heatmaps", type=Path, default=None,
                        help="Generate heatmaps to this directory after evaluation")
    parser.add_argument("--cross-task", action="store_true",
                        help="Cross-task mode: inject every skill into every task. "
                             "Default is one-to-one (only the matching skill per task "
                             "via source_task_uids; unmatched tasks run baseline only).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config-driven path (--config): LiteLLM routing from YAML config
    # ------------------------------------------------------------------
    if args.config:
        config = load_model_config(args.config)

        # Determine which model aliases to evaluate
        if args.models:
            requested = [m.strip() for m in args.models.split(",") if m.strip()]
            missing = [m for m in requested if m not in config.models]
            if missing:
                parser.error(f"Model aliases not found in config: {', '.join(missing)}")
            model_aliases = requested
        else:
            model_aliases = config.model_names

        # Create judge from config
        judge_entry = config.get_judge_entry()
        judge_provider = _create_openai_provider(
            judge_entry.litellm_model,
            base_url=judge_entry.api_base,
            api_key=judge_entry.api_key or "lmproxy",
        )
        judge = LLMJudgeEvaluator(judge_provider)

        tasks = load_extracted_tasks(args.tasks)
        skills = load_extracted_skills(args.skills)

        if args.verbose:
            print(f"Config: {args.config}")
            print(f"Tasks: {len(tasks)}, Skills: {len(skills)}, "
                  f"Models: {model_aliases}, Mode: {args.mode}")

        all_episodes = []
        for model_index, model_alias in enumerate(model_aliases):
            entry = config.models[model_alias]
            model_provider = _create_openai_provider(
                entry.litellm_model,
                base_url=entry.api_base,
                api_key=entry.api_key or "lmproxy",
            )

            if args.verbose:
                print(f"\n{'='*60}")
                print(f"Model {model_index + 1}/{len(model_aliases)}: "
                      f"{model_alias} (mode={args.mode})")
                print(f"{'='*60}")

            episodes = run_corpus_evaluation(
                tasks, skills, model_provider, judge,
                model_name=model_alias, mode=args.mode,
                max_steps=args.max_steps, cross_task=args.cross_task,
                verbose=args.verbose,
            )
            all_episodes.extend(episodes)

            baseline = [e for e in episodes if e.condition == "baseline"]
            skilled = [e for e in episodes if e.condition == "curated"]
            bl_rate = (sum(1 for e in baseline if e.passed) / len(baseline)
                       if baseline else 0)
            sk_rate = (sum(1 for e in skilled if e.passed) / len(skilled)
                       if skilled else 0)
            print(f"  {model_alias}: baseline={bl_rate:.1%} "
                  f"skill={sk_rate:.1%} delta={sk_rate - bl_rate:+.1%}")

        if args.output:
            _save_episodes(all_episodes, args.output)

        print(f"\n{'='*60}")
        print(f"OVERALL: {len(all_episodes)} episodes, "
              f"{len(model_aliases)} models, mode={args.mode}")
        baseline = [e for e in all_episodes if e.condition == "baseline"]
        skilled = [e for e in all_episodes if e.condition == "curated"]
        bl_rate = (sum(1 for e in baseline if e.passed) / len(baseline)
                   if baseline else 0)
        sk_rate = (sum(1 for e in skilled if e.passed) / len(skilled)
                   if skilled else 0)
        print(f"  Baseline: {bl_rate:.1%}  Skill: {sk_rate:.1%}  "
              f"Delta: {sk_rate - bl_rate:+.1%}")
        print(f"  Saved to {args.output}")

        if args.heatmaps:
            from c3_skillsbench.visualization import (
                generate_uplift_heatmap,
                generate_pass_rate_heatmap,
                generate_combined_heatmap,
                load_episodes,
            )
            heatmap_episodes = load_episodes(results_file=args.output)
            args.heatmaps.mkdir(parents=True, exist_ok=True)
            generate_uplift_heatmap(
                heatmap_episodes, args.heatmaps / "uplift_heatmap.png")
            generate_pass_rate_heatmap(
                heatmap_episodes, args.heatmaps / "baseline_pass_rate.png")
            generate_combined_heatmap(
                heatmap_episodes, args.heatmaps / "combined_heatmap.png")

        return

    # ------------------------------------------------------------------
    # Legacy path (--provider / --base-url / --models)
    # ------------------------------------------------------------------
    if not args.models:
        parser.error("--models is required when --config is not provided")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    judge_provider = _create_openai_provider(args.judge_model, base_url=args.base_url)
    judge = LLMJudgeEvaluator(judge_provider)

    tasks = load_extracted_tasks(args.tasks)
    skills = load_extracted_skills(args.skills)

    if args.verbose:
        print(f"Tasks: {len(tasks)}, Skills: {len(skills)}, Models: {model_names}, Mode: {args.mode}")

    episodes = run_multi_model_evaluation(
        tasks, skills, model_names,
        provider_name=args.provider, base_url=args.base_url,
        judge=judge, mode=args.mode, max_steps=args.max_steps,
        cross_task=args.cross_task,
        output_path=args.output, verbose=args.verbose,
    )

    print(f"\n{'='*60}")
    print(f"OVERALL: {len(episodes)} episodes, {len(model_names)} models, mode={args.mode}")
    baseline = [e for e in episodes if e.condition == "baseline"]
    skilled = [e for e in episodes if e.condition == "curated"]
    bl_rate = sum(1 for e in baseline if e.passed) / len(baseline) if baseline else 0
    sk_rate = sum(1 for e in skilled if e.passed) / len(skilled) if skilled else 0
    print(f"  Baseline: {bl_rate:.1%}  Skill: {sk_rate:.1%}  Delta: {sk_rate - bl_rate:+.1%}")
    print(f"  Saved to {args.output}")

    if args.heatmaps:
        from c3_skillsbench.visualization import (
            generate_uplift_heatmap,
            generate_pass_rate_heatmap,
            generate_combined_heatmap,
            load_episodes,
        )
        heatmap_episodes = load_episodes(results_file=args.output)
        args.heatmaps.mkdir(parents=True, exist_ok=True)
        generate_uplift_heatmap(heatmap_episodes, args.heatmaps / "uplift_heatmap.png")
        generate_pass_rate_heatmap(heatmap_episodes, args.heatmaps / "baseline_pass_rate.png")
        generate_combined_heatmap(heatmap_episodes, args.heatmaps / "combined_heatmap.png")


if __name__ == "__main__":
    main()
