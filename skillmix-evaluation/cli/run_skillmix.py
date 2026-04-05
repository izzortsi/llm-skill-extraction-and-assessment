"""
run_skillmix.py

CLI command: run SkillMix benchmark experiment.

Parses arguments and delegates to skillmix.runner functional APIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from providers.providers import create_provider
from evaluation.llm_judge import LLMJudgeEvaluator
from skillmix.runner import run_skillmix_experiment


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the run-skillmix subcommand."""
    parser = subparsers.add_parser(
        "run-skillmix",
        help="Run SkillMix benchmark (multi-model, LLM-as-judge)",
    )
    parser.add_argument("--tasks", type=Path, required=True, help="Path to tasks JSON")
    parser.add_argument("--skills-dir", type=Path, help="Directory with skill .md files (recursive)")
    parser.add_argument("--atomic-dir", type=Path, help="Directory with atomic skill .md files")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model aliases")
    parser.add_argument("--config", type=Path, help="Path to models.yaml config (resolves aliases to providers)")
    parser.add_argument("--base-url", type=str, default="http://localhost:11434/v1", help="Ollama base URL (legacy, ignored when --config is set)")
    parser.add_argument("--provider", type=str, default="openai", help="Model provider (legacy, ignored when --config is set)")
    parser.add_argument("--judge-provider", type=str, default="anthropic")
    parser.add_argument("--judge-model", type=str, default="claude-opus-4-6")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("stage5-corpus-evaluation/skillmix-results"))
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the run-skillmix command."""
    # Load tasks
    with open(args.tasks) as f:
        tasks = json.load(f)

    # Load skills
    skills = {}
    if args.skills_dir and args.skills_dir.exists():
        for md_file in sorted(args.skills_dir.rglob("*.md")):
            skills[md_file.stem] = md_file.read_text(encoding="utf-8")
    elif args.atomic_dir and args.atomic_dir.exists():
        for md_file in sorted(args.atomic_dir.glob("*.md")):
            if md_file.name != "extracted-skills-inventory.txt":
                skills[md_file.stem] = md_file.read_text(encoding="utf-8")

    # Build model configs
    model_aliases = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.config:
        if not args.config.exists():
            print(f"ERROR: config file not found: {args.config}")
            return

        from providers.model_config import load_model_config
        config = load_model_config(str(args.config))

        missing = [m for m in model_aliases if m not in config.models]
        if missing:
            print(f"ERROR: model aliases not found in config: {', '.join(missing)}")
            print(f"Available aliases: {', '.join(config.model_names)}")
            return

        model_configs = []
        for alias in model_aliases:
            entry = config.models[alias]
            model_configs.append({
                "provider": entry.provider,
                "model": entry.litellm_model,
                "base_url": entry.api_base,
                "api_key": entry.api_key,
                "alias": alias,
            })

        # Create judge from config if judge section exists
        judge_entry = None
        if config.judge_model_name:
            judge_entry = config.get_judge_entry()
        if judge_entry:
            judge_provider = create_provider(
                judge_entry.provider, judge_entry.litellm_model,
                base_url=judge_entry.api_base, api_key=judge_entry.api_key,
            )
        else:
            judge_provider = create_provider(args.judge_provider, args.judge_model)
    else:
        model_configs = [
            {"provider": args.provider, "model": m, "base_url": args.base_url}
            for m in model_aliases
        ]
        judge_provider = create_provider(args.judge_provider, args.judge_model)

    judge = LLMJudgeEvaluator(judge_provider)

    # Run
    result = run_skillmix_experiment(
        tasks, skills, model_configs, judge,
        output_dir=args.output_dir, verbose=args.verbose,
    )

    print(f"\nExperiment complete: {len(result.episodes)} episodes")
    for model, stats in result.summary.items():
        print(f"  {model}: baseline={stats['baseline_mean_score']:.3f} skill={stats['skill_mean_score']:.3f} delta={stats['delta']:+.3f}")
