"""
run_skillmix.py

CLI command: run SkillMix benchmark experiment.

Parses arguments and delegates to c3_skillmix.runner functional APIs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import openai

from c2_evaluation.llm_judge import LLMJudgeEvaluator
from c3_skillmix.runner import run_skillmix_experiment


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the run-skillmix subcommand."""
    parser = subparsers.add_parser(
        "run-skillmix",
        help="Run SkillMix benchmark (multi-model, LLM-as-judge)",
    )
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
    model_names = [m.strip() for m in args.models.split(",")]
    model_configs = [
        {"provider": args.provider, "model": m, "base_url": args.base_url}
        for m in model_names
    ]

    # Create judge (uses same lmproxy session as the runner)
    from c3_skillmix.runner import _OpenAIProvider, _ensure_lmproxy_session, _create_lmproxy_client
    worker_id = _ensure_lmproxy_session("skillmix-judge")
    judge_client = _create_lmproxy_client(worker_id)
    judge_provider = _OpenAIProvider(judge_client, args.judge_model)
    judge = LLMJudgeEvaluator(judge_provider)

    # Run
    result = run_skillmix_experiment(
        tasks, skills, model_configs, judge,
        output_dir=args.output_dir, verbose=args.verbose,
    )

    print(f"\nExperiment complete: {len(result.episodes)} episodes")
    for model, stats in result.summary.items():
        print(f"  {model}: baseline={stats['baseline_mean_score']:.3f} skill={stats['skill_mean_score']:.3f} delta={stats['delta']:+.3f}")
