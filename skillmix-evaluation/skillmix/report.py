"""
report.py

SkillMix experiment reporting and visualization.

Usage:
    python -m skillmix.report --results-dir skillmix-results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def generate_report(results_dir: Path) -> str:
    """Generate text report from SkillMix results."""
    episodes_path = results_dir / "episodes.json"
    summary_path = results_dir / "summary.json"

    with open(episodes_path) as f:
        episodes = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    lines = []
    lines.append("=" * 60)
    lines.append("SKILLMIX EXPERIMENT REPORT")
    lines.append("=" * 60)
    lines.append(f"\nTotal episodes: {len(episodes)}")

    # Per-model summary
    lines.append("\nPer-Model Results:")
    lines.append("-" * 60)
    lines.append(f"{'Model':<30} {'Baseline':>10} {'Skill':>10} {'Delta':>10}")
    lines.append("-" * 60)

    for model, stats in sorted(summary.items()):
        baseline = stats.get("baseline_mean_score", 0)
        skill = stats.get("skill_mean_score", 0)
        delta = stats.get("delta", 0)
        lines.append(f"{model:<30} {baseline:>10.3f} {skill:>10.3f} {delta:>+10.3f}")

    # Per-skill breakdown
    skill_scores = {}
    for ep in episodes:
        if ep.get("condition") == "skill_injected" and "score" in ep:
            sn = ep.get("skill_name", "")
            if sn not in skill_scores:
                skill_scores[sn] = []
            skill_scores[sn].append(ep["score"])

    if skill_scores:
        lines.append(f"\nPer-Skill Mean Scores (across all models):")
        lines.append("-" * 60)
        for sn in sorted(skill_scores.keys()):
            scores = skill_scores[sn]
            mean = sum(scores) / len(scores)
            lines.append(f"  {sn}: {mean:.3f} (n={len(scores)})")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for backwards compatibility.

    Delegates to cli.report for argument parsing and execution.
    Prefer using ``python -m cli.main report`` instead.
    """
    from cli.report import run

    parser = argparse.ArgumentParser(description="Generate SkillMix report")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory with experiment results")
    parser.add_argument("--output", "-o", type=Path, help="Output report file (default: stdout)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
