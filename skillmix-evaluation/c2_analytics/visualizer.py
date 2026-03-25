"""
visualizer.py

Generate charts from SkillMix experiment results. Reads episodes.json
and summary.json produced by run-skillmix and generates PNG visualizations.

Charts:
    baseline_vs_skill.png   grouped bar: baseline vs skill-injected per model
    delta_by_model.png      bar chart: uplift (delta) per model
    skill_heatmap.png       heatmap: mean score per skill x model
    win_loss.png            stacked bar: win/tie/loss per model
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = ("#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
           "#64B5CD", "#DD8452", "#A1C9F4")


def _load_data(results_dir: Path) -> tuple:
    """Load episodes.json and summary.json from a results directory."""
    with open(results_dir / "episodes.json", "r", encoding="utf-8") as f:
        episodes = json.load(f)
    with open(results_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    return episodes, summary


def generate_baseline_vs_skill_bar(
    summary: Dict[str, Any],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Grouped bar chart: baseline vs skill-injected mean score per model."""
    models = sorted(summary.keys())
    baselines = [summary[m]["baseline_mean_score"] for m in models]
    skills = [summary[m]["skill_mean_score"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2), 5))
    bars_b = ax.bar(x - width / 2, baselines, width, label="Baseline", color=PALETTE[0])
    bars_s = ax.bar(x + width / 2, skills, width, label="Skill-Injected", color=PALETTE[1])

    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title("Baseline vs Skill-Injected Score by Model", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    # value labels above bars
    for bar in list(bars_b) + list(bars_s):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_delta_by_model_bar(
    summary: Dict[str, Any],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Bar chart: delta (uplift) per model."""
    models = sorted(summary.keys(), key=lambda m: summary[m]["delta"], reverse=True)
    deltas = [summary[m]["delta"] for m in models]

    colors = [PALETTE[1] if d >= 0 else PALETTE[2] for d in deltas]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    bars = ax.bar(range(len(models)), deltas, color=colors)

    ax.set_ylabel("Score Delta (Skill - Baseline)", fontsize=11)
    ax.set_title("Skill Injection Uplift by Model", fontsize=13, pad=12)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    for bar, d in zip(bars, deltas):
        ypos = bar.get_height()
        offset = 3 if d >= 0 else -10
        ax.annotate(f"{d:+.3f}", xy=(bar.get_x() + bar.get_width() / 2, ypos),
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_skill_heatmap(
    episodes: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Heatmap: mean score per skill x model (skill-injected episodes only)."""
    # group scores by (skill, model)
    scores = defaultdict(lambda: defaultdict(list))
    models_set = set()
    for ep in episodes:
        if ep.get("condition") != "skill_injected":
            continue
        sn = ep.get("skill_name", "")
        model = ep.get("model", "")
        score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
        scores[sn][model].append(score)
        models_set.add(model)

    if not scores:
        return

    skills = sorted(scores.keys())
    models = sorted(models_set)

    data = np.zeros((len(skills), len(models)))
    for i, sk in enumerate(skills):
        for j, m in enumerate(models):
            vals = scores[sk][m]
            data[i, j] = sum(vals) / len(vals) if vals else 0.0

    figsize = (max(6, len(models) * 1.5 + 3), max(4, len(skills) * 0.5 + 2))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(skills)))
    ax.set_yticklabels(skills, fontsize=8)
    ax.set_title("Skill-Injected Score by Skill x Model", fontsize=13, pad=12)

    # text overlay
    for i in range(len(skills)):
        for j in range(len(models)):
            val = data[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Score", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_win_loss_bar(
    episodes: List[Dict],
    summary: Dict[str, Any],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Stacked bar: wins/ties/losses per model.

    For each task, a win occurs when skill_injected score > baseline score,
    a loss when skill_injected < baseline, and a tie when equal.
    """
    # group by (model, task) -> {baseline: [scores], skill: [scores]}
    by_model_task = defaultdict(lambda: defaultdict(lambda: {"baseline": [], "skill_injected": []}))
    for ep in episodes:
        model = ep.get("model", "")
        task = ep.get("task_uid", ep.get("task_id", ""))
        cond = ep.get("condition", "")
        score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
        by_model_task[model][task][cond].append(score)

    models = sorted(by_model_task.keys())
    wins = []
    ties = []
    losses = []

    for m in models:
        w, t, l = 0, 0, 0
        for task, conds in by_model_task[m].items():
            b_scores = conds.get("baseline", [])
            s_scores = conds.get("skill_injected", [])
            if not b_scores or not s_scores:
                continue
            b_mean = sum(b_scores) / len(b_scores)
            s_mean = sum(s_scores) / len(s_scores)
            if s_mean > b_mean + 0.001:
                w += 1
            elif s_mean < b_mean - 0.001:
                l += 1
            else:
                t += 1
        wins.append(w)
        ties.append(t)
        losses.append(l)

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))

    ax.bar(x, wins, label="Win", color=PALETTE[1])
    ax.bar(x, ties, bottom=wins, label="Tie", color="#999999")
    bottoms = [w + t for w, t in zip(wins, ties)]
    ax.bar(x, losses, bottom=bottoms, label="Loss", color=PALETTE[2])

    ax.set_ylabel("Number of Tasks", fontsize=11)
    ax.set_title("Win / Tie / Loss per Model (Skill vs Baseline)", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_all(
    results_dir: Path,
    output_dir: Path,
    dpi: int = 150,
) -> List[str]:
    """Generate all SkillMix visualization charts.

    Args:
        results_dir: directory containing episodes.json and summary.json
        output_dir: directory to write PNG files into
        dpi: output image resolution

    Returns:
        list of generated file paths
    """
    episodes, summary = _load_data(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    generate_baseline_vs_skill_bar(summary, output_dir / "baseline_vs_skill.png", dpi)
    generated.append(str(output_dir / "baseline_vs_skill.png"))

    generate_delta_by_model_bar(summary, output_dir / "delta_by_model.png", dpi)
    generated.append(str(output_dir / "delta_by_model.png"))

    generate_skill_heatmap(episodes, output_dir / "skill_heatmap.png", dpi)
    generated.append(str(output_dir / "skill_heatmap.png"))

    generate_win_loss_bar(episodes, summary, output_dir / "win_loss.png", dpi)
    generated.append(str(output_dir / "win_loss.png"))

    return generated


def main() -> None:
    """CLI entry point for visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate SkillMix charts")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory with episodes.json and summary.json")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="Output directory for PNG files")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150)")
    args = parser.parse_args()

    generated = generate_all(args.results_dir, args.output_dir, args.dpi)
    for path in generated:
        print(f"  generated: {path}")


if __name__ == "__main__":
    main()
