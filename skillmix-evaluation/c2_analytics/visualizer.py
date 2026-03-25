"""
visualizer.py

Generate charts from SkillMix experiment results. Reads episodes.json
and summary.json produced by run-skillmix and generates PNG visualizations.

SkillMix-specific dimensions parsed from skill_name:
    operator    seq / par / cond / atomic (no prefix = atomic)
    k           composition size (number of atomic skills combined)

Charts:
    score_by_k.png              line: mean score by k-value, one line per model
    operator_heatmap.png        heatmap: operator x model, mean score
    uplift_heatmap.png          heatmap: skill x model, delta from baseline (diverging)
    k_operator_heatmap.png      heatmap: (k, operator) x model, mean score
    baseline_vs_skill.png       grouped bar: baseline vs skill-injected per model
    win_loss.png                stacked bar: win/tie/loss per model
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


PALETTE = ("#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
           "#64B5CD", "#DD8452", "#A1C9F4")

OPERATOR_COLORS = {"atomic": "#4C72B0", "seq": "#55A868", "par": "#DD8452", "cond": "#C44E52"}


def _load_data(results_dir: Path) -> tuple:
    """Load episodes.json and summary.json from a results directory."""
    with open(results_dir / "episodes.json", "r", encoding="utf-8") as f:
        episodes = json.load(f)
    with open(results_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    return episodes, summary


def _parse_skill_name(skill_name: str) -> Tuple[str, int]:
    """Extract operator type and k-value from a composed skill name.

    Naming conventions:
        atomic skill (no prefix):  "extract-parallel-claims"         -> ("atomic", 1)
        seq composition:           "seq-skill1-then-skill2"          -> ("seq", 2)
        par composition:           "par-skill1-and-skill2-and-s3"    -> ("par", 3)
        cond composition:          "cond-skill1-then-skill2-and-s3"  -> ("cond", 3)

    Returns:
        (operator, k) tuple
    """
    if not skill_name:
        return ("baseline", 0)

    for prefix in ("seq-", "par-", "cond-"):
        if skill_name.startswith(prefix):
            operator = prefix[:-1]
            body = skill_name[len(prefix):]
            separators = len(re.findall(r"-(?:then|and)-", body))
            k = separators + 1
            return (operator, k)

    return ("atomic", 1)


def generate_score_by_k(
    episodes: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Line chart: mean score by k-value, one line per model."""
    # group by (model, k) -> scores
    by_model_k = defaultdict(lambda: defaultdict(list))
    for ep in episodes:
        if ep.get("condition") != "skill_injected":
            continue
        model = ep.get("model", "")
        _, k = _parse_skill_name(ep.get("skill_name", ""))
        score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
        by_model_k[model][k].append(score)

    if not by_model_k:
        return

    models = sorted(by_model_k.keys())
    all_k = sorted(set(k for m in models for k in by_model_k[m].keys()))

    # also compute baseline per model for reference line
    baseline_by_model = defaultdict(list)
    for ep in episodes:
        if ep.get("condition") == "baseline":
            model = ep.get("model", "")
            score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
            baseline_by_model[model].append(score)

    fig, ax = plt.subplots(figsize=(max(6, len(all_k) * 1.5 + 2), 5))

    for idx, model in enumerate(models):
        k_values = []
        means = []
        for k in all_k:
            vals = by_model_k[model].get(k, [])
            if vals:
                k_values.append(k)
                means.append(sum(vals) / len(vals))
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(k_values, means, marker="o", label=model, color=color, linewidth=2)

        # baseline reference
        b_vals = baseline_by_model.get(model, [])
        if b_vals:
            b_mean = sum(b_vals) / len(b_vals)
            ax.axhline(y=b_mean, color=color, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("Composition Size (k)", fontsize=11)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_title("Score by Composition Size (k)", fontsize=13, pad=12)
    ax.set_xticks(all_k)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_operator_heatmap(
    episodes: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Heatmap: operator type x model, cell = mean score (skill-injected only)."""
    scores = defaultdict(lambda: defaultdict(list))
    models_set = set()
    for ep in episodes:
        if ep.get("condition") != "skill_injected":
            continue
        model = ep.get("model", "")
        operator, _ = _parse_skill_name(ep.get("skill_name", ""))
        score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
        scores[operator][model].append(score)
        models_set.add(model)

    if not scores:
        return

    operators = sorted(scores.keys())
    models = sorted(models_set)

    data = np.zeros((len(operators), len(models)))
    for i, op in enumerate(operators):
        for j, m in enumerate(models):
            vals = scores[op][m]
            data[i, j] = sum(vals) / len(vals) if vals else 0.0

    figsize = (max(6, len(models) * 1.5 + 3), max(3, len(operators) * 0.8 + 2))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(operators)))
    ax.set_yticklabels(operators, fontsize=10)
    ax.set_title("Mean Score by Operator Type x Model", fontsize=13, pad=12)

    for i in range(len(operators)):
        for j in range(len(models)):
            val = data[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Score", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_uplift_heatmap(
    episodes: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Heatmap: skill x model, cell = delta (skill score - baseline score).

    Uses a diverging RdBu colormap: blue = positive uplift, red = negative.
    """
    # compute baseline mean per (model, task)
    baseline_scores = defaultdict(lambda: defaultdict(list))
    skill_scores = defaultdict(lambda: defaultdict(list))
    models_set = set()

    for ep in episodes:
        model = ep.get("model", "")
        score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
        models_set.add(model)
        if ep.get("condition") == "baseline":
            task = ep.get("task_uid", ep.get("task_id", ""))
            baseline_scores[model][task].append(score)
        elif ep.get("condition") == "skill_injected":
            sn = ep.get("skill_name", "")
            skill_scores[sn][model].append(score)

    if not skill_scores:
        return

    # compute baseline mean per model (across all tasks)
    baseline_mean = {}
    for m in models_set:
        all_b = [s for task_scores in baseline_scores[m].values() for s in task_scores]
        baseline_mean[m] = sum(all_b) / len(all_b) if all_b else 0.0

    skills = sorted(skill_scores.keys())
    models = sorted(models_set)

    data = np.zeros((len(skills), len(models)))
    for i, sk in enumerate(skills):
        for j, m in enumerate(models):
            vals = skill_scores[sk][m]
            sk_mean = sum(vals) / len(vals) if vals else 0.0
            data[i, j] = sk_mean - baseline_mean.get(m, 0.0)

    # shorten skill labels for display
    labels = []
    for sk in skills:
        op, k = _parse_skill_name(sk)
        if op == "atomic":
            label = sk[:30]
        else:
            label = f"[{op} k={k}] {sk[len(op)+1:30]}"
        labels.append(label)

    vmax = max(abs(data.min()), abs(data.max()), 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    figsize = (max(6, len(models) * 1.5 + 3), max(4, len(skills) * 0.45 + 2))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap="RdBu", norm=norm, aspect="auto")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(skills)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Uplift Heatmap: Score Delta (Skill - Baseline)", fontsize=13, pad=12)

    for i in range(len(skills)):
        for j in range(len(models)):
            val = data[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Score Delta", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_k_operator_heatmap(
    episodes: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Heatmap: rows = (k, operator) combinations, columns = models."""
    scores = defaultdict(lambda: defaultdict(list))
    models_set = set()
    for ep in episodes:
        if ep.get("condition") != "skill_injected":
            continue
        model = ep.get("model", "")
        operator, k = _parse_skill_name(ep.get("skill_name", ""))
        score = ep.get("score", 1.0 if ep.get("passed") else 0.0)
        row_key = f"k={k} {operator}"
        scores[row_key][model].append(score)
        models_set.add(model)

    if not scores:
        return

    row_keys = sorted(scores.keys())
    models = sorted(models_set)

    data = np.zeros((len(row_keys), len(models)))
    for i, rk in enumerate(row_keys):
        for j, m in enumerate(models):
            vals = scores[rk][m]
            data[i, j] = sum(vals) / len(vals) if vals else 0.0

    figsize = (max(6, len(models) * 1.5 + 3), max(3, len(row_keys) * 0.6 + 2))
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(row_keys, fontsize=9)
    ax.set_title("Mean Score by (k, Operator) x Model", fontsize=13, pad=12)

    for i in range(len(row_keys)):
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

    for bar in list(bars_b) + list(bars_s):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

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

    charts = [
        ("score_by_k.png", lambda p: generate_score_by_k(episodes, p, dpi)),
        ("operator_heatmap.png", lambda p: generate_operator_heatmap(episodes, p, dpi)),
        ("uplift_heatmap.png", lambda p: generate_uplift_heatmap(episodes, p, dpi)),
        ("k_operator_heatmap.png", lambda p: generate_k_operator_heatmap(episodes, p, dpi)),
        ("baseline_vs_skill.png", lambda p: generate_baseline_vs_skill_bar(summary, p, dpi)),
        ("win_loss.png", lambda p: generate_win_loss_bar(episodes, summary, p, dpi)),
    ]

    for filename, gen_fn in charts:
        path = output_dir / filename
        gen_fn(path)
        generated.append(str(path))

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
