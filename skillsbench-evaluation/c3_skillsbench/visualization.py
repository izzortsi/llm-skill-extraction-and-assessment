"""
visualization.py

Generate heatmap visualizations for benchmark results.
Supports both SkillsBench corpus results and SkillMix experiment results.

Three heatmap types:
  1. Uplift heatmap: delta (skill-injected minus baseline) per task x model
  2. Pass rate heatmap: pass rate per task x model for a single condition
  3. Combined heatmap: baseline pass rate (left) + uplift (right) side-by-side

Usage:
    python -m c3_skillsbench.visualization --results ../llm-skills.shared-data/260323.skilleval-corpus-judge-results/corpus_results.json -o heatmaps/
    python -m c3_skillsbench.visualization --results-dir ../llm-skills.shared-data/skilleval-skillmix-results/ -o heatmaps/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_episodes_from_corpus_results(filepath: Path) -> List[dict]:
    """Load episodes from SkillsBench corpus_results.json format."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # corpus_results.json is a list of episode dicts with:
    # task_id, model, condition, skill_name, passed, score, ...
    return data


def _load_episodes_from_skillmix_results(results_dir: Path) -> List[dict]:
    """Load episodes from SkillMix episodes.json format."""
    episodes_path = results_dir / "episodes.json"
    if not episodes_path.exists():
        return []
    with open(episodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # SkillMix episodes use "skill_injected" as condition name;
    # normalize to "curated" for consistency with heatmap logic
    for ep in data:
        if ep.get("condition") == "skill_injected":
            ep["condition"] = "curated"
    return data


def load_episodes(
    results_file: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    mode_filter: str = "",
) -> List[dict]:
    """Load episodes from either format, optionally filtered by mode.

    Args:
        results_file: Path to corpus_results.json (SkillsBench)
        results_dir: Path to skillmix-results/ directory (SkillMix)
        mode_filter: If non-empty, only return episodes matching this mode
                     (singlecall, stepwise, skillselect). Empty = all modes.

    Returns:
        List of episode dicts with task_id, model, condition, passed, score
    """
    episodes = []
    if results_file and results_file.exists():
        episodes.extend(_load_episodes_from_corpus_results(results_file))
    if results_dir and results_dir.exists():
        episodes.extend(_load_episodes_from_skillmix_results(results_dir))

    if mode_filter:
        episodes = [ep for ep in episodes if ep.get("mode", "singlecall") == mode_filter]

    return episodes


def build_task_model_uplift_matrix(
    episodes: List[dict],
) -> Dict[str, Dict[str, float]]:
    """Build a task x model uplift matrix from episode dicts.

    For each (task, model) pair, computes:
        delta = mean(curated_scores) - mean(baseline_scores)

    Args:
        episodes: List of episode dicts with task_id, model, condition, score/passed

    Returns:
        Nested dict: {task_id: {model: delta}}
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for ep in episodes:
        task_id = ep.get("task_uid", ep.get("task_id", ep.get("problem_id", "")))
        model = ep.get("model", "")
        condition = ep.get("condition", "")
        # Use passed (binary) if score not available
        if "score" in ep:
            score = float(ep["score"])
        elif "passed" in ep:
            score = 1.0 if ep["passed"] else 0.0
        else:
            continue

        grouped[(task_id, model)][condition].append(score)

    matrix = {}
    for (task_id, model), conditions in grouped.items():
        baseline_scores = conditions.get("baseline", [])
        curated_scores = conditions.get("curated", [])

        if not baseline_scores or not curated_scores:
            continue

        baseline_mean = sum(baseline_scores) / len(baseline_scores)
        curated_mean = sum(curated_scores) / len(curated_scores)
        delta = curated_mean - baseline_mean

        if task_id not in matrix:
            matrix[task_id] = {}
        matrix[task_id][model] = round(delta, 4)

    return matrix


def _shorten_task_uid(task_uid: str) -> str:
    """Shorten task UIDs for display labels."""
    if task_uid.startswith("ext-"):
        return task_uid[4:]
    if task_uid.startswith("ipc-"):
        return task_uid[4:]
    if len(task_uid) > 25:
        return task_uid[:22] + "..."
    return task_uid


def generate_uplift_heatmap(
    episodes: List[dict],
    output_path: Path,
    title: str = "Skill Uplift per Task (with Skill - without Skill)",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
) -> Path:
    """Generate an uplift heatmap (delta between skill-injected and baseline).

    Rows = tasks, Columns = models, Cells = score delta.
    Blue = positive uplift, Red = negative uplift.

    Args:
        episodes: List of episode dicts
        output_path: Path to save PNG
        title: Plot title
        figsize: Figure size (width, height) in inches; auto-computed if None
        dpi: Output resolution

    Returns:
        Path to saved image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    matrix = build_task_model_uplift_matrix(episodes)

    if not matrix:
        print("WARNING: No uplift data to plot (need both baseline and curated conditions)")
        return output_path

    all_models = set()
    for task_deltas in matrix.values():
        all_models.update(task_deltas.keys())

    # Sort models by aggregate uplift (descending)
    model_avg_uplift = {}
    for model in all_models:
        deltas = [matrix[t].get(model, 0.0) for t in matrix if model in matrix[t]]
        model_avg_uplift[model] = sum(deltas) / len(deltas) if deltas else 0.0

    models = sorted(all_models, key=lambda m: model_avg_uplift[m], reverse=True)
    tasks = sorted(matrix.keys())

    data = np.zeros((len(tasks), len(models)))
    for i, task_id in enumerate(tasks):
        for j, model in enumerate(models):
            data[i, j] = matrix.get(task_id, {}).get(model, 0.0)

    if figsize is None:
        width = max(6, len(models) * 1.5 + 3)
        height = max(4, len(tasks) * 0.4 + 2)
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)

    vmax = max(abs(data.min()), abs(data.max()), 0.1)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu

    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    for i in range(len(tasks)):
        for j in range(len(models)):
            val = data[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(tasks)))
    task_labels = [_shorten_task_uid(t) for t in tasks]
    ax.set_yticklabels(task_labels, fontsize=8)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Task", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Score Delta", fontsize=10)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Heatmap saved: {output_path}")
    return output_path


def generate_pass_rate_heatmap(
    episodes: List[dict],
    output_path: Path,
    condition: str = "baseline",
    title: str = "Pass Rate per Task per Model",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
) -> Path:
    """Generate a pass-rate heatmap for a single condition.

    Args:
        episodes: List of episode dicts
        output_path: Path to save PNG
        condition: Which condition to show ("baseline" or "curated")
        title: Plot title
        figsize: Figure dimensions
        dpi: Output resolution

    Returns:
        Path to saved image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grouped = defaultdict(list)
    for ep in episodes:
        if ep.get("condition") == condition:
            task_id = ep.get("task_uid", ep.get("task_id", ep.get("problem_id", "")))
            model = ep.get("model", "")
            is_passed = ep.get("passed", False)
            grouped[(task_id, model)].append(1.0 if is_passed else 0.0)

    if not grouped:
        print(f"WARNING: No records for condition '{condition}'")
        return output_path

    all_tasks = sorted(set(k[0] for k in grouped))
    all_models = sorted(set(k[1] for k in grouped))

    data = np.zeros((len(all_tasks), len(all_models)))
    for i, task_id in enumerate(all_tasks):
        for j, model in enumerate(all_models):
            scores = grouped.get((task_id, model), [])
            data[i, j] = sum(scores) / len(scores) if scores else 0.0

    if figsize is None:
        width = max(6, len(all_models) * 1.5 + 3)
        height = max(4, len(all_tasks) * 0.4 + 2)
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0,
                   interpolation="nearest")

    for i in range(len(all_tasks)):
        for j in range(len(all_models)):
            val = data[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(all_models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(all_tasks)))
    task_labels = [_shorten_task_uid(t) for t in all_tasks]
    ax.set_yticklabels(task_labels, fontsize=8)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Task", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Pass Rate", fontsize=10)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Heatmap saved: {output_path}")
    return output_path


def generate_combined_heatmap(
    episodes: List[dict],
    output_path: Path,
    baseline_title: str = "Baseline Pass Rate",
    uplift_title: str = "Skill Uplift (curated \u2212 baseline)",
    dpi: int = 150,
) -> Path:
    """Generate side-by-side figure: baseline pass rate (left) + uplift (right).

    Args:
        episodes: List of episode dicts
        output_path: Path to save PNG
        baseline_title: Left pane title
        uplift_title: Right pane title
        dpi: Output resolution

    Returns:
        Path to saved image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Build pass-rate data (baseline condition)
    pr_grouped = defaultdict(list)
    for ep in episodes:
        if ep.get("condition") == "baseline":
            task_id = ep.get("task_uid", ep.get("task_id", ep.get("problem_id", "")))
            model = ep.get("model", "")
            pr_grouped[(task_id, model)].append(1.0 if ep.get("passed", False) else 0.0)

    if not pr_grouped:
        print("WARNING: No baseline records for combined heatmap")
        return output_path

    all_tasks = sorted(set(k[0] for k in pr_grouped))
    all_models = sorted(set(k[1] for k in pr_grouped))

    pr_data = np.zeros((len(all_tasks), len(all_models)))
    for i, task_id in enumerate(all_tasks):
        for j, model in enumerate(all_models):
            scores = pr_grouped.get((task_id, model), [])
            pr_data[i, j] = sum(scores) / len(scores) if scores else 0.0

    # Build uplift data
    uplift_matrix = build_task_model_uplift_matrix(episodes)
    up_data = np.zeros((len(all_tasks), len(all_models)))
    for i, task_id in enumerate(all_tasks):
        for j, model in enumerate(all_models):
            up_data[i, j] = uplift_matrix.get(task_id, {}).get(model, 0.0)

    # Figure with two subplots
    n_tasks = len(all_tasks)
    n_models = len(all_models)
    pane_w = max(5, n_models * 1.2 + 2)
    pane_h = max(4, n_tasks * 0.45 + 1.5)
    fig, (ax_pr, ax_up) = plt.subplots(1, 2, figsize=(pane_w * 2 + 1.5, pane_h),
                                        sharey=True)

    task_labels = [_shorten_task_uid(t) for t in all_tasks]

    # Left pane: baseline pass rate
    im_pr = ax_pr.imshow(pr_data, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0,
                         interpolation="nearest")
    for i in range(n_tasks):
        for j in range(n_models):
            val = pr_data[i, j]
            color = "white" if val > 0.6 else "black"
            ax_pr.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=8, color=color)
    ax_pr.set_xticks(range(n_models))
    ax_pr.set_xticklabels(all_models, rotation=45, ha="right", fontsize=9)
    ax_pr.set_yticks(range(n_tasks))
    ax_pr.set_yticklabels(task_labels, fontsize=8)
    ax_pr.set_ylabel("Task", fontsize=11)
    ax_pr.set_xlabel("Model", fontsize=11)
    ax_pr.set_title(baseline_title, fontsize=12, pad=10)
    cbar_pr = fig.colorbar(im_pr, ax=ax_pr, shrink=0.8, pad=0.02)
    cbar_pr.set_label("Pass Rate", fontsize=9)

    # Right pane: uplift
    vmax = max(abs(up_data.min()), abs(up_data.max()), 0.1)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im_up = ax_up.imshow(up_data, aspect="auto", cmap=plt.cm.RdBu, norm=norm,
                         interpolation="nearest")
    for i in range(n_tasks):
        for j in range(n_models):
            val = up_data[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax_up.text(j, i, f"{val:+.2f}", ha="center", va="center",
                       fontsize=8, color=color)
    ax_up.set_xticks(range(n_models))
    ax_up.set_xticklabels(all_models, rotation=45, ha="right", fontsize=9)
    ax_up.set_xlabel("Model", fontsize=11)
    ax_up.set_title(uplift_title, fontsize=12, pad=10)
    cbar_up = fig.colorbar(im_up, ax=ax_up, shrink=0.8, pad=0.02)
    cbar_up.set_label("Score Delta", fontsize=9)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Combined heatmap saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Chart: grouped bar -- delta by model, grouped by mode
# ---------------------------------------------------------------------------

def generate_delta_by_mode_bar(
    episodes: List[dict],
    output_path: Path,
    title: str = "Skill Delta by Model and Mode",
    dpi: int = 150,
) -> Path:
    """Grouped bar chart: one group per model, one bar per mode.

    Args:
        episodes: List of episode dicts (must include 'mode' field).
        output_path: Path to save PNG.
        title: Plot title.
        dpi: Output resolution.

    Returns:
        Path to saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes = sorted(set(ep.get("mode", "singlecall") for ep in episodes))
    models = sorted(set(ep.get("model", "") for ep in episodes))

    deltas: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        deltas[mode] = {}
        for model in models:
            bl = [ep["score"] for ep in episodes
                  if ep.get("model") == model and ep.get("condition") == "baseline"
                  and ep.get("mode", "singlecall") == mode]
            cu = [ep["score"] for ep in episodes
                  if ep.get("model") == model and ep.get("condition") == "curated"
                  and ep.get("mode") == mode]
            bl_avg = sum(bl) / len(bl) if bl else 0
            cu_avg = sum(cu) / len(cu) if cu else 0
            deltas[mode][model] = cu_avg - bl_avg

    x = np.arange(len(models))
    bar_width = 0.8 / len(modes)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))

    for i, mode in enumerate(modes):
        vals = [deltas[mode].get(m, 0) for m in models]
        offset = (i - len(modes) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=mode,
                      color=colors[i % len(colors)], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            y_pos = bar.get_height()
            va = "bottom" if y_pos >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:+.3f}", ha="center", va=va, fontsize=7)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score Delta (curated - baseline)", fontsize=10)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(title="Mode", fontsize=9, title_fontsize=10)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Delta-by-mode bar chart saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Chart: scatter -- baseline score vs curated score per episode
# ---------------------------------------------------------------------------

def generate_baseline_vs_curated_scatter(
    episodes: List[dict],
    output_path: Path,
    title: str = "Baseline vs Curated Score per Task",
    dpi: int = 150,
) -> Path:
    """Scatter plot: baseline score (x) vs curated score (y), color-coded by model.

    Points above the diagonal show skill benefit; below show skill harm.

    Args:
        episodes: List of episode dicts.
        output_path: Path to save PNG.
        title: Plot title.
        dpi: Output resolution.

    Returns:
        Path to saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for ep in episodes:
        key = (ep.get("task_uid", ep.get("task_id", "")), ep.get("model", ""), ep.get("mode", "singlecall"))
        if key not in grouped:
            grouped[key] = {}
        grouped[key][ep.get("condition", "")] = ep.get("score", 0)

    models = sorted(set(k[1] for k in grouped))
    color_map = {}
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
               "#64B5CD", "#DD8452", "#A1C9F4"]
    for i, model in enumerate(models):
        color_map[model] = palette[i % len(palette)]

    fig, ax = plt.subplots(figsize=(7, 7))

    for (task_id, model, mode), scores in grouped.items():
        bl = scores.get("baseline")
        cu = scores.get("curated")
        if bl is not None and cu is not None:
            ax.scatter(bl, cu, c=color_map[model], s=50, alpha=0.7,
                       edgecolors="white", linewidths=0.5)

    ax.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle="--", alpha=0.6)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color_map[m], markersize=8, label=m)
               for m in models]
    ax.legend(handles=handles, title="Model", fontsize=8, title_fontsize=9,
              loc="lower right")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Baseline Score", fontsize=11)
    ax.set_ylabel("Curated Score", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_aspect("equal")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Baseline vs curated scatter saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Chart: stacked bar -- win/loss/tie per model per mode
# ---------------------------------------------------------------------------

def generate_win_loss_bar(
    episodes: List[dict],
    output_path: Path,
    title: str = "Win/Loss/Tie per Model and Mode",
    threshold: float = 0.01,
    dpi: int = 150,
) -> Path:
    """Stacked bar chart: wins (green), ties (gray), losses (red) per model/mode.

    A 'win' is a task where curated score exceeds baseline score by more than
    the threshold. A 'loss' is the opposite. A 'tie' is within the threshold.

    Args:
        episodes: List of episode dicts.
        output_path: Path to save PNG.
        title: Plot title.
        threshold: Minimum absolute delta to count as win or loss.
        dpi: Output resolution.

    Returns:
        Path to saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes = sorted(set(ep.get("mode", "singlecall") for ep in episodes))
    models = sorted(set(ep.get("model", "") for ep in episodes))

    counts: Dict[str, Dict[str, List[int]]] = {}
    for mode in modes:
        counts[mode] = {}
        for model in models:
            task_scores: Dict[str, Dict[str, float]] = {}
            for ep in episodes:
                if ep.get("model") != model:
                    continue
                if ep.get("mode", "singlecall") != mode and ep.get("condition") != "baseline":
                    continue
                tid = ep.get("task_uid", ep.get("task_id", ""))
                if tid not in task_scores:
                    task_scores[tid] = {}
                task_scores[tid][ep.get("condition", "")] = ep.get("score", 0)

            wins = losses = ties = 0
            for tid, sc in task_scores.items():
                bl = sc.get("baseline")
                cu = sc.get("curated")
                if bl is not None and cu is not None:
                    delta = cu - bl
                    if delta > threshold:
                        wins += 1
                    elif delta < -threshold:
                        losses += 1
                    else:
                        ties += 1
            counts[mode][model] = [wins, ties, losses]

    labels = []
    for mode in modes:
        for model in models:
            labels.append(f"{model}\n({mode})")

    wins_list = []
    ties_list = []
    losses_list = []
    for mode in modes:
        for model in models:
            w, t, l = counts[mode].get(model, [0, 0, 0])
            wins_list.append(w)
            ties_list.append(t)
            losses_list.append(l)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))

    ax.bar(x, wins_list, color="#55A868", label="Win (skill helps)", edgecolor="white")
    ax.bar(x, ties_list, bottom=wins_list, color="#CCCCCC", label="Tie", edgecolor="white")
    bottoms = [w + t for w, t in zip(wins_list, ties_list)]
    ax.bar(x, losses_list, bottom=bottoms, color="#C44E52", label="Loss (skill hurts)", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Number of Tasks", fontsize=10)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=9)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Win/loss bar chart saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Chart: scatter -- token cost vs delta
# ---------------------------------------------------------------------------

def generate_token_vs_delta_scatter(
    episodes: List[dict],
    output_path: Path,
    title: str = "Token Cost vs Score Delta",
    dpi: int = 150,
) -> Path:
    """Scatter plot: curated token usage (x) vs score delta (y), color-coded by mode.

    Shows cost-effectiveness of each scaffolding mode.

    Args:
        episodes: List of episode dicts (must include 'tokens' field).
        output_path: Path to save PNG.
        title: Plot title.
        dpi: Output resolution.

    Returns:
        Path to saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: Dict[Tuple[str, str, str], Dict[str, dict]] = {}
    for ep in episodes:
        key = (ep.get("task_uid", ep.get("task_id", "")), ep.get("model", ""), ep.get("mode", "singlecall"))
        if key not in grouped:
            grouped[key] = {}
        grouped[key][ep.get("condition", "")] = {
            "score": ep.get("score", 0),
            "tokens": ep.get("tokens", 0),
        }

    modes = sorted(set(k[2] for k in grouped))
    mode_colors = {}
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for i, mode in enumerate(modes):
        mode_colors[mode] = palette[i % len(palette)]

    fig, ax = plt.subplots(figsize=(9, 6))

    for (task_id, model, mode), conds in grouped.items():
        bl = conds.get("baseline")
        cu = conds.get("curated")
        if bl is not None and cu is not None:
            delta = cu["score"] - bl["score"]
            cu_tokens = cu["tokens"]
            if cu_tokens > 0:
                ax.scatter(cu_tokens, delta, c=mode_colors[mode], s=40,
                           alpha=0.6, edgecolors="white", linewidths=0.5)

    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=mode_colors[m], markersize=8, label=m)
               for m in modes]
    ax.legend(handles=handles, title="Mode", fontsize=9, title_fontsize=10)

    ax.set_xlabel("Curated Episode Tokens", fontsize=11)
    ax.set_ylabel("Score Delta (curated - baseline)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Token vs delta scatter saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Chart: radar -- per-skill delta by mode
# ---------------------------------------------------------------------------

def generate_skill_radar(
    episodes: List[dict],
    output_path: Path,
    title: str = "Per-Skill Delta by Mode",
    dpi: int = 150,
) -> Path:
    """Radar chart: one axis per skill, one line per mode.

    Shows which skills benefit most from which scaffolding mode.
    Only includes episodes for Ollama models (excludes frontier models
    that saturate at 1.0 baseline).

    Args:
        episodes: List of episode dicts (must include 'skill_name' and 'mode').
        output_path: Path to save PNG.
        title: Plot title.
        dpi: Output resolution.

    Returns:
        Path to saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    frontier_models = {"claude-opus-4-6", "glm-5-turbo"}
    filtered = [ep for ep in episodes if ep.get("model", "") not in frontier_models]

    modes = sorted(set(ep.get("mode", "singlecall") for ep in filtered))
    skills = sorted(set(ep.get("skill_name", "") for ep in filtered
                        if ep.get("skill_name", "") and ep.get("condition") == "curated"))

    if not skills or not modes:
        print("WARNING: Not enough skill/mode data for radar chart")
        return output_path

    mode_deltas: Dict[str, List[float]] = {}
    for mode in modes:
        deltas = []
        for skill in skills:
            bl_scores = [ep["score"] for ep in filtered
                         if ep.get("condition") == "baseline"
                         and ep.get("mode", "singlecall") == mode]
            cu_scores = [ep["score"] for ep in filtered
                         if ep.get("condition") == "curated"
                         and ep.get("mode") == mode
                         and ep.get("skill_name") == skill]

            bl_avg = sum(bl_scores) / len(bl_scores) if bl_scores else 0
            cu_avg = sum(cu_scores) / len(cu_scores) if cu_scores else 0
            deltas.append(cu_avg - bl_avg)
        mode_deltas[mode] = deltas

    angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
    angles.append(angles[0])

    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    for i, mode in enumerate(modes):
        values = mode_deltas[mode] + [mode_deltas[mode][0]]
        ax.plot(angles, values, "o-", linewidth=2, markersize=5,
                label=mode, color=palette[i % len(palette)])
        ax.fill(angles, values, alpha=0.1, color=palette[i % len(palette)])

    skill_labels = [s[:25] + "..." if len(s) > 25 else s for s in skills]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(skill_labels, fontsize=8)
    ax.set_title(title, fontsize=13, pad=20)
    ax.legend(title="Mode", loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=9, title_fontsize=10)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Skill radar chart saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Heatmap orchestration (original)
# ---------------------------------------------------------------------------

def _generate_heatmaps_for_episodes(
    episodes: List[dict],
    output_dir: Path,
    heatmap_type: str,
    suffix: str = "",
    dpi: int = 150,
) -> None:
    """Generate heatmaps and charts for a set of episodes.

    Args:
        episodes: List of episode dicts
        output_dir: Output directory for PNGs
        heatmap_type: Which outputs: all, uplift, baseline, combined, charts
        suffix: Filename suffix (e.g., "_stepwise") for mode-specific outputs
        dpi: Output resolution
    """
    if not episodes:
        return

    is_all = heatmap_type == "all"

    if is_all or heatmap_type == "uplift":
        generate_uplift_heatmap(
            episodes,
            output_dir / f"uplift_heatmap{suffix}.png",
            title=f"Skill Uplift{' (' + suffix.strip('_') + ')' if suffix else ''}",
            dpi=dpi,
        )

    if is_all or heatmap_type == "baseline":
        generate_pass_rate_heatmap(
            episodes,
            output_dir / f"baseline_pass_rate{suffix}.png",
            condition="baseline",
            title=f"Baseline Pass Rate{' (' + suffix.strip('_') + ')' if suffix else ''}",
            dpi=dpi,
        )

    if is_all or heatmap_type == "combined":
        generate_combined_heatmap(
            episodes,
            output_dir / f"combined_heatmap{suffix}.png",
            dpi=dpi,
        )

    if is_all or heatmap_type == "charts":
        has_multiple_modes = len(set(ep.get("mode", "singlecall") for ep in episodes)) > 1
        if has_multiple_modes:
            generate_delta_by_mode_bar(
                episodes,
                output_dir / f"delta_by_mode{suffix}.png",
                dpi=dpi,
            )
        generate_baseline_vs_curated_scatter(
            episodes,
            output_dir / f"baseline_vs_curated{suffix}.png",
            dpi=dpi,
        )
        if has_multiple_modes:
            generate_win_loss_bar(
                episodes,
                output_dir / f"win_loss{suffix}.png",
                dpi=dpi,
            )
            generate_token_vs_delta_scatter(
                episodes,
                output_dir / f"token_vs_delta{suffix}.png",
                dpi=dpi,
            )
            generate_skill_radar(
                episodes,
                output_dir / f"skill_radar{suffix}.png",
                dpi=dpi,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate heatmap visualizations from benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All modes combined
  python -m c3_skillsbench.visualization --results ../llm-skills.shared-data/260323.skilleval-corpus-judge-results/corpus_results.json -o heatmaps/

  # Filter to stepwise mode only
  python -m c3_skillsbench.visualization --results ../llm-skills.shared-data/results.json -o heatmaps/ --mode stepwise

  # Separate heatmaps per mode (auto-detected from data)
  python -m c3_skillsbench.visualization --results ../llm-skills.shared-data/results.json -o heatmaps/ --mode per-mode

  # From SkillMix results
  python -m c3_skillsbench.visualization --results-dir ../llm-skills.shared-data/skilleval-skillmix-results/ -o heatmaps/
""",
    )
    parser.add_argument("--results", type=Path, nargs="+", help="Path(s) to corpus_results.json (multiple files merged)")
    parser.add_argument("--results-dir", type=Path, help="Path to skillmix-results/ directory")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("stage6-visualization/heatmaps"), help="Output directory for PNGs")
    parser.add_argument("--type", choices=["all", "uplift", "baseline", "combined", "charts"],
                        default="all", help="Which output(s) to generate: all, uplift, baseline, combined, charts (default: all)")
    parser.add_argument("--mode", type=str, default="",
                        help="Filter by mode: singlecall, stepwise, skillselect, "
                             "per-mode (separate heatmaps per mode), or empty for all")
    parser.add_argument("--dpi", type=int, default=150, help="Output resolution (default: 150)")
    args = parser.parse_args()

    if not args.results and not args.results_dir:
        parser.error("At least one of --results or --results-dir is required")

    # Load all episodes first (unfiltered), merging multiple result files
    all_episodes = []
    if args.results:
        for results_file in args.results:
            all_episodes.extend(load_episodes(results_file=results_file))
    if args.results_dir:
        all_episodes.extend(load_episodes(results_dir=args.results_dir))
    if not all_episodes:
        print("ERROR: No episodes loaded. Check input paths.")
        return

    # Detect available modes
    modes_present = sorted(set(ep.get("mode", "singlecall") for ep in all_episodes))
    print(f"Loaded {len(all_episodes)} episodes, modes: {modes_present}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Baselines are always mode=singlecall. When filtering by a non-baseline
    # mode (stepwise, skillselect), we must include the singlecall baselines
    # so that uplift deltas can be computed.
    baseline_episodes = [ep for ep in all_episodes if ep.get("condition") == "baseline"]

    if args.mode == "per-mode":
        # Generate separate heatmaps for each non-baseline mode
        treatment_modes = [m for m in modes_present if m != "singlecall"]
        if not treatment_modes:
            treatment_modes = modes_present  # only singlecall present
        for mode_name in treatment_modes:
            mode_episodes = [ep for ep in all_episodes if ep.get("mode", "singlecall") == mode_name]
            # Merge baselines so uplift can be computed
            combined = baseline_episodes + [ep for ep in mode_episodes if ep.get("condition") != "baseline"]
            if not combined:
                continue
            print(f"\n--- Mode: {mode_name} ({len(mode_episodes)} treatment + {len(baseline_episodes)} baseline) ---")
            _generate_heatmaps_for_episodes(
                combined, args.output_dir, args.type,
                suffix=f"_{mode_name}", dpi=args.dpi,
            )
    elif args.mode:
        # Filter to specific mode, but always include baselines
        mode_episodes = [ep for ep in all_episodes if ep.get("mode", "singlecall") == args.mode]
        if not mode_episodes:
            print(f"ERROR: No episodes with mode '{args.mode}'. Available: {modes_present}")
            return
        combined = baseline_episodes + [ep for ep in mode_episodes if ep.get("condition") != "baseline"]
        print(f"Filtered: {len(mode_episodes)} {args.mode} + {len(baseline_episodes)} baselines = {len(combined)} total")
        _generate_heatmaps_for_episodes(
            combined, args.output_dir, args.type,
            suffix=f"_{args.mode}", dpi=args.dpi,
        )
    else:
        # All modes combined
        _generate_heatmaps_for_episodes(
            all_episodes, args.output_dir, args.type, dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
