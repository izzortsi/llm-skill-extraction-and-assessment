"""
summary.py

Statistical aggregation utilities for evaluation episodes.
"""

from __future__ import annotations

from typing import Any, Dict, List


def compute_summary(episodes: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics from episodes.

    Groups episodes by model and condition, then computes mean scores
    and deltas between baseline and skill-injected conditions.

    Args:
        episodes: List of episode dicts, each with model, condition, and
                  optional score fields.

    Returns:
        Dict mapping model name to summary statistics.
    """
    by_model: Dict[str, Dict[str, List[float]]] = {}
    for ep in episodes:
        model = ep.get("model", "unknown")
        condition = ep.get("condition", "")
        if model not in by_model:
            by_model[model] = {"baseline": [], "skill_injected": []}
        if "score" in ep:
            by_model[model].setdefault(condition, []).append(ep["score"])

    summary: Dict[str, Any] = {}
    for model, conditions in by_model.items():
        baseline_scores = conditions.get("baseline", [])
        skill_scores = conditions.get("skill_injected", [])

        baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        skill_mean = sum(skill_scores) / len(skill_scores) if skill_scores else 0.0

        summary[model] = {
            "baseline_mean_score": round(baseline_mean, 4),
            "skill_mean_score": round(skill_mean, 4),
            "delta": round(skill_mean - baseline_mean, 4),
            "n_baseline": len(baseline_scores),
            "n_skill": len(skill_scores),
        }

    return summary
