"""
effectiveness.py

Compute pass rate deltas and aggregate effectiveness metrics.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Any

from providers.stat_utils import (
    pass_rate,
    pass_rate_delta_pp,
    bootstrap_ci,
    permutation_test,
)
from config.trial_result import BenchmarkRecord


def compute_pass_rate_delta(
    baseline_records: List[BenchmarkRecord],
    treatment_records: List[BenchmarkRecord],
) -> Dict[str, Any]:
    """Compute pass rate delta between baseline and treatment.

    Args:
        baseline_records: Records from baseline condition
        treatment_records: Records from treatment condition

    Returns:
        Dict with delta_pp, baseline_rate, treatment_rate, p_value, ci
    """
    baseline_outcomes = [r.passed for r in baseline_records]
    treatment_outcomes = [r.passed for r in treatment_records]

    baseline_rate = pass_rate(baseline_outcomes)
    treatment_rate = pass_rate(treatment_outcomes)
    delta_pp = pass_rate_delta_pp(baseline_outcomes, treatment_outcomes)

    # Permutation test on scores
    baseline_scores = [r.score for r in baseline_records]
    treatment_scores = [r.score for r in treatment_records]
    _, p_value = permutation_test(treatment_scores, baseline_scores)

    # Bootstrap CI on treatment pass rate
    treatment_floats = [1.0 if o else 0.0 for o in treatment_outcomes]
    _, ci_lower, ci_upper = bootstrap_ci(treatment_floats)

    return {
        "delta_pp": round(delta_pp, 2),
        "baseline_rate": round(baseline_rate, 4),
        "treatment_rate": round(treatment_rate, 4),
        "p_value": round(p_value, 4),
        "treatment_ci_lower": round(ci_lower, 4),
        "treatment_ci_upper": round(ci_upper, 4),
        "n_baseline": len(baseline_records),
        "n_treatment": len(treatment_records),
    }


def _group_records(
    records: List[BenchmarkRecord],
    key_fn,
) -> Dict[str, List[BenchmarkRecord]]:
    """Group records by a key function."""
    groups = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    return dict(groups)


def aggregate_by_skill(records: List[BenchmarkRecord]) -> Dict[str, Dict[str, Any]]:
    """Aggregate effectiveness per skill.

    Computes delta between baseline and each curated skill.

    Args:
        records: All benchmark records

    Returns:
        Dict mapping skill_name -> effectiveness metrics
    """
    baseline = [r for r in records if r.condition == "baseline"]
    curated = [r for r in records if r.condition == "curated"]

    # Group baseline by task
    baseline_by_task = _group_records(baseline, lambda r: r.problem_id)

    # Group curated by (task, skill)
    skill_groups = _group_records(curated, lambda r: r.skill_name)

    results = {}
    for skill_name, skill_records in skill_groups.items():
        # Find matching baseline records (same tasks)
        task_ids = set(r.problem_id for r in skill_records)
        matching_baseline = [r for r in baseline if r.problem_id in task_ids]

        if matching_baseline:
            results[skill_name] = compute_pass_rate_delta(matching_baseline, skill_records)
        else:
            results[skill_name] = {
                "delta_pp": 0.0,
                "baseline_rate": 0.0,
                "treatment_rate": pass_rate([r.passed for r in skill_records]),
                "p_value": 1.0,
                "n_baseline": 0,
                "n_treatment": len(skill_records),
            }

    return results


def aggregate_by_domain(records: List[BenchmarkRecord]) -> Dict[str, Dict[str, Any]]:
    """Aggregate effectiveness per domain."""
    baseline = [r for r in records if r.condition == "baseline"]
    curated = [r for r in records if r.condition == "curated"]

    baseline_by_domain = _group_records(baseline, lambda r: r.domain)
    curated_by_domain = _group_records(curated, lambda r: r.domain)

    results = {}
    all_domains = set(list(baseline_by_domain.keys()) + list(curated_by_domain.keys()))

    for domain in sorted(all_domains):
        b = baseline_by_domain.get(domain, [])
        t = curated_by_domain.get(domain, [])
        if b and t:
            results[domain] = compute_pass_rate_delta(b, t)
        else:
            results[domain] = {
                "delta_pp": 0.0,
                "n_baseline": len(b),
                "n_treatment": len(t),
            }

    return results


def aggregate_by_model(records: List[BenchmarkRecord]) -> Dict[str, Dict[str, Any]]:
    """Aggregate effectiveness per model."""
    baseline = [r for r in records if r.condition == "baseline"]
    curated = [r for r in records if r.condition == "curated"]

    baseline_by_model = _group_records(baseline, lambda r: r.model)
    curated_by_model = _group_records(curated, lambda r: r.model)

    results = {}
    all_models = set(list(baseline_by_model.keys()) + list(curated_by_model.keys()))

    for model in sorted(all_models):
        b = baseline_by_model.get(model, [])
        t = curated_by_model.get(model, [])
        if b and t:
            results[model] = compute_pass_rate_delta(b, t)
        else:
            results[model] = {
                "delta_pp": 0.0,
                "n_baseline": len(b),
                "n_treatment": len(t),
            }

    return results


def aggregate_by_k_value(records: List[BenchmarkRecord]) -> Dict[int, Dict[str, Any]]:
    """Aggregate effectiveness per composition depth (k_value)."""
    baseline = [r for r in records if r.condition == "baseline"]
    curated = [r for r in records if r.condition == "curated"]

    curated_by_k = _group_records(curated, lambda r: str(r.k_value))

    results = {}
    for k_str, k_records in sorted(curated_by_k.items()):
        task_ids = set(r.problem_id for r in k_records)
        matching_baseline = [r for r in baseline if r.problem_id in task_ids]
        if matching_baseline:
            results[int(k_str)] = compute_pass_rate_delta(matching_baseline, k_records)
        else:
            results[int(k_str)] = {
                "delta_pp": 0.0,
                "n_baseline": 0,
                "n_treatment": len(k_records),
            }

    return results


def aggregate_by_composition_type(records: List[BenchmarkRecord]) -> Dict[str, Dict[str, Any]]:
    """Aggregate effectiveness per composition type."""
    baseline = [r for r in records if r.condition == "baseline"]
    curated = [r for r in records if r.condition == "curated"]

    curated_by_type = _group_records(curated, lambda r: r.composition_type)

    results = {}
    for comp_type, type_records in sorted(curated_by_type.items()):
        if not comp_type:
            continue
        task_ids = set(r.problem_id for r in type_records)
        matching_baseline = [r for r in baseline if r.problem_id in task_ids]
        if matching_baseline:
            results[comp_type] = compute_pass_rate_delta(matching_baseline, type_records)
        else:
            results[comp_type] = {
                "delta_pp": 0.0,
                "n_baseline": 0,
                "n_treatment": len(type_records),
            }

    return results


def compute_overall_summary(records: List[BenchmarkRecord]) -> Dict[str, Any]:
    """Compute overall experiment summary.

    Args:
        records: All benchmark records

    Returns:
        Summary dict with total_trials, curated_delta_pp, self_generated_delta_pp, p_values
    """
    baseline = [r for r in records if r.condition == "baseline"]
    curated = [r for r in records if r.condition == "curated"]
    self_generated = [r for r in records if r.condition == "self_generated"]

    summary = {"total_trials": len(records)}

    if baseline and curated:
        curated_result = compute_pass_rate_delta(baseline, curated)
        summary["curated_delta_pp"] = curated_result["delta_pp"]
        summary["curated_p_value"] = curated_result["p_value"]
    else:
        summary["curated_delta_pp"] = 0.0
        summary["curated_p_value"] = 1.0

    if baseline and self_generated:
        self_gen_result = compute_pass_rate_delta(baseline, self_generated)
        summary["self_generated_delta_pp"] = self_gen_result["delta_pp"]
        summary["self_generated_p_value"] = self_gen_result["p_value"]
    else:
        summary["self_generated_delta_pp"] = 0.0
        summary["self_generated_p_value"] = 1.0

    return summary
