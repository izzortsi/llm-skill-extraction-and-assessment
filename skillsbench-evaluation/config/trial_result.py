"""
trial_result.py

Trial result and benchmark record data structures.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TrialResult:
    """Result of a single trial execution."""

    # Identification
    trial_key: str
    task_uid: str
    domain: str
    model: str
    provider: str
    condition: str              # baseline, curated, self_generated
    skill_name: str
    k_value: int
    composition_type: str
    repetition_index: int

    # Outcome
    passed: bool
    score: float                # 0.0 to 1.0
    episode_id: str

    # Cost
    total_tokens: int
    elapsed_s: float
    steps: int

    # Optional metadata
    verification: Dict[str, Any] = field(default_factory=dict)
    error: str = ""             # Non-empty if trial failed with error


@dataclass
class BenchmarkRecord:
    """Single record for progress JSONL output.

    Flat structure optimized for streaming write and analysis.
    """

    problem_id: str
    domain: str
    model: str
    provider: str
    condition: str
    skill_name: str
    k_value: int
    composition_type: str
    repetition: int
    passed: bool               # renamed from 'pass' to avoid keyword conflict
    score: float
    tokens: int
    elapsed_s: float
    steps: int
    episode_id: str
    error: str = ""

    def to_json_line(self) -> str:
        """Serialize to a single JSON line."""
        d = asdict(self)
        # Rename 'passed' to 'pass' in output for compatibility with plan spec
        d["pass"] = d.pop("passed")
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_trial_result(cls, result: TrialResult) -> "BenchmarkRecord":
        """Create BenchmarkRecord from TrialResult."""
        return cls(
            problem_id=result.task_uid,
            domain=result.domain,
            model=result.model,
            provider=result.provider,
            condition=result.condition,
            skill_name=result.skill_name,
            k_value=result.k_value,
            composition_type=result.composition_type,
            repetition=result.repetition_index,
            passed=result.passed,
            score=result.score,
            tokens=result.total_tokens,
            elapsed_s=result.elapsed_s,
            steps=result.steps,
            episode_id=result.episode_id,
            error=result.error,
        )


def write_progress_record(filepath: Path, record: BenchmarkRecord) -> None:
    """Append a single record to progress JSONL file.

    Args:
        filepath: Path to progress.jsonl
        record: Record to append
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(record.to_json_line() + "\n")
        f.flush()


def load_progress_records(filepath: Path) -> List[BenchmarkRecord]:
    """Load all records from progress JSONL file.

    Args:
        filepath: Path to progress.jsonl

    Returns:
        List of BenchmarkRecord objects
    """
    records = []
    if not filepath.exists():
        return records

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Rename 'pass' back to 'passed'
            if "pass" in d:
                d["passed"] = d.pop("pass")
            records.append(BenchmarkRecord(**d))

    return records
