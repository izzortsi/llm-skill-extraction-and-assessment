"""
extracted_skill.py

Shared data types and I/O for extracted skills.

ExtractedSkill is the data contract between the task-skill-extraction-pipeline
(which produces skills) and the evaluation pipelines (which consume them).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ExtractedSkill:
    """A reusable procedural skill extracted from reasoning traces."""

    skill_uid: str
    name: str
    description: str
    procedure: List[str]
    when_to_use: str
    constraints: List[str]
    source_task_uids: List[str]
    source_trace_uids: List[str] = field(default_factory=list)
    extraction_method: str = ""


def save_extracted_skills(skills: List[ExtractedSkill], output_path: Path) -> None:
    """Save extracted skills to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for skill in skills:
        data.append({
            "skill_uid": skill.skill_uid,
            "name": skill.name,
            "description": skill.description,
            "procedure": skill.procedure,
            "when_to_use": skill.when_to_use,
            "constraints": skill.constraints,
            "source_task_uids": skill.source_task_uids,
            "source_trace_uids": skill.source_trace_uids,
            "extraction_method": skill.extraction_method,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_extracted_skills(filepath: Path) -> List[ExtractedSkill]:
    """Load extracted skills from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    skills = []
    for entry in data:
        skill_uid = entry.get("skill_uid", entry.get("skill_id", ""))
        source_task_uids = entry.get("source_task_uids", entry.get("source_task_ids", []))
        source_trace_uids = entry.get("source_trace_uids", entry.get("source_trace_ids", []))
        skills.append(ExtractedSkill(
            skill_uid=skill_uid,
            name=entry["name"],
            description=entry.get("description", ""),
            procedure=entry.get("procedure", []),
            when_to_use=entry.get("when_to_use", ""),
            constraints=entry.get("constraints", []),
            source_task_uids=source_task_uids,
            source_trace_uids=source_trace_uids,
            extraction_method=entry.get("extraction_method", ""),
        ))
    return skills
