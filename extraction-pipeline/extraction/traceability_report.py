"""
traceability_report.py

Generate a traceability report showing the full provenance chain:
    source_document_uid -> passage_uid -> task_uid -> skill_uid

Identifies orphaned tasks (no source document link) and orphaned skills
(no source task link).

Usage:
    python -m c2_skill_extraction.traceability_report \
        --tasks tasks.json --skills verified_skills.json \
        --passages passages.json -o traceability-report.txt

    python -m c2_skill_extraction.traceability_report \
        --tasks tasks.json --skills verified_skills.json \
        -o traceability-report.txt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class TraceabilityRecord:
    """One row in the traceability chain."""

    source_document_uid: str
    source_artifact: str
    passage_uid: str
    task_uid: str
    task_title: str
    skill_uid: str
    skill_name: str


def generate_traceability_report(
    tasks_path: Path,
    skills_path: Path,
    passages_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> str:
    """Generate a traceability report from pipeline output files.

    Args:
        tasks_path: Path to tasks JSON (stage 1 output).
        skills_path: Path to verified skills JSON (stage 4 output).
        passages_path: Optional path to passages JSON (from passage_extractor).
        output_path: Optional path to write report file.
        verbose: Print progress.

    Returns:
        Report text as a string.
    """
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks_data = json.load(f)

    with open(skills_path, "r", encoding="utf-8") as f:
        skills_data = json.load(f)

    passages_data = []
    if passages_path and passages_path.exists():
        with open(passages_path, "r", encoding="utf-8") as f:
            passages_data = json.load(f)

    # Build task lookup
    task_by_uid: Dict[str, dict] = {}
    for task in tasks_data:
        uid = task.get("task_uid", task.get("task_id", ""))
        task_by_uid[uid] = task

    # Build passage lookup
    passage_by_uid: Dict[str, dict] = {}
    for passage in passages_data:
        uid = passage.get("passage_uid", "")
        passage_by_uid[uid] = passage

    # Build skill -> task mapping
    skill_to_tasks: Dict[str, List[str]] = {}
    for skill in skills_data:
        skill_uid = skill.get("skill_uid", skill.get("skill_id", ""))
        source_uids = skill.get("source_task_uids", skill.get("source_task_ids", []))
        skill_to_tasks[skill_uid] = source_uids

    # Build traceability records
    records: List[TraceabilityRecord] = []

    for skill in skills_data:
        skill_uid = skill.get("skill_uid", skill.get("skill_id", ""))
        skill_name = skill.get("name", "")
        source_task_uids = skill.get("source_task_uids", skill.get("source_task_ids", []))

        for task_uid in source_task_uids:
            task = task_by_uid.get(task_uid, {})
            source_doc_uid = task.get("source_document_uid", "")
            source_artifact = task.get("source_artifact", "")

            records.append(TraceabilityRecord(
                source_document_uid=source_doc_uid,
                source_artifact=source_artifact,
                passage_uid="",
                task_uid=task_uid,
                task_title=task.get("title", ""),
                skill_uid=skill_uid,
                skill_name=skill_name,
            ))

    # Identify orphaned tasks (no source_document_uid)
    orphaned_tasks: List[str] = []
    for uid, task in task_by_uid.items():
        if not task.get("source_document_uid", ""):
            orphaned_tasks.append(uid)

    # Identify orphaned skills (no source task link)
    orphaned_skills: List[str] = []
    for skill in skills_data:
        skill_uid = skill.get("skill_uid", skill.get("skill_id", ""))
        source_uids = skill.get("source_task_uids", skill.get("source_task_ids", []))
        if not source_uids:
            orphaned_skills.append(skill_uid)

    # Identify tasks not linked to any skill
    tasks_with_skills: Set[str] = set()
    for source_uids in skill_to_tasks.values():
        tasks_with_skills.update(source_uids)
    unlinked_tasks = [uid for uid in task_by_uid if uid not in tasks_with_skills]

    # Build unique documents
    documents: Dict[str, str] = {}
    for task in tasks_data:
        doc_uid = task.get("source_document_uid", "")
        artifact = task.get("source_artifact", "")
        if doc_uid:
            documents[doc_uid] = artifact

    # Generate report text
    lines = []
    lines.append("=" * 80)
    lines.append("TRACEABILITY REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Tasks file: {tasks_path}")
    lines.append(f"Skills file: {skills_path}")
    if passages_path:
        lines.append(f"Passages file: {passages_path}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"Source documents:      {len(documents)}")
    lines.append(f"Passages:              {len(passages_data)}")
    lines.append(f"Tasks:                 {len(tasks_data)}")
    lines.append(f"Skills:                {len(skills_data)}")
    lines.append(f"Traceability records:  {len(records)}")
    lines.append(f"Orphaned tasks:        {len(orphaned_tasks)} of {len(tasks_data)} (no source_document_uid)")
    lines.append(f"Orphaned skills:       {len(orphaned_skills)} of {len(skills_data)} (no source_task_uids)")
    lines.append(f"Unlinked tasks:        {len(unlinked_tasks)} of {len(tasks_data)} (not referenced by any skill)")
    lines.append("")

    lines.append("-" * 80)
    lines.append("SOURCE DOCUMENTS")
    lines.append("-" * 80)
    lines.append("")
    for doc_uid, artifact in sorted(documents.items()):
        task_count = sum(1 for t in tasks_data if t.get("source_document_uid") == doc_uid)
        lines.append(f"  {doc_uid}  {artifact}  ({task_count} tasks)")
    lines.append("")

    lines.append("-" * 80)
    lines.append("TRACEABILITY CHAIN")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"  {'Document UID':<22} {'Task UID':<22} {'Skill UID':<22} {'Skill Name'}")
    lines.append(f"  {'-'*20}   {'-'*20}   {'-'*20}   {'-'*20}")
    for r in records:
        doc_short = r.source_document_uid[:19] if r.source_document_uid else "(none)"
        task_short = r.task_uid[:19] if r.task_uid else "(none)"
        skill_short = r.skill_uid[:19] if r.skill_uid else "(none)"
        lines.append(f"  {doc_short:<22} {task_short:<22} {skill_short:<22} {r.skill_name}")
    lines.append("")

    if orphaned_tasks:
        lines.append("-" * 80)
        lines.append(f"ORPHANED TASKS ({len(orphaned_tasks)})")
        lines.append("-" * 80)
        lines.append("")
        for uid in orphaned_tasks:
            task = task_by_uid[uid]
            lines.append(f"  {uid}  {task.get('title', '')}")
        lines.append("")

    if orphaned_skills:
        lines.append("-" * 80)
        lines.append(f"ORPHANED SKILLS ({len(orphaned_skills)} of {len(skills_data)})")
        lines.append("-" * 80)
        lines.append("")
        for uid in orphaned_skills:
            skill = next((s for s in skills_data
                          if s.get("skill_uid", s.get("skill_id")) == uid), {})
            lines.append(f"  {uid}  {skill.get('name', '')}")
        lines.append("")

    if unlinked_tasks:
        lines.append("-" * 80)
        lines.append(f"UNLINKED TASKS ({len(unlinked_tasks)} of {len(tasks_data)} - not referenced by any skill)")
        lines.append("-" * 80)
        lines.append("")
        for uid in unlinked_tasks:
            task = task_by_uid[uid]
            lines.append(f"  {uid}  {task.get('title', '')}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_text = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")
        if verbose:
            print(f"Traceability report written to {output_path}")

    return report_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate traceability report for pipeline output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tasks", "-t", type=Path, required=True,
                        help="Input tasks JSON")
    parser.add_argument("--skills", "-s", type=Path, required=True,
                        help="Input skills or verified_skills JSON")
    parser.add_argument("--passages", "-p", type=Path, default=None,
                        help="Input passages JSON (optional)")
    parser.add_argument("--output", "-o", type=Path, default=Path("traceability-report.txt"),
                        help="Output report path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    report = generate_traceability_report(
        tasks_path=args.tasks,
        skills_path=args.skills,
        passages_path=args.passages,
        output_path=args.output,
        verbose=args.verbose,
    )

    print(report)


if __name__ == "__main__":
    main()
