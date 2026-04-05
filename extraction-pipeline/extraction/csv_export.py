"""
csv_export.py

Generate CSV output artifacts alongside JSON pipeline output.

Four CSV files per standard-pipeline-csv-schema.txt:
    source_documents.csv  - registry of ingested source documents
    passages.csv          - extracted passages linked to source documents
    skills.csv            - extracted skills with instance counts
    skill_instances.csv   - skill-to-task mapping

Usage:
    python -m c2_skill_extraction.csv_export \
        --tasks tasks.json --skills verified_skills.json \
        --passages passages.json -o stage1-task-extraction/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def export_source_documents_csv(
    tasks_data: List[Dict[str, Any]],
    output_path: Path,
) -> int:
    """Generate source_documents.csv from tasks data.

    Columns: document_uid, source_type, source_name

    Args:
        tasks_data: List of task dicts (from tasks.json).
        output_path: Path to output CSV file.

    Returns:
        Number of unique documents written.
    """
    documents: Dict[str, Dict[str, str]] = {}
    for task in tasks_data:
        doc_uid = task.get("source_document_uid", "")
        if not doc_uid or doc_uid in documents:
            continue
        source_artifact = task.get("source_artifact", "")
        if source_artifact.startswith("dataset:"):
            source_type = "dataset"
        elif source_artifact.endswith(".pdf"):
            source_type = "pdf"
        else:
            source_type = "file"
        documents[doc_uid] = {
            "document_uid": doc_uid,
            "source_type": source_type,
            "source_name": source_artifact,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["document_uid", "source_type", "source_name"])
        writer.writeheader()
        for doc in sorted(documents.values(), key=lambda d: d["document_uid"]):
            writer.writerow(doc)

    return len(documents)


def export_passages_csv(
    passages_data: List[Dict[str, Any]],
    output_path: Path,
) -> int:
    """Generate passages.csv from passages data.

    Columns: passage_uid, document_uid, text, chunk_index, total_chunks

    Args:
        passages_data: List of passage dicts (from passages.json).
        output_path: Path to output CSV file.

    Returns:
        Number of passages written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["passage_uid", "document_uid", "text", "chunk_index", "total_chunks"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in passages_data:
            writer.writerow({
                "passage_uid": p.get("passage_uid", ""),
                "document_uid": p.get("source_document_uid", ""),
                "text": p.get("text", "")[:500],
                "chunk_index": p.get("chunk_index", 0),
                "total_chunks": p.get("total_chunks", 1),
            })

    return len(passages_data)


def export_skills_csv(
    skills_data: List[Dict[str, Any]],
    output_path: Path,
) -> int:
    """Generate skills.csv from skills data.

    Columns: skill_uid, name, description, procedure_steps, instance_count

    Args:
        skills_data: List of skill dicts (from skills.json or verified_skills.json).
        output_path: Path to output CSV file.

    Returns:
        Number of skills written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["skill_uid", "name", "description", "procedure_steps", "instance_count"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for skill in skills_data:
            skill_uid = skill.get("skill_uid", skill.get("skill_id", ""))
            source_uids = skill.get("source_task_uids", skill.get("source_task_ids", []))
            procedure = skill.get("procedure", [])
            writer.writerow({
                "skill_uid": skill_uid,
                "name": skill.get("name", ""),
                "description": skill.get("description", ""),
                "procedure_steps": len(procedure),
                "instance_count": len(source_uids),
            })

    return len(skills_data)


def export_skill_instances_csv(
    skills_data: List[Dict[str, Any]],
    output_path: Path,
) -> int:
    """Generate skill_instances.csv from skills data.

    Columns: skill_uid, task_uid

    One row per (skill, task) pair.

    Args:
        skills_data: List of skill dicts.
        output_path: Path to output CSV file.

    Returns:
        Number of instance rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["skill_uid", "task_uid"]

    row_count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for skill in skills_data:
            skill_uid = skill.get("skill_uid", skill.get("skill_id", ""))
            source_uids = skill.get("source_task_uids", skill.get("source_task_ids", []))
            for task_uid in source_uids:
                writer.writerow({"skill_uid": skill_uid, "task_uid": task_uid})
                row_count += 1

    return row_count


def export_all_csvs(
    tasks_path: Path,
    skills_path: Path,
    output_dir: Path,
    passages_path: Path = None,
    verbose: bool = False,
) -> None:
    """Generate all 4 CSV artifacts from pipeline JSON output.

    Args:
        tasks_path: Path to tasks JSON.
        skills_path: Path to skills or verified_skills JSON.
        output_dir: Directory to write CSV files.
        passages_path: Optional path to passages JSON.
        verbose: Print progress.
    """
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks_data = json.load(f)

    with open(skills_path, "r", encoding="utf-8") as f:
        skills_data = json.load(f)

    passages_data = []
    if passages_path and passages_path.exists():
        with open(passages_path, "r", encoding="utf-8") as f:
            passages_data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    n_docs = export_source_documents_csv(tasks_data, output_dir / "source_documents.csv")
    if verbose:
        print(f"source_documents.csv: {n_docs} documents")

    if passages_data:
        n_passages = export_passages_csv(passages_data, output_dir / "passages.csv")
        if verbose:
            print(f"passages.csv: {n_passages} passages")

    n_skills = export_skills_csv(skills_data, output_dir / "skills.csv")
    if verbose:
        print(f"skills.csv: {n_skills} skills")

    n_instances = export_skill_instances_csv(skills_data, output_dir / "skill_instances.csv")
    if verbose:
        print(f"skill_instances.csv: {n_instances} skill-task pairs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CSV output artifacts from pipeline JSON files",
    )
    parser.add_argument("--tasks", "-t", type=Path, required=True, help="Input tasks JSON")
    parser.add_argument("--skills", "-s", type=Path, required=True, help="Input skills JSON")
    parser.add_argument("--passages", "-p", type=Path, default=None, help="Input passages JSON (optional)")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("stage1-task-extraction"),
                        help="Output directory for CSV files")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    export_all_csvs(
        tasks_path=args.tasks,
        skills_path=args.skills,
        output_dir=args.output_dir,
        passages_path=args.passages,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
