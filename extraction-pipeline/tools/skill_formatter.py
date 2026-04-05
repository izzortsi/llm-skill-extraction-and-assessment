"""
skill_formatter.py

Bidirectional converter between JSON pipeline format and markdown
skill-creator template format for both tasks and skills.

JSON format: used by the extraction pipeline (dataclass-based I/O)
Markdown format: used by the skill registry and composition engine

Functions:
    skill_to_markdown     -- ExtractedSkill -> .md string
    markdown_to_skill     -- .md string -> ExtractedSkill
    task_to_markdown      -- ExtractedTask -> .md string
    markdown_to_task      -- .md string -> ExtractedTask
    skills_json_to_dir    -- skills.json -> directory of .md files
    skills_dir_to_json    -- directory of .md files -> skills.json
    tasks_json_to_dir     -- tasks.json -> directory of .md files
    tasks_dir_to_json     -- directory of .md files -> tasks.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from schemas.extracted_skill import ExtractedSkill, load_extracted_skills, save_extracted_skills
from schemas.extracted_task import ExtractedTask, load_extracted_tasks, save_extracted_tasks


# ---------------------------------------------------------------------------
# Skill: JSON <-> Markdown
# ---------------------------------------------------------------------------

def skill_to_markdown(skill: ExtractedSkill) -> str:
    """Convert an ExtractedSkill to markdown skill-creator template format.

    Output format matches what skill_registry.parse_skill_file() expects:
        ---
        name: kebab-case-name
        description: "Use when ... one-line description"
        ---
        ## When to Use
        - bullet point
        ## Procedure
        1. step one
        2. step two
        ## Constraints
        - constraint one
    """
    frontmatter = {
        "name": skill.name,
        "description": f"Use when {skill.when_to_use}" if not skill.when_to_use.startswith("Use when") else skill.when_to_use,
    }

    # optional metadata fields
    if skill.skill_uid:
        frontmatter["skill_uid"] = skill.skill_uid
    if skill.source_task_uids:
        frontmatter["source_task_uids"] = skill.source_task_uids
    if skill.source_trace_uids:
        frontmatter["source_trace_uids"] = skill.source_trace_uids
    if skill.extraction_method:
        frontmatter["extraction_method"] = skill.extraction_method

    lines = []
    lines.append("---")
    lines.append(yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).rstrip())
    lines.append("---")
    lines.append("")

    # description
    lines.append(f"# {skill.name}")
    lines.append("")
    lines.append(skill.description)
    lines.append("")

    # when to use
    lines.append("## When to Use")
    lines.append("")
    lines.append(f"- {skill.when_to_use}")
    lines.append("")

    # procedure
    lines.append("## Procedure")
    lines.append("")
    for i, step in enumerate(skill.procedure, 1):
        lines.append(f"{i}. {step}")
    lines.append("")

    # constraints
    if skill.constraints:
        lines.append("## Constraints")
        lines.append("")
        for constraint in skill.constraints:
            lines.append(f"- {constraint}")
        lines.append("")

    return "\n".join(lines)


def markdown_to_skill(content: str, source_file: str = "") -> ExtractedSkill:
    """Convert a markdown skill-creator template to an ExtractedSkill.

    Parses YAML frontmatter for name, description, and optional metadata.
    Parses body sections for when_to_use, procedure, and constraints.
    """
    # parse YAML frontmatter
    frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError(f"No YAML frontmatter found in markdown content")

    frontmatter = yaml.safe_load(frontmatter_match.group(1))
    body = content[frontmatter_match.end():]

    name = frontmatter.get("name", "")
    description_raw = frontmatter.get("description", "")

    # extract when_to_use from description or body
    when_to_use = ""
    if description_raw.startswith("Use when "):
        when_to_use = description_raw[len("Use when "):]
    else:
        when_bullets = _extract_bullets(body, "When to Use")
        when_to_use = when_bullets[0] if when_bullets else description_raw

    # extract procedure steps (numbered list)
    procedure = _extract_numbered(body, "Procedure")

    # extract constraints (bullet list)
    constraints = _extract_bullets(body, "Constraints")

    # extract description from body (first paragraph after heading)
    description = description_raw
    desc_match = re.search(r"^# .+\n\n(.+?)(?:\n\n|\n## )", body, re.DOTALL)
    if desc_match:
        description = desc_match.group(1).strip()

    return ExtractedSkill(
        skill_uid=frontmatter.get("skill_uid", ""),
        name=name,
        description=description,
        procedure=procedure,
        when_to_use=when_to_use,
        constraints=constraints,
        source_task_uids=frontmatter.get("source_task_uids", []),
        source_trace_uids=frontmatter.get("source_trace_uids", []),
        extraction_method=frontmatter.get("extraction_method", ""),
    )


# ---------------------------------------------------------------------------
# Task: JSON <-> Markdown
# ---------------------------------------------------------------------------

def task_to_markdown(task: ExtractedTask) -> str:
    """Convert an ExtractedTask to markdown format.

    Output format:
        ---
        task_uid: xxxx-xxxx-xxxx-xxxx
        title: Task Title
        domain: language-skills
        query_type: YES_NO_VERIFICATION
        difficulty: intermediate
        ---
        # Task Title

        ## Question
        The question text...

        ## Passage
        The input passage text...

        ## Expected Output
        The expected answer...

        ## Acceptance Criteria
        ### Must Identify
        - item 1
        - item 2
        ### Correct Conclusion
        The conclusion...
    """
    frontmatter = {
        "task_uid": task.task_uid,
        "title": task.title,
        "domain": task.domain,
        "query_type": task.query_type,
        "difficulty": task.difficulty,
        "source_artifact": task.source_artifact,
        "source_document_uid": task.source_document_uid,
    }
    if task.extraction_method:
        frontmatter["extraction_method"] = task.extraction_method

    lines = []
    lines.append("---")
    lines.append(yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).rstrip())
    lines.append("---")
    lines.append("")
    lines.append(f"# {task.title}")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append(task.question)
    lines.append("")
    lines.append("## Passage")
    lines.append("")
    lines.append(task.input)
    lines.append("")
    lines.append("## Expected Output")
    lines.append("")
    lines.append(task.output)
    lines.append("")

    ac = task.acceptance_criteria
    if ac:
        lines.append("## Acceptance Criteria")
        lines.append("")
        must_identify = ac.get("must_identify", [])
        if must_identify:
            lines.append("### Must Identify")
            lines.append("")
            for item in must_identify:
                lines.append(f"- {item}")
            lines.append("")
        conclusion = ac.get("correct_conclusion", "")
        if conclusion:
            lines.append("### Correct Conclusion")
            lines.append("")
            lines.append(conclusion)
            lines.append("")

    return "\n".join(lines)


def markdown_to_task(content: str) -> ExtractedTask:
    """Convert a markdown task file to an ExtractedTask."""
    frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError("No YAML frontmatter found in task markdown")

    frontmatter = yaml.safe_load(frontmatter_match.group(1))
    body = content[frontmatter_match.end():]

    question = _extract_section_text(body, "Question")
    passage = _extract_section_text(body, "Passage")
    output = _extract_section_text(body, "Expected Output")

    must_identify = _extract_bullets(body, "Must Identify")
    conclusion = _extract_section_text(body, "Correct Conclusion")

    acceptance_criteria = {}
    if must_identify:
        acceptance_criteria["must_identify"] = must_identify
    if conclusion:
        acceptance_criteria["correct_conclusion"] = conclusion

    return ExtractedTask(
        task_uid=frontmatter.get("task_uid", ""),
        title=frontmatter.get("title", ""),
        domain=frontmatter.get("domain", ""),
        source_artifact=frontmatter.get("source_artifact", ""),
        source_document_uid=frontmatter.get("source_document_uid", ""),
        question=question,
        input=passage,
        output=output,
        difficulty=frontmatter.get("difficulty", ""),
        acceptance_criteria=acceptance_criteria,
        query_type=frontmatter.get("query_type", "FREE_FORM"),
        extraction_method=frontmatter.get("extraction_method", ""),
    )


# ---------------------------------------------------------------------------
# Batch: JSON file <-> directory of .md files
# ---------------------------------------------------------------------------

def skills_json_to_dir(json_path: Path, output_dir: Path) -> int:
    """Convert a skills JSON file to a directory of markdown files.

    Each skill becomes {skill-name}.md in the output directory.
    Returns the number of files written.
    """
    skills = load_extracted_skills(json_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for skill in skills:
        filename = f"{skill.name}.md"
        filepath = output_dir / filename
        md_content = skill_to_markdown(skill)
        filepath.write_text(md_content, encoding="utf-8")
        count += 1

    return count


def skills_dir_to_json(input_dir: Path, output_path: Path) -> int:
    """Convert a directory of markdown skill files to a JSON file.

    Returns the number of skills converted.
    """
    skills = []
    for md_file in sorted(input_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        skill = markdown_to_skill(content, source_file=str(md_file))
        skills.append(skill)

    save_extracted_skills(skills, output_path)
    return len(skills)


def tasks_json_to_dir(json_path: Path, output_dir: Path) -> int:
    """Convert a tasks JSON file to a directory of markdown files.

    Each task becomes {task-uid}.md in the output directory.
    Returns the number of files written.
    """
    tasks = load_extracted_tasks(json_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for task in tasks:
        filename = f"{task.task_uid}.md"
        filepath = output_dir / filename
        md_content = task_to_markdown(task)
        filepath.write_text(md_content, encoding="utf-8")
        count += 1

    return count


def tasks_dir_to_json(input_dir: Path, output_path: Path) -> int:
    """Convert a directory of markdown task files to a JSON file.

    Returns the number of tasks converted.
    """
    tasks = []
    for md_file in sorted(input_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        task = markdown_to_task(content)
        tasks.append(task)

    save_extracted_tasks(tasks, output_path)
    return len(tasks)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_bullets(content: str, section_name: str) -> List[str]:
    """Extract bullet list items from a markdown section."""
    pattern = rf"##+ {re.escape(section_name)}\n(.*?)(?=\n##+ |\n---|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    return re.findall(r"^[\s]*[-*]\s+(.+)$", match.group(1), re.MULTILINE)


def _extract_numbered(content: str, section_name: str) -> List[str]:
    """Extract numbered list items from a markdown section."""
    pattern = rf"## {re.escape(section_name)}\n(.*?)(?=\n## |\n---|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    return re.findall(r"^\s*\d+\.\s+(.+)$", match.group(1), re.MULTILINE)


def _extract_section_text(content: str, section_name: str) -> str:
    """Extract the text content of a markdown section (non-list)."""
    pattern = rf"##+ {re.escape(section_name)}\n\n(.*?)(?=\n##+ |\n---|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI for format conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert between JSON and markdown formats for tasks and skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Skills JSON -> markdown directory
  python -m tools.skill_formatter skills-to-md --input skills.json --output-dir skills-md/

  # Skills markdown directory -> JSON
  python -m tools.skill_formatter skills-to-json --input-dir skills-md/ --output skills.json

  # Tasks JSON -> markdown directory
  python -m tools.skill_formatter tasks-to-md --input tasks.json --output-dir tasks-md/

  # Tasks markdown directory -> JSON
  python -m tools.skill_formatter tasks-to-json --input-dir tasks-md/ --output tasks.json
""",
    )
    parser.add_argument("command", choices=["skills-to-md", "skills-to-json", "tasks-to-md", "tasks-to-json"])
    parser.add_argument("--input", type=Path, default=None, help="Input JSON file")
    parser.add_argument("--input-dir", type=Path, default=None, help="Input markdown directory")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output markdown directory")

    args = parser.parse_args()

    if args.command == "skills-to-md":
        count = skills_json_to_dir(args.input, args.output_dir)
        print(f"Converted {count} skills to {args.output_dir}/")

    elif args.command == "skills-to-json":
        count = skills_dir_to_json(args.input_dir, args.output)
        print(f"Converted {count} skills to {args.output}")

    elif args.command == "tasks-to-md":
        count = tasks_json_to_dir(args.input, args.output_dir)
        print(f"Converted {count} tasks to {args.output_dir}/")

    elif args.command == "tasks-to-json":
        count = tasks_dir_to_json(args.input_dir, args.output)
        print(f"Converted {count} tasks to {args.output}")


if __name__ == "__main__":
    main()
