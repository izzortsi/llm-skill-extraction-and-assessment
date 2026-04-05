"""
task_loader.py

Load tasks from verification_tasks.json and grpt problem JSONs into a unified format.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class BenchTask:
    """Unified task format for benchmarking."""

    task_uid: str
    title: str
    domain: str
    difficulty: str
    prompt: str                              # The challenge/problem text
    context: str                             # Passage or setup context
    required_skills: List[str]               # Skill names needed
    acceptance_criteria: Dict[str, Any]      # Structured criteria for verification
    k_target: int = 0                        # Target skill composition depth
    source: str = ""                         # Source file path
    test_command: str = ""                   # For coding tasks: test suite command
    setup_commands: List[str] = field(default_factory=list)  # Setup commands
    system_prompt: str = ""                  # Optional system prompt override


def load_verification_tasks(filepath: Path) -> List[BenchTask]:
    """Load tasks from skillmix extraction verification_tasks.json.

    Args:
        filepath: Path to verification_tasks.json

    Returns:
        List of BenchTask objects
    """
    with open(filepath, "r", encoding="utf-8") as f:
        tasks_data = json.load(f)

    tasks = []
    for entry in tasks_data:
        # Build prompt from passage + challenge
        prompt = f"Passage: {entry['passage']}\n\nChallenge: {entry['challenge']}"

        task = BenchTask(
            task_uid=entry["id"],
            title=entry["title"],
            domain="reading_comprehension",
            difficulty=entry.get("difficulty", "unknown"),
            prompt=prompt,
            context=entry["passage"],
            required_skills=entry.get("required_skills", []),
            acceptance_criteria=entry.get("acceptance_criteria", {}),
            k_target=entry.get("k_target", 0),
            source=str(filepath),
        )
        tasks.append(task)

    return tasks


def load_problem_json(filepath: Path) -> BenchTask:
    """Load a single problem from grpt-training-data-pipeline JSON format.

    Args:
        filepath: Path to problem JSON file

    Returns:
        BenchTask object
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return BenchTask(
        task_uid=data.get("problem_id", filepath.stem),
        title=data.get("problem_id", filepath.stem),
        domain=data.get("domain", "coding"),
        difficulty=data.get("difficulty", "unknown"),
        prompt=data.get("prompt", ""),
        context="",
        required_skills=[],
        acceptance_criteria={},
        source=str(filepath),
        test_command=data.get("test_command", ""),
        setup_commands=data.get("setup_commands", []),
        system_prompt=data.get("system_prompt", ""),
    )


def load_problem_directory(directory: Path) -> List[BenchTask]:
    """Load all problem JSONs from a directory.

    Args:
        directory: Directory containing problem JSON files

    Returns:
        List of BenchTask objects
    """
    tasks = []
    for json_file in sorted(directory.glob("*.json")):
        try:
            task = load_problem_json(json_file)
            tasks.append(task)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: skipping {json_file}: {e}")
    return tasks


@dataclass
class TaskLoader:
    """Unified task loader from multiple sources."""

    tasks: Dict[str, BenchTask] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        verification_tasks_path: Optional[Path] = None,
        problem_dirs: Optional[List[Path]] = None,
    ) -> "TaskLoader":
        """Load tasks from configured sources.

        Args:
            verification_tasks_path: Path to verification_tasks.json
            problem_dirs: List of directories with problem JSONs

        Returns:
            TaskLoader with all loaded tasks
        """
        loader = cls()

        if verification_tasks_path is not None and verification_tasks_path.exists():
            for task in load_verification_tasks(verification_tasks_path):
                loader.tasks[task.task_uid] = task

        if problem_dirs:
            for directory in problem_dirs:
                if directory.exists():
                    for task in load_problem_directory(directory):
                        loader.tasks[task.task_uid] = task

        return loader

    def get_task(self, task_uid: str) -> Optional[BenchTask]:
        """Get task by UID."""
        return self.tasks.get(task_uid)

    def list_task_uids(self) -> List[str]:
        """List all task UIDs."""
        return sorted(self.tasks.keys())

    def filter_by_domain(self, domain: str) -> List[BenchTask]:
        """Get tasks filtered by domain."""
        return [t for t in self.tasks.values() if t.domain == domain]

    def filter_by_difficulty(self, difficulty: str) -> List[BenchTask]:
        """Get tasks filtered by difficulty."""
        return [t for t in self.tasks.values() if t.difficulty == difficulty]

    def filter_by_required_skills(self, skill_names: List[str]) -> List[BenchTask]:
        """Get tasks that require any of the given skills."""
        skill_set = set(skill_names)
        return [
            t for t in self.tasks.values()
            if skill_set.intersection(set(t.required_skills))
        ]

    @classmethod
    def from_extracted_tasks(cls, filepath: Path) -> "TaskLoader":
        """Load tasks from corpus extraction pipeline output.

        Maps ExtractedTask format (corpus_extractor.py) to BenchTask.

        Args:
            filepath: Path to extracted tasks JSON (from extract_pipeline.py)

        Returns:
            TaskLoader with all loaded tasks
        """
        loader = cls()

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            prompt = f"Passage: {entry['passage']}\n\nChallenge: {entry['challenge']}"
            task = BenchTask(
                task_uid=entry.get("task_uid", entry.get("task_id", "")),
                title=entry["title"],
                domain=entry.get("domain", "reading_comprehension"),
                difficulty=entry.get("difficulty", "intermediate"),
                prompt=prompt,
                context=entry["passage"],
                required_skills=[],
                acceptance_criteria=entry.get("acceptance_criteria", {}),
                source=str(filepath),
            )
            loader.tasks[task.task_uid] = task

        return loader

    def get_task_count(self) -> int:
        """Number of loaded tasks."""
        return len(self.tasks)
