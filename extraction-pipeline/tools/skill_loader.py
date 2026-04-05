"""
skill_loader.py

Load atomic and composed skills from markdown files.
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from tools.skill_registry import Skill, SkillRegistry, parse_skill_file


@dataclass
class ComposedSkill:
    """A composed skill loaded from experiment output."""

    name: str
    description: str
    k_value: int
    composition_type: str
    source_skills: List[str]
    raw_content: str
    source_file: Path

    procedure: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    when_to_use: List[str] = field(default_factory=list)


def _parse_composed_skill_file(filepath: Path, k_value: int, composition_type: str) -> ComposedSkill:
    """Parse a composed skill markdown file."""
    content = filepath.read_text(encoding="utf-8")

    frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        try:
            frontmatter = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError:
            frontmatter = _parse_frontmatter_manual(frontmatter_text)
        if not isinstance(frontmatter, dict):
            frontmatter = {}
        name = frontmatter.get("name", filepath.stem)
        description = frontmatter.get("description", "")
        source_skills = frontmatter.get("source_skills", [])
        if not source_skills:
            source_skills = frontmatter.get("component_skills", [])
        body = content[frontmatter_match.end():]
    else:
        name = filepath.stem
        description = ""
        source_skills = []
        body = content

    if not source_skills:
        source_skills = _infer_source_skills(name, composition_type)

    procedure = _extract_numbered_section(body, "Procedure")
    constraints = _extract_bullet_section(body, "Constraints")
    when_to_use = _extract_bullet_section(body, "When to Use")

    return ComposedSkill(
        name=name,
        description=description,
        k_value=k_value,
        composition_type=composition_type,
        source_skills=source_skills,
        raw_content=content,
        source_file=filepath,
        procedure=procedure,
        constraints=constraints,
        when_to_use=when_to_use,
    )


def _parse_frontmatter_manual(text: str) -> Dict:
    """Manually parse YAML-like frontmatter when yaml.safe_load fails."""
    result = {}
    for line in text.split("\n"):
        colon_idx = line.find(":")
        if colon_idx > 0:
            key = line[:colon_idx].strip()
            value = line[colon_idx + 1:].strip()
            if key in ("name", "description"):
                result[key] = value
            elif key in ("source_skills", "component_skills"):
                if value.startswith("["):
                    try:
                        result[key] = yaml.safe_load(value)
                    except yaml.YAMLError:
                        result[key] = []
    return result


def _infer_source_skills(composed_name: str, composition_type: str) -> List[str]:
    """Infer source skill names from composed skill filename."""
    remainder = composed_name
    if remainder.startswith(composition_type + "-"):
        remainder = remainder[len(composition_type) + 1:]

    connectors = ["-then-", "-and-", "-if-", "-with-", "-or-"]
    for connector in connectors:
        if connector in remainder:
            parts = remainder.split(connector)
            return [p.strip() for p in parts if p.strip()]

    return [remainder]


def _extract_numbered_section(content: str, section_name: str) -> List[str]:
    """Extract numbered steps from a markdown section."""
    pattern = rf"## {re.escape(section_name)}\n(.*?)(?=\n## |\n---|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    section = match.group(1)
    steps = re.findall(r"^\d+\.\s+(.+)$", section, re.MULTILINE)
    return steps


def _extract_bullet_section(content: str, section_name: str) -> List[str]:
    """Extract bullet items from a markdown section."""
    pattern = rf"## {re.escape(section_name)}\n(.*?)(?=\n## |\n---|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []
    section = match.group(1)
    bullets = re.findall(r"^[\s]*[-*]\s+(.+)$", section, re.MULTILINE)
    return bullets


@dataclass
class SkillLoader:
    """Unified loader for atomic and composed skills."""

    atomic_skills: Dict[str, Skill] = field(default_factory=dict)
    composed_skills: Dict[str, ComposedSkill] = field(default_factory=dict)
    _registry: Optional[SkillRegistry] = field(default=None, repr=False)

    @classmethod
    def from_directories(
        cls,
        atomic_dir: Path,
        composed_dir: Optional[Path] = None,
        k_values: Optional[List[int]] = None,
        composition_types: Optional[List[str]] = None,
    ) -> "SkillLoader":
        """Load skills from directories."""
        loader = cls()

        registry = SkillRegistry.from_directory(atomic_dir)
        loader._registry = registry
        loader.atomic_skills = dict(registry.skills)

        if composed_dir is not None and composed_dir.exists():
            available_k = sorted([
                int(d.name[1:])
                for d in composed_dir.iterdir()
                if d.is_dir() and d.name.startswith("k")
            ])

            target_k = k_values if k_values else available_k

            for k in target_k:
                k_dir = composed_dir / f"k{k}"
                if not k_dir.exists():
                    continue

                available_types = sorted([d.name for d in k_dir.iterdir() if d.is_dir()])
                target_types = composition_types if composition_types else available_types

                for comp_type in target_types:
                    type_dir = k_dir / comp_type
                    if not type_dir.exists():
                        continue

                    for skill_file in sorted(type_dir.glob("*.md")):
                        composed = _parse_composed_skill_file(skill_file, k, comp_type)
                        loader.composed_skills[composed.name] = composed

        return loader

    def get_skill_content(self, skill_name: str) -> Optional[str]:
        """Get raw content for any skill (atomic or composed)."""
        if skill_name in self.atomic_skills:
            return self.atomic_skills[skill_name].raw_content
        if skill_name in self.composed_skills:
            return self.composed_skills[skill_name].raw_content
        return None

    def get_skill_description(self, skill_name: str) -> Optional[str]:
        """Get description for any skill."""
        if skill_name in self.atomic_skills:
            return self.atomic_skills[skill_name].description
        if skill_name in self.composed_skills:
            return self.composed_skills[skill_name].description
        return None

    def list_all_skills(self) -> List[str]:
        """List all skill names (atomic + composed)."""
        return sorted(list(self.atomic_skills.keys()) + list(self.composed_skills.keys()))

    def find_skills_for_task(self, required_skills: List[str]) -> List[str]:
        """Find composed skills whose source skills cover the required skills."""
        required_set = set(required_skills)
        matches = []

        for name, skill in self.composed_skills.items():
            source_set = set(skill.source_skills)
            if required_set.issubset(source_set):
                matches.append(name)

        return sorted(matches)

    def get_atomic_count(self) -> int:
        return len(self.atomic_skills)

    def get_composed_count(self) -> int:
        return len(self.composed_skills)
