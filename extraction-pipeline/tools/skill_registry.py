"""
skill_registry.py

Parser and registry for atomic skills in skill-creator template format.

Loads skills from sample-extracted-skills/, parses YAML frontmatter and
body sections, and provides access for composition operations.
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import yaml


@dataclass
class Skill:
    """A skill in skill-creator template format."""

    # YAML frontmatter
    name: str
    description: str

    # Body sections
    when_to_use: List[str]
    procedure: List[str]
    constraints: List[str]
    examples: List[Dict[str, str]]
    related_skills: List[str]

    # Source info
    source_file: Path
    raw_content: str

    def __post_init__(self):
        """Validate skill after initialization."""
        if not self.name:
            raise ValueError("Skill name cannot be empty")
        if "Use when" not in self.description:
            raise ValueError(f"Description must include 'Use when': {self.description}")


@dataclass
class SkillRegistry:
    """Registry of atomic skills with dependency graph."""

    skills: Dict[str, Skill] = field(default_factory=dict)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)

    @classmethod
    def from_directory(cls, directory: Path) -> "SkillRegistry":
        """Load all skills from a directory."""
        registry = cls()

        for skill_file in directory.glob("*.md"):
            if skill_file.name == "extracted-skills-inventory.txt":
                continue

            skill = parse_skill_file(skill_file)
            registry.skills[skill.name] = skill

            # Build dependency graph from related_skills
            registry.dependency_graph[skill.name] = set(skill.related_skills)

        return registry

    def get(self, name: str) -> Optional[Skill]:
        """Get skill by name."""
        return self.skills.get(name)

    def list_all(self) -> List[str]:
        """List all skill names."""
        return list(self.skills.keys())

    def get_dependencies(self, name: str, recursive: bool = False) -> Set[str]:
        """Get dependencies for a skill."""
        if name not in self.dependency_graph:
            return set()

        deps = self.dependency_graph[name].copy()

        if recursive:
            for dep in self.dependency_graph[name]:
                deps.update(self.get_dependencies(dep, recursive=True))

        return deps

    def validate_references(self) -> Dict[str, List[str]]:
        """Validate that all related skills exist.

        Returns:
            Dict mapping skill name to list of invalid references.
        """
        invalid_refs = {}

        for name, skill in self.skills.items():
            invalid = []
            for ref in skill.related_skills:
                if ref not in self.skills:
                    invalid.append(ref)

            if invalid:
                invalid_refs[name] = invalid

        return invalid_refs


def parse_skill_file(filepath: Path) -> Skill:
    """Parse a skill file in skill-creator template format.

    Args:
        filepath: Path to .md skill file

    Returns:
        Skill object with parsed content
    """
    content = filepath.read_text(encoding="utf-8")

    # Extract YAML frontmatter
    frontmatter_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError(f"No YAML frontmatter found in {filepath}")

    frontmatter_yaml = frontmatter_match.group(1)
    frontmatter = yaml.safe_load(frontmatter_yaml)

    name = frontmatter.get("name", "")
    description = frontmatter.get("description", "")

    # Parse body sections
    body_content = content[frontmatter_match.end():]

    when_to_use = extract_section(body_content, "When to Use")
    procedure = extract_procedure_steps(body_content)
    constraints = extract_section(body_content, "Constraints")
    examples = extract_examples(body_content)
    related_skills = extract_related_skills(body_content)

    return Skill(
        name=name,
        description=description,
        when_to_use=when_to_use,
        procedure=procedure,
        constraints=constraints,
        examples=examples,
        related_skills=related_skills,
        source_file=filepath,
        raw_content=content,
    )


def extract_section(content: str, section_name: str) -> List[str]:
    """Extract bullet list items from a section.

    Args:
        content: Markdown content
        section_name: Name of section to extract (e.g., "When to Use")

    Returns:
        List of bullet items (without the bullet marker)
    """
    # Find the section
    section_pattern = rf"## {re.escape(section_name)}\n(.*?)(?=\n## |\n---|\Z)"
    section_match = re.search(section_pattern, content, re.DOTALL)

    if not section_match:
        return []

    section_content = section_match.group(1)

    # Extract bullet items
    bullets = re.findall(r"^[\s]*[-*]\s+(.+)$", section_content, re.MULTILINE)
    return bullets


def extract_procedure_steps(content: str) -> List[str]:
    """Extract numbered procedure steps.

    Args:
        content: Markdown content

    Returns:
        List of procedure steps (without the number marker)
    """
    # Find the Procedure section
    section_pattern = r"## Procedure\n(.*?)(?=\n## |\n---|\Z)"
    section_match = re.search(section_pattern, content, re.DOTALL)

    if not section_match:
        return []

    section_content = section_match.group(1)

    # Extract numbered steps
    steps = re.findall(r"^\d+\.\s+(.+)$", section_content, re.MULTILINE)
    return steps


def extract_examples(content: str) -> List[Dict[str, str]]:
    """Extract example sections with Input/Process/Output.

    Args:
        content: Markdown content

    Returns:
        List of dicts with 'title', 'input', 'process', 'output' keys
    """
    examples = []

    # Find all Example subsections
    # Format: ### Example 1: Title or ### Example 1: **Title**
    example_pattern = r"### Example \d+[: ]+(?:\*\*)?(.+?)(?:\*\*)?\n(.*?)(?=### Example|\n## |\Z)"
    for match in re.finditer(example_pattern, content, re.DOTALL):
        title = match.group(1).strip()
        example_content = match.group(2)

        example = {
            "title": title,
            "input": extract_subsection(example_content, "Input"),
            "process": extract_subsection(example_content, "Process"),
            "output": extract_subsection(example_content, "Output"),
        }

        examples.append(example)

    return examples


def extract_subsection(content: str, subsection_name: str) -> str:
    """Extract content from a named subsection.

    Args:
        content: Example section content
        subsection_name: Name of subsection (e.g., "Input")

    Returns:
        Subsection content with leading/trailing whitespace stripped
    """
    # Pattern matches **Input:** content (content may be on same line or next line)
    # Case 1: Content on same line: **Input:** content here...
    # Case 2: Content on next line: **Input:**\ncontent here...
    pattern = rf"\*\*{subsection_name}:\*\*\s*(.*?)(?=\n\*\*|\n###|\Z)"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        result = match.group(1).strip()
        # Remove leading bullets if present (common in Process sections)
        result = re.sub(r"^[-*]\s*", "", result, flags=re.MULTILINE)
        # Remove newlines within the content for cleaner output
        result = re.sub(r"\n+", " ", result)
        return result

    return ""


def extract_related_skills(content: str) -> List[str]:
    """Extract related skill names from Related Skills section.

    Args:
        content: Markdown content

    Returns:
        List of related skill names (in hyphen-case format)
    """
    # Find the Related Skills section
    section_pattern = r"## Related Skills\n(.*?)(?=\n## |\n---|\Z)"
    section_match = re.search(section_pattern, content, re.DOTALL)

    if not section_match:
        return []

    section_content = section_match.group(1)

    # Extract skill names from bullet items
    # Format: "- skill-name: description" or "- skill-name"
    related = []
    for bullet in re.findall(r"^[\s]*[-*]\s+(.+)$", section_content, re.MULTILINE):
        # Extract name (before colon or end of line)
        name = bullet.split(":")[0].strip()
        related.append(name)

    return related


if __name__ == "__main__":
    # Test loading the sample-extracted-skills
    registry = SkillRegistry.from_directory(
        Path(__file__).parent.parent.parent / "sample-extracted-skills"
    )

    print(f"Loaded {len(registry.skills)} atomic skills:")
    for name in sorted(registry.list_all()):
        skill = registry.get(name)
        deps = registry.get_dependencies(name)
        print(f"  - {name} ({len(skill.procedure)} steps, {len(deps)} deps)")

    # Validate references
    invalid = registry.validate_references()
    if invalid:
        print("\nInvalid references found:")
        for name, refs in invalid.items():
            print(f"  - {name}: {refs}")
    else:
        print("\nAll related skill references are valid.")
