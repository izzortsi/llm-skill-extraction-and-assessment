"""
skill_injection.py

Format skill content for system prompt injection, matching the SkillsBench paper method:
"We provide Skills as system-level context preceding the task instruction."
"""

from __future__ import annotations

from typing import Optional


def format_skill_for_system_prompt(
    base_system_prompt: str,
    skill_name: str,
    skill_content: str,
) -> str:
    """Format skill content as system-level context preceding the task.

    Produces:
        [base system prompt]

        --- SKILL: {skill_name} ---
        {skill content}
        --- END SKILL ---

    Args:
        base_system_prompt: The base system prompt (agent instructions)
        skill_name: Name of the skill being injected
        skill_content: Full skill markdown content (procedure, constraints, examples)

    Returns:
        Complete system prompt with skill injected
    """
    skill_block = (
        f"\n\n--- SKILL: {skill_name} ---\n"
        f"{skill_content.strip()}\n"
        f"--- END SKILL ---"
    )

    return base_system_prompt.rstrip() + skill_block


def format_self_generation_prompt(
    base_system_prompt: str,
    task_description: str,
) -> str:
    """Format prompt for self-generated skill condition.

    Matching SkillsBench: "Generate relevant procedural knowledge before solving the task."
    The agent is instructed to generate its own skill before attempting the task.

    Args:
        base_system_prompt: The base system prompt
        task_description: Brief description of the task domain

    Returns:
        System prompt with self-generation instruction
    """
    self_gen_block = (
        "\n\n--- SELF-GENERATED SKILL INSTRUCTION ---\n"
        "Before solving the task, generate relevant procedural knowledge that "
        "would help you approach this type of problem systematically.\n\n"
        "Structure your procedural knowledge as:\n"
        "1. When to Use: Conditions that indicate this procedure applies\n"
        "2. Procedure: Numbered steps to follow\n"
        "3. Constraints: Important limitations or pitfalls to avoid\n\n"
        f"Task domain: {task_description}\n\n"
        "First output your procedural knowledge, then apply it to solve the task.\n"
        "--- END INSTRUCTION ---"
    )

    return base_system_prompt.rstrip() + self_gen_block


# Default base system prompt for reading comprehension tasks
DEFAULT_READING_COMPREHENSION_PROMPT = (
    "You are an expert in reading comprehension and cognitive science. "
    "Analyze the given passage and challenge carefully. "
    "Provide a thorough, well-structured answer that addresses all aspects of the challenge. "
    "Support your analysis with specific evidence from the passage."
)

# Default base system prompt for coding tasks
DEFAULT_CODING_PROMPT = (
    "You are an expert software developer. "
    "Solve the given programming task step by step. "
    "You have access to a bash tool to execute commands."
)


def get_default_system_prompt(domain: str) -> str:
    """Get the default base system prompt for a domain.

    Args:
        domain: Task domain (reading_comprehension, coding, etc.)

    Returns:
        Default system prompt string
    """
    if domain == "reading_comprehension":
        return DEFAULT_READING_COMPREHENSION_PROMPT
    if domain in ("coding", "code_bugfix"):
        return DEFAULT_CODING_PROMPT
    return DEFAULT_READING_COMPREHENSION_PROMPT


def format_extracted_skill_for_system_prompt(
    base_system_prompt: str,
    skill,
) -> str:
    """Format an ExtractedSkill (from skill_extractor.py) as system-level context.

    Converts the structured skill fields into a markdown skill block
    compatible with the existing --- SKILL: ... --- format.

    Args:
        base_system_prompt: The base system prompt
        skill: ExtractedSkill instance with name, description, procedure, etc.

    Returns:
        Complete system prompt with skill injected
    """
    parts = []
    if skill.description:
        parts.append(skill.description)
        parts.append("")

    if skill.when_to_use:
        parts.append("## When to Use")
        parts.append(skill.when_to_use)
        parts.append("")

    if skill.procedure:
        parts.append("## Procedure")
        for i, step in enumerate(skill.procedure):
            parts.append(f"{i+1}. {step}")
        parts.append("")

    if skill.constraints:
        parts.append("## Constraints")
        for constraint in skill.constraints:
            parts.append(f"- {constraint}")

    skill_content = "\n".join(parts)
    return format_skill_for_system_prompt(base_system_prompt, skill.name, skill_content)


def format_skill_for_user_message(
    user_prompt: str,
    skill,
) -> str:
    """Append a minimal skill hint (description only) to the user message.

    Keeps the system prompt clean and injects only the skill description
    after the passage and challenge in the user message.

    Args:
        user_prompt: The existing user message (passage + challenge).
        skill: ExtractedSkill instance with at least a name and description.

    Returns:
        User message with skill hint appended.
    """
    return f"{user_prompt}\n\nSkill hint ({skill.name}): {skill.description}"
