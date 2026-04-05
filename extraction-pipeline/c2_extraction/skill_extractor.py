"""
skill_extractor.py

Extract reusable procedural skills from reasoning traces using Opus.
Phase 3 of the skill distillation pipeline: Reasoning Traces -> Skills.

Usage:
    python -m c2_skill_extraction.skill_extractor --traces traces.jsonl --output skills.json
    python -m c2_skill_extraction.skill_extractor --traces traces.jsonl --output skills.json --max-skills 10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from c0_utils.uid import generate_uid
from c0_utils.text_utils import strip_markdown_fences
from c2_extraction.trace_capturer import ReasoningTrace, load_traces


@dataclass
class ExtractedSkill:
    """A reusable procedural skill extracted from reasoning traces."""

    skill_uid: str
    name: str                  # kebab-case skill name
    description: str
    procedure: List[str]       # numbered procedure steps
    when_to_use: str
    constraints: List[str]
    source_task_uids: List[str]
    source_trace_uids: List[str] = field(default_factory=list)
    extraction_method: str = ""


SKILL_EXTRACTION_PROMPT = """You are an expert at extracting reusable procedural skills from action traces.

A skill is a sequence of procedural actions — not declarative knowledge. Each procedure step must be an action (what to do), not an explanation (why to do it).

The 6 procedural action primitives are:
- OBSERVE: read, perceive, or scan input
- DISTINGUISH: classify, differentiate, or categorize
- ENUMERATE: list possible next actions or options
- COMPARE: evaluate between enumerated options
- SELECT: choose one action from the compared options
- APPLY: execute the selected action on the input

Given the following action traces from solving different tasks, identify recurring action sequences and extract them as reusable procedural skills.

## Action Traces

{traces_text}

## Instructions

1. Identify recurring action sequences across the traces
2. Name each skill using kebab-case (e.g., "concept-comparison-synthesis")
3. Write each procedure step as a concrete action using one of the 6 primitives (observe, distinguish, enumerate, compare, select, apply)
4. Each step must name the actor and the object acted upon
5. If two action sequences describe the same procedure, merge them into one skill

## Output Format

Return ONLY valid JSON (no markdown, no explanation) as a list of skills:

[
  {{
    "name": "<kebab-case-skill-name>",
    "description": "<one-sentence description of what action sequence this skill performs>",
    "procedure": [
      "The analyst observes [specific input]",
      "The analyst distinguishes [X from Y]",
      "The analyst enumerates [possible actions]",
      "The analyst compares [option A against option B]",
      "The analyst selects [chosen action]",
      "The analyst applies [action] to [object]"
    ],
    "when_to_use": "<observable conditions that trigger this action sequence>",
    "constraints": [
      "Constraint or limitation 1",
      "Constraint or limitation 2"
    ],
    "source_task_ids": ["<task_uid_1>", "<task_uid_2>"]
  }}
]

Extract between 1 and {max_skills} skills. Quality over quantity: only extract skills with clear, actionable procedures where every step is a procedural action.

CRITICAL — ATOMICITY: Extract the MOST ATOMIC skills possible.
- Each skill must address exactly ONE reasoning sub-operation.
- If a skill involves two distinct cognitive steps (e.g., "identify contradictions" AND "resolve contradictions"), split the skill into two separate skills.
- Prefer many small skills over few compound skills.
- A skill is atomic if removing any single procedure step makes the skill incomplete for its stated purpose.
"""


def validate_trace_input(trace: ReasoningTrace) -> None:
    """Validate that a trace has all required fields before skill extraction.

    Raises:
        ValueError: If any required field is missing or empty.
    """
    if not trace.task_uid or not str(trace.task_uid).strip():
        raise ValueError(
            f"Trace is missing required field 'task_uid': "
            f"model={getattr(trace, 'model', '')!r}"
        )
    if not trace.procedural_steps or not isinstance(trace.procedural_steps, list):
        raise ValueError(
            f"Trace {trace.task_uid!r} has missing or empty 'procedural_steps': "
            f"procedural_steps must be a non-empty list"
        )


def _generate_skill_uid(name: str, source_task_uids: List[str]) -> str:
    """Generate a deterministic skill UID in company standard format."""
    seed = "|".join(sorted(source_task_uids)) + "|" + name
    return generate_uid(seed)


def _extract_response_text(result) -> str:
    """Extract text content from a provider chat result."""
    if isinstance(result.message.get("content"), str):
        return result.message["content"]
    if isinstance(result.message.get("content"), list):
        parts = []
        for block in result.message["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


_strip_markdown_fences = strip_markdown_fences


def _format_traces_for_prompt(traces: List[ReasoningTrace]) -> str:
    """Format procedural traces as text for the extraction prompt."""
    parts = []
    for i, trace in enumerate(traces):
        parts.append(f"### Trace {i+1} (Task: {trace.task_uid})")
        parts.append(f"Challenge: {trace.user_prompt[:500]}")
        parts.append(f"Procedural actions taken:")
        for j, step in enumerate(trace.procedural_steps):
            parts.append(f"  {j+1}. {step[:300]}")
        parts.append(f"Conclusion reached: {trace.conclusion[:300]}")
        parts.append("")
    return "\n".join(parts)


def _deduplicate_skills(skills: List[ExtractedSkill]) -> List[ExtractedSkill]:
    """Remove duplicate skills by merging those with very similar names.

    Two skills are considered duplicates if their names share >60% of words.
    """
    if len(skills) <= 1:
        return skills

    deduplicated = []
    seen_name_words = []

    for skill in skills:
        skill_words = set(skill.name.split("-"))
        is_duplicate = False

        for i, existing_words in enumerate(seen_name_words):
            overlap = len(skill_words & existing_words)
            total = max(len(skill_words), len(existing_words))
            if total > 0 and overlap / total > 0.6:
                # Merge source_task_uids into existing skill
                existing_skill = deduplicated[i]
                merged_uids = list(set(existing_skill.source_task_uids + skill.source_task_uids))
                deduplicated[i] = ExtractedSkill(
                    skill_uid=existing_skill.skill_uid,
                    name=existing_skill.name,
                    description=existing_skill.description,
                    procedure=existing_skill.procedure,
                    when_to_use=existing_skill.when_to_use,
                    constraints=existing_skill.constraints,
                    source_task_uids=merged_uids,
                    source_trace_uids=existing_skill.source_trace_uids,
                )
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(skill)
            seen_name_words.append(skill_words)

    return deduplicated


def extract_skills_from_traces(
    traces: List[ReasoningTrace],
    provider,
    max_skills: int = 10,
    deduplicate: bool = True,
    verbose: bool = False,
) -> List[ExtractedSkill]:
    """Extract reusable skills from a set of reasoning traces.

    Args:
        traces: List of reasoning traces to analyze
        provider: LLM provider with chat() method (Opus for extraction)
        max_skills: Maximum number of skills to extract
        deduplicate: Merge duplicate skills
        verbose: Enable verbose logging

    Returns:
        List of ExtractedSkill objects
    """
    if not traces:
        return []

    for trace in traces:
        validate_trace_input(trace)

    traces_text = _format_traces_for_prompt(traces)
    prompt = SKILL_EXTRACTION_PROMPT.format(
        traces_text=traces_text,
        max_skills=max_skills,
    )

    messages = [{"role": "user", "content": prompt}]

    if verbose:
        print(f"Extracting skills from {len(traces)} traces...")

    result = provider.chat(messages)
    response_text = _extract_response_text(result)
    json_text = _strip_markdown_fences(response_text)

    try:
        skills_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        if verbose:
            print(f"  ERROR: Failed to parse JSON: {e}")
            print(f"  Raw response: {response_text[:500]}")
        return []

    if not isinstance(skills_data, list):
        skills_data = [skills_data]

    skills = []
    model_name = getattr(provider, "model_name", getattr(provider, "model", "unknown"))
    # TODO: import from llm-providers
    import re as _re
    short_model = _re.sub(r"^claude-", "", model_name)
    short_model = _re.sub(r"[:/]", "-", short_model)
    extraction_method = f"{short_model}-skill-extraction-v1"

    for entry in skills_data:
        name = entry.get("name", "unnamed-skill")
        source_task_uids = entry.get("source_task_ids", entry.get("source_task_uids", []))
        skill = ExtractedSkill(
            skill_uid=_generate_skill_uid(name, source_task_uids),
            name=name,
            description=entry.get("description", ""),
            procedure=entry.get("procedure", []),
            when_to_use=entry.get("when_to_use", ""),
            constraints=entry.get("constraints", []),
            source_task_uids=source_task_uids,
            extraction_method=extraction_method,
        )
        skills.append(skill)

        if verbose:
            print(f"  -> Extracted skill: {skill.name} ({len(skill.procedure)} steps)")

    if deduplicate:
        before_count = len(skills)
        skills = _deduplicate_skills(skills)
        if verbose and len(skills) < before_count:
            print(f"  Deduplicated: {before_count} -> {len(skills)} skills")

    return skills


def save_extracted_skills(skills: List[ExtractedSkill], output_path: Path) -> None:
    """Save extracted skills to a JSON file.

    Args:
        skills: List of ExtractedSkill objects
        output_path: Path to output JSON file
    """
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
    """Load extracted skills from a JSON file.

    Args:
        filepath: Path to extracted skills JSON

    Returns:
        List of ExtractedSkill objects
    """
    # TODO: import from llm-providers
    # from c1_providers.schema_validator import validate_skills_json

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # validate_skills_json(data)  # TODO: import from llm-providers

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract skills from reasoning traces")
    parser.add_argument("--traces", "-t", type=Path, required=True, help="Input traces JSONL file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output skills JSON path")
    parser.add_argument("--max-skills", type=int, default=10, help="Maximum number of skills to extract")
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication")
    parser.add_argument("--provider", type=str, default="anthropic", help="LLM provider (anthropic, openai, mock)")
    parser.add_argument("--model", type=str, default="claude-opus-4-6", help="Model to use")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.traces.exists():
        raise FileNotFoundError(f"Traces file not found: {args.traces}")

    # TODO: import from llm-providers
    from c1_providers.providers import create_provider  # noqa: requires llm-providers on sys.path
    provider = create_provider(args.provider, args.model)

    traces = load_traces(args.traces)
    if args.verbose:
        print(f"Loaded {len(traces)} traces from {args.traces}")

    skills = extract_skills_from_traces(
        traces, provider,
        max_skills=args.max_skills,
        deduplicate=not args.no_dedup,
        verbose=args.verbose,
    )
    save_extracted_skills(skills, args.output)
    print(f"Saved {len(skills)} skills to {args.output}")


if __name__ == "__main__":
    main()
