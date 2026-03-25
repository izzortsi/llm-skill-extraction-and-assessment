"""
skill_verifier.py

Verify extracted skills on two dimensions:
1. Language defects -- 6 types
2. Procedural sufficiency -- each step must be an action, chain must be complete

Skills are procedures (action sequences), not declarative knowledge.
Each procedure step must be a concrete action: observe, distinguish, enumerate,
compare, select, or apply. Procedural sufficiency means the chain of actions
is traceable from input observation to conclusion.

Rule-based checks detect defects. An LLM revision loop can fix defective
skills by rewriting them to comply with language rules.

Usage:
    python -m c2_skill_extraction.skill_verifier --skills skills.json --output verified_skills.json
    python -m c2_skill_extraction.skill_verifier --skills skills.json --output verified_skills.json --revise --provider anthropic
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import openai

from c0_utils.text_utils import strip_markdown_fences


class _LMProxyResult:
    def __init__(self, response):
        choice = response.choices[0]
        self.message = {"role": "assistant", "content": choice.message.content}
        self.usage = {}
        if response.usage:
            self.usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }


class _LMProxyClient:
    def __init__(self, model, base_url="", api_key=""):
        self.model = model
        self._client = openai.OpenAI(
            base_url=base_url or os.environ.get("LMPROXY_BASE_URL", "http://localhost:8080"),
            api_key=api_key or "lmproxy",
        )

    @property
    def model_name(self):
        return self.model

    def chat(self, messages, tools=None):
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = self._client.chat.completions.create(**kwargs)
        return _LMProxyResult(response)
from c2_extraction.skill_extractor import ExtractedSkill, load_extracted_skills, save_extracted_skills


# 6 language defect types + 1 procedural sufficiency type
LANGUAGE_DEFECT_TYPES = [
    "undefined_language",          # term used without definition
    "implicit_subject",            # imperative with implied "you"
    "passive_voice",               # hides the actor
    "not_stated_with_specificity", # missing N of M, valid enumeration, boundaries
    "vaguely_specified_operation", # action without depth/granularity/acceptance
    "imprecise_external_reference" # reference without enough info to locate
]

PROCEDURAL_DEFECT_TYPES = [
    "procedural_sufficiency",      # step is not an action, or chain is incomplete
]

DEFECT_TYPES = LANGUAGE_DEFECT_TYPES + PROCEDURAL_DEFECT_TYPES


@dataclass
class LanguageDefect:
    """A single language defect found in a skill."""

    defect_type: str       # one of DEFECT_TYPES
    location: str          # which field (name, description, procedure step N)
    description: str       # what is wrong
    suggestion: str        # how to fix


@dataclass
class VerificationResult:
    """Result of verifying a single skill."""

    skill_uid: str
    defects: List[LanguageDefect] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.defects) == 0

    @property
    def defect_count(self) -> int:
        return len(self.defects)


# Passive voice patterns: "is/was/were/are/been/being + past participle"
_PASSIVE_PATTERN = re.compile(
    r"\b(is|was|were|are|been|being|gets|got)\s+(\w+ed|(\w+en))\b",
    re.IGNORECASE,
)

# Vague quantifiers that should be specific counts
_VAGUE_QUANTIFIERS = [
    "some", "various", "several", "many", "multiple",
    "numerous", "a few", "a number of", "etc.", "and so on",
    "and more", "among others",
]

# Undefined referents (pronouns without clear antecedent)
_UNDEFINED_REFERENTS = re.compile(
    r"\b(it|this|that|these|those|they|them)\b(?!\s+(is|was|were|are|has|have|will|can|should|would|could|may|might)\s+(a|an|the|not)\b)",
    re.IGNORECASE,
)

# Vague operation verbs (too general without further specification)
_VAGUE_OPERATIONS = [
    "handle", "process", "manage", "deal with", "address",
    "take care of", "work with", "do", "perform",
]

# Imprecise reference patterns
_IMPRECISE_REFS = [
    "the standard", "the specification", "the documentation",
    "the relevant", "the appropriate", "as needed",
    "as necessary", "if applicable", "when relevant",
]


def validate_skill_input(skill: ExtractedSkill) -> None:
    """Validate that a skill has all required fields before verification.

    Raises:
        ValueError: If any required field is missing or empty.
    """
    if not skill.skill_uid or not str(skill.skill_uid).strip():
        raise ValueError(
            f"Skill is missing required field 'skill_uid': "
            f"name={getattr(skill, 'name', '')!r}"
        )
    if not skill.name or not str(skill.name).strip():
        raise ValueError(
            f"Skill {skill.skill_uid!r} has missing or empty 'name'"
        )
    if not skill.procedure or not isinstance(skill.procedure, list):
        raise ValueError(
            f"Skill {skill.skill_uid!r} ({skill.name!r}) has missing or empty "
            f"'procedure': procedure must be a non-empty list"
        )


def _check_passive_voice(text: str, location: str) -> List[LanguageDefect]:
    """Detect passive voice constructions."""
    defects = []
    matches = _PASSIVE_PATTERN.finditer(text)
    for match in matches:
        phrase = match.group(0)
        # Skip common false positives
        if phrase.lower() in ("is used", "is called", "is named", "is defined"):
            continue
        defects.append(LanguageDefect(
            defect_type="passive_voice",
            location=location,
            description=f"Passive voice: \"{phrase}\" hides the actor",
            suggestion=f"Rewrite \"{phrase}\" to name the actor explicitly",
        ))
    return defects


def _check_implicit_subject(text: str, location: str) -> List[LanguageDefect]:
    """Detect imperative sentences with implied subject (starts with verb)."""
    defects = []
    sentences = re.split(r'[.!?]\s+', text)
    imperative_verbs = {
        "identify", "find", "check", "verify", "ensure", "determine",
        "analyze", "examine", "compare", "evaluate", "apply", "use",
        "consider", "note", "remember", "look", "review", "read",
    }

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        first_word = sentence.split()[0].lower() if sentence.split() else ""
        if first_word in imperative_verbs:
            defects.append(LanguageDefect(
                defect_type="implicit_subject",
                location=location,
                description=f"Imperative \"{sentence[:60]}...\" has implied subject",
                suggestion=f"Add explicit subject: \"The analyst {first_word}s...\"",
            ))
    return defects


def _check_undefined_language(text: str, location: str) -> List[LanguageDefect]:
    """Detect terms used without definition (vague referents)."""
    defects = []

    # Check for isolated pronouns that likely lack antecedent
    # Only flag "it" and "this" at sentence start (higher confidence of missing antecedent)
    sentences = re.split(r'[.!?]\s+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        first_word = sentence.split()[0].lower() if sentence.split() else ""
        if first_word in ("it", "this", "that"):
            # "This" at start of a sentence is a common undefined referent
            defects.append(LanguageDefect(
                defect_type="undefined_language",
                location=location,
                description=f"Undefined referent \"{first_word}\" at sentence start: \"{sentence[:60]}...\"",
                suggestion=f"Replace \"{first_word}\" with the specific noun it refers to",
            ))

    return defects


def _check_specificity(text: str, location: str) -> List[LanguageDefect]:
    """Detect vague quantifiers that should be specific counts."""
    defects = []
    text_lower = text.lower()

    for vague in _VAGUE_QUANTIFIERS:
        if vague in text_lower:
            defects.append(LanguageDefect(
                defect_type="not_stated_with_specificity",
                location=location,
                description=f"Vague quantifier \"{vague}\" used without specific count",
                suggestion=f"Replace \"{vague}\" with an exact count (e.g., \"3 of 5\")",
            ))

    return defects


def _check_vague_operations(text: str, location: str) -> List[LanguageDefect]:
    """Detect vaguely specified operations."""
    defects = []
    text_lower = text.lower()

    for vague_op in _VAGUE_OPERATIONS:
        if vague_op in text_lower:
            defects.append(LanguageDefect(
                defect_type="vaguely_specified_operation",
                location=location,
                description=f"Vague operation \"{vague_op}\" lacks depth or acceptance criteria",
                suggestion=f"Replace \"{vague_op}\" with a specific action and expected outcome",
            ))

    return defects


def _check_imprecise_references(text: str, location: str) -> List[LanguageDefect]:
    """Detect imprecise external references."""
    defects = []
    text_lower = text.lower()

    for imprecise in _IMPRECISE_REFS:
        if imprecise in text_lower:
            defects.append(LanguageDefect(
                defect_type="imprecise_external_reference",
                location=location,
                description=f"Imprecise reference \"{imprecise}\" without enough info to locate",
                suggestion=f"Replace \"{imprecise}\" with exact name, path, or section reference",
            ))

    return defects


# The 6 procedural action primitives. Each procedure step must map to one of these.
PROCEDURAL_PRIMITIVES = [
    "observe",     # read, perceive, scan, inspect, examine input
    "distinguish", # classify, differentiate, categorize, identify, recognize
    "enumerate",   # list, generate options, brainstorm, collect candidates
    "compare",     # evaluate, weigh, contrast, rank, assess between options
    "select",      # choose, pick, decide, commit to one option
    "apply",       # execute, perform, carry out, produce, transform
]

# Verb sets that map to each procedural primitive
_OBSERVE_VERBS = {
    "observe", "read", "perceive", "scan", "inspect", "examine",
    "look", "view", "survey", "notice", "detect", "find", "locate",
    "extract", "gather", "collect", "note", "identify", "see", "check",
    "review", "study", "reads", "observes", "examines", "inspects",
    "scans", "identifies", "finds", "locates", "extracts", "notes",
    "checks", "reviews", "studies", "detects", "gathers", "collects",
}

_DISTINGUISH_VERBS = {
    "distinguish", "classify", "differentiate", "categorize", "label",
    "recognize", "separate", "sort", "group", "tag", "type", "mark",
    "divide", "partition", "filter", "distinguishes", "classifies",
    "differentiates", "categorizes", "labels", "recognizes", "separates",
    "sorts", "groups", "tags", "marks", "divides", "filters",
}

_ENUMERATE_VERBS = {
    "enumerate", "list", "generate", "brainstorm", "collect",
    "inventory", "catalog", "compile", "assemble", "create",
    "produce", "outline", "map", "chart", "enumerates", "lists",
    "generates", "compiles", "assembles", "creates", "outlines",
    "maps", "charts", "catalogs",
}

_COMPARE_VERBS = {
    "compare", "evaluate", "weigh", "contrast", "rank", "assess",
    "measure", "rate", "score", "benchmark", "judge", "test",
    "validate", "verify", "match", "compares", "evaluates",
    "weighs", "contrasts", "ranks", "assesses", "measures",
    "rates", "scores", "judges", "tests", "validates", "verifies",
    "matches",
}

_SELECT_VERBS = {
    "select", "choose", "pick", "decide", "commit", "determine",
    "opt", "prefer", "accept", "reject", "adopt", "selects",
    "chooses", "picks", "decides", "commits", "determines",
    "opts", "prefers", "accepts", "rejects", "adopts",
}

_APPLY_VERBS = {
    "apply", "execute", "perform", "carry", "produce", "transform",
    "compute", "calculate", "derive", "construct", "build", "write",
    "output", "return", "submit", "format", "render", "state",
    "conclude", "synthesize", "combine", "merge", "integrate",
    "applies", "executes", "performs", "produces", "transforms",
    "computes", "calculates", "derives", "constructs", "builds",
    "writes", "outputs", "returns", "formats", "renders", "states",
    "concludes", "synthesizes", "combines", "merges", "integrates",
}

_ALL_ACTION_VERBS = (
    _OBSERVE_VERBS | _DISTINGUISH_VERBS | _ENUMERATE_VERBS
    | _COMPARE_VERBS | _SELECT_VERBS | _APPLY_VERBS
)


def _classify_step_primitive(step_text: str) -> Optional[str]:
    """Classify which procedural primitive a step maps to.

    Returns the primitive name, or None if the step does not map to any.
    """
    words = step_text.lower().split()
    # Check first 5 words for action verbs (the action verb is typically near the start)
    check_words = words[:5] if len(words) >= 5 else words

    for word in check_words:
        cleaned = word.strip(".,;:!?()[]\"'")
        if cleaned in _OBSERVE_VERBS:
            return "observe"
        if cleaned in _DISTINGUISH_VERBS:
            return "distinguish"
        if cleaned in _ENUMERATE_VERBS:
            return "enumerate"
        if cleaned in _COMPARE_VERBS:
            return "compare"
        if cleaned in _SELECT_VERBS:
            return "select"
        if cleaned in _APPLY_VERBS:
            return "apply"

    return None


def _check_procedural_sufficiency(skill: ExtractedSkill) -> List[LanguageDefect]:
    """Verify procedural sufficiency of a skill.

    Checks:
    1. Each procedure step must be a procedural action (maps to a primitive)
    2. Procedure must not be empty
    3. Procedure should start with observation and end with application/selection
    """
    defects = []

    # Check: procedure must not be empty
    if not skill.procedure:
        defects.append(LanguageDefect(
            defect_type="procedural_sufficiency",
            location="procedure",
            description="Procedure is empty -- a skill must have at least 1 procedural action step",
            suggestion="Add numbered action steps using procedural primitives: observe, distinguish, enumerate, compare, select, apply",
        ))
        return defects

    # Check each step maps to a procedural primitive
    step_primitives = []
    for i, step in enumerate(skill.procedure):
        primitive = _classify_step_primitive(step)
        step_primitives.append(primitive)

        if primitive is None:
            # Check if the step contains ANY action verb at all
            words = set(w.strip(".,;:!?()[]\"'").lower() for w in step.split())
            has_any_action = bool(words & _ALL_ACTION_VERBS)

            if not has_any_action:
                defects.append(LanguageDefect(
                    defect_type="procedural_sufficiency",
                    location=f"procedure step {i+1}",
                    description=f"Step is not a procedural action: \"{step[:80]}\"",
                    suggestion="Rewrite as an action using a procedural primitive verb (observe, distinguish, enumerate, compare, select, apply)",
                ))

    # Check: first step should involve observation (reading/perceiving input)
    if step_primitives and step_primitives[0] not in ("observe", None):
        # Not a hard error, just a warning-level note -- only flag if no observe anywhere
        has_observe = "observe" in step_primitives
        if not has_observe:
            defects.append(LanguageDefect(
                defect_type="procedural_sufficiency",
                location="procedure",
                description="Procedure has no observation step -- actions should begin with observing the input",
                suggestion="Add an initial step that observes/reads/scans the input",
            ))

    return defects


def verify_skill(skill: ExtractedSkill) -> VerificationResult:
    """Verify a single skill on two dimensions:
    1. Language defects (6 types)
    2. Procedural sufficiency (each step is an action, chain is complete)

    Args:
        skill: The extracted skill to verify

    Returns:
        VerificationResult with list of defects found
    """
    validate_skill_input(skill)

    all_defects = []

    # --- Procedural sufficiency checks ---
    all_defects.extend(_check_procedural_sufficiency(skill))

    # --- Language defect checks ---

    # Check description
    if skill.description:
        all_defects.extend(_check_passive_voice(skill.description, "description"))
        all_defects.extend(_check_undefined_language(skill.description, "description"))
        all_defects.extend(_check_specificity(skill.description, "description"))
        all_defects.extend(_check_vague_operations(skill.description, "description"))
        all_defects.extend(_check_imprecise_references(skill.description, "description"))

    # Check each procedure step
    for i, step in enumerate(skill.procedure):
        location = f"procedure step {i+1}"
        all_defects.extend(_check_passive_voice(step, location))
        all_defects.extend(_check_implicit_subject(step, location))
        all_defects.extend(_check_undefined_language(step, location))
        all_defects.extend(_check_specificity(step, location))
        all_defects.extend(_check_vague_operations(step, location))
        all_defects.extend(_check_imprecise_references(step, location))

    # Check when_to_use
    if skill.when_to_use:
        all_defects.extend(_check_passive_voice(skill.when_to_use, "when_to_use"))
        all_defects.extend(_check_undefined_language(skill.when_to_use, "when_to_use"))
        all_defects.extend(_check_specificity(skill.when_to_use, "when_to_use"))
        all_defects.extend(_check_imprecise_references(skill.when_to_use, "when_to_use"))

    # Check constraints
    for i, constraint in enumerate(skill.constraints):
        location = f"constraint {i+1}"
        all_defects.extend(_check_passive_voice(constraint, location))
        all_defects.extend(_check_undefined_language(constraint, location))
        all_defects.extend(_check_specificity(constraint, location))
        all_defects.extend(_check_vague_operations(constraint, location))

    return VerificationResult(
        skill_uid=skill.skill_uid,
        defects=all_defects,
    )


def verify_skills(skills: List[ExtractedSkill], verbose: bool = False) -> List[VerificationResult]:
    """Verify a list of skills against the 6 language defect types.

    Args:
        skills: List of skills to verify
        verbose: Enable verbose logging

    Returns:
        List of VerificationResult objects
    """
    results = []
    for skill in skills:
        result = verify_skill(skill)
        results.append(result)

        if verbose:
            status = "VALID" if result.is_valid else f"{result.defect_count} defects"
            print(f"  {skill.name}: {status}")
            if not result.is_valid:
                for defect in result.defects:
                    print(f"    [{defect.defect_type}] {defect.location}: {defect.description}")

    return results


def save_verification_results(
    skills: List[ExtractedSkill],
    results: List[VerificationResult],
    output_path: Path,
) -> None:
    """Save verified skills with their verification results.

    Args:
        skills: List of skills
        results: Corresponding verification results
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for skill, result in zip(skills, results):
        entry = {
            "skill_uid": skill.skill_uid,
            "name": skill.name,
            "description": skill.description,
            "procedure": skill.procedure,
            "when_to_use": skill.when_to_use,
            "constraints": skill.constraints,
            "source_task_uids": skill.source_task_uids,
            "extraction_method": skill.extraction_method,
            "is_valid": result.is_valid,
            "defect_count": result.defect_count,
            "defects": [
                {
                    "defect_type": d.defect_type,
                    "location": d.location,
                    "description": d.description,
                    "suggestion": d.suggestion,
                }
                for d in result.defects
            ],
        }
        data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# Optional standards loading
# =============================================================================

def load_standards_text(standards_dir: Optional[Path] = None) -> str:
    """Load language defect definitions from a standards directory.

    Reads standard language files if they exist. Returns the combined text
    for use in LLM revision prompts. Returns empty string if no standards
    directory is provided or files are not found.

    Args:
        standards_dir: Path to a standards directory containing language
            definition files. Purely optional -- if None or if the directory
            does not exist, returns empty string.

    Returns:
        Combined standards text, or empty string if not available.
    """
    if standards_dir is None:
        return ""

    standards_dir = Path(standards_dir)
    if not standards_dir.is_dir():
        return ""

    # Look for common standards file names
    candidate_files = [
        "claude-language-usage-instructions.txt",
        "standards-report-writing.txt",
    ]

    parts = []
    for filename in candidate_files:
        filepath = standards_dir / filename
        if filepath.exists():
            parts.append(filepath.read_text(encoding="utf-8"))

    return "\n\n".join(parts)


# =============================================================================
# LLM-based skill revision (fix defective skills)
# =============================================================================

_REVISION_PROMPT = """You are a skill editor. Fix the defects in this skill so it complies with the language standards.

## Language Standards

{standards_text}

## Skill to Fix

Name: {skill_name}
Description: {skill_description}
Procedure:
{skill_procedure}
When to use: {skill_when_to_use}
Constraints:
{skill_constraints}

## Defects Found

{defects_text}

## Instructions

Rewrite the skill to fix ALL listed defects. Rules:
1. Every sentence must have an explicit, named subject (no imperatives, no implied "you")
2. No passive voice -- every verb must have a stated actor
3. No undefined referents ("it", "this", "that" without antecedent)
4. No vague quantifiers ("some", "various", "etc.") -- use exact counts
5. No vague operations ("handle", "process", "manage") -- use specific actions
6. No imprecise references ("the relevant", "as needed") -- use exact names
7. Every procedure step must be a procedural action using one of 6 primitives: observe, distinguish, enumerate, compare, select, apply

Return ONLY valid JSON (no markdown, no explanation):

{{
  "name": "{skill_name}",
  "description": "<fixed description>",
  "procedure": ["<fixed step 1>", "<fixed step 2>"],
  "when_to_use": "<fixed when_to_use>",
  "constraints": ["<fixed constraint 1>", "<fixed constraint 2>"]
}}
"""


def _format_defects_for_prompt(defects: List[LanguageDefect]) -> str:
    """Format defects list for the revision prompt."""
    lines = []
    for i, d in enumerate(defects):
        lines.append(f"{i+1}. [{d.defect_type}] {d.location}: {d.description}")
        lines.append(f"   Suggestion: {d.suggestion}")
    return "\n".join(lines)


def _format_procedure_for_prompt(procedure: List[str]) -> str:
    """Format procedure steps for the revision prompt."""
    return "\n".join(f"  {i+1}. {step}" for i, step in enumerate(procedure))


def _format_constraints_for_prompt(constraints: List[str]) -> str:
    """Format constraints for the revision prompt."""
    return "\n".join(f"  - {c}" for c in constraints)


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


def revise_skill(
    skill: ExtractedSkill,
    defects: List[LanguageDefect],
    provider,
    standards_text: str = "",
    verbose: bool = False,
) -> ExtractedSkill:
    """Use an LLM to rewrite a defective skill to fix all defects.

    Args:
        skill: The skill with defects
        defects: List of defects to fix
        provider: LLM provider with chat() method
        standards_text: Standards text to include in prompt
        verbose: Enable verbose logging

    Returns:
        Revised ExtractedSkill with defects fixed
    """
    prompt = _REVISION_PROMPT.format(
        standards_text=standards_text[:3000] if standards_text else "(standards not loaded)",
        skill_name=skill.name,
        skill_description=skill.description,
        skill_procedure=_format_procedure_for_prompt(skill.procedure),
        skill_when_to_use=skill.when_to_use,
        skill_constraints=_format_constraints_for_prompt(skill.constraints),
        defects_text=_format_defects_for_prompt(defects),
    )

    messages = [{"role": "user", "content": prompt}]
    result = provider.chat(messages)
    response_text = _extract_response_text(result)
    json_text = _strip_markdown_fences(response_text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        if verbose:
            print(f"    REVISION FAILED: could not parse JSON for {skill.name}")
        return skill

    return ExtractedSkill(
        skill_uid=skill.skill_uid,
        name=data.get("name", skill.name),
        description=data.get("description", skill.description),
        procedure=data.get("procedure", skill.procedure),
        when_to_use=data.get("when_to_use", skill.when_to_use),
        constraints=data.get("constraints", skill.constraints),
        source_task_uids=skill.source_task_uids,
        source_trace_uids=skill.source_trace_uids,
    )


def verify_and_revise(
    skills: List[ExtractedSkill],
    provider=None,
    max_revisions: int = 2,
    standards_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Tuple[List[ExtractedSkill], List[VerificationResult]]:
    """Verify skills and revise defective ones using an LLM.

    Runs verify -> revise -> verify loop up to max_revisions times per skill.
    Skills that pass verification are returned unchanged.

    Args:
        skills: List of skills to verify and fix
        provider: LLM provider for revision (None = verify only, no revision)
        max_revisions: Maximum revision attempts per skill
        standards_dir: Optional path to a standards directory for loading
            language definitions. If None, revision prompts use a placeholder.
        verbose: Enable verbose logging

    Returns:
        (revised_skills, final_verification_results) tuple
    """
    standards_text = load_standards_text(standards_dir) if provider else ""

    final_skills = []
    final_results = []

    for skill in skills:
        current_skill = skill
        current_result = verify_skill(current_skill)

        revision_count = 0
        while not current_result.is_valid and provider is not None and revision_count < max_revisions:
            revision_count += 1
            if verbose:
                print(f"  {current_skill.name}: {current_result.defect_count} defects, revision {revision_count}/{max_revisions}")

            current_skill = revise_skill(
                current_skill,
                current_result.defects,
                provider,
                standards_text,
                verbose,
            )
            current_result = verify_skill(current_skill)

            if verbose:
                status = "VALID" if current_result.is_valid else f"{current_result.defect_count} defects remain"
                print(f"    After revision {revision_count}: {status}")

        final_skills.append(current_skill)
        final_results.append(current_result)

    return final_skills, final_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and optionally revise extracted skills")
    parser.add_argument("--skills", "-s", type=Path, required=True, help="Input skills JSON file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output verified skills JSON path")
    parser.add_argument("--revise", action="store_true", help="Enable LLM-based revision of defective skills")
    parser.add_argument("--max-revisions", type=int, default=2, help="Maximum revision attempts per skill")
    parser.add_argument("--standards-dir", type=Path, default=None, help="Optional path to standards directory")
    parser.add_argument("--provider", type=str, default="anthropic", help="LLM provider for revision (anthropic, openai, mock)")
    parser.add_argument("--model", type=str, default="claude-opus-4-6", help="Model for revision")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.skills.exists():
        raise FileNotFoundError(f"Skills file not found: {args.skills}")

    skills = load_extracted_skills(args.skills)
    if args.verbose:
        print(f"Loaded {len(skills)} skills from {args.skills}")

    if args.revise:
        provider = _LMProxyClient(args.model)

        revised_skills, results = verify_and_revise(
            skills, provider,
            max_revisions=args.max_revisions,
            standards_dir=args.standards_dir,
            verbose=args.verbose,
        )
        save_verification_results(revised_skills, results, args.output)

        valid_count = sum(1 for r in results if r.is_valid)
        print(f"Verified {len(skills)} skills: {valid_count} valid, {len(skills) - valid_count} with defects")
        print(f"Saved to {args.output}")
    else:
        results = verify_skills(skills, verbose=args.verbose)
        save_verification_results(skills, results, args.output)

        valid_count = sum(1 for r in results if r.is_valid)
        total_defects = sum(r.defect_count for r in results)
        print(f"Verified {len(skills)} skills: {valid_count} valid, {len(skills) - valid_count} with defects ({total_defects} total defects)")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
