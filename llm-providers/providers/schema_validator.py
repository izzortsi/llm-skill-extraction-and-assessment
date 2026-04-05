"""
schema_validator.py

Strict schema validation for pipeline data files. Every pipeline stage
calls validate_* before processing input to catch malformed data at load
time rather than deep in processing.
"""

from __future__ import annotations

from typing import Any, Dict, List


class SchemaValidationError(ValueError):
    """Raised when loaded data does not conform to the expected schema."""


def _check_field(
    entry: Dict[str, Any],
    field_name: str,
    expected_type: type,
    entry_index: int,
    entity_name: str,
    is_required: bool = True,
) -> None:
    """Check that a single field exists and has the expected type.

    Args:
        entry: Dict to validate.
        field_name: Name of the field.
        expected_type: Expected Python type (str, int, float, list, dict, bool).
        entry_index: Index of the entry in the list (for error messages).
        entity_name: Name of the entity (e.g., "ExtractedTask") for error messages.
        is_required: If True, missing field raises an error. If False, missing field is skipped.

    Raises:
        SchemaValidationError: If the field is missing (when required) or has the wrong type.
    """
    if field_name not in entry:
        if is_required:
            raise SchemaValidationError(
                f"{entity_name}[{entry_index}] missing required field '{field_name}'"
            )
        return

    value = entry[field_name]
    if not isinstance(value, expected_type):
        raise SchemaValidationError(
            f"{entity_name}[{entry_index}].{field_name}: expected {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def validate_tasks_json(data: List[Dict[str, Any]]) -> None:
    """Validate a list of ExtractedTask dicts loaded from tasks.json.

    Args:
        data: List of task dicts.

    Raises:
        SchemaValidationError: If any entry fails validation.
    """
    if not isinstance(data, list):
        raise SchemaValidationError(f"tasks.json root must be a list, got {type(data).__name__}")

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise SchemaValidationError(f"ExtractedTask[{i}] must be a dict, got {type(entry).__name__}")

        _check_field(entry, "title", str, i, "ExtractedTask")
        _check_field(entry, "domain", str, i, "ExtractedTask")

        has_question = "question" in entry or "challenge" in entry
        has_input = "input" in entry or "passage" in entry
        if not has_question:
            raise SchemaValidationError(
                f"ExtractedTask[{i}] missing required field 'question' (or legacy 'challenge')"
            )
        if not has_input:
            raise SchemaValidationError(
                f"ExtractedTask[{i}] missing required field 'input' (or legacy 'passage')"
            )

        _check_field(entry, "acceptance_criteria", dict, i, "ExtractedTask", is_required=False)

        has_task_uid = "task_uid" in entry
        has_task_id = "task_id" in entry
        if not has_task_uid and not has_task_id:
            raise SchemaValidationError(
                f"ExtractedTask[{i}] missing required field 'task_uid' (or legacy 'task_id')"
            )

        _check_field(entry, "difficulty", str, i, "ExtractedTask", is_required=False)
        _check_field(entry, "query_type", str, i, "ExtractedTask", is_required=False)
        _check_field(entry, "extraction_method", str, i, "ExtractedTask", is_required=False)
        _check_field(entry, "source_document_uid", str, i, "ExtractedTask", is_required=False)


def validate_traces_jsonl(entries: List[Dict[str, Any]]) -> None:
    """Validate a list of ReasoningTrace dicts loaded from traces.jsonl.

    Args:
        entries: List of trace dicts (one per JSONL line).

    Raises:
        SchemaValidationError: If any entry fails validation.
    """
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SchemaValidationError(f"ReasoningTrace[{i}] must be a dict, got {type(entry).__name__}")

        has_task_uid = "task_uid" in entry
        has_task_id = "task_id" in entry
        if not has_task_uid and not has_task_id:
            raise SchemaValidationError(
                f"ReasoningTrace[{i}] missing required field 'task_uid' (or legacy 'task_id')"
            )

        _check_field(entry, "model", str, i, "ReasoningTrace")
        _check_field(entry, "system_prompt", str, i, "ReasoningTrace")
        _check_field(entry, "user_prompt", str, i, "ReasoningTrace")
        _check_field(entry, "response", str, i, "ReasoningTrace")
        _check_field(entry, "procedural_steps", list, i, "ReasoningTrace")
        _check_field(entry, "conclusion", str, i, "ReasoningTrace")
        _check_field(entry, "tokens", int, i, "ReasoningTrace")


def validate_skills_json(data: List[Dict[str, Any]]) -> None:
    """Validate a list of ExtractedSkill dicts loaded from skills.json.

    Args:
        data: List of skill dicts.

    Raises:
        SchemaValidationError: If any entry fails validation.
    """
    if not isinstance(data, list):
        raise SchemaValidationError(f"skills.json root must be a list, got {type(data).__name__}")

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise SchemaValidationError(f"ExtractedSkill[{i}] must be a dict, got {type(entry).__name__}")

        has_skill_uid = "skill_uid" in entry
        has_skill_id = "skill_id" in entry
        if not has_skill_uid and not has_skill_id:
            raise SchemaValidationError(
                f"ExtractedSkill[{i}] missing required field 'skill_uid' (or legacy 'skill_id')"
            )

        _check_field(entry, "name", str, i, "ExtractedSkill")
        _check_field(entry, "description", str, i, "ExtractedSkill", is_required=False)
        _check_field(entry, "procedure", list, i, "ExtractedSkill")
        _check_field(entry, "constraints", list, i, "ExtractedSkill", is_required=False)


def validate_verified_skills_json(data: List[Dict[str, Any]]) -> None:
    """Validate a list of verified skill dicts loaded from verified_skills.json.

    Verified skills have the same fields as ExtractedSkill plus is_valid
    and defects.

    Args:
        data: List of verified skill dicts.

    Raises:
        SchemaValidationError: If any entry fails validation.
    """
    validate_skills_json(data)

    for i, entry in enumerate(data):
        _check_field(entry, "is_valid", bool, i, "VerifiedSkill")
        _check_field(entry, "defect_count", int, i, "VerifiedSkill")
        _check_field(entry, "defects", list, i, "VerifiedSkill")
