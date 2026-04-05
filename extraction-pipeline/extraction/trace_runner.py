"""
trace_runner.py

Run extracted tasks through an LLM to capture reasoning traces.
Alternative to load-traces (which reads LinearHarness captures):
this module generates traces directly by prompting the model with
structured JSON output.

Usage:
    python -m c2_skill_extraction.trace_runner --tasks tasks.json --output traces.jsonl
    python -m c2_skill_extraction.trace_runner --tasks tasks.json --output traces.jsonl --provider anthropic --model claude-opus-4-6
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, List

from schemas.extracted_task import ExtractedTask, load_extracted_tasks, save_extracted_tasks
from extraction.trace_capturer import (
    ReasoningTrace,
    _parse_structured_trace,
    _strip_markdown_fences,
    extract_thinking,
    save_traces,
)


_TRACE_SYSTEM_PROMPT = """\
You are an expert analyst. Analyze the passage and answer the challenge.

Return ONLY valid JSON with this structure:
{
  "steps": [
    {"primitive": "observe", "action": "description of what you observed"},
    {"primitive": "distinguish", "action": "description of distinction made"},
    {"primitive": "compare", "action": "description of comparison"},
    {"primitive": "select", "action": "description of selection"},
    {"primitive": "apply", "action": "description of application"}
  ],
  "conclusion": "your final answer"
}

Valid primitives: observe, distinguish, enumerate, compare, select, apply.
Use 3-8 steps. Each step must name what you act on.\
"""


def validate_task_input(task: ExtractedTask) -> None:
    """Validate that a task has all required fields before trace capture.

    Raises:
        ValueError: If any required field is missing or empty.
    """
    required_fields = {
        "task_uid": task.task_uid,
        "question": task.question,
        "input": task.input,
        "output": task.output,
        "query_type": task.query_type,
    }
    for field_name, value in required_fields.items():
        if not value or not str(value).strip():
            raise ValueError(
                f"Task is missing required field '{field_name}': "
                f"task_uid={task.task_uid!r}, title={getattr(task, 'title', '')!r}"
            )


def _extract_raw_steps(response_text: str) -> List[dict]:
    """Extract raw step dicts from a structured JSON response before normalization.

    Args:
        response_text: Model response text (after thinking extraction).

    Returns:
        List of raw step dicts with 'primitive' and 'action' keys, or empty list
        if JSON parsing fails.
    """
    import json
    json_text = _strip_markdown_fences(response_text)
    data = None
    try:
        data = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(response_text[start:end])
            except (json.JSONDecodeError, ValueError):
                pass

    if data is None or not isinstance(data, dict):
        return []

    steps = data.get("steps", [])
    if not isinstance(steps, list):
        return []

    raw = []
    for step_obj in steps:
        if isinstance(step_obj, dict):
            raw.append({
                "primitive": str(step_obj.get("primitive", "")).lower().strip(),
                "action": str(step_obj.get("action", "")).strip(),
            })
    return raw


def _extract_response_text(message: dict) -> str:
    """Extract plain text from a provider response message."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def run_task_for_trace(
    task: ExtractedTask,
    provider: Any,
    verbose: bool = False,
) -> ReasoningTrace:
    """Run a single extracted task to capture a reasoning trace.

    Args:
        task: ExtractedTask with passage, challenge, acceptance_criteria
        provider: LLM provider with .chat(messages) method
        verbose: Print progress

    Returns:
        ReasoningTrace with procedural steps and conclusion
    """
    validate_task_input(task)

    user_prompt = (
        f"## Passage\n\n{task.passage}\n\n"
        f"## Challenge\n\n{task.challenge}"
    )

    messages = [
        {"role": "system", "content": _TRACE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if verbose:
        print(f"  Capturing trace for: {task.task_uid} ({task.title})")

    start_time = time.time()
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    response_text = ""
    usage = {}

    try:
        result = provider.chat(messages)
        usage = result.usage
        total_tokens = usage.get("total_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        response_text = _extract_response_text(result.message)
    except Exception as e:
        import logging
        logging.error("LLM call failed for task %s: %s", task.task_uid, e)
        raise RuntimeError(
            f"LLM call failed for task {task.task_uid}: {e}"
        ) from e

    elapsed = time.time() - start_time

    # Extract <think> blocks before parsing structured trace
    thinking, clean_response = extract_thinking(response_text)

    # Parse structured trace from cleaned response
    procedural_steps, conclusion = _parse_structured_trace(clean_response)

    # Extract raw step dicts before normalization
    raw_steps = _extract_raw_steps(clean_response)

    model_name = getattr(provider, "model_name", getattr(provider, "model", "unknown"))
    # TODO: import from llm-providers
    import re as _re
    short_model = _re.sub(r"^claude-", "", model_name)
    short_model = _re.sub(r"[:/]", "-", short_model)
    extraction_method = f"{short_model}-trace-capture-v1"

    return ReasoningTrace(
        task_uid=task.task_uid,
        model=model_name,
        system_prompt=_TRACE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response=response_text,
        procedural_steps=procedural_steps,
        conclusion=conclusion,
        tokens=total_tokens,
        elapsed_s=round(elapsed, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        thinking=thinking,
        raw_steps=raw_steps,
        extraction_method=extraction_method,
    )


def run_tasks_for_traces(
    tasks: List[ExtractedTask],
    provider: Any,
    verbose: bool = False,
) -> List[ReasoningTrace]:
    """Run multiple extracted tasks and collect reasoning traces.

    Args:
        tasks: List of ExtractedTask objects
        provider: LLM provider with .chat(messages) method
        verbose: Print progress

    Returns:
        List of ReasoningTrace objects (one per task)
    """
    traces = []
    for i, task in enumerate(tasks):
        if verbose:
            print(f"[{i+1}/{len(tasks)}] {task.task_uid}")
        trace = run_task_for_trace(task, provider, verbose=verbose)
        traces.append(trace)
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tasks through LLM to capture reasoning traces")
    parser.add_argument("--tasks", "-t", type=Path, required=True, help="Input tasks JSON file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output traces JSONL path")
    parser.add_argument("--provider", type=str, default="anthropic", help="LLM provider (anthropic, openai, mock)")
    parser.add_argument("--model", type=str, default="claude-opus-4-6", help="Model to use")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.tasks.exists():
        raise FileNotFoundError(f"Tasks file not found: {args.tasks}")

    # TODO: import from llm-providers
    from providers.providers import create_provider  # noqa: requires llm-providers on sys.path
    provider = create_provider(args.provider, args.model)

    tasks = load_extracted_tasks(args.tasks)
    if args.verbose:
        print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    traces = run_tasks_for_traces(tasks, provider, verbose=args.verbose)
    save_traces(traces, args.output)
    print(f"Saved {len(traces)} traces to {args.output}")


if __name__ == "__main__":
    main()
