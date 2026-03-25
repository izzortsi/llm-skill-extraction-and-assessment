"""
trace_capturer.py

Read LinearHarness capture files and convert to ReasoningTrace objects.
Phase 2 of the skill distillation pipeline: Tasks -> Procedural Traces.

LinearHarness captures full agent episodes (multi-step reasoning with tools).
This module reads those captures and extracts structured procedural traces
for downstream skill extraction.

Usage:
    python -m c2_skill_extraction.trace_capturer --capture-dir ./captures --output traces.jsonl
    python -m c2_skill_extraction.trace_capturer --capture-dir ./captures --output traces.jsonl --category science
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from c0_utils.text_utils import (
    extract_thinking as _c0_extract_thinking,
    strip_markdown_fences as _c0_strip_markdown_fences,
)


@dataclass
class ReasoningTrace:
    """A captured procedural trace from running a task on a model.

    Each step is a procedural action (what was done), not a declarative
    justification (why it was done). The trace captures the sequence of
    actions taken to arrive at the answer.
    """

    task_uid: str
    model: str
    system_prompt: str
    user_prompt: str
    response: str                    # full model response (raw JSON or text)
    procedural_steps: List[str]      # procedural actions (what was done, in order)
    conclusion: str
    tokens: int
    elapsed_s: float
    prompt_tokens: int = 0           # input/prompt token count
    completion_tokens: int = 0       # output/completion token count
    thinking: str = ""               # content of <think>...</think> blocks
    raw_steps: List[dict] = field(default_factory=list)  # raw parsed step dicts before normalization
    extraction_method: str = ""


VALID_PRIMITIVES = {"observe", "distinguish", "enumerate", "compare", "select", "apply"}


# Delegate to c0_utils canonical implementations; keep names for backwards compatibility.
extract_thinking = _c0_extract_thinking
_strip_markdown_fences = _c0_strip_markdown_fences


def _parse_structured_trace(response_text: str) -> tuple:
    """Parse structured JSON trace from model response.

    Returns (procedural_steps: List[str], conclusion: str).
    Each step is formatted as "PRIMITIVE: action description".

    Falls back to best-effort extraction if JSON parsing fails.
    """
    json_text = _strip_markdown_fences(response_text)
    data = None

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        # Fallback: try to find JSON object in the response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass

    if data is None or not isinstance(data, dict):
        # Final fallback: treat whole response as unstructured
        return _fallback_parse(response_text)

    steps = []
    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    for step_obj in raw_steps:
        if isinstance(step_obj, dict):
            primitive = str(step_obj.get("primitive", "")).lower().strip()
            action = str(step_obj.get("action", "")).strip()
            if primitive in VALID_PRIMITIVES and action:
                steps.append(f"{primitive.upper()}: {action}")
            elif action:
                steps.append(action)
        elif isinstance(step_obj, str):
            steps.append(step_obj)

    conclusion = str(data.get("conclusion", "")).strip()

    if not steps:
        return _fallback_parse(response_text)

    return steps, conclusion


def _fallback_parse(response_text: str) -> tuple:
    """Fallback: extract steps from unstructured text when JSON parsing fails.

    Returns (procedural_steps, conclusion) with best-effort extraction.
    """
    lines = [line.strip() for line in response_text.split("\n") if line.strip()]

    # Try to find a conclusion marker
    conclusion = ""
    step_lines = []
    for line in lines:
        lower = line.lower()
        if lower.startswith(("conclusion:", "in conclusion", "therefore:", "answer:")):
            conclusion = line.split(":", 1)[-1].strip() if ":" in line else line
        else:
            step_lines.append(line)

    if not conclusion and step_lines:
        conclusion = step_lines[-1]
        step_lines = step_lines[:-1]

    # Take up to 10 lines as pseudo-steps
    steps = step_lines[:10] if step_lines else [response_text[:500]]

    return steps, conclusion


# ---------------------------------------------------------------------------
# JSONL serialization
# ---------------------------------------------------------------------------

def save_traces(traces: List[ReasoningTrace], output_path: Path) -> None:
    """Save reasoning traces to a JSONL file (one trace per line).

    Args:
        traces: List of ReasoningTrace objects
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for trace in traces:
            record = {
                "task_uid": trace.task_uid,
                "model": trace.model,
                "system_prompt": trace.system_prompt,
                "user_prompt": trace.user_prompt,
                "response": trace.response,
                "procedural_steps": trace.procedural_steps,
                "conclusion": trace.conclusion,
                "tokens": trace.tokens,
                "elapsed_s": trace.elapsed_s,
                "prompt_tokens": trace.prompt_tokens,
                "completion_tokens": trace.completion_tokens,
                "thinking": trace.thinking,
                "raw_steps": trace.raw_steps,
                "extraction_method": trace.extraction_method,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_traces(filepath: Path) -> List[ReasoningTrace]:
    """Load reasoning traces from a JSONL file.

    Args:
        filepath: Path to traces JSONL file

    Returns:
        List of ReasoningTrace objects
    """
    # TODO: import from llm-skills.llm-providers
    # from c1_providers.schema_validator import validate_traces_jsonl

    raw_entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_entries.append(json.loads(line))

    # validate_traces_jsonl(raw_entries)  # TODO: import from llm-skills.llm-providers

    traces = []
    for d in raw_entries:
        traces.append(ReasoningTrace(
            task_uid=d.get("task_uid", d.get("task_id", "")),
            model=d["model"],
            system_prompt=d["system_prompt"],
            user_prompt=d["user_prompt"],
            response=d["response"],
            procedural_steps=d["procedural_steps"],
            conclusion=d["conclusion"],
            tokens=d["tokens"],
            elapsed_s=d["elapsed_s"],
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            thinking=d.get("thinking", ""),
            raw_steps=d.get("raw_steps", []),
            extraction_method=d.get("extraction_method", ""),
        ))

    return traces


# ---------------------------------------------------------------------------
# LinearHarness capture file readers
# ---------------------------------------------------------------------------

def load_harness_steps(steps_path: Path) -> List[dict]:
    """Load raw step records from a LinearHarness steps JSONL file.

    Each line is a JSON object with fields: type, run_id, episode_id, step,
    model, request_messages, response_message, response_reasoning, usage,
    raw_response.

    Args:
        steps_path: Path to a steps JSONL file
            (named like YYMMDD-steps-{category}-{run_id}-{NNNN}.jsonl)

    Returns:
        List of step record dicts, in file order.
    """
    records = []
    with open(steps_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _find_steps_file_for_episode(episode_path: Path) -> Path:
    """Given an episode JSONL path, find the matching steps JSONL file.

    Episode files:  YYMMDD-episode-{category}-{run_id}-{NNNN}.jsonl
    Steps files:    YYMMDD-steps-{category}-{run_id}-{NNNN}.jsonl

    Replace '-episode-' with '-steps-' in the filename.
    """
    name = episode_path.name.replace("-episode-", "-steps-", 1)
    return episode_path.parent / name


def _extract_response_from_messages(messages: List[dict]) -> str:
    """Extract the last assistant message content from a message list.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        The text content of the last assistant message, or empty string.
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                return "".join(parts)
    return ""


def _extract_response_from_steps(steps: List[dict]) -> str:
    """Extract full response text from step records.

    Concatenates response_message content from all steps.
    If only one step, returns that step's response_message content.

    Args:
        steps: List of step record dicts.

    Returns:
        Combined response text from all steps.
    """
    parts = []
    for step in steps:
        resp_msg = step.get("response_message", {})
        if isinstance(resp_msg, dict):
            content = resp_msg.get("content", "")
            if isinstance(content, str) and content:
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            parts.append(text)
    return "\n".join(parts)


def load_harness_episode(episode_path: Path) -> List[ReasoningTrace]:
    """Load traces from a single LinearHarness episode JSONL file.

    Each line in the episode file is an episode record with fields:
    type, run_id, episode_id, problem_id, domain, model, messages,
    steps, total_tokens, elapsed_s, verification.

    For each episode record, extracts:
    - problem_id as task_id
    - model
    - system prompt from first message (if role=='system')
    - user prompt from messages
    - total_tokens and elapsed_s
    - Response text from corresponding steps file (if available) or
      from the messages list (last assistant message)

    The response is parsed via _parse_structured_trace() to extract
    procedural_steps and conclusion.

    Args:
        episode_path: Path to an episode JSONL file.

    Returns:
        List of ReasoningTrace objects (one per episode record in the file).
    """
    traces = []

    # Try to load corresponding steps file for richer response data
    steps_path = _find_steps_file_for_episode(episode_path)
    step_records = []
    if steps_path.exists():
        step_records = load_harness_steps(steps_path)

    # Index step records by episode_id for lookup
    steps_by_episode = {}
    for rec in step_records:
        eid = rec.get("episode_id", "")
        if eid not in steps_by_episode:
            steps_by_episode[eid] = []
        steps_by_episode[eid].append(rec)

    with open(episode_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)

            task_uid = episode.get("problem_id", "")
            model = episode.get("model", "")
            messages = episode.get("messages", [])
            total_tokens = episode.get("total_tokens", 0)
            elapsed_s = episode.get("elapsed_s", 0.0)
            episode_id = episode.get("episode_id", "")

            # Extract system prompt from first message
            system_prompt = ""
            user_prompt = ""
            for msg in messages:
                if msg.get("role") == "system" and not system_prompt:
                    content = msg.get("content", "")
                    system_prompt = content if isinstance(content, str) else ""
                elif msg.get("role") == "user" and not user_prompt:
                    content = msg.get("content", "")
                    user_prompt = content if isinstance(content, str) else ""

            # Get response text: prefer steps file, fall back to messages
            episode_steps = steps_by_episode.get(episode_id, [])
            if episode_steps:
                response_text = _extract_response_from_steps(episode_steps)
            else:
                response_text = _extract_response_from_messages(messages)

            procedural_steps, conclusion = _parse_structured_trace(response_text)

            traces.append(ReasoningTrace(
                task_uid=task_uid,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response_text,
                procedural_steps=procedural_steps,
                conclusion=conclusion,
                tokens=total_tokens,
                elapsed_s=round(elapsed_s, 2),
            ))

    return traces


def load_harness_traces(capture_dir: Path, category: str = "") -> List[ReasoningTrace]:
    """Read LinearHarness JSONL capture files and return ReasoningTrace objects.

    Scans capture_dir for episode JSONL files and converts each episode
    record into a ReasoningTrace.

    LinearHarness files are named like:
        YYMMDD-episode-{category}-{run_id}-{NNNN}.jsonl
        YYMMDD-steps-{category}-{run_id}-{NNNN}.jsonl

    Args:
        capture_dir: Directory containing LinearHarness capture files.
        category: If non-empty, only load episode files whose name contains
            this category string. Empty string matches all episode files.

    Returns:
        List of ReasoningTrace objects from all matching episode files.
    """
    capture_dir = Path(capture_dir)
    if not capture_dir.is_dir():
        return []

    # Find episode JSONL files
    episode_files = sorted(capture_dir.glob("*-episode-*.jsonl"))

    # Filter by category if specified
    if category:
        episode_files = [
            f for f in episode_files
            if f"-episode-{category}-" in f.name
        ]

    traces = []
    for ep_file in episode_files:
        traces.extend(load_harness_episode(ep_file))

    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and convert reasoning traces")
    parser.add_argument("--capture-dir", type=Path, required=True, help="Directory with LinearHarness capture files")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--category", type=str, default="", help="Filter by category")
    args = parser.parse_args()

    traces = load_harness_traces(args.capture_dir, category=args.category)
    save_traces(traces, args.output)
    print(f"Loaded {len(traces)} traces -> {args.output}")


if __name__ == "__main__":
    main()
