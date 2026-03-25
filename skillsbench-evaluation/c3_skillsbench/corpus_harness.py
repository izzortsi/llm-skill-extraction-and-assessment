"""
corpus_harness.py

SkillsBench evaluation harness for general domain tasks.
Three scaffolding modes:
  - singlecall: one LLM call per episode (default)
  - stepwise:   multi-turn, model reasons one step at a time using action primitives
  - guided:     multi-turn, each turn follows one step of the skill's procedure

Supports multiple models in a single run.

Usage:
    python -m c3_skillsbench.corpus_harness \
      --tasks tasks.json --skills skills.json \
      --models qwen3:0.6b,qwen2.5:1.5b,llama3.2:1b \
      --base-url http://host.docker.internal:11434/v1 \
      --mode guided
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

from openai import OpenAI

from c0_config.skill_injection import (
    format_skill_for_user_message,
    get_default_system_prompt,
)
from c2_evaluation.llm_judge import LLMJudgeEvaluator, JudgeResult
from c1_types.extracted_task import ExtractedTask
from c1_types.extracted_skill import ExtractedSkill


# ---------------------------------------------------------------------------
# OpenAI-SDK-based provider (replaces c1_providers.providers.create_provider)
# ---------------------------------------------------------------------------

_LMPROXY_BASE_URL = os.environ.get("LMPROXY_BASE_URL", "http://localhost:8080")


def _create_openai_provider(model: str, base_url: str = "", api_key: str = "lmproxy"):
    """Return a thin wrapper around the OpenAI SDK pointed at lmproxy."""
    effective_base = base_url or _LMPROXY_BASE_URL
    client = OpenAI(base_url=effective_base, api_key=api_key)

    class _Provider:
        def __init__(self):
            self.model_name = model
            self._client = client

        def chat(self, messages, tools=None):
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
                "total_tokens": resp.usage.total_tokens if resp.usage else 0,
            }

            class _R:
                pass
            r = _R()
            r.message = {"role": "assistant", "content": content}
            r.usage = usage
            return r

    return _Provider()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepTrace:
    """One step in a stepwise or guided episode."""

    step_index: int
    user_prompt: str
    raw_response: str
    parsed_action: str
    thinking: str = ""
    tokens: int = 0
    elapsed_s: float = 0.0
    selected_skill: str = ""


@dataclass
class CorpusEpisode:
    """Result of one corpus evaluation episode."""

    task_uid: str
    model: str
    condition: str
    skill_name: str
    mode: str
    response: str
    passed: bool
    score: float
    tokens: int
    elapsed_s: float
    judge_rationale: str = ""
    steps: List[StepTrace] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Response text extraction
# ---------------------------------------------------------------------------

def _extract_response_text(message: Dict[str, Any]) -> str:
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


def _extract_thinking(text: str) -> tuple:
    """Extract <think>...</think> reasoning and remaining text."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        remaining = text[:think_match.start()] + text[think_match.end():]
        return thinking, remaining.strip()
    return "", text


# ---------------------------------------------------------------------------
# Mode: singlecall
# ---------------------------------------------------------------------------

def run_singlecall_episode(
    task: ExtractedTask,
    model_provider,
    judge: LLMJudgeEvaluator,
    model_name: str = "",
    skill: Optional[ExtractedSkill] = None,
    verbose: bool = False,
) -> CorpusEpisode:
    """Run one singlecall episode: one LLM call, judge scores the response."""
    system_prompt = get_default_system_prompt(task.domain)
    user_prompt = f"Passage: {task.passage}\n\nChallenge: {task.challenge}"

    if skill:
        user_prompt = format_skill_for_user_message(user_prompt, skill)
        condition = "curated"
        skill_name = skill.name
    else:
        condition = "baseline"
        skill_name = ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    start = time.time()
    tokens = 0
    response_text = ""

    try:
        result = model_provider.chat(messages)
        tokens = result.usage.get("total_tokens", 0)
        response_text = _extract_response_text(result.message)
    except Exception as e:
        if verbose:
            print(f"    ERROR: {e}")

    elapsed = time.time() - start

    judge_result = judge.evaluate(
        response=response_text,
        passage=task.passage,
        challenge=task.challenge,
        acceptance_criteria=task.acceptance_criteria,
        query_type=getattr(task, "query_type", "FREE_FORM"),
    )

    resolved_name = model_name or getattr(model_provider, "model_name",
                                           getattr(model_provider, "model", "unknown"))

    return CorpusEpisode(
        task_uid=task.task_uid,
        model=resolved_name,
        condition=condition,
        skill_name=skill_name,
        mode="singlecall",
        response=response_text,
        passed=judge_result.passed,
        score=judge_result.score,
        tokens=tokens,
        elapsed_s=round(elapsed, 2),
        judge_rationale=judge_result.rationale,
    )


# ---------------------------------------------------------------------------
# Mode: stepwise
# ---------------------------------------------------------------------------

_STEPWISE_SYSTEM = """\
You are an expert analyst. You will solve the challenge one reasoning step at a time.

Each turn, output exactly ONE reasoning step using one of these action primitives:
  OBSERVE: read or identify something specific in the passage
  DISTINGUISH: classify, differentiate, or categorize elements
  ENUMERATE: list possible options or relevant items
  COMPARE: evaluate or contrast between options
  SELECT: choose one interpretation or answer
  APPLY: produce a conclusion or synthesis

Format: PRIMITIVE: description of what you do

When you have reached your final answer, output: CONCLUDE: your final answer

Rules:
- ONE step per turn, nothing else (no explanation beyond the action)
- Each step must reference specific content from the passage
- Use 3-8 steps total
"""


def _format_stepwise_state(
    passage: str,
    challenge: str,
    steps_so_far: List[str],
    skill_context: str = "",
) -> str:
    """Format the current reasoning state as a user message for stepwise mode."""
    parts = []
    if skill_context:
        parts.append(skill_context)
        parts.append("")
    parts.append(f"Passage: {passage}")
    parts.append(f"\nChallenge: {challenge}")
    parts.append("\nReasoning so far:")
    if steps_so_far:
        for i, step in enumerate(steps_so_far, 1):
            parts.append(f"  {i}. {step}")
    else:
        parts.append("  (no steps yet)")
    parts.append("\nWrite the next reasoning step:")
    return "\n".join(parts)


def _parse_stepwise_response(text: str) -> str:
    """Extract a single reasoning step from model response."""
    _, clean = _extract_thinking(text)
    clean = clean.strip()

    if re.match(r"^CONCLUDE:", clean, re.IGNORECASE):
        return clean

    # Check for primitive-prefixed lines
    for line in clean.split("\n"):
        line = line.strip()
        if re.match(r"^(OBSERVE|DISTINGUISH|ENUMERATE|COMPARE|SELECT|APPLY|CONCLUDE):", line, re.IGNORECASE):
            return line

    # Fallback: return first non-empty line if short
    if len(clean) < 300:
        first_line = clean.split("\n")[0].strip()
        if first_line:
            return first_line

    return ""


def run_stepwise_episode(
    task: ExtractedTask,
    model_provider,
    judge: LLMJudgeEvaluator,
    model_name: str = "",
    skill: Optional[ExtractedSkill] = None,
    max_steps: int = 8,
    verbose: bool = False,
) -> CorpusEpisode:
    """Run a stepwise episode: model reasons one step at a time.

    Each turn the model sees passage + challenge + previous steps, and outputs
    one action primitive step. After CONCLUDE or max_steps, the full chain
    is judged.
    """
    start_time = time.time()

    if skill:
        condition = "curated"
        skill_name = skill.name
        skill_context = f"Skill hint ({skill.name}): {skill.description}"
    else:
        condition = "baseline"
        skill_name = ""
        skill_context = ""

    messages = [{"role": "system", "content": _STEPWISE_SYSTEM}]

    reasoning_steps = []
    step_traces = []
    total_tokens = 0

    for step_idx in range(max_steps):
        step_start = time.time()

        user_msg = _format_stepwise_state(
            task.passage, task.challenge, reasoning_steps, skill_context,
        )
        messages.append({"role": "user", "content": user_msg})

        step_tokens = 0
        response_text = ""
        try:
            result = model_provider.chat(messages)
            step_tokens = result.usage.get("total_tokens", 0)
            total_tokens += step_tokens
            response_text = _extract_response_text(result.message)
        except Exception as e:
            if verbose:
                print(f"      step {step_idx + 1} ERROR: {e}")
            break

        step_elapsed = time.time() - step_start
        thinking, _ = _extract_thinking(response_text)
        parsed = _parse_stepwise_response(response_text)

        if verbose:
            print(f"      step {step_idx + 1}: {parsed[:100]}")

        step_traces.append(StepTrace(
            step_index=step_idx + 1,
            user_prompt=user_msg,
            raw_response=response_text,
            parsed_action=parsed,
            thinking=thinking,
            tokens=step_tokens,
            elapsed_s=round(step_elapsed, 2),
        ))

        messages.append({"role": "assistant", "content": response_text})

        if not parsed or parsed.upper().startswith("CONCLUDE:"):
            if parsed:
                reasoning_steps.append(parsed)
            break

        reasoning_steps.append(parsed)

    elapsed = time.time() - start_time

    # Assemble full response for judge
    full_response = "\n".join(f"{i}. {s}" for i, s in enumerate(reasoning_steps, 1))

    judge_result = judge.evaluate(
        response=full_response,
        passage=task.passage,
        challenge=task.challenge,
        acceptance_criteria=task.acceptance_criteria,
    )

    resolved_name = model_name or getattr(model_provider, "model_name",
                                           getattr(model_provider, "model", "unknown"))

    return CorpusEpisode(
        task_uid=task.task_uid,
        model=resolved_name,
        condition=condition,
        skill_name=skill_name,
        mode="stepwise",
        response=full_response,
        passed=judge_result.passed,
        score=judge_result.score,
        tokens=total_tokens,
        elapsed_s=round(elapsed, 2),
        judge_rationale=judge_result.rationale,
        steps=step_traces,
    )


# ---------------------------------------------------------------------------
# Mode: guided
# ---------------------------------------------------------------------------

_GUIDED_SYSTEM = """\
You are an expert analyst. You will solve the challenge by following a \
procedure one step at a time.

Each turn you receive a specific instruction. Carry it out using evidence \
from the passage and write your finding in 2-4 sentences. Reference \
specific words, phrases, or claims from the passage.

On the final turn you will be asked to synthesize your findings into a \
conclusion. Write a concise paragraph that directly answers the challenge.
"""


def _format_guided_user(
    passage: str,
    challenge: str,
    procedure_step: str,
    step_index: int,
    total_steps: int,
    findings_so_far: List[str],
) -> str:
    """Format the user message for one guided-mode turn."""
    parts = [f"Passage: {passage}", f"\nChallenge: {challenge}"]

    if findings_so_far:
        parts.append("\nFindings so far:")
        for i, finding in enumerate(findings_so_far, 1):
            parts.append(f"  {i}. {finding}")

    parts.append(f"\nStep {step_index} of {total_steps}: {procedure_step}")
    return "\n".join(parts)


def run_guided_episode(
    task: ExtractedTask,
    model_provider,
    judge: LLMJudgeEvaluator,
    model_name: str = "",
    skill: Optional[ExtractedSkill] = None,
    verbose: bool = False,
) -> CorpusEpisode:
    """Run a guided episode: one LLM call per skill procedure step.

    Each turn gives the model one concrete instruction from the skill's
    procedure. The final turn asks for a synthesized conclusion. Without
    a skill, falls back to a single-call episode.
    """
    if not skill or not skill.procedure:
        return run_singlecall_episode(
            task, model_provider, judge, model_name=model_name, verbose=verbose,
        )

    start_time = time.time()

    procedure = list(skill.procedure)
    procedure.append(
        "Synthesize your findings into a single paragraph that directly "
        "answers the challenge."
    )
    total_steps = len(procedure)

    findings: List[str] = []
    step_traces: List[StepTrace] = []
    total_tokens = 0

    for step_idx, step_instruction in enumerate(procedure, 1):
        step_start = time.time()

        user_msg = _format_guided_user(
            task.passage, task.challenge, step_instruction,
            step_idx, total_steps, findings,
        )
        messages = [
            {"role": "system", "content": _GUIDED_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        step_tokens = 0
        response_text = ""
        try:
            result = model_provider.chat(messages)
            step_tokens = result.usage.get("total_tokens", 0)
            total_tokens += step_tokens
            response_text = _extract_response_text(result.message)
        except Exception as e:
            if verbose:
                print(f"      step {step_idx} ERROR: {e}")
            break

        step_elapsed = time.time() - step_start
        thinking, clean = _extract_thinking(response_text)
        finding = clean.strip()

        if verbose:
            print(f"      step {step_idx}/{total_steps}: {finding[:100]}")

        step_traces.append(StepTrace(
            step_index=step_idx,
            user_prompt=user_msg,
            raw_response=response_text,
            parsed_action=finding,
            thinking=thinking,
            tokens=step_tokens,
            elapsed_s=round(step_elapsed, 2),
            selected_skill=skill.name,
        ))

        findings.append(finding)

    elapsed = time.time() - start_time

    full_response = "\n".join(f"{i}. {s}" for i, s in enumerate(findings, 1))

    judge_result = judge.evaluate(
        response=full_response,
        passage=task.passage,
        challenge=task.challenge,
        acceptance_criteria=task.acceptance_criteria,
    )

    resolved_name = model_name or getattr(model_provider, "model_name",
                                           getattr(model_provider, "model", "unknown"))

    return CorpusEpisode(
        task_uid=task.task_uid,
        model=resolved_name,
        condition="curated",
        skill_name=skill.name,
        mode="guided",
        response=full_response,
        passed=judge_result.passed,
        score=judge_result.score,
        tokens=total_tokens,
        elapsed_s=round(elapsed, 2),
        judge_rationale=judge_result.rationale,
        steps=step_traces,
    )


# ---------------------------------------------------------------------------
# Skill-task matching (one-to-one mode)
# ---------------------------------------------------------------------------

def build_task_skill_map(
    tasks: List[ExtractedTask],
    skills: List[ExtractedSkill],
) -> Dict[str, ExtractedSkill]:
    """Build a 1:1 mapping from task_uid to its best-matching skill.

    Uses source_task_uids: if a skill lists task_uid in its source_task_uids,
    that skill is the match. When multiple skills match, the function picks
    the skill with the fewest source tasks (most specific to the task).

    Returns:
        Dict mapping task_uid -> ExtractedSkill (only for matched tasks)
    """
    task_to_skill: Dict[str, ExtractedSkill] = {}

    for task in tasks:
        candidates = []
        for skill in skills:
            source_uids = getattr(skill, "source_task_uids", []) or []
            if task.task_uid in source_uids:
                candidates.append(skill)

        if candidates:
            best = min(candidates, key=lambda s: len(getattr(s, "source_task_uids", []) or []))
            task_to_skill[task.task_uid] = best

    return task_to_skill


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_corpus_evaluation(
    tasks: List[ExtractedTask],
    skills: List[ExtractedSkill],
    model_provider,
    judge: LLMJudgeEvaluator,
    model_name: str = "",
    mode: str = "singlecall",
    max_steps: int = 8,
    cross_task: bool = False,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> List[CorpusEpisode]:
    """Run full corpus evaluation for one model.

    For singlecall, stepwise, and guided: runs baseline + matching skill per task
    (one-to-one via source_task_uids). Unmatched tasks run baseline only.
    With cross_task=True: runs baseline + ALL skills per task (N*M episodes).

    Args:
        tasks: List of ExtractedTask objects
        skills: List of ExtractedSkill objects
        model_provider: LLM provider for the model being evaluated
        judge: LLMJudgeEvaluator for scoring
        model_name: Display name for the model
        mode: Scaffolding mode (singlecall, stepwise, guided)
        max_steps: Max steps per episode for stepwise
        cross_task: If True, inject every skill into every task (not just matching)
        output_path: If provided, save episodes to JSON
        verbose: Print progress

    Returns:
        List of CorpusEpisode objects
    """
    episodes = []

    if mode == "guided":
        episode_fn = run_guided_episode
    elif mode == "stepwise":
        episode_fn = run_stepwise_episode
    else:
        episode_fn = run_singlecall_episode

    if cross_task:
        task_skill_map = None
        total = len(tasks) * (1 + len(skills))
    else:
        task_skill_map = build_task_skill_map(tasks, skills)
        matched = sum(1 for t in tasks if t.task_uid in task_skill_map)
        total = len(tasks) + matched
        if verbose:
            print(f"  one-to-one mode: {matched}/{len(tasks)} tasks have matching skills")

    count = 0

    for task in tasks:
        count += 1
        if verbose:
            print(f"  [{count}/{total}] {task.task_uid} baseline")
        ep = episode_fn(task, model_provider, judge, model_name=model_name,
                        verbose=verbose)
        episodes.append(ep)

        if cross_task:
            for skill in skills:
                count += 1
                if verbose:
                    print(f"  [{count}/{total}] {task.task_uid} + {skill.name}")
                ep = episode_fn(task, model_provider, judge, model_name=model_name,
                                skill=skill, verbose=verbose)
                episodes.append(ep)
        else:
            matching = task_skill_map.get(task.task_uid) if task_skill_map else None
            if matching:
                count += 1
                if verbose:
                    print(f"  [{count}/{total}] {task.task_uid} + {matching.name}")
                ep = episode_fn(task, model_provider, judge, model_name=model_name,
                                skill=matching, verbose=verbose)
                episodes.append(ep)
            elif verbose:
                print(f"  [{count}/{total}] {task.task_uid} (no matching skill, baseline only)")

    if output_path:
        _save_episodes(episodes, output_path)

    return episodes


def run_multi_model_evaluation(
    tasks: List[ExtractedTask],
    skills: List[ExtractedSkill],
    model_names: List[str],
    provider_name: str,
    base_url: str,
    judge: LLMJudgeEvaluator,
    mode: str = "singlecall",
    max_steps: int = 8,
    cross_task: bool = False,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> List[CorpusEpisode]:
    """Run corpus evaluation across multiple models."""
    all_episodes = []

    for model_index, model_name in enumerate(model_names):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Model {model_index + 1}/{len(model_names)}: {model_name} (mode={mode})")
            print(f"{'='*60}")

        model_provider = _create_openai_provider(model_name, base_url=base_url)

        episodes = run_corpus_evaluation(
            tasks, skills, model_provider, judge,
            model_name=model_name, mode=mode, max_steps=max_steps,
            cross_task=cross_task, verbose=verbose,
        )
        all_episodes.extend(episodes)

        baseline = [e for e in episodes if e.condition == "baseline"]
        skilled = [e for e in episodes if e.condition == "curated"]
        bl_rate = sum(1 for e in baseline if e.passed) / len(baseline) if baseline else 0
        sk_rate = sum(1 for e in skilled if e.passed) / len(skilled) if skilled else 0
        print(f"  {model_name}: baseline={bl_rate:.1%} skill={sk_rate:.1%} delta={sk_rate - bl_rate:+.1%}")

    if output_path:
        _save_episodes(all_episodes, output_path)

    return all_episodes


def _save_episodes(episodes: List[CorpusEpisode], output_path: Path) -> None:
    """Save episodes to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for ep in episodes:
        entry = {
            "task_uid": ep.task_uid, "model": ep.model,
            "condition": ep.condition, "skill_name": ep.skill_name,
            "mode": ep.mode, "passed": ep.passed, "score": ep.score,
            "tokens": ep.tokens, "elapsed_s": ep.elapsed_s,
            "judge_rationale": ep.judge_rationale,
        }
        if ep.steps:
            entry["num_steps"] = len(ep.steps)
            entry["steps"] = [
                {
                    "step_index": s.step_index,
                    "parsed_action": s.parsed_action,
                    "selected_skill": s.selected_skill,
                    "tokens": s.tokens,
                    "elapsed_s": s.elapsed_s,
                }
                for s in ep.steps
            ]
        data.append(entry)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI (delegates to c4_cli.run_skillsbench; kept here for backwards compat)
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point -- delegates to c4_cli.run_skillsbench.main()."""
    from c4_cli.run_skillsbench import main as _cli_main
    _cli_main()


if __name__ == "__main__":
    main()
