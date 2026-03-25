"""
harness.py

SkillMix evaluation harness: evaluate composed skills using LLM-as-judge.
Tests whether models can apply composed skill procedures to tasks.

Usage:
    python -m c3_skillmix.harness --skills-dir composed-skills/ --tasks tasks.json --provider openai --model qwen3:0.6b --base-url http://localhost:11434/v1
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from c2_evaluation.skill_injection import format_skill_for_system_prompt, get_default_system_prompt
from c2_evaluation.llm_judge import LLMJudgeEvaluator, JudgeResult


@dataclass
class SkillMixEpisode:
    """Result of evaluating one composed skill on one task with one model."""

    task_uid: str
    skill_name: str
    model: str
    condition: str         # baseline, skill_injected
    response: str
    judge_result: Optional[JudgeResult]
    tokens: int
    elapsed_s: float


def run_skillmix_episode(
    task: dict,
    skill_content: Optional[str],
    skill_name: str,
    model_provider,
    judge: Optional[LLMJudgeEvaluator],
    verbose: bool = False,
) -> SkillMixEpisode:
    """Run a single SkillMix episode: model answers task, judge evaluates.

    Args:
        task: Dict with passage, challenge, acceptance_criteria
        skill_content: Skill markdown to inject (None for baseline)
        skill_name: Name of skill being tested
        model_provider: LLM provider for the model being evaluated
        judge: LLM judge evaluator (or None to skip judging)
        verbose: Print progress

    Returns:
        SkillMixEpisode with results
    """
    domain = task.get("domain", "reading_comprehension")
    base_prompt = get_default_system_prompt(domain)

    if skill_content:
        system_prompt = format_skill_for_system_prompt(base_prompt, skill_name, skill_content)
        condition = "skill_injected"
    else:
        system_prompt = base_prompt
        condition = "baseline"

    passage = task.get("passage", task.get("input", ""))
    challenge = task.get("challenge", task.get("question", ""))
    user_prompt = f"Passage: {passage}\n\nChallenge: {challenge}"
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
        content = result.message.get("content", "")
        if isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            response_text = "".join(parts)
    except Exception as e:
        if verbose:
            print(f"    ERROR: {e}")

    elapsed = time.time() - start

    judge_result = None
    if judge and response_text:
        judge_result = judge.evaluate(
            response=response_text,
            passage=passage,
            challenge=challenge,
            acceptance_criteria=task.get("acceptance_criteria", {}),
        )

    model_name = getattr(model_provider, "model_name", getattr(model_provider, "model", "unknown"))

    return SkillMixEpisode(
        task_uid=task.get("task_uid", task.get("task_id", "")),
        skill_name=skill_name,
        model=model_name,
        condition=condition,
        response=response_text,
        judge_result=judge_result,
        tokens=tokens,
        elapsed_s=round(elapsed, 2),
    )
