"""
llm_judge.py

LLM-as-judge evaluator for reading comprehension tasks.
Uses a powerful model (e.g., Opus) to score responses against acceptance criteria,
replacing or supplementing keyword-based evaluation.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# Inline mock provider (replaces c1_providers.mock_provider.MockProvider)
# ---------------------------------------------------------------------------

class _MockProvider:
    def __init__(self, model="mock-model", seed=42):
        self.model_name = model

    def chat(self, messages, tools=None):
        class _R:
            message = {"role": "assistant", "content": "Mock response."}
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return _R()


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


JUDGE_PROMPT_TEMPLATE = """You are a precise evaluator for a reading comprehension benchmark.

## Task Passage
{passage}

## Challenge
{challenge}

## Acceptance Criteria

Must identify:
{must_identify_formatted}

Correct conclusion: {correct_conclusion}

## Model Response
{response}

## Instructions

Evaluate whether the model's response adequately addresses the challenge. Score each criterion:

1. For each "must identify" item, determine if the response identifies or addresses it (even with different wording/synonyms).
2. Determine if the response reaches the correct conclusion (even if phrased differently).

Return ONLY valid JSON (no markdown, no explanation):

{{
  "score": <float 0.0 to 1.0>,
  "passed": <true if score >= 0.6>,
  "criteria_met": <number of criteria met out of total>,
  "criteria_total": <total criteria count>,
  "rationale": "<1-2 sentence explanation>"
}}
"""


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""

    passed: bool
    score: float
    rationale: str
    criteria_met: int = 0
    criteria_total: int = 0
    raw_response: str = ""


class LLMJudgeEvaluator:
    """Evaluates reading comprehension responses using an LLM as judge."""

    def __init__(self, provider):
        """Initialize with an LLM provider instance.

        Args:
            provider: Any object with a chat(messages) method returning
                      a result with result.message["content"]
        """
        self._provider = provider

    def evaluate(
        self,
        response: str,
        passage: str,
        challenge: str,
        acceptance_criteria: dict,
        query_type: str = "FREE_FORM",
    ) -> JudgeResult:
        """Evaluate a model response against acceptance criteria.

        Args:
            response: The model's response text
            passage: The task passage
            challenge: The task challenge/question
            acceptance_criteria: Dict with must_identify and correct_conclusion

        Returns:
            JudgeResult with score, passed, and rationale
        """
        if not acceptance_criteria:
            return JudgeResult(passed=True, score=1.0, rationale="No criteria to evaluate")

        # Deterministic scoring for non-free-form query types
        if query_type in ("YES_NO", "YES_NO_VERIFICATION", "SINGLE_WORD", "RANKING"):
            return _score_deterministic(response, acceptance_criteria, query_type)

        must_identify = acceptance_criteria.get("must_identify", [])
        correct_conclusion = acceptance_criteria.get("correct_conclusion", "")

        must_identify_formatted = "\n".join(f"- {item}" for item in must_identify)
        if not must_identify_formatted:
            must_identify_formatted = "(none specified)"

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            passage=passage,
            challenge=challenge,
            must_identify_formatted=must_identify_formatted,
            correct_conclusion=correct_conclusion or "(none specified)",
            response=response,
        )

        messages = [
            {"role": "user", "content": prompt},
        ]

        try:
            result = self._provider.chat(messages)
            response_text = _extract_text(result)
            return _parse_judge_response(response_text)
        except Exception as e:
            return JudgeResult(
                passed=False,
                score=0.0,
                rationale=f"Judge evaluation failed: {e}",
                raw_response="",
            )

    @classmethod
    def from_config(cls, judge_config) -> Optional["LLMJudgeEvaluator"]:
        """Create a judge evaluator from JudgeConfig.

        Uses _create_openai_provider (OpenAI SDK via lmproxy) instead of
        the former c1_providers dependency.

        Args:
            judge_config: JudgeConfig with provider, model, base_url

        Returns:
            LLMJudgeEvaluator if configured, None otherwise
        """
        if not judge_config.enabled:
            return None

        model_spec = judge_config.to_model_spec()

        if model_spec.provider == "mock":
            provider = _MockProvider(model=model_spec.model, seed=99)
        else:
            provider = _create_openai_provider(
                model=model_spec.model,
                base_url=model_spec.base_url or "",
            )

        return cls(provider)


def _extract_text(result) -> str:
    """Extract text from a chat result."""
    if isinstance(result.message.get("content"), str):
        return result.message["content"]
    elif isinstance(result.message.get("content"), list):
        parts = []
        for block in result.message["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _parse_judge_response(response_text: str) -> JudgeResult:
    """Parse judge response JSON into JudgeResult."""
    json_text = response_text.strip()

    # Strip markdown code fence if present
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        start = 1 if lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        json_text = "\n".join(lines[start:end])

    try:
        data = json.loads(json_text)
        return JudgeResult(
            passed=bool(data.get("passed", False)),
            score=float(data.get("score", 0.0)),
            rationale=str(data.get("rationale", "")),
            criteria_met=int(data.get("criteria_met", 0)),
            criteria_total=int(data.get("criteria_total", 0)),
            raw_response=response_text,
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        # Fallback: try to extract score from text
        return JudgeResult(
            passed=False,
            score=0.0,
            rationale=f"Failed to parse judge response: {response_text[:200]}",
            raw_response=response_text,
        )


def _score_deterministic(
    response: str,
    acceptance_criteria: dict,
    query_type: str,
) -> JudgeResult:
    """Score deterministic query types (YES_NO, YES_NO_VERIFICATION, SINGLE_WORD, RANKING).

    Uses exact match against the expected output in acceptance_criteria.correct_conclusion.

    Args:
        response: The model's response text.
        acceptance_criteria: Dict with correct_conclusion containing the expected answer.
        query_type: One of YES_NO, YES_NO_VERIFICATION, SINGLE_WORD, RANKING.

    Returns:
        JudgeResult with score 1.0 (match) or 0.0 (no match).
    """
    expected = acceptance_criteria.get("correct_conclusion", "").strip().lower()
    actual = response.strip().lower()

    # Extract the first meaningful word/phrase from the response
    # Models often add explanation after the answer
    first_line = actual.split("\n")[0].strip().rstrip(".")

    if query_type == "YES_NO":
        is_match = first_line in ("yes", "no") and first_line == expected
    elif query_type == "YES_NO_VERIFICATION":
        is_match = first_line in ("correct", "incorrect") and first_line == expected
    elif query_type == "SINGLE_WORD":
        first_word = first_line.split()[0] if first_line.split() else ""
        is_match = first_word == expected or first_line == expected
    elif query_type == "RANKING":
        is_match = first_line in ("a", "b") and first_line == expected
    else:
        is_match = False

    score = 1.0 if is_match else 0.0
    is_passed = is_match

    rationale = (
        f"Exact match: expected '{expected}', got '{first_line}'"
        if is_match
        else f"No match: expected '{expected}', got '{first_line}'"
    )

    return JudgeResult(
        passed=is_passed,
        score=score,
        rationale=rationale,
        criteria_met=1 if is_match else 0,
        criteria_total=1,
    )
