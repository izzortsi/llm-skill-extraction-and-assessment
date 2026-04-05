"""
mock_provider.py

Mock LLM provider for testing the pipeline without API keys.
Returns deterministic responses based on the task content and acceptance criteria.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MockChatResult:
    """Mimics grpt ChatResult interface."""

    message: Dict[str, Any]
    usage: Dict[str, int]
    raw_response: Dict[str, Any]
    reasoning: Optional[str] = None
    history_message: Optional[Dict[str, Any]] = None


class MockProvider:
    """Deterministic mock provider that generates responses from acceptance criteria.

    For reading comprehension tasks, extracts key terms from the system prompt
    and user message to produce a response that partially satisfies acceptance
    criteria. The pass rate varies by condition (baseline vs skill-injected)
    based on a deterministic hash of the input.
    """

    def __init__(self, model: str = "mock-model", seed: int = 42):
        self._model = model
        self._seed = seed

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def uses_native_tools(self) -> bool:
        return False

    @property
    def message_format(self) -> str:
        return "openai"

    @property
    def is_external(self) -> bool:
        return False

    def chat(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> MockChatResult:
        """Generate a deterministic response based on message content.

        When a SKILL block is present in the system prompt, the response
        incorporates more key terms from the task, simulating the effect
        of skill-guided reasoning.
        """
        system_text = ""
        user_text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            elif role == "user":
                user_text = content

        has_skill = "--- SKILL:" in system_text
        has_self_gen = "SELF-GENERATED SKILL INSTRUCTION" in system_text

        response = self._build_response(user_text, system_text, has_skill, has_self_gen)

        token_count = len(response.split())
        return MockChatResult(
            message={"role": "assistant", "content": response},
            usage={
                "prompt_tokens": len(system_text.split()) + len(user_text.split()),
                "completion_tokens": token_count,
                "total_tokens": len(system_text.split()) + len(user_text.split()) + token_count,
            },
            raw_response={"mock": True, "model": self._model},
        )

    def _build_response(
        self,
        user_text: str,
        system_text: str,
        has_skill: bool,
        has_self_gen: bool,
    ) -> str:
        """Build a deterministic response that partially matches acceptance criteria.

        Strategy:
        - Extract key terms from the user prompt (passage + challenge)
        - When a skill is injected, include more analytical terms that map to
          acceptance criteria keywords (schema, activation, ambiguous, etc.)
        - Use a hash to deterministically vary quality across trials
        """
        # Derive a deterministic quality factor from input hash
        input_hash = hashlib.sha256(
            f"{user_text}|{system_text}|{self._seed}".encode()
        ).hexdigest()
        quality = int(input_hash[:4], 16) / 0xFFFF  # 0.0 to 1.0

        parts = ["Based on my analysis of the passage:\n"]

        # Always echo some passage content
        passage_sentences = [s.strip() for s in user_text.split(".") if len(s.strip()) > 20]
        if passage_sentences:
            parts.append(f"The passage states: \"{passage_sentences[0]}.\"")

        # Extract words from the challenge for targeted response
        challenge_start = user_text.find("Challenge:")
        challenge_text = user_text[challenge_start:] if challenge_start >= 0 else user_text

        if has_skill:
            # Skill-injected: produce more analytical, criteria-aligned response
            parts.append(
                "\nApplying the procedural knowledge from the skill, "
                "I can systematically analyze this:"
            )

            # Include cognitive science terms commonly found in acceptance criteria
            analysis_terms = [
                "ambiguous", "meaning", "context", "schema", "activation",
                "interpretation", "semantic", "syntactic", "resolution",
                "inference", "proposition", "reference", "coreference",
                "pronoun", "antecedent", "working memory", "lexical",
                "comprehension", "encoding", "role", "structure",
            ]

            # With skill, include more terms (higher coverage of criteria)
            threshold = 0.3 if quality > 0.4 else 0.5
            included = [t for t in analysis_terms if _term_relevant(t, user_text, threshold)]

            if included:
                parts.append(
                    f"\nKey cognitive processes involved: {', '.join(included[:8])}."
                )

            # Echo key phrases from the challenge back
            parts.append(f"\n{_paraphrase_challenge(challenge_text)}")

            # Add structured conclusion
            parts.append(
                "\nIn conclusion, the reader selects the correct interpretation "
                "based on the activated schema and contextual constraints. "
                "The context words prime related concepts in the mental lexicon, "
                "enabling faster direct lexical access. "
                "The syntactic structure links back to the appropriate antecedent."
            )

        elif has_self_gen:
            # Self-generated: produce procedural knowledge then a moderate answer
            parts.append(
                "\nFirst, let me generate relevant procedural knowledge:\n"
                "1. Identify the key elements in the passage\n"
                "2. Determine what cognitive processes are involved\n"
                "3. Apply those processes to the challenge\n"
            )
            parts.append(f"\nNow applying this to the task: {_paraphrase_challenge(challenge_text)}")

        else:
            # Baseline: simpler, less targeted response
            parts.append(f"\nRegarding the challenge: {_paraphrase_challenge(challenge_text)}")

            if quality > 0.6:
                parts.append(
                    "\nThe passage provides context that helps determine "
                    "the correct interpretation."
                )

        return "\n".join(parts)


def _term_relevant(term: str, text: str, threshold: float) -> bool:
    """Check if a term is relevant to the text, with some hash-based variation."""
    text_lower = text.lower()
    if term.lower() in text_lower:
        return True
    h = int(hashlib.md5(f"{term}|{text[:50]}".encode()).hexdigest()[:4], 16) / 0xFFFF
    return h < threshold


def _paraphrase_challenge(challenge_text: str) -> str:
    """Extract and lightly rephrase challenge content."""
    # Pull key noun phrases from the challenge
    words = challenge_text.split()
    # Take meaningful chunks, skip very short words
    meaningful = [w for w in words if len(w) > 3]
    if len(meaningful) > 15:
        meaningful = meaningful[:15]
    return "The analysis shows that " + " ".join(meaningful) + "."
