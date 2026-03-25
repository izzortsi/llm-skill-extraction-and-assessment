"""
claude_code_provider.py

LLM provider adapter that shells out to `claude -p` CLI.
Enables zero-setup pipeline runs on machines with Claude Code installed.
"""

from __future__ import annotations

import json
import subprocess
from typing import Dict, List


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert a messages list to a single text prompt for claude -p."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


class ClaudeCodeProvider:
    """LLM provider that wraps `claude -p --output-format json`."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model_name = model

    def chat(self, messages):
        prompt = _messages_to_prompt(messages)
        cmd = ["claude", "-p", "--output-format", "json", "--model", self.model_name]
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude -p failed (exit {result.returncode}): {result.stderr[:500]}"
            )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            data = {"result": result.stdout.strip()}

        content = data.get("result", data.get("content", result.stdout.strip()))
        usage = data.get("usage", {})

        class _Result:
            pass

        r = _Result()
        r.message = {"content": content}
        r.usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }
        return r
