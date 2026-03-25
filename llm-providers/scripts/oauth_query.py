"""
oauth_query.py

Minimal script demonstrating Anthropic API access via OAuth token
(Claude Pro/Max subscription) using the anthropic-oauth library.

First run launches a browser for authentication and prompts for a code.
Subsequent runs reuse saved tokens with automatic refresh.

Prerequisites:
    pip install "git+https://github.com/izzortsi/anthropic-oauth"

Usage:
    python scripts/oauth_query.py
    python scripts/oauth_query.py "What is the capital of France?"
    python scripts/oauth_query.py --model claude-sonnet-4-5-20250929 "Explain gradient descent"
"""

import argparse
import sys

from anthropic_oauth import create_oauth_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Anthropic via OAuth")
    parser.add_argument("prompt", nargs="?", default="Say hello in three languages.",
                        help="User prompt to send")
    parser.add_argument("--model", type=str, default="claude-opus-4-6",
                        help="Model name (default: claude-opus-4-6)")
    parser.add_argument("--max-tokens", type=int, default=16000,
                        help="Max tokens in response (default: 16000)")
    args = parser.parse_args()

    client = create_oauth_client()

    response = client.messages.create(
        model=args.model,
        max_tokens=args.max_tokens,
        messages=[{"role": "user", "content": args.prompt}],
    )

    for block in response.content:
        if block.type == "text":
            print(block.text)

    print(f"\n--- tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out ---")


if __name__ == "__main__":
    main()
