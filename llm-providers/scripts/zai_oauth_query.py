"""
oauth_query.py

Query GLM models via zhipuai SDK using a subscription endpoint at api.z.ai.

Prerequisites:
    pip install zhipuai

Usage:
    python oauth_query.py
    python oauth_query.py "What is the capital of France?"
    python oauth_query.py --model glm-5-turbo "Explain gradient descent"
"""

import argparse
import os
import sys

from zhipuai import ZhipuAI


BASE_URL = "https://api.z.ai/api/coding/paas/v4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Query z.ai via zhipuai SDK")
    parser.add_argument("prompt", nargs="?", default="Say hello in three languages.",
                        help="User prompt to send")
    parser.add_argument("--model", type=str, default="glm-5-turbo",
                        help="Model name (default: glm-5-turbo)")
    parser.add_argument("--max-tokens", type=int, default=16000,
                        help="Max tokens in response (default: 16000)")
    args = parser.parse_args()

    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        print("Error: ZHIPU_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = ZhipuAI(api_key=api_key, base_url=BASE_URL)

    response = client.chat.completions.create(
        model=args.model,
        max_tokens=args.max_tokens,
        messages=[{"role": "user", "content": args.prompt}],
    )

    choice = response.choices[0]
    print(choice.message.content)

    usage = response.usage
    print(f"\n--- tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out ---")


if __name__ == "__main__":
    main()
