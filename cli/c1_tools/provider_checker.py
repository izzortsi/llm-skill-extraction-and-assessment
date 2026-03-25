"""
provider_checker.py

Pre-flight checks for pipeline execution. Validates provider connectivity
and required Python packages before running stages.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import List


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


def check_ollama(url: str) -> CheckResult:
    """Check if Ollama is reachable at the given URL."""
    try:
        import urllib.request
        # strip /v1 suffix for API check
        base = url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        req = urllib.request.Request(f"{base}/api/tags", method="GET")
        req.add_header("User-Agent", "llm-skills-cli")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return CheckResult("Ollama", True, f"reachable at {base}")
    except Exception as e:
        return CheckResult("Ollama", False, f"not reachable at {url}: {e}")
    return CheckResult("Ollama", False, f"unexpected response from {url}")


def check_lmproxy(url: str) -> CheckResult:
    """Check lmproxy health endpoint."""
    try:
        base = url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        import urllib.request
        req = urllib.request.Request(f"{base}/health", method="GET")
        resp = urllib.request.urlopen(req, timeout=5)
        if resp.status == 200:
            return CheckResult("lmproxy", True, f"reachable at {url}")
        return CheckResult("lmproxy", False, f"health returned {resp.status}")
    except Exception as exc:
        return CheckResult("lmproxy", False, f"unreachable: {exc}")


def check_anthropic_key() -> CheckResult:
    """Check if Anthropic API key or OAuth is configured."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return CheckResult("Anthropic API key", True, "ANTHROPIC_API_KEY is set")

    # check for OAuth token file
    oauth_path = os.path.expanduser("~/.anthropic/oauth_token.json")
    if os.path.exists(oauth_path):
        return CheckResult("Anthropic OAuth", True, f"OAuth token found at {oauth_path}")

    return CheckResult("Anthropic credentials", False,
                       "ANTHROPIC_API_KEY not set and no OAuth token found")


def check_zhipu_key() -> CheckResult:
    """Check if Zhipu AI (ZAI) API key is set."""
    if os.environ.get("ZHIPU_API_KEY"):
        return CheckResult("Zhipu AI key", True, "ZHIPU_API_KEY is set")
    return CheckResult("Zhipu AI key", False, "ZHIPU_API_KEY not set (optional)")


def check_python_package(package_name: str) -> CheckResult:
    """Check if a Python package is importable."""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        return CheckResult(f"Python: {package_name}", True, "installed")
    return CheckResult(f"Python: {package_name}", False, "not installed")


def run_preflight_checks(profile) -> List[CheckResult]:
    """Run all pre-flight checks for a given profile.

    Returns a list of CheckResult. Caller decides whether to abort on failures.
    """
    results = []

    # required packages
    for pkg in ["yaml", "json", "pathlib"]:
        results.append(check_python_package(pkg))

    # optional but recommended packages
    for pkg in ["rich", "datasets", "matplotlib"]:
        results.append(check_python_package(pkg))

    # provider checks
    if profile.extraction_provider == "anthropic" or profile.judge_provider == "anthropic":
        results.append(check_anthropic_key())

    if profile.ollama_url:
        results.append(check_ollama(profile.ollama_url))

    if profile.zai_model:
        results.append(check_zhipu_key())

    # lmproxy check
    if hasattr(profile, "lmproxy_base_url") and profile.lmproxy_base_url:
        uses_lmproxy = (
            getattr(profile, "extraction_provider", "") == "lmproxy"
            or getattr(profile, "trace_provider", "") == "lmproxy"
            or getattr(profile, "judge_provider", "") == "lmproxy"
            or any(
                entry.get("provider") == "lmproxy"
                for entry in getattr(profile, "eval_models", [])
                if isinstance(entry, dict)
            )
        )
        if uses_lmproxy:
            results.append(check_lmproxy(profile.lmproxy_base_url))

    return results
