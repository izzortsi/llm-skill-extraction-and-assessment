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


def check_lm_studio(url: str) -> CheckResult:
    """Check LM Studio local server."""
    try:
        import urllib.request
        base = url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        endpoint = f"{base}/v1/models"
        req = urllib.request.Request(endpoint, method="GET")
        resp = urllib.request.urlopen(req, timeout=5)
        if resp.status == 200:
            return CheckResult("lm-studio", True, f"reachable at {url}")
        return CheckResult("lm-studio", False, f"{endpoint} returned {resp.status}")
    except Exception as exc:
        return CheckResult("lm-studio", False, f"unreachable: {exc}")


def check_zai(url: str) -> CheckResult:
    """Check Z.AI (Zhipu) API."""
    try:
        api_key = os.environ.get("ZHIPU_API_KEY", "")
        if not api_key:
            return CheckResult("zai", False, "no ZHIPU_API_KEY found")
        import urllib.request
        req = urllib.request.Request(f"{url}/models", method="GET")
        req.add_header("Authorization", f"Bearer {api_key}")
        resp = urllib.request.urlopen(req, timeout=5)
        if resp.status == 200:
            return CheckResult("zai", True, f"reachable at {url}")
        return CheckResult("zai", False, f"/models returned {resp.status}")
    except Exception as exc:
        return CheckResult("zai", False, f"unreachable: {exc}")


def check_iosys(url: str) -> CheckResult:
    """Check iosys LLM inference API."""
    try:
        import urllib.request
        api_key = os.environ.get("IOSYS_API_KEY", "")
        req = urllib.request.Request(f"{url}/models", method="GET")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        resp = urllib.request.urlopen(req, timeout=5)
        if resp.status == 200:
            return CheckResult("iosys", True, f"reachable at {url}")
        return CheckResult("iosys", False, f"/v1/models returned {resp.status}")
    except Exception as exc:
        return CheckResult("iosys", False, f"unreachable: {exc}")


def check_anthropic_oauth() -> CheckResult:
    """Check if anthropic-oauth package is installed with valid tokens."""
    try:
        from anthropic_oauth import OAuthManager
    except ImportError:
        return CheckResult("anthropic-oauth", False, "anthropic-oauth not installed")
    try:
        manager = OAuthManager()
        if manager.has_valid_tokens():
            return CheckResult("anthropic-oauth", True, "OAuth tokens valid")
        return CheckResult("anthropic-oauth", False, "no valid OAuth tokens (run: anthropic-oauth)")
    except Exception as exc:
        return CheckResult("anthropic-oauth", False, f"error checking OAuth tokens: {exc}")


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
    uses_anthropic_oauth = (
        getattr(profile, "extraction_provider", "") == "anthropic-oauth"
        or getattr(profile, "trace_provider", "") == "anthropic-oauth"
        or getattr(profile, "judge_provider", "") == "anthropic-oauth"
        or any(
            entry.get("provider") == "anthropic-oauth"
            for entry in getattr(profile, "eval_models", [])
            if isinstance(entry, dict)
        )
    )
    if uses_anthropic_oauth:
        results.append(check_anthropic_oauth())

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

    # iosys check
    if hasattr(profile, "iosys_base_url") and profile.iosys_base_url:
        uses_iosys = (
            getattr(profile, "extraction_provider", "") == "iosys"
            or getattr(profile, "trace_provider", "") == "iosys"
            or getattr(profile, "judge_provider", "") == "iosys"
            or any(
                entry.get("provider") == "iosys"
                for entry in getattr(profile, "eval_models", [])
                if isinstance(entry, dict)
            )
        )
        if uses_iosys:
            results.append(check_iosys(profile.iosys_base_url))

    # lm-studio check
    if hasattr(profile, "lm_studio_url") and profile.lm_studio_url:
        uses_lm_studio = (
            getattr(profile, "extraction_provider", "") == "lm-studio"
            or getattr(profile, "trace_provider", "") == "lm-studio"
            or getattr(profile, "judge_provider", "") == "lm-studio"
            or any(
                entry.get("provider") == "lm-studio"
                for entry in getattr(profile, "eval_models", [])
                if isinstance(entry, dict)
            )
        )
        if uses_lm_studio:
            results.append(check_lm_studio(profile.lm_studio_url))

    # zai check
    if hasattr(profile, "zai_url") and profile.zai_url:
        uses_zai = (
            getattr(profile, "extraction_provider", "") == "zai"
            or getattr(profile, "trace_provider", "") == "zai"
            or getattr(profile, "judge_provider", "") == "zai"
            or any(
                entry.get("provider") == "zai"
                for entry in getattr(profile, "eval_models", [])
                if isinstance(entry, dict)
            )
        )
        if uses_zai:
            results.append(check_zai(profile.zai_url))

    return results
