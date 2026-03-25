"""
provider_discovery.py

Runtime discovery of LLM providers: probes lmproxy, Ollama, and checks
API key environment variables for Anthropic and OpenAI.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import requests


@dataclass
class ProviderStatus:
    name: str
    reachable: bool
    models: List[str] = field(default_factory=list)
    base_url: str = ""
    message: str = ""
    upstream_providers: List[str] = field(default_factory=list)


def _strip_v1(url: str) -> str:
    """Strip trailing /v1 or /v1/ from URL to get base."""
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def _probe_lmproxy(lmproxy_url: str) -> ProviderStatus:
    """Probe lmproxy health and upstream providers."""
    base = _strip_v1(lmproxy_url)
    status = ProviderStatus(name="lmproxy", reachable=False, base_url=lmproxy_url)
    try:
        health = requests.get(f"{base}/health", timeout=3)
        if health.status_code != 200:
            status.message = f"health returned {health.status_code}"
            return status
        status.reachable = True
        health_data = health.json()
        key_count = health_data.get("keys", 0)
        session_count = health_data.get("sessions", 0)
    except Exception as exc:
        status.message = str(exc)
        return status

    upstream_names = []
    try:
        providers_resp = requests.get(f"{base}/llmproxy/llm-providers", timeout=3)
        if providers_resp.status_code == 200:
            providers_data = providers_resp.json()
            if isinstance(providers_data, list):
                upstream_names = [p.get("name", "") for p in providers_data if p.get("name")]
            status.upstream_providers = upstream_names
    except Exception:
        pass

    if upstream_names:
        status.message = f"{key_count} keys, {session_count} sessions (-> {', '.join(upstream_names)})"
    else:
        status.message = f"{key_count} keys, {session_count} sessions"
    return status


def _probe_ollama(ollama_url: str) -> ProviderStatus:
    """Probe Ollama /api/tags for local model list."""
    base = _strip_v1(ollama_url)
    status = ProviderStatus(name="ollama", reachable=False, base_url=ollama_url)
    try:
        resp = requests.get(f"{base}/api/tags", timeout=3)
        if resp.status_code != 200:
            status.message = f"/api/tags returned {resp.status_code}"
            return status
        data = resp.json()
        models = data.get("models", [])
        status.models = [m.get("name", "") for m in models if m.get("name")]
        status.reachable = True
        status.message = f"{len(status.models)} models"
    except Exception as exc:
        status.message = str(exc)
    return status


def _probe_lm_studio(lm_studio_url: str) -> ProviderStatus:
    """Probe LM Studio local server for available models."""
    status = ProviderStatus(name="lm-studio", reachable=False, base_url=lm_studio_url)
    # Normalize: ensure URL ends with /v1 for the models endpoint
    base = _strip_v1(lm_studio_url)
    models_endpoint = f"{base}/v1/models"
    try:
        resp = requests.get(models_endpoint, timeout=3)
        if resp.status_code != 200:
            status.message = f"{models_endpoint} returned {resp.status_code}"
            return status
        data = resp.json()
        model_list = data.get("data", [])
        status.models = [m.get("id", "") for m in model_list if m.get("id")]
        status.reachable = True
        status.message = f"{len(status.models)} models"
    except Exception as exc:
        status.message = str(exc)
    return status


def _probe_zai(zai_url: str) -> ProviderStatus:
    """Probe Z.AI (Zhipu) API for available models."""
    status = ProviderStatus(name="zai", reachable=False, base_url=zai_url)
    api_key = os.environ.get("ZHIPU_API_KEY", "")
    if not api_key:
        status.message = "no ZHIPU_API_KEY found"
        return status
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(f"{zai_url}/models", headers=headers, timeout=5)
        if resp.status_code != 200:
            status.message = f"/models returned {resp.status_code}"
            return status
        data = resp.json()
        model_list = data.get("data", [])
        status.models = [m.get("id", "") for m in model_list if m.get("id")]
        status.reachable = True
        status.message = f"{len(status.models)} models"
    except Exception as exc:
        status.message = str(exc)
    return status


def _probe_iosys(iosys_url: str) -> ProviderStatus:
    """Probe iosys LLM inference API for available models."""
    status = ProviderStatus(name="iosys", reachable=False, base_url=iosys_url)
    api_key = os.environ.get("IOSYS_API_KEY", "")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(f"{iosys_url}/models", headers=headers, timeout=3)
        if resp.status_code != 200:
            status.message = f"/v1/models returned {resp.status_code}"
            return status
        data = resp.json()
        model_list = data.get("data", [])
        status.models = [m.get("id", "") for m in model_list if m.get("id")]
        status.reachable = True
        status.message = f"{len(status.models)} models"
        if not api_key:
            status.message += " (set IOSYS_API_KEY for auth)"
    except Exception as exc:
        status.message = str(exc)
    return status


def _check_anthropic() -> ProviderStatus:
    """Check Anthropic API key availability (no network call)."""
    status = ProviderStatus(name="anthropic", reachable=False, base_url="https://api.anthropic.com")
    if os.environ.get("ANTHROPIC_API_KEY"):
        status.reachable = True
        status.models = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
        status.message = "API key found"
        return status
    oauth_path = Path.home() / ".anthropic" / "oauth_token.json"
    if oauth_path.exists():
        status.reachable = True
        status.models = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
        status.message = "OAuth token found"
        return status
    status.message = "no API key found"
    return status


def _check_openai() -> ProviderStatus:
    """Check OpenAI API key availability (no network call)."""
    status = ProviderStatus(name="openai", reachable=False, base_url="https://api.openai.com")
    if os.environ.get("OPENAI_API_KEY"):
        status.reachable = True
        status.models = ["gpt-4o", "gpt-4o-mini"]
        status.message = "API key found"
    else:
        status.message = "no API key found"
    return status


def collect_lmproxy_models(lmproxy_url: str, config_file: str) -> List[str]:
    """Collect model aliases from models.yaml that route through lmproxy."""
    config_path = Path(config_file)
    if not config_path.exists():
        return []
    try:
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        models_dict = data.get("models", {})
        matched = []
        for alias, entry in models_dict.items():
            if not isinstance(entry, dict):
                continue
            api_base = entry.get("api_base", "")
            if api_base.rstrip("/") == lmproxy_url.rstrip("/"):
                matched.append(alias)
        return matched
    except Exception:
        return []


def _check_claude_code() -> ProviderStatus:
    """Check if claude CLI is available on PATH."""
    status = ProviderStatus(name="claude-code", reachable=False)
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            status.reachable = True
            status.models = ["claude-code"]
            status.message = f"CLI available ({result.stdout.strip()})"
        else:
            status.message = "claude CLI returned error"
    except FileNotFoundError:
        status.message = "claude CLI not found on PATH"
    except Exception as exc:
        status.message = str(exc)
    return status


def discover_providers(
    lmproxy_url: str = "http://localhost:8080/v1",
    ollama_url: str = "http://localhost:11434/v1",
    iosys_url: str = "http://llm.iosys.net/v1",
    lm_studio_url: str = "http://localhost:1234/v1",
    zai_url: str = "https://open.bigmodel.cn/api/paas/v4",
    config_file: str = "",
) -> List[ProviderStatus]:
    """Probe all known providers and return their status."""
    lmproxy_status = _probe_lmproxy(lmproxy_url)

    if lmproxy_status.reachable and config_file:
        lmproxy_status.models = collect_lmproxy_models(lmproxy_url, config_file)
        if lmproxy_status.models:
            lmproxy_status.message += f", {len(lmproxy_status.models)} models"

    ollama_status = _probe_ollama(ollama_url)
    iosys_status = _probe_iosys(iosys_url)
    lm_studio_status = _probe_lm_studio(lm_studio_url)
    zai_status = _probe_zai(zai_url)
    anthropic_status = _check_anthropic()
    openai_status = _check_openai()
    claude_status = _check_claude_code()

    return [lmproxy_status, ollama_status, iosys_status, lm_studio_status, zai_status, anthropic_status, openai_status, claude_status]
