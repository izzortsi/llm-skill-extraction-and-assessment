import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c1_tools.provider_discovery import (
    ProviderStatus, discover_providers, _strip_v1, _probe_lmproxy,
    _probe_ollama, _check_anthropic, _check_openai,
)


def test_strip_v1():
    assert _strip_v1("http://localhost:8080/v1") == "http://localhost:8080"
    assert _strip_v1("http://localhost:8080/v1/") == "http://localhost:8080"
    assert _strip_v1("http://localhost:8080") == "http://localhost:8080"


def test_probe_lmproxy_reachable():
    health_resp = MagicMock()
    health_resp.status_code = 200
    health_resp.json.return_value = {"status": "ok", "keys": 2, "sessions": 1}

    providers_resp = MagicMock()
    providers_resp.status_code = 200
    providers_resp.json.return_value = [
        {"name": "anthropic", "base_url": "https://api.anthropic.com"},
        {"name": "zai", "base_url": "https://api.z.ai"},
    ]

    with patch("c1_tools.provider_discovery.requests.get") as mock_get:
        mock_get.side_effect = [health_resp, providers_resp]
        status = _probe_lmproxy("http://localhost:8080/v1")

    assert status.reachable is True
    assert "anthropic" in status.message.lower() or "2" in status.message


def test_probe_lmproxy_unreachable():
    with patch("c1_tools.provider_discovery.requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        status = _probe_lmproxy("http://localhost:8080/v1")

    assert status.reachable is False


def test_probe_ollama_reachable():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"models": [
        {"name": "qwen2.5:3b"},
        {"name": "llama3.2:1b"},
    ]}

    with patch("c1_tools.provider_discovery.requests.get", return_value=resp):
        status = _probe_ollama("http://localhost:11434/v1")

    assert status.reachable is True
    assert "qwen2.5:3b" in status.models
    assert "llama3.2:1b" in status.models


def test_probe_ollama_unreachable():
    with patch("c1_tools.provider_discovery.requests.get") as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        status = _probe_ollama("http://localhost:11434/v1")

    assert status.reachable is False
    assert status.models == []


def test_check_anthropic_with_key():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
        status = _check_anthropic()
    assert status.reachable is True
    assert len(status.models) > 0


def test_check_anthropic_no_key():
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with patch.dict("os.environ", env, clear=True):
        with patch("c1_tools.provider_discovery.Path.exists", return_value=False):
            status = _check_anthropic()
    assert status.reachable is False


def test_discover_providers_returns_all():
    with patch("c1_tools.provider_discovery._probe_lmproxy") as m1, \
         patch("c1_tools.provider_discovery._probe_ollama") as m2, \
         patch("c1_tools.provider_discovery._probe_iosys") as m2b, \
         patch("c1_tools.provider_discovery._check_anthropic") as m3, \
         patch("c1_tools.provider_discovery._check_openai") as m4, \
         patch("c1_tools.provider_discovery._check_claude_code") as m5:
        m1.return_value = ProviderStatus("lmproxy", False, [], "", "down")
        m2.return_value = ProviderStatus("ollama", False, [], "", "down")
        m2b.return_value = ProviderStatus("iosys", False, [], "", "down")
        m3.return_value = ProviderStatus("anthropic", False, [], "", "no key")
        m4.return_value = ProviderStatus("openai", False, [], "", "no key")
        m5.return_value = ProviderStatus("claude-code", False, [], "", "not found")
        results = discover_providers()

    assert len(results) == 6
    names = [r.name for r in results]
    assert "iosys" in names
    assert "lmproxy" in names
    assert "ollama" in names
