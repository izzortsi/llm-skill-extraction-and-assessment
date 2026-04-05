import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.claude_code_provider import ClaudeCodeProvider, _messages_to_prompt


def test_messages_to_prompt_single_user():
    messages = [{"role": "user", "content": "Hello"}]
    prompt = _messages_to_prompt(messages)
    assert "Hello" in prompt


def test_messages_to_prompt_system_and_user():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    prompt = _messages_to_prompt(messages)
    assert "You are a helpful assistant." in prompt
    assert "What is 2+2?" in prompt


def test_chat_returns_content():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({
        "result": "The answer is 4.",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    with patch("tools.claude_code_provider.subprocess.run", return_value=mock_result):
        provider = ClaudeCodeProvider()
        result = provider.chat([{"role": "user", "content": "What is 2+2?"}])
    assert result.message["content"] == "The answer is 4."


def test_chat_passes_model_to_cli():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"result": "ok"})
    with patch("tools.claude_code_provider.subprocess.run", return_value=mock_result) as mock_run:
        provider = ClaudeCodeProvider(model="claude-opus-4-6")
        provider.chat([{"role": "user", "content": "test"}])
    cmd = mock_run.call_args[0][0]
    assert "--model" in cmd
    model_idx = cmd.index("--model")
    assert cmd[model_idx + 1] == "claude-opus-4-6"


def test_default_model_is_haiku():
    provider = ClaudeCodeProvider()
    assert provider.model_name == "claude-haiku-4-5-20251001"


def test_chat_raises_on_nonzero_exit():
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "claude: command not found"
    with patch("tools.claude_code_provider.subprocess.run", return_value=mock_result):
        provider = ClaudeCodeProvider()
        try:
            provider.chat([{"role": "user", "content": "test"}])
            assert False, "Should have raised"
        except RuntimeError as exc:
            assert "claude" in str(exc).lower()


def test_check_claude_code_available():
    from tools.provider_discovery import _check_claude_code
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "claude 1.0.0"
    with patch("tools.provider_discovery.subprocess.run", return_value=mock_result):
        status = _check_claude_code()
    assert status.reachable is True
    assert "claude-code" in status.models


def test_check_claude_code_not_available():
    from tools.provider_discovery import _check_claude_code
    with patch("tools.provider_discovery.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("claude not found")
        status = _check_claude_code()
    assert status.reachable is False
