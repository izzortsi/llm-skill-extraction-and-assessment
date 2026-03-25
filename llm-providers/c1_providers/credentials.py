"""
credentials.py

Load OAuth credentials from Claude Code's local config.

Claude Code stores OAuth tokens in ~/.claude/.credentials.json.
This module reads that file so other tools can reuse the same
authentication without requiring the anthropic_oauth package
or a separate API key.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def get_claude_credentials_path() -> Path:
    """return path to Claude Code's credentials file."""
    return Path.home() / ".claude" / ".credentials.json"


def load_claude_credentials() -> dict[str, Any] | None:
    """load OAuth credentials from Claude Code's local config.

    returns the parsed credentials dict, or None if the file
    doesn't exist or can't be read.
    """
    path = get_claude_credentials_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_claude_oauth_token() -> str | None:
    """extract the OAuth access token from Claude Code credentials.

    returns the access token string, or None if credentials are
    not available.
    """
    creds = load_claude_credentials()
    if not creds:
        return None
    oauth = creds.get("claudeAiOauth", {})
    return oauth.get("accessToken")


def get_anthropic_api_key() -> str | None:
    """get an Anthropic API key from any available source.

    priority:
    1. ANTHROPIC_API_KEY environment variable
    2. Claude Code OAuth token (~/.claude/.credentials.json)
    3. CLAUDE_CODE_OAUTH_TOKEN environment variable

    returns the key/token string, or None if nothing is available.
    """
    # 1. explicit env var
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    # 2. claude code local credentials
    token = get_claude_oauth_token()
    if token:
        return token

    # 3. fallback env var
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if token:
        return token

    return None
