"""
uid.py

Company standard UID generation and extraction method naming.

UID format: xxxx-xxxx-xxxx-xxxx (64-bit).
Extraction method format: {model}-{operation}-v{version}.
"""

import hashlib
import re


def generate_uid(seed_string: str) -> str:
    """Generate a company standard UID from a seed string.

    Format: xxxx-xxxx-xxxx-xxxx (4 groups of 4 lowercase hex digits).
    Uses SHA-256 truncated to 64 bits for deterministic output.

    Args:
        seed_string: Input string to hash. Must combine enough context
                     to be unique (e.g., domain + title + index for tasks).

    Returns:
        UID string in format xxxx-xxxx-xxxx-xxxx.
    """
    hash_bytes = hashlib.sha256(seed_string.encode()).digest()
    uid_int = int.from_bytes(hash_bytes[:8], byteorder="big")
    hex_str = f"{uid_int:016x}"
    return f"{hex_str[0:4]}-{hex_str[4:8]}-{hex_str[8:12]}-{hex_str[12:16]}"


def format_extraction_method(model: str, operation: str, version: int = 1) -> str:
    """Format an extraction method identifier per standard-extraction-method-naming.txt.

    Format: {model}-{operation}-v{version}

    Args:
        model: LLM model name (e.g., "claude-opus-4-6", "qwen3:0.6b").
               Colons and slashes are replaced with hyphens.
        operation: What the extraction does (e.g., "task-extraction", "trace-capture").
        version: Integer version of the prompt/method.

    Returns:
        Method identifier string (e.g., "opus-task-extraction-v1").
    """
    short_model = re.sub(r"^claude-", "", model)
    short_model = re.sub(r"[:/]", "-", short_model)
    return f"{short_model}-{operation}-v{version}"
