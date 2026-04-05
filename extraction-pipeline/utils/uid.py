"""
uid.py

Canonical UID generation for the skill extraction pipeline.

Format: xxxx-xxxx-xxxx-xxxx (4 groups of 4 lowercase hex digits).
Uses SHA-256 truncated to 64 bits for deterministic output.
"""

from __future__ import annotations

import hashlib


def generate_uid(seed_string: str) -> str:
    """Generate a company standard UID from a seed string.

    Format: xxxx-xxxx-xxxx-xxxx (4 groups of 4 lowercase hex digits).
    Uses SHA-256 truncated to 64 bits for deterministic output.
    """
    hash_bytes = hashlib.sha256(seed_string.encode()).digest()
    uid_int = int.from_bytes(hash_bytes[:8], byteorder="big")
    hex_str = f"{uid_int:016x}"
    return f"{hex_str[0:4]}-{hex_str[4:8]}-{hex_str[8:12]}-{hex_str[12:16]}"
