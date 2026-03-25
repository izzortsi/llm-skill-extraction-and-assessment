"""
data_structures.py

Pure dataclasses used across the extraction pipeline.
No domain-specific imports -- only stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Block:
    """A parsed markdown block."""
    block_type: str
    level: Optional[int]
    text: str
    raw: str


@dataclass
class DocumentExport:
    """Container for extracted document data."""
    markdown: str
    plain_text: str
    blocks: list
    images: dict
    metadata: dict


@dataclass
class TransformConfig:
    """Controls which text cleanup transforms are applied."""
    should_rejoin_hyphens: bool = True
    should_collapse_lettertracks: bool = True
    should_normalize_block_spacing: bool = True
    should_clean_markdown_artifacts: bool = True
    should_normalize_whitespace: bool = True
    is_markdown: bool = False
    should_use_dictionary: bool = False
    dictionary_path: Optional[str] = None
    min_lettertrack_length: int = 4


@dataclass
class ProcessingResult:
    """Status report from processing a single file."""
    input_path: Path
    output_path: Path
    format: str
    is_success: bool = True
    errors: list = field(default_factory=list)
