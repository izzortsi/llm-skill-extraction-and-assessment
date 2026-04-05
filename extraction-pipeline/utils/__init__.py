"""utils -- pure utility functions with no domain imports."""

from utils.uid import generate_uid
from utils.text_utils import (
    parse_markdown_blocks,
    markdown_to_plain_text,
    collapse_lettertracks,
    rejoin_hyphens,
    normalize_block_spacing,
    normalize_whitespace,
    clean_markdown_artifacts,
    apply_transforms,
    extract_thinking,
    strip_markdown_fences,
)
from utils.data_structures import (
    Block,
    DocumentExport,
    TransformConfig,
    ProcessingResult,
)

__all__ = [
    # uid
    "generate_uid",
    # text_utils
    "parse_markdown_blocks",
    "markdown_to_plain_text",
    "collapse_lettertracks",
    "rejoin_hyphens",
    "normalize_block_spacing",
    "normalize_whitespace",
    "clean_markdown_artifacts",
    "apply_transforms",
    "extract_thinking",
    "strip_markdown_fences",
    # data_structures
    "Block",
    "DocumentExport",
    "TransformConfig",
    "ProcessingResult",
]
