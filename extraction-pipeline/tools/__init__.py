"""tools -- loaders and text extraction utilities."""

__all__: list[str] = []

try:
    from tools.skill_loader import SkillLoader
    from tools.task_loader import TaskLoader, BenchTask, load_verification_tasks
    __all__ += ["SkillLoader", "TaskLoader", "BenchTask", "load_verification_tasks"]
except ImportError:
    pass

try:
    from tools.text_extractor import process_file, process_directory
    __all__ += ["process_file", "process_directory"]
except ImportError:
    pass

# Backwards compatibility: re-export data structures and text utils that
# were previously defined directly in tools.text_extractor.
try:
    from utils.data_structures import Block, DocumentExport, TransformConfig, ProcessingResult
    from utils.text_utils import (
        parse_markdown_blocks,
        markdown_to_plain_text,
        collapse_lettertracks,
        rejoin_hyphens,
        normalize_block_spacing,
        normalize_whitespace,
        clean_markdown_artifacts,
        apply_transforms,
    )
    __all__ += [
        "Block", "DocumentExport", "TransformConfig", "ProcessingResult",
        "parse_markdown_blocks", "markdown_to_plain_text",
        "collapse_lettertracks", "rejoin_hyphens",
        "normalize_block_spacing", "normalize_whitespace",
        "clean_markdown_artifacts", "apply_transforms",
    ]
except ImportError:
    pass
