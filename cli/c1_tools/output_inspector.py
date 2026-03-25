"""
output_inspector.py

Check pipeline run directory for existing stage outputs. Enables
crash recovery by detecting which stages have already completed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from c0_config.pipeline_stage import PipelineStage
from c0_config.stage_registry import STAGES


@dataclass
class StageStatus:
    stage_id: str
    name: str
    is_complete: bool
    output_paths: List[str]  # paths that exist
    missing_paths: List[str]  # paths that do not exist


def inspect_run_dir(run_dir: Path) -> List[StageStatus]:
    """Check all stages for existing output in the run directory.

    Returns a list of StageStatus for each stage, indicating whether
    the stage's expected output files exist.
    """
    results = []

    for stage in STAGES:
        existing = []
        missing = []

        for output_file in stage.output_files:
            if stage.output_dir:
                path = run_dir / stage.output_dir / output_file
            else:
                path = run_dir / output_file

            if path.exists():
                existing.append(str(path))
            else:
                missing.append(str(path))

        is_complete = len(stage.output_files) > 0 and len(missing) == 0

        results.append(StageStatus(
            stage_id=stage.stage_id,
            name=stage.name,
            is_complete=is_complete,
            output_paths=existing,
            missing_paths=missing,
        ))

    return results


def check_dependencies_met(stage: PipelineStage, run_dir: Path) -> List[str]:
    """Check if all dependencies for a stage have output on disk.

    Returns a list of missing dependency stage IDs. Empty list means
    all dependencies are met.
    """
    statuses = inspect_run_dir(run_dir)
    status_map = {s.stage_id: s for s in statuses}

    missing = []
    for dep_id in stage.depends_on:
        dep_status = status_map.get(dep_id)
        if dep_status is None or not dep_status.is_complete:
            missing.append(dep_id)

    return missing
