"""
stage_runner.py

Execute a single pipeline stage as a subprocess. Each stage runs inside
its pipeline directory (cd isolation) to avoid Python module shadowing
between pipelines that share package names (c0_utils, c2_extraction).
"""

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class StageResult:
    stage_id: str
    command: str
    exit_code: int
    duration_seconds: float
    log_path: str


def run_stage_command(
    pipeline_dir: Path,
    command: str,
    args: List[str],
    log_path: Path,
    verbose: bool = False,
) -> StageResult:
    """Run a single CLI command inside a pipeline directory.

    Executes: python3 -m c4_cli.main <command> <args...>
    with cwd set to pipeline_dir.

    Args:
        pipeline_dir: absolute path to the pipeline project directory
        command: the CLI subcommand (e.g., "extract-passages")
        args: list of CLI arguments to pass after the command
        log_path: path to write combined stdout/stderr log
        verbose: if True, stream output to console in real time

    Returns:
        StageResult with exit code and timing
    """
    full_cmd = [sys.executable, "-u", "-m", "c4_cli.main", command] + args

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # PYTHONUNBUFFERED=1 forces line-buffered output from the subprocess
    # so that verbose streaming shows progress in real time instead of
    # waiting for the default 4KB pipe buffer to fill.
    import os
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    start_time = time.time()

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            full_cmd,
            cwd=str(pipeline_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
        )

        for line in process.stdout:
            log_file.write(line)
            if verbose:
                sys.stdout.write(line)
                sys.stdout.flush()

        process.wait()

    duration = time.time() - start_time

    return StageResult(
        stage_id="",
        command=command,
        exit_code=process.returncode,
        duration_seconds=duration,
        log_path=str(log_path),
    )
