"""
stage_output_wirer.py

Build CLI argument lists for each pipeline stage. Maps profile config
and prior stage outputs to the specific --flag values each stage's
CLI expects.

This module is the Python equivalent of the variable assignments and
argument construction in run_full_pipeline.sh.
"""

from pathlib import Path
from typing import Dict, List

from c0_config.pipeline_profile import PipelineProfile


def build_stage_args(
    stage_id: str,
    profile: PipelineProfile,
    run_dir: Path,
    repo_root: Path,
    stage_outputs: Dict[str, Dict[str, str]],
    mode: str = "",
) -> List[str]:
    """Build the CLI argument list for a specific stage.

    Args:
        stage_id: which stage to build args for
        profile: the experiment profile
        run_dir: absolute path to the pipeline run directory
        repo_root: absolute path to the repository root
        stage_outputs: map of stage_id -> {"file_key": "absolute_path"}
        mode: evaluation mode (for stage 5 only)

    Returns:
        list of string arguments to pass after the CLI command
    """
    stage1_dir = run_dir / "stage1-task-extraction"
    stage2_dir = run_dir / "stage2-trace-capture"
    stage3_dir = run_dir / "stage3-skill-extraction"
    stage4_dir = run_dir / "stage4-skill-verification"
    stage5_dir = run_dir / "stage5-corpus-evaluation"
    stage6_dir = run_dir / "stage6-visualization"

    if stage_id == "1a":
        return [
            "--dataset", profile.dataset,
            "--subset", profile.subset,
            "--chunk-size", str(profile.chunk_size),
            "--max-chunks", str(profile.max_chunks),
            "--output", str(stage1_dir / "passages.json"),
            "-v",
        ]

    if stage_id == "1b":
        return [
            "--passages", stage_outputs.get("1a", {}).get("passages", str(stage1_dir / "passages.json")),
            "--domain", profile.domain,
            "--tasks-per-chunk", str(profile.tasks_per_chunk),
            "--provider", profile.extraction_provider,
            "--model", profile.extraction_model,
            "--output", str(stage1_dir / "tasks.json"),
            "-v",
        ]

    if stage_id == "2":
        return [
            "--tasks", stage_outputs.get("1b", {}).get("tasks", str(stage1_dir / "tasks.json")),
            "--output", str(stage2_dir / "traces.jsonl"),
            "--provider", profile.trace_provider,
            "--model", profile.trace_model,
            "-v",
        ]

    if stage_id == "3":
        return [
            "--traces", stage_outputs.get("2", {}).get("traces", str(stage2_dir / "traces.jsonl")),
            "--output", str(stage3_dir / "skills.json"),
            "--max-skills", str(profile.max_skills),
            "--provider", profile.extraction_provider,
            "--model", profile.extraction_model,
            "-v",
        ]

    if stage_id == "4":
        return [
            "--skills", stage_outputs.get("3", {}).get("skills", str(stage3_dir / "skills.json")),
            "--output", str(stage4_dir / "verified_skills.json"),
            "-v",
        ]

    if stage_id == "5":
        config_path = repo_root / profile.config_file
        if config_path.exists():
            args = [
                "--tasks", stage_outputs.get("1b", {}).get("tasks", str(stage1_dir / "tasks.json")),
                "--skills", stage_outputs.get("4", {}).get("skills", str(stage4_dir / "verified_skills.json")),
                "--config", str(config_path),
                "--mode", mode,
                "--output", str(stage5_dir / mode / "results-all.json"),
                "-v",
            ]
            # pass --models to restrict evaluation to profile's eval_models subset
            if profile.eval_models:
                args += ["--models", ",".join(profile.eval_models)]
            return args
        else:
            raise ValueError(
                f"Stage 5 config file not found at '{config_path}'. "
                "Ensure profile.config_file points to a valid models.yaml."
            )

    if stage_id == "6":
        return [
            "--results", str(stage5_dir / "results-all-merged.json"),
            "-o", str(stage6_dir / "cross-mode"),
            "--type", "charts",
            "--dpi", "200",
        ]

    if stage_id == "7":
        # stage 7 has two commands; return args for the first (traceability-report)
        # export-csv args are built separately via build_stage7_csv_args()
        return [
            "--tasks", stage_outputs.get("1b", {}).get("tasks", str(stage1_dir / "tasks.json")),
            "--skills", stage_outputs.get("4", {}).get("skills", str(stage4_dir / "verified_skills.json")),
            "--passages", stage_outputs.get("1a", {}).get("passages", str(stage1_dir / "passages.json")),
            "--output", str(run_dir / "traceability-report.txt"),
            "-v",
        ]

    raise ValueError(f"Unknown stage_id: {stage_id}")


def build_stage7_csv_args(
    profile: PipelineProfile,
    run_dir: Path,
    stage_outputs: Dict[str, Dict[str, str]],
) -> List[str]:
    """Build args for stage 7's second command (export-csv)."""
    stage1_dir = run_dir / "stage1-task-extraction"
    stage4_dir = run_dir / "stage4-skill-verification"

    return [
        "--tasks", stage_outputs.get("1b", {}).get("tasks", str(stage1_dir / "tasks.json")),
        "--skills", stage_outputs.get("4", {}).get("skills", str(stage4_dir / "verified_skills.json")),
        "--passages", stage_outputs.get("1a", {}).get("passages", str(stage1_dir / "passages.json")),
        "--output-dir", str(run_dir / "csv"),
        "-v",
    ]


def register_stage_outputs(stage_id: str, run_dir: Path) -> Dict[str, str]:
    """Register the output file paths for a completed stage."""
    stage1_dir = run_dir / "stage1-task-extraction"
    stage2_dir = run_dir / "stage2-trace-capture"
    stage3_dir = run_dir / "stage3-skill-extraction"
    stage4_dir = run_dir / "stage4-skill-verification"

    outputs = {
        "1a": {"passages": str(stage1_dir / "passages.json")},
        "1b": {"tasks": str(stage1_dir / "tasks.json")},
        "2": {"traces": str(stage2_dir / "traces.jsonl")},
        "3": {"skills": str(stage3_dir / "skills.json")},
        "4": {"skills": str(stage4_dir / "verified_skills.json")},
    }

    return outputs.get(stage_id, {})
