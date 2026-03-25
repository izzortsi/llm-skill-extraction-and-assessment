"""
pipeline_executor.py

Orchestrate multi-stage pipeline execution. Resolves dependencies,
checks for existing outputs (crash recovery), and runs stages
sequentially via subprocess isolation.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List

from c0_config.pipeline_profile import PipelineProfile
from c0_config.stage_registry import get_stage, parse_stage_range, STAGES
from c1_tools.output_inspector import check_dependencies_met, inspect_run_dir
from c1_tools.stage_runner import StageResult, run_stage_command
from c2_orchestration.stage_output_wirer import (
    build_stage_args,
    build_stage7_csv_args,
    register_stage_outputs,
)


def execute_pipeline(
    profile: PipelineProfile,
    stage_range: str,
    repo_root: Path,
    clean: bool = False,
    verbose: bool = True,
    print_fn=None,
) -> List[StageResult]:
    """Execute a sequence of pipeline stages.

    Args:
        profile: experiment configuration
        stage_range: stage range string ("all", "1-4", "5-7", etc.)
        repo_root: absolute path to kcg-ml-llm repository root
        clean: if True, wipe the run directory before starting
        verbose: stream subprocess output to console
        print_fn: callable for status messages (default: print)

    Returns:
        list of StageResult for each stage executed
    """
    if print_fn is None:
        print_fn = print

    run_dir = Path(profile.run_dir)
    if not run_dir.is_absolute():
        run_dir = repo_root / run_dir

    # clean mode: wipe run directory
    if clean and run_dir.exists():
        print_fn(f"CLEAN: removing {run_dir}")
        shutil.rmtree(run_dir)

    # create output directories
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # parse stage range
    stage_ids = parse_stage_range(stage_range)
    print_fn(f"Stages: {', '.join(stage_ids)}")

    # accumulate outputs from completed stages
    stage_outputs: Dict[str, Dict[str, str]] = {}

    # pre-populate outputs from existing files (for partial runs)
    for sid in ["1a", "1b", "2", "3", "4"]:
        stage_outputs[sid] = register_stage_outputs(sid, run_dir)

    results = []

    for stage_id in stage_ids:
        stage = get_stage(stage_id)
        pipeline_dir = repo_root / stage.pipeline_dir

        print_fn(f"\n=== Stage {stage_id}: {stage.description} ===")

        # check dependencies
        missing_deps = check_dependencies_met(stage, run_dir)
        if len(missing_deps) > 0:
            print_fn(f"  ERROR: missing dependencies: stages {', '.join(missing_deps)}")
            print_fn(f"  Run those stages first, or use --stages all")
            break

        # check if output already exists (crash recovery)
        if _stage_output_exists(stage, run_dir, profile):
            print_fn(f"  [skip] output already exists")
            results.append(StageResult(
                stage_id=stage_id,
                command="(skipped)",
                exit_code=0,
                duration_seconds=0.0,
                log_path="",
            ))
            continue

        # ensure output directories exist
        if stage.output_dir:
            (run_dir / stage.output_dir).mkdir(parents=True, exist_ok=True)

        # execute stage
        if stage_id == "5":
            stage_results = _execute_stage5(
                stage, profile, run_dir, repo_root, pipeline_dir,
                logs_dir, stage_outputs, verbose, print_fn,
            )
            results.extend(stage_results)
        elif stage_id == "6":
            stage_result = _execute_stage6(
                stage, profile, run_dir, repo_root, pipeline_dir,
                logs_dir, stage_outputs, verbose, print_fn,
            )
            results.append(stage_result)
        elif stage_id == "7":
            stage_results = _execute_stage7(
                stage, profile, run_dir, repo_root, pipeline_dir,
                logs_dir, stage_outputs, verbose, print_fn,
            )
            results.extend(stage_results)
        else:
            args = build_stage_args(stage_id, profile, run_dir, repo_root, stage_outputs)
            log_path = logs_dir / f"stage{stage_id}-{stage.name}.log"

            result = run_stage_command(
                pipeline_dir=pipeline_dir,
                command=stage.commands[0],
                args=args,
                log_path=log_path,
                verbose=verbose,
            )
            result.stage_id = stage_id

            if result.exit_code != 0:
                print_fn(f"  FAILED (exit code {result.exit_code}). See {result.log_path}")
                results.append(result)
                break

            print_fn(f"  completed in {result.duration_seconds:.1f}s")
            results.append(result)

        # register outputs
        stage_outputs[stage_id] = register_stage_outputs(stage_id, run_dir)

    return results


def _stage_output_exists(stage, run_dir: Path, profile: PipelineProfile) -> bool:
    """Check if a stage's output already exists on disk."""
    if not stage.output_files:
        # stages with dynamic output (5, 6) -- check first mode dir
        if stage.stage_id == "5":
            first_mode = profile.modes[0] if profile.modes else "singlecall"
            return (run_dir / stage.output_dir / first_mode / "results-all.json").exists()
        if stage.stage_id == "6":
            return (run_dir / stage.output_dir / "cross-mode").exists()
        return False

    for output_file in stage.output_files:
        if stage.output_dir:
            path = run_dir / stage.output_dir / output_file
        else:
            path = run_dir / output_file
        if not path.exists():
            return False
    return True


def _execute_stage5(stage, profile, run_dir, repo_root, pipeline_dir,
                    logs_dir, stage_outputs, verbose, print_fn):
    """Execute stage 5 (corpus evaluation) once per mode."""
    results = []
    stage5_dir = run_dir / "stage5-corpus-evaluation"

    for mode in profile.modes:
        print_fn(f"  --- mode: {mode} ---")
        (stage5_dir / mode).mkdir(parents=True, exist_ok=True)

        result_file = stage5_dir / mode / "results-all.json"
        if result_file.exists():
            print_fn(f"  [skip] {mode} results exist")
            continue

        args = build_stage_args("5", profile, run_dir, repo_root, stage_outputs, mode=mode)
        log_path = logs_dir / f"stage5-{mode}.log"

        result = run_stage_command(
            pipeline_dir=pipeline_dir,
            command=stage.commands[0],
            args=args,
            log_path=log_path,
            verbose=verbose,
        )
        result.stage_id = "5"

        if result.exit_code != 0:
            print_fn(f"  FAILED mode={mode} (exit code {result.exit_code}). See {result.log_path}")
            results.append(result)
            return results

        print_fn(f"  {mode} completed in {result.duration_seconds:.1f}s")
        results.append(result)

    return results


def _execute_stage6(stage, profile, run_dir, repo_root, pipeline_dir,
                    logs_dir, stage_outputs, verbose, print_fn):
    """Execute stage 6 (visualization): merge results then generate charts."""
    stage5_dir = run_dir / "stage5-corpus-evaluation"
    stage6_dir = run_dir / "stage6-visualization"

    # merge results from all modes (in-process, no subprocess needed)
    all_merged_path = stage5_dir / "results-all-merged.json"
    all_episodes = []
    for mode in profile.modes:
        mode_files = sorted((stage5_dir / mode).glob("results-*.json"))
        for f in mode_files:
            with open(f, "r", encoding="utf-8") as fh:
                all_episodes.extend(json.load(fh))
    with open(all_merged_path, "w", encoding="utf-8") as fh:
        json.dump(all_episodes, fh, indent=2)
    print_fn(f"  merged {len(all_episodes)} episodes from {len(profile.modes)} modes")

    # per-mode heatmaps
    for mode in profile.modes:
        mode_merged = stage5_dir / f"results-{mode}-merged.json"
        mode_episodes = []
        for f in sorted((stage5_dir / mode).glob("results-*.json")):
            with open(f, "r", encoding="utf-8") as fh:
                mode_episodes.extend(json.load(fh))
        with open(mode_merged, "w", encoding="utf-8") as fh:
            json.dump(mode_episodes, fh, indent=2)

        mode_dir = stage6_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        args = [
            "--results", str(mode_merged),
            "-o", str(mode_dir),
            "--type", "all",
            "--dpi", "200",
        ]
        log_path = logs_dir / f"stage6-{mode}.log"
        run_stage_command(
            pipeline_dir=pipeline_dir,
            command="heatmaps",
            args=args,
            log_path=log_path,
            verbose=verbose,
        )

    # cross-mode charts
    cross_dir = stage6_dir / "cross-mode"
    cross_dir.mkdir(parents=True, exist_ok=True)

    args = build_stage_args("6", profile, run_dir, repo_root, stage_outputs)
    log_path = logs_dir / "stage6-cross-mode.log"

    result = run_stage_command(
        pipeline_dir=pipeline_dir,
        command="heatmaps",
        args=args,
        log_path=log_path,
        verbose=verbose,
    )
    result.stage_id = "6"
    print_fn(f"  visualization completed in {result.duration_seconds:.1f}s")
    return result


def _execute_stage7(stage, profile, run_dir, repo_root, pipeline_dir,
                    logs_dir, stage_outputs, verbose, print_fn):
    """Execute stage 7 (traceability report + CSV export)."""
    results = []

    # command 1: traceability-report
    args = build_stage_args("7", profile, run_dir, repo_root, stage_outputs)
    log_path = logs_dir / "stage7-traceability.log"
    result = run_stage_command(
        pipeline_dir=pipeline_dir,
        command="traceability-report",
        args=args,
        log_path=log_path,
        verbose=verbose,
    )
    result.stage_id = "7"
    results.append(result)

    if result.exit_code != 0:
        print_fn(f"  traceability-report FAILED. See {result.log_path}")
        return results

    # command 2: export-csv
    csv_args = build_stage7_csv_args(profile, run_dir, stage_outputs)
    csv_log_path = logs_dir / "stage7-csv.log"
    csv_result = run_stage_command(
        pipeline_dir=pipeline_dir,
        command="export-csv",
        args=csv_args,
        log_path=csv_log_path,
        verbose=verbose,
    )
    csv_result.stage_id = "7"
    results.append(csv_result)

    print_fn(f"  traceability + CSV completed")
    return results
