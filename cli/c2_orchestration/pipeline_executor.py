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

try:
    from c4_cli.rich_ui import (
        print_stage_start,
        print_stage_skip,
        print_stage_complete,
        print_stage_fail,
        print_stage_mode,
        print_stage_info,
        print_dependency_error,
    )
    _HAS_UI = True
except ImportError:
    _HAS_UI = False


def _plain_print(msg):
    print(msg)


def ui_stage_start(sid, desc):
    if _HAS_UI:
        print_stage_start(sid, desc)
    else:
        _plain_print(f"\n=== Stage {sid}: {desc} ===")


def ui_stage_skip(sid):
    if _HAS_UI:
        print_stage_skip(sid)
    else:
        _plain_print(f"  [skip] output already exists")


def ui_stage_complete(sid, duration):
    if _HAS_UI:
        print_stage_complete(sid, duration)
    else:
        _plain_print(f"  completed in {duration:.1f}s")


def ui_stage_fail(sid, code, log):
    if _HAS_UI:
        print_stage_fail(sid, code, log)
    else:
        _plain_print(f"  FAILED (exit code {code}). See {log}")


def ui_dep_error(sid, missing):
    if _HAS_UI:
        print_dependency_error(sid, missing)
    else:
        _plain_print(f"  ERROR: missing dependencies: stages {', '.join(missing)}")


def ui_mode(mode):
    if _HAS_UI:
        print_stage_mode(mode)
    else:
        _plain_print(f"  --- mode: {mode} ---")


def ui_info(msg):
    if _HAS_UI:
        print_stage_info(msg)
    else:
        _plain_print(f"  {msg}")


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
        ui_info(f"CLEAN: removing {run_dir}")
        shutil.rmtree(run_dir)

    # create output directories
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # parse stage range
    stage_ids = parse_stage_range(stage_range)

    # accumulate outputs from completed stages
    stage_outputs: Dict[str, Dict[str, str]] = {}

    # pre-populate outputs from existing files (for partial runs)
    for sid in ["1a", "1b", "2", "3", "4"]:
        stage_outputs[sid] = register_stage_outputs(sid, run_dir)

    results = []

    for stage_id in stage_ids:
        stage = get_stage(stage_id)
        pipeline_dir = repo_root / stage.pipeline_dir

        ui_stage_start(stage_id, stage.description)

        # check dependencies
        missing_deps = check_dependencies_met(stage, run_dir)
        if len(missing_deps) > 0:
            ui_dep_error(stage_id, missing_deps)
            break

        # check if output already exists (crash recovery)
        if _stage_output_exists(stage, run_dir, profile):
            ui_stage_skip(stage_id)
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
        if stage_id == "4b":
            stage_result = _execute_stage4b(
                stage, profile, run_dir, repo_root, pipeline_dir,
                logs_dir, stage_outputs, verbose, print_fn,
            )
            results.append(stage_result)
            if stage_result.exit_code != 0:
                break
            stage_outputs[stage_id] = register_stage_outputs(stage_id, run_dir)
            continue
        elif stage_id == "5":
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
                ui_stage_fail(stage_id, result.exit_code, result.log_path)
                results.append(result)
                break

            ui_stage_complete(stage_id, result.duration_seconds)
            results.append(result)

        # post-stage: convert JSON outputs to markdown for reuse
        _convert_stage_outputs_to_markdown(stage_id, run_dir, pipeline_dir, logs_dir, verbose)

        # register outputs
        stage_outputs[stage_id] = register_stage_outputs(stage_id, run_dir)

    return results


def _stage_output_exists(stage, run_dir: Path, profile: PipelineProfile) -> bool:
    """Check if a stage's output already exists on disk."""
    if not stage.output_files:
        # stages with dynamic output -- check all expected per-mode outputs
        if stage.stage_id == "5":
            modes = profile.modes if profile.modes else ["singlecall"]
            for mode in modes:
                if not (run_dir / stage.output_dir / mode / "results-all.json").exists():
                    return False
            return True
        if stage.stage_id == "6":
            cross_dir = run_dir / stage.output_dir / "cross-mode"
            if not cross_dir.exists():
                return False
            png_files = list(cross_dir.glob("*.png"))
            return len(png_files) > 0
        return False

    for output_file in stage.output_files:
        if stage.output_dir:
            path = run_dir / stage.output_dir / output_file
        else:
            path = run_dir / output_file
        if not path.exists():
            return False
    return True


_MARKDOWN_CONVERSIONS = {
    "1b": ("tasks", "stage1-task-extraction/tasks.json", "stage1-task-extraction/tasks-md", "tasks-to-md"),
    "3": ("skills", "stage3-skill-extraction/skills.json", "stage3-skill-extraction/skills-md", "skills-to-md"),
    "4": ("skills", "stage4-skill-verification/verified_skills.json", "stage4-skill-verification/verified-skills-md", "skills-to-md"),
}


def _convert_stage_outputs_to_markdown(stage_id, run_dir, pipeline_dir, logs_dir, verbose):
    """Convert JSON stage output to markdown files for human review and reuse."""
    if stage_id not in _MARKDOWN_CONVERSIONS:
        return

    entity_type, json_rel, md_dir_rel, format_cmd = _MARKDOWN_CONVERSIONS[stage_id]
    json_path = run_dir / json_rel
    md_dir = run_dir / md_dir_rel

    if not json_path.exists():
        return
    if md_dir.exists() and list(md_dir.glob("*.md")):
        return  # already converted

    ui_info(f"converting {entity_type} to markdown: {md_dir_rel}/")
    format_args = [format_cmd, "--input", str(json_path), "--output-dir", str(md_dir)]
    log_path = logs_dir / f"stage{stage_id}-format-md.log"
    run_stage_command(
        pipeline_dir=pipeline_dir,
        command="format",
        args=format_args,
        log_path=log_path,
        verbose=False,
    )


def _execute_stage4b(stage, profile, run_dir, repo_root, pipeline_dir,
                     logs_dir, stage_outputs, verbose, print_fn):
    """Execute stage 4b: convert verified skills to markdown, then compose."""
    stage4_dir = run_dir / "stage4-skill-verification"
    stage4b_dir = run_dir / "stage4b-skill-composition"
    stage4b_dir.mkdir(parents=True, exist_ok=True)
    atomic_md_dir = stage4b_dir / "atomic-skills-md"

    # step 1: convert verified_skills.json -> markdown directory
    skills_json = stage_outputs.get("4", {}).get("skills", str(stage4_dir / "verified_skills.json"))
    ui_info(f"converting {skills_json} to markdown")

    format_args = [
        "skills-to-md",
        "--input", str(skills_json),
        "--output-dir", str(atomic_md_dir),
    ]
    format_log = logs_dir / "stage4b-format.log"
    format_result = run_stage_command(
        pipeline_dir=pipeline_dir,
        command="format",
        args=format_args,
        log_path=format_log,
        verbose=verbose,
    )
    if format_result.exit_code != 0:
        ui_stage_fail("4b", format_result.exit_code, format_result.log_path)
        format_result.stage_id = "4b"
        return format_result

    # step 2: run compose-skills on the markdown directory
    args = build_stage_args("4b", profile, run_dir, repo_root, stage_outputs)
    log_path = logs_dir / "stage4b-compose.log"

    result = run_stage_command(
        pipeline_dir=pipeline_dir,
        command=stage.commands[0],
        args=args,
        log_path=log_path,
        verbose=verbose,
    )
    result.stage_id = "4b"

    if result.exit_code != 0:
        ui_stage_fail("4b", result.exit_code, result.log_path)
        return result

    ui_stage_complete("4b", format_result.duration_seconds + result.duration_seconds)
    return result


def _execute_stage5(stage, profile, run_dir, repo_root, pipeline_dir,
                    logs_dir, stage_outputs, verbose, print_fn):
    """Execute stage 5 (corpus evaluation) once per mode."""
    results = []
    stage5_dir = run_dir / "stage5-corpus-evaluation"

    for mode in profile.modes:
        ui_mode(mode)
        (stage5_dir / mode).mkdir(parents=True, exist_ok=True)

        result_file = stage5_dir / mode / "results-all.json"
        if result_file.exists():
            ui_info(f"[skip] {mode} results exist")
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
            ui_stage_fail("5", result.exit_code, result.log_path)
            results.append(result)
            return results

        ui_stage_complete("5", result.duration_seconds)
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
    ui_info(f"merged {len(all_episodes)} episodes from {len(profile.modes)} modes")

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
        mode_result = run_stage_command(
            pipeline_dir=pipeline_dir,
            command="heatmaps",
            args=args,
            log_path=log_path,
            verbose=verbose,
        )
        if mode_result.exit_code != 0:
            ui_stage_fail("6", mode_result.exit_code, mode_result.log_path)
            mode_result.stage_id = "6"
            return mode_result

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

    if result.exit_code != 0:
        ui_stage_fail("6", result.exit_code, result.log_path)
        return result

    ui_stage_complete("6", result.duration_seconds)
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
        ui_stage_fail("7", result.exit_code, result.log_path)
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

    ui_stage_complete("7", result.duration_seconds + csv_result.duration_seconds)
    return results
