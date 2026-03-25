"""
command_run.py

The "run" command: execute pipeline stages with profile-based configuration.

Usage:
    python -m c4_cli.main run [options]

Options:
    --profile NAME       Use saved profile (default: "default")
    --stages RANGE       Stage range: all, 1-4, 5-7, extraction, evaluation (default: all)
    --clean              Wipe previous output and re-run
    --minimal            Override profile with minimal settings
    --run-dir PATH       Override run directory
    --verbose            Verbose output (default: True)
    --quiet              Suppress stage output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from c0_config.pipeline_profile import PipelineProfile, apply_minimal
from c1_tools.profile_loader import load_profile, PROFILES_DIR
from c2_orchestration.pipeline_executor import execute_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-skills run",
        description="Run pipeline stages",
    )
    parser.add_argument("--profile", type=str, default="",
                        help="Named profile from profiles/ directory")
    parser.add_argument("--stages", type=str, default="all",
                        help="Stage range: all, 1-4, 5-7, extraction, evaluation, 1a,3,5")
    parser.add_argument("--clean", action="store_true",
                        help="Wipe previous output and re-run all stages")
    parser.add_argument("--minimal", action="store_true",
                        help="Override with minimal settings (fewest API calls)")
    parser.add_argument("--run-dir", type=str, default="",
                        help="Override run directory path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress stage subprocess output")

    args = parser.parse_args()

    # resolve repo root (parent of llm-skills.cli/)
    repo_root = Path(__file__).resolve().parent.parent.parent

    # load or create profile
    if args.profile:
        try:
            profile = load_profile(args.profile)
        except FileNotFoundError:
            print(f"Profile '{args.profile}' not found in {PROFILES_DIR}")
            print(f"Available profiles: {', '.join(_list_profiles())}")
            sys.exit(1)
    else:
        profile = PipelineProfile()

    # apply overrides
    if args.minimal:
        apply_minimal(profile)

    if args.run_dir:
        profile.run_dir = args.run_dir

    verbose = not args.quiet

    # print header
    print(f"llm-skills pipeline")
    print(f"  profile:  {profile.profile_name}")
    print(f"  stages:   {args.stages}")
    print(f"  run_dir:  {profile.run_dir}")
    if args.minimal:
        print(f"  mode:     MINIMAL (1 chunk, 1 task/chunk, 3 skills, singlecall, 1 model)")
    if args.clean:
        print(f"  clean:    yes (wipe previous output)")
    print()

    # execute
    results = execute_pipeline(
        profile=profile,
        stage_range=args.stages,
        repo_root=repo_root,
        clean=args.clean,
        verbose=verbose,
    )

    # summary
    print()
    print("=" * 60)
    print("Pipeline Summary")
    print("=" * 60)

    total_time = 0.0
    failed = 0
    skipped = 0

    for r in results:
        status = "OK"
        if r.exit_code != 0:
            status = "FAILED"
            failed += 1
        elif r.command == "(skipped)":
            status = "SKIP"
            skipped += 1

        total_time += r.duration_seconds
        print(f"  stage {r.stage_id:<4} {status:<8} {r.duration_seconds:>6.1f}s  {r.command}")

    print(f"\nTotal: {len(results)} stages, {failed} failed, {skipped} skipped, {total_time:.1f}s")


def _list_profiles():
    from c1_tools.profile_loader import list_profiles
    return list_profiles()
