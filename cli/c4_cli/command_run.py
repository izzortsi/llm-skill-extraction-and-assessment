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
    --quiet              Suppress stage output
    --interactive        Build profile interactively before running
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from c0_config.pipeline_profile import PipelineProfile, apply_minimal
from c1_tools.profile_loader import load_profile, save_profile, PROFILES_DIR
from c2_orchestration.pipeline_executor import execute_pipeline
from c4_cli.rich_ui import print_header, print_summary


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
    parser.add_argument("--clean-stages", action="store_true",
                        help="Wipe output of only the requested stages before running")
    parser.add_argument("--minimal", action="store_true",
                        help="Override with minimal settings (fewest API calls)")
    parser.add_argument("--run-dir", type=str, default="",
                        help="Override run directory path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress stage subprocess output")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Build profile interactively before running")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent

    # interactive mode: build profile via prompts
    if args.interactive:
        from c4_cli.interactive import build_profile_interactive, prompt_confirm
        profile = build_profile_interactive()
        if prompt_confirm("Save this profile?", True):
            path = save_profile(profile)
            print(f"Saved to {path}")
        if not prompt_confirm("Run pipeline now?", True):
            return
    elif args.profile:
        try:
            profile = load_profile(args.profile)
        except FileNotFoundError:
            print(f"Profile '{args.profile}' not found in {PROFILES_DIR}")
            sys.exit(1)
    else:
        profile = PipelineProfile()

    if args.minimal:
        apply_minimal(profile)

    if args.run_dir:
        profile.run_dir = args.run_dir

    verbose = not args.quiet

    print_header(
        profile_name=profile.profile_name,
        stages=args.stages,
        run_dir=profile.run_dir,
        is_minimal=args.minimal,
        is_clean=args.clean,
    )

    results = execute_pipeline(
        profile=profile,
        stage_range=args.stages,
        repo_root=repo_root,
        clean=args.clean,
        clean_stages=args.clean_stages,
        verbose=verbose,
    )

    print_summary(results)
