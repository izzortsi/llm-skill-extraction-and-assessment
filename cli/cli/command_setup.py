"""
command_setup.py

The "setup" command: pre-flight checks and first-run wizard.

Usage:
    python -m cli.main setup [--profile NAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config.pipeline_profile import PipelineProfile
from tools.provider_checker import run_preflight_checks
from cli.rich_ui import console, HAS_RICH


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-skills setup",
        description="Pre-flight checks and environment validation",
    )
    parser.add_argument("--profile", type=str, default="",
                        help="Profile to validate against")

    args = parser.parse_args()

    if args.profile:
        from tools.profile_loader import load_profile
        profile = load_profile(args.profile)
    else:
        profile = PipelineProfile()

    if HAS_RICH:
        console.print("[bold]llm-skills setup[/bold]\n")
    else:
        print("llm-skills setup\n")

    results = run_preflight_checks(profile)

    passed = 0
    failed = 0
    for r in results:
        if r.passed:
            passed += 1
            if HAS_RICH:
                console.print(f"  [green]PASS[/green]  {r.name}: {r.message}")
            else:
                print(f"  PASS  {r.name}: {r.message}")
        else:
            failed += 1
            if HAS_RICH:
                console.print(f"  [red]FAIL[/red]  {r.name}: {r.message}")
            else:
                print(f"  FAIL  {r.name}: {r.message}")

    if HAS_RICH:
        console.print(f"\n[bold]{passed}[/bold] passed, [bold]{failed}[/bold] failed out of {len(results)} checks")
    else:
        print(f"\n{passed} passed, {failed} failed out of {len(results)} checks")

    if failed > 0:
        if HAS_RICH:
            console.print("[yellow]Some checks failed. The pipeline may not run correctly.[/yellow]")
        else:
            print("Some checks failed. The pipeline may not run correctly.")
