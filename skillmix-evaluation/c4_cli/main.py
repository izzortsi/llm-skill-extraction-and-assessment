"""
main.py

CLI entry point for llm-skills.skillmix-evaluation.

Usage:
    python -m c4_cli.main <command> [options]

Commands:
    run-skillmix    Run SkillMix benchmark (multi-model, LLM-as-judge)
    report          Generate SkillMix report from results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_repo = _project.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import _bootstrap
_bootstrap.setup_project(_project)
_bootstrap.setup_skillsbench_evaluation()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="llm-skills.skillmix-evaluation: SkillMix benchmarking CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register subcommands
    from c4_cli.run_skillmix import add_parser as add_run_skillmix
    from c4_cli.report import add_parser as add_report

    add_run_skillmix(subparsers)
    add_report(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to the subcommand's run function
    args.func(args)


if __name__ == "__main__":
    main()
