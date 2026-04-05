"""
report.py

CLI command: generate SkillMix report from results.

Parses arguments and delegates to skillmix.report functional APIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from skillmix.report import generate_report


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the report subcommand."""
    parser = subparsers.add_parser(
        "report",
        help="Generate SkillMix report from results",
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory with experiment results")
    parser.add_argument("--output", "-o", type=Path, help="Output report file (default: stdout)")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the report command."""
    report = generate_report(args.results_dir)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)
