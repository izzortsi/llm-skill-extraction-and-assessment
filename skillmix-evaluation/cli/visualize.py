"""
visualize.py

CLI command: generate SkillMix charts from experiment results.

Parses arguments and delegates to analytics.visualizer functional APIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analytics.visualizer import generate_all


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the visualize subcommand."""
    parser = subparsers.add_parser(
        "visualize",
        help="Generate charts from SkillMix results",
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory with episodes.json and summary.json")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="Output directory for PNG files")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150)")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Execute the visualize command."""
    generated = generate_all(args.results_dir, args.output_dir, args.dpi)
    print(f"Generated {len(generated)} charts in {args.output_dir}")
    for path in generated:
        print(f"  {path}")
