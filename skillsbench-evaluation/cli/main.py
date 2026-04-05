"""
main.py

CLI entry point for skillsbench-evaluation.
Evaluation-related commands only (stages 5a, 6, 7).

Usage:
    python -m cli.main <command> [options]

Commands:
    run-skillsbench  Run SkillsBench corpus evaluation (Stage 5a)
    heatmaps         Generate heatmap visualizations from results (Stage 6)
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
_bootstrap.setup_providers()
_bootstrap.setup_shared_data()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="skillsbench-evaluation: Skill evaluation and benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  Stage 5a - SkillsBench Evaluation:
    run-skillsbench  Run SkillsBench corpus evaluation (baseline + skill-injected)

  Stage 6 - Visualization:
    heatmaps         Generate heatmap visualizations from results
""",
    )
    parser.add_argument("command", help="Command to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Command arguments")

    args = parser.parse_args()

    # Route to appropriate module
    if args.command == "run-skillsbench":
        sys.argv = ["run-skillsbench"] + args.args
        from cli.run_skillsbench import main as cmd_main
        cmd_main()

    elif args.command == "heatmaps":
        sys.argv = ["heatmaps"] + args.args
        from skillsbench.visualization import main as cmd_main
        cmd_main()

    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
