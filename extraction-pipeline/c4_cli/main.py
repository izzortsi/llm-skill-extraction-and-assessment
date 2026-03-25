"""
main.py

CLI entry point for the llm-skills extraction library.

Usage:
    python -m c4_cli.main <command> [options]

Commands:
    extract-text       Extract text from PDF (Module 1)
    clean-text         Post-process extracted text (Module 1)
    extract-passages   Extract passages from source documents (Stage 1a)
    extract-tasks      Extract tasks from text artifact (Stage 1b)
    capture-traces     Run tasks to capture reasoning traces (Stage 2)
    extract-skills     Extract skills from reasoning traces (Stage 3)
    verify-skills      Verify extracted skills (Stage 4)
    compose-skills     Generate skill compositions (SkillMix)
    traceability-report  Generate traceability report
    export-csv         Generate CSV artifacts
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="llm-skills: extraction pipeline library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  Module 1 - Text Extraction:
    extract-text    Extract text from PDF via Marker
    clean-text      Post-process extracted text files

  Stage 1 - Passage & Task Extraction:
    extract-passages Extract passages from source documents
    extract-tasks   Extract evaluation tasks from text artifacts or passages

  Stage 2 - Trace Capture:
    capture-traces  Run tasks through LLM to capture reasoning traces

  Stage 3 - Skill Extraction:
    extract-skills  Extract reusable skills from reasoning traces

  Stage 4 - Skill Verification:
    verify-skills   Verify extracted skills for quality

  Composition:
    compose-skills  Generate skill compositions (seq, par, cond, sem)

  Reporting:
    traceability-report  Generate traceability report (source doc -> skill)
    export-csv      Generate CSV artifacts
""",
    )
    parser.add_argument("command", help="Command to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Command arguments")

    args = parser.parse_args()

    if args.command == "extract-text":
        sys.argv = ["extract-text"] + args.args
        from c1_tools.text_extractor import main as cmd_main
        cmd_main()

    elif args.command == "clean-text":
        sys.argv = ["clean-text"] + args.args
        from c1_tools.text_extractor import main as cmd_main
        cmd_main()

    elif args.command == "extract-passages":
        sys.argv = ["extract-passages"] + args.args
        from c2_extraction.passage_extractor import main as cmd_main
        cmd_main()

    elif args.command == "extract-tasks":
        sys.argv = ["extract-tasks"] + args.args
        from c2_extraction.task_extractor import main as cmd_main
        cmd_main()

    elif args.command == "capture-traces":
        sys.argv = ["capture-traces"] + args.args
        from c2_extraction.trace_runner import main as cmd_main
        cmd_main()

    elif args.command == "extract-skills":
        sys.argv = ["extract-skills"] + args.args
        from c2_extraction.skill_extractor import main as cmd_main
        cmd_main()

    elif args.command == "verify-skills":
        sys.argv = ["verify-skills"] + args.args
        from c2_extraction.skill_verifier import main as cmd_main
        cmd_main()

    elif args.command == "compose-skills":
        sys.argv = ["compose-skills"] + args.args
        from c2_composition.generator import main as cmd_main
        cmd_main()

    elif args.command == "traceability-report":
        sys.argv = ["traceability-report"] + args.args
        from c2_extraction.traceability_report import main as cmd_main
        cmd_main()

    elif args.command == "export-csv":
        sys.argv = ["export-csv"] + args.args
        from c2_extraction.csv_export import main as cmd_main
        cmd_main()

    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
