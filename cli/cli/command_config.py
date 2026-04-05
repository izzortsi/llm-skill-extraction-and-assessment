"""
command_config.py

The "config" command: manage experiment profiles.

Usage:
    python -m cli.main config <subcommand> [args]

Subcommands:
    create NAME     Create a new profile with default values
    list            List all saved profiles
    show NAME       Display profile contents
    delete NAME     Delete a profile
"""

from __future__ import annotations

import argparse
import sys

from config.pipeline_profile import PipelineProfile
from tools.profile_loader import (
    load_profile,
    save_profile,
    list_profiles,
    delete_profile,
    PROFILES_DIR,
)
from cli.rich_ui import console, print_profile, print_profiles_list


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-skills config",
        description="Manage experiment profiles",
    )
    parser.add_argument("subcommand", choices=["create", "list", "show", "delete"],
                        help="Config subcommand")
    parser.add_argument("name", nargs="?", default="",
                        help="Profile name (for create, show, delete)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Create profile interactively (for 'create' subcommand)")

    args = parser.parse_args()

    if args.subcommand == "list":
        print_profiles_list(list_profiles())

    elif args.subcommand == "create":
        if args.interactive:
            from pathlib import Path
            from cli.interactive import build_profile_interactive
            from tools.provider_discovery import discover_providers
            repo_root = Path(__file__).resolve().parent.parent.parent
            profile = PipelineProfile()
            cfg_path = repo_root / profile.config_file
            providers = discover_providers(
                lmproxy_url=profile.lmproxy_base_url,
                ollama_url=profile.ollama_url,
                config_file=str(cfg_path) if cfg_path.exists() else "",
            )
            profile = build_profile_interactive(providers)
            if args.name:
                profile.profile_name = args.name
            path = save_profile(profile)
            console.print(f"Created profile '[cyan]{profile.profile_name}[/cyan]' at {path}")
        else:
            if not args.name:
                console.print("Usage: llm-skills config create <name>")
                console.print("  or:  llm-skills config create --interactive")
                sys.exit(1)
            profile = PipelineProfile(profile_name=args.name)
            path = save_profile(profile)
            console.print(f"Created profile '[cyan]{args.name}[/cyan]' at {path}")
            console.print(f"Edit the YAML file, then run: [bold]python3 -m cli.main run --profile {args.name}[/bold] (from cli/)")

    elif args.subcommand == "show":
        if not args.name:
            console.print("Usage: llm-skills config show <name>")
            sys.exit(1)
        try:
            profile = load_profile(args.name)
        except FileNotFoundError:
            console.print(f"Profile '{args.name}' not found in {PROFILES_DIR}")
            sys.exit(1)
        print_profile(profile)

    elif args.subcommand == "delete":
        if not args.name:
            console.print("Usage: llm-skills config delete <name>")
            sys.exit(1)
        if delete_profile(args.name):
            console.print(f"Deleted profile '[cyan]{args.name}[/cyan]'")
        else:
            console.print(f"Profile '{args.name}' not found")
            sys.exit(1)
