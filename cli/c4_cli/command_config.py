"""
command_config.py

The "config" command: manage experiment profiles.

Usage:
    python -m c4_cli.main config <subcommand> [args]

Subcommands:
    create NAME     Create a new profile with default values
    list            List all saved profiles
    show NAME       Display profile contents
    delete NAME     Delete a profile
"""

from __future__ import annotations

import argparse
import sys

from c0_config.pipeline_profile import PipelineProfile
from c1_tools.profile_loader import (
    load_profile,
    save_profile,
    list_profiles,
    delete_profile,
    PROFILES_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-skills config",
        description="Manage experiment profiles",
    )
    parser.add_argument("subcommand", choices=["create", "list", "show", "delete"],
                        help="Config subcommand")
    parser.add_argument("name", nargs="?", default="",
                        help="Profile name (for create, show, delete)")

    args = parser.parse_args()

    if args.subcommand == "list":
        _cmd_list()
    elif args.subcommand == "create":
        if not args.name:
            print("Usage: llm-skills config create <name>")
            sys.exit(1)
        _cmd_create(args.name)
    elif args.subcommand == "show":
        if not args.name:
            print("Usage: llm-skills config show <name>")
            sys.exit(1)
        _cmd_show(args.name)
    elif args.subcommand == "delete":
        if not args.name:
            print("Usage: llm-skills config delete <name>")
            sys.exit(1)
        _cmd_delete(args.name)


def _cmd_list() -> None:
    profiles = list_profiles()
    if not profiles:
        print(f"No profiles found in {PROFILES_DIR}")
        print(f"Create one: llm-skills config create <name>")
        return

    print(f"Profiles ({len(profiles)}):")
    for name in profiles:
        print(f"  {name}")


def _cmd_create(name: str) -> None:
    profile = PipelineProfile(profile_name=name)
    path = save_profile(profile)
    print(f"Created profile '{name}' at {path}")
    print(f"Edit the YAML file to customize, then run: llm-skills run --profile {name}")


def _cmd_show(name: str) -> None:
    try:
        profile = load_profile(name)
    except FileNotFoundError:
        print(f"Profile '{name}' not found in {PROFILES_DIR}")
        sys.exit(1)

    print(f"Profile: {profile.profile_name}")
    print(f"  run_dir:              {profile.run_dir}")
    print()
    print(f"  Source:")
    print(f"    dataset:            {profile.dataset}")
    print(f"    subset:             {profile.subset}")
    print(f"    domain:             {profile.domain}")
    print(f"    max_chunks:         {profile.max_chunks}")
    print(f"    chunk_size:         {profile.chunk_size}")
    print(f"    tasks_per_chunk:    {profile.tasks_per_chunk}")
    print()
    print(f"  Extraction:")
    print(f"    provider:           {profile.extraction_provider}")
    print(f"    model:              {profile.extraction_model}")
    print(f"    max_skills:         {profile.max_skills}")
    print()
    print(f"  Trace capture:")
    print(f"    provider:           {profile.trace_provider}")
    print(f"    model:              {profile.trace_model}")
    print()
    print(f"  Evaluation:")
    print(f"    config_file:        {profile.config_file}")
    print(f"    modes:              {', '.join(profile.modes)}")
    print(f"    eval_models:        {', '.join(profile.eval_models)}")
    print(f"    ollama_url:         {profile.ollama_url}")
    print()
    print(f"  Judge:")
    print(f"    provider:           {profile.judge_provider}")
    print(f"    model:              {profile.judge_model}")


def _cmd_delete(name: str) -> None:
    if delete_profile(name):
        print(f"Deleted profile '{name}'")
    else:
        print(f"Profile '{name}' not found")
        sys.exit(1)
