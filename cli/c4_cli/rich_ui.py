"""
rich_ui.py

Rich-powered UI components for the llm-skills CLI. All Rich usage is
centralized here. If Rich is not installed, all functions degrade to
plain print() output.

Components:
    console         -- pre-configured Rich Console (or plain fallback)
    print_header    -- pipeline run header with profile summary
    print_stage     -- stage start/skip/complete/fail indicators
    print_summary   -- final results table
    print_status    -- run directory status table
    print_profile   -- profile config display
    print_profiles  -- profile list display
"""

from __future__ import annotations

import re
import sys
from typing import List

# graceful import: Rich is optional
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ---------------------------------------------------------------------------
# Console singleton
# ---------------------------------------------------------------------------

if HAS_RICH:
    console = Console()
else:
    class _PlainConsole:
        # known Rich style tags to strip; keeps non-style brackets like [skip], [done]
        _RICH_TAGS = re.compile(
            r"\[/?"
            r"(?:bold|italic|underline|strike|dim|reverse|blink|"
            r"red|green|blue|yellow|magenta|cyan|white|black|"
            r"bold red|bold cyan|bold green)"
            r"\]"
        )

        def print(self, *args, **kwargs):
            text = " ".join(str(a) for a in args)
            text = self._RICH_TAGS.sub("", text)
            print(text)
    console = _PlainConsole()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def print_header(profile_name: str, stages: str, run_dir: str,
                 is_minimal: bool, is_clean: bool) -> None:
    """Print pipeline run header."""
    if HAS_RICH:
        lines = []
        lines.append(f"[bold]profile:[/bold]  {profile_name}")
        lines.append(f"[bold]stages:[/bold]   {stages}")
        lines.append(f"[bold]run_dir:[/bold]  {run_dir}")
        if is_minimal:
            lines.append("[yellow]mode:     MINIMAL (1 chunk, 1 task/chunk, 3 skills, singlecall, 1 model)[/yellow]")
        if is_clean:
            lines.append("[red]clean:    wiping previous output[/red]")
        content = "\n".join(lines)
        console.print(Panel(content, title="[bold cyan]llm-skills pipeline[/bold cyan]", border_style="cyan"))
    else:
        console.print(f"llm-skills pipeline")
        console.print(f"  profile:  {profile_name}")
        console.print(f"  stages:   {stages}")
        console.print(f"  run_dir:  {run_dir}")
        if is_minimal:
            console.print(f"  mode:     MINIMAL")
        if is_clean:
            console.print(f"  clean:    yes")
        console.print("")


# ---------------------------------------------------------------------------
# Stage indicators
# ---------------------------------------------------------------------------

def print_stage_start(stage_id: str, description: str) -> None:
    """Print stage start header."""
    if HAS_RICH:
        console.print(Rule(f"[bold]Stage {stage_id}[/bold]: {description}", style="blue"))
    else:
        console.print(f"\n=== Stage {stage_id}: {description} ===")


def print_stage_skip(stage_id: str) -> None:
    """Print stage skip indicator."""
    if HAS_RICH:
        console.print(f"  [dim]\\[skip][/dim] output already exists")
    else:
        console.print(f"  [skip] output already exists")


def print_stage_complete(stage_id: str, duration: float) -> None:
    """Print stage completion indicator."""
    if HAS_RICH:
        console.print(f"  [green]\\[done][/green] completed in {duration:.1f}s")
    else:
        console.print(f"  [done] completed in {duration:.1f}s")


def print_stage_fail(stage_id: str, exit_code: int, log_path: str) -> None:
    """Print stage failure indicator."""
    if HAS_RICH:
        console.print(f"  [bold red]\\[FAIL][/bold red] exit code {exit_code}")
        console.print(f"  [dim]log: {log_path}[/dim]")
    else:
        console.print(f"  [FAIL] exit code {exit_code}")
        console.print(f"  log: {log_path}")


def print_stage_mode(mode: str) -> None:
    """Print evaluation mode sub-header."""
    if HAS_RICH:
        console.print(f"  [cyan]--- mode: {mode} ---[/cyan]")
    else:
        console.print(f"  --- mode: {mode} ---")


def print_stage_info(message: str) -> None:
    """Print informational message within a stage."""
    if HAS_RICH:
        console.print(f"  [dim]{message}[/dim]")
    else:
        console.print(f"  {message}")


def print_dependency_error(stage_id: str, missing: List[str]) -> None:
    """Print missing dependency error."""
    if HAS_RICH:
        console.print(f"  [bold red]ERROR:[/bold red] missing dependencies: stages {', '.join(missing)}")
        console.print(f"  [dim]run those stages first, or use --stages all[/dim]")
    else:
        console.print(f"  ERROR: missing dependencies: stages {', '.join(missing)}")
        console.print(f"  run those stages first, or use --stages all")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list) -> None:
    """Print pipeline completion summary table."""
    total_time = 0.0
    failed = 0
    skipped = 0

    if HAS_RICH:
        table = Table(title="Pipeline Summary", border_style="cyan")
        table.add_column("Stage", style="bold", width=6)
        table.add_column("Status", width=8)
        table.add_column("Time", justify="right", width=8)
        table.add_column("Command", style="dim")

        for r in results:
            total_time += r.duration_seconds

            if r.exit_code != 0:
                status = "[bold red]FAIL[/bold red]"
                failed += 1
            elif r.command == "(skipped)":
                status = "[dim]SKIP[/dim]"
                skipped += 1
            else:
                status = "[green]OK[/green]"

            time_str = f"{r.duration_seconds:.1f}s" if r.duration_seconds > 0 else "-"
            table.add_row(r.stage_id, status, time_str, r.command)

        console.print("")
        console.print(table)

        total_count = len(results)
        ok_count = total_count - failed - skipped
        summary = f"[bold]{total_count}[/bold] stages"
        if ok_count > 0:
            summary += f", [green]{ok_count} ok[/green]"
        if skipped > 0:
            summary += f", [dim]{skipped} skipped[/dim]"
        if failed > 0:
            summary += f", [bold red]{failed} failed[/bold red]"
        summary += f", {total_time:.1f}s total"
        console.print(summary)
    else:
        console.print("")
        console.print("=" * 60)
        console.print("Pipeline Summary")
        console.print("=" * 60)

        for r in results:
            total_time += r.duration_seconds
            status = "OK"
            if r.exit_code != 0:
                status = "FAILED"
                failed += 1
            elif r.command == "(skipped)":
                status = "SKIP"
                skipped += 1
            console.print(f"  stage {r.stage_id:<4} {status:<8} {r.duration_seconds:>6.1f}s  {r.command}")

        console.print(f"\nTotal: {len(results)} stages, {failed} failed, {skipped} skipped, {total_time:.1f}s")


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------

def print_status_table(run_dir: str, statuses: list) -> None:
    """Print run directory status as a table."""
    if HAS_RICH:
        console.print(f"[bold]Run directory:[/bold] {run_dir}\n")

        table = Table(border_style="dim")
        table.add_column("Stage", style="bold", width=6)
        table.add_column("Name", width=22)
        table.add_column("Status", width=10)
        table.add_column("Output", style="dim")

        for s in statuses:
            if s.is_complete:
                status = "[green]DONE[/green]"
                output = ", ".join(s.output_paths)
            elif len(s.output_paths) > 0:
                status = "[yellow]PARTIAL[/yellow]"
                output = f"{len(s.output_paths)} of {len(s.output_paths) + len(s.missing_paths)} files"
            else:
                status = "[dim]PENDING[/dim]"
                output = ""

            table.add_row(s.stage_id, s.name, status, output)

        console.print(table)

        done = sum(1 for s in statuses if s.is_complete)
        console.print(f"\n[bold]{done}[/bold] of {len(statuses)} stages complete")
    else:
        console.print(f"Run directory: {run_dir}\n")
        console.print(f"{'Stage':<6} {'Name':<22} {'Status':<10} {'Output'}")
        console.print("-" * 70)
        for s in statuses:
            if s.is_complete:
                status_str = "DONE"
                output_str = ", ".join(s.output_paths)
            elif len(s.output_paths) > 0:
                status_str = "PARTIAL"
                output_str = f"{len(s.output_paths)} of {len(s.output_paths) + len(s.missing_paths)} files"
            else:
                status_str = "PENDING"
                output_str = ""
            console.print(f"{s.stage_id:<6} {s.name:<22} {status_str:<10} {output_str}")
        done = sum(1 for s in statuses if s.is_complete)
        console.print(f"\n{done} of {len(statuses)} stages complete")


# ---------------------------------------------------------------------------
# Profile display
# ---------------------------------------------------------------------------

def print_profile(profile) -> None:
    """Print a profile's configuration."""
    if HAS_RICH:
        table = Table(title=f"Profile: {profile.profile_name}", border_style="cyan",
                      show_header=False, padding=(0, 2))
        table.add_column("Key", style="bold", width=22)
        table.add_column("Value")

        table.add_row("run_dir", profile.run_dir)
        table.add_row("", "")
        table.add_row("[cyan]Source[/cyan]", "")
        table.add_row("  dataset", profile.dataset)
        table.add_row("  subset", profile.subset)
        table.add_row("  domain", profile.domain)
        table.add_row("  max_chunks", str(profile.max_chunks))
        table.add_row("  chunk_size", str(profile.chunk_size))
        table.add_row("  tasks_per_chunk", str(profile.tasks_per_chunk))
        table.add_row("", "")
        table.add_row("[cyan]Extraction[/cyan]", "")
        table.add_row("  provider", profile.extraction_provider)
        table.add_row("  model", profile.extraction_model)
        table.add_row("  max_skills", str(profile.max_skills))
        table.add_row("", "")
        table.add_row("[cyan]Trace Capture[/cyan]", "")
        table.add_row("  provider", profile.trace_provider)
        table.add_row("  model", profile.trace_model)
        table.add_row("", "")
        table.add_row("[cyan]Evaluation[/cyan]", "")
        table.add_row("  config_file", profile.config_file)
        table.add_row("  modes", ", ".join(profile.modes))
        table.add_row("  eval_models", ", ".join(profile.eval_models))
        table.add_row("  ollama_url", profile.ollama_url)
        table.add_row("", "")
        table.add_row("[cyan]Judge[/cyan]", "")
        table.add_row("  provider", profile.judge_provider)
        table.add_row("  model", profile.judge_model)

        console.print(table)
    else:
        # plain fallback (existing implementation)
        console.print(f"Profile: {profile.profile_name}")
        console.print(f"  run_dir:              {profile.run_dir}")
        console.print(f"  dataset:              {profile.dataset}")
        console.print(f"  subset:               {profile.subset}")
        console.print(f"  domain:               {profile.domain}")
        console.print(f"  extraction_model:     {profile.extraction_model}")
        console.print(f"  trace_model:          {profile.trace_model}")
        console.print(f"  modes:                {', '.join(profile.modes)}")
        console.print(f"  judge_model:          {profile.judge_model}")


def print_profiles_list(profiles: List[str]) -> None:
    """Print list of available profiles."""
    if not profiles:
        console.print("No profiles found. Create one: llm-skills config create <name>")
        return

    if HAS_RICH:
        console.print(f"[bold]Profiles[/bold] ({len(profiles)}):")
        for name in profiles:
            console.print(f"  [cyan]{name}[/cyan]")
    else:
        console.print(f"Profiles ({len(profiles)}):")
        for name in profiles:
            console.print(f"  {name}")
