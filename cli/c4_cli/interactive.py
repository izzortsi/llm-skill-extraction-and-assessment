"""
interactive.py

Interactive prompts for the llm-skills CLI. Uses Rich prompts when
available, falls back to plain input() otherwise.

Functions:
    prompt_provider     -- select LLM provider
    prompt_model        -- select model for a provider
    prompt_dataset      -- select data source
    prompt_modes        -- select evaluation scaffolding modes
    prompt_stages       -- select stage range
    prompt_confirm      -- yes/no confirmation
    prompt_profile_name -- name for a new profile
    build_profile_interactive -- full interactive profile builder
"""

from __future__ import annotations

from typing import List

from c0_config.pipeline_profile import PipelineProfile

try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.panel import Panel
    _console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _ask(question: str, default: str = "") -> str:
    if HAS_RICH:
        return Prompt.ask(question, default=default, console=_console)
    else:
        suffix = f" [{default}]" if default else ""
        return input(f"{question}{suffix}: ").strip() or default


def _ask_int(question: str, default: int = 0) -> int:
    if HAS_RICH:
        return IntPrompt.ask(question, default=default, console=_console)
    else:
        raw = input(f"{question} [{default}]: ").strip()
        return int(raw) if raw else default


def _confirm(question: str, default: bool = True) -> bool:
    if HAS_RICH:
        return Confirm.ask(question, default=default, console=_console)
    else:
        suffix = " [Y/n]" if default else " [y/N]"
        raw = input(f"{question}{suffix}: ").strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes")


def _choose(question: str, choices: List[str], default: str = "") -> str:
    if HAS_RICH:
        choices_str = ", ".join(choices)
        _console.print(f"  Options: [cyan]{choices_str}[/cyan]")
        return Prompt.ask(question, default=default, console=_console)
    else:
        print(f"  Options: {', '.join(choices)}")
        return input(f"{question} [{default}]: ").strip() or default


def _multi_choose(question: str, choices: List[str], defaults: List[str] = None) -> List[str]:
    if defaults is None:
        defaults = []
    defaults_str = ",".join(defaults)

    if HAS_RICH:
        choices_str = ", ".join(choices)
        _console.print(f"  Options: [cyan]{choices_str}[/cyan]")
        raw = Prompt.ask(f"{question} (comma-separated)", default=defaults_str, console=_console)
    else:
        print(f"  Options: {', '.join(choices)}")
        raw = input(f"{question} (comma-separated) [{defaults_str}]: ").strip() or defaults_str

    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# High-level prompts
# ---------------------------------------------------------------------------

def prompt_provider(label: str, default: str = "anthropic") -> str:
    return _choose(f"{label} provider", ["anthropic", "openai", "ollama"], default)


def prompt_model(label: str, provider: str, default: str = "claude-opus-4-6") -> str:
    if provider == "anthropic":
        choices = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
    elif provider in ("openai", "ollama"):
        choices = ["qwen2.5:3b", "qwen2.5:7b", "qwen3:0.6b", "llama3.2:1b", "glm-5-turbo"]
    else:
        choices = [default]
    return _choose(f"{label} model", choices, default)


def prompt_dataset() -> tuple:
    dataset = _ask("Dataset (HuggingFace identifier)", "wikimedia/wikipedia")
    subset = _ask("Subset", "20231101.en")
    domain = _ask("Domain label", "language-skills")
    return dataset, subset, domain


def prompt_modes() -> List[str]:
    return _multi_choose(
        "Evaluation modes",
        ["singlecall", "stepwise", "guided"],
        ["singlecall", "stepwise", "guided"],
    )


def prompt_stages() -> str:
    return _choose(
        "Stage range",
        ["all", "extraction (1-4)", "evaluation (5-7)", "1a", "1-4", "5-7"],
        "all",
    )


def prompt_confirm(message: str, default: bool = True) -> bool:
    return _confirm(message, default)


def prompt_profile_name() -> str:
    return _ask("Profile name", "default")


# ---------------------------------------------------------------------------
# Full interactive profile builder
# ---------------------------------------------------------------------------

def build_profile_interactive() -> PipelineProfile:
    """Walk the user through building a complete pipeline profile."""
    if HAS_RICH:
        _console.print(Panel("[bold]Pipeline Configuration[/bold]", border_style="cyan"))
    else:
        print("\n=== Pipeline Configuration ===\n")

    profile = PipelineProfile()

    # source data
    if HAS_RICH:
        _console.print("\n[bold cyan]1. Source Data[/bold cyan]")
    else:
        print("\n1. Source Data")

    profile.dataset, profile.subset, profile.domain = prompt_dataset()
    profile.max_chunks = _ask_int("Max chunks", 5)
    profile.chunk_size = _ask_int("Chunk size (chars)", 4000)
    profile.tasks_per_chunk = _ask_int("Tasks per chunk", 2)

    # extraction model
    if HAS_RICH:
        _console.print("\n[bold cyan]2. Extraction Model[/bold cyan]")
    else:
        print("\n2. Extraction Model")

    profile.extraction_provider = prompt_provider("Extraction", "anthropic")
    profile.extraction_model = prompt_model("Extraction", profile.extraction_provider, "claude-opus-4-6")
    profile.max_skills = _ask_int("Max skills to extract", 8)

    # trace capture
    if HAS_RICH:
        _console.print("\n[bold cyan]3. Trace Capture[/bold cyan]")
    else:
        print("\n3. Trace Capture")

    use_same = _confirm("Use same provider/model for trace capture?", True)
    if use_same:
        profile.trace_provider = profile.extraction_provider
        profile.trace_model = profile.extraction_model
    else:
        profile.trace_provider = prompt_provider("Trace", "anthropic")
        profile.trace_model = prompt_model("Trace", profile.trace_provider, "claude-opus-4-6")

    # evaluation
    if HAS_RICH:
        _console.print("\n[bold cyan]4. Evaluation[/bold cyan]")
    else:
        print("\n4. Evaluation")

    profile.modes = prompt_modes()
    profile.ollama_url = _ask("Ollama URL", "http://localhost:11434/v1")

    # judge
    if HAS_RICH:
        _console.print("\n[bold cyan]5. Judge[/bold cyan]")
    else:
        print("\n5. Judge")

    profile.judge_provider = prompt_provider("Judge", "anthropic")
    profile.judge_model = prompt_model("Judge", profile.judge_provider, "claude-opus-4-6")

    # profile name
    if HAS_RICH:
        _console.print("\n[bold cyan]6. Save[/bold cyan]")
    else:
        print("\n6. Save")

    profile.profile_name = prompt_profile_name()

    return profile
