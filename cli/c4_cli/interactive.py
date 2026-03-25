"""
interactive.py

Interactive prompts for the llm-skills CLI. Uses InquirerPy when available
for rich list/checkbox selection, Rich for styled text, and falls back to
plain input() when neither is installed.

Functions:
    build_profile_interactive -- full interactive profile builder (takes provider_statuses)
"""

from __future__ import annotations

from typing import Dict, List

from c0_config.pipeline_profile import PipelineProfile

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    _console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
    HAS_INQUIRER = True
except ImportError:
    HAS_INQUIRER = False

    # Minimal Choice stub so isinstance() checks work in fallback paths
    class Choice:
        def __init__(self, value="", name="", enabled=True):
            self.value = value
            self.name = name
            self.enabled = enabled


# ---------------------------------------------------------------------------
# Primitive input helpers
# ---------------------------------------------------------------------------

def _print(text: str) -> None:
    """Print text, using Rich console when available."""
    if HAS_RICH:
        _console.print(text)
    else:
        print(text)


def _section(number: int, title: str) -> None:
    """Print a numbered section header."""
    if HAS_RICH:
        _console.print(f"\n[bold cyan]{number}. {title}[/bold cyan]")
    else:
        print(f"\n{number}. {title}")


def _select_one(message: str, choices: list, default: str = "") -> str:
    """Single select -- InquirerPy list or numbered input()."""
    if HAS_INQUIRER:
        result = inquirer.select(
            message=message,
            choices=choices,
            default=default if default else None,
        ).execute()
        return result

    # Fallback: numbered list
    print(f"\n  {message}")
    valid_indices = []
    for i, ch in enumerate(choices):
        if isinstance(ch, Choice):
            if ch.enabled is False:
                print(f"    {i + 1}. {ch.name}  (disabled)")
            else:
                print(f"    {i + 1}. {ch.name}")
                valid_indices.append(i)
        else:
            print(f"    {i + 1}. {ch}")
            valid_indices.append(i)

    default_display = ""
    if default:
        for i, ch in enumerate(choices):
            val = ch.value if isinstance(ch, Choice) else ch
            if val == default:
                default_display = str(i + 1)
                break

    while True:
        raw = input(f"  Enter number [{default_display}]: ").strip()
        if not raw and default_display:
            idx = int(default_display) - 1
            ch = choices[idx]
            return ch.value if isinstance(ch, Choice) else ch
        try:
            idx = int(raw) - 1
            if idx in valid_indices:
                ch = choices[idx]
                return ch.value if isinstance(ch, Choice) else ch
        except (ValueError, IndexError):
            pass
        print("  Invalid choice, try again.")


def _select_many(message: str, choices: list, defaults: list = None) -> list:
    """Multi select -- InquirerPy checkbox or comma-separated input()."""
    if defaults is None:
        defaults = []

    if HAS_INQUIRER:
        # Mark defaults as enabled
        iq_choices = []
        for ch in choices:
            if isinstance(ch, Choice):
                val = ch.value
                ch.enabled = val in defaults
                iq_choices.append(ch)
            else:
                iq_choices.append(Choice(value=ch, name=ch, enabled=(ch in defaults)))

        result = inquirer.checkbox(
            message=message,
            choices=iq_choices,
        ).execute()
        return result

    # Fallback: numbered list with comma-separated selection
    print(f"\n  {message}")
    for i, ch in enumerate(choices):
        name = ch.name if isinstance(ch, Choice) else ch
        val = ch.value if isinstance(ch, Choice) else ch
        marker = "*" if val in defaults else " "
        print(f"    {i + 1}. [{marker}] {name}")

    default_indices = []
    for d in defaults:
        for i, ch in enumerate(choices):
            val = ch.value if isinstance(ch, Choice) else ch
            if val == d:
                default_indices.append(str(i + 1))
                break
    default_display = ",".join(default_indices)

    raw = input(f"  Enter numbers (comma-separated) [{default_display}]: ").strip()
    if not raw:
        return list(defaults)
    selected = []
    for part in raw.split(","):
        part = part.strip()
        try:
            idx = int(part) - 1
            if 0 <= idx < len(choices):
                ch = choices[idx]
                selected.append(ch.value if isinstance(ch, Choice) else ch)
        except (ValueError, IndexError):
            pass
    return selected


def _text_input(message: str, default: str = "") -> str:
    """Text input with default."""
    if HAS_INQUIRER:
        return inquirer.text(message=message, default=default).execute()
    suffix = f" [{default}]" if default else ""
    return input(f"  {message}{suffix}: ").strip() or default


def _int_input(message: str, default: int = 0) -> int:
    """Integer input with default."""
    if HAS_INQUIRER:
        raw = inquirer.text(message=message, default=str(default)).execute()
        try:
            return int(raw)
        except ValueError:
            return default
    raw = input(f"  {message} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _confirm(message: str, default: bool = True) -> bool:
    """Yes/no confirmation."""
    if HAS_INQUIRER:
        return inquirer.confirm(message=message, default=default).execute()
    suffix = " [Y/n]" if default else " [y/N]"
    raw = input(f"  {message}{suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


# ---------------------------------------------------------------------------
# Discovery display
# ---------------------------------------------------------------------------

def _display_discovery_summary(providers: list) -> None:
    """Print a summary of discovered providers."""
    if HAS_RICH:
        _console.print(Panel("[bold]Provider Discovery[/bold]", border_style="cyan"))
    else:
        print("\n=== Provider Discovery ===")

    for p in providers:
        status_icon = "OK" if p.reachable else "--"
        line = f"  [{status_icon}] {p.name}: {p.message}"
        if HAS_RICH:
            color = "green" if p.reachable else "red"
            _console.print(f"  [{color}]{status_icon}[/{color}] {p.name}: {p.message}")
        else:
            print(line)


# ---------------------------------------------------------------------------
# Provider and model selection helpers
# ---------------------------------------------------------------------------

def _provider_by_name(providers: list, name: str):
    """Find a ProviderStatus by name, or None."""
    for p in providers:
        if p.name == name:
            return p
    return None


def select_provider(role_label: str, providers: list, default_provider: str = "") -> str:
    """Select provider for a role. Unreachable shown dimmed/disabled."""
    if HAS_INQUIRER:
        iq_choices = []
        for p in providers:
            suffix = "" if p.reachable else " (unreachable)"
            iq_choices.append(Choice(
                value=p.name,
                name=f"{p.name}{suffix}",
                enabled=p.reachable if not p.reachable else True,
            ))
            # InquirerPy uses 'enabled' to disable checkbox items;
            # for select (list) we need the 'disabled' key instead
        # Rebuild with disabled for list prompt
        iq_choices = []
        for p in providers:
            if p.reachable:
                iq_choices.append(Choice(value=p.name, name=p.name))
            else:
                iq_choices.append(Choice(value=p.name, name=f"{p.name} (unreachable)", enabled=False))

        result = inquirer.select(
            message=f"{role_label} provider",
            choices=iq_choices,
            default=default_provider if default_provider else None,
        ).execute()
        return result

    # Fallback
    print(f"\n  {role_label} provider")
    valid_indices = []
    for i, p in enumerate(providers):
        if p.reachable:
            print(f"    {i + 1}. {p.name}")
            valid_indices.append(i)
        else:
            print(f"    {i + 1}. {p.name} (unreachable)")

    default_display = ""
    if default_provider:
        for i, p in enumerate(providers):
            if p.name == default_provider:
                default_display = str(i + 1)
                break

    while True:
        raw = input(f"  Enter number [{default_display}]: ").strip()
        if not raw and default_display:
            idx = int(default_display) - 1
            return providers[idx].name
        try:
            idx = int(raw) - 1
            if idx in valid_indices:
                return providers[idx].name
        except (ValueError, IndexError):
            pass
        print("  Invalid choice (provider unreachable or out of range).")


CLAUDE_CODE_TIERS = [
    ("haiku  (claude-haiku-4-5-20251001)", "claude-haiku-4-5-20251001"),
    ("sonnet (claude-sonnet-4-6)", "claude-sonnet-4-6"),
    ("opus   (claude-opus-4-6)", "claude-opus-4-6"),
]


def _select_claude_code_tier(role_label: str, default_model: str = "") -> str:
    """Pick a Claude Code model tier (haiku/sonnet/opus)."""
    tier_names = [t[0] for t in CLAUDE_CODE_TIERS]
    tier_values = [t[1] for t in CLAUDE_CODE_TIERS]

    if not default_model or default_model == "claude-code":
        default_model = "claude-haiku-4-5-20251001"

    if HAS_INQUIRER:
        iq_choices = [Choice(value=v, name=n) for n, v in CLAUDE_CODE_TIERS]
        result = inquirer.select(
            message=f"{role_label} model (claude-code)",
            choices=iq_choices,
            default=default_model if default_model in tier_values else None,
        ).execute()
        return result

    # Fallback
    print(f"\n  {role_label} model (claude-code)")
    for i, name in enumerate(tier_names):
        print(f"    {i + 1}. {name}")

    default_display = ""
    if default_model in tier_values:
        default_display = str(tier_values.index(default_model) + 1)

    while True:
        raw = input(f"  Enter number [{default_display}]: ").strip()
        if not raw and default_display:
            return tier_values[int(default_display) - 1]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(tier_values):
                return tier_values[idx]
        except (ValueError, IndexError):
            pass
        print("  Invalid choice, try again.")


def select_model(role_label: str, provider_status, default_model: str = "") -> str:
    """Select one model from a provider's model list + custom option."""
    # claude-code: show tier picker (no custom option — tiers are exhaustive)
    if provider_status and provider_status.name == "claude-code":
        return _select_claude_code_tier(role_label, default_model)

    CUSTOM_SENTINEL = "__custom__"
    models = list(provider_status.models) if provider_status.models else []
    choices = models + ["[ enter custom model name ]"]
    choice_values = models + [CUSTOM_SENTINEL]

    if HAS_INQUIRER:
        iq_choices = []
        for m, v in zip(choices, choice_values):
            iq_choices.append(Choice(value=v, name=m))

        result = inquirer.select(
            message=f"{role_label} model ({provider_status.name})",
            choices=iq_choices,
            default=default_model if default_model in models else None,
        ).execute()

        if result == CUSTOM_SENTINEL:
            return _text_input(f"Custom model name for {role_label}")
        return result

    # Fallback
    print(f"\n  {role_label} model ({provider_status.name})")
    for i, name in enumerate(choices):
        print(f"    {i + 1}. {name}")

    default_display = ""
    if default_model and default_model in models:
        default_display = str(models.index(default_model) + 1)

    while True:
        raw = input(f"  Enter number [{default_display}]: ").strip()
        if not raw and default_display:
            return models[int(default_display) - 1]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choice_values):
                val = choice_values[idx]
                if val == CUSTOM_SENTINEL:
                    return _text_input(f"Custom model name for {role_label}")
                return val
        except (ValueError, IndexError):
            pass
        print("  Invalid choice, try again.")


def select_eval_models(providers: list) -> List[Dict[str, str]]:
    """Multi-select eval models from all reachable providers."""
    choices = []
    choice_data = []  # parallel list of dicts

    for p in providers:
        if not p.reachable or not p.models:
            continue
        for m in p.models:
            label = f"{p.name}/{m}"
            entry = {"provider": p.name, "model": m}
            # Store canonical litellm_model for ollama
            if p.name == "ollama":
                entry["litellm_model"] = f"openai/{m}"
            elif p.name == "lmproxy":
                entry["litellm_model"] = m
            choices.append(label)
            choice_data.append(entry)

    if not choices:
        _print("  No models discovered. Enter models manually.")
        raw = _text_input("Eval models (comma-separated provider/model)")
        result = []
        for part in raw.split(","):
            part = part.strip()
            if "/" in part:
                prov, model = part.split("/", 1)
                result.append({"provider": prov, "model": model})
            elif part:
                result.append({"provider": "lmproxy", "model": part})
        return result

    if HAS_INQUIRER:
        iq_choices = []
        for label, data in zip(choices, choice_data):
            iq_choices.append(Choice(value=label, name=label))

        selected_labels = inquirer.checkbox(
            message="Eval models (space to toggle, enter to confirm)",
            choices=iq_choices,
        ).execute()

        result = []
        for label in selected_labels:
            idx = choices.index(label)
            result.append(choice_data[idx])
        return result

    # Fallback
    print("\n  Eval models (select from discovered models)")
    for i, label in enumerate(choices):
        print(f"    {i + 1}. {label}")

    raw = input("  Enter numbers (comma-separated): ").strip()
    if not raw:
        return []
    result = []
    for part in raw.split(","):
        part = part.strip()
        try:
            idx = int(part) - 1
            if 0 <= idx < len(choice_data):
                result.append(choice_data[idx])
        except (ValueError, IndexError):
            pass
    return result


# ---------------------------------------------------------------------------
# Re-probe helper
# ---------------------------------------------------------------------------

def _reprobe_if_changed(providers: list, profile: PipelineProfile) -> list:
    """Re-run discovery if the user changed any provider URL."""
    from c1_tools.provider_discovery import discover_providers
    from pathlib import Path

    url_checks = [
        ("lmproxy", "lmproxy_base_url"),
        ("ollama", "ollama_url"),
        ("iosys", "iosys_base_url"),
        ("lm-studio", "lm_studio_url"),
    ]
    changed = False
    for provider_name, profile_field in url_checks:
        p = _provider_by_name(providers, provider_name)
        old_url = p.base_url if p else ""
        new_url = getattr(profile, profile_field, "")
        if new_url != old_url:
            changed = True
            break

    if not changed:
        return providers

    _print("  URLs changed, re-probing providers...")
    repo_root = Path(__file__).resolve().parent.parent.parent
    cfg_path = repo_root / profile.config_file
    new_providers = discover_providers(
        lmproxy_url=profile.lmproxy_base_url,
        ollama_url=profile.ollama_url,
        iosys_url=profile.iosys_base_url,
        lm_studio_url=profile.lm_studio_url,
        config_file=str(cfg_path) if cfg_path.exists() else "",
    )
    _display_discovery_summary(new_providers)
    return new_providers


# ---------------------------------------------------------------------------
# Full interactive profile builder
# ---------------------------------------------------------------------------

def build_profile_interactive(provider_statuses: list) -> PipelineProfile:
    """Walk the user through building a complete pipeline profile."""
    if HAS_RICH:
        _console.print(Panel("[bold]Pipeline Configuration[/bold]", border_style="cyan"))
    else:
        print("\n=== Pipeline Configuration ===")

    # Step 0: discovery summary
    _display_discovery_summary(provider_statuses)

    profile = PipelineProfile()

    # Determine defaults: lmproxy if reachable, else first reachable provider
    lmproxy_p = _provider_by_name(provider_statuses, "lmproxy")
    default_provider_name = ""
    if lmproxy_p and lmproxy_p.reachable:
        default_provider_name = "lmproxy"
    else:
        for p in provider_statuses:
            if p.reachable:
                default_provider_name = p.name
                break

    HINT = "(ENTER to confirm default)"

    # Step 1: URLs
    _section(1, f"Provider URLs  {HINT}")
    profile.lmproxy_base_url = _text_input("lmproxy URL", profile.lmproxy_base_url)
    profile.ollama_url = _text_input("Ollama URL", profile.ollama_url)
    profile.iosys_base_url = _text_input("iosys URL", profile.iosys_base_url)
    profile.lm_studio_url = _text_input("LM Studio URL", profile.lm_studio_url)

    # Re-probe if URLs changed
    provider_statuses = _reprobe_if_changed(provider_statuses, profile)

    # Step 2: Extraction provider + model
    _section(2, f"Extraction Model  {HINT}")
    profile.extraction_provider = select_provider("Extraction", provider_statuses, default_provider_name)
    ext_p = _provider_by_name(provider_statuses, profile.extraction_provider)
    ext_default_model = ext_p.models[0] if (ext_p and ext_p.models) else ""
    profile.extraction_model = select_model("Extraction", ext_p, ext_default_model)

    # Step 3: Trace provider + model (default: same as extraction)
    _section(3, f"Trace Capture Model  {HINT}")
    profile.trace_provider = select_provider("Trace", provider_statuses, profile.extraction_provider)
    trace_p = _provider_by_name(provider_statuses, profile.trace_provider)
    trace_default = profile.extraction_model if profile.trace_provider == profile.extraction_provider else ""
    if not trace_default and trace_p and trace_p.models:
        trace_default = trace_p.models[0]
    profile.trace_model = select_model("Trace", trace_p, trace_default)

    # Step 4: Judge provider + model (default: same as trace)
    _section(4, f"Judge Model  {HINT}")
    profile.judge_provider = select_provider("Judge", provider_statuses, profile.trace_provider)
    judge_p = _provider_by_name(provider_statuses, profile.judge_provider)
    judge_default = profile.trace_model if profile.judge_provider == profile.trace_provider else ""
    if not judge_default and judge_p and judge_p.models:
        judge_default = judge_p.models[0]
    profile.judge_model = select_model("Judge", judge_p, judge_default)

    # Step 5: Eval models (multi-select)
    _section(5, "Evaluation Models  (SPACE to toggle, ENTER to confirm)")
    profile.eval_models = select_eval_models(provider_statuses)

    # Step 6: Source data
    _section(6, f"Source Data  {HINT}")
    profile.dataset = _text_input("Dataset (HuggingFace identifier)", profile.dataset)
    profile.subset = _text_input("Subset", profile.subset)
    profile.domain = _text_input("Domain label", profile.domain)
    profile.max_chunks = _int_input("Max chunks", profile.max_chunks)
    profile.chunk_size = _int_input("Chunk size (chars)", profile.chunk_size)
    profile.tasks_per_chunk = _int_input("Tasks per chunk", profile.tasks_per_chunk)

    # Step 7: Skills config
    _section(7, f"Skills Configuration  {HINT}")
    profile.max_skills = _int_input("Max skills to extract", profile.max_skills)

    compose_k_raw = _text_input("Compose k values (comma-separated ints)", ",".join(str(k) for k in profile.compose_k_values))
    parsed_k = []
    for part in compose_k_raw.split(","):
        part = part.strip()
        if part:
            try:
                parsed_k.append(int(part))
            except ValueError:
                pass
    profile.compose_k_values = parsed_k if parsed_k else [2, 3]

    compose_ops_raw = _text_input("Compose operators (comma-separated)", ",".join(profile.compose_operators))
    profile.compose_operators = [s.strip() for s in compose_ops_raw.split(",") if s.strip()]

    # Step 8: Modes
    _section(8, "Evaluation Modes  (SPACE to toggle, ENTER to confirm)")
    all_modes = ["singlecall", "stepwise", "guided"]
    profile.modes = _select_many("Select evaluation modes", all_modes, profile.modes)

    # Step 9: Profile name
    _section(9, f"Save  {HINT}")
    profile.profile_name = _text_input("Profile name", profile.profile_name)

    return profile
