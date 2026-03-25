"""
stage_output_wirer.py

Build CLI argument lists for each pipeline stage. Maps profile config
and prior stage outputs to the specific --flag values each stage's
CLI expects.

This module is the Python equivalent of the variable assignments and
argument construction in run_full_pipeline.sh.
"""

from pathlib import Path
from typing import Dict, List

from c0_config.pipeline_profile import PipelineProfile


def _provider_args(provider, model, profile):
    """Translate a profile provider value into CLI args for a stage.

    For providers that need a base_url (lmproxy, ollama, iosys, lm-studio), we pass
    --provider openai --model <model> and rely on OPENAI_BASE_URL env var
    (set by provider_env()) rather than --base-url, because not all stages
    accept --base-url as a CLI flag.
    """
    if provider in ("lmproxy", "ollama", "iosys", "lm-studio"):
        return ["--provider", "openai", "--model", model]
    if provider == "claude-code":
        return ["--provider", "claude-code", "--model", model]
    return ["--provider", provider, "--model", model]


def provider_env(provider, profile):
    """Return env vars dict to set for a stage subprocess based on provider.

    Sets OPENAI_BASE_URL so that create_provider("openai", model) inside the
    subprocess connects to the right endpoint.
    """
    if provider == "lmproxy":
        return {"OPENAI_BASE_URL": profile.lmproxy_base_url}
    if provider == "ollama":
        return {"OPENAI_BASE_URL": profile.ollama_url}
    if provider == "iosys":
        env = {"OPENAI_BASE_URL": profile.iosys_base_url}
        import os
        iosys_key = os.environ.get("IOSYS_API_KEY", "")
        if iosys_key:
            env["OPENAI_API_KEY"] = iosys_key
        return env
    if provider == "lm-studio":
        return {"OPENAI_BASE_URL": profile.lm_studio_url}
    return {}


def build_stage_args(
    stage_id: str,
    profile: PipelineProfile,
    run_dir: Path,
    repo_root: Path,
    stage_outputs: Dict[str, Dict[str, str]],
    mode: str = "",
) -> List[str]:
    """Build the CLI argument list for a specific stage.

    Args:
        stage_id: which stage to build args for
        profile: the experiment profile
        run_dir: absolute path to the pipeline run directory
        repo_root: absolute path to the repository root
        stage_outputs: map of stage_id -> {"file_key": "absolute_path"}
        mode: evaluation mode (for stage 5 only)

    Returns:
        list of string arguments to pass after the CLI command
    """
    stage1_dir = run_dir / "stage1-task-extraction"
    stage2_dir = run_dir / "stage2-trace-capture"
    stage3_dir = run_dir / "stage3-skill-extraction"
    stage4_dir = run_dir / "stage4-skill-verification"
    stage5_dir = run_dir / "stage5-corpus-evaluation"
    stage6_dir = run_dir / "stage6-visualization"

    if stage_id == "1a":
        return [
            "--dataset", profile.dataset,
            "--subset", profile.subset,
            "--chunk-size", str(profile.chunk_size),
            "--max-chunks", str(profile.max_chunks),
            "--output", str(stage1_dir / "passages.json"),
            "-v",
        ]

    if stage_id == "1b":
        args = [
            "--passages", stage_outputs.get("1a", {}).get("passages", str(stage1_dir / "passages.json")),
            "--domain", profile.domain,
            "--tasks-per-chunk", str(profile.tasks_per_chunk),
        ]
        args += _provider_args(profile.extraction_provider, profile.extraction_model, profile)
        args += [
            "--output", str(stage1_dir / "tasks.json"),
            "-v",
        ]
        return args

    if stage_id == "2":
        args = [
            "--tasks", stage_outputs.get("1b", {}).get("tasks", str(stage1_dir / "tasks.json")),
            "--output", str(stage2_dir / "traces.jsonl"),
        ]
        args += _provider_args(profile.trace_provider, profile.trace_model, profile)
        args.append("-v")
        return args

    if stage_id == "3":
        args = [
            "--traces", stage_outputs.get("2", {}).get("traces", str(stage2_dir / "traces.jsonl")),
            "--output", str(stage3_dir / "skills.json"),
            "--max-skills", str(profile.max_skills),
        ]
        args += _provider_args(profile.extraction_provider, profile.extraction_model, profile)
        args.append("-v")
        return args

    if stage_id == "4":
        args = [
            "--skills", stage_outputs.get("3", {}).get("skills", str(stage3_dir / "skills.json")),
            "--output", str(stage4_dir / "verified_skills.json"),
            "--revise",
        ]
        args += _provider_args(profile.extraction_provider, profile.extraction_model, profile)
        args += [
            "--standards-dir", str(repo_root / "b0.standards"),
            "-v",
        ]
        return args

    if stage_id == "4b":
        stage4b_dir = run_dir / "stage4b-skill-composition"
        atomic_md_dir = stage4b_dir / "atomic-skills-md"
        args = [
            "--atomic-dir", str(atomic_md_dir),
            "--output-dir", str(stage4b_dir),
            "--k",
        ]
        args += [str(k) for k in profile.compose_k_values]
        args += ["--operators"]
        args += profile.compose_operators
        args.append("-v")
        return args

    if stage_id == "5":
        generated_config = run_dir / "models.yaml"
        config_path = generated_config if generated_config.exists() else repo_root / profile.config_file
        if config_path.exists():
            args = [
                "--tasks", stage_outputs.get("1b", {}).get("tasks", str(stage1_dir / "tasks.json")),
                "--skills", stage_outputs.get("4", {}).get("skills", str(stage4_dir / "verified_skills.json")),
                "--config", str(config_path),
                "--mode", mode,
                "--output", str(stage5_dir / mode / "results-all.json"),
                "-v",
            ]
            # pass --models to restrict evaluation to profile's eval_models subset
            if profile.eval_models:
                args += ["--models", ",".join(entry["model"] for entry in profile.eval_models)]
            return args
        else:
            raise ValueError(
                f"Stage 5 config file not found at '{config_path}'. "
                "Ensure profile.config_file points to a valid models.yaml."
            )

    if stage_id == "6":
        return [
            "--results", str(stage5_dir / "results-all-merged.json"),
            "-o", str(stage6_dir / "cross-mode"),
            "--type", "charts",
            "--dpi", "200",
        ]

    if stage_id == "7":
        # stage 7 has two commands; return args for the first (traceability-report)
        # export-csv args are built separately via build_stage7_csv_args()
        return [
            "--tasks", stage_outputs.get("1b", {}).get("tasks", str(stage1_dir / "tasks.json")),
            "--skills", stage_outputs.get("4", {}).get("skills", str(stage4_dir / "verified_skills.json")),
            "--passages", stage_outputs.get("1a", {}).get("passages", str(stage1_dir / "passages.json")),
            "--output", str(run_dir / "traceability-report.txt"),
            "-v",
        ]

    if stage_id == "8":
        # stage 8 has three commands; return args for the first (run-skillmix)
        # report and visualize args are built separately
        stage4b_dir = run_dir / "stage4b-skill-composition"
        stage8_dir = run_dir / "stage8-skillmix-evaluation"
        generated_config = run_dir / "models.yaml"
        config_path = generated_config if generated_config.exists() else repo_root / profile.config_file
        args = [
            "--tasks", stage_outputs.get("1b", {}).get("tasks", str(stage1_dir / "tasks.json")),
            "--skills-dir", str(stage4b_dir),
            "--models", ",".join(entry["model"] for entry in profile.eval_models),
            "--output-dir", str(stage8_dir),
            "-v",
        ]
        if config_path.exists():
            args += ["--config", str(config_path)]
        else:
            args += _provider_args(profile.judge_provider, profile.judge_model, profile)
        return args

    if stage_id == "9":
        stage8_dir = run_dir / "stage8-skillmix-evaluation"
        stage9_dir = run_dir / "stage9-skillmix-visualization"
        return [
            "--results-dir", str(stage8_dir),
            "--output-dir", str(stage9_dir),
            "--dpi", "200",
        ]

    raise ValueError(f"Unknown stage_id: {stage_id}")


def build_stage7_csv_args(
    profile: PipelineProfile,
    run_dir: Path,
    stage_outputs: Dict[str, Dict[str, str]],
) -> List[str]:
    """Build args for stage 7's second command (export-csv)."""
    stage1_dir = run_dir / "stage1-task-extraction"
    stage4_dir = run_dir / "stage4-skill-verification"

    return [
        "--tasks", stage_outputs.get("1b", {}).get("tasks", str(stage1_dir / "tasks.json")),
        "--skills", stage_outputs.get("4", {}).get("skills", str(stage4_dir / "verified_skills.json")),
        "--passages", stage_outputs.get("1a", {}).get("passages", str(stage1_dir / "passages.json")),
        "--output-dir", str(run_dir / "csv"),
        "-v",
    ]


def build_stage8_report_args(
    run_dir: Path,
) -> List[str]:
    """Build args for stage 8's second command (report)."""
    stage8_dir = run_dir / "stage8-skillmix-evaluation"
    return [
        "--results-dir", str(stage8_dir),
        "--output", str(stage8_dir / "report.txt"),
    ]


def register_stage_outputs(stage_id: str, run_dir: Path) -> Dict[str, str]:
    """Register the output file paths for a completed stage."""
    stage1_dir = run_dir / "stage1-task-extraction"
    stage2_dir = run_dir / "stage2-trace-capture"
    stage3_dir = run_dir / "stage3-skill-extraction"
    stage4_dir = run_dir / "stage4-skill-verification"

    outputs = {
        "1a": {"passages": str(stage1_dir / "passages.json")},
        "1b": {"tasks": str(stage1_dir / "tasks.json")},
        "2": {"traces": str(stage2_dir / "traces.jsonl")},
        "3": {"skills": str(stage3_dir / "skills.json")},
        "4": {"skills": str(stage4_dir / "verified_skills.json")},
    }

    return outputs.get(stage_id, {})
