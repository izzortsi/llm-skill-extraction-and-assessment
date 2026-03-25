"""
stage_registry.py

Registry of all pipeline stages. The single source of truth for stage
ordering, pipeline directory mapping, CLI commands, output files, and
dependency relationships.
"""

from c0_config.pipeline_stage import PipelineStage


STAGES = [
    PipelineStage(
        stage_id="1a",
        name="extract-passages",
        description="Extract text passages from source documents or datasets",
        pipeline_dir="llm-skills",
        commands=["extract-passages"],
        output_dir="stage1-task-extraction",
        output_files=["passages.json"],
        depends_on=[],
    ),
    PipelineStage(
        stage_id="1b",
        name="extract-tasks",
        description="Generate evaluation tasks from passages using LLM",
        pipeline_dir="llm-skills",
        commands=["extract-tasks"],
        output_dir="stage1-task-extraction",
        output_files=["tasks.json"],
        depends_on=["1a"],
    ),
    PipelineStage(
        stage_id="2",
        name="capture-traces",
        description="Run tasks through LLM to capture reasoning traces",
        pipeline_dir="llm-skills",
        commands=["capture-traces"],
        output_dir="stage2-trace-capture",
        output_files=["traces.jsonl"],
        depends_on=["1b"],
    ),
    PipelineStage(
        stage_id="3",
        name="extract-skills",
        description="Extract procedural skills from reasoning traces",
        pipeline_dir="llm-skills",
        commands=["extract-skills"],
        output_dir="stage3-skill-extraction",
        output_files=["skills.json"],
        depends_on=["2"],
    ),
    PipelineStage(
        stage_id="4",
        name="verify-skills",
        description="Verify extracted skill quality via rule-based checks",
        pipeline_dir="llm-skills",
        commands=["verify-skills"],
        output_dir="stage4-skill-verification",
        output_files=["verified_skills.json"],
        depends_on=["3"],
    ),
    PipelineStage(
        stage_id="5",
        name="corpus-evaluation",
        description="Evaluate skill injection across models and scaffolding modes",
        pipeline_dir="llm-skills.skillsbench-evaluation",
        commands=["run-skillsbench"],
        output_dir="stage5-corpus-evaluation",
        output_files=[],  # dynamic: {mode}/results-all.json per mode
        depends_on=["1b", "4"],
    ),
    PipelineStage(
        stage_id="6",
        name="visualization",
        description="Generate heatmaps and charts from evaluation results",
        pipeline_dir="llm-skills.skillsbench-evaluation",
        commands=["heatmaps"],
        output_dir="stage6-visualization",
        output_files=[],  # dynamic: per-mode directories with PNGs
        depends_on=["5"],
    ),
    PipelineStage(
        stage_id="7",
        name="traceability",
        description="Generate traceability report and CSV exports",
        pipeline_dir="llm-skills",
        commands=["traceability-report", "export-csv"],
        output_dir="",  # outputs to run_dir root and run_dir/csv
        output_files=["traceability-report.txt", "csv/skills.csv"],
        depends_on=["1a", "1b", "4"],
    ),
]


STAGE_MAP = {stage.stage_id: stage for stage in STAGES}

ALL_STAGE_IDS = [stage.stage_id for stage in STAGES]


def get_stage(stage_id: str) -> PipelineStage:
    """Look up a stage by stage_id. Raises KeyError if stage_id not found."""
    if stage_id not in STAGE_MAP:
        valid_ids = ", ".join(ALL_STAGE_IDS)
        raise KeyError(f"Unknown stage_id '{stage_id}'. Valid IDs: {valid_ids}")
    return STAGE_MAP[stage_id]


def _resolve_stage_index(stage_str: str, range_str: str, find_first: bool) -> int:
    """Resolve a stage string to an index in ALL_STAGE_IDS.

    Supports both exact IDs ("1a", "4") and numeric prefixes ("1" -> "1a"/"1b").
    When find_first=True, returns the first matching index (for range start).
    When find_first=False, returns the last matching index (for range end).
    """
    if stage_str in ALL_STAGE_IDS:
        return ALL_STAGE_IDS.index(stage_str)

    # numeric prefix: "1" matches "1a" and "1b"
    if stage_str.isdigit():
        matches = [i for i, sid in enumerate(ALL_STAGE_IDS) if sid.startswith(stage_str)]
        if matches:
            return matches[0] if find_first else matches[-1]

    label = "start" if find_first else "end"
    raise ValueError(f"Unknown {label} stage '{stage_str}' in range '{range_str}'")


def parse_stage_range(range_str: str) -> list:
    """Parse a stage range string into a list of stage IDs.

    Accepted formats:
        "all"          -> ["1a", "1b", "2", "3", "4", "5", "6", "7"]
        "1-4"          -> ["1a", "1b", "2", "3", "4"]
        "5-7"          -> ["5", "6", "7"]
        "1a,1b,5"      -> ["1a", "1b", "5"]
        "3"            -> ["3"]
        "extraction"   -> ["1a", "1b", "2", "3", "4"]
        "evaluation"   -> ["5", "6", "7"]
    """
    range_str = range_str.strip().lower()

    if range_str == "all":
        return list(ALL_STAGE_IDS)

    if range_str == "extraction":
        return ["1a", "1b", "2", "3", "4"]

    if range_str == "evaluation":
        return ["5", "6", "7"]

    # comma-separated list
    if "," in range_str:
        ids = [s.strip() for s in range_str.split(",") if s.strip()]
        for sid in ids:
            if sid not in STAGE_MAP:
                raise ValueError(f"Unknown stage ID '{sid}' in range '{range_str}'")
        return ids

    # range: "1-4", "5-7", etc.
    if "-" in range_str:
        parts = range_str.split("-", 1)
        start_str = parts[0].strip()
        end_str = parts[1].strip()

        start_idx = _resolve_stage_index(start_str, range_str, find_first=True)
        end_idx = _resolve_stage_index(end_str, range_str, find_first=False)

        if start_idx > end_idx:
            raise ValueError(f"Start stage '{start_str}' comes after end stage '{end_str}'")

        return ALL_STAGE_IDS[start_idx:end_idx + 1]

    # single stage
    if range_str in STAGE_MAP:
        return [range_str]

    raise ValueError(f"Cannot parse stage range '{range_str}'. Use: all, 1-4, 5-7, 1a,1b,5, extraction, evaluation")
