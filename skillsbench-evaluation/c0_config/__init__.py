"""c0_config -- pure data structures, config classes, and string formatting.

No domain imports. Only stdlib dependencies (dataclasses, json, pathlib, enum, typing).
"""

from c0_config.experiment_config import (
    ConditionType,
    ModelSpec,
    SkillSourceConfig,
    TaskSourceConfig,
    SelfGeneratedConfig,
    JudgeConfig,
    ExperimentConfig,
    TrialSpec,
)
from c0_config.trial_result import (
    TrialResult,
    BenchmarkRecord,
    write_progress_record,
    load_progress_records,
)
from c0_config.skill_injection import (
    format_skill_for_system_prompt,
    format_self_generation_prompt,
    format_extracted_skill_for_system_prompt,
    format_skill_for_user_message,
    get_default_system_prompt,
    DEFAULT_READING_COMPREHENSION_PROMPT,
    DEFAULT_CODING_PROMPT,
)

__all__ = [
    # experiment_config
    "ConditionType",
    "ModelSpec",
    "SkillSourceConfig",
    "TaskSourceConfig",
    "SelfGeneratedConfig",
    "JudgeConfig",
    "ExperimentConfig",
    "TrialSpec",
    # trial_result
    "TrialResult",
    "BenchmarkRecord",
    "write_progress_record",
    "load_progress_records",
    # skill_injection
    "format_skill_for_system_prompt",
    "format_self_generation_prompt",
    "format_extracted_skill_for_system_prompt",
    "format_skill_for_user_message",
    "get_default_system_prompt",
    "DEFAULT_READING_COMPREHENSION_PROMPT",
    "DEFAULT_CODING_PROMPT",
]
