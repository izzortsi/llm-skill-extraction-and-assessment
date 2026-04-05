"""evaluation -- skill evaluation components.

LLM-as-judge, proof verification, skill injection, effectiveness metrics,
trial results, and experiment configuration.

Config/data classes re-exported from config for backwards compatibility.
"""

__all__: list[str] = []

try:
    from config.experiment_config import (
        ConditionType,
        ModelSpec,
        ExperimentConfig,
        TrialSpec,
    )
    __all__ += ["ConditionType", "ModelSpec", "ExperimentConfig", "TrialSpec"]
except ImportError:
    pass

try:
    from config.trial_result import TrialResult, BenchmarkRecord, load_progress_records
    __all__ += ["TrialResult", "BenchmarkRecord", "load_progress_records"]
except ImportError:
    pass

try:
    from evaluation.llm_judge import LLMJudgeEvaluator, JudgeResult
    __all__ += ["LLMJudgeEvaluator", "JudgeResult"]
except ImportError:
    pass

try:
    from evaluation.proof_verifier import ProofVerifier
    __all__ += ["ProofVerifier"]
except ImportError:
    pass

try:
    from config.skill_injection import (
        format_skill_for_system_prompt,
        format_self_generation_prompt,
        format_extracted_skill_for_system_prompt,
        get_default_system_prompt,
    )
    __all__ += [
        "format_skill_for_system_prompt",
        "format_self_generation_prompt",
        "format_extracted_skill_for_system_prompt",
        "get_default_system_prompt",
    ]
except ImportError:
    pass

try:
    from evaluation.effectiveness import (
        compute_pass_rate_delta,
        aggregate_by_skill,
        aggregate_by_model,
    )
    __all__ += ["compute_pass_rate_delta", "aggregate_by_skill", "aggregate_by_model"]
except ImportError:
    pass
