"""skillsbench -- SkillsBench evaluation harness and visualization.

Corpus harness for running evaluation episodes and heatmap/chart generation.
"""

__all__: list[str] = []

try:
    from skillsbench.corpus_harness import run_corpus_evaluation, run_multi_model_evaluation
    __all__ += ["run_corpus_evaluation", "run_multi_model_evaluation"]
except ImportError:
    pass

try:
    from skillsbench.visualization import generate_uplift_heatmap, generate_pass_rate_heatmap
    __all__ += ["generate_uplift_heatmap", "generate_pass_rate_heatmap"]
except ImportError:
    pass
