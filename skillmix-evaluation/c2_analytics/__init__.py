"""c2_analytics -- statistical aggregation for evaluation results."""

__all__: list[str] = []

try:
    from c2_analytics.summary import compute_summary
    __all__ += ["compute_summary"]
except ImportError:
    pass
