"""analytics -- statistical aggregation for evaluation results."""

__all__: list[str] = []

try:
    from analytics.summary import compute_summary
    __all__ += ["compute_summary"]
except ImportError:
    pass
