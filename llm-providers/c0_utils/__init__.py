"""c0_utils -- pure utility functions with no external dependencies.

Provides UID generation and statistical utilities used across layers.
"""

from c0_utils.uid import generate_uid, format_extraction_method
from c0_utils.stat_utils import (
    bootstrap_ci,
    mean,
    pass_rate,
    pass_rate_delta_pp,
    permutation_test,
)

__all__ = [
    "generate_uid",
    "format_extraction_method",
    "bootstrap_ci",
    "mean",
    "pass_rate",
    "pass_rate_delta_pp",
    "permutation_test",
]
