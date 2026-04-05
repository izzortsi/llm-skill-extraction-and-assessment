"""
stat_utils.py -- backwards-compatibility shim.

Canonical location: utils.stat_utils
"""

from utils.stat_utils import (  # noqa: F401
    mean,
    bootstrap_ci,
    permutation_test,
    pass_rate,
    pass_rate_delta_pp,
)

__all__ = ["mean", "bootstrap_ci", "permutation_test", "pass_rate", "pass_rate_delta_pp"]
