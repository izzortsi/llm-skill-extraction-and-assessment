"""
trial_result.py  (backwards-compatibility shim)

Canonical location: c0_config.trial_result
"""

from c0_config.trial_result import (  # noqa: F401
    TrialResult,
    BenchmarkRecord,
    write_progress_record,
    load_progress_records,
)
