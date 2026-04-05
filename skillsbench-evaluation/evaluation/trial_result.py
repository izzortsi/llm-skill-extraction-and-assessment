"""
trial_result.py  (backwards-compatibility shim)

Canonical location: config.trial_result
"""

from config.trial_result import (  # noqa: F401
    TrialResult,
    BenchmarkRecord,
    write_progress_record,
    load_progress_records,
)
