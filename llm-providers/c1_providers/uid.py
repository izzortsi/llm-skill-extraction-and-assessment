"""
uid.py -- backwards-compatibility shim.

Canonical location: c0_utils.uid
"""

from c0_utils.uid import generate_uid, format_extraction_method  # noqa: F401

__all__ = ["generate_uid", "format_extraction_method"]
