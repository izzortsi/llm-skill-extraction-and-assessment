"""
uid.py -- backwards-compatibility shim.

Canonical location: utils.uid
"""

from utils.uid import generate_uid, format_extraction_method  # noqa: F401

__all__ = ["generate_uid", "format_extraction_method"]
