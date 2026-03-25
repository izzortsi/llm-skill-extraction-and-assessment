"""
skill_injection.py  (backwards-compatibility shim)

Canonical location: c0_config.skill_injection
"""

from c0_config.skill_injection import (  # noqa: F401
    format_skill_for_system_prompt,
    format_self_generation_prompt,
    format_extracted_skill_for_system_prompt,
    format_skill_for_user_message,
    get_default_system_prompt,
    DEFAULT_READING_COMPREHENSION_PROMPT,
    DEFAULT_CODING_PROMPT,
)
