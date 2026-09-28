"""System-prompt wording failure: the one exception its contract raises.

``SystemPromptTemplateError`` is the single failure type of the system-prompt
wording contract.  By §12.2 class-path mirroring its file name is the class
name in snake_case; ``ard.core.system_prompt`` re-exports it, so callers are
unaffected by the split.
"""

__all__ = ["SystemPromptTemplateError"]


class SystemPromptTemplateError(RuntimeError):
    """Raised when a mode's wording file is missing, empty or malformed.

    The message always names the path and what was expected.  It never contains
    credential material: a wording path and a mode name are not secret (§15).
    """
