"""System-prompt wording loader — the facility half of :mod:`ard.core.system_prompt`.

Responsibility: locate and read one ``<system_prompt_mode>.md`` wording file off
disk, then hand the text to the pure contract in
:mod:`ard.core.system_prompt`.  Reading a file is facility work (§1.3 计算与设施
分离), so it lives here beside the other facilities — ``core/`` stays free of
filesystem, network and subprocess access.

Everything is checked loudly and nothing is swallowed (§2.3):

* a missing wording directory, a missing file and an unreadable file are all
  :class:`~ard.core.system_prompt.SystemPromptTemplateError` naming the path;
* empty wording and a placeholder the renderer cannot fill are rejected by
  :func:`~ard.core.system_prompt.validate_system_prompt_template`;
* the absence case (``none``) is rejected before any lookup.

There is no fallback wording: a silent default would be a second source of truth
(§1.4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ard.core.system_prompt import (
    SYSTEM_PROMPT_TEMPLATE_SUFFIX,
    SystemPromptTemplateError,
    render_system_prompt_prompt,
    require_present_mode,
    system_prompt_template_path,
    validate_system_prompt_template,
)

__all__ = ["build_system_prompt_prompt", "load_system_prompt_template"]


def load_system_prompt_template(mode: str, directory: str | Path | None = None) -> str:
    """Read one mode's wording template from disk.

    Args:
        mode: A ``system_prompt_mode`` value (any value except ``none``).
        directory: Wording directory override; defaults to the directory the
            ontology declares (:data:`~ard.core.system_prompt.SYSTEM_PROMPT_TEMPLATE_DIR`).

    Returns:
        The validated file content, with surrounding whitespace removed.

    Raises:
        SystemPromptTemplateError: If *mode* is the absence case, the directory
            is missing, or the file is missing, unreadable, empty, or uses a
            disallowed placeholder.
    """
    require_present_mode(mode)
    path = system_prompt_template_path(mode, directory)
    if not path.parent.is_dir():
        raise SystemPromptTemplateError(
            f"system-prompt wording directory not found: {path.parent} "
            f"(expected one {SYSTEM_PROMPT_TEMPLATE_SUFFIX} file per "
            f"system_prompt_mode value, including {path.name!r})"
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemPromptTemplateError(
            f"no wording file for system_prompt_mode {mode!r}: expected {path}"
        ) from exc
    except OSError as exc:
        raise SystemPromptTemplateError(
            f"cannot read system-prompt wording file {path}: {exc.strerror}"
        ) from exc
    return validate_system_prompt_template(raw, path)


def build_system_prompt_prompt(
    anchor_meta: dict[str, Any],
    mode: str,
    directory: str | Path | None = None,
) -> str:
    """Build the input-generator prompt that writes one system prompt.

    Reads the mode's wording file (see :mod:`ard.core.system_prompt`) and fills
    its placeholders from the anchor's own metadata.

    Args:
        anchor_meta: Anchor metadata (``language`` / ``knowledge_domain`` /
            ``capability``).
        mode: A system-prompt style (any value except ``none``).
        directory: Wording directory override (tests only).

    Returns:
        Prompt text for the input generator.

    Raises:
        SystemPromptTemplateError: If *mode* is the absence case, or if its
            wording file is missing, empty or malformed.  There is no fallback
            wording: a silent default would be a second source of truth.
    """
    template = load_system_prompt_template(mode, directory)
    return render_system_prompt_prompt(anchor_meta, mode, template)
