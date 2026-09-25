"""System-prompt sampling dimension and prompt-text generation.

The system prompt is a **sampling dimension**, not a global switch: real
conversation data is mixed (many turns carry no system prompt at all), so the
generator has to cover both "no system prompt" and several styles of one.

The vocabulary lives in the v4 ontology, on a single axis
(``system_prompt_mode``) whose five values already merge presence and style:
``none`` is the absence case of the same coordinate, the other four name how a
system prompt is written.  That axis is the single source of truth for *which
modes exist* (§1.4).

The **wording** that turns a present mode into an instruction for the input
generator is not code: it lives as data, one file per mode, in the directory
the ontology declares for it (``wording_policy`` — see
:data:`SYSTEM_PROMPT_TEMPLATE_DIR`).  This module owns only the contract: where
the files are, how a template is read, and how failures are reported.  A
missing, empty or malformed file is a hard error naming the path; there is
deliberately no built-in fallback string, which would restore a second source of
truth (§1.4, §2.3).

``anchor_meta["system_prompt_mode"]`` is the chosen coordinate value, and is
the single value downstream routes and counts on: :data:`SYSTEM_PROMPT_NONE`
when the anchor has no system message, otherwise the style name.
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Any, TypeAlias

__all__ = [
    "SYSTEM_PROMPT_NONE",
    "SYSTEM_PROMPT_TEMPLATE_DIR",
    "SYSTEM_PROMPT_TEMPLATE_FIELDS",
    "SYSTEM_PROMPT_TEMPLATE_SUFFIX",
    "SystemPromptTemplateError",
    "build_system_prompt_prompt",
    "system_prompt_template_path",
]

#: The ``system_prompt_mode`` value that means "this anchor has no system
#: message".  It is the absence case of the ontology axis, not a separate
#: presence dimension (v4 merged the two).
SYSTEM_PROMPT_NONE = "none"

#: The wording directory, stated **once** in the runtime.  It is the location
#: the v4 ontology declares in
#: ``wording_policy.prompt_wording_location_recommendation.target``
#: (``ontology/anchor_ontology.v4.json:1319-1323``), whose
#: ``<system_prompt_mode>`` placeholder is this directory's per-mode file name.
#: A contract test holds the two together, so an ontology change cannot silently
#: point somewhere else (§1.4).
SYSTEM_PROMPT_TEMPLATE_DIR = Path("configs/prompts/system_prompt")

#: Wording files are plain Markdown: ``<mode>.md``.
SYSTEM_PROMPT_TEMPLATE_SUFFIX = ".md"

#: The only placeholders a wording file may use.  They are filled from the
#: anchor's own metadata (``language`` / ``capability`` / ``knowledge_domain``),
#: so the system prompt echoes the conversation it belongs to instead of being a
#: fixed sentence reused everywhere.  Any other placeholder is a load error.
SYSTEM_PROMPT_TEMPLATE_FIELDS: tuple[str, ...] = (
    "language",
    "capability",
    "domain",
)

#: One placeholder's source: the metadata key it reads and its default.
FieldSource: TypeAlias = tuple[str, str]

#: Metadata key and default for each placeholder.
_FIELD_SOURCES: dict[str, FieldSource] = {
    "language": ("language", "English"),
    "capability": ("capability", "qa"),
    "domain": ("knowledge_domain", "general"),
}

#: A mode names a file, never a path: rejecting separators, dots and empty
#: values keeps a bad ontology or metadata value from reading — or reporting —
#: a file outside the wording directory (§2.3 边界校验).
_MODE_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


class SystemPromptTemplateError(RuntimeError):
    """Raised when a mode's wording file is missing, empty or malformed.

    The message always names the path and what was expected.  It never contains
    credential material: a wording path and a mode name are not secret (§15).
    """


def system_prompt_template_path(mode: str, directory: str | Path | None = None) -> Path:
    """Return the wording file a ``system_prompt_mode`` value must have.

    Args:
        mode: A ``system_prompt_mode`` value (any value except ``none``).
        directory: Wording directory override; defaults to
            :data:`SYSTEM_PROMPT_TEMPLATE_DIR`.  Tests point it at a scratch
            directory to exercise the failure paths.

    Returns:
        ``<directory>/<mode>.md``.  The path is returned even when the file does
        not exist, so an error message can name what was expected.

    Raises:
        SystemPromptTemplateError: If *mode* is not a bare file-name value.
    """
    if _MODE_PATTERN.fullmatch(mode) is None:
        raise SystemPromptTemplateError(
            f"invalid system_prompt_mode {mode!r}: a mode names a wording file, "
            f"so it must match {_MODE_PATTERN.pattern!r}"
        )
    base = Path(directory) if directory is not None else SYSTEM_PROMPT_TEMPLATE_DIR
    return base / f"{mode}{SYSTEM_PROMPT_TEMPLATE_SUFFIX}"


def _check_placeholders(body: str, path: Path) -> None:
    """Reject placeholder syntax the renderer cannot fill.

    Raises:
        SystemPromptTemplateError: On unbalanced braces, or on a placeholder
            outside :data:`SYSTEM_PROMPT_TEMPLATE_FIELDS`.
    """
    try:
        parsed = list(string.Formatter().parse(body))
    except ValueError as exc:
        raise SystemPromptTemplateError(f"malformed placeholder syntax in {path}: {exc}") from exc
    used = {name for _, name, _, _ in parsed if name is not None}
    unknown = sorted(used - set(SYSTEM_PROMPT_TEMPLATE_FIELDS))
    if unknown:
        raise SystemPromptTemplateError(
            f"unknown placeholder(s) {unknown} in {path}: allowed placeholders "
            f"are {list(SYSTEM_PROMPT_TEMPLATE_FIELDS)}"
        )


def _read_template(mode: str, directory: str | Path | None = None) -> str:
    """Read one mode's wording template.

    Args:
        mode: A ``system_prompt_mode`` value.
        directory: Wording directory override (tests only).

    Returns:
        The file's content with surrounding whitespace removed.

    Raises:
        SystemPromptTemplateError: If the directory is missing, or the file is
            missing, unreadable, empty, or uses a disallowed placeholder.
    """
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
    body = raw.strip()
    if not body:
        raise SystemPromptTemplateError(
            f"system-prompt wording file is empty: {path} "
            f"(expected the wording the input generator is asked to follow)"
        )
    _check_placeholders(body, path)
    return body


def build_system_prompt_prompt(
    anchor_meta: dict[str, Any],
    mode: str,
    directory: str | Path | None = None,
) -> str:
    """Build the input-generator prompt that writes one system prompt.

    The wording is the mode's data file (see the module docstring); the text has
    to fit the conversation it will be attached to, so the template's
    ``{language}`` / ``{capability}`` / ``{domain}`` placeholders are filled from
    the anchor's own metadata — a system prompt that contradicts the dialogue it
    precedes produces a low-quality anchor (the failure mode scheme §5.4 warns
    about).

    Args:
        anchor_meta: Anchor metadata (``language`` / ``knowledge_domain`` /
            ``capability``).
        mode: A system-prompt style (any value except
            :data:`SYSTEM_PROMPT_NONE`).
        directory: Wording directory override (tests only).

    Returns:
        Prompt text for the input generator.

    Raises:
        SystemPromptTemplateError: If *mode* is the absence case, or if its
            wording file is missing, empty or malformed.  There is no fallback
            wording: a silent default would be a second source of truth.
    """
    if mode == SYSTEM_PROMPT_NONE:
        raise SystemPromptTemplateError(
            f"system_prompt_mode {SYSTEM_PROMPT_NONE!r} is the absence case: such "
            f"an anchor carries no system message, so no wording is written for it"
        )
    template = _read_template(mode, directory)
    values = {
        placeholder: anchor_meta.get(key, default)
        for placeholder, (key, default) in _FIELD_SOURCES.items()
    }
    return template.format(**values)
