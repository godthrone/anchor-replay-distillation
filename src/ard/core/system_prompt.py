"""System-prompt sampling dimension: wording contract and prompt rendering.

The system prompt is a **sampling dimension**, not a global switch: real
conversation data is mixed (many turns carry no system prompt at all), so the
generator has to cover both "no system prompt" and several styles of one.

The vocabulary lives in the v4 ontology, on a single axis
(``system_prompt_mode``) whose five values already merge presence and style:
``none`` is the absence case of the same coordinate, the other four name how a
system prompt is written.  That axis is the single source of truth for *which
modes exist* (§1.4).

The **wording** that turns a present mode into an instruction for the input
generator is not code: it lives as data, one file per mode, in the directory the
ontology declares for it (``wording_policy`` — see
:data:`SYSTEM_PROMPT_TEMPLATE_DIR`).  This module owns only the **pure** half of
that contract: where the files *are* (the path), what a template may contain, and
how a rendered prompt is assembled from already-read text.  Reading a file is
facility work (§1.3) and lives in
:mod:`ard.backends.prompt_loader`, which validates what it read with
:func:`validate_system_prompt_template` and renders through
:func:`render_system_prompt_prompt`.

A missing, empty or malformed template is a hard error naming the path; there is
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
    "render_system_prompt_prompt",
    "require_present_mode",
    "system_prompt_template_path",
    "validate_system_prompt_template",
]

#: The ``system_prompt_mode`` value that means "this anchor has no system
#: message".  It is the absence case of the ontology axis, not a separate
#: presence dimension (v4 merged the two).
SYSTEM_PROMPT_NONE = "none"

#: The wording directory, stated **once** in the runtime.  It is the location
#: the v4 ontology declares in
#: ``wording_policy.prompt_wording_location_recommendation.target``
#: (in ``ontology/anchor_ontology.v4.json``), whose
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

    Pure path arithmetic: the path is returned even when the file does not
    exist, so an error message can name what was expected and nothing is opened.

    Args:
        mode: A ``system_prompt_mode`` value (any value except ``none``).
        directory: Wording directory override; defaults to
            :data:`SYSTEM_PROMPT_TEMPLATE_DIR`.

    Returns:
        ``<directory>/<mode>.md``.

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


def require_present_mode(mode: str) -> None:
    """Reject the absence case before any wording is looked up.

    ``none`` is a coordinate value, not a style, so it has no generator-facing
    wording.  Stating the rejection once means the facility loader and the pure
    renderer cannot drift into two different messages for the same defect (§1.4).

    Raises:
        SystemPromptTemplateError: If *mode* is :data:`SYSTEM_PROMPT_NONE`.
    """
    if mode == SYSTEM_PROMPT_NONE:
        raise SystemPromptTemplateError(
            f"system_prompt_mode {SYSTEM_PROMPT_NONE!r} is the absence case: such "
            f"an anchor carries no system message, so no wording is written for it"
        )


def validate_system_prompt_template(body: str, path: str | Path) -> str:
    """Validate already-read wording text and return it stripped.

    Pure: the caller has read *body*; *path* is used for error messages only.

    Args:
        body: The raw text of a ``<mode>.md`` wording file.
        path: The file it came from, named in every error message.

    Returns:
        *body* with surrounding whitespace removed.

    Raises:
        SystemPromptTemplateError: If the wording is empty, uses an unbalanced
            brace, or uses a placeholder outside
            :data:`SYSTEM_PROMPT_TEMPLATE_FIELDS`.
    """
    where = Path(path)
    stripped = body.strip()
    if not stripped:
        raise SystemPromptTemplateError(
            f"system-prompt wording file is empty: {where} "
            f"(expected the wording the input generator is asked to follow)"
        )
    _check_placeholders(stripped, where)
    return stripped


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


def render_system_prompt_prompt(
    anchor_meta: dict[str, Any],
    mode: str,
    template: str,
) -> str:
    """Fill a validated wording template from the anchor's own metadata.

    The text has to fit the conversation it will be attached to, so the
    template's ``{language}`` / ``{capability}`` / ``{domain}`` placeholders are
    filled from the anchor's own metadata — a system prompt that contradicts the
    dialogue it precedes produces a low-quality anchor (the failure mode scheme
    §5.4 warns about).

    Args:
        anchor_meta: Anchor metadata (``language`` / ``knowledge_domain`` /
            ``capability``).
        mode: A system-prompt style (any value except
            :data:`SYSTEM_PROMPT_NONE`).
        template: Already-read, already-validated wording text, as returned by
            :func:`validate_system_prompt_template`.

    Returns:
        Prompt text for the input generator.

    Raises:
        SystemPromptTemplateError: If *mode* is the absence case.
    """
    require_present_mode(mode)
    values = {
        placeholder: anchor_meta.get(key, default)
        for placeholder, (key, default) in _FIELD_SOURCES.items()
    }
    return template.format(**values)
