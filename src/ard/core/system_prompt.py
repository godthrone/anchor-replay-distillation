"""System-prompt sampling dimension and prompt-text generation.

The system prompt is a **sampling dimension**, not a global switch: real
conversation data is mixed (many turns carry no system prompt at all), so the
generator has to cover both "no system prompt" and several styles of one.

Two ontology dimensions, deliberately kept apart (they are orthogonal):

* ``system_prompt_presence`` — ``none`` / ``present``: is there a system
  message at all?
* ``system_prompt_style`` — how it is written when there is one.

``anchor_meta["system_prompt_mode"]`` is their merge, and is the single value
downstream routes and counts on: :data:`SYSTEM_PROMPT_NONE` when the anchor has
no system message, otherwise the style name.

This module owns that contract (vocabulary + prompt text) so the anchor
generator and the domain layer read the same definition (§1.4).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SYSTEM_PROMPT_GENERATION_INSTRUCTIONS",
    "SYSTEM_PROMPT_NONE",
    "build_system_prompt_prompt",
    "get_system_prompt_values",
    "system_prompt_mode",
]

#: The presence value that means "this anchor has no system message".
SYSTEM_PROMPT_NONE = "none"

#: The style value paired with :data:`SYSTEM_PROMPT_NONE`.  A style describes
#: *how* a system prompt is written, so it is meaningless without one — pairing
#: the absent case with the vocabulary's own absence marker keeps the two
#: dimensions from admitting nonsensical combinations such as "no system
#: prompt, but a detailed persona".
SYSTEM_PROMPT_STYLE_ABSENT = SYSTEM_PROMPT_NONE

#: How the input generator is asked to write each style.  The mode gives the
#: **style specification**; the concrete text is generated at run time from
#: this instruction plus the anchor's own ``knowledge_domain`` /
#: ``capability``, so the system prompt echoes the conversation it belongs to
#: instead of being a fixed sentence reused everywhere.
SYSTEM_PROMPT_GENERATION_INSTRUCTIONS: dict[str, str] = {
    "minimal_persona": (
        "Write ONE short sentence (a single clause) that only gives the "
        "assistant a role identity, and nothing else. Do not mention output "
        "format, length or domain expertise."
    ),
    "detailed_persona": (
        "Write a detailed persona: the assistant's role, its background or "
        "experience, and how it behaves. Use two to four sentences. Do not "
        "impose output-format or length constraints."
    ),
    "task_constraint": (
        "Write a task-constraint system prompt: state output-format, "
        "language, length and must-do / must-not-do rules. Use two to four "
        "sentences and keep the wording imperative."
    ),
    "domain_style": (
        "Write a domain-style system prompt that fits the assistant's "
        "professional field and working style for this specific task. Use two "
        "to four sentences. Do not repeat domain names mechanically."
    ),
}


def get_system_prompt_values(ontology: dict[str, Any]) -> list[tuple[str, str]]:
    """Return the ``(presence, style)`` pairs the sampler may choose from.

    Args:
        ontology: Loaded ontology dict.

    Returns:
        List of ``(presence, style)`` pairs — one pair per distinct
        system-prompt mode.  An ontology without the new sections (older
        ontologies, synthetic test fixtures) degrades to a single
        ``("none", "none")`` pair, i.e. "no system prompt", rather than
        inventing a dimension the ontology does not declare.
    """
    presence = ontology.get("system_prompt_presence")
    styles = ontology.get("system_prompt_style")
    if not presence or not styles:
        return [(SYSTEM_PROMPT_NONE, SYSTEM_PROMPT_STYLE_ABSENT)]

    pairs: list[tuple[str, str]] = []
    for presence_value in presence:
        if presence_value == SYSTEM_PROMPT_NONE:
            pairs.append((SYSTEM_PROMPT_NONE, SYSTEM_PROMPT_STYLE_ABSENT))
            continue
        for style_value in styles:
            pairs.append((presence_value, style_value))
    return pairs


def system_prompt_mode(presence: str, style: str) -> str:
    """Merge the two system-prompt sampling dimensions into one routing label.

    Args:
        presence: One ``system_prompt_presence`` value.
        style: One ``system_prompt_style`` value.

    Returns:
        :data:`SYSTEM_PROMPT_NONE` when there is no system prompt, otherwise
        the style name.
    """
    if presence == SYSTEM_PROMPT_NONE:
        return SYSTEM_PROMPT_NONE
    return style


def build_system_prompt_prompt(anchor_meta: dict[str, Any], mode: str) -> str:
    """Build the input-generator prompt that writes one system prompt.

    The generated text has to fit the conversation it will be attached to, so
    the instruction is combined with the anchor's own ``knowledge_domain`` and
    ``capability``: a system prompt that contradicts the dialogue it precedes
    produces a low-quality anchor (the failure mode scheme §5.4 warns about).

    Args:
        anchor_meta: Anchor metadata (``language`` / ``knowledge_domain`` /
            ``capability``).
        mode: A system-prompt style (any value except
            :data:`SYSTEM_PROMPT_NONE`).

    Returns:
        Prompt text for the input generator.

    Raises:
        KeyError: If *mode* is not a known style.
    """
    style_instruction = SYSTEM_PROMPT_GENERATION_INSTRUCTIONS[mode]
    language = anchor_meta.get("language", "English")
    domain = anchor_meta.get("knowledge_domain", "general")
    capability = anchor_meta.get("capability", "qa")
    return (
        "You are writing a system prompt for an assistant, not a message to "
        "the user. "
        f"{style_instruction} "
        f"Write it in {language}. "
        f"The assistant will handle a '{capability}' task in the field of "
        f"'{domain}', so the system prompt must fit that field and task — a "
        f"generic persona that could belong to any conversation is not "
        f"acceptable. "
        "Output the system prompt text only, with no quotes, no label and no "
        "explanation."
    )
