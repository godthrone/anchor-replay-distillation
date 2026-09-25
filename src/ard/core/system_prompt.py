"""System-prompt sampling dimension and prompt-text generation.

The system prompt is a **sampling dimension**, not a global switch: real
conversation data is mixed (many turns carry no system prompt at all), so the
generator has to cover both "no system prompt" and several styles of one.

The vocabulary lives in the v4 ontology, on a single axis
(``system_prompt_mode``) whose five values already merge presence and style:
``none`` is the absence case of the same coordinate, the other four name how a
system prompt is written.  That axis is the single source of truth (§1.4) —
this module never restates the value list, it reads it from the ontology and
owns only the *prompt text* that turns a present mode into an instruction for
the input generator.

``anchor_meta["system_prompt_mode"]`` is the chosen coordinate value, and is
the single value downstream routes and counts on: :data:`SYSTEM_PROMPT_NONE`
when the anchor has no system message, otherwise the style name.
"""

from __future__ import annotations

from typing import Any

from ard.core.ontology import OntologyV4

__all__ = [
    "SYSTEM_PROMPT_GENERATION_INSTRUCTIONS",
    "SYSTEM_PROMPT_NONE",
    "build_system_prompt_prompt",
    "get_system_prompt_values",
]

#: The ``system_prompt_mode`` value that means "this anchor has no system
#: message".  It is the absence case of the ontology axis, not a separate
#: presence dimension (v4 merged the two).
SYSTEM_PROMPT_NONE = "none"

#: How the input generator is asked to write each style.  The mode gives the
#: **style specification**; the concrete text is generated at run time from
#: this instruction plus the anchor's own ``knowledge_domain`` /
#: ``capability``, so the system prompt echoes the conversation it belongs to
#: instead of being a fixed sentence reused everywhere.
#:
#: Keyed by exactly the non-:data:`SYSTEM_PROMPT_NONE` values of the ontology's
#: ``system_prompt_mode`` axis.  The contract test holds the two together, so a
#: mode added to the ontology without an instruction fails loudly (§1.4).
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


def get_system_prompt_values(ontology: OntologyV4) -> list[str]:
    """Return the system-prompt modes the sampler may choose from.

    The modes are the values of the v4 ``system_prompt_mode`` axis: the absence
    case plus one value per style (five in the shipped ontology).  Reading them
    from the ontology keeps the vocabulary in one place (§1.4) — this module
    restates no value, so a value added to the axis is usable here without a
    code change.

    Args:
        ontology: The loaded v4 ontology.

    Returns:
        One mode string per axis value, in the axis's declaration order.

    Raises:
        KeyError: If the ontology declares no ``system_prompt_mode`` axis —
            a schema-level defect the v4 loader rejects before this point.
    """
    return list(ontology.axis_values("system_prompt_mode"))


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
