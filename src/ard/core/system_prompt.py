"""System-prompt sampling dimension and prompt-text generation (v3.0.0).

The system prompt is a **sampling dimension**, not a global switch: real
conversation data is mixed (many turns carry no system prompt at all), so the
generator has to cover both "no system prompt" and several styles of one, and
FPS has to be able to spread the choice over the ontology like any other
dimension.

Two ontology dimensions, deliberately kept apart (they are orthogonal):

* ``system_prompt_presence`` — ``none`` / ``present``: is there a system
  message at all?
* ``system_prompt_style`` — how it is written when there is one.

``anchor_meta["system_prompt_mode"]`` is their merge, and is the single value
downstream routes and counts on: :data:`SYSTEM_PROMPT_NONE` when the anchor has
no system message, otherwise the style name.

This module owns that contract (vocabulary + prompt text) so the sampler, the
FPS layer and the anchor generator all read the same definition (§1.4).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "SYSTEM_PROMPT_GENERATION_INSTRUCTIONS",
    "SYSTEM_PROMPT_NONE",
    "SYSTEM_PROMPT_STYLE_ORDER",
    "build_system_prompt_prompt",
    "get_system_prompt_values",
    "named_style_vector",
    "style_centroid_direction",
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

#: The system-prompt style names, in the order the ontology declares them.
#: Used to check that the sampling code and the ontology have not drifted.
SYSTEM_PROMPT_STYLE_ORDER: tuple[str, ...] = (
    "minimal_persona",
    "detailed_persona",
    "task_constraint",
    "domain_style",
)

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
        inventing a dimension that has no embeddings.
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


def named_style_vector(style: str, dimension: int, n_styles: int) -> np.ndarray:
    """Return the composition-slot vector of a system-prompt *style*.

    Styles are a **categorical** choice — a "detailed persona" and a "task
    constraint" are not two points in semantic space, they are two different
    kinds of instruction — so the slot uses one axis per style instead of a
    semantic vector.  Orthogonal axes are what make two anchors that differ
    only in style genuinely far apart; the semantic style vectors that the
    embeddings file carries are near-parallel (cosine ≈ 0.96-0.99), which left
    the choice of style almost invisible to FPS (measured: 1/500 anchors picked
    the rarest style).

    The absent case has **no** style, and that is exactly what it contributes:
    :data:`SYSTEM_PROMPT_STYLE_ABSENT` maps to the zero vector.  Its own
    direction lives in the presence slot, which is orthogonal to this one.

    Args:
        style: A ``system_prompt_style`` value (the absent marker is allowed).
        dimension: Width of the composition slot (the embedding dimension).
        n_styles: Number of one-hot axes needed (the real styles, excluding
            the absent marker).

    Returns:
        Vector of shape ``(dimension,)`` — a unit one-hot, or all zeros for the
        absent case.

    Raises:
        ValueError: If *n_styles* does not fit in *dimension*.
    """
    if n_styles > dimension:
        raise ValueError(
            f"cannot one-hot encode {n_styles} system-prompt styles in "
            f"{dimension} dimensions"
        )
    vector = np.zeros(dimension, dtype=np.float64)
    if style == SYSTEM_PROMPT_STYLE_ABSENT:
        return vector
    vector[SYSTEM_PROMPT_STYLE_ORDER.index(style)] = 1.0
    return vector


def style_centroid_direction(style_vectors: list[np.ndarray]) -> np.ndarray:
    """Return the direction that is maximally far from every system-prompt style.

    Used for the ``none`` **presence** value.  "No system prompt" is not a
    style; it is the absence of the whole dimension, so it has no natural
    position among the style vectors.  Giving it the *antipode* of the style
    centroid is the one choice that states that in geometry: the absent case is
    as far from every style as the style space allows, instead of sitting at
    the centroid where it is equidistant from all of them — which is what made
    FPS unable to distinguish "no system prompt" from "some system prompt"
    (measured before this change: ``none`` was picked 1 time in 500).

    Args:
        style_vectors: Normalised style vectors (any number ≥ 1).

    Returns:
        Unit vector pointing away from the style centroid.

    Raises:
        ValueError: If *style_vectors* is empty.
    """
    if not style_vectors:
        raise ValueError("style_centroid_direction needs at least one style vector")
    centroid = np.mean(style_vectors, axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        raise ValueError("style vectors cancel out; no centroid direction exists")
    return -(centroid / norm)


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
