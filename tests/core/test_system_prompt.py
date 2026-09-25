"""Tests for the system-prompt sampling dimension (:mod:`ard.core.system_prompt`).

Covers the vocabulary contract — the modes come from the v4 ontology's
``system_prompt_mode`` axis, all five of them — the alignment between that axis
and the generation-instruction table, and the generation prompt handed to the
input generator.
"""

from __future__ import annotations

import pytest

from ard.core.ontology import FlatAxisWithDefinitions, OntologyV4
from ard.core.system_prompt import (
    SYSTEM_PROMPT_GENERATION_INSTRUCTIONS,
    SYSTEM_PROMPT_NONE,
    build_system_prompt_prompt,
    get_system_prompt_values,
)

# ── Vocabulary: the v4 axis is the single source ────────────────────────────


def test_modes_are_the_five_axis_values(ontology: OntologyV4) -> None:
    """The sampler's modes are the ontology axis values — all five of them."""
    values = get_system_prompt_values(ontology)
    assert values == list(ontology.axis_values("system_prompt_mode"))
    assert len(values) == 5
    assert len(set(values)) == 5
    assert values[0] == SYSTEM_PROMPT_NONE
    # The v3 shape fell back to a single ``("none", "none")`` *pair*; the v4
    # modes are plain coordinate values, never tuples.
    assert all(isinstance(value, str) for value in values)


def test_axis_defines_every_mode_the_module_hands_out(ontology: OntologyV4) -> None:
    """Each mode the module offers is documented on the axis it came from."""
    axis = ontology.axes.spec("system_prompt_mode")
    assert isinstance(axis, FlatAxisWithDefinitions)
    for mode in get_system_prompt_values(ontology):
        assert axis.value_definitions.get(mode, "").strip()


def test_generation_instructions_cover_every_present_mode(ontology: OntologyV4) -> None:
    """The instruction table is keyed by exactly the non-``none`` axis values."""
    present = set(get_system_prompt_values(ontology)) - {SYSTEM_PROMPT_NONE}
    assert set(SYSTEM_PROMPT_GENERATION_INSTRUCTIONS) == present
    for instruction in SYSTEM_PROMPT_GENERATION_INSTRUCTIONS.values():
        assert instruction.strip()


# ── Generation prompt ───────────────────────────────────────────────────────


def test_generation_prompt_echoes_domain_and_capability() -> None:
    """The prompt ties the system text to the anchor's own domain/capability."""
    prompt = build_system_prompt_prompt(
        {
            "language": "日本語",
            "knowledge_domain": "software_engineering",
            "capability": "debugging",
        },
        "task_constraint",
    )
    assert "software_engineering" in prompt
    assert "debugging" in prompt
    assert "日本語" in prompt
    assert "task-constraint" in prompt


def test_generation_prompt_rejects_unknown_mode() -> None:
    """An unknown mode cannot silently fall back to a default style."""
    with pytest.raises(KeyError):
        build_system_prompt_prompt({}, "casual_vibes")
