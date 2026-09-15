"""Tests for the system-prompt sampling dimension (:mod:`ard.core.system_prompt`).

Covers the vocabulary contract, the merge of the two ontology dimensions into
the routing label, the composition-slot geometry (one-hot styles, antipodal
absent direction), and the generation prompt handed to the input generator.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ard.core.sampler import _build_all_combinations
from ard.core.system_prompt import (
    SYSTEM_PROMPT_NONE,
    SYSTEM_PROMPT_STYLE_ABSENT,
    SYSTEM_PROMPT_STYLE_ORDER,
    build_system_prompt_prompt,
    get_system_prompt_values,
    named_style_vector,
    style_centroid_direction,
    system_prompt_mode,
)

ONTOLOGY_PATH = "ontology/anchor_ontology.json"


def _real_ontology() -> dict:
    with open(ONTOLOGY_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _minimal_ontology() -> dict:
    return {
        "languages": ["English"],
        "knowledge_domains": {"science": {"sub": ["physics"]}},
        "capabilities": {"knowledge_response": ["qa"]},
        "conversation_types": {"single_turn": ["single_turn"]},
    }


# ── Vocabulary ──────────────────────────────────────────────────────────────


def test_ontology_declares_the_style_vocabulary() -> None:
    """The ontology's style keys are exactly the vocabulary the code encodes."""
    ontology = _real_ontology()
    assert tuple(ontology["system_prompt_style"]) == SYSTEM_PROMPT_STYLE_ORDER


def test_ontology_presence_values_and_absent_marker() -> None:
    """Presence is binary and the absent marker is the vocabulary's ``none``."""
    ontology = _real_ontology()
    assert ontology["system_prompt_presence"] == [SYSTEM_PROMPT_NONE, "present"]
    assert SYSTEM_PROMPT_STYLE_ABSENT == SYSTEM_PROMPT_NONE


def test_generation_instructions_cover_every_style() -> None:
    """Every style the ontology declares has a generation instruction."""
    from ard.core.system_prompt import SYSTEM_PROMPT_GENERATION_INSTRUCTIONS

    assert set(SYSTEM_PROMPT_GENERATION_INSTRUCTIONS) == set(SYSTEM_PROMPT_STYLE_ORDER)
    for instruction in SYSTEM_PROMPT_GENERATION_INSTRUCTIONS.values():
        assert instruction.strip()


# ── (presence, style) pairs ─────────────────────────────────────────────────


def test_pairs_cover_every_mode_exactly_once() -> None:
    """One pair per mode: absent once, one per style for the present case."""
    pairs = get_system_prompt_values(_real_ontology())
    assert pairs[0] == (SYSTEM_PROMPT_NONE, SYSTEM_PROMPT_STYLE_ABSENT)
    assert [style for presence, style in pairs if presence != SYSTEM_PROMPT_NONE] == list(
        SYSTEM_PROMPT_STYLE_ORDER
    )
    assert len(pairs) == 1 + len(SYSTEM_PROMPT_STYLE_ORDER)


def test_pairs_fall_back_for_ontology_without_the_section() -> None:
    """An ontology without the new sections degrades to "no system prompt"."""
    assert get_system_prompt_values(_minimal_ontology()) == [
        (SYSTEM_PROMPT_NONE, SYSTEM_PROMPT_STYLE_ABSENT)
    ]


def test_system_prompt_mode_merges_presence_and_style() -> None:
    """The routing label is the style when present, ``none`` when absent."""
    assert system_prompt_mode("present", "task_constraint") == "task_constraint"
    assert system_prompt_mode(SYSTEM_PROMPT_NONE, SYSTEM_PROMPT_STYLE_ABSENT) == (
        SYSTEM_PROMPT_NONE
    )


# ── Composition-slot geometry ───────────────────────────────────────────────


def test_named_style_vector_is_one_hot_and_orthogonal() -> None:
    """Each style gets its own axis; different styles are orthogonal."""
    n = len(SYSTEM_PROMPT_STYLE_ORDER)
    vectors = [
        named_style_vector(style, dimension=16, n_styles=n)
        for style in SYSTEM_PROMPT_STYLE_ORDER
    ]
    for vector in vectors:
        assert np.linalg.norm(vector) == pytest.approx(1.0)
    for i, first in enumerate(vectors):
        for j, second in enumerate(vectors):
            expected = 1.0 if i == j else 0.0
            assert float(first @ second) == pytest.approx(expected)


def test_absent_style_contributes_no_direction() -> None:
    """The absent case has no style, so its style slot is the zero vector."""
    vector = named_style_vector(
        SYSTEM_PROMPT_STYLE_ABSENT,
        dimension=16,
        n_styles=len(SYSTEM_PROMPT_STYLE_ORDER),
    )
    assert not vector.any()


def test_named_style_vector_rejects_unknown_style() -> None:
    """An unknown style is a vocabulary violation, not an implicit zero."""
    with pytest.raises(ValueError):
        named_style_vector("casual_vibes", dimension=16, n_styles=4)


def test_named_style_vector_rejects_too_narrow_slot() -> None:
    """A slot narrower than the vocabulary cannot encode it."""
    with pytest.raises(ValueError, match="one-hot"):
        named_style_vector("domain_style", dimension=2, n_styles=4)


def test_centroid_direction_is_opposite_the_styles() -> None:
    """The absent presence direction points away from every style vector."""
    styles = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
    ]
    direction = style_centroid_direction(styles)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    for style in styles:
        unit_style = style / np.linalg.norm(style)
        assert float(direction @ unit_style) < 0


def test_centroid_direction_needs_vectors() -> None:
    """Without style vectors there is no centroid to point away from."""
    with pytest.raises(ValueError):
        style_centroid_direction([])


# ── Combinations carry the dimension ────────────────────────────────────────


def test_combinations_include_system_prompt_keys() -> None:
    """Every combination carries the two dimensions plus the merged label."""
    ontology = _minimal_ontology()
    ontology["system_prompt_presence"] = ["none", "present"]
    ontology["system_prompt_style"] = {
        style: {"anchor_concepts": ["qa"]} for style in SYSTEM_PROMPT_STYLE_ORDER
    }
    combos = _build_all_combinations(ontology, languages=[], task_types=[])
    assert len(combos) == 1 + len(SYSTEM_PROMPT_STYLE_ORDER)
    for combo in combos:
        assert set(combo) == {
            "language",
            "knowledge_domain",
            "capability",
            "conversation_type",
            "system_prompt_presence",
            "system_prompt_style",
            "system_prompt_mode",
        }
        assert combo["system_prompt_mode"] == system_prompt_mode(
            combo["system_prompt_presence"], combo["system_prompt_style"]
        )


def test_absent_presence_never_claims_a_style() -> None:
    """``none`` is paired with the absent marker only, never with a real style."""
    combos = _build_all_combinations(_real_ontology(), languages=[], task_types=[])
    absent = [c for c in combos if c["system_prompt_presence"] == SYSTEM_PROMPT_NONE]
    assert absent, "the real ontology declares the absent presence"
    for combo in absent:
        assert combo["system_prompt_style"] == SYSTEM_PROMPT_STYLE_ABSENT
        assert combo["system_prompt_mode"] == SYSTEM_PROMPT_NONE


def test_combinations_modes_are_unique() -> None:
    """The merged label does not collide across combinations."""
    combos = _build_all_combinations(_real_ontology(), languages=[], task_types=[])
    modes = {
        (combo["language"], combo["knowledge_domain"], combo["capability"],
         combo["conversation_type"], combo["system_prompt_mode"])
        for combo in combos
    }
    assert len(modes) == len(combos)


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
