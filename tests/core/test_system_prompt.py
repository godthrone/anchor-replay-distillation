"""Tests for the system-prompt sampling dimension (:mod:`ard.core.system_prompt`).

Covers the vocabulary contract, the merge of the two ontology dimensions into
the routing label, and the generation prompt handed to the input generator.
"""

from __future__ import annotations

import json

import pytest

from ard.core.system_prompt import (
    SYSTEM_PROMPT_NONE,
    SYSTEM_PROMPT_STYLE_ABSENT,
    build_system_prompt_prompt,
    get_system_prompt_values,
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


def test_ontology_presence_values_and_absent_marker() -> None:
    """Presence is binary and the absent marker is the vocabulary's ``none``."""
    ontology = _real_ontology()
    assert ontology["system_prompt_presence"] == [SYSTEM_PROMPT_NONE, "present"]
    assert SYSTEM_PROMPT_STYLE_ABSENT == SYSTEM_PROMPT_NONE


def test_generation_instructions_cover_every_ontology_style() -> None:
    """The instruction table is keyed by exactly the ontology's style vocabulary."""
    from ard.core.system_prompt import SYSTEM_PROMPT_GENERATION_INSTRUCTIONS

    ontology = _real_ontology()
    assert set(SYSTEM_PROMPT_GENERATION_INSTRUCTIONS) == set(ontology["system_prompt_style"])
    for instruction in SYSTEM_PROMPT_GENERATION_INSTRUCTIONS.values():
        assert instruction.strip()


# ── (presence, style) pairs ─────────────────────────────────────────────────


def test_pairs_cover_every_mode_exactly_once() -> None:
    """One pair per mode: absent once, one per style for the present case."""
    ontology = _real_ontology()
    pairs = get_system_prompt_values(ontology)
    assert pairs[0] == (SYSTEM_PROMPT_NONE, SYSTEM_PROMPT_STYLE_ABSENT)
    assert [style for presence, style in pairs if presence != SYSTEM_PROMPT_NONE] == list(
        ontology["system_prompt_style"]
    )
    assert len(pairs) == 1 + len(ontology["system_prompt_style"])


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
