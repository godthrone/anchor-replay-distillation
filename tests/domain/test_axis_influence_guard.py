"""Guard: no axis may be a mascot — every axis must change what generation sees.

The WP-S14 audit found six of the twelve v4 axes (``response_style``,
``output_format``, ``difficulty``, ``context_length``, ``input_condition``,
``answer_mode``) reached the sampler's coordinates but never a prompt: two specs
differing only on one of them issued the *same* generator-side request.  The
block-level consequence was 935 blocks projecting onto only 112 prompt-effective
restricted combinations.

This guard walks all twelve axes: for each one it takes two coordinates that
differ **only** on that axis and asserts their generator-side requests differ.
The compared tuple is the one the audit pinned —

* ``I`` — the input generator's first-turn system string
  (:func:`~ard.domain.text_anchor._build_user_prompt`);
* ``S`` — the system-message generation request (``""`` for mode ``none``);
* ``T`` — the spec turn count for the coordinate's ``conversation_type``;
* ``V`` — the ``visual_domain`` leaf (``""`` for text-only).

A guard that cannot fail proves nothing, so the negative control removes the six
axes' wording and asserts the guard then names **exactly** those six.
"""

from __future__ import annotations

from typing import Any

import pytest

from ard.backends.axis_instruction_loader import load_axis_instructions
from ard.backends.prompt_loader import build_system_prompt_prompt
from ard.core.axis_instruction import INSTRUCTION_AXES
from ard.core.ontology import OntologyV4
from ard.core.sampling import turn_counts_by_conversation_type
from ard.core.system_prompt import SYSTEM_PROMPT_NONE
from ard.core.types import TurnSpec
from ard.domain.text_anchor import _build_user_prompt

#: The axes that must reach the prompt as wording (WP-S17).  Imported, not
#: re-listed, so the guard cannot drift from the module it guards.
WORDING_AXES = INSTRUCTION_AXES


def _base_meta(ontology: OntologyV4) -> dict[str, Any]:
    """One text-only coordinate: first value of every axis but ``visual_domain``."""
    meta: dict[str, Any] = {
        axis: ontology.axis_values(axis)[0]
        for axis in ontology.axis_names()
        if axis != "visual_domain"
    }
    meta["modality"] = "text_only"
    return meta


def _input_generator_system(meta: dict[str, Any]) -> str:
    messages = _build_user_prompt(TurnSpec(turn_index=0, role="user", is_final=True), [], meta)
    content = messages[0]["content"]
    assert isinstance(content, str)
    return content


def _signature(meta: dict[str, Any], turn_counts: dict[str, int]) -> tuple[str, str, int, str]:
    """The deterministic generator-side requests a coordinate fixes (I, S, T, V)."""
    mode = meta["system_prompt_mode"]
    system_request = "" if mode == SYSTEM_PROMPT_NONE else build_system_prompt_prompt(meta, mode)
    return (
        _input_generator_system(meta),
        system_request,
        turn_counts[meta["conversation_type"]],
        meta.get("visual_domain") or "",
    )


def _pair(
    ontology: OntologyV4, axis: str, base: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two coordinates differing only on *axis* (modality is set for the image axis)."""
    values = ontology.axis_values(axis)
    assert len(values) >= 2, axis
    first = dict(base)
    second = dict(base)
    first[axis] = values[0]
    second[axis] = values[1]
    if axis == "visual_domain":
        first["modality"] = second["modality"] = "image"
    return first, second


def _axes_without_a_difference(ontology: OntologyV4) -> list[str]:
    """Axes whose two values render the identical (I, S, T, V) tuple.

    Empty means no mascot axis.  The negative control shows what a failure looks
    like: the six wording axes reappear here, by name.
    """
    turn_counts = turn_counts_by_conversation_type(ontology)
    base = _base_meta(ontology)
    without: list[str] = []
    for axis in ontology.axis_names():
        first, second = _pair(ontology, axis, base)
        if _signature(first, turn_counts) == _signature(second, turn_counts):
            without.append(axis)
    return without


def test_every_axis_changes_the_generated_requests(ontology: OntologyV4) -> None:
    """No axis may be a mascot: two coordinates differing only on it must differ."""
    without = _axes_without_a_difference(ontology)
    assert without == [], (
        f"these axes do not change the generator-side requests, so they are inert: {without}"
    )


def test_negative_control_without_the_wording_names_exactly_the_six_axes(
    ontology: OntologyV4, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remove the six axes' wording: the guard must fail, naming exactly those six.

    This is the proof that the guard above can fail.  With
    :func:`~ard.backends.axis_instruction_loader.build_axis_requirements`
    neutralised — the pre-fix state — only the six instruction axes lose their
    prompt difference; the other six axes still differ through I/S/T/V.
    """
    monkeypatch.setattr(
        "ard.domain.text_anchor.build_axis_requirements",
        lambda anchor_meta, directory=None: "",
    )
    assert _axes_without_a_difference(ontology) == list(WORDING_AXES)


@pytest.mark.parametrize("axis", WORDING_AXES)
def test_each_wording_axis_changes_the_prompt_text(axis: str, ontology: OntologyV4) -> None:
    """The six axes change the input-generator prompt itself, sentence by sentence."""
    base = _base_meta(ontology)
    first, second = _pair(ontology, axis, base)
    sentences = load_axis_instructions(axis)
    first_prompt = _input_generator_system(first)
    second_prompt = _input_generator_system(second)
    assert sentences[first[axis]] in first_prompt
    assert sentences[second[axis]] in second_prompt
    # Each prompt carries only its own value's sentence, never the other's.
    assert sentences[second[axis]] not in first_prompt
    assert sentences[first[axis]] not in second_prompt


def test_legacy_coordinate_sentence_is_still_rendered_byte_for_byte(
    ontology: OntologyV4,
) -> None:
    """WP-S17 adds the requirement clause; it does not rewrite the legacy words."""
    meta = _base_meta(ontology)
    expected = (
        f"Generate a realistic user message in {meta['language']} "
        f"on the topic of {meta['knowledge_domain']}. "
        f"The user is asking for a {meta['capability']} task. "
        f"The conversation style is {meta['conversation_type']}."
    )
    assert expected in _input_generator_system(meta)
