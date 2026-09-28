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
* ``U`` — the user half of that call, carrying the picture's data URI when the
  coordinate has one;
* ``S`` — the system-message generation request (``""`` for mode ``none``);
* ``T`` — the spec turn count for the coordinate's ``conversation_type``;
* ``V`` — the *resolved* picture address (``""`` for text-only).

**Every axis is probed under both modalities.**  WP-26 showed why that matters:
``_build_user_prompt`` chooses its instruction from ``image_data_url``, not from
``modality``, so a probe that only flips the ``modality`` field while passing no
picture never renders the image branch at all — the whole branch was untested, and
``knowledge_domain`` was silently inert for image coordinates whose
``system_prompt_mode`` is ``none``.  The picture therefore comes from
:func:`domain_images <tests.conftest.domain_images>` — real, byte-distinct PNGs
resolved by the production resolver — and an image coordinate without one is an
error, never a silent fall back to the text branch (§2.3).

A guard that cannot fail proves nothing, so the negative control removes the six
axes' wording and asserts the guard then names **exactly** those six.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ard.backends.axis_instruction_loader import load_axis_instructions
from ard.backends.prompt_loader import build_system_prompt_prompt
from ard.core.axis_instruction import INSTRUCTION_AXES
from ard.core.ontology import OntologyV4
from ard.core.sampling import turn_counts_by_conversation_type
from ard.core.system_prompt import SYSTEM_PROMPT_NONE
from ard.core.types import TurnSpec
from ard.domain.text_anchor import _build_user_prompt

if TYPE_CHECKING:
    from tests.conftest import DomainImages

#: The axes that must reach the prompt as wording (WP-S17).  Imported, not
#: re-listed, so the guard cannot drift from the module it guards.
WORDING_AXES = INSTRUCTION_AXES

#: The two modalities every axis is probed under.  ``image`` uses the ontology's
#: first ``visual_domain`` leaf, so the coordinate is legal in both directions.
MODALITIES = ("text_only", "image")


def _base_meta(ontology: OntologyV4, modality: str = "text_only") -> dict[str, Any]:
    """One coordinate: first value of every axis, in the requested modality.

    ``visual_domain`` is set only for the image modality — the ontology declares
    it ``{"requires_modality": "image"}``, so a text coordinate carrying a leaf
    would not be a legal coordinate.
    """
    meta: dict[str, Any] = {
        axis: ontology.axis_values(axis)[0]
        for axis in ontology.axis_names()
        if axis != "visual_domain"
    }
    meta["modality"] = modality
    if modality == "image":
        meta["visual_domain"] = ontology.axis_values("visual_domain")[0]
    return meta


def _image_data_url(meta: dict[str, Any], domain_images: DomainImages) -> str | None:
    """The data URI of the picture *meta* carries, or ``None`` for a text coordinate.

    An image coordinate whose picture did not resolve is a fixture failure and is
    reported as one — falling back to the text branch would make this guard pass
    while testing nothing.
    """
    if meta.get("modality") != "image":
        return None
    url = domain_images.data_url(meta.get("visual_domain"))
    assert url is not None, (
        f"image coordinate {meta.get('visual_domain')!r} resolved to no picture: "
        "the image branch would go unprobed"
    )
    return url


def _input_generator_system(meta: dict[str, Any], domain_images: DomainImages) -> str:
    """The input generator's system string for *meta* — the channel ``I``."""
    turn = TurnSpec(turn_index=0, role="user", is_final=True)
    messages = _build_user_prompt(
        turn, [], meta, image_data_url=_image_data_url(meta, domain_images)
    )
    content = messages[0]["content"]
    assert isinstance(content, str)
    return content


def _signature(
    meta: dict[str, Any], turn_counts: dict[str, int], domain_images: DomainImages
) -> tuple[Any, ...]:
    """The deterministic generator-side requests a coordinate fixes (I, U, S, T, V)."""
    mode = meta["system_prompt_mode"]
    system_request = "" if mode == SYSTEM_PROMPT_NONE else build_system_prompt_prompt(meta, mode)
    turn = TurnSpec(turn_index=0, role="user", is_final=True)
    messages = _build_user_prompt(
        turn, [], meta, image_data_url=_image_data_url(meta, domain_images)
    )
    return (
        messages[0]["content"],
        messages[1]["content"],
        system_request,
        turn_counts[meta["conversation_type"]],
        domain_images.address(meta.get("visual_domain")),
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


def _inert_probes(ontology: OntologyV4, domain_images: DomainImages) -> list[tuple[str, str]]:
    """``(axis, modality)`` probes whose two values render the identical tuple.

    Empty means no mascot axis, in either modality.
    """
    turn_counts = turn_counts_by_conversation_type(ontology)
    inert: list[tuple[str, str]] = []
    for modality in MODALITIES:
        base = _base_meta(ontology, modality)
        for axis in ontology.axis_names():
            first, second = _pair(ontology, axis, base)
            if _signature(first, turn_counts, domain_images) == _signature(
                second, turn_counts, domain_images
            ):
                inert.append((axis, modality))
    return inert


def _axes_without_a_difference(ontology: OntologyV4, domain_images: DomainImages) -> list[str]:
    """Axis names that are inert in at least one modality, in ontology order.

    The negative control shows what a failure looks like: the six wording axes
    reappear here, by name.
    """
    inert = {axis for axis, _ in _inert_probes(ontology, domain_images)}
    return [axis for axis in ontology.axis_names() if axis in inert]


def test_every_axis_changes_the_generated_requests(
    ontology: OntologyV4, domain_images: DomainImages
) -> None:
    """No axis may be a mascot: two coordinates differing only on it must differ.

    The probe runs under ``text_only`` **and** ``image``; the image one renders a
    real picture, so the image branch is genuinely exercised (WP-26's blind spot).
    """
    inert = _inert_probes(ontology, domain_images)
    assert inert == [], (
        "these (axis, modality) probes do not change the generator-side requests, "
        f"so they are inert there: {inert}"
    )


def test_negative_control_without_the_wording_names_exactly_the_six_axes(
    ontology: OntologyV4, domain_images: DomainImages, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remove the six axes' wording: the guard must fail, naming exactly those six.

    This is the proof that the guard above can fail.  With
    :func:`~ard.backends.axis_instruction_loader.build_axis_requirements`
    neutralised — the pre-fix state — only the six instruction axes lose their
    prompt difference; the other six axes still differ through I/U/S/T/V.
    """
    monkeypatch.setattr(
        "ard.domain.text_anchor.build_axis_requirements",
        lambda anchor_meta, directory=None: "",
    )
    assert _axes_without_a_difference(ontology, domain_images) == list(WORDING_AXES)


@pytest.mark.parametrize("axis", WORDING_AXES)
def test_each_wording_axis_changes_the_prompt_text(
    axis: str, ontology: OntologyV4, domain_images: DomainImages
) -> None:
    """The six axes change the input-generator prompt itself, sentence by sentence."""
    base = _base_meta(ontology)
    first, second = _pair(ontology, axis, base)
    sentences = load_axis_instructions(axis)
    first_prompt = _input_generator_system(first, domain_images)
    second_prompt = _input_generator_system(second, domain_images)
    assert sentences[first[axis]] in first_prompt
    assert sentences[second[axis]] in second_prompt
    # Each prompt carries only its own value's sentence, never the other's.
    assert sentences[second[axis]] not in first_prompt
    assert sentences[first[axis]] not in second_prompt


def test_legacy_coordinate_sentence_is_still_rendered_byte_for_byte(
    ontology: OntologyV4, domain_images: DomainImages
) -> None:
    """WP-S17 adds the requirement clause; it does not rewrite the legacy words."""
    meta = _base_meta(ontology)
    expected = (
        f"Generate a realistic user message in {meta['language']} "
        f"on the topic of {meta['knowledge_domain']}. "
        f"The user is asking for a {meta['capability']} task. "
        f"The conversation style is {meta['conversation_type']}."
    )
    assert expected in _input_generator_system(meta, domain_images)
