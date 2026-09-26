# test_acceptance_prompt_signature.py — Guard the honesty of the structure readout.
# Responsibility: pin the two WP-S18 readings against the code they describe:
#   * `acceptance.PROMPT_SIGNATURE_AXES` is exactly the anchor_meta fields the
#     generator-side assembly reads (no mascot, and no unlisted field matters);
#   * `prompt_signature_distinct` / `effective_projection_distinct` count
#     *distinct values*, not plan entries (the negative control behind WP-S18 ④);
#   * the real plan lands on the 935 / 891 WP-S14's audit measured;
#   * the noise band's repeat groups are prompt signatures, so a field no
#     consumer reads cannot split one cell.

from __future__ import annotations

from typing import Any

import pytest

from ard.backends.embedding_client import EmbeddingMatrix
from ard.backends.prompt_loader import build_system_prompt_prompt
from ard.core import acceptance, constraints, sampling
from ard.core import coverage as ruler
from ard.core.ontology import OntologyV4
from ard.core.sampling import sample_anchors, turn_counts_by_conversation_type
from ard.core.system_prompt import SYSTEM_PROMPT_NONE
from ard.core.types import AnchorGenerationConfig, TurnSpec
from ard.domain.text_anchor import _build_user_prompt

#: The anchor_meta fields that are plan bookkeeping, not prompt input.  WP-S18
#: claims they never reach the generated request; this file renders both ways
#: to keep that claim honest.
NON_PROMPT_FIELDS = ("modality", "has_image", "image_count")


def _vector_set(name: str, rows: EmbeddingMatrix) -> ruler.VectorSet:
    return ruler.VectorSet(name=name, vectors=rows)


def _meta(ontology: OntologyV4, modality: str = "text_only") -> dict[str, Any]:
    """One full coordinate: first value of every axis, in ontology order."""
    meta: dict[str, Any] = {axis: ontology.axis_values(axis)[0] for axis in ontology.axis_names()}
    meta["modality"] = modality
    return meta


def _rendered_request(meta: dict[str, Any], turn_counts: dict[str, int]) -> tuple[Any, ...]:
    """The generator-side request tuple WP-S14 used as its signature witness.

    ``I`` = the input generator's system string; ``S`` = the system-message
    generation request (``""`` for mode ``none``); ``T`` = the spec turn count;
    ``V`` = the image-selecting ``visual_domain`` leaf (``""`` when absent).
    """
    messages = _build_user_prompt(TurnSpec(turn_index=0, role="user", is_final=True), [], meta)
    first = messages[0]["content"]
    assert isinstance(first, str)
    mode = meta["system_prompt_mode"]
    mode_request = "" if mode == SYSTEM_PROMPT_NONE else build_system_prompt_prompt(meta, mode)
    return (
        first,
        mode_request,
        str(turn_counts[meta["conversation_type"]]),
        meta.get("visual_domain") or "",
    )


def test_prompt_signature_axes_are_exactly_the_fields_the_renderers_read(
    ontology: OntologyV4,
) -> None:
    """Every listed axis changes the render; no unlisted field does."""
    turn_counts = turn_counts_by_conversation_type(ontology)
    base = _meta(ontology)
    baseline = _rendered_request(base, turn_counts)

    for axis in acceptance.PROMPT_SIGNATURE_AXES:
        values = ontology.axis_values(axis)
        assert len(values) >= 2, f"{axis} cannot be probed with {len(values)} value(s)"
        other = {**base, axis: values[1]}
        assert other != base, f"{axis} is listed but changing it changed nothing"
        assert _rendered_request(other, turn_counts) != baseline, (
            f"{axis} is in PROMPT_SIGNATURE_AXES but does not change the generator-side "
            "request — a mascot axis must not be counted as prompt-effective"
        )

    for field in NON_PROMPT_FIELDS:
        stamped = {**base, field: "image" if field == "modality" else 1}
        assert _rendered_request(stamped, turn_counts) == baseline, (
            f"{field} is not in PROMPT_SIGNATURE_AXES but changes the generator-side "
            "request: the signature is missing a field the assembly consumes"
        )


def test_prompt_signature_axes_cover_every_ontology_axis(ontology: OntologyV4) -> None:
    """The signature is the whole axis set — an inert axis would show up here."""
    assert set(acceptance.PROMPT_SIGNATURE_AXES) == set(ontology.axis_names())
    assert len(acceptance.PROMPT_SIGNATURE_AXES) == len(set(acceptance.PROMPT_SIGNATURE_AXES))


def test_effective_projection_is_the_restricted_axes_that_reach_the_prompt() -> None:
    """The projection is derived, and in this tree it is the whole restricted set."""
    derived = tuple(
        axis for axis in constraints.RESTRICTED_AXES if axis in acceptance.PROMPT_SIGNATURE_AXES
    )
    assert acceptance.EFFECTIVE_PROJECTION_AXES == derived
    # The three restricted axes WP-S14 found inert (and WP-S17 rendered) must be
    # in the projection: dropping them is exactly the 112/102 collapse.
    assert set(constraints.RESTRICTED_AXES) <= set(acceptance.PROMPT_SIGNATURE_AXES)
    assert acceptance.EFFECTIVE_PROJECTION_AXES == constraints.RESTRICTED_AXES


def test_distinct_counts_count_distinct_values_not_plan_entries(ontology: OntologyV4) -> None:
    """Negative-control target: a nominal count must not pass as a distinct count.

    Two entries that share one prompt signature (they differ only in fields no
    prompt consumer reads) are two plan entries but **one** effective
    specification; a "distinct" reading that returns the entry count would
    report 2 here.
    """
    meta = _meta(ontology, modality="image")
    plan = [{**meta, "has_image": True, "image_count": 1}, {**meta}]
    readout = acceptance.structure_readout(plan)

    assert readout.plan_image_entries == 2
    assert readout.image_block_count == 1
    assert readout.prompt_signature_distinct.image == 1
    assert readout.effective_projection_distinct.image == 1
    assert readout.duplicate_coordinates == 0  # the two entries are not identical
    assert readout.within_rule is False


def test_distinct_counts_fall_when_one_prompt_axis_is_shared(ontology: OntologyV4) -> None:
    """Two images specs differing on no prompt axis are one signature."""
    meta = _meta(ontology)
    forced = {**meta, "conversation_type": ontology.axis_values("conversation_type")[0]}
    readout = acceptance.structure_readout([dict(forced), dict(forced)])
    assert readout.prompt_signature_distinct.text_only == 1
    assert readout.effective_projection_distinct.text_only == 1


def test_the_real_plan_lands_on_the_s14_numbers(ontology: OntologyV4) -> None:
    """Regression pin: 935 / 891, the numbers WP-S14's audit script measured."""
    plan = sample_anchors(ontology, AnchorGenerationConfig(seed=42))
    readout = acceptance.structure_readout([spec.anchor_meta for spec in plan])

    assert readout.text_block_count == 935
    assert readout.image_block_count == 891
    assert readout.prompt_signature_distinct.text_only == 935
    assert readout.prompt_signature_distinct.image == 891
    assert readout.effective_projection_distinct.text_only == 935
    assert readout.effective_projection_distinct.image == 891
    assert readout.within_rule is True
    assert acceptance.structure_mismatch(readout) is None


def test_repeat_groups_key_on_the_prompt_signature_not_the_whole_mapping(
    ontology: OntologyV4,
) -> None:
    """Fields no consumer reads must not split a repeat group."""
    meta = _meta(ontology, modality="image")
    stamped = {**meta, "has_image": True, "image_count": 1}
    unstamped = dict(meta)  # ``quota.allocate_images`` stamps nothing for an empty pool

    assert stamped != unstamped
    assert acceptance.repeat_groups([stamped, unstamped]) == [[0, 1]]


def test_repeat_groups_split_when_a_prompt_signature_differs(ontology: OntologyV4) -> None:
    """A genuine prompt axis (instruction wording) is not one cell."""
    meta = _meta(ontology)
    other = ontology.axis_values("response_style")[1]
    assert acceptance.repeat_groups([dict(meta), {**meta, "response_style": other}]) == []


def test_noise_band_is_available_for_a_repeated_prompt_signature(ontology: OntologyV4) -> None:
    """Positive use case: two identical signatures ⇒ a band with numbers."""
    meta = _meta(ontology)
    groups = acceptance.repeat_groups([dict(meta), dict(meta)])
    assert groups == [[0, 1]]
    anchors = _vector_set("anchors", [[1.0, 0.0], [0.6, 0.8]])
    section = acceptance.noise_section(anchors, groups)

    assert section.available is True
    assert section.reason is None
    assert section.n_repeat_groups == 1
    assert section.n_pairs == 1
    assert section.band is not None
    assert section.band.lower == pytest.approx(0.4)
    assert section.band.upper == pytest.approx(0.4)


def test_noise_band_stays_unavailable_without_a_repeated_signature(
    ontology: OntologyV4,
) -> None:
    """Negative use case: no repeat ⇒ the literal ``unavailable`` statement."""
    meta = _meta(ontology)
    other = ontology.axis_values("answer_mode")[1]
    groups = acceptance.repeat_groups([dict(meta), {**meta, "answer_mode": other}])
    assert groups == []
    section = acceptance.noise_section(_vector_set("anchors", [[1.0, 0.0], [0.0, 1.0]]), groups)

    assert section.available is False
    assert section.band is None
    assert section.reason == acceptance.NOISE_UNAVAILABLE_REASON
    assert section.reason == "noise band unavailable (no repeated generation data provided)"


def test_diversity_declaration_is_carried_by_the_readout() -> None:
    """The counts travel with their definition (§1.4)."""
    readout = acceptance.structure_readout([])
    assert readout.diversity.prompt_signature_axes == acceptance.PROMPT_SIGNATURE_AXES
    assert readout.diversity.effective_projection_axes == acceptance.EFFECTIVE_PROJECTION_AXES
    assert "ordered tuple" in readout.diversity.prompt_signature_definition
    assert readout.diversity.counted_over.startswith("plan entries")
    assert sampling.MODALITY_TEXT in readout.diversity.counted_over
