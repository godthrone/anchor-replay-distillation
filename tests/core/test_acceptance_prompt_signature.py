# test_acceptance_prompt_signature.py — Guard the honesty of the structure readout.
# Responsibility: pin the two diversity readings against the code they describe:
#   * `acceptance.PROMPT_SIGNATURE_AXES` is exactly the anchor_meta fields the
#     generator-side assembly reads (no mascot, and no unlisted field matters) —
#     probed in **both** modalities, with a real picture, so the image branch is
#     rendered rather than merely named;
#   * `prompt_signature_distinct` / `effective_projection_distinct` count
#     *distinct values*, not plan entries (the negative control below);
#   * the real plan lands on 935 / 891;
#   * two coordinates share a signature **iff** their rendered requests are
#     byte-identical, over the materialised legal block set.

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ard.backends.prompt_loader import build_system_prompt_prompt
from ard.core import acceptance, constraints, sampling
from ard.core.ontology import OntologyV4
from ard.core.sampling import sample_anchors, turn_counts_by_conversation_type
from ard.core.system_prompt import SYSTEM_PROMPT_NONE
from ard.core.types import AnchorGenerationConfig, TurnSpec
from ard.domain.text_anchor import _build_user_prompt

if TYPE_CHECKING:
    from tests.conftest import DomainImages

#: The anchor_meta fields that are plan bookkeeping, not prompt input.  The
#: claim is that they never reach the generated request; this file renders
#: both ways to keep that claim honest.
NON_PROMPT_FIELDS = ("modality", "has_image", "image_count")

#: The modalities a coordinate is rendered under.  Both are probed because
#: ``_build_user_prompt`` chooses its instruction from ``image_data_url``
#: rather than from ``modality``, so probing only the text branch left the whole
#: image branch — and the ``knowledge_domain`` axis inside it — unrendered.
MODALITIES = (sampling.MODALITY_TEXT, sampling.MODALITY_IMAGE)


def _meta(ontology: OntologyV4, modality: str = sampling.MODALITY_TEXT) -> dict[str, Any]:
    """One coordinate: first value of every axis, in the requested modality.

    ``visual_domain`` is carried only by an image coordinate — the ontology makes
    it conditional on ``modality == "image"`` and
    :meth:`ard.core.sampling.Coordinate.as_dict` omits the key from a text-only
    coordinate, so a text coordinate naming a leaf is not a legal coordinate.
    """
    meta: dict[str, Any] = {
        axis: ontology.axis_values(axis)[0]
        for axis in ontology.axis_names()
        if modality == sampling.MODALITY_IMAGE or axis != "visual_domain"
    }
    meta["modality"] = modality
    return meta


def _image_data_url(meta: dict[str, Any], domain_images: DomainImages) -> str | None:
    """The data URI of the picture *meta* carries, or ``None`` for a text coordinate.

    ``visual_domain`` is the image-selecting free axis, so its presence decides the
    branch here.  A named domain whose picture did not resolve is reported as a
    fixture failure — silently rendering the text branch would make every assertion
    below pass while testing nothing.
    """
    visual_domain = meta.get("visual_domain")
    if visual_domain is None:
        return None
    url = domain_images.data_url(visual_domain)
    assert url is not None, (
        f"no fixture picture for {visual_domain!r}: the image branch would go unrendered"
    )
    return url


def _rendered_request(
    meta: dict[str, Any], turn_counts: dict[str, int], domain_images: DomainImages
) -> tuple[Any, ...]:
    """The generator-side request tuple used here as the signature witness.

    ``I`` = the input generator's system string; ``U`` = the user half of that call,
    carrying the picture's data URI when the coordinate has one; ``S`` = the
    system-message generation request (``""`` for mode ``none``); ``T`` = the spec
    turn count; ``V`` = the address of the picture the coordinate **resolved to**
    (``""`` when absent) — the resolved picture, not the raw leaf name, so ``V``
    stands for what the generator actually receives.
    """
    turn = TurnSpec(turn_index=0, role="user", is_final=True)
    messages = _build_user_prompt(
        turn, [], meta, image_data_url=_image_data_url(meta, domain_images)
    )
    mode = meta["system_prompt_mode"]
    mode_request = "" if mode == SYSTEM_PROMPT_NONE else build_system_prompt_prompt(meta, mode)
    return (
        messages[0]["content"],
        messages[1]["content"],
        mode_request,
        str(turn_counts[meta["conversation_type"]]),
        domain_images.address(meta.get("visual_domain")),
    )


def _probe_pair(ontology: OntologyV4, base: dict[str, Any], axis: str) -> dict[str, Any]:
    """*base* with *axis* moved to a different value, legal in the same modality.

    ``visual_domain`` is conditional on the image modality, so its second value can
    only exist on an image coordinate; the contrast moves the modality with it and
    stays legal (§2.3).  Every other axis keeps *base*'s modality.
    """
    values = ontology.axis_values(axis)
    other = {**base, axis: values[-1]}
    if axis == "visual_domain":
        other["modality"] = sampling.MODALITY_IMAGE
    return other


def test_prompt_signature_axes_are_exactly_the_fields_the_renderers_read(
    ontology: OntologyV4, domain_images: DomainImages
) -> None:
    """Every listed axis changes the render; no unlisted field does.

    Probed under **both** modalities.  The image probe renders a real picture
    (byte-distinct per domain, resolved by the production resolver), which is what
    covers the image branch at all.
    """
    turn_counts = turn_counts_by_conversation_type(ontology)

    for modality in MODALITIES:
        base = _meta(ontology, modality)
        baseline = _rendered_request(base, turn_counts, domain_images)

        for axis in acceptance.PROMPT_SIGNATURE_AXES:
            values = ontology.axis_values(axis)
            assert len(values) >= 2, f"{axis} cannot be probed with {len(values)} value(s)"
            other = _probe_pair(ontology, base, axis)
            assert other != base, f"{axis} is listed but changing it changed nothing"
            assert _rendered_request(other, turn_counts, domain_images) != baseline, (
                f"{axis} is in PROMPT_SIGNATURE_AXES but does not change the generator-side "
                f"request under modality {modality!r} — a mascot axis must not be counted "
                "as prompt-effective"
            )

        for field in NON_PROMPT_FIELDS:
            stamped = {**base, field: "image" if field == "modality" else 1}
            assert _rendered_request(stamped, turn_counts, domain_images) == baseline, (
                f"{field} is not in PROMPT_SIGNATURE_AXES but changes the generator-side "
                "request: the signature is missing a field the assembly consumes"
            )


def _legal_block_coordinates(ontology: OntologyV4) -> list[dict[str, Any]]:
    """Every legal block, materialised, with the free axes decorrelated.

    One base coordinate per ``(modality, legal restricted block)`` coverage unit,
    plus — for each free axis the modality actually carries — the same coordinate
    with that axis moved to a *different* value.  The result contains, inside every
    legal block, a contrast on every free axis, so an axis that is inert in any one
    region shows up as a pair whose signatures differ while its requests are
    byte-identical.

    ``visual_domain`` is swept only on image coordinates: the ontology makes it
    conditional on the image modality, and only image-capable capabilities form
    legal image blocks (R5), so it has no text-side contrast to move.

    Enumerating the whole free-axis product (≈ 5·10⁷ coordinates) is not feasible;
    this set is exhaustive over the blocks and one-contrast-per-axis inside each.
    """
    coordinates: list[dict[str, Any]] = []
    for modality, blocks in (
        (sampling.MODALITY_TEXT, constraints.enumerate_legal_blocks(ontology)),
        (
            sampling.MODALITY_IMAGE,
            constraints.enumerate_legal_blocks(ontology, image_capable_only=True),
        ),
    ):
        swept = constraints.FREE_AXES
        if modality == sampling.MODALITY_IMAGE:
            swept = (*constraints.FREE_AXES, "visual_domain")
        for block in blocks:
            base = {**_meta(ontology, modality), **block.as_dict()}
            coordinates.append(base)
            for axis in swept:
                coordinates.append({**base, axis: ontology.axis_values(axis)[-1]})
    return coordinates


def test_shared_prompt_signature_iff_byte_identical_request(
    ontology: OntologyV4, domain_images: DomainImages
) -> None:
    """★ Two coordinates share a prompt signature **iff** their requests are identical.

    ``PROMPT_SIGNATURE_AXES`` names the fields the assembly "actually reads", and
    ``PROMPT_SIGNATURE_DEFINITION`` promises the signature is shared exactly when the
    generator-side request is the same.  This pins the second direction, the one that
    can fail: coordinates equal on the signature agree on every axis the renderer
    reads, so equal bytes ⇒ equal signature is the direction an inert axis breaks —
    it makes two byte-identical requests carry two different signatures, splitting
    the noise band's repeat groups.

    Method: materialise the legal block set (:func:`_legal_block_coordinates`),
    render each coordinate once, and compare the two partitions of the same index
    set — ``{prompt_signature}`` and ``{rendered request bytes}``.  The invariant
    holds iff the partitions are equal.
    """
    turn_counts = turn_counts_by_conversation_type(ontology)
    coordinates = _legal_block_coordinates(ontology)

    by_signature: dict[tuple[Any, ...], list[int]] = {}
    by_bytes: dict[str, list[int]] = {}
    signature_of: dict[int, tuple[Any, ...]] = {}
    for index, meta in enumerate(coordinates):
        signature = acceptance.prompt_signature(meta)
        wire = json.dumps(_rendered_request(meta, turn_counts, domain_images), ensure_ascii=False)
        by_signature.setdefault(signature, []).append(index)
        by_bytes.setdefault(wire, []).append(index)
        signature_of[index] = signature

    # The two partitions are equal iff the invariant holds.  On failure, name the
    # first few conflicting groups (an inert axis in a wide region conflicts in
    # hundreds of groups; listing them all would bury the report).
    signature_partition = {frozenset(group) for group in by_signature.values()}
    byte_partition = {frozenset(group) for group in by_bytes.values()}

    n_conflicting = 0
    offenders: list[str] = []
    for group in by_bytes.values():
        signatures = {signature_of[index] for index in group}
        if len(signatures) <= 1:
            continue
        n_conflicting += 1
        if len(offenders) >= 3:
            continue
        first, second = sorted(group)[:2]
        differing = [
            axis
            for axis in acceptance.PROMPT_SIGNATURE_AXES
            if coordinates[first].get(axis) != coordinates[second].get(axis)
        ]
        offenders.append(
            f"{len(group)} coordinates render identically but hold {len(signatures)} "
            f"signatures; {first} vs {second} differ only on {differing} "
            f"(modality {coordinates[first].get('modality')!r})"
        )
    assert signature_partition == byte_partition, (
        "prompt signature and rendered request disagree over the legal block set: "
        f"{n_conflicting} rendered-request group(s) hold more than one signature — "
        + "; ".join(offenders)
    )

    # Non-vacuity: the swept set is large, and every coordinate in it holds a
    # distinct signature — so the two partitions can only coincide because the
    # render really is a function of the signature, not because the set collapsed.
    assert len(coordinates) > 10_000
    assert len(by_signature) == len(coordinates)


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
    # The three restricted axes that were once inert (and are now rendered) must be
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
    readout = acceptance.structure_readout(plan, ontology=ontology)

    assert readout.plan_image_entries == 2
    assert readout.image_block_count == 1
    assert readout.prompt_signature_distinct.image == 1
    assert readout.effective_projection_distinct.image == 1
    assert readout.coverage.distinct_coordinates == 2  # not identical: the two differ
    assert readout.within_rule is False  # 2 entries is not this run's count


def test_distinct_counts_fall_when_one_prompt_axis_is_shared(ontology: OntologyV4) -> None:
    """Two images specs differing on no prompt axis are one signature."""
    meta = _meta(ontology)
    forced = {**meta, "conversation_type": ontology.axis_values("conversation_type")[0]}
    readout = acceptance.structure_readout([dict(forced), dict(forced)], ontology=ontology)
    assert readout.prompt_signature_distinct.text_only == 1
    assert readout.effective_projection_distinct.text_only == 1


def test_the_real_plan_lands_on_the_s14_numbers(ontology: OntologyV4) -> None:
    """Regression pin: the real plan's 935 / 891 block counts."""
    plan = sample_anchors(ontology, AnchorGenerationConfig(seed=42))
    readout = acceptance.structure_readout([spec.anchor_meta for spec in plan], ontology=ontology)

    assert readout.text_block_count == 935
    assert readout.image_block_count == 891
    assert readout.prompt_signature_distinct.text_only == 935
    assert readout.prompt_signature_distinct.image == 891
    assert readout.effective_projection_distinct.text_only == 935
    assert readout.effective_projection_distinct.image == 891
    assert readout.within_rule is True
    assert acceptance.structure_mismatch(readout) is None


def test_diversity_declaration_is_carried_by_the_readout(ontology: OntologyV4) -> None:
    """The counts travel with their definition (§1.4)."""
    readout = acceptance.structure_readout([], ontology=ontology)
    assert readout.diversity.prompt_signature_axes == acceptance.PROMPT_SIGNATURE_AXES
    assert readout.diversity.effective_projection_axes == acceptance.EFFECTIVE_PROJECTION_AXES
    assert "ordered tuple" in readout.diversity.prompt_signature_definition
    assert readout.diversity.counted_over.startswith("plan entries")
    assert sampling.MODALITY_TEXT in readout.diversity.counted_over
