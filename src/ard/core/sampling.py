"""Deterministic cycle-shuffle sampling for ontology v4 (ARD v5 sampling core).

Responsibility: turn a validated v4 ontology into the run's anchor coordinates
(:class:`Coordinate`) and :class:`~ard.core.types.AnchorSpec` objects, using the
**cycle-shuffle** rule of ``docs/algorithm.md`` §2.

The rule, in words:

* a **coverage unit** is one ``(modality, legal restricted block)`` pair.  The
  unit list is the runtime enumeration of
  :meth:`~ard.core.constraints.ConstraintEvaluator.enumerate_legal_blocks` —
  text units first, then image units, in ontology declaration order.  No count
  in this module is hard-coded: adding a legal block, a leaf or an ontology
  value changes the plan automatically (§1.4 单一真相源).
* the plan is indexed by ``i``.  ``c, pos = divmod(i, U)`` selects cycle ``c``
  and the unit at position ``pos`` of ``order(c)``, a seeded shuffle of the
  unit list.  The cycle index enters the shuffle seed, so a changed ontology
  cannot masquerade as "the order did not change".
* ``knowledge_domain`` and (for image units) ``visual_domain`` rotate with the
  cycle: ``index = (b(u) + c) % leaf_count``, where ``b(u)`` is the unit's
  fixed index in the unit list.  Step ``1`` (not a coprime-looking step like
  ``37``) keeps the rotation valid for any leaf count a user adds.
* the four free axes (``language`` / ``response_style`` / ``difficulty`` /
  ``context_length``) are drawn from a per-plan-index seeded generator.

Because the plan is exactly ``[coordinate(i) for i in range(N)]``,
``plan(seed, N1)`` is a prefix of ``plan(seed, N2)`` for ``N1 <= N2`` — the
load-bearing property behind "coverage never decreases as N grows".

There is no maximum N.  ``count`` may exceed ``max_plan_size`` (one full
rotation of every unit through every knowledge leaf); the plan simply runs more
cycles, and a coordinate that recurs is a **new sample**, not a duplicate — the
id is a plan-position serial number (:func:`format_anchor_id`), so both
occurrences are kept.  Only ``count >= 1`` is enforced here as a defensive guard
(§2.3) — rejecting an unusable request belongs to the configuration boundary,
and :func:`plan_rounds` gives the caller the round decomposition it needs to
*report* a large request instead of refusing it.  Nothing is ever silently
truncated (§3.4).

Pure computation: the only I/O is reading the ontology through
:mod:`ard.core.ontology`.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from ard.core.constraints import ConstraintEvaluator, RestrictedBlock
from ard.core.ontology import FlatAxisWithAttributes, OntologyV4
from ard.core.types import (
    AnchorGenerationConfig,
    AnchorSpec,
    AxisValuesByAxis,
    TurnSpec,
)

#: The sampling algorithm's auditable name.  It enters a run's :func:`run_key`
#: and a plan's identity (:class:`PlanIdentity`), so a future algorithm change
#: cannot impersonate this one.  ``v1`` is the cycle-shuffle rule defined here.
SAMPLING_ALGORITHM = "cycle-shuffle/v1"

MODALITY_TEXT = "text_only"
"""The ``modality`` sample field of a text-only coordinate."""

MODALITY_IMAGE = "image"
"""The ``modality`` sample field of a coordinate that carries an image."""

# ── The free axes the rule assigns ──────────────────────────────────────────

#: The free axes the rule assigns by rotation.  Documented as data so the
#: rotation rule is readable in one place; the rotation itself is in
#: :func:`_iter_plan`.
ROTATED_FREE_AXES: tuple[str, ...] = ("knowledge_domain", "visual_domain")

#: The free axes the rule draws from the per-index seeded generator.  The order
#: is the draw order and therefore part of the plan's byte-level identity.
RANDOM_FREE_AXES: tuple[str, ...] = (
    "language",
    "response_style",
    "difficulty",
    "context_length",
)

#: The documented fallback when an ontology declares no ``language`` value.
#: A text-only ontology need not declare one; a silently empty coordinate would
#: violate §2.2, so the fallback is explicit.
DEFAULT_LANGUAGE = "English"

# ── Turn counts: from the ontology, never from a configuration knob ─────────
#
# ``conversation_type.value_attributes.turns`` is the single source of a plan
# entry's turn count (§1.4).  The ontology counts *exchanges* — a user turn plus
# the reply it gets — while :class:`~ard.core.types.AnchorSpec` carries the
# conversation *prefix* that ends on the final user question (the answer is the
# generation target, not a spec turn).  So ``n`` ontology turns are
# ``2 * n - 1`` spec turns: ``single_turn``(1) → 1, ``clarification``(2) → 3,
# ``troubleshooting``(3) → 5, ``constraint_update``(4) → 7.  Every resolved count
# is therefore odd and ends on a user turn, which is exactly what
# :class:`~ard.core.types.AnchorSpec` validates.

MULTI_TURN_LITERAL = "multi"
"""The ontology's marker for a conversation type whose turn count is variable."""

MULTI_TURN_DEFAULT = 4
"""Ontology turns planned for a ``"multi"`` conversation type.

``tool_assisted`` and ``source_review`` declare ``turns: "multi"`` — a variable
count, because it depends on how many tool calls or revision rounds the exchange
needs.  A deterministic plan still needs one number per entry; this is the
single place that number is defined.

Basis: **the largest turn count the v4 ontology itself declares**
(``constraint_update = 4``).  Taking the ontology's own maximum invents no
number outside the single source of truth, and makes a ``"multi"`` type planned
at least as long as any fixed multi-turn type.  Exposed as one named constant so
a reviewer can ratify or change it in one line; the guard in
:func:`_spec_turns` refuses to run if the ontology ever declares something larger
behind it.
"""


class SamplingError(ValueError):
    """The ontology cannot produce the plan the sampling rule describes.

    Raised for an unusable ``count``, a duplicated anchor id, or an ontology
    that cannot supply a coordinate's axis values.  Never raised for a *large*
    plan: N has no ceiling (§2.3 — the guard rejects illegal requests, it does
    not truncate legal ones).
    """


# ── Coverage units: the runtime enumeration that defines U ──────────────────


@dataclass(frozen=True, slots=True)
class CoverageUnit:
    """One ``(modality, legal restricted block)`` pair — the sampling atom.

    Attributes:
        modality: :data:`MODALITY_TEXT` or :data:`MODALITY_IMAGE`.
        block: The six restricted axes, as enumerated by
            :class:`~ard.core.constraints.ConstraintEvaluator`.  The image units
            are the image-capable subset of the text blocks, so a block can
            appear twice — once per modality.
    """

    modality: str
    block: RestrictedBlock


def coverage_units(ontology: OntologyV4) -> list[CoverageUnit]:
    """Return the plan's coverage units, in ontology exhaustion order.

    Text units come first, then image units; within each group the order is
    :meth:`~ard.core.constraints.ConstraintEvaluator.enumerate_legal_blocks`
    order.  This list is the single source of ``U`` (§1.4): nothing in this
    module hard-codes how many units there are.

    The result is a fresh list over a memoised, immutable unit tuple, so a
    caller may reorder it without disturbing the next call.

    Args:
        ontology: A validated v4 ontology.

    Returns:
        The units, text then image.
    """
    return list(_unit_tuple(ontology))


#: Bounded memo of the enumerated unit tuple, keyed by ontology fingerprint.
#: The enumeration expands the raw restricted-block product and validates one
#: block per combination, so every counter (``unit_total`` / ``max_plan_size`` /
#: ``plan_rounds`` / ``_iter_plan``) would otherwise redo the same half-second of
#: work.  The key is content-derived and the ontology model is frozen, so a
#: cache hit is exactly the value a recomputation would produce (§5 先正确，
#: 后可优化 — the cache changes cost, never the result).
_UNIT_MEMO: dict[str, tuple[CoverageUnit, ...]] = {}
_UNIT_MEMO_LIMIT = 4


def _unit_tuple(ontology: OntologyV4) -> tuple[CoverageUnit, ...]:
    """Return the units as an immutable tuple, memoised by ontology fingerprint."""
    fingerprint = ontology_sha256(ontology)
    cached = _UNIT_MEMO.get(fingerprint)
    if cached is not None:
        return cached
    evaluator = ConstraintEvaluator(ontology)
    units = tuple(
        CoverageUnit(modality=MODALITY_TEXT, block=block)
        for block in evaluator.enumerate_legal_blocks()
    ) + tuple(
        CoverageUnit(modality=MODALITY_IMAGE, block=block)
        for block in evaluator.enumerate_legal_blocks(image_capable_only=True)
    )
    if len(_UNIT_MEMO) >= _UNIT_MEMO_LIMIT:
        _UNIT_MEMO.clear()
    _UNIT_MEMO[fingerprint] = units
    return units


def unit_total(ontology: OntologyV4) -> int:
    """Return ``U`` — how many coverage units one full cycle visits."""
    return len(_unit_tuple(ontology))


def knowledge_leaf_count(ontology: OntologyV4) -> int:
    """Return ``K`` — the number of ``knowledge_domain`` leaves the ontology declares."""
    return len(ontology.axis_values("knowledge_domain"))


def visual_leaf_count(ontology: OntologyV4) -> int:
    """Return ``V`` — the number of ``visual_domain`` leaves the ontology declares."""
    return len(ontology.axis_values("visual_domain"))


def max_plan_size(ontology: OntologyV4) -> int:
    """Return ``K * U`` — how many coordinates one full rotation spans.

    ``K`` cycles over ``U`` units is the point at which the knowledge axis has
    visited every leaf for every unit, so coordinate content can recur after it.
    Recurrence is **not** an error and nothing is dropped: the same content asked
    again is a new sample.  This count is a reporting quantity (and the boundary
    the design documents), not a limit — :func:`sample_coordinates` accepts
    larger counts on purpose, and an anchor's id
    (:func:`format_anchor_id`) is a plan-position serial number that does not
    depend on this number at all.
    """
    return knowledge_leaf_count(ontology) * unit_total(ontology)


def plan_rounds(ontology: OntologyV4, count: int) -> tuple[int, int]:
    """Split ``count`` into full cycles plus the tail of the last cycle.

    This is the reporting helper for a large N: the caller logs "X full cycles,
    Y coordinates in the final cycle, N coordinates expected" instead of
    refusing the request.

    Args:
        ontology: A validated v4 ontology.
        count: The requested number of coordinates (``>= 1``).

    Returns:
        ``(full_cycles, last_cycle_size)`` = ``divmod(count, U)``.  A
        ``last_cycle_size`` of ``0`` means the plan ends exactly on a cycle
        boundary: with ``U`` cycles' worth of units, ``count = U`` gives
        ``(1, 0)`` and ``count = U + 4`` gives ``(1, 4)``.

    Raises:
        SamplingError: If ``count`` is not an integer ``>= 1``.
    """
    _require_positive_count(ontology, count)
    return divmod(count, unit_total(ontology))


# ── Plan identity inputs: the ontology fingerprint and the seeded generators ─


def ontology_sha256(ontology: OntologyV4) -> str:
    """Return the content fingerprint of ``ontology``.

    The fingerprint is the SHA-256 of the ontology's canonical JSON form (keys
    sorted, ASCII, no insignificant whitespace).  It is a pure function of the
    ontology's content: loading the same file yields the same fingerprint in any
    process, under any ``PYTHONHASHSEED``, and editing any field — including
    adding one knowledge-domain leaf — yields a different one.  It is hashed
    into every coordinate's shuffle and free-axis generators and recorded in a
    plan's identity, so "the ontology changed" can never be mistaken for "the
    plan did not change".
    """
    canonical = json.dumps(
        ontology.model_dump(mode="json"),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hash_seed(*parts: object) -> int:
    """Derive a deterministic RNG seed from ``parts``.

    The parts are canonicalised as a JSON array of strings (ASCII, no
    insignificant whitespace) and hashed with SHA-256, so the seed is
    reproducible across processes and hash seeds — unlike Python's built-in
    ``hash``, which is salted by ``PYTHONHASHSEED``.
    """
    canonical = json.dumps([str(part) for part in parts], ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest(), "big")


# ── Coordinates ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Coordinate:
    """One anchor's full coordinate: the six restricted axes plus the free ones.

    ``modality`` is the v4 *sample field* that switches the conditional axis on
    and off (``reading_rules.axis_vs_sample_field``): a text-only coordinate has
    ``visual_domain is None``, an image coordinate always names a visual leaf.
    The fields are unchanged from v4, so bank records and downstream consumers
    are unaffected.  A coordinate is **content**, not identity: the same
    coordinate may legitimately be sampled again in a later cycle, and the
    anchor id says *where in the plan* a sample sits (see
    :func:`format_anchor_id`), never *what it contains*.
    """

    modality: str
    language: str
    knowledge_domain: str
    capability: str
    system_prompt_mode: str
    conversation_type: str
    response_style: str
    output_format: str
    difficulty: str
    context_length: str
    input_condition: str
    answer_mode: str
    visual_domain: str | None = None

    def as_dict(self) -> dict[str, str]:
        """Return the coordinate as an ordinary ``axis -> value`` mapping.

        The ``visual_domain`` key is omitted for a text-only coordinate, so the
        mapping holds exactly the axes that sample has — no ``None`` placeholder
        a consumer could mistake for a value.
        """
        meta = {
            "modality": self.modality,
            "language": self.language,
            "knowledge_domain": self.knowledge_domain,
            "capability": self.capability,
            "system_prompt_mode": self.system_prompt_mode,
            "conversation_type": self.conversation_type,
            "response_style": self.response_style,
            "output_format": self.output_format,
            "difficulty": self.difficulty,
            "context_length": self.context_length,
            "input_condition": self.input_condition,
            "answer_mode": self.answer_mode,
        }
        if self.visual_domain is not None:
            meta["visual_domain"] = self.visual_domain
        return meta


def _draw(values: tuple[str, ...], rng: random.Random) -> str:
    """Draw one value from ``values`` using the plan's seeded generator."""
    return values[rng.randrange(len(values))]


def _build_coordinate(
    block: RestrictedBlock,
    *,
    modality: str,
    knowledge_domain: str,
    visual_domain: str | None,
    rng: random.Random,
    free_values: AxisValuesByAxis,
) -> Coordinate:
    """Combine one legal restricted block with its free-axis choices."""
    languages = free_values["language"]
    return Coordinate(
        modality=modality,
        language=_draw(languages, rng) if languages else DEFAULT_LANGUAGE,
        knowledge_domain=knowledge_domain,
        capability=block.capability,
        system_prompt_mode=block.system_prompt_mode,
        conversation_type=block.conversation_type,
        response_style=_draw(free_values["response_style"], rng),
        output_format=block.output_format,
        difficulty=_draw(free_values["difficulty"], rng),
        context_length=_draw(free_values["context_length"], rng),
        input_condition=block.input_condition,
        answer_mode=block.answer_mode,
        visual_domain=visual_domain,
    )


def _free_rng(seed: int, fingerprint: str, plan_index: int) -> random.Random:
    """Return the generator of one plan index's four free axes.

    The free draws depend on the plan index ``i`` — **and only on ``i``**, never
    on N.  That is what makes ``plan(seed, N1)`` a prefix of ``plan(seed, N2)``:
    a coordinate is constructed independently of how many coordinates follow it.
    Kept as a named seam so a test can inject a count-dependent derivation and
    prove the prefix-property test catches it (see the negative controls in
    ``tests/core/test_sampling.py``).
    """
    return random.Random(_hash_seed(seed, fingerprint, "free", plan_index))


def _units_by_modality(units: tuple[CoverageUnit, ...]) -> tuple[list[int], list[int]]:
    """Return the unit indices of each modality, in unit-list order."""
    text = [index for index, unit in enumerate(units) if unit.modality == MODALITY_TEXT]
    image = [index for index, unit in enumerate(units) if unit.modality == MODALITY_IMAGE]
    return text, image


def _require_positive_count(ontology: OntologyV4, count: object) -> None:
    """Refuse a ``count`` that is not an integer ``>= 1`` (§2.3 防线).

    A large count is *not* refused: N has no ceiling.  The message still quotes
    the plan's shape so a caller reading the failure knows what it asked for.
    """
    if isinstance(count, int) and not isinstance(count, bool) and count >= 1:
        return
    raise SamplingError(
        f"count must be at least 1 (received: {count!r}); one full cycle is "
        f"{unit_total(ontology)} coordinates and one full rotation is "
        f"{max_plan_size(ontology)} = {knowledge_leaf_count(ontology)} x "
        f"{unit_total(ontology)}, so a request of {count!r} does not reach round 1. "
        "Larger counts are allowed — they simply run more cycles."
    )


def _validate_per_modality(
    ontology: OntologyV4,
    per_modality: object,
    text_count: int,
    image_count: int,
) -> tuple[int, int]:
    """Validate the smoke-only ``per_modality`` subset request."""
    if not isinstance(per_modality, tuple) or len(per_modality) != 2:
        raise SamplingError(
            f"per_modality must be a (text, image) tuple of unit counts, received: {per_modality!r}"
        )
    text, image = per_modality
    for name, value, limit in (("text", text, text_count), ("image", image, image_count)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SamplingError(
                f"per_modality {name} count must be an integer >= 0, received: {value!r}"
            )
        if value > limit:
            raise SamplingError(
                f"per_modality {name} count {value} exceeds the {limit} legal {name} "
                f"units this ontology declares ({unit_total(ontology)} units in total)"
            )
    if text + image < 1:
        raise SamplingError("per_modality=(0, 0) is not a plan: it would sample no anchor")
    return text, image


def _iter_plan(
    ontology: OntologyV4,
    seed: int,
    *,
    count: int | None,
    per_modality: tuple[int, int] | None,
) -> Iterator[tuple[Coordinate, int]]:
    """Yield ``(coordinate, cycle)`` in plan order — the rule's single loop.

    This is the one construction path — :func:`sample_coordinates` and every
    test consume it, so no second sampling rule can drift away from this one
    (§18.1 不留负债).  It yields ``(coordinate, cycle)`` rather than the bare
    coordinate because the cycle is needed to place the sample in the plan.

    Args:
        ontology: A validated v4 ontology.
        seed: The run seed.
        count: The requested coordinate count, or ``None`` for one full cycle.
            Mutually exclusive with ``per_modality``.
        per_modality: Smoke-only ``(text, image)`` request: take the first *k*
            units of each modality from cycle 0's shuffle order.  Mutually
            exclusive with ``count``.

    Yields:
        ``(coordinate, cycle)``, the cycle index being ``divmod(i, U)[0]``.

    Raises:
        SamplingError: If the request is unusable (``count`` below 1, both
            selectors given, or a ``per_modality`` count out of range).
    """
    if per_modality is not None and count is not None:
        raise SamplingError(
            "count and per_modality are mutually exclusive: count selects a plan "
            "length, per_modality selects a per-modality subset of cycle 0"
        )

    units = _unit_tuple(ontology)
    total = len(units)
    leaves_k = ontology.axis_values("knowledge_domain")
    leaves_v = ontology.axis_values("visual_domain")
    free_values: AxisValuesByAxis = {axis: ontology.axis_values(axis) for axis in RANDOM_FREE_AXES}
    fingerprint = ontology_sha256(ontology)
    orders: dict[int, list[int]] = {}

    def order(cycle: int) -> list[int]:
        """Return the shuffled unit-index order of ``cycle`` (cached per cycle)."""
        cached = orders.get(cycle)
        if cached is None:
            positions = list(range(total))
            random.Random(_hash_seed(seed, fingerprint, "cycle", cycle)).shuffle(positions)
            orders[cycle] = positions
            cached = positions
        return cached

    def coordinate(unit_index: int, cycle: int, plan_index: int) -> Coordinate:
        unit = units[unit_index]
        knowledge = leaves_k[(unit_index + cycle) % len(leaves_k)]
        visual = None
        if unit.modality == MODALITY_IMAGE:
            visual = leaves_v[(unit_index + cycle) % len(leaves_v)]
        rng = _free_rng(seed, fingerprint, plan_index)
        return _build_coordinate(
            unit.block,
            modality=unit.modality,
            knowledge_domain=knowledge,
            visual_domain=visual,
            rng=rng,
            free_values=free_values,
        )

    if per_modality is not None:
        text_units, image_units = _units_by_modality(units)
        wanted_text, wanted_image = _validate_per_modality(
            ontology, per_modality, len(text_units), len(image_units)
        )
        first_cycle = order(0)
        selected = [index for index in first_cycle if units[index].modality == MODALITY_TEXT][
            :wanted_text
        ]
        selected.extend(
            [index for index in first_cycle if units[index].modality == MODALITY_IMAGE][
                :wanted_image
            ]
        )
        for plan_index, unit_index in enumerate(selected):
            yield coordinate(unit_index, 0, plan_index), 0
        return

    resolved = total if count is None else count
    _require_positive_count(ontology, resolved)
    for plan_index in range(resolved):
        cycle, position = divmod(plan_index, total)
        yield coordinate(order(cycle)[position], cycle, plan_index), cycle


def sample_coordinates(
    ontology: OntologyV4,
    seed: int,
    *,
    count: int | None = None,
    per_modality: tuple[int, int] | None = None,
) -> list[Coordinate]:
    """Build the run's coordinate plan.

    The plan is ``[coordinate(i) for i in range(N)]``:

    * ``c, pos = divmod(i, U)``; the unit is ``order(c)[pos]``, where
      ``order(c)`` is the unit list shuffled with
      ``H(seed, ontology_sha256, "cycle", c)``;
    * ``knowledge_domain`` is ``leaves[(b(u) + c) % K]``; image units also take
      ``visual_domain = leaves_v[(b(u) + c) % V]``;
    * the four free axes come from ``H(seed, ontology_sha256, "free", i)``.

    **No uniqueness gate is applied to coordinates.**  A coordinate is content,
    not identity: the same content can be asked again in a later cycle and both
    samples are legitimate (the generation model is stochastic — the same
    labels yield a different question and a different answer), so dropping a
    repeat would throw away a valid sample and would contradict the very idea of
    cycling.  In-cycle unit and coordinate distinctness is a *test* invariant;
    it is deliberately not a runtime check.

    Args:
        ontology: A validated v4 ontology.
        seed: The run seed driving the shuffle, the rotation and the free draws.
            It comes from the config snapshot, never from a module constant.
        count: How many coordinates to plan; ``None`` means exactly one full
            cycle (``unit_total(ontology)``).  Any value ``>= 1`` is accepted —
            there is no ceiling.  Mutually exclusive with ``per_modality``.
        per_modality: Smoke-only ``(text, image)`` request, e.g. ``(4, 4)`` for
            four text plus four image coordinates taken from cycle 0's shuffle
            order.  Used by ``--smoke``; mutually exclusive with ``count``.

    Returns:
        The plan, in plan order, as a list of :class:`Coordinate`.  Every
        requested index is present; nothing is filtered out.

    Raises:
        SamplingError: If ``count`` is below 1, if both selectors are given, or
            if a ``per_modality`` count is out of range.
    """
    return [
        coordinate
        for coordinate, _ in _iter_plan(ontology, seed, count=count, per_modality=per_modality)
    ]


# ── Turn counts ─────────────────────────────────────────────────────────────


def _spec_turns(declared: int | str | None, value: str) -> int:
    """Translate one ontology ``turns`` attribute into a spec turn count.

    Args:
        declared: The ``turns`` attribute of one ``conversation_type`` value.
        value: The conversation-type value, used to name a violation.

    Returns:
        The number of :class:`~ard.core.types.AnchorSpec` turns — always odd,
        i.e. ``2 * ontology_turns - 1`` (see :data:`MULTI_TURN_DEFAULT`).

    Raises:
        SamplingError: If the attribute is neither a positive integer nor
            :data:`MULTI_TURN_LITERAL`, or if it exceeds
            :data:`MULTI_TURN_DEFAULT` — which would silently invalidate the
            documented basis of the ``"multi"`` bound.
    """
    if declared == MULTI_TURN_LITERAL:
        return 2 * MULTI_TURN_DEFAULT - 1
    if not isinstance(declared, int) or isinstance(declared, bool) or declared < 1:
        raise SamplingError(
            f"conversation_type {value!r} declares an unusable turns attribute: "
            f"{declared!r} (expected a positive integer or "
            f"{MULTI_TURN_LITERAL!r})"
        )
    if declared > MULTI_TURN_DEFAULT:
        raise SamplingError(
            f"conversation_type {value!r} declares turns={declared}, above "
            f"MULTI_TURN_DEFAULT={MULTI_TURN_DEFAULT}: the 'multi' bound is no longer "
            "the ontology's largest declared turn count — re-derive it."
        )
    return 2 * declared - 1


def turn_counts_by_conversation_type(ontology: OntologyV4) -> dict[str, int]:
    """Return the spec turn count of every ``conversation_type`` value.

    The counts come from ``conversation_type.value_attributes.turns`` and from
    nothing else — there is no configuration knob for turns (see
    :data:`MULTI_TURN_DEFAULT` for how the literal ``"multi"`` resolves).

    Args:
        ontology: A validated v4 ontology.

    Returns:
        ``conversation_type value -> number of AnchorSpec turns``, one entry per
        declared value, in ontology order.

    Raises:
        SamplingError: If the axis carries no per-value turn attributes, if a
            value carries none, or if a declared count is unusable.
    """
    spec = ontology.axes.spec("conversation_type")
    if not isinstance(spec, FlatAxisWithAttributes):
        raise SamplingError(
            "conversation_type must carry per-value turn attributes to derive the "
            f"plan's turn counts, got {type(spec).__name__}"
        )
    counts: dict[str, int] = {}
    for value in ontology.axis_values("conversation_type"):
        attributes = spec.value_attributes.get(value)
        if attributes is None:
            raise SamplingError(f"conversation_type {value!r} declares no turn attributes")
        counts[value] = _spec_turns(attributes.root.get("turns"), value)
    return counts


# ── Anchor ids: the plan position, not the coordinate ───────────────────────
#
# An anchor id is a *serial number*: it says where in the plan a sample sits.
# It carries no coordinate content, so two cycles over the same coordinate get
# two different ids and both samples survive — a stochastic generator asked
# twice with the same labels produces a different question and a different
# answer, and discarding the repeat would discard a valid sample.


def run_key(ontology: OntologyV4, seed: int) -> str:
    """Return the 8-hex-digit key that scopes a run's anchor ids.

    ``run_key = H(ontology_sha256(ontology), seed, SAMPLING_ALGORITHM)[:8]``.

    The key answers "which run does this id belong to?":

    * **content-derived**: it depends only on the ontology's fingerprint, the
      seed and the sampling algorithm's name, so it is a pure function of the
      run's inputs and is reproducible in any process, under any
      ``PYTHONHASHSEED``;
    * **it does not contain the plan length N** — on purpose.  If N entered the
      key, raising N on a resume would rewrite every id already written to the
      bank and break the prefix (see :func:`format_anchor_id`);
    * **run identity**: two runs with a different seed or a different ontology
      (or a future sampling algorithm) get a different key, so their ids cannot
      be mistaken for each other.

    Args:
        ontology: A validated v4 ontology.
        seed: The run seed.

    Returns:
        The first 8 hex digits of the SHA-256 digest of the canonical triple.
    """
    canonical = json.dumps(
        [ontology_sha256(ontology), seed, SAMPLING_ALGORITHM],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def format_anchor_id(run: str, cycle: int, position: int) -> str:
    """Return the id of the anchor at ``(cycle, position)`` in run ``run``.

    The id is ``f"{run}-c{cycle:05d}p{position:05d}"`` — for example
    ``"1a2b3c4d-c00000p00137"`` is the 138th sample of the first cycle.  The
    four properties the id must have, and where each comes from:

    (a) **Stable prefix as N grows.**  The id depends on the plan index only
        through ``divmod(i, U)`` and on the run only through the seed, the
        ontology and the algorithm — never on N.  Raising N therefore leaves
        every already-generated id byte-identical, so a resumed run can append
        to the bank without rewriting it.
    (b) **Deterministic.**  Every input is an integer or a digest rendered by a
        fixed format, and ``run`` in turn is a SHA-256 over a canonical JSON
        triple — no ``hash()``, no ``set`` iteration, no locale, no float.  The
        same ``(ontology, seed, N)`` yields the same ids in any process, under
        any ``PYTHONHASHSEED``.
    (c) **Distinguishable across runs.**  ``run`` mixes the ontology
        fingerprint, the seed and the sampling algorithm, so ids from a
        different seed, ontology or algorithm do not collide (up to the 32-bit
        prefix of the digest — the same width a run directory name needs, and
        the bank's uniqueness gate lives inside one run).
    (d) **Readable and locatable.**  ``cNNNNN`` is the zero-based cycle and
        ``pNNNNN`` the position inside that cycle, both zero-padded to five
        digits, so an id in a log or a bank line says exactly which sample of
        which cycle it is and can be rebuilt from the plan index alone.

    The id is the bank's primary key, and the plan positions are distinct by
    construction — a cycle has ``U`` positions and a plan index maps to exactly
    one ``(cycle, position)`` pair — so ids are pairwise distinct within a run
    without any runtime uniqueness check.

    Args:
        run: The run key from :func:`run_key`.
        cycle: The zero-based cycle index, ``divmod(i, U)[0]``.
        position: The zero-based position inside the cycle, ``divmod(i, U)[1]``.

    Returns:
        ``"<run>-c<cycle:05d>p<position:05d>"``.

    Raises:
        ValueError: If ``cycle`` or ``position`` is negative or does not fit the
            five-digit field — a silently truncated field would let two
            positions share an id.
    """
    for name, value in (("cycle", cycle), ("position", position)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"anchor id {name} must be a non-negative integer, got {value!r}")
        if value > 99999:
            raise ValueError(
                f"anchor id {name} {value} does not fit the five-digit field; "
                "widen the field before two positions can share an id"
            )
    return f"{run}-c{cycle:05d}p{position:05d}"


# ── Anchor specs ────────────────────────────────────────────────────────────


def _build_spec(
    coordinate: Coordinate, num_turns: int, *, run: str, cycle: int, position: int
) -> AnchorSpec:
    """Build one :class:`AnchorSpec` from a coordinate and its plan position.

    The id is the plan position's serial number (:func:`format_anchor_id`); the
    coordinate goes into ``anchor_meta`` unchanged, so the record format is
    exactly the v4 one and the bank and its consumers are unaffected (§1.4 —
    the id is a label, the metadata is the coordinate).
    """
    meta = coordinate.as_dict()
    turns = [
        TurnSpec(
            turn_index=index,
            role="user" if index % 2 == 0 else "assistant",
            generation_instruction=None,
            is_final=(index == num_turns - 1),
        )
        for index in range(num_turns)
    ]
    return AnchorSpec(
        id=format_anchor_id(run, cycle, position),
        anchor_meta=meta,
        turns=turns,
        input_generator_id=None,
    )


def build_specs(
    coordinates: Iterable[Coordinate],
    *,
    ontology: OntologyV4,
    seed: int,
) -> list[AnchorSpec]:
    """Turn a coordinate plan into the run's :class:`AnchorSpec` list.

    Each spec's turn count is the one its own ``conversation_type`` declares in
    the ontology (see :func:`turn_counts_by_conversation_type`); its id is the
    serial number of its plan position, ``(index // U, index % U)``.  A
    ``per_modality`` smoke subset never spans more than one cycle, so its
    positions are ``(0, index)``.  No randomness is involved, so the same
    ``(coordinates, ontology, seed)`` yields the same specs.

    Args:
        coordinates: The plan from :func:`sample_coordinates`, in plan order.
        ontology: The validated v4 ontology the coordinates came from.
        seed: The run seed (enters :func:`run_key`; the ids of an existing
            prefix do not depend on how many coordinates follow).

    Returns:
        One spec per coordinate, in the same order.

    Raises:
        SamplingError: If a coordinate's ``conversation_type`` has no declared
            turn count, or a declared count is unusable.
    """
    ordered = list(coordinates)
    turn_counts = turn_counts_by_conversation_type(ontology)
    unknown = {
        coordinate.conversation_type
        for coordinate in ordered
        if coordinate.conversation_type not in turn_counts
    }
    if unknown:
        raise SamplingError(
            f"coordinates carry conversation_type values the ontology does not "
            f"declare: {sorted(unknown)}"
        )
    total = unit_total(ontology)
    run = run_key(ontology, seed)
    return [
        _build_spec(
            coordinate,
            num_turns=turn_counts[coordinate.conversation_type],
            run=run,
            cycle=index // total,
            position=index % total,
        )
        for index, coordinate in enumerate(ordered)
    ]


def sample_anchors(
    ontology: OntologyV4,
    config: AnchorGenerationConfig,
    *,
    count: int | None = None,
    per_modality: tuple[int, int] | None = None,
) -> list[AnchorSpec]:
    """Sample the run's plan from ``ontology`` (§12.1 entry point).

    ``count`` is the run's N and comes from the caller (the pipeline passes the
    ``[generation] count`` value); ``None`` means exactly one full cycle.  A
    caller that needs a prefix (the checkpoint/resume path asks only for the
    anchors still missing) slices the result — the plan itself is never
    shortened here, because a truncated plan would silently skip coordinates
    (§3.4), and the ids of the first N samples do not depend on N.

    Args:
        ontology: A validated v4 ontology.
        config: Generation configuration; only ``seed`` is read.  The turn count
            of each entry comes from the ontology, not from the config.
        count: How many coordinates to plan; ``None`` means one full cycle.
        per_modality: Smoke-only ``(text, image)`` subset of cycle 0.

    Returns:
        The plan's :class:`AnchorSpec` objects, in plan order.

    Raises:
        SamplingError: If the request is unusable, or the ontology cannot supply
            a turn count for one of the coordinates.
    """
    coordinates = sample_coordinates(
        ontology,
        config.seed,
        count=count,
        per_modality=per_modality,
    )
    return build_specs(coordinates, ontology=ontology, seed=config.seed)


# ── Plan identity: the auditable name of a plan ─────────────────────────────
#
# ``seed`` is a *configuration* value, not a plan's name.  Two runs can share a
# seed and build different plans (a different ontology, a different N), and — the
# failure this recipe exists for — the seed recorded in a run directory can be
# overwritten by a later invocation that sampled nothing, while the bank still
# holds the *first* invocation's plan.  A plan's auditable name is therefore a
# digest over the plan itself plus the inputs that produced it:
# ``PlanIdentity.of(plan, ontology_sha256=..., seed=..., count=..., unit_total=...)``.
#
# The digest is a pure function of the ordered coordinate list and those inputs:
# the axes are read in the explicit order below (never in ``dict``/``set``
# iteration order), the canonical form is ASCII JSON with sorted keys, and the
# recipe version and the sampling algorithm name are framed into the hashed
# payload — so the same plan yields the same digest in any process, under any
# ``PYTHONHASHSEED``, and a future recipe or algorithm change cannot collide with
# this one.  The four things a resume guard must tell apart — a different
# ontology, a different seed, a different N, a different algorithm version — are
# all in the payload.

PLAN_IDENTITY_VERSION = 2
"""Version of the digest recipe below; bump it when the canonical form changes."""

PLAN_IDENTITY_ALGORITHM = "sha256"
"""The one hash function :meth:`PlanIdentity.of` uses."""

PLAN_IDENTITY_AXES: tuple[str, ...] = (
    "modality",
    "language",
    "knowledge_domain",
    "capability",
    "system_prompt_mode",
    "conversation_type",
    "response_style",
    "output_format",
    "difficulty",
    "context_length",
    "input_condition",
    "answer_mode",
    "visual_domain",
)
"""The coordinate axes, in digest order — fixed here, not derived from a mapping."""


@dataclass(frozen=True, slots=True)
class PlanIdentity:
    """The stable, reproducible name of a plan.

    Attributes:
        algorithm: The hash function, always :data:`PLAN_IDENTITY_ALGORITHM`.
        version: The recipe version, always :data:`PLAN_IDENTITY_VERSION`.
        sampling: The sampling algorithm, always :data:`SAMPLING_ALGORITHM`.
        ontology_sha256: The ontology fingerprint the plan was built from (see
            :func:`ontology_sha256`).
        seed: The run seed.
        count: The requested N exactly as the caller gave it; ``None`` means
            "one full cycle".
        unit_total: ``U`` for that ontology, so a reader can decompose
            ``plan_size`` into cycles without re-loading the ontology.
        plan_size: How many coordinates the plan holds.
        digest: Hex SHA-256 of the canonical, ordered coordinate list and the
            inputs above.
    """

    algorithm: str
    version: int
    sampling: str
    ontology_sha256: str
    seed: int
    count: int | None
    unit_total: int
    plan_size: int
    digest: str

    @classmethod
    def of(
        cls,
        specs: Iterable[AnchorSpec],
        *,
        ontology_sha256: str,
        seed: int,
        count: int | None,
        unit_total: int,
    ) -> PlanIdentity:
        """Return the identity of *specs*, in the order given.

        The digest covers each spec's id, its coordinate values (the axes it
        carries, in :data:`PLAN_IDENTITY_AXES` order) and its turn count, plus
        the ontology fingerprint, the seed, the requested ``count``, ``U``, the
        plan size and the algorithm/recipe version.  Two plans with the same
        coordinates and ids get the same digest; reordering them, dropping one,
        changing one value, or changing any of those inputs changes it.

        Args:
            specs: The plan, as :class:`~ard.core.types.AnchorSpec` objects.
            ontology_sha256: The ontology fingerprint (see
                :func:`ontology_sha256`).
            seed: The run seed.
            count: The requested N as given to :func:`sample_coordinates`;
                ``None`` means "one full cycle".
            unit_total: ``U`` for the ontology (``unit_total(ontology)``).

        Returns:
            The plan's :class:`PlanIdentity`.
        """
        ordered = list(specs)
        rows: list[list[Any]] = []
        for spec in ordered:
            meta = spec.anchor_meta or {}
            rows.append(
                [
                    spec.id,
                    [[axis, str(meta[axis])] for axis in PLAN_IDENTITY_AXES if axis in meta],
                    len(spec.turns),
                ]
            )
        plan_size = len(ordered)
        payload = json.dumps(
            {
                "sampling": SAMPLING_ALGORITHM,
                "version": PLAN_IDENTITY_VERSION,
                "algorithm": PLAN_IDENTITY_ALGORITHM,
                "ontology_sha256": ontology_sha256,
                "seed": seed,
                "count": count,
                "unit_total": unit_total,
                "plan_size": plan_size,
                "rows": rows,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return cls(
            algorithm=PLAN_IDENTITY_ALGORITHM,
            version=PLAN_IDENTITY_VERSION,
            sampling=SAMPLING_ALGORITHM,
            ontology_sha256=ontology_sha256,
            seed=seed,
            count=count,
            unit_total=unit_total,
            plan_size=plan_size,
            digest=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the identity as the mapping a run artifact records."""
        return {
            "algorithm": self.algorithm,
            "version": self.version,
            "sampling": self.sampling,
            "ontology_sha256": self.ontology_sha256,
            "seed": self.seed,
            "count": self.count,
            "unit_total": self.unit_total,
            "plan_size": self.plan_size,
            "digest": self.digest,
        }
