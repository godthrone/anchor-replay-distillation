"""Deterministic anchor-coordinate sampling for ontology v4.

Responsibility: turn a validated v4 ontology into the anchor coordinates and
:class:`~ard.core.types.AnchorSpec` objects of a run, using the **construction
rule** — every legal restricted block exactly once, the ``knowledge_domain``
axis rotated over its tree leaves, the ``visual_domain`` axis rotated over its
leaves in the image modality, and the four remaining free axes (``language`` /
``response_style`` / ``difficulty`` / ``context_length``) drawn from the run
seed.

Two rules are absolute here:

* the sample **count** is derived from the rule and the ontology, never from a
  caller-supplied number — adding or removing a legal block changes the plan;
* every count the rule depends on is checked against the ontology before any
  coordinate is accepted, and a mismatch raises :class:`SamplingError` instead
  of silently emitting a shorter (or longer) plan (§2.3 边界校验即防呆).

Pure computation: the only I/O is reading the ontology through
:mod:`ard.core.ontology`.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from ard.core.constraints import ConstraintEvaluator, RestrictedBlock
from ard.core.ontology import FlatAxisWithAttributes, OntologyV4
from ard.core.types import AnchorGenerationConfig, AnchorSpec, TurnSpec

#: The anchor-id dimensions, in hash order.  The set is the legacy one (v3.0.0
#: ids), kept deliberately: the bank's uniqueness gate and the checkpoint/resume
#: path both key on these ids, and changing the hash input would make every
#: already-written record look like a new anchor.  On the v4 plan the five
#: dimensions are still injective — two coordinates that differ only in a
#: non-hashed axis cannot occur, because each legal restricted block contributes
#: exactly one sample (see the contract tests).
ANCHOR_ID_DIMENSIONS: tuple[str, ...] = (
    "language",
    "knowledge_domain",
    "capability",
    "conversation_type",
    "system_prompt_mode",
)

# ── The construction rule, as data ─────────────────────────────────────────
#
# These are the counts the rule *must* observe.  They are not knobs: a value
# that differs means the ontology changed and the plan would silently cover a
# different coordinate set, so :func:`sample_anchors` refuses instead.

#: The free axes the rule assigns positionally (rotating) rather than randomly.
ROTATED_FREE_AXES: tuple[str, ...] = ("knowledge_domain", "visual_domain")
#: The free axes the rule draws from the run seed.
RANDOM_FREE_AXES: tuple[str, ...] = (
    "language",
    "response_style",
    "difficulty",
    "context_length",
)

EXPECTED_TEXT_BLOCKS = 935
"""Legal restricted blocks reachable by a text-only sample (all 20 capabilities)."""

EXPECTED_IMAGE_BLOCKS = 891
"""Legal restricted blocks reachable when ``capability`` is image-capable (18)."""

EXPECTED_KNOWLEDGE_DOMAINS = 209
"""Leaves of the ``knowledge_domain`` tree — one sample per leaf, rotating."""

EXPECTED_VISUAL_DOMAINS = 21
"""Leaves of the ``visual_domain`` axis — one sample per leaf, rotating."""

EXPECTED_TOTAL = EXPECTED_TEXT_BLOCKS + EXPECTED_IMAGE_BLOCKS
"""Total plan size of one run: 935 text-only + 891 multimodal = 1,826."""

MODALITY_TEXT = "text_only"
"""The ``modality`` sample field of a text-only coordinate."""

MODALITY_IMAGE = "image"
"""The ``modality`` sample field of a coordinate that carries an image."""

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

#: The rule's four expectations as an ordered table, so one failure names every
#: mismatch at once (one run, one fix).
_COUNT_EXPECTATIONS: tuple[tuple[str, int], ...] = (
    ("knowledge_domain leaves", EXPECTED_KNOWLEDGE_DOMAINS),
    ("visual_domain leaves", EXPECTED_VISUAL_DOMAINS),
    ("legal restricted blocks (text)", EXPECTED_TEXT_BLOCKS),
    ("legal restricted blocks (image-capable)", EXPECTED_IMAGE_BLOCKS),
)


class SamplingError(ValueError):
    """The ontology cannot produce the plan the construction rule describes.

    Raised for a wrong block/leaf count or a duplicated coordinate.  Never
    raised for a *shorter* plan: the rule either produces exactly the documented
    coordinate set or it fails (§2.3).
    """


@dataclass(frozen=True, slots=True)
class AnchorCoordinate:
    """One anchor's full coordinate: the six restricted axes plus the free ones.

    ``modality`` is the v4 *sample field* that switches the conditional axis on
    and off (``reading_rules.axis_vs_sample_field``): a text-only coordinate has
    ``visual_domain is None``, an image coordinate always names a visual leaf.
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

    def identity(self) -> tuple[tuple[str, str], ...]:
        """Return a hashable identity for duplicate detection."""
        return tuple(self.as_dict().items())


def _verify_rule_counts(evaluator: ConstraintEvaluator) -> None:
    """Refuse an ontology whose counts do not match the construction rule."""
    actual = (
        len(evaluator.ontology.axis_values("knowledge_domain")),
        len(evaluator.ontology.axis_values("visual_domain")),
        len(evaluator.enumerate_legal_blocks()),
        len(evaluator.enumerate_legal_blocks(image_capable_only=True)),
    )
    expected = tuple(value for _, value in _COUNT_EXPECTATIONS)
    if actual == expected:
        return
    problems = "; ".join(
        f"{name}: expected {want} (received: {got})"
        for (name, want), got in zip(_COUNT_EXPECTATIONS, actual, strict=True)
        if got != want
    )
    raise SamplingError(
        "ontology does not match the construction rule — "
        f"{problems}. The rule produces one sample per legal restricted block "
        "and one sample per rotated leaf; a count mismatch would silently "
        "change the plan."
    )


def _rotating(values: tuple[str, ...]) -> Iterator[str]:
    """Yield ``values`` over and over, so index ``i`` receives ``values[i % n]``."""
    while True:
        yield from values


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
    free_values: dict[str, tuple[str, ...]],
) -> AnchorCoordinate:
    """Combine one legal restricted block with its free-axis choices."""
    languages = free_values["language"]
    return AnchorCoordinate(
        modality=modality,
        # A text-only ontology need not declare any language; ``English`` is the
        # documented fallback rather than a silently empty coordinate (§2.2).
        language=_draw(languages, rng) if languages else "English",
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


def _reject_duplicates(coordinates: Iterable[AnchorCoordinate]) -> None:
    """Raise :class:`SamplingError` naming the first duplicated coordinate."""
    seen: set[tuple[tuple[str, str], ...]] = set()
    for index, coordinate in enumerate(coordinates):
        identity = coordinate.identity()
        if identity in seen:
            raise SamplingError(
                f"duplicate coordinate at plan index {index}: {coordinate.as_dict()}"
            )
        seen.add(identity)


def sample_coordinates(ontology: OntologyV4, seed: int) -> tuple[AnchorCoordinate, ...]:
    """Build the run's coordinates from the construction rule.

    The plan is, in order:

    1. the 935 legal restricted blocks over all capabilities — text-only, with
       ``knowledge_domain`` rotating over its 209 leaves;
    2. the 891 legal restricted blocks whose capability is image-capable, with
       ``visual_domain`` rotating over its 21 leaves.

    Every block appears exactly once; within each group the free axes rotate or
    are drawn from ``random.Random(seed)``, so the same ``(ontology, seed)``
    yields a byte-identical coordinate tuple and a different seed does not.

    Args:
        ontology: A validated v4 ontology.
        seed: The run seed driving the random free axes.  It comes from the
            config snapshot, never from a module constant.

    Returns:
        The plan, as a tuple of :class:`AnchorCoordinate`.

    Raises:
        SamplingError: If the ontology's counts do not match the rule or if two
            coordinates repeat.
    """
    evaluator = ConstraintEvaluator(ontology)
    _verify_rule_counts(evaluator)

    rng = random.Random(seed)
    free_values: dict[str, tuple[str, ...]] = {
        axis: ontology.axis_values(axis) for axis in RANDOM_FREE_AXES
    }
    knowledge_domains = _rotating(ontology.axis_values("knowledge_domain"))
    visual_domains = _rotating(ontology.axis_values("visual_domain"))

    coordinates: list[AnchorCoordinate] = []
    for block in evaluator.enumerate_legal_blocks():
        coordinates.append(
            _build_coordinate(
                block,
                modality=MODALITY_TEXT,
                knowledge_domain=next(knowledge_domains),
                visual_domain=None,
                rng=rng,
                free_values=free_values,
            )
        )
    for block in evaluator.enumerate_legal_blocks(image_capable_only=True):
        coordinates.append(
            _build_coordinate(
                block,
                modality=MODALITY_IMAGE,
                knowledge_domain=next(knowledge_domains),
                visual_domain=next(visual_domains),
                rng=rng,
                free_values=free_values,
            )
        )

    _reject_duplicates(coordinates)
    return tuple(coordinates)


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


def generate_anchor_id(meta: dict[str, Any]) -> str:
    """Return the stable id of an anchor with metadata ``meta``.

    The id is a hash over :data:`ANCHOR_ID_DIMENSIONS` in that order.  Every
    dimension that can make two anchors different must be part of it: two
    anchors that differ only in their system prompt are different training
    samples, and collapsing them onto one id would make the bank's uniqueness
    gate drop a legitimate record.

    Args:
        meta: The anchor's coordinate mapping (``modality`` and ``visual_domain``
            are present for image coordinates but are not hashed — see the
            note on :data:`ANCHOR_ID_DIMENSIONS`).

    Returns:
        ``"anchor_"`` plus the first 16 hex digits of the SHA-256 digest.
    """
    raw = "|".join(str(meta.get(key, "")) for key in ANCHOR_ID_DIMENSIONS)
    return "anchor_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_spec(coordinate: AnchorCoordinate, num_turns: int) -> AnchorSpec:
    """Build one :class:`AnchorSpec` from a coordinate and its turn count."""
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
        id=generate_anchor_id(meta),
        anchor_meta=meta,
        turns=turns,
        input_generator_id=None,
    )


def build_specs(
    coordinates: tuple[AnchorCoordinate, ...],
    *,
    ontology: OntologyV4,
) -> list[AnchorSpec]:
    """Turn a coordinate plan into the run's :class:`AnchorSpec` list.

    Each spec's turn count is the one its own ``conversation_type`` declares in
    the ontology (see :func:`turn_counts_by_conversation_type`); no randomness
    is involved, so the same ``(coordinates, ontology)`` yields the same specs.

    Args:
        coordinates: The plan from :func:`sample_coordinates`, in plan order.
        ontology: The validated v4 ontology the coordinates came from.

    Returns:
        One spec per coordinate, in the same order.

    Raises:
        SamplingError: If a coordinate's ``conversation_type`` has no declared
            turn count, or a declared count is unusable.
    """
    turn_counts = turn_counts_by_conversation_type(ontology)
    unknown = {
        coordinate.conversation_type
        for coordinate in coordinates
        if coordinate.conversation_type not in turn_counts
    }
    if unknown:
        raise SamplingError(
            f"coordinates carry conversation_type values the ontology does not "
            f"declare: {sorted(unknown)}"
        )
    return [
        _build_spec(coordinate, num_turns=turn_counts[coordinate.conversation_type])
        for coordinate in coordinates
    ]


def sample_anchors(
    ontology: OntologyV4,
    config: AnchorGenerationConfig,
) -> list[AnchorSpec]:
    """Sample the run's full plan from ``ontology`` (§12.1 entry point).

    The count is **derived** from the construction rule: the returned list has
    exactly :data:`EXPECTED_TOTAL` entries.  A caller that needs a prefix (the
    checkpoint/resume path asks only for the anchors still missing) slices the
    result — the plan itself is never shortened here, because a truncated plan
    would silently skip coordinates.

    Args:
        ontology: A validated v4 ontology.
        config: Generation configuration; only ``seed`` is read.  The turn count
            of each entry comes from the ontology, not from the config.

    Returns:
        The plan's :class:`AnchorSpec` objects, in plan order.

    Raises:
        SamplingError: If the ontology cannot produce the rule's coordinate set,
            or cannot supply a turn count for one of them.
    """
    coordinates = sample_coordinates(ontology, config.seed)
    return build_specs(coordinates, ontology=ontology)
