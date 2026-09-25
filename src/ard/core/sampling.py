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
from ard.core.ontology import OntologyV4
from ard.core.quota import compute_turn_distribution
from ard.core.types import AnchorGenerationConfig, AnchorSpec, TurnSpec

#: The anchor-id dimensions, in hash order.  The set is the legacy one (v3.0.0
#: ids), kept deliberately: the bank's uniqueness gate and the checkpoint/resume
#: path both key on these ids, and changing the hash input would make every
#: already-written record look like a new anchor.  On the v4 plan the five
#: dimensions are still injective — two coordinates that differ only in a
#: non-hashed axis cannot occur, because each legal restricted block contributes
#: exactly one sample (see the contract tests).
#:
#: WP-S2b note: when :mod:`ard.core.sampler` is deleted this function (and the
#: tuple below) must survive *here*; the legacy tests that import it from
#: ``ard.core.sampler`` move to this module in the same step.
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


def turn_counts_for(plan_size: int, max_turns: int, rng: random.Random) -> list[int]:
    """Distribute ``plan_size`` anchors over the odd turn counts up to ``max_turns``.

    Only odd turn counts are valid — a conversation must end with a user turn —
    so bucket ``i`` holds the anchors with ``2 * i + 1`` turns.  The
    distribution is as even as possible and the remainder is spread by ``rng``;
    with the shipped ``max_turns = 1`` every anchor is single-turn.

    Args:
        plan_size: Number of anchors to distribute.
        max_turns: Largest allowed turn count (1..10).
        rng: Seeded generator; consumed here so the plan's later consumers keep
            their positions on the same stream.

    Returns:
        A list of length ``max_turns`` whose index ``n - 1`` holds the number of
        anchors with ``n`` turns.

    Raises:
        SamplingError: If ``max_turns`` is not a positive turn count.
    """
    if max_turns < 1:
        raise SamplingError(f"max_turns must be >= 1, received {max_turns}")
    num_odd_buckets = (max_turns + 1) // 2
    raw_turn_counts = compute_turn_distribution(plan_size, num_odd_buckets, rng)
    turn_counts = [0] * max_turns
    for index, count in enumerate(raw_turn_counts):
        turn_counts[2 * index] = count
    return turn_counts


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
    max_turns: int,
    rng: random.Random,
) -> list[AnchorSpec]:
    """Turn a coordinate plan into the run's :class:`AnchorSpec` list.

    Args:
        coordinates: The plan from :func:`sample_coordinates`, in plan order.
        max_turns: Largest allowed turn count.
        rng: Seeded generator used only for the turn distribution.

    Returns:
        One spec per coordinate, in the same order.

    Raises:
        SamplingError: If the distribution does not cover every coordinate.
    """
    counts = turn_counts_for(len(coordinates), max_turns, rng)
    specs: list[AnchorSpec] = []
    coordinate_index = 0
    for turn_count, count in enumerate(counts, start=1):
        for _ in range(count):
            if coordinate_index >= len(coordinates):
                break
            specs.append(_build_spec(coordinates[coordinate_index], turn_count))
            coordinate_index += 1
    if coordinate_index != len(coordinates):
        raise SamplingError(
            f"turn distribution covered {coordinate_index} of {len(coordinates)} "
            f"coordinates (counts={counts}, max_turns={max_turns})"
        )
    return specs


def sample_anchors(
    ontology: OntologyV4,
    config: AnchorGenerationConfig,
    rng: random.Random,
) -> list[AnchorSpec]:
    """Sample the run's full plan from ``ontology`` (§12.1 entry point).

    The count is **derived** from the construction rule: the returned list has
    exactly :data:`EXPECTED_TOTAL` entries and ``config.target_count`` is
    deliberately ignored.  A caller that needs a prefix (the checkpoint/resume
    path asks only for the anchors still missing) slices the result — the plan
    itself is never shortened here, because a truncated plan would silently skip
    coordinates.

    Args:
        ontology: A validated v4 ontology.
        config: Generation configuration; only ``max_turns`` and
            ``max_turns_with_image`` are read.  ``seed`` is unused because the
            caller passes the already-seeded ``rng``, which keeps any work that
            runs after this call (image allocation) on the same stream.
        rng: The single source of randomness for the plan.

    Returns:
        The plan's :class:`AnchorSpec` objects, in plan order.

    Raises:
        SamplingError: If the ontology cannot produce the rule's coordinate set.
    """
    coordinates = sample_coordinates(ontology, config.seed)
    return build_specs(coordinates, max_turns=config.max_turns, rng=rng)
