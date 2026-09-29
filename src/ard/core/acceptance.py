"""Acceptance readout for one ARD run: the structure readout.

Responsibility: assemble the acceptance report of a run — the **structure
readout**: what the plan covers, next to what this run's own ``N`` and round
decomposition expect.  Pure computation: pydantic only — this module touches no
file, no network, no endpoint, and no other ``ard`` module beyond the pure
``core`` modules it reads its 口径 from.

The embedding-side ruler (``q95`` / ``q50`` / ``q90`` / ``r_max`` /
``Extent(ε)``, the ε sensitivity band, the noise band and the target set) has
been removed: measuring a run against an embedder cost an endpoint, a key and a
vector store, and the structure readout answers the question this project
actually asks — *did the plan cover what the construction rule says it must?* —
with zero model calls and zero numpy.

Two rules are absolute here:

* every expected count is derived at call time from the run's ontology and its
  count ``N`` (`:func:`ard.core.sampling.coverage_units` /
  :func:`~ard.core.sampling.plan_rounds` / ``axis_values``) — no rule count is
  written down in this module, so an ontology change or a larger ``N`` cannot
  leave a stale expectation behind;
* an input that cannot be read (an unusable count, a plan entry that is not an
  ``axis -> value`` mapping) is refused with :class:`AcceptanceError` instead of
  yielding a meaningless number (§2.3 边界校验即防呆).

Two readings exist so that a count of *planned blocks* cannot be mistaken for a
count of *effective specifications* (WP-S14 audit, WP-S18):

* :data:`PROMPT_SIGNATURE_AXES` names the ``anchor_meta`` fields the
  generator-side prompt assembly actually reads.  ``prompt_signature_distinct``
  is defined on that tuple — grouping by the whole ``anchor_meta`` would split
  two records that issue the identical request and therefore **overstate** the
  effective diversity;
* ``effective_projection_distinct`` counts the distinct projections onto the
  restricted axes that reach the prompt, so a block count that includes axes no
  prompt consumer reads cannot be reported as "effective diversity".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from ard.core import axis_instruction, constraints, sampling
from ard.core.ontology import OntologyV4
from ard.core.types import JsonObjectSequence, StringPairs

#: Version of the acceptance report schema — bumped when a field changes meaning.
#: ``ard-acceptance-4`` removes the whole metric half (``metrics`` and its space
#: declaration): the report is now the structure readout plus its warnings, so a
#: reader of ``ard-acceptance-3`` must not look for ``metrics`` here.
#:
#: ``ard-acceptance-3`` (WP-5) replaced the v4 rule "every count equals the full
#: cycle's constant" with this run's own ``N`` and round decomposition, added
#: ``structure.coverage`` (coverage by coordinate, density by entry), and
#: **removed** ``structure.duplicate_coordinates``: under the v5 id rule a
#: coordinate is content, not identity, so the same coordinate sampled in a later
#: round is a new sample and is never reported as a duplicate.
REPORT_SCHEMA: Final[str] = "ard-acceptance-4"

#: The ``anchor_meta`` fields the generator-side prompt assembly **actually
#: reads**, in signature order.  Two coordinates that agree on all of them issue
#: the byte-identical generator-side request, so they are one cell for the noise
#: band and one prompt for the diversity readout.
#:
#: Evidence in the current tree — one entry per consumer, no inference:
#:
#: * ``language`` / ``knowledge_domain`` / ``capability`` / ``conversation_type``
#:   — ``ard.domain.text_anchor._build_user_prompt`` reads exactly these four from
#:   ``anchor_meta`` and turns them into the instruction (the image branch drops
#:   ``knowledge_domain``, which is why the four are one group), then assembles
#:   the system string in the same function.
#: * ``system_prompt_mode`` — chooses the wording file
#:   (``ard.backends.prompt_loader.build_system_prompt_prompt``), and the mode's
#:   template is filled from ``language`` / ``capability`` /
#:   ``knowledge_domain``
#:   (``ard.core.system_prompt._FIELD_SOURCES``).
#: * the six instruction axes (:data:`ard.core.axis_instruction.INSTRUCTION_AXES`,
#:   imported below, not re-listed) — ``ard.domain.text_anchor._build_user_prompt``
#:   turns every one of them into a requirement clause; the per-axis read is
#:   ``ard.backends.axis_instruction_loader.build_axis_requirements``.
#: * ``visual_domain`` — selects ``<image_dir>/<visual_domain>``
#:   (``ard.domain.image_store.domain_directory``), i.e. *which* image the request
#:   carries; for a text-only coordinate the axis is absent, which is itself the
#:   distinction from an image coordinate.
#:
#: Deliberately **not** in the tuple: ``modality`` / ``has_image`` /
#: ``image_count``.  They are plan bookkeeping (``modality`` selects the branch,
#: ``has_image`` / ``image_count`` are stamped by ``ard.core.quota.allocate_images`` /
#: ``ard.pipeline._assign_images_by_domain``); ``image_count`` is ``min(#user turns,
#: IMAGES_PER_ANCHOR)`` with ``IMAGES_PER_ANCHOR = 1``
#: (``ard.pipeline.IMAGES_PER_ANCHOR``), so it carries no information the tuple does
#: not already carry.  The contract test
#: ``tests/core/test_acceptance_prompt_signature.py`` renders both sides and
#: fails if any listed axis is a mascot or any unlisted field changes the render.
PROMPT_SIGNATURE_AXES: Final[tuple[str, ...]] = (
    "language",
    "knowledge_domain",
    "capability",
    "conversation_type",
    "system_prompt_mode",
    *axis_instruction.INSTRUCTION_AXES,
    "visual_domain",
)

#: The restricted axes whose value reaches the prompt: the intersection of
#: :data:`ard.core.constraints.RESTRICTED_AXES` with
#: :data:`PROMPT_SIGNATURE_AXES`.  It is derived, not written down, so an axis
#: that stops reaching the prompt leaves this projection on its own.
EFFECTIVE_PROJECTION_AXES: Final[tuple[str, ...]] = tuple(
    axis for axis in constraints.RESTRICTED_AXES if axis in PROMPT_SIGNATURE_AXES
)

#: The definition of the prompt signature, stated in every report.
PROMPT_SIGNATURE_DEFINITION: Final[str] = (
    "ordered tuple of the values of the axes listed here (a missing axis is None) "
    "over one plan entry's anchor_meta; two entries share a signature iff the "
    "generator-side request they fix is the same (input-generator system string, "
    "system-message generation request, spec turn count, and the image the "
    "coordinate carries)"
)

#: The definition of the effective restricted projection, stated in every report.
EFFECTIVE_PROJECTION_DEFINITION: Final[str] = (
    "ordered tuple of the values of the restricted axes (constraints.RESTRICTED_AXES) "
    "that reach the prompt; it is the number of distinct restricted specifications "
    "that can actually change what is generated, so it can never exceed the legal "
    "block count and must equal it when no legal block is inert"
)

#: What the two distinct counts are counted over, stated in every report.
DIVERSITY_COUNTED_OVER: Final[str] = (
    "plan entries, split by their 'modality' value (text_only / image); an entry "
    "of any other modality is counted in neither"
)

#: One restricted-axis value tuple: a sampled block's identity in the plan.
BlockKey: TypeAlias = tuple[Any, ...]

#: One structure check: its label, the measured count, the expected count, and
#: the comparison that must hold.  ``>=`` expresses the cycle rule's *guarantee*
#: (a lower bound); ``==`` expresses a count the rule fixes exactly.
CheckRelation: TypeAlias = Literal["==", ">="]
CountCheck: TypeAlias = tuple[str, int, int, CheckRelation]


class AcceptanceError(Exception):
    """An input to the acceptance readout cannot be measured as declared."""


class Conventions(BaseModel):
    """The two 口径 statements every report must carry (read from sampling).

    Attributes:
        sampling_module: module the constants were read from.
        multi_turn_default: ``sampling.MULTI_TURN_DEFAULT`` at report time.
        multi_turn_literal: the ontology literal that maps onto that default.
        turn_count_mapping: how a declared ``turns`` attribute becomes a spec
            turn count — ``2 * n - 1``, so every spec has an odd turn count.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sampling_module: str
    multi_turn_default: int
    multi_turn_literal: str
    turn_count_mapping: str


class ModalityDistinct(BaseModel):
    """One distinct-class count per modality.

    A ``ModalityDistinct`` is deliberately not a single integer: the plan's
    blocks, signatures and projections are already reported per modality
    (``plan_text_entries`` / ``plan_image_entries``,
    ``text_block_count`` / ``image_block_count``), and a single pooled count
    would hide a collapse that happens in only one of the two.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text_only: int
    image: int


class DiversityDeclaration(BaseModel):
    """The definition behind the two distinct counts — stated, not implied.

    A number called "distinct" is meaningless without the tuple it is distinct
    over, so the readout carries the axes, the join rule and the population it
    was counted on (``docs/algorithm.md`` §8).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    counted_over: str
    prompt_signature_axes: tuple[str, ...]
    prompt_signature_definition: str
    effective_projection_axes: tuple[str, ...]
    effective_projection_definition: str


class PlanCoverage(BaseModel):
    """How much of the sampling space one plan visits, and over how many rounds.

    Two 口径 live here and must not be confused:

    * **coverage** counts *coordinates*: ``min(distinct, U) / U``.  The same
      coordinate sampled in a later round is **not** credited a second time, so
      coverage saturates at ``1.0`` once every unit has been visited and a
      repetition can never push it above one;
    * **density** counts *entries*: ``plan_total / U``.  Every sampled entry
      counts, including a coordinate that recurs across rounds, so density keeps
      growing with ``N`` (``N = 2U`` ⇒ ``2.0``).

    Attributes:
        unit_total: ``U`` — how many coverage units one full round visits
            (text units then image units).
        text_unit_total: the text-only units of ``U``.
        image_unit_total: the image-modality units of ``U``.
        knowledge_leaf_total: ``K`` — how many ``knowledge_domain`` leaves the
            ontology declares.
        visual_leaf_total: ``V`` — how many ``visual_domain`` leaves it declares.
        plan_count: the run's count ``N`` as the caller declared it.
        distinct_coordinates: distinct coordinate identities in the plan.
        coverage: ``min(distinct_coordinates, unit_total) / unit_total``.
        density: ``plan_total / unit_total`` — the entry口径.
        full_rounds: complete cycles the plan holds, ``divmod(N, U)[0]``.
        last_round_size: entries in the final, partial cycle.
        rounds: how many cycles the plan touches —
            ``full_rounds + (1 if last_round_size else 0)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_total: int
    text_unit_total: int
    image_unit_total: int
    knowledge_leaf_total: int
    visual_leaf_total: int
    plan_count: int
    distinct_coordinates: int
    coverage: float
    density: float
    full_rounds: int
    last_round_size: int
    rounds: int


class StructureReadout(BaseModel):
    """What the plan covers, next to what this run's ``N`` and round rule expect.

    ``within_rule`` is the conjunction of the checks :func:`structure_checks`
    derives from *this* readout; it is a reading, not a gate: a plan assembled by
    a test double (or an ontology whose rule changed) is reported as inconsistent
    instead of aborting the run that already paid for its anchors.

    **Coordinate repetition is not a check.**  Under v5 a coordinate is content,
    not identity: sampling the same coordinate again in a later round is a new
    sample (at temperature 0.8 the same labels yield a different question), so no
    field here counts "duplicates" and no reading reports a repeat as wrong.

    Attributes:
        plan_total: number of coordinates in the plan.
        plan_text_entries: plan entries in the text-only modality.
        plan_image_entries: plan entries in the image modality.
        text_block_count: distinct legal restricted blocks covered by the text
            plan entries.
        image_block_count: distinct legal restricted blocks covered by the image
            plan entries.
        prompt_signature_distinct: distinct prompt signatures
            (:data:`PROMPT_SIGNATURE_AXES`), per modality — how many
            *different generator-side requests* the plan's entries fix, as
            opposed to how many entries it has.
        effective_projection_distinct: distinct projections onto
            :data:`EFFECTIVE_PROJECTION_AXES` (the restricted axes that reach the
            prompt), per modality.  A legal block whose extra restricted axes no
            prompt consumer reads collapses here; the count can never exceed
            ``text_block_count`` / ``image_block_count``.
        knowledge_domain_leaves: distinct ``knowledge_domain`` leaves covered.
        visual_domain_leaves: distinct ``visual_domain`` leaves covered.
        coverage: the ``N`` / ``U`` / rounds / coverage / density readout.
        expected_total: the run's count ``N`` (``U`` when the caller declares
            none) — the rule's plan holds exactly this many entries.
        expected_distinct_coordinates: ``min(N, U)`` — the cycle rule's coverage
            lower bound.  One round visits every unit once, so the first ``U``
            coordinates are distinct; only a cross-round recurrence (which the
            rule permits, see :mod:`ard.core.sampling`) can lower the count below
            this bound.  It is always compared with ``>=``, never with ``==``.
        expected_text_blocks / expected_image_blocks: the text / image unit
            totals when ``N >= U`` (a full round visits every unit), else
            ``None``.  Below one round the shuffle order — not the rule — sets
            the per-modality split, so the rule guarantees no block count.
        expected_knowledge_domains / expected_visual_domains: ``K`` / ``V`` when
            ``N >= U``, else ``None``.  A full round feeds every unit index
            through ``(index + cycle) % K`` (and the image unit indices through
            ``% V``), and ``U >= K``, so every leaf is visited; below ``U``
            shuffled indices can share a residue, so no leaf count is guaranteed
            (measured on today's ontology: ``N = K = 209`` reaches 136 of 209
            leaves).
        conventions: the ``MULTI_TURN_DEFAULT`` / turn-mapping statements.
        diversity: the definition of the two distinct counts above.
        within_rule: every check in :func:`structure_checks` holds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_total: int
    plan_text_entries: int
    plan_image_entries: int
    text_block_count: int
    image_block_count: int
    prompt_signature_distinct: ModalityDistinct
    effective_projection_distinct: ModalityDistinct
    knowledge_domain_leaves: int
    visual_domain_leaves: int
    coverage: PlanCoverage
    expected_total: int
    expected_distinct_coordinates: int
    expected_text_blocks: int | None
    expected_image_blocks: int | None
    expected_knowledge_domains: int | None
    expected_visual_domains: int | None
    conventions: Conventions
    diversity: DiversityDeclaration
    within_rule: bool


class AcceptanceReport(BaseModel):
    """The whole acceptance report of one run: the structure readout, warnings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_schema: str = REPORT_SCHEMA
    structure: StructureReadout
    warnings: list[str]


def conventions() -> Conventions:
    """Return the run's 口径 statements, read from :mod:`ard.core.sampling`.

    Read at call time (not import time) on purpose: a change to
    ``MULTI_TURN_DEFAULT`` must change the report, and a test can prove the
    number is not copied into this module by patching the constant.
    """
    multi_turn_default = sampling.MULTI_TURN_DEFAULT
    literal = sampling.MULTI_TURN_LITERAL
    return Conventions(
        sampling_module=sampling.__name__,
        multi_turn_default=multi_turn_default,
        multi_turn_literal=literal,
        turn_count_mapping=(
            "declared turns n -> 2*n-1 spec turns; the literal "
            f"{literal!r} -> 2*MULTI_TURN_DEFAULT-1 = {2 * multi_turn_default - 1}"
        ),
    )


def prompt_signature(meta: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the prompt signature of one coordinate: the tuple of
    :data:`PROMPT_SIGNATURE_AXES` values, in signature order.

    Two coordinates with the same signature issue the same generator-side
    request, so they are one cell for the noise band.  A missing axis becomes
    ``None`` rather than a default: ``None`` is "the coordinate does not carry
    the axis", which is not the same as carrying the axis' first value.

    Pure tuple arithmetic — this module declares *which* fields matter; the
    renderers remain the only place the prompts are built (§1.4 单一真相源).
    """
    return tuple(meta.get(axis) for axis in PROMPT_SIGNATURE_AXES)


def structure_readout(
    plan: JsonObjectSequence,
    *,
    ontology: OntologyV4,
    count: int | None = None,
) -> StructureReadout:
    """Summarise *plan*: plan coordinates in, coverage counts out.

    Every expectation is a function of the run's **own** sampling space and
    count, computed here from *ontology* and *count* — nothing is compared
    against a written-down full-cycle constant, so a smoke plan (8 entries) is
    not reported as a violation of a 1,826-entry rule.

    Args:
        plan: one ``axis -> value`` mapping per planned anchor (the
            ``AnchorSpec.anchor_meta`` of each entry, in plan order).
        ontology: the run's validated v4 ontology — the single source of ``U``
            (:func:`ard.core.sampling.coverage_units`), ``K`` and ``V``
            (``axis_values``).  Required: this module holds no fallback count
            (§1.4 单一真相源).
        count: the run's ``N``, i.e. the count the plan was requested with.
            ``None`` means one full round, ``U`` — the same convention as
            :func:`ard.core.sampling.sample_coordinates`.  A ``per_modality``
            smoke plan is **not** one full round: its caller passes
            ``count=len(plan)``.

    Returns:
        The structure readout, with every expectation derived from *ontology*
        and *count* at call time and ``within_rule`` stating whether the plan
        matches them.

    Raises:
        AcceptanceError: if ``count`` is unusable (not an integer ``>= 1``), or a
            plan entry is not an ``axis -> value`` mapping.
    """
    if count is None:
        count = sampling.unit_total(ontology)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise AcceptanceError(
            f"structure_readout needs the run's count N as an integer >= 1 "
            f"(received {count!r}); a resumed run passes the full plan's count and a "
            "per_modality smoke plan passes len(plan)"
        )

    restricted_axes = constraints.RESTRICTED_AXES
    effective_axes = EFFECTIVE_PROJECTION_AXES
    knowledge_axis, visual_axis = sampling.ROTATED_FREE_AXES
    text_modality = sampling.MODALITY_TEXT
    image_modality = sampling.MODALITY_IMAGE

    units = sampling.coverage_units(ontology)
    unit_total = len(units)
    text_units = sum(1 for unit in units if unit.modality == text_modality)
    image_units = unit_total - text_units
    knowledge_total = sampling.knowledge_leaf_count(ontology)
    visual_total = sampling.visual_leaf_count(ontology)
    full_rounds, last_round_size = sampling.plan_rounds(ontology, count)

    text_entries = 0
    image_entries = 0
    text_blocks: set[BlockKey] = set()
    image_blocks: set[BlockKey] = set()
    signatures: dict[str, set[tuple[Any, ...]]] = {text_modality: set(), image_modality: set()}
    projections: dict[str, set[BlockKey]] = {text_modality: set(), image_modality: set()}
    knowledge_leaves: set[str] = set()
    visual_leaves: set[str] = set()
    identities: set[StringPairs] = set()

    for index, meta in enumerate(plan):
        if not isinstance(meta, Mapping):
            raise AcceptanceError(
                f"plan entry {index} is {type(meta).__name__}, not an axis -> value mapping"
            )
        identities.add(tuple(sorted((str(key), str(value)) for key, value in meta.items())))
        modality = meta.get("modality")
        if modality in signatures:
            signatures[modality].add(prompt_signature(meta))
            projections[modality].add(tuple(meta.get(axis) for axis in effective_axes))
            if modality == text_modality:
                text_entries += 1
                text_blocks.add(tuple(meta.get(axis) for axis in restricted_axes))
            else:
                image_entries += 1
                image_blocks.add(tuple(meta.get(axis) for axis in restricted_axes))
        knowledge = meta.get(knowledge_axis)
        if knowledge is not None:
            knowledge_leaves.add(str(knowledge))
        visual = meta.get(visual_axis)
        if visual is not None:
            visual_leaves.add(str(visual))

    plan_total = len(plan)
    # A full round visits every unit, so it reaches every text/image block and
    # every leaf.  Below one round the shuffle order decides what is reached;
    # there the rule guarantees nothing, and ``None`` says so instead of
    # inventing a bound a legitimate plan could miss.
    reaches_full_round = count >= unit_total
    signature_distinct = ModalityDistinct(
        text_only=len(signatures[text_modality]),
        image=len(signatures[image_modality]),
    )
    projection_distinct = ModalityDistinct(
        text_only=len(projections[text_modality]),
        image=len(projections[image_modality]),
    )
    readout = StructureReadout(
        plan_total=plan_total,
        plan_text_entries=text_entries,
        plan_image_entries=image_entries,
        text_block_count=len(text_blocks),
        image_block_count=len(image_blocks),
        prompt_signature_distinct=signature_distinct,
        effective_projection_distinct=projection_distinct,
        knowledge_domain_leaves=len(knowledge_leaves),
        visual_domain_leaves=len(visual_leaves),
        coverage=PlanCoverage(
            unit_total=unit_total,
            text_unit_total=text_units,
            image_unit_total=image_units,
            knowledge_leaf_total=knowledge_total,
            visual_leaf_total=visual_total,
            plan_count=count,
            distinct_coordinates=len(identities),
            coverage=min(len(identities), unit_total) / unit_total,
            density=plan_total / unit_total,
            full_rounds=full_rounds,
            last_round_size=last_round_size,
            rounds=full_rounds + (1 if last_round_size else 0),
        ),
        expected_total=count,
        expected_distinct_coordinates=min(count, unit_total),
        expected_text_blocks=text_units if reaches_full_round else None,
        expected_image_blocks=image_units if reaches_full_round else None,
        expected_knowledge_domains=knowledge_total if reaches_full_round else None,
        expected_visual_domains=visual_total if reaches_full_round else None,
        conventions=conventions(),
        diversity=DiversityDeclaration(
            counted_over=DIVERSITY_COUNTED_OVER,
            prompt_signature_axes=PROMPT_SIGNATURE_AXES,
            prompt_signature_definition=PROMPT_SIGNATURE_DEFINITION,
            effective_projection_axes=EFFECTIVE_PROJECTION_AXES,
            effective_projection_definition=EFFECTIVE_PROJECTION_DEFINITION,
        ),
        within_rule=True,
    )
    # One source for the reading and the warning: ``within_rule`` is exactly
    # "structure_mismatch finds nothing" (§1.4).
    return readout.model_copy(update={"within_rule": structure_mismatch(readout) is None})


def structure_checks(structure: StructureReadout) -> list[CountCheck]:
    """Return every count comparison the rule makes, in report order.

    Each entry is ``(label, measured, expected, relation)``.  Both
    :func:`structure_mismatch` and :func:`render_markdown` consume this one list,
    so the published table and the WARNING cannot disagree (§1.4).

    A check whose expected value is ``None`` is omitted: the rule guarantees
    nothing about that count at this ``N``, and an omitted check can never be
    reported as a mismatch.  The two projection checks are the inert-axis guard —
    every restricted axis must reach the prompt, so the projection count must
    equal the block count (WP-S18).
    """
    checks: list[CountCheck] = [
        ("plan entries", structure.plan_total, structure.expected_total, "=="),
        (
            "entries with a known modality",
            structure.plan_text_entries + structure.plan_image_entries,
            structure.plan_total,
            "==",
        ),
        (
            "distinct coordinates",
            structure.coverage.distinct_coordinates,
            structure.expected_distinct_coordinates,
            ">=",
        ),
        (
            "effective restricted projections (text_only)",
            structure.effective_projection_distinct.text_only,
            structure.text_block_count,
            "==",
        ),
        (
            "effective restricted projections (image)",
            structure.effective_projection_distinct.image,
            structure.image_block_count,
            "==",
        ),
    ]
    for label, measured, expected in (
        ("legal text blocks", structure.text_block_count, structure.expected_text_blocks),
        ("legal image blocks", structure.image_block_count, structure.expected_image_blocks),
        (
            "knowledge_domain leaves",
            structure.knowledge_domain_leaves,
            structure.expected_knowledge_domains,
        ),
        (
            "visual_domain leaves",
            structure.visual_domain_leaves,
            structure.expected_visual_domains,
        ),
    ):
        if expected is not None:
            checks.append((label, measured, expected, ">="))
    return checks


def structure_mismatch(structure: StructureReadout) -> str | None:
    """Return one line naming every structure count that disagrees, or ``None``.

    The acceptance phase publishes this as a WARNING instead of aborting a run
    whose anchors are already on disk: "the plan is not the rule's plan" is a
    reading about the artifact, and the reader must be able to see it.

    A ``>=`` check is the cycle rule's guarantee, so a legitimate cross-round
    coordinate recurrence — which can lower the distinct-coordinate count below
    ``N`` — is **not** a mismatch.  Repetition is never named here.
    """
    failed: list[str] = []
    for name, measured, expected, relation in structure_checks(structure):
        if relation == ">=":
            if measured < expected:
                failed.append(f"{name}: {measured} (expected >= {expected})")
        elif measured != expected:
            failed.append(f"{name}: {measured} (expected {expected})")
    if not failed:
        return None
    return "plan does not match the construction rule — " + "; ".join(failed)


def _status(measured: int, expected: int, relation: CheckRelation) -> str:
    """``OK`` / ``MISMATCH`` marker for one counted-vs-expected row."""
    ok = measured >= expected if relation == ">=" else measured == expected
    return "OK" if ok else "MISMATCH"


def render_markdown(report: AcceptanceReport) -> str:
    """Render *report* as the human-readable ``coverage.md`` companion."""
    structure = report.structure
    coverage = structure.coverage
    signature = structure.prompt_signature_distinct
    projection = structure.effective_projection_distinct
    lines: list[str] = [
        "# ARD acceptance readout",
        "",
        f"- report schema: `{report.report_schema}`",
        f"- structure readout within the construction rule: "
        f"`{'yes' if structure.within_rule else 'no'}`",
        "",
        "## Structure readout (zero model calls)",
        "",
        "### Coverage and rounds",
        "",
        "| reading | value |",
        "|---|---:|",
        f"| coverage unit total U | {coverage.unit_total} |",
        f"| U = text units + image units | {coverage.text_unit_total} + "
        f"{coverage.image_unit_total} |",
        f"| planned count N | {coverage.plan_count} |",
        f"| plan entries | {structure.plan_total} |",
        f"| full rounds | {coverage.full_rounds} |",
        f"| coordinates in the final round | {coverage.last_round_size} |",
        f"| rounds the plan touches | {coverage.rounds} |",
        f"| distinct coordinates | {coverage.distinct_coordinates} |",
        f"| **coverage** = min(distinct coordinates, U) / U | **{coverage.coverage:.6f}** |",
        f"| **density** = plan entries / U | **{coverage.density:.6f}** |",
        "",
        "- reading note: a coordinate sampled again in a later round counts "
        "**once** for coverage (a repeat never pushes it above 1.0) and **every "
        "entry** counts for density. Coordinate repetition is a legitimate "
        "resample under the v5 id rule, not a defect, and is never reported as "
        "one.",
        "",
        "### Rule checks",
        "",
        "| reading | measured | expected | status |",
        "|---|---:|---:|---|",
    ]
    for name, measured, expected, relation in structure_checks(structure):
        wanted = f">= {expected}" if relation == ">=" else f"{expected}"
        lines.append(
            f"| {name} | {measured} | {wanted} | {_status(measured, expected, relation)} |"
        )
    lines += [
        "",
        "### Reported readings",
        "",
        "| reading | value |",
        "|---|---:|",
        f"| text-only plan entries | {structure.plan_text_entries} |",
        f"| image plan entries | {structure.plan_image_entries} |",
        f"| prompt signature distinct (text_only) | {signature.text_only} |",
        f"| prompt signature distinct (image) | {signature.image} |",
        f"| effective restricted projection distinct (text_only) | {projection.text_only} |",
        f"| effective restricted projection distinct (image) | {projection.image} |",
        f"| knowledge_domain leaves | {structure.knowledge_domain_leaves} of "
        f"{coverage.knowledge_leaf_total} |",
        f"| visual_domain leaves | {structure.visual_domain_leaves} of "
        f"{coverage.visual_leaf_total} |",
        "",
    ]
    if structure.expected_text_blocks is None:
        lines += [
            "- reading note: below one full round the shuffle order — not the "
            "rule — sets the per-modality split and the leaves reached, so no "
            "per-modality block count and no leaf count is expected here; those "
            "checks are omitted rather than guessed.",
            "",
        ]
    lines += [
        "## Conventions",
        "",
        f"- constants read from `{structure.conventions.sampling_module}` at report time",
        f"- `MULTI_TURN_DEFAULT = {structure.conventions.multi_turn_default}`",
        f"- turn-count mapping: {structure.conventions.turn_count_mapping}",
        "",
        "## Diversity declaration (what the two distinct counts mean)",
        "",
        f"- counted over: {structure.diversity.counted_over}",
        f"- prompt signature = {structure.diversity.prompt_signature_definition}",
        f"  axes: `{'`, `'.join(structure.diversity.prompt_signature_axes)}`",
        f"- effective restricted projection = "
        f"{structure.diversity.effective_projection_definition}",
        f"  axes: `{'`, `'.join(structure.diversity.effective_projection_axes)}`",
        "- reading note: a *legal block count* is a count of planned cells; only "
        "these two counts say how many of them are different specifications. "
        "`MISMATCH` above means a planned cell does **not** produce its own "
        "prompt, so the block count must not be read as effective diversity "
        "(`docs/algorithm.md` §8).",
        "",
    ]
    lines += ["## Warnings", ""]
    if report.warnings:
        lines += [f"- {warning}" for warning in report.warnings]
    else:
        lines += ["- none"]
    lines.append("")
    return "\n".join(lines)
