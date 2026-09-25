"""Acceptance readouts for one ARD run: structure counts and metric readouts.

Responsibility: assemble the acceptance report of a run — the **structure
readout** (what the plan covers, versus what the construction rule expects) and
the **metric readout** (``q95`` / ``q50`` / ``q90`` / ``r_max`` / ``Extent(ε)``
and the ε sensitivity band, measured with :mod:`ard.core.coverage`).  Pure
computation: numpy + pydantic only — this module touches no file, no network, no
endpoint and no other ``ard`` module beyond the two pure ``core`` modules it
reads its口径 from.

Two rules are absolute here:

* every expected count and every convention is read from
  :mod:`ard.core.sampling` at call time — no rule count is written down in this
  module, so an ontology change cannot leave a stale expectation behind;
* an input that cannot be measured (an anchor record without a final user turn,
  a run whose bank is empty, a target set that is too small to have an intrinsic
  scale) is refused with :class:`AcceptanceError` instead of yielding a
  meaningless number (§2.3 边界校验即防呆).

The metric readout never invents its own maths: distances, quantiles, Extent and
the ε band all come from :mod:`ard.core.coverage`.  What this module adds is the
**space declaration** (which field of which artifact was embedded, ``|A|``,
``|T|``, and the embedder identity — model and dimension only, never an endpoint
or a key), the noise-band assembly, and the report rendering.

The declared anchor field is the final user turn's **text parts only**
(:data:`ANCHOR_TEXT_FIELD`): an image-modality anchor participates through the
text of its request, and its image pixels never enter the metric space.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict

from ard.core import constraints, sampling
from ard.core import coverage as ruler
from ard.core.types import JsonObjectSequence, StringPairs

#: Version of the acceptance report schema — bumped when a field changes meaning.
REPORT_SCHEMA: Final[str] = "ard-acceptance-1"

#: Which artifact field the anchor side embeds, declared in every report.
#: The ``(text parts only)`` qualifier is load-bearing, not decoration: an
#: image-modality anchor's final user turn is a multimodal part list, and this
#: ruler embeds **only its ``text`` parts**.  Image pixels never enter the
#: metric space, so the declaration must say so wherever the space is read
#: (``docs/measurement.md`` §5).
ANCHOR_TEXT_FIELD: Final[str] = "messages[last].content(text parts only)"

#: The multimodal part type whose ``text`` field is the embeddable text.
#: Parts of any other type (``image``, ``image_url``, …) contribute no text.
TEXT_PART_TYPE: Final[str] = "text"

#: How the text parts of one turn are joined when it carries more than one.
#: The writer emits a single text part, so this is a boundary default rather
#: than a live behaviour; it is named so it cannot drift silently.
TEXT_PART_SEPARATOR: Final[str] = "\n"

#: Which field of a target-set entry this project embeds.
TARGET_TEXT_FIELD: Final[str] = "text"

#: Statement written whenever no repeat-generation data is available.
NOISE_UNAVAILABLE_REASON: Final[str] = (
    "noise band unavailable (no repeated generation data provided)"
)

#: One restricted-axis value tuple: a sampled block's identity in the plan.
BlockKey: TypeAlias = tuple[Any, ...]

#: Record indices of anchors that share one coordinate.
IndexGroup: TypeAlias = list[int]

#: One structure check: its label, the measured count, the expected count.
CountCheck: TypeAlias = tuple[str, int, int]

_STRICT = ConfigDict(extra="forbid", frozen=True)


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

    model_config = _STRICT

    sampling_module: str
    multi_turn_default: int
    multi_turn_literal: str
    turn_count_mapping: str


class StructureReadout(BaseModel):
    """What the plan covers, next to what the construction rule expects.

    ``within_rule`` is the conjunction of every comparison below; it is a
    reading, not a gate: a plan assembled by a test double (or an ontology whose
    rule changed) is reported as inconsistent instead of aborting the run that
    already paid for its anchors.

    Attributes:
        plan_total: number of coordinates in the plan.
        plan_text_entries: plan entries in the text-only modality.
        plan_image_entries: plan entries in the image modality.
        text_block_count: distinct legal restricted blocks covered by the text
            plan entries.
        image_block_count: distinct legal restricted blocks covered by the image
            plan entries.
        knowledge_domain_leaves: distinct ``knowledge_domain`` leaves covered.
        visual_domain_leaves: distinct ``visual_domain`` leaves covered.
        duplicate_coordinates: plan entries sharing another entry's full
            coordinate identity.
        expected_*: the rule's own counts, read from :mod:`ard.core.sampling`.
        conventions: the ``MULTI_TURN_DEFAULT`` / turn-mapping statements.
        within_rule: every measured count equals its expected count.
    """

    model_config = _STRICT

    plan_total: int
    plan_text_entries: int
    plan_image_entries: int
    text_block_count: int
    image_block_count: int
    knowledge_domain_leaves: int
    visual_domain_leaves: int
    duplicate_coordinates: int
    expected_total: int
    expected_text_blocks: int
    expected_image_blocks: int
    expected_knowledge_domains: int
    expected_visual_domains: int
    conventions: Conventions
    within_rule: bool


class EmbedderIdentity(BaseModel):
    """Who produced the vectors — model and dimension, never a key or endpoint."""

    model_config = _STRICT

    model: str
    dimension: int
    normalize: bool


class SpaceDeclaration(BaseModel):
    """The space the metric readout was measured in — stated, not implied.

    Attributes:
        anchors_source: where the anchor vectors' texts came from.
        anchor_field: which field of that artifact was embedded —
            ``ANCHOR_TEXT_FIELD``, which for a multimodal turn names the text
            parts only, so the declaration cannot be read as "the image too".
        targets_source: path of the target-set file.
        target_field: which field of an entry was embedded.
        n_anchor: ``|A|``.
        n_target: ``|T|``.
        embedder: model + dimension + normalisation flag.
        distance: the distance definition (``1 - cos``).
        quantile_method: the quantile definition used (``linear`` = type-7).
        epsilon: the coverage radius the ``Extent`` was measured at.
        epsilon_source: where ``epsilon`` came from — the target-set header, or
            the target set's own intrinsic scale.
    """

    model_config = _STRICT

    anchors_source: str
    anchor_field: str
    targets_source: str
    target_field: str
    n_anchor: int
    n_target: int
    embedder: EmbedderIdentity
    distance: str
    quantile_method: str
    epsilon: float
    epsilon_source: str


class NoiseSection(BaseModel):
    """The same-cell repeat-generation noise band, or why it is missing.

    Attributes:
        available: whether a band could be measured.
        reason: the explicit unavailability statement when ``available`` is
            ``False`` — the band is never silently omitted (§3.2).
        band: ``[q50, max]`` of the repeat-generation pair distances.
        n_repeat_groups: coordinates that appeared more than once.
        n_pairs: repeat-generation pairs measured.
    """

    model_config = _STRICT

    available: bool
    reason: str | None
    band: ruler.NoiseBand | None
    n_repeat_groups: int
    n_pairs: int


class MetricReadout(BaseModel):
    """One anchor set measured against one target set, with its space declared."""

    model_config = _STRICT

    space: SpaceDeclaration
    quantiles: ruler.DistanceQuantiles
    extent: float
    epsilon_band: ruler.EpsilonSensitivity
    noise: NoiseSection


class AcceptanceReport(BaseModel):
    """The whole acceptance report of one run: structure, metrics, warnings."""

    model_config = _STRICT

    report_schema: str = REPORT_SCHEMA
    structure: StructureReadout
    metrics: MetricReadout | None
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


def structure_readout(plan: JsonObjectSequence) -> StructureReadout:
    """Summarise *plan*: plan coordinates in, coverage counts out.

    Args:
        plan: one ``axis -> value`` mapping per planned anchor (the
            ``AnchorSpec.anchor_meta`` of each entry, in plan order).

    Returns:
        The structure readout, with every expected count read from
        :mod:`ard.core.sampling` at call time and ``within_rule`` stating
        whether the plan matches them.
    """
    restricted_axes = constraints.RESTRICTED_AXES
    knowledge_axis, visual_axis = sampling.ROTATED_FREE_AXES
    text_modality = sampling.MODALITY_TEXT
    image_modality = sampling.MODALITY_IMAGE

    text_entries = 0
    image_entries = 0
    text_blocks: set[BlockKey] = set()
    image_blocks: set[BlockKey] = set()
    knowledge_leaves: set[str] = set()
    visual_leaves: set[str] = set()
    identities: set[StringPairs] = set()

    for meta in plan:
        identities.add(tuple(sorted((str(key), str(value)) for key, value in meta.items())))
        modality = meta.get("modality")
        if modality == text_modality:
            text_entries += 1
            text_blocks.add(tuple(meta.get(axis) for axis in restricted_axes))
        elif modality == image_modality:
            image_entries += 1
            image_blocks.add(tuple(meta.get(axis) for axis in restricted_axes))
        knowledge = meta.get(knowledge_axis)
        if knowledge is not None:
            knowledge_leaves.add(str(knowledge))
        visual = meta.get(visual_axis)
        if visual is not None:
            visual_leaves.add(str(visual))

    plan_total = len(plan)
    duplicates = plan_total - len(identities)
    expected_total = sampling.EXPECTED_TOTAL
    expected_text = sampling.EXPECTED_TEXT_BLOCKS
    expected_image = sampling.EXPECTED_IMAGE_BLOCKS
    expected_knowledge = sampling.EXPECTED_KNOWLEDGE_DOMAINS
    expected_visual = sampling.EXPECTED_VISUAL_DOMAINS
    within_rule = (
        plan_total == expected_total
        and text_entries == expected_text
        and image_entries == expected_image
        and len(text_blocks) == expected_text
        and len(image_blocks) == expected_image
        and len(knowledge_leaves) == expected_knowledge
        and len(visual_leaves) == expected_visual
        and duplicates == 0
    )
    return StructureReadout(
        plan_total=plan_total,
        plan_text_entries=text_entries,
        plan_image_entries=image_entries,
        text_block_count=len(text_blocks),
        image_block_count=len(image_blocks),
        knowledge_domain_leaves=len(knowledge_leaves),
        visual_domain_leaves=len(visual_leaves),
        duplicate_coordinates=duplicates,
        expected_total=expected_total,
        expected_text_blocks=expected_text,
        expected_image_blocks=expected_image,
        expected_knowledge_domains=expected_knowledge,
        expected_visual_domains=expected_visual,
        conventions=conventions(),
        within_rule=within_rule,
    )


def structure_mismatch(structure: StructureReadout) -> str | None:
    """Return one line naming every structure count that disagrees, or ``None``.

    The acceptance phase publishes this as a WARNING instead of aborting a run
    whose anchors are already on disk: "the plan is not the rule's plan" is a
    reading about the artifact, and the reader must be able to see it.
    """
    checks: tuple[CountCheck, ...] = (
        ("plan entries", structure.plan_total, structure.expected_total),
        ("text-only plan entries", structure.plan_text_entries, structure.expected_text_blocks),
        ("image plan entries", structure.plan_image_entries, structure.expected_image_blocks),
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
        ("duplicate coordinates", structure.duplicate_coordinates, 0),
    )
    mismatches = [
        f"{name}: {measured} (expected {expected})"
        for name, measured, expected in checks
        if measured != expected
    ]
    if not mismatches:
        return None
    return "plan does not match the construction rule — " + "; ".join(mismatches)


def user_turn_text(content: object, where: str) -> str:
    """Return the embeddable text of one final user turn's ``content``.

    The writer emits two shapes and this ruler accepts both:

    * a plain ``str`` — every text-only anchor, byte-identical to before;
    * a multimodal part list — the image-modality anchors.  Only the parts
      whose ``type`` is :data:`TEXT_PART_TYPE` and whose ``text`` is a
      non-blank string contribute; parts of any other type (``image``,
      ``image_url``, …) are ignored, because **this ruler embeds text only and
      image pixels never enter the metric space**.  The contributing parts are
      joined with :data:`TEXT_PART_SEPARATOR` in list order.

    A multimodal turn with no usable text part is refused, naming the record and
    what it did carry: silently skipping it would publish a ``q95`` over a
    smaller anchor set than the run produced, and substituting an empty string
    would fabricate a point in the space.

    Args:
        content: the ``content`` field of the final user-role message.
        where: a human-readable record label used in every error message.

    Returns:
        The embeddable text, stripped of no characters (the raw string wins as
        written; only the emptiness test looks at whitespace).

    Raises:
        AcceptanceError: if *content* is missing/blank, is neither a string nor
            a part list, or is a part list with no non-blank text part.
    """
    if content is None:
        raise AcceptanceError(f"{where}: the final user turn has blank content")
    if isinstance(content, str):
        if not content.strip():
            raise AcceptanceError(f"{where}: the final user turn has blank content")
        return content
    if not isinstance(content, list):
        raise AcceptanceError(
            f"{where}: the final user turn's content must be a string or a multimodal part "
            f"list, got {type(content).__name__}"
        )
    texts = [
        part["text"]
        for part in content
        if isinstance(part, Mapping)
        and part.get("type") == TEXT_PART_TYPE
        and isinstance(part.get("text"), str)
        and part["text"].strip()
    ]
    if not texts:
        carried = sorted({str(part.get("type")) for part in content if isinstance(part, Mapping)})
        raise AcceptanceError(
            f"{where}: the final user turn carries no text part "
            f"({len(content)} part(s), types {carried}) — the metric space embeds the text "
            "parts only, so this anchor has nothing to measure and is not skipped silently"
        )
    return TEXT_PART_SEPARATOR.join(texts)


def anchor_texts(records: JsonObjectSequence) -> list[str]:
    """Return the embedded text of every bank record, in file order.

    The anchor side is the record's **final user-role turn**: that is the
    question the anchor actually poses, and it is the only text a generated
    record is guaranteed to carry.  (The plan itself holds coordinates only, so
    it cannot supply this — see the space declaration in every report.)

    An image-modality anchor's final user turn is a multimodal part list; its
    text parts are extracted by :func:`user_turn_text`, so image anchors take
    part in the readout **as text** while their pixels stay out of the metric
    space.

    Raises:
        AcceptanceError: if the bank is empty, a record has no ``messages``
            list, its last message is not a user turn, or its content yields no
            text (blank string, or a part list with no text part).
    """
    if not records:
        raise AcceptanceError(
            "anchor bank holds no records: a nearest-anchor distance needs at least one anchor text"
        )
    texts: list[str] = []
    for index, record in enumerate(records):
        anchor_id = record.get("id")
        messages = record.get("messages")
        where = f"anchor record {index} (id={anchor_id!r})"
        if not isinstance(messages, list) or not messages:
            raise AcceptanceError(f"{where} has no non-empty 'messages' list")
        last = messages[-1]
        if not isinstance(last, Mapping) or last.get("role") != "user":
            raise AcceptanceError(
                f"{where}: the last message is not a user turn "
                f"(role={last.get('role') if isinstance(last, Mapping) else type(last).__name__!r})"
            )
        texts.append(user_turn_text(last.get("content"), where))
    return texts


def repeat_groups(coordinates: JsonObjectSequence) -> list[IndexGroup]:
    """Group record indices that carry the **same** full coordinate.

    The v4 plan has no repeated coordinate, so this list is normally empty; it
    becomes non-empty only when the bank holds repeated generations of one cell,
    which is exactly what the noise band needs.

    Raises:
        AcceptanceError: if a record carries no ``anchor_meta`` mapping.
    """
    groups: dict[StringPairs, IndexGroup] = {}
    for index, meta in enumerate(coordinates):
        if not isinstance(meta, Mapping) or not meta:
            raise AcceptanceError(
                f"anchor record {index} carries no 'anchor_meta' coordinate mapping, so "
                "repeat-generation groups cannot be formed"
            )
        key = tuple(sorted((str(axis), str(value)) for axis, value in meta.items()))
        groups.setdefault(key, []).append(index)
    return [indices for indices in groups.values() if len(indices) >= 2]


def intrinsic_epsilon(targets: ruler.VectorSet) -> float:
    """Return the target set's own scale: median nearest-neighbour distance.

    Used only when the target-set file does not declare an ``epsilon``.  The
    scale is computed from the target points themselves (no convention value is
    hard-coded anywhere), and the report names this source explicitly.

    Raises:
        AcceptanceError: if the target set has fewer than two rows, so no
            nearest neighbour exists.
    """
    vectors = targets.vectors
    if vectors.shape[0] < 2:
        raise AcceptanceError(
            f"target set {targets.name!r} has {vectors.shape[0]} row(s): at least 2 are "
            "needed to calibrate epsilon from the target set's own scale — declare "
            "'epsilon' in the target-set header instead"
        )
    distances = 1.0 - vectors @ vectors.T
    np.fill_diagonal(distances, np.inf)
    nearest = np.clip(distances.min(axis=1), 0.0, ruler.DISTANCE_UPPER_BOUND)
    return float(np.percentile(nearest, ruler.QUANTILE_LEVELS[0], method=ruler.PERCENTILE_METHOD))


def noise_section(anchors: ruler.VectorSet, groups: Sequence[IndexGroup]) -> NoiseSection:
    """Assemble the noise band from repeat-generation groups, or state its absence.

    Args:
        anchors: the embedded anchor set, indexed like the bank records.
        groups: index groups of records sharing one coordinate.

    Returns:
        An available band when at least one coordinate has repeats, otherwise an
        explicit :data:`NOISE_UNAVAILABLE_REASON` statement.
    """
    if not groups:
        return NoiseSection(
            available=False,
            reason=NOISE_UNAVAILABLE_REASON,
            band=None,
            n_repeat_groups=0,
            n_pairs=0,
        )
    samples = [
        ruler.pairwise_distances(
            ruler.VectorSet(
                name=f"repeat-group-{position}",
                vectors=anchors.vectors[list(indices)],
            )
        )
        for position, indices in enumerate(groups)
    ]
    values = np.concatenate([sample.values for sample in samples])
    combined = ruler.DistanceSample(
        label="same-cell repeat generations",
        values=values,
        unit_id="repeat-group",
    )
    return NoiseSection(
        available=True,
        reason=None,
        band=ruler.noise_band(combined),
        n_repeat_groups=len(groups),
        n_pairs=int(values.shape[0]),
    )


def metric_readout(
    anchors: ruler.VectorSet,
    targets: ruler.VectorSet,
    *,
    epsilon: float,
    epsilon_source: str,
    anchors_source: str,
    targets_source: str,
    embedder: EmbedderIdentity,
    noise: NoiseSection,
) -> MetricReadout:
    """Measure *anchors* against *targets* and declare the space it happened in.

    The statistics come from :func:`ard.core.coverage.acceptance_readout`; this
    function only attaches the space declaration and the noise section.
    """
    measured = ruler.acceptance_readout(targets, anchors, epsilon=epsilon, label="ard-run")
    return MetricReadout(
        space=SpaceDeclaration(
            anchors_source=anchors_source,
            anchor_field=ANCHOR_TEXT_FIELD,
            targets_source=targets_source,
            target_field=TARGET_TEXT_FIELD,
            n_anchor=anchors.vectors.shape[0],
            n_target=targets.vectors.shape[0],
            embedder=embedder,
            distance="1 - cos",
            quantile_method=ruler.PERCENTILE_METHOD,
            epsilon=epsilon,
            epsilon_source=epsilon_source,
        ),
        quantiles=measured.quantiles,
        extent=measured.extent,
        epsilon_band=measured.epsilon_band,
        noise=noise,
    )


def _status(measured: int, expected: int) -> str:
    """``OK`` / ``MISMATCH`` marker for one counted-vs-expected row."""
    return "OK" if measured == expected else "MISMATCH"


def render_markdown(report: AcceptanceReport) -> str:
    """Render *report* as the human-readable ``coverage.md`` companion."""
    structure = report.structure
    lines: list[str] = [
        "# ARD acceptance readout",
        "",
        f"- report schema: `{report.report_schema}`",
        f"- structure readout within the construction rule: "
        f"`{'yes' if structure.within_rule else 'no'}`",
        "",
        "## Structure readout (zero model calls)",
        "",
        "| reading | measured | expected | status |",
        "|---|---:|---:|---|",
        f"| plan entries | {structure.plan_total} | {structure.expected_total} | "
        f"{_status(structure.plan_total, structure.expected_total)} |",
        f"| text-only plan entries | {structure.plan_text_entries} | "
        f"{structure.expected_text_blocks} | "
        f"{_status(structure.plan_text_entries, structure.expected_text_blocks)} |",
        f"| image plan entries | {structure.plan_image_entries} | "
        f"{structure.expected_image_blocks} | "
        f"{_status(structure.plan_image_entries, structure.expected_image_blocks)} |",
        f"| legal restricted blocks (text) | {structure.text_block_count} | "
        f"{structure.expected_text_blocks} | "
        f"{_status(structure.text_block_count, structure.expected_text_blocks)} |",
        f"| legal restricted blocks (image) | {structure.image_block_count} | "
        f"{structure.expected_image_blocks} | "
        f"{_status(structure.image_block_count, structure.expected_image_blocks)} |",
        f"| knowledge_domain leaves | {structure.knowledge_domain_leaves} | "
        f"{structure.expected_knowledge_domains} | "
        f"{_status(structure.knowledge_domain_leaves, structure.expected_knowledge_domains)} |",
        f"| visual_domain leaves | {structure.visual_domain_leaves} | "
        f"{structure.expected_visual_domains} | "
        f"{_status(structure.visual_domain_leaves, structure.expected_visual_domains)} |",
        f"| duplicate coordinates | {structure.duplicate_coordinates} | 0 | "
        f"{_status(structure.duplicate_coordinates, 0)} |",
        "",
        "## Conventions",
        "",
        f"- constants read from `{structure.conventions.sampling_module}` at report time",
        f"- `MULTI_TURN_DEFAULT = {structure.conventions.multi_turn_default}`",
        f"- turn-count mapping: {structure.conventions.turn_count_mapping}",
        "",
    ]
    metrics = report.metrics
    if metrics is None:
        lines += ["## Metric readout", "", "not computed — structure readout only.", ""]
    else:
        space = metrics.space
        band = metrics.epsilon_band
        lines += [
            "## Metric readout",
            "",
            f"- anchors: `{space.anchors_source}`, field `{space.anchor_field}`, "
            f"|A| = {space.n_anchor}",
            f"- targets: `{space.targets_source}`, field `{space.target_field}`, "
            f"|T| = {space.n_target}",
            f"- embedder: model `{space.embedder.model}`, dimension "
            f"{space.embedder.dimension}, normalize `{space.embedder.normalize}` "
            "(no endpoint or key is recorded)",
            f"- distance `{space.distance}`, quantile method `{space.quantile_method}`",
            f"- epsilon = {space.epsilon:.6f} (source: {space.epsilon_source})",
            "",
            "| reading | value |",
            "|---|---:|",
            f"| q50 | {metrics.quantiles.q50:.6f} |",
            f"| q90 | {metrics.quantiles.q90:.6f} |",
            f"| **q95** | **{metrics.quantiles.q95:.6f}** |",
            f"| r_max | {metrics.quantiles.r_max:.6f} |",
            f"| Extent(epsilon) | {metrics.extent:.6f} |",
            f"| Extent(0.95 * epsilon) | {band.extent_at_095:.6f} |",
            f"| Extent(1.00 * epsilon) | {band.extent_at_100:.6f} |",
            f"| Extent(1.05 * epsilon) | {band.extent_at_105:.6f} |",
            "",
        ]
        noise = metrics.noise
        if noise.available and noise.band is not None:
            lines += [
                f"- noise band `[{noise.band.lower:.6f}, {noise.band.upper:.6f}]` from "
                f"{noise.n_repeat_groups} repeated coordinate(s), {noise.n_pairs} pair(s)",
                "",
            ]
        else:
            lines += [f"- noise band: **unavailable** — {noise.reason}", ""]
    lines += ["## Warnings", ""]
    if report.warnings:
        lines += [f"- {warning}" for warning in report.warnings]
    else:
        lines += ["- none"]
    lines.append("")
    return "\n".join(lines)
