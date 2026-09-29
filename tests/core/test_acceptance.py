# test_acceptance.py — Unit tests for the pure structure readout assembly.
# Responsibility: freeze the structure readout (counts come from ard.core.sampling
# at call time, never from this module), its mismatch reporting, and the
# markdown rendering of the report.

from __future__ import annotations

import pytest

from ard.core import acceptance, sampling
from ard.core.ontology import OntologyV4
from ard.core.sampling import sample_anchors
from ard.core.types import AnchorGenerationConfig

# ── Conventions and structure readout ───────────────────────────────────────


def _rules(ontology: OntologyV4, *, seed: int = 42, **kwargs: object) -> list[dict[str, str]]:
    """The ``anchor_meta`` of one sampled plan, as the readout consumes it."""
    return [
        coordinate.as_dict()
        for coordinate in sampling.sample_coordinates(ontology, seed, **kwargs)  # type: ignore[arg-type]
    ]


def _unit_split(ontology: OntologyV4) -> tuple[int, int]:
    """``(text units, image units)`` — the runtime source of the block counts."""
    units = sampling.coverage_units(ontology)
    text = sum(1 for unit in units if unit.modality == sampling.MODALITY_TEXT)
    return text, len(units) - text


def test_conventions_read_multi_turn_default_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MULTI_TURN_DEFAULT`` is read when the report is built, not imported as a copy."""
    monkeypatch.setattr(sampling, "MULTI_TURN_DEFAULT", 9)
    conventions = acceptance.conventions()
    assert conventions.multi_turn_default == 9
    assert "= 17" in conventions.turn_count_mapping
    assert conventions.multi_turn_literal == sampling.MULTI_TURN_LITERAL


def test_structure_readout_matches_the_real_rule_plan(ontology: OntologyV4) -> None:
    """The production full-round plan lands on every count the rule derives."""
    unit_total = sampling.unit_total(ontology)
    text_units, image_units = _unit_split(ontology)
    plan = sample_anchors(ontology, AnchorGenerationConfig(seed=11))
    readout = acceptance.structure_readout([spec.anchor_meta for spec in plan], ontology=ontology)

    assert readout.within_rule is True
    assert acceptance.structure_mismatch(readout) is None
    assert readout.plan_total == unit_total
    assert readout.expected_total == unit_total
    assert readout.text_block_count == text_units
    assert readout.image_block_count == image_units
    assert readout.knowledge_domain_leaves == sampling.knowledge_leaf_count(ontology)
    assert readout.visual_domain_leaves == sampling.visual_leaf_count(ontology)
    assert readout.coverage.coverage == 1.0
    assert readout.coverage.density == 1.0
    assert readout.coverage.rounds == 1


def test_structure_readout_expectations_follow_the_run_count(ontology: OntologyV4) -> None:
    """N drives the expectations: 8 entries expect 8 entries, not a 1,826 constant."""
    unit_total = sampling.unit_total(ontology)
    plan = _rules(ontology, per_modality=(4, 4))

    explicit = acceptance.structure_readout(plan, ontology=ontology, count=8)
    assert explicit.expected_total == 8
    assert explicit.expected_distinct_coordinates == 8
    # Below one full round the shuffle order, not the rule, sets the split and
    # the leaves reached, so those expectations are ``None`` (not guessed).
    assert explicit.expected_text_blocks is None
    assert explicit.expected_image_blocks is None
    assert explicit.expected_knowledge_domains is None
    assert explicit.expected_visual_domains is None

    default = acceptance.structure_readout(plan, ontology=ontology)
    assert default.expected_total == unit_total
    assert default.expected_distinct_coordinates == unit_total
    assert default.expected_text_blocks is not None
    assert default.expected_knowledge_domains == sampling.knowledge_leaf_count(ontology)


def test_smoke_plan_raises_no_mismatch_warning(ontology: OntologyV4) -> None:
    """Regression: a normal 8-entry smoke artifact used to print MISMATCH rows."""
    plan = _rules(ontology, per_modality=(4, 4))
    readout = acceptance.structure_readout(plan, ontology=ontology, count=len(plan))

    assert readout.within_rule is True
    assert acceptance.structure_mismatch(readout) is None
    assert readout.plan_total == 8
    assert readout.plan_text_entries == 4
    assert readout.plan_image_entries == 4
    assert readout.coverage.coverage == pytest.approx(8 / sampling.unit_total(ontology))
    assert readout.coverage.density == pytest.approx(8 / sampling.unit_total(ontology))
    assert readout.coverage.rounds == 1


def test_a_full_round_saturates_coverage_and_density(ontology: OntologyV4) -> None:
    unit_total = sampling.unit_total(ontology)
    plan = _rules(ontology, count=unit_total)
    readout = acceptance.structure_readout(plan, ontology=ontology, count=unit_total)

    assert readout.coverage.distinct_coordinates == unit_total
    assert readout.coverage.coverage == 1.0
    assert readout.coverage.density == 1.0
    assert (readout.coverage.full_rounds, readout.coverage.last_round_size) == (1, 0)
    assert readout.coverage.rounds == 1
    assert readout.knowledge_domain_leaves == sampling.knowledge_leaf_count(ontology)
    assert readout.visual_domain_leaves == sampling.visual_leaf_count(ontology)
    assert readout.within_rule is True


def test_a_second_round_grows_density_but_not_coverage(ontology: OntologyV4) -> None:
    unit_total = sampling.unit_total(ontology)
    plan = _rules(ontology, count=2 * unit_total)
    readout = acceptance.structure_readout(plan, ontology=ontology, count=2 * unit_total)

    assert readout.coverage.density == 2.0
    assert readout.coverage.coverage == 1.0
    assert readout.coverage.distinct_coordinates == 2 * unit_total
    assert readout.coverage.rounds == 2
    assert readout.within_rule is True


def test_a_cross_round_repeated_coordinate_is_not_an_error(ontology: OntologyV4) -> None:
    """Same coordinate ≠ sample duplicate: counted once for coverage, twice for density.

    Round 1 re-sampling round 0's first coordinate is a legitimate v5 sample, so
    it must raise no warning of any kind and must not be named a duplicate.
    """
    unit_total = sampling.unit_total(ontology)
    coordinates = sampling.sample_coordinates(ontology, 42, count=unit_total)
    plan = [coordinate.as_dict() for coordinate in coordinates]
    plan.append(coordinates[0].as_dict())

    readout = acceptance.structure_readout(plan, ontology=ontology, count=unit_total + 1)

    # The "count by coordinate" reading no longer exists as a field or a check.
    assert "duplicate_coordinates" not in acceptance.StructureReadout.model_fields
    assert all(
        banned not in label.lower()
        for label, _measured, _expected, _relation in acceptance.structure_checks(readout)
        for banned in ("duplicate", "repeat")
    )
    assert readout.within_rule is True
    assert acceptance.structure_mismatch(readout) is None

    # Coverage credits the repeated coordinate once (still exactly U) while
    # density counts both occurrences (U + 1 entries).
    assert readout.plan_total == unit_total + 1
    assert readout.coverage.distinct_coordinates == unit_total
    assert readout.coverage.coverage == 1.0
    assert readout.coverage.density == pytest.approx((unit_total + 1) / unit_total)


def test_structure_readout_counts_blocks_leaves_and_modalities(ontology: OntologyV4) -> None:
    """Block / leaf / modality counting on a hand-built plan."""
    plan = [
        {"modality": "text_only", "knowledge_domain": "k1", "capability": "c1"},
        {"modality": "text_only", "knowledge_domain": "k2", "capability": "c9"},
        {"modality": "image", "knowledge_domain": "k1", "capability": "c2", "visual_domain": "v1"},
        {"modality": "text_only", "knowledge_domain": "k1", "capability": "c1"},
        {"modality": "video", "knowledge_domain": "k1", "capability": "c1"},
    ]
    readout = acceptance.structure_readout(plan, ontology=ontology, count=5)

    assert readout.plan_total == 5
    assert readout.plan_text_entries == 3
    assert readout.plan_image_entries == 1
    assert readout.text_block_count == 2  # entries 0/3 share one block, entry 1 another
    assert readout.image_block_count == 1
    assert readout.knowledge_domain_leaves == 2
    assert readout.visual_domain_leaves == 1
    assert readout.coverage.distinct_coordinates == 4  # entries 0 and 3 are one coordinate
    assert readout.within_rule is False  # entry 4 carries an unknown modality


def test_structure_readout_refuses_an_unusable_count(ontology: OntologyV4) -> None:
    for bad in (0, -1, True, 1.5):
        with pytest.raises(acceptance.AcceptanceError, match="count N"):
            acceptance.structure_readout([], ontology=ontology, count=bad)


def test_structure_readout_refuses_a_non_mapping_entry(ontology: OntologyV4) -> None:
    with pytest.raises(acceptance.AcceptanceError, match="plan entry 0"):
        acceptance.structure_readout([42], ontology=ontology, count=1)


def test_structure_mismatch_names_every_disagreeing_count(ontology: OntologyV4) -> None:
    readout = acceptance.structure_readout([], ontology=ontology)
    message = acceptance.structure_mismatch(readout)
    assert message is not None
    assert "plan entries: 0" in message
    assert str(sampling.unit_total(ontology)) in message


def test_render_markdown_renders_the_structure_readout(ontology: OntologyV4) -> None:
    report = acceptance.AcceptanceReport(
        structure=acceptance.structure_readout([], ontology=ontology), warnings=[]
    )
    markdown = acceptance.render_markdown(report)
    assert "## Structure readout (zero model calls)" in markdown
    assert "within the construction rule" in markdown
    assert "**coverage**" in markdown and "**density**" in markdown
    assert acceptance.REPORT_SCHEMA in markdown
    assert "## Warnings" in markdown
    assert "- none" in markdown
