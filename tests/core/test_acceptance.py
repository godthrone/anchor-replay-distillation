# test_acceptance.py — Unit tests for the pure acceptance readout assembly.
# Responsibility: freeze the structure readout (counts come from ard.core.sampling
# at call time, never from this module), the anchor-text / repeat-group
# extraction contracts, the noise-band statement, and the space declaration.

from __future__ import annotations

import pytest

from ard.backends.embedding_client import EmbeddingMatrix
from ard.core import acceptance, sampling
from ard.core import coverage as ruler
from ard.core.ontology import OntologyV4
from ard.core.sampling import sample_anchors
from ard.core.types import AnchorGenerationConfig


def _vector_set(name: str, rows: EmbeddingMatrix) -> ruler.VectorSet:
    return ruler.VectorSet(name=name, vectors=rows)


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
    """Regression (WP-5): a normal 8-entry smoke artifact used to print MISMATCH rows."""
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


# ── Anchor text extraction ──────────────────────────────────────────────────


def _record(anchor_id: str, messages: list[dict], meta: dict | None = None) -> dict:
    return {
        "id": anchor_id,
        "messages": messages,
        "anchor_meta": meta if meta is not None else {"knowledge_domain": "k"},
    }


def test_anchor_texts_uses_the_final_user_turn() -> None:
    records = [
        _record(
            "a1",
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "final question"},
            ],
        )
    ]
    assert acceptance.anchor_texts(records) == ["final question"]


@pytest.mark.parametrize(
    ("records", "fragment"),
    [
        ([], "holds no records"),
        ([_record("a", [])], "non-empty 'messages' list"),
        ([_record("a", [{"role": "assistant", "content": "x"}])], "not a user turn"),
        ([_record("a", [{"role": "user", "content": "   "}])], "blank content"),
    ],
)
def test_anchor_texts_refuses_unusable_records(records: list[dict], fragment: str) -> None:
    with pytest.raises(acceptance.AcceptanceError, match=fragment):
        acceptance.anchor_texts(records)


def test_anchor_field_declares_the_text_parts_only_space() -> None:
    """The declared anchor field must say the image pixels are not embedded."""
    assert acceptance.ANCHOR_TEXT_FIELD == "messages[last].content(text parts only)"


def test_anchor_texts_takes_the_text_part_of_a_multimodal_turn() -> None:
    """An image-modality anchor is measured through its request text."""
    records = [
        _record(
            "img1",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "images/aesthetics/sample_01.jpg"},
                        {"type": "text", "text": "What is happening in this painting?"},
                    ],
                }
            ],
        )
    ]
    assert acceptance.anchor_texts(records) == ["What is happening in this painting?"]


def test_anchor_texts_joins_every_text_part_and_ignores_non_text_parts() -> None:
    records = [
        _record(
            "img2",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
                        {"type": "text", "text": "first half"},
                        {"type": "image", "image": "images/x/y.jpg"},
                        {"type": "text", "text": "   "},
                        {"type": "text", "text": "second half"},
                    ],
                }
            ],
        )
    ]
    assert acceptance.anchor_texts(records) == [
        "first half" + acceptance.TEXT_PART_SEPARATOR + "second half"
    ]


def test_anchor_texts_refuses_a_turn_without_any_text_part() -> None:
    """An image-only anchor is named and refused — never skipped, never blank-filled."""
    records = [
        _record("a1", [{"role": "user", "content": "text anchor"}]),
        _record(
            "img-only",
            [{"role": "user", "content": [{"type": "image", "image": "images/x/y.jpg"}]}],
        ),
    ]
    with pytest.raises(acceptance.AcceptanceError) as excinfo:
        acceptance.anchor_texts(records)
    message = str(excinfo.value)
    assert "anchor record 1" in message
    assert "img-only" in message
    assert "no text part" in message
    assert "image" in message


def test_anchor_texts_refuses_blank_text_parts() -> None:
    records = [
        _record(
            "img-blank",
            [{"role": "user", "content": [{"type": "text", "text": "   "}]}],
        )
    ]
    with pytest.raises(acceptance.AcceptanceError, match="no text part"):
        acceptance.anchor_texts(records)


def test_user_turn_text_refuses_a_non_string_non_list_content() -> None:
    with pytest.raises(acceptance.AcceptanceError, match="must be a string"):
        acceptance.user_turn_text(42, "anchor record 0 (id='a')")


# ── Repeat groups and the noise band ────────────────────────────────────────


def test_repeat_groups_only_returns_coordinates_with_repeats() -> None:
    groups = acceptance.repeat_groups(
        [
            {"knowledge_domain": "k1"},
            {"knowledge_domain": "k2"},
            {"knowledge_domain": "k1"},
        ]
    )
    assert groups == [[0, 2]]


def test_repeat_groups_refuses_a_record_without_coordinates() -> None:
    with pytest.raises(acceptance.AcceptanceError, match="anchor_meta"):
        acceptance.repeat_groups([{}])


def test_noise_section_states_unavailability_without_repeats() -> None:
    section = acceptance.noise_section(_vector_set("anchors", [[1.0, 0.0]]), [])
    assert section.available is False
    assert section.band is None
    assert section.reason == acceptance.NOISE_UNAVAILABLE_REASON
    assert section.n_pairs == 0


def test_noise_section_measures_a_repeat_group() -> None:
    anchors = _vector_set("anchors", [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    section = acceptance.noise_section(anchors, [[0, 1, 2]])
    assert section.available is True
    assert section.reason is None
    assert section.n_repeat_groups == 1
    assert section.n_pairs == 3
    assert section.band is not None
    # pair distances: a-b = 1, a-c = 0, b-c = 1 ⇒ q50 = 1, max = 1
    assert section.band.lower == pytest.approx(1.0)
    assert section.band.upper == pytest.approx(1.0)


def test_intrinsic_epsilon_is_the_median_nearest_neighbour_distance() -> None:
    targets = _vector_set("targets", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    # nearest neighbours: 1, 1, 1 ⇒ median 1.0
    assert acceptance.intrinsic_epsilon(targets) == pytest.approx(1.0)


def test_intrinsic_epsilon_needs_two_targets() -> None:
    with pytest.raises(acceptance.AcceptanceError, match="at least 2"):
        acceptance.intrinsic_epsilon(_vector_set("targets", [[1.0, 0.0]]))


# ── Metric readout and rendering ────────────────────────────────────────────


def _metric_readout() -> acceptance.MetricReadout:
    anchors = _vector_set("anchors", [[1.0, 0.0], [0.0, 1.0]])
    targets = _vector_set("targets", [[1.0, 0.0], [0.7071067811865476, 0.7071067811865476]])
    return acceptance.metric_readout(
        anchors,
        targets,
        epsilon=0.5,
        epsilon_source="unit-test source",
        anchors_source="out/anchor_bank.jsonl",
        targets_source="targets.jsonl",
        embedder=acceptance.EmbedderIdentity(model="mock-embed", dimension=2, normalize=True),
        noise=acceptance.noise_section(anchors, []),
    )


def test_metric_readout_declares_its_space_and_matches_the_ruler() -> None:
    anchors = _vector_set("anchors", [[1.0, 0.0], [0.0, 1.0]])
    targets = _vector_set("targets", [[1.0, 0.0], [0.7071067811865476, 0.7071067811865476]])
    readout = _metric_readout()
    expected = ruler.acceptance_readout(targets, anchors, epsilon=0.5, label="ard-run")

    assert readout.quantiles == expected.quantiles
    assert readout.extent == pytest.approx(expected.extent)
    assert readout.epsilon_band == expected.epsilon_band
    assert readout.space.n_anchor == 2
    assert readout.space.n_target == 2
    assert readout.space.anchor_field == acceptance.ANCHOR_TEXT_FIELD
    assert readout.space.target_field == acceptance.TARGET_TEXT_FIELD
    assert readout.space.embedder.model == "mock-embed"
    assert readout.space.embedder.dimension == 2
    assert readout.space.epsilon_source == "unit-test source"


def test_render_markdown_states_the_missing_noise_band(ontology: OntologyV4) -> None:
    report = acceptance.AcceptanceReport(
        structure=acceptance.structure_readout([], ontology=ontology),
        metrics=_metric_readout(),
        warnings=["metric readout not measured: coverage.target_set_path is unset"],
    )
    markdown = acceptance.render_markdown(report)
    assert "unavailable" in markdown
    assert acceptance.NOISE_UNAVAILABLE_REASON in markdown
    assert "metric readout not measured" in markdown
    assert "q95" in markdown
    assert "coverage" in markdown and "density" in markdown
    assert acceptance.REPORT_SCHEMA in markdown


def test_render_markdown_says_not_computed_without_metrics(ontology: OntologyV4) -> None:
    report = acceptance.AcceptanceReport(
        structure=acceptance.structure_readout([], ontology=ontology), metrics=None, warnings=[]
    )
    markdown = acceptance.render_markdown(report)
    assert "not computed" in markdown
    assert "- none" in markdown
