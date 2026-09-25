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


def test_structure_readout_matches_the_real_rule_plan(ontology: OntologyV4) -> None:
    """The production plan lands exactly on every rule count (no hard-coding)."""
    plan = sample_anchors(ontology, AnchorGenerationConfig(seed=11))
    readout = acceptance.structure_readout([spec.anchor_meta for spec in plan])

    assert readout.within_rule is True
    assert acceptance.structure_mismatch(readout) is None
    assert readout.plan_total == sampling.EXPECTED_TOTAL
    assert readout.text_block_count == sampling.EXPECTED_TEXT_BLOCKS
    assert readout.image_block_count == sampling.EXPECTED_IMAGE_BLOCKS
    assert readout.knowledge_domain_leaves == sampling.EXPECTED_KNOWLEDGE_DOMAINS
    assert readout.visual_domain_leaves == sampling.EXPECTED_VISUAL_DOMAINS
    assert readout.duplicate_coordinates == 0


def test_structure_readout_reads_expected_counts_from_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patching the constant changes the report — the number is not copied here."""
    monkeypatch.setattr(sampling, "EXPECTED_TOTAL", 1)
    monkeypatch.setattr(sampling, "EXPECTED_TEXT_BLOCKS", 1)
    readout = acceptance.structure_readout([{"modality": sampling.MODALITY_TEXT}])
    assert readout.expected_total == 1
    assert readout.expected_text_blocks == 1


def test_conventions_read_multi_turn_default_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MULTI_TURN_DEFAULT`` is read when the report is built, not imported as a copy."""
    monkeypatch.setattr(sampling, "MULTI_TURN_DEFAULT", 9)
    conventions = acceptance.conventions()
    assert conventions.multi_turn_default == 9
    assert "= 17" in conventions.turn_count_mapping
    assert conventions.multi_turn_literal == sampling.MULTI_TURN_LITERAL


def test_structure_readout_counts_blocks_leaves_and_duplicates() -> None:
    """Block / leaf / duplicate counting on a hand-built plan."""
    plan = [
        {"modality": "text_only", "knowledge_domain": "k1", "capability": "c1"},
        {"modality": "text_only", "knowledge_domain": "k2", "capability": "c9"},
        {"modality": "image", "knowledge_domain": "k1", "capability": "c2", "visual_domain": "v1"},
        {"modality": "text_only", "knowledge_domain": "k1", "capability": "c1"},
    ]
    readout = acceptance.structure_readout(plan)

    assert readout.plan_total == 4
    assert readout.plan_text_entries == 3
    assert readout.plan_image_entries == 1
    assert readout.text_block_count == 2  # entries 0/3 share one block, entry 1 another
    assert readout.image_block_count == 1
    assert readout.knowledge_domain_leaves == 2
    assert readout.visual_domain_leaves == 1
    assert readout.duplicate_coordinates == 1  # entries 0 and 3: same full coordinate
    assert readout.within_rule is False


def test_structure_mismatch_names_every_disagreeing_count() -> None:
    readout = acceptance.structure_readout([])
    message = acceptance.structure_mismatch(readout)
    assert message is not None
    assert "plan entries: 0" in message
    assert str(sampling.EXPECTED_TOTAL) in message


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


def test_render_markdown_states_the_missing_noise_band() -> None:
    report = acceptance.AcceptanceReport(
        structure=acceptance.structure_readout([]),
        metrics=_metric_readout(),
        warnings=["metric readout not measured: coverage.target_set_path is unset"],
    )
    markdown = acceptance.render_markdown(report)
    assert "unavailable" in markdown
    assert acceptance.NOISE_UNAVAILABLE_REASON in markdown
    assert "metric readout not measured" in markdown
    assert "q95" in markdown


def test_render_markdown_says_not_computed_without_metrics() -> None:
    report = acceptance.AcceptanceReport(
        structure=acceptance.structure_readout([]), metrics=None, warnings=[]
    )
    markdown = acceptance.render_markdown(report)
    assert "not computed" in markdown
    assert "- none" in markdown
