"""Contract tests for the v4 construction rule in :mod:`ard.core.sampling`.

The rule is a *count contract*: one sample per legal restricted block, one per
rotated ``knowledge_domain`` leaf, one per rotated ``visual_domain`` leaf in the
image modality.  These tests pin the numbers the rule produces against the real
shipped ontology (935 + 891 = 1,826) and pin the failure behaviour when an
ontology drifts away from them — a silently shorter plan is the one outcome the
rule must never allow (§2.3 边界校验即防呆).
"""

from __future__ import annotations

import json

import pytest

from ard.config import load_config
from ard.core.constraints import ConstraintEvaluator, LegalBlockCounts, RestrictedBlock
from ard.core.ontology import OntologyV4, load_ontology_v4
from ard.core.sampling import (
    EXPECTED_IMAGE_BLOCKS,
    EXPECTED_KNOWLEDGE_DOMAINS,
    EXPECTED_TEXT_BLOCKS,
    EXPECTED_TOTAL,
    EXPECTED_VISUAL_DOMAINS,
    MULTI_TURN_DEFAULT,
    AnchorCoordinate,
    SamplingError,
    sample_anchors,
    sample_coordinates,
    turn_counts_by_conversation_type,
)
from ard.core.types import AnchorGenerationConfig

ONTOLOGY_PATH = "ontology/anchor_ontology.v4.json"
RESTRICTED_AXES = (
    "capability",
    "system_prompt_mode",
    "conversation_type",
    "output_format",
    "input_condition",
    "answer_mode",
)


# ── fixtures (the ontology is parsed once per session) ──────────────────────


@pytest.fixture(scope="module")
def ontology() -> OntologyV4:
    return load_ontology_v4(ONTOLOGY_PATH)


@pytest.fixture(scope="module")
def plan(ontology: OntologyV4) -> tuple[AnchorCoordinate, ...]:
    return sample_coordinates(ontology, seed=20260925)


def _block_key(coordinate: AnchorCoordinate) -> tuple[str, ...]:
    values = coordinate.as_dict()
    return tuple(values[axis] for axis in RESTRICTED_AXES)


def _text_plan(plan: tuple[AnchorCoordinate, ...]) -> list[AnchorCoordinate]:
    return [c for c in plan if c.modality == "text_only"]


def _image_plan(plan: tuple[AnchorCoordinate, ...]) -> list[AnchorCoordinate]:
    return [c for c in plan if c.modality == "image"]


# ── the four counts + the total ─────────────────────────────────────────────


def test_rule_total_is_1826(plan: tuple[AnchorCoordinate, ...]) -> None:
    assert len(plan) == EXPECTED_TOTAL == 1826


def test_text_blocks_cover_935_of_935(plan: tuple[AnchorCoordinate, ...]) -> None:
    text = _text_plan(plan)
    blocks = {_block_key(c) for c in text}
    assert len(text) == EXPECTED_TEXT_BLOCKS == 935
    assert len(blocks) == EXPECTED_TEXT_BLOCKS, "every text block exactly once"
    assert all(c.visual_domain is None for c in text)


def test_image_blocks_cover_891_of_891(plan: tuple[AnchorCoordinate, ...]) -> None:
    image = _image_plan(plan)
    blocks = {_block_key(c) for c in image}
    assert len(image) == EXPECTED_IMAGE_BLOCKS == 891
    assert len(blocks) == EXPECTED_IMAGE_BLOCKS, "every image block exactly once"


def test_knowledge_domain_covers_209_of_209(
    ontology: OntologyV4, plan: tuple[AnchorCoordinate, ...]
) -> None:
    covered = {c.knowledge_domain for c in plan}
    declared = set(ontology.axis_values("knowledge_domain"))
    assert len(covered) == EXPECTED_KNOWLEDGE_DOMAINS == 209
    assert covered == declared, "the rotation must visit every declared leaf"


def test_visual_domain_covers_21_of_21(
    ontology: OntologyV4, plan: tuple[AnchorCoordinate, ...]
) -> None:
    covered = {c.visual_domain for c in _image_plan(plan)}
    declared = set(ontology.axis_values("visual_domain"))
    assert len(covered) == EXPECTED_VISUAL_DOMAINS == 21
    assert covered == declared


def test_no_duplicate_coordinates(plan: tuple[AnchorCoordinate, ...]) -> None:
    identities = [tuple(c.as_dict().items()) for c in plan]
    assert len(set(identities)) == len(identities) == EXPECTED_TOTAL


def test_image_blocks_are_the_image_capable_subspace(
    ontology: OntologyV4, plan: tuple[AnchorCoordinate, ...]
) -> None:
    """The 891 image blocks are exactly the evaluator's image-capable subset."""
    evaluator = ConstraintEvaluator(ontology)
    expected = {
        _block_key(
            AnchorCoordinate(
                modality="image",
                language="ignored",
                knowledge_domain="ignored",
                capability=block.capability,
                system_prompt_mode=block.system_prompt_mode,
                conversation_type=block.conversation_type,
                response_style="ignored",
                output_format=block.output_format,
                difficulty="ignored",
                context_length="ignored",
                input_condition=block.input_condition,
                answer_mode=block.answer_mode,
                visual_domain="ignored",
            )
        )
        for block in evaluator.enumerate_legal_blocks(image_capable_only=True)
    }
    assert {_block_key(c) for c in _image_plan(plan)} == expected


# ── determinism (same seed → identical plan; different seed → not) ───────────


def test_same_seed_yields_an_identical_plan(ontology: OntologyV4) -> None:
    first = sample_coordinates(ontology, seed=7)
    second = sample_coordinates(ontology, seed=7)
    assert json.dumps([c.as_dict() for c in first], sort_keys=True) == json.dumps(
        [c.as_dict() for c in second], sort_keys=True
    )


def test_a_different_seed_changes_the_plan(ontology: OntologyV4) -> None:
    assert sample_coordinates(ontology, seed=7) != sample_coordinates(ontology, seed=8)


def test_spec_plan_is_identical_for_the_same_seed(ontology: OntologyV4) -> None:
    config = AnchorGenerationConfig(seed=99)

    def fingerprint() -> str:
        specs = sample_anchors(ontology, config)
        return json.dumps(
            [
                {
                    "id": spec.id,
                    "meta": spec.anchor_meta,
                    "turns": [(t.turn_index, t.role, t.is_final) for t in spec.turns],
                }
                for spec in specs
            ],
            sort_keys=True,
            ensure_ascii=False,
        )

    assert fingerprint() == fingerprint()


def test_spec_plan_has_one_spec_per_coordinate_and_unique_ids(
    ontology: OntologyV4, plan: tuple[AnchorCoordinate, ...]
) -> None:
    specs = sample_anchors(ontology, AnchorGenerationConfig(seed=3))
    assert len(specs) == EXPECTED_TOTAL
    assert len({spec.id for spec in specs}) == EXPECTED_TOTAL
    assert [spec.anchor_meta["modality"] for spec in specs] == [
        c.modality for c in plan
    ], "plan order must be preserved (the resume path slices this list)"


# ── turn counts: derived from the ontology ──────────────────────────────────


def test_turn_counts_come_from_the_conversation_type_attribute(
    ontology: OntologyV4,
) -> None:
    """Every declared count becomes an odd spec turn count ending on ``user``.

    The ontology counts *exchanges*; an AnchorSpec carries the conversation
    prefix that ends on the final user question, i.e. ``2 * n - 1`` turns.
    """
    counts = turn_counts_by_conversation_type(ontology)
    assert set(counts) == set(ontology.axis_values("conversation_type"))
    assert counts == {
        "single_turn": 1,
        "clarification": 3,
        "troubleshooting": 5,
        "iterative_revision": 5,
        "constraint_update": 7,
        "tool_assisted": 2 * MULTI_TURN_DEFAULT - 1,
        "source_review": 2 * MULTI_TURN_DEFAULT - 1,
    }
    assert all(count % 2 == 1 and count >= 1 for count in counts.values())


def test_multi_turn_bound_is_the_ontologys_largest_declared_count(
    ontology: OntologyV4,
) -> None:
    """``"multi"`` resolves to the largest explicit count the ontology declares.

    That is the documented basis of :data:`MULTI_TURN_DEFAULT`; if the ontology
    ever declares something larger, the constant is stale and the resolution must
    refuse rather than silently plan a shorter conversation.
    """
    spec = ontology.axes.spec("conversation_type")
    declared = [
        attributes.root["turns"]
        for attributes in getattr(spec, "value_attributes").values()
        if isinstance(attributes.root["turns"], int)
    ]
    assert isinstance(declared, list) and declared
    assert max(declared) == MULTI_TURN_DEFAULT


def test_an_unusable_turn_attribute_is_refused() -> None:
    """A count that is neither a positive int nor the marker is not guessed at."""
    from ard.core import sampling

    with pytest.raises(SamplingError, match="unusable turns attribute"):
        sampling._spec_turns("many", "tool_assisted")
    with pytest.raises(SamplingError, match="unusable turns attribute"):
        sampling._spec_turns(None, "tool_assisted")
    with pytest.raises(SamplingError, match="unusable turns attribute"):
        sampling._spec_turns(0, "single_turn")


def test_a_declared_count_above_the_multi_bound_is_refused() -> None:
    """A count above the ``"multi"`` bound would invalidate its documented basis."""
    from ard.core import sampling

    with pytest.raises(SamplingError, match="no longer"):
        sampling._spec_turns(MULTI_TURN_DEFAULT + 1, "constraint_update")


# ── no silent failure: counts and duplicates ────────────────────────────────


def test_a_leaf_removal_is_refused(ontology: OntologyV4, monkeypatch: pytest.MonkeyPatch) -> None:
    """An ontology edited without its counts must fail loudly, not sample short."""
    tree = ontology.knowledge_domain_tree.root
    first_domain = next(iter(tree))
    first_subdomain = next(iter(tree[first_domain].root))
    removed = tree[first_domain].root[first_subdomain].pop()
    assert removed, "the fixture must actually remove a leaf"
    try:
        with pytest.raises(SamplingError) as excinfo:
            sample_coordinates(ontology, seed=1)
    finally:
        tree[first_domain].root[first_subdomain].append(removed)

    message = str(excinfo.value)
    assert "knowledge_domain leaves" in message
    assert "expected 209 (received: 208)" in message


def test_truncated_block_enumeration_is_refused(
    ontology: OntologyV4, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong *block* count (not just a wrong leaf count) is refused too."""
    real_enumerate = ConstraintEvaluator.enumerate_legal_blocks

    def truncating(
        self: ConstraintEvaluator, *, image_capable_only: bool = False
    ) -> tuple[RestrictedBlock, ...]:
        return real_enumerate(self, image_capable_only=image_capable_only)[:100]

    monkeypatch.setattr(ConstraintEvaluator, "enumerate_legal_blocks", truncating)

    with pytest.raises(SamplingError) as excinfo:
        sample_coordinates(ontology, seed=1)

    message = str(excinfo.value)
    assert "legal restricted blocks (text)" in message
    assert "expected 935 (received: 100)" in message


def test_duplicate_coordinates_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A collision must be reported, never sampled twice."""
    block = RestrictedBlock(
        capability="qa",
        system_prompt_mode="none",
        conversation_type="single_turn",
        output_format="paragraph",
        input_condition="clean",
        answer_mode="direct_answer",
    )
    coordinates = [
        AnchorCoordinate(
            modality="text_only",
            language="English",
            knowledge_domain="x",
            capability=block.capability,
            system_prompt_mode=block.system_prompt_mode,
            conversation_type=block.conversation_type,
            response_style="concise",
            output_format=block.output_format,
            difficulty="intermediate",
            context_length="medium",
            input_condition=block.input_condition,
            answer_mode=block.answer_mode,
        ),
        AnchorCoordinate(
            modality="text_only",
            language="English",
            knowledge_domain="x",
            capability=block.capability,
            system_prompt_mode=block.system_prompt_mode,
            conversation_type=block.conversation_type,
            response_style="concise",
            output_format=block.output_format,
            difficulty="intermediate",
            context_length="medium",
            input_condition=block.input_condition,
            answer_mode=block.answer_mode,
        ),
    ]
    from ard.core import sampling

    with pytest.raises(SamplingError, match="duplicate coordinate"):
        sampling._reject_duplicates(coordinates)


# ── end-to-end wiring: shipped config → v4 ontology → plan ──────────────────


def test_shipped_config_points_at_the_v4_ontology() -> None:
    config = load_config("configs/config.toml")
    assert config.ontology.path == ONTOLOGY_PATH == "ontology/anchor_ontology.v4.json"


def test_shipped_config_has_no_count_or_fps_fields() -> None:
    generation = load_config("configs/config.toml").generation
    for removed in (
        "target_count",
        "criterion",
        "embeddings_path",
        "task_types",
        "languages",
        "max_turns",
        "max_turns_with_image",
    ):
        assert not hasattr(generation, removed), f"{removed} must not be a config field"


def test_pipeline_plan_matches_the_rule_contract(ontology: OntologyV4) -> None:
    from ard.pipeline import sample_specs

    config = load_config("configs/config.toml")
    specs = sample_specs(config)
    assert len(specs) == EXPECTED_TOTAL
    assert [spec.anchor_meta["modality"] for spec in specs].count("text_only") == (
        EXPECTED_TEXT_BLOCKS
    )
    assert [spec.anchor_meta["modality"] for spec in specs].count("image") == (
        EXPECTED_IMAGE_BLOCKS
    )


def test_free_axis_product_is_unchanged_by_the_rule(ontology: OntologyV4) -> None:
    """The evaluator's five-free-axis product is untouched by this module."""
    evaluator = ConstraintEvaluator(ontology)
    assert evaluator.free_axis_product() == 52668
    counts: LegalBlockCounts = evaluator.legal_block_counts()
    assert counts.legal_restricted_block == EXPECTED_TEXT_BLOCKS
    assert counts.legal_restricted_block_image_capable == EXPECTED_IMAGE_BLOCKS
