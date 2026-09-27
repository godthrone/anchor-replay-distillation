# tests/core/test_constraints.py — constraint evaluation contract tests.
# Responsibility: lock the v4 restricted-block counts (100800 / 935 / 891 /
# 52668), derive every axis / block distribution from the coordinate lists the
# ontology declares (it carries no handwritten count block — §1.4), and cover
# constraint-evaluation boundaries including unsolvable pairs.

import json
from collections import Counter
from math import prod
from pathlib import Path
from typing import Any

import pytest

from ard.backends.ontology_loader import load_ontology_v4
from ard.core.constraints import (
    FREE_AXES,
    RESTRICTED_AXES,
    ConstraintEvaluationError,
    ConstraintEvaluator,
    RestrictedBlock,
)
from ard.core.ontology import GroupedAxis, OntologyV4

ONTOLOGY_V4_PATH = Path("ontology/anchor_ontology.v4.json")


@pytest.fixture(scope="module")
def ontology() -> OntologyV4:
    """The validated real v4 ontology."""
    return load_ontology_v4(ONTOLOGY_V4_PATH)


@pytest.fixture(scope="module")
def evaluator(ontology: OntologyV4) -> ConstraintEvaluator:
    """A constraint evaluator over the real v4 ontology."""
    return ConstraintEvaluator(ontology)


@pytest.fixture(scope="module")
def legal_blocks(evaluator: ConstraintEvaluator) -> tuple[RestrictedBlock, ...]:
    """Every legal restricted block (text modality)."""
    return evaluator.enumerate_legal_blocks()


# ── The four contract counts ────────────────────────────────────────────────
#
# Every number pinned in this section is a **shipped-ontology contract**: it
# holds for the ontology published with this revision of the repository, and a
# changed ontology is expected to move it.  The runtime never depends on these
# numbers — every runtime count is enumerated from the ontology's leaf / value
# lists (``ConstraintEvaluator.enumerate_legal_blocks`` and friends) — so
# editing the ontology means updating the numbers here, never the production
# code (§1.4 single source of truth).


def test_restricted_axis_cardinalities(evaluator: ConstraintEvaluator) -> None:
    """The six restricted axes have 20/5/7/6/6/4 values.

    **shipped-ontology contract**：只对当前随仓库发布的这份本体成立；运行时
    不依赖它 —— 运行时计数一律从叶子清单穷举。
    """
    assert [len(evaluator.axis_values[axis]) for axis in RESTRICTED_AXES] == [20, 5, 7, 6, 6, 4]


def test_raw_restricted_block_is_100800(evaluator: ConstraintEvaluator) -> None:
    """Unconstrained product of the six restricted axes.

    **shipped-ontology contract**：100800 只对当前随仓库发布的这份本体成立；
    运行时不依赖它 —— 运行时计数一律从叶子清单穷举。
    """
    assert evaluator.raw_restricted_block_count() == 100800


def test_free_axis_product_is_52668(evaluator: ConstraintEvaluator) -> None:
    """Product of the five free (R7 orthogonal) axes.

    **shipped-ontology contract**：52668 只对当前随仓库发布的这份本体成立；
    运行时不依赖它 —— 运行时计数一律从叶子清单穷举。
    """
    assert evaluator.free_axis_product() == 52668


def test_axis_roles_match_declared_orthogonality(
    ontology: OntologyV4, evaluator: ConstraintEvaluator
) -> None:
    """The free/restricted split matches R7 and partitions the 11 universal axes."""
    orthogonality = next(c for c in ontology.constraints if c.id == "R7")
    assert set(orthogonality.axes) == set(FREE_AXES)  # type: ignore[union-attr]
    assert set(RESTRICTED_AXES).isdisjoint(FREE_AXES)
    assert set(RESTRICTED_AXES) | set(FREE_AXES) == set(ontology.axis_names()) - {"visual_domain"}


def test_legal_restricted_block_is_935(evaluator: ConstraintEvaluator) -> None:
    """Text-modality legal restricted blocks (visual_domain gated off by R5).

    **shipped-ontology contract**：935 只对当前随仓库发布的这份本体成立；
    运行时不依赖它 —— 运行时计数一律从叶子清单穷举。
    """
    assert len(evaluator.enumerate_legal_blocks()) == 935


def test_legal_restricted_block_image_capable_is_891(
    evaluator: ConstraintEvaluator,
) -> None:
    """Image-modality legal restricted blocks (18 image-capable capabilities).

    **shipped-ontology contract**：891 只对当前随仓库发布的这份本体成立；
    运行时不依赖它 —— 运行时计数一律从叶子清单穷举。
    """
    assert len(evaluator.enumerate_legal_blocks(image_capable_only=True)) == 891


def test_counts_model_matches_enumeration(evaluator: ConstraintEvaluator) -> None:
    """The aggregate counts model reports the four contract numbers.

    **shipped-ontology contract**：100800 / 935 / 891 / 52668 只对当前随仓库
    发布的这份本体成立；运行时不依赖它们 —— 运行时计数一律从叶子清单穷举。
    """
    counts = evaluator.legal_block_counts()
    assert counts.raw_restricted_block == 100800
    assert counts.legal_restricted_block == 935
    assert counts.legal_restricted_block_image_capable == 891
    assert counts.free_axis_product == 52668


# ── Counts derived from the declared coordinate lists ───────────────────────
#
# The ontology file carries no handwritten count block (§1.4): the leaf / value
# lists are the single authority. The tests below derive every number from those
# declarations and cross-check it against exhaustive enumeration, so adding a
# leaf needs no other field to be touched.


@pytest.mark.parametrize("axis", RESTRICTED_AXES)
def test_restricted_axis_values_come_from_the_declaration(
    ontology: OntologyV4, evaluator: ConstraintEvaluator, axis: str
) -> None:
    """Enumeration reads exactly the values the ontology declares for the axis."""
    assert evaluator.axis_values[axis] == ontology.axis_values(axis)


def test_knowledge_domain_values_are_the_flattened_leaf_list(ontology: OntologyV4) -> None:
    """The hierarchical axis value set is exactly the tree's deepest leaves."""
    leaves = [
        leaf
        for subdomains in ontology.knowledge_domain_tree.root.values()
        for group in subdomains.root.values()
        for leaf in group
    ]
    assert ontology.axis_values("knowledge_domain") == tuple(leaves)
    assert len(leaves) == len(set(leaves))


@pytest.mark.parametrize("axis", ["capability", "visual_domain"])
def test_grouped_axis_values_are_the_concatenated_groups(ontology: OntologyV4, axis: str) -> None:
    """A grouped axis value set is derived from its own group value lists."""
    spec = ontology.axes.spec(axis)
    assert isinstance(spec, GroupedAxis)
    declared = tuple(value for group in spec.groups.values() for value in group.values)
    assert ontology.axis_values(axis) == declared


def test_raw_restricted_block_is_the_product_of_declared_cardinalities(
    ontology: OntologyV4, evaluator: ConstraintEvaluator
) -> None:
    """The unconstrained product comes from the declared value counts alone."""
    declared = [len(ontology.axis_values(axis)) for axis in RESTRICTED_AXES]
    assert prod(declared) == evaluator.raw_restricted_block_count()


def test_legal_block_distributions_sum_to_the_enumerated_total(
    evaluator: ConstraintEvaluator, legal_blocks: tuple[RestrictedBlock, ...]
) -> None:
    """Per-axis distributions are derived from enumeration, not from the file."""
    total = len(evaluator.enumerate_legal_blocks())
    for axis in ("capability", "input_condition", "answer_mode"):
        counter = Counter(getattr(block, axis) for block in legal_blocks)
        assert sum(counter.values()) == total
        assert set(counter) <= set(evaluator.axis_values[axis])


# ── Enumeration properties ──────────────────────────────────────────────────


def test_every_enumerated_block_is_legal(
    evaluator: ConstraintEvaluator, legal_blocks: tuple[RestrictedBlock, ...]
) -> None:
    """Enumeration never emits a block that the legality check rejects."""
    assert all(evaluator.is_legal_block(block) for block in legal_blocks)


def test_enumerated_blocks_are_unique(legal_blocks: tuple[RestrictedBlock, ...]) -> None:
    """No restricted coordinate appears twice.

    **shipped-ontology contract**：尾部的 935 只对当前随仓库发布的这份本体成立；
    运行时不依赖它 —— 运行时计数一律从叶子清单穷举。
    """
    coordinates = [tuple(block.as_dict().values()) for block in legal_blocks]
    assert len(set(coordinates)) == len(coordinates) == 935


def test_block_axis_names_are_the_six_restricted_axes() -> None:
    """A block carries exactly the six restricted coordinates."""
    block = RestrictedBlock(
        capability="qa",
        system_prompt_mode="none",
        conversation_type="single_turn",
        output_format="paragraph",
        input_condition="clean",
        answer_mode="direct_answer",
    )
    assert tuple(block.as_dict()) == RESTRICTED_AXES
    assert "capability=qa" in block.label()


def test_image_capable_excludes_text_only_capabilities(
    evaluator: ConstraintEvaluator,
) -> None:
    """R5 text_only capabilities are absent from image enumeration.

    **shipped-ontology contract**：18 / 935 / 891 / 44 只对当前随仓库发布的
    这份本体成立；运行时不依赖它们 —— 运行时计数一律从叶子清单穷举。
    """
    assert evaluator.text_only == {"tool_use", "ask_clarification"}
    image_capabilities = {
        block.capability for block in evaluator.enumerate_legal_blocks(image_capable_only=True)
    }
    assert image_capabilities.isdisjoint(evaluator.text_only)
    assert len(image_capabilities) == 18
    # 891 = 935 minus the legal blocks of the two text-only capabilities (12 + 32).
    assert 935 - 891 == 44


def test_none_system_prompt_is_wildcard_for_every_capability(
    evaluator: ConstraintEvaluator,
) -> None:
    """R1 maps `none` to every capability."""
    table = next(t for t in evaluator.tables if t.constraint_id == "R1")
    assert table.targets["none"] == evaluator.axis_values["capability"]


# ── Legality boundaries ─────────────────────────────────────────────────────


def _block(**overrides: str) -> RestrictedBlock:
    """Build a legal baseline block with targeted overrides."""
    base = {
        "capability": "qa",
        "system_prompt_mode": "none",
        "conversation_type": "single_turn",
        "output_format": "paragraph",
        "input_condition": "clean",
        "answer_mode": "direct_answer",
    }
    base.update(overrides)
    return RestrictedBlock(**base)


def test_known_legal_block_is_accepted(evaluator: ConstraintEvaluator) -> None:
    """The baseline block satisfies every allowed-pairs constraint."""
    assert evaluator.is_legal_block(_block())


def test_capability_answer_mode_violation_is_rejected(evaluator: ConstraintEvaluator) -> None:
    """R4a forbids `ask_before_answering` for `qa` even on clean input."""
    probe = _block(answer_mode="ask_before_answering")
    assert probe.input_condition == "clean"  # R4b alone would allow it
    assert not evaluator.is_legal_block(probe)


def test_system_prompt_capability_violation_is_rejected(evaluator: ConstraintEvaluator) -> None:
    """R1 forbids pairing `detailed_persona` with `coding`."""
    probe = _block(
        capability="coding", system_prompt_mode="detailed_persona", output_format="code block"
    )
    assert not evaluator.is_legal_block(probe)


def test_capability_output_format_violation_is_rejected(evaluator: ConstraintEvaluator) -> None:
    """R3 forbids `coding` rendering as `paragraph`."""
    probe = _block(capability="coding", output_format="paragraph")
    assert not evaluator.is_legal_block(probe)


@pytest.mark.parametrize(
    ("capability", "input_condition"),
    [
        ("qa", "missing_info"),
        ("tool_use", "ambiguous"),
        ("creative_writing", "edge_case"),
        ("ask_clarification", "irrelevant_details"),
    ],
)
def test_unsolvable_pairs_have_no_legal_block(
    evaluator: ConstraintEvaluator,
    legal_blocks: tuple[RestrictedBlock, ...],
    capability: str,
    input_condition: str,
) -> None:
    """A pair absent from every legal block cannot be realised by any block."""
    matching = [
        block
        for block in legal_blocks
        if block.capability == capability and block.input_condition == input_condition
    ]
    assert matching == []
    assert (capability, input_condition) in evaluator.unsolvable_capability_input_pairs()


def test_unsolvable_pairs_are_the_complement_of_enumeration(
    evaluator: ConstraintEvaluator, legal_blocks: tuple[RestrictedBlock, ...]
) -> None:
    """An unsolvable pair is exactly a (capability, input_condition) with no block."""
    realised = {(block.capability, block.input_condition) for block in legal_blocks}
    all_pairs = {
        (capability, input_condition)
        for capability in evaluator.axis_values["capability"]
        for input_condition in evaluator.axis_values["input_condition"]
    }
    assert set(evaluator.unsolvable_capability_input_pairs()) == all_pairs - realised
    assert realised | (all_pairs - realised) == all_pairs


def test_r4b_relaxations_are_effective(evaluator: ConstraintEvaluator) -> None:
    """R4b relaxations: ask_clarification may ask on clean input."""
    assert evaluator.is_legal_block(
        _block(capability="ask_clarification", answer_mode="ask_before_answering")
    )
    assert evaluator.is_legal_block(
        _block(capability="qa", input_condition="ambiguous", answer_mode="explain_then_answer")
    )


def test_missing_info_admits_only_ask_before_answering(
    evaluator: ConstraintEvaluator, legal_blocks: tuple[RestrictedBlock, ...]
) -> None:
    """R4b makes `missing_info` reachable only through `ask_before_answering`."""
    answer_modes = {
        block.answer_mode for block in legal_blocks if block.input_condition == "missing_info"
    }
    assert answer_modes == {"ask_before_answering"}


# ── Fail-loud on a drifting constraint table ────────────────────────────────


def test_incomplete_from_axis_coverage_fails_loudly(tmp_path: Path) -> None:
    """A from-value with no rule row is an error, not a silently empty set."""
    payload: dict[str, Any] = json.loads(ONTOLOGY_V4_PATH.read_text(encoding="utf-8"))
    payload["constraints"][1]["rules"] = [
        rule for rule in payload["constraints"][1]["rules"] if rule["from_value"] != "coding"
    ]
    path = tmp_path / "drifting.v4.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConstraintEvaluationError, match="coding"):
        ConstraintEvaluator(load_ontology_v4(path))


def test_loader_builds_evaluator() -> None:
    """The facility loader plus the pure evaluator reproduce the block count.

    **shipped-ontology contract**：935 只对当前随仓库发布的这份本体成立；
    运行时不依赖它 —— 运行时计数一律从叶子清单穷举。
    """
    evaluator = ConstraintEvaluator(load_ontology_v4(ONTOLOGY_V4_PATH))
    assert len(evaluator.enumerate_legal_blocks()) == 935
