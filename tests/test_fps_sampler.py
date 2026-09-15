"""Integration tests for the ARD sampler module.

Tests :mod:`ard.core.sampler` functions: combination building, and the
public ``sample_anchors`` entry point.
Uses synthetic ontologies — no external data files required.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest
from ard.core.sampler import (
    _build_all_combinations,
    _get_leaf_conversation_types,
    sample_anchors,
)
from ard.core.types import AnchorGenerationConfig


# ---------------------------------------------------------------------------
# Synthetic ontology builders
# ---------------------------------------------------------------------------


def _make_minimal_ontology() -> dict[str, Any]:
    """Construct a minimal ontology dict for testing.

    Returns an ontology with:
    - 3 languages
    - 2 knowledge domains (each with sub_categories containing leaf_topics)
    - 3 capabilities
    - 2 conversation types
    """
    return {
        "languages": ["English", "简体中文", "Español"],
        "knowledge_domains": {
            "science": {
                "sub_categories": {
                    "physics": {"leaf_topics": ["mechanics", "thermodynamics"]},
                    "biology": {"leaf_topics": ["genetics", "ecology"]},
                }
            },
            "humanities": {
                "sub_categories": {
                    "history": {"leaf_topics": ["ancient", "modern"]},
                    "philosophy": {"leaf_topics": ["ethics", "logic"]},
                }
            },
        },
        "capabilities": {
            "reasoning": ["deduction", "induction"],
            "generation": ["summarization"],
        },
        "conversation_types": {
            "single_turn": ["single_turn"],
            "multi_turn": ["multi_turn"],
        },
    }


def _make_flat_ontology() -> dict[str, Any]:
    """Ontology without sub_categories — domains are flat leaf lists."""
    return {
        "languages": ["English", "简体中文"],
        "knowledge_domains": {
            "math": {"sub_categories": {"algebra": {"leaf_topics": ["linear", "abstract"]}}},
            "cs": {"sub_categories": {"algorithms": {"leaf_topics": ["sorting", "graphs"]}}},
        },
        "capabilities": {
            "analysis": ["classification", "regression"],
        },
        "conversation_types": {
            "single_turn": ["single_turn"],
        },
    }


# ---------------------------------------------------------------------------
# _get_leaf_conversation_types
# ---------------------------------------------------------------------------


def test_get_leaf_conversation_types():
    """应从 ontology 中提取所有叶子会话类型。"""
    ontology = _make_minimal_ontology()
    leaves = _get_leaf_conversation_types(ontology)
    assert set(leaves) == {"single_turn", "multi_turn"}


def test_get_leaf_conversation_types_empty():
    """ontology 无 conversation_types 时应返回默认值。"""
    leaves = _get_leaf_conversation_types({"languages": ["English"]})
    assert leaves == ["single_turn", "multi_turn"]


def test_get_leaf_conversation_types_malformed():
    """conversation_types 值非 list 时应返回默认值。"""
    ontology: dict[str, Any] = {
        "conversation_types": {"bad": "not_a_list"}
    }
    leaves = _get_leaf_conversation_types(ontology)
    assert leaves == ["single_turn", "multi_turn"]


# ---------------------------------------------------------------------------
# _build_all_combinations
# ---------------------------------------------------------------------------


def test_build_all_combinations_count():
    """组合数 = languages × domains × capabilities × conv_types × system prompt 模式。

    合成本体没有 ``system_prompt_presence`` / ``system_prompt_style``，
    system prompt 维度退化为单值（``none``），倍率仍为 1。
    """
    ontology = _make_minimal_ontology()
    combos = _build_all_combinations(ontology, languages=[], task_types=[])
    # 3 languages × 2 domains × 3 capabilities × 2 conv_types = 36
    assert len(combos) == 36


def test_build_all_combinations_system_prompt_multiplies():
    """带 system prompt 维度的本体，组合数按模式数倍增。"""
    ontology = _make_minimal_ontology()
    ontology["system_prompt_presence"] = ["none", "present"]
    ontology["system_prompt_style"] = {
        "minimal_persona": {"anchor_concepts": ["deduction"]},
        "detailed_persona": {"anchor_concepts": ["induction"]},
    }
    combos = _build_all_combinations(ontology, languages=[], task_types=[])
    # 36 × (1 absent + 2 styles) = 108
    assert len(combos) == 108
    modes = {c["system_prompt_mode"] for c in combos}
    assert modes == {"none", "minimal_persona", "detailed_persona"}


def test_build_all_combinations_language_filter():
    """language 过滤应只保留指定语言。"""
    ontology = _make_minimal_ontology()
    combos = _build_all_combinations(ontology, languages=["English"], task_types=[])
    langs = {c["language"] for c in combos}
    assert langs == {"English"}


def test_build_all_combinations_task_type_filter():
    """task_type 过滤应只保留指定能力。"""
    ontology = _make_minimal_ontology()
    combos = _build_all_combinations(
        ontology, languages=[], task_types=["deduction"]
    )
    caps = {c["capability"] for c in combos}
    assert caps == {"deduction"}


def test_build_all_combinations_structure():
    """每个组合应包含全部必需字段（含 system prompt 三个字段）。"""
    ontology = _make_minimal_ontology()
    combos = _build_all_combinations(ontology, languages=[], task_types=[])
    for combo in combos:
        assert set(combo.keys()) == {
            "language",
            "knowledge_domain",
            "capability",
            "conversation_type",
            "system_prompt_presence",
            "system_prompt_style",
            "system_prompt_mode",
        }


# ---------------------------------------------------------------------------
# sample_anchors (public API)
# ---------------------------------------------------------------------------


def test_sample_anchors_returns_list():
    """sample_anchors 应返回 list[AnchorSpec]."""
    ontology = json.load(open("ontology/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        target_count=4, seed=42,
        embeddings_path="ontology/anchor_ontology_embeddings.json",
    )
    rng = random.Random(42)
    result = sample_anchors(ontology, config, rng)
    assert isinstance(result, list)
    assert len(result) == 4
    for item in result:
        assert hasattr(item, "anchor_meta")
        assert hasattr(item, "turns")


def test_sample_anchors_items_have_required_keys():
    """每个返回的 AnchorSpec 的 anchor_meta 应包含知识域、语言、能力、会话类型。"""
    ontology = json.load(open("ontology/anchor_ontology.json"))
    config = AnchorGenerationConfig(
        target_count=4, seed=42,
        embeddings_path="ontology/anchor_ontology_embeddings.json",
    )
    rng = random.Random(42)
    result = sample_anchors(ontology, config, rng)
    for item in result:
        meta = item.anchor_meta
        for key in ["knowledge_domain", "language", "capability", "conversation_type"]:
            assert key in meta, f"Missing key '{key}' in {meta}"


# ============================================================================
# Farthest-point sampling (FPS) tests — real ontology + embeddings
# ============================================================================


# ---------------------------------------------------------------------------
# Shared helpers for FPS tests
# ---------------------------------------------------------------------------

from ard.core._fps import (  # noqa: E402
    _compose_combination_vectors,
    _farthest_domain_order,
    _load_dimension_embeddings,
    _sample_farthest,
)


def _load_real_ontology() -> dict[str, Any]:
    """Load the real ontology from the data directory."""
    return json.load(open("ontology/anchor_ontology.json"))


def _load_real_embeddings() -> dict[str, Any]:
    """Load the real embeddings from the data directory."""
    return json.load(open("ontology/anchor_ontology_embeddings.json"))


# ---------------------------------------------------------------------------
# Layer 1: _farthest_domain_order
# ---------------------------------------------------------------------------


def test_farthest_domain_order_returns_18_domains():
    """_farthest_domain_order 应返回 18 个知识域 category 名。"""
    ontology = _load_real_ontology()
    result = _farthest_domain_order(
        "ontology/anchor_ontology_embeddings.json", 42, ontology
    )
    assert len(result) == 18
    assert all(isinstance(x, str) for x in result)


def test_farthest_domain_order_no_duplicates():
    """_farthest_domain_order 结果应无重复。"""
    ontology = _load_real_ontology()
    result = _farthest_domain_order(
        "ontology/anchor_ontology_embeddings.json", 42, ontology
    )
    assert len(result) == len(set(result))


def test_farthest_domain_order_deterministic():
    """相同 seed 应产生相同结果。"""
    ontology = _load_real_ontology()
    r1 = _farthest_domain_order(
        "ontology/anchor_ontology_embeddings.json", 42, ontology
    )
    r2 = _farthest_domain_order(
        "ontology/anchor_ontology_embeddings.json", 42, ontology
    )
    assert r1 == r2


def test_farthest_domain_order_all_in_ontology():
    """所有返回值应在 ontology 的 knowledge_domains 键中。"""
    ontology = _load_real_ontology()
    result = _farthest_domain_order(
        "ontology/anchor_ontology_embeddings.json", 42, ontology
    )
    onto_domains = set(ontology["knowledge_domains"].keys())
    for domain in result:
        assert domain in onto_domains, f"Domain '{domain}' not in ontology"


def test_farthest_domain_order_missing_embeddings_raises():
    """embeddings_path 不存在时应抛 ValueError。"""
    ontology = _load_real_ontology()
    with pytest.raises(ValueError):
        _farthest_domain_order("ontology/nonexistent_file.json", 42, ontology)


# ---------------------------------------------------------------------------
# Layer 2: _sample_farthest
# ---------------------------------------------------------------------------


def test_sample_farthest_deterministic():
    """相同 seed 应产生相同结果。"""
    ontology = _load_real_ontology()
    config1 = AnchorGenerationConfig(
        target_count=50, seed=42,
        embeddings_path="ontology/anchor_ontology_embeddings.json",
    )
    config2 = AnchorGenerationConfig(
        target_count=50, seed=42,
        embeddings_path="ontology/anchor_ontology_embeddings.json",
    )
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    result1 = _sample_farthest(ontology, config1, rng1)
    result2 = _sample_farthest(ontology, config2, rng2)
    assert result1 == result2


def test_sample_farthest_coverage_n50():
    """n=50 时 domain 覆盖率应 ≥ 50%（至少 9 个不同 domain）。"""
    ontology = _load_real_ontology()
    config = AnchorGenerationConfig(
        target_count=50, seed=42,
        embeddings_path="ontology/anchor_ontology_embeddings.json",
    )
    rng = random.Random(42)
    result = _sample_farthest(ontology, config, rng)
    domains = {item["knowledge_domain"] for item in result}
    assert len(domains) >= 9, (
        f"Expected at least 9 domains, got {len(domains)}: {domains}"
    )


def test_sample_farthest_coverage_n500():
    """n=500 时 domain 覆盖率应 ≥ 95%（至少 17 个 domain）。"""
    ontology = _load_real_ontology()
    config = AnchorGenerationConfig(
        target_count=500, seed=42,
        embeddings_path="ontology/anchor_ontology_embeddings.json",
    )
    rng = random.Random(42)
    result = _sample_farthest(ontology, config, rng)
    domains = {item["knowledge_domain"] for item in result}
    actual_count = len(result)
    assert len(domains) >= 17, (
        f"Expected at least 17 domains, got {len(domains)} "
        f"(returned {actual_count} items): {domains}"
    )


def test_sample_farthest_no_duplicates():
    """结果应无重复（去重键含 system_prompt_mode 这第 5 维）。

    Two anchors that share the first four dimensions but differ in their system
    prompt are different training samples, so ``system_prompt_mode`` belongs in
    the key — leaving it out is what makes a correct result look duplicated.
    """
    ontology = _load_real_ontology()
    config = AnchorGenerationConfig(
        target_count=50, seed=42,
        embeddings_path="ontology/anchor_ontology_embeddings.json",
    )
    rng = random.Random(42)
    result = _sample_farthest(ontology, config, rng)
    keys = [
        (item["language"], item["knowledge_domain"],
         item["capability"], item["conversation_type"], item["system_prompt_mode"])
        for item in result
    ]
    assert len(keys) == len(set(keys)), (
        f"Found {len(keys) - len(set(keys))} duplicate(s)"
    )


def test_sample_farthest_covers_the_system_prompt_dimension():
    """FPS 采样应覆盖到 system prompt 维度（含「无 system」这一取值）。"""
    ontology = _load_real_ontology()
    config = AnchorGenerationConfig(
        target_count=100, seed=42,
        embeddings_path="ontology/anchor_ontology_embeddings.json",
    )
    rng = random.Random(42)
    result = _sample_farthest(ontology, config, rng)
    modes = {item["system_prompt_mode"] for item in result}
    assert modes == {
        "none",
        "minimal_persona",
        "detailed_persona",
        "task_constraint",
        "domain_style",
    }, modes


def test_sample_farthest_correct_count():
    """_sample_farthest 应返回正确数量的锚点。"""
    ontology = _load_real_ontology()
    for target in [5, 10, 50]:
        config = AnchorGenerationConfig(
            target_count=target, seed=42,
            embeddings_path="ontology/anchor_ontology_embeddings.json",
        )
        rng = random.Random(42)
        result = _sample_farthest(ontology, config, rng)
        assert len(result) == target, (
            f"Expected {target}, got {len(result)}"
        )


# ---------------------------------------------------------------------------
# End-to-end: sample_anchors with FPS
# ---------------------------------------------------------------------------


def test_sample_anchors_fps_not_crash():
    """通过公开 API sample_anchors 使用真实 ontology + embeddings 不崩溃。"""
    ontology = _load_real_ontology()
    config = AnchorGenerationConfig(
        target_count=10, seed=42, max_turns=3,
        embeddings_path="ontology/anchor_ontology_embeddings.json",
    )
    rng = random.Random(42)
    result = sample_anchors(ontology, config, rng)
    assert isinstance(result, list)
    assert len(result) > 0


def test_sample_anchors_fps_correct_count():
    """sample_anchors 返回正确数量的 AnchorSpec。"""
    ontology = _load_real_ontology()
    for target in [5, 10, 20]:
        config = AnchorGenerationConfig(
            target_count=target, seed=42, max_turns=3,
            embeddings_path="ontology/anchor_ontology_embeddings.json",
        )
        rng = random.Random(42)
        result = sample_anchors(ontology, config, rng)
        assert len(result) == target, (
            f"Expected {target}, got {len(result)}"
        )


def test_sample_anchors_fps_deterministic():
    """相同 seed 应产生相同结果。"""
    ontology = _load_real_ontology()
    config1 = AnchorGenerationConfig(
        target_count=10, seed=42, max_turns=3,
        embeddings_path="ontology/anchor_ontology_embeddings.json",
    )
    config2 = AnchorGenerationConfig(
        target_count=10, seed=42, max_turns=3,
        embeddings_path="ontology/anchor_ontology_embeddings.json",
    )
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    result1 = sample_anchors(ontology, config1, rng1)
    result2 = sample_anchors(ontology, config2, rng2)
    # Compare ids for determinism
    ids1 = [item.id for item in result1]
    ids2 = [item.id for item in result2]
    assert ids1 == ids2