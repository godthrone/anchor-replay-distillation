"""Tests for embedding data consistency and validation.

Verifies that the embedding data file is well-formed, has the expected
section counts and dimensions, and that the composition functions produce
correctly shaped output.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from ard.core._fps import _compose_combination_vectors, _load_dimension_embeddings

# ---------------------------------------------------------------------------
# Test 1: file existence
# ---------------------------------------------------------------------------


def test_embedding_file_exists() -> None:
    """确认 ``ontology/anchor_ontology_embeddings.json`` 存在。"""
    path = os.path.join("ontology", "anchor_ontology_embeddings.json")
    assert os.path.exists(path), f"Embeddings file not found: {path}"


# ---------------------------------------------------------------------------
# Test 2: section counts
# ---------------------------------------------------------------------------


def test_embedding_section_counts() -> None:
    """验证嵌入文件各 section 的 key 数量符合预期。"""
    with open(os.path.join("ontology", "anchor_ontology_embeddings.json"), encoding="utf-8") as f:
        data: dict = json.load(f)

    items = data["items"]
    assert len(items["knowledge_domains"]) == 18, (
        f"knowledge_domains: expected 18, got {len(items['knowledge_domains'])}"
    )
    assert len(items["capabilities"]) == 20, (
        f"capabilities: expected 20, got {len(items['capabilities'])}"
    )
    assert len(items["languages"]) == 4, (
        f"languages: expected 4, got {len(items['languages'])}"
    )
    assert len(items["conversation_types"]) == 7, (
        f"conversation_types: expected 7, got {len(items['conversation_types'])}"
    )
    assert len(items["system_prompt"]) == 6, (
        f"system_prompt: expected 6 (2 presence + 4 style), "
        f"got {len(items['system_prompt'])}"
    )

    total = (
        len(items["knowledge_domains"])
        + len(items["capabilities"])
        + len(items["languages"])
        + len(items["conversation_types"])
        + len(items["system_prompt"])
    )
    assert total == 55, f"Total embeddings: expected 55, got {total}"


def test_system_prompt_embeddings_match_the_ontology_vocabulary() -> None:
    """system_prompt 段覆盖的取值应与 ontology 声明的词表完全一致。"""
    with open(os.path.join("ontology", "anchor_ontology_embeddings.json"), encoding="utf-8") as f:
        embed_data: dict = json.load(f)
    with open(os.path.join("ontology", "anchor_ontology.json"), encoding="utf-8") as f:
        onto_data: dict = json.load(f)

    expected = set(onto_data["system_prompt_presence"]) | set(
        onto_data["system_prompt_style"]
    )
    assert set(embed_data["items"]["system_prompt"].keys()) == expected


def test_absent_presence_points_away_from_every_style() -> None:
    """``none`` 应指向风格质心的反方向（与每种风格余弦相似度 < 0）。"""
    with open(os.path.join("ontology", "anchor_ontology_embeddings.json"), encoding="utf-8") as f:
        embed_data: dict = json.load(f)
    with open(os.path.join("ontology", "anchor_ontology.json"), encoding="utf-8") as f:
        onto_data: dict = json.load(f)

    vectors = embed_data["items"]["system_prompt"]
    absent = np.array(vectors["none"], dtype=np.float64)
    absent = absent / np.linalg.norm(absent)
    for style in onto_data["system_prompt_style"]:
        style_vector = np.array(vectors[style], dtype=np.float64)
        style_vector = style_vector / np.linalg.norm(style_vector)
        assert float(absent @ style_vector) < 0.0, (
            f"the absent presence direction should point away from {style}"
        )


# ---------------------------------------------------------------------------
# Test 3: dimension consistency
# ---------------------------------------------------------------------------


def test_embedding_dimension_consistent() -> None:
    """遍历所有 section 的所有向量，确认每个都是 1024-dim。"""
    with open(os.path.join("ontology", "anchor_ontology_embeddings.json"), encoding="utf-8") as f:
        data: dict = json.load(f)

    items = data["items"]
    for section_name, section_data in items.items():
        for name, vec in section_data.items():
            assert len(vec) == 1024, (
                f"{section_name}.{name}: expected 1024, got {len(vec)}"
            )


# ---------------------------------------------------------------------------
# Test 4: KD names in ontology
# ---------------------------------------------------------------------------


def test_kd_names_in_ontology() -> None:
    """验证 knowledge_domains 嵌入的每个 key 与 ontology 双向一致。"""
    with open(os.path.join("ontology", "anchor_ontology_embeddings.json"), encoding="utf-8") as f:
        embed_data: dict = json.load(f)
    with open(os.path.join("ontology", "anchor_ontology.json"), encoding="utf-8") as f:
        onto_data: dict = json.load(f)

    embed_kd_names = set(embed_data["items"]["knowledge_domains"].keys())
    onto_kd_names = set(onto_data["knowledge_domains"].keys())

    missing_in_onto = embed_kd_names - onto_kd_names
    missing_in_embed = onto_kd_names - embed_kd_names

    assert not missing_in_onto, (
        f"Embedding KD keys not in ontology: {missing_in_onto}"
    )
    assert not missing_in_embed, (
        f"Ontology KD keys missing embeddings: {missing_in_embed}"
    )


# ---------------------------------------------------------------------------
# Test 5: _compose_combination_vectors shape
# ---------------------------------------------------------------------------


def test_compose_combination_vectors_shape() -> None:
    """验证 ``_compose_combination_vectors`` 返回 (len(pool), 6*1024) 形状。

    v3.0.0 adds the two system-prompt slots (presence + style), so the composed
    vector is 6 × 1024 wide; the shape assertion is what catches a dimension
    that stops being composed at all.
    """
    with open(os.path.join("ontology", "anchor_ontology_embeddings.json"), encoding="utf-8") as f:
        embed_data: dict = json.load(f)

    dim_embeddings = _load_dimension_embeddings(embed_data)

    # Pick one real key from each dimension
    cap_key = list(embed_data["items"]["capabilities"].keys())[0]  # e.g. "qa"
    lang_key = list(embed_data["items"]["languages"].keys())[0]  # e.g. "English"
    conv_key = list(embed_data["items"]["conversation_types"].keys())[0]  # e.g. "single_turn"

    pool: list[dict[str, str]] = [
        {
            "capability": cap_key,
            "language": lang_key,
            "conversation_type": conv_key,
            "system_prompt_presence": "present",
            "system_prompt_style": "minimal_persona",
        },
        {
            "capability": cap_key,
            "language": lang_key,
            "conversation_type": conv_key,
            "system_prompt_presence": "none",
            "system_prompt_style": "none",
        },
    ]

    # Use the first knowledge domain's embedding
    first_kd = list(embed_data["items"]["knowledge_domains"].keys())[0]
    domain_emb = np.array(
        embed_data["items"]["knowledge_domains"][first_kd],
        dtype=np.float64,
    )

    result = _compose_combination_vectors(pool, dim_embeddings, domain_emb)

    assert result.shape == (2, 6144), (
        f"Expected shape (2, 6144), got {result.shape}"
    )
    assert result.dtype == np.float64, (
        f"Expected dtype float64, got {result.dtype}"
    )


def test_every_composed_slot_is_unit_length() -> None:
    """每个槽位都是单位长度（各维等权，不被某一维的模长主导）。

    The composed vector itself is not re-normalised — the domain slot has to
    keep carrying the domain identity — so what is pinned here is the *slot*
    norms: five unit slots when the anchor has a system prompt, and a zero
    style slot when it does not (absence has no style direction).
    """
    with open(os.path.join("ontology", "anchor_ontology_embeddings.json"), encoding="utf-8") as f:
        embed_data: dict = json.load(f)

    dim_embeddings = _load_dimension_embeddings(embed_data)
    first_kd = list(embed_data["items"]["knowledge_domains"].keys())[0]
    domain_emb = np.array(embed_data["items"]["knowledge_domains"][first_kd], dtype=np.float64)
    dimension = len(embed_data["items"]["capabilities"][next(iter(embed_data["items"]["capabilities"]))])

    combos = [
        {
            "capability": list(embed_data["items"]["capabilities"])[0],
            "language": list(embed_data["items"]["languages"])[0],
            "conversation_type": list(embed_data["items"]["conversation_types"])[0],
            "system_prompt_presence": presence,
            "system_prompt_style": style,
        }
        for presence, style in (("present", "domain_style"), ("none", "none"))
    ]

    result = _compose_combination_vectors(combos, dim_embeddings, domain_emb)
    assert result.shape == (2, 6 * dimension)
    for row_index, expected_style_norm in ((0, 1.0), (1, 0.0)):
        row = result[row_index]
        slot_norms = [
            float(np.linalg.norm(row[index * dimension:(index + 1) * dimension]))
            for index in range(6)
        ]
        assert slot_norms[:5] == pytest.approx([1.0] * 5)
        assert slot_norms[5] == pytest.approx(expected_style_norm)