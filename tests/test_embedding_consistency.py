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

    total = (
        len(items["knowledge_domains"])
        + len(items["capabilities"])
        + len(items["languages"])
        + len(items["conversation_types"])
    )
    assert total == 49, f"Total embeddings: expected 49, got {total}"


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
    """验证 ``_compose_combination_vectors`` 返回 (len(pool), 4*1024) 形状。"""
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
        }
    ]

    # Use the first knowledge domain's embedding
    first_kd = list(embed_data["items"]["knowledge_domains"].keys())[0]
    domain_emb = np.array(
        embed_data["items"]["knowledge_domains"][first_kd],
        dtype=np.float64,
    )

    result = _compose_combination_vectors(pool, dim_embeddings, domain_emb)

    assert result.shape == (1, 4096), (
        f"Expected shape (1, 4096), got {result.shape}"
    )
    assert result.dtype == np.float64, (
        f"Expected dtype float64, got {result.dtype}"
    )