"""Farthest-point sampling (FPS) for anchor selection.

Pure computation layer: Layer 1 FPS over knowledge domain embeddings,
Layer 2 FPS within each domain over composed 4096-dim vectors.  No network,
no GPU management, no file-system side-effects beyond the JSON load path.

NOTE: This file is ~347 lines (exceeds the 300-line guideline) because it
bundles FPS algorithm, embedding composition, and validation which are
tightly coupled and share the same domain concept.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np

from ard.core.embeddings import farthest_point_sampling


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_embeddings_for_ontology(
    embeddings_path: str,
    ontology: dict[str, Any],
) -> dict[str, Any]:
    """Eager validation: 8 项校验。失败抛 ValueError。成功返回解析后的嵌入数据。

    Checks performed:
    V1: File exists
    V2: Valid JSON
    V3: ``items`` non-empty
    V4: ``knowledge_domains`` section present
    V5: Consistent embedding dimension across all sections
    V6: Every KD embedding key exists in ontology, and vice versa
    V7: Intersection non-empty
    V8: ``items`` is a dict (new format), not a list (old format)

    Args:
        embeddings_path: Path to ``anchor_ontology_embeddings.json``.
        ontology: Loaded ontology dict.

    Returns:
        Parsed embeddings data dict.

    Raises:
        ValueError: If any validation check fails.
    """
    if not os.path.exists(embeddings_path):
        raise ValueError(f"Embeddings file not found: {embeddings_path}")

    with open(embeddings_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    items = data.get("items", {})

    # V3: items non-empty
    if not items:
        raise ValueError("Embeddings file has no 'items'")

    # V8: items must be a dict (new format)
    if not isinstance(items, dict):
        raise ValueError(
            "Embeddings 'items' must be a dict (new format), "
            f"got {type(items).__name__}"
        )

    # V4: knowledge_domains section exists
    kd = items.get("knowledge_domains", {})
    if not kd:
        raise ValueError("Embeddings missing 'knowledge_domains' section")

    # V5: consistent dimension across all sections
    dims: set[int] = set()
    for _section_name, section_data in items.items():
        for _name, vec in section_data.items():
            dims.add(len(vec))
    if len(dims) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {dims}")

    # V6: KD embedding keys match ontology domain keys
    onto_domains = set(ontology.get("knowledge_domains", {}).keys())
    embed_domains = set(kd.keys())
    missing = embed_domains - onto_domains
    extra = onto_domains - embed_domains
    if missing:
        raise ValueError(f"Embedding domains not in ontology: {missing}")
    if extra:
        raise ValueError(f"Ontology domains missing embeddings: {extra}")

    # V7: intersection non-empty
    if not (embed_domains & onto_domains):
        raise ValueError(
            "No overlap between embedding domains and ontology domains"
        )

    return data


# ---------------------------------------------------------------------------
# Layer 1: domain ordering
# ---------------------------------------------------------------------------


def _farthest_domain_order(
    embeddings_path: str,
    seed: int,
    ontology: dict[str, Any],
) -> list[str]:
    """返回 18 个知识域 category 名的 FPS 多样性排序。

    Validates the embeddings file against *ontology*, then runs
    farthest-point sampling over the knowledge_domain vectors to produce
    a diversity-ordered list of domain names.

    Args:
        embeddings_path: Path to ``anchor_ontology_embeddings.json``.
        seed: Random seed for FPS initial point selection.
        ontology: Loaded ontology dict.

    Returns:
        List of knowledge domain category names ordered by FPS diversity
        (most diverse first).
    """
    data = validate_embeddings_for_ontology(embeddings_path, ontology)
    domain_embeddings: dict[str, list[float]] = data["items"]["knowledge_domains"]

    # Deterministic ordering
    names = sorted(domain_embeddings.keys())
    vectors = np.array(
        [domain_embeddings[name] for name in names],
        dtype=np.float64,
    )
    order = farthest_point_sampling(vectors, n=len(vectors), seed=seed)
    return [names[i] for i in order]


# ---------------------------------------------------------------------------
# Layer 2: within-domain FPS helpers
# ---------------------------------------------------------------------------


def _load_dimension_embeddings(
    embeddings_data: dict[str, Any],
) -> dict[str, dict[str, list[float]]]:
    """从嵌入数据中加载各维度的 name→vector 映射表。

    Returns a dict with keys ``"capability"``, ``"language"``,
    ``"conversation_type"``, each mapping to ``{name: vector}``.
    """
    items = embeddings_data["items"]
    return {
        "capability": items["capabilities"],
        "language": items["languages"],
        "conversation_type": items["conversation_types"],
    }


def _get_domain_embedding(
    domain: str,
    embeddings_data: dict[str, Any],
) -> np.ndarray:
    """获取知识域的嵌入向量。

    Args:
        domain: Knowledge domain category name (e.g. ``"science_exploration"``).
        embeddings_data: Parsed embeddings data dict.

    Returns:
        1-D numpy array of shape ``(D,)``.
    """
    return np.array(
        embeddings_data["items"]["knowledge_domains"][domain],
        dtype=np.float64,
    )


def _compose_combination_vectors(
    pool: list[dict[str, Any]],
    dim_embeddings: dict[str, dict[str, list[float]]],
    domain_emb: np.ndarray,
) -> np.ndarray:
    """为域内每个组合拼接四维嵌入。

    Concatenates ``[domain_emb, cap_emb, lang_emb, conv_emb]`` for each
    combination in *pool*, producing a ``(len(pool), 4*D)`` matrix.

    Args:
        pool: List of combination dicts with keys ``capability``,
            ``language``, ``conversation_type``.
        dim_embeddings: Dimension embedding tables from
            :func:`_load_dimension_embeddings`.
        domain_emb: 1-D domain embedding vector.

    Returns:
        Array of shape ``(len(pool), 4*D)``.
    """
    vectors: list[np.ndarray] = []
    for combo in pool:
        cap_vec = np.array(
            dim_embeddings["capability"][combo["capability"]],
            dtype=np.float64,
        )
        lang_vec = np.array(
            dim_embeddings["language"][combo["language"]],
            dtype=np.float64,
        )
        conv_vec = np.array(
            dim_embeddings["conversation_type"][combo["conversation_type"]],
            dtype=np.float64,
        )
        combo_vec = np.concatenate([domain_emb, cap_vec, lang_vec, conv_vec])
        vectors.append(combo_vec)
    return np.array(vectors, dtype=np.float64)


# ---------------------------------------------------------------------------
# Hierarchical FPS sampler
# ---------------------------------------------------------------------------


def _sample_farthest(
    ontology: dict[str, Any],
    config: Any,  # AnchorGenerationConfig
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Hierarchical FPS: Layer 1 selects diverse domains, Layer 2 balances
    within each domain using composed 4096-dim embedding vectors.

    Steps:
    1. Validate embeddings, load dimension tables.
    2. Get FPS-ordered domain list via :func:`_farthest_domain_order`.
    3. Build all ontology combinations, group by domain.
    4. Distribute ``target_count`` evenly across the ordered domains.
    5. Within each domain, compose 4-part vectors and select via FPS.
    6. Fill any remainder from leftover combinations using global FPS.

    Args:
        ontology: Loaded ontology dict.
        config: Generation configuration.  Requires ``embeddings_path``
            and ``target_count`` attributes.
        rng: Seeded :class:`random.Random` instance (used only for
            final trimming when ``len(result) > target_count``).

    Returns:
        List of sampled anchor meta dicts.

    Raises:
        ValueError: If embeddings validation fails or no domains overlap.
    """
    from ard.core.sampler import _build_all_combinations  # lazy import

    embeddings_path: str = config.embeddings_path

    # Validate and load embeddings
    embeddings_data = validate_embeddings_for_ontology(embeddings_path, ontology)
    dim_embeddings = _load_dimension_embeddings(embeddings_data)

    # Layer 1: domain ordering
    domain_order = _farthest_domain_order(embeddings_path, config.seed, ontology)

    # Build all combinations
    combos = _build_all_combinations(ontology, config.languages, config.task_types)

    # Group combinations by knowledge domain
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for combo in combos:
        domain = combo["knowledge_domain"]
        by_domain.setdefault(domain, []).append(combo)

    # Only use domains that appear in both the FPS order and the ontology
    available_domains = [d for d in domain_order if d in by_domain]
    if not available_domains:
        raise ValueError(
            f"FPS domain order has no overlap with ontology domains. "
            f"FPS domains: {domain_order}, "
            f"Ontology domains: {list(by_domain.keys())}"
        )

    n_domains = len(available_domains)
    per_domain = max(1, config.target_count // n_domains)

    # Helper to produce a stable key for deduplication
    def _combo_key(c: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            c["language"],
            c["knowledge_domain"],
            c["capability"],
            c["conversation_type"],
        )

    result: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str, str, str]] = set()

    # Layer 2: per-domain FPS
    for domain in available_domains:
        pool = by_domain[domain]
        n = min(per_domain, len(pool))
        if n > 0:
            domain_emb = _get_domain_embedding(domain, embeddings_data)
            vectors = _compose_combination_vectors(pool, dim_embeddings, domain_emb)
            order = farthest_point_sampling(vectors, n=n, seed=config.seed)
            for idx in order:
                combo = pool[idx]
                result.append(combo)
                used_keys.add(_combo_key(combo))

    # Fill remainder from leftover combinations using global FPS
    if len(result) < config.target_count:
        remaining = [c for c in combos if _combo_key(c) not in used_keys]
        n_missing = config.target_count - len(result)
        if remaining:
            # Compose vectors for all remaining combinations
            rem_vectors: list[np.ndarray] = []
            for combo in remaining:
                domain = combo["knowledge_domain"]
                domain_emb = _get_domain_embedding(domain, embeddings_data)
                cap_vec = np.array(
                    dim_embeddings["capability"][combo["capability"]],
                    dtype=np.float64,
                )
                lang_vec = np.array(
                    dim_embeddings["language"][combo["language"]],
                    dtype=np.float64,
                )
                conv_vec = np.array(
                    dim_embeddings["conversation_type"][combo["conversation_type"]],
                    dtype=np.float64,
                )
                combo_vec = np.concatenate([domain_emb, cap_vec, lang_vec, conv_vec])
                rem_vectors.append(combo_vec)
            rem_vectors_arr = np.array(rem_vectors, dtype=np.float64)
            order = farthest_point_sampling(
                rem_vectors_arr,
                n=min(n_missing, len(remaining)),
                seed=config.seed,
            )
            for idx in order:
                result.append(remaining[idx])

    # Trim if we somehow exceeded target_count
    if len(result) > config.target_count:
        result = rng.sample(result, config.target_count)

    return result