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
from ard.core.system_prompt import (
    SYSTEM_PROMPT_STYLE_ABSENT,
    SYSTEM_PROMPT_STYLE_ORDER,
    named_style_vector,
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_embeddings_for_ontology(
    embeddings_path: str,
    ontology: dict[str, Any],
) -> dict[str, Any]:
    """Eager validation: 10 项校验。失败抛 ValueError。成功返回解析后的嵌入数据。

    Checks performed:
    V1: File exists
    V2: Valid JSON
    V3: ``items`` non-empty
    V4: ``knowledge_domains`` section present
    V5: Consistent embedding dimension across all sections
    V6: Every KD embedding key exists in ontology, and vice versa
    V7: Intersection non-empty
    V8: ``items`` is a dict (new format), not a list (old format)
    V9: ``system_prompt`` section present (v3.0.0 sampling dimension)
    V10: Every system-prompt value in the ontology has an embedding

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

    # V9/V10: the system-prompt sampling dimension must have vectors.  Checked
    # here (not at first use) because the failure mode otherwise is a KeyError
    # deep inside the FPS loop, far away from the misconfiguration.
    system_prompt_vectors = items.get("system_prompt", {})
    if not system_prompt_vectors:
        raise ValueError(
            "Embeddings missing 'system_prompt' section — regenerate it with "
            "scripts/generate_ontology_embeddings.py"
        )
    from ard.core.system_prompt import get_system_prompt_values  # lazy import

    required_values = {
        value
        for pair in get_system_prompt_values(ontology)
        for value in pair
    }
    missing_values = required_values - set(system_prompt_vectors)
    if missing_values:
        raise ValueError(
            f"Ontology system-prompt values missing embeddings: {missing_values}"
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
) -> dict[str, Any]:
    """从嵌入数据中加载各维度的 name→vector 映射表。

    Returns a dict with keys ``"capability"``, ``"language"``,
    ``"conversation_type"``, ``"system_prompt"`` (each mapping to
    ``{name: vector}``) and ``"system_prompt_style_order"`` (the style names,
    which is what positions the one-hot style slot).

    ``system_prompt`` covers the two *presence* values of the system-prompt
    dimension (see ``ontology/anchor_ontology.json``).  Their vectors are
    derived from the capability and style vectors by
    ``scripts/generate_ontology_embeddings.derive_system_prompt_embeddings``
    because the values have no embeddable text of their own — one of them *is*
    the absence of a system prompt.  Deriving them from real ontology vectors
    keeps the dimension in the same space as the ones FPS already uses (§1.4
    单一真相源: the derivation has exactly one implementation, which both builds
    the file and checks it).

    Raises:
        KeyError: If a section is missing — the caller's validation reports
            that with a readable message before this point.
    """
    items = embeddings_data["items"]
    return {
        "capability": items["capabilities"],
        "language": items["languages"],
        "conversation_type": items["conversation_types"],
        "system_prompt": items["system_prompt"],
        "system_prompt_style_order": list(SYSTEM_PROMPT_STYLE_ORDER),
    }


def _unit(vector: Any) -> np.ndarray:
    """Return *vector* as a unit-length float64 array.

    Every dimension enters the composed vector at **equal weight**.  Concatenating
    raw vectors instead would weight each dimension by its own norm, which is
    arbitrary (an artifact of the embedding model's scale, not of the design),
    and it let the knowledge-domain vector — whose norm dominates — drown out
    the differences this layer exists to spread.  Normalising makes the
    dimensions comparable, which is what "diversity over the ontology" means
    (§1.3: the rule is part of the computation, not a data accident).

    Args:
        vector: Any sequence of floats (embedding, list, or ndarray).

    Returns:
        Array of shape ``(D,)`` with L2 norm 1.

    Raises:
        ValueError: If the vector has zero norm (a zero vector has no
            direction, so it cannot be compared by cosine distance).
    """
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("cannot normalise a zero-length vector")
    return array / norm


def _build_dimension_unit_table(
    dim_embeddings: dict[str, Any],
    embeddings_data: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Normalise every ontology value once, and index the domain vectors by name.

    Composing 50k combinations used to convert and normalise the same ~55
    embedding lists over and over (once per combination).  The vectors are
    properties of the ontology, not of a combination, so they are normalised
    once here and only *looked up* afterwards — the same numbers, without the
    repeated work.

    Args:
        dim_embeddings: Dimension embedding tables from
            :func:`_load_dimension_embeddings`.
        embeddings_data: Parsed embeddings data (for the domain vectors).

    Returns:
        ``(unit_table, domain_vectors)`` where *unit_table* maps
        ``"capability"`` / ``"language"`` / ``"conversation_type"`` /
        ``"presence"`` / ``"style"`` to ``{value: unit vector}`` and
        *domain_vectors* maps each knowledge domain to its unit vector.
    """
    presence_dimension = next(
        iter(dim_embeddings["system_prompt"].values())
    ).__len__()
    return (
        {
            "capability": {
                name: _unit(vector)
                for name, vector in dim_embeddings["capability"].items()
            },
            "language": {
                name: _unit(vector)
                for name, vector in dim_embeddings["language"].items()
            },
            "conversation_type": {
                name: _unit(vector)
                for name, vector in dim_embeddings["conversation_type"].items()
            },
            "presence": {
                name: _unit(vector)
                for name, vector in dim_embeddings["system_prompt"].items()
            },
            "style": {
                name: named_style_vector(
                    name,
                    dimension=presence_dimension,
                    n_styles=len(dim_embeddings["system_prompt_style_order"]),
                )
                for name in (SYSTEM_PROMPT_STYLE_ABSENT,)
                + tuple(dim_embeddings["system_prompt_style_order"])
            },
        },
        {
            name: _unit(vector)
            for name, vector in embeddings_data["items"]["knowledge_domains"].items()
        },
    )


def _compose_one_vector(
    combo: dict[str, Any],
    dim_embeddings: dict[str, Any],
    domain_emb: np.ndarray,
) -> np.ndarray:
    """Compose the embedding vector of a single combination.

    The single implementation of combination-vector composition: the
    whole-ontology precomputation (:func:`_compose_combination_vectors`) builds
    its matrix through it, so the sampler can never disagree with itself about
    which dimensions an anchor is separated on (§1.4 单一真相源).

    The last two slots are the two system-prompt dimensions:

    * **presence** uses the embeddings file's ``none`` / ``present`` vectors.
      ``none`` is the direction *away from every style* (see
      :func:`ard.core.system_prompt.style_centroid_direction`), so "no system
      prompt" is maximally distinguishable from any style instead of sitting
      among them.
    * **style** is a categorical one-hot over the real styles, and zero for the
      absent case — orthogonal by construction, so two anchors that differ only
      in style are genuinely far apart (see
      :func:`ard.core.system_prompt.named_style_vector`).

    Args:
        combo: One combination dict.
        dim_embeddings: Dimension embedding tables from
            :func:`_load_dimension_embeddings`.
        domain_emb: 1-D domain embedding vector.

    Returns:
        1-D array of shape ``(6 * D,)``.

    Raises:
        KeyError: If a value has no vector — a combination naming an unknown
            capability / language / conversation type / system-prompt value
            must fail loudly rather than be silently dropped (§2.3).
    """
    cap_vec = _unit(dim_embeddings["capability"][combo["capability"]])
    lang_vec = _unit(dim_embeddings["language"][combo["language"]])
    conv_vec = _unit(dim_embeddings["conversation_type"][combo["conversation_type"]])
    presence_vec = _unit(
        dim_embeddings["system_prompt"][combo["system_prompt_presence"]]
    )
    style_vec = named_style_vector(
        combo["system_prompt_style"],
        dimension=presence_vec.shape[0],
        n_styles=len(dim_embeddings["system_prompt_style_order"]),
    )
    return np.concatenate(
        [_unit(domain_emb), cap_vec, lang_vec, conv_vec, presence_vec, style_vec]
    )


def _compose_combination_vectors(
    pool: list[dict[str, Any]],
    dim_embeddings: dict[str, dict[str, list[float]]],
    domain_emb: np.ndarray,
) -> np.ndarray:
    """为域内每个组合拼接各维嵌入（域内一次性的便捷入口）。

    Concatenates the unit-normalised
    ``[domain, capability, language, conversation_type, presence, style]``
    slots for each combination in *pool*, producing a matrix of width
    ``6 * D`` where ``D`` is the embedding dimension.  The arrangement of the
    two system-prompt dimensions is described in
    :func:`_compose_one_vector`.

    :func:`_sample_farthest` does **not** call this per domain: it composes the
    whole ontology once through :func:`_build_dimension_unit_table`, because
    re-normalising the same dimension vectors for every domain was pure
    repeated work.

    Args:
        pool: List of combination dicts with keys ``capability``,
            ``language``, ``conversation_type``, ``system_prompt_presence``
            and ``system_prompt_style``.
        dim_embeddings: Dimension embedding tables from
            :func:`_load_dimension_embeddings`.
        domain_emb: 1-D domain embedding vector.

    Returns:
        Array of shape ``(len(pool), 6 * D)``.

    Raises:
        KeyError: If a value has no vector — a combination naming an unknown
            capability / language / conversation type / system-prompt value
            must fail loudly rather than be silently dropped (§2.3).
    """
    vectors: list[np.ndarray] = []
    for combo in pool:
        vectors.append(_compose_one_vector(combo, dim_embeddings, domain_emb))
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
    within each domain using composed embedding vectors.

    Steps:
    1. Validate embeddings, load dimension tables.
    2. Get FPS-ordered domain list via :func:`_farthest_domain_order`.
    3. Build all ontology combinations, group by domain.
    4. Compose **every** combination's vector once (see below).
    5. Distribute ``target_count`` evenly across the ordered domains and run
       FPS inside each domain's slice.
    6. Fill any remainder from leftover combinations using global FPS.

    Step 4 composes the whole ontology up front instead of re-composing per
    domain: the capability / language / conversation-type / system-prompt
    vectors do not depend on the domain, so composing per domain normalised the
    same vectors again and again, and step 6 composed the leftovers a third
    time.  Both steps now only *slice* the precomputed matrix, which also
    guarantees that a combination looks identical to step 5 and step 6.

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

    # Group combinations by knowledge domain (index into ``combos``)
    by_domain: dict[str, list[int]] = {}
    for position, combo in enumerate(combos):
        by_domain.setdefault(combo["knowledge_domain"], []).append(position)

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

    # Step 4: compose every combination once.  Every dimension vector is
    # normalised once (in the unit table) and then only looked up, so the cost
    # does not grow with the number of combinations.  The slots are written into
    # a preallocated matrix slice by slice — six times cheaper than building a
    # fresh concatenated array per combination.
    unit_table, domain_vectors = _build_dimension_unit_table(
        dim_embeddings, embeddings_data
    )
    embedding_dimension = len(next(iter(unit_table["capability"].values())))
    slot_bounds = [(index * embedding_dimension, (index + 1) * embedding_dimension)
                   for index in range(6)]
    vectors = np.empty((len(combos), 6 * embedding_dimension), dtype=np.float64)
    for position, combo in enumerate(combos):
        slots = (
            domain_vectors[combo["knowledge_domain"]],
            unit_table["capability"][combo["capability"]],
            unit_table["language"][combo["language"]],
            unit_table["conversation_type"][combo["conversation_type"]],
            unit_table["presence"][combo["system_prompt_presence"]],
            unit_table["style"][combo["system_prompt_style"]],
        )
        for (start, end), vector in zip(slot_bounds, slots):
            vectors[position, start:end] = vector

    result: list[dict[str, Any]] = []
    # Bookkeeping is by *position* in ``combos``: two combinations are distinct
    # exactly when their positions differ, because one combination is built per
    # (language, domain, capability, conversation type, system-prompt mode)
    # tuple.  A key-based set of only the first four dimensions — what this used
    # to be — silently collapsed the system-prompt dimension and would drop
    # legitimate leftovers from the remainder fill.
    used_positions: set[int] = set()

    # Layer 2: per-domain FPS, over that domain's slice of the precomputed rows
    for domain in available_domains:
        positions = by_domain[domain]
        n = min(per_domain, len(positions))
        if n > 0:
            vectors_for_domain = vectors[np.array(positions)]
            order = farthest_point_sampling(vectors_for_domain, n=n, seed=config.seed)
            for idx in order:
                position = positions[idx]
                result.append(combos[position])
                used_positions.add(position)

    # Fill remainder from leftover combinations using global FPS
    if len(result) < config.target_count:
        remaining_positions = [
            position for position in range(len(combos)) if position not in used_positions
        ]
        n_missing = config.target_count - len(result)
        if remaining_positions:
            remaining_vectors = vectors[np.array(remaining_positions)]
            order = farthest_point_sampling(
                remaining_vectors,
                n=min(n_missing, len(remaining_positions)),
                seed=config.seed,
            )
            for idx in order:
                result.append(combos[remaining_positions[idx]])

    # Trim if we somehow exceeded target_count
    if len(result) > config.target_count:
        result = rng.sample(result, config.target_count)

    return result