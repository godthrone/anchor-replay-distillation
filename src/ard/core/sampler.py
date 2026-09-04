"""Anchor sampling strategies for ARD.

Provides balanced sampling of ontology combinations.
Core layer — no network or API dependencies.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from ard.core.types import AnchorGenerationConfig, AnchorSpec, TurnSpec


def sample_anchors(
    ontology: dict[str, Any],
    config: AnchorGenerationConfig,
) -> list[dict[str, Any]]:
    """Sample anchor prompts from the ontology using balanced strategy.

    Args:
        ontology: Loaded ontology dict.
        config: Generation configuration.

    Returns:
        List of sampled anchor meta dicts.
    """
    rng = random.Random(config.seed)
    return _sample_balanced(ontology, config, rng)


def _get_leaf_conversation_types(ontology: dict[str, Any]) -> list[str]:
    """Extract all leaf conversation type names from the ontology.

    The ontology ``conversation_types`` section maps categories to lists of
    leaf type names (e.g. ``"single_turn": ["single_turn"]``).  This helper
    collects every leaf value into a flat list.
    """
    conv_types = ontology.get("conversation_types", {})
    if not conv_types:
        return ["single_turn", "multi_turn"]
    leaves: list[str] = []
    for leaves_list in conv_types.values():
        if isinstance(leaves_list, list):
            leaves.extend(leaves_list)
    return leaves if leaves else ["single_turn", "multi_turn"]


def _build_all_combinations(
    ontology: dict[str, Any],
    languages: list[str],
    task_types: list[str],
) -> list[dict[str, Any]]:
    """Build all possible ontology combinations."""
    available_langs = ontology.get("languages", [])
    if not available_langs:
        available_langs = ["English"]
    if languages:
        available_langs = [lang for lang in available_langs if lang in languages]

    domains = ontology.get("knowledge_domains", {})
    caps = ontology.get("capabilities", {})

    all_caps: list[str] = []
    for category, leaves in caps.items():
        if isinstance(leaves, list):
            all_caps.extend(leaves)
    if task_types:
        all_caps = [c for c in all_caps if c in task_types]

    # Read conversation types dynamically from ontology
    conv_types = _get_leaf_conversation_types(ontology)

    combinations: list[dict[str, Any]] = []
    for lang in available_langs:
        for domain_name in domains:
            for cap in all_caps:
                for conv_name in conv_types:
                    combinations.append({
                        "language": lang,
                        "knowledge_domain": domain_name,
                        "capability": cap,
                        "conversation_type": conv_name,
                    })
    return combinations


def _sample_balanced(
    ontology: dict[str, Any],
    config: AnchorGenerationConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    combos = _build_all_combinations(ontology, config.languages, config.task_types)
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for combo in combos:
        domain = combo["knowledge_domain"]
        by_domain.setdefault(domain, []).append(combo)

    domains = list(by_domain.keys())
    per_domain = max(1, config.target_count // len(domains))

    result: list[dict[str, Any]] = []
    for domain in domains:
        pool = by_domain[domain]
        n = min(per_domain, len(pool))
        result.extend(rng.sample(pool, n))

    if len(result) < config.target_count:
        # Integer division may leave a shortfall (e.g. 100 // 18 = 5, 18 × 5 = 90).
        remaining = [c for c in combos if c not in result]
        n_missing = config.target_count - len(result)
        if remaining:
            result.extend(rng.sample(remaining, min(n_missing, len(remaining))))

    if len(result) > config.target_count:
        result = rng.sample(result, config.target_count)
    return result


def sample_anchor_specs(
    ontology: dict[str, Any],
    config: AnchorGenerationConfig,
    rng: random.Random,
) -> list[AnchorSpec]:
    """Sample AnchorSpec objects from the ontology.

    Builds a balanced sample of anchor meta dicts, then distributes turn
    counts across the sample using :func:`ard.core.quota.compute_turn_distribution`.

    Args:
        ontology: Loaded ontology dict.
        config: Generation configuration (must include ``max_turns``).
        rng: Seeded random.Random instance.

    Returns:
        List of :class:`AnchorSpec` objects ready for generation.
    """
    from ard.core.quota import compute_turn_distribution

    # Sample base meta dicts
    meta_dicts = _sample_balanced(ontology, config, rng)

    max_turns = getattr(config, "max_turns", 1)
    # Only odd turn counts are valid (last turn must be user).
    # Map: bucket index i → actual turn count = 2*i + 1 (1, 3, 5, ...)
    num_odd_buckets = (max_turns + 1) // 2
    raw_turn_counts = compute_turn_distribution(
        len(meta_dicts), num_odd_buckets, rng
    )
    # Build turn_counts indexed by actual turn count (1..max_turns)
    turn_counts: list[int] = [0] * max_turns
    for i in range(num_odd_buckets):
        actual_turns = 2 * i + 1
        turn_counts[actual_turns - 1] = raw_turn_counts[i]

    # Build AnchorSpec objects
    specs: list[AnchorSpec] = []
    meta_idx = 0
    for num_turns_minus_1, count in enumerate(turn_counts):
        num_turns = num_turns_minus_1 + 1
        for _ in range(count):
            if meta_idx >= len(meta_dicts):
                break
            meta = meta_dicts[meta_idx]
            meta_idx += 1

            # Build turns: user/assistant alternating, always ending with user.
            # Even turn counts were already merged into odd buckets above.
            turns: list[TurnSpec] = []
            for i in range(num_turns):
                role = "user" if i % 2 == 0 else "assistant"
                turn = TurnSpec(
                    turn_index=i,
                    role=role,
                    generation_instruction="",
                    is_final=(i == num_turns - 1),
                )
                turns.append(turn)

            spec = AnchorSpec(
                id=generate_anchor_id(meta),
                anchor_meta=meta,
                turns=turns,
                input_generator_id="",
            )
            specs.append(spec)

    return specs


def generate_anchor_id(meta: dict[str, Any]) -> str:
    """Generate a stable anchor ID from meta dict."""
    raw = (
        f"{meta.get('language', '')}|"
        f"{meta.get('knowledge_domain', '')}|"
        f"{meta.get('capability', '')}|"
        f"{meta.get('conversation_type', '')}"
    )
    return "anchor_" + hashlib.sha256(raw.encode()).hexdigest()[:16]