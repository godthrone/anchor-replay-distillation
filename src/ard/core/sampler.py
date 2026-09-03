"""Anchor sampling strategies for ARD.

Provides balanced sampling of ontology combinations.
Core layer — no network or API dependencies.
"""

import hashlib
import random
from typing import Any

from ard.core.types import AnchorGenerationConfig


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
        available_langs = [l for l in available_langs if l in languages]

    domains = ontology.get("knowledge_domains", {})
    caps = ontology.get("capabilities", {})

    all_caps: list[str] = []
    for category, leaves in caps.items():
        if isinstance(leaves, list):
            all_caps.extend(leaves)
    if task_types:
        all_caps = [c for c in all_caps if c in task_types]

    combinations: list[dict[str, Any]] = []
    for lang in available_langs:
        for domain_name in domains:
            for cap in all_caps:
                for conv_name in ["single_turn", "multi_turn"]:
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

    if len(result) > config.target_count:
        result = rng.sample(result, config.target_count)
    return result


def generate_anchor_id(meta: dict[str, Any]) -> str:
    """Generate a stable anchor ID from meta dict."""
    raw = (
        f"{meta.get('language', '')}|"
        f"{meta.get('knowledge_domain', '')}|"
        f"{meta.get('capability', '')}"
    )
    return "anchor_" + hashlib.sha256(raw.encode()).hexdigest()[:16]