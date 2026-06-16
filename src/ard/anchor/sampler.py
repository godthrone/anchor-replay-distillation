from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from ard.anchor.bank import AnchorPrompt
from ard.anchor.embeddings import OntologyEmbeddings
from ard.anchor.ontology import Ontology


SamplingStrategy = Literal["farthest", "balanced", "random"]

TECHNICAL_RESEARCH_DOMAINS = {
    "software_engineering",
    "systems_devops",
    "data_ai_ml",
    "math_logic",
    "science_engineering",
}
PRACTICAL_RESEARCH_DOMAINS = {
    "business_operations",
    "finance_economics",
    "law_policy_institutions",
    "medicine_health",
    "agent_tool_use",
}

MULTI_TURN_CONVERSATIONS = {
    "clarification_2_turn",
    "troubleshooting_3_turn",
    "iterative_revision_3_turn",
    "constraint_update_4_turn",
    "tool_assisted_multi_turn",
    "refusal_or_boundary_multi_turn",
}


@dataclass(slots=True)
class AnchorGenerationConfig:
    languages: list[str]
    task_types: list[str]
    count: int = 100
    seed: int = 42
    language_features_per_prompt: int = 2
    input_generator_model: str = "unspecified_input_generator_model"
    target_model: str = "unspecified_target_model"
    sampling_strategy: SamplingStrategy = "balanced"
    ontology_embeddings: OntologyEmbeddings | None = None
    ontology_sha256: str | None = None
    embedding_model: str | None = None
    embedding_distance: str | None = None
    system_personas: list[str] | None = None


def _group_by_top_level(leaves: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for leaf in leaves:
        grouped.setdefault(str(leaf.get("top_level", "default")), []).append(leaf)
    return grouped


def _language_dimensions(language_features: list[dict[str, Any]]) -> dict[str, str]:
    by_top = {
        str(feature.get("top_level", "")): str(feature.get("leaf", ""))
        for feature in language_features
    }
    return {
        "style": by_top.get("style", "concise"),
        "format": by_top.get("format", "paragraph"),
        "difficulty": by_top.get("difficulty", "intermediate"),
        "context_length": by_top.get("context_length", "short"),
        "noise": by_top.get("noise", "clean"),
        "answer_expectation": by_top.get("answer_expectation", "direct_answer"),
    }


def _bucket_for_domain(domain_key: str) -> str:
    if domain_key in TECHNICAL_RESEARCH_DOMAINS:
        return "technical_research"
    if domain_key in PRACTICAL_RESEARCH_DOMAINS:
        return "practical_research"
    return "general_research"


def _is_multi_turn(conversation_type: str) -> bool:
    return conversation_type in MULTI_TURN_CONVERSATIONS or not conversation_type.startswith(
        "single_turn"
    )


def _leaf_key(leaf: dict[str, Any]) -> str:
    return str(leaf["full_path"])


def _embedding_key(path: list[str], leaf: str) -> str:
    return " -> ".join([*path, leaf])


def _cosine_distance(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    return 1.0 - dot / (left_norm * right_norm)


def _farthest_domain_order(
    leaves: list[dict[str, Any]],
    embeddings: OntologyEmbeddings | None,
    seed: int,
) -> list[dict[str, Any]]:
    if embeddings is None:
        raise ValueError(
            "farthest sampling requires ontology embeddings. "
            "Run `ard ontology-embed --ontology <path> --output <embeddings>` "
            "or use `--sampling-strategy balanced`."
        )
    vector_by_key = {
        _embedding_key(item.path, item.leaf): item.embedding
        for item in embeddings.items_for_section("knowledge_domains")
    }
    candidates = [leaf for leaf in leaves if _leaf_key(leaf) in vector_by_key]
    if len(candidates) != len(leaves):
        raise ValueError(
            "ontology embeddings do not cover all knowledge domain leaves. "
            "Run `ard ontology-embed --ontology <path> --output <embeddings>`."
        )
    rng = random.Random(seed)
    remaining = candidates[:]
    first = rng.choice(remaining)
    order = [first]
    remaining.remove(first)
    nearest_distance = {
        _leaf_key(leaf): _cosine_distance(
            vector_by_key[_leaf_key(leaf)], vector_by_key[_leaf_key(first)]
        )
        for leaf in remaining
    }
    while remaining:
        best = max(
            remaining,
            key=lambda leaf: (nearest_distance[_leaf_key(leaf)], str(leaf["full_path"])),
        )
        order.append(best)
        remaining.remove(best)
        best_vector = vector_by_key[_leaf_key(best)]
        for leaf in remaining:
            key = _leaf_key(leaf)
            nearest_distance[key] = min(
                nearest_distance[key],
                _cosine_distance(vector_by_key[key], best_vector),
            )
    return order


def _least_used(
    leaves: list[dict[str, Any]],
    leaf_counts: Counter[str],
    top_counts: Counter[str],
    rng: random.Random,
) -> dict[str, Any]:
    min_top = min(top_counts[str(leaf.get("top_level", ""))] for leaf in leaves)
    top_candidates = [
        leaf for leaf in leaves if top_counts[str(leaf.get("top_level", ""))] == min_top
    ]
    min_leaf = min(leaf_counts[_leaf_key(leaf)] for leaf in top_candidates)
    leaf_candidates = [leaf for leaf in top_candidates if leaf_counts[_leaf_key(leaf)] == min_leaf]
    return rng.choice(sorted(leaf_candidates, key=lambda leaf: str(leaf["full_path"])))


def _least_used_value(values: list[str], counts: Counter[str], rng: random.Random) -> str:
    min_count = min(counts[value] for value in values)
    candidates = [value for value in values if counts[value] == min_count]
    return rng.choice(sorted(candidates))


def _sample_language_features(
    language: Ontology,
    feature_counts: Counter[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    grouped = _group_by_top_level(language.leaves)
    preferred = ["style", "format", "difficulty", "context_length", "noise", "answer_expectation"]
    features: list[dict[str, Any]] = []
    for key in preferred:
        candidates = grouped.get(key)
        if not candidates:
            continue
        chosen = _least_used(candidates, feature_counts, Counter(), rng)
        feature_counts[_leaf_key(chosen)] += 1
        features.append(chosen)
    return features


def build_anchor_prompt(
    knowledge_leaf: dict[str, Any],
    language_features: list[dict[str, Any]],
    language: str,
    task_type: str,
    conversation_type: str = "single_turn",
    input_generator_model: str = "unspecified_input_generator_model",
    target_model: str = "unspecified_target_model",
) -> str:
    features = "; ".join(feature["full_path"] for feature in language_features)
    dims = _language_dimensions(language_features)
    domain = knowledge_leaf["full_path"]
    turn_instruction = (
        "Generate one realistic user message only."
        if not _is_multi_turn(conversation_type)
        else (
            "Generate a realistic multi-turn conversation context as a JSON array of messages. "
            "Messages must alternate user and assistant where applicable, and the final message must be from user. "
            "Do not include the final assistant answer."
        )
    )
    return (
        f"You are the input generator model '{input_generator_model}' creating realistic user-side input for an ARD anchor. "
        f"The target model '{target_model}' will later answer this input to produce the supervised target. "
        f"Create the user-side prompt in {language}. "
        f"Domain: {domain}. Capability: {task_type}. "
        f"Conversation type: {conversation_type}. Difficulty: {dims['difficulty']}. "
        f"Context length: {dims['context_length']}. Noise: {dims['noise']}. "
        f"Style: {dims['style']}. Output format expectation: {dims['format']}. "
        f"Answer expectation: {dims['answer_expectation']}. "
        f"Additional language features: {features}. "
        f"{turn_instruction}"
    )


def generate_anchor_prompts(
    knowledge: Ontology,
    language: Ontology,
    capability: Ontology,
    conversation: Ontology,
    config: AnchorGenerationConfig,
) -> list[AnchorPrompt]:
    if not knowledge.leaves:
        raise ValueError("knowledge ontology has no leaves")
    if not language.leaves:
        raise ValueError("language ontology has no leaves")
    if not capability.leaves:
        raise ValueError("capability ontology has no leaves")
    if not conversation.leaves:
        raise ValueError("conversation ontology has no leaves")

    rng = random.Random(config.seed)
    allowed = set(config.task_types)
    capability_leaves = [leaf for leaf in capability.leaves if str(leaf.get("leaf", "")) in allowed]
    if not capability_leaves:
        raise ValueError(
            "task_types did not match any capability leaves: " + ", ".join(config.task_types)
        )

    domain_order = (
        _farthest_domain_order(knowledge.leaves, config.ontology_embeddings, config.seed)
        if config.sampling_strategy == "farthest"
        else []
    )
    domain_leaf_counts: Counter[str] = Counter()
    domain_top_counts: Counter[str] = Counter()
    capability_leaf_counts: Counter[str] = Counter()
    capability_top_counts: Counter[str] = Counter()
    conversation_leaf_counts: Counter[str] = Counter()
    conversation_top_counts: Counter[str] = Counter()
    system_persona_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    prompts: list[AnchorPrompt] = []

    for idx in range(config.count):
        if config.sampling_strategy == "farthest":
            knowledge_leaf = domain_order[idx % len(domain_order)]
        elif config.sampling_strategy == "random":
            knowledge_leaf = rng.choice(knowledge.leaves)
        else:
            knowledge_leaf = _least_used(
                knowledge.leaves, domain_leaf_counts, domain_top_counts, rng
            )
        domain_key = str(knowledge_leaf["top_level"])
        domain_leaf_counts[_leaf_key(knowledge_leaf)] += 1
        domain_top_counts[domain_key] += 1

        output_language = _least_used_value(config.languages, language_counts, rng)
        language_counts[output_language] += 1
        language_features = _sample_language_features(language, feature_counts, rng)
        language_dims = _language_dimensions(language_features)

        capability_leaf = _least_used(
            capability_leaves, capability_leaf_counts, capability_top_counts, rng
        )
        capability_leaf_counts[_leaf_key(capability_leaf)] += 1
        capability_top_counts[str(capability_leaf.get("top_level", ""))] += 1

        conversation_leaf = _least_used(
            conversation.leaves, conversation_leaf_counts, conversation_top_counts, rng
        )
        conversation_leaf_counts[_leaf_key(conversation_leaf)] += 1
        conversation_top_counts[str(conversation_leaf.get("top_level", ""))] += 1

        capability_name = str(capability_leaf["leaf"])
        conversation_type = str(conversation_leaf["leaf"])

        if config.system_personas:
            system_persona = rng.choice(config.system_personas)
            system_persona_counts[system_persona] += 1
        else:
            system_persona = "none"

        prompt = build_anchor_prompt(
            knowledge_leaf=knowledge_leaf,
            language_features=language_features,
            language=output_language,
            task_type=capability_name,
            conversation_type=conversation_type,
            input_generator_model=config.input_generator_model,
            target_model=config.target_model,
        )
        meta = {
            "language": output_language,
            "knowledge_domain": knowledge_leaf["full_path"],
            "knowledge_top_level": domain_key,
            "capability": capability_name,
            "capability_top_level": capability_leaf.get("top_level", "unknown"),
            "task_type": capability_name,
            "conversation_type": conversation_type,
            "conversation_top_level": conversation_leaf.get("top_level", "unknown"),
            "is_multi_turn": _is_multi_turn(conversation_type),
            "sampling_bucket": _bucket_for_domain(domain_key),
            "sampling_strategy": config.sampling_strategy,
            "ontology_sha256": config.ontology_sha256,
            "embedding_model": config.embedding_model,
            "embedding_distance": config.embedding_distance,
            "language_features": [feature["full_path"] for feature in language_features],
            **language_dims,
            "input_generator_model": config.input_generator_model,
            "target_model": config.target_model,
            "source": "two_hop_anchor_generator",
            "system_persona": system_persona,
            "seed": config.seed,
        }
        prompts.append(AnchorPrompt.from_prompt(prompt=prompt, meta=meta, salt=str(config.seed)))

    return prompts
