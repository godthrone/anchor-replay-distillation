from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ard.anchor.bank import AnchorPrompt
from ard.anchor.ontology import Ontology


DEFAULT_TASK_TYPES = (
    "qa",
    "explanation",
    "reasoning",
    "coding",
    "translation",
    "summarization",
    "tool_use",
)

DEFAULT_LANGUAGES = ("English", "简体中文", "bilingual_zh_en")

CODE_MATH_REASONING_DOMAINS = {
    "software_engineering",
    "systems_devops",
    "data_ai_ml",
    "math_logic",
}
BUSINESS_AGENT_DOMAINS = {"business_operations", "agent_tool_use"}
SAFETY_DOMAINS = {"finance_economics", "law_policy_safety", "medicine_health_safety"}

SAFETY_CAPABILITIES = {"refusal_safety", "uncertainty_handling", "ask_clarification"}
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
    count: int = 100
    seed: int = 42
    languages: list[str] = field(default_factory=lambda: list(DEFAULT_LANGUAGES))
    task_types: list[str] = field(default_factory=lambda: list(DEFAULT_TASK_TYPES))
    language_features_per_prompt: int = 2
    input_generator_model: str = "unspecified_input_generator_model"
    target_model: str = "unspecified_target_model"


def _group_by_top_level(leaves: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for leaf in leaves:
        grouped.setdefault(str(leaf.get("top_level", "default")), []).append(leaf)
    return grouped


def _leaf_from_value(value: str, top_level: str = "default") -> dict[str, Any]:
    return {
        "full_path": value,
        "top_level": top_level,
        "path": [top_level],
        "leaf": value,
    }


def _fallback_ontology(values: list[str], top_level: str) -> Ontology:
    return Ontology(leaves=[_leaf_from_value(value, top_level) for value in values])


def _leaves_by_name(
    ontology: Ontology | None, fallback_values: list[str], top_level: str
) -> list[dict[str, Any]]:
    if ontology and ontology.leaves:
        return ontology.leaves
    return _fallback_ontology(fallback_values, top_level).leaves


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


def _sample_language_features(
    language: Ontology,
    rng: random.Random,
    fallback_count: int,
) -> list[dict[str, Any]]:
    grouped = _group_by_top_level(language.leaves)
    preferred = ["style", "format", "difficulty", "context_length", "noise", "answer_expectation"]
    features: list[dict[str, Any]] = []
    for key in preferred:
        if key in grouped:
            features.append(rng.choice(grouped[key]))
    if features:
        return features
    k = min(fallback_count, len(language.leaves))
    return rng.sample(language.leaves, k)


def _bucket_for_domain(domain_key: str) -> str:
    if domain_key in CODE_MATH_REASONING_DOMAINS:
        return "code_math_reasoning"
    if domain_key in BUSINESS_AGENT_DOMAINS:
        return "business_agent"
    if domain_key in SAFETY_DOMAINS:
        return "safety_refusal"
    return "general"


def _target_bucket(idx: int) -> str:
    # 60% general, 20% code/math/reasoning, 15% business/agent, 5% safety.
    slot = idx % 20
    if slot < 12:
        return "general"
    if slot < 16:
        return "code_math_reasoning"
    if slot < 19:
        return "business_agent"
    return "safety_refusal"


def _conversation_target(idx: int) -> str:
    # 70% single-turn, 30% multi-turn: 10% clarification, 10% troubleshooting,
    # 5% revision, 5% safety/tool-assisted.
    slot = idx % 20
    if slot < 14:
        return "single_turn"
    if slot < 16:
        return "clarification"
    if slot < 18:
        return "troubleshooting"
    if slot == 18:
        return "revision"
    return "tool_and_safety"


def _pick_from_target(
    grouped: dict[str, list[dict[str, Any]]],
    target: str,
    rng: random.Random,
) -> dict[str, Any]:
    candidates = grouped.get(target)
    if candidates:
        return rng.choice(candidates)
    all_candidates = [leaf for leaves in grouped.values() for leaf in leaves]
    return rng.choice(all_candidates)


def _pick_domain(
    grouped_domains: dict[str, list[dict[str, Any]]],
    idx: int,
    rng: random.Random,
) -> tuple[str, dict[str, Any], str]:
    target = _target_bucket(idx)
    domain_keys = sorted(grouped_domains)
    target_keys = [key for key in domain_keys if _bucket_for_domain(key) == target]
    if not target_keys:
        target_keys = domain_keys
    domain_key = target_keys[idx % len(target_keys)]
    return target, rng.choice(grouped_domains[domain_key]), domain_key


def _pick_capability(
    capability: Ontology | None,
    config: AnchorGenerationConfig,
    idx: int,
    bucket: str,
    rng: random.Random,
) -> dict[str, Any]:
    leaves = _leaves_by_name(capability, list(config.task_types), "capability")
    grouped = _group_by_top_level(leaves)
    if bucket == "code_math_reasoning":
        preferred = ["reasoning", "coding_and_data"]
    elif bucket == "business_agent":
        preferred = ["agentic_behavior", "knowledge_response"]
    elif bucket == "safety_refusal":
        preferred = ["agentic_behavior"]
    else:
        preferred = ["knowledge_response", "language_work", "reasoning"]
    for key in preferred:
        if key in grouped:
            return rng.choice(grouped[key])
    return leaves[idx % len(leaves)]


def _pick_conversation(
    conversation: Ontology | None, idx: int, rng: random.Random
) -> dict[str, Any]:
    leaves = _leaves_by_name(conversation, ["single_turn"], "single_turn")
    grouped = _group_by_top_level(leaves)
    return _pick_from_target(grouped, _conversation_target(idx), rng)


def _pick_safety(
    safety: Ontology | None,
    domain_key: str,
    capability_leaf: str,
    conversation_leaf: str,
    idx: int,
    rng: random.Random,
) -> dict[str, Any]:
    leaves = _leaves_by_name(safety, ["standard_helpful_answer"], "normal")
    grouped = _group_by_top_level(leaves)
    if domain_key in SAFETY_DOMAINS:
        target = "regulated_domain"
    elif capability_leaf in SAFETY_CAPABILITIES or "refusal" in conversation_leaf:
        target = "boundary"
    elif idx % 20 == 19:
        target = "boundary"
    else:
        target = "normal"
    return _pick_from_target(grouped, target, rng)


def _is_multi_turn(conversation_type: str) -> bool:
    return conversation_type in MULTI_TURN_CONVERSATIONS or not conversation_type.startswith(
        "single_turn"
    )


def build_anchor_prompt(
    knowledge_leaf: dict[str, Any],
    language_features: list[dict[str, Any]],
    language: str,
    task_type: str,
    conversation_type: str = "single_turn",
    safety_boundary: str = "standard_helpful_answer",
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
    safety_instruction = (
        "For medical, legal, financial, unsafe, or uncertain situations, keep the request within safe, "
        "non-authoritative guidance boundaries and avoid asking for dangerous procedural details."
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
        f"Safety boundary: {safety_boundary}. {safety_instruction} "
        f"{turn_instruction}"
    )


def generate_anchor_prompts(
    knowledge: Ontology,
    language: Ontology,
    config: AnchorGenerationConfig,
    capability: Ontology | None = None,
    conversation: Ontology | None = None,
    safety: Ontology | None = None,
) -> list[AnchorPrompt]:
    if not knowledge.leaves:
        raise ValueError("knowledge ontology has no leaves")
    if not language.leaves:
        raise ValueError("language ontology has no leaves")

    rng = random.Random(config.seed)
    grouped_domains = _group_by_top_level(knowledge.leaves)
    prompts: list[AnchorPrompt] = []

    for idx in range(config.count):
        bucket, knowledge_leaf, domain_key = _pick_domain(grouped_domains, idx, rng)
        language_features = _sample_language_features(
            language, rng, config.language_features_per_prompt
        )
        language_dims = _language_dimensions(language_features)
        output_language = config.languages[idx % len(config.languages)]
        capability_leaf = _pick_capability(capability, config, idx, bucket, rng)
        conversation_leaf = _pick_conversation(conversation, idx, rng)
        capability_name = str(capability_leaf["leaf"])
        conversation_type = str(conversation_leaf["leaf"])
        safety_leaf = _pick_safety(
            safety=safety,
            domain_key=domain_key,
            capability_leaf=capability_name,
            conversation_leaf=conversation_type,
            idx=idx,
            rng=rng,
        )
        safety_boundary = str(safety_leaf["leaf"])
        prompt = build_anchor_prompt(
            knowledge_leaf=knowledge_leaf,
            language_features=language_features,
            language=output_language,
            task_type=capability_name,
            conversation_type=conversation_type,
            safety_boundary=safety_boundary,
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
            "safety_boundary": safety_boundary,
            "safety_top_level": safety_leaf.get("top_level", "unknown"),
            "sampling_bucket": bucket,
            "language_features": [feature["full_path"] for feature in language_features],
            **language_dims,
            "input_generator_model": config.input_generator_model,
            "target_model": config.target_model,
            "source": "two_hop_anchor_generator",
            "seed": config.seed,
        }
        prompts.append(AnchorPrompt.from_prompt(prompt=prompt, meta=meta, salt=str(config.seed)))

    return prompts
