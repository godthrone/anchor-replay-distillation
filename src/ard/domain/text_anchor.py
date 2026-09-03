"""Text anchor generation pipeline — simple per-sample loop."""

import random
from typing import Any, Optional

from ard.core.types import Anchor, AnchorGenerationConfig
from ard.core.sampler import sample_anchors, generate_anchor_id
from ard.backends.api_client import ChatAPIClient


def build_input_prompt(meta: dict[str, Any]) -> str:
    language = meta.get("language", "English")
    domain = meta.get("knowledge_domain", "general")
    capability = meta.get("capability", "qa")
    conv_type = meta.get("conversation_type", "single_turn")
    return (
        f"You are a helpful assistant simulating a real user. "
        f"Generate a single realistic user message in {language} "
        f"on the topic of {domain}. "
        f"The user is asking for a {capability} task. "
        f"The conversation style is {conv_type}. "
        f"Only output the user message, nothing else."
    )


def build_target_prompt(meta: dict[str, Any]) -> str:
    return (
        "You are a knowledgeable assistant. "
        "Answer the user's question accurately and helpfully."
    )


def generate_text_anchors(
    ontology: dict[str, Any],
    config: AnchorGenerationConfig,
    input_client: ChatAPIClient,
    target_client: ChatAPIClient,
    input_model_name: str,
    target_model_name: str,
    min_answer_chars: int = 8,
    max_answer_chars: Optional[int] = None,
) -> list[Anchor]:
    metas = sample_anchors(ontology, config)
    rng = random.Random(config.seed)
    rng.shuffle(metas)
    anchors: list[Anchor] = []

    for meta in metas:
        anchor_id = generate_anchor_id(meta)
        sys_prompt = build_input_prompt(meta)

        # Step 1: Generate user question via input_client
        try:
            user_msg = input_client.chat(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "Generate a user message as instructed."},
                ],
                temperature=0.7,
            )
        except Exception:
            continue
        user_msg = user_msg.strip()
        if not user_msg:
            continue

        # Step 2: Generate answer with logprobs via target_client
        try:
            result = target_client.chat_with_logprobs(
                [
                    {"role": "system", "content": build_target_prompt(meta)},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
        except Exception:
            continue

        target_answer = result["content"].strip()
        logprobs = result["logprobs"]

        if len(target_answer) < min_answer_chars:
            continue
        if max_answer_chars is not None and len(target_answer) > max_answer_chars:
            continue

        anchor = Anchor(
            id=anchor_id,
            messages=[{"role": "user", "content": user_msg}],
            target_answer=target_answer,
            target_model=target_model_name,
            input_generator_model=input_model_name,
            anchor_meta=meta,
            logprobs=logprobs,
        )
        anchors.append(anchor)

        if len(anchors) >= config.target_count:
            break

    return anchors