"""Text anchor generation pipeline — concurrent per-Anchorspec loop via ThreadPoolExecutor.

Phase 2: Two-step generation per turn (OPD role correction).
1. input_generator generates user message (teacher)
2. target_model generates assistant reply (student)
Last turn: target_model generates with logprobs.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ard.backends.api_client import ChatAPIClient
from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec
from ard.domain.bank import append_anchor


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


def _build_user_prompt(
    turn: TurnSpec,
    messages: list[dict[str, Any]],
    anchor_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build prompt messages for the input generator to produce the next user turn.

    Args:
        turn: Current turn specification with generation instruction.
        messages: Conversation history so far.
        anchor_meta: Anchor metadata (language, domain, capability, etc.).

    Returns:
        A list of message dicts to send to the input generator.
    """
    language = anchor_meta.get("language", "English")
    domain = anchor_meta.get("knowledge_domain", "general")
    capability = anchor_meta.get("capability", "qa")
    conv_type = anchor_meta.get("conversation_type", "single_turn")

    # Build conversation history text
    history_text = ""
    if messages:
        history_parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                # Extract text from multimodal content parts
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(text_parts)
            history_parts.append(f"{role}: {content}")
        history_text = "\n".join(history_parts)

    instruction = turn.generation_instruction or (
        f"Generate a {'follow-up ' if messages else ''}realistic user message in {language} "
        f"on the topic of {domain}. "
        f"The user is asking for a {capability} task. "
        f"The conversation style is {conv_type}."
    )

    system_prompt = (
        f"You are a helpful assistant simulating a real user. "
        f"{instruction} "
        f"Only output the user message, nothing else."
    )

    if history_text:
        user_content = (
            f"Conversation so far:\n{history_text}\n\n"
            f"Generate the next user message as instructed."
        )
    else:
        user_content = "Generate a user message as instructed."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _generate_one_anchor(
    spec: AnchorSpec,
    input_client: ChatAPIClient,
    target_client: ChatAPIClient,
    input_model_name: str,
    target_model_name: str,
    min_answer_chars: int = 8,
    max_answer_chars: int | None = None,
) -> GeneratedAnchor | None:
    """Generate one anchor from an AnchorSpec.

    Two-step generation per turn:
    1. input_generator generates user message (teacher)
    2. target_model generates assistant reply (student)

    Last turn: target_model generates with logprobs.

    Args:
        spec: Anchor specification with turns and metadata.
        input_client: API client for generating user messages.
        target_client: API client for generating target answers.
        input_model_name: Name of the input-generator model.
        target_model_name: Name of the target model.
        min_answer_chars: Minimum answer length in characters.
        max_answer_chars: Optional maximum answer length.

    Returns:
        A :class:`GeneratedAnchor` on success, or ``None`` if skipped.
    """
    messages: list[dict[str, Any]] = []

    for turn in spec.turns:
        # Step 1: input_generator generates the user message (teacher)
        try:
            user_msg = input_client.chat(
                _build_user_prompt(turn, messages, spec.anchor_meta),
                temperature=0.7,
            )
        except Exception as exc:
            print(f"Warning: skipping anchor due to error: {exc}", file=sys.stderr)
            return None
        user_msg = user_msg.strip()
        if not user_msg:
            return None

        # Build message content (possibly with image for multimodal turns)
        msg_content: list[dict[str, Any]] = []
        if turn.image_path:
            from ard.backends.api_client import encode_image_to_base64

            data_url = encode_image_to_base64(turn.image_path)
            msg_content.append({"type": "image_url", "image_url": {"url": data_url}})
        msg_content.append({"type": "text", "text": user_msg})
        messages.append(
            {
                "role": "user",
                "content": msg_content if len(msg_content) > 1 else user_msg,
            }
        )

        if turn.is_final:
            # Final turn: target_model generates with logprobs
            try:
                result = target_client.chat_with_logprobs(messages, temperature=0.0)
            except Exception as exc:
                print(f"Warning: skipping anchor due to error: {exc}", file=sys.stderr)
                return None
            target_answer = result["content"].strip()
            logprobs = result["logprobs"]
            if len(target_answer) < min_answer_chars:
                return None
            if max_answer_chars is not None and len(target_answer) > max_answer_chars:
                return None
            return GeneratedAnchor(
                id=spec.id,
                messages=messages,
                target_answer=target_answer,
                target_model=target_model_name,
                input_generator_model=input_model_name,
                anchor_meta=spec.anchor_meta,
                logprobs=logprobs,
            )
        else:
            # Intermediate turn: target_model generates without logprobs
            try:
                assist_msg = target_client.chat(messages, temperature=0.0)
            except Exception as exc:
                print(f"Warning: skipping anchor due to error: {exc}", file=sys.stderr)
                return None
            assist_msg = assist_msg.strip()
            if not assist_msg:
                return None
            messages.append({"role": "assistant", "content": assist_msg})

    return None  # Should not reach here (last turn always returns)


def generate_text_anchors(
    specs: list[AnchorSpec],
    input_client: ChatAPIClient,
    target_client: ChatAPIClient,
    input_model_name: str,
    target_model_name: str,
    concurrency: int = 4,
    min_answer_chars: int = 8,
    max_answer_chars: int | None = None,
    output_path: Path | None = None,
) -> list[GeneratedAnchor]:
    """Generate text anchors from a list of AnchorSpec objects.

    Each spec is processed concurrently via ThreadPoolExecutor.  Results
    are streamed to *output_path* (if provided) as they complete.

    Args:
        specs: List of :class:`AnchorSpec` objects to generate.
        input_client: API client for generating user questions.
        target_client: API client for generating target answers.
        input_model_name: Name of the input-generator model.
        target_model_name: Name of the target model.
        concurrency: Maximum number of concurrent generations.
        min_answer_chars: Minimum answer length in characters.
        max_answer_chars: Optional maximum answer length.
        output_path: If provided, each anchor is appended to this JSONL
            file immediately after generation (streaming write).

    Returns:
        List of generated :class:`GeneratedAnchor` objects.
    """
    target_count = len(specs)
    anchors: list[GeneratedAnchor] = []

    pbar = tqdm(total=target_count, desc="Text anchors", unit="anchor")

    executor = ThreadPoolExecutor(max_workers=concurrency)
    futures = [
        executor.submit(
            _generate_one_anchor,
            spec,
            input_client,
            target_client,
            input_model_name,
            target_model_name,
            min_answer_chars,
            max_answer_chars,
        )
        for spec in specs
    ]
    try:
        for future in as_completed(futures):
            anchor = future.result()
            if anchor is not None:
                anchors.append(anchor)
                if output_path is not None:
                    append_anchor(anchor, output_path)
                pbar.update(1)
                if len(anchors) >= target_count:
                    break
    finally:
        pbar.close()
        executor.shutdown(wait=False, cancel_futures=True)

    return anchors