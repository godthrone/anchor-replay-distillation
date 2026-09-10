"""Text anchor generation pipeline — concurrent per-Anchorspec loop via ThreadPoolExecutor.

Phase 2: Two-step generation per turn (OPD role correction).
1. input_generator generates user message (teacher)
2. target_model generates assistant reply (student)
Last turn: target_model generates with logprobs.
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from tqdm import tqdm

from ard.backends.api_client import ARDTimeoutError, ChatAPIClient
from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec

logger = logging.getLogger(__name__)
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
    image_path: str | None = None,
    image_data_url: str | None = None,
) -> list[dict[str, Any]]:
    """Build prompt messages for the input generator to produce the next user turn.

    When an image is provided via *image_data_url*, the image is included as a
    multimodal ``image_url`` content part so the input generator can see the
    image and generate a question related to it.

    .. note::

        Image encoding must be done by the caller (``_generate_one_anchor``)
        **before** the API try/except block so that encoding failures propagate
        as fatal errors rather than being silently caught.

    Args:
        turn: Current turn specification with generation instruction.
        messages: Conversation history so far.
        anchor_meta: Anchor metadata (language, domain, capability, etc.).
        image_path: Optional path to an image file.  Kept for backward
            compatibility; image encoding is now handled by the caller.
            Ignored if *image_data_url* is provided.
        image_data_url: Optional pre-encoded base64 data URI for the image.
            This is the **only** way to include an image — the function no
            longer performs on-demand encoding.

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

    # Build instruction — when an image is provided, focus on the image content
    # Image encoding is done by the caller (_generate_one_anchor) before the
    # API try/except block so that encoding failures propagate as fatal errors.
    if image_data_url is not None:
        if turn.generation_instruction:
            instruction = turn.generation_instruction
        else:
            instruction = (
                f"Look at the image and generate a "
                f"{'follow-up ' if messages else ''}realistic user message in {language} "
                f"about what you see in the image. "
                f"The user is asking for a {capability} task. "
                f"The conversation style is {conv_type}."
            )
    else:
        image_data_url = None
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

    # Build user message content (may include image)
    if image_data_url is not None:
        user_content_parts: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
        if history_text:
            user_content_parts.append(
                {"type": "text", "text": f"Conversation so far:\n{history_text}\n\nGenerate the next user message as instructed."}
            )
        else:
            user_content_parts.append(
                {"type": "text", "text": "Generate a user message as instructed."}
            )
        user_content: str | list[dict[str, Any]] = user_content_parts
    else:
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


def _convert_images_to_paths(
    messages: list[dict[str, Any]],
    spec: AnchorSpec,
) -> list[dict[str, Any]]:
    """Convert base64 image_url to image type with relative path for output.

    API calls need base64-encoded images, but the output JSONL should use
    ``{"type": "image", "image": "images/xxx.jpg"}`` format (Graspo-compatible).
    """
    result: list[dict[str, Any]] = []
    msg_idx = 0
    for turn in spec.turns:
        if msg_idx >= len(messages):
            break
        msg = dict(messages[msg_idx])  # shallow copy
        if turn.image_path and isinstance(msg.get("content"), list):
            new_content: list[dict[str, Any]] = []
            for item in msg["content"]:
                if item.get("type") == "image_url":
                    rel = _abs_to_rel_path(turn.image_path)
                    new_content.append({"type": "image", "image": rel})
                else:
                    new_content.append(item)
            msg["content"] = new_content
        result.append(msg)
        msg_idx += 1  # user message
        if not turn.is_final and msg_idx < len(messages):
            result.append(messages[msg_idx])  # assistant message (no image)
            msg_idx += 1
    return result


def _abs_to_rel_path(abs_path: str) -> str:
    """Extract relative path (images/xxx.jpg) from absolute path."""
    idx = abs_path.rfind("/images/")
    if idx >= 0:
        return abs_path[idx + 1:]  # images/xxx.jpg
    return abs_path


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

    total_turns = len(spec.turns)
    for turn_idx, turn in enumerate(spec.turns):
        # Pre-encode image (if any) — encoding failures are fatal and
        # must NOT be caught by the API error handler below.
        image_data_url: str | None = None
        if turn.image_path:
            from ard.backends.api_client import encode_image_to_base64

            image_data_url = encode_image_to_base64(turn.image_path)

        # Step 1: input_generator generates the user message (teacher)
        # Pass the image to the input generator so it can generate a
        # question related to the image content (multimodal anchor).
        try:
            user_msg = input_client.chat(
                _build_user_prompt(
                    turn, messages, spec.anchor_meta,
                    image_path=turn.image_path,
                    image_data_url=image_data_url,
                ),
                temperature=0.7,
            )
        except ARDTimeoutError:
            if len(messages) <= 1:
                return None
            logger.warning(
                "Timeout in turn %d/%d, skipping this turn and continuing",
                turn_idx + 1, total_turns,
            )
            continue
        except Exception as exc:
            logger.warning("skipping anchor due to error: %s", exc)
            return None
        user_msg = user_msg.strip()
        if not user_msg:
            return None

        # Build message content (possibly with image for multimodal turns)
        msg_content: list[dict[str, Any]] = []
        if image_data_url is not None:
            msg_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
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
            except ARDTimeoutError:
                if len(messages) <= 1:
                    return None
                logger.warning(
                    "Timeout in turn %d/%d, skipping this turn and continuing",
                    turn_idx + 1, total_turns,
                )
                continue
            except Exception as exc:
                logger.warning("skipping anchor due to error: %s", exc)
                return None
            target_answer = result["content"].strip()
            logprobs = result["logprobs"]
            if len(target_answer) < min_answer_chars:
                return None
            if max_answer_chars is not None and len(target_answer) > max_answer_chars:
                return None
            return GeneratedAnchor(
                id=spec.id,
                messages=_convert_images_to_paths(messages, spec),
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
            except ARDTimeoutError:
                if len(messages) <= 1:
                    return None
                logger.warning(
                    "Timeout in turn %d/%d, skipping this turn and continuing",
                    turn_idx + 1, total_turns,
                )
                continue
            except Exception as exc:
                logger.warning("skipping anchor due to error: %s", exc)
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
    backpressure_threshold: int = 3,
    backpressure_cooldown: float = 60.0,
) -> list[GeneratedAnchor]:
    """Generate text anchors from a list of AnchorSpec objects.

    Each spec is processed concurrently via ThreadPoolExecutor.  Results
    are streamed to *output_path* (if provided) as they complete.

    Backpressure: when ``ARDTimeoutError`` is raised by any worker,
    a consecutive timeout counter is incremented.  After
    *backpressure_threshold* consecutive timeouts the pipeline pauses for
    *backpressure_cooldown* seconds to let vLLM recover from overload.

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
        backpressure_threshold: Consecutive timeout count that triggers a
            cooldown pause.  Default 3.
        backpressure_cooldown: Seconds to pause when backpressure triggers.
            Default 60.

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

    consecutive_timeouts = 0

    try:
        for future in as_completed(futures):
            try:
                anchor = future.result()
            except ARDTimeoutError:
                consecutive_timeouts += 1
                logger.warning(
                    "Timeout generating anchor (consecutive: %d/%d)",
                    consecutive_timeouts, backpressure_threshold,
                )
                pbar.update(1)
                if consecutive_timeouts >= backpressure_threshold:
                    logger.warning(
                        "Backpressure: %d consecutive timeouts, pausing %ds...",
                        consecutive_timeouts, backpressure_cooldown,
                    )
                    time.sleep(backpressure_cooldown)
                    consecutive_timeouts = 0
                continue

            if anchor is not None:
                anchors.append(anchor)
                if output_path is not None:
                    append_anchor(anchor, output_path)
                consecutive_timeouts = 0
            pbar.update(1)
            if len(anchors) >= target_count:
                break
    finally:
        pbar.close()
        executor.shutdown(wait=False, cancel_futures=True)

    return anchors