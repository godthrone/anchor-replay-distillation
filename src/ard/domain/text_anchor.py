"""Text anchor generation pipeline — concurrent per-Anchorspec loop via ThreadPoolExecutor.

Role-driven generation: every :class:`TurnSpec` yields exactly one message.
1. ``role == "user"`` → input_generator generates the user message (teacher).
2. ``role == "assistant"`` → target_model generates the assistant reply.

The final turn is always a ``user`` turn (guaranteed by
:mod:`ard.core.sampler`): there the target model answers, and — when thinking is
enabled for it — its reasoning trace is kept as a separate field, because
thinking is not the answer (``targets[0].output.content`` vs
``targets[0].output.reasoning``).  The message list therefore always starts
with ``user``, ends with ``user`` and alternates — the same invariant
:class:`ard.core.types.AnchorSpec` enforces on the way in, and which
:func:`message_shape_error` re-checks on the way out.

Failure policy (one anchor = one unit of work):

* A turn that fails **abandons the whole anchor** — never "skip the turn",
  because skipping would leave two consecutive messages with the same role
  (``UAUU``) and break the alternation invariant.
* The failure is **re-raised** to :func:`generate_text_anchors` with
  ``logger.exception`` evidence, so the scheduler can count it.  Swallowing it
  here is what left the backpressure counter unreachable (the counter below is
  the single place where consecutive failures are observed, §1.4).

Backpressure lives in :func:`generate_text_anchors`; the observations it
produces are returned as :class:`AnchorGenerationStats` so the pipeline can
publish them (§3.2 透明退路).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from ard.backends.api_client import (
    ARDEmptyContentError,
    ARDTimeoutError,
    ChatAPIClient,
)
from ard.core.types import AnchorSpec, GeneratedAnchor, TurnSpec
from ard.domain.anchor_shape import expected_message_roles, message_shape_error
from ard.domain.bank import AppendOutcome, append_anchor

logger = logging.getLogger(__name__)


#: Abandon reasons that mean "the server could not serve us", i.e. the ones the
#: backpressure counter is built on.
SERVER_INSTABILITY_REASONS: frozenset[str] = frozenset({"timeout", "transport_error"})

#: Abandon reasons that mean "the model answered unusably".  They are counted
#: and published, but they must **not** trigger a cooldown: sleeping cannot
#: change a deterministic model-output failure, it only wastes wall clock.
MODEL_OUTPUT_REASONS: frozenset[str] = frozenset(
    {
        "empty_content",
        "empty_user_message",
        "empty_assistant_message",
        "answer_too_short",
        "answer_too_long",
        "invalid_shape",
    }
)


@dataclass(slots=True)
class AnchorGenerationStats:
    """Observable outcome of one :func:`generate_text_anchors` run.

    Exists because the run's failures used to be visible only as scattered log
    lines: the caller got a (possibly short) anchor list and no way to tell a
    clean run from one that dropped 40% of its anchors (§3.2).  Every counter
    is a plain integer — no reasoning text, no answer text, so the object is
    safe to publish into ``manifest.json``.

    Attributes:
        requested: Number of anchor specs the run was asked for.
        succeeded: Anchors produced in memory (before the persistence gates).
        abandoned_total: Anchors dropped by :func:`_generate_one_anchor`.
        abandoned_by_reason: ``abandoned_total`` split by machine-readable tag
            (see :data:`SERVER_INSTABILITY_REASONS` /
            :data:`MODEL_OUTPUT_REASONS`).
        written: Anchors accepted by :func:`append_anchor` (0 when no
            ``output_path`` was given).
        rejected_invalid_shape: Anchors refused by the bank's shape gate.
        duplicate_ids: Anchors refused by the bank's id-uniqueness gate.
        backpressure_events: Cooldowns triggered (each one paused the run).
        consecutive_server_failures: Counter value at the end of the run; it is
            reset to 0 after every cooldown and after every success.
    """

    requested: int = 0
    succeeded: int = 0
    abandoned_total: int = 0
    abandoned_by_reason: dict[str, int] = field(default_factory=dict)
    written: int = 0
    rejected_invalid_shape: int = 0
    duplicate_ids: int = 0
    backpressure_events: int = 0
    consecutive_server_failures: int = 0

    def record_abandon(self, reason: str) -> None:
        """Count one dropped anchor under *reason*."""
        self.abandoned_total += 1
        self.abandoned_by_reason[reason] = self.abandoned_by_reason.get(reason, 0) + 1

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable view published in ``manifest.json``."""
        return {
            "requested": self.requested,
            "succeeded": self.succeeded,
            "abandoned_total": self.abandoned_total,
            "abandoned_by_reason": dict(sorted(self.abandoned_by_reason.items())),
            "written": self.written,
            "rejected_invalid_shape": self.rejected_invalid_shape,
            "duplicate_ids": self.duplicate_ids,
            "backpressure_events": self.backpressure_events,
        }


def failure_reason(exc: BaseException) -> str:
    """Classify *exc* into a machine-readable abandon reason.

    Classification is by exception **type**, never by string matching, so a
    renamed tag cannot silently move a failure into the wrong bucket
    (§2.2 显式即防呆).  The order of the branches is the classification
    contract:

    1. :exc:`ARDEmptyContentError` — model-output failure: deterministic for the
       same prompt, no cooldown.
    2. :exc:`ARDTimeoutError` — the layered timeout fired: server instability.
    3. :class:`httpx.HTTPError` — transport/server status failure: instability.
    4. anything else — unexpected; counted as ``unexpected_error`` and never
       treated as a server signal (a bug must not be disguised as load).
    """
    if isinstance(exc, ARDEmptyContentError):
        return "empty_content"
    if isinstance(exc, ARDTimeoutError):
        return "timeout"
    if isinstance(exc, httpx.HTTPError):
        return "transport_error"
    return "unexpected_error"


def is_server_instability(exc_or_reason: BaseException | str) -> bool:
    """Whether *exc_or_reason* justifies a backpressure cooldown."""
    reason = (
        exc_or_reason
        if isinstance(exc_or_reason, str)
        else failure_reason(exc_or_reason)
    )
    return reason in SERVER_INSTABILITY_REASONS


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
                {
                    "type": "text",
                    "text": (
                        f"Conversation so far:\n{history_text}\n\n"
                        f"Generate the next user message as instructed."
                    ),
                }
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

    One message per :class:`TurnSpec`, in order, so message *i* belongs to
    ``spec.turns[i]``: only a turn that owns an ``image_path`` has image parts
    to rewrite.  Messages beyond the turn list are carried over unchanged
    rather than dropped, so the function is length-preserving and never
    silently truncates an anchor's history.
    """
    result: list[dict[str, Any]] = []
    for msg_idx, raw_msg in enumerate(messages):
        msg = dict(raw_msg)  # shallow copy
        turn = spec.turns[msg_idx] if msg_idx < len(spec.turns) else None
        if turn is not None and turn.image_path and isinstance(msg.get("content"), list):
            new_content: list[dict[str, Any]] = []
            for item in msg["content"]:
                if item.get("type") == "image_url":
                    rel = _abs_to_rel_path(turn.image_path)
                    new_content.append({"type": "image", "image": rel})
                else:
                    new_content.append(item)
            msg["content"] = new_content
        result.append(msg)
    return result


def _abs_to_rel_path(abs_path: str) -> str:
    """Extract relative path (images/xxx.jpg) from absolute path."""
    idx = abs_path.rfind("/images/")
    if idx >= 0:
        return abs_path[idx + 1:]  # images/xxx.jpg
    return abs_path


def _user_message(
    user_msg: str,
    image_data_url: str | None,
) -> dict[str, Any]:
    """Build a ``user`` message, optionally carrying an image.

    With an image the content is the multimodal part list (image first, then
    text) that the API expects; without one it is the plain string, keeping
    text-only anchors byte-identical to what they have always been.
    """
    if image_data_url is None:
        return {"role": "user", "content": user_msg}
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": user_msg},
        ],
    }


def _generate_one_anchor(
    spec: AnchorSpec,
    input_client: ChatAPIClient,
    target_client: ChatAPIClient,
    input_model_name: str,
    target_model_name: str,
    min_answer_chars: int = 8,
    max_answer_chars: int | None = None,
    stats: AnchorGenerationStats | None = None,
) -> GeneratedAnchor | None:
    """Generate one anchor from an AnchorSpec.

    Role-driven generation — each turn appends exactly one message:
    user turns are produced by *input_client*, assistant turns by
    *target_client*.  The final (user) turn is answered by *target_client*; its
    answer becomes ``targets[0].output.content`` and its reasoning trace (present
    only when thinking is enabled for the target model) becomes
    ``targets[0].output.reasoning``.

    Any turn that fails (timeout, empty content, unknown role) abandons the
    whole anchor: the conversation is dropped and logged rather than left in
    a state where two messages share a role.  Failures **propagate** as
    exceptions (after being logged with their traceback) instead of being
    converted into a ``None`` return: that is the only way the scheduler can
    count them and apply backpressure (§3.2 透明退路).

    Args:
        spec: Anchor specification with turns and metadata.
        input_client: API client for generating user messages.
        target_client: API client for generating target answers.
        input_model_name: Name of the input-generator model.
        target_model_name: Name of the target model.
        min_answer_chars: Minimum answer length in characters.
        max_answer_chars: Optional maximum answer length.
        stats: Optional counters; soft abandons (empty / too short / unknown
            role) are recorded here because they raise nothing.

    Returns:
        A :class:`GeneratedAnchor` on success, or ``None`` if abandoned
        without an exception (empty, too short/long, no final turn).

    Raises:
        ARDTimeoutError: A turn exceeded the layered timeout.
        ARDEmptyContentError: A turn returned no assistant content.
        httpx.HTTPError: Transport/server failure.
    """
    if stats is None:
        stats = AnchorGenerationStats()
    messages: list[dict[str, Any]] = []
    expected_roles = expected_message_roles(spec)

    total_turns = len(spec.turns)
    for turn_idx, turn in enumerate(spec.turns):
        # Pre-encode image (if any) — encoding failures are fatal and
        # must NOT be caught by the API error handler below.
        image_data_url: str | None = None
        if turn.image_path:
            from ard.backends.api_client import encode_image_to_base64

            image_data_url = encode_image_to_base64(turn.image_path)

        if turn.role == "user":
            # ── user turn: input_generator produces the user message ───────
            # _build_user_prompt asks the input generator to *impersonate* the
            # user, so it is only meaningful for user turns.
            try:
                user_msg = input_client.chat(
                    _build_user_prompt(
                        turn, messages, spec.anchor_meta,
                        image_path=turn.image_path,
                        image_data_url=image_data_url,
                    ),
                    temperature=0.7,
                ).content.strip()
            except ARDTimeoutError:
                # Timeout mid-conversation: abandon the whole anchor.  Retrying
                # the turn would cost an extra request for a different sample
                # (§3.1 同效退路 does not apply — sampling is not idempotent),
                # and skipping it would leave two consecutive user messages,
                # i.e. a broken conversation.  The anchor is dropped and
                # announced; the pipeline's resume path supplies a replacement.
                logger.exception(
                    "Anchor %s: timeout generating user turn %d/%d — "
                    "abandoning this anchor to preserve role alternation",
                    spec.id, turn_idx + 1, total_turns,
                )
                raise
            except Exception as exc:
                # Not swallowed: the scheduler counts the abandon reason, and a
                # generator bug must not look like an empty anchor bank.
                logger.exception("Anchor %s: error generating user turn: %s", spec.id, exc)
                raise
            if not user_msg:
                logger.warning(
                    "Anchor %s: empty user message at turn %d/%d — abandoning anchor",
                    spec.id, turn_idx + 1, total_turns,
                )
                stats.record_abandon("empty_user_message")
                return None
            messages.append(_user_message(user_msg, image_data_url))
        elif turn.role == "assistant":
            # ── assistant turn: target_model produces the assistant reply ──
            # No input_client call: an assistant turn is a model answer, not a
            # user question.  Generating one here (the historical bug) shifted
            # every later message and produced UAUAU-shaped output.
            try:
                assist_msg = target_client.chat(messages, temperature=0.0).content.strip()
            except ARDTimeoutError:
                logger.exception(
                    "Anchor %s: timeout generating assistant turn %d/%d — "
                    "abandoning this anchor to preserve role alternation",
                    spec.id, turn_idx + 1, total_turns,
                )
                raise
            except Exception as exc:
                logger.exception(
                    "Anchor %s: error generating assistant turn: %s", spec.id, exc
                )
                raise
            if not assist_msg:
                logger.warning(
                    "Anchor %s: empty assistant message at turn %d/%d — abandoning anchor",
                    spec.id, turn_idx + 1, total_turns,
                )
                stats.record_abandon("empty_assistant_message")
                return None
            messages.append({"role": "assistant", "content": assist_msg})
        else:
            logger.warning(
                "Anchor %s: unknown turn role %r at turn %d/%d — abandoning anchor",
                spec.id, turn.role, turn_idx + 1, total_turns,
            )
            stats.record_abandon("unknown_role")
            return None

        # Fail fast: a turn that did not append its message would silently
        # shift the whole conversation.  Cheap, and it catches generator bugs
        # at the turn boundary rather than at the persistence boundary.
        if len(messages) != turn_idx + 1:
            logger.warning(
                "Anchor %s: after turn %d/%d the history has %d message(s), "
                "expected %d — abandoning anchor",
                spec.id, turn_idx + 1, total_turns, len(messages), turn_idx + 1,
            )
            stats.record_abandon("history_not_advanced")
            return None

        if not turn.is_final:
            continue

        # ── Final turn (always a user turn): target_model answers ───────────
        # The response carries the answer (``content``) and, when thinking is
        # enabled, the teacher's reasoning trace (``reasoning``).  The two are
        # kept apart all the way to the bank: thinking is not the answer
        # (§1.2 契约 2).  Check the accumulated roles against the spec before
        # spending the request: the answer is only meaningful for the intended
        # turn.
        #
        # v3.0.0 D1: an optional *single leading* ``system`` message is allowed
        # and is validated by the shared contract (:func:`message_shape_error`),
        # which rejects a misplaced or repeated system.  The turn-derived
        # ``expected_roles`` never contains a system, so before comparing them
        # the (already validated) system is stripped from the actual roles.
        actual_roles = [m["role"] for m in messages]
        conversation_roles = [r for r in actual_roles if r != "system"]
        if (
            message_shape_error(messages) is not None
            or conversation_roles != expected_roles
        ):
            logger.warning(
                "Anchor %s: message roles %r do not match spec roles %r — "
                "abandoning anchor",
                spec.id, actual_roles, expected_roles,
            )
            stats.record_abandon("role_mismatch")
            return None
        try:
            response = target_client.chat(messages, temperature=0.0)
        except ARDTimeoutError:
            logger.exception(
                "Anchor %s: timeout generating final user turn %d/%d — "
                "abandoning this anchor",
                spec.id, turn_idx + 1, total_turns,
            )
            raise
        except Exception as exc:
            logger.exception(
                "Anchor %s: error generating final user turn: %s", spec.id, exc
            )
            raise
        target_answer = response.content.strip()
        reasoning = response.reasoning
        if reasoning is not None:
            reasoning = reasoning.strip() or None
        if len(target_answer) < min_answer_chars:
            logger.warning(
                "Anchor %s: target answer too short (%d < %d chars) — skipping",
                spec.id, len(target_answer), min_answer_chars,
            )
            stats.record_abandon("answer_too_short")
            return None
        if max_answer_chars is not None and len(target_answer) > max_answer_chars:
            logger.warning(
                "Anchor %s: target answer too long (%d > %d chars) — skipping",
                spec.id, len(target_answer), max_answer_chars,
            )
            stats.record_abandon("answer_too_long")
            return None

        converted = _convert_images_to_paths(messages, spec)
        shape_error = message_shape_error(converted)
        if shape_error is not None:
            # Unreachable by construction unless a turn generator misbehaves;
            # kept as the last in-process gate so that no malformed anchor is
            # ever constructed (§2.3 边界校验即防呆).
            logger.warning(
                "Anchor %s: produced invalid message shape (%s) — abandoning anchor",
                spec.id, shape_error,
            )
            stats.record_abandon("invalid_shape")
            return None

        return GeneratedAnchor(
            id=spec.id,
            messages=converted,
            target_answer=target_answer,
            target_model=target_model_name,
            input_generator_model=input_model_name,
            anchor_meta=spec.anchor_meta,
            reasoning=reasoning,
        )

    # Only reachable if spec.turns had no is_final turn; AnchorSpec allows it
    # (it only requires first/last role), so warn instead of failing silently.
    logger.warning(
        "Anchor %s: no final turn produced an answer (%d turns) — no anchor",
        spec.id, total_turns,
    )
    stats.record_abandon("no_final_turn")
    return None


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
    stats: AnchorGenerationStats | None = None,
    sleep: Callable[[float], None] = time.sleep,
    disable_progress: bool = False,
) -> list[GeneratedAnchor]:
    """Generate text anchors from a list of AnchorSpec objects.

    Each spec is processed concurrently via ThreadPoolExecutor.  Results
    are streamed to *output_path* (if provided) as they complete.

    Backpressure (this function is the **only** place that can see it): one
    future = one anchor, and ``future.result()`` raises exactly when that
    anchor was abandoned.  A counter of *consecutive abandons caused by server
    instability* (timeout / transport error — see
    :data:`SERVER_INSTABILITY_REASONS`) is kept as results are collected; when
    it reaches *backpressure_threshold* the run pauses *backpressure_cooldown*
    seconds to let vLLM drain its queue, then the counter restarts from zero.
    Anchors abandoned for *model-output* reasons (empty answer, too short) are
    counted but never trigger a cooldown: the same prompt would fail
    identically, so sleeping only wastes wall clock.

    Whatever persists an anchor here (``append_anchor``) validates its message
    shape and de-duplicates by id; both rejections are counted and announced
    at the end of the run so a short anchor bank is never silent (§3.2).

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
        backpressure_threshold: Consecutive server-instability abandons that
            trigger a cooldown pause.  Default 3.
        backpressure_cooldown: Seconds to pause when backpressure triggers.
            Default 60.
        stats: Optional counters to fill in; when omitted an internal one is
            used (and discarded) so existing callers keep working.
        sleep: Injection point for the cooldown pause (tests pass a stub).
        disable_progress: Suppress the tqdm bar (keeps test output readable).

    Returns:
        List of generated :class:`GeneratedAnchor` objects.  The per-run
        failure accounting is available on *stats*.

    Raises:
        httpx.HTTPError: Propagated from a worker if the whole run fails this
            way outside a cooldown path (a worker's failure is normally folded
            into *stats*, not raised).
    """
    if stats is None:
        stats = AnchorGenerationStats()
    target_count = len(specs)
    stats.requested = target_count
    anchors: list[GeneratedAnchor] = []

    # Persistence-gate tallies (exit-boundary defence, §2.3 / §3.2).
    written = 0
    invalid_shape: list[tuple[str, str]] = []
    duplicate_ids: list[str] = []

    pbar = tqdm(total=target_count, desc="Text anchors", unit="anchor", disable=disable_progress)

    # Single-writer invariant: only this collector thread mutates the counter
    # (workers only *return* an outcome), so it needs no lock — and stating the
    # invariant is what lets the next reader trust that.
    consecutive_server_failures = 0

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
            stats,
        )
        for spec in specs
    ]

    try:
        for future in as_completed(futures):
            try:
                anchor = future.result()
            except Exception as exc:  # noqa: BLE001
                # Deliberately broad: the worker re-raises whatever abandoned
                # the anchor, and the collector must classify and count all of
                # it.  This is not a swallow (§13.1): the exception is
                # classified by type, counted, logged and published.
                reason = failure_reason(exc)
                stats.record_abandon(reason)
                if is_server_instability(reason):
                    consecutive_server_failures += 1
                # A deterministic model-output failure is neither evidence of
                # overload nor evidence of health, so it leaves the counter
                # untouched instead of incrementing or resetting it.
                counted = consecutive_server_failures
                logger.warning(
                    "Anchor abandoned (%s): %s "
                    "(%d consecutive server failure(s), threshold %d)",
                    reason, exc, counted, backpressure_threshold,
                )
                pbar.update(1)
                if counted >= backpressure_threshold:
                    # Announce *before* sleeping: the cooldown is the observable
                    # part of backpressure and must be visible even if the run
                    # is interrupted during it (§3.2 透明退路).
                    logger.warning(
                        "Backpressure triggered: %d consecutive server failures "
                        "(timeout/transport), pausing generation for %.1fs to let the "
                        "server recover.",
                        counted, backpressure_cooldown,
                    )
                    sleep(backpressure_cooldown)
                    stats.backpressure_events += 1
                    logger.warning(
                        "Backpressure cooldown finished after %.1fs; resetting the "
                        "consecutive-failure counter and resuming generation.",
                        backpressure_cooldown,
                    )
                    consecutive_server_failures = 0
                continue

            if anchor is not None:
                anchors.append(anchor)
                if output_path is not None:
                    outcome = append_anchor(anchor, output_path)
                    if outcome is AppendOutcome.APPENDED:
                        written += 1
                    elif outcome is AppendOutcome.DUPLICATE_SKIPPED:
                        duplicate_ids.append(anchor.id)
                    elif outcome is AppendOutcome.INVALID_SHAPE_SKIPPED:
                        invalid_shape.append(
                            (anchor.id, message_shape_error(anchor.messages) or "unknown")
                        )
                # A produced anchor is positive evidence that the server is
                # serving again, so it clears the streak.
                consecutive_server_failures = 0
            pbar.update(1)
            if len(anchors) >= target_count:
                break
    finally:
        pbar.close()
        executor.shutdown(wait=False, cancel_futures=True)

    stats.succeeded = len(anchors)
    stats.written = written
    stats.rejected_invalid_shape = len(invalid_shape)
    stats.duplicate_ids = len(duplicate_ids)
    stats.consecutive_server_failures = consecutive_server_failures

    if stats.abandoned_total:
        # A run that dropped anchors must not look like a clean run (§3.2).
        logger.warning(
            "Anchor generation abandoned %d/%d anchor(s): %s",
            stats.abandoned_total, target_count,
            ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(stats.abandoned_by_reason.items())
            ),
        )
    if invalid_shape or duplicate_ids:
        logger.warning(
            "Anchor bank %s accepted %d anchor(s); rejected %d with an invalid "
            "message shape and %d duplicate id(s).",
            output_path, written, len(invalid_shape), len(duplicate_ids),
        )
    for anchor_id, reason in invalid_shape:
        logger.warning("  rejected %s: %s", anchor_id, reason)
    for duplicate_id in duplicate_ids:
        logger.warning(
            "  duplicate id skipped: %s (an anchor with this id is already in %s)",
            duplicate_id, output_path,
        )

    return anchors
