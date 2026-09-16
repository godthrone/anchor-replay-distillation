"""OpenAI-compatible API client with httpx streaming (SSE) and layered timeouts.

Supports text and multimodal (base64-encoded image) chat requests,
with concurrent batch execution via ThreadPoolExecutor.

All requests use streaming SSE: :meth:`ChatAPIClient.chat` returns a
:class:`ChatResponse`, which carries the assistant's answer and — when the
server emitted one — the teacher's reasoning trace, as **two separate fields**.

Reasoning is *persisted output*, not accounting: it arrives in
``delta.reasoning`` (Qwen3-style chat templates: an **undefined**
``enable_thinking`` means thinking is ON) while ``delta.content`` stays
``null`` for the whole thinking phase.  It must never be concatenated into
the answer — thinking and answer are different supervision targets (§1.2
契约 2) — so the two streams are collected into separate fields and reach the
dataset as ``targets[0].output.content`` and ``targets[0].output.reasoning``.

Reasoning is also a *budget consumer*: a response whose whole budget went into
thinking arrives with ``delta.content`` never set.  This module counts that
(``ChatAPIStats``) and refuses to hand an empty string to the caller as if it
were an answer (:exc:`ARDEmptyContentError`).

The whole timeout policy is enforced by :func:`_iter_lines_with_timeout`;
this module deliberately sets **no** httpx-level read timeout (see
:func:`_send_streaming_request`).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import socket
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import httpx

logger = logging.getLogger(__name__)

#: How long the reader thread is given to exit after ``response.close()``
#: before its socket is forced down (see :func:`_iter_lines_with_timeout`).
_READER_GRACE_TIMEOUT = 0.25

#: Bound on the join *after* the socket has been forced down.  Deliberately far
#: below the timeout granularity: the forced ``shutdown(SHUT_RDWR)`` is what
#: releases the reader (measured: it exits within microseconds of the shutdown),
#: so this bound only caps the degraded path where the socket could not be
#: captured.  A large value here silently adds itself to every timeout — the
#: version this constant replaced joined with 5 s unconditionally, turning
#: ``first_token_timeout=300`` into 305 s and pinning a thread/connection slot
#: for 5 extra seconds per timed-out request.
_READER_JOIN_TIMEOUT = 0.5


# ── Exception types ─────────────────────────────────────────────────────────


class ARDTimeoutError(RuntimeError):
    """Raised when a streaming request exceeds the configured timeout."""


class ARDEmptyContentError(RuntimeError):
    """Raised when a completion produced no assistant text at all.

    This is a *model-output* failure — the token budget was consumed by hidden
    reasoning, or the server returned an unusable completion — and it is
    deliberately a different type from :exc:`ARDTimeoutError` and from the
    generic ``RuntimeError`` used for transport/server failures, so callers can
    classify it without string matching (§2.2 显式即防呆).

    The alternative — returning ``""`` — is what let the reasoning-truncation
    failure reach the dataset generator as an "empty target answer" that the
    domain layer discarded for an unrelated reason (``if not assist_msg: return
    None``), i.e. a silent loss (§3.2 透明退路: an affected result must be
    announced, never swallowed).

    Attributes:
        reason: Human-readable, greppable cause.
        stats: The :class:`ChatAPIStats` of the failed response, so the caller
            can tell "nothing came back at all" from "reasoning ate the budget".
    """

    def __init__(self, reason: str, stats: ChatAPIStats) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stats = stats


# ── Reasoning observability (WP-F3) ────────────────────────────────────────
#
# Reasoning tokens arrive in ``delta.reasoning`` (measured against vLLM 0.22 +
# Qwen3.8-27B; ``delta.reasoning_content`` does not exist there).  They are
# persisted as the teacher's reasoning trace (``targets[0].output.reasoning``)
# *and* they are a budget consumer: a run where they ate the whole
# ``max_tokens`` budget must not look healthy.  The counters below are the
# observable signal; ``pipeline.run()`` prints their delta around anchor
# generation.

_REASONING_STATS: dict[str, int] = {}
_REASONING_STATS_LOCK = threading.Lock()

#: ``finish_reason`` produced when the server stops at ``max_tokens``.
TRUNCATED_BY_MAX_TOKENS = "Response truncated by max_tokens limit"

#: Raised reason when the server stopped normally but sent no assistant text.
NO_CONTENT_RETURNED = "Response contained no assistant content"


def _record_reasoning_event(key: str, count: int = 1) -> None:
    """Increment one reasoning-observability counter (thread-safe).

    ``count=0`` is recorded rather than skipped: a key that is present with the
    value ``0`` says "measured, nothing seen", which is what a caller reading
    :func:`reasoning_stats` needs in order to tell it apart from "this run
    reported nothing at all".
    """
    with _REASONING_STATS_LOCK:
        _REASONING_STATS[key] = _REASONING_STATS.get(key, 0) + count


def reasoning_stats() -> dict[str, int]:
    """Return a copy of the reasoning-observability counters.

    Keys (cumulative since :func:`reset_reasoning_stats`):

    * ``responses`` — completed streamed responses.
    * ``reasoning_responses`` — responses that carried reasoning tokens.
    * ``reasoning_chars`` — total reasoning characters seen.  The text itself
      reaches the dataset through :attr:`ChatResponse.reasoning`; these counters
      are the run-level view of the same stream, so they stay safe to log.
    * ``reasoning_only_responses`` — responses that delivered reasoning and no
      content: every anchor built from one is guaranteed empty.
    * ``empty_content`` — responses whose assistant content was empty (a
      superset of ``reasoning_only_responses``).
    * ``truncated_empty`` — empty **and** ``finish_reason == "length"``: the
      signature of "thinking ate the budget".
    """
    with _REASONING_STATS_LOCK:
        return dict(_REASONING_STATS)


def reset_reasoning_stats() -> None:
    """Clear the reasoning-observability counters (call once at run start)."""
    with _REASONING_STATS_LOCK:
        _REASONING_STATS.clear()


@dataclass(slots=True)
class ChatAPIStats:
    """Per-response reasoning/content accounting (observability only).

    Deliberately carries **no** reasoning text: only counts, so a stats object
    (or a log line built from it) can never leak reasoning prose into a log
    file or into ``manifest.json``.  The text has exactly one home — the
    :attr:`ChatResponse.reasoning` field of the response it came from.
    """

    content_chars: int = 0
    reasoning_chars: int = 0
    reasoning_chunks: int = 0
    finish_reason: str | None = None

    @property
    def has_reasoning(self) -> bool:
        """True when this response delivered any reasoning token."""
        return self.reasoning_chunks > 0

    @property
    def empty_content(self) -> bool:
        """True when no assistant content arrived (reasoning may still have)."""
        return self.content_chars == 0

    def describe(self) -> str:
        """One-line, greppable summary for logs and error messages."""
        return (
            f"content_chars={self.content_chars}, reasoning_chars={self.reasoning_chars}, "
            f"reasoning_chunks={self.reasoning_chunks}, "
            f"finish_reason={self.finish_reason!r}"
        )


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ChatAPIConfig:
    """Configuration for an OpenAI-compatible chat completions endpoint."""

    api_base: str
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int | None = None  # None means omit from request (use provider default)
    connect_timeout: float = 10.0  # connection / TLS handshake
    first_token_timeout: float = 60.0  # prefill / time-to-first-token
    inter_token_timeout: float = 15.0  # stalls between tokens during generation
    max_retries: int = 2
    retry_on_timeout: bool = False  # True: retry timeouts; False: fail fast
    enable_thinking: bool = False  # two-state on purpose: see __post_init__

    def __post_init__(self) -> None:
        if self.api_base is None:
            raise ValueError("api_base must not be None")
        if self.model_name is None:
            raise ValueError("model_name must not be None")
        # §2.1 契约即防呆: ``enable_thinking`` is a **two-state** bool, not a
        # tri-state.  The measured server behaviour makes the third state
        # (`null` / "do not send the key") actively harmful:
        #
        #   * key sent as ``false``  → thinking off (25 prompt tokens measured);
        #   * key sent as ``true``   → thinking on  (65 prompt tokens measured);
        #   * key **absent**         → thinking ON  (the template reads
        #     ``enable_thinking is undefined`` as true);
        #   * key sent as ``null``   → thinking ON *and* the server's reasoning
        #     splitter stops splitting, so reasoning prose is delivered as
        #     ``message.content`` — reasoning text silently becomes the answer.
        #
        # So "not configured" is not a safe third state to expose: the only two
        # honest answers are True and False, and the annotation says exactly
        # that.  ``bool`` is checked by identity because ``isinstance(1, int)``
        # is True: only ``True``/``False`` pass (§2.2 — no implicit coercion).
        if self.enable_thinking is not True and self.enable_thinking is not False:
            raise TypeError(
                "enable_thinking must be True or False, got "
                f"{self.enable_thinking!r} ({type(self.enable_thinking).__name__}). "
                "None is rejected: on vLLM an explicit null both enables thinking "
                "and disables the reasoning splitter, which leaks reasoning text "
                "into the answer, while omitting the key enables thinking too. "
                "Use False to disable thinking."
            )

    @property
    def chat_completions_url(self) -> str:
        return self.api_base.rstrip("/") + "/chat/completions"


@dataclass(slots=True)
class ChatRequest:
    """A single chat request to be sent to the API."""

    messages: list[dict[str, Any]]
    temperature: float | None = None


@dataclass(slots=True)
class ChatResult:
    """Result of a single chat completion request."""

    content: str
    success: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A completed chat completion: the answer, and the reasoning behind it.

    Two fields on purpose.  ``reasoning`` is the teacher's thinking trace and
    ``content`` is its answer; they are **different supervision targets** and
    downstream code decides independently whether it wants the reasoning (§1.2
    契约 2 — the generator does not make that decision for the consumer).
    Concatenating them, or returning one where the other is expected, is the
    failure this type exists to make impossible.

    Attributes:
        content: The assistant's answer (the model's ``delta.content`` stream,
            stripped of the surrounding whitespace at the call site).
        reasoning: The thinking trace (``delta.reasoning``), or ``None`` when
            the response carried none — i.e. when thinking was disabled for the
            request.  Empty and absent are the same thing (§2.2: ``None`` is the
            only legal empty value), so a server that sent no reasoning yields
            ``None`` rather than ``""``.
    """

    content: str
    reasoning: str | None = None


# ── Image encoding utility ─────────────────────────────────────────────────


def encode_image_to_base64(path: str | Path) -> str:
    """Read an image file and return a ``data:image/...;base64,...`` data URI.

    Supports JPEG, PNG, WebP, and GIF formats.  The MIME type is
    derived from the file extension (case-insensitive).

    Args:
        path: Path to the image file on disk.

    Returns:
        A base64 data URI string suitable for use in an OpenAI-compatible
        multimodal message ``image_url`` field.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not a recognised image format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    extension_to_mime: dict[str, str] = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        
    }
    suffix = path.suffix.lower()
    mime = extension_to_mime.get(suffix)
    if mime is None:
        raise ValueError(
            f"Unsupported image format: {suffix!r}. "
            f"Supported: {', '.join(extension_to_mime)}"
        )

    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# ── Internal helpers ───────────────────────────────────────────────────────


def _reasoning_text_of(delta: dict[str, Any]) -> str:
    """Return the reasoning fragment carried by one SSE ``delta`` (or ``""``).

    Measured reality on the production endpoint (vLLM 0.22.0 + Qwen3.8-27B):
    reasoning arrives in ``delta.reasoning`` while ``delta.content`` stays
    ``null`` for the whole thinking phase.  ``reasoning_content`` (the DeepSeek
    spelling) is accepted as a fallback so this parser survives a serving-stack
    change; a stray empty string counts as "absent" and falls through, so a
    server that pads the unused field cannot mask the real one.

    The returned text is accumulated into :attr:`ChatResponse.reasoning` and
    persisted as ``targets[0].output.reasoning``.  It must never be appended to
    the content parts: reasoning is not part of the distilled answer, and mixing
    it into ``messages`` would de-sync the emitted conversation from the schema
    written to disk.
    """
    for field in ("reasoning", "reasoning_content"):
        value = delta.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _log_empty_completion(response_label: str, stats: ChatAPIStats) -> str:
    """Announce an empty completion and return the greppable reason string.

    Called when a streamed response yielded no assistant content.  Two shapes
    are told apart explicitly (§2.2 显式即防呆) — they have different causes and
    different fixes, and neither is a network error:

    * ``finish_reason == "length"``: the budget ran out.  If reasoning was
      present, that is the smoking gun: reasoning consumed ``max_tokens``
      before the answer began, so raise ``max_tokens`` or turn thinking off.
    * anything else: the server finished (or the stream ended) without ever
      emitting content.
    """
    if stats.finish_reason == "length":
        detail = (
            f"{TRUNCATED_BY_MAX_TOKENS}: the token budget was exhausted before any "
            f"assistant content arrived"
        )
        if stats.has_reasoning:
            detail += (
                f" — reasoning consumed it ({stats.reasoning_chars} reasoning chars over "
                f"{stats.reasoning_chunks} chunk(s), 0 content chars). Raise max_tokens or "
                f"disable thinking (enable_thinking=false)."
            )
    else:
        detail = (
            f"{NO_CONTENT_RETURNED}: the response finished with "
            f"{stats.finish_reason!r} and no assistant content"
        )
        if stats.has_reasoning:
            detail += (
                f" (reasoning-only response: {stats.reasoning_chars} reasoning chars over "
                f"{stats.reasoning_chunks} chunk(s))"
            )
    logger.warning("Empty completion discarded [%s]: %s", response_label, detail)
    return detail


def _assert_enable_thinking_is_bool(value: Any, *, where: str) -> bool:
    """Return *value* if it is a real ``bool``, else raise ``TypeError`` (§2.1).

    Shared by the two layers that must agree on the contract.  ``bool`` is
    checked by identity so that ``1``/``0``/``"true"``/``None`` cannot slip
    through as ints or strings (§2.2 — no implicit coercion).  Returns the value
    so callers can use it directly.
    """
    if value is not True and value is not False:
        raise TypeError(
            f"{where}: enable_thinking must be True or False, got "
            f"{value!r} ({type(value).__name__})."
        )
    return value


def _build_payload(
    config: ChatAPIConfig,
    messages: list[dict[str, Any]],
    temperature: float | None,
    *,
    stream: bool = True,
) -> dict[str, Any]:
    """Build the JSON payload for a chat completions request.

    ``chat_template_kwargs.enable_thinking`` is **always** emitted, and always
    as a real ``bool`` (``config.enable_thinking`` is two-state by contract —
    see :meth:`ChatAPIConfig.__post_init__`).  Both of the other candidates are
    measured-bad on vLLM 0.22 + Qwen3.8-27B:

    * ``null`` → thinking stays on *and* the server's reasoning splitter stops
      splitting, so reasoning prose arrives as ``message.content`` and the
      dataset would learn it as the answer;
    * omitting the key → ``enable_thinking is undefined`` in the chat template,
      which enables thinking: the run silently starts paying for reasoning
      tokens (and can exhaust ``max_tokens`` before any answer).

    Emitting an explicit bool is therefore the only payload shape whose meaning
    is unambiguous.  The value is re-validated here so the helper cannot be
    used to forge an invalid payload (§2.3 边界校验即防呆).
    """
    payload: dict[str, Any] = {
        "model": config.model_name,
        "messages": messages,
        "temperature": temperature if temperature is not None else config.temperature,
        "stream": stream,
    }
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    payload["chat_template_kwargs"] = {
        "enable_thinking": _assert_enable_thinking_is_bool(
            config.enable_thinking, where="payload construction"
        )
    }
    return payload


def _stream_socket(response: httpx.Response) -> socket.socket | None:
    """Return an *independent handle* to the socket carrying *response*'s body.

    Walks the third-party chain
    ``response.stream._stream._httpcore_stream._stream._connection._network_stream._sock``
    and returns a duplicate of that socket, because the handle must stay usable
    after ``response.close()`` has closed the original (see
    :func:`_force_release_reader`).  ``os.dup`` shares the open file
    description, so shutting the duplicate down affects the very connection the
    reader thread is parked on.

    Every step is guarded: the chain is httpx/httpcore internals and may differ
    between versions, in which case we return ``None`` and the caller degrades to
    a bounded join (no crash).  Introspecting a third-party library for runtime
    capability detection is the allowed form of attribute probing (§2.2); this
    module's own interfaces are never probed.

    Args:
        response: The streaming response whose socket is wanted.

    Returns:
        A duplicated ``socket.socket`` the caller owns and must close, or
        ``None`` if the socket could not be reached or duplicated.
    """
    node: Any = response
    for attribute in ("stream", "_stream", "_httpcore_stream", "_stream",
                      "_connection", "_network_stream"):
        node = getattr(node, attribute, None)
        if node is None:
            return None
    sock = getattr(node, "_sock", None)
    if not isinstance(sock, socket.socket):
        return None
    try:
        return socket.socket(fileno=os.dup(sock.fileno()))
    except OSError:
        return None


def _force_release_reader(sock: socket.socket | None) -> None:
    """Turn a blocked ``recv()`` into an immediate return (§3.1 同效退路).

    Three steps, in order:

    1. ``shutdown(SHUT_RDWR)`` — this is what makes an in-flight ``recv()`` on
       the reader thread return *now* instead of when the server finally writes.
    2. ``SO_LINGER`` with a zero timeout — makes the subsequent ``close()``
       discard the send buffer and emit an RST instead of a graceful FIN, so the
       server notices the aborted request instead of finishing a generation
       nobody is reading.
    3. ``close()`` — release the duplicated descriptor.

    All three tolerate ``OSError``: the peer may already be gone, or the original
    descriptor may already have been closed by httpx.
    """
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError as exc:
        # Expected when the peer is already gone or httpx has closed the
        # descriptor; recorded rather than swallowed (§13.1 异常不吞).
        logger.debug("Forced release: shutdown failed (already closed): %s", exc)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    except OSError as exc:
        logger.debug("Forced release: set SO_LINGER failed (already closed): %s", exc)
    try:
        sock.close()
    except OSError as exc:
        logger.debug("Forced release: close failed (already closed): %s", exc)


def _iter_lines_with_timeout(
    response: httpx.Response,
    first_token_timeout: float,
    inter_token_timeout: float,
) -> Generator[str, None, None]:
    """Iterate over SSE ``data:`` lines with per-phase timeout control.

    Uses a background thread that reads lines from the response and pushes
    them into a queue; the main generator thread consumes them with the
    appropriate timeout per phase:

    - Before the first ``data:`` line arrives → *first_token_timeout*
    - After the first ``data:`` line → *inter_token_timeout*

    On timeout, signals the reader thread to stop, **actively releases** it, and
    raises :exc:`ARDTimeoutError`.

    **Why no httpx-level read timeout is needed (and why closing is not enough).**
    The queue timeout caps *every* wait the caller can experience: when it fires
    we raise.  But raising alone would leave the reader thread parked in
    ``recv()`` — right up until the server decides to write — which would keep
    the HTTP request alive on the server side (a zombie request) and leak one
    thread per timed-out call.  Measured on httpx 0.28 / httpcore 1.x against a
    server that sends headers then stalls 5 s:

    * ``response.close()`` returns in 0.000 s but does **not** release the
      blocked ``recv()`` (the caller's ``finally`` then burned the full 5 s join
      budget waiting for a thread that was still parked);
    * with no read deadline at all, "the response body will not be read" is not
      by itself a release — the peer is never told.

    So the ordering is: ``stop_event`` → ``response.close()`` (supported path,
    releases the common cases) → **``socket.shutdown(SHUT_RDWR)`` on the socket
    captured before the first read**, which is the only call that turns an
    in-flight ``recv()`` into an immediate return, and tells the peer to stop
    (``SO_LINGER(1, 0)`` makes it an RST, so the server abandons the generation
    instead of finishing a response nobody reads).

    **Why the join bounds are small (0.25 s + 0.5 s) and why that is safe.**
    The release is performed *by the shutdown*, not by the join — the joins only
    observe it.  Measured on a 5 s-stalling server with
    ``first_token_timeout=0.5``: the call returns in 0.86 s, and the reader
    thread is already gone by then (``is_alive()`` is False after every join,
    including a 0 s join).  A large join bound therefore buys nothing and costs
    real latency: it adds itself to *every* timeout (``first_token_timeout=300``
    would behave as 305 s) and holds a thread/connection slot for the remainder.
    The residual case is only "the socket could not be captured" (unknown
    httpx/httpcore internals), where the reader is left as a daemon thread that
    exits whenever the server finally writes or closes; the caller still gets a
    prompt :exc:`ARDTimeoutError` and the queue timer still caps every wait, so
    the degraded outcome is a lingering daemon thread, never a stuck caller.

    Args:
        response: An ``httpx.Response`` from a streaming request.
        first_token_timeout: Maximum seconds to wait for the first SSE line.
        inter_token_timeout: Maximum seconds to wait between lines after the first.

    Yields:
        Raw line strings (including ``"data: ..."`` prefix and ``"[DONE]"``).

    Raises:
        ARDTimeoutError: If a timeout is exceeded in either phase.
    """
    line_queue: queue.Queue[tuple[str, str | Exception | None]] = queue.Queue()
    stop_event = threading.Event()
    # Captured BEFORE the reader starts consuming, so the socket path is walked
    # on a quiescent response object rather than one being mutated by another
    # thread.
    sock = _stream_socket(response)

    def _read_lines() -> None:
        try:
            for line in response.iter_lines():
                line_queue.put(("line", line))
                if stop_event.is_set():
                    break
        except Exception as exc:
            line_queue.put(("error", exc))
        line_queue.put(("done", None))

    reader_thread = threading.Thread(target=_read_lines, daemon=True)
    reader_thread.start()

    first_token = True
    try:
        while True:
            try:
                timeout = first_token_timeout if first_token else inter_token_timeout
                kind, value = line_queue.get(timeout=timeout)
            except queue.Empty:
                stop_event.set()
                phase = "first token (prefill)" if first_token else "inter-token"
                raise ARDTimeoutError(
                    f"Streaming request timed out waiting for {phase} "
                    f"(timeout={timeout:.0f}s)"
                )

            if kind == "done":
                break
            if kind == "error":
                raise value  # type: ignore[misc]
            # kind == "line"
            assert isinstance(value, str) or value is None
            if value is not None:
                first_token = False
                yield value
    finally:
        stop_event.set()
        try:
            # Supported path: normal completion, server-side [DONE], and any
            # case where the body stream is not currently blocked in recv().
            response.close()
        except httpx.TransportError:
            # Closing a stream whose peer has already gone away raises
            # ``httpx.ReadError``/``RemoteProtocolError`` (both are
            # ``TransportError``).  Nothing left to do about it here, and this
            # is a cleanup path — swallowing a transport error must not mask the
            # ARDTimeoutError/parse error that is already propagating.
            pass
        # Give the reader a moment; if it is still parked in recv(), force the
        # socket down so the thread exits now instead of when the server feels
        # like finishing (zombie-request defense, §3.1).
        reader_thread.join(timeout=_READER_GRACE_TIMEOUT)
        if reader_thread.is_alive():
            _force_release_reader(sock)
            reader_thread.join(timeout=_READER_JOIN_TIMEOUT)


def _send_streaming_request(
    config: ChatAPIConfig,
    payload: dict[str, Any],
) -> tuple[str, str | None, str | None, ChatAPIStats]:
    """Send a streaming chat completions request via SSE.

    Returns ``(content, reasoning, finish_reason, stats)``.

    Uses ``httpx.Client.stream()`` to POST *payload* with ``stream: true``,
    then parses SSE ``data:`` chunks.  Timeouts are layered:

    1. ``connect_timeout`` — TCP connection + TLS handshake (httpx-level)
    2. ``first_token_timeout`` — wait for the first ``data:`` line (our loop).
       This covers prefill, so it must also cover the first bytes of the
       response body: no httpx-level ``read`` timeout is set, otherwise the
       read timer would start ticking on the wait for the first token and
       would cut long prefills short (see :func:`_iter_lines_with_timeout`).
    3. ``inter_token_timeout`` — wait between lines during generation (our loop)

    Args:
        config: Endpoint and timeout configuration.
        payload: Full JSON request body (must include ``"stream": true``).

    Returns:
        A tuple of ``(full_content, reasoning, finish_reason, stats)``.
        *reasoning* is the accumulated ``delta.reasoning`` text, or ``None``
        when the response carried none.  *finish_reason* is ``None`` if none was
        observed.  *stats* counts what the response actually contained
        (see :class:`ChatAPIStats`).

    Raises:
        httpx.TimeoutException: On connect-timeout.
        RuntimeError: On HTTP errors or SSE parse errors.
        ARDTimeoutError: On streaming timeouts (from :func:`_iter_lines_with_timeout`).
        TypeError: If *payload* would carry a non-bool ``enable_thinking`` (the
            last checkpoint before the request leaves the process).
    """
    # Last checkpoint before the wire (§2.1/§2.3): whatever built this payload,
    # validate the one field whose serialized form (``null``) silently corrupts
    # the dataset.  Cheap, and it fails before a token is generated rather than
    # after a contaminated answer is written to the anchor bank.
    template_kwargs = payload.get("chat_template_kwargs")
    if not isinstance(template_kwargs, dict) or "enable_thinking" not in template_kwargs:
        raise TypeError(
            "payload must carry chat_template_kwargs.enable_thinking as a bool; "
            f"got {template_kwargs!r}. The key is always sent explicitly: omitting "
            "it enables thinking on vLLM, and null additionally leaks reasoning "
            "text into message.content."
        )
    _assert_enable_thinking_is_bool(
        template_kwargs["enable_thinking"], where="request boundary"
    )

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    # No httpx read timeout on purpose (zombie-request defense is provided by
    # the queue timer below, not by a socket-level read deadline — see
    # _iter_lines_with_timeout).
    http_timeout = httpx.Timeout(None, connect=config.connect_timeout, read=None)

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    # Reasoning accounting (WP-F3).  The text is accumulated into
    # ``reasoning_parts`` — never into ``content_parts`` (see _reasoning_text_of).
    reasoning_chars = 0
    reasoning_chunks = 0

    with httpx.Client(timeout=http_timeout) as client:
        with client.stream("POST", config.chat_completions_url,
                           json=payload, headers=headers) as response:
            if response.status_code != 200:
                try:
                    body = response.read().decode("utf-8", errors="replace")
                except Exception:
                    body = "<unreadable>"
                raise RuntimeError(
                    f"Chat completions request failed with HTTP {response.status_code}: {body}"
                )

            for line in _iter_lines_with_timeout(
                response, config.first_token_timeout, config.inter_token_timeout
            ):
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].lstrip()  # remove "data:" and optional leading space
                if data_str == "[DONE]":
                    break
                try:
                    chunk: dict[str, Any] = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.debug("Skipping unparseable SSE line: %s", line[:120])
                    continue

                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice0 = choices[0]
                if not isinstance(choice0, dict):
                    continue

                delta = choice0.get("delta")
                if isinstance(delta, dict):
                    token = delta.get("content")
                    if isinstance(token, str) and token:
                        content_parts.append(token)
                    # Reasoning is a *separate output stream*.  It is collected
                    # into its own field because the dataset persists it as
                    # ``targets[0].output.reasoning``; appending it to
                    # content_parts would merge thinking into the answer.
                    reasoning = _reasoning_text_of(delta)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        reasoning_chars += len(reasoning)
                        reasoning_chunks += 1

                fr = choice0.get("finish_reason")
                if fr is not None:
                    finish_reason = fr

    content = "".join(content_parts)
    # Empty and absent are the same thing (§2.2): a response that carried no
    # reasoning yields None, never "".
    reasoning_text = "".join(reasoning_parts) or None
    stats = ChatAPIStats(
        content_chars=len(content),
        reasoning_chars=reasoning_chars,
        reasoning_chunks=reasoning_chunks,
        finish_reason=finish_reason,
    )

    # ── Observability: reasoning vs. content (WP-F3) ──────────────────────
    # Public counters, so the pipeline can report "N responses lost their whole
    # budget to thinking" instead of leaving it to be inferred from logs.
    _record_reasoning_event("responses")
    if stats.has_reasoning:
        _record_reasoning_event("reasoning_responses")
        _record_reasoning_event("reasoning_chars", reasoning_chars)
        if stats.empty_content:
            _record_reasoning_event("reasoning_only_responses")
    if stats.empty_content:
        _record_reasoning_event("empty_content")
        if finish_reason == "length":
            _record_reasoning_event("truncated_empty")

    return content, reasoning_text, finish_reason, stats


def _is_timeout_error(exc: Exception) -> bool:
    """Check if an exception is a timeout-related error.

    Checks for :class:`ARDTimeoutError` (our own timeout),
    :class:`httpx.TimeoutException` (httpx-level timeout), or a
    ``"timed out"`` substring in the message (a text fallback for
    error messages that do not map to a known exception type).
    """
    if isinstance(exc, ARDTimeoutError):
        return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    msg = str(exc).lower()
    return "timed out" in msg


# ── Public API ─────────────────────────────────────────────────────────────


class ChatAPIClient:
    """OpenAI-compatible chat completions API client.

    Supports text and multimodal messages.  All requests use ``httpx`` with
    streaming SSE and per-phase timeouts (connect → first_token → inter_token).
    Every response comes back as a :class:`ChatResponse` — answer and reasoning
    in separate fields; a response that delivered reasoning and no content is a
    failure (:exc:`ARDEmptyContentError`), not an empty answer.
    Concurrency is handled by ``ThreadPoolExecutor``.

    Parameters:
        config: Endpoint, model, auth, and timeout configuration.
    """

    def __init__(self, config: ChatAPIConfig) -> None:
        self._config = config

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> ChatResponse:
        """Send a single streaming chat request and return a :class:`ChatResponse`.

        Uses SSE streaming with per-phase timeouts (connect → first_token →
        inter_token).  On timeout the request is NOT retried when
        ``retry_on_timeout`` is ``False`` (default).

        Args:
            messages: A list of message dicts in OpenAI format.
                For multimodal requests, the ``content`` field of a user
                message may be a ``list[dict]`` with ``{"type": "image_url",
                "image_url": {"url": "<data URI>"}}`` and ``{"type": "text",
                "text": "..."}`` parts.
            temperature: Optional per-request temperature override.
                When ``None``, the client's default temperature is used.

        Returns:
            A :class:`ChatResponse` whose ``content`` is the answer and whose
            ``reasoning`` is the thinking trace (``None`` when the response
            carried none).  The two are separate fields: reasoning is never
            part of ``content``, and merging them is the caller's decision to
            make, not this client's.

        Raises:
            ARDEmptyContentError: If the completion carried no assistant content
                (e.g. reasoning consumed the whole ``max_tokens`` budget).  Not
                retried — the cause is deterministic for the same prompt.
            RuntimeError: After exhausting all retries.
        """
        payload = _build_payload(self._config, messages, temperature, stream=True)
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                content, reasoning, finish_reason, stats = _send_streaming_request(
                    self._config, payload
                )
                # Order matters: "no content at all" is checked first so the
                # caller always sees the precise cause (reasoning ate the
                # budget) rather than the generic truncation message.
                if stats.empty_content:
                    raise ARDEmptyContentError(_log_empty_completion("chat", stats), stats)
                if finish_reason == "length":
                    logger.warning(
                        "Response truncated by max_tokens (finish_reason=length), discarding"
                    )
                    raise RuntimeError(TRUNCATED_BY_MAX_TOKENS)
                if finish_reason is not None and finish_reason != "stop":
                    logger.warning("Unusual finish_reason: %s", finish_reason)
                return ChatResponse(content=content.strip(), reasoning=reasoning)
            except ARDEmptyContentError:
                # Model-output failure, not a transport failure: the same
                # prompt with the same budget would fail identically, so retry
                # would only burn time and tokens.  The counter and the WARNING
                # above already carry the observable signal (§3.2).
                raise
            except RuntimeError as exc:
                if _is_timeout_error(exc) and not self._config.retry_on_timeout:
                    logger.warning(
                        "Streaming request timed out (retry_on_timeout=false, failing fast). "
                        "Set retry_on_timeout=true to enable automatic retries."
                    )
                    raise
                last_error = exc
                if attempt < self._config.max_retries:
                    delay = min(2.0 ** attempt, 30.0)
                    if _is_timeout_error(exc):
                        logger.warning(
                            "Attempt %d/%d failed (timeout). Retrying in %.1fs...",
                            attempt + 1, self._config.max_retries + 1, delay,
                        )
                    else:
                        logger.warning(
                            "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                            attempt + 1, self._config.max_retries + 1, exc, delay,
                        )
                    time.sleep(delay)
            except httpx.TimeoutException as exc:
                if not self._config.retry_on_timeout:
                    logger.warning(
                        "Streaming request timed out (retry_on_timeout=false, failing fast). "
                        "Set retry_on_timeout=true to enable automatic retries."
                    )
                    # Fail-fast: exactly one attempt is made, so the message must
                    # not claim max_retries+1 attempts (§2.2 显式即防呆).
                    raise ARDTimeoutError(
                        "Streaming request timed out "
                        "(retry_on_timeout=false, single attempt, no retry)"
                    ) from exc
                last_error = exc
                if attempt < self._config.max_retries:
                    delay = min(2.0 ** attempt, 30.0)
                    logger.warning(
                        "Attempt %d/%d failed (timeout). Retrying in %.1fs...",
                        attempt + 1, self._config.max_retries + 1, delay,
                    )
                    time.sleep(delay)

        logger.error(
            "All %d attempts failed: %s",
            self._config.max_retries + 1, last_error,
        )
        raise RuntimeError(
            f"Chat request failed after {self._config.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    def chat_batch(
        self,
        requests: list[ChatRequest],
        concurrency: int = 4,
    ) -> list[ChatResult]:
        """Execute multiple chat requests concurrently.

        Args:
            requests: A list of :class:`ChatRequest` objects.
            concurrency: Maximum number of concurrent requests.

        Returns:
            A list of :class:`ChatResult` objects in the same order as
            *requests*.  Results for failed requests have
            ``success=False`` and a non-``None`` ``error`` field.
        """
        results: list[ChatResult | None] = [None] * len(requests)

        def _worker(index: int, req: ChatRequest) -> None:
            try:
                # Batch callers want the answers as a list of ChatResult; the
                # reasoning trace is dropped here on purpose (ChatResult has no
                # field for it — a caller that needs reasoning uses chat()).
                response = self.chat(req.messages, req.temperature)
                results[index] = ChatResult(content=response.content, success=True)
            except Exception as exc:
                results[index] = ChatResult(
                    content="",
                    success=False,
                    error=str(exc),
                )

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures: list[Future[None]] = []
            for i, req in enumerate(requests):
                futures.append(executor.submit(_worker, i, req))
            for future in as_completed(futures):
                future.result()  # propagate unexpected exceptions

        # At this point all slots are filled — the cast is safe.
        return results  # type: ignore[return-value]
