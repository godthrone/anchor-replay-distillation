"""OpenAI-compatible API client with httpx streaming (SSE) and layered timeouts.

Supports text and multimodal (base64-encoded image) chat requests,
with concurrent batch execution via ThreadPoolExecutor.

All requests use streaming SSE:
- ``chat()`` returns content via SSE with per-phase timeouts.
- ``chat_with_logprobs()`` returns content + log-probs via SSE,
  collecting ``logprobs.content`` from each delta chunk.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import socket
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import httpx

logger = logging.getLogger(__name__)


# ── Exception types ─────────────────────────────────────────────────────────


class ARDTimeoutError(RuntimeError):
    """Raised when a streaming request exceeds the configured timeout."""


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
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if self.api_base is None:
            raise ValueError("api_base must not be None")
        if self.model_name is None:
            raise ValueError("model_name must not be None")

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


@dataclass(slots=True)
class ChatResultWithLogprobs(ChatResult):
    """Result of a chat completion request with log-prob data."""

    logprobs: dict[str, Any] | None = None


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


def _build_payload(
    config: ChatAPIConfig,
    messages: list[dict[str, Any]],
    temperature: float | None,
    *,
    logprobs: bool = False,
    top_logprobs: int = 1,
    stream: bool = True,
) -> dict[str, Any]:
    """Build the JSON payload for a chat completions request."""
    payload: dict[str, Any] = {
        "model": config.model_name,
        "messages": messages,
        "temperature": temperature if temperature is not None else config.temperature,
        "stream": stream,
    }
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    if config.enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    return payload


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

    On timeout, signals the reader thread to stop, attempts to shut down
    the underlying socket to interrupt a blocked ``recv()``, and raises
    :exc:`ARDTimeoutError`.  A ``finally`` block ensures the response is
    closed and the reader thread is joined.

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
                # Attempt to interrupt a blocked recv() on the underlying socket
                # so the reader thread can exit promptly (zombie-request defense).
                try:
                    sock = response._transport._proxy._sock
                    sock.shutdown(socket.SHUT_RD)
                except Exception:
                    pass
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
            response.close()
        except Exception:
            pass
        reader_thread.join(timeout=5.0)


def _send_streaming_request(
    config: ChatAPIConfig,
    payload: dict[str, Any],
    *,
    collect_logprobs: bool = False,
) -> tuple[str, str | None, dict | None]:
    """Send a streaming chat completions request via SSE and return (content, finish_reason, logprobs).

    Uses ``httpx.Client.stream()`` to POST *payload* with ``stream: true``,
    then parses SSE ``data:`` chunks.  Timeouts are layered:

    1. ``connect_timeout`` — TCP connection + TLS handshake (httpx-level)
    2. ``read=30.0`` — httpx-level per-read timeout as a safety net
       against permanently blocked receive operations
    3. ``first_token_timeout`` — wait for the first ``data:`` line (our loop)
    4. ``inter_token_timeout`` — wait between lines during generation (our loop)

    Args:
        config: Endpoint and timeout configuration.
        payload: Full JSON request body (must include ``"stream": true``).
        collect_logprobs: If True, collect log-probabilities from each SSE chunk
            and return them as a dict with ``token_ids`` and ``log_probs`` lists.
            Defaults to False for plain ``chat()`` calls.

    Returns:
        A tuple of ``(full_content, finish_reason, logprobs)``.  *finish_reason* is
        ``None`` if none was observed.  *logprobs* is ``None`` when *collect_logprobs*
        is ``False``, otherwise a dict ``{"token_ids": [...], "log_probs": [...]}``.

    Raises:
        httpx.TimeoutException: On connect-timeout or read-timeout.
        RuntimeError: On HTTP errors or SSE parse errors.
        ARDTimeoutError: On streaming timeouts (from :func:`_iter_lines_with_timeout`).
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    # Per-phase timeouts are enforced by _iter_lines_with_timeout.
    # httpx-level read=30.0 acts as a safety net — if no data arrives for 30 s
    # the read thread will not block indefinitely (zombie-request defense).
    http_timeout = httpx.Timeout(None, connect=config.connect_timeout, read=30.0)

    content_parts: list[str] = []
    finish_reason: str | None = None
    token_ids: list[int | str] = []
    log_probs: list[float] = []

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
                if isinstance(choices, list) and choices:
                    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                    if isinstance(delta, dict):
                        token = delta.get("content")
                        if isinstance(token, str) and token:
                            content_parts.append(token)
                    if collect_logprobs:
                        chunk_logprobs = chunk.get("logprobs")
                        if chunk_logprobs is not None and isinstance(chunk_logprobs, dict):
                            content_lps = chunk_logprobs.get("content")
                            if isinstance(content_lps, list):
                                for lp in content_lps:
                                    if isinstance(lp, dict):
                                        tid = lp.get("token") or lp.get("token_id", "")
                                        if isinstance(tid, (int, str)):
                                            token_ids.append(tid)
                                        lp_val = lp.get("logprob")
                                        if isinstance(lp_val, (int, float)):
                                            log_probs.append(float(lp_val))
                    fr = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
                    if fr is not None:
                        finish_reason = fr

    logprobs_result: dict | None = None
    if collect_logprobs:
        logprobs_result = {"token_ids": token_ids, "log_probs": log_probs}
    return "".join(content_parts), finish_reason, logprobs_result


def _is_timeout_error(exc: Exception) -> bool:
    """Check if an exception is a timeout-related error.

    Checks for :class:`ARDTimeoutError` (our own timeout),
    :class:`httpx.TimeoutException` (httpx-level timeout), or a
    ``"timed out"`` substring in the message (string fallback for
    backward compatibility).
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
    Log-probs are collected from SSE delta chunks when requested.
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
        *,
        logprobs: bool = False,
        top_logprobs: int = 1,
    ) -> str:
        """Send a single streaming chat request and return the content text.

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
            logprobs: If True, request log-probabilities from the API.
                Note: This method only returns the content string.
                Use :meth:`chat_with_logprobs` to also retrieve log-probs.

        Returns:
            The text content of the assistant's response.

        Raises:
            RuntimeError: After exhausting all retries.
        """
        payload = _build_payload(
            self._config, messages, temperature,
            logprobs=logprobs, top_logprobs=top_logprobs,
            stream=True,
        )
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                content, finish_reason, _ = _send_streaming_request(self._config, payload)
                if finish_reason == "length":
                    logger.warning(
                        "Response truncated by max_tokens (finish_reason=length), discarding"
                    )
                    raise RuntimeError("Response truncated by max_tokens limit")
                if finish_reason is not None and finish_reason != "stop":
                    logger.warning("Unusual finish_reason: %s", finish_reason)
                return content.strip()
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
                    raise ARDTimeoutError(
                        f"Streaming request timed out after "
                        f"{self._config.max_retries + 1} attempt(s)"
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

    def chat_with_logprobs(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        *,
        top_logprobs: int = 1,
    ) -> dict[str, Any]:
        """Send a streaming chat request and return content + log-probabilities.

        Uses SSE streaming mode with per-phase timeouts (connect → first_token →
        inter_token).  Log-probs are collected from each SSE delta chunk via
        ``collect_logprobs=True``.

        Args:
            messages: A list of message dicts in OpenAI format.
            temperature: Optional per-request temperature override.
            top_logprobs: Number of top log-probs to return per token (default 1).

        Returns:
            A dict with keys:
            - ``content`` (str): The assistant's response text.
            - ``logprobs`` (dict): ``{"token_ids": [...], "log_probs": [...]}``.

        Raises:
            RuntimeError: After exhausting all retries.
        """
        payload = _build_payload(
            self._config, messages, temperature,
            logprobs=True, top_logprobs=top_logprobs,
            stream=True,
        )
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                content, finish_reason, logprobs = _send_streaming_request(
                    self._config, payload, collect_logprobs=True,
                )
                if finish_reason == "length":
                    logger.warning(
                        "Response truncated by max_tokens (finish_reason=length), discarding"
                    )
                    raise RuntimeError("Response truncated by max_tokens limit")
                if finish_reason is not None and finish_reason != "stop":
                    logger.warning("Unusual finish_reason: %s", finish_reason)
                return {
                    "content": content.strip(),
                    "logprobs": logprobs or {"token_ids": [], "log_probs": []},
                }
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
                    raise ARDTimeoutError(
                        f"Streaming request timed out after "
                        f"{self._config.max_retries + 1} attempt(s)"
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
                content = self.chat(req.messages, req.temperature)
                results[index] = ChatResult(content=content, success=True)
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