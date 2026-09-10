"""OpenAI-compatible API client with httpx streaming (SSE) and layered timeouts.

Supports text and multimodal (base64-encoded image) chat requests,
with concurrent batch execution via ThreadPoolExecutor.

P0 changes:
- httpx replaces urllib; streaming SSE with per-phase timeouts
- ChatAPIConfig gains connect_timeout, first_token_timeout, inter_token_timeout
- retry_on_timeout=False prevents timeout amplification
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import httpx

logger = logging.getLogger(__name__)

# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ChatAPIConfig:
    """Configuration for an OpenAI-compatible chat completions endpoint."""

    api_base: str
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int | None = None  # None means omit from request (use provider default)
    timeout: float = 60.0  # legacy — retained for backward compatibility
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

    Args:
        response: An ``httpx.Response`` from a streaming request.
        first_token_timeout: Maximum seconds to wait for the first SSE line.
        inter_token_timeout: Maximum seconds to wait between lines after the first.

    Yields:
        Raw line strings (including ``"data: ..."`` prefix and ``"[DONE]"``).

    Raises:
        RuntimeError: If a timeout is exceeded in either phase.
    """
    line_queue: queue.Queue[tuple[str, str | Exception | None]] = queue.Queue()

    def _read_lines() -> None:
        try:
            for line in response.iter_lines():
                line_queue.put(("line", line))
        except Exception as exc:
            line_queue.put(("error", exc))
        line_queue.put(("done", None))

    reader_thread = threading.Thread(target=_read_lines, daemon=True)
    reader_thread.start()

    first_token = True
    while True:
        try:
            timeout = first_token_timeout if first_token else inter_token_timeout
            kind, value = line_queue.get(timeout=timeout)
        except queue.Empty:
            phase = "first token (prefill)" if first_token else "inter-token"
            raise RuntimeError(
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


def _send_streaming_request(
    config: ChatAPIConfig,
    payload: dict[str, Any],
) -> tuple[str, str | None]:
    """Send a streaming chat completions request via SSE and return (content, finish_reason).

    Uses ``httpx.Client.stream()`` to POST *payload* with ``stream: true``,
    then parses SSE ``data:`` chunks.  Timeouts are layered:

    1. ``connect_timeout`` — TCP connection + TLS handshake (httpx-level)
    2. ``first_token_timeout`` — wait for the first ``data:`` line (our loop)
    3. ``inter_token_timeout`` — wait between lines during generation (our loop)

    Args:
        config: Endpoint and timeout configuration.
        payload: Full JSON request body (must include ``"stream": true``).

    Returns:
        A tuple of ``(full_content, finish_reason)``.  *finish_reason* is
        ``None`` if none was observed.

    Raises:
        httpx.TimeoutException: On connect-timeout.
        RuntimeError: On HTTP errors, SSE parse errors, or streaming timeouts.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    # Disable httpx-level total/read timeouts — our _iter_lines_with_timeout
    # enforces the per-phase timeouts directly.
    http_timeout = httpx.Timeout(connect=config.connect_timeout)

    content_parts: list[str] = []
    finish_reason: str | None = None

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
                    fr = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
                    if fr is not None:
                        finish_reason = fr

    return "".join(content_parts), finish_reason


def _send_non_streaming_request(
    config: ChatAPIConfig,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Send a non-streaming chat completions request and return the parsed JSON.

    Used for ``chat_with_logprobs`` where streaming logprobs are unreliable
    on many vLLM configurations.

    Raises:
        httpx.TimeoutException: On connect/read timeout.
        RuntimeError: On HTTP errors.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    http_timeout = httpx.Timeout(
        connect=config.connect_timeout,
        read=config.timeout,  # fall back to legacy timeout for non-streaming
    )

    with httpx.Client(timeout=http_timeout) as client:
        response = client.post(
            config.chat_completions_url,
            json=payload,
            headers=headers,
        )
        if response.status_code != 200:
            body = response.text[:1000]
            raise RuntimeError(
                f"Chat completions request failed with HTTP {response.status_code}: {body}"
            )
        return response.json()  # type: ignore[no-any-return]


def _get_finish_reason(response_payload: dict[str, Any]) -> str | None:
    """Extract finish_reason from a chat completions response."""
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason")
    return None


def _extract_content(response_payload: dict[str, Any]) -> str:
    """Extract the message content string from a chat completions response."""
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Chat completions response did not contain choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError("Chat completions response did not contain message content")
    content = message.get("content")
    if content is None:
        # Fallback: Qwen3.8-27B may return reasoning in the "reasoning" field
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
        raise RuntimeError("Chat completions response message had no content")
    if isinstance(content, str):
        return content.strip()
    # content may be a list (multimodal response) — flatten to string
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts).strip()
    return str(content).strip()


def _extract_logprobs(response_payload: dict[str, Any]) -> dict[str, Any]:
    """Extract log-prob data from a chat completions response.

    Returns a dict with ``token_ids`` and ``log_probs`` lists.
    """
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"token_ids": [], "log_probs": []}

    logprobs_content = None
    if isinstance(choices[0], dict):
        logprobs_content = choices[0].get("logprobs")
    if logprobs_content is None:
        return {"token_ids": [], "log_probs": []}

    content_list = logprobs_content.get("content") if isinstance(logprobs_content, dict) else None
    if not isinstance(content_list, list):
        return {"token_ids": [], "log_probs": []}

    token_ids: list[int | str] = []
    log_probs: list[float] = []
    for item in content_list:
        if isinstance(item, dict):
            # vLLM returns token as a string (e.g., "The", " dilemma");
            # some OpenAI-compatible servers also return a numeric token_id
            tid = item.get("token_id") or item.get("token")
            if isinstance(tid, (int, str)):
                token_ids.append(tid)
            lp = item.get("logprob")
            if isinstance(lp, (int, float)):
                log_probs.append(float(lp))

    return {"token_ids": token_ids, "log_probs": log_probs}


def _is_timeout_error(exc: Exception) -> bool:
    """Check if an exception is a timeout-related error."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    msg = str(exc).lower()
    return "timed out" in msg


# ── Public API ─────────────────────────────────────────────────────────────


class ChatAPIClient:
    """OpenAI-compatible chat completions API client.

    Supports text and multimodal messages.  Uses ``httpx`` with streaming
    SSE for text generation and non-streaming for logprobs retrieval.
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
                content, finish_reason = _send_streaming_request(self._config, payload)
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
                    raise
                last_error = exc
                if attempt < self._config.max_retries:
                    time.sleep(min(2.0 ** attempt, 30.0))
            except httpx.TimeoutException as exc:
                if not self._config.retry_on_timeout:
                    raise RuntimeError(
                        f"Streaming request timed out after "
                        f"{self._config.max_retries + 1} attempt(s)"
                    ) from exc
                last_error = exc
                if attempt < self._config.max_retries:
                    time.sleep(min(2.0 ** attempt, 30.0))

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
        """Send a non-streaming chat request and return content + log-probabilities.

        Uses non-streaming mode because streaming logprobs are unreliable on
        many vLLM configurations.  This is only used for the final turn
        where logprobs are needed, so the latency impact is acceptable.

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
            stream=False,
        )
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = _send_non_streaming_request(self._config, payload)
                finish_reason = _get_finish_reason(response)
                if finish_reason == "length":
                    logger.warning(
                        "Response truncated by max_tokens (finish_reason=length), discarding"
                    )
                    raise RuntimeError("Response truncated by max_tokens limit")
                content = _extract_content(response)
                logprobs_data = _extract_logprobs(response)
                return {"content": content, "logprobs": logprobs_data}
            except RuntimeError as exc:
                if _is_timeout_error(exc) and not self._config.retry_on_timeout:
                    raise
                last_error = exc
                if attempt < self._config.max_retries:
                    time.sleep(min(2.0 ** attempt, 30.0))
            except httpx.TimeoutException as exc:
                if not self._config.retry_on_timeout:
                    raise RuntimeError(
                        f"Logprobs request timed out after "
                        f"{self._config.max_retries + 1} attempt(s)"
                    ) from exc
                last_error = exc
                if attempt < self._config.max_retries:
                    time.sleep(min(2.0 ** attempt, 30.0))

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