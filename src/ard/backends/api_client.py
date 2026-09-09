"""OpenAI-compatible API client using only stdlib urllib.

Supports text and multimodal (base64-encoded image) chat requests,
with concurrent batch execution via ThreadPoolExecutor.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    timeout: float = 60.0
    max_retries: int = 2
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
) -> dict[str, Any]:
    """Build the JSON payload for a chat completions request."""
    payload: dict[str, Any] = {
        "model": config.model_name,
        "messages": messages,
        "temperature": temperature if temperature is not None else config.temperature,
    }
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    if config.enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    return payload


def _send_request(
    config: ChatAPIConfig,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Send a single chat completions request and return the parsed JSON response.

    Raises:
        RuntimeError: On HTTP errors or network failures.
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = urllib.request.Request(
        config.chat_completions_url,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            response_payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Chat completions request failed with HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Chat completions request failed: {exc.reason}") from exc
    return response_payload


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


# ── Public API ─────────────────────────────────────────────────────────────


class ChatAPIClient:
    """OpenAI-compatible chat completions API client.

    Supports text and multimodal messages.  Uses only ``urllib.request``
    (no third-party HTTP libraries).  Concurrency is handled by
    ``ThreadPoolExecutor``.

    Parameters:
        config: Endpoint, model, and auth configuration.
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
        """Send a single chat request and return the content text.

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
        )
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = _send_request(self._config, payload)
                finish_reason = _get_finish_reason(response)
                if finish_reason == "length":
                    logger.warning(
                        "Response truncated by max_tokens (finish_reason=length), discarding"
                    )
                    raise RuntimeError("Response truncated by max_tokens limit")
                return _extract_content(response)
            except RuntimeError as exc:
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
        """Send a chat request and return content + log-probabilities.

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
        )
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = _send_request(self._config, payload)
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
