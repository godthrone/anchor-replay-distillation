"""OpenAI-compatible embedding facility: text batches in, vectors out.

Responsibility: turn a batch of texts into embedding vectors by calling an
OpenAI-compatible ``POST {api_base}/embeddings`` endpoint over ``httpx``.
This is the *facility* layer (§1.1 计算与设施分离): it owns network I/O,
batching, retry/backoff and response-boundary validation, and nothing else.

It deliberately does **not**:

* read any config file — every setting is an explicit constructor argument,
  handed in by the caller (the wiring lives one layer up);
* write anything to disk — vectors are returned, never persisted here;
* normalise silently — L2 normalisation happens only when the caller asks for
  it via ``normalize=True``, and the returned :class:`EmbeddingBatch` always
  states ``normalized`` explicitly.  ``coverage.VectorSet`` (the acceptance
  ruler) requires unit-norm rows, so the wiring layer must request it, but
  this module never decides that on the caller's behalf.

Retry / error-handling policy mirrors :mod:`ard.backends.api_client`:
``for attempt in range(max_retries + 1)``, exponential backoff
``min(2.0 ** attempt, 30.0)`` with :func:`time.sleep`, and a final exception
that names the batch, the attempt count and the HTTP status.  Only timeouts,
transport (connect/read) errors, ``429`` and ``5xx`` are retried; every other
``4xx`` fails immediately.  A response whose shape, count, dimension or
finiteness violates the contract is never retried and never silently trimmed.

The API key is private state: it is never logged, never interpolated into an
exception message, never part of ``repr``, and any response-body excerpt that
reaches a message is redacted against it (§15).
"""

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, TypeAlias

import httpx

logger = logging.getLogger(__name__)

#: Default number of texts sent in one HTTP request.
DEFAULT_BATCH_SIZE = 32
#: Default TCP connect / TLS handshake budget, in seconds.
DEFAULT_CONNECT_TIMEOUT = 10.0
#: Default whole-response read budget, in seconds.
DEFAULT_READ_TIMEOUT = 60.0
#: Default number of *retries* (so ``max_retries + 1`` HTTP attempts).
DEFAULT_MAX_RETRIES = 2
#: Upper bound on the exponential backoff delay, in seconds.
MAX_BACKOFF_SECONDS = 30.0
#: Substituted for the API key wherever diagnostic text is built.
_REDACTED = "<redacted>"


# ── Shared types ────────────────────────────────────────────────────────────

EmbeddingRow: TypeAlias = list[float]
"""One embedding vector: ``dimension`` floats, in model order."""

EmbeddingMatrix: TypeAlias = list[EmbeddingRow]
"""One batch of embedding rows — one row per requested text, in input order.

Named instead of written out inline as an anonymous nested container type: the
shape *is* a contract (one row per text, input order, ``dimension`` long —
§2.1), and a name states it once instead of at every signature (§12.1 forbids
anonymous nested container types).
"""


# ── Exception types ─────────────────────────────────────────────────────────


class EmbeddingError(Exception):
    """Base class for every failure raised by this module."""


class EmptyInputError(EmbeddingError):
    """The caller handed in an empty text list.

    A batch of zero texts is a caller bug, not an embedding of "nothing": the
    endpoint would answer ``data: []``, which downstream would become a zero-row
    matrix — exactly the silent-empty-result failure §2.3/§13.1 forbid.
    """


class EmbeddingResponseError(EmbeddingError):
    """The endpoint answered 200 with a body that violates the contract.

    Never retried: the same request would come back with the same broken body,
    so retrying only burns time.  Attributes:
        batch_index: 0-based index of the request batch that failed.
    """

    def __init__(self, message: str, *, batch_index: int) -> None:
        super().__init__(message)
        self.batch_index = batch_index


class EmbeddingCountMismatchError(EmbeddingResponseError):
    """The response carried a different number of vectors than texts sent."""


class EmbeddingDimensionMismatchError(EmbeddingResponseError):
    """A returned vector's length differs from the configured ``dimension``."""


class NonFiniteEmbeddingError(EmbeddingResponseError):
    """A returned vector contains NaN/infinity or cannot be L2-normalised."""


class EmbeddingRequestError(EmbeddingError):
    """The request failed at the HTTP/transport level.

    Covers both terminal ``4xx`` (raised after a single attempt) and exhausted
    retries on timeouts, connect errors, ``429`` or ``5xx``.

    Attributes:
        batch_index: 0-based index of the request batch that failed.
        attempts: total HTTP attempts made (1 means "no retry was made").
        status_code: HTTP status of the last response, or ``None`` when the
            failure was a transport error with no response at all.
    """

    def __init__(
        self,
        message: str,
        *,
        batch_index: int,
        attempts: int,
        status_code: int | None,
    ) -> None:
        super().__init__(message)
        self.batch_index = batch_index
        self.attempts = attempts
        self.status_code = status_code

    @property
    def retries(self) -> int:
        """Number of retries performed (``attempts - 1``)."""
        return self.attempts - 1


class _RetryableStatusError(Exception):
    """Internal signal: a retryable HTTP status (429 / 5xx) was received."""

    def __init__(self, status_code: int, body_snippet: str) -> None:
        super().__init__(f"HTTP {status_code}: {body_snippet}")
        self.status_code = status_code


# ── Result container ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Vectors returned for one :meth:`EmbeddingClient.embed_texts` call.

    Attributes:
        vectors: ``count`` rows, each ``dimension`` floats, in input order.
        model: the model name the vectors were requested from.
        dimension: the configured embedding dimension (already validated
            against every row).
        normalized: whether every row was L2-normalised by the client.
            ``coverage.VectorSet`` needs ``True``; the flag is explicit so the
            wiring layer can assert it instead of guessing (§2.2 显式即防呆).
    """

    vectors: EmbeddingMatrix
    model: str
    dimension: int
    normalized: bool

    @property
    def count(self) -> int:
        """Number of vectors (equals the number of texts requested)."""
        return len(self.vectors)


# ── Client ──────────────────────────────────────────────────────────────────


class EmbeddingClient:
    """OpenAI-compatible ``/embeddings`` client with batching and retries.

    Every constructor argument is explicit — the client never reads a config
    file (§7.1: the caller owns configuration).  ``None`` means "not provided":
    for the tunables that selects the documented module default, for ``api_key``
    it means "no authorization header", and for ``api_base`` / ``model`` /
    ``dimension`` it is a hard error (§2.2 ``None`` 是唯一合法空值).

    Args:
        api_base: Endpoint base **including** ``/v1``, e.g.
            ``"http://localhost:8000/v1"`` (same convention as
            :class:`ard.backends.api_client.ChatAPIConfig`).  The client POSTs
            to ``api_base.rstrip("/") + "/embeddings"``.
        api_key: Bearer token, or ``None`` for servers that need none.
            Never logged and never part of ``repr``.
        model: Embedding model name sent in the request body.
        dimension: Expected vector length; every returned row is validated
            against it.
        batch_size: Texts per HTTP request; ``None`` → 32.
        connect_timeout: TCP connect / TLS handshake budget in seconds;
            ``None`` → 10.0.
        read_timeout: Response read budget in seconds; ``None`` → 60.0.
        max_retries: Retries after the first attempt; ``None`` → 2.  A value
            of ``0`` means exactly one attempt.
        normalize: When ``True`` (default) each returned row is L2-normalised
            so that ``dot == cos`` for :class:`ard.core.coverage.VectorSet`.
            When ``False`` the raw server vectors are returned untouched and
            :attr:`EmbeddingBatch.normalized` is ``False``.
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        max_retries: int | None = None,
        normalize: bool = True,
    ) -> None:
        if api_base is None or api_base == "":
            raise ValueError("api_base must be a non-empty URL base (e.g. 'http://host:8000/v1')")
        if model is None or model == "":
            raise ValueError("model must be a non-empty embedding model name")
        if dimension is None or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError(f"dimension must be a positive int, got {dimension!r}")
        if batch_size is not None and (isinstance(batch_size, bool) or batch_size <= 0):
            raise ValueError(f"batch_size must be a positive int or None, got {batch_size!r}")
        if connect_timeout is not None and connect_timeout <= 0:
            raise ValueError(f"connect_timeout must be > 0 or None, got {connect_timeout!r}")
        if read_timeout is not None and read_timeout <= 0:
            raise ValueError(f"read_timeout must be > 0 or None, got {read_timeout!r}")
        if max_retries is not None and (isinstance(max_retries, bool) or max_retries < 0):
            raise ValueError(f"max_retries must be >= 0 or None, got {max_retries!r}")
        if normalize is not True and normalize is not False:
            raise TypeError(f"normalize must be True or False, got {normalize!r}")

        # §2.2: ``""`` is not a credential.  Normalise it to None so exactly one
        # thing is tested later ("was a key supplied?").
        self._api_base: str = api_base
        self._api_key: str | None = api_key if api_key != "" else None
        self._model: str = model
        self._dimension: int = dimension
        self._batch_size: int = DEFAULT_BATCH_SIZE if batch_size is None else batch_size
        self._connect_timeout: float = (
            DEFAULT_CONNECT_TIMEOUT if connect_timeout is None else connect_timeout
        )
        self._read_timeout: float = DEFAULT_READ_TIMEOUT if read_timeout is None else read_timeout
        self._max_retries: int = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        self._normalize: bool = normalize

    # ── Introspection (key-safe) ───────────────────────────────────────────

    def __repr__(self) -> str:
        """Repr that can never leak the API key (§15)."""
        return (
            f"EmbeddingClient(api_base={self._api_base!r}, model={self._model!r}, "
            f"dimension={self._dimension!r}, batch_size={self._batch_size!r}, "
            f"connect_timeout={self._connect_timeout!r}, read_timeout={self._read_timeout!r}, "
            f"max_retries={self._max_retries!r}, normalize={self._normalize!r}, "
            f"api_key={self._describe_api_key()})"
        )

    def _describe_api_key(self) -> str:
        """Report *whether* a key is set, never the key itself."""
        return "<unset>" if self._api_key is None else _REDACTED

    @property
    def embeddings_url(self) -> str:
        """Full endpoint URL the client POSTs to."""
        return self._api_base.rstrip("/") + "/embeddings"

    @property
    def normalize(self) -> bool:
        """Whether returned vectors are L2-normalised before being handed back."""
        return self._normalize

    # ── Public API ─────────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        """Embed *texts*, splitting the request into ``batch_size`` chunks.

        Args:
            texts: Non-empty list of strings.  Order is preserved in the result.

        Returns:
            An :class:`EmbeddingBatch` whose ``count`` equals ``len(texts)``.

        Raises:
            EmptyInputError: If *texts* is empty.
            TypeError: If *texts* is not a list of strings.
            EmbeddingResponseError: If a 200 response violates the count,
                dimension or finiteness contract.
            EmbeddingRequestError: If a request fails at the HTTP/transport
                level after retries, or immediately on a non-retryable 4xx.
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be a list of str, got {type(texts).__name__}")
        if len(texts) == 0:
            raise EmptyInputError("texts is empty: an embedding of zero texts is undefined")
        for position, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{position}] must be str, got {type(text).__name__}")

        vectors: EmbeddingMatrix = []
        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start : start + self._batch_size]
            batch_index = start // self._batch_size
            vectors.extend(self._embed_batch(chunk, batch_index))

        # Belt-and-braces: per-batch count checks already ran, but the caller's
        # contract is "one vector per text", so it is re-asserted at the exit
        # boundary rather than assumed (§2.3).
        if len(vectors) != len(texts):
            raise EmbeddingCountMismatchError(
                f"embedding produced {len(vectors)} vector(s) for {len(texts)} text(s)",
                batch_index=len(texts) // self._batch_size,
            )
        return EmbeddingBatch(
            vectors=vectors,
            model=self._model,
            dimension=self._dimension,
            normalized=self._normalize,
        )

    # ── One batch with retries ─────────────────────────────────────────────

    def _embed_batch(self, texts: list[str], batch_index: int) -> EmbeddingMatrix:
        """Send one batch, retrying the retryable failures only.

        The retry policy mirrors :meth:`ard.backends.api_client.ChatAPIClient.chat`:
        ``max_retries + 1`` attempts, backoff ``min(2.0 ** attempt, 30.0)``,
        ``logger.warning`` per failed attempt, and a terminal
        :class:`EmbeddingRequestError` naming the batch, attempts and status.
        """
        total_attempts = self._max_retries + 1
        last_error: Exception | None = None
        last_status: int | None = None

        for attempt in range(total_attempts):
            try:
                return self._post_embeddings(texts, batch_index)
            except EmbeddingResponseError:
                # Deterministic contract violation: the same request would
                # produce the same bad body, so retrying is pure waste.
                raise
            except _RetryableStatusError as exc:
                last_error, last_status = exc, exc.status_code
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # Timeouts and connect/read errors carry no HTTP status.
                last_error, last_status = exc, None

            if attempt < self._max_retries:
                delay = min(2.0**attempt, MAX_BACKOFF_SECONDS)
                logger.warning(
                    "Embedding batch %d attempt %d/%d failed: %s. Retrying in %.1fs...",
                    batch_index,
                    attempt + 1,
                    total_attempts,
                    last_error,
                    delay,
                )
                time.sleep(delay)

        logger.error(
            "Embedding batch %d failed after %d attempt(s): %s",
            batch_index,
            total_attempts,
            last_error,
        )
        status_note = (
            f", last HTTP status {last_status}" if last_status is not None else ", no HTTP response"
        )
        raise EmbeddingRequestError(
            f"embedding batch {batch_index} failed after {total_attempts} attempt(s) "
            f"({total_attempts - 1} retr{'y' if total_attempts == 2 else 'ies'}{status_note}): "
            f"{last_error}",
            batch_index=batch_index,
            attempts=total_attempts,
            status_code=last_status,
        ) from last_error

    # ── One HTTP request ───────────────────────────────────────────────────

    def _post_embeddings(self, texts: list[str], batch_index: int) -> EmbeddingMatrix:
        """POST one batch and validate the 200 response body.

        Raises:
            _RetryableStatusError: On 429 / 5xx (the caller retries).
            EmbeddingRequestError: On any other non-200 status (no retry).
            EmbeddingResponseError: On an unparsable or contract-violating body.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "input": list(texts),
            # Pin the encoding: the OpenAI default is float, but leaving it
            # implicit would let a base64-answering server slip non-numeric rows
            # into the pipeline as a confusing shape error (§2.2).
            "encoding_format": "float",
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"

        http_timeout = httpx.Timeout(timeout=self._read_timeout, connect=self._connect_timeout)
        with httpx.Client(timeout=http_timeout) as client:
            response = client.post(self.embeddings_url, json=payload, headers=headers)

        status = response.status_code
        if status == 429 or 500 <= status < 600:
            raise _RetryableStatusError(status, self._redact(_body_snippet(response)))
        if status != 200:
            raise EmbeddingRequestError(
                f"embedding batch {batch_index} received non-retryable HTTP {status}: "
                f"{self._redact(_body_snippet(response))}",
                batch_index=batch_index,
                attempts=1,
                status_code=status,
            )
        return self._parse_embeddings(response, batch_index, len(texts))

    def _parse_embeddings(
        self, response: httpx.Response, batch_index: int, expected_count: int
    ) -> EmbeddingMatrix:
        """Turn a 200 body into validated rows; raise on any contract violation."""
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise EmbeddingResponseError(
                f"embedding batch {batch_index}: response body is not valid JSON: {exc}",
                batch_index=batch_index,
            ) from exc
        if not isinstance(payload, dict):
            raise EmbeddingResponseError(
                f"embedding batch {batch_index}: response is not a JSON object",
                batch_index=batch_index,
            )
        items = payload.get("data")
        if not isinstance(items, list):
            raise EmbeddingResponseError(
                f"embedding batch {batch_index}: response carries no 'data' list",
                batch_index=batch_index,
            )
        if len(items) != expected_count:
            raise EmbeddingCountMismatchError(
                f"embedding batch {batch_index}: requested {expected_count} embedding(s) "
                f"but the response carried {len(items)}",
                batch_index=batch_index,
            )

        vectors: EmbeddingMatrix = []
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                raise EmbeddingResponseError(
                    f"embedding batch {batch_index} item {position}: entry is not an object",
                    batch_index=batch_index,
                )
            raw = item.get("embedding")
            if not isinstance(raw, list) or len(raw) == 0:
                raise EmbeddingResponseError(
                    f"embedding batch {batch_index} item {position}: 'embedding' is not a "
                    "non-empty list (a base64 payload is not accepted)",
                    batch_index=batch_index,
                )
            if len(raw) != self._dimension:
                raise EmbeddingDimensionMismatchError(
                    f"embedding batch {batch_index} item {position}: expected dimension "
                    f"{self._dimension}, got {len(raw)}",
                    batch_index=batch_index,
                )
            vectors.append(self._to_unit_vector(raw, batch_index, position))
        return vectors

    def _to_unit_vector(self, raw: list[Any], batch_index: int, position: int) -> EmbeddingRow:
        """Validate one raw row and optionally L2-normalise it."""
        values: EmbeddingRow = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingResponseError(
                    f"embedding batch {batch_index} item {position}: non-numeric element "
                    f"{value!r} ({type(value).__name__})",
                    batch_index=batch_index,
                )
            number = float(value)
            if not math.isfinite(number):
                raise NonFiniteEmbeddingError(
                    f"embedding batch {batch_index} item {position}: contains a non-finite "
                    f"value ({number!r}); NaN/infinity cannot enter the vector set",
                    batch_index=batch_index,
                )
            values.append(number)

        if not self._normalize:
            return values

        norm = math.sqrt(sum(component * component for component in values))
        if norm == 0.0 or not math.isfinite(norm):
            raise NonFiniteEmbeddingError(
                f"embedding batch {batch_index} item {position}: L2 norm is {norm!r}, so the "
                "row cannot be normalised to unit length",
                batch_index=batch_index,
            )
        return [component / norm for component in values]

    def _redact(self, text: str) -> str:
        """Replace the API key anywhere it appears in diagnostic text (§15)."""
        if self._api_key is None:
            return text
        return text.replace(self._api_key, _REDACTED)


# ── Module helpers ──────────────────────────────────────────────────────────


def _body_snippet(response: httpx.Response, limit: int = 200) -> str:
    """First *limit* characters of the response body, for diagnostics only.

    Decoded with ``errors="replace"`` so a binary or mis-encoded error body can
    never raise while an exception is being built.
    """
    return response.content.decode("utf-8", errors="replace")[:limit]
