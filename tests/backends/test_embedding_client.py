"""Tests for :mod:`ard.backends.embedding_client` — no network, mock transport only.

Every test installs an :class:`httpx.MockTransport` as the module's
``httpx.Client``, so the suite can never dial a real endpoint: the only base URL
used is ``https://mock.invalid/v1``, which is not resolvable and is answered
entirely by the in-process mock transport.
"""

import json
import logging
import math
from collections.abc import Callable

import httpx
import pytest

from ard.backends import embedding_client as ec

#: Captured before any monkeypatching: the module under test builds its own
#: ``httpx.Client``, so a test that replaces ``httpx.Client`` must still be able
#: to construct a real client (bound to a MockTransport) from this reference.
_REAL_HTTPX_CLIENT = httpx.Client

#: Unresolvable host on purpose: a MockTransport intercepts every request, and
#: anything that slipped through would fail to resolve instead of hitting a
#: real embedding endpoint.
API_BASE = "https://mock.invalid/v1"
MODEL = "mock-embed"
DIMENSION = 3
#: Assembled from parts so the tracked source carries no key-shaped literal
#: (§15.1).  It must stay a distinctive string: the "key never leaks" assertions
#: below are only meaningful while *this* exact value could not occur by accident.
API_KEY = "sk-" + "test-embedding-client-not-a-real-key"


# ── Mock endpoint ───────────────────────────────────────────────────────────


class MockEndpoint:
    """A recording ``httpx.MockTransport`` installed as ``ec.httpx.Client``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        self.requests: list[httpx.Request] = []
        self.payloads: list[dict] = []
        self.attempts = 0

    def install(self, responder: Callable[[int, httpx.Request], httpx.Response]) -> "MockEndpoint":
        """Route every module request through *responder(attempt, request)*."""

        def handler(request: httpx.Request) -> httpx.Response:
            self.attempts += 1
            self.requests.append(request)
            try:
                self.payloads.append(json.loads(request.content))
            except (json.JSONDecodeError, UnicodeDecodeError):  # pragma: no cover
                self.payloads.append({})
            return responder(self.attempts, request)

        def fake_client(*args: object, **kwargs: object) -> httpx.Client:
            kwargs.pop("timeout", None)
            return _REAL_HTTPX_CLIENT(
                transport=httpx.MockTransport(handler), timeout=httpx.Timeout(None)
            )

        self._monkeypatch.setattr(ec.httpx, "Client", fake_client)
        return self

    @property
    def last_authorization(self) -> str | None:
        return self.requests[-1].headers.get("authorization")


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch) -> MockEndpoint:
    return MockEndpoint(monkeypatch)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff delays instead of actually sleeping."""
    delays: list[float] = []
    monkeypatch.setattr(ec.time, "sleep", delays.append)
    return delays


def make_client(**overrides: object) -> ec.EmbeddingClient:
    params: dict[str, object] = {
        "api_base": API_BASE,
        "api_key": None,
        "model": MODEL,
        "dimension": DIMENSION,
    }
    params.update(overrides)
    return ec.EmbeddingClient(**params)  # type: ignore[arg-type]


def embeddings_response(vectors: ec.EmbeddingMatrix, *, status: int = 200) -> httpx.Response:
    """A well-formed OpenAI-compatible embeddings body for *vectors*.

    ``json.dumps`` is used with ``allow_nan=True`` so the non-finite cases can
    put a literal ``NaN`` / ``Infinity`` on the wire (``json=`` would refuse).
    """
    body = json.dumps(
        {
            "object": "list",
            "model": MODEL,
            "data": [
                {"object": "embedding", "index": i, "embedding": list(row)}
                for i, row in enumerate(vectors)
            ],
        },
        allow_nan=True,
    )
    return httpx.Response(status, content=body.encode("utf-8"))


def echo_vectors(attempt: int, request: httpx.Request) -> httpx.Response:
    """Answer each input string ``t<i>`` with the unit vector for ``i``."""
    payload = json.loads(request.content)
    vectors = [[float(int(text[1:])), 1.0, 0.0] for text in payload["input"]]
    return embeddings_response(vectors)


# ── Construction and key hygiene ────────────────────────────────────────────


def test_constructor_defaults_and_url():
    client = make_client()
    assert client.embeddings_url == "https://mock.invalid/v1/embeddings"
    assert client.normalize is True
    shown = repr(client)
    assert "batch_size=32" in shown
    assert "connect_timeout=10.0" in shown
    assert "read_timeout=60.0" in shown
    assert "max_retries=2" in shown
    assert "api_key=<unset>" in shown


def test_constructor_rejects_missing_required_fields():
    with pytest.raises(ValueError):
        ec.EmbeddingClient(api_base=None, model=MODEL, dimension=DIMENSION)
    with pytest.raises(ValueError):
        ec.EmbeddingClient(api_base=API_BASE, model=None, dimension=DIMENSION)
    with pytest.raises(ValueError):
        ec.EmbeddingClient(api_base=API_BASE, model=MODEL, dimension=None)
    with pytest.raises(ValueError):
        make_client(dimension=0)
    with pytest.raises(ValueError):
        make_client(max_retries=-1)


def test_constructor_normalizes_empty_key_to_none():
    client = make_client(api_key="")
    assert "api_key=<unset>" in repr(client)
    assert client._api_key is None


# ── Boundary validation (input) ─────────────────────────────────────────────


def test_empty_input_raises_and_hits_no_endpoint(endpoint: MockEndpoint):
    endpoint.install(echo_vectors)
    with pytest.raises(ec.EmptyInputError):
        make_client().embed_texts([])
    assert endpoint.attempts == 0


def test_non_string_input_is_rejected(endpoint: MockEndpoint):
    endpoint.install(echo_vectors)
    with pytest.raises(TypeError):
        make_client().embed_texts([1, 2])  # type: ignore[list-item]
    assert endpoint.attempts == 0


# ── Normal batching ─────────────────────────────────────────────────────────


def test_batches_by_batch_size_and_preserves_order(endpoint: MockEndpoint):
    endpoint.install(echo_vectors)
    batch = make_client(batch_size=2).embed_texts([f"t{i}" for i in range(5)])

    assert batch.count == 5
    assert batch.model == MODEL
    assert batch.dimension == DIMENSION
    assert batch.normalized is True
    assert [payload["input"] for payload in endpoint.payloads] == [
        ["t0", "t1"],
        ["t2", "t3"],
        ["t4"],
    ]
    assert batch.vectors[0] == [0.0, 1.0, 0.0]
    assert batch.vectors[1] == pytest.approx([1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0])


def test_request_contract_sent_to_endpoint(endpoint: MockEndpoint):
    endpoint.install(echo_vectors)
    make_client().embed_texts(["t0"])
    request = endpoint.requests[0]
    assert str(request.url) == "https://mock.invalid/v1/embeddings"
    assert request.headers["content-type"] == "application/json"
    assert "authorization" not in request.headers
    assert endpoint.payloads[0]["model"] == MODEL
    assert endpoint.payloads[0]["encoding_format"] == "float"


# ── Boundary validation (response) ──────────────────────────────────────────


def test_count_mismatch_raises_and_is_not_retried(endpoint: MockEndpoint, no_sleep: list[float]):
    endpoint.install(lambda attempt, request: embeddings_response([[1.0, 2.0, 3.0]]))

    with pytest.raises(ec.EmbeddingCountMismatchError) as excinfo:
        make_client().embed_texts(["a", "b"])

    assert excinfo.value.batch_index == 0
    assert endpoint.attempts == 1
    assert no_sleep == []


def test_dimension_mismatch_raises(endpoint: MockEndpoint, no_sleep: list[float]):
    endpoint.install(lambda attempt, request: embeddings_response([[1.0, 2.0]]))

    with pytest.raises(ec.EmbeddingDimensionMismatchError) as excinfo:
        make_client().embed_texts(["a"])

    assert "expected dimension 3, got 2" in str(excinfo.value)
    assert endpoint.attempts == 1
    assert no_sleep == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_raise(endpoint: MockEndpoint, bad: float, no_sleep: list[float]):
    endpoint.install(lambda attempt, request: embeddings_response([[1.0, bad, 3.0]]))

    with pytest.raises(ec.NonFiniteEmbeddingError) as excinfo:
        make_client().embed_texts(["a"])

    assert excinfo.value.batch_index == 0
    assert endpoint.attempts == 1
    assert no_sleep == []


def test_zero_vector_cannot_be_normalized(endpoint: MockEndpoint):
    endpoint.install(lambda attempt, request: embeddings_response([[0.0, 0.0, 0.0]]))

    with pytest.raises(ec.NonFiniteEmbeddingError):
        make_client(normalize=True).embed_texts(["a"])


def test_unparsable_body_raises(endpoint: MockEndpoint):
    endpoint.install(lambda attempt, request: httpx.Response(200, text="<html>nope</html>"))

    with pytest.raises(ec.EmbeddingResponseError):
        make_client().embed_texts(["a"])


# ── Normalisation is explicit, never silent ─────────────────────────────────


def test_normalization_on_returns_unit_vectors(endpoint: MockEndpoint):
    endpoint.install(lambda attempt, request: embeddings_response([[3.0, 4.0, 0.0]]))
    batch = make_client(normalize=True).embed_texts(["a"])

    assert batch.normalized is True
    assert batch.vectors[0] == pytest.approx([0.6, 0.8, 0.0])
    assert math.dist(batch.vectors[0], [0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_normalization_off_returns_raw_vectors(endpoint: MockEndpoint):
    endpoint.install(lambda attempt, request: embeddings_response([[3.0, 4.0, 0.0]]))
    batch = make_client(normalize=False).embed_texts(["a"])

    assert batch.normalized is False
    assert batch.vectors[0] == [3.0, 4.0, 0.0]


def test_normalization_off_accepts_zero_vector(endpoint: MockEndpoint):
    endpoint.install(lambda attempt, request: embeddings_response([[0.0, 0.0, 0.0]]))
    batch = make_client(normalize=False).embed_texts(["a"])

    assert batch.normalized is False
    assert batch.vectors[0] == [0.0, 0.0, 0.0]


# ── Retry policy ────────────────────────────────────────────────────────────


def test_5xx_is_retried_then_succeeds(endpoint: MockEndpoint, no_sleep: list[float]):
    def responder(attempt: int, request: httpx.Request) -> httpx.Response:
        if attempt < 3:
            return httpx.Response(503, text=f"upstream busy {attempt}")
        return echo_vectors(attempt, request)

    endpoint.install(responder)
    batch = make_client(max_retries=2).embed_texts(["t0"])

    assert batch.count == 1
    assert endpoint.attempts == 3
    assert no_sleep == [1.0, 2.0]


def test_retries_exhausted_raises_request_error_with_context(
    endpoint: MockEndpoint, no_sleep: list[float]
):
    endpoint.install(lambda attempt, request: httpx.Response(500, text="boom"))

    with pytest.raises(ec.EmbeddingRequestError) as excinfo:
        make_client(max_retries=2).embed_texts(["a", "b"])

    error = excinfo.value
    assert error.batch_index == 0
    assert error.attempts == 3
    assert error.retries == 2
    assert error.status_code == 500
    assert "HTTP 500" in str(error)
    assert "3 attempt(s)" in str(error)
    assert endpoint.attempts == 3
    assert no_sleep == [1.0, 2.0]


def test_429_is_retried(endpoint: MockEndpoint, no_sleep: list[float]):
    def responder(attempt: int, request: httpx.Request) -> httpx.Response:
        if attempt == 1:
            return httpx.Response(429, text="slow down")
        return echo_vectors(attempt, request)

    endpoint.install(responder)
    batch = make_client(max_retries=2).embed_texts(["t0"])

    assert batch.count == 1
    assert endpoint.attempts == 2
    assert no_sleep == [1.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_other_4xx_is_not_retried(endpoint: MockEndpoint, no_sleep: list[float], status: int):
    endpoint.install(lambda attempt, request: httpx.Response(status, text="bad request"))

    with pytest.raises(ec.EmbeddingRequestError) as excinfo:
        make_client(max_retries=3).embed_texts(["a"])

    assert excinfo.value.attempts == 1
    assert excinfo.value.retries == 0
    assert excinfo.value.status_code == status
    assert "non-retryable" in str(excinfo.value)
    assert endpoint.attempts == 1
    assert no_sleep == []


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("connection refused"), httpx.ReadTimeout("read timed out")],
)
def test_timeout_and_connect_errors_are_retried(
    endpoint: MockEndpoint, no_sleep: list[float], failure: Exception
):
    def responder(attempt: int, request: httpx.Request) -> httpx.Response:
        if attempt == 1:
            raise failure
        return echo_vectors(attempt, request)

    endpoint.install(responder)
    batch = make_client(max_retries=2).embed_texts(["t0"])

    assert batch.count == 1
    assert endpoint.attempts == 2
    assert no_sleep == [1.0]


def test_transport_failure_exhausted_reports_no_http_response(
    endpoint: MockEndpoint, no_sleep: list[float]
):
    def responder(attempt: int, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connection refused (attempt {attempt})")

    endpoint.install(responder)

    with pytest.raises(ec.EmbeddingRequestError) as excinfo:
        make_client(max_retries=2).embed_texts(["a"])

    assert excinfo.value.status_code is None
    assert excinfo.value.attempts == 3
    assert "no HTTP response" in str(excinfo.value)
    assert endpoint.attempts == 3


def test_max_retries_zero_makes_exactly_one_attempt(endpoint: MockEndpoint):
    endpoint.install(lambda attempt, request: httpx.Response(500, text="boom"))

    with pytest.raises(ec.EmbeddingRequestError) as excinfo:
        make_client(max_retries=0).embed_texts(["a"])

    assert excinfo.value.attempts == 1
    assert excinfo.value.retries == 0
    assert endpoint.attempts == 1


# ── Key never leaks ─────────────────────────────────────────────────────────


def test_api_key_not_in_repr():
    client = make_client(api_key=API_KEY)
    assert API_KEY not in repr(client)
    assert "api_key=<redacted>" in repr(client)


def test_api_key_used_but_absent_from_exception_and_logs(
    endpoint: MockEndpoint, caplog: pytest.LogCaptureFixture
):
    # The (hostile) mock echoes the Authorization header back in its error body;
    # the client must redact it before it reaches any message or log record.
    def responder(attempt: int, request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=f"upstream echo: {request.headers['authorization']}")

    endpoint.install(responder)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ec.EmbeddingRequestError) as excinfo:
            make_client(api_key=API_KEY, max_retries=1).embed_texts(["a"])

    assert endpoint.last_authorization == f"Bearer {API_KEY}"
    assert API_KEY not in str(excinfo.value)
    assert "<redacted>" in str(excinfo.value)
    assert API_KEY not in caplog.text
    assert API_KEY not in repr(make_client(api_key=API_KEY))


def test_api_key_absent_from_non_retryable_4xx_message(endpoint: MockEndpoint):
    def responder(attempt: int, request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"denied for token {request.headers['authorization']}")

    endpoint.install(responder)

    with pytest.raises(ec.EmbeddingRequestError) as excinfo:
        make_client(api_key=API_KEY).embed_texts(["a"])

    assert API_KEY not in str(excinfo.value)
    assert "<redacted>" in str(excinfo.value)
